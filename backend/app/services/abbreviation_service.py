"""Abbreviation expansion service — lookup, suffix expansion, glossary generation.

Caches the compiled lookup in-process for 30 seconds (same pattern as settings_service).
All expansion is deterministic (regex + CSV lookup). No LLM calls.
"""
from __future__ import annotations

import csv
import io
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.models.abbreviation import Abbreviation, AbbreviationList
from app.services.settings_service import get_setting

logger = logging.getLogger(__name__)

_CACHE_TTL = 30  # seconds
_cache: dict[tuple[Optional[int], str], tuple["AbbreviationLookup", float]] = {}


@dataclass
class AbbreviationLookup:
    """Compiled abbreviation lookup: {abbr: [form1, form2, ...]} with pre-compiled regex."""
    forward: Dict[str, List[str]] = field(default_factory=dict)
    all_abbrs_sorted: List[str] = field(default_factory=list)
    compiled_patterns: Dict[str, re.Pattern] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.forward


def _invalidate_cache(org_id: Optional[int] = None) -> None:
    """Invalidate cache for a specific org, or all cache if org_id is None."""
    if org_id is None:
        _cache.clear()
    else:
        keys_to_remove = [k for k in _cache if k[0] == org_id or k[0] is None]
        for k in keys_to_remove:
            del _cache[k]


def _get_active_list_ids(db: Session, org_id: Optional[int]) -> List[int]:
    """Get IDs of enabled abbreviation lists for this org (universal + org-specific)."""
    query = db.query(AbbreviationList.id).filter(AbbreviationList.is_enabled.is_(True))
    if org_id is not None:
        query = query.filter(
            (AbbreviationList.org_id.is_(None)) | (AbbreviationList.org_id == org_id)
        )
    else:
        query = query.filter(AbbreviationList.org_id.is_(None))
    return [row[0] for row in query.all()]


def build_lookup(db: Session, org_id: Optional[int] = None) -> AbbreviationLookup:
    """Build the compiled abbreviation lookup, cached for 30 seconds.

    Merges universal lists (org_id=NULL) with org-specific lists.
    Returns an empty AbbreviationLookup if no lists are configured or expansion is disabled.
    """
    # Check if expansion is enabled
    if not get_setting(db, "ABBREVIATION_EXPANSION_ENABLED", org_id):
        return AbbreviationLookup()

    cache_key = (org_id, "lookup")
    now = time.time()
    cached = _cache.get(cache_key)
    if cached and (now - cached[1]) < _CACHE_TTL:
        return cached[0]

    list_ids = _get_active_list_ids(db, org_id)
    if not list_ids:
        lookup = AbbreviationLookup()
        _cache[cache_key] = (lookup, now)
        return lookup

    # Build forward mapping: {abbr: [form1, form2, ...]}
    rows = (
        db.query(Abbreviation)
        .filter(Abbreviation.list_id.in_(list_ids))
        .all()
    )
    forward: Dict[str, List[str]] = {}
    for row in rows:
        abbr = row.abbreviation.strip()
        form = row.expanded_form.strip()
        if abbr and form:
            if abbr not in forward:
                forward[abbr] = []
            if form not in forward[abbr]:
                forward[abbr].append(form)

    # Sort by length descending (longest first) to avoid partial matches
    all_abbrs_sorted = sorted(forward.keys(), key=len, reverse=True)

    # Pre-compile regex patterns
    compiled_patterns: Dict[str, re.Pattern] = {}
    for abbr in all_abbrs_sorted:
        compiled_patterns[abbr] = re.compile(r'\b' + re.escape(abbr) + r'\b', re.IGNORECASE)

    lookup = AbbreviationLookup(
        forward=forward,
        all_abbrs_sorted=all_abbrs_sorted,
        compiled_patterns=compiled_patterns,
    )
    _cache[cache_key] = (lookup, now)
    return lookup


def find_abbrs_in_text(text: str, lookup: AbbreviationLookup) -> Dict[str, List[str]]:
    """Find all abbreviations in text. Returns {abbr: [forms]}."""
    if lookup.is_empty:
        return {}
    found: Dict[str, List[str]] = {}
    for abbr in lookup.all_abbrs_sorted:
        if lookup.compiled_patterns[abbr].search(text):
            found[abbr] = lookup.forward[abbr]
    return found


def expand_suffix(text: str, lookup: AbbreviationLookup) -> str:
    """Append [Expansions: abbr=form1 form2; ...] to text.

    Preserves the original text and adds all expanded forms as a suffix block.
    """
    if lookup.is_empty:
        return text
    found = find_abbrs_in_text(text, lookup)
    if not found:
        return text
    parts = [f"{a}={' '.join(f)}" for a, f in found.items()]
    return f"{text} [Expansions: {'; '.join(parts)}]"


def expand_query_suffix(query: str, lookup: AbbreviationLookup) -> str:
    """Bidirectional query expansion: append all forms for found abbreviations.

    Also appends the abbreviation for any full form found in the query.
    """
    if lookup.is_empty:
        return query
    found = find_abbrs_in_text(query, lookup)
    if not found:
        return query
    expansions: List[str] = []
    for abbr, forms in found.items():
        expansions.extend(forms)
    return f"{query} {' '.join(expansions)}"


def build_glossary(text: str, lookup: AbbreviationLookup) -> str:
    """Build a glossary block from abbreviations found in text.

    Format:
        CO = Commanding Officer
        bns = Battalions
        wdr = Withdraw, Withdrawal, Withdrawing
    """
    if lookup.is_empty:
        return ""
    found = find_abbrs_in_text(text, lookup)
    if not found:
        return ""
    lines = []
    for abbr in sorted(found.keys(), key=lambda x: x.lower()):
        forms = ", ".join(found[abbr])
        lines.append(f"{abbr} = {forms}")
    return "\n".join(lines)


def build_glossary_from_texts(texts: List[str], lookup: AbbreviationLookup) -> str:
    """Build a glossary from abbreviations found across multiple texts."""
    if lookup.is_empty:
        return ""
    all_found: Dict[str, List[str]] = {}
    for text in texts:
        found = find_abbrs_in_text(text, lookup)
        for abbr, forms in found.items():
            if abbr not in all_found:
                all_found[abbr] = forms
    if not all_found:
        return ""
    lines = []
    for abbr in sorted(all_found.keys(), key=lambda x: x.lower()):
        forms = ", ".join(all_found[abbr])
        lines.append(f"{abbr} = {forms}")
    return "\n".join(lines)


def parse_csv_content(content: bytes) -> List[Dict[str, str]]:
    """Parse CSV content with columns: abbreviation, expanded_form, category.

    Returns list of dicts with keys 'abbreviation', 'expanded_form', 'category'.
    Skips blank rows and rows with empty abbreviation or expanded_form.
    """
    text = content.decode("utf-8-sig")  # handle BOM
    reader = csv.DictReader(io.StringIO(text))
    rows: List[Dict[str, str]] = []
    for row in reader:
        abbr = (row.get("abbreviation") or "").strip()
        form = (row.get("expanded_form") or "").strip()
        category = (row.get("category") or "").strip() or None
        if abbr and form:
            rows.append({
                "abbreviation": abbr,
                "expanded_form": form,
                "category": category,
            })
    return rows
