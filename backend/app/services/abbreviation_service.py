"""Abbreviation expansion service — lookup, suffix expansion, glossary generation.

Caches the compiled lookup in-process for 30 seconds (same pattern as settings_service).
All expansion is deterministic (flashtext2 + CSV lookup). No LLM calls.

Case-sensitivity rules (2 tiers, derived from CSV casing):
- UPPERCASE abbrs (CO, DA, HQ, QMG): always written in capitals → exact (case-sensitive) match.
- All other abbrs (lowercase, mixed-case: bns, wdr, Comd, Dy, met): case-insensitive match.
  This includes qualification abbrs (psc, ndc) — they match case-insensitively like prose.

Single-letter abbreviations (A, D, C, F, etc.) are excluded from standalone matching.
They are used as prefixes (e.g. A/Comd = Acting Commander) and matching them standalone
causes massive false positives in prose.

flashtext2 (Rust-based) is used for O(text) matching that natively supports multi-word
keywords (e.g. "GREN RIF PRAC 94 ENERGA WITH* CART", "commanding officer" → CO).
"""
from __future__ import annotations

import csv
import io
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from flashtext2 import KeywordProcessor
from sqlalchemy.orm import Session

from app.models.abbreviation import Abbreviation, AbbreviationList
from app.services.settings_service import get_setting

logger = logging.getLogger(__name__)

_CACHE_TTL = 30  # seconds
_cache: dict[tuple[Optional[int], str], tuple["AbbreviationLookup", float]] = {}

# Minimum expanded-form length for reverse lookup (full-form → abbreviation).
# Forms shorter than this are likely common English words, not expanded forms
# that need reverse mapping.
_REVERSE_MIN_FORM_LEN = 5

# Possessive suffix pattern: CO's → CO s, officer's → officer s.
# flashtext2 uses Unicode UAX #29 word segmentation which treats apostrophe as
# a word boundary, preventing matches like "CO's" → CO. We strip the possessive
# "'s" before passing text to flashtext2. Original text is never modified —
# this only affects the string sent to extract_keywords().
_POSSESSIVE_RE = re.compile(r"'s\b")


def _is_uppercase(s: str) -> bool:
    """True if string is all uppercase and contains at least one cased char."""
    return s == s.upper() and s != s.lower()




@dataclass
class AbbreviationLookup:
    """Compiled abbreviation lookup using flashtext2 KeywordProcessors.

    Two forward processors for case-sensitivity tiers:
    - kp_exact: all-uppercase abbrs (case_sensitive=True, exact match)
    - kp_prose: all other abbrs — lowercase, mixed-case (case_sensitive=False)

    One reverse processor for full-form → abbreviation lookup (case-insensitive,
    multi-word capable).
    """
    forward: Dict[str, List[str]] = field(default_factory=dict)
    kp_exact: Optional[KeywordProcessor] = None     # uppercase only
    kp_prose: Optional[KeywordProcessor] = None     # lowercase + mixed-case
    # Reverse: lowercase expanded_form → [abbr, ...]
    reverse: Dict[str, List[str]] = field(default_factory=dict)
    kp_reverse: Optional[KeywordProcessor] = None   # reverse lookup processor

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


def _build_forward_mapping(rows) -> Dict[str, List[str]]:
    forward: Dict[str, List[str]] = {}
    for row in rows:
        abbr = row.abbreviation.strip()
        form = row.expanded_form.strip()
        if abbr and form:
            if abbr not in forward:
                forward[abbr] = []
            if form not in forward[abbr]:
                forward[abbr].append(form)
    return forward


def _classify_abbrs(forward: Dict[str, List[str]]) -> tuple[List[str], List[str]]:
    exact_abbrs: List[str] = []
    prose_abbrs: List[str] = []
    for abbr in forward:
        if len(abbr.strip()) <= 1:
            continue
        if _is_uppercase(abbr):
            exact_abbrs.append(abbr)
        else:
            prose_abbrs.append(abbr)
    return exact_abbrs, prose_abbrs


def _build_keyword_processors(
    exact_abbrs: List[str], prose_abbrs: List[str]
) -> tuple[KeywordProcessor, KeywordProcessor]:
    kp_exact = KeywordProcessor(case_sensitive=True)
    for abbr in exact_abbrs:
        kp_exact.add_keyword(abbr, abbr)
    kp_prose = KeywordProcessor(case_sensitive=False)
    for abbr in prose_abbrs:
        kp_prose.add_keyword(abbr, abbr)
    return kp_exact, kp_prose


def _build_reverse_mapping(forward: Dict[str, List[str]]) -> tuple[Dict[str, List[str]], KeywordProcessor]:
    reverse: Dict[str, List[str]] = {}
    for abbr, forms in forward.items():
        if len(abbr.strip()) <= 1:
            continue
        for form in forms:
            key = form.lower()
            if len(key) < _REVERSE_MIN_FORM_LEN:
                continue
            if key not in reverse:
                reverse[key] = []
            if abbr not in reverse[key]:
                reverse[key].append(abbr)
    kp_reverse = KeywordProcessor(case_sensitive=False)
    for form_lower in reverse:
        kp_reverse.add_keyword(form_lower, form_lower)
    return reverse, kp_reverse


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

    rows = (
        db.query(Abbreviation)
        .filter(Abbreviation.list_id.in_(list_ids))
        .all()
    )
    forward = _build_forward_mapping(rows)
    exact_abbrs, prose_abbrs = _classify_abbrs(forward)
    kp_exact, kp_prose = _build_keyword_processors(exact_abbrs, prose_abbrs)
    reverse, kp_reverse = _build_reverse_mapping(forward)

    lookup = AbbreviationLookup(
        forward=forward,
        kp_exact=kp_exact,
        kp_prose=kp_prose,
        reverse=reverse,
        kp_reverse=kp_reverse,
    )
    _cache[cache_key] = (lookup, now)
    return lookup


def find_abbrs_in_text(text: str, lookup: AbbreviationLookup) -> Dict[str, List[str]]:
    """Find all abbreviations in text using flashtext2. Returns {abbr: [forms]}.

    Matching rules (2 tiers based on CSV casing):
    - UPPERCASE abbrs (CO, DA, HQ, QMG): exact case match only.
    - All other abbrs (lowercase, mixed-case: bns, wdr, Comd, Dy, met): case-insensitive.
    - Single-letter abbrs (A, D, C) are never matched standalone.
    """
    if lookup.is_empty:
        return {}
    found: Dict[str, List[str]] = {}
    # Normalize possessive "'s" so flashtext2 can match the base abbreviation.
    match_text = _POSSESSIVE_RE.sub(" ", text)
    # Tier 1: exact match (uppercase only)
    if lookup.kp_exact:
        for abbr in lookup.kp_exact.extract_keywords(match_text):
            if abbr in lookup.forward and abbr not in found:
                found[abbr] = lookup.forward[abbr]
    # Tier 2: prose match (lowercase + mixed-case, case-insensitive)
    if lookup.kp_prose:
        for abbr in lookup.kp_prose.extract_keywords(match_text):
            if abbr in lookup.forward and abbr not in found:
                found[abbr] = lookup.forward[abbr]
    return found


def find_forms_in_text(text: str, lookup: AbbreviationLookup) -> Dict[str, List[str]]:
    """Find all expanded forms in text via reverse lookup. Returns {abbr: [forms]}.

    If text contains "battalions", returns {"bns": ["Battalions", ...]}.
    If text contains "commanding officer", returns {"CO": ["Commanding Officer"]}.

    Uses flashtext2 for multi-word form matching (e.g. "commanding officer" → CO).
    Only forms ≥ _REVERSE_MIN_FORM_LEN chars are indexed. Stopwords are excluded.
    """
    if lookup.is_empty or not lookup.reverse or not lookup.kp_reverse:
        return {}
    found_abbrs: set[str] = set()
    # Normalize possessive "'s" so flashtext2 can match the base form.
    match_text = _POSSESSIVE_RE.sub(" ", text)
    for form_lower in lookup.kp_reverse.extract_keywords(match_text):
        abbrs = lookup.reverse.get(form_lower)
        if abbrs:
            found_abbrs.update(abbrs)
    if not found_abbrs:
        return {}
    return {abbr: lookup.forward[abbr] for abbr in found_abbrs}


def expand_suffix(text: str, lookup: AbbreviationLookup) -> str:
    """Append an [Abbreviation Glossary] block to text.

    Preserves the original text and adds all expanded forms as a suffix block.
    Forward-only (used during ingestion).

    Format:
        {original text}

        [Abbreviation Glossary]
        CO = Commanding Officer
        bns = Battalions
    """
    if lookup.is_empty:
        return text
    found = find_abbrs_in_text(text, lookup)
    if not found:
        return text
    lines = []
    for abbr in sorted(found.keys(), key=lambda x: x.lower()):
        forms = ", ".join(found[abbr])
        lines.append(f"{abbr} = {forms}")
    return f"{text}\n\n[Abbreviation Glossary]\n" + "\n".join(lines)


def expand_query_suffix(query: str, lookup: AbbreviationLookup) -> str:
    """Bidirectional query expansion in glossary suffix format.

    Finds abbreviations in the query (forward) AND full forms (reverse),
    then appends an [Abbreviation Glossary] block.

    "bns wdr"             → "bns wdr\n\n[Abbreviation Glossary]\nbns = Battalions\nwdr = Withdraw, ..."
    "battalions withdrew" → "battalions withdrew\n\n[Abbreviation Glossary]\nbns = Battalions\nwdr = Withdraw, ..."
    "commanding officer"  → "commanding officer\n\n[Abbreviation Glossary]\nCO = Commanding Officer"
    """
    if lookup.is_empty:
        return query
    found_abbrs = find_abbrs_in_text(query, lookup)
    found_forms = find_forms_in_text(query, lookup)
    merged = dict(found_abbrs)
    for abbr, forms in found_forms.items():
        if abbr not in merged:
            merged[abbr] = forms
    if not merged:
        return query
    lines = []
    for abbr in sorted(merged.keys(), key=lambda x: x.lower()):
        forms = ", ".join(merged[abbr])
        lines.append(f"{abbr} = {forms}")
    return f"{query}\n\n[Abbreviation Glossary]\n" + "\n".join(lines)


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
