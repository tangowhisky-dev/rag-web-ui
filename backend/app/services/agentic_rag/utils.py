"""Helper functions for the LangGraph agent graph."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, List

from app.services.infrastructure.reasoning_tags import (
    build_strip_patterns as _build_reasoning_patterns,
    strip_reasoning_tags,
)

logger = logging.getLogger(__name__)


def _format_doc_parts(pruned_docs: list[dict], file_markdown: str | None) -> list[str]:
    parts: list[str] = []
    for i, doc in enumerate(pruned_docs, 1):
        # Use original_text from metadata if available (clean prose for generation)
        metadata = doc.get("metadata", {})
        content = metadata.get("original_text", doc.get("page_content", "")).strip()
        source = metadata.get("source", "")
        title = metadata.get("title", "")
        citation_ref = metadata.get("citation_ref") or {}
        # When citation_ref is present, use evidence-based [E1] labeling.
        # Otherwise fall back to legacy [KB-N] labeling for backward compat.
        if citation_ref:
            citation_id = citation_ref.get("citation_id") or f"E{i}"
            kind = citation_ref.get("citation_kind", "chunk")
            header_parts = [f'document="{title or metadata.get("file_name", "")}"',
                            f"kind={kind}"]
            if citation_ref.get("chunk_index") is not None:
                header_parts.append(f"chunk={citation_ref['chunk_index']}")
            if citation_ref.get("page") is not None:
                header_parts.append(f"page={citation_ref['page']}")
            if citation_ref.get("section"):
                header_parts.append(f"section={citation_ref['section']}")
            if citation_ref.get("start_line") is not None and citation_ref.get("end_line") is not None:
                header_parts.append(f"lines={citation_ref['start_line']}-{citation_ref['end_line']}")
            if citation_ref.get("match_line") is not None:
                header_parts.append(f"line={citation_ref['match_line']}")
            if citation_ref.get("source_tool"):
                header_parts.append(f"source={citation_ref['source_tool']}")
            header = f"[{citation_id}] " + ", ".join(header_parts)
            parts.append(f'{header}\n     "{content}"')
        else:
            # Legacy [KB-N] labeling (no citation_ref metadata)
            if title and title != source:
                header = f"[KB-{i}] {title} ({source})"
            else:
                header = f"[KB-{i}]" + (f" ({source})" if source else "")
            parts.append(f"{header}\n{content}")
    if file_markdown:
        parts.append(f"[File Content]\n{file_markdown}")
    return parts


def _build_chunk_glossary(db: Any, org_id: Any, pruned_docs: list[dict]) -> str:
    if db is None:
        return ""
    from app.services.abbreviation_service import build_lookup, build_glossary_from_texts
    abbr_lookup = build_lookup(db, org_id)
    if not abbr_lookup or abbr_lookup.is_empty:
        return ""
    texts = []
    for doc in (pruned_docs or []):
        md = doc.get("metadata", {})
        texts.append(md.get("original_text", doc.get("page_content", "")))
    return build_glossary_from_texts(texts, abbr_lookup)


def _merge_glossaries(*glossaries: str) -> str:
    merged_lines = []
    seen_abbrs = set()
    for g in glossaries:
        if not g:
            continue
        for line in g.split("\n"):
            abbr = line.split("=", 1)[0].strip() if "=" in line else line.strip()
            if abbr and abbr not in seen_abbrs:
                seen_abbrs.add(abbr)
                merged_lines.append(line)
    if merged_lines:
        return f"[Abbreviation Glossary]\n" + "\n".join(merged_lines)
    return ""


def _build_glossary_section(db: Any, org_id: Any, query_glossary: str, pruned_docs: list[dict]) -> str:
    if db is None and not query_glossary:
        return ""
    try:
        chunk_glossary = _build_chunk_glossary(db, org_id, pruned_docs)
        return _merge_glossaries(query_glossary, chunk_glossary)
    except Exception:
        if query_glossary:
            return f"[Abbreviation Glossary]\n{query_glossary}"
    return ""


def format_context_string(
    docs: list[dict],
    file_markdown: str | None = None,
    db: Any = None,
    org_id: Any = None,
    query_glossary: str = "",
) -> str:
    """Format a list of serialized documents into a context string for the LLM.

    Each doc becomes ``[KB-N] (source)\\ncontent``.  If *file_markdown* is
    provided it is appended after a ``[File Content]`` header.

    Uses ``original_text`` from doc metadata when available (for clean prose
    in generation). Appends a scoped abbreviation glossary when abbreviation
    expansion is enabled. The glossary merges *query_glossary* (pre-built by
    the abbreviation service) with any additional abbreviations found in the chunk
    texts.

    Contiguous chunks from the same document have their overlap pruned
    so the LLM doesn't see duplicated text (300 chars per adjacent pair
    at 20% overlap). Citation indices are unaffected — pruning only
    shortens the content, not the chunk's position in the list.
    """
    from app.services.agentic_rag.agent_graph import _prune_contiguous_overlaps

    pruned_docs = _prune_contiguous_overlaps(docs) if docs else docs
    parts = _format_doc_parts(pruned_docs, file_markdown)
    glossary = _build_glossary_section(db, org_id, query_glossary, pruned_docs)
    if glossary:
        parts.append(glossary)
    return "\n\n---\n\n".join(parts)


def group_docs_by_document(docs: list[dict]) -> list[dict]:
    """Reorder docs so chunks from the same document are contiguous.

    Groups are ordered by their highest-scoring chunk (descending
    ``_reranker_score``). Within a group, chunks are ordered by
    ``chunk_index`` ascending so the LLM reads them in document order.

    Chunks without a ``document_id`` (e.g. graph-expanded docs with
    missing metadata) are kept at the end in their original relative order.

    This reordering affects citation indices — the caller must use the
    returned list for both ``format_context_string`` and
    ``normalize_citations`` so [KB-N] markers map correctly.
    """
    if not docs:
        return docs

    # Partition into grouped (has document_id) and ungrouped.
    grouped: dict[Any, list[dict]] = {}
    ungrouped: list[dict] = []
    for doc in docs:
        meta = doc.get("metadata", {}) if isinstance(doc, dict) else {}
        doc_id = meta.get("document_id")
        if doc_id is None:
            ungrouped.append(doc)
        else:
            grouped.setdefault(doc_id, []).append(doc)

    # Order groups by their best chunk's reranker score (descending).
    def _best_score(group: list[dict]) -> float:
        return max(
            (d.get("_reranker_score", d.get("score", 0.0)) or 0.0)
            for d in group
        )

    ordered_groups = sorted(grouped.values(), key=_best_score, reverse=True)

    # Within each group, sort by chunk_index ascending.
    result: list[dict] = []
    for group in ordered_groups:
        group.sort(key=lambda d: d.get("metadata", {}).get("chunk_index", 0))
        result.extend(group)
    result.extend(ungrouped)
    return result


def normalize_evidence_citations(answer: str, evidence: list[dict]) -> tuple[str, list[dict]]:
    """Validate, deduplicate, and renumber citations in an LLM answer.

    Handles two citation formats:
    - [E1], [E2] — evidence label format
    - [N](N) — markdown link format where N is the numeric portion of E-N

    Returns (rewritten_answer, cited_evidence_in_display_order).
    Each item in cited_evidence is the evidence dict (with citation_ref in metadata).
    """
    if not answer:
        return answer or "", []
    if not evidence:
        # Strip both [E1] and [N](N) citation formats
        cleaned = re.sub(r"\[E\d+\]", "", answer, flags=re.IGNORECASE)
        cleaned = re.sub(r"\[\d+\]\(\d+\)", "", cleaned)
        return cleaned.strip(), []

    max_e = len(evidence)

    # Split out code blocks
    _code_segments: list[str] = []
    def _extract_code(m: re.Match) -> str:
        _code_segments.append(m.group(0))
        return f"\x00CODE{len(_code_segments) - 1}\x00"
    answer = re.sub(r"```[\s\S]*?```", _extract_code, answer)
    answer = re.sub(r"`[^`]*`", _extract_code, answer)

    # Split out reasoning sections
    _reasoning_segments: list[str] = []
    def _extract_reasoning(m: re.Match) -> str:
        _reasoning_segments.append(m.group(0))
        return f"\x00REASONING{len(_reasoning_segments) - 1}\x00"
    _full_patterns, _ = _build_reasoning_patterns()
    for pat in _full_patterns:
        answer = pat.sub(_extract_reasoning, answer)

    # Collect unique E-numbers in first-appearance order.
    # Match both [E1] and [1](1) formats (N refers to the E-N label).
    valid_cited: list[int] = []
    seen: set[int] = set()
    # First pass: [E1] format
    for match in re.finditer(r"\[E(\d+)\]", answer, re.IGNORECASE):
        n = int(match.group(1))
        if 1 <= n <= max_e and n not in seen:
            valid_cited.append(n)
            seen.add(n)
    # Second pass: [N](N) format (only if not already seen as [E1])
    for match in re.finditer(r"\[(\d+)\]\(\d+\)", answer):
        n = int(match.group(1))
        if 1 <= n <= max_e and n not in seen:
            valid_cited.append(n)
            seen.add(n)

    # Renumber: first cited → [1], second → [2], etc.
    index_map = {orig: new for new, orig in enumerate(valid_cited, start=1)}

    def _replace_marker(match: re.Match) -> str:
        n = int(match.group(1))
        if n in index_map:
            return f"[{index_map[n]}]"
        return ""
    # Replace [E1] format
    normalized = re.sub(r"\[E(\d+)\]", _replace_marker, answer, flags=re.IGNORECASE)
    # Replace [N](N) format → [M](M) with renumbered M
    def _replace_link(match: re.Match) -> str:
        n = int(match.group(1))
        if n in index_map:
            new_n = index_map[n]
            return f"[{new_n}]({new_n})"
        return ""
    normalized = re.sub(r"\[(\d+)\]\(\d+\)", _replace_link, normalized)

    # Restore code blocks
    normalized = re.sub(r"\x00CODE(\d+)\x00", lambda m: _code_segments[int(m.group(1))], normalized)

    # Restore reasoning sections with citations stripped
    def _strip_reasoning_citations(text: str) -> str:
        text = re.sub(r"\[E\d+\]", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\[\d+\]\(\d+\)", "", text)
        return text
    normalized = re.sub(
        r"\x00REASONING(\d+)\x00",
        lambda m: _strip_reasoning_citations(_reasoning_segments[int(m.group(1))]),
        normalized,
    )

    cited_evidence = [evidence[i - 1] for i in valid_cited]
    return normalized, cited_evidence


def normalize_citations(answer: str, docs: list) -> tuple[str, list[int]]:
    """Validate, deduplicate, and renumber inline citations in an LLM answer.

    - Parses [N](N) markdown citation links.
    - Normalizes [citation](N) and [citation](N)(N) variants to [N](N).
    - Removes any citation whose index is outside the provided docs range.
    - Renumbers remaining citations 1..M by first appearance in the answer.
    - Skips reasoning/thinking sections when collecting cited indices so
      that citations in the reasoning (which often reference every chunk)
      don't dilute the answer's renumbering.
    - Returns the rewritten answer and the list of original 1-based doc indices
      in display order.
    """
    if not answer:
        return answer or "", []

    # Strip any existing citation markers when no docs are available.
    if not docs:
        cleaned = re.sub(r"\[citation\]\(\d+\)\(\d+\)", "", answer)
        cleaned = re.sub(r"\[citation\]\(\d+\)", "", cleaned)
        cleaned = re.sub(r"\[(?:KB-)?\d+\]\((?:KB-)?\d+\)", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip(), []

    # Normalize common malformed variants emitted by some models to [N](N).
    answer = re.sub(r"\[citation\]\((\d+)\)\((\d+)\)", r"[\1](\1)", answer)
    answer = re.sub(r"\[citation\]\((\d+)\)", r"[\1](\1)", answer)
    # Some models cite using the full "KB-N" label instead of the bare
    # numeral (e.g. [KB-2](KB-2)) despite being instructed otherwise —
    # strip the prefix on either side so it's treated as a normal [N](N).
    answer = re.sub(r"\[KB-(\d+)\]\(KB-(\d+)\)", r"[\1](\2)", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\[KB-(\d+)\]\((\d+)\)", r"[\1](\2)", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\[(\d+)\]\(KB-(\d+)\)", r"[\1](\2)", answer, flags=re.IGNORECASE)
    # Bare [KB-N] (no parenthetical) and combined [KB-N, KB-M] — the model
    # copied the context label verbatim.  The "KB-" prefix is unambiguous
    # (never appears in prose/code/URLs), so this is safe.  Negative
    # lookahead (?!\() prevents double-matching the parenthetical variants
    # already handled above.
    answer = re.sub(
        r"\[(KB-\d+(?:[,\s]+KB-\d+)*)\](?!\()",
        lambda m: "".join(
            f"[{int(x)}]({int(x)})"
            for x in re.findall(r"KB-(\d+)", m.group(1), flags=re.IGNORECASE)
        ),
        answer,
        flags=re.IGNORECASE,
    )
    # Bare [N] without KB- prefix or parenthetical — the model used a
    # shorthand citation.  Convert to [N](N) only when N is a valid doc
    # index (1..max_index) so we don't touch [0], array indices in code,
    # or unrelated numbers in prose.  Code blocks are protected by
    # splitting them out before processing and restoring afterwards.
    _max = len(docs)
    if _max > 0:
        # Split out fenced code blocks so bare [N] inside them is untouched.
        _code_segments: list[str] = []
        def _extract_code(m: re.Match) -> str:
            _code_segments.append(m.group(0))
            return f"\x00CODE{len(_code_segments) - 1}\x00"
        answer = re.sub(r"```[\s\S]*?```", _extract_code, answer)
        answer = re.sub(r"`[^`]*`", _extract_code, answer)

        answer = re.sub(
            r"\[(\d{1,3})\](?!\()",
            lambda m: f"[{m.group(1)}]({m.group(1)})" if 1 <= int(m.group(1)) <= _max else m.group(0),
            answer,
        )

        # Restore code blocks.
        answer = re.sub(
            r"\x00CODE(\d+)\x00",
            lambda m: _code_segments[int(m.group(1))],
            answer,
        )

    # Split out reasoning sections so citations inside them don't affect
    # the answer's renumbering.  Uses the shared pattern definitions from
    # reasoning_tags.py — the single source of truth for tag formats.
    # Preserves original tags (think, reasoning, channel) in the output.
    _reasoning_segments: list[str] = []
    def _extract_reasoning_block(m: re.Match) -> str:
        _reasoning_segments.append(m.group(0))
        return f"\x00REASONING{len(_reasoning_segments) - 1}\x00"
    _full_patterns, _ = _build_reasoning_patterns()
    for pat in _full_patterns:
        answer = pat.sub(_extract_reasoning_block, answer)

    max_index = len(docs)
    valid_cited: list[int] = []
    seen: set[int] = set()

    # Collect unique valid original indices in first-appearance order
    # from the answer section only (reasoning sections are extracted).
    for match in re.finditer(r"\[(\d+)\]\((\d+)\)", answer):
        n = int(match.group(1))
        # Guard against mismatched brackets like [1](2) — require both numbers equal.
        if n != int(match.group(2)):
            continue
        if 1 <= n <= max_index and n not in seen:
            valid_cited.append(n)
            seen.add(n)

    index_map = {orig: new for new, orig in enumerate(valid_cited, start=1)}

    def _replace_marker(match: re.Match) -> str:
        n = int(match.group(1))
        m = int(match.group(2))
        # Only rewrite well-formed [N](N); strip malformed or out-of-range markers.
        if n == m and n in index_map:
            new_idx = index_map[n]
            return f"[{new_idx}]({new_idx})"
        return ""

    # Renumber citations in the answer section.
    normalized = re.sub(r"\[(\d+)\]\((\d+)\)", _replace_marker, answer)

    # Strip citations from reasoning sections — they are internal reasoning
    # displayed in a collapsible UI section, not clickable citations.
    def _strip_reasoning_citations(text: str) -> str:
        text = re.sub(r"\[(?:KB-)?\d+\]\((?:KB-)?\d+\)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\[citation\]\(\d+\)\(\d+\)", "", text)
        text = re.sub(r"\[citation\]\(\d+\)", "", text)
        return text

    # Restore reasoning sections with citations stripped, preserving
    # the original tag format.
    normalized = re.sub(
        r"\x00REASONING(\d+)\x00",
        lambda m: _strip_reasoning_citations(_reasoning_segments[int(m.group(1))]),
        normalized,
    )
    return normalized, valid_cited

