"""Helper functions for the LangGraph agent graph."""

from __future__ import annotations

import json
import re
from typing import Any, List

from app.services.agentic_rag.prompts import REWRITE_SYSTEM_PROMPT


def estimate_context_tokens(text: str) -> int:
    """Rough token estimation from character count.
    
    Uses ~1 token per 4 chars for English text.
    """
    return int(len(text) * 0.25)


def estimate_messages_tokens(messages: list) -> int:
    """Estimate total tokens for a list of messages."""
    total = 0
    for msg in messages:
        content = getattr(msg, "content", str(msg)) if hasattr(msg, "content") else msg.get("content", "")
        total += estimate_context_tokens(str(content))
    return total


def strip_reasoning_tags(text: str) -> str:
    """Strip <think>...</think> tags from text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _format_doc_parts(pruned_docs: list[dict], file_markdown: str | None) -> list[str]:
    parts: list[str] = []
    for i, doc in enumerate(pruned_docs, 1):
        # Use original_text from metadata if available (clean prose for generation)
        metadata = doc.get("metadata", {})
        content = metadata.get("original_text", doc.get("page_content", "")).strip()
        source = metadata.get("source", "")
        title = metadata.get("title", "")
        # Header shows title as primary identifier, filename in parentheses.
        # Falls back to filename only when no title is available.
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
    expand_query_node) with any additional abbreviations found in the chunk
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


def normalize_citations(answer: str, docs: list) -> tuple[str, list[int]]:
    """Validate, deduplicate, and renumber inline citations in an LLM answer.

    - Parses [N](N) markdown citation links.
    - Normalizes [citation](N) and [citation](N)(N) variants to [N](N).
    - Removes any citation whose index is outside the provided docs range.
    - Renumbers remaining citations 1..M by first appearance in the answer.
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
    # already handled above.  Bare [N] without "KB-" is intentionally NOT
    # handled — it's too common in code, URLs, and prose.
    answer = re.sub(
        r"\[(KB-\d+(?:[,\s]+KB-\d+)*)\](?!\()",
        lambda m: "".join(
            f"[{int(x)}]({int(x)})"
            for x in re.findall(r"KB-(\d+)", m.group(1), flags=re.IGNORECASE)
        ),
        answer,
        flags=re.IGNORECASE,
    )

    max_index = len(docs)
    valid_cited: list[int] = []
    seen: set[int] = set()

    # Collect unique valid original indices in first-appearance order.
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

    normalized = re.sub(r"\[(\d+)\]\((\d+)\)", _replace_marker, answer)
    return normalized, valid_cited


# Markers that indicate the message may depend on earlier turns. Only these
# trigger the resolver LLM call; everything else passes through unchanged.
_REFERENCE_MARKERS = re.compile(
    r"\b("
    r"it|its|it's|they|them|their|this|that|these|those|there|"
    r"he|she|his|her|him|"
    r"one|ones|former|latter|above|previous|prior|earlier|"
    r"first|second|third|fourth|fifth|last|next|"
    r"same|other|others|another|else|"
    r"instead|also|too|again|more|further|"
    r"yours?|you\s+said|you\s+mentioned|mentioned"
    r")\b",
    re.IGNORECASE,
)

# Elliptical fragments ("what about X?", "and Y?", "why?") are context-dependent
# even without an explicit anaphor.
_ELLIPSIS_MARKERS = re.compile(
    r"^\s*(and|but|or|so|what about|how about|why|why not|what if|ok|okay|yes|no)\b",
    re.IGNORECASE,
)

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "in", "is", "it", "its", "of",
    "on", "or", "that", "the", "their", "them", "there", "these", "they", "this",
    "those", "to", "was", "were", "what", "when", "where", "which", "who", "why",
    "will", "with", "you", "your", "about", "into", "than", "then", "explain",
    "describe", "tell", "me", "give", "list", "show", "summarise", "summarize",
}


def _content_tokens(text: str) -> set[str]:
    """Lowercase alphanumeric tokens with stopwords and short tokens removed."""
    return {
        t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(t) > 2 and t not in _STOPWORDS
    }


def needs_reference_resolution(query: str, has_history: bool) -> bool:
    """True if *query* plausibly refers to something from an earlier turn."""
    if not has_history:
        return False
    if not query or not query.strip():
        return False
    return bool(_REFERENCE_MARKERS.search(query) or _ELLIPSIS_MARKERS.match(query))


def validate_resolution_provenance(
    original_query: str,
    rewritten: str,
    provenance_sources: list[str],
) -> tuple[bool, list[str]]:
    """Check that every term introduced by the rewrite is traceable.

    Resolving "it" legitimately *adds* an entity, so a "no new words" rule is
    wrong. The correct invariant is provenance: any content token present in
    the rewrite but absent from the original query must appear in one of the
    supplied sources (recent turns, compaction summary, previous answer,
    clarification text).

    Returns (ok, unsupported_tokens).
    """
    added = _content_tokens(rewritten) - _content_tokens(original_query)
    if not added:
        return True, []
    supported = set()
    for source in provenance_sources:
        supported |= _content_tokens(source)
    unsupported = sorted(added - supported)
    return not unsupported, unsupported


async def resolve_retrieval_query(
    query: str,
    original_query: str,
    recent_history: list,
    provenance_sources: list[str] | None = None,
    api_base: str | None = None,
    query_model: str | None = None,
    openai_api_key: str = "",
    openai_api_base: str = "",
    glossary: str = "",
    retrieved_titles: list[str] | None = None,
) -> tuple[str, dict]:
    """Resolve *query* into a standalone retrieval string.

    Returns ``(retrieval_query, provenance)``. ``provenance`` records whether
    resolution ran and, when it did, which tokens it introduced — so a bad
    rewrite is auditable rather than silently authoritative.

    Falls back to *query* on skip, timeout, LLM error, or failed provenance
    validation. The retrieval query is never allowed to become free text the
    pipeline cannot account for.
    """
    provenance_sources = provenance_sources or []
    has_history = bool(recent_history)

    if not needs_reference_resolution(query, has_history):
        return query, {"resolved": False, "reason": "self_contained"}

    try:
        raw_rewrite = await _call_rewriter(
            query=query,
            recent_history=recent_history,
            api_base=api_base,
            query_model=query_model,
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base,
            glossary=glossary,
            retrieved_titles=retrieved_titles,
        )
    except Exception as exc:  # network, timeout, provider error
        return query, {"resolved": False, "reason": f"resolver_failed: {exc}"}

    standalone = _clean_rewrite(raw_rewrite) or query
    if standalone == query:
        return query, {"resolved": False, "reason": "unchanged"}

    ok, unsupported = validate_resolution_provenance(
        original_query=query,
        rewritten=standalone,
        provenance_sources=provenance_sources,
    )
    if not ok:
        return query, {
            "resolved": False,
            "reason": "provenance_rejected",
            "unsupported_terms": unsupported,
            "rejected_query": standalone,
        }

    return standalone, {
        "resolved": True,
        "reason": "reference_resolved",
        "original_query": original_query,
    }


def _clean_rewrite(raw_rewrite: str) -> str:
    """Strip reasoning tags, meta-commentary preambles, and answer echoes."""
    standalone = strip_reasoning_tags(raw_rewrite).strip()
    if not standalone:
        return ""

    # Strip a meta-commentary preamble ("Rewritten standalone query: ...").
    # Only split on a colon that is part of such a preamble prefix, so a
    # legitimate query containing a colon is left intact.
    meta_prefix = re.match(
        r"^[^:\n]{0,80}?\b(rewritten|standalone|search)\s+(query|question)?\s*:\s*",
        standalone,
        re.IGNORECASE,
    )
    if meta_prefix:
        candidate = standalone[meta_prefix.end():].strip()
        if len(candidate) > 5:
            standalone = candidate

    # Guard: the rewriter echoed an answer instead of rewriting.
    answer_patterns = [
        r"\bthere\s+is\s+no\s+information\b",
        r"\bthe\s+context\s+does?\s+not\s+contain\b",
        r"\bi\s+cannot\s+answer\b",
        r"\bi\s+don't\s+have\s+enough\b",
        r"\bno\s+information\s+found\b",
    ]
    if any(re.search(p, standalone, re.IGNORECASE) for p in answer_patterns):
        return ""

    return standalone.strip().strip('"')


async def _call_rewriter(
    query: str,
    recent_history: list,
    api_base: str | None,
    query_model: str | None,
    openai_api_key: str,
    openai_api_base: str,
    glossary: str = "",
    retrieved_titles: list[str] | None = None,
) -> str:
    """Single rewriter LLM call. Raises on provider failure."""
    system_msg = REWRITE_SYSTEM_PROMPT.format(memory_section="")

    messages: list[dict] = [{"role": "system", "content": system_msg}]
    from langchain_core.messages import HumanMessage, AIMessage

    for m in recent_history:
        if isinstance(m, HumanMessage):
            messages.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            messages.append({"role": "assistant", "content": m.content})
    user_content = query
    if glossary:
        user_content += f"\n\n[Abbreviation Glossary]\n{glossary}"
    if retrieved_titles:
        titles_text = "\n".join(f"- {t}" for t in retrieved_titles)
        user_content += f"\n\n[Retrieved Document Titles]\n{titles_text}"
    messages.append({"role": "user", "content": user_content})

    from openai import AsyncOpenAI as _AsyncOAI
    client = _AsyncOAI(api_key=openai_api_key, base_url=api_base or openai_api_base)
    resp = await client.chat.completions.create(
        model=query_model or "default",
        # 60 tokens truncated rewrites mid-phrase; the prompt caps output at
        # 30 words, so 160 leaves headroom without inviting an essay.
        max_tokens=160,
        messages=messages,
        temperature=0,
        stream=False,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return (resp.choices[0].message.content or "").strip()

