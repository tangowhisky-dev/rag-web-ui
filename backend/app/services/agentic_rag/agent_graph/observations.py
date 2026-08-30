"""Observation formatting and text helpers.

Formats tool observations for LLM context in three modes:
- Full: complete page_content of all docs (deduplicated by content_hash).
- Compact: doc count, confidence, top doc preview.
- Metadata-only: rag_retrieve gets metadata only; non-retrieval tools get
  full results.

Also contains overlap pruning for contiguous chunks from the same document,
and the deterministic stage-1 observation compaction (shrink tool outputs
in-place without an LLM call).
"""

from __future__ import annotations

import json
from typing import Any

from app.core.settings_registry import get_def
from app.services.agentic_rag.schemas import Observation

from .helpers import _coerce_observation


# How many docs to keep per rag_retrieve observation when compacting.
_COMPACT_KEEP_DOCS = 5
# How many stdout lines to keep for code_execute when compacting.
_COMPACT_KEEP_STDOUT_LINES = 20


def _tool_descriptions_text(tools: list) -> str:
    lines = []
    for t in tools:
        lines.append(f"- {t.name}: {t.description}")
        # Include the args schema so the LLM knows the exact field names and
        # types. Essential for json_text mode where bind_tools is not called;
        # harmless in native mode (the schema is redundant but consistent).
        schema = t.args_schema.model_json_schema()
        props = schema.get("properties", {})
        required = schema.get("required", [])
        field_lines = []
        for fname, finfo in props.items():
            ftype = finfo.get("type", "any")
            desc = finfo.get("description", "")
            req = " (required)" if fname in required else ""
            field_lines.append(f"    {fname}: {ftype}{req} — {desc}")
        if field_lines:
            lines.append("  args:")
            lines.extend(field_lines)
    return "\n".join(lines)


def _strip_overlap(prev: str, curr: str, max_search: int) -> str:
    """Strip the overlapping prefix from *curr* that duplicates the tail of *prev*.

    Searches for the longest suffix of *prev* (up to *max_search* chars) that
    appears as a prefix of *curr* and strips it. Returns *curr* unchanged if
    no overlap is found.
    """
    search_len = min(len(prev), len(curr), max_search)
    for length in range(search_len, 0, -1):
        if prev[-length:] == curr[:length]:
            return curr[length:]
    return curr


def _prune_contiguous_overlaps(docs: list[dict]) -> list[dict]:
    """Prune overlap text from contiguous chunks.

    Chunks are created with OVERLAP_PERCENTAGE (default 20% = 300 chars at
    CHUNK_SIZE=1500). When two adjacent chunks from the same document are
    both retrieved, the overlapping region appears twice. This function:

    1. Groups docs by document_id.
    2. Sorts by chunk_index within each group.
    3. For contiguous chunks (chunk_index differs by 1), strips the overlap
       from the later chunk using _strip_overlap.
    4. Non-contiguous chunks are left unchanged.

    Returns a new list with pruned page_content. Original metadata (chunk_index,
    document_id, content_hash) is preserved for citations — pruning only
    affects the text shown to the LLM, not the citation mapping.
    """
    if not docs:
        return docs

    chunk_size = get_def("CHUNK_SIZE").default
    overlap_pct = get_def("OVERLAP_PERCENTAGE").default
    max_overlap = max(200, int(chunk_size * overlap_pct) * 2)

    # Group by document_id, sort by chunk_index within each group.
    by_doc: dict[Any, list[dict]] = {}
    for doc in docs:
        meta = doc.get("metadata", {}) if isinstance(doc, dict) else {}
        doc_id = meta.get("document_id")
        if doc_id is None:
            continue
        by_doc.setdefault(doc_id, []).append(doc)

    # Build a set of (doc_id, chunk_index) → pruned content.
    pruned_content: dict[int, str] = {}  # id() of doc → pruned text
    for doc_id, group in by_doc.items():
        group.sort(key=lambda d: d.get("metadata", {}).get("chunk_index", 0))
        prev_text = None
        prev_idx = None
        for doc in group:
            chunk_idx = doc.get("metadata", {}).get("chunk_index", 0)
            content = doc.get("page_content", "")
            # Position must come from enumeration order: list.index() resolves
            # by dict equality and returns the wrong neighbour when two chunk
            # dicts compare equal.
            if prev_text is not None and prev_idx is not None and prev_idx + 1 == chunk_idx:
                pruned = _strip_overlap(prev_text, content, max_overlap)
                pruned_content[id(doc)] = pruned
                prev_text = pruned
            else:
                prev_text = content
            prev_idx = chunk_idx

    if not pruned_content:
        return docs

    # Build result with pruned content where applicable.
    result = []
    for doc in docs:
        if id(doc) in pruned_content:
            doc_copy = dict(doc)
            doc_copy["page_content"] = pruned_content[id(doc)]
            result.append(doc_copy)
        else:
            result.append(doc)
    return result


def _format_retrieval_obs_full(docs, doc_count, confidence, sufficient_text, seen_hashes):
    from app.services.infrastructure import content_hash as _ch

    unique_docs = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        h = doc.get("metadata", {}).get("content_hash") or _ch(doc.get("page_content", ""))
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_docs.append(doc)
    parts = [f"  doc_count={doc_count} unique_so_far={len(seen_hashes)} confidence={confidence}{sufficient_text}"]
    pruned_docs = _prune_contiguous_overlaps(unique_docs)
    for j, doc in enumerate(pruned_docs, 1):
        content = str(doc.get("page_content", ""))
        parts.append(f"  doc_{j}: {content}")
    return parts


def _format_retrieval_obs_compact(docs, doc_count, confidence, sufficient_text):
    parts = [f"  doc_count={doc_count} confidence={confidence}{sufficient_text}"]
    if docs and isinstance(docs[0], dict):
        preview = str(docs[0].get("page_content", ""))[:300]
        parts.append(f"  top_doc_preview: {preview}")
    return parts


def _observations_text(observations: list[Observation], full: bool = False) -> str:
    """Format observations for LLM context.

    When full=True, include the complete page_content of all docs per
    observation (deduplicated across observations by content_hash) so
    think_node can judge whether the retrieval actually answers the query.
    Chunks are 1500 chars (CHUNK_SIZE). With dedup, the worst case
    (3 rag_retrieve calls returning the same 29 docs) is 29 unique docs
    = ~43k chars = ~10.9k tokens — well within budget. The
    _compact_if_needed helper handles overflow if unique docs accumulate
    across many iterations with different queries.

    When full=False, include a compact summary (doc count, confidence, top
    doc preview) to keep reflect/finalize prompts small.
    """
    parts = []
    seen_hashes: set[str] = set()
    for i, raw_obs in enumerate(observations, 1):
        obs = _coerce_observation(raw_obs)
        parts.append(f"Observation {i}: tool={obs.tool} args={obs.arguments}")
        if obs.error:
            parts.append(f"  error: {obs.error}")
            continue
        result = obs.result if isinstance(obs.result, dict) else {}
        if "docs" not in result:
            # Non-retrieval tools (code_execute, chart_generate, extract_data,
            # file_read, etc.) don't use the docs/confidence shape — render
            # their result directly. Without this, the LLM never sees these
            # tools' output and re-issues the same call repeatedly, believing
            # it got nothing back.
            max_len = 2000 if full else 300
            summary = json.dumps(result, default=str)[:max_len]
            parts.append(f"  result: {summary}")
            continue
        docs = result.get("docs", [])
        doc_count = len(docs)
        confidence = result.get("confidence", "N/A")
        sufficient = result.get("sufficient")
        sufficient_text = f" sufficient={sufficient}" if sufficient is not None else ""
        if full:
            parts.extend(_format_retrieval_obs_full(docs, doc_count, confidence, sufficient_text, seen_hashes))
        else:
            parts.extend(_format_retrieval_obs_compact(docs, doc_count, confidence, sufficient_text))
    return "\n".join(parts)


def _non_retrieval_observations_text(observations: list[Observation]) -> str:
    """Format only non-retrieval tool results (code_execute, chart_generate,
    extract_data, file_read, etc.).

    Retrieval (rag_retrieve) results are already in ``retrieved_docs`` via
    ``format_context_string``; including them again here would duplicate the
    same chunks in a different format. Non-retrieval tool outputs are NOT in
    ``retrieved_docs`` and must be surfaced to the LLM for answer synthesis.
    """
    parts = []
    for i, raw_obs in enumerate(observations, 1):
        obs = _coerce_observation(raw_obs)
        result = obs.result if isinstance(obs.result, dict) else {}
        if "docs" in result:
            continue  # retrieval — already in retrieved_docs
        parts.append(f"Observation {i}: tool={obs.tool} args={obs.arguments}")
        if obs.error:
            parts.append(f"  error: {obs.error}")
            continue
        summary = json.dumps(result, default=str)
        parts.append(f"  result: {summary}")
    return "\n".join(parts)


def _observations_metadata_text(observations: list[Observation]) -> str:
    """Format observations for think_node: metadata-only for rag_retrieve,
    full result for non-retrieval tools.

    rag_retrieve: the reranker already determined relevance. think_node only
    needs to know *what was found* (doc_count, confidence, sufficient) to
    decide whether to call another tool or finalize — not the chunk content.

    Non-retrieval tools (code_execute, chart_generate, extract_data, file_read):
    the LLM needs the full result to decide the next step.
    """
    parts = []
    for i, raw_obs in enumerate(observations, 1):
        obs = _coerce_observation(raw_obs)
        parts.append(f"Observation {i}: tool={obs.tool} args={obs.arguments}")
        if obs.error:
            parts.append(f"  error: {obs.error}")
            continue
        result = obs.result if isinstance(obs.result, dict) else {}
        if "docs" not in result:
            # Non-retrieval tool — full result needed for next-step reasoning.
            # Truncate kb_read content to avoid bloating the think prompt.
            if obs.tool == "kb_read" and "content" in result:
                content_preview = str(result.get("content", ""))[:300]
                parts.append(f"  document_id={result.get('document_id')} section={result.get('section')}")
                parts.append(f"  content_preview: {content_preview}…")
                continue
            if obs.tool == "kb_grep" and "matches" in result:
                match_count = len(result.get("matches", []))
                first_matches = result.get("matches", [])[:5]
                parts.append(f"  total_matches={result.get('total_matches')} documents_searched={result.get('documents_searched')}")
                for m in first_matches:
                    parts.append(f"    doc={m.get('document_id')} line={m.get('line_number')}: {m.get('line_text', '')[:80]}")
                continue
            summary = json.dumps(result, default=str)
            parts.append(f"  result: {summary}")
            continue
        # rag_retrieve — metadata only, no chunk content.
        doc_count = len(result.get("docs", []))
        confidence = result.get("confidence", "N/A")
        sufficient = result.get("sufficient")
        sufficient_text = f" sufficient={sufficient}" if sufficient is not None else ""
        missing = result.get("missing", "")
        missing_text = f" missing={missing}" if not sufficient and missing else ""
        rewritten = result.get("query_rewritten")
        rewrite_text = f" query_rewritten=true used={result.get('query_used', '')}" if rewritten else ""
        parts.append(f"  doc_count={doc_count} confidence={confidence}{sufficient_text}{missing_text}{rewrite_text}")
    return "\n".join(parts)


def _tried_rag_retrieve_queries(observations: list[Observation]) -> list[str]:
    """Exact query strings already sent to rag_retrieve, in order tried.

    The ladder inside rag_retrieve already exhausts every relaxation level
    for a given query string, so resubmitting the identical text can never
    yield a better result — it only wastes an iteration (the dedup layer in
    tool_node reuses the prior observation instead of re-running it).
    """
    seen: list[str] = []
    for raw_obs in observations:
        obs = _coerce_observation(raw_obs)
        if obs.tool == "rag_retrieve":
            query = obs.arguments.get("query")
            if query and query not in seen:
                seen.append(query)
    return seen


def _compact_observations(observations: list[Observation]) -> list[Observation]:
    """Stage 1 (deterministic): shrink tool observations in-place.

    Per the design doc (05-context-memory.md §4.4):
    - rag_retrieve: keep only top 5 chunks by score (already sorted by reranker).
    - code_execute: keep result + last 20 lines of stdout.
    - file_read: keep only a summary line.

    Returns a new list; original observations are not mutated.
    """
    compacted = []
    for raw_obs in observations:
        obs = _coerce_observation(raw_obs)
        if obs.error:
            compacted.append(obs)
            continue
        if obs.tool == "rag_retrieve":
            docs = obs.result.get("docs", [])
            if len(docs) > _COMPACT_KEEP_DOCS:
                new_result = dict(obs.result)
                new_result["docs"] = docs[:_COMPACT_KEEP_DOCS]
                compacted.append(Observation(
                    tool=obs.tool, observation_id=obs.observation_id,
                    arguments=obs.arguments, result=new_result,
                    error=obs.error, tokens=obs.tokens,
                ))
            else:
                compacted.append(obs)
        elif obs.tool == "code_execute":
            stdout = obs.result.get("stdout", "")
            if stdout and stdout.count("\n") > _COMPACT_KEEP_STDOUT_LINES:
                lines = stdout.split("\n")
                trimmed = "\n".join(lines[-_COMPACT_KEEP_STDOUT_LINES:])
                new_result = dict(obs.result)
                new_result["stdout"] = f"[...trimmed {len(lines) - _COMPACT_KEEP_STDOUT_LINES} lines...]\n{trimmed}"
                compacted.append(Observation(
                    tool=obs.tool, observation_id=obs.observation_id,
                    arguments=obs.arguments, result=new_result,
                    error=obs.error, tokens=obs.tokens,
                ))
            else:
                compacted.append(obs)
        else:
            compacted.append(obs)
    return compacted
