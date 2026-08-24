"""Agent loop graph for the enterprise agent.

Replaces the rigid RAG pipeline with a tool-calling loop:
  load_context → rewrite_query → plan → think → [tool → think ...] → finalize → save_memory
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from functools import partial
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from app.core.config import settings
from app.models.chat import ChatFile, Message
from app.services.agentic_rag.schemas import Observation
from langgraph.config import get_stream_writer


def _coerce_observation(obs: Observation | dict) -> Observation:
    """Coerce a dict (e.g. restored from Redis checkpoint) to an Observation.

    Redis checkpoint serializes Pydantic models as LangChain constructor
    dicts: {"lc": 2, "type": "constructor", "id": [...], "kwargs": {...}}.
    The actual fields live under "kwargs".
    """
    if isinstance(obs, Observation):
        return obs
    if isinstance(obs, dict):
        if "kwargs" in obs and "lc" in obs:
            return Observation(**obs["kwargs"])
        return Observation(**obs)
    return Observation(tool=str(obs))


def _writer():
    """Return a stream writer if one is available, else a no-op."""
    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda x: None


# Per-turn call caps for tools that can be invoked in a tight loop.
# Resolved per-request via the settings service (org-overridable).
def _tool_call_budget(db, org_id) -> dict:
    return {
        "rag_retrieve": get_setting(db, "AGENT_MAX_RETRIEVALS", org_id),
        "code_execute": get_setting(db, "AGENT_MAX_CODE_EXEC", org_id),
    }

# Error patterns that indicate a transient infrastructure failure rather
# than a bad-argument error.  Transient failures retry with the same
# arguments (plus backoff); argument failures call the correction LLM.
_TRANSIENT_ERROR_PATTERNS = (
    "timeout", "timed out", "connection", "network", "unreachable",
    "temporarily", "broken pipe", "reset by peer", "i/o error",
    "errno 5", "errno 11", "errno 104", "errno 110",
)

# Tool-specific hints appended to the correction prompt so the LLM knows
# how to fix common errors without guessing.
_TOOL_ERROR_HINTS: dict[str, dict[str, str]] = {
    "code_execute": {
        "_iter_unpack_sequence_": "List comprehensions and tuple unpacking in for-loops are not supported. Rewrite as explicit for-loops with .append().",
        "_unpack_sequence_": "Tuple unpacking (a, b = ...) is not supported. Use indexing: a = x[0]; b = x[1].",
        "_inplacevar_": "Augmented assignment (+=, *=) is not supported. Use explicit assignment: x = x + 1.",
    },
    "chart_generate": {
        "No numeric values": "Each data item must have a 'value' key with a numeric value. Check your data items.",
        "No data provided": "The 'data' field must be a non-empty list of objects with 'label' and 'value' keys.",
    },
    "extract_data": {
        "JSON": "Try a different source or simplify the focus parameter.",
    },
    "file_read": {
        "not found": "Check the file_id and section parameter.",
    },
}


def _is_transient_error(error: str) -> bool:
    return any(p in (error or "").lower() for p in _TRANSIENT_ERROR_PATTERNS)


def _correction_hints(tool_name: str, error: str) -> str:
    hints = _TOOL_ERROR_HINTS.get(tool_name, {})
    matched = [h for pat, h in hints.items() if pat in (error or "")]
    if matched:
        return "\n".join(matched)
    return "Fix the arguments based on the error message."


def _extract_balanced(text: str, chars: tuple[str, str]) -> str | None:
    """Return the first balanced *chars* region in *text* while respecting strings."""
    start_char, end_char = chars
    start = text.find(start_char)
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def _substitute_chart_markers(text: str, chart_options: list[dict]) -> str:
    """Replace [[CHART_N]] placeholders with the real ECharts fence.

    Any chart whose marker the model omitted is appended at the end, so a
    chart is never silently dropped even if placement wasn't followed.
    """
    result = text
    for i, option in enumerate(chart_options, start=1):
        marker = f"[[CHART_{i}]]"
        fence = f"```echarts\n{json.dumps(option)}\n```"
        if marker in result:
            result = result.replace(marker, fence, 1)
        else:
            result = f"{result}\n\n{fence}"
    return result


def _extract_json_block(text: str) -> str | None:
    """Return the first well-formed JSON object or array string from *text*.

    Tries markdown fenced blocks first, then scans for balanced braces or brackets.
    """
    if not text:
        return None
    # Prefer a fenced ```json ... ``` block.
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        block = _extract_balanced(m.group(1), ("{", "}")) or _extract_balanced(m.group(1), ("[", "]"))
        if block:
            return block
    # Fall back to the first inline balanced object or array.
    return _extract_balanced(text, ("{", "}")) or _extract_balanced(text, ("[", "]"))


from app.services.agentic_rag.llm_factory import build_chat_llm
from app.services.settings_service import get_setting
from app.core.settings_registry import get_def
from app.services.agentic_rag.nodes import (
    _agent_step,
    answer_evaluation_node,
    expand_query_node,
    history_to_text,
    rewrite_query_node,
    select_recent_history,
)
from app.services.agentic_rag.prompts import (
    AGENT_SYSTEM_PROMPT,
    FINALIZE_ANSWER_PROMPT,
    FINALIZE_GUARDRAIL_PROMPT,
    LAST_ANSWER_EXTRACT_PROMPT,
    PLAN_SYSTEM_PROMPT,
    THINK_SYSTEM_PROMPT,
)
from app.services.agentic_rag.schemas import LastAnswerObject, Observation, Plan, Subtask
from app.services.agentic_rag.tool_call_parser import parse_think_response
from app.services.agentic_rag.tool_context import ToolContext, write_audit
from app.services.agentic_rag.tools import applicable_tools
from app.services.agentic_rag.token_budget import count_tokens
from app.services.agentic_rag.utils import format_context_string, normalize_citations

from .graph_state import AgentState

logger = logging.getLogger(__name__)


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
    from app.services.infrastructure import content_hash as _ch

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
            unique_docs = []
            for doc in docs:
                if not isinstance(doc, dict):
                    continue
                h = doc.get("metadata", {}).get("content_hash") or _ch(doc.get("page_content", ""))
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    unique_docs.append(doc)
            parts.append(f"  doc_count={doc_count} unique_so_far={len(seen_hashes)} confidence={confidence}{sufficient_text}")
            # Prune overlap from contiguous chunks so the LLM doesn't see
            # duplicated text (300 chars per adjacent pair at 20% overlap).
            pruned_docs = _prune_contiguous_overlaps(unique_docs)
            for j, doc in enumerate(pruned_docs, 1):
                content = str(doc.get("page_content", ""))
                parts.append(f"  doc_{j}: {content}")
        else:
            parts.append(f"  doc_count={doc_count} confidence={confidence}{sufficient_text}")
            if docs and isinstance(docs[0], dict):
                preview = str(docs[0].get("page_content", ""))[:300]
                parts.append(f"  top_doc_preview: {preview}")
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
            summary = json.dumps(result, default=str)
            parts.append(f"  result: {summary}")
            continue
        # rag_retrieve — metadata only, no chunk content.
        doc_count = len(result.get("docs", []))
        confidence = result.get("confidence", "N/A")
        sufficient = result.get("sufficient")
        sufficient_text = f" sufficient={sufficient}" if sufficient is not None else ""
        parts.append(f"  doc_count={doc_count} confidence={confidence}{sufficient_text}")
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


# ---------------------------------------------------------------------------
# Runtime compaction — referral-style context budget check
# ---------------------------------------------------------------------------

# How many docs to keep per rag_retrieve observation when compacting.
_COMPACT_KEEP_DOCS = 5
# Doc preview length when compacting (shorter than the 800 used in full mode).
_COMPACT_DOC_PREVIEW_CHARS = 200
# How many stdout lines to keep for code_execute when compacting.
_COMPACT_KEEP_STDOUT_LINES = 20
# Stable id for the conversation-summary message so repeated compactions
# replace the previous summary instead of stacking summaries.
_COMPACTION_SUMMARY_ID = "agentic-compaction-summary"


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


async def _compact_messages_llm(
    messages: list,
    ctx: Optional["ToolContext"] = None,
) -> tuple[list, list, str | None]:
    """Summarize the oldest messages into one structured summary message.

    Returns ``(message_updates, resolved_messages, summary_text)``.
    ``message_updates`` is a LangGraph message-channel update that *replaces*
    the history — the ``add_messages`` reducer appends, so a plain
    ``[summary] + recent`` list would grow the checkpoint instead of shrinking
    it. Uses ``RemoveMessage(REMOVE_ALL_MESSAGES)`` plus a stable summary id so
    repeated compactions replace the previous summary rather than stacking.
    """
    from langchain_core.messages import RemoveMessage
    from langgraph.graph.message import REMOVE_ALL_MESSAGES

    from app.services.agentic_rag.nodes import _messages_to_conversation_text
    from app.services.agentic_rag.prompts import COMPACTION_SYSTEM_PROMPT, COMPACTION_USER_PROMPT

    keep_recent = get_setting(ctx.db, "COMPACTION_KEEP_RECENT", ctx.org_id) if ctx else get_def("COMPACTION_KEEP_RECENT").default
    recent = messages[-keep_recent:] if keep_recent else []
    old = messages[:len(messages) - len(recent)]
    # Drop a previous summary from *old* so it is re-summarised rather than
    # quoted verbatim inside the new summary.
    old = [m for m in old if getattr(m, "id", None) != _COMPACTION_SUMMARY_ID]
    if not old:
        return [], list(messages), None

    conversation_text = _messages_to_conversation_text(old)
    try:
        llm = _build_compaction_llm(ctx)
        response = await llm.ainvoke([
            {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
            {"role": "user", "content": COMPACTION_USER_PROMPT.format(conversation=conversation_text)},
        ])
        summary = str(response.content).strip()
        max_chars = get_setting(ctx.db, "COMPACTION_SUMMARY_MAX_CHARS", ctx.org_id) if ctx else get_def("COMPACTION_SUMMARY_MAX_CHARS").default
        if len(summary) > max_chars:
            summary = summary[:max_chars] + "\n\n[...summary truncated]"
        summary_msg = HumanMessage(
            content=f"[Conversation summary]\n{summary}", id=_COMPACTION_SUMMARY_ID
        )
        resolved = [summary_msg, *recent]
        return [RemoveMessage(id=REMOVE_ALL_MESSAGES), *resolved], resolved, summary
    except Exception as exc:
        logger.warning("[_compact_messages_llm] failed: %s — keeping full history", exc)
        return [], list(messages), None


def _build_compaction_llm(ctx: Optional["ToolContext"]):
    """Return the org-configured summarisation LLM, falling back to globals."""
    if ctx is not None:
        try:
            return build_chat_llm(ctx.org_id, ctx.db, role="query", temperature=0.0)
        except Exception as exc:
            logger.warning("[compaction] org LLM unavailable (%s) — falling back to global config", exc)
    from app.services.agentic_rag.nodes import _get_llm

    return _get_llm(temperature=0.0, streaming=False)


def _trim_docs_to_budget(docs: list[dict], overflow_tokens: int) -> list[dict]:
    """Drop the lowest-scoring chunks until roughly *overflow_tokens* are freed.

    ``finalize_node``'s prompt is dominated by ``retrieved_docs``; summarising
    conversation history cannot fix an evidence-payload overflow. Chunks are
    removed lowest ``_reranker_score`` first so the strongest evidence — and
    therefore the citation set — survives.
    """
    if not docs or overflow_tokens <= 0:
        return docs

    scored = sorted(
        enumerate(docs),
        key=lambda pair: pair[1].get("_reranker_score", pair[1].get("score", 0.0) or 0.0),
    )
    drop: set[int] = set()
    freed = 0
    # Always keep at least one chunk — an empty context guarantees a refusal.
    for idx, doc in scored[:-1]:
        if freed >= overflow_tokens:
            break
        freed += count_tokens(str(doc.get("page_content", "")))
        drop.add(idx)

    if not drop:
        return docs
    logger.info("[_trim_docs_to_budget] dropped %d/%d chunks (~%d tokens)", len(drop), len(docs), freed)
    return [d for i, d in enumerate(docs) if i not in drop]


async def _compact_if_needed(
    state: AgentState,
    prompt_text: str,
    system_overhead: int = 0,
    ctx: Optional["ToolContext"] = None,
    trim_docs: bool = False,
) -> tuple[dict, dict]:
    """Single context-budget guard. Called before any LLM call with variable context.

    Stages, applied only while the rebuilt prompt is still over budget:
      1. Deterministic: shrink tool observations (keep top docs, trim stdout).
      2. Evidence packing: drop the lowest-scoring retrieved chunks
         (``trim_docs=True``, i.e. the finalize prompt only).
      3. LLM call: summarise the oldest messages into one summary message.

    Returns ``(graph_updates, local_view)``. ``graph_updates`` is returned from
    the node and speaks the channels' reducer contracts (``__reset__`` markers,
    ``RemoveMessage``). ``local_view`` holds the same data already resolved, so
    the caller can rebuild its prompt without having to interpret reducer
    markers. Both are empty when no compaction was needed.
    """
    if not (get_setting(ctx.db, "COMPACTION_ENABLED", ctx.org_id) if ctx else get_def("COMPACTION_ENABLED").default):
        return {}, {}

    from app.services.agentic_rag.token_budget import ContextBudget

    budget = ContextBudget(db=ctx.db if ctx else None, org_id=ctx.org_id if ctx else None)
    budget.add(count_tokens(prompt_text))
    budget.add(system_overhead)

    if not budget.needs_compaction():
        return {}, {}

    logger.info(
        "[_compact_if_needed] over budget | used=%d threshold=%d — compacting",
        budget.used, budget.compaction_threshold,
    )

    updates: dict[str, Any] = {"compaction_triggered": True}
    local: dict[str, Any] = {}

    # Stage 1: compact observations (deterministic, no LLM call).
    # The `observations` channel uses the append-style `accumulate` reducer,
    # so a replacement must be sent through the `__reset__` marker contract.
    observations = [_coerce_observation(o) for o in state.get("observations", [])]
    compacted_obs = _compact_observations(observations)
    obs_tokens_before = sum(count_tokens(json.dumps(o.result, default=str)) for o in observations)
    obs_tokens_after = sum(count_tokens(json.dumps(o.result, default=str)) for o in compacted_obs)
    savings = obs_tokens_before - obs_tokens_after
    if savings > 0:
        updates["observations"] = [{"__reset__": True}, *compacted_obs]
        local["observations"] = compacted_obs
        budget.used -= savings
        logger.info("[_compact_if_needed] stage 1 (observations) saved %d tokens", savings)

    if not budget.needs_compaction():
        return updates, local

    # Stage 2: evidence packing — only meaningful where retrieved_docs are
    # actually rendered into the prompt (finalize).
    if trim_docs:
        docs = list(state.get("retrieved_docs", []) or [])
        overflow = budget.used - budget.compaction_threshold
        trimmed = _trim_docs_to_budget(docs, overflow)
        if len(trimmed) < len(docs):
            freed = count_tokens(format_context_string(docs)) - count_tokens(format_context_string(trimmed))
            updates["retrieved_docs"] = trimmed
            local["retrieved_docs"] = trimmed
            budget.used -= max(freed, 0)
            logger.info("[_compact_if_needed] stage 2 (evidence) saved %d tokens", freed)

    if not budget.needs_compaction():
        return updates, local

    # Stage 3: summarise old messages (LLM call).
    messages = list(state.get("messages", []))
    if len(messages) > (get_setting(ctx.db, "COMPACTION_KEEP_RECENT", ctx.org_id) if ctx else get_def("COMPACTION_KEEP_RECENT").default):
        message_updates, resolved, summary = await _compact_messages_llm(messages, ctx=ctx)
        if summary is not None:
            updates["messages"] = message_updates
            updates["compaction_summary"] = summary
            local["messages"] = resolved
            local["compaction_summary"] = summary
            logger.info("[_compact_if_needed] stage 3 (messages) summarized %d old messages",
                        len(messages) - len(resolved) + 1)

    return updates, local


async def load_context_node(state: AgentState, ctx: ToolContext) -> dict:
    """Load previous-answer object, recalled memory, and file metadata into state."""
    with _agent_step("load_context"):
        last_obj: Optional[LastAnswerObject] = None
        if ctx.chat_id and ctx.message_id:
            # The current assistant message may already exist; find the previous assistant message.
            prev = (
                ctx.db.query(Message)
                .filter(Message.chat_id == ctx.chat_id, Message.role == "assistant")
                .filter(Message.id != ctx.message_id)
                .order_by(Message.id.desc())
                .first()
            )
            if prev and prev.last_answer_object:
                try:
                    last_obj = LastAnswerObject(**prev.last_answer_object)
                except Exception:
                    last_obj = None
    
        recalled: list[dict] = []
        if ctx.redis_memory and getattr(ctx.redis_memory, "search_memory", None):
            try:
                recalled = await ctx.redis_memory.search_memory(
                    query=state.get("original_query", ""),
                    user_id=ctx.user_id,
                    chat_id=ctx.chat_id,
                    limit=3,
                )
            except Exception as exc:
                logger.warning("[load_context] memory search failed: %s", exc)
    
        return {
            "last_answer_object": last_obj,
            # Recalled conversational memory is NOT citable evidence: it stays
            # out of retrieved_docs so it can never be rendered as a [KB-n]
            # chunk, cited, or scored for faithfulness.
            "recalled_memories": recalled,
            "org_id": ctx.org_id,
            "user_id": ctx.user_id,
            "chat_id": ctx.chat_id,
            "message_id": ctx.message_id,
            "started_at": time.monotonic(),
            # Reset per-turn loop state; the checkpointer otherwise carries it
            # over from the previous turn (e.g. force_finalize would silently
            # kill tool calls this turn; observations would leak last turn's
            # doc chunks into this turn's think_node prompt).
            "observations": [{"__reset__": True}],
            "iteration": 0,
            "tool_call_count": {},
            "force_finalize": False,
            "precomputed_tool_calls": [],
            "reflection_final": None,
            "precomputed_answer": "",
            "tool_calls": [],
            "retrieved_docs": [],
            "all_scored_docs": [],
            "cited_doc_indices": [],
            "retrieval_confidence": 0.0,
            "compaction_triggered": False,
            "answer_evaluation_attempts": 0,
            "evaluation_flags": [],
            "adaptive_reran": False,
            "answer_usage": None,
            "final_answer": "",
            "answer": "",
            "clarification_count": 0,
            "clarification_response": "",
            "needs_clarification": False,
            "resolution_provenance": None,
        }
    
    
async def plan_node(state: AgentState, ctx: ToolContext) -> dict:
    """Produce a structured plan for the current turn."""
    with _agent_step("plan"):
        writer = _writer()
        original = state.get("original_query", "")
        rewritten = state.get("rewritten_query", "") or original
    
        file_meta = []
        if ctx.chat_id:
            files = ctx.db.query(ChatFile).filter(ChatFile.chat_id == ctx.chat_id).all()
            file_meta = [{"id": f.id, "name": f.file_name, "type": f.content_type} for f in files]
    
        last_summary = ""
        lao = state.get("last_answer_object")
        if lao and hasattr(lao, "summary"):
            last_summary = lao.summary
            if getattr(lao, "chart_options", None):
                last_summary += f" (Previous answer includes {len(lao.chart_options)} chart(s) with structured data.)"
    
        recalled = state.get("recalled_memories", []) or []
        recalled_text = "\n".join(d.get("page_content", "") for d in recalled[:3])
        clarification = (state.get("clarification_response") or "").strip()
    
        system = AGENT_SYSTEM_PROMPT + "\n\n" + PLAN_SYSTEM_PROMPT

        # Glossary was built once by expand_query_node — reuse it.
        glossary = state.get("abbreviation_glossary", "")

        user = (
            f"User message: {original}\n"
            f"Retrieval query: {rewritten}\n"
            + (f"User clarification: {clarification}\n" if clarification else "")
            + (f"[Abbreviation Glossary]\n{glossary}\n\n" if glossary else "")
            + f"Previous answer summary: {last_summary}\n"
            f"Recalled long-term memory (context only, not evidence):\n{recalled_text}\n\n"
            f"Attached files: {json.dumps(file_meta)}\n\n"
            "Produce a plan JSON matching the schema."
        )
    
        try:
            llm = build_chat_llm(ctx.org_id, ctx.db, role="query", temperature=0.0)
            structured = llm.with_structured_output(Plan, method="json_schema", include_raw=True)
            resp = await structured.ainvoke([
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ])
            # include_raw=True returns a dict with 'raw', 'parsed', 'parsing_error'
            if isinstance(resp, dict):
                plan = resp.get("parsed")
                if plan is None or resp.get("parsing_error"):
                    raise resp.get("parsing_error") or ValueError("structured output parsed to None")
            else:
                plan = resp.parsed if hasattr(resp, "parsed") else resp
        except Exception as exc:
            logger.warning("[plan_node] structured output failed: %s; using JSON parse fallback", exc)
            llm = build_chat_llm(ctx.org_id, ctx.db, role="query", temperature=0.0)
            resp = await llm.ainvoke([
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ])
            raw = str(resp.content)
            block = _extract_json_block(raw)
            try:
                plan = Plan.model_validate_json(block) if block else Plan()
            except Exception as parse_exc:
                logger.warning("[plan_node] JSON parse failed: %s", parse_exc)
                plan = Plan(intent="rag", subtasks=[Subtask(id="a", description=rewritten, tool_hint="rag_retrieve")])
    
        writer({"event": "plan", "plan": plan.model_dump() if isinstance(plan, Plan) else plan})

        # Clarification budget: without a cap, a model that keeps setting
        # needs_clarification=true loops plan → clarify → plan until the
        # recursion limit. Past the cap, answer with the ambiguity stated.
        needs_clarification = bool(getattr(plan, "needs_clarification", False))
        if needs_clarification and state.get("clarification_count", 0) >= get_setting(ctx.db, "AGENT_MAX_CLARIFICATIONS", ctx.org_id):
            logger.info("[plan_node] clarification budget exhausted — proceeding without asking")
            needs_clarification = False
            if isinstance(plan, Plan):
                plan.needs_clarification = False

        return {
            "plan": plan,
            "needs_clarification": needs_clarification,
            "clarification_question": getattr(plan, "clarification_question", None),
        }
    
    
async def think_node(state: AgentState, ctx: ToolContext) -> dict:
    """Decide the next action: emit one or more tool calls or a final answer."""
    with _agent_step("think"):
        ctx.state = state
        iteration = state.get("iteration", 0) + 1
        max_iter = get_setting(ctx.db, "AGENT_MAX_ITERATIONS", ctx.org_id)
    
        if state.get("force_finalize"):
            return {"iteration": iteration, "tool_calls": [], "precomputed_answer": ""}

        # Pre-think sufficiency check: if the plan is already deterministically
        # satisfied, don't spend an LLM call asking the model whether to stop —
        # it isn't reliable at noticing this on its own (see tool_node's matching
        # post-round check for the same reasoning).
        ready, _reasoning = _verify_execution(_build_execution_summary(state))
        if ready:
            return {"iteration": iteration, "tool_calls": [], "precomputed_answer": ""}

        precomputed = state.get("precomputed_tool_calls", [])
        if precomputed:
            return {"iteration": iteration, "tool_calls": list(precomputed), "precomputed_tool_calls": []}
    
        query = state.get("rewritten_query", "") or state.get("original_query", "")
        original = state.get("original_query", "") or query
        plan = state.get("plan") or Plan()
        observations = state.get("observations", [])
        # Expose current state to tools so applicable_tools() and tool reads
        # (last_answer_object, retrieved_docs, kb_ids, file_markdown) see live data.
        ctx.state = state
        tools = applicable_tools(ctx)
        tools_text = _tool_descriptions_text(tools)

        # Build conversation context from the same shared projection every
        # other node uses (select_recent_history), so rewrite/think/finalize
        # can't disagree about what "recent history" means.
        recent = select_recent_history(state.get("messages", []), max_pairs=get_setting(ctx.db, "AGENT_HISTORY_PAIRS", ctx.org_id))
        history_text = history_to_text(recent)
        summary_text = state.get("compaction_summary") or ""

        # Include last_answer_object summary so "summarize it" / "chart it" work.
        lao = state.get("last_answer_object")
        lao_text = ""
        if lao and hasattr(lao, "summary"):
            lao_text = f"  Previous answer summary: {lao.summary[:300]}\n"
            if lao.key_points:
                lao_text += f"  Key points: {'; '.join(lao.key_points[:5])}\n"

        # If reflect_final sent us back, include its reasoning so the agent
        # knows exactly what was missing and can act on it.
        reflection = state.get("reflection_final")
        reflection_text = ""
        if reflection and isinstance(reflection, dict) and not reflection.get("ready", True):
            reflection_text = (
                f"  NOTE — the verification module rejected your previous final_answer because:\n"
                f"  {reflection.get('reasoning', '')}\n"
                "  Do NOT reference this feedback in your answer. Use it only as guidance to\n"
                "  decide which tool to call next, then emit a clean final_answer.\n"
            )

        system = AGENT_SYSTEM_PROMPT + "\n\n" + THINK_SYSTEM_PROMPT
        tried_queries = _tried_rag_retrieve_queries(observations)
        tried_queries_text = (
            f"  Already tried (do NOT resubmit these exact strings to rag_retrieve): {tried_queries}\n"
            if tried_queries else ""
        )

        # Glossary was built once by expand_query_node — reuse it.
        glossary = state.get("abbreviation_glossary", "")

        def _build_think_user_prompt() -> str:
            return (
                f"Iteration: {iteration}/{max_iter}\n"
                f"User message: {original}\n"
                f"Retrieval query: {query}\n"
                + (f"[Abbreviation Glossary]\n{glossary}\n\n" if glossary else "")
                + (f"Earlier conversation summary:\n{summary_text}\n" if summary_text else "")
                + f"Conversation history (recent turns):\n{history_text or '  (none)'}\n"
                f"Previous answer context:\n{lao_text or '  (none)'}\n"
                f"Verification feedback:\n{reflection_text or '  (none)'}\n"
                f"{tried_queries_text}"
                f"Plan: {json.dumps(plan.model_dump() if isinstance(plan, Plan) else plan, default=str)}\n"
                f"Observations so far:\n{_observations_metadata_text(observations)}\n\n"
                f"Available tools:\n{tools_text}\n\n"
                "Emit either {\"tool_calls\": [...]} or {\"final_answer\": true}."
            )

        user = _build_think_user_prompt()

        # Runtime compaction: check if the prompt exceeds the context budget.
        # If so, compact observations (deterministic) and/or messages (LLM call),
        # then rebuild the prompt from the compacted state.
        compaction_updates, compaction_local = await _compact_if_needed(
            state, user, system_overhead=count_tokens(system), ctx=ctx,
        )
        if compaction_local:
            state = {**state, **compaction_local}
            observations = state.get("observations", [])
            recent = select_recent_history(state.get("messages", []), max_pairs=get_setting(ctx.db, "AGENT_HISTORY_PAIRS", ctx.org_id))
            history_text = history_to_text(recent)
            summary_text = state.get("compaction_summary") or ""
            tried_queries = _tried_rag_retrieve_queries(observations)
            tried_queries_text = (
                f"  Already tried (do NOT resubmit these exact strings to rag_retrieve): {tried_queries}\n"
                if tried_queries else ""
            )
            user = _build_think_user_prompt()
    
        mode = get_setting(ctx.db, "TOOL_CALL_MODE", None)
        try:
            # Tool selection is a classification decision — temperature 0.
            # Creative sampling belongs in finalize_node's prose, not here.
            if mode == "json_text":
                llm = build_chat_llm(ctx.org_id, ctx.db, role="chat", temperature=0.0)
                resp = await llm.ainvoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
            else:
                # native or auto: bind tools; parser falls back to JSON-text if native call absent.
                llm = build_chat_llm(ctx.org_id, ctx.db, role="chat", temperature=0.0)
                resp = await llm.bind_tools(tools).ainvoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
        except Exception as exc:
            logger.warning("[think_node] LLM call failed: %s", exc)
            return {"iteration": iteration, "tool_calls": [], "precomputed_answer": f"LLM error: {exc}"}
    
        parsed = parse_think_response(resp, mode=mode)
        tool_calls = parsed.tool_calls
        final_answer = parsed.final_answer
    
        if iteration >= max_iter:
            tool_calls = []
    
        # Dependency guard: only allow independent tool calls in one message.
        allowed = list(tool_calls)
    
        if tool_calls and not final_answer:
            return {**compaction_updates, "iteration": iteration, "tool_calls": allowed}

        # final_answer can be:
        #   - True (boolean signal from {"final_answer": true}) — think is done,
        #     finalize_node will generate the answer with streaming.
        #   - str (Tier 3 fallback — LLM wrote plain text instead of JSON) — pass
        #     it through as precomputed since the text was already generated.
        if isinstance(final_answer, bool) and final_answer:
            return {**compaction_updates, "iteration": iteration, "tool_calls": [], "precomputed_answer": ""}
        return {**compaction_updates, "iteration": iteration, "tool_calls": [], "precomputed_answer": final_answer or ""}
    
    
def _wall_clock_exceeded(state: AgentState) -> bool:
    started_at = state.get("started_at")
    if started_at is None:
        return False
    from app.db.session import SessionLocal
    org_id = state.get("org_id")
    _db = SessionLocal()
    try:
        max_seconds = get_setting(_db, "AGENT_MAX_WALL_SECONDS", org_id)
    finally:
        _db.close()
    return (time.monotonic() - started_at) >= max_seconds


def route_think(state: AgentState) -> str:
    iteration = state.get("iteration", 0)
    from app.db.session import SessionLocal
    org_id = state.get("org_id")
    _db = SessionLocal()
    try:
        max_iter = get_setting(_db, "AGENT_MAX_ITERATIONS", org_id)
    finally:
        _db.close()
    if iteration >= max_iter or _wall_clock_exceeded(state):
        return "reflect_final"
    if state.get("tool_calls"):
        return "tool"
    return "reflect_final"


def route_tool(state: AgentState) -> str:
    """After a tool round: skip reflect+think entirely if already satisfied."""
    if state.get("force_finalize"):
        return "reflect_final"
    return "reflect"


def route_reflect_final(state: AgentState) -> str:
    """Route after final verification: ready → finalize, not ready → think."""
    reflection = state.get("reflection_final", {})
    ready = reflection.get("ready", True) if isinstance(reflection, dict) else True
    iteration = state.get("iteration", 0)
    from app.db.session import SessionLocal
    org_id = state.get("org_id")
    _db = SessionLocal()
    try:
        max_iter = get_setting(_db, "AGENT_MAX_ITERATIONS", org_id)
    finally:
        _db.close()
    if not ready and iteration < max_iter and not _wall_clock_exceeded(state):
        return "think"
    return "finalize"


async def _correct_tool_args(
    tool_name: str,
    original_args: dict,
    error: str,
    tools: dict,
    ctx: "ToolContext",
) -> dict | None:
    """Call the correction LLM to produce fixed arguments for a failed tool call."""
    tool = tools.get(tool_name)
    if tool is None:
        return None
    schema = {}
    try:
        schema = tool.args_schema.model_json_schema()
    except Exception:
        pass
    prompt = (
        f"The {tool_name} tool failed with this error:\n{error}\n\n"
        f"Original arguments: {json.dumps(original_args, default=str)}\n\n"
        f"Tool schema: {json.dumps(schema, default=str)}\n\n"
        f"Generate corrected arguments as a JSON object matching the schema.\n"
        f"{_correction_hints(tool_name, error)}\n"
        "Return ONLY the JSON object, no explanation."
    )
    try:
        llm = build_chat_llm(ctx.org_id, ctx.db, role="query", temperature=0.0)
        response = await llm.ainvoke([{"role": "user", "content": prompt}])
        raw = str(response.content)
        block = _extract_json_block(raw)
        if block:
            corrected = json.loads(block)
            if isinstance(corrected, dict):
                return corrected
    except Exception as exc:
        logger.debug("[_correct_tool_args] correction LLM failed: %s", exc)
    return None


async def tool_node(state: AgentState, ctx: ToolContext) -> dict:
    """Dispatch tool calls, run them (in parallel when independent), record observations."""
    with _agent_step("tool"):
        writer = _writer()
        tool_calls = state.get("tool_calls", [])
        if not tool_calls:
            return {}
    
        # Expose current state to tools so they can read last_answer_object,
        # retrieved_docs, kb_ids, file_markdown, message_id, iteration, etc.
        ctx.state = state
        tools = {t.name: t for t in applicable_tools(ctx)}
        prior_observations = [_coerce_observation(o) for o in state.get("observations", [])]
        new_observations: list[Observation] = []
        counts = dict(state.get("tool_call_count", {}))

        # Idempotency guard: the think LLM sometimes re-emits an identical
        # tool_call (same tool + same arguments) across iterations even
        # when instructed not to. Reuse the prior observation instead of
        # re-running an expensive retrieval/tool call for nothing.
        def _call_signature(name: str, args: dict) -> tuple[str, str]:
            return (name, json.dumps(args, sort_keys=True, default=str))

        prior_signatures: dict[tuple[str, str], Observation] = {}
        for obs in prior_observations:
            prior_signatures.setdefault(_call_signature(obs.tool, obs.arguments), obs)

        async def _budget_exceeded(name, args, cap):
            return {"tool": name, "arguments": args, "result": {}, "error": f"Budget exceeded: {name} call cap is {cap}", "tokens": 0}

        async def _reuse_prior(prior: Observation):
            return {
                "tool": prior.tool,
                "arguments": prior.arguments,
                "result": prior.result,
                "error": prior.error,
                "tokens": 0,
            }

        coros = []
        for tc in tool_calls:
            name = tc.get("tool")
            args = tc.get("arguments", {})
            tool_obj = tools.get(name)
            label = getattr(tool_obj, "ui_label", None) if tool_obj else None
            writer({"event": "tool_call", "tool": name, "arguments": args, "label": label or name})
            prior = prior_signatures.get(_call_signature(name, args))
            if prior is not None:
                logger.info("[tool_node] duplicate call skipped, reusing prior observation: tool=%s args=%s", name, args)
                coros.append(_reuse_prior(prior))
                continue
            cap = _tool_call_budget(ctx.db, ctx.org_id).get(name)
            current = counts.get(name, 0)
            if cap is not None and current >= cap:
                coros.append(_budget_exceeded(name, args, cap))
                continue
            tool = tools.get(name)
            if tool is None:
                async def _missing(name=name, args=args):
                    return {"tool": name, "arguments": args, "result": {}, "error": f"Tool {name} not available", "tokens": 0}
                coros.append(_missing())
            else:
                coros.append(_run_tool(tool, name, args))
    
        results = await asyncio.gather(*coros, return_exceptions=True)
        for i, tc in enumerate(tool_calls):
            res = results[i]
            if isinstance(res, Exception):
                obs = Observation(
                    tool=tc["tool"],
                    arguments=tc.get("arguments", {}),
                    result={},
                    error=str(res),
                    tokens=0,
                )
            else:
                obs = Observation(
                    tool=res["tool"],
                    arguments=res["arguments"],
                    result=res.get("result", {}),
                    error=res.get("error"),
                    tokens=res.get("tokens", 0),
                )
            new_observations.append(obs)
            writer({"event": "tool_observation", **obs.model_dump()})
            counts[obs.tool] = counts.get(obs.tool, 0) + 1

        # Retry failed tool calls: transient errors retry with the same
        # arguments + backoff; argument errors call the correction LLM to
        # generate new arguments.  Retries do NOT count against the per-tool
        # call budget (_TOOL_CALL_BUDGET) — that budget limits how many times
        # the *think* LLM can choose to call a tool, not how many times a
        # single failed call can be retried.
        max_retries = get_setting(ctx.db, "AGENT_MAX_TOOL_RETRIES", ctx.org_id)
        if max_retries > 0:
            for idx, obs in enumerate(new_observations):
                if obs.error is None:
                    continue
                tool_name = obs.tool
                tool = tools.get(tool_name)
                if tool is None:
                    continue
                for attempt in range(max_retries):
                    if _is_transient_error(obs.error):
                        await asyncio.sleep(get_setting(ctx.db, "AGENT_RETRY_BACKOFF_BASE", ctx.org_id) * (2 ** attempt))
                        retry_args = obs.arguments
                    else:
                        retry_args = await _correct_tool_args(
                            tool_name, obs.arguments, obs.error, tools, ctx,
                        )
                        if retry_args is None:
                            break
                    retry_result = await _run_tool(tool, tool_name, retry_args)
                    retry_obs = Observation(
                        tool=retry_result["tool"],
                        arguments=retry_result["arguments"],
                        result=retry_result.get("result", {}),
                        error=retry_result.get("error"),
                        tokens=retry_result.get("tokens", 0),
                    )
                    writer({
                        "event": "tool_retry",
                        "tool": tool_name,
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                        "success": retry_obs.error is None,
                        "error": retry_obs.error,
                    })
                    if retry_obs.error is None:
                        new_observations[idx] = retry_obs
                        break
                    obs = retry_obs
    
        # Promote all rag_retrieve docs into graph state (deduplicated across
        # observations by content_hash) so finalize_node, answer_evaluation_node,
        # extract_data(source="retrieved_docs"), and the citations payload in
        # agent_runner all see the full set of retrieved chunks — not just the
        # first call's docs.
        #
        # `observations` uses the append-style `accumulate` reducer, so this
        # node must return ONLY the observations it created. Returning
        # prior + new made the channel grow 1 → 3 → 7 → 15 across tool rounds.
        state_update: dict = {
            "tool_calls": [],
            "observations": new_observations,
            "tool_call_count": counts,
        }
        all_observations = prior_observations + new_observations
        from app.services.infrastructure import content_hash as _ch
        merged_docs: list[dict] = []
        seen_hashes: set[str] = set()
        best_confidence = 0.0
        # Seed with docs already promoted this turn so a later rag_retrieve
        # call doesn't discard earlier ones. Recalled conversational memory is
        # deliberately NOT seeded here — it lives in `recalled_memories` and
        # must never become citable evidence.
        for doc in state.get("retrieved_docs", []) or []:
            if not isinstance(doc, dict):
                continue
            h = doc.get("metadata", {}).get("content_hash") or _ch(doc.get("page_content", ""))
            if h not in seen_hashes:
                seen_hashes.add(h)
                merged_docs.append(doc)
        for obs in all_observations:
            if obs.tool == "rag_retrieve" and not obs.error:
                docs = obs.result.get("docs")
                if isinstance(docs, list):
                    for doc in docs:
                        if not isinstance(doc, dict):
                            continue
                        h = doc.get("metadata", {}).get("content_hash") or _ch(doc.get("page_content", ""))
                        if h not in seen_hashes:
                            seen_hashes.add(h)
                            merged_docs.append(doc)
                    conf = obs.result.get("confidence", 0.0)
                    if conf > best_confidence:
                        best_confidence = conf
        if merged_docs:
            state_update["retrieved_docs"] = merged_docs
            state_update["retrieval_confidence"] = best_confidence

        # Root cause: the acting LLM alone decides when to stop calling tools,
        # and small/local models don't reliably follow "stop once sufficient"
        # / "don't repeat calls" prompt rules — they keep re-emitting tool_calls
        # (often exact duplicates) past the point the plan is already
        # deterministically satisfied. reflect_final already verifies this
        # deterministically, but only once the LLM itself stops requesting
        # tools. Run the same check here after every tool round so a
        # completed plan short-circuits immediately instead of waiting on
        # the LLM to notice.
        probe_state = {**state, **state_update, "observations": all_observations}
        ready, reasoning = _verify_execution(_build_execution_summary(probe_state))
        if ready:
            logger.info("[tool_node] plan deterministically satisfied after this tool round — forcing finalize: %s", reasoning[:200])
            state_update["force_finalize"] = True

        return state_update
    
    
async def _run_tool(tool, name: str, args: dict) -> dict:
    try:
        raw = await tool.arun(args)
        # Tools return {"ok": bool, "result": {...}, "error": str|None, "tokens": int}.
        # Unwrap the envelope so obs.result is the inner payload (e.g. {"docs": [...], ...}).
        if isinstance(raw, dict) and "result" in raw:
            return {
                "tool": name,
                "arguments": args,
                "result": raw.get("result", {}),
                "error": raw.get("error"),
                "tokens": raw.get("tokens", 0),
            }
        return {"tool": name, "arguments": args, "result": raw, "error": None, "tokens": 0}
    except Exception as exc:
        logger.warning("[_run_tool] %s failed: %s", name, exc)
        return {"tool": name, "arguments": args, "result": {}, "error": str(exc), "tokens": 0}


def route_plan(state: AgentState) -> str:
    if state.get("needs_clarification"):
        return "clarify_interrupt"
    return "think"


async def finalize_node(state: AgentState, ctx: ToolContext) -> dict:
    """Generate final answer if not precomputed; extract LastAnswerObject."""
    with _agent_step("finalize"):
        writer = _writer()
        precomputed = state.get("precomputed_answer", "")
        # The answer model always sees the user's exact wording; only
        # retrieval sees the resolved standalone query.
        query = state.get("original_query", "") or state.get("rewritten_query", "")
        retrieval_query = state.get("rewritten_query", "") or query
        observations = state.get("observations", [])
        docs = state.get("retrieved_docs", [])
        plan = state.get("plan")
        compaction_updates: dict = {}
        answer_usage: Optional[dict] = None

        # Collect every valid chart_generate result up front so both the
        # prompt instructions and the post-generation substitution use the
        # same list, regardless of the precomputed/streamed branch below.
        chart_options: list[dict] = []
        for raw_obs in observations:
            obs = _coerce_observation(raw_obs)
            if obs.tool == "chart_generate" and obs.result.get("chart_option"):
                chart_options.append(obs.result["chart_option"])

        if precomputed:
            final = precomputed
        else:
            context_text = format_context_string(docs, state.get("file_markdown"), db=ctx.db, org_id=ctx.org_id, query_glossary=state.get("abbreviation_glossary", ""))
            # Non-retrieval tool results (code_execute, chart_generate, etc.)
            # are not in retrieved_docs; surface them separately. Retrieval
            # results are already in context_text — don't duplicate.
            non_rag_text = _non_retrieval_observations_text(observations)

            # Chart docs are only appended when the plan intent is "chart" or
            # a chart_generate observation exists.
            plan_intent = plan.intent if isinstance(plan, Plan) else ""
            include_charts = plan_intent == "chart" or bool(chart_options)

            answer_prompt = FINALIZE_ANSWER_PROMPT
            if chart_options:
                # Valid chart JSON already exists — have the model place a
                # marker instead of freehand-writing (and risking malformed) JSON.
                from app.services.prompts.loader import append_chart_placeholder_instructions
                answer_prompt = append_chart_placeholder_instructions(answer_prompt, len(chart_options))
            elif include_charts:
                from app.services.prompts.loader import append_chart_instructions
                answer_prompt = append_chart_instructions(answer_prompt)

            system = FINALIZE_GUARDRAIL_PROMPT + "\n\n" + answer_prompt

            def _build_finalize_user_prompt() -> str:
                # Priority order is explicit: retrieved documents are the
                # evidence, the conversation is the intent. Without this block
                # conversational instructions ("shorter", "in a table",
                # "compare with your last answer") are unanswerable.
                parts = [f"User query: {query}\n\n"]
                if retrieval_query and retrieval_query != query:
                    parts.append(f"Resolved retrieval query: {retrieval_query}\n\n")
                if summary_text:
                    parts.append(f"Earlier conversation summary:\n{summary_text}\n\n")
                if history_text:
                    parts.append(
                        "Conversation so far (intent only — cite nothing from here):\n"
                        f"{history_text}\n\n"
                    )
                parts.append(f"Retrieved context (the only citable evidence):\n{context_text}\n\n")
                if non_rag_text:
                    parts.append(f"Tool results:\n{non_rag_text}\n\n")
                parts.append("Provide a concise, accurate answer.")
                return "".join(parts)

            recent = select_recent_history(state.get("messages", []), max_pairs=get_setting(ctx.db, "AGENT_HISTORY_PAIRS", ctx.org_id))
            history_text = history_to_text(recent)
            summary_text = state.get("compaction_summary") or ""
            user = _build_finalize_user_prompt()

            # Runtime compaction before the generation LLM call. trim_docs=True
            # because this prompt is dominated by retrieved_docs — summarising
            # conversation history cannot fix an evidence-payload overflow.
            compaction_updates, compaction_local = await _compact_if_needed(
                state, user, system_overhead=count_tokens(system), ctx=ctx, trim_docs=True,
            )
            if compaction_local:
                state = {**state, **compaction_local}
                observations = state.get("observations", [])
                docs = state.get("retrieved_docs", docs)
                context_text = format_context_string(docs, state.get("file_markdown"), db=ctx.db, org_id=ctx.org_id, query_glossary=state.get("abbreviation_glossary", ""))
                non_rag_text = _non_retrieval_observations_text(observations)
                recent = select_recent_history(state.get("messages", []), max_pairs=get_setting(ctx.db, "AGENT_HISTORY_PAIRS", ctx.org_id))
                history_text = history_to_text(recent)
                summary_text = state.get("compaction_summary") or ""
                user = _build_finalize_user_prompt()
            try:
                gen_temp = get_setting(ctx.db, "GENERATION_TEMPERATURE", ctx.org_id)
                llm = build_chat_llm(ctx.org_id, ctx.db, role="chat", temperature=gen_temp, streaming=True)
                final = ""
                async for chunk in llm.astream([
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]):
                    # Capture provider-reported usage when the backend sends
                    # it, so reported tokens are measured rather than guessed.
                    chunk_usage = getattr(chunk, "usage_metadata", None)
                    if chunk_usage:
                        answer_usage = {
                            "input_tokens": chunk_usage.get("input_tokens", 0),
                            "output_tokens": chunk_usage.get("output_tokens", 0),
                            "total_tokens": chunk_usage.get("total_tokens", 0),
                        }
                        # Calibrate the token estimator using the exact
                        # prompt_tokens reported by the provider.
                        from app.services.agentic_rag.token_budget import record_usage
                        record_usage(system + user, chunk_usage.get("input_tokens", 0))
                    content = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                    if content:
                        writer({"event": "token", "content": content})
                        final += content
                if not final:
                    final = "I'm sorry, I couldn't generate a response at this time."
            except Exception as exc:
                logger.warning("[finalize_node] generation failed: %s", exc)
                final = "I'm sorry, I couldn't generate a response at this time."

        # Extraction (Call 4, below) wants the raw marker text — not the
        # substituted chart JSON — so it isn't fed a large embedded blob.
        # Keep that copy before substituting, then rewrite citations and
        # stream the display-ready answer immediately, without waiting on
        # Call 4 (last_answer_object extraction) or Call 5 (confidence score).
        raw_for_extraction = final
        # Append a readable summary of chart data so the extraction LLM can
        # see the actual values (the marker text alone says "[[CHART_1]]").
        if chart_options:
            chart_parts: list[str] = []
            for i, opt in enumerate(chart_options, 1):
                series = opt.get("series", [])
                xaxis = opt.get("xAxis", {})
                labels = xaxis.get("data", []) if isinstance(xaxis, dict) else []
                for s in series:
                    values = s.get("data", [])
                    pairs = ", ".join(
                        f"{labels[j]}={values[j]}"
                        for j in range(min(len(labels), len(values)))
                    )
                    chart_parts.append(f"Chart {i} ({s.get('type', 'chart')}): {pairs}")
            raw_for_extraction += "\n\nChart data:\n" + "\n".join(chart_parts)
        final = _substitute_chart_markers(final, chart_options)
        final, cited_doc_indices = normalize_citations(final, docs)
        cited_docs = [docs[i - 1] for i in cited_doc_indices]
        writer({"event": "answer_rewrite", "content": final, "citations": cited_docs})

        # Build a lightweight LastAnswerObject. Try LLM extraction for data/chart.
        lao = LastAnswerObject(
            summary=final[:500],
            key_points=[s.strip("- ") for s in final.splitlines() if s.strip()][:8],
            data=None,
            citations=[],
            chart_option=None,
            chart_options=[],
            followups=[],
        )
    
        # Use a structured extraction for data if any numeric content; otherwise cheap.
        llm_query = build_chat_llm(ctx.org_id, ctx.db, role="query", temperature=0.0)
        extracted: Optional[LastAnswerObject] = None
        for attempt in range(2):
            try:
                raw = await llm_query.ainvoke([
                    {"role": "user", "content": LAST_ANSWER_EXTRACT_PROMPT.format(answer=raw_for_extraction[:3000])},
                ])
                block = _extract_json_block(str(raw.content))
                if block:
                    extracted = LastAnswerObject.model_validate_json(block)
                    break
            except Exception as exc:
                logger.debug("[finalize_node] last_answer_object extraction attempt %d failed: %s", attempt + 1, exc)
        if extracted:
            lao = extracted

        lao.chart_options = chart_options
        lao.chart_option = chart_options[0] if chart_options else None
    
        writer({"event": "last_answer", "last_answer_object": lao.model_dump()})

        # Persist the assistant turn into the checkpointed conversation.
        # Without this the thread holds user questions only, so every
        # downstream consumer of history (reference resolution, think,
        # compaction) reads half a conversation and invents the rest.
        # The `add_messages` reducer appends, and the id is stable per
        # assistant message row so a resume-after-interrupt replaces rather
        # than duplicates the turn.
        message_id = state.get("message_id")
        answer_message = AIMessage(
            content=final,
            id=f"assistant-{message_id}" if message_id else None,
        )

        updates = {
            **compaction_updates,
            "final_answer": final,
            "answer": final,
            "last_answer_object": lao,
            "retrieved_docs": docs,
            "cited_doc_indices": cited_doc_indices,
            "messages": [*compaction_updates.get("messages", []), answer_message],
        }
        if answer_usage:
            updates["answer_usage"] = answer_usage
        return updates
    
    
async def save_memory_node(state: AgentState, ctx: ToolContext) -> dict:
    """Persist final answer, last_answer_object, and tool calls to the DB message row."""
    with _agent_step("save_memory"):
        message_id = state.get("message_id")
        if not message_id:
            return {}
    
        msg = ctx.db.query(Message).filter(Message.id == message_id).first()
        if not msg:
            return {}
    
        msg.content = state.get("final_answer", "")
        plan = state.get("plan")
        if plan:
            msg.plan = plan.model_dump() if isinstance(plan, Plan) else plan
        lao = state.get("last_answer_object")
        if lao:
            msg.last_answer_object = lao.model_dump() if isinstance(lao, LastAnswerObject) else lao
        observations = state.get("observations", [])
        msg.tool_calls = [_coerce_observation(obs).model_dump() for obs in observations]
        msg.final_confidence = state.get("final_confidence")
        msg.final_confidence_level = state.get("confidence_level")
        msg.faithfulness = state.get("faithfulness")
        msg.completeness = state.get("completeness")
    
        try:
            ctx.db.commit()
        except Exception as exc:
            logger.warning("[save_memory_node] failed to commit message updates: %s", exc)
            ctx.db.rollback()
        return {}
    
    
async def reflect_node(state: AgentState, ctx: ToolContext) -> dict:
    """Periodic reflection: concrete deterministic recovery rules only."""
    with _agent_step("reflect"):
        iteration = state.get("iteration", 0)
        if iteration == 0 or iteration % get_setting(ctx.db, "AGENT_REFLECT_EVERY", ctx.org_id) != 0:
            return {}
    
        observations = state.get("observations", [])
        counts = state.get("tool_call_count", {})
        precomputed: list[dict] = []
    
        # Concrete replanning rules =================================================
        # NOTE: rag_retrieve now runs its own internal graduated relaxation ladder
        # (loosening leg/reranker thresholds across multiple levels, see
        # tools/rag_retrieve.py) before returning. A zero-doc / insufficient
        # observation therefore already reflects the best the retrieval system
        # could do for that exact query string — automatically re-issuing the
        # *same* query here would just repeat the same ladder and return
        # identical results. Do not auto-retry rag_retrieve with an unchanged
        # query; leave that decision (and any query reformulation) to LLM
        # discretion below, which sees the sufficiency/doc-count signal.
        for raw_obs in observations:
            obs = _coerce_observation(raw_obs)
            if obs.tool == "chart_generate" and obs.error:
                if counts.get("extract_data", 0) < get_setting(ctx.db, "AGENT_MAX_RETRIEVALS", ctx.org_id):
                    precomputed.append({
                        "tool": "extract_data",
                        "arguments": {"source": "retrieved_docs"},
                    })
            if obs.tool == "code_execute" and obs.error:
                if counts.get("code_execute", 0) < get_setting(ctx.db, "AGENT_MAX_CODE_EXEC", ctx.org_id):
                    precomputed.append({
                        "tool": "extract_data",
                        "arguments": {"source": "retrieved_docs"},
                    })
    
        if precomputed:
            return {"precomputed_tool_calls": precomputed}

        # No concrete rule fired — nothing to do. Termination is decided
        # deterministically (tool_node's post-round check and think_node's
        # pre-think check, both backed by _verify_execution), not by LLM
        # discretion here.
        return {}
    
    
async def clarify_interrupt_node(state: AgentState) -> dict:
    """Pause execution and ask the user for clarification; resumes on response."""
    with _agent_step("clarify_interrupt"):
        plan = state.get("plan") or Plan()
        question = ""
        if isinstance(plan, Plan):
            question = plan.clarification_question or ""
        if not question:
            question = "Could you clarify what you need?"

        # No try/except and no pre-emitted custom event here.
        # `interrupt()` raises GraphInterrupt, which subclasses Exception —
        # catching it swallowed the pause and let the graph run on with an
        # empty answer. Emitting a custom "interrupt" event *before* the call
        # also let the consumer close the stream before LangGraph could
        # persist the interrupt checkpoint, so the resume had nothing to
        # resume. The interrupt is surfaced from the graph's own
        # `__interrupt__` update in agent_runner instead.
        user_response = interrupt({"question": question})

        response_text = str(user_response) if user_response else ""
        return {
            # add_messages appends: return only the new message.
            "messages": [HumanMessage(content=response_text)],
            "clarification_response": response_text,
            "clarification_count": state.get("clarification_count", 0) + 1,
            "needs_clarification": False,
        }
    
    
async def answer_scoring_node(state: AgentState, ctx: "ToolContext") -> dict:
    """Evaluate the final answer quality."""
    with _agent_step("answer_scoring"):
        return await answer_evaluation_node(state, ctx=ctx)
    
    
def _build_execution_summary(state: AgentState) -> dict:
    """Build a structured execution summary for deterministic verification."""
    plan = state.get("plan") or Plan()
    observations = state.get("observations", [])
    counts = dict(state.get("tool_call_count", {}))
    iteration = state.get("iteration", 0)

    # Map subtask tool_hints to whether we have a matching observation.
    # Matching is by *count*, not by mere presence: three subtasks that all
    # hint rag_retrieve need three distinct successful retrievals. Matching on
    # presence alone marked a three-part question complete after one retrieval
    # and silently capped multi-hop questions at a single hop.
    coerced = [_coerce_observation(o) for o in observations]
    successful_by_tool: dict[str, int] = {}
    for o in coerced:
        if not o.error:
            successful_by_tool[o.tool] = successful_by_tool.get(o.tool, 0) + 1
    any_successful = sum(successful_by_tool.values())

    consumed: dict[str, int] = {}
    consumed_any = 0
    subtask_status = []
    for st in plan.subtasks:
        hint = st.tool_hint
        if hint == "none":
            # The subtask needs no tool call (e.g. pure conversation).
            completed = True
        elif hint == "any":
            completed = consumed_any < any_successful
            if completed:
                consumed_any += 1
        else:
            used = consumed.get(hint, 0)
            completed = used < successful_by_tool.get(hint, 0)
            if completed:
                consumed[hint] = used + 1
        subtask_status.append({
            "id": st.id,
            "description": st.description,
            "tool_hint": hint,
            "completed": completed,
        })

    # Retrieval stats.
    retrieval_queries = counts.get("rag_retrieve", 0)
    total_docs = 0
    for o in coerced:
        if o.tool == "rag_retrieve" and not o.error:
            total_docs += len(o.result.get("docs", []))

    # Tool failures.
    failures = []
    for o in coerced:
        if o.error:
            failures.append({"tool": o.tool, "error": o.error})

    # Remaining retrieval budget.
    from app.db.session import SessionLocal
    org_id = state.get("org_id")
    _db = SessionLocal()
    try:
        max_retrievals = get_setting(_db, "AGENT_MAX_RETRIEVALS", org_id)
        max_iterations = get_setting(_db, "AGENT_MAX_ITERATIONS", org_id)
        max_wall_seconds = get_setting(_db, "AGENT_MAX_WALL_SECONDS", org_id)
    finally:
        _db.close()
    retrieval_budget_left = max_retrievals - retrieval_queries

    started_at = state.get("started_at")
    elapsed_seconds = round(time.monotonic() - started_at, 1) if started_at else 0.0

    return {
        "user_goal": state.get("original_query", ""),
        "intent": plan.intent,
        "subtasks": subtask_status,
        "retrieval": {
            "queries": retrieval_queries,
            "documents": total_docs,
        },
        "tool_failures": failures,
        "remaining_budget": {
            "retrieval": retrieval_budget_left,
            "iterations": max_iterations - iteration,
            "seconds": round(max_wall_seconds - elapsed_seconds, 1),
        },
    }


def _verify_execution(summary: dict) -> tuple[bool, str]:
    """Deterministic execution verification.

    Returns (ready, reasoning). ready=True means the agent has done enough
    work to generate an answer. ready=False means a required step is missing
    and another iteration is likely to help.
    """
    issues = []

    # 1. No observations at all — nothing was attempted.
    if not summary["subtasks"] and summary["retrieval"]["queries"] == 0:
        # No plan subtasks and no retrieval — could be a conversation query.
        # If intent is conversation, that's fine.
        if summary.get("intent") not in ("conversation",):
            issues.append("No tool calls were made and no subtasks were planned.")
        else:
            return True, "Conversation intent — no tools needed."

    # 2. Uncompleted subtasks that still have budget.
    for st in summary["subtasks"]:
        if not st["completed"]:
            issues.append(f"Subtask '{st['id']}' ({st['description'][:60]}) has no successful tool result.")

    # 3. Retrieval returned zero docs and budget remains.
    if summary["retrieval"]["queries"] > 0 and summary["retrieval"]["documents"] == 0:
        if summary["remaining_budget"]["retrieval"] > 0:
            issues.append("Retrieval returned 0 documents; another query may help.")
        else:
            # No budget left — can't fix this, proceed with what we have.
            pass

    # 4. Tool failures that could be retried.
    for f in summary["tool_failures"]:
        tool = f["tool"]
        if "Budget exceeded" in f["error"]:
            continue  # Budget-exceeded failures can't be retried.
        issues.append(f"Tool '{tool}' failed: {f['error'][:80]}")

    if issues:
        return False, "; ".join(issues)
    return True, "All planned steps have supporting tool results."


async def reflect_final_node(state: AgentState, ctx: ToolContext) -> dict:
    """Final pre-finalize verification: deterministic execution completeness check."""
    with _agent_step("reflect_final"):
        iteration = state.get("iteration", 0)
        max_iter = get_setting(ctx.db, "AGENT_MAX_ITERATIONS", ctx.org_id)

        summary = _build_execution_summary(state)
        ready, reasoning = _verify_execution(summary)

        # Force ready when iteration cap OR wall-clock budget is reached — no
        # more retries possible/worthwhile.
        if not ready and (iteration >= max_iter or _wall_clock_exceeded(state)):
            logger.info("[reflect_final_node] not ready but iteration/wall-clock cap reached (%d/%d) — forcing finalize", iteration, max_iter)
            ready = True
            reasoning = f"Forced finalize at iteration/time cap. Pending issues: {reasoning}"

        logger.info("[reflect_final_node] ready=%s reasoning=%s", ready, reasoning[:200])

        writer = _writer()
        writer({"event": "progress", "phase": "reflect_final", "ready": ready, "reasoning": reasoning})
        return {"reflection_final": {"ready": ready, "reasoning": reasoning}}
    
    
def build_agent_graph(ctx: ToolContext):
    """Compile and return the agent loop graph."""
    graph = StateGraph(AgentState)

    graph.add_node("load_context", partial(load_context_node, ctx=ctx))
    # Resolve query-role LLM config for rewrite_query_node
    from app.services.agentic_rag.llm_factory import get_org_llm
    query_cfg = get_org_llm(ctx.org_id, ctx.db, role="query")
    graph.add_node("rewrite_query", partial(rewrite_query_node,
                                            api_base=query_cfg["api_base"],
                                            api_key=query_cfg["api_key"],
                                            query_model=query_cfg["model_name"],
                                            db=ctx.db,
                                            org_id=ctx.org_id))
    graph.add_node("expand_query", partial(expand_query_node, db=ctx.db, org_id=ctx.org_id))
    graph.add_node("plan", partial(plan_node, ctx=ctx))
    graph.add_node("clarify_interrupt", clarify_interrupt_node)
    graph.add_node("think", partial(think_node, ctx=ctx))
    graph.add_node("tool", partial(tool_node, ctx=ctx))
    graph.add_node("reflect", partial(reflect_node, ctx=ctx))
    graph.add_node("reflect_final", partial(reflect_final_node, ctx=ctx))
    graph.add_node("finalize", partial(finalize_node, ctx=ctx))
    graph.add_node("answer_scoring", partial(answer_scoring_node, ctx=ctx))
    graph.add_node("save_memory", partial(save_memory_node, ctx=ctx))

    graph.set_entry_point("load_context")
    graph.add_edge("load_context", "expand_query")
    graph.add_edge("expand_query", "rewrite_query")
    graph.add_edge("rewrite_query", "plan")
    graph.add_conditional_edges("plan", route_plan)
    # Back through expansion + rewrite, not straight to plan: the clarification
    # answer has to reach the retrieval query, which was computed from the
    # original ambiguous message.
    graph.add_edge("clarify_interrupt", "expand_query")
    graph.add_conditional_edges("think", route_think)
    graph.add_conditional_edges("tool", route_tool)
    graph.add_edge("reflect", "think")
    graph.add_conditional_edges("reflect_final", route_reflect_final)
    graph.add_edge("finalize", "answer_scoring")
    graph.add_edge("answer_scoring", "save_memory")
    graph.add_edge("save_memory", END)

    checkpointer = getattr(ctx.redis_memory, "checkpointer", None) if ctx.redis_memory else None
    return graph.compile(checkpointer=checkpointer)
