"""Runtime context compaction — referral-style context budget check.

Single guard called before any LLM call with variable context. Three
stages, applied only while the rebuilt prompt is still over budget:

1. Deterministic: shrink tool observations (keep top docs, trim stdout).
2. Evidence packing: drop the lowest-scoring retrieved chunks
   (trim_docs=True, i.e. the finalize prompt only).
3. LLM call: summarise the oldest messages into one summary message.

Returns (graph_updates, local_view). graph_updates speaks the channels'
reducer contracts (__reset__ markers, RemoveMessage). local_view holds the
same data already resolved so the caller can rebuild its prompt without
interpreting reducer markers.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from langchain_core.messages import HumanMessage

from app.core.settings_registry import get_def
from app.services.agentic_rag.llm_factory import build_chat_llm
from app.services.agentic_rag.schemas import Observation
from app.services.agentic_rag.token_budget import count_tokens
from app.services.agentic_rag.utils import format_context_string
from app.services.settings_service import get_setting

from .helpers import _coerce_observation
from .observations import _compact_observations

logger = logging.getLogger(__name__)

# Doc preview length when compacting (shorter than the 800 used in full mode).
_COMPACT_DOC_PREVIEW_CHARS = 200
# Stable id for the conversation-summary message so repeated compactions
# replace the previous summary instead of stacking summaries.
_COMPACTION_SUMMARY_ID = "agentic-compaction-summary"


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
    logger.debug("[_trim_docs_to_budget] dropped %d/%d chunks (~%d tokens)", len(drop), len(docs), freed)
    return [d for i, d in enumerate(docs) if i not in drop]


def _compact_stage1_observations(state, budget):
    updates: dict[str, Any] = {}
    local: dict[str, Any] = {}
    observations = [_coerce_observation(o) for o in state.get("observations", [])]
    compacted_obs = _compact_observations(observations)
    obs_tokens_before = sum(count_tokens(json.dumps(o.result, default=str)) for o in observations)
    obs_tokens_after = sum(count_tokens(json.dumps(o.result, default=str)) for o in compacted_obs)
    savings = obs_tokens_before - obs_tokens_after
    if savings > 0:
        updates["observations"] = [{"__reset__": True}, *compacted_obs]
        local["observations"] = compacted_obs
        budget.used -= savings
        logger.debug("[_compact_if_needed] stage 1 (observations) saved %d tokens", savings)
    return updates, local


def _compact_stage2_docs(state, budget):
    updates: dict[str, Any] = {}
    local: dict[str, Any] = {}
    docs = list(state.get("retrieved_docs", []) or [])
    overflow = budget.used - budget.compaction_threshold
    trimmed = _trim_docs_to_budget(docs, overflow)
    if len(trimmed) < len(docs):
        freed = count_tokens(format_context_string(docs)) - count_tokens(format_context_string(trimmed))
        updates["retrieved_docs"] = trimmed
        local["retrieved_docs"] = trimmed
        budget.used -= max(freed, 0)
        logger.debug("[_compact_if_needed] stage 2 (evidence) saved %d tokens", freed)
    return updates, local


async def _compact_stage3_messages(state, ctx):
    updates: dict[str, Any] = {}
    local: dict[str, Any] = {}
    messages = list(state.get("messages", []))
    keep_recent = get_setting(ctx.db, "COMPACTION_KEEP_RECENT", ctx.org_id) if ctx else get_def("COMPACTION_KEEP_RECENT").default
    if len(messages) > keep_recent:
        message_updates, resolved, summary = await _compact_messages_llm(messages, ctx=ctx)
        if summary is not None:
            updates["messages"] = message_updates
            updates["compaction_summary"] = summary
            local["messages"] = resolved
            local["compaction_summary"] = summary
            logger.debug("[_compact_if_needed] stage 3 (messages) summarized %d old messages",
                        len(messages) - len(resolved) + 1)
    return updates, local


async def _compact_if_needed(
    state,
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

    logger.debug(
        "[_compact_if_needed] over budget | used=%d threshold=%d — compacting",
        budget.used, budget.compaction_threshold,
    )

    updates: dict[str, Any] = {"compaction_triggered": True}
    local: dict[str, Any] = {}

    u1, l1 = _compact_stage1_observations(state, budget)
    updates.update(u1)
    local.update(l1)

    if not budget.needs_compaction():
        return updates, local

    if trim_docs:
        u2, l2 = _compact_stage2_docs(state, budget)
        updates.update(u2)
        local.update(l2)

    if not budget.needs_compaction():
        return updates, local

    u3, l3 = await _compact_stage3_messages(state, ctx)
    updates.update(u3)
    local.update(l3)

    return updates, local
