"""Think node for agentic-v2.

One LLM call with all tools bound. The LLM either:
- Emits tool calls → graph goes to tool node, then loops back.
- Emits plain text (no tool calls) → that text IS the final answer.

No separate planner, sufficiency checker, or finalizer. The LLM reasons
about what to do, calls tools until it has enough evidence, then writes
the answer as its final message.
"""

from __future__ import annotations

import logging
import time

from langchain_core.messages import AIMessage

from app.services.agentic_rag.kb_profile import format_profile_summary
from app.services.agentic_rag.llm_factory import build_chat_llm
from app.services.agentic_rag.nodes import _agent_step, history_to_text, select_recent_history
from app.services.agentic_rag.prompts_v2 import get_agent_v2_system_prompt
from app.services.agentic_rag.tool_call_parser import parse_think_response
from app.services.agentic_rag.tools import applicable_tools
from app.services.agentic_rag.token_budget import count_tokens
from app.services.infrastructure import is_cancelled
from app.services.settings_service import get_setting

from ..agent_graph.compaction import _compact_if_needed
from ..agent_graph.helpers import _coerce_observation, _total_tool_budget, _wall_clock_exceeded, _writer
from ..agent_graph.observations import (
    _observations_metadata_text,
    _tried_search_queries,
)

logger = logging.getLogger(__name__)


def _format_retrieved_docs_for_think(docs: list[dict], max_docs: int = 10, max_chars: int = 400) -> str:
    """Format retrieved docs with content previews for the think prompt.

    In v2, the think node IS the finalizer — the LLM needs to see the actual
    evidence content to decide if it's sufficient and to write the answer.
    """
    if not docs:
        return ""
    parts: list[str] = []
    for i, doc in enumerate(docs[:max_docs], 1):
        if not isinstance(doc, dict):
            continue
        content = (doc.get("page_content") or "")[:max_chars]
        meta = doc.get("metadata") or {}
        title = meta.get("title") or meta.get("file_name") or "Unknown"
        score = meta.get("_reranker_score", meta.get("score", 0))
        score_str = f" score={score:.3f}" if score else ""
        parts.append(f"[E{i}] {title}{score_str}\n  {content}")
    return "\n\n".join(parts)


def _available_tools_text(tools: list) -> str:
    """Return a short 'Available this turn' list of tool names."""
    names = [t.name for t in tools]
    return "\n".join(f"- {name}" for name in names)


def _build_v2_user_prompt(
    iteration: int,
    tool_budget: int,
    tool_calls_used: int,
    original: str,
    summary_text: str,
    history_text: str,
    lao,
    observations: list,
    retrieved_docs: list,
    available_tools_text: str,
    kb_profile_text: str,
    file_markdown: str | None,
) -> str:
    """Build the user-turn prompt for the think LLM call."""
    lao_text = ""
    if lao and hasattr(lao, "summary") and lao.summary:
        lao_text = f"  Previous answer summary: {lao.summary[:300]}\n"
        if lao.key_points:
            lao_text += f"  Key points: {'; '.join(lao.key_points[:5])}\n"

    tried_queries = _tried_search_queries(observations)
    tried_queries_text = (
        f"  Already tried (do NOT repeat these exact queries): {tried_queries}\n"
        if tried_queries else ""
    )

    obs_text = _observations_metadata_text(observations)
    docs_text = _format_retrieved_docs_for_think(retrieved_docs)

    parts: list[str] = []
    if kb_profile_text:
        parts.append(f"{kb_profile_text}\n")
    if summary_text:
        parts.append(f"Earlier conversation summary:\n{summary_text}\n\n")
    if history_text:
        parts.append(f"Conversation history (recent turns, not citable):\n{history_text}\n\n")
    if lao_text:
        parts.append(f"Previous answer context:\n{lao_text}\n\n")
    if file_markdown:
        parts.append(f"Attached file metadata:\n{file_markdown[:2000]}\n\n")
    if tried_queries_text:
        parts.append(tried_queries_text)
    if obs_text:
        parts.append(f"Tool observations so far:\n{obs_text}\n\n")
    if docs_text:
        parts.append(f"Retrieved evidence (cite these as [N](N) in your answer):\n{docs_text}\n\n")
    parts.append(f"Available this turn:\n{available_tools_text}\n\n")
    parts.append(f"User message: {original}\n")

    # Forceful reminder: if the user asked to create/generate a document and
    # create_office_document hasn't been called yet, remind the LLM to call it.
    _office_keywords = ("create", "generate", "make", "build", "produce")
    _office_targets = ("document", "word", "docx", "powerpoint", "pptx", "slide",
                       "excel", "xlsx", "spreadsheet", "presentation", "deck")
    original_lower = original.lower()
    asks_for_office = any(k in original_lower for k in _office_keywords) and \
                      any(t in original_lower for t in _office_targets)
    office_called = any(
        _coerce_observation(o).tool == "create_office_document"
        for o in observations
    )
    if asks_for_office and not office_called and iteration < max_iter:
        parts.append(
            "\n⚠ IMPORTANT: The user asked to CREATE a document. You MUST call "
            "create_office_document to actually create the file. Do NOT just describe "
            "what you would create — call the tool."
        )

    remaining = tool_budget - tool_calls_used
    parts.append(f"\nTool calls remaining: {remaining}/{tool_budget}\n")
    if remaining <= 0:
        parts.append(
            "You have exhausted your tool-call budget. Write your answer now using the evidence gathered. "
            "Do not call any more tools."
        )
    else:
        parts.append(
            "Call the next tool(s) to gather evidence, or write your final answer as plain text "
            "(no tool calls) when you have enough to respond. "
            "When writing your answer, cite evidence using [N](N) format where N matches the evidence item number."
        )

    return "".join(parts)


async def think_node_v2(state, ctx) -> dict:
    """Unified think node: LLM reasons and either calls tools or writes the answer."""
    with _agent_step("think"):
        ctx.state = state
        iteration = state.get("iteration", 0) + 1
        tool_budget = _total_tool_budget(ctx.db, ctx.org_id)
        tool_calls_used = sum(state.get("tool_call_counts", {}).values())

        # Wall-clock check: force finalize if time exceeded.
        if _wall_clock_exceeded(state):
            logger.debug("[think_v2] wall clock exceeded, forcing answer")
            return {"iteration": iteration, "tool_calls": [], "force_finalize": True}

        query = state.get("original_query", "")
        observations = state.get("observations", [])
        tools = applicable_tools(ctx)
        available_tools_text = _available_tools_text(tools)

        recent = select_recent_history(
            state.get("messages", []),
            max_pairs=get_setting(ctx.db, "AGENT_HISTORY_PAIRS", ctx.org_id),
        )
        history_text = history_to_text(recent)
        summary_text = state.get("compaction_summary") or ""
        kb_profile_text = format_profile_summary(state.get("kb_profile", {}))

        system = get_agent_v2_system_prompt()
        retrieved_docs = state.get("retrieved_docs", [])
        user = _build_v2_user_prompt(
            iteration, tool_budget, tool_calls_used, query, summary_text, history_text,
            state.get("last_answer_object"), observations, retrieved_docs,
            available_tools_text, kb_profile_text, state.get("file_markdown"),
        )

        # Compaction: if the prompt exceeds context budget, compact before calling LLM.
        compaction_updates, compaction_local = await _compact_if_needed(
            state, user, system_overhead=count_tokens(system), ctx=ctx, trim_docs=True,
        )
        if compaction_local:
            state = {**state, **compaction_local}
            observations = state.get("observations", [])
            retrieved_docs = state.get("retrieved_docs", [])
            recent = select_recent_history(
                state.get("messages", []),
                max_pairs=get_setting(ctx.db, "AGENT_HISTORY_PAIRS", ctx.org_id),
            )
            history_text = history_to_text(recent)
            summary_text = state.get("compaction_summary") or ""
            user = _build_v2_user_prompt(
                iteration, tool_budget, tool_calls_used, query, summary_text, history_text,
                state.get("last_answer_object"), observations, retrieved_docs,
                tools_text, kb_profile_text, state.get("file_markdown"),
            )

        mode = get_setting(ctx.db, "TOOL_CALL_MODE", None)

        # Cancellation check before the expensive LLM call.
        # Return no tool calls and empty precomputed_answer so the graph
        # routes to post_process, which detects cancellation and skips
        # generation/persistence entirely.
        chat_id = ctx.chat_id if ctx is not None else None
        if chat_id is not None and is_cancelled(chat_id):
            logger.debug("[think_v2] cancelled before LLM call | chat_id=%s", chat_id)
            return {"iteration": iteration, "tool_calls": [], "precomputed_answer": ""}

        # Emit "thinking..." event so the frontend shows the thinking indicator.
        writer = _writer()
        writer({"event": "thinking", "content": "", "done": False})
        think_start = time.monotonic()

        try:
            tool_temp = get_setting(ctx.db, "TOOL_CALL_TEMPERATURE", ctx.org_id)
            if mode == "json_text":
                llm = build_chat_llm(ctx.org_id, ctx.db, role="chat", temperature=tool_temp)
                resp = await llm.ainvoke([
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ])
            else:
                llm = build_chat_llm(ctx.org_id, ctx.db, role="chat", temperature=tool_temp)
                resp = await llm.bind_tools(tools).ainvoke([
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ])
        except Exception as exc:
            logger.warning("[think_v2] LLM call failed: %s", exc)
            return {"iteration": iteration, "tool_calls": [], "force_finalize": True}

        # Emit "thought for N seconds" with reasoning content (if any).
        think_elapsed = time.monotonic() - think_start
        parsed = parse_think_response(resp, mode=mode)
        if parsed.reasoning:
            writer({
                "event": "thinking",
                "content": parsed.reasoning,
                "done": True,
                "elapsed": round(think_elapsed, 1),
            })
        else:
            # No reasoning content — close the thinking indicator with
            # elapsed time but empty content (non-thinking model).
            writer({
                "event": "thinking",
                "content": "",
                "done": True,
                "elapsed": round(think_elapsed, 1),
            })

        tool_calls = parsed.tool_calls
        final_answer_text = parsed.final_answer

        # If tool budget exhausted, force answer even if LLM emitted tool calls.
        budget_exhausted = tool_calls_used >= tool_budget
        if budget_exhausted:
            tool_calls = []

        if tool_calls:
            return {**compaction_updates, "iteration": iteration, "tool_calls": tool_calls}

        # No tool calls → the LLM wrote the answer (or signaled final_answer).
        # If final_answer is a string, it's the answer text (Tier 3 fallback).
        # If final_answer is True (boolean), the LLM signaled done but didn't
        # write text — the post_process node will generate from evidence.
        answer_text = ""
        if isinstance(final_answer_text, str) and final_answer_text.strip():
            answer_text = final_answer_text
        return {
            **compaction_updates,
            "iteration": iteration,
            "tool_calls": [],
            "precomputed_answer": answer_text,
        }


def route_think_v2(state) -> str:
    """Route after think: tool if there are tool calls, otherwise post_process."""
    from app.db.session import SessionLocal
    from app.services.settings_service import get_setting as _gs
    org_id = state.get("org_id")
    _db = SessionLocal()
    try:
        tool_budget = _gs(_db, "AGENT_TOTAL_TOOL_BUDGET", org_id)
    finally:
        _db.close()

    tool_calls_used = sum(state.get("tool_call_counts", {}).values())

    if tool_calls_used >= tool_budget or _wall_clock_exceeded(state):
        return "post_process"

    if state.get("tool_calls"):
        return "tool"

    # No tool calls and budget remains → the LLM wrote the answer.
    return "post_process"
