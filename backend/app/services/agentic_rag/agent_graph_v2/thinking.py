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

from langchain_core.messages import AIMessage, AIMessageChunk

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
from ..agent_graph.helpers import _coerce_observation, _emit_timeline, _total_tool_budget, _wall_clock_exceeded, _writer, debug_emit
from ..agent_graph.observations import (
    _observations_metadata_text,
    _prune_contiguous_overlaps,
    _tried_search_queries,
)
from ..utils import _authority_markers, group_docs_by_document

logger = logging.getLogger(__name__)


def _format_retrieved_docs_for_think(docs: list[dict], max_docs: int = 20, max_chars: int = 400) -> str:
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
        markers = _authority_markers(meta)
        marker_str = f" [{', '.join(markers)}]" if markers else ""
        parts.append(f"[E{i}] {title}{score_str}{marker_str}\n  {content}")
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
        parts.append(f"Retrieved evidence (cite these as [E1], [E2], etc. in your answer):\n{docs_text}\n\n")
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
    if asks_for_office and not office_called and iteration < tool_budget:
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
            "When writing your answer, cite evidence using [E1], [E2], etc. where the number matches the evidence item number."
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
        # Group chunks by document (consecutive chunks together) and prune
        # chunking overlap so the LLM sees clean, ordered evidence.
        retrieved_docs = group_docs_by_document(retrieved_docs)
        retrieved_docs = _prune_contiguous_overlaps(retrieved_docs)
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
            retrieved_docs = group_docs_by_document(retrieved_docs)
            retrieved_docs = _prune_contiguous_overlaps(retrieved_docs)
            recent = select_recent_history(
                state.get("messages", []),
                max_pairs=get_setting(ctx.db, "AGENT_HISTORY_PAIRS", ctx.org_id),
            )
            history_text = history_to_text(recent)
            summary_text = state.get("compaction_summary") or ""
            user = _build_v2_user_prompt(
                iteration, tool_budget, tool_calls_used, query, summary_text, history_text,
                state.get("last_answer_object"), observations, retrieved_docs,
                available_tools_text, kb_profile_text, state.get("file_markdown"),
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

        # Debug stream: the literal (post-compaction) prompt the think LLM
        # consumes — obs_text, evidence preview, history — so evaluators can
        # verify the agent saw what it needed.
        debug_emit("think_input", {
            "iteration": iteration,
            "prompt": user[:12000],
            "n_observations": len(observations),
            "n_retrieved_docs": len(retrieved_docs),
        })

        # Emit timeline thinking step — inline in the CoT at its actual position.
        writer = _writer()
        think_step_id = _emit_timeline(type="thinking", content="", status="active")
        think_start = time.monotonic()

        try:
            tool_temp = get_setting(ctx.db, "TOOL_CALL_TEMPERATURE", ctx.org_id)
            if mode == "json_text":
                llm = build_chat_llm(ctx.org_id, ctx.db, role="chat", temperature=tool_temp, streaming=True)
                stream = llm.astream([
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ])
            else:
                llm = build_chat_llm(ctx.org_id, ctx.db, role="chat", temperature=tool_temp, streaming=True)
                stream = llm.bind_tools(tools).astream([
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ])

            # Stream content tokens to the frontend as they arrive.
            # Tool-call chunks (native mode) carry no content, so no tokens
            # are emitted for tool-call iterations — only for the final
            # answer iteration.
            accumulated: AIMessageChunk | None = None
            reasoning_accumulated = ""
            finalize_phase_id: str | None = None
            async for chunk in stream:
                if chat_id is not None and is_cancelled(chat_id):
                    logger.debug("[think_v2] cancelled during LLM stream | chat_id=%s", chat_id)
                    break
                if not isinstance(chunk, AIMessageChunk):
                    continue
                content = chunk.content if isinstance(chunk.content, str) else ""
                if content:
                    # Emit "Finalizing answer" phase before the first content
                    # token so the timeline shows the phase BEFORE the answer
                    # starts streaming (not after, which was the bug when
                    # post_process emitted it).
                    if finalize_phase_id is None:
                        finalize_phase_id = _emit_timeline(
                            type="phase", label="Finalizing answer", status="active")
                    writer({"event": "token", "content": content})
                # Stream reasoning content live for thinking models that expose
                # it via additional_kwargs.reasoning_content (DeepSeek, Qwen, etc.)
                chunk_reasoning = ""
                if chunk.additional_kwargs:
                    chunk_reasoning = chunk.additional_kwargs.get("reasoning_content", "") or ""
                if chunk_reasoning:
                    reasoning_accumulated += chunk_reasoning
                    _emit_timeline(id=think_step_id, type="thinking", content=reasoning_accumulated, status="active")
                accumulated = chunk if accumulated is None else accumulated + chunk

            # Close the "Finalizing answer" phase if we opened one.
            if finalize_phase_id is not None:
                _emit_timeline(id=finalize_phase_id, type="phase",
                               label="Finalizing answer", status="complete")

            # Reconstruct an AIMessage-like object for the parser.
            # AIMessageChunk supports .content and .tool_calls just like AIMessage.
            resp = accumulated if accumulated is not None else AIMessage(content="")
        except Exception as exc:
            logger.warning("[think_v2] LLM call failed: %s", exc)
            return {"iteration": iteration, "tool_calls": [], "force_finalize": True}

        # Emit "thought for N seconds" with reasoning content (if any).
        think_elapsed = time.monotonic() - think_start
        parsed = parse_think_response(resp, mode=mode)
        # Use parsed.reasoning if available, otherwise fall back to what we
        # accumulated from streaming chunks (some providers don't surface
        # reasoning_content on the final accumulated message).
        final_reasoning = parsed.reasoning or (reasoning_accumulated if reasoning_accumulated else None)
        if final_reasoning:
            _emit_timeline(id=think_step_id, type="thinking", content=final_reasoning,
                           status="complete", elapsed=round(think_elapsed, 1))
        else:
            # No reasoning content — close the step with empty content.
            _emit_timeline(id=think_step_id, type="thinking", content="",
                           status="complete", elapsed=round(think_elapsed, 1))

        tool_calls = parsed.tool_calls
        final_answer_text = parsed.final_answer

        # If tool budget exhausted, force answer even if LLM emitted tool calls.
        budget_exhausted = tool_calls_used >= tool_budget
        if budget_exhausted:
            tool_calls = []

        if tool_calls:
            # Clear any content tokens that were streamed before we detected
            # tool calls (happens in json_text mode where the LLM writes the
            # JSON tool-call as content). In native mode, tool-call chunks
            # carry no content, so nothing was streamed.
            if resp.content:
                writer({"event": "answer_rewrite", "content": "", "citations": []})
            return {**compaction_updates, "iteration": iteration, "tool_calls": tool_calls}

        # No tool calls → the LLM wrote the answer (or signaled final_answer).
        # If final_answer is a string, it's the answer text (Tier 3 fallback).
        # If final_answer is True (boolean), the LLM signaled done but didn't
        # write text — the post_process node will generate from evidence.
        answer_text = ""
        if isinstance(final_answer_text, str) and final_answer_text.strip():
            answer_text = final_answer_text
        elif resp.content:
            # LLM emitted {"final_answer": true} or similar JSON signal as
            # content. Clear the streamed tokens so the fallback generator
            # in post_process starts from a clean slate.
            writer({"event": "answer_rewrite", "content": "", "citations": []})
        return {
            **compaction_updates,
            "iteration": iteration,
            "tool_calls": [],
            "precomputed_answer": answer_text,
            "reasoning_content": final_reasoning or "",
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
