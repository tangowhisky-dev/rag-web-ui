"""Reflection, clarification, scoring, and final-verification nodes.

reflect_node: periodic deterministic recovery rules (no LLM call).
clarify_interrupt_node: pauses execution to ask the user for clarification;
  resumes on response via LangGraph interrupt/resume.
answer_scoring_node: delegates to the LLM-based answer evaluation.
reflect_final_node: deterministic execution-completeness check before
  finalize; sends the agent back to think if required steps are missing.

Also contains the execution-summary builder and verifier used by both
think_node and tool_node for early-exit / force-finalize decisions.
"""

from __future__ import annotations

import logging
import time

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from app.services.agentic_rag.nodes import _agent_step, answer_evaluation_node
from app.services.agentic_rag.schemas import Plan
from app.services.settings_service import get_setting

from .helpers import _coerce_observation, _wall_clock_exceeded, _writer

logger = logging.getLogger(__name__)


async def reflect_node(state, ctx) -> dict:
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


async def clarify_interrupt_node(state) -> dict:
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


async def answer_scoring_node(state, ctx: "ToolContext") -> dict:
    """Evaluate the final answer quality."""
    with _agent_step("answer_scoring"):
        return await answer_evaluation_node(state, ctx=ctx)


def _count_successful_by_tool(coerced):
    successful_by_tool: dict[str, int] = {}
    for o in coerced:
        if not o.error:
            successful_by_tool[o.tool] = successful_by_tool.get(o.tool, 0) + 1
    return successful_by_tool, sum(successful_by_tool.values())


def _build_subtask_status(plan, successful_by_tool, any_successful):
    consumed: dict[str, int] = {}
    consumed_any = 0
    subtask_status = []
    for st in plan.subtasks:
        hint = st.tool_hint
        if hint == "none":
            completed = True
        elif hint == "any":
            completed = consumed_any < any_successful
            if completed:
                consumed_any += 1
        else:
            used = consumed.get(hint, 0)
            # kb_search_documents satisfies rag_retrieve hints — it's a
            # document-level retrieval that covers the same intent.
            available = successful_by_tool.get(hint, 0)
            if hint == "rag_retrieve" and not available:
                available = successful_by_tool.get("kb_search_documents", 0)
            completed = used < available
            if completed:
                consumed[hint] = used + 1
        subtask_status.append({
            "id": st.id,
            "description": st.description,
            "tool_hint": hint,
            "completed": completed,
        })
    return subtask_status


def _retrieval_doc_count(coerced):
    total_docs = 0
    for o in coerced:
        if o.tool in ("rag_retrieve", "kb_search_documents") and not o.error:
            total_docs += len(o.result.get("docs", []))
    return total_docs


def _collect_tool_failures(coerced):
    failures = []
    for o in coerced:
        if o.error:
            failures.append({"tool": o.tool, "error": o.error})
    return failures


def _build_execution_summary(state) -> dict:
    """Build a structured execution summary for deterministic verification."""
    plan = state.get("plan") or Plan()
    observations = state.get("observations", [])
    counts = dict(state.get("tool_call_count", {}))
    iteration = state.get("iteration", 0)

    coerced = [_coerce_observation(o) for o in observations]
    successful_by_tool, any_successful = _count_successful_by_tool(coerced)
    subtask_status = _build_subtask_status(plan, successful_by_tool, any_successful)

    retrieval_queries = counts.get("rag_retrieve", 0)
    total_docs = _retrieval_doc_count(coerced)
    failures = _collect_tool_failures(coerced)

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


async def reflect_final_node(state, ctx) -> dict:
    """Final pre-finalize verification: deterministic execution completeness check."""
    with _agent_step("reflect_final"):
        iteration = state.get("iteration", 0)
        max_iter = get_setting(ctx.db, "AGENT_MAX_ITERATIONS", ctx.org_id)

        summary = _build_execution_summary(state)
        ready, reasoning = _verify_execution(summary)

        # Force ready when iteration cap OR wall-clock budget is reached — no
        # more retries possible/worthwhile.
        if not ready and (iteration >= max_iter or _wall_clock_exceeded(state)):
            logger.debug("[reflect_final_node] not ready but iteration/wall-clock cap reached (%d/%d) — forcing finalize", iteration, max_iter)
            ready = True
            reasoning = f"Forced finalize at iteration/time cap. Pending issues: {reasoning}"

        logger.debug("[reflect_final_node] ready=%s reasoning=%s", ready, reasoning[:200])

        writer = _writer()
        writer({"event": "progress", "phase": "reflect_final", "ready": ready, "reasoning": reasoning})
        return {"reflection_final": {"ready": ready, "reasoning": reasoning}}
