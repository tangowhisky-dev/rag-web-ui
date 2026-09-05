"""Execution completeness check — moved from reflection.py.

Deterministic verification that the agent has done enough work to generate
an answer. Used by tool_node (after every tool round) and sufficiency_check
(the replacement for reflect_final).

Rewritten for atomic tools: counts search_exact/search_sparse/search_dense
instead of the legacy composite retrieval tool.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from app.services.agentic_rag.schemas import Observation, Plan
from app.services.settings_service import get_setting

from .helpers import _coerce_observation

logger = logging.getLogger(__name__)

# Tools that count as "search" for retrieval budget tracking.
_SEARCH_TOOLS = frozenset({
    "search_exact", "search_sparse", "search_dense",
    "rerank_results", "graph_expand",
})


def _count_successful_by_tool(coerced: list[Observation]) -> tuple[dict[str, int], int]:
    successful_by_tool: dict[str, int] = {}
    for o in coerced:
        if not o.error:
            successful_by_tool[o.tool] = successful_by_tool.get(o.tool, 0) + 1
    return successful_by_tool, sum(successful_by_tool.values())


def _build_subtask_status(plan: Plan, successful_by_tool: dict[str, int], any_successful: int) -> list[dict]:
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
            available = successful_by_tool.get(hint, 0)
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


def _retrieval_hit_count(coerced: list[Observation]) -> int:
    """Count total hits/docs from search and retrieval tools."""
    total = 0
    for o in coerced:
        if o.tool in _SEARCH_TOOLS and not o.error:
            hits = o.result.get("hits", [])
            total += len(hits)
        elif o.tool == "kb_search_documents" and not o.error:
            total += len(o.result.get("docs", []))
    return total


def _collect_tool_failures(coerced: list[Observation]) -> list[dict]:
    failures = []
    for o in coerced:
        if o.error:
            failures.append({"tool": o.tool, "error": o.error})
    return failures


def _build_execution_summary(state: dict) -> dict:
    """Build a structured execution summary for deterministic verification."""
    plan = state.get("plan") or Plan()
    observations = state.get("observations", [])
    counts = dict(state.get("tool_call_counts", {}))
    iteration = state.get("iteration", 0)

    coerced = [_coerce_observation(o) for o in observations]
    successful_by_tool, any_successful = _count_successful_by_tool(coerced)
    subtask_status = _build_subtask_status(plan, successful_by_tool, any_successful)

    # Count search tool calls across all atomic search tools
    search_calls = sum(counts.get(t, 0) for t in _SEARCH_TOOLS)
    total_hits = _retrieval_hit_count(coerced)
    failures = _collect_tool_failures(coerced)

    from app.db.session import SessionLocal
    org_id = state.get("org_id")
    _db = SessionLocal()
    try:
        max_iterations = get_setting(_db, "AGENT_MAX_ITERATIONS", org_id)
        max_wall_seconds = get_setting(_db, "AGENT_MAX_WALL_SECONDS", org_id)
        total_budget = get_setting(_db, "AGENT_TOTAL_TOOL_BUDGET", org_id)
    finally:
        _db.close()

    total_tool_calls = sum(counts.values())
    started_at = state.get("started_at")
    elapsed_seconds = round(time.monotonic() - started_at, 1) if started_at else 0.0

    return {
        "user_goal": state.get("original_query", ""),
        "intent": plan.intent,
        "subtasks": subtask_status,
        "search": {
            "calls": search_calls,
            "hits": total_hits,
        },
        "tool_failures": failures,
        "remaining_budget": {
            "total": total_budget - total_tool_calls,
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
    if not summary["subtasks"] and summary["search"]["calls"] == 0:
        if summary.get("intent") not in ("conversation",):
            issues.append("No tool calls were made and no subtasks were planned.")
        else:
            return True, "Conversation intent — no tools needed."

    # 2. Uncompleted subtasks that still have budget.
    for st in summary["subtasks"]:
        if not st["completed"]:
            issues.append(f"Subtask '{st['id']}' ({st['description'][:60]}) has no successful tool result.")

    # 3. Search returned zero hits and budget remains.
    if summary["search"]["calls"] > 0 and summary["search"]["hits"] == 0:
        if summary["remaining_budget"]["total"] > 0:
            issues.append("Search returned 0 hits; another search may help.")
        else:
            pass  # No budget left — proceed with what we have.

    # 4. Tool failures that could be retried.
    for f in summary["tool_failures"]:
        tool = f["tool"]
        if "Budget exceeded" in f["error"]:
            continue
        issues.append(f"Tool '{tool}' failed: {f['error'][:80]}")

    if issues:
        return False, "; ".join(issues)
    return True, "All planned steps have supporting tool results."
