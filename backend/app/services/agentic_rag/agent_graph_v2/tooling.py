"""Tool node for agentic-v2.

Dispatches tool calls in parallel, records observations, and loops back
to think. No duplicate detection, no consecutive same-tool limits, no
reranker confidence short-circuits, no plan satisfaction checks.

The LLM sees its own prior observations in the think prompt, so it won't
repeat calls unless it has a reason. If it does repeat, the tool just
runs again — simpler and more honest than trying to outsmart the LLM.

Kept from the current system:
- Parallel dispatch of independent tool calls
- Per-tool call budget (caps)
- Total tool-call budget
- Transient error retry with backoff
- Non-transient errors returned as observations (isError pattern)
- Retrieved docs merging into state
- accumulated_data / generated_files propagation
"""

from __future__ import annotations

import asyncio
import logging

from app.services.agentic_rag.nodes import _agent_step
from app.services.agentic_rag.schemas import Observation
from app.services.agentic_rag.tools import applicable_tools
from app.services.settings_service import get_setting

from ..agent_graph.helpers import (
    _coerce_observation,
    _is_transient_error,
    _tool_call_budget,
    _total_tool_budget,
    _writer,
)
from ..agent_graph.tooling import (
    _merge_retrieved_docs,
    _run_tool,
    _summarize_result,
    _tool_label,
)

logger = logging.getLogger(__name__)


async def _dispatch_v2(
    tool_calls: list[dict],
    tools: dict,
    counts: dict,
    ctx,
) -> tuple[list[Observation], dict, bool]:
    """Dispatch tool calls in parallel. Returns (observations, updated_counts, should_terminate)."""
    writer = _writer()
    new_observations: list[Observation] = []
    total_budget = _total_tool_budget(ctx.db, ctx.org_id)
    total_calls = sum(counts.values())
    caps = _tool_call_budget(ctx.db, ctx.org_id)

    coros = []
    executed_flags: list[bool] = []

    for tc in tool_calls:
        name = tc.get("tool")
        args = tc.get("arguments", {})
        tool_obj = tools.get(name)
        if tool_obj and hasattr(tool_obj, "prepare_arguments"):
            args = tool_obj.prepare_arguments(args)
        label = getattr(tool_obj, "ui_label", None) if tool_obj else None
        writer({"event": "tool_call", "tool": name, "arguments": args, "label": label or name})

        # Budget checks
        if total_calls >= total_budget:
            async def _budget_exceeded(name=name, args=args, cap=total_budget):
                return {"tool": name, "arguments": args, "result": {},
                        "error": f"Total tool-call budget ({cap}) exceeded. Write your answer now.", "tokens": 0}
            coros.append(_budget_exceeded())
            executed_flags.append(False)
            continue
        cap = caps.get(name)
        current = counts.get(name, 0)
        if cap is not None and current >= cap:
            async def _cap_exceeded(name=name, args=args, cap=cap):
                return {"tool": name, "arguments": args, "result": {},
                        "error": f"Tool '{name}' call cap ({cap}) reached. Use a different tool or write your answer.", "tokens": 0}
            coros.append(_cap_exceeded())
            executed_flags.append(False)
            continue

        total_calls += 1
        tool = tools.get(name)
        if tool is None:
            async def _missing(name=name, args=args):
                return {"tool": name, "arguments": args, "result": {},
                        "error": f"Tool '{name}' is not available.", "tokens": 0}
            coros.append(_missing())
        else:
            coros.append(_run_tool(tool, name, args))
        executed_flags.append(True)

    results = await asyncio.gather(*coros, return_exceptions=True)
    should_terminate = False

    # GraphInterrupt from a tool (e.g. clarify) must propagate to LangGraph
    # so it can checkpoint and pause. Do not turn it into an error observation.
    from langgraph.errors import GraphInterrupt
    for res in results:
        if isinstance(res, GraphInterrupt):
            raise res

    for i, tc in enumerate(tool_calls):
        res = results[i]
        if isinstance(res, Exception):
            obs = Observation(
                tool=tc["tool"], arguments=tc.get("arguments", {}),
                result={}, error=str(res), tokens=0,
            )
        else:
            obs = Observation(
                tool=res["tool"], arguments=res["arguments"],
                result=res.get("result", {}), error=res.get("error"),
                tokens=res.get("tokens", 0),
            )
            if res.get("terminate"):
                should_terminate = True
        new_observations.append(obs)
        writer({
            "event": "tool_observation",
            "tool": obs.tool,
            "label": _tool_label(obs.tool, tool_calls),
            "summary": _summarize_result(obs),
            "error": obs.error,
        })
        if executed_flags[i]:
            counts[obs.tool] = counts.get(obs.tool, 0) + 1

    return new_observations, counts, should_terminate


async def _retry_transient_v2(
    new_observations: list[Observation],
    tool_calls: list[dict],
    tools: dict,
    ctx,
) -> bool:
    """Retry transient failures with backoff. Returns True if any retried call set terminate."""
    max_retries = get_setting(ctx.db, "AGENT_MAX_TOOL_RETRIES", ctx.org_id)
    if max_retries <= 0:
        return False
    writer = _writer()
    retry_terminate = False
    for idx, obs in enumerate(new_observations):
        if obs.error is None or not _is_transient_error(obs.error):
            continue
        tool = tools.get(obs.tool)
        if tool is None:
            continue
        for attempt in range(max_retries):
            await asyncio.sleep(
                get_setting(ctx.db, "AGENT_RETRY_BACKOFF_BASE", ctx.org_id) * (2 ** attempt)
            )
            retry_result = await _run_tool(tool, obs.tool, obs.arguments)
            retry_obs = Observation(
                tool=retry_result["tool"], arguments=retry_result["arguments"],
                result=retry_result.get("result", {}), error=retry_result.get("error"),
                tokens=retry_result.get("tokens", 0),
            )
            if retry_result.get("terminate"):
                retry_terminate = True
            writer({
                "event": "tool_retry", "tool": obs.tool,
                "attempt": attempt + 1, "max_retries": max_retries,
                "success": retry_obs.error is None, "error": retry_obs.error,
            })
            if retry_obs.error is None:
                new_observations[idx] = retry_obs
                break
            if not _is_transient_error(retry_obs.error):
                break
    return retry_terminate


async def tool_node_v2(state, ctx) -> dict:
    """Dispatch tool calls, record observations, merge docs. Loop back to think."""
    with _agent_step("tool"):
        tool_calls = state.get("tool_calls", [])
        if not tool_calls:
            return {}

        ctx.state = state
        tools = {t.name: t for t in applicable_tools(ctx)}
        prior_observations = [_coerce_observation(o) for o in state.get("observations", [])]
        counts = dict(state.get("tool_call_counts", {}))

        new_observations, counts, should_terminate = await _dispatch_v2(
            tool_calls, tools, counts, ctx,
        )

        retry_terminate = await _retry_transient_v2(new_observations, tool_calls, tools, ctx)
        if retry_terminate:
            should_terminate = True

        state_update: dict = {
            "tool_calls": [],
            "observations": new_observations,
            "tool_call_counts": counts,
        }

        if should_terminate:
            state_update["force_finalize"] = True

        # Merge retrieved docs from all observations (prior + new).
        all_observations = prior_observations + new_observations
        merged_docs, best_confidence = _merge_retrieved_docs(
            all_observations, state.get("retrieved_docs", []),
        )
        if merged_docs:
            state_update["retrieved_docs"] = merged_docs
            state_update["best_retrieval_confidence"] = best_confidence

        # Propagate accumulated_data and generated_files from tool state.
        if "accumulated_data" in ctx.state:
            state_update["accumulated_data"] = ctx.state["accumulated_data"]
        if "generated_files" in ctx.state:
            state_update["generated_files"] = ctx.state["generated_files"]

        return state_update
