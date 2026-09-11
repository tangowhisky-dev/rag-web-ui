"""Tool node for agentic-v2.

Dispatches tool calls in parallel, records observations, and loops back
to think. Includes idempotency guard (reuse prior observation for duplicate
tool+args), consecutive same-tool repeat guard, and total tool-call budget.
No per-tool caps — the agent is free to call any tool within the total budget.

The LLM sees its own prior observations in the think prompt, so it won't
repeat calls unless it has a reason. If it does repeat, the idempotency
guard reuses the prior result instead of re-running.

Kept from the current system:
- Parallel dispatch of independent tool calls
- Total tool-call budget
- Same-tool repeat guard
- Transient error retry with backoff
- Non-transient errors returned as observations (isError pattern)
- Retrieved docs merging into state
- accumulated_data / generated_files propagation
"""

from __future__ import annotations

import asyncio
import json
import logging

from app.services.agentic_rag.nodes import _agent_step
from app.services.agentic_rag.schemas import Observation
from app.services.agentic_rag.tools import applicable_tools
from app.services.infrastructure import is_cancelled
from app.services.settings_service import get_setting

from ..agent_graph.helpers import (
    _coerce_observation,
    _compact_args,
    _emit_timeline,
    _is_transient_error,
    _result_brief,
    _tool_call_budget,
    _total_tool_budget,
    _writer,
    debug_emit,
)
from ..agent_graph.tooling import (
    _hit_to_doc_dict,
    _merge_retrieved_docs,
    _run_tool,
    _summarize_result,
)

logger = logging.getLogger(__name__)


def _call_signature(name: str, args: dict) -> tuple[str, str]:
    """Stable hash key for a tool call (tool name + sorted JSON of args)."""
    import json
    return (name, json.dumps(args, sort_keys=True, default=str))


async def _dispatch_v2(
    tool_calls: list[dict],
    tools: dict,
    counts: dict,
    ctx,
    prior_observations: list | None = None,
) -> tuple[list[Observation], dict, bool]:
    """Dispatch tool calls in parallel. Returns (observations, updated_counts, should_terminate).

    Guards:
    1. Total tool-call budget (AGENT_TOTAL_TOOL_BUDGET) — forces answer when exhausted.
    2. Clarify cap (AGENT_MAX_CLARIFY) — limits human-in-the-loop rounds per user query.
    3. Same-argument repeat limit (AGENT_MAX_SAME_TOOL_REPEAT) — blocks the
       (limit + 1)th consecutive call with the exact same arguments; earlier
       duplicates are idempotently reused. Same tool with different arguments
       is allowed and is not guarded.
    No other per-tool caps — the agent is free to call any tool within the total budget.
    """
    new_observations: list[Observation] = []
    total_budget = _total_tool_budget(ctx.db, ctx.org_id)
    total_calls = sum(counts.values())
    max_same_repeat = get_setting(ctx.db, "AGENT_MAX_SAME_TOOL_REPEAT", ctx.org_id)
    caps = _tool_call_budget(ctx.db, ctx.org_id)
    prior_observations = prior_observations or []

    def _call_signature(name: str, args: dict) -> tuple[str, str]:
        return (name, json.dumps(args, sort_keys=True, default=str))

    prior_signatures: dict[tuple[str, str], Observation] = {}
    for obs in prior_observations:
        # Only cache successful observations — failed calls should be
        # retried, not served from cache.
        if obs.error:
            continue
        prior_signatures.setdefault(_call_signature(obs.tool, obs.arguments), obs)

    def _consecutive_same_signature_count(signature: tuple[str, str]) -> int:
        count = 0
        for obs in reversed(prior_observations):
            if _call_signature(obs.tool, obs.arguments) == signature:
                count += 1
            else:
                break
        return count

    async def _reuse_prior(prior: Observation):
        # Debug stream: dedup replays never reach _run_tool, so emit the
        # replayed I/O explicitly — the agent still "saw" this output.
        debug_emit("tool_observation", {
            "tool": prior.tool,
            "arguments": _compact_args(prior.arguments),
            "result": _result_brief(prior.result),
            "error": prior.error,
            "dedup_replay": True,
            "tokens": 0,
        })
        return {
            "tool": prior.tool,
            "arguments": prior.arguments,
            "result": prior.result,
            "error": prior.error,
            "tokens": 0,
        }

    _dup_attempt_counts: dict[tuple[str, str], int] = {}

    coros = []
    executed_flags: list[bool] = []
    tool_step_ids: list[str] = []
    tool_labels: list[str] = []

    for tc in tool_calls:
        name = tc.get("tool")
        args = tc.get("arguments", {})
        tool_obj = tools.get(name)
        if tool_obj and hasattr(tool_obj, "prepare_arguments"):
            args = tool_obj.prepare_arguments(args)
        label = getattr(tool_obj, "ui_label", None) if tool_obj else None
        ui_label = label or name
        step_id = _emit_timeline(type="tool_call", tool=name, label=ui_label,
                                 status="active", arguments=_compact_args(args))
        tool_step_ids.append(step_id)
        tool_labels.append(ui_label)

        sig = _call_signature(name, args)

        # Idempotency: reuse prior observation for exact duplicate calls.
        # The call still counts against the tool-call budget — the model
        # made a tool call, we just served it from cache.
        prior = prior_signatures.get(sig)
        if prior is not None:
            logger.debug("[tool_node_v2] duplicate call skipped, reusing prior observation: tool=%s args=%s", name, args)
            prior_dups = _consecutive_same_signature_count(sig)
            current_dups = _dup_attempt_counts.get(sig, 0)
            total_consecutive = prior_dups + current_dups + 1
            _dup_attempt_counts[sig] = current_dups + 1
            if total_consecutive > max_same_repeat:
                logger.debug("[tool_node_v2] same-argument repeat limit (%d) exceeded for %s (%d consecutive) — forcing strategy change", max_same_repeat, name, total_consecutive)
                async def _dup_repeat_exceeded(name=name, args=args, total=total_consecutive, limit=max_same_repeat):
                    return {"tool": name, "arguments": args, "result": {},
                            "error": f"Tool '{name}' called {total} times consecutively with the same arguments. "
                                     f"Max allowed is {limit}. You already have the result — use it and proceed to the next step. "
                                     f"Change strategy: use a different tool, finalize, or ask for clarification.",
                            "tokens": 0}
                coros.append(_dup_repeat_exceeded())
            else:
                coros.append(_reuse_prior(prior))
            total_calls += 1
            executed_flags.append(True)
            continue

        # Total tool-call budget guard.
        if total_calls >= total_budget:
            async def _budget_exceeded(name=name, args=args, cap=total_budget):
                return {"tool": name, "arguments": args, "result": {},
                        "error": f"Total tool-call budget ({cap}) exceeded. Write your answer now.", "tokens": 0}
            coros.append(_budget_exceeded())
            executed_flags.append(False)
            continue

        # Clarify cap (AGENT_MAX_CLARIFY) — per-query human-in-the-loop safety.
        cap = caps.get(name)
        if cap is not None and counts.get(name, 0) >= cap:
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
        _emit_timeline(
            id=tool_step_ids[i],
            type="tool_result",
            tool=obs.tool,
            label=tool_labels[i],
            summary=_summarize_result(obs),
            details=obs.result.get("ui_details") if isinstance(obs.result, dict) else None,
            error=obs.error,
            status="complete",
        )
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

        # Cancellation check before dispatching tools.
        chat_id = ctx.chat_id if ctx is not None else None
        if chat_id is not None and is_cancelled(chat_id):
            logger.debug("[tool_v2] cancelled before dispatch | chat_id=%s", chat_id)
            return {"tool_calls": [], "force_finalize": True}

        ctx.state = state
        tools = {t.name: t for t in applicable_tools(ctx)}
        prior_observations = [_coerce_observation(o) for o in state.get("observations", [])]
        counts = dict(state.get("tool_call_counts", {}))

        new_observations, counts, should_terminate = await _dispatch_v2(
            tool_calls, tools, counts, ctx,
            prior_observations=prior_observations,
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
