"""Tool node — dispatches tool calls and records observations.

Dispatches tool calls in parallel (when independent), runs them, records
observations, and retries failed calls. Transient errors retry with
backoff; argument errors call the correction LLM for new arguments.
Retries do not count against the per-tool call budget.

After every tool round, runs the deterministic execution-completeness
check so a completed plan short-circuits immediately instead of waiting
on the LLM to notice.

Also contains route_tool and route_reflect_final, the conditional edges
after the tool and reflect_final nodes.
"""

from __future__ import annotations

import asyncio
import json
import logging

from app.services.agentic_rag.llm_factory import build_chat_llm
from app.services.agentic_rag.nodes import _agent_step
from app.services.agentic_rag.prompts import TOOL_CORRECTION_PROMPT
from app.services.agentic_rag.schemas import Observation
from app.services.agentic_rag.tools import applicable_tools
from app.services.settings_service import get_setting

from .helpers import (
    _coerce_observation,
    _correction_hints,
    _extract_json_block,
    _is_transient_error,
    _tool_call_budget,
    _wall_clock_exceeded,
    _writer,
)
from .reflection import _build_execution_summary, _verify_execution

logger = logging.getLogger(__name__)


def route_tool(state) -> str:
    """After a tool round: skip reflect+think entirely if already satisfied."""
    if state.get("force_finalize"):
        return "reflect_final"
    return "reflect"


def route_reflect_final(state) -> str:
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
    prompt = TOOL_CORRECTION_PROMPT.format(
        tool_name=tool_name,
        error=error,
        original_args=json.dumps(original_args, default=str),
        schema=json.dumps(schema, default=str),
        hints=_correction_hints(tool_name, error),
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


async def _dispatch_tool_calls(
    tool_calls: list[dict],
    tools: dict,
    prior_observations: list[Observation],
    counts: dict,
    ctx: "ToolContext",
) -> tuple[list[Observation], dict]:
    """Dispatch tool calls in parallel, returning new observations and updated counts."""
    writer = _writer()
    new_observations: list[Observation] = []

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
            logger.debug("[tool_node] duplicate call skipped, reusing prior observation: tool=%s args=%s", name, args)
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

    return new_observations, counts


async def _retry_failed_calls(
    new_observations: list[Observation],
    tool_calls: list[dict],
    tools: dict,
    max_retries: int,
    ctx: "ToolContext",
) -> None:
    """Retry failed tool calls in place: transient errors retry with backoff;
    argument errors call the correction LLM for new arguments.

    Retries do NOT count against the per-tool call budget (_TOOL_CALL_BUDGET)
    \u2014 that budget limits how many times the *think* LLM can choose to call a
    tool, not how many times a single failed call can be retried.
    """
    writer = _writer()
    if max_retries <= 0:
        return
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


def _seed_existing_docs(existing_docs, seen_hashes, merged_docs):
    from app.services.infrastructure import content_hash as _ch
    for doc in existing_docs or []:
        if not isinstance(doc, dict):
            continue
        h = doc.get("metadata", {}).get("content_hash") or _ch(doc.get("page_content", ""))
        if h not in seen_hashes:
            seen_hashes.add(h)
            merged_docs.append(doc)


def _merge_observation_docs(all_observations, seen_hashes, merged_docs):
    from app.services.infrastructure import content_hash as _ch
    best_confidence = 0.0
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
    return best_confidence


def _merge_retrieved_docs(
    all_observations: list[Observation],
    existing_docs: list[dict],
) -> tuple[list[dict], float]:
    """Promote all rag_retrieve docs into graph state (deduplicated across
    observations by content_hash).

    `observations` uses the append-style `accumulate` reducer, so tool_node
    must return ONLY the observations it created. Returning prior + new made
    the channel grow 1 \u2192 3 \u2192 7 \u2192 15 across tool rounds.
    """
    merged_docs: list[dict] = []
    seen_hashes: set[str] = set()
    _seed_existing_docs(existing_docs, seen_hashes, merged_docs)
    best_confidence = _merge_observation_docs(all_observations, seen_hashes, merged_docs)
    return merged_docs, best_confidence


async def tool_node(state, ctx) -> dict:
    """Dispatch tool calls, run them (in parallel when independent), record observations."""
    with _agent_step("tool"):
        tool_calls = state.get("tool_calls", [])
        if not tool_calls:
            return {}

        # Expose current state to tools so they can read last_answer_object,
        # retrieved_docs, kb_ids, file_markdown, message_id, iteration, etc.
        ctx.state = state
        tools = {t.name: t for t in applicable_tools(ctx)}
        prior_observations = [_coerce_observation(o) for o in state.get("observations", [])]
        counts = dict(state.get("tool_call_count", {}))

        new_observations, counts = await _dispatch_tool_calls(
            tool_calls, tools, prior_observations, counts, ctx,
        )

        max_retries = get_setting(ctx.db, "AGENT_MAX_TOOL_RETRIES", ctx.org_id)
        await _retry_failed_calls(new_observations, tool_calls, tools, max_retries, ctx)

        state_update: dict = {
            "tool_calls": [],
            "observations": new_observations,
            "tool_call_count": counts,
        }
        all_observations = prior_observations + new_observations
        merged_docs, best_confidence = _merge_retrieved_docs(
            all_observations, state.get("retrieved_docs", []),
        )
        if merged_docs:
            state_update["retrieved_docs"] = merged_docs
            state_update["retrieval_confidence"] = best_confidence

        # Root cause: the acting LLM alone decides when to stop calling tools,
        # and small/local models don\u2019t reliably follow "stop once sufficient"
        # / "don\u2019t repeat calls" prompt rules \u2014 they keep re-emitting tool_calls
        # (often exact duplicates) past the point the plan is already
        # deterministically satisfied. reflect_final already verifies this
        # deterministically, but only once the LLM itself stops requesting
        # tools. Run the same check here after every tool round so a
        # completed plan short-circuits immediately instead of waiting on
        # the LLM to notice.
        probe_state = {**state, **state_update, "observations": all_observations}
        ready, reasoning = _verify_execution(_build_execution_summary(probe_state))
        if ready:
            logger.debug("[tool_node] plan deterministically satisfied after this tool round \u2014 forcing finalize: %s", reasoning[:200])
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
