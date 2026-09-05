"""Tool node — dispatches tool calls and records observations.

Dispatches tool calls in parallel (when independent), runs them, records
observations, and retries transient failures with backoff. Argument errors
are returned to the LLM via the observation (isError pattern) — the LLM
fixes them on the next think iteration, eliminating the correction LLM call.

After every tool round, runs the deterministic execution-completeness
check so a completed plan short-circuits immediately instead of waiting
on the LLM to notice.
"""

from __future__ import annotations

import asyncio
import json
import logging

from app.services.agentic_rag.nodes import _agent_step
from app.services.agentic_rag.schemas import Observation
from app.services.agentic_rag.tools import applicable_tools
from app.services.settings_service import get_setting

from .execution_check import _build_execution_summary, _verify_execution
from .helpers import (
    _coerce_observation,
    _is_transient_error,
    _tool_call_budget,
    _total_tool_budget,
    _wall_clock_exceeded,
    _writer,
)

logger = logging.getLogger(__name__)


def _tool_label(tool_name: str, tool_calls: list[dict]) -> str:
    """Get the UI label for a tool from the tool_calls list."""
    for tc in tool_calls:
        if tc.get("tool") == tool_name:
            return tc.get("label") or tool_name
    return tool_name


def _summarize_result(obs: Observation) -> str:
    """Build a one-line summary of a tool observation result for the UI."""
    if obs.error:
        return obs.error[:120]
    r = obs.result
    if not isinstance(r, dict):
        return ""
    if "hits" in r and isinstance(r["hits"], list):
        n = len(r["hits"])
        return f"{n} {'hit' if n == 1 else 'hits'} retrieved"
    if "docs" in r and isinstance(r["docs"], list):
        n = len(r["docs"])
        return f"{n} {'doc' if n == 1 else 'docs'} retrieved"
    if "matches" in r and isinstance(r["matches"], list):
        n = len(r["matches"])
        return f"{n} {'match' if n == 1 else 'matches'} found"
    if "content" in r:
        tokens = r.get("total_tokens", "?")
        return f"Read {tokens} tokens"
    if "headings" in r and isinstance(r["headings"], list):
        n = len(r["headings"])
        return f"{n} {'heading' if n == 1 else 'headings'}"
    if "points" in r and isinstance(r["points"], list):
        n = len(r["points"])
        return f"{n} {'data point' if n == 1 else 'data points'} extracted"
    if "chart_option" in r:
        return "Chart generated"
    if "file_id" in r and "format" in r:
        fmt = r["format"].upper()
        name = r.get("file_name", "")
        charts = r.get("chart_count", 0)
        suffix = f" ({charts} charts)" if charts else ""
        return f"{fmt} generated: {name}{suffix}"
    if "mode" in r and r.get("mode") in ("issues", "screenshot", "outline", "validate", "annotated", "text", "get", "query"):
        mode = r["mode"]
        output = r.get("output", "")
        if mode == "issues":
            issue_count = output.count("issue") if isinstance(output, str) else 0
            return f"QA: {issue_count} issues" if issue_count else "QA: no issues"
        return f"Inspected ({mode})"
    if "commands_applied" in r:
        return f"Edited: {r['commands_applied']} commands applied"
    if "result" in r and isinstance(r["result"], str):
        return r["result"][:120]
    return ""


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

    # Consecutive same-tool repeat guard: local models sometimes loop
    # calling the same tool with slightly different arguments (not exact
    # duplicates, so the idempotency guard doesn't catch them). Count
    # consecutive calls to the same tool; if the count exceeds the
    # configured limit, return an error telling the LLM to change strategy.
    max_same_repeat = get_setting(ctx.db, "AGENT_MAX_SAME_TOOL_REPEAT", ctx.org_id)
    # Count consecutive same-tool calls from the end of prior_observations.
    consecutive_counts: dict[str, int] = {}
    for obs in reversed(prior_observations):
        # Stop counting when we hit a different tool — only consecutive runs matter.
        # But we need per-tool counts, so count the tail run of each tool.
        pass
    # Simpler: count how many of the last N observations are the same tool.
    # We check this per-tool in the loop below.
    def _consecutive_same_tool_count(tool_name: str) -> int:
        count = 0
        for obs in reversed(prior_observations):
            if obs.tool == tool_name:
                count += 1
            else:
                break
        return count

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
    executed_flags = []  # True = actually executed, False = reused/budget-exceeded
    total_budget = _total_tool_budget(ctx.db, ctx.org_id)
    total_calls = sum(counts.values())
    for tc in tool_calls:
        name = tc.get("tool")
        args = tc.get("arguments", {})
        # Normalize nested dict keys before any processing — some LLM providers
        # return keys with extra quotes (e.g. '"title"' instead of 'title').
        tool_obj = tools.get(name)
        if tool_obj and hasattr(tool_obj, "prepare_arguments"):
            args = tool_obj.prepare_arguments(args)
        label = getattr(tool_obj, "ui_label", None) if tool_obj else None
        writer({"event": "tool_call", "tool": name, "arguments": args, "label": label or name})
        prior = prior_signatures.get(_call_signature(name, args))
        if prior is not None:
            logger.debug("[tool_node] duplicate call skipped, reusing prior observation: tool=%s args=%s", name, args)
            # Track consecutive duplicates for the same-tool repeat guard.
            # If the LLM keeps calling the same tool with the same args, it's
            # stuck in a loop — return an error to force a strategy change.
            _consecutive_counts[name] = _consecutive_counts.get(name, 0) + 1
            if _consecutive_counts[name] >= max_same_repeat:
                logger.debug("[tool_node] same-tool repeat limit (%d) reached for %s via duplicates — forcing strategy change", max_same_repeat, name)
                async def _dup_repeat_exceeded(name=name, args=args, limit=max_same_repeat):
                    return {"tool": name, "arguments": args, "result": {},
                            "error": f"Tool '{name}' called {limit} times consecutively (including duplicates). "
                                     f"You already have the result — use it and proceed to the next step. "
                                     f"For office generation: call extract_data(source='retrieved_docs') next, then office_generate.",
                            "tokens": 0}
                coros.append(_dup_repeat_exceeded())
            else:
                coros.append(_reuse_prior(prior))
            executed_flags.append(False)
            continue
        # Consecutive same-tool repeat guard: if the same tool was called
        # consecutively too many times (with different args), force the LLM
        # to change strategy. This catches local models that loop with
        # slightly different arguments.
        if _consecutive_same_tool_count(name) >= max_same_repeat:
            async def _repeat_exceeded(name=name, args=args, limit=max_same_repeat):
                return {"tool": name, "arguments": args, "result": {},
                        "error": f"Tool '{name}' called {limit} times consecutively with different arguments. "
                                 f"Change strategy: use a different tool, finalize, or ask for clarification.",
                        "tokens": 0}
            coros.append(_repeat_exceeded())
            executed_flags.append(False)
            continue
        # Total budget check (across all tools)
        if total_calls >= total_budget:
            coros.append(_budget_exceeded(name, args, total_budget))
            executed_flags.append(False)
            continue
        cap = _tool_call_budget(ctx.db, ctx.org_id).get(name)
        current = counts.get(name, 0)
        if cap is not None and current >= cap:
            coros.append(_budget_exceeded(name, args, cap))
            executed_flags.append(False)
            continue
        total_calls += 1
        tool = tools.get(name)
        if tool is None:
            async def _missing(name=name, args=args):
                return {"tool": name, "arguments": args, "result": {}, "error": f"Tool {name} not available", "tokens": 0}
            coros.append(_missing())
        else:
            coros.append(_run_tool(tool, name, args))
        executed_flags.append(True)

    results = await asyncio.gather(*coros, return_exceptions=True)
    should_terminate = False
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
            if res.get("terminate"):
                should_terminate = True
        new_observations.append(obs)
        writer({"event": "tool_observation", "tool": obs.tool, "label": _tool_label(obs.tool, tool_calls), "summary": _summarize_result(obs), "error": obs.error})
        # Only count actually-executed calls toward the budget.
        if executed_flags[i]:
            counts[obs.tool] = counts.get(obs.tool, 0) + 1

    return new_observations, counts, should_terminate


async def _retry_failed_calls(
    new_observations: list[Observation],
    tool_calls: list[dict],
    tools: dict,
    max_retries: int,
    ctx: "ToolContext",
) -> bool:
    """Retry failed tool calls in place: transient errors retry with backoff.

    Argument errors are NOT retried via a correction LLM anymore (isError pattern).
    Instead, the error is returned to the LLM via the observation, and the LLM
    fixes the arguments on its next think iteration. This removes the extra
    LLM call per failed tool invocation.

    Retries do NOT count against the per-tool call budget (_tool_call_budget)
    \u2014 that budget limits how many times the *think* LLM can choose to call a
    tool, not how many times a single failed call can be retried.

    Returns True if any retried call set terminate=True.
    """
    writer = _writer()
    if max_retries <= 0:
        return False
    retry_terminate = False
    for idx, obs in enumerate(new_observations):
        if obs.error is None:
            continue
        if not _is_transient_error(obs.error):
            # Non-transient error: return to LLM via observation (isError pattern).
            # The LLM will see the error and fix the arguments on the next think.
            continue
        tool_name = obs.tool
        tool = tools.get(tool_name)
        if tool is None:
            continue
        for attempt in range(max_retries):
            await asyncio.sleep(get_setting(ctx.db, "AGENT_RETRY_BACKOFF_BASE", ctx.org_id) * (2 ** attempt))
            retry_result = await _run_tool(tool, tool_name, obs.arguments)
            retry_obs = Observation(
                tool=retry_result["tool"],
                arguments=retry_result["arguments"],
                result=retry_result.get("result", {}),
                error=retry_result.get("error"),
                tokens=retry_result.get("tokens", 0),
            )
            if retry_result.get("terminate"):
                retry_terminate = True
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
            if not _is_transient_error(retry_obs.error):
                break  # non-transient error, return to LLM
            obs = retry_obs
    return retry_terminate


def _seed_existing_docs(existing_docs, seen_hashes, merged_docs):
    from app.services.infrastructure import content_hash as _ch
    for doc in existing_docs or []:
        if not isinstance(doc, dict):
            continue
        h = doc.get("metadata", {}).get("content_hash") or _ch(doc.get("page_content", ""))
        if h not in seen_hashes:
            seen_hashes.add(h)
            merged_docs.append(doc)


# Tools that return hits in the new atomic search format: {"hits": [...]}
_SEARCH_TOOLS = frozenset({"search_exact", "search_sparse", "search_dense", "rerank_results", "graph_expand"})


def _hit_to_doc_dict(hit: dict) -> dict:
    """Convert a search tool hit (flat dict) to the standard doc dict shape."""
    return {
        "page_content": hit.get("content", ""),
        "metadata": {
            "document_id": hit.get("document_id"),
            "chunk_index": hit.get("chunk_index"),
            "page": hit.get("page"),
            "title": hit.get("title", ""),
            "file_name": hit.get("file_name", ""),
            "content_hash": hit.get("content_hash", ""),
            "qdrant_point_id": hit.get("qdrant_point_id", ""),
            "_reranker_score": hit.get("_reranker_score", hit.get("score", 0.0)),
            "citation_ref": hit.get("citation_ref", {}),
        },
    }


def _merge_observation_docs(all_observations, seen_hashes, merged_docs):
    from app.services.infrastructure import content_hash as _ch
    best_confidence = 0.0
    for obs in all_observations:
        if obs.tool in _SEARCH_TOOLS and not obs.error:
            hits = obs.result.get("hits")
            if isinstance(hits, list):
                for hit in hits:
                    if not isinstance(hit, dict):
                        continue
                    doc_dict = _hit_to_doc_dict(hit)
                    h = doc_dict["metadata"].get("content_hash") or _ch(doc_dict.get("page_content", ""))
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        merged_docs.append(doc_dict)
                # Search hits with reranker scores or dense scores contribute confidence.
                # _reranker_score (from rerank_results) is a cross-encoder score
                # that can be negative; normalize via sigmoid to 0-1.
                # score from search_dense is cosine similarity (0-1).
                # score from search_sparse is SPLADE dot product (0-10+); clamp to 0-1.
                # score from search_exact is MySQL FTS score (0-10+); clamp to 0-1.
                for h in hits:
                    rs = h.get("_reranker_score")
                    if rs is not None:
                        norm = 1.0 / (1.0 + pow(2.718281828, -rs))
                    else:
                        raw_score = h.get("score", 0.0)
                        # Dense cosine similarity is 0-1; SPLADE/FTS scores can be >1.
                        # Clamp to 0-1 range.
                        norm = min(raw_score, 1.0) if raw_score > 0 else 0.0
                    if norm > best_confidence:
                        best_confidence = norm
                logger.debug(
                    "[tool_node] merged search hits: tool=%s hits=%d best_confidence=%.3f",
                    obs.tool, len(hits), best_confidence,
                )
        elif obs.tool == "kb_search_documents" and not obs.error:
            docs = obs.result.get("docs")
            if isinstance(docs, list):
                for doc in docs:
                    if not isinstance(doc, dict):
                        continue
                    h = doc.get("metadata", {}).get("content_hash") or _ch(doc.get("page_content", ""))
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        merged_docs.append(doc)
                # Document-level matches are high-confidence by definition.
                if best_confidence < 0.9:
                    best_confidence = 0.9
        elif obs.tool == "kb_read" and not obs.error:
            # kb_read returns a single document's content, not a docs list.
            # Convert to the standard doc dict shape so it gets a [KB-N]
            # label in the finalize prompt and becomes citable evidence.
            content = obs.result.get("content", "")
            if content:
                citation_ref = obs.result.get("citation_ref", {})
                doc_dict = {
                    "page_content": content,
                    "metadata": {
                        "document_id": obs.result.get("document_id"),
                        "title": obs.result.get("title") or obs.result.get("file_name"),
                        "file_name": obs.result.get("file_name"),
                        "section": obs.result.get("section"),
                        "source": "kb_read",
                        "_reranker_score": 1.0,
                        "truncated": obs.result.get("truncated", False),
                        "citation_ref": citation_ref,
                    },
                }
                h = _ch(content)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    merged_docs.append(doc_dict)
                if best_confidence < 0.9:
                    best_confidence = 0.9
    return best_confidence


def _merge_retrieved_docs(
    all_observations: list[Observation],
    existing_docs: list[dict],
) -> tuple[list[dict], float]:
    """Promote all search/read docs into graph state (deduplicated across
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
        counts = dict(state.get("tool_call_counts", {}))

        new_observations, counts, should_terminate = await _dispatch_tool_calls(
            tool_calls, tools, prior_observations, counts, ctx,
        )

        max_retries = get_setting(ctx.db, "AGENT_MAX_TOOL_RETRIES", ctx.org_id)
        retry_terminate = await _retry_failed_calls(new_observations, tool_calls, tools, max_retries, ctx)
        if retry_terminate:
            should_terminate = True

        state_update: dict = {
            "tool_calls": [],
            "observations": new_observations,
            "tool_call_counts": counts,
        }

        # If any tool returned terminate=True, force finalize immediately.
        if should_terminate:
            state_update["force_finalize"] = True
            logger.debug("[tool_node] tool requested termination, forcing finalize")

        all_observations = prior_observations + new_observations
        merged_docs, best_confidence = _merge_retrieved_docs(
            all_observations, state.get("retrieved_docs", []),
        )
        if merged_docs:
            state_update["retrieved_docs"] = merged_docs
            state_update["best_retrieval_confidence"] = best_confidence

        # Propagate accumulated_data changes from extract_data back into
        # graph state. extract_data writes to ctx.state["accumulated_data"]
        # directly (append semantics); tool_node must surface it so the
        # _last_value reducer picks it up.
        if "accumulated_data" in ctx.state:
            state_update["accumulated_data"] = ctx.state["accumulated_data"]

        # Propagate generated_files from office_generate back into graph state.
        # office_generate writes to ctx.state["generated_files"] directly;
        # tool_node must surface it so office_inspect/office_edit can see it.
        if "generated_files" in ctx.state:
            state_update["generated_files"] = ctx.state["generated_files"]

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
            logger.debug("[tool_node] plan deterministically satisfied after this tool round, forcing finalize: %s", reasoning[:200])
            state_update["force_finalize"] = True
        else:
            # Confidence short-circuit: if the reranker is highly confident
            # (top-1 score >= threshold AND gap to tail >= gap_threshold),
            # skip reflection and go straight to finalize. Saves ~2-5s of
            # reflect+think LLM latency for confident retrievals.
            # BUT: skip for office/chart intent — those require post-retrieval
            # tool calls (extract_data, office_generate, chart_generate) that
            # haven't happened yet.
            plan = state.get("plan")
            plan_intent = ""
            if hasattr(plan, "intent"):
                plan_intent = plan.intent
            elif isinstance(plan, dict):
                plan_intent = plan.get("intent", "")
            counts = state.get("tool_call_counts", {})
            has_office_generate = counts.get("office_generate", 0) > 0
            has_chart_generate = counts.get("chart_generate", 0) > 0
            needs_post_retrieval = plan_intent in ("office", "chart")
            if needs_post_retrieval and not (has_office_generate or has_chart_generate):
                logger.debug("[tool_node] skipping reranker confidence short-circuit — %s intent requires post-retrieval tools", plan_intent)
            elif _reranker_confident(merged_docs, ctx):
                logger.debug("[tool_node] reranker confidence short-circuit, forcing finalize")
                state_update["force_finalize"] = True

        return state_update

def _reranker_confident(merged_docs: list[dict], ctx) -> bool:
    """Check if the reranker is confident enough to skip reflection.

    Returns True when:
    - There are at least 2 docs with reranker scores
    - Top-1 score >= RERANKER_CONFIDENCE_THRESHOLD (default 0.8)
    - Gap between top-1 and the tail (last doc) >= RERANKER_CONFIDENCE_GAP (default 0.3)

    This mirrors the Cohere confidence short-circuit from retrievalagent.
    """
    if not merged_docs or len(merged_docs) < 2:
        return False

    scores = [
        d.get("metadata", {}).get("_reranker_score", -float("inf"))
        for d in merged_docs
        if d.get("metadata", {}).get("_reranker_score") is not None
    ]
    if len(scores) < 2:
        return False

    scores.sort(reverse=True)
    top1 = scores[0]
    tail = scores[-1]
    gap = top1 - tail

    top_threshold = get_setting(ctx.db, "RERANKER_CONFIDENCE_THRESHOLD", ctx.org_id)
    gap_threshold = get_setting(ctx.db, "RERANKER_CONFIDENCE_GAP", ctx.org_id)

    confident = top1 >= top_threshold and gap >= gap_threshold
    if confident:
        logger.debug("[reranker_confident] top1=%.3f >= %.2f, gap=%.3f >= %.2f — confident",
                     top1, top_threshold, gap, gap_threshold)
    return confident


async def _run_tool(tool, name: str, args: dict) -> dict:
    try:
        # Normalize nested dict keys — some LLM providers return keys with
        # extra quotes (e.g. '"title"' instead of 'title'). Call the tool's
        # prepare_arguments if it exists, otherwise pass through.
        if hasattr(tool, "prepare_arguments"):
            args = tool.prepare_arguments(args)
        raw = await tool.arun(args)
        # Tools return {"ok": bool, "result": {...}, "error": str|None, "tokens": int, "terminate": bool}.
        # Unwrap the envelope so obs.result is the inner payload (e.g. {"docs": [...], ...}).
        if isinstance(raw, dict) and "result" in raw:
            return {
                "tool": name,
                "arguments": args,
                "result": raw.get("result", {}),
                "error": raw.get("error"),
                "tokens": raw.get("tokens", 0),
                "terminate": raw.get("terminate", False),
            }
        return {"tool": name, "arguments": args, "result": raw, "error": None, "tokens": 0, "terminate": False}
    except Exception as exc:
        logger.warning("[_run_tool] %s failed: %s", name, exc)
        return {"tool": name, "arguments": args, "result": {}, "error": str(exc), "tokens": 0, "terminate": False}
