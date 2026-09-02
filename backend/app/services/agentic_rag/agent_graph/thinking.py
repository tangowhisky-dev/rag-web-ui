"""Think node — decides the next action in the agent loop.

Emits one or more tool calls or a final-answer signal. Builds the think
prompt from iteration count, user message, plan, observations, and
available tools. Runs context compaction before the LLM call. Short-
circuits without an LLM call when the plan is already deterministically
satisfied or when precomputed tool calls exist.

Also contains route_think, the conditional edge after the think node.
"""

from __future__ import annotations

import json
import logging

from app.services.agentic_rag.kb_profile import format_profile_summary
from app.services.agentic_rag.llm_factory import build_chat_llm
from app.services.agentic_rag.nodes import _agent_step, history_to_text, select_recent_history
from app.services.agentic_rag.prompts import AGENT_SYSTEM_PROMPT, THINK_SYSTEM_PROMPT
from app.services.agentic_rag.schemas import Plan
from app.services.agentic_rag.tool_call_parser import parse_think_response
from app.services.agentic_rag.tools import applicable_tools
from app.services.agentic_rag.token_budget import count_tokens
from app.services.settings_service import get_setting

from .compaction import _compact_if_needed
from .helpers import _wall_clock_exceeded, _writer
from .observations import _observations_metadata_text, _tool_descriptions_text, _tried_rag_retrieve_queries
from .reflection import _build_execution_summary, _verify_execution

logger = logging.getLogger(__name__)


def _build_think_prompt(
    iteration: int,
    max_iter: int,
    original: str,
    query: str,
    glossary: str,
    summary_text: str,
    history_text: str,
    lao,
    reflection,
    observations: list,
    plan,
    tools_text: str,
    kb_profile_text: str = "",
    query_intent: dict | None = None,
) -> str:
    # Include last_answer_object summary so "summarize it" / "chart it" work.
    lao_text = ""
    if lao and hasattr(lao, "summary"):
        lao_text = f"  Previous answer summary: {lao.summary[:300]}\n"
        if lao.key_points:
            lao_text += f"  Key points: {'; '.join(lao.key_points[:5])}\n"

    # If reflect_final sent us back, include its reasoning so the agent
    # knows exactly what was missing and can act on it.
    reflection_text = ""
    if reflection and isinstance(reflection, dict) and not reflection.get("ready", True):
        reflection_text = (
            f"  NOTE \u2014 the verification module rejected your previous final_answer because:\n"
            f"  {reflection.get('reasoning', '')}\n"
            "  Do NOT reference this feedback in your answer. Use it only as guidance to\n"
            "  decide which tool to call next, then emit a clean final_answer.\n"
        )

    tried_queries = _tried_rag_retrieve_queries(observations)
    tried_queries_text = (
        f"  Already tried (do NOT resubmit these exact strings to rag_retrieve): {tried_queries}\n"
        if tried_queries else ""
    )

    return (
        f"Iteration: {iteration}/{max_iter}\n"
        f"User message: {original}\n"
        f"Retrieval query: {query}\n"
        + (f"[Abbreviation Glossary]\n{glossary}\n\n" if glossary else "")
        + (f"{kb_profile_text}\n\n" if kb_profile_text else "")
        + (f"[Query Intent] {json.dumps(query_intent)}\n\n" if query_intent else "")
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


def _parse_tool_calls(resp, mode: str, iteration: int, max_iter: int):
    parsed = parse_think_response(resp, mode=mode)
    tool_calls = parsed.tool_calls
    final_answer = parsed.final_answer
    if iteration >= max_iter:
        tool_calls = []
    # Dependency guard: only allow independent tool calls in one message.
    allowed = list(tool_calls)
    return allowed, final_answer


def _think_early_exit(state, iteration):
    if state.get("force_finalize"):
        return {"iteration": iteration, "tool_calls": [], "precomputed_answer": ""}
    ready, _reasoning = _verify_execution(_build_execution_summary(state))
    if ready:
        return {"iteration": iteration, "tool_calls": [], "precomputed_answer": ""}
    precomputed = state.get("precomputed_tool_calls", [])
    if precomputed:
        return {"iteration": iteration, "tool_calls": list(precomputed), "precomputed_tool_calls": []}
    return None


async def _invoke_think_llm(ctx, system, user, tools, mode):
    if mode == "json_text":
        llm = build_chat_llm(ctx.org_id, ctx.db, role="chat", temperature=0.0)
        return await llm.ainvoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
    llm = build_chat_llm(ctx.org_id, ctx.db, role="chat", temperature=0.0)
    return await llm.bind_tools(tools).ainvoke([{"role": "system", "content": system}, {"role": "user", "content": user}])


def _rebuild_think_after_compaction(state, compaction_local, ctx, iteration, max_iter, original, query, glossary, plan, tools_text):
    state = {**state, **compaction_local}
    observations = state.get("observations", [])
    recent = select_recent_history(state.get("messages", []), max_pairs=get_setting(ctx.db, "AGENT_HISTORY_PAIRS", ctx.org_id))
    history_text = history_to_text(recent)
    summary_text = state.get("compaction_summary") or ""
    kb_profile_text = format_profile_summary(state.get("kb_profile", {}))
    query_intent = state.get("query_intent")
    user = _build_think_prompt(
        iteration, max_iter, original, query, glossary, summary_text,
        history_text, state.get("last_answer_object"),
        state.get("reflection_final"), observations, plan, tools_text,
        kb_profile_text, query_intent,
    )
    return state, observations, user


async def think_node(state, ctx) -> dict:
    """Decide the next action: emit one or more tool calls or a final answer."""
    with _agent_step("think"):
        ctx.state = state
        iteration = state.get("iteration", 0) + 1
        max_iter = get_setting(ctx.db, "AGENT_MAX_ITERATIONS", ctx.org_id)

        early = _think_early_exit(state, iteration)
        if early is not None:
            return early

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

        # Glossary was built once by expand_query_node — reuse it.
        glossary = state.get("abbreviation_glossary", "")

        system = AGENT_SYSTEM_PROMPT + "\n\n" + THINK_SYSTEM_PROMPT
        kb_profile_text = format_profile_summary(state.get("kb_profile", {}))
        query_intent = state.get("query_intent")
        user = _build_think_prompt(
            iteration, max_iter, original, query, glossary, summary_text,
            history_text, state.get("last_answer_object"),
            state.get("reflection_final"), observations, plan, tools_text,
            kb_profile_text, query_intent,
        )

        # Runtime compaction: check if the prompt exceeds the context budget.
        # If so, compact observations (deterministic) and/or messages (LLM call),
        # then rebuild the prompt from the compacted state.
        compaction_updates, compaction_local = await _compact_if_needed(
            state, user, system_overhead=count_tokens(system), ctx=ctx,
        )
        if compaction_local:
            state, observations, user = _rebuild_think_after_compaction(
                state, compaction_local, ctx, iteration, max_iter, original, query, glossary, plan, tools_text,
            )

        mode = get_setting(ctx.db, "TOOL_CALL_MODE", None)
        try:
            resp = await _invoke_think_llm(ctx, system, user, tools, mode)
        except Exception as exc:
            logger.warning("[think_node] LLM call failed: %s", exc)
            return {"iteration": iteration, "tool_calls": [], "precomputed_answer": f"LLM error: {exc}"}

        allowed, final_answer = _parse_tool_calls(resp, mode, iteration, max_iter)

        if allowed and not final_answer:
            return {**compaction_updates, "iteration": iteration, "tool_calls": allowed}

        # final_answer can be:
        #   - True (boolean signal from {"final_answer": true}) — think is done,
        #     finalize_node will generate the answer with streaming.
        #   - str (Tier 3 fallback — LLM wrote plain text instead of JSON) — pass
        #     it through as precomputed since the text was already generated.
        if isinstance(final_answer, bool) and final_answer:
            return {**compaction_updates, "iteration": iteration, "tool_calls": [], "precomputed_answer": ""}
        return {**compaction_updates, "iteration": iteration, "tool_calls": [], "precomputed_answer": final_answer or ""}


def route_think(state) -> str:
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
