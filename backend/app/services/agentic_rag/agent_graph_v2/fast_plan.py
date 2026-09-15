"""Fast-path planner + step resolver for the fast pipeline.

Graph:  load_context → fast_plan → fast_step ⇄ tool → post_process → END

The ReAct loop is replaced by a committed plan:

- ``fast_plan`` runs ONE utility-model call producing the full ordered plan:
  intent + resolved query + steps[] (a search round plus optional processing
  steps like extract_data/chart_generate/code_execute/summarize).
- ``fast_step`` runs between tool rounds: it materializes the next step's
  ``tool_calls``. Literal args pass through unchanged; steps marked
  ``needs_prior`` get their args compiled by one utility-model call that sees
  the prior observations (the cheap substitute for a think call).
- ``tool`` is the existing ``tool_node_v2`` — same dispatch, dedup, timeline
  and debug events, and retrieved_docs merge as the agentic loop.
- Repair-once: if a round's observations all error, fast_step recompiles the
  same step's args with the error in context exactly once, then stops.

Bounded by design: ≤3 tool rounds, no mid-round reasoning, no clarification,
no sub-agents (retrieve_parallel / create_office_document never appear).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app.services.agentic_rag.nodes import (
    _agent_step,
    history_to_text,
    select_recent_history,
)
from app.services.settings_service import get_setting

from ..agent_graph.helpers import _writer, debug_emit
from ..llm_factory import build_chat_llm

logger = logging.getLogger(__name__)

# Hard caps — the plan is committed upfront so rounds are bounded.
_MAX_SUB_QUERIES = 3
_MAX_FILE_READS = 3
_MAX_TOOL_CALLS = 8        # per round
_MAX_STEPS = 4             # tool rounds total: search + extract + code + chart

# Step tools the fast pipeline may execute. Everything else in
# applicable_tools is either loop-bound (clarify, retrieve_parallel,
# create_office_document) or an office-internal tool.
_STEP_TOOLS = {
    "file_read", "kb_metadata", "kb_outline", "kb_grep", "graph_expand",
    "extract_data", "chart_generate", "code_execute", "summarize",
    "current_datetime", "file_extract_table", "title_search",
}
_SEARCH_TOOLS = {"keyword_search", "semantic_search"}

FAST_PLAN_PROMPT = """\
You plan a bounded pipeline for a fast RAG system. Your plan is executed
verbatim — there is no re-planning. Output ONLY a JSON object.

Conversation history (recent turns):
{history}

Previous answer summary: {last_answer_summary}
Previously cited document IDs: {cited_doc_ids}
Attached file this turn: {has_file}

KB profile:
{kb_profile}

Today's date: {today}

Abbreviations in the question (expand these in sub_queries — use BOTH the
short form and the full form, they embed and keyword-match differently):
{abbreviations}

User question: {query}

Output ONLY:
{{
  "intent": "retrieve" | "answer_from_history" | "direct",
  "resolved_query": "the question rewritten standalone (see rewrite rules)",
  "steps": [
    {{"tool": "search", "sub_queries": [
      {{"query": "...", "tool_hint": "any|title_search|file_read",
        "filters": {{}}, "document_ids": []}}]}},
    {{"tool": "<step tool>", "args": {{...}}}},
    {{"tool": "<step tool>", "spec": {{...}}, "needs_prior": true}}
  ]
}}

Intent rules:
- "direct": answerable without documents — arithmetic, definitions, chitchat,
  or text the user pasted into the message itself. Emit no steps. (For
  non-trivial math prefer a code_execute step instead of direct.)
- "answer_from_history": the question refers to the previous answer
  ("summarize this", "explain that", "shorter") and history suffices.
- "retrieve": needs documents — emit steps.

Step rules:
- steps[0] is normally a "search" round with 1-{_MAX_SUB_QUERIES} sub_queries.
  Use ONE sub_query for single-topic questions — only split when the question
  has genuinely independent facets (e.g. comparing two different things).
  Each sub_query becomes keyword_search + semantic_search (hybrid).
  tool_hint narrows it:
  "title_search" only as a companion (titles, not content — searches still run);
  "file_read" only with a known document_id (e.g. previously cited docs).
- Later steps may use: extract_data, chart_generate, code_execute, summarize,
  kb_grep, kb_outline, kb_metadata, graph_expand, file_extract_table,
  title_search, current_datetime. Put literal args in "args" when they do not
  depend on earlier results; use "spec" + needs_prior when args must be
  computed from prior output (e.g. code for code_execute, text for summarize).
- "spec" describes what the step needs, e.g.
  {{"task": "compute mean of extracted values"}} — a later call resolves it.
- filters may use: title_contains (short fragment only), document_status
  (draft|active|superseded), effective_as_of (ISO date),
  effective_window_start/end, version, owner, document_ids. For current-state
  questions use {{"document_status":"active","effective_as_of":"<today>"}};
  leave filters empty for history/comparison questions — evidence carries
  status tags so old versions stay citable.
- resolved_query is always required. Rewrite rules: resolve pronouns and
  references ("it", "that report", "the company") using conversation history;
  expand abbreviations using the glossary above; make relative dates/times
  absolute using today's date; preserve the user's intent — a reader seeing
  ONLY resolved_query must understand the question. Do not answer it.

Example — "extract the X values, compute Y, and chart them":
{{
  "intent": "retrieve",
  "resolved_query": "...",
  "steps": [
    {{"tool": "search", "sub_queries": [{{"query": "X values", "tool_hint": "any"}}]}},
    {{"tool": "extract_data", "args": {{"source": "retrieved_docs", "focus": "X"}}}},
    {{"tool": "code_execute", "spec": {{"task": "compute Y from the extracted points"}}, "needs_prior": true}},
    {{"tool": "chart_generate", "args": {{"chart_type": "bar", "title": "X comparison"}}}}
  ]
}}
For compute-then-chart, code_execute args need output_as_data=true so its
result list appends to accumulated_data for chart_generate.
"""


def _parse_plan_json(text: str) -> Optional[dict]:
    """Extract the plan JSON from model output.

    Tolerates markdown fences and <think>…</think> reasoning blocks (the
    utility role may resolve to a thinking model when UTILITY_MODEL unset).
    """
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _prior_cited_doc_ids(state) -> list[int]:
    lao = state.get("last_answer_object")
    if lao is None:
        return []
    citations = getattr(lao, "citations", None) or []
    ids = []
    for c in citations:
        did = getattr(c, "document_id", None)
        if did is not None and did not in ids:
            ids.append(did)
    return ids


def _normalize_steps(plan: dict) -> list[dict]:
    """Normalize planner output into the step list.

    Accepts explicit ``steps`` or the legacy flat ``sub_queries`` form
    (treated as one search round).
    """
    steps = plan.get("steps")
    if isinstance(steps, list) and steps:
        return [s for s in steps if isinstance(s, dict)][: _MAX_STEPS]
    sq = plan.get("sub_queries")
    if isinstance(sq, list) and sq:
        return [{"tool": "search", "sub_queries": sq}]
    return []


def _search_step_calls(step: dict, kb_ids: list[int]) -> list[dict]:
    """Expand a search step into tool_calls: hybrid pair per sub-query."""
    calls: list[dict] = []
    seen: set[str] = set()

    def _add(tool: str, args: dict) -> None:
        # Sub-queries may overlap (e.g. same title_contains per part) — an
        # identical call is idempotent, so emit it once.
        sig = tool + json.dumps(args, sort_keys=True, default=str)
        if sig in seen or len(calls) >= _MAX_TOOL_CALLS:
            return
        seen.add(sig)
        calls.append({"tool": tool, "arguments": args})

    # Drop token-identical sub-queries — the planner sometimes emits
    # reorderings of the same terms ("GMR-2 attacks" / "attacks on GMR-2"),
    # each fanning out to keyword+semantic for near-duplicate hits.
    seen_sq_keys: set = set()
    for sq in (step.get("sub_queries") or [])[:_MAX_SUB_QUERIES]:
        if not isinstance(sq, dict):
            continue
        query = (sq.get("query") or "").strip()
        sq_key = " ".join(sorted(query.lower().split()))
        if sq_key in seen_sq_keys:
            continue
        seen_sq_keys.add(sq_key)
        hint = (sq.get("tool_hint") or sq.get("tool") or "").strip()
        filters = sq.get("filters") if isinstance(sq.get("filters"), dict) else None
        doc_ids = [d for d in (sq.get("document_ids") or []) if isinstance(d, int)][: _MAX_FILE_READS]

        if hint == "file_read" or (doc_ids and hint not in _SEARCH_TOOLS):
            for did in doc_ids:
                _add("file_read", {"document_id": did})
            if hint == "file_read" and not doc_ids:
                continue
            if not query:
                continue

        if hint == "title_search":
            # metadata_only returns titles, not citable content — pair with
            # hybrid searches on the same query. title_contains must be a
            # short fragment, never the full question.
            frag = (filters or {}).get("title_contains") or query
            _add("title_search", {"title_contains": frag, "kb_ids": kb_ids,
                                  "metadata_only": True})

        for name in ("keyword_search", "semantic_search"):
            args = {"query": query, "kb_ids": kb_ids, "top_k": 10}
            if filters:
                args["filters"] = {k: v for k, v in filters.items() if k != "title_contains"}
            if doc_ids:
                args["document_ids"] = doc_ids
            _add(name, args)

    return calls


def _tool_args_schema_text(tool_name: str, ctx, state) -> str:
    """Field names + descriptions for the tool's args_schema, e.g.
    'text (str): Text to summarize.' — injected into the compile prompt so
    generated arg keys match the schema exactly."""
    from ..tools import applicable_tools
    ctx.state = state
    try:
        tools = {t.name: t for t in applicable_tools(ctx)}
    except Exception:
        tools = {}
    tool_obj = tools.get(tool_name)
    schema = getattr(tool_obj, "args_schema", None)
    if schema is None or not hasattr(schema, "model_fields"):
        return "(schema unavailable — use documented arg names)"
    lines = []
    for name, field in schema.model_fields.items():
        desc = getattr(field, "description", "") or ""
        req = "required" if field.is_required() else "optional"
        lines.append(f"  {name} ({req}): {desc}")
    return "\n".join(lines) or "(no args)"


async def _compile_step_args(step: dict, state, ctx, error_context: str = "") -> dict:
    """Resolve a needs_prior step's args via one utility-model call.

    The compiler sees the tool's arg schema, the step spec, prior-observation
    briefs (with extracted data rows surfaced verbatim for code/data steps)
    and turn context — and returns concrete args JSON for the named tool.
    """
    tool = step.get("tool", "")
    spec = step.get("spec") or {}
    from ..agent_graph.helpers import _result_brief
    obs_briefs = []
    data_blocks = []
    # All observations — the pipeline is bounded (≤4 rounds, ≤8 calls/round)
    # so this stays small; a window could drop the extract rows a code step
    # needs from two rounds back.
    for o in (state.get("observations") or []):
        if isinstance(o, dict):
            tname, res = o.get("tool", "?"), o.get("result") or {}
        else:
            tname, res = getattr(o, "tool", "?"), getattr(o, "result", None) or {}
        brief = _result_brief(res) if isinstance(res, dict) else str(res)
        obs_briefs.append(f"- {tname}: {json.dumps(brief, default=str)[:600]}")
        # Surface extracted rows verbatim — code_execute's compiler needs the
        # real numbers, not a depth-2-compacted preview.
        if isinstance(res, dict) and res.get("data"):
            data_blocks.append(json.dumps(res["data"], default=str)[:2000])
    has_file = bool(state.get("file_markdown"))
    lao = state.get("last_answer_object")
    last_summary = getattr(lao, "summary", "") if lao else ""

    prompt = (
        "Produce the arguments JSON for the tool call below. Reply with ONLY the JSON object.\n\n"
        f"Tool: {tool}\n"
        f"Arguments schema:\n{_tool_args_schema_text(tool, ctx, state)}\n"
        f"Step spec: {json.dumps(spec, default=str)[:1500]}\n"
        f"User question: {state.get('original_query','')}\n"
        + (f"Resolved question: {(state.get('fast_plan') or {}).get('resolved_query')}\n"
           if (state.get('fast_plan') or {}).get('resolved_query')
           and (state.get('fast_plan') or {}).get('resolved_query') != state.get('original_query')
           else "")
        + f"Attached file present: {has_file}\n"
        + f"Previous answer summary: {last_summary or '(none)'}\n"
        + f"Prior tool observations:\n" + ("\n".join(obs_briefs) or "(none)") + "\n"
        + ("Extracted data (use these values in code/data args):\n" + "\n".join(data_blocks) + "\n" if data_blocks else "")
        + (f"Previous attempt failed with: {error_context}\nFix the arguments.\n" if error_context else "")
        + f"\nToday's date: {datetime.now(timezone.utc).date().isoformat()}\n"
        "Args JSON:"
    )
    llm = build_chat_llm(ctx.org_id, ctx.db, role="utility", temperature=0.0)
    resp = await llm.ainvoke([
        SystemMessage(content="You output only valid JSON."),
        HumanMessage(content=prompt),
    ])
    raw = getattr(resp, "content", "") or ""
    debug_emit("fast_arg_compile", {"tool": tool, "prompt": prompt[:6000], "raw_output": raw[:2000]})
    parsed = _parse_plan_json(raw)
    return parsed if isinstance(parsed, dict) else {}


async def _step_to_tool_calls(step: dict, state, ctx, error_context: str = "") -> list[dict]:
    """Materialize one plan step into tool_calls for tool_node_v2."""
    tool = step.get("tool", "")
    kb_ids = state.get("kb_ids", [])

    if tool == "search":
        return _search_step_calls(step, kb_ids)

    if tool not in _STEP_TOOLS and tool not in _SEARCH_TOOLS:
        logger.debug("[fast_step] skipping non-fast tool %r", tool)
        return []

    # Applicability gating — e.g. chart_generate needs data in state; an
    # inapplicable planned step is skipped rather than erroring in dispatch.
    from ..tools import applicable_tools
    ctx.state = state
    try:
        tools = {t.name: t for t in applicable_tools(ctx)}
    except Exception:
        tools = {}
    if tools and tool not in tools:
        logger.debug("[fast_step] skipping inapplicable tool %r", tool)
        return []

    args = step.get("args")
    if not isinstance(args, dict):
        args = {}
    if step.get("needs_prior") or not args:
        try:
            args = await _compile_step_args(step, state, ctx, error_context)
        except Exception as exc:
            logger.warning("[fast_step] arg compile failed for %s: %s", tool, exc)
            return []
        if not args:
            return []
    return [{"tool": tool, "arguments": args}]


async def fast_plan_node(state, ctx) -> dict:
    """One-shot planner: history + last-answer + KB profile → steps + round 1."""
    with _agent_step("fast_plan"):
        writer = _writer()
        query = state.get("original_query", "")
        kb_ids = state.get("kb_ids", [])

        recent = select_recent_history(
            state.get("messages", []),
            max_pairs=get_setting(ctx.db, "AGENT_HISTORY_PAIRS", ctx.org_id),
        )
        history_text = history_to_text(recent) or "(no prior turns)"
        lao = state.get("last_answer_object")
        last_summary = getattr(lao, "summary", "") if lao else ""
        cited_ids = _prior_cited_doc_ids(state)
        from app.services.agentic_rag.kb_profile import format_profile_summary
        kb_profile_text = format_profile_summary(state.get("kb_profile", {}))
        today = datetime.now(timezone.utc).date().isoformat()

        # Abbreviation glossary for the raw question — the planner phrases
        # sub_queries with canonical forms (semantic rewards fluent terms;
        # keyword_search also expands deterministically at match time).
        abbr_text = "(none)"
        try:
            from app.services.abbreviation_service import (
                build_lookup, find_abbrs_in_text, find_forms_in_text)
            _lk = build_lookup(ctx.db, ctx.org_id)
            if not _lk.is_empty:
                _merged = dict(find_abbrs_in_text(query, _lk))
                for _a, _fs in find_forms_in_text(query, _lk).items():
                    _merged.setdefault(_a, _fs)
                if _merged:
                    abbr_text = "\n".join(
                        f"{a} = {', '.join(fs)}"
                        for a, fs in sorted(_merged.items(), key=lambda kv: kv[0].lower()))
        except Exception as exc:
            logger.debug("[fast_plan] abbreviation lookup failed: %s", exc)

        prompt = FAST_PLAN_PROMPT.format(
            history=history_text,
            last_answer_summary=last_summary or "(none)",
            cited_doc_ids=cited_ids or "(none)",
            has_file="yes" if state.get("file_markdown") else "no",
            kb_profile=kb_profile_text or "(empty)",
            today=today,
            abbreviations=abbr_text,
            query=query,
            _MAX_SUB_QUERIES=_MAX_SUB_QUERIES,
        )

        plan: Optional[dict] = None
        raw = ""
        try:
            llm = build_chat_llm(ctx.org_id, ctx.db, role="utility", temperature=0.0)
            resp = await llm.ainvoke([
                SystemMessage(content="You output only valid JSON."),
                HumanMessage(content=prompt),
            ])
            raw = getattr(resp, "content", "") or ""
            plan = _parse_plan_json(raw)
        except Exception as exc:
            logger.warning("[fast_plan] planner call failed: %s", exc)

        intent = (plan or {}).get("intent") or "retrieve"
        resolved = (plan or {}).get("resolved_query") or query
        steps = _normalize_steps(plan or {}) if intent in ("retrieve", "answer_from_history") else []

        # Planner produced steps for a history intent anyway — honor the steps
        # (it decided grounding was needed).
        fast_plan = {
            "intent": intent, "resolved_query": resolved, "steps": steps,
            "cursor": 0, "repairs": 0, "last_emitted": 0, "raw": raw[:2000],
        }

        # Compile the first step inline; later steps go through fast_step.
        tool_calls: list[dict] = []
        if steps:
            tool_calls = await _step_to_tool_calls(steps[0], {**state, "fast_plan": fast_plan}, ctx)
            fast_plan["cursor"] = 1
            fast_plan["last_emitted"] = len(tool_calls)

        # Fallback: planner failed or produced nothing for a retrieval intent —
        # one hybrid pair on the resolved query so the turn still retrieves.
        if not tool_calls and intent == "retrieve":
            tool_calls = _search_step_calls(
                {"tool": "search", "sub_queries": [{"query": resolved}]}, kb_ids)
            fast_plan["last_emitted"] = len(tool_calls)

        debug_emit("fast_plan", {
            "prompt": prompt[:8000],
            "raw_output": raw[:4000],
            "parsed": plan,
            "tool_calls": tool_calls,
        })
        writer({"event": "plan", "plan": {
            "mode": "fast",
            "intent": intent,
            "resolved_query": resolved,
            "steps": steps,
            "tool_calls": [{"tool": c["tool"], "arguments": c["arguments"]} for c in tool_calls],
        }})

        return {"tool_calls": tool_calls, "fast_plan": fast_plan}


async def fast_step_node(state, ctx) -> dict:
    """Resolve the next planned step (or a one-shot repair) into tool_calls."""
    with _agent_step("fast_step"):
        return await _fast_step_inner(state, ctx)


async def _fast_step_inner(state, ctx) -> dict:
    plan = dict(state.get("fast_plan") or {})
    steps = plan.get("steps") or []
    cursor = plan.get("cursor", 1)
    repairs = plan.get("repairs", 0)
    last_n = plan.get("last_emitted", 0)

    # Inspect the just-finished round's observations.
    all_obs = state.get("observations") or []
    prev_obs = all_obs[-last_n:] if last_n else []
    prev_errors = [
        getattr(o, "error", None) if not isinstance(o, dict) else o.get("error")
        for o in prev_obs
    ]
    prev_all_errored = bool(prev_obs) and all(prev_errors)

    if prev_all_errored:
        prev_step = steps[cursor - 1] if cursor > 0 else {}
        # Repair-once only helps when args can change — a needs_prior/spec
        # step gets recompiled with the error in context. Re-emitting literal
        # args would deterministically fail again, so skip straight to the
        # answer with the partial evidence already merged.
        repairable = repairs == 0 and (
            prev_step.get("needs_prior") or isinstance(prev_step.get("spec"), dict))
        if repairable:
            err_txt = "; ".join(str(e) for e in prev_errors if e)[:800]
            fixed = await _step_to_tool_calls(prev_step, state, ctx, error_context=err_txt)
            plan["repairs"] = 1
            if fixed:
                plan["last_emitted"] = len(fixed)
                return {"tool_calls": fixed, "fast_plan": plan}
        return {"tool_calls": [], "fast_plan": plan}

    if cursor >= len(steps):
        return {"tool_calls": [], "fast_plan": plan}

    step = steps[cursor]
    tool_calls = await _step_to_tool_calls(step, state, ctx)
    plan["cursor"] = cursor + 1
    plan["last_emitted"] = len(tool_calls)
    plan["repairs"] = repairs
    if not tool_calls:
        # Step produced nothing (skipped/bad compile) — try the next step once.
        if plan["cursor"] < len(steps):
            tool_calls = await _step_to_tool_calls(steps[plan["cursor"]], state, ctx)
            plan["cursor"] += 1
            plan["last_emitted"] = len(tool_calls)
    return {"tool_calls": tool_calls, "fast_plan": plan}


def route_fast_plan(state) -> str:
    """fast_plan → tool when round 1 produced calls, else post_process."""
    return "tool" if state.get("tool_calls") else "post_process"


def route_fast_step(state) -> str:
    """fast_step → tool for another round, else post_process."""
    return "tool" if state.get("tool_calls") else "post_process"
