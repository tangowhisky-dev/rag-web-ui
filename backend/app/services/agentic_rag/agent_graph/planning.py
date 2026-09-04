"""Planning node — produces a structured plan for the current turn.

Calls the plan LLM with the user message, rewritten query, clarification,
glossary, previous answer summary, recalled memory, and attached file
metadata. Falls back to JSON parsing if structured output fails. Enforces
a clarification budget so the agent doesn't ask the user indefinitely.
"""

from __future__ import annotations

import json
import logging

from app.models.chat import ChatFile
from app.services.agentic_rag.kb_profile import format_profile_summary
from app.services.agentic_rag.llm_factory import build_chat_llm
from app.services.agentic_rag.nodes import _agent_step
from app.services.agentic_rag.prompts import AGENT_SYSTEM_PROMPT, PLAN_SYSTEM_PROMPT
from app.services.agentic_rag.schemas import Plan, Subtask
from app.services.settings_service import get_setting

from .helpers import _extract_json_block, _writer

logger = logging.getLogger(__name__)


def _build_plan_user_prompt(original, rewritten, clarification, glossary, last_summary, recalled_text, file_meta, kb_profile_text=""):
    # Order: stable → volatile for prefix cache reuse.
    # KB profile is stable within a KB. File metadata is stable within a chat.
    # Glossary is stable within a turn. Last summary / recalled / query change every turn.
    return (
        (f"{kb_profile_text}\n\n" if kb_profile_text else "")
        + f"Attached files: {json.dumps(file_meta)}\n\n"
        + (f"[Abbreviation Glossary]\n{glossary}\n\n" if glossary else "")
        + f"Previous answer summary: {last_summary}\n"
        f"Recalled long-term memory (context only, not evidence):\n{recalled_text}\n\n"
        + (f"User clarification: {clarification}\n" if clarification else "")
        + f"Retrieval query: {rewritten}\n"
        f"User message: {original}\n"
        "Produce a plan JSON matching the schema."
    )


async def _invoke_plan_llm(ctx, system, user, rewritten):
    try:
        llm = build_chat_llm(ctx.org_id, ctx.db, role="query", temperature=0.0)
        structured = llm.with_structured_output(Plan, method="json_schema", include_raw=True)
        resp = await structured.ainvoke([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
        # include_raw=True returns a dict with 'raw', 'parsed', 'parsing_error'
        if isinstance(resp, dict):
            plan = resp.get("parsed")
            if plan is None or resp.get("parsing_error"):
                raise resp.get("parsing_error") or ValueError("structured output parsed to None")
        else:
            plan = resp.parsed if hasattr(resp, "parsed") else resp
    except Exception as exc:
        logger.warning("[plan_node] structured output failed: %s; using JSON parse fallback", exc)
        llm = build_chat_llm(ctx.org_id, ctx.db, role="query", temperature=0.0)
        resp = await llm.ainvoke([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
        raw = str(resp.content)
        block = _extract_json_block(raw)
        try:
            plan = Plan.model_validate_json(block) if block else Plan()
        except Exception as parse_exc:
            logger.warning("[plan_node] JSON parse failed: %s", parse_exc)
            plan = Plan(intent="rag", subtasks=[Subtask(id="a", description=rewritten, tool_hint="rag_retrieve")])
    return plan


def _check_clarification_budget(plan, state, ctx):
    needs_clarification = bool(getattr(plan, "needs_clarification", False))
    if needs_clarification and state.get("clarification_count", 0) >= get_setting(ctx.db, "AGENT_MAX_CLARIFICATIONS", ctx.org_id):
        logger.debug("[plan_node] clarification budget exhausted — proceeding without asking")
        needs_clarification = False
        if isinstance(plan, Plan):
            plan.needs_clarification = False
    return needs_clarification


_AGGREGATE_KEYWORDS = frozenset({
    "how many", "count", "summarize all", "compare", "table", "chart",
    "list all", "aggregate", "breakdown", "statistics", "all of", "every ",
    "each ", "trend", "distribution", "summary of all", "overview of all",
})


def _is_simple_lookup(original: str) -> bool:
    """Heuristic: is this a single-document lookup that can skip the plan LLM?

    Fast-track is appropriate for queries like "what is in the latest weekly
    update" or "show me the Q3 report". It is NOT appropriate for aggregate
    queries ("how many weekly updates this year"), comparison queries, or
    queries that need chart/table generation — those need the plan LLM to
    decompose into multiple subtasks.
    """
    if len(original.split()) > 12:
        return False
    lower = original.lower()
    if any(kw in lower for kw in _AGGREGATE_KEYWORDS):
        return False
    return True


async def plan_node(state, ctx) -> dict:
    """Produce a structured plan for the current turn."""
    with _agent_step("plan"):
        writer = _writer()
        original = state.get("original_query", "")
        rewritten = state.get("rewritten_query", "") or original

        # ── Tier-0 fast-track ────────────────────────────────────────────
        # When query_intent already has title_contains AND the query is a
        # simple single-document lookup, skip the plan LLM entirely —
        # create a minimal plan and pre-populate a kb_search_documents call.
        # This saves ~2-5s of plan LLM latency for title-specific queries.
        # Aggregate/analysis queries fall through to the plan LLM so it can
        # decompose them into discovery → retrieval → extraction → chart.
        qi = state.get("query_intent") or {}
        title_contains = (qi.get("suggested_filters") or {}).get("title_contains")
        if title_contains and not state.get("clarification_response") and _is_simple_lookup(original):
            suggested_sort = qi.get("suggested_sort")
            sort_field = "file_modified_at"
            sort_direction = "desc"
            if suggested_sort and isinstance(suggested_sort, dict):
                sort_field = suggested_sort.get("field", "file_modified_at")
                sort_direction = suggested_sort.get("direction", "desc")
            fast_plan = Plan(
                intent="rag",
                subtasks=[Subtask(
                    id="a",
                    description=f"Read document matching title '{title_contains}'",
                    tool_hint="kb_search_documents",
                    suggested_filters=qi.get("suggested_filters"),
                    suggested_sort=suggested_sort,
                )],
            )
            call: dict = {
                "tool": "kb_search_documents",
                "arguments": {
                    "title_contains": title_contains,
                    "sort_field": sort_field,
                    "sort_direction": sort_direction,
                    "top_n": 3,
                    "max_tokens_per_doc": 16000,
                },
            }
            logger.debug("[plan_node] Tier-0 fast-track: title_contains=%r, skipping plan LLM", title_contains)
            writer({"event": "plan", "plan": fast_plan.model_dump(), "fast_track": True})
            return {
                "plan": fast_plan,
                "needs_clarification": False,
                "clarification_question": None,
                "precomputed_tool_calls": [call],
            }

        file_meta = []
        if ctx.chat_id:
            files = ctx.db.query(ChatFile).filter(ChatFile.chat_id == ctx.chat_id).all()
            file_meta = [{"id": f.id, "name": f.file_name, "type": f.content_type} for f in files]

        last_summary = ""
        lao = state.get("last_answer_object")
        if lao and hasattr(lao, "summary"):
            last_summary = lao.summary
            if getattr(lao, "chart_options", None):
                last_summary += f" (Previous answer includes {len(lao.chart_options)} chart(s) with structured data.)"

        recalled = state.get("recalled_memories", []) or []
        recalled_text = "\n".join(d.get("page_content", "") for d in recalled[:3])
        clarification = (state.get("clarification_response") or "").strip()

        system = AGENT_SYSTEM_PROMPT + "\n\n" + PLAN_SYSTEM_PROMPT

        # Inject current date so the plan LLM can produce correct date filters
        # (e.g. file_modified_after for "this year" queries) without needing
        # a current_datetime tool call first.
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        system += f"\n\n[Current Date: {now.strftime('%Y-%m-%d')} UTC — use this when producing date filters in suggested_filters]"

        # Glossary was built once by expand_query_node — reuse it.
        glossary = state.get("abbreviation_glossary", "")

        kb_profile_text = format_profile_summary(state.get("kb_profile", {}))
        user = _build_plan_user_prompt(original, rewritten, clarification, glossary, last_summary, recalled_text, file_meta, kb_profile_text)

        plan = await _invoke_plan_llm(ctx, system, user, rewritten)

        writer({"event": "plan", "plan": plan.model_dump() if isinstance(plan, Plan) else plan})

        needs_clarification = _check_clarification_budget(plan, state, ctx)

        # Pre-populate tool calls for ALL independent subtasks (no
        # depends_on) that use rag_retrieve or kb_search_documents.
        # This ensures the first round of retrieval uses the correct
        # strategy without depending on the think LLM noticing the
        # [Query Intent] JSON or per-subtask params in its prompt.
        #
        # Priority for retrieval params:
        #   1. Per-subtask suggested_filters/sort/legs (from the plan LLM)
        #   2. Global query_intent (from the rewrite LLM)
        #
        # When title_contains is present, use kb_search_documents
        # (document-level retrieval) instead of rag_retrieve (chunk-level).
        precomputed_tool_calls: list[dict] = []
        if not needs_clarification and isinstance(plan, Plan):
            qi = state.get("query_intent") or {}
            rewritten = state.get("rewritten_query", "") or original

            for st in plan.subtasks:
                if st.depends_on:
                    continue  # dependent subtasks wait for their deps

                # current_datetime: no args needed.
                if st.tool_hint == "current_datetime":
                    call = {"tool": "current_datetime", "arguments": {}}
                    precomputed_tool_calls.append(call)
                    logger.debug("[plan_node] pre-populated current_datetime for subtask %s", st.id)
                    continue

                # kb_metadata: pre-populate based on action hints.
                if st.tool_hint == "kb_metadata":
                    call = {"tool": "kb_metadata", "arguments": {"action": "list_documents", "limit": 50}}
                    suggested_filters = st.suggested_filters or {}
                    if suggested_filters.get("title_contains"):
                        call["arguments"]["value_contains"] = suggested_filters["title_contains"]
                    precomputed_tool_calls.append(call)
                    logger.debug("[plan_node] pre-populated kb_metadata for subtask %s: %s", st.id, call["arguments"])
                    continue

                if st.tool_hint not in ("rag_retrieve", "kb_search_documents", "any"):
                    continue

                # Merge per-subtask params with global query_intent.
                # Per-subtask params take priority.
                suggested_filters = st.suggested_filters or qi.get("suggested_filters") or {}
                suggested_sort = st.suggested_sort or qi.get("suggested_sort")
                suggested_legs = st.suggested_legs or qi.get("suggested_legs")
                title_contains = suggested_filters.get("title_contains") if suggested_filters else None

                if title_contains or st.tool_hint == "kb_search_documents":
                    # Document-level retrieval: read the full file, skip
                    # chunks and reranker entirely.
                    sort_field = "file_modified_at"
                    sort_direction = "desc"
                    if suggested_sort and isinstance(suggested_sort, dict):
                        sort_field = suggested_sort.get("field", "file_modified_at")
                        sort_direction = suggested_sort.get("direction", "desc")
                    call: dict = {
                        "tool": "kb_search_documents",
                        "arguments": {
                            "sort_field": sort_field,
                            "sort_direction": sort_direction,
                            "top_n": st.suggested_top_n or 3,
                            "max_tokens_per_doc": 16000,
                        },
                    }
                    if title_contains:
                        call["arguments"]["title_contains"] = title_contains
                    if st.suggested_metadata_only:
                        call["arguments"]["metadata_only"] = True
                    # Pass date filters from suggested_filters
                    if suggested_filters.get("file_modified_after"):
                        call["arguments"]["modified_after"] = suggested_filters["file_modified_after"]
                    if suggested_filters.get("file_modified_before"):
                        call["arguments"]["modified_before"] = suggested_filters["file_modified_before"]
                    if suggested_filters.get("content_type"):
                        call["arguments"]["content_type"] = suggested_filters["content_type"]
                    precomputed_tool_calls.append(call)
                    logger.debug(
                        "[plan_node] pre-populated kb_search_documents for subtask %s: %s",
                        st.id, call["arguments"],
                    )
                elif suggested_filters or suggested_sort or suggested_legs or st.suggested_query:
                    # Chunk-level retrieval with filters/sort/legs.
                    call = {"tool": "rag_retrieve", "arguments": {"query": st.suggested_query or rewritten}}
                    if suggested_filters:
                        call["arguments"]["filters"] = suggested_filters
                    if suggested_sort:
                        call["arguments"]["sort"] = suggested_sort
                    if suggested_legs:
                        call["arguments"]["legs"] = suggested_legs
                    precomputed_tool_calls.append(call)
                    logger.debug(
                        "[plan_node] pre-populated rag_retrieve for subtask %s: %s",
                        st.id, call["arguments"],
                    )
                # If no params and tool_hint is "any" or "rag_retrieve",
                # don't precompute — let the think LLM decide the strategy.
                # Only the first subtask gets a fallback from query_intent
                # to preserve the original behavior for simple queries.
                elif not precomputed_tool_calls and _has_intent_params(qi) and st.id == plan.subtasks[0].id:
                    call = {"tool": "rag_retrieve", "arguments": {"query": rewritten}}
                    if qi.get("suggested_filters"):
                        call["arguments"]["filters"] = qi["suggested_filters"]
                    if qi.get("suggested_sort"):
                        call["arguments"]["sort"] = qi["suggested_sort"]
                    if qi.get("suggested_legs"):
                        call["arguments"]["legs"] = qi["suggested_legs"]
                    precomputed_tool_calls.append(call)
                    logger.debug(
                        "[plan_node] pre-populated rag_retrieve for first subtask from query_intent: %s",
                        call["arguments"],
                    )

        result = {
            "plan": plan,
            "needs_clarification": needs_clarification,
            "clarification_question": getattr(plan, "clarification_question", None),
        }
        if precomputed_tool_calls:
            result["precomputed_tool_calls"] = precomputed_tool_calls
        return result


def route_plan(state) -> str:
    if state.get("needs_clarification"):
        return "clarify_interrupt"
    return "think"


def _has_intent_params(qi: dict) -> bool:
    """Check whether query_intent has any actionable filters/sort/legs."""
    return bool(qi.get("suggested_filters") or qi.get("suggested_sort") or qi.get("suggested_legs"))
