"""Planning node — produces a structured plan for the current turn.

Calls the plan LLM with the user message, clarification, previous answer
summary, recalled memory, and attached file metadata. Falls back to JSON
parsing if structured output fails. Enforces a clarification budget so the
agent doesn't ask the user indefinitely.
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


def _build_plan_user_prompt(original, clarification, last_summary, recalled_text, file_meta, kb_profile_text=""):
    # Order: stable → volatile for prefix cache reuse.
    # KB profile is stable within a KB. File metadata is stable within a chat.
    # Last summary / recalled / query change every turn.
    return (
        (f"{kb_profile_text}\n\n" if kb_profile_text else "")
        + f"Attached files: {json.dumps(file_meta)}\n\n"
        + f"Previous answer summary: {last_summary}\n"
        f"Recalled long-term memory (context only, not evidence):\n{recalled_text}\n\n"
        + (f"User clarification: {clarification}\n" if clarification else "")
        + f"User message: {original}\n"
        "Produce a plan JSON matching the schema."
    )


async def _invoke_plan_llm(ctx, system, user, original):
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
            plan = Plan(intent="rag", subtasks=[Subtask(id="a", description=original, tool_hint="search_dense")])
    return plan


def _check_clarification_budget(plan, state, ctx):
    needs_clarification = bool(getattr(plan, "needs_clarification", False))
    if needs_clarification and state.get("clarification_count", 0) >= get_setting(ctx.db, "AGENT_MAX_CLARIFICATIONS", ctx.org_id):
        logger.debug("[plan_node] clarification budget exhausted — proceeding without asking")
        needs_clarification = False
        if isinstance(plan, Plan):
            plan.needs_clarification = False
    return needs_clarification


async def plan_node(state, ctx) -> dict:
    """Produce a structured plan for the current turn."""
    with _agent_step("plan"):
        writer = _writer()
        original = state.get("original_query", "")

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

        kb_profile_text = format_profile_summary(state.get("kb_profile", {}))
        user = _build_plan_user_prompt(original, clarification, last_summary, recalled_text, file_meta, kb_profile_text)

        plan = await _invoke_plan_llm(ctx, system, user, original)

        writer({"event": "plan", "plan": plan.model_dump() if isinstance(plan, Plan) else plan})

        needs_clarification = _check_clarification_budget(plan, state, ctx)

        # Pre-populate tool calls for independent subtasks (no depends_on)
        # that use atomic search tools or kb_search_documents. This ensures
        # the first round of retrieval uses the correct strategy without
        # depending on the think LLM noticing per-subtask params in its prompt.
        precomputed_tool_calls: list[dict] = []
        if not needs_clarification and isinstance(plan, Plan):
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

                # kb_search_documents: document-level retrieval.
                if st.tool_hint == "kb_search_documents":
                    suggested_filters = st.suggested_filters or {}
                    suggested_sort = st.suggested_sort
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
                    title_contains = suggested_filters.get("title_contains")
                    if title_contains:
                        call["arguments"]["title_contains"] = title_contains
                    if st.suggested_metadata_only:
                        call["arguments"]["metadata_only"] = True
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
                    continue

                # Atomic search tools: pre-populate from suggested_query/filters.
                if st.tool_hint in ("search_exact", "search_sparse", "search_dense"):
                    query = st.suggested_query or original
                    call = {"tool": st.tool_hint, "arguments": {"query": query}}
                    suggested_filters = st.suggested_filters
                    if suggested_filters:
                        call["arguments"]["filters"] = suggested_filters
                    if st.suggested_top_n:
                        call["arguments"]["top_k"] = st.suggested_top_n
                    precomputed_tool_calls.append(call)
                    logger.debug(
                        "[plan_node] pre-populated %s for subtask %s: %s",
                        st.tool_hint, st.id, call["arguments"],
                    )
                    continue

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
