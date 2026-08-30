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
from app.services.agentic_rag.llm_factory import build_chat_llm
from app.services.agentic_rag.nodes import _agent_step
from app.services.agentic_rag.prompts import AGENT_SYSTEM_PROMPT, PLAN_SYSTEM_PROMPT
from app.services.agentic_rag.schemas import Plan, Subtask
from app.services.settings_service import get_setting

from .helpers import _extract_json_block, _writer

logger = logging.getLogger(__name__)


def _build_plan_user_prompt(original, rewritten, clarification, glossary, last_summary, recalled_text, file_meta):
    return (
        f"User message: {original}\n"
        f"Retrieval query: {rewritten}\n"
        + (f"User clarification: {clarification}\n" if clarification else "")
        + (f"[Abbreviation Glossary]\n{glossary}\n\n" if glossary else "")
        + f"Previous answer summary: {last_summary}\n"
        f"Recalled long-term memory (context only, not evidence):\n{recalled_text}\n\n"
        f"Attached files: {json.dumps(file_meta)}\n\n"
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
        logger.info("[plan_node] clarification budget exhausted — proceeding without asking")
        needs_clarification = False
        if isinstance(plan, Plan):
            plan.needs_clarification = False
    return needs_clarification


async def plan_node(state, ctx) -> dict:
    """Produce a structured plan for the current turn."""
    with _agent_step("plan"):
        writer = _writer()
        original = state.get("original_query", "")
        rewritten = state.get("rewritten_query", "") or original

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

        # Glossary was built once by expand_query_node — reuse it.
        glossary = state.get("abbreviation_glossary", "")

        user = _build_plan_user_prompt(original, rewritten, clarification, glossary, last_summary, recalled_text, file_meta)

        plan = await _invoke_plan_llm(ctx, system, user, rewritten)

        writer({"event": "plan", "plan": plan.model_dump() if isinstance(plan, Plan) else plan})

        needs_clarification = _check_clarification_budget(plan, state, ctx)

        return {
            "plan": plan,
            "needs_clarification": needs_clarification,
            "clarification_question": getattr(plan, "clarification_question", None),
        }


def route_plan(state) -> str:
    if state.get("needs_clarification"):
        return "clarify_interrupt"
    return "think"
