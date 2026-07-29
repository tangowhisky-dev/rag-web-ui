"""Agent loop graph for the enterprise agent.

Replaces the rigid RAG pipeline with a tool-calling loop:
  load_context → rewrite_query → compaction → plan → think → [tool → think ...] → finalize → save_memory
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from functools import partial
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from app.core.config import settings
from app.models.chat import ChatFile, Message
from langgraph.config import get_stream_writer


def _writer():
    """Return a stream writer if one is available, else a no-op."""
    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda x: None


# Per-turn call caps for tools that can be invoked in a tight loop.
_TOOL_CALL_BUDGET = {
    "rag_retrieve": settings.AGENT_MAX_RETRIEVALS,
    "code_execute": settings.AGENT_MAX_CODE_EXEC,
}


def _extract_balanced(text: str, chars: tuple[str, str]) -> str | None:
    """Return the first balanced *chars* region in *text* while respecting strings."""
    start_char, end_char = chars
    start = text.find(start_char)
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def _extract_json_block(text: str) -> str | None:
    """Return the first well-formed JSON object or array string from *text*.

    Tries markdown fenced blocks first, then scans for balanced braces or brackets.
    """
    if not text:
        return None
    # Prefer a fenced ```json ... ``` block.
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        block = _extract_balanced(m.group(1), ("{", "}")) or _extract_balanced(m.group(1), ("[", "]"))
        if block:
            return block
    # Fall back to the first inline balanced object or array.
    return _extract_balanced(text, ("{", "}")) or _extract_balanced(text, ("[", "]"))


from app.services.agentic_rag.llm_factory import build_chat_llm
from app.services.agentic_rag.nodes import _agent_step, answer_evaluation_node, compaction_node, rewrite_query_node, select_recent_history
from app.services.agentic_rag.prompts import (
    AGENT_SYSTEM_PROMPT,
    ANSWER_SYSTEM_PROMPT_BASE,
    LAST_ANSWER_EXTRACT_PROMPT,
    PLAN_SYSTEM_PROMPT,
    REFLECT_SYSTEM_PROMPT,
    THINK_SYSTEM_PROMPT,
)
from app.services.agentic_rag.schemas import LastAnswerObject, Observation, Plan, Subtask
from app.services.agentic_rag.tool_call_parser import parse_think_response
from app.services.agentic_rag.tool_context import ToolContext, write_audit
from app.services.agentic_rag.tools import applicable_tools
from app.services.agentic_rag.token_budget import count_tokens
from app.services.agentic_rag.utils import format_context_string

from .graph_state import AgentState

logger = logging.getLogger(__name__)


def _tool_descriptions_text(tools: list) -> str:
    lines = []
    for t in tools:
        lines.append(f"- {t.name}: {t.description}")
        # Include the args schema so the LLM knows the exact field names and
        # types. Essential for json_text mode where bind_tools is not called;
        # harmless in native mode (the schema is redundant but consistent).
        schema = t.args_schema.model_json_schema()
        props = schema.get("properties", {})
        required = schema.get("required", [])
        field_lines = []
        for fname, finfo in props.items():
            ftype = finfo.get("type", "any")
            desc = finfo.get("description", "")
            req = " (required)" if fname in required else ""
            field_lines.append(f"    {fname}: {ftype}{req} — {desc}")
        if field_lines:
            lines.append("  args:")
            lines.extend(field_lines)
    return "\n".join(lines)


def _observations_text(observations: list[Observation]) -> str:
    parts = []
    for i, obs in enumerate(observations, 1):
        parts.append(f"Observation {i}: tool={obs.tool} args={obs.arguments}")
        if obs.error:
            parts.append(f"  error: {obs.error}")
        else:
            summary = json.dumps(obs.result, default=str)[:500]
            parts.append(f"  result: {summary}")
    return "\n".join(parts)


async def load_context_node(state: AgentState, ctx: ToolContext) -> dict:
    """Load previous-answer object, recalled memory, and file metadata into state."""
    with _agent_step("load_context"):
        last_obj: Optional[LastAnswerObject] = None
        if ctx.chat_id and ctx.message_id:
            # The current assistant message may already exist; find the previous assistant message.
            prev = (
                ctx.db.query(Message)
                .filter(Message.chat_id == ctx.chat_id, Message.role == "assistant")
                .filter(Message.id != ctx.message_id)
                .order_by(Message.id.desc())
                .first()
            )
            if prev and prev.last_answer_object:
                try:
                    last_obj = LastAnswerObject(**prev.last_answer_object)
                except Exception:
                    last_obj = None
    
        recalled: list[dict] = []
        if ctx.redis_memory and getattr(ctx.redis_memory, "search_memory", None):
            try:
                recalled = await ctx.redis_memory.search_memory(
                    query=state.get("original_query", ""),
                    user_id=ctx.user_id,
                    chat_id=ctx.chat_id,
                    limit=3,
                )
            except Exception as exc:
                logger.warning("[load_context] memory search failed: %s", exc)
    
        return {
            "last_answer_object": last_obj,
            "retrieved_docs": recalled,
            "org_id": ctx.org_id,
            "user_id": ctx.user_id,
            "chat_id": ctx.chat_id,
            "message_id": ctx.message_id,
        }
    
    
async def plan_node(state: AgentState, ctx: ToolContext) -> dict:
    """Produce a structured plan for the current turn."""
    with _agent_step("plan"):
        writer = _writer()
        original = state.get("original_query", "")
        rewritten = state.get("rewritten_query", original)
    
        file_meta = []
        if ctx.chat_id:
            files = ctx.db.query(ChatFile).filter(ChatFile.chat_id == ctx.chat_id).all()
            file_meta = [{"id": f.id, "name": f.file_name, "type": f.content_type} for f in files]
    
        last_summary = ""
        lao = state.get("last_answer_object")
        if lao and hasattr(lao, "summary"):
            last_summary = lao.summary
    
        recalled = state.get("retrieved_docs", [])
        recalled_text = "\n".join(d.get("page_content", "") for d in recalled[:3])
    
        system = AGENT_SYSTEM_PROMPT + "\n\n" + PLAN_SYSTEM_PROMPT
        user = (
            f"Original query: {original}\n"
            f"Rewritten query: {rewritten}\n"
            f"Previous answer summary: {last_summary}\n"
            f"Recalled long-term memory:\n{recalled_text}\n\n"
            f"Attached files: {json.dumps(file_meta)}\n\n"
            "Produce a plan JSON matching the schema."
        )
    
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
                plan = Plan(intent="rag", subtasks=[Subtask(id="a", description=original, tool_hint="rag_retrieve")])
    
        writer({"event": "plan", "plan": plan.model_dump() if isinstance(plan, Plan) else plan})
    
        return {"plan": plan, "needs_clarification": plan.needs_clarification}
    
    
async def think_node(state: AgentState, ctx: ToolContext) -> dict:
    """Decide the next action: emit one or more tool calls or a final answer."""
    with _agent_step("think"):
        ctx.state = state
        iteration = state.get("iteration", 0) + 1
        max_iter = settings.AGENT_MAX_ITERATIONS
    
        if state.get("force_finalize"):
            return {"iteration": iteration, "tool_calls": [], "precomputed_answer": ""}
    
        precomputed = state.get("precomputed_tool_calls", [])
        if precomputed:
            return {"iteration": iteration, "tool_calls": list(precomputed), "precomputed_tool_calls": []}
    
        original = state.get("original_query", "")
        plan = state.get("plan") or Plan()
        observations = state.get("observations", [])
        # Expose current state to tools so applicable_tools() and tool reads
        # (last_answer_object, retrieved_docs, kb_ids, file_markdown) see live data.
        ctx.state = state
        tools = applicable_tools(ctx)
        tools_text = _tool_descriptions_text(tools)

        # Build conversation context so the agent can handle multi-turn references.
        recent = select_recent_history(state.get("messages", []), max_pairs=3)
        history_text = ""
        for msg in recent:
            role = "User" if msg.type == "human" else "Assistant"
            content = str(msg.content)[:500]
            history_text += f"  {role}: {content}\n"

        # Include last_answer_object summary so "summarize it" / "chart it" work.
        lao = state.get("last_answer_object")
        lao_text = ""
        if lao and hasattr(lao, "summary"):
            lao_text = f"  Previous answer summary: {lao.summary[:300]}\n"
            if lao.key_points:
                lao_text += f"  Key points: {'; '.join(lao.key_points[:5])}\n"

        system = AGENT_SYSTEM_PROMPT + "\n\n" + THINK_SYSTEM_PROMPT
        user = (
            f"Iteration: {iteration}/{max_iter}\n"
            f"User query: {original}\n"
            f"Conversation history (recent turns):\n{history_text or '  (none)'}\n"
            f"Previous answer context:\n{lao_text or '  (none)'}\n"
            f"Plan: {json.dumps(plan.model_dump() if isinstance(plan, Plan) else plan, default=str)}\n"
            f"Observations so far:\n{_observations_text(observations)}\n\n"
            f"Available tools:\n{tools_text}\n\n"
            "Emit either {\"tool_calls\": [...]} or {\"final_answer\": \"...\"}."
        )
    
        mode = settings.TOOL_CALL_MODE
        try:
            if mode == "json_text":
                llm = build_chat_llm(ctx.org_id, ctx.db, role="chat", temperature=0.7)
                resp = await llm.ainvoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
            else:
                # native or auto: bind tools; parser falls back to JSON-text if native call absent.
                llm = build_chat_llm(ctx.org_id, ctx.db, role="chat", temperature=0.7)
                resp = await llm.bind_tools(tools).ainvoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
        except Exception as exc:
            logger.warning("[think_node] LLM call failed: %s", exc)
            return {"iteration": iteration, "tool_calls": [], "precomputed_answer": f"LLM error: {exc}"}
    
        parsed = parse_think_response(resp, mode=mode)
        tool_calls = parsed.tool_calls
        final_answer = parsed.final_answer
    
        if iteration >= max_iter:
            tool_calls = []
    
        # Dependency guard: only allow independent tool calls in one message.
        allowed = list(tool_calls)
    
        if tool_calls and not final_answer:
            return {"iteration": iteration, "tool_calls": allowed}
        return {"iteration": iteration, "tool_calls": [], "precomputed_answer": final_answer or ""}
    
    
def route_think(state: AgentState) -> str:
    iteration = state.get("iteration", 0)
    if iteration >= settings.AGENT_MAX_ITERATIONS:
        return "reflect_final"
    if state.get("tool_calls"):
        return "tool"
    return "reflect_final"


async def tool_node(state: AgentState, ctx: ToolContext) -> dict:
    """Dispatch tool calls, run them (in parallel when independent), record observations."""
    with _agent_step("tool"):
        writer = _writer()
        tool_calls = state.get("tool_calls", [])
        if not tool_calls:
            return {}
    
        # Expose current state to tools so they can read last_answer_object,
        # retrieved_docs, kb_ids, file_markdown, message_id, iteration, etc.
        ctx.state = state
        tools = {t.name: t for t in applicable_tools(ctx)}
        observations = list(state.get("observations", []))
        counts = dict(state.get("tool_call_count", {}))
    
        async def _budget_exceeded(name, args, cap):
            return {"tool": name, "arguments": args, "result": {}, "error": f"Budget exceeded: {name} call cap is {cap}", "tokens": 0}

        coros = []
        for tc in tool_calls:
            name = tc.get("tool")
            args = tc.get("arguments", {})
            writer({"event": "tool_call", "tool": name, "arguments": args})
            cap = _TOOL_CALL_BUDGET.get(name)
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
            observations.append(obs)
            writer({"event": "tool_observation", **obs.model_dump()})
            counts[obs.tool] = counts.get(obs.tool, 0) + 1
    
        # Promote the latest rag_retrieve docs/confidence into graph state so
        # finalize_node, answer_evaluation_node, extract_data(source="retrieved_docs"),
        # and the citations payload in agent_runner all see the retrieved chunks.
        state_update: dict = {"tool_calls": [], "observations": observations, "tool_call_count": counts}
        for obs in observations:
            if obs.tool == "rag_retrieve" and not obs.error:
                docs = obs.result.get("docs")
                if isinstance(docs, list) and docs:
                    state_update["retrieved_docs"] = docs
                    state_update["retrieval_confidence"] = obs.result.get("confidence", 0.0)
                    break
    
        return state_update
    
    
async def _run_tool(tool, name: str, args: dict) -> dict:
    try:
        result = await tool.arun(args)
        return {"tool": name, "arguments": args, "result": result, "error": None, "tokens": 0}
    except Exception as exc:
        return {"tool": name, "arguments": args, "result": {}, "error": str(exc), "tokens": 0}


def route_plan(state: AgentState) -> str:
    if state.get("needs_clarification"):
        return "clarify_interrupt"
    return "think"


async def finalize_node(state: AgentState, ctx: ToolContext) -> dict:
    """Generate final answer if not precomputed; extract LastAnswerObject."""
    with _agent_step("finalize"):
        writer = _writer()
        precomputed = state.get("precomputed_answer", "")
        original = state.get("original_query", "")
        observations = state.get("observations", [])
        docs = state.get("retrieved_docs", [])
    
        if precomputed:
            final = precomputed
        else:
            context_text = format_context_string(docs, state.get("file_markdown"))
            system = (
                AGENT_SYSTEM_PROMPT
                + "\n\n"
                + ANSWER_SYSTEM_PROMPT_BASE
                + "\n\n"
                + "You are the final answer synthesizer. Use the retrieved context and tool observations below to answer the user query."
            )
            user = (
                f"User query: {original}\n\n"
                f"Retrieved context:\n{context_text}\n\n"
                f"Tool observations:\n{_observations_text(observations)}\n\n"
                "Provide a concise, accurate answer. Cite the retrieved document chunks that support each factual claim."
            )
            try:
                llm = build_chat_llm(ctx.org_id, ctx.db, role="chat", temperature=0.7)
                final = ""
                async for chunk in llm.astream([
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]):
                    content = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                    if content:
                        writer({"event": "token", "content": content})
                        final += content
                if not final:
                    final = "I'm sorry, I couldn't generate a response at this time."
            except Exception as exc:
                logger.warning("[finalize_node] generation failed: %s", exc)
                final = "I'm sorry, I couldn't generate a response at this time."
    
        # Build a lightweight LastAnswerObject. Try LLM extraction for data/chart.
        lao = LastAnswerObject(
            summary=final[:500],
            key_points=[s.strip("- ") for s in final.splitlines() if s.strip()][:8],
            data=None,
            citations=[],
            chart_option=None,
            followups=[],
        )
    
        # Use a structured extraction for data if any numeric content; otherwise cheap.
        llm_query = build_chat_llm(ctx.org_id, ctx.db, role="query", temperature=0.0)
        extracted: Optional[LastAnswerObject] = None
        for attempt in range(2):
            try:
                raw = await llm_query.ainvoke([
                    {"role": "user", "content": LAST_ANSWER_EXTRACT_PROMPT.format(answer=final[:3000])},
                ])
                block = _extract_json_block(str(raw.content))
                if block:
                    extracted = LastAnswerObject.model_validate_json(block)
                    break
            except Exception as exc:
                logger.debug("[finalize_node] last_answer_object extraction attempt %d failed: %s", attempt + 1, exc)
        if extracted:
            lao = extracted
    
        # Preserve chart from chart_generate observation if present.
        for obs in observations:
            if obs.tool == "chart_generate" and obs.result.get("chart_option"):
                lao.chart_option = obs.result["chart_option"]
                break
    
        writer({"event": "last_answer", "last_answer_object": lao.model_dump()})
    
        return {
            "final_answer": final,
            "answer": final,
            "last_answer_object": lao,
            "retrieved_docs": docs,
        }
    
    
async def save_memory_node(state: AgentState, ctx: ToolContext) -> dict:
    """Persist final answer, last_answer_object, and tool calls to the DB message row."""
    with _agent_step("save_memory"):
        message_id = state.get("message_id")
        if not message_id:
            return {}
    
        msg = ctx.db.query(Message).filter(Message.id == message_id).first()
        if not msg:
            return {}
    
        msg.content = state.get("final_answer", "")
        plan = state.get("plan")
        if plan:
            msg.plan = plan.model_dump() if isinstance(plan, Plan) else plan
        lao = state.get("last_answer_object")
        if lao:
            msg.last_answer_object = lao.model_dump() if isinstance(lao, LastAnswerObject) else lao
        observations = state.get("observations", [])
        msg.tool_calls = [obs.model_dump() for obs in observations]
        msg.final_confidence = state.get("final_confidence")
        msg.final_confidence_level = state.get("confidence_level")
        msg.faithfulness = state.get("faithfulness")
        msg.completeness = state.get("completeness")
    
        try:
            ctx.db.commit()
        except Exception as exc:
            logger.warning("[save_memory_node] failed to commit message updates: %s", exc)
            ctx.db.rollback()
        return {}
    
    
async def reflect_node(state: AgentState, ctx: ToolContext) -> dict:
    """Periodic reflection: concrete replanning rules + LLM discretion."""
    with _agent_step("reflect"):
        iteration = state.get("iteration", 0)
        if iteration == 0 or iteration % settings.AGENT_REFLECT_EVERY != 0:
            return {}
    
        observations = state.get("observations", [])
        counts = state.get("tool_call_count", {})
        original = state.get("original_query", "")
        rewritten = state.get("rewritten_query", original)
        precomputed: list[dict] = []
    
        # Concrete replanning rules =================================================
        for obs in observations:
            if obs.tool == "rag_retrieve" and len(obs.result.get("docs", [])) == 0:
                if counts.get("rag_retrieve", 0) < settings.AGENT_MAX_RETRIEVALS:
                    precomputed.append({
                        "tool": "rag_retrieve",
                        "arguments": {
                            "query": rewritten or original,
                            "legs": ["dense", "sparse", "exact"],
                            "min_confidence": 0.1,
                        },
                    })
            if obs.tool == "chart_generate" and obs.error:
                if counts.get("extract_data", 0) < settings.AGENT_MAX_RETRIEVALS:
                    precomputed.append({
                        "tool": "extract_data",
                        "arguments": {"source": "retrieved_docs"},
                    })
            if obs.tool == "code_execute" and obs.error:
                if counts.get("code_execute", 0) < settings.AGENT_MAX_CODE_EXEC:
                    precomputed.append({
                        "tool": "extract_data",
                        "arguments": {"source": "retrieved_docs"},
                    })
    
        if precomputed:
            return {
                "reflection": {"action": "retry", "reasoning": "Concrete replanning rule triggered."},
                "precomputed_tool_calls": precomputed,
            }
    
        # LLM discretion ============================================================
        plan = state.get("plan") or Plan()
        system = AGENT_SYSTEM_PROMPT + "\n\n" + REFLECT_SYSTEM_PROMPT
        user = (
            f"Iteration: {iteration}/{settings.AGENT_MAX_ITERATIONS}\n"
            f"Plan: {json.dumps(plan.model_dump() if isinstance(plan, Plan) else plan, default=str)}\n"
            f"Observations:\n{_observations_text(observations)}\n\n"
            "Return a JSON object with: { 'action': 'continue|finalize', 'reasoning': '...' }"
        )
    
        action = "continue"
        reasoning = ""
        try:
            llm = build_chat_llm(ctx.org_id, ctx.db, role="query", temperature=0.0)
            resp = await llm.ainvoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
            raw = str(resp.content)
            block = _extract_json_block(raw)
            if block:
                parsed = json.loads(block)
                action = parsed.get("action", "continue")
                reasoning = parsed.get("reasoning", "")
        except Exception as exc:
            logger.warning("[reflect_node] reflection failed: %s", exc)
    
        writer = _writer()
        writer({"event": "progress", "phase": "reflect", "action": action, "reasoning": reasoning})
        return {"reflection": {"action": action, "reasoning": reasoning}, "force_finalize": action == "finalize"}
    
    
async def clarify_interrupt_node(state: AgentState) -> dict:
    """Pause execution and ask the user for clarification; resumes on response."""
    with _agent_step("clarify_interrupt"):
        plan = state.get("plan") or Plan()
        question = ""
        if isinstance(plan, Plan):
            question = plan.clarification_question or ""
        if not question:
            question = "Could you clarify what you need?"
    
        writer = _writer()
        writer({"event": "interrupt", "question": question})
    
        try:
            user_response = interrupt({"question": question})
        except Exception as exc:
            logger.warning("[clarify_interrupt_node] interrupt not supported or failed: %s", exc)
            user_response = ""
    
        if not user_response:
            user_response = ""
        return {
            "messages": list(state.get("messages", [])) + [HumanMessage(content=str(user_response))],
            "needs_clarification": False,
        }
    
    
async def answer_scoring_node(state: AgentState) -> dict:
    """Evaluate the final answer quality."""
    with _agent_step("answer_scoring"):
        return await answer_evaluation_node(state)
    
    
async def reflect_final_node(state: AgentState, ctx: ToolContext) -> dict:
    """Final pre-finalize reflection: one last satisfaction check."""
    with _agent_step("reflect_final"):
        plan = state.get("plan") or Plan()
        observations = state.get("observations", [])
        original = state.get("original_query", "")
    
        system = AGENT_SYSTEM_PROMPT + "\n\n" + REFLECT_SYSTEM_PROMPT
        user = (
            f"This is the FINAL reflection before answering.\n"
            f"Original query: {original}\n"
            f"Plan: {json.dumps(plan.model_dump() if isinstance(plan, Plan) else plan, default=str)}\n"
            f"Observations:\n{_observations_text(observations)}\n\n"
            "Return JSON: { 'ready': true|false, 'reasoning': '...' }"
        )
    
        ready = True
        reasoning = ""
        try:
            llm = build_chat_llm(ctx.org_id, ctx.db, role="query", temperature=0.0)
            resp = await llm.ainvoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
            raw = str(resp.content)
            block = _extract_json_block(raw)
            if block:
                parsed = json.loads(block)
                ready = parsed.get("ready", True)
                reasoning = parsed.get("reasoning", "")
        except Exception as exc:
            logger.warning("[reflect_final_node] reflection failed: %s", exc)
    
        writer = _writer()
        writer({"event": "progress", "phase": "reflect_final", "ready": ready, "reasoning": reasoning})
        return {"reflection_final": {"ready": ready, "reasoning": reasoning}}
    
    
def build_agent_graph(ctx: ToolContext):
    """Compile and return the agent loop graph."""
    graph = StateGraph(AgentState)

    graph.add_node("load_context", partial(load_context_node, ctx=ctx))
    graph.add_node("rewrite_query", partial(rewrite_query_node, api_base=ctx.org_llm_config.get("api_base")))
    graph.add_node("compaction", compaction_node)
    graph.add_node("plan", partial(plan_node, ctx=ctx))
    graph.add_node("clarify_interrupt", clarify_interrupt_node)
    graph.add_node("think", partial(think_node, ctx=ctx))
    graph.add_node("tool", partial(tool_node, ctx=ctx))
    graph.add_node("reflect", partial(reflect_node, ctx=ctx))
    graph.add_node("reflect_final", partial(reflect_final_node, ctx=ctx))
    graph.add_node("finalize", partial(finalize_node, ctx=ctx))
    graph.add_node("answer_scoring", answer_scoring_node)
    graph.add_node("save_memory", partial(save_memory_node, ctx=ctx))

    graph.set_entry_point("load_context")
    graph.add_edge("load_context", "rewrite_query")
    graph.add_edge("rewrite_query", "compaction")
    graph.add_edge("compaction", "plan")
    graph.add_conditional_edges("plan", route_plan)
    graph.add_edge("clarify_interrupt", "plan")
    graph.add_conditional_edges("think", route_think)
    graph.add_edge("tool", "reflect")
    graph.add_edge("reflect", "think")
    graph.add_edge("reflect_final", "finalize")
    graph.add_edge("finalize", "answer_scoring")
    graph.add_edge("answer_scoring", "save_memory")
    graph.add_edge("save_memory", END)

    checkpointer = getattr(ctx.redis_memory, "checkpointer", None) if ctx.redis_memory else None
    return graph.compile(checkpointer=checkpointer)
