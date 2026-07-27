"""Agent loop graph for the enterprise agent.

Replaces the rigid RAG pipeline with a tool-calling loop:
  load_context → rewrite_query → compaction → plan → think → [tool → think ...] → finalize → save_memory
"""

from __future__ import annotations

import json
import logging
import re
from functools import partial
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph

from app.core.config import settings
from app.models.chat import ChatFile, Message
from app.services.agentic_rag.callbacks import get_stream_writer


def _writer():
    """Return a stream writer if one is available, else a no-op."""
    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda x: None
from app.services.agentic_rag.llm_factory import build_chat_llm
from app.services.agentic_rag.nodes import compaction_node, rewrite_query_node
from app.services.agentic_rag.prompts import (
    AGENT_SYSTEM_PROMPT,
    LAST_ANSWER_EXTRACT_PROMPT,
    PLAN_SYSTEM_PROMPT,
    REFLECT_SYSTEM_PROMPT,
    THINK_SYSTEM_PROMPT,
)
from app.services.agentic_rag.schemas import LastAnswerObject, Observation, Plan, Subtask
from app.services.agentic_rag.tool_context import ToolContext, write_audit
from app.services.agentic_rag.tools import applicable_tools
from app.services.agentic_rag.token_budget import count_tokens

from .graph_state import AgentState

logger = logging.getLogger(__name__)


def _tool_descriptions_text(tools: list) -> str:
    lines = []
    for t in tools:
        lines.append(f"- {t.name}: {t.description}")
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
    """Load previous-answer object and file metadata into state."""
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

    return {
        "last_answer_object": last_obj,
        "org_id": ctx.org_id,
        "user_id": ctx.user_id,
        "chat_id": ctx.chat_id,
        "message_id": ctx.message_id,
    }


async def plan_node(state: AgentState, ctx: ToolContext) -> dict:
    """Produce a structured plan for the current turn."""
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

    system = AGENT_SYSTEM_PROMPT + "\n\n" + PLAN_SYSTEM_PROMPT
    user = (
        f"Original query: {original}\n"
        f"Rewritten query: {rewritten}\n"
        f"Previous answer summary: {last_summary}\n"
        f"Attached files: {json.dumps(file_meta)}\n\n"
        "Produce a plan JSON matching the schema."
    )

    try:
        llm = build_chat_llm(ctx.org_id, ctx.db, role="query", temperature=0.0)
        structured = llm.with_structured_output(Plan, method="json_mode", include_raw=True)
        resp = await structured.ainvoke([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
        plan = resp.parsed if hasattr(resp, "parsed") else resp
    except Exception as exc:
        logger.warning("[plan_node] structured output failed: %s; using JSON parse fallback", exc)
        llm = build_chat_llm(ctx.org_id, ctx.db, role="query", temperature=0.0)
        resp = await llm.ainvoke([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
        raw = str(resp.content)
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            plan = Plan.model_validate_json(match.group(0)) if match else Plan()
        except Exception as parse_exc:
            logger.warning("[plan_node] JSON parse failed: %s", parse_exc)
            plan = Plan(intent="rag", subtasks=[Subtask(id="a", description=original, tool_hint="rag_retrieve")])

    writer({"event": "plan", "plan": plan.model_dump() if isinstance(plan, Plan) else plan})

    if plan.needs_clarification and plan.clarification_question:
        # Clarification is handled as an interrupt by the runner; emit interrupt event for it.
        writer({"event": "interrupt", "question": plan.clarification_question})

    return {"plan": plan, "needs_clarification": plan.needs_clarification}


async def think_node(state: AgentState, ctx: ToolContext) -> dict:
    """Decide the next action: emit one or more tool calls or a final answer."""
    iteration = state.get("iteration", 0) + 1
    max_iter = settings.AGENT_MAX_ITERATIONS

    original = state.get("original_query", "")
    plan = state.get("plan") or Plan()
    observations = state.get("observations", [])
    tools = applicable_tools(ctx)
    tools_text = _tool_descriptions_text(tools)

    system = AGENT_SYSTEM_PROMPT + "\n\n" + THINK_SYSTEM_PROMPT
    user = (
        f"Iteration: {iteration}/{max_iter}\n"
        f"User query: {original}\n"
        f"Plan: {json.dumps(plan.model_dump() if isinstance(plan, Plan) else plan, default=str)}\n"
        f"Observations so far:\n{_observations_text(observations)}\n\n"
        f"Available tools:\n{tools_text}\n\n"
        "Emit either {\"tool_calls\": [...]} or {\"final_answer\": \"...\"}."
    )

    tool_calls: list[dict] = []
    final_answer: Optional[str] = None

    mode = settings.TOOL_CALL_MODE
    try_native = mode in ("native", "auto")
    if try_native:
        try:
            llm = build_chat_llm(ctx.org_id, ctx.db, role="chat", temperature=0.7)
            llm_bound = llm.bind_tools(tools)
            resp = await llm_bound.ainvoke([
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ])
            if getattr(resp, "tool_calls", None):
                for tc in resp.tool_calls:
                    tool_calls.append({"tool": tc.get("name"), "arguments": tc.get("args", {})})
            else:
                final_answer = str(resp.content)
        except Exception as exc:
            logger.warning("[think_node] native tool-calling failed: %s", exc)
            if mode == "native":
                final_answer = f"Tool-calling error: {exc}"

    if not tool_calls and not final_answer:
        # JSON-text fallback
        llm = build_chat_llm(ctx.org_id, ctx.db, role="chat", temperature=0.7)
        resp = await llm.ainvoke([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
        raw = str(resp.content)
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            parsed = json.loads(match.group(0)) if match else {}
            if "tool_calls" in parsed:
                tool_calls = parsed["tool_calls"]
            elif "tool" in parsed:
                tool_calls = [parsed]
            elif "final_answer" in parsed:
                final_answer = parsed["final_answer"]
            else:
                final_answer = raw
        except Exception as exc:
            logger.warning("[think_node] JSON fallback parse failed: %s", exc)
            final_answer = raw

    if iteration >= max_iter:
        tool_calls = []

    # Dependency guard: only allow independent tool calls in one message.
    allowed = []
    completed_ids = {obs.observation_id for obs in observations}
    for tc in tool_calls:
        # No depends tracking on actual calls for now; run all emitted calls.
        allowed.append(tc)

    if tool_calls and not final_answer:
        return {"iteration": iteration, "tool_calls": allowed}
    return {"iteration": iteration, "tool_calls": [], "precomputed_answer": final_answer or ""}


def route_think(state: AgentState) -> str:
    iteration = state.get("iteration", 0)
    if iteration >= settings.AGENT_MAX_ITERATIONS:
        return "finalize"
    if state.get("tool_calls"):
        return "tool"
    return "finalize"


async def tool_node(state: AgentState, ctx: ToolContext) -> dict:
    """Dispatch tool calls, run them (in parallel when independent), record observations."""
    writer = _writer()
    tool_calls = state.get("tool_calls", [])
    if not tool_calls:
        return {}

    tools = {t.name: t for t in applicable_tools(ctx)}
    observations = list(state.get("observations", []))
    counts = dict(state.get("tool_call_count", {}))

    coros = []
    for tc in tool_calls:
        name = tc.get("tool")
        args = tc.get("arguments", {})
        writer({"event": "tool_call", "tool": name, "arguments": args})
        tool = tools.get(name)
        if tool is None:
            async def _missing(name=name):
                return {"tool": name, "arguments": {}, "result": {}, "error": f"Tool {name} not available", "tokens": 0}
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

    return {"tool_calls": [], "observations": observations, "tool_call_count": counts}


import asyncio


async def _run_tool(tool, name: str, args: dict) -> dict:
    try:
        result = await tool.arun(args)
        return {"tool": name, "arguments": args, "result": result, "error": None, "tokens": 0}
    except Exception as exc:
        return {"tool": name, "arguments": args, "result": {}, "error": str(exc), "tokens": 0}


def route_plan(state: AgentState) -> str:
    if state.get("needs_clarification"):
        return END
    return "think"


async def finalize_node(state: AgentState, ctx: ToolContext) -> dict:
    """Generate final answer if not precomputed; extract LastAnswerObject."""
    writer = _writer()
    precomputed = state.get("precomputed_answer", "")
    original = state.get("original_query", "")
    observations = state.get("observations", [])
    docs = state.get("retrieved_docs", [])

    if precomputed:
        final = precomputed
    else:
        system = (
            AGENT_SYSTEM_PROMPT
            + "\n\n"
            + "You are the final answer synthesizer. Use the observations below to answer the user query."
        )
        user = (
            f"User query: {original}\n"
            f"Observations:\n{_observations_text(observations)}\n\n"
            "Provide a concise, accurate answer. Cite documents with [1], [2] etc if applicable."
        )
        try:
            llm = build_chat_llm(ctx.org_id, ctx.db, role="chat", temperature=0.7)
            resp = await llm.ainvoke([
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ])
            final = str(resp.content)
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
    try:
        llm_query = build_chat_llm(ctx.org_id, ctx.db, role="query", temperature=0.0)
        raw = await llm_query.ainvoke([
            {"role": "user", "content": LAST_ANSWER_EXTRACT_PROMPT.format(answer=final[:3000])},
        ])
        match = re.search(r"\{.*\}", str(raw.content), re.DOTALL)
        if match:
            extracted = LastAnswerObject.model_validate_json(match.group(0))
            lao = extracted
    except Exception as exc:
        logger.debug("[finalize_node] last_answer_object extraction skipped: %s", exc)

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
    message_id = state.get("message_id")
    if not message_id:
        return {}

    msg = ctx.db.query(Message).filter(Message.id == message_id).first()
    if not msg:
        return {}

    msg.content = state.get("final_answer", "")
    lao = state.get("last_answer_object")
    if lao:
        msg.last_answer_object = lao.model_dump() if isinstance(lao, LastAnswerObject) else lao
    observations = state.get("observations", [])
    msg.tool_calls = [obs.model_dump() for obs in observations]

    ctx.db.commit()
    return {}


def build_agent_graph(ctx: ToolContext):
    """Compile and return the agent loop graph."""
    graph = StateGraph(AgentState)

    graph.add_node("load_context", partial(load_context_node, ctx=ctx))
    graph.add_node("rewrite_query", partial(rewrite_query_node, api_base=ctx.org_llm_config.get("api_base")))
    graph.add_node("compaction", compaction_node)
    graph.add_node("plan", partial(plan_node, ctx=ctx))
    graph.add_node("think", partial(think_node, ctx=ctx))
    graph.add_node("tool", partial(tool_node, ctx=ctx))
    graph.add_node("finalize", partial(finalize_node, ctx=ctx))
    graph.add_node("save_memory", partial(save_memory_node, ctx=ctx))

    graph.set_entry_point("load_context")
    graph.add_edge("load_context", "rewrite_query")
    graph.add_edge("rewrite_query", "compaction")
    graph.add_edge("compaction", "plan")
    graph.add_conditional_edges("plan", route_plan)
    graph.add_conditional_edges("think", route_think)
    graph.add_edge("tool", "think")
    graph.add_edge("finalize", "save_memory")
    graph.add_edge("save_memory", END)

    checkpointer = getattr(ctx.redis_memory, "checkpointer", None) if ctx.redis_memory else None
    return graph.compile(checkpointer=checkpointer, interrupt_after=["plan"] if False else None)
