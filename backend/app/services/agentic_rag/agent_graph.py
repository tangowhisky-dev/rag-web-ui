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


def _observations_text(observations: list[Observation], full: bool = False) -> str:
    """Format observations for LLM context.

    When full=True, include doc previews (first 800 chars of each doc's
    page_content) so think_node can judge whether the retrieval actually
    answers the query. When full=False, include a compact summary (doc
    count, confidence, top doc preview) to keep reflect/finalize prompts
    small.
    """
    parts = []
    for i, obs in enumerate(observations, 1):
        parts.append(f"Observation {i}: tool={obs.tool} args={obs.arguments}")
        if obs.error:
            parts.append(f"  error: {obs.error}")
            continue
        result = obs.result if isinstance(obs.result, dict) else {}
        docs = result.get("docs", [])
        doc_count = len(docs)
        confidence = result.get("confidence", "N/A")
        if full:
            parts.append(f"  doc_count={doc_count} confidence={confidence}")
            for j, doc in enumerate(docs[:10], 1):
                content = str(doc.get("page_content", ""))[:800] if isinstance(doc, dict) else str(doc)[:800]
                parts.append(f"  doc_{j}: {content}")
        else:
            parts.append(f"  doc_count={doc_count} confidence={confidence}")
            if docs and isinstance(docs[0], dict):
                preview = str(docs[0].get("page_content", ""))[:300]
                parts.append(f"  top_doc_preview: {preview}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Runtime compaction — referral-style context budget check
# ---------------------------------------------------------------------------

# How many docs to keep per rag_retrieve observation when compacting.
_COMPACT_KEEP_DOCS = 5
# Doc preview length when compacting (shorter than the 800 used in full mode).
_COMPACT_DOC_PREVIEW_CHARS = 200
# How many stdout lines to keep for code_execute when compacting.
_COMPACT_KEEP_STDOUT_LINES = 20


def _compact_observations(observations: list[Observation]) -> list[Observation]:
    """Stage 1 (deterministic): shrink tool observations in-place.

    Per the design doc (05-context-memory.md §4.4):
    - rag_retrieve: keep only top 5 chunks by score (already sorted by reranker).
    - code_execute: keep result + last 20 lines of stdout.
    - file_read: keep only a summary line.

    Returns a new list; original observations are not mutated.
    """
    compacted = []
    for obs in observations:
        if obs.error:
            compacted.append(obs)
            continue
        if obs.tool == "rag_retrieve":
            docs = obs.result.get("docs", [])
            if len(docs) > _COMPACT_KEEP_DOCS:
                new_result = dict(obs.result)
                new_result["docs"] = docs[:_COMPACT_KEEP_DOCS]
                compacted.append(Observation(
                    tool=obs.tool, observation_id=obs.observation_id,
                    arguments=obs.arguments, result=new_result,
                    error=obs.error, tokens=obs.tokens,
                ))
            else:
                compacted.append(obs)
        elif obs.tool == "code_execute":
            stdout = obs.result.get("stdout", "")
            if stdout and stdout.count("\n") > _COMPACT_KEEP_STDOUT_LINES:
                lines = stdout.split("\n")
                trimmed = "\n".join(lines[-_COMPACT_KEEP_STDOUT_LINES:])
                new_result = dict(obs.result)
                new_result["stdout"] = f"[...trimmed {len(lines) - _COMPACT_KEEP_STDOUT_LINES} lines...]\n{trimmed}"
                compacted.append(Observation(
                    tool=obs.tool, observation_id=obs.observation_id,
                    arguments=obs.arguments, result=new_result,
                    error=obs.error, tokens=obs.tokens,
                ))
            else:
                compacted.append(obs)
        else:
            compacted.append(obs)
    return compacted


async def _compact_messages_llm(messages: list, api_base: str | None = None) -> tuple[list, str | None]:
    """Stage 2 (LLM call): summarize oldest messages into a structured summary.

    Keeps the most recent COMPACTION_KEEP_RECENT messages verbatim.
    Returns (compacted_messages, summary_text).
    """
    from app.services.agentic_rag.nodes import _messages_to_conversation_text
    from app.services.agentic_rag.prompts import COMPACTION_SYSTEM_PROMPT, COMPACTION_USER_PROMPT
    from app.services.agentic_rag.nodes import _get_llm

    keep_recent = settings.COMPACTION_KEEP_RECENT
    recent = messages[-keep_recent:]
    old = messages[:len(messages) - keep_recent]
    if not old:
        return messages, None

    conversation_text = _messages_to_conversation_text(old)
    try:
        llm = _get_llm(
            model_name=settings.effective_query_model,
            temperature=0.0,
            streaming=False,
            api_base=api_base,
        )
        response = llm.invoke([
            {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
            {"role": "user", "content": COMPACTION_USER_PROMPT.format(conversation=conversation_text)},
        ])
        summary = str(response.content).strip()
        if len(summary) > settings.COMPACTION_SUMMARY_MAX_CHARS:
            summary = summary[:settings.COMPACTION_SUMMARY_MAX_CHARS] + "\n\n[...summary truncated]"
        # Prepend the summary as a system-like HumanMessage so the LLM sees it.
        compacted = [HumanMessage(content=f"[Conversation summary]\n{summary}")] + recent
        return compacted, summary
    except Exception as exc:
        logger.warning("[_compact_messages_llm] failed: %s — keeping full history", exc)
        return messages, None


async def _compact_if_needed(
    state: AgentState,
    prompt_text: str,
    system_overhead: int = 0,
    api_base: str | None = None,
) -> dict:
    """Runtime compaction referral. Called before any LLM call with variable-length context.

    Checks if system_overhead + prompt_text exceeds the context budget.
    If so, applies two stages:
      1. Deterministic: compact tool observations (keep top 5 docs, trim stdout).
      2. LLM call: summarize oldest messages (only if stage 1 wasn't enough).

    Returns a state update dict (observations and/or messages) that the caller
    should merge into its local state before building the LLM prompt.
    Empty dict if no compaction was needed.
    """
    if not settings.COMPACTION_ENABLED:
        return {}

    from app.services.agentic_rag.token_budget import ContextBudget

    budget = ContextBudget()
    budget.add(count_tokens(prompt_text))
    budget.add(system_overhead)

    if not budget.needs_compaction():
        return {}

    logger.info(
        "[_compact_if_needed] over budget | used=%d threshold=%d — compacting",
        budget.used, budget.compaction_threshold,
    )

    updates: dict[str, Any] = {}

    # Stage 1: Compact observations (deterministic, no LLM call).
    observations = list(state.get("observations", []))
    compacted_obs = _compact_observations(observations)

    # Re-check budget with compacted observations.
    # We need to recompute the prompt text with the compacted observations
    # to see if stage 1 was sufficient. The caller will rebuild the prompt
    # from the updated state, so we estimate the savings.
    obs_tokens_before = sum(count_tokens(json.dumps(o.result, default=str)) for o in observations)
    obs_tokens_after = sum(count_tokens(json.dumps(o.result, default=str)) for o in compacted_obs)
    savings = obs_tokens_before - obs_tokens_after

    if savings > 0:
        updates["observations"] = compacted_obs
        budget.used -= savings
        logger.info("[_compact_if_needed] stage 1 (observations) saved %d tokens", savings)

    if not budget.needs_compaction():
        return updates

    # Stage 2: Compact messages (LLM call).
    messages = list(state.get("messages", []))
    if len(messages) > settings.COMPACTION_KEEP_RECENT:
        compacted_msgs, summary = await _compact_messages_llm(messages, api_base=api_base)
        if summary is not None:
            updates["messages"] = compacted_msgs
            updates["compaction_summary"] = summary
            logger.info("[_compact_if_needed] stage 2 (messages) summarized %d old messages",
                        len(messages) - settings.COMPACTION_KEEP_RECENT)

    return updates


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

        # If reflect_final sent us back, include its reasoning so the agent
        # knows exactly what was missing and can act on it.
        reflection = state.get("reflection_final")
        reflection_text = ""
        if reflection and isinstance(reflection, dict) and not reflection.get("ready", True):
            reflection_text = (
                f"  NOTE — the verification module rejected your previous final_answer because:\n"
                f"  {reflection.get('reasoning', '')}\n"
                "  Do NOT reference this feedback in your answer. Use it only as guidance to\n"
                "  decide which tool to call next, then emit a clean final_answer.\n"
            )

        system = AGENT_SYSTEM_PROMPT + "\n\n" + THINK_SYSTEM_PROMPT
        user = (
            f"Iteration: {iteration}/{max_iter}\n"
            f"User query: {original}\n"
            f"Conversation history (recent turns):\n{history_text or '  (none)'}\n"
            f"Previous answer context:\n{lao_text or '  (none)'}\n"
            f"Verification feedback:\n{reflection_text or '  (none)'}\n"
            f"Plan: {json.dumps(plan.model_dump() if isinstance(plan, Plan) else plan, default=str)}\n"
            f"Observations so far:\n{_observations_text(observations, full=True)}\n\n"
            f"Available tools:\n{tools_text}\n\n"
            "Emit either {\"tool_calls\": [...]} or {\"final_answer\": true}."
        )

        # Runtime compaction: check if the prompt exceeds the context budget.
        # If so, compact observations (deterministic) and/or messages (LLM call),
        # then rebuild the prompt from the compacted state.
        compaction_updates = await _compact_if_needed(
            state, user, system_overhead=count_tokens(system), api_base=ctx.org_llm_config.get("api_base"),
        )
        if compaction_updates:
            state = {**state, **compaction_updates}
            observations = state.get("observations", [])
            recent = select_recent_history(state.get("messages", []), max_pairs=3)
            history_text = ""
            for msg in recent:
                role = "User" if msg.type == "human" else "Assistant"
                content = str(msg.content)[:500]
                history_text += f"  {role}: {content}\n"
            user = (
                f"Iteration: {iteration}/{max_iter}\n"
                f"User query: {original}\n"
                f"Conversation history (recent turns):\n{history_text or '  (none)'}\n"
                f"Previous answer context:\n{lao_text or '  (none)'}\n"
                f"Verification feedback:\n{reflection_text or '  (none)'}\n"
                f"Plan: {json.dumps(plan.model_dump() if isinstance(plan, Plan) else plan, default=str)}\n"
                f"Observations so far:\n{_observations_text(observations, full=True)}\n\n"
                f"Available tools:\n{tools_text}\n\n"
                "Emit either {\"tool_calls\": [...]} or {\"final_answer\": true}."
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
            return {**compaction_updates, "iteration": iteration, "tool_calls": allowed}

        # final_answer can be:
        #   - True (boolean signal from {"final_answer": true}) — think is done,
        #     finalize_node will generate the answer with streaming.
        #   - str (Tier 3 fallback — LLM wrote plain text instead of JSON) — pass
        #     it through as precomputed since the text was already generated.
        if isinstance(final_answer, bool) and final_answer:
            return {**compaction_updates, "iteration": iteration, "tool_calls": [], "precomputed_answer": ""}
        return {**compaction_updates, "iteration": iteration, "tool_calls": [], "precomputed_answer": final_answer or ""}
    
    
def route_think(state: AgentState) -> str:
    iteration = state.get("iteration", 0)
    if iteration >= settings.AGENT_MAX_ITERATIONS:
        return "reflect_final"
    if state.get("tool_calls"):
        return "tool"
    return "reflect_final"


def route_reflect_final(state: AgentState) -> str:
    """Route after final verification: ready → finalize, not ready → think."""
    reflection = state.get("reflection_final", {})
    ready = reflection.get("ready", True) if isinstance(reflection, dict) else True
    iteration = state.get("iteration", 0)
    if not ready and iteration < settings.AGENT_MAX_ITERATIONS:
        return "think"
    return "finalize"


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
        compaction_updates: dict = {}

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

            # Runtime compaction before the generation LLM call.
            compaction_updates = await _compact_if_needed(
                state, user, system_overhead=count_tokens(system),
                api_base=ctx.org_llm_config.get("api_base"),
            )
            if compaction_updates:
                state = {**state, **compaction_updates}
                observations = state.get("observations", [])
                docs = state.get("retrieved_docs", docs)
                context_text = format_context_string(docs, state.get("file_markdown"))
                user = (
                    f"User query: {original}\n\n"
                    f"Retrieved context:\n{context_text}\n\n"
                    f"Tool observations:\n{_observations_text(observations)}\n\n"
                    "Provide a concise, accurate answer. Cite the retrieved document chunks that support each factual claim."
                )
            try:
                llm = build_chat_llm(ctx.org_id, ctx.db, role="chat", temperature=0.7, streaming=True)
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
            **compaction_updates,
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

        # Runtime compaction before the reflect LLM call.
        compaction_updates = await _compact_if_needed(
            state, user, system_overhead=count_tokens(system),
            api_base=ctx.org_llm_config.get("api_base"),
        )
        if compaction_updates:
            state = {**state, **compaction_updates}
            observations = state.get("observations", [])
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
        return {**compaction_updates, "reflection": {"action": action, "reasoning": reasoning}, "force_finalize": action == "finalize"}
    
    
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
    
    
def _build_execution_summary(state: AgentState) -> dict:
    """Build a structured execution summary for deterministic verification."""
    plan = state.get("plan") or Plan()
    observations = state.get("observations", [])
    counts = dict(state.get("tool_call_count", {}))
    iteration = state.get("iteration", 0)

    # Map subtask tool_hints to whether we have a matching observation.
    subtask_status = []
    for st in plan.subtasks:
        hint = st.tool_hint
        if hint == "any":
            has_obs = len(observations) > 0
        else:
            has_obs = any(o.tool == hint and not o.error for o in observations)
        subtask_status.append({
            "id": st.id,
            "description": st.description,
            "tool_hint": hint,
            "completed": has_obs,
        })

    # Retrieval stats.
    retrieval_queries = counts.get("rag_retrieve", 0)
    total_docs = 0
    for o in observations:
        if o.tool == "rag_retrieve" and not o.error:
            total_docs += len(o.result.get("docs", []))

    # Tool failures.
    failures = []
    for o in observations:
        if o.error:
            failures.append({"tool": o.tool, "error": o.error})

    # Remaining retrieval budget.
    retrieval_budget_left = settings.AGENT_MAX_RETRIEVALS - retrieval_queries

    return {
        "user_goal": state.get("original_query", ""),
        "intent": plan.intent,
        "subtasks": subtask_status,
        "retrieval": {
            "queries": retrieval_queries,
            "documents": total_docs,
        },
        "tool_failures": failures,
        "remaining_budget": {
            "retrieval": retrieval_budget_left,
            "iterations": settings.AGENT_MAX_ITERATIONS - iteration,
        },
    }


def _verify_execution(summary: dict) -> tuple[bool, str]:
    """Deterministic execution verification.

    Returns (ready, reasoning). ready=True means the agent has done enough
    work to generate an answer. ready=False means a required step is missing
    and another iteration is likely to help.
    """
    issues = []

    # 1. No observations at all — nothing was attempted.
    if not summary["subtasks"] and summary["retrieval"]["queries"] == 0:
        # No plan subtasks and no retrieval — could be a conversation query.
        # If intent is conversation, that's fine.
        if summary.get("intent") not in ("conversation",):
            issues.append("No tool calls were made and no subtasks were planned.")
        else:
            return True, "Conversation intent — no tools needed."

    # 2. Uncompleted subtasks that still have budget.
    for st in summary["subtasks"]:
        if not st["completed"]:
            issues.append(f"Subtask '{st['id']}' ({st['description'][:60]}) has no successful tool result.")

    # 3. Retrieval returned zero docs and budget remains.
    if summary["retrieval"]["queries"] > 0 and summary["retrieval"]["documents"] == 0:
        if summary["remaining_budget"]["retrieval"] > 0:
            issues.append("Retrieval returned 0 documents; another query may help.")
        else:
            # No budget left — can't fix this, proceed with what we have.
            pass

    # 4. Tool failures that could be retried.
    for f in summary["tool_failures"]:
        tool = f["tool"]
        if "Budget exceeded" in f["error"]:
            continue  # Budget-exceeded failures can't be retried.
        issues.append(f"Tool '{tool}' failed: {f['error'][:80]}")

    if issues:
        return False, "; ".join(issues)
    return True, "All planned steps have supporting tool results."


async def reflect_final_node(state: AgentState, ctx: ToolContext) -> dict:
    """Final pre-finalize verification: deterministic execution completeness check."""
    with _agent_step("reflect_final"):
        iteration = state.get("iteration", 0)
        max_iter = settings.AGENT_MAX_ITERATIONS

        summary = _build_execution_summary(state)
        ready, reasoning = _verify_execution(summary)

        # Force ready when iteration cap is reached — no more retries possible.
        if not ready and iteration >= max_iter:
            logger.info("[reflect_final_node] not ready but iteration cap reached (%d/%d) — forcing finalize", iteration, max_iter)
            ready = True
            reasoning = f"Forced finalize at iteration cap. Pending issues: {reasoning}"

        logger.info("[reflect_final_node] ready=%s reasoning=%s", ready, reasoning[:200])

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
    graph.add_conditional_edges("reflect_final", route_reflect_final)
    graph.add_edge("finalize", "answer_scoring")
    graph.add_edge("answer_scoring", "save_memory")
    graph.add_edge("save_memory", END)

    checkpointer = getattr(ctx.redis_memory, "checkpointer", None) if ctx.redis_memory else None
    return graph.compile(checkpointer=checkpointer)
