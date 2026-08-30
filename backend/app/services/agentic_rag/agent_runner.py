"""Runner for the enterprise agent loop graph."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, AsyncGenerator, Optional

from langchain_core.messages import HumanMessage

from app.core.config import settings
from app.services.agentic_rag.agent_graph import build_agent_graph
from app.services.agentic_rag.graph_state import AgentState
from app.services.agentic_rag.llm_factory import get_org_llm
from app.services.agentic_rag.prompts import (
    AGENT_SYSTEM_PROMPT,
    FINALIZE_ANSWER_PROMPT,
    FINALIZE_GUARDRAIL_PROMPT,
    PLAN_SYSTEM_PROMPT,
    THINK_SYSTEM_PROMPT,
)
from app.services.agentic_rag.redis_memory import get_redis_memory
from app.services.agentic_rag.token_budget import count_tokens
from app.services.agentic_rag.tool_context import ToolContext

logger = logging.getLogger(__name__)


class _LoopState:
    __slots__ = (
        "full_answer",
        "citations",
        "observations",
        "plan_obj",
        "think_iterations",
        "provider_usage",
        "usage",
    )

    def __init__(self, message_id: Optional[int]) -> None:
        self.full_answer = ""
        self.citations: list = []
        self.observations: list[dict] = []
        self.plan_obj = None
        self.think_iterations = 0
        self.provider_usage: dict | None = None
        self.usage = {"promptTokens": 0, "completionTokens": 0, "messageId": message_id}


def _handle_custom_event(payload: Any, full_answer: str) -> tuple[str, Optional[dict]]:
    if not isinstance(payload, dict):
        return full_answer, None
    if payload.get("event") == "token":
        token = payload.get("content", "")
        if token:
            return full_answer + token, payload
        return full_answer, None
    elif payload.get("event") == "answer_rewrite":
        # Emitted by finalize_node right after generation, before
        # Call 4 (last_answer_object extraction) or Call 5
        # (confidence scoring) — keep full_answer in sync for the
        # token accounting below.
        return payload.get("content", full_answer), payload
    else:
        return full_answer, payload


def _handle_tool_update(state: _LoopState, update: dict) -> Optional[dict]:
    if update.get("observations"):
        for obs in update["observations"]:
            observation_payload = obs.model_dump() if hasattr(obs, "model_dump") else obs
            state.observations.append(observation_payload)
    return None


def _handle_plan_update(state: _LoopState, update: dict) -> Optional[dict]:
    state.plan_obj = update.get("plan")
    return None


def _handle_think_update(state: _LoopState, update: dict) -> Optional[dict]:
    state.think_iterations = max(state.think_iterations, update.get("iteration", 0))
    return None


def _handle_finalize_update(state: _LoopState, update: dict) -> Optional[dict]:
    # Citation-normalized content already arrived via the
    # earlier "answer_rewrite" custom event (emitted by
    # finalize_node itself, before Call 4/5) — this update
    # only carries retrieved_docs for token accounting.
    final = update.get("final_answer", "")
    if final:
        state.full_answer = final
    state.citations = update.get("retrieved_docs", [])
    if isinstance(update.get("answer_usage"), dict):
        state.provider_usage = update["answer_usage"]
    return {"event": "progress", "phase": "finalize", "message": "Finalising answer"}


def _handle_scoring_update(state: _LoopState, update: dict) -> Optional[dict]:
    state.usage["final_confidence"] = update.get("final_confidence")
    state.usage["confidence_level"] = update.get("confidence_level")
    state.usage["faithfulness"] = update.get("faithfulness")
    state.usage["completeness"] = update.get("completeness")
    state.usage["retrieval_score"] = update.get("retrieval_score")
    return None


NODE_HANDLERS = {
    "tool": _handle_tool_update,
    "plan": _handle_plan_update,
    "think": _handle_think_update,
    "finalize": _handle_finalize_update,
    "answer_scoring": _handle_scoring_update,
}


def _handle_node_update(node: str, update: dict, state: _LoopState) -> Optional[dict]:
    handler = NODE_HANDLERS.get(node)
    if handler is None:
        return None
    return handler(state, update)


def _estimate_token_usage(
    state: _LoopState,
    query: str,
    file_markdown: Optional[str],
) -> list[dict]:
    events: list[dict] = []
    if not state.full_answer:
        state.full_answer = "I'm sorry, I could not produce an answer."
        events.append({"event": "answer_rewrite", "content": state.full_answer, "citations": []})

    # ── Token accounting ────────────────────────────────────────────────
    # Prefer provider-reported usage from the generation call. When the
    # backend doesn't report usage, fall back to a reconstruction and mark it
    # as an estimate — it counts system prompts, the query, all retrieved
    # docs and observations once each, which is a lower bound, not a measured
    # figure.
    if state.provider_usage:
        state.usage["promptTokens"] = state.provider_usage.get("input_tokens", 0)
        state.usage["completionTokens"] = state.provider_usage.get("output_tokens", 0)
        state.usage["estimated"] = False
        events.append({"event": "done", "usage": state.usage})
        return events

    # System prompt overhead (constant per call type)
    plan_sys_tokens = count_tokens(AGENT_SYSTEM_PROMPT) + count_tokens(PLAN_SYSTEM_PROMPT)
    think_sys_tokens = count_tokens(AGENT_SYSTEM_PROMPT) + count_tokens(THINK_SYSTEM_PROMPT)
    finalize_sys_tokens = count_tokens(FINALIZE_GUARDRAIL_PROMPT) + count_tokens(FINALIZE_ANSWER_PROMPT)

    prompt_tokens = plan_sys_tokens  # 1 plan call
    prompt_tokens += think_sys_tokens * max(state.think_iterations, 1)  # think calls
    prompt_tokens += finalize_sys_tokens  # 1 finalize call

    # User query
    prompt_tokens += count_tokens(query)

    # All retrieved docs (not just cited subset)
    prompt_tokens += sum(
        count_tokens(d.get("page_content", "")) for d in state.citations
    )

    # Tool observations
    prompt_tokens += sum(
        count_tokens(json.dumps(o.model_dump() if hasattr(o, "model_dump") else o, default=str))
        for o in state.observations
    )

    # Plan
    if state.plan_obj:
        prompt_tokens += count_tokens(
            json.dumps(state.plan_obj.model_dump() if hasattr(state.plan_obj, "model_dump") else state.plan_obj, default=str)
        )

    # File markdown
    if file_markdown:
        prompt_tokens += count_tokens(file_markdown)
    completion_tokens = count_tokens(state.full_answer)
    state.usage["promptTokens"] = prompt_tokens
    state.usage["completionTokens"] = completion_tokens
    state.usage["estimated"] = True
    events.append({
        "event": "done",
        "usage": state.usage,
    })
    return events


def _extract_interrupt_event(payload: dict, thread_id: str) -> dict:
    interrupts = payload["__interrupt__"]
    value = interrupts[0].value if interrupts else None
    question = value.get("question", "") if isinstance(value, dict) else str(value or "")
    return {"event": "interrupt", "question": question, "thread_id": thread_id}


def _process_node_updates(payload: dict, state: _LoopState) -> list[dict]:
    events: list[dict] = []
    for node, update in payload.items():
        if not isinstance(update, dict):
            continue
        event = _handle_node_update(node, update, state)
        if event is not None:
            events.append(event)
    return events


async def run_agent_loop(
    query: str,
    kb_ids: list[int],
    db: Any,
    file_markdown: Optional[str] = None,
    org_id: Optional[int] = None,
    chat_id: Optional[int] = None,
    user_id: Optional[int] = None,
    message_id: Optional[int] = None,
    display_query: Optional[str] = None,
) -> AsyncGenerator[dict, None]:
    """Run the new enterprise agent loop and stream SSE-style events.

    Model, temperature and endpoint are resolved per organization by
    ``build_chat_llm`` / ``get_org_llm``; there are deliberately no
    per-call overrides here.
    """
    memory = await get_redis_memory()
    thread_id = f"chat-{chat_id}" if chat_id else f"anon-{uuid.uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id}}

    org_cfg = get_org_llm(org_id, db, role="chat")
    ctx = ToolContext(
        db=db,
        user_id=user_id,
        org_id=org_id,
        chat_id=chat_id,
        message_id=message_id,
        qdrant_client=None,
        redis_memory=memory,
        org_llm_config=org_cfg,
        state=None,
    )

    graph = build_agent_graph(ctx)

    initial_state = AgentState(
        messages=[HumanMessage(content=query)],
        original_query=display_query or query,
        kb_ids=kb_ids,
        org_id=org_id,
        user_id=user_id,
        chat_id=chat_id,
        message_id=message_id,
        file_markdown=file_markdown,
    )

    state = _LoopState(message_id)

    async for chunk in graph.astream(initial_state, config, stream_mode=["updates", "custom"]):
        kind, payload = chunk if isinstance(chunk, tuple) else ("updates", chunk)

        if kind == "custom":
            state.full_answer, event = _handle_custom_event(payload, state.full_answer)
            if event is not None:
                yield event
            continue

        if kind != "updates" or not isinstance(payload, dict):
            continue

        if "__interrupt__" in payload:
            yield _extract_interrupt_event(payload, thread_id)
            return

        for event in _process_node_updates(payload, state):
            yield event

    for event in _estimate_token_usage(state, query, file_markdown):
        yield event
