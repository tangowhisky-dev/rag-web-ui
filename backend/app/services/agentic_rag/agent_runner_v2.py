"""Runner for the agentic-v2 pipeline.

Uses the v2 graph (unified think ⇄ tool loop, no planner/sufficiency/finalizer nodes).
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncGenerator, Optional

from langchain_core.messages import HumanMessage

from app.services.agentic_rag.agent_graph_v2 import build_agent_graph_v2
from app.services.agentic_rag.graph_state import AgentState
from app.services.agentic_rag.llm_factory import get_org_llm
from app.services.agentic_rag.prompts_v2 import AGENT_V2_PROMPT
from app.services.agentic_rag.redis_memory import get_redis_memory
from app.services.agentic_rag.token_budget import count_tokens
from app.services.agentic_rag.tool_context import ToolContext
from app.services.infrastructure import is_cancelled

logger = logging.getLogger(__name__)

# Keep a hard reference to background extraction tasks so they aren't
# garbage collected before they run.
_BACKGROUND_TASKS: set[asyncio.Task] = set()


class _V2LoopState:
    __slots__ = (
        "full_answer",
        "citations",
        "observations",
        "think_iterations",
        "provider_usage",
        "usage",
    )

    def __init__(self, message_id: Optional[int]) -> None:
        self.full_answer = ""
        self.citations: list = []
        self.observations: list[dict] = []
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
        return payload.get("content", full_answer), payload
    else:
        return full_answer, payload


def _handle_node_update(node: str, update: dict, state: _V2LoopState) -> Optional[dict]:
    if node == "tool":
        if update.get("observations"):
            for obs in update["observations"]:
                obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs
                state.observations.append(obs_dict)
        return None
    if node == "think":
        state.think_iterations = max(state.think_iterations, update.get("iteration", 0))
        return None
    if node == "post_process":
        final = update.get("final_answer", "")
        if final:
            state.full_answer = final
        state.citations = update.get("cited_docs", [])
        if isinstance(update.get("answer_usage"), dict):
            state.provider_usage = update["answer_usage"]
        # Scoring is done inside post_process; extract confidence/followups
        # for the done event and last_answer event.
        if "final_confidence" in update:
            state.usage["final_confidence"] = update.get("final_confidence")
            state.usage["confidence_level"] = update.get("confidence_level")
            state.usage["faithfulness"] = update.get("faithfulness")
            state.usage["completeness"] = update.get("completeness")
            state.usage["retrieval_score"] = update.get("retrieval_score")
        lao = update.get("last_answer_object")
        if lao is not None:
            lao_dict = lao.model_dump() if hasattr(lao, "model_dump") else lao
            return {"event": "last_answer", "last_answer_object": lao_dict}
        return {"event": "progress", "phase": "finalize", "message": "Finalising answer"}
    # answer_evaluation runs inside post_process in v2, but its updates
    # are merged into the post_process node update. Keep as a fallback for
    # any other node that emits scoring fields.
    if "final_confidence" in update:
        state.usage["final_confidence"] = update.get("final_confidence")
        state.usage["confidence_level"] = update.get("confidence_level")
        state.usage["faithfulness"] = update.get("faithfulness")
        state.usage["completeness"] = update.get("completeness")
        state.usage["retrieval_score"] = update.get("retrieval_score")
        lao = update.get("last_answer_object")
        if lao is not None:
            lao_dict = lao.model_dump() if hasattr(lao, "model_dump") else lao
            return {"event": "last_answer", "last_answer_object": lao_dict}
    return None


def _process_node_updates(payload: dict, state: _V2LoopState) -> list[dict]:
    events: list[dict] = []
    for node, update in payload.items():
        if not isinstance(update, dict):
            continue
        event = _handle_node_update(node, update, state)
        if event is not None:
            events.append(event)
    return events


def _estimate_token_usage(state: _V2LoopState, query: str, file_markdown: Optional[str]) -> list[dict]:
    events: list[dict] = []
    if not state.full_answer:
        state.full_answer = "I'm sorry, I could not produce an answer."
        events.append({"event": "answer_rewrite", "content": state.full_answer, "citations": []})

    if state.provider_usage:
        state.usage["promptTokens"] = state.provider_usage.get("input_tokens", 0)
        state.usage["completionTokens"] = state.provider_usage.get("output_tokens", 0)
        state.usage["estimated"] = False
        events.append({"event": "done", "usage": state.usage})
        return events

    # Estimate: system prompt once per think iteration + user prompt + observations.
    system_tokens = count_tokens(AGENT_V2_PROMPT)
    prompt_tokens = system_tokens * max(state.think_iterations, 1)
    prompt_tokens += count_tokens(query)
    prompt_tokens += sum(
        count_tokens(json.dumps(o, default=str)) for o in state.observations
    )
    if file_markdown:
        prompt_tokens += count_tokens(file_markdown)
    for d in state.citations:
        prompt_tokens += count_tokens(d.get("page_content", ""))
    completion_tokens = count_tokens(state.full_answer)
    state.usage["promptTokens"] = prompt_tokens
    state.usage["completionTokens"] = completion_tokens
    state.usage["estimated"] = True
    events.append({"event": "done", "usage": state.usage})
    return events


async def _background_extract_and_persist(
    message_id: int,
    answer: str,
    org_id: Optional[int],
    eval_kwargs: dict,
) -> None:
    """Run structured extraction in the background and update the DB."""
    try:
        from app.services.agentic_rag.evaluator import extract_structured
        from app.db.session import SessionLocal
        from app.models.chat import Message
        from app.services.agentic_rag.schemas import LastAnswerObject, DataPoint

        extraction = await extract_structured(answer=answer, **eval_kwargs)
        db = SessionLocal()
        try:
            msg = db.query(Message).filter(Message.id == message_id).first()
            if not msg or not msg.last_answer_object:
                return
            lao = LastAnswerObject(**msg.last_answer_object)
            lao.summary = extraction.summary
            lao.key_points = extraction.key_points
            if extraction.data:
                try:
                    lao.data = [DataPoint(**d) if isinstance(d, dict) else d for d in extraction.data]
                except Exception:
                    lao.data = None
            msg.last_answer_object = lao.model_dump()
            db.commit()
        finally:
            db.close()
    except Exception as exc:
        logger.warning("[BG_EXTRACT_V2] failed for message %d: %s", message_id, exc)


async def run_agent_loop_v2(
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
    """Run the agentic-v2 loop and stream SSE-style events."""
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

    graph = build_agent_graph_v2(ctx)

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

    state = _V2LoopState(message_id)

    async for chunk in graph.astream(initial_state, config, stream_mode=["updates", "custom"]):
        if chat_id is not None and is_cancelled(chat_id):
            logger.debug("[agent_runner_v2] cancel detected | chat_id=%d", chat_id)
            break
        kind, payload = chunk if isinstance(chunk, tuple) else ("updates", chunk)

        if kind == "custom":
            state.full_answer, event = _handle_custom_event(payload, state.full_answer)
            if event is not None:
                yield event
            continue

        if kind != "updates" or not isinstance(payload, dict):
            continue

        # Detect LangGraph interrupt (from the clarify tool) and emit
        # an interrupt event. The graph is checkpointed and paused —
        # the /clarification endpoint resumes it with Command(resume=...).
        if "__interrupt__" in payload:
            interrupts = payload["__interrupt__"]
            value = interrupts[0].value if interrupts else None
            question = value.get("question", "") if isinstance(value, dict) else str(value or "")
            yield {"event": "interrupt", "question": question, "thread_id": thread_id}
            return

        for event in _process_node_updates(payload, state):
            yield event

    for event in _estimate_token_usage(state, query, file_markdown):
        yield event

    # Background structured extraction (fire-and-forget).
    if message_id and state.full_answer:
        try:
            eval_kwargs: dict = {}
            try:
                utility_cfg = get_org_llm(org_id, db, role="utility")
                eval_kwargs = {
                    "api_base": utility_cfg["api_base"],
                    "api_key": utility_cfg["api_key"],
                    "utility_model": utility_cfg["model_name"],
                }
            except Exception:
                pass
            task = asyncio.create_task(
                _background_extract_and_persist(
                    message_id=message_id,
                    answer=state.full_answer,
                    org_id=org_id,
                    eval_kwargs=eval_kwargs,
                )
            )
            _BACKGROUND_TASKS.add(task)
            task.add_done_callback(_BACKGROUND_TASKS.discard)
            logger.info("[agent_runner_v2] spawned background extraction | message_id=%s", message_id)
        except Exception as exc:
            logger.warning("[agent_runner_v2] failed to spawn background extraction: %s", exc)


async def resume_agent_loop_v2(
    resume_value: str,
    chat_id: int,
    db: Any,
    org_id: Optional[int] = None,
    user_id: Optional[int] = None,
    message_id: Optional[int] = None,
) -> AsyncGenerator[dict, None]:
    """Resume a paused v2 graph after user clarification.

    Takes the user's clarification response and resumes the paused graph
    via Command(resume=...). Yields the same SSE-style events as
    run_agent_loop_v2.
    """
    from langgraph.types import Command

    memory = await get_redis_memory()
    thread_id = f"chat-{chat_id}"
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

    graph = build_agent_graph_v2(ctx)
    state = _V2LoopState(message_id)

    async for chunk in graph.astream(
        Command(resume=resume_value), config, stream_mode=["updates", "custom"],
    ):
        if chat_id is not None and is_cancelled(chat_id):
            logger.debug("[agent_runner_v2] cancel detected during resume | chat_id=%d", chat_id)
            break
        kind, payload = chunk if isinstance(chunk, tuple) else ("updates", chunk)

        if kind == "custom":
            state.full_answer, event = _handle_custom_event(payload, state.full_answer)
            if event is not None:
                yield event
            continue

        if kind != "updates" or not isinstance(payload, dict):
            continue

        # Detect re-interrupt (clarify called again after resume).
        if "__interrupt__" in payload:
            interrupts = payload["__interrupt__"]
            value = interrupts[0].value if interrupts else None
            question = value.get("question", "") if isinstance(value, dict) else str(value or "")
            yield {"event": "interrupt", "question": question, "thread_id": thread_id}
            return

        for event in _process_node_updates(payload, state):
            yield event

    for event in _estimate_token_usage(state, "", None):
        yield event

    # Background structured extraction (fire-and-forget).
    if message_id and state.full_answer:
        try:
            eval_kwargs: dict = {}
            try:
                utility_cfg = get_org_llm(org_id, db, role="utility")
                eval_kwargs = {
                    "api_base": utility_cfg["api_base"],
                    "api_key": utility_cfg["api_key"],
                    "utility_model": utility_cfg["model_name"],
                }
            except Exception:
                pass
            task = asyncio.create_task(
                _background_extract_and_persist(
                    message_id=message_id,
                    answer=state.full_answer,
                    org_id=org_id,
                    eval_kwargs=eval_kwargs,
                )
            )
            _BACKGROUND_TASKS.add(task)
            task.add_done_callback(_BACKGROUND_TASKS.discard)
            logger.info("[agent_runner_v2] spawned background extraction | message_id=%s", message_id)
        except Exception as exc:
            logger.warning("[agent_runner_v2] failed to spawn background extraction: %s", exc)
