"""Runner for the enterprise agent loop graph."""

from __future__ import annotations

import logging
import uuid
from typing import Any, AsyncGenerator, Optional

from langchain_core.messages import HumanMessage

from app.core.config import settings
from app.services.agentic_rag.agent_graph import build_agent_graph
from app.services.agentic_rag.graph_state import AgentState
from app.services.agentic_rag.llm_factory import get_org_llm
from app.services.agentic_rag.redis_memory import get_redis_memory
from app.services.agentic_rag.token_budget import count_tokens
from app.services.agentic_rag.tool_context import ToolContext

logger = logging.getLogger(__name__)


async def run_agent_loop(
    query: str,
    kb_ids: list[int],
    db: Any,
    file_markdown: Optional[str] = None,
    temperature: float = 0.0,
    model_name: Optional[str] = None,
    api_base: Optional[str] = None,
    org_id: Optional[int] = None,
    chat_id: Optional[int] = None,
    user_id: Optional[int] = None,
    message_id: Optional[int] = None,
    display_query: Optional[str] = None,
    query_model: Optional[str] = None,
) -> AsyncGenerator[dict, None]:
    """Run the new enterprise agent loop and stream SSE-style events."""
    memory = await get_redis_memory()
    thread_id = f"chat-{chat_id}" if chat_id else f"anon-{uuid.uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id}}

    org_cfg = get_org_llm(org_id, db, role="chat")
    ctx = ToolContext(
        db=db,
        user_id=user_id,
        org_id=org_id,
        chat_id=chat_id,
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

    full_answer = ""
    citations = []
    observations: list[dict] = []
    usage = {"promptTokens": 0, "completionTokens": 0, "messageId": message_id}

    try:
        async for chunk in graph.astream(initial_state, config, stream_mode=["updates", "custom"]):
            kind, payload = chunk if isinstance(chunk, tuple) else ("updates", chunk)

            if kind == "custom":
                if not isinstance(payload, dict):
                    continue
                if payload.get("event") == "token":
                    token = payload.get("content", "")
                    if token:
                        full_answer += token
                        yield payload
                else:
                    yield payload
                continue

            if kind != "updates" or not isinstance(payload, dict):
                continue

            for node, update in payload.items():
                if not isinstance(update, dict):
                    continue

                # plan, tool_call, and tool_observation events are emitted via
                # the custom stream (writer() calls in agent_graph nodes). The
                # update stream is used only to accumulate state needed for
                # token accounting below — no events yielded here to avoid
                # duplicating every pl:/tc:/to: event on the frontend.
                if node == "tool" and update.get("observations"):
                    for obs in update["observations"]:
                        observation_payload = obs.model_dump() if hasattr(obs, "model_dump") else obs
                        observations.append(observation_payload)

                elif node == "finalize":
                    final = update.get("final_answer", "")
                    if final:
                        full_answer = final
                    citations = update.get("retrieved_docs", [])
                    yield {"event": "progress", "phase": "finalize", "message": "Finalising answer"}

                elif node == "answer_scoring":
                    usage["final_confidence"] = update.get("final_confidence")
                    usage["confidence_level"] = update.get("confidence_level")
                    usage["faithfulness"] = update.get("faithfulness")
                    usage["completeness"] = update.get("completeness")

    except Exception as exc:
        # Handle LangGraph interrupts for human-in-the-loop clarification.
        exc_name = type(exc).__name__
        if exc_name == "GraphInterrupt":
            value = getattr(exc, "value", None)
            if isinstance(value, list) and value:
                value = value[0]
            ivalue = getattr(value, "value", value)
            question = ivalue.get("question", "") if isinstance(ivalue, dict) else str(ivalue)
            yield {"event": "interrupt", "question": question}
            return
        raise

    if not full_answer:
        full_answer = "I'm sorry, I could not produce an answer."
    yield {"event": "answer_rewrite", "content": full_answer, "citations": citations}

    prompt_tokens = count_tokens(str(initial_state.messages)) + count_tokens(str(citations)) + count_tokens(str(observations))
    completion_tokens = count_tokens(full_answer)
    usage["promptTokens"] = prompt_tokens
    usage["completionTokens"] = completion_tokens
    yield {
        "event": "done",
        "usage": usage,
    }
