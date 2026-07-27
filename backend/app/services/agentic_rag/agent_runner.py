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
    usage = {"promptTokens": 0, "completionTokens": 0, "messageId": message_id}

    try:
        async for chunk in graph.astream(initial_state, config, stream_mode="updates"):
            for node, update in chunk.items():
                if not isinstance(update, dict):
                    continue

                if node == "plan" and update.get("plan"):
                    plan = update.get("plan")
                    data = plan.model_dump() if hasattr(plan, "model_dump") else plan
                    yield {"event": "plan", "plan": data}

                elif node == "think" and update.get("tool_calls"):
                    for tc in update["tool_calls"]:
                        yield {"event": "tool_call", "tool": tc.get("tool"), "arguments": tc.get("arguments", {})}

                elif node == "tool" and update.get("observations"):
                    for obs in update["observations"]:
                        payload = obs.model_dump() if hasattr(obs, "model_dump") else obs
                        yield {"event": "tool_observation", **payload}

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

                elif node == "save_memory":
                    if full_answer:
                        yield {"event": "answer_rewrite", "content": full_answer, "citations": citations}

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
    yield {
        "event": "done",
        "usage": {
            "promptTokens": 1,
            "completionTokens": 1,
            **usage,
        },
    }
