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

    full_answer = ""
    citations = []
    observations: list[dict] = []
    plan_obj = None
    think_iterations = 0
    provider_usage: dict | None = None
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
                elif payload.get("event") == "answer_rewrite":
                    # Emitted by finalize_node right after generation, before
                    # Call 4 (last_answer_object extraction) or Call 5
                    # (confidence scoring) — keep full_answer in sync for the
                    # token accounting below.
                    full_answer = payload.get("content", full_answer)
                    yield payload
                else:
                    yield payload
                continue

            if kind != "updates" or not isinstance(payload, dict):
                continue

            # Human-in-the-loop clarification. `astream` does NOT raise
            # GraphInterrupt — it emits an `__interrupt__` update once the
            # interrupt checkpoint has been persisted. Surfacing the event from
            # here (rather than from a custom writer call inside the node) is
            # what makes `Command(resume=...)` have something to resume.
            if "__interrupt__" in payload:
                interrupts = payload["__interrupt__"]
                value = interrupts[0].value if interrupts else None
                question = value.get("question", "") if isinstance(value, dict) else str(value or "")
                yield {"event": "interrupt", "question": question, "thread_id": thread_id}
                return

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

                elif node == "plan":
                    plan_obj = update.get("plan")

                elif node == "think":
                    think_iterations = max(think_iterations, update.get("iteration", 0))

                elif node == "finalize":
                    # Citation-normalized content already arrived via the
                    # earlier "answer_rewrite" custom event (emitted by
                    # finalize_node itself, before Call 4/5) — this update
                    # only carries retrieved_docs for token accounting.
                    final = update.get("final_answer", "")
                    if final:
                        full_answer = final
                    citations = update.get("retrieved_docs", [])
                    if isinstance(update.get("answer_usage"), dict):
                        provider_usage = update["answer_usage"]
                    yield {"event": "progress", "phase": "finalize", "message": "Finalising answer"}

                elif node == "answer_scoring":
                    usage["final_confidence"] = update.get("final_confidence")
                    usage["confidence_level"] = update.get("confidence_level")
                    usage["faithfulness"] = update.get("faithfulness")
                    usage["completeness"] = update.get("completeness")
                    usage["retrieval_score"] = update.get("retrieval_score")

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
        yield {"event": "answer_rewrite", "content": full_answer, "citations": []}

    # ── Token accounting ────────────────────────────────────────────────
    # Prefer provider-reported usage from the generation call. When the
    # backend doesn't report usage, fall back to a reconstruction and mark it
    # as an estimate — it counts system prompts, the query, all retrieved
    # docs and observations once each, which is a lower bound, not a measured
    # figure.
    if provider_usage:
        usage["promptTokens"] = provider_usage.get("input_tokens", 0)
        usage["completionTokens"] = provider_usage.get("output_tokens", 0)
        usage["estimated"] = False
        yield {"event": "done", "usage": usage}
        return

    # System prompt overhead (constant per call type)
    plan_sys_tokens = count_tokens(AGENT_SYSTEM_PROMPT) + count_tokens(PLAN_SYSTEM_PROMPT)
    think_sys_tokens = count_tokens(AGENT_SYSTEM_PROMPT) + count_tokens(THINK_SYSTEM_PROMPT)
    finalize_sys_tokens = count_tokens(FINALIZE_GUARDRAIL_PROMPT) + count_tokens(FINALIZE_ANSWER_PROMPT)

    prompt_tokens = plan_sys_tokens  # 1 plan call
    prompt_tokens += think_sys_tokens * max(think_iterations, 1)  # think calls
    prompt_tokens += finalize_sys_tokens  # 1 finalize call

    # User query
    prompt_tokens += count_tokens(query)

    # All retrieved docs (not just cited subset)
    prompt_tokens += sum(
        count_tokens(d.get("page_content", "")) for d in citations
    )

    # Tool observations
    prompt_tokens += sum(
        count_tokens(json.dumps(o.model_dump() if hasattr(o, "model_dump") else o, default=str))
        for o in observations
    )

    # Plan
    if plan_obj:
        prompt_tokens += count_tokens(
            json.dumps(plan_obj.model_dump() if hasattr(plan_obj, "model_dump") else plan_obj, default=str)
        )

    # File markdown
    if file_markdown:
        prompt_tokens += count_tokens(file_markdown)
    completion_tokens = count_tokens(full_answer)
    usage["promptTokens"] = prompt_tokens
    usage["completionTokens"] = completion_tokens
    usage["estimated"] = True
    yield {
        "event": "done",
        "usage": usage,
    }
