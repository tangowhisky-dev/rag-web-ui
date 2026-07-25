"""LangGraph-based pipeline runner — the single pipeline implementation.

Uses compiled LangGraph StateGraph for execution.
Routes through nodes: rewrite -> classify -> [Send(agent, ...)] -> synthesize.
Streams SSE events (p:/t:/th:/0:/1:/2:/3:/4:/d:) in real-time.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, AsyncGenerator, List, Optional

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command

from app.core.config import settings

from .graph import build_main_graph
from .graph_state import AgentState
from .redis_memory import get_redis_memory
from .streaming import AgenticRAGTransformer
from .utils import format_context_string

logger = logging.getLogger(__name__)


def _final_answer_to_string(fa_raw: Any) -> str:
    """Normalize final_answer (string, list of dicts, or other) to a string."""
    if isinstance(fa_raw, list):
        return "".join(
            part.get("text", "")
            if isinstance(part, dict) and part.get("type") == "text"
            else str(part)
            for part in fa_raw
        )
    return str(fa_raw) if fa_raw else ""


async def run_agentic_rag(
    query: str,
    kb_ids: List[int],
    db: Any,
    file_markdown: Optional[str] = None,
    temperature: float = 0.0,
    model_name: Optional[str] = None,
    api_base: Optional[str] = None,
    org_id: Optional[int] = None,
    chat_id: Optional[int] = None,
    user_id: Optional[int] = None,
    generate_answer: bool = True,
) -> AsyncGenerator[dict, None]:
    """Run the agentic RAG pipeline using a compiled LangGraph StateGraph.

    The compiled graph owns the routing logic; this runner only:
    1. Builds the initial state from history
    2. Executes the graph via astream_events(..., version="v3") with a custom
       stream transformer that consumes the raw protocol stream
    3. Handles interrupt() pauses for clarification (human-in-the-loop)
    4. Extracts the final state to emit context, done, evaluation, and usage events

    Note: ``db`` is accepted for API compatibility but NOT passed to the pipeline.
    The retrieval functions create their own fresh sessions internally. Passing the
    caller's ``db`` session risks corrupting the caller's ORM state if any pipeline
    operation triggers a rollback or session invalidation.
    """
    t0 = time.monotonic()

    # The durable Redis checkpointer is the single source of truth for thread
    # state. All prior turns are loaded from the checkpoint; we only append the
    # current user query here.
    memory = await get_redis_memory()
    thread_id = f"chat-{chat_id}" if chat_id else f"anon-{uuid.uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id}}

    messages: list = [HumanMessage(content=query)]

    initial_state: AgentState = AgentState(
        messages=messages,
        original_query=query,
        kb_ids=kb_ids,
        org_id=org_id,
        chat_id=chat_id,
        user_id=user_id,
        file_markdown=file_markdown,
        generate_answer=generate_answer,
    )

    # Build the compiled graph with the shared Redis checkpointer + store.
    # NOTE: db is NOT passed to the pipeline — retrieval functions create their
    # own fresh sessions internally. Passing the caller's db risks corrupting
    # the caller's ORM state (e.g. detached bot_message after a rollback).
    graph = build_main_graph(
        db=None,
        kb_ids=kb_ids,
        org_id=org_id,
        file_markdown=file_markdown,
        api_base=api_base,
        generate_answer=generate_answer,
        checkpointer=memory.checkpointer,
        store=memory.store,
    )

    try:
        stream = await graph.astream_events(
            initial_state,
            config=config,
            version="v3",
            transformers=[AgenticRAGTransformer],
        )

        # The transformer instance is exposed on stream.extensions["agentic"]
        # because init() returns {"agentic": self}.
        transformer = stream.extensions["agentic"]
        events_channel = stream.extensions["events"]

        # The raw protocol stream must be consumed to drive the transformer,
        # but we only care about the transformed "events" channel. Run raw
        # consumption in a background task and yield from the events channel.
        async def _drain_raw() -> None:
            async for _ in stream:
                pass

        raw_task = asyncio.create_task(_drain_raw())

        try:
            async for event in events_channel:
                if event:
                    yield event
        finally:
            # Ensure the raw stream finishes even if the consumer cancels early.
            if not raw_task.done():
                raw_task.cancel()
                try:
                    await raw_task
                except asyncio.CancelledError:
                    pass

        # Check if the run was interrupted (human-in-the-loop)
        if stream.interrupted:
            # Extract interrupt payloads for the frontend
            interrupt_payloads = []
            for intp in stream.interrupts:
                interrupt_payloads.append(str(intp.value))
            
            logger.info(
                "[GRAPH] interrupted for clarification | thread_id=%s | question=%s",
                thread_id, interrupt_payloads[0][:200] if interrupt_payloads else "",
            )
            
            yield {
                "event": "interrupt",
                "question": interrupt_payloads[0] if interrupt_payloads else "",
                "thread_id": thread_id,
            }
            return

        # Final state from the run
        final_output = await stream.output()
        final_state: dict = (
            final_output
            if isinstance(final_output, dict)
            else getattr(final_output, "values", {}) or {}
        )

        fa_raw = final_state.get("final_answer") or final_state.get("answer", "")
        final_answer = _final_answer_to_string(fa_raw)

        # Aggregate retrieved docs from final state; fall back to the docs the
        # transformer already streamed, in case the graph state reducer doesn't
        # surface them in the root final state.
        retrieved_docs = final_state.get("retrieved_docs", [])
        all_docs = retrieved_docs if isinstance(retrieved_docs, list) else []
        if not all_docs and transformer is not None and transformer._all_docs:
            all_docs = transformer._all_docs
        conf = final_state.get("retrieval_confidence", 0.0)

        # Usage is collected by the transformer from message-finish events.
        # OpenAI-compatible endpoints often omit usage in streaming mode, so we
        # fall back to usage_metadata attached to the final AIMessage.
        input_tokens = getattr(transformer, "_input_tokens", 0) if transformer is not None else 0
        completion_tokens = getattr(transformer, "_output_tokens", 0) if transformer is not None else 0
        if input_tokens == 0 and completion_tokens == 0:
            # Fallback 1: usage captured during generation streaming.
            answer_usage = final_state.get("answer_usage")
            if isinstance(answer_usage, dict):
                input_tokens = answer_usage.get("input_tokens", 0) or 0
                completion_tokens = answer_usage.get("output_tokens", 0) or 0
            # Fallback 2: usage attached to final state messages.
            if input_tokens == 0 and completion_tokens == 0:
                for msg in final_state.get("messages", []):
                    usage = getattr(msg, "usage_metadata", None)
                    if isinstance(usage, dict):
                        input_tokens += usage.get("input_tokens", 0) or 0
                        completion_tokens += usage.get("output_tokens", 0) or 0

        # Build the final citation list in display order (1..M) from the cited
        # original doc indices and the full retrieved docs.
        cited_doc_indices = final_state.get("cited_doc_indices", [])
        cited_docs: list[dict] = []
        for idx in cited_doc_indices:
            if 1 <= idx <= len(all_docs):
                doc = all_docs[idx - 1]
                # Serialize to the same shape the frontend already expects.
                doc_dict = (
                    {"page_content": doc.page_content, "metadata": doc.metadata}
                    if hasattr(doc, "page_content") else doc
                )
                cited_docs.append(doc_dict)

        # Emit the answer_rewrite event with the normalized answer + cited docs.
        # This replaces the raw streamed text in the UI and provides the exact
        # citation list that corresponds to the [1], [2], ... markers.
        if final_answer:
            yield {
                "event": "answer_rewrite",
                "content": final_answer,
                "citations": cited_docs,
            }

        # If there is no final answer and no docs, stream whatever we have as a token
        if final_answer and not all_docs:
            yield {"event": "token", "content": final_answer}

        # Answer quality evaluation — already computed by answer_evaluation_node
        usage: dict[str, Any] = {
            "promptTokens": input_tokens,
            "completionTokens": completion_tokens,
            "final_confidence": final_state.get("final_confidence", 0.0),
            "confidence_level": final_state.get("confidence_level", "none"),
            "faithfulness": final_state.get("faithfulness", 0),
            "completeness": final_state.get("completeness", 0),
            "citation_quality": final_state.get("citation_quality", 0),
            "confidence_match": final_state.get("confidence_match", True),
            "flags": final_state.get("evaluation_flags", []),
        }

        yield {"event": "done", "full_response": final_answer, "usage": usage}

        # The completed turn is persisted to long-term memory by the
        # "save_memory" node inside the compiled graph, so we do not duplicate
        # the write here.

    except Exception as exc:
        logger.error("[GRAPH] pipeline failed: %s", exc, exc_info=True)
        yield {"event": "error", "message": str(exc)}
        yield {"event": "done", "full_response": "", "usage": {}}

    logger.info(
        "[GRAPH] total latency=%.1fms query=%r",
        (time.monotonic() - t0) * 1000,
        query[:80],
    )
