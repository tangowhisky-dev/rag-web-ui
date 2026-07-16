"""LangGraph-based pipeline runner — the single pipeline implementation.

Uses compiled LangGraph StateGraph for execution.
Routes through nodes: summarize_history → rewrite → classify → [Send(agent, ...)] → synthesize.
Streams SSE events (p:/t:/th:/0:/1:/2:/3:/4:/d:) in real-time.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, List, Optional

from langchain_core.messages import HumanMessage, AIMessage

from app.core.config import settings

from .graph import build_main_graph
from .graph_state import AgentState
from .streaming import AgenticRAGTransformer
from .evaluator import evaluate_answer
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
    recent_lc_history: list,
    existing_summary: Optional[str] = None,
    file_markdown: Optional[str] = None,
    temperature: float = 0.0,
    model_name: Optional[str] = None,
    api_base: Optional[str] = None,
    org_id: Optional[int] = None,
    chat_id: Optional[int] = None,
    generate_answer: bool = True,
) -> AsyncGenerator[dict, None]:
    """Run the agentic RAG pipeline using a compiled LangGraph StateGraph.

    The compiled graph owns the routing logic; this runner only:
    1. Builds the initial state from history
    2. Executes the graph via astream_events(..., version="v3") with a custom
       stream transformer that consumes the raw protocol stream
    3. Handles interrupt() pauses for clarification
    4. Extracts the final state to emit context, done, evaluation, and usage events
    """
    t0 = time.monotonic()

    # Build initial messages from history
    messages: list = []
    for m in recent_lc_history:
        if isinstance(m, HumanMessage):
            messages.append(HumanMessage(content=m.content))
        elif isinstance(m, AIMessage):
            messages.append(AIMessage(content=m.content[:400]))
    messages.append(HumanMessage(content=query))

    initial_state: AgentState = AgentState(
        messages=messages,
        original_query=query,
        existing_summary=existing_summary or "",
        kb_ids=kb_ids,
        org_id=org_id,
        file_markdown=file_markdown,
        generate_answer=generate_answer,
    )

    # Build the compiled graph
    graph = build_main_graph(
        db=db,
        kb_ids=kb_ids,
        org_id=org_id,
        file_markdown=file_markdown,
        api_base=api_base,
        generate_answer=generate_answer,
    )

    # Config — thread_id enables checkpointing per chat
    config = {
        "configurable": {"thread_id": str(chat_id) if chat_id else None},
    }

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

        # Emit final context event if docs exist and transformer didn't already.
        if all_docs and (transformer is None or not transformer._all_docs):
            yield {
                "event": "context",
                "docs": all_docs,
                "confidence": "high" if conf > 0.7 else "medium" if conf > 0.3 else "low",
                "score": int(conf * 100),
                "synthesis_mode": len(final_state.get("subtask_contexts", [])) > 1,
            }

        # If there is no final answer and no docs, stream whatever we have as a token
        if final_answer and not all_docs:
            yield {"event": "token", "content": final_answer}

        # Answer quality evaluation
        usage: dict[str, Any] = {
            "promptTokens": input_tokens,
            "completionTokens": completion_tokens,
            "final_confidence": final_state.get("final_confidence", 0.0),
            "confidence_level": final_state.get("confidence_level", "none"),
            "faithfulness": final_state.get("faithfulness", 0),
            "completeness": final_state.get("completeness", 0),
        }

        if settings.ANSWER_QUALITY_GRADING_ENABLED and final_answer and all_docs:
            try:
                ct = format_context_string(all_docs, final_state.get("file_markdown"))
                cle = (
                    "very_high"
                    if conf > 0.8
                    else "high" if conf > 0.6 else "medium" if conf > 0.3 else "low"
                )
                ev = await evaluate_answer(
                    query=final_state.get("original_query", ""),
                    answer=final_answer,
                    context_preview=ct,
                    confidence_level=cle,
                )
                usage["evaluation"] = {
                    "faithfulness": ev.faithfulness,
                    "completeness": ev.completeness,
                    "citation_quality": ev.citation_quality,
                    "confidence_match": ev.confidence_match,
                    "flags": ev.flags,
                }
                yield {
                    "event": "evaluation",
                    "faithfulness": ev.faithfulness,
                    "completeness": ev.completeness,
                    "citation_quality": ev.citation_quality,
                    "confidence_match": ev.confidence_match,
                    "flags": ev.flags,
                }
                logger.info(
                    "[EVAL] faithfulness=%s completeness=%s citation_quality=%s confidence_match=%s flags=%s",
                    ev.faithfulness, ev.completeness, ev.citation_quality,
                    ev.confidence_match, ev.flags,
                )
            except Exception as exc:
                logger.warning("[EVAL] evaluation skipped: %s", exc)

        yield {"event": "done", "full_response": final_answer, "usage": usage}

    except Exception as exc:
        logger.error("[GRAPH] pipeline failed: %s", exc, exc_info=True)
        yield {"event": "error", "message": str(exc)}
        yield {"event": "done", "full_response": "", "usage": {}}

    logger.info(
        "[GRAPH] total latency=%.1fms query=%r",
        (time.monotonic() - t0) * 1000,
        query[:80],
    )



