"""LangGraph-based pipeline runner — the single pipeline implementation.

Uses compiled LangGraph StateGraph for execution.
Routes through nodes: rewrite → classify → [direct_retrieval | agent_subgraph] → synthesize.
Streams SSE events (p:/t:/th:/0:/1:/2:/3:/4:/d:) in real-time.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncGenerator, List, Optional

from langchain_core.messages import HumanMessage, AIMessage

from app.core.config import settings

from .callbacks import SSEEventEmitter
from .graph import build_main_graph
from .graph_state import AgentState

logger = logging.getLogger(__name__)


async def run_agentic_rag(
    query: str,
    kb_ids: List[int],
    db: Any,
    recent_lc_history: list,
    existing_summary: Optional[str] = None,
    file_markdown: Optional[str] = None,
    use_dense: bool = True,
    use_sparse: bool = True,
    use_exact: bool = True,
    use_graph_rag: bool = False,
    temperature: float = 0.0,
    model_name: Optional[str] = None,
    api_base: Optional[str] = None,
    org_id: Optional[int] = None,
    chat_id: Optional[int] = None,
) -> AsyncGenerator[dict, None]:
    """Run the agentic RAG pipeline using a compiled LangGraph StateGraph.

    The compiled graph owns the routing logic; this runner only:
    1. Builds the initial state from history
    2. Executes the graph via astream_events to stream tokens
    3. Extracts the final state to emit context, done, and usage events
    """
    t0 = time.monotonic()
    emitter = SSEEventEmitter()

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
    )

    # Build the compiled graph
    graph = build_main_graph(
        db=db,
        kb_ids=kb_ids,
        org_id=org_id,
        file_markdown=file_markdown,
        use_dense=use_dense,
        use_sparse=use_sparse,
        use_exact=use_exact,
        use_graph_rag=use_graph_rag,
        api_base=api_base,
    )

    # Config — thread_id enables checkpointing per chat
    config = {
        "configurable": {"thread_id": str(chat_id) if chat_id else None},
    }

    all_docs: List[dict] = []

    try:
        async for event in graph.astream_events(initial_state, config=config, version="v2"):
            event_type = event.get("event", "")
            name = event.get("name", "")
            data = event.get("data", {})

            # ── Node entry/exit for agent_step events ─────────────────
            if event_type == "on_chain_start" and name in (
                "rewrite_query", "classify_query", "request_clarification",
                "synthesize", "direct_retrieval", "agent_subgraph",
                "sufficiency_check", "generating", "adaptive_reranking",
                "chart_validation",
            ):
                await _emit_agent_step(emitter, name, "active")

            elif event_type == "on_chain_end" and name in (
                "rewrite_query", "classify_query", "request_clarification",
                "synthesize", "direct_retrieval", "agent_subgraph",
                "sufficiency_check", "generating", "adaptive_reranking",
                "chart_validation",
            ):
                elapsed_ms = 0  # astream_events doesn't give per-node timing
                await _emit_agent_step(emitter, name, "done", elapsed_ms)

            # ── LLM token streaming ───────────────────────────────────
            elif event_type == "on_llm_stream":
                content = data.get("content", "")
                if content:
                    await emitter.emit_token(content)
                    yield {"event": "token", "content": content}

        # ── Extract final state from the graph ──────────────────────
        final_state = await graph.aget_state(config)
        if isinstance(final_state, dict):
            state = final_state.get("values", final_state) if isinstance(final_state.get("values"), dict) else final_state
        else:
            state = dict(final_state) if final_state else {}

        # Emit context events from collected docs
        if state.get("retrieved_docs"):
            docs = state["retrieved_docs"]
            all_docs.extend(docs)
            conf = state.get("retrieval_confidence", 0.0)
            conf_level = "high" if conf > 0.7 else ("medium" if conf > 0.3 else "low")

            yield {
                "event": "context",
                "docs": docs,
                "confidence": conf_level,
                "score": int(conf * 100),
            }

        # Emit context for all collected docs
        if all_docs:
            await emitter.emit_context(
                docs=all_docs[:10],
                confidence="high" if len(all_docs) > 5 else ("medium" if all_docs else "low"),
                synthesis_mode=len(state.get("subtask_answers", [])) > 1,
            )

        # Final answer and usage from graph state
        final_answer = state.get("final_answer") or state.get("answer", "")
        thinking_chunks = state.get("thinking_chunks", [])
        usage = {"promptTokens": 0, "completionTokens": 0}

        # ── In-pipeline answer quality evaluation ───────────────────
        if settings.ANSWER_QUALITY_GRADING_ENABLED and final_answer and all_docs:
            try:
                from .evaluator import evaluate_answer, summarize_evaluation
                from .utils import format_context_string

                context_text = format_context_string(all_docs, state.get("file_markdown"))
                conf_score = state.get("retrieval_confidence", 0.0)
                conf_level = "very_high" if conf_score > 0.8 else (
                    "high" if conf_score > 0.6 else (
                        "medium" if conf_score > 0.3 else "low"
                    )
                )

                evaluation = await evaluate_answer(
                    query=state.get("original_query", ""),
                    answer=final_answer,
                    context_preview=context_text,
                    confidence_level=conf_level,
                )
                eval_summary = summarize_evaluation(evaluation)
                usage["evaluation"] = {
                    "faithfulness": evaluation.faithfulness,
                    "completeness": evaluation.completeness,
                    "citation_quality": evaluation.citation_quality,
                    "confidence_match": evaluation.confidence_match,
                    "flags": evaluation.flags,
                }
                # Emit evaluation as a standalone event before done
                await emitter.emit_evaluation(
                    faithfulness=evaluation.faithfulness,
                    completeness=evaluation.completeness,
                    citation_quality=evaluation.citation_quality,
                    confidence_match=evaluation.confidence_match,
                    flags=evaluation.flags,
                )
                logger.info("[EVAL] answer_quality=%s", eval_summary)
            except Exception as exc:
                logger.warning("[EVAL] evaluation skipped: %s", exc)

        yield {"event": "done", "full_response": final_answer, "usage": usage}

    except Exception as exc:
        logger.error("[GRAPH] pipeline failed: %s", exc, exc_info=True)
        await emitter.emit_error(str(exc))
        yield {"event": "error", "message": str(exc)}
        yield {"event": "done", "full_response": "", "usage": {}}

    finally:
        async for event in emitter.drain():
            yield event

    logger.info(
        "[GRAPH] total latency=%.1fms query=%r",
        (time.monotonic() - t0) * 1000,
        query[:80],
    )


async def _emit_agent_step(
    emitter: SSEEventEmitter,
    node: str,
    status: str,
    latency_ms: int = 0,
) -> None:
    """Emit a 4: agent_step SSE event."""
    step = {"node": node, "status": status, "latency_ms": latency_ms}
    await emitter.emit(json.dumps(step))
