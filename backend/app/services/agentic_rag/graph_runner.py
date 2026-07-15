"""LangGraph-based pipeline runner — the single pipeline implementation.

Uses compiled LangGraph StateGraph for execution.
Routes through nodes: summarize_history → rewrite → classify → [Send(agent, ...)] → synthesize.
Streams SSE events (p:/t:/th:/0:/1:/2:/3:/4:/d:) in real-time.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncGenerator, List, Optional

from langchain_core.messages import HumanMessage, AIMessage

from app.core.config import settings

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
    2. Executes the graph via astream_events to stream tokens
    3. Handles interrupt() pauses for clarification
    4. Extracts the final state to emit context, done, and usage events
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

    all_docs: List[dict] = []
    interrupted = False

    # ── Task list tracking (for complex queries with subtasks) ───────────
    _task_texts: List[str] = []
    _completed_subtasks: int = 0

    def _make_task_list(texts: List[str], done_count: int) -> List[dict]:
        tasks = []
        for i, text in enumerate(texts):
            if i < done_count:
                status = "done"
            elif i == done_count:
                status = "active"
            else:
                status = "pending"
            tasks.append({"id": i, "text": text, "status": status})
        return tasks

    try:
        async for event in graph.astream_events(initial_state, config=config, version="v2"):
            event_type = event.get("event", "")
            name = event.get("name", "")
            data = event.get("data", {})

            # ── Task list: emit after classify_query to show subtasks ────
            if event_type == "on_chain_end" and name == "classify_query":
                output = data.get("output", {}) or {}
                subtasks = output.get("subtasks", [])
                if isinstance(subtasks, list) and len(subtasks) > 1:
                    _task_texts = subtasks
                    _completed_subtasks = 0
                    yield {"event": "task_list", "tasks": _make_task_list(_task_texts, 0)}

            # ── Task list: update as each agent_subgraph (subtask) finishes ─
            if event_type == "on_chain_end" and name == "agent_subgraph" and _task_texts:
                _completed_subtasks = min(_completed_subtasks + 1, len(_task_texts))
                yield {"event": "task_list", "tasks": _make_task_list(_task_texts, _completed_subtasks)}

            # ── Clarification interrupt: only check when classify_query ends ─
            # (aget_state is expensive; no need to call it for every node end)
            if event_type == "on_chain_end" and name == "classify_query" and not interrupted:
                try:
                    graph_state = await graph.aget_state(config)
                    if (
                        hasattr(graph_state, "next")
                        and graph_state.next
                        and "request_clarification" in graph_state.next
                    ):
                        interrupted = True
                        logger.info("[GRAPH] interrupted for clarification")
                        msgs = getattr(graph_state, "values", {}).get("messages", [])
                        clarification_msg = msgs[-1].content if msgs else "Please provide clarification."
                        yield {"event": "progress", "phase": "clarification", "message": clarification_msg}
                except Exception:
                    pass

            # ── Agent step events (node start / finish) ───────────────
            _STEP_NODES = frozenset((
                "load_historical_memory", "summarize_history",
                "rewrite_query", "classify_query", "request_clarification",
                "rewrite_subtask_query",
                "dense_retrieval", "sparse_retrieval", "exact_retrieval",
                "merge", "sufficiency_check", "graph_expansion",
                "reranking", "adaptive_reranking", "collect_context",
                "prepare_final_context", "generating", "chart_validation",
                "answer_evaluation", "finalize_answer",
            ))
            if name in _STEP_NODES:
                if event_type == "on_chain_start":
                    yield {"event": "agent_step", "node": name, "status": "active", "latency_ms": 0}
                elif event_type == "on_chain_end":
                    yield {"event": "agent_step", "node": name, "status": "done", "latency_ms": 0}

            # ── LLM token streaming ───────────────────────────────────
            # LangChain 1.x emits on_chat_model_stream (not on_llm_stream).
            # Content lives in data["chunk"].content; handle both str and list.
            if event_type == "on_chat_model_stream":
                chunk = data.get("chunk")
                raw = getattr(chunk, "content", "") if chunk else ""
                if isinstance(raw, list):
                    # Some models return [{"type": "text", "text": "..."}]
                    content = "".join(
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in raw
                    )
                else:
                    content = raw or ""
                if content:
                    yield {"event": "token", "content": content}

        # ── Extract final state from the graph ──────────────────────
        final_state = await graph.aget_state(config)
        if isinstance(final_state, dict):
            state = final_state.get("values", final_state) if isinstance(final_state.get("values"), dict) else final_state
        else:
            # StateSnapshot (from aget_state) has a .values attribute that
            # is a dict of all state fields.
            state = getattr(final_state, "values", {})
            if not isinstance(state, dict):
                state = {}

        # ── Emit retrieved context (docs + confidence) ───────────────
        retrieved_docs = state.get("retrieved_docs", [])
        if retrieved_docs:
            all_docs.extend(retrieved_docs)
        conf = state.get("retrieval_confidence", 0.0)
        conf_level = "high" if conf > 0.7 else ("medium" if conf > 0.3 else "low")
        is_synthesis = len(state.get("subtask_answers", [])) > 1
        if all_docs:
            yield {
                "event": "context",
                "docs": all_docs,
                "confidence": conf_level,
                "score": int(conf * 100),
                "synthesis_mode": is_synthesis,
            }

        # ── Final answer ─────────────────────────────────────────────
        final_answer = state.get("final_answer") or state.get("answer", "")
        usage = {
            "promptTokens": 0,
            "completionTokens": 0,
            "final_confidence": state.get("final_confidence", 0.0),
            "confidence_level": state.get("confidence_level", "none"),
            "faithfulness": state.get("faithfulness", 0),
            "completeness": state.get("completeness", 0),
        }

        # If nothing was streamed (e.g., no docs or generation disabled) but
        # a final answer exists, emit it as a single token.
        if final_answer and not all_docs:
            yield {"event": "token", "content": final_answer}

        # ── Optional post-hoc answer quality evaluation ────────────────
        # Note: evaluation is now done inside the graph via answer_evaluation_node.
        # This block is kept for legacy external evaluation (e.g. eval endpoint).
        if settings.ANSWER_QUALITY_GRADING_ENABLED and final_answer and all_docs:
            try:
                from .evaluator import evaluate_answer, summarize_evaluation
                from .utils import format_context_string
                context_text = format_context_string(all_docs, state.get("file_markdown"))
                conf_level_eval = (
                    "very_high" if conf > 0.8 else
                    "high"      if conf > 0.6 else
                    "medium"    if conf > 0.3 else "low"
                )
                evaluation = await evaluate_answer(
                    query=state.get("original_query", ""),
                    answer=final_answer,
                    context_preview=context_text,
                    confidence_level=conf_level_eval,
                )
                usage["evaluation"] = {
                    "faithfulness": evaluation.faithfulness,
                    "completeness": evaluation.completeness,
                    "citation_quality": evaluation.citation_quality,
                    "confidence_match": evaluation.confidence_match,
                    "flags": evaluation.flags,
                }
                yield {
                    "event": "evaluation",
                    "faithfulness": evaluation.faithfulness,
                    "completeness": evaluation.completeness,
                    "citation_quality": evaluation.citation_quality,
                    "confidence_match": evaluation.confidence_match,
                    "flags": evaluation.flags,
                }
                logger.info("[EVAL] answer_quality=%s", summarize_evaluation(evaluation))
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



