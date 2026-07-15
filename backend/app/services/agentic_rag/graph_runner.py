"""LangGraph-based pipeline runner — the single pipeline implementation.

Uses compiled LangGraph StateGraph for execution.
Routes through nodes: summarize_history → rewrite → classify → [Send(agent, ...)] → synthesize.
Streams SSE events (p:/t:/th:/0:/1:/2:/3:/4:/d:) in real-time.
"""

from __future__ import annotations

import json
import logging
import time
import asyncio
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
        # ── LangGraph v3: concurrent projections ──────────────────────
        # stream.messages  → AsyncChatModelStream (iterate to get dict chunks)
        # stream.subgraphs → subgraph lifecycle (agent_subgraph start/done)
        # stream.values    → state snapshots (dict)
        # stream.output    → final state after completion
        stream = await graph.astream_events(initial_state, config=config, version="v3")

        q = asyncio.Queue()
        all_docs: List[dict] = []
        interrupted = False

        async def _tok():
            """Read LLM token deltas from the generating node only."""
            async for msg_proj in stream.messages:
                if msg_proj.node != "generating":
                    continue
                async for chunk in msg_proj:
                    if isinstance(chunk, dict) and chunk.get('event') == 'content-block-delta':
                        delta = chunk.get('delta', {})
                        if delta.get('type') == 'text-delta':
                            text = delta.get('text', '')
                            if text:
                                await q.put({"event": "token", "content": text})

        async def _sub():
            """Read subgraph lifecycle events."""
            nonlocal _completed_subtasks
            async for sub in stream.subgraphs:
                status = getattr(sub, 'status', '')
                gname = getattr(sub, 'graph_name', '')
                if gname == 'agent_subgraph':
                    await q.put({"event": "agent_step", "node": "agent_subgraph", "status": "active" if status == "started" else "done", "latency_ms": 0})
                    if status == "started" and _task_texts:
                        _completed_subtasks = 0
                        await q.put({"event": "task_list", "tasks": _make_task_list(_task_texts, 0)})
                    elif status == "done" and _task_texts:
                        _completed_subtasks = min(_completed_subtasks + 1, len(_task_texts))
                        await q.put({"event": "task_list", "tasks": _make_task_list(_task_texts, _completed_subtasks)})

        async def _val():
            """Read state snapshots from stream.values."""
            async for snapshot in stream.values:
                if not isinstance(snapshot, dict):
                    continue
                rd = snapshot.get("retrieved_docs", [])
                if rd:
                    all_docs.extend(rd)
                    conf = snapshot.get("retrieval_confidence", 0.0)
                    synthesis = len(snapshot.get("subtask_answers", [])) > 1
                    await q.put({
                        "event": "context",
                        "docs": all_docs,
                        "confidence": "high" if conf > 0.7 else "medium" if conf > 0.3 else "low",
                        "score": int(conf * 100),
                        "synthesis_mode": synthesis,
                    })

        async def _raw():
            """Read raw lifecycle protocol events."""
            nonlocal interrupted
            async for event in stream:
                method = event.get("method", "")
                ns = event.get("params", {}).get("namespace", [])
                pd = event.get("params", {}).get("data", {})
                name = ns[0] if ns else ""
                if method == "lifecycle":
                    et = pd.get("event", "") if isinstance(pd, dict) else ""
                    _SN = frozenset((
                        "load_historical_memory","summarize_history",
                        "rewrite_query","classify_query","request_clarification",
                        "rewrite_subtask_query",
                        "dense_retrieval","sparse_retrieval","exact_retrieval",
                        "merge","sufficiency_check","graph_expansion",
                        "reranking","adaptive_reranking","collect_context",
                        "prepare_final_context","generating","chart_validation",
                        "answer_evaluation","finalize_answer",
                    ))
                    if name in _SN:
                        await q.put({"event": "agent_step", "node": name, "status": "active" if et == "started" else "done", "latency_ms": 0})
                    if name == "classify_query" and et == "ended":
                        out = pd.get("output", {}) if isinstance(pd, dict) else {}
                        if isinstance(out, dict):
                            subtasks = out.get("subtasks", [])
                            if isinstance(subtasks, list) and len(subtasks) > 1:
                                _task_texts.extend(subtasks)
                                await q.put({"event": "task_list", "tasks": _make_task_list(_task_texts, 0)})
                        if not interrupted:
                            try:
                                gs = await graph.aget_state(config)
                                if hasattr(gs, "next") and gs.next and "request_clarification" in gs.next:
                                    interrupted = True
                                    msgs = getattr(gs, "values", {}).get("messages", [])
                                    mc = msgs[-1].content if msgs else "Please provide clarification."
                                    if isinstance(mc, list):
                                        mc = str(mc)
                                    await q.put({"event": "progress", "phase": "clarification", "message": mc})
                            except Exception:
                                pass

        async def _out():
            """Read final state and emit done/evaluation."""
            nonlocal all_docs
            fs = await stream.output()
            st = fs if isinstance(fs, dict) else getattr(fs, "values", {})
            if not isinstance(st, dict):
                st = {}
            fa_raw = st.get("final_answer") or st.get("answer", "")
            # Convert list content (multimodal messages) to string
            if isinstance(fa_raw, list):
                fa = "".join(
                    part.get("text", "") if isinstance(part, dict) and part.get("type") == "text" else ""
                    for part in fa_raw
                )
            else:
                fa = str(fa_raw) if fa_raw else ""
            # Handle list-type final_answer (from chat messages)
            if isinstance(fa, list):
                fa = ''.join(
                    p.get('text', '') if isinstance(p, dict) else str(p)
                    for p in fa
                )
            elif not isinstance(fa, str):
                fa = str(fa) if fa else ""
            conf = st.get("retrieval_confidence", 0.0)
            usage = {
                "promptTokens": 0, "completionTokens": 0,
                "final_confidence": st.get("final_confidence", 0.0),
                "confidence_level": st.get("confidence_level", "none"),
                "faithfulness": st.get("faithfulness", 0),
                "completeness": st.get("completeness", 0),
            }
            if fa and not all_docs:
                await q.put({"event": "token", "content": fa})
            if settings.ANSWER_QUALITY_GRADING_ENABLED and fa and all_docs:
                try:
                    ct = format_context_string(all_docs, st.get("file_markdown"))
                    cle = "very_high" if conf > 0.8 else "high" if conf > 0.6 else "medium" if conf > 0.3 else "low"
                    ev = await evaluate_answer(query=st.get("original_query", ""), answer=fa, context_preview=ct, confidence_level=cle)
                    usage["evaluation"] = {
                        "faithfulness": ev.faithfulness, "completeness": ev.completeness,
                        "citation_quality": ev.citation_quality, "confidence_match": ev.confidence_match, "flags": ev.flags,
                    }
                    await q.put({"event": "evaluation", "faithfulness": ev.faithfulness, "completeness": ev.completeness, "citation_quality": ev.citation_quality, "confidence_match": ev.confidence_match, "flags": ev.flags})
                    logger.info("[EVAL] answer_quality=%s", summarize_evaluation(ev))
                except Exception as exc:
                    logger.warning("[EVAL] evaluation skipped: %s", exc)
            await q.put({"event": "done", "full_response": fa, "usage": usage})

        # Launch all projections concurrently
        t1 = asyncio.create_task(_tok())
        t2 = asyncio.create_task(_sub())
        t3 = asyncio.create_task(_val())
        t4 = asyncio.create_task(_raw())
        t5 = asyncio.create_task(_out())

        # Drain queue: yield items as they arrive from any projection
        tasks = [t1, t2, t3, t4, t5]
        while tasks:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            tasks = [t for t in tasks if t not in done]
            while not q.empty():
                try:
                    item = q.get_nowait()
                    if item:
                        yield item
                except asyncio.QueueEmpty:
                    break
        # Final drain after all tasks done
        while not q.empty():
            try:
                item = q.get_nowait()
                if item:
                    yield item
            except asyncio.QueueEmpty:
                break

    except Exception as exc:
        logger.error("[GRAPH] pipeline failed: %s", exc, exc_info=True)
        yield {"event": "error", "message": str(exc)}
        yield {"event": "done", "full_response": "", "usage": {}}

    logger.info(
        "[GRAPH] total latency=%.1fms query=%r",
        (time.monotonic() - t0) * 1000,
        query[:80],
    )



