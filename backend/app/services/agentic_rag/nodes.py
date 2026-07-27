"""LangGraph node implementations for the agentic RAG pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from contextlib import contextmanager
from typing import Any, Generator, List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.config import get_stream_writer
from langgraph.types import Command, interrupt

from app.core.config import settings

from app.services.infrastructure import content_hash
from app.services.infrastructure.utils import _serialise_doc
from app.services.retrieval import score_retrieval, rerank
from app.services.retrieval import (
    get_effective_datastore_ids,
    dense_search_docs,
    sparse_search_docs,
    exact_search_docs,
)
from app.services.retrieval.reranker import _get_cross_encoder

from .graph_state import AgentState
from .prompts import (
    COMPACTION_SYSTEM_PROMPT,
    COMPACTION_USER_PROMPT,
    EVALUATION_SYSTEM_PROMPT,
)
from .redis_memory import get_redis_memory
from .token_budget import ContextBudget, count_tokens
from .utils import format_context_string

logger = logging.getLogger(__name__)


def select_recent_history(messages: list, max_pairs: int = 3) -> list:
    """Return up to ``max_pairs`` of recent user/assistant turns.

    The last message is assumed to be the current user query and is excluded.
    """
    from langchain_core.messages import HumanMessage, AIMessage

    history: list = []
    # Skip the final message (current query).
    for m in messages[:-1]:
        if isinstance(m, HumanMessage):
            history.append(HumanMessage(content=m.content))
        elif isinstance(m, AIMessage):
            history.append(AIMessage(content=m.content[:400]))
    # Keep the most recent max_pairs * 2 messages.
    return history[-(max_pairs * 2):]


# ---------------------------------------------------------------------------
# Compaction / Summarization Node
# ---------------------------------------------------------------------------

def _messages_to_conversation_text(messages: list) -> str:
    """Convert LangChain messages to a readable conversation text for summarization."""
    parts = []
    for m in messages:
        if isinstance(m, HumanMessage):
            content = str(m.content)
            if len(content) > 500:
                content = content[:500] + "..."
            parts.append(f"User: {content}")
        elif isinstance(m, AIMessage):
            content = str(m.content)
            if len(content) > 800:
                content = content[:800] + "..."
            parts.append(f"Assistant: {content}")
    return "\n\n".join(parts)


async def compaction_node(state: AgentState) -> dict:
    """Compact conversation history when it grows too long.

    When the number of messages exceeds COMPACTION_HISTORY_THRESHOLD,
    summarize older messages into a structured checkpoint. The checkpoint
    preserves RAG-specific context (topics covered, documents retrieved,
    key findings) so the model can continue the conversation fluently.

    This node runs after rewrite_query but before classification/routing,
    so the query rewriter still sees full history.
    """
    if not settings.COMPACTION_ENABLED:
        return {"compaction_summary": None, "compaction_triggered": False}

    keep_recent = settings.COMPACTION_KEEP_RECENT
    max_summary_chars = settings.COMPACTION_SUMMARY_MAX_CHARS

    messages = state.get("messages", [])
    budget = ContextBudget()
    budget.add(count_tokens(_messages_to_conversation_text(messages)))
    if not budget.needs_compaction():
        return {"compaction_summary": None, "compaction_triggered": False}

    # Split: keep recent messages, summarize older ones
    recent_messages = messages[-keep_recent:]
    old_messages = messages[:len(messages) - keep_recent]

    if not old_messages:
        return {"compaction_summary": None, "compaction_triggered": False}

    logger.info(
        "[COMPACTION] triggered | total_tokens=%d | total_msgs=%d | recent=%d | summarizing=%d",
        budget.used, len(messages), len(recent_messages), len(old_messages),
    )

    # Build conversation text from old messages
    conversation_text = _messages_to_conversation_text(old_messages)

    # Build the prompt
    user_prompt = COMPACTION_USER_PROMPT.format(conversation=conversation_text)

    # Call LLM for summarization
    try:
        llm = _get_llm(
            model_name=settings.effective_query_model,  # Use query model for cheaper summarization
            temperature=0.0,
            streaming=False,
        )

        response = llm.invoke([
            {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])

        summary = str(response.content).strip()

        # Truncate summary if too long
        if len(summary) > max_summary_chars:
            summary = summary[:max_summary_chars] + "\n\n[...summary truncated for space]"

        logger.info(
            "[COMPACTION] summary generated | chars=%d",
            len(summary),
        )

        return {
            "compaction_summary": summary,
            "compaction_triggered": True,
        }

    except Exception as exc:
        logger.warning("[COMPACTION] failed: %s — continuing without summary", exc)
        return {"compaction_summary": None, "compaction_triggered": False}


@contextmanager
def _agent_step(name: str) -> Generator[None, None, None]:
    """Emit agent_step active/done lifecycle events around a node.

    No-op when called outside a LangGraph runnable context (e.g. unit tests).
    """
    writer = _safe_writer()
    if writer is not None:
        writer({"event": "agent_step", "node": name, "status": "active", "latency_ms": 0})
    try:
        yield
    finally:
        if writer is not None:
            writer({"event": "agent_step", "node": name, "status": "done", "latency_ms": 0})


def _safe_writer():
    """Return the stream writer if inside a graph context, else None.

    Retrieval nodes are called both from the graph (where stream events work)
    and from the rag_retrieve tool (where there is no graph context). This
    helper lets nodes emit progress events safely in both cases.
    """
    try:
        return get_stream_writer()
    except RuntimeError:
        return None


# ── Answer Generation ──────────────────────────────────────────────────────


def _get_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    api_base: Optional[str] = None,
    streaming: bool = False,
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name or settings.OPENAI_MODEL,
        temperature=temperature,
        openai_api_base=api_base or settings.OPENAI_API_BASE,
        openai_api_key=settings.OPENAI_API_KEY,
        streaming=streaming,
    )


# ---------------------------------------------------------------------------
# Node: rewrite_query
# ---------------------------------------------------------------------------

async def rewrite_query_node(
    state: AgentState,
    api_base: Optional[str] = None,
) -> dict:
    """Rewrite query using recent checkpoint history."""
    from .utils import rewrite_query as _rewrite_query

    with _agent_step("rewrite_query"):
        messages = state.get("messages", [])
        query = state.get("original_query", "")
        recent_history = select_recent_history(messages, max_pairs=3)

        rewritten = _rewrite_query(
            query=query,
            recent_history=recent_history,
            memory_context="",
            api_base=api_base,
            query_model=settings.effective_query_model,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_API_BASE,
        )

        # Stream the rewritten query so the UI can display it immediately.
        writer = _safe_writer()
        if writer:
            writer({"event": "rewritten_query", "query": rewritten})

        return {"rewritten_query": rewritten}


# ---------------------------------------------------------------------------
# Node: neo4j_expansion (always runs after merge)
# ---------------------------------------------------------------------------

async def neo4j_expansion_node(
    state: AgentState,
    db: Any,
    kb_ids: List[int] | None = None,
    org_id: int | None = None,
    file_markdown: str | None = None,
) -> dict:
    """Expand retrieved docs via Neo4j graph relationships."""
    kb_ids = kb_ids or state.get("kb_ids", [])
    org_id = org_id if org_id is not None else state.get("org_id")
    file_markdown = file_markdown or state.get("file_markdown")

    with _agent_step("neo4j_expansion"):
        docs = state.get("retrieved_docs", [])
        if not docs:
            return {"graph_docs": [], "graph_expansion_done": True}

        try:
            from langchain_core.documents import Document as LangchainDocument
            from app.services.graph import expand_docs_via_graph

            lc_docs = [
                LangchainDocument(page_content=d.get("page_content", ""), metadata=d.get("metadata", {}))
                for d in docs
            ]
            loop = asyncio.get_event_loop()
            expanded = await loop.run_in_executor(None, lambda: expand_docs_via_graph(lc_docs, kb_ids))
            existing_hashes = {content_hash(d.page_content) for d in lc_docs}
            new_docs = [
                _serialise_doc(d)
                for d in expanded
                if content_hash(d.page_content) not in existing_hashes
            ]
        except Exception as exc:
            logger.warning("[NEO4J_EXPANSION] failed: %s", exc)
            new_docs = []

        merged = docs + new_docs

        return {
            "graph_docs": new_docs,
            "retrieved_docs": merged,
            "graph_expansion_done": True,
        }


# ---------------------------------------------------------------------------
# Node: reranking (scores all docs with -inf threshold)
# ---------------------------------------------------------------------------

def reranking_node(
    state: AgentState,
) -> dict:
    """Rerank merged docs with the cross-encoder using -inf threshold.

    Scores ALL docs so they can be re-filtered later by adaptive reranking
    without re-running the cross-encoder.
    """
    query = state.get("rewritten_query", state.get("original_query", ""))
    docs = state.get("retrieved_docs", [])

    with _agent_step("reranking"):
        if not docs:
            return {
                "retrieved_docs": [],
                "all_scored_docs": [],
                "retrieval_confidence": 0.0,
            }

        try:
            from langchain_core.documents import Document as LangchainDocument
            lc_docs = [
                LangchainDocument(page_content=d.get("page_content", ""), metadata=d.get("metadata", {}))
                for d in docs
            ]
            reranked = rerank(query=query, docs=lc_docs, score_threshold=float("-inf"))
            serialised = [_serialise_doc(d) for d in reranked]
        except Exception as exc:
            logger.warning("[RERANKING] failed: %s", exc)
            serialised = docs

        conf_result = score_retrieval(serialised, {}) if serialised else None
        conf_score = conf_result.score / 100.0 if conf_result else 0.0

        return {
            "retrieved_docs": serialised,
            "all_scored_docs": serialised,
            "retrieval_confidence": conf_score,
        }


# ---------------------------------------------------------------------------
# Node: filter (applies RERANKER_SCORE_THRESHOLD)
# ---------------------------------------------------------------------------

def filter_node(state: AgentState) -> dict:
    """Filter scored docs by RERANKER_SCORE_THRESHOLD.

    Keeps only docs whose _reranker_score >= threshold.
    Unscored docs (graph expansion without score) are excluded.
    """
    with _agent_step("filter"):
        docs = state.get("all_scored_docs", [])
        if not docs:
            return {"retrieved_docs": []}

        threshold = settings.RERANKER_SCORE_THRESHOLD

        filtered = [
            d for d in docs
            if d.get("metadata", {}).get("_reranker_score", -float("inf")) >= threshold
        ]
        filtered.sort(
            key=lambda d: d.get("metadata", {}).get("_reranker_score", -float("inf")),
            reverse=True,
        )

        logger.info("[FILTER] threshold=%.2f | input=%d | passed=%d", threshold, len(docs), len(filtered))

        return {
            "retrieved_docs": filtered,
        }


# ---------------------------------------------------------------------------
# Node: adaptive_reranking (re-filter with lower threshold)
# ---------------------------------------------------------------------------

def adaptive_reranking_node(state: Any = None, db: Any = None) -> dict:
    """Adaptive reranking: re-filter all_scored_docs with lower threshold.

    Since all docs already have _reranker_score from the initial run,
    this just re-filters with ADAPTIVE_RETRIEVAL_RERANKER_THRESHOLD (-5.0).
    """
    with _agent_step("adaptive_reranking"):
        all_docs = state.get("all_scored_docs", [])
        if not all_docs:
            return {"adaptive_rerunning": False, "adaptive_reran": True}

        threshold = settings.ADAPTIVE_RETRIEVAL_RERANKER_THRESHOLD

        filtered = [
            d for d in all_docs
            if d.get("metadata", {}).get("_reranker_score", -float("inf")) >= threshold
        ]
        filtered.sort(
            key=lambda d: d.get("metadata", {}).get("_reranker_score", -float("inf")),
            reverse=True,
        )

        logger.info("[ADAPTIVE_FILTER] threshold=%.2f | input=%d | passed=%d", threshold, len(all_docs), len(filtered))

        conf_result = score_retrieval(filtered, {}) if filtered else None
        new_conf = conf_result.score / 100.0 if conf_result else 0.0

        return {
            "adaptive_rerunning": True,
            "adaptive_reran": True,
            "retrieved_docs": filtered,
            "retrieval_confidence": new_conf,
        }


# ---------------------------------------------------------------------------
# Helper: merge per-leg docs (deduplicate only, no ranking)
# ---------------------------------------------------------------------------

def _merge_docs(
    leg_docs: dict[str, list[dict]],
) -> list[dict]:
    """Merge docs from multiple retrieval legs into a single deduplicated list."""
    seen_hashes: set[str] = set()
    merged: list[dict] = []

    for docs in leg_docs.values():
        for doc in docs:
            h = doc.get("metadata", {}).get("content_hash") or content_hash(
                doc.get("page_content", "")
            )
            if h not in seen_hashes:
                seen_hashes.add(h)
                merged.append(doc)

    return merged





# ---------------------------------------------------------------------------
# Node: dense_retrieval
# ---------------------------------------------------------------------------

async def dense_retrieval_node(
    state: AgentState,
    db: Any,
    kb_ids: List[int] | None = None,
    org_id: int | None = None,
    file_markdown: str | None = None,
) -> dict:
    """Run the dense retrieval leg for the current subtask."""
    with _agent_step("dense_retrieval"):
        kb_ids = kb_ids or state.get("kb_ids", [])
        org_id = org_id if org_id is not None else state.get("org_id")
        file_markdown = file_markdown or state.get("file_markdown")

        query = state.get("rewritten_query", state.get("original_query", ""))
        datastore_ids = get_effective_datastore_ids(kb_ids, org_id, db) if db else []

        writer = _safe_writer()
        if writer:
            writer({"event": "progress", "phase": "dense_retrieval", "message": "Running dense vector retrieval..."})

        try:
            docs = dense_search_docs(query=query, kb_ids=kb_ids, datastore_ids=datastore_ids)
            failed = False
        except Exception as exc:
            logger.warning("[DENSE_RETRIEVAL] failed: %s", exc)
            docs = []
            failed = True

        serialised = [_serialise_doc(d) for d in docs]

        return {
            "dense_docs": serialised,
            "leg_results": {"dense": {"status": "failed" if failed else "ok", "count": len(serialised)}},
            "failed_legs": ["dense"] if failed else [],
            "leg_doc_counts": {"dense": len(serialised)},
        }


# ---------------------------------------------------------------------------
# Node: sparse_retrieval
# ---------------------------------------------------------------------------

async def sparse_retrieval_node(
    state: AgentState,
    db: Any,
    kb_ids: List[int] | None = None,
    org_id: int | None = None,
    file_markdown: str | None = None,
) -> dict:
    """Run the sparse retrieval leg for the current subtask."""
    with _agent_step("sparse_retrieval"):
        kb_ids = kb_ids or state.get("kb_ids", [])
        org_id = org_id if org_id is not None else state.get("org_id")
        file_markdown = file_markdown or state.get("file_markdown")

        query = state.get("rewritten_query", state.get("original_query", ""))
        datastore_ids = get_effective_datastore_ids(kb_ids, org_id, db) if db else []

        writer = _safe_writer()
        if writer:
            writer({"event": "progress", "phase": "sparse_retrieval", "message": "Running sparse keyword retrieval..."})

        try:
            docs = sparse_search_docs(query=query, kb_ids=kb_ids, datastore_ids=datastore_ids)
            failed = False
        except Exception as exc:
            logger.warning("[SPARSE_RETRIEVAL] failed: %s", exc)
            docs = []
            failed = True

        serialised = [_serialise_doc(d) for d in docs]

        return {
            "sparse_docs": serialised,
            "leg_results": {"sparse": {"status": "failed" if failed else "ok", "count": len(serialised)}},
            "failed_legs": ["sparse"] if failed else [],
            "leg_doc_counts": {"sparse": len(serialised)},
        }


# ---------------------------------------------------------------------------
# Node: exact_retrieval
# ---------------------------------------------------------------------------

async def exact_retrieval_node(
    state: AgentState,
    db: Any,
    kb_ids: List[int] | None = None,
    org_id: int | None = None,
    file_markdown: str | None = None,
) -> dict:
    """Run the exact (MySQL FTS) retrieval leg for the current subtask."""
    with _agent_step("exact_retrieval"):
        kb_ids = kb_ids or state.get("kb_ids", [])
        org_id = org_id if org_id is not None else state.get("org_id")
        file_markdown = file_markdown or state.get("file_markdown")

        query = state.get("rewritten_query", state.get("original_query", ""))
        datastore_ids = get_effective_datastore_ids(kb_ids, org_id, db) if db else []

        writer = _safe_writer()
        if writer:
            writer({"event": "progress", "phase": "exact_retrieval", "message": "Running exact full-text retrieval..."})

        try:
            docs = exact_search_docs(query=query, kb_ids=kb_ids, datastore_ids=datastore_ids, db=db)
            failed = False
        except Exception as exc:
            logger.warning("[EXACT_RETRIEVAL] failed: %s", exc)
            docs = []
            failed = True

        serialised = [_serialise_doc(d) for d in docs]

        return {
            "exact_docs": serialised,
            "leg_results": {"exact": {"status": "failed" if failed else "ok", "count": len(serialised)}},
            "failed_legs": ["exact"] if failed else [],
            "leg_doc_counts": {"exact": len(serialised)},
        }


# ---------------------------------------------------------------------------
# Node: merge
# ---------------------------------------------------------------------------

def merge_node(
    state: AgentState,
    file_markdown: str | None = None,
) -> dict:
    """Merge per-leg retrieval results into a single deduplicated doc list."""
    with _agent_step("merge"):
        file_markdown = file_markdown or state.get("file_markdown")

        leg_docs = {
            "dense": state.get("dense_docs", []),
            "sparse": state.get("sparse_docs", []),
            "exact": state.get("exact_docs", []),
        }

        merged = _merge_docs(leg_docs)

        # Stream the merged candidate docs as they become available.
        if merged:
            writer = _safe_writer()
            if writer:
                writer({"event": "context", "docs": merged})

        return {
            "retrieved_docs": merged,
        }


# ---------------------------------------------------------------------------
# Node: answer_evaluation
# ---------------------------------------------------------------------------

async def answer_evaluation_node(
    state: AgentState,
    llm: ChatOpenAI | None = None,
) -> dict:
    """Evaluate final answer quality and compute final confidence score."""
    with _agent_step("answer_evaluation"):
        from .evaluator import evaluate_answer

        answer = state.get("answer", "")
        query = state.get("original_query", "")
        docs = state.get("retrieved_docs", [])
        retrieval_conf = state.get("retrieval_confidence", 0.0)

        if not answer or not docs:
            return {
                "answer_evaluation_attempts": state.get("answer_evaluation_attempts", 0) + 1,
                "final_confidence": 0.0,
                "confidence_level": "none",
                "faithfulness": 0,
                "completeness": 0,
            }

        context_text = format_context_string(docs, state.get("file_markdown"))
        conf_level = (
            "very_high" if retrieval_conf > 0.8 else
            "high" if retrieval_conf > 0.6 else
            "medium" if retrieval_conf > 0.3 else "low"
        )

        try:
            evaluation = await evaluate_answer(
                query=query,
                answer=answer,
                context_preview=context_text,
                confidence_level=conf_level,
            )
            faithfulness = evaluation.faithfulness
            completeness = evaluation.completeness
            citation_quality = evaluation.citation_quality
            confidence_match = evaluation.confidence_match
            eval_flags = evaluation.flags
        except Exception as exc:
            logger.warning("[ANSWER_EVALUATION] failed: %s", exc)
            faithfulness = 50
            completeness = 50
            citation_quality = 50
            confidence_match = True
            eval_flags = ["Evaluation unavailable"]

        retrieval_score = retrieval_conf * 100
        final_confidence = (
            0.4 * retrieval_score +
            0.3 * faithfulness +
            0.3 * completeness
        )
        final_confidence = round(final_confidence / 100.0, 3)

        confidence_level = (
            "very_high" if final_confidence > 0.8 else
            "high" if final_confidence > 0.6 else
            "medium" if final_confidence > 0.3 else "low" if final_confidence > 0 else "none"
        )

        return {
            "answer_evaluation_attempts": state.get("answer_evaluation_attempts", 0) + 1,
            "final_confidence": final_confidence,
            "confidence_level": confidence_level,
            "faithfulness": faithfulness,
            "completeness": completeness,
            "citation_quality": citation_quality,
            "confidence_match": confidence_match,
            "evaluation_flags": eval_flags,
        }


