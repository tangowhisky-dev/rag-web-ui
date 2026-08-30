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
from app.core.settings_registry import get_def

from app.services.infrastructure import content_hash
from app.services.infrastructure.utils import _serialise_doc
from app.services.retrieval import score_retrieval, rerank
from app.services.retrieval import (
    get_effective_datastore_ids,
    dense_search_docs,
    sparse_search_docs,
    exact_search_docs,
    dedup_by_content_hash,
    semantic_dedup,
)
from app.services.retrieval.reranker import _get_cross_encoder

from .graph_state import AgentState
from .prompts import EVALUATION_SYSTEM_PROMPT
from .redis_memory import get_redis_memory
from .token_budget import ContextBudget, count_tokens
from .utils import format_context_string

logger = logging.getLogger(__name__)


def select_recent_history(messages: list, max_pairs: int | None = None, db: Any = None, org_id: Any = None) -> list:
    """Return up to ``max_pairs`` of recent user/assistant turns.

    The last message is assumed to be the current user query and is excluded.
    Assistant turns are not truncated: the resolver, think and finalize nodes
    need the full prior answer to resolve references and compare approaches.
    Truncating here loses facts that cannot be recovered downstream.
    """
    from langchain_core.messages import HumanMessage, AIMessage

    if max_pairs is None:
        from app.services.settings_service import get_setting
        max_pairs = get_setting(db, "AGENT_HISTORY_PAIRS", org_id) if db is not None else get_def("AGENT_HISTORY_PAIRS").default

    history: list = []
    # Skip the final message (current query).
    for m in messages[:-1]:
        if isinstance(m, HumanMessage):
            history.append(HumanMessage(content=m.content))
        elif isinstance(m, AIMessage):
            history.append(AIMessage(content=m.content))
    # Keep the most recent max_pairs * 2 messages.
    return history[-(max_pairs * 2):]


def history_to_text(messages: list) -> str:
    """Render selected history messages as a ``User:``/``Assistant:`` block."""
    lines = []
    for msg in messages:
        role = "User" if getattr(msg, "type", "") == "human" else "Assistant"
        lines.append(f"  {role}: {msg.content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Conversation text helper (shared by the single compaction implementation)
# ---------------------------------------------------------------------------

def _messages_to_conversation_text(messages: list) -> str:
    """Convert LangChain messages to readable conversation text for summarization.

    No per-turn truncation: the summarizer cannot preserve facts that were
    removed before it ever saw them.
    """
    parts = []
    for m in messages:
        if isinstance(m, HumanMessage):
            parts.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            parts.append(f"Assistant: {m.content}")
    return "\n\n".join(parts)


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
    except (RuntimeError, KeyError):
        return None


# ── Answer Generation ──────────────────────────────────────────────────────


def _get_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    streaming: bool = False,
) -> ChatOpenAI:
    """Build a ChatOpenAI from explicit overrides or app-level settings.

    Prefer build_chat_llm() / get_org_llm() for per-org resolution.
    This fallback path is used when org context is unavailable (e.g. compaction
    fallback in agent_graph.py).
    """
    # Resolve from app-level settings when not explicitly provided
    if model_name is None or api_base is None or api_key is None:
        from app.services.settings_service import get_setting
        from app.db.session import SessionLocal
        _db = SessionLocal()
        try:
            if model_name is None:
                model_name = get_setting(_db, "QUERY_MODEL", None) or get_setting(_db, "OPENAI_MODEL", None)
            if api_base is None:
                api_base = get_setting(_db, "OPENAI_API_BASE", None)
            if api_key is None:
                api_key = get_setting(_db, "OPENAI_API_KEY", None)
        finally:
            _db.close()
    # Local servers (LM Studio, Ollama) don't require a key, but the OpenAI
    # client rejects None/empty — supply a placeholder when unset.
    if not api_key:
        api_key = "not-required"
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_base=api_base,
        openai_api_key=api_key,
        streaming=streaming,
    )


# ---------------------------------------------------------------------------
# Node: rewrite_query
# ---------------------------------------------------------------------------

def _collect_provenance_sources(
    state: AgentState,
    recent_history: list,
    clarification: str,
) -> tuple[list[str], Any]:
    """Build the list of provenance sources a resolver may draw terms from."""
    lao = state.get("last_answer_object")
    provenance_sources: list[str] = [history_to_text(recent_history)]
    if clarification:
        provenance_sources.append(clarification)
    if state.get("compaction_summary"):
        provenance_sources.append(str(state["compaction_summary"]))
    if lao is not None:
        summary = getattr(lao, "summary", "") or ""
        key_points = getattr(lao, "key_points", None) or []
        provenance_sources.append(summary)
        provenance_sources.extend(str(k) for k in key_points)
    for doc in state.get("recalled_memories", []) or []:
        if isinstance(doc, dict):
            provenance_sources.append(str(doc.get("page_content", "")))
    return provenance_sources, lao


def _lookup_cited_titles(lao: Any, db: Any) -> list[str]:
    """Look up document titles from the last answer's citations."""
    if lao is None or db is None:
        return []
    try:
        from app.models.knowledge import Document
        cited_doc_ids = {c.document_id for c in (getattr(lao, "citations", None) or [])}
        if cited_doc_ids:
            docs = db.query(Document).filter(Document.id.in_(cited_doc_ids)).all()
            return [d.title for d in docs if d.title]
    except Exception:
        pass
    return []


async def rewrite_query_node(
    state: AgentState,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    query_model: Optional[str] = None,
    db: Any = None,
    org_id: Any = None,
) -> dict:
    """Resolve the user's message into a standalone *retrieval* query.

    The rewrite is conditional and provenance-bound:
    - Self-contained messages pass through byte-for-byte (no LLM call).
    - Otherwise the resolver may only introduce terms that appear in the
      expanded query, the recent verbatim turns, the compaction summary,
      or the previous answer object. A rewrite that invents terms is
      rejected and the expanded query is used.

    Uses ``expanded_query`` (abbreviation-expanded original) as resolver
    input so the LLM can see expanded forms during pronoun resolution.
    Falls back to ``original_query`` if expansion was not run.
    """
    from .utils import resolve_retrieval_query

    with _agent_step("rewrite_query"):
        messages = state.get("messages", [])
        original = state.get("original_query", "")
        query = state.get("expanded_query", "") or original
        recent_history = select_recent_history(messages, db=db, org_id=org_id)

        # Glossary was built once by expand_query_node — reuse it.
        glossary = state.get("abbreviation_glossary", "")

        # A clarification answer is part of the request, not a new turn:
        # fold it into the text sent to the resolver.
        clarification = (state.get("clarification_response") or "").strip()
        resolver_input = query
        if clarification:
            resolver_input = f"{query}\n\n[User clarification: {clarification}]"

        provenance_sources, lao = _collect_provenance_sources(state, recent_history, clarification)

        retrieved_titles = _lookup_cited_titles(lao, db)
        if retrieved_titles:
            provenance_sources.extend(retrieved_titles)

        rewritten, provenance = await resolve_retrieval_query(
            query=resolver_input,
            original_query=query,
            recent_history=recent_history,
            provenance_sources=provenance_sources,
            api_base=api_base,
            query_model=query_model,
            openai_api_key=api_key,
            openai_api_base=api_base,
            glossary=glossary,
            retrieved_titles=retrieved_titles,
        )

        if provenance.get("reason") == "provenance_rejected":
            logger.warning(
                "[rewrite_query] rejected rewrite %r — unsupported terms %s; using original query",
                provenance.get("rejected_query"), provenance.get("unsupported_terms"),
            )

        # Stream the rewritten query so the UI can display it immediately.
        writer = _safe_writer()
        if writer:
            writer({"event": "rewritten_query", "query": rewritten})

        return {"rewritten_query": rewritten, "resolution_provenance": provenance}


# ---------------------------------------------------------------------------
# Node: abbreviation expansion (runs before rewrite)
# ---------------------------------------------------------------------------

def expand_query_node(
    state: AgentState,
    db: Any = None,
    org_id: Any = None,
) -> dict:
    """Expand the original query with bidirectional abbreviation suffix expansion.

    Runs BEFORE rewrite_query_node so the LLM rewriter can see expanded forms
    during pronoun/reference resolution. The expanded query is also used for
    dense/sparse/exact retrieval. The reranker uses the rewritten query (not
    expanded) to preserve user intent.
    """
    from app.services.abbreviation_service import build_lookup, expand_query_suffix, build_glossary

    original = state.get("original_query", "")
    org_id = org_id if org_id is not None else state.get("org_id")

    glossary = ""
    try:
        abbr_lookup = build_lookup(db, org_id) if db else None
        if abbr_lookup and not abbr_lookup.is_empty:
            expanded = expand_query_suffix(original, abbr_lookup)
            glossary = build_glossary(original, abbr_lookup)
        else:
            expanded = original
    except Exception as exc:
        logger.warning("[ABBREV_EXPAND] failed: %s", exc)
        expanded = original

    # Stream the expanded query so the UI can show it as a tooltip on the user bubble.
    # Only emit when something was actually expanded (differs from original).
    if expanded != original:
        writer = _safe_writer()
        if writer:
            writer({"event": "expanded_query", "query": expanded})

    return {"expanded_query": expanded, "abbreviation_glossary": glossary}


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

            datastore_ids = get_effective_datastore_ids(kb_ids, org_id, db) if db else []
            lc_docs = [
                LangchainDocument(page_content=d.get("page_content", ""), metadata=d.get("metadata", {}))
                for d in docs
            ]
            loop = asyncio.get_event_loop()
            expanded = await loop.run_in_executor(None, lambda: expand_docs_via_graph(lc_docs, kb_ids, None, None, datastore_ids))
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

def filter_node(state: AgentState, threshold: Optional[float] = None, db: Any = None, org_id: Any = None) -> dict:
    """Filter scored docs by RERANKER_SCORE_THRESHOLD (or an override).

    Keeps only docs whose _reranker_score >= threshold.
    Unscored docs (graph expansion without score) are excluded.
    """
    with _agent_step("filter"):
        docs = state.get("all_scored_docs", [])
        if not docs:
            return {"retrieved_docs": []}

        if threshold is None:
            if db is not None:
                from app.services.settings_service import get_setting
                threshold = get_setting(db, "RERANKER_SCORE_THRESHOLD", org_id)
            else:
                threshold = get_def("RERANKER_SCORE_THRESHOLD").default

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
# Helper: fetch modified_at for documents and attach to serialized docs
# ---------------------------------------------------------------------------

def _enrich_with_modified_at(docs: list[dict], db: Any) -> None:
    """Fetch COALESCE(modified_at, created_at) for unique document_ids and
    store as ``_modified_at`` (ISO string) in each doc's metadata.

    Skips docs that already have ``_modified_at`` (e.g. exact leg gets it
    from the SQL JOIN). Uses a single indexed query.
    """
    from app.models.knowledge import Document
    from sqlalchemy import func

    needed: set[int] = set()
    for doc in docs:
        meta = doc.get("metadata", {})
        if meta.get("_modified_at"):
            continue
        did = meta.get("document_id")
        if did is not None:
            needed.add(int(did))

    if not needed:
        return

    rows = db.query(
        Document.id,
        func.coalesce(Document.modified_at, Document.created_at),
    ).filter(Document.id.in_(needed)).all()

    mtime_map: dict[int, str] = {}
    for row in rows:
        if row[1] is not None:
            mtime_map[row[0]] = row[1].isoformat() if hasattr(row[1], "isoformat") else str(row[1])

    for doc in docs:
        meta = doc.get("metadata", {})
        if meta.get("_modified_at"):
            continue
        did = meta.get("document_id")
        if did is not None and int(did) in mtime_map:
            meta["_modified_at"] = mtime_map[int(did)]





# ---------------------------------------------------------------------------
# Node: dense_retrieval
# ---------------------------------------------------------------------------

async def dense_retrieval_node(
    state: AgentState,
    db: Any,
    kb_ids: List[int] | None = None,
    org_id: int | None = None,
    file_markdown: str | None = None,
    min_score: Optional[float] = None,
) -> dict:
    """Run the dense retrieval leg for the current subtask."""
    with _agent_step("dense_retrieval"):
        kb_ids = kb_ids or state.get("kb_ids", [])
        org_id = org_id if org_id is not None else state.get("org_id")
        file_markdown = file_markdown or state.get("file_markdown")

        query = state.get("expanded_query", state.get("rewritten_query", state.get("original_query", "")))
        datastore_ids = get_effective_datastore_ids(kb_ids, org_id, db) if db else []

        writer = _safe_writer()
        if writer:
            writer({"event": "progress", "phase": "dense_retrieval", "message": "Running dense vector retrieval..."})

        try:
            docs = dense_search_docs(query=query, kb_ids=kb_ids, datastore_ids=datastore_ids, db=db, org_id=org_id, min_score=min_score)
            failed = False
        except Exception as exc:
            logger.warning("[DENSE_RETRIEVAL] failed: %s", exc)
            docs = []
            failed = True

        serialised = [_serialise_doc(d) for d in docs]
        _enrich_with_modified_at(serialised, db)
        serialised = dedup_by_content_hash(serialised)

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
    min_score: Optional[float] = None,
) -> dict:
    """Run the sparse retrieval leg for the current subtask."""
    with _agent_step("sparse_retrieval"):
        kb_ids = kb_ids or state.get("kb_ids", [])
        org_id = org_id if org_id is not None else state.get("org_id")
        file_markdown = file_markdown or state.get("file_markdown")

        query = state.get("expanded_query", state.get("rewritten_query", state.get("original_query", "")))
        datastore_ids = get_effective_datastore_ids(kb_ids, org_id, db) if db else []

        writer = _safe_writer()
        if writer:
            writer({"event": "progress", "phase": "sparse_retrieval", "message": "Running sparse keyword retrieval..."})

        try:
            docs = sparse_search_docs(query=query, kb_ids=kb_ids, datastore_ids=datastore_ids, db=db, org_id=org_id, min_score=min_score)
            failed = False
        except Exception as exc:
            logger.warning("[SPARSE_RETRIEVAL] failed: %s", exc)
            docs = []
            failed = True

        serialised = [_serialise_doc(d) for d in docs]
        _enrich_with_modified_at(serialised, db)
        serialised = dedup_by_content_hash(serialised)

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
    min_score: Optional[float] = None,
) -> dict:
    """Run the exact (MySQL FTS) retrieval leg for the current subtask."""
    with _agent_step("exact_retrieval"):
        kb_ids = kb_ids or state.get("kb_ids", [])
        org_id = org_id if org_id is not None else state.get("org_id")
        file_markdown = file_markdown or state.get("file_markdown")

        query = state.get("expanded_query", state.get("rewritten_query", state.get("original_query", "")))
        datastore_ids = get_effective_datastore_ids(kb_ids, org_id, db) if db else []

        writer = _safe_writer()
        if writer:
            writer({"event": "progress", "phase": "exact_retrieval", "message": "Running exact full-text retrieval..."})

        try:
            docs = exact_search_docs(query=query, kb_ids=kb_ids, datastore_ids=datastore_ids, db=db, org_id=org_id, min_score=min_score)
            failed = False
        except Exception as exc:
            logger.warning("[EXACT_RETRIEVAL] failed: %s", exc)
            docs = []
            failed = True

        serialised = [_serialise_doc(d) for d in docs]
        # Exact leg already has _modified_at from the SQL JOIN — no fetch needed.
        serialised = dedup_by_content_hash(serialised)

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
    db: Any = None,
    org_id: int | None = None,
) -> dict:
    """Merge per-leg retrieval results into a single deduplicated doc list.

    Two-stage dedup:
      1. Exact content_hash dedup (recency-aware: latest modified_at wins).
      2. Semantic dedup (>threshold cosine similarity, keep latest).
    """
    from app.services.settings_service import get_setting

    with _agent_step("merge"):
        file_markdown = file_markdown or state.get("file_markdown")

        all_docs: list[dict] = []
        for leg in ("dense_docs", "sparse_docs", "exact_docs"):
            all_docs.extend(state.get(leg, []))

        # Stage 1: exact content_hash dedup (recency-aware)
        merged = dedup_by_content_hash(all_docs)

        # Stage 2: semantic dedup (cosine > threshold, keep latest)
        threshold = get_setting(db, "DEDUP_SEMANTIC_THRESHOLD", org_id) if db else 0.95
        if threshold < 1.0 and len(merged) > 1:
            merged = semantic_dedup(merged, threshold)

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

def _retrieval_confidence_level(retrieval_conf: float) -> str:
    if retrieval_conf > 0.8:
        return "very_high"
    if retrieval_conf > 0.6:
        return "high"
    if retrieval_conf > 0.3:
        return "medium"
    return "low"


def _resolve_eval_kwargs(ctx: Any) -> dict:
    if ctx is None:
        return {}
    try:
        from app.services.agentic_rag.llm_factory import get_org_llm
        query_cfg = get_org_llm(ctx.org_id, ctx.db, role="query")
        return {
            "api_base": query_cfg["api_base"],
            "api_key": query_cfg["api_key"],
            "query_model": query_cfg["model_name"],
        }
    except Exception:
        return {}


def _final_confidence_level(final_confidence: float) -> str:
    if final_confidence > 0.8:
        return "very_high"
    if final_confidence > 0.6:
        return "high"
    if final_confidence > 0.3:
        return "medium"
    if final_confidence > 0:
        return "low"
    return "none"


async def answer_evaluation_node(
    state: AgentState,
    llm: ChatOpenAI | None = None,
    ctx: Any = None,
) -> dict:
    """Evaluate final answer quality and compute final confidence score."""
    with _agent_step("answer_evaluation"):
        from .evaluator import evaluate_answer

        answer = state.get("answer", "")
        # Evaluate against the user's exact request, not the retrieval rewrite:
        # completeness is a property of what was asked, not what was searched.
        query = state.get("original_query", "") or state.get("rewritten_query", "")
        docs = state.get("retrieved_docs", [])
        retrieval_conf = state.get("retrieval_confidence", 0.0)

        if not answer:
            return {
                "answer_evaluation_attempts": state.get("answer_evaluation_attempts", 0) + 1,
                "final_confidence": 0.0,
                "confidence_level": "none",
                "faithfulness": 0,
                "completeness": 0,
                "retrieval_score": 0,
            }

        _db = ctx.db if ctx is not None else None
        _org_id = ctx.org_id if ctx is not None else None
        context_text = format_context_string(docs, state.get("file_markdown"), db=_db, org_id=_org_id, query_glossary=state.get("abbreviation_glossary", ""))
        conf_level = _retrieval_confidence_level(retrieval_conf)

        eval_kwargs = _resolve_eval_kwargs(ctx)

        try:
            evaluation = await evaluate_answer(
                query=query,
                answer=answer,
                context_preview=context_text,
                confidence_level=conf_level,
                **eval_kwargs,
            )
            faithfulness = evaluation.faithfulness
            completeness = evaluation.completeness
            confidence_match = evaluation.confidence_match
            eval_flags = evaluation.flags
        except Exception as exc:
            logger.warning("[ANSWER_EVALUATION] failed: %s", exc)
            faithfulness = 50
            completeness = 50
            confidence_match = True
            eval_flags = ["Evaluation unavailable"]

        retrieval_score = retrieval_conf * 100
        final_confidence = (
            0.4 * retrieval_score +
            0.3 * faithfulness +
            0.3 * completeness
        )
        final_confidence = round(final_confidence / 100.0, 3)

        confidence_level = _final_confidence_level(final_confidence)

        return {
            "answer_evaluation_attempts": state.get("answer_evaluation_attempts", 0) + 1,
            "final_confidence": final_confidence,
            "confidence_level": confidence_level,
            "faithfulness": faithfulness,
            "completeness": completeness,
            "retrieval_score": int(retrieval_score),
            "confidence_match": confidence_match,
            "evaluation_flags": eval_flags,
        }


