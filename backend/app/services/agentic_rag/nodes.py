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


# ---------------------------------------------------------------------------
# Negation extraction (regex-only, DE/EN/FR/IT — ported from retrievalagent)
# ---------------------------------------------------------------------------

_NEGATION_PATTERNS = [
    # German: "aber nicht (von) X", "nicht von X", "ohne X", "ausser X", "außer X", "keine X"
    re.compile(
        r"\baber\s+nicht\s+(?:(?:von|der|aus|mit|von\s+der)\s+)?([^\s,.;:!?]+(?:\s+[A-Za-z0-9][^\s,.;:!?]*){0,2})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bnicht\s+(?:von|der|von der|aus|mit)\s+([^\s,.;:!?]+(?:\s+[A-Za-z0-9][^\s,.;:!?]*){0,2})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bohne\s+([^\s,.;:!?]+(?:\s+[A-Za-z0-9][^\s,.;:!?]*){0,2})", re.IGNORECASE
    ),
    re.compile(
        r"\b(?:auss?er|außer)\s+([^\s,.;:!?]+(?:\s+[A-Za-z0-9][^\s,.;:!?]*){0,2})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bkeine?\s+([^\s,.;:!?]+(?:\s+[A-Za-z0-9][^\s,.;:!?]*){0,2})", re.IGNORECASE
    ),
    # English: "but not X", "not from X", "without X", "except X"
    re.compile(
        r"\bbut\s+not\s+(?:from\s+)?([^\s,.;:!?]+(?:\s+[A-Za-z0-9][^\s,.;:!?]*){0,2})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bnot\s+from\s+([^\s,.;:!?]+(?:\s+[A-Za-z0-9][^\s,.;:!?]*){0,2})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwithout\s+([^\s,.;:!?]+(?:\s+[A-Za-z0-9][^\s,.;:!?]*){0,2})", re.IGNORECASE
    ),
    re.compile(
        r"\bexcept\s+(?:for\s+)?([^\s,.;:!?]+(?:\s+[A-Za-z0-9][^\s,.;:!?]*){0,2})",
        re.IGNORECASE,
    ),
    # French: "sans X", "mais pas X"
    re.compile(
        r"\bsans\s+([^\s,.;:!?]+(?:\s+[A-Za-z0-9][^\s,.;:!?]*){0,2})", re.IGNORECASE
    ),
    re.compile(
        r"\bmais\s+pas\s+(?:de\s+)?([^\s,.;:!?]+(?:\s+[A-Za-z0-9][^\s,.;:!?]*){0,2})",
        re.IGNORECASE,
    ),
    # Italian: "ma non X", "senza X"
    re.compile(
        r"\bma\s+non\s+([^\s,.;:!?]+(?:\s+[A-Za-z0-9][^\s,.;:!?]*){0,2})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bsenza\s+([^\s,.;:!?]+(?:\s+[A-Za-z0-9][^\s,.;:!?]*){0,2})", re.IGNORECASE
    ),
]

_NEGATION_STOP = frozenset({
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einem", "einen",
    "the", "a", "an", "of", "from", "this", "that", "these", "those",
    "le", "la", "les", "un", "une", "du", "de", "il", "lo", "i", "gli", "una",
})

_NEGATION_HALT = frozenset({
    "aber", "und", "oder", "but", "and", "or", "mais", "ou", "et", "ma", "o", "e",
})


def _extract_negation_terms(query: str) -> list[str]:
    """Deterministic regex negation extractor (DE/EN/FR/IT).

    Returns excluded terms (1-3 words each, stopwords stripped).
    Zero latency, no LLM call.
    """
    if not query:
        return []
    raw: list[str] = []
    seen_lc: set[str] = set()
    for pat in _NEGATION_PATTERNS:
        for m in pat.finditer(query):
            term = (m.group(1) or "").strip(" ,.;:!?\"'()[]")
            if not term:
                continue
            tokens = term.split()
            while tokens and tokens[0].lower() in _NEGATION_STOP:
                tokens = tokens[1:]
            for i, tok in enumerate(tokens):
                if tok.lower() in _NEGATION_HALT:
                    tokens = tokens[:i]
                    break
            if not tokens:
                continue
            term = " ".join(tokens)
            key = term.lower()
            if key in seen_lc:
                continue
            seen_lc.add(key)
            raw.append(term)
    # Drop entries whose lowercased form has another extracted term as
    # a whole-word prefix — the shorter form is the conservative target.
    raw_sorted = sorted(raw, key=len)
    out: list[str] = []
    for term in raw_sorted:
        tl = term.lower()
        if any(tl != prev.lower() and tl.startswith(prev.lower() + " ") for prev in out):
            continue
        out.append(term)
    return out


def _content_contains_exclusion(text: str, value: str) -> bool:
    """Check if text contains an excluded term (case-insensitive, camelCase-aware)."""
    low = text.lower()
    val = value.lower()
    if val in low:
        return True
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value).lower()
    if spaced != val and spaced in low:
        return True
    words = val.split()
    if len(words) >= 3:
        prefix2 = " ".join(words[:2])
        if prefix2 in low:
            return True
    return False


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

        # Build KB profile text for intent extraction (folded into rewrite call).
        from app.services.agentic_rag.kb_profile import format_profile_summary
        kb_profile_text = format_profile_summary(state.get("kb_profile", {}))

        rewritten, provenance, query_intent = await resolve_retrieval_query(
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
            kb_profile_text=kb_profile_text,
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

        # Extract negated terms via deterministic regex (zero latency, no LLM).
        # Runs on the original user query, not the rewritten one — the user's
        # exclusion intent is in their wording, not in the pronoun-resolved form.
        excluded = _extract_negation_terms(original)
        if excluded:
            logger.debug("[rewrite_query] extracted excluded_terms: %s", excluded)

        result = {"rewritten_query": rewritten, "resolution_provenance": provenance,
                  "excluded_terms": excluded}
        if query_intent is not None:
            logger.debug("[rewrite_query] extracted query_intent: %s", query_intent)
            result["query_intent"] = query_intent
        return result


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

def _elbow_cut(sorted_docs: list[dict]) -> list[dict]:
    """Elbow cutoff: find the largest consecutive score drop and cut there.

    After sorting docs by reranker score (descending), find the position
    where the score drops most sharply between consecutive docs. Cut there
    to keep only the high-scoring cluster, adapting to the score
    distribution per query rather than using a fixed threshold.

    Also applies an absolute floor (RERANKER_SCORE_THRESHOLD) — no doc
    below the floor survives even if there's no sharp elbow.
    """
    if not sorted_docs:
        return []

    scores = [d.get("metadata", {}).get("_reranker_score", -float("inf")) for d in sorted_docs]

    # Need at least 3 docs to detect an elbow meaningfully
    if len(scores) < 3:
        # Just apply the floor
        floor = get_def("RERANKER_SCORE_THRESHOLD").default
        return [d for d in sorted_docs if d.get("metadata", {}).get("_reranker_score", -float("inf")) >= floor]

    # Find the largest consecutive score drop
    max_drop = -float("inf")
    elbow_idx = len(scores)  # default: keep all
    for i in range(1, len(scores)):
        drop = scores[i - 1] - scores[i]
        if drop > max_drop:
            max_drop = drop
            elbow_idx = i  # keep docs[0..i-1], cut from i onward

    # Apply the absolute floor as well — no doc below floor survives
    floor = get_def("RERANKER_SCORE_THRESHOLD").default
    result = []
    for i, d in enumerate(sorted_docs):
        if i >= elbow_idx:
            break
        score = d.get("metadata", {}).get("_reranker_score", -float("inf"))
        if score >= floor:
            result.append(d)

    logger.debug("[elbow_cut] %d docs → elbow at idx %d (drop=%.2f) → %d passed (floor=%.2f)",
                 len(sorted_docs), elbow_idx, max_drop, len(result), floor)
    return result


def filter_node(state: AgentState, threshold: Optional[float] = None, db: Any = None, org_id: Any = None) -> dict:
    """Filter scored docs by RERANKER_SCORE_THRESHOLD or adaptive elbow cutoff.

    When ELBOW_CUT_ENABLED is True, uses adaptive elbow cutoff: finds the
    largest consecutive score drop and cuts there, while still applying an
    absolute floor. This adapts to the score distribution per query.
    Otherwise, uses the traditional flat threshold.
    """
    with _agent_step("filter"):
        docs = state.get("all_scored_docs", [])
        if not docs:
            return {"retrieved_docs": []}

        # Sort by reranker score descending
        sorted_docs = sorted(
            docs,
            key=lambda d: d.get("metadata", {}).get("_reranker_score", -float("inf")),
            reverse=True,
        )

        if db is not None:
            from app.services.settings_service import get_setting
            elbow_enabled = get_setting(db, "ELBOW_CUT_ENABLED", org_id)
        else:
            elbow_enabled = get_def("ELBOW_CUT_ENABLED").default

        if elbow_enabled:
            filtered = _elbow_cut(sorted_docs)
        else:
            # Traditional flat threshold
            if threshold is None:
                if db is not None:
                    from app.services.settings_service import get_setting
                    threshold = get_setting(db, "RERANKER_SCORE_THRESHOLD", org_id)
                else:
                    threshold = get_def("RERANKER_SCORE_THRESHOLD").default
            filtered = [
                d for d in sorted_docs
                if d.get("metadata", {}).get("_reranker_score", -float("inf")) >= threshold
            ]

        logger.debug("[FILTER] elbow=%s | input=%d | passed=%d", elbow_enabled, len(docs), len(filtered))

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
    doc_ids: Optional[List[int]] = None,
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
            docs = dense_search_docs(query=query, kb_ids=kb_ids, datastore_ids=datastore_ids, db=db, org_id=org_id, min_score=min_score, doc_ids=doc_ids)
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
    doc_ids: Optional[List[int]] = None,
    extra_queries: Optional[List[str]] = None,
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
            docs = sparse_search_docs(query=query, kb_ids=kb_ids, datastore_ids=datastore_ids, db=db, org_id=org_id, min_score=min_score, doc_ids=doc_ids, extra_queries=extra_queries)
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
    doc_ids: Optional[List[int]] = None,
    extra_queries: Optional[List[str]] = None,
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
            docs = exact_search_docs(query=query, kb_ids=kb_ids, datastore_ids=datastore_ids, db=db, org_id=org_id, min_score=min_score, doc_ids=doc_ids, extra_queries=extra_queries)
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

def collapse_same_title_versions(docs: list[dict]) -> list[dict]:
    """Collapse chunks from same-title documents, keeping only the latest version.

    When multiple documents share the same title (e.g. "Weekly Update" uploaded
    on different dates), keep only chunks from the document with the latest
    created_at. This prevents older versions from polluting retrieval results
    for "latest" / "most recent" queries.

    Documents without a title or with unique titles are passed through unchanged.
    """
    if not docs:
        return docs

    # Group document_ids by title, tracking the latest created_at per title.
    doc_meta: dict[int, dict] = {}  # document_id -> {title, created_at}
    for d in docs:
        meta = d.get("metadata", {})
        doc_id = meta.get("document_id")
        if doc_id is None:
            continue
        title = (meta.get("title") or meta.get("_title") or "").strip()
        created = meta.get("_created_at") or meta.get("created_at") or ""
        if doc_id not in doc_meta:
            doc_meta[doc_id] = {"title": title, "created_at": created}
        else:
            # Keep the latest created_at if we see it on a later chunk.
            if created > doc_meta[doc_id]["created_at"]:
                doc_meta[doc_id]["created_at"] = created

    # For each title, find the document_id with the latest created_at.
    title_to_latest_doc: dict[str, int] = {}
    for doc_id, info in doc_meta.items():
        title = info["title"]
        if not title:
            continue
        existing = title_to_latest_doc.get(title)
        if existing is None:
            title_to_latest_doc[title] = doc_id
        else:
            existing_created = doc_meta[existing]["created_at"]
            if info["created_at"] > existing_created:
                title_to_latest_doc[title] = doc_id

    if not title_to_latest_doc:
        return docs

    # Build the set of document_ids to drop (older versions of same-title docs).
    dropped_doc_ids: set[int] = set()
    for title, latest_doc_id in title_to_latest_doc.items():
        for doc_id, info in doc_meta.items():
            if info["title"] == title and doc_id != latest_doc_id:
                dropped_doc_ids.add(doc_id)

    if not dropped_doc_ids:
        return docs

    result = [d for d in docs if d.get("metadata", {}).get("document_id") not in dropped_doc_ids]
    logger.debug(
        "[collapse_same_title] %d titles with multiple versions | dropped %d doc_ids | %d → %d chunks",
        len(title_to_latest_doc), len(dropped_doc_ids), len(docs), len(result),
    )
    return result


def _rrf_fuse_legs(all_docs: list[dict], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion across retrieval legs.

    Each doc's metadata contains ``_legs`` (list of legs that found it) and
    ``_leg_rank`` (rank within the first leg that found it). We use the
    per-leg rank to compute RRF scores across legs, then sort by fused score.

    Docs from a single leg get a score from that leg's rank only.
    Docs found by multiple legs get the sum of reciprocal ranks — naturally
    boosting consensus hits.

    Returns the docs sorted by fused RRF score (descending).
    """
    if not all_docs:
        return all_docs

    from app.services.infrastructure import content_hash as _ch

    # Build per-leg ranked lists from _legs metadata
    leg_docs: dict[str, list[dict]] = {}
    for doc in all_docs:
        meta = doc.get("metadata", {})
        legs = meta.get("_legs", [])
        for leg in legs:
            leg_docs.setdefault(leg, []).append(doc)

    # Sort each leg's docs by _leg_rank
    for leg in leg_docs:
        leg_docs[leg].sort(key=lambda d: d.get("metadata", {}).get("_leg_rank", 9999))

    # Compute RRF scores
    scores: dict[str, float] = {}
    for leg, docs in leg_docs.items():
        for rank, doc in enumerate(docs):
            meta = doc.get("metadata", {})
            h = meta.get("content_hash") or _ch(doc.get("page_content", ""))
            scores[h] = scores.get(h, 0.0) + 1.0 / (k + rank)

    # Sort all_docs by their fused score
    def _score(doc: dict) -> float:
        meta = doc.get("metadata", {})
        h = meta.get("content_hash") or _ch(doc.get("page_content", ""))
        return scores.get(h, 0.0)

    return sorted(all_docs, key=_score, reverse=True)


def _bow_jaccard(text_a: str, text_b: str) -> float:
    """Bag-of-words Jaccard similarity between two text snippets.

    Tokenizes on whitespace, lowercases. Fast and embedding-free.
    """
    set_a = set(text_a.lower().split())
    set_b = set(text_b.lower().split())
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def _mmr_diverse(docs: list[dict], lam: float, max_results: int | None = None) -> list[dict]:
    """Maximal Marginal Relevance diversification.

    Greedily selects docs that are both relevant (high RRF/reranker score)
    and diverse (low lexical overlap with already-selected docs).

    score = lam * relevance - (1 - lam) * max_sim_to_selected

    ``lam`` controls the relevance/diversity tradeoff:
      1.0 = pure relevance (no diversification)
      0.7 = balanced (default from retrievalagent)
      0.0 = pure diversity

    Uses bag-of-words Jaccard for similarity (no embeddings needed).
    """
    if not docs or lam >= 1.0:
        return docs

    n = max_results or len(docs)
    selected: list[dict] = []
    selected_texts: list[str] = []
    remaining = list(docs)

    # Use RRF rank as relevance proxy (position in the input list,
    # which is already sorted by RRF score from _rrf_fuse_legs).
    max_idx = len(remaining) - 1

    while remaining and len(selected) < n:
        best_score = -float("inf")
        best_idx = 0
        for i, doc in enumerate(remaining):
            # Relevance: normalized inverse rank
            relevance = 1.0 - (len(selected) + i) / (max_idx + 1) if max_idx > 0 else 1.0
            # Diversity penalty: max similarity to any selected doc
            text = doc.get("page_content", "")[:500]
            max_sim = 0.0
            for sel_text in selected_texts:
                sim = _bow_jaccard(text, sel_text)
                if sim > max_sim:
                    max_sim = sim
            mmr_score = lam * relevance - (1.0 - lam) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        selected_texts.append(chosen.get("page_content", "")[:500])

    return selected


def merge_node(
    state: AgentState,
    file_markdown: str | None = None,
    db: Any = None,
    org_id: int | None = None,
) -> dict:
    """Merge per-leg retrieval results into a single deduplicated doc list.

    Four-stage processing:
      1. Exact content_hash dedup (recency-aware: latest modified_at wins).
      2. RRF fusion across legs (boosts docs found by multiple legs).
      3. Semantic dedup (>threshold cosine similarity, keep latest).
      4. Same-title version collapse (keep only latest created_at per title).
    """
    from app.services.settings_service import get_setting

    with _agent_step("merge"):
        file_markdown = file_markdown or state.get("file_markdown")

        all_docs: list[dict] = []
        for leg in ("dense_docs", "sparse_docs", "exact_docs"):
            all_docs.extend(state.get(leg, []))

        # Stage 1: exact content_hash dedup (recency-aware)
        merged = dedup_by_content_hash(all_docs)

        # Stage 2: RRF fusion across legs (before semantic dedup so
        # consensus-ranked docs survive even if semantically similar)
        rrf_enabled = get_setting(db, "RRF_FUSION_ENABLED", org_id) if db else True
        if rrf_enabled and len(merged) > 1:
            merged = _rrf_fuse_legs(merged)

        # Stage 3: semantic dedup (cosine > threshold, keep latest)
        threshold = get_setting(db, "DEDUP_SEMANTIC_THRESHOLD", org_id) if db else 0.95
        if threshold < 1.0 and len(merged) > 1:
            merged = semantic_dedup(merged, threshold)

        # Stage 4: collapse same-title document versions (keep latest)
        collapse_enabled = get_setting(db, "COLLAPSE_SAME_TITLE_VERSIONS", org_id) if db else True
        if collapse_enabled:
            merged = collapse_same_title_versions(merged)

        # Stage 5: MMR diversity — reduce redundant context by penalizing
        # lexical similarity. Uses bag-of-words Jaccard (no embeddings needed).
        # lambda=1.0 = pure relevance, 0.0 = pure diversity.
        mmr_lambda = get_setting(db, "MERGE_MMR_LAMBDA", org_id) if db else 1.0
        if mmr_lambda < 1.0 and len(merged) > 2:
            merged = _mmr_diverse(merged, mmr_lambda)

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
        all_docs = state.get("retrieved_docs", [])
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

        # Use only cited docs for evaluation, not the full retrieved set.
        # The evaluator checks faithfulness against the evidence the answer
        # actually cites — feeding 50 uncited chunks just inflates the prompt
        # and slows the LLM call without improving the assessment.
        cited_indices = state.get("cited_doc_indices", [])
        if cited_indices:
            docs = [all_docs[i - 1] for i in cited_indices if 0 < i <= len(all_docs)]
        else:
            docs = all_docs

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


