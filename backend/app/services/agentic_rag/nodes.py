"""LangGraph node implementations for the agentic RAG pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

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
from app.services.prompts.loader import append_chart_instructions
from app.services.retrieval.reranker import _get_cross_encoder

from .graph_state import AgentState
from .schemas import QueryAnalysis
from .utils import estimate_messages_tokens, strip_reasoning_tags, format_context_string

logger = logging.getLogger(__name__)

_ANSWER_SYSTEM_PROMPT = append_chart_instructions("""\
You are a helpful assistant. Answer the user's question using ONLY the provided context.
If the context is insufficient, say so clearly.

FORMATTING RULES:
- Use ### headers to divide multi-part answers (e.g., "### 1. Definition", "### 2. How It Works").
- Use numbered lists for sequential steps or algorithms.
- Use bullet points with **bold terms** for features, attributes, or comparisons.
- Use inline code for variable names, identifiers, and technical terms (e.g., `wait()`, `Available[j]`).
- For simple single-concept questions, plain prose is fine - do not force structure.

CITATION RULES:
When you use information from a chunk, cite it as a markdown link with ONLY the number as both text and href:
  Example: process scheduling [1](1) involves saving the CPU state [2](2).
The number must match the [KB-N] label of the chunk you are citing.
Do NOT invent citations. Only cite chunks you actually used.

IMPORTANT: Do NOT repeat the user's question in your answer. Just provide the answer directly.
""")

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
    """Rewrite query using chat history."""
    from .utils import rewrite_query as _rewrite_query

    messages = state.get("messages", [])
    query = state.get("original_query", "")
    rewritten = _rewrite_query(
        query=query,
        recent_history=messages,
        api_base=api_base,
        query_model=settings.effective_query_model,
        openai_api_key=settings.OPENAI_API_KEY,
        openai_api_base=settings.OPENAI_API_BASE,
    )
    return {"rewritten_query": rewritten}


# ---------------------------------------------------------------------------
# Node: classify_query (LLM-based classification)
# ---------------------------------------------------------------------------

async def classify_query_node(
    state: AgentState,
) -> dict:
    """Classify query using structured LLM output."""
    rewritten = state.get("rewritten_query", "")
    query = state.get("original_query", "")

    try:
        llm = _get_llm(streaming=False)
        llm_structured = llm.with_structured_output(QueryAnalysis)

        response = llm_structured.invoke([
            {"role": "system", "content": (
                "You are a query classifier. Analyze the user's question and respond with structured data.\n\n"
                "Rules:\n"
                "- is_clear: true if the question is clear and answerable from documents.\n"
                "- questions: list of self-contained questions extracted from the query (1 if simple, 2-5 if complex).\n"
                "- clarification_needed: explanation of missing info, or empty string if clear.\n"
                "Output ONLY a JSON object with keys: is_clear, questions, clarification_needed."
            )},
            {"role": "user", "content": rewritten},
        ])
        is_clear = getattr(response, "is_clear", True)
        questions = getattr(response, "questions", [rewritten]) or [rewritten]
    except Exception as exc:
        logger.warning("[CLASSIFY] structured classification failed: %s - using fallback", exc)
        is_clear = True
        questions = [rewritten]

    subtasks = questions if len(questions) > 1 else [rewritten]

    return {
        "question_is_clear": is_clear,
        "subtasks": subtasks,
        "is_complex": len(subtasks) > 1,
    }


# ---------------------------------------------------------------------------
# Node: request_clarification
# ---------------------------------------------------------------------------

def request_clarification_node(state: AgentState) -> dict:
    """Ask the user for clarification when the query is unclear."""
    pending = state.get("pending_query", "")
    clarifications = state.get("clarification_questions", [])

    if clarifications:
        context = "\n".join(f"{i+1}. {c}" for i, c in enumerate(clarifications))
        clarification_msg = (
            f"I need more information to answer your question about: '{pending}'\n\n"
            f"Please clarify:\n{context}"
        )
    else:
        clarification_msg = f"I need more information to understand your question: '{pending}'"

    return {
        "messages": [AIMessage(content=clarification_msg, name="clarification")],
    }


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
# Node: sufficiency_check (after filter)
# ---------------------------------------------------------------------------

def sufficiency_check_node(state: AgentState) -> dict:
    """Check if filtered docs are sufficient.

    Routes to adaptive_reranking if confidence is low or doc count is small.
    """
    conf_score = state.get("retrieval_confidence", 0.0)
    doc_count = len(state.get("retrieved_docs", []))
    leg_results = state.get("leg_results", {})

    confidence_met = conf_score > 0.3
    docs_met = doc_count >= 3

    sufficiency_met = confidence_met and docs_met

    if leg_results:
        dense_count = leg_results.get("dense", {}).get("count", 0)
        sparse_count = leg_results.get("sparse", {}).get("count", 0)
        exact_count = leg_results.get("exact", {}).get("count", 0)

        if dense_count > 0 or sparse_count > 0 or exact_count > 0:
            sufficiency_met = True

    needs_adaptive = not sufficiency_met

    if sufficiency_met:
        message = f"Retrieval sufficient: confidence={conf_score:.2f}, docs={doc_count}"
    else:
        message = f"Retrieval insufficient: confidence={conf_score:.2f}, docs={doc_count}, checking adaptive reranking"

    return {
        "sufficiency_met": sufficiency_met,
        "sufficiency_message": message,
        "needs_adaptive_reranking": needs_adaptive,
    }


# ---------------------------------------------------------------------------
# Node: adaptive_reranking (re-filter with lower threshold)
# ---------------------------------------------------------------------------

def adaptive_reranking_node(state: AgentState) -> dict:
    """Adaptive reranking: re-filter all_scored_docs with lower threshold.

    Since all docs already have _reranker_score from the initial run,
    this just re-filters with ADAPTIVE_RETRIEVAL_RERANKER_THRESHOLD (-5.0).
    """
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
# Node: generating (actual LLM answer generation)
# ---------------------------------------------------------------------------

async def generating_node(
    state: AgentState,
    llm: ChatOpenAI | None = None,
    api_base: Optional[str] = None,
) -> dict:
    """Generate answer using retrieved context."""
    llm_instance = llm or _get_llm(streaming=True)

    retrieved_docs = state.get("retrieved_docs", [])
    original_query = state.get("original_query", "")
    file_markdown = state.get("file_markdown")

    if not retrieved_docs:
        response = await llm_instance.ainvoke([
            {"role": "system", "content": (
                "You are a helpful assistant. The user asked a question but no "
                "relevant documents were found in the knowledge base. Answer from "
                "your general knowledge, but clearly state that no documents were found."
            )},
            {"role": "user", "content": original_query},
        ])
        answer = getattr(response, "content", "") or ""
        return {"answer": answer, "is_chart_query": False}

    context_text = format_context_string(retrieved_docs, file_markdown)

    messages = [
        {"role": "system", "content": _ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Query: {original_query}\n\nContext:\n{context_text}"},
    ]

    answer = ""
    async for chunk in llm_instance.astream(messages):
        chunk_text = getattr(chunk, "content", "") or ""
        if isinstance(chunk_text, list):
            chunk_text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in chunk_text
            )
        answer += chunk_text

    is_chart = bool(re.search(r"\b(chart|graph|plot|visuali[zs]|trend|distribution)\b", original_query.lower()))

    return {
        "answer": answer,
        "is_chart_query": is_chart,
        "thinking_chunks": [],
    }


# ---------------------------------------------------------------------------
# Node: chart_validation (conditional, if charts detected)
# ---------------------------------------------------------------------------

def chart_validation_node(state: AgentState) -> dict:
    """Validate chart-related queries and add chart context."""
    is_chart = state.get("is_chart_query", False)

    if not is_chart:
        return {"chart_validated": False, "chart_data": None}

    retries = state.get("chart_retries", 0)

    answer = state.get("answer", "")
    if not answer:
        retrieved_docs = state.get("retrieved_docs", [])
        has_enough_context = len(retrieved_docs) >= 2
        chart_data = {
            "is_chart_query": True,
            "valid": has_enough_context,
            "has_context": has_enough_context,
            "doc_count": len(retrieved_docs),
            "validation_message": (
                "Chart query validated — sufficient context available"
                if has_enough_context
                else "Chart query validated — limited context available, may need clarification"
            ),
        }
        return {"chart_validated": True, "chart_data": chart_data}

    valid, message = validate_echarts_json(answer)
    if valid:
        return {
            "chart_validated": True,
            "chart_data": {
                "valid": True,
                "validation_message": message,
            },
        }

    if retries < 3:
        return {
            "chart_validated": False,
            "chart_data": {"valid": False, "validation_message": message},
            "chart_retries": retries + 1,
        }

    return {
        "chart_validated": True,
        "chart_data": {"valid": False, "validation_message": message},
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
# Node: load_historical_memory
# ---------------------------------------------------------------------------

async def load_historical_memory_node(
    state: AgentState,
    db: Any,
) -> dict:
    """Load relevant past assistant messages as historical memory docs."""
    chat_id = state.get("chat_id")
    if not chat_id:
        return {"historical_memory_docs": []}

    from app.services.chat import retrieve_historical_memory

    try:
        docs = retrieve_historical_memory(
            chat_id=chat_id,
            query=state.get("original_query", ""),
            db=db,
            top_k=settings.HISTORICAL_MEMORY_TOP_K,
            score_threshold=settings.HISTORICAL_MEMORY_SCORE_THRESHOLD,
        )
    except Exception as exc:
        logger.warning("[LOAD_HISTORICAL_MEMORY] failed: %s", exc)
        docs = []

    return {"historical_memory_docs": docs}


# ---------------------------------------------------------------------------
# Node: summarize_history
# ---------------------------------------------------------------------------

async def summarize_history_node(
    state: AgentState,
    llm: ChatOpenAI | None = None,
) -> dict:
    """Reduce conversation history using rolling summaries."""
    messages = state.get("messages", [])
    plain_msgs = [m for m in messages if not getattr(m, "tool_calls", None)]
    keep_count = 4
    older = plain_msgs[:-keep_count] if len(plain_msgs) > keep_count else []
    if not older:
        return {}

    existing_summary = state.get("existing_summary", "").strip()
    conversation = f"Existing summary:\n{existing_summary or '(none)'}\n\n"
    conversation += "New messages:\n" + "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content[:200]}"
        for m in older
    )

    llm_instance = llm or _get_llm(streaming=False)
    response = await llm_instance.ainvoke([
        {"role": "system", "content": (
            "You are a conversation summarizer. Provide a concise summary of key facts, "
            "decisions, and context. Max 200 words."
        )},
        {"role": "user", "content": conversation},
    ])

    summary = response.content.strip() if hasattr(response, "content") else str(response)
    return {"existing_summary": summary}


# ---------------------------------------------------------------------------
# Node: rewrite_subtask_query
# ---------------------------------------------------------------------------

async def rewrite_subtask_query_node(
    state: AgentState,
    api_base: Optional[str] = None,
) -> dict:
    """Rewrite a subtask query into a self-contained search query."""
    from .utils import rewrite_query as _rewrite_query

    subtasks = state.get("subtasks", [])
    idx = state.get("current_subtask_index", 0)
    subtask = subtasks[idx] if 0 <= idx < len(subtasks) else state.get("original_query", "")

    rewritten = _rewrite_query(
        query=subtask,
        recent_history=[],
        api_base=api_base,
        query_model=settings.effective_query_model,
        openai_api_key=settings.OPENAI_API_KEY,
        openai_api_base=settings.OPENAI_API_BASE,
    )
    return {"rewritten_query": rewritten}


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
    kb_ids = kb_ids or state.get("kb_ids", [])
    org_id = org_id if org_id is not None else state.get("org_id")
    file_markdown = file_markdown or state.get("file_markdown")

    query = state.get("rewritten_query", state.get("original_query", ""))
    datastore_ids = get_effective_datastore_ids(kb_ids, org_id, db) if db else []

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
    kb_ids = kb_ids or state.get("kb_ids", [])
    org_id = org_id if org_id is not None else state.get("org_id")
    file_markdown = file_markdown or state.get("file_markdown")

    query = state.get("rewritten_query", state.get("original_query", ""))
    datastore_ids = get_effective_datastore_ids(kb_ids, org_id, db) if db else []

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
    kb_ids = kb_ids or state.get("kb_ids", [])
    org_id = org_id if org_id is not None else state.get("org_id")
    file_markdown = file_markdown or state.get("file_markdown")

    query = state.get("rewritten_query", state.get("original_query", ""))
    datastore_ids = get_effective_datastore_ids(kb_ids, org_id, db) if db else []

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
    file_markdown = file_markdown or state.get("file_markdown")

    leg_docs = {
        "dense": state.get("dense_docs", []),
        "sparse": state.get("sparse_docs", []),
        "exact": state.get("exact_docs", []),
    }

    merged = _merge_docs(leg_docs)

    return {
        "retrieved_docs": merged,
    }


# ---------------------------------------------------------------------------
# Node: collect_context
# ---------------------------------------------------------------------------

def collect_context_node(state: AgentState) -> dict:
    """Collect a subagent's retrieved context and add it to subtask_contexts."""
    return {
        "subtask_contexts": [{
            "question": state.get("original_query", ""),
            "rewritten_query": state.get("rewritten_query", ""),
            "retrieved_docs": state.get("retrieved_docs", []),
            "retrieved_contexts": state.get("retrieved_contexts", []),
            "retrieval_confidence": state.get("retrieval_confidence", 0.0),
            "leg_results": state.get("leg_results", {}),
            "failed_legs": state.get("failed_legs", []),
            "all_scored_docs": state.get("all_scored_docs", []),
        }],
    }


# ---------------------------------------------------------------------------
# Node: prepare_final_context
# ---------------------------------------------------------------------------

def prepare_final_context_node(state: AgentState) -> dict:
    """Aggregate all subtask contexts into the final retrieval state.

    Deduplicates docs by content_hash across subtasks so the final LLM context
    contains no duplicate chunks. Tracks how many duplicates were removed.
    """
    contexts = state.get("subtask_contexts", [])
    if not contexts:
        return {}

    all_docs: list[dict] = []
    all_scored: list[dict] = []
    all_contexts: list[str] = []
    confidences: list[float] = []

    seen_hashes: set[str] = set()
    doc_deduped = 0
    scored_deduped = 0

    for ctx in contexts:
        for doc in ctx.get("retrieved_docs", []):
            h = doc.get("metadata", {}).get("content_hash") or content_hash(
                doc.get("page_content", "")
            )
            if h not in seen_hashes:
                seen_hashes.add(h)
                all_docs.append(doc)
            else:
                doc_deduped += 1
        for doc in ctx.get("all_scored_docs", []):
            h = doc.get("metadata", {}).get("content_hash") or content_hash(
                doc.get("page_content", "")
            )
            if h not in seen_hashes:
                seen_hashes.add(h)
                all_scored.append(doc)
            else:
                scored_deduped += 1
        all_contexts.extend(ctx.get("retrieved_contexts", []))
        confidences.append(ctx.get("retrieval_confidence", 0.0))

    is_complex = len(contexts) > 1
    if is_complex and (doc_deduped or scored_deduped):
        logger.info(
            "[DEDUP] complex query: %d subtasks | "
            "retrieved_deduped=%d | scored_deduped=%d | "
            "final_retrieved=%d | final_scored=%d",
            len(contexts), doc_deduped, scored_deduped,
            len(all_docs), len(all_scored),
        )

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "retrieved_docs": all_docs,
        "all_scored_docs": all_scored,
        "retrieved_contexts": all_contexts,
        "retrieval_confidence": avg_conf,
        "is_complex": is_complex,
    }


# ---------------------------------------------------------------------------
# Node: finalize_answer
# ---------------------------------------------------------------------------

def finalize_answer_node(state: AgentState) -> dict:
    """Promote the generated answer to final_answer."""
    answer = state.get("answer", "")
    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)] if answer else [],
    }


# ---------------------------------------------------------------------------
# Node: answer_evaluation
# ---------------------------------------------------------------------------

async def answer_evaluation_node(
    state: AgentState,
    llm: ChatOpenAI | None = None,
) -> dict:
    """Evaluate final answer quality and compute final confidence score."""
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
            "needs_retry": False,
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
    except Exception as exc:
        logger.warning("[ANSWER_EVALUATION] failed: %s", exc)
        faithfulness = 50
        completeness = 50

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
        "needs_retry": False,
    }


# ---------------------------------------------------------------------------
# Helper: validate ECharts JSON
# ---------------------------------------------------------------------------

def validate_echarts_json(answer_text: str) -> tuple[bool, str]:
    """Validate an ECharts option JSON embedded in the answer text.

    Returns (valid, message).
    """
    import json

    text = answer_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        option = json.loads(text)
    except Exception as exc:
        return False, f"Invalid JSON: {exc}"

    if not isinstance(option, dict):
        return False, "ECharts option must be a JSON object"

    series = option.get("series")
    if not series:
        return False, "Missing required 'series' key"

    chart_type = (series[0].get("type") if isinstance(series, list) and series else None) or series.get("type")
    if not chart_type:
        return False, "Series type missing"

    cartesian_types = {"line", "bar", "scatter"}
    if chart_type in cartesian_types:
        if not option.get("xAxis") or not option.get("yAxis"):
            return False, "Cartesian charts require both xAxis and yAxis"

    return True, "ECharts JSON is valid"
