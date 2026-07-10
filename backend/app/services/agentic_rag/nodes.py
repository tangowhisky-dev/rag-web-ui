"""LangGraph node implementations for the agentic RAG pipeline."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings

from app.services.infrastructure.utils import _serialise_doc
from app.services.retrieval import score_retrieval, rerank
from app.services.retrieval import hybrid_search_with_legs, get_effective_datastore_ids
from app.services.prompts.loader import append_chart_instructions
from app.services.retrieval.reranker import _get_cross_encoder

from .graph_state import AgentState
from .schemas import QueryAnalysis
from .utils import estimate_messages_tokens, strip_reasoning_tags

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
    """Rewrite query using chat history.

    Delegates to the shared ``rewrite_query`` in utils.py.
    """
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
# Node: direct retrieval (simple path)
# ---------------------------------------------------------------------------

async def direct_retrieval_node(
    state: AgentState,
    db: Any,
    kb_ids: List[int] | None = None,
    org_id: int | None = None,
    file_markdown: str | None = None,
    use_dense: bool = True,
    use_sparse: bool = True,
    use_exact: bool = True,
    use_graph_rag: bool = False,
) -> dict:
    """Simple path: search + rerank. Returns state update with retrieved docs."""
    kb_ids = kb_ids or state.get("kb_ids", [])
    org_id = org_id if org_id is not None else state.get("org_id")
    file_markdown = file_markdown or state.get("file_markdown")

    rewritten = state.get("rewritten_query", state.get("original_query", ""))

    datastore_ids = get_effective_datastore_ids(kb_ids, org_id, db)

    retrieval_result = await hybrid_search_with_legs(
        query=rewritten,
        kb_ids=kb_ids,
        db=db,
        use_dense=use_dense,
        use_sparse=use_sparse,
        use_exact=use_exact,
        use_graph_rag=use_graph_rag,
        datastore_ids=datastore_ids,
        return_full_pool=True,
    )

    docs = retrieval_result.get("docs", [])
    retrieval_info = retrieval_result.get("retrieval_info", {})
    failed_legs = retrieval_info.get("failed_legs", [])
    legs = retrieval_info.get("legs", {})

    conf_result = score_retrieval(docs, retrieval_info) if docs else None
    conf_score = conf_result.score if conf_result else 0

    # Extract per-leg doc counts for sufficiency check
    leg_doc_counts = {leg: info.get("count", 0) for leg, info in legs.items()}

    from .utils import format_context_string

    serialised = [_serialise_doc(d) for d in docs]
    context_text = format_context_string(serialised, file_markdown)

    return {
        "retrieved_docs": serialised,
        "retrieved_contexts": [context_text],
        "retrieval_confidence": conf_score / 100.0 if conf_score else 0.0,
        "retrieval_iterations": 1,
        "leg_results": legs,
        "failed_legs": failed_legs,
        "leg_doc_counts": leg_doc_counts,
    }


# ---------------------------------------------------------------------------
# Node: orchestrator (subgraph entry point)
# ---------------------------------------------------------------------------

async def orchestrator_node(
    state: AgentState,
    llm: ChatOpenAI,
    api_base: Optional[str] = None,
) -> dict:
    """Agent subgraph orchestrator."""
    messages = state.get("messages", [])

    tool_call_count = sum(
        1 for m in messages if hasattr(m, "tool_calls") and m.tool_calls
    )
    iteration_count = state.get("retrieval_iterations", 0)

    if iteration_count >= 8 or tool_call_count >= 20:
        return {"_orchestrator_result": "fallback"}

    return {"_orchestrator_result": "generate", "current_subtask_index": 0}


# ---------------------------------------------------------------------------
# Node: collect_answer
# ---------------------------------------------------------------------------

def collect_answer_node(state: AgentState) -> dict:
    """Collect the answer from the agent subgraph."""
    answer = state.get("answer", "")
    return {
        "subtask_answers": [{"answer": answer}],
    }


# ---------------------------------------------------------------------------
# Node: synthesize
# ---------------------------------------------------------------------------

async def synthesize_node(
    state: AgentState,
    llm: ChatOpenAI | None = None,
    api_base: Optional[str] = None,
) -> dict:
    """Synthesize final answer from subtask answers or direct answer."""
    subtask_answers = state.get("subtask_answers", [])
    final_answer = ""

    if len(subtask_answers) > 1:
        synthesis_parts = []
        for i, sa in enumerate(subtask_answers):
            answer_text = sa.get("answer", "") if isinstance(sa, dict) else str(sa)
            synthesis_parts.append(f"### Answer {i+1}\n\n{answer_text}")
        combined = "\n\n---\n\n".join(synthesis_parts)
        final_answer = combined
    elif subtask_answers:
        first = subtask_answers[0]
        final_answer = first.get("answer", first) if isinstance(first, dict) else first
    else:
        final_answer = state.get("answer", "")

    return {
        "final_answer": final_answer,
    }


# ---------------------------------------------------------------------------
# Node: fallback response
# ---------------------------------------------------------------------------

def fallback_response_node(state: AgentState) -> dict:
    """Fallback when budget exceeded or retrieval fails completely."""
    question = state.get("rewritten_query", state.get("original_query", ""))
    return {
        "messages": [AIMessage(content=(
            f"I wasn't able to find sufficient information in the documents "
            f"to fully answer your question about '{question}'. "
            f"You might want to try rephrasing or providing more context."
        ), name="fallback")],
    }


# ---------------------------------------------------------------------------
# Node: compress context (between retrieval iterations)
# ---------------------------------------------------------------------------

def compress_context_node(state: AgentState) -> dict:
    """Compress accumulated retrieval context to free token budget."""
    retrieval_keys = set(state.get("retrieval_keys", set()) or set())

    for doc in state.get("retrieved_docs", []):
        source = doc.get("metadata", {}).get("source", "")
        if source:
            retrieval_keys.add(f"source:{source}")

    return {
        "retrieval_keys": list(retrieval_keys),
    }


# ---------------------------------------------------------------------------
# Node: sufficiency_check (between retrieval and graph expansion)
# ---------------------------------------------------------------------------

def sufficiency_check_node(state: AgentState) -> dict:
    """Check if retrieved docs are sufficient before graph expansion.

    Uses confidence score as primary metric, with fallback to doc count.
    Routes to graph_expansion if confidence is low or doc count is small.
    """
    conf_score = state.get("retrieval_confidence", 0.0)
    doc_count = len(state.get("retrieved_docs", []))
    leg_results = state.get("leg_results", {})

    # Primary: confidence score threshold
    # Medium confidence = some relevant docs found (30-55)
    # Low confidence = very few or marginal docs (0-30)
    confidence_met = conf_score > 0.3
    docs_met = doc_count >= 3  # At least 3 relevant chunks

    sufficiency_met = confidence_met and docs_met

    # Check leg-specific info for better routing decisions
    if leg_results:
        dense_count = leg_results.get("dense", {}).get("count", 0)
        sparse_count = leg_results.get("sparse", {}).get("count", 0)
        exact_count = leg_results.get("exact", {}).get("count", 0)

        # If any leg returned results, we likely have some signal
        if dense_count > 0 or sparse_count > 0 or exact_count > 0:
            sufficiency_met = True

    # Determine if we need graph expansion
    needs_graph = not sufficiency_met

    # Build sufficiency message for logging/debugging
    if sufficiency_met:
        message = f"Retrieval sufficient: confidence={conf_score:.2f}, docs={doc_count}"
    else:
        message = f"Retrieval insufficient: confidence={conf_score:.2f}, docs={doc_count}, checking graph expansion"

    return {
        "sufficiency_met": sufficiency_met,
        "sufficiency_message": message,
        "needs_graph_expansion": needs_graph,
    }


# ---------------------------------------------------------------------------
# Node: generating (actual LLM answer generation)
# ---------------------------------------------------------------------------

async def generating_node(
    state: AgentState,
    llm: ChatOpenAI | None = None,
    api_base: Optional[str] = None,
) -> dict:
    """Generate answer using retrieved context.

    This is the actual answer generation step that calls the LLM
    with the retrieved documents and query to produce the final response.
    """
    llm_instance = llm or _get_llm(streaming=False)

    retrieved_docs = state.get("retrieved_docs", [])
    original_query = state.get("original_query", "")
    file_markdown = state.get("file_markdown")

    if not retrieved_docs:
        # No docs retrieved — answer from intrinsic knowledge with disclaimer
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

    # Build context string from retrieved docs
    from .utils import format_context_string
    context_text = format_context_string(retrieved_docs, file_markdown)

    # Generate answer using the retrieved context
    messages = [
        {"role": "system", "content": _ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Query: {original_query}\n\nContext:\n{context_text}"},
    ]

    response = await llm_instance.ainvoke(messages)
    answer = getattr(response, "content", "") or ""

    # Check if this is a chart question (simple heuristic)
    is_chart = bool(re.search(r"\b(chart|graph|plot|visuali[zs]|trend|distribution)\b", original_query.lower()))

    return {
        "answer": answer,
        "is_chart_query": is_chart,
        "thinking_chunks": [],
    }


# ---------------------------------------------------------------------------
# Node: adaptive_reranking (conditional, if confidence low)
# ---------------------------------------------------------------------------

async def adaptive_reranking_node(
    state: AgentState,
    db: Any,
    kb_ids: List[int] | None = None,
    org_id: int | None = None,
    file_markdown: str | None = None,
    use_dense: bool = True,
    use_sparse: bool = True,
    use_exact: bool = True,
    use_graph_rag: bool = False,
) -> dict:
    """Adaptive reranking when retrieval confidence is low.

    When initial retrieval confidence is below threshold, reruns retrieval
    with modified parameters to improve recall.
    """
    kb_ids = kb_ids or state.get("kb_ids", [])
    org_id = org_id if org_id is not None else state.get("org_id")
    file_markdown = file_markdown or state.get("file_markdown")

    conf_score = state.get("retrieval_confidence", 0.0)
    current_docs = state.get("retrieved_docs", [])

    # Only adapt if confidence is low and we have some docs
    if conf_score >= 0.3 or not current_docs:
        return {"adaptive_rerunning": False, "confidence": conf_score}

    rewritten = state.get("rewritten_query", state.get("original_query", ""))
    datastore_ids = get_effective_datastore_ids(kb_ids, org_id, db) if db else []

    # Run full retrieval again with lower threshold (return_full_pool=True)
    retrieval_result = await hybrid_search_with_legs(
        query=rewritten,
        kb_ids=kb_ids,
        db=db,
        use_dense=use_dense,
        use_sparse=use_sparse,
        use_exact=use_exact,
        use_graph_rag=use_graph_rag,
        datastore_ids=datastore_ids,
        return_full_pool=True,
    )

    new_docs = retrieval_result.get("docs", [])
    new_info = retrieval_result.get("retrieval_info", {})
    new_conf_result = score_retrieval(new_docs, new_info) if new_docs else None
    new_conf = new_conf_result.score / 100.0 if new_conf_result else 0.0

    # Merge new docs with existing ones (deduplication)
    existing_hashes = {d.get("metadata", {}).get("content_hash") or d.get("page_content", "") for d in current_docs}
    new_unique = [d for d in new_docs if (d.get("metadata", {}).get("content_hash") or d.get("page_content", "")) not in existing_hashes]

    merged_docs = current_docs + new_unique
    merged_contexts = [format_context_string(merged_docs, file_markdown)]

    return {
        "adaptive_rerunning": True,
        "retrieved_docs": merged_docs,
        "retrieved_contexts": merged_contexts,
        "retrieval_confidence": new_conf,
        "new_doc_count": len(new_unique),
    }


# ---------------------------------------------------------------------------
# Node: chart_validation (conditional, if charts detected)
# ---------------------------------------------------------------------------

def chart_validation_node(state: AgentState) -> dict:
    """Validate chart-related queries and add chart context."""
    is_chart = state.get("is_chart_query", False)

    if not is_chart:
        return {"chart_validated": False, "chart_data": None}

    # Simple validation: ensure we have context data for chart generation
    retrieved_docs = state.get("retrieved_docs", [])
    has_enough_context = len(retrieved_docs) >= 2

    chart_data = {
        "is_chart_query": True,
        "has_context": has_enough_context,
        "doc_count": len(retrieved_docs),
        "validation_message": "Chart query validated — sufficient context available"
        if has_enough_context
        else "Chart query validated — limited context available, may need clarification",
    }

    return {
        "chart_validated": True,
        "chart_data": chart_data,
    }


# ---------------------------------------------------------------------------
# Node: compress context (between retrieval iterations)
# ---------------------------------------------------------------------------

def compress_context_node(state: AgentState) -> dict:
    """Compress accumulated retrieval context to free token budget."""
    retrieval_keys = set(state.get("retrieval_keys", set()) or set())

    for doc in state.get("retrieved_docs", []):
        source = doc.get("metadata", {}).get("source", "")
        if source:
            retrieval_keys.add(f"source:{source}")

    return {
        "retrieval_keys": list(retrieval_keys),
    }


# ---------------------------------------------------------------------------
# Node: should_compress_context (routing decision)
# ---------------------------------------------------------------------------

def should_compress_context(state: AgentState) -> str:
    """Decide whether to compress context based on token budget."""
    messages = state.get("messages", [])
    current_tokens = estimate_messages_tokens(messages)
    max_allowed = settings.OPENAI_MODEL_CONTEXT_SIZE * 0.8

    if current_tokens > max_allowed:
        return "compress"
    return "next"
