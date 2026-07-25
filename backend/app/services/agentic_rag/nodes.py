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
    ANSWER_SYSTEM_PROMPT_BASE,
    CHAT_ONLY_SYSTEM_PROMPT,
    CHAT_ONLY_SYSTEM_PROMPT_BASE,
    CLASSIFY_SYSTEM_PROMPT,
    COMPACTION_SYSTEM_PROMPT,
    COMPACTION_USER_PROMPT,
    EVALUATION_SYSTEM_PROMPT,
    RETRIEVED_CONTEXT_TEMPLATE,
)
from .redis_memory import get_redis_memory
from .schemas import QueryAnalysis
from .utils import estimate_context_tokens, strip_reasoning_tags, format_context_string

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

    threshold = settings.COMPACTION_HISTORY_THRESHOLD
    keep_recent = settings.COMPACTION_KEEP_RECENT
    max_summary_chars = settings.COMPACTION_SUMMARY_MAX_CHARS

    messages = state.get("messages", [])
    if len(messages) <= threshold:
        return {"compaction_summary": None, "compaction_triggered": False}

    # Split: keep recent messages, summarize older ones
    recent_messages = messages[-keep_recent:]
    old_messages = messages[:len(messages) - keep_recent]

    if not old_messages:
        return {"compaction_summary": None, "compaction_triggered": False}

    logger.info(
        "[COMPACTION] triggered | total_msgs=%d | recent=%d | summarizing=%d",
        len(messages), len(recent_messages), len(old_messages),
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
    try:
        writer = get_stream_writer()
    except RuntimeError:
        writer = None
    if writer is not None:
        writer({"event": "agent_step", "node": name, "status": "active", "latency_ms": 0})
    try:
        yield
    finally:
        if writer is not None:
            writer({"event": "agent_step", "node": name, "status": "done", "latency_ms": 0})


# ── Answer Generation ──────────────────────────────────────────────────────
# ANSWER_SYSTEM_PROMPT_BASE, RETRIEVED_CONTEXT_TEMPLATE,
# CHAT_ONLY_SYSTEM_PROMPT_BASE, CHAT_ONLY_SYSTEM_PROMPT imported from prompts.py


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
        writer = get_stream_writer()
        writer({"event": "rewritten_query", "query": rewritten})

        return {"rewritten_query": rewritten}


def _enrich_query(subtask: str, dependencies: list[int], prior_contexts: dict[int, str]) -> str:
    """Enrich a subtask query with context from its dependencies.

    For each dependency index, append the prior subtask's context so the
    model can resolve references like "that" or "the second one".
    """
    query = subtask
    for dep_idx in sorted(dependencies):
        if dep_idx in prior_contexts:
            ctx = prior_contexts[dep_idx][:500]  # cap per-reference
            query += f"\n\n[Reference to previous subtask {dep_idx}:\n{ctx}]"
    return query


# ---------------------------------------------------------------------------
# Node: enrich_subtask_query (lightweight enrichment, no LLM call)
# ---------------------------------------------------------------------------

def enrich_subtask_query_node(state: AgentState) -> dict:
    """Enrich the rewritten query with context from dependency subtasks.

    This replaces the old rewrite_subtask_query_node for the sequential loop.
    No LLM call — just text injection so dependent subtasks can resolve
    references like "the second one" or "that".
    """
    subtasks = state.get("subtasks", [])
    idx = state.get("current_subtask_index", 0)
    subtask = subtasks[idx] if 0 <= idx < len(subtasks) else state.get("original_query", "")

    dependencies = state.get("subtask_dependencies", [])[idx] if idx < len(state.get("subtask_dependencies", [])) else []
    prior_contexts = state.get("subtask_contexts", [])
    dep_context = {}
    for i, ctx in enumerate(prior_contexts):
        contexts = ctx.get("retrieved_contexts", [])
        dep_context[i] = "\n\n".join(contexts) if contexts else ""

    enriched = _enrich_query(subtask, dependencies, dep_context)

    return {"rewritten_query": enriched}


# ── Query Classification ───────────────────────────────────────────────────
# CLASSIFY_SYSTEM_PROMPT imported from prompts.py


async def classify_query_node(
    state: AgentState,
) -> dict:
    """Classify query using structured LLM output with per-subtask routing."""
    rewritten = state.get("rewritten_query", "")
    query = state.get("original_query", "")
    pending_query = ""  # Set when is_clear=False for clarification flow

    try:
        llm = _get_llm(streaming=False)
        llm_structured = llm.with_structured_output(QueryAnalysis)

        response = llm_structured.invoke([
            {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
            {"role": "user", "content": rewritten},
        ])
        is_clear = getattr(response, "is_clear", True)
        questions = getattr(response, "questions", [rewritten]) or [rewritten]
        clarification_needed = getattr(response, "clarification_needed", "")
        clarification_questions = getattr(response, "clarification_questions", [])
        raw_deps = getattr(response, "subtask_dependencies", None)
        subtask_deps = raw_deps if raw_deps is not None else [[] for _ in questions]

        # Extract per-subtask routing flags
        subtask_routing = getattr(response, "subtask_routing", None)
        if subtask_routing is None or len(subtask_routing) != len(questions):
            # LLM didn't return per-subtask routing — fall back to global flags
            logger.warning(
                "[CLASSIFY] subtask_routing length %d != questions length %d, falling back to global",
                len(subtask_routing) if subtask_routing else 0, len(questions),
            )
            gr = getattr(response, "needs_retrieval", True)
            gc = getattr(response, "needs_file_content", False)
            gm = getattr(response, "needs_file_metadata", False)
            subtask_routing = [
                {"needs_retrieval": gr, "needs_file_content": gc, "needs_file_metadata": gm}
                for _ in questions
            ]

        # Sanity check: override is_clear=False for clearly factual/definitional
        # queries. The LLM sometimes misclassifies these as unclear, which routes
        # to request_clarification and silently kills the response.
        query_lower = rewritten.lower()
        factual_patterns = (
            "explain ", "what is ", "what's ", "define ", "describe ",
            "how does ", "how do ", "difference between ", "vs ", "versus ",
            "means ", "what does ", "what does ", "meaning of ",
            "tell me about ", "give me an overview of ", "overview of ",
        )
        is_factual = any(query_lower.startswith(p) for p in factual_patterns)
        if is_factual and not is_clear:
            logger.warning(
                "[CLASSIFY] factual query misclassified as unclear — overriding is_clear=True | query=%s",
                rewritten[:80],
            )
            is_clear = True
            clarification_needed = ""

        # Set pending_query for clarification flow
        pending_query = rewritten if not is_clear else ""
    except Exception as exc:
        logger.warning("[CLASSIFY] structured classification failed: %s - using fallback", exc)
        is_clear = True
        questions = [rewritten]
        clarification_needed = ""
        clarification_questions = []
        subtask_deps = [[] for _ in questions]
        subtask_routing = [
            {"needs_retrieval": True, "needs_file_content": False, "needs_file_metadata": False}
            for _ in questions
        ]

    with _agent_step("classify_query"):
        subtasks = questions if len(questions) > 1 else [rewritten]
        # Align dependencies and routing with subtasks list
        subtask_deps = subtask_deps[:len(subtasks)] if len(subtask_deps) == len(subtasks) else [[] for _ in subtasks]
        subtask_routing = subtask_routing[:len(subtasks)] if len(subtask_routing) == len(subtasks) else [
            {"needs_retrieval": True, "needs_file_content": False, "needs_file_metadata": False}
            for _ in subtasks
        ]

        # Emit explicit task_list event for multi-subtask queries so the runner
        # doesn't have to reverse-engineer it from state updates.
        if len(subtasks) > 1:
            writer = get_stream_writer()
            writer({
                "event": "task_list",
                "tasks": [
                    {"id": i, "text": text, "status": "pending"}
                    for i, text in enumerate(subtasks)
                ],
            })

        # Compute legacy global flags for backward compatibility (first subtask wins)
        if subtask_routing:
            first_routing = subtask_routing[0]
            needs_retrieval = getattr(first_routing, "needs_retrieval", True)
            needs_file_content = getattr(first_routing, "needs_file_content", False)
            needs_file_metadata = getattr(first_routing, "needs_file_metadata", False)
        else:
            needs_retrieval = True
            needs_file_content = False
            needs_file_metadata = False

        return {
            "question_is_clear": is_clear,
            "clarification_needed": clarification_needed,
            "clarification_questions": clarification_questions,
            "pending_query": pending_query,
            "subtasks": subtasks,
            "is_complex": len(subtasks) > 1,
            "needs_retrieval": needs_retrieval,
            "needs_file_content": needs_file_content,
            "needs_file_metadata": needs_file_metadata,
            "subtask_dependencies": subtask_deps,
            "subtask_routing": subtask_routing,  # NEW: per-subtask routing
        }


# ---------------------------------------------------------------------------
# Node: request_clarification
# ---------------------------------------------------------------------------

def request_clarification_node(state: AgentState) -> dict | Command:
    """Ask the user for clarification when the query is unclear.

    Pauses graph execution via interrupt() and waits for user input.
    On resume, stores the response and routes back to classify_query
    so the full pipeline re-runs with the clarification context.
    """
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

    # Emit progress event before interrupting (streaming writer works inside node)
    with _agent_step("request_clarification"):
        writer = get_stream_writer()
        writer({"event": "progress", "phase": "clarification", "message": clarification_msg})

    # Pause execution and wait for user response.
    # The interrupt payload surfaces on stream.interrupts when using v3 stream_events.
    # On resume, Command(resume=...) becomes the return value of this call.
    user_response = interrupt(clarification_msg)

    # Store the user's response and allow re-classification to proceed.
    # Route to classify_query — the user's clarification response (e.g.
    # "Technical" or "Computer science") is already a meaningful query.
    # Setting rewritten_query ensures classify_query classifies the right thing
    # instead of re-classifying the original pending_query.
    return Command(
        update={
            "clarification_response": str(user_response),
            "rewritten_query": str(user_response),
            "question_is_clear": True,
        },
        goto="classify_query",
    )


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
# Node: sufficiency_check (after filter)
# ---------------------------------------------------------------------------

def sufficiency_check_node(state: AgentState) -> dict:
    """Check if filtered docs are sufficient.

    Routes to adaptive_reranking if confidence is low or doc count is small.
    """
    conf_score = state.get("retrieval_confidence", 0.0)
    doc_count = len(state.get("retrieved_docs", []))

    confidence_met = conf_score > 0.3
    docs_met = doc_count >= 3

    sufficiency_met = confidence_met and docs_met

    needs_adaptive = not sufficiency_met

    if sufficiency_met:
        message = f"Retrieval sufficient: confidence={conf_score:.2f}, docs={doc_count}"
    else:
        message = f"Retrieval insufficient: confidence={conf_score:.2f}, docs={doc_count}, checking adaptive reranking"

    with _agent_step("sufficiency_check"):
        return {
            "sufficiency_met": sufficiency_met,
            "sufficiency_message": message,
            "needs_adaptive_reranking": needs_adaptive,
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
# Node: generating (actual LLM answer generation)
# CHAT_ONLY_SYSTEM_PROMPT_BASE, CHAT_ONLY_SYSTEM_PROMPT imported from prompts.py


def _build_generation_messages(state: AgentState) -> list[dict]:
    """Build the LLM message list from bounded history plus retrieved context.

    The Redis checkpointer accumulates the full conversation. We pass:
    - The compaction summary (if present) — structured checkpoint of older turns
    - Recent user/assistant pairs from the checkpoint (assistant capped at
      COMPACTION_ASSISTANT_MAX_CHARS to prevent context poisoning while
      preserving conversational continuity)
    - The current query

    This lets the model reference its own prior answers ("as I mentioned above")
    while keeping the context window bounded.

    Routing-aware:
    - needs_retrieval=True: document context injected into system prompt
    - needs_retrieval=False: chat-only mode, no document context
    - needs_file_content=True: file content injected as user message
    - needs_file_metadata=True: file names/descriptions injected as user message
    """
    retrieved_docs = state.get("retrieved_docs", [])
    file_markdown = state.get("file_markdown")
    compaction_summary = state.get("compaction_summary")
    needs_retrieval = state.get("needs_retrieval", True)
    needs_file_content = state.get("needs_file_content", False)
    needs_file_metadata = state.get("needs_file_metadata", False)

    # ── System messages ──────────────────────────────────────────────────
    # Static behavioral instructions (first system message, never changes)
    # followed by a dynamic context message (second system message, per-query).
    #
    # This pattern is preferred in production RAG: the model's behavior
    # instructions stay constant while retrieved context changes per query,
    # enabling better token caching and clearer separation of concerns.
    system_messages: list[dict] = [{"role": "system", "content": ANSWER_SYSTEM_PROMPT_BASE}]

    if needs_retrieval and retrieved_docs:
        context_text = format_context_string(retrieved_docs, file_markdown)
        system_messages.append({
            "role": "system",
            "content": RETRIEVED_CONTEXT_TEMPLATE.format(retrieved_context=context_text),
        })
    elif needs_retrieval:
        # Retrieval was requested but no docs found
        system_messages.append({
            "role": "system",
            "content": RETRIEVED_CONTEXT_TEMPLATE.format(
                retrieved_context="The user asked a question but no relevant documents were found in the knowledge base. Answer from your general knowledge, but clearly state that no documents were found."
            ),
        })
    else:
        # Chat-only mode: split into base + context
        file_context_parts = []
        if needs_file_content and file_markdown:
            file_context_parts.append(
                f"Attached file content:\n{file_markdown}"
            )
        if needs_file_metadata and file_markdown:
            file_context_parts.append(
                "Attached file content is available above."
            )
        if not file_context_parts:
            file_context_parts.append("No attached files.")
        system_messages.append({
            "role": "system",
            "content": CHAT_ONLY_SYSTEM_PROMPT.format(
                file_context="\n\n".join(file_context_parts)
            ),
        })

    # ── Compaction summary — injected as a third system message ──────────
    # When present, tell the model the summary is the authoritative view of
    # earlier turns. Putting it in system (not user) means the model treats
    # it as an instruction to follow, not just another piece of context to
    # weigh against the raw messages below.
    if compaction_summary:
        system_messages.append({
            "role": "system",
            "content": (
                "# Conversation Checkpoint\n"
                "Previous conversation summary (for context, do NOT repeat this):\n"
                "<conversation_checkpoint>\n"
                f"{compaction_summary}\n"
                "</conversation_checkpoint>\n\n"
                "Use this summary as the authoritative view of earlier turns. "
                "Do NOT revisit details that are already covered in the summary."
            ),
        })

    # ── Merge all system messages into one (TGI only supports a single
    #    system message slot). ─────────────────────────────────────────────
    if len(system_messages) > 1:
        system_messages = [
            {"role": "system", "content": "\n\n".join(s["content"] for s in system_messages)}
        ]

    messages: list[dict] = system_messages

    # ── Compaction summary also injected as user message — NOPE, removed.
    # The model now sees it in the system prompt where it is treated as an
    # instruction, not as peer context. Keeping it as both system + user
    # creates the exact duplication problem we are trying to avoid.

    # ── File content from subtask context ────────────────────────────────
    # When subtasks run the file_context_subgraph, their captured file
    # content is available here via prepare_final_context's file_contents.
    file_contents = state.get("file_contents", [])
    if file_contents:
        for i, fc in enumerate(file_contents):
            messages.append({
                "role": "user",
                "content": f"[Subtask {i + 1} — File Content]\n{fc}",
            })
    # ── File content injection (legacy path) ─────────────────────────────
    elif needs_file_content and file_markdown:
        messages.append({
            "role": "user",
            "content": f"Attached file content:\n{file_markdown}",
        })
    elif needs_file_metadata and file_markdown:
        messages.append({
            "role": "user",
            "content": "An attached file is available in the conversation.",
        })

    # ── Conversation history (bounded by context budget) ─────────────────
    # Compaction splits messages into old (summarized) + recent.
    # When compaction is off, budget-driven selection picks the most recent
    # messages that fit within available context window.
    compaction_summary = state.get("compaction_summary")
    compaction_triggered = state.get("compaction_triggered", False)

    # ── Filter state.messages: only real conversation turns, no system/context ─
    # The MessagesState reducer can accumulate system messages that leaked in
    # from prior _build_generation_messages() calls. Also deduplicate by
    # content so the same query doesn't appear twice.
    from langchain_core.messages import HumanMessage, AIMessage

    seen_content: set[str] = set()
    clean_messages: list = []
    for m in state.get("messages", []):
        if isinstance(m, dict):
            role = m.get("role", "")
            if role not in ("user", "assistant"):
                continue
            clean_messages.append(m)
        elif isinstance(m, (HumanMessage, AIMessage)):
            content = str(m.content).strip()
            if content in seen_content:
                continue  # skip duplicates
            seen_content.add(content)
            clean_messages.append(m)
        else:
            continue

    if compaction_triggered:
        # Compaction already defined the split. Include only the recent ones
        # (the old ones are covered by the summary).
        keep_recent = settings.COMPACTION_KEEP_RECENT
        prior_messages = clean_messages[-keep_recent:]
    else:
        # No compaction — estimate budget and pick the most recent messages.
        # Reserve ~25% of the context window for system prompt, file content,
        # and retrieved docs. Walk backwards from the end to keep only the
        # most recent messages that fit.
        estimated_doc_chars = sum(
            estimate_context_tokens(str(d.get("page_content", ""))) * 4
            for d in retrieved_docs
        )
        # Account for file contents from subtasks in budget estimation
        estimated_file_chars = sum(
            estimate_context_tokens(fc) * 4 for fc in file_contents
        )
        estimated_sys_chars = 1500  # system prompt + file context + user message formatting
        estimated_summary_chars = (
            estimate_context_tokens(compaction_summary) * 4 if compaction_summary else 0
        )
        available_chars = max(
            0,
            int(
                settings.OPENAI_MODEL_CONTEXT_SIZE
                - estimated_doc_chars
                - estimated_file_chars
                - estimated_sys_chars
                - estimated_summary_chars
                - 500  # headroom
            ),
        )

        prior_messages = []
        current_chars = 0
        # Use clean_messages (filtered/deduplicated) instead of raw state.messages.
        # This prevents system messages and duplicates from leaking into the prompt.
        for m in reversed(clean_messages[:-1]):  # walk backwards, skip current query
            if current_chars > available_chars:
                break
            content = str(m.content)
            if isinstance(m, HumanMessage):
                prior_messages.insert(0, m)
                current_chars += len(content)
            elif isinstance(m, AIMessage):
                prior_messages.insert(0, m)
                current_chars += len(content)
            elif isinstance(m, dict):
                content = str(m.get("content", ""))
                prior_messages.insert(0, m)
                current_chars += len(content)

    # ── Emit the history ─────────────────────────────────────────────────
    for m in prior_messages:
        if isinstance(m, HumanMessage):
            messages.append({"role": "user", "content": str(m.content)})
        elif isinstance(m, AIMessage):
            messages.append({
                "role": "assistant",
                "content": str(m.content),
            })
        elif isinstance(m, dict):
            role = m.get("role")
            content = str(m.get("content", ""))
            if role == "user":
                messages.append({"role": "user", "content": content})
            elif role == "assistant":
                messages.append({"role": "assistant", "content": content})

    # Always append the current user query (last message in state)
    current = state.get("messages", [])[-1] if state.get("messages") else None
    if current is not None:
        if isinstance(current, HumanMessage):
            messages.append({"role": "user", "content": str(current.content)})
        elif isinstance(current, AIMessage):
            messages.append({"role": "assistant", "content": str(current.content)})
        elif isinstance(current, dict):
            role = current.get("role")
            content = str(current.get("content", ""))
            if role and content:
                entry = {"role": role, "content": content}
                messages.append(entry)

    return messages


async def generating_node(
    state: AgentState,
    llm: ChatOpenAI | None = None,
    api_base: Optional[str] = None,
) -> dict:
    """Generate answer using retrieved context and conversation memory.

    Streams tokens to the client in real-time via LangGraph's stream writer so
    the answer appears word-by-word in the UI. The final accumulated answer is
    still returned in state for downstream nodes and persistence.
    """
    _ = api_base  # kept for API compatibility
    with _agent_step("generating"):
        llm_instance = llm or _get_llm(streaming=True)
        writer = get_stream_writer()

        messages = _build_generation_messages(state)
        logger.info("[GENERATING] messages=%d | last_user=%s", len(messages), messages[-1].get("content", "") if messages else "")
        # print complete messages for debugging
        for i, msg in enumerate(messages):
            logger.info("[GENERATING] message %d: role=%s | content=%s", i, msg.get("role"), msg.get("content", "")[:200])
            
        original_query = state.get("original_query", "")

        answer = ""
        usage_metadata: Optional[dict] = None
        # stream_usage=True asks OpenAI-compatible endpoints to emit token usage
        # on the final streaming chunk. This is the only reliable way to get usage
        # when streaming through v3.
        async for chunk in llm_instance.astream(messages, stream_usage=True):
            chunk_text = getattr(chunk, "content", "") or ""
            if isinstance(chunk_text, list):
                chunk_text = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in chunk_text
                )
            if chunk_text:
                answer += chunk_text
                writer({"event": "token", "content": chunk_text})
            if getattr(chunk, "usage_metadata", None):
                usage_metadata = chunk.usage_metadata

    is_chart = bool(re.search(r"\b(chart|graph|plot|visuali[zs]|trend|distribution)\b", original_query.lower()))

    result: dict = {
        "answer": answer,
        "is_chart_query": is_chart,
        "thinking_chunks": [],
        "messages": [AIMessage(content=answer)] if answer else [],
    }
    if usage_metadata:
        result["answer_usage"] = usage_metadata
    return result

# ---------------------------------------------------------------------------
# Node: chart_validation (conditional, if charts detected)
# ---------------------------------------------------------------------------

def chart_validation_node(state: AgentState) -> dict:
    """Validate chart-related queries and add chart context."""
    with _agent_step("chart_validation"):
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
# Node: rewrite_subtask_query
# ---------------------------------------------------------------------------

async def rewrite_subtask_query_node(
    state: AgentState,
    api_base: Optional[str] = None,
) -> dict:
    """Rewrite a subtask query into a self-contained search query."""
    with _agent_step("rewrite_subtask_query"):
        from .utils import rewrite_query as _rewrite_query

        subtasks = state.get("subtasks", [])
        idx = state.get("current_subtask_index", 0)
        subtask = subtasks[idx] if 0 <= idx < len(subtasks) else state.get("original_query", "")

        # Enrich with dependency context for sequential subtasks
        dependencies = state.get("subtask_dependencies", [])[idx] if idx < len(state.get("subtask_dependencies", [])) else []
        prior_contexts = state.get("subtask_contexts", [])
        # Build a dict mapping dependency index → joined context string
        dep_context = {}
        for i, ctx in enumerate(prior_contexts):
            contexts = ctx.get("retrieved_contexts", [])
            dep_context[i] = "\n\n".join(contexts) if contexts else ""
        enriched_subtask = _enrich_query(subtask, dependencies, dep_context)

        # Carry conversation context so subtasks can resolve references like
        # "the second option" or "the previous paper".
        # Use the bounded subgraph_history supplied by route_by_dependencies
        # instead of the MessagesState ``messages`` channel to avoid subgraph
        # state being merged back into the parent conversation.
        subgraph_history = state.get("subgraph_history", [])
        if subgraph_history:
            # Already trimmed by route_by_dependencies; cap to 4 messages just in case.
            recent_history = subgraph_history[-4:]
        else:
            recent_history = select_recent_history(state.get("messages", []), max_pairs=2)

        rewritten = _rewrite_query(
            query=enriched_subtask,
            recent_history=recent_history,
            memory_context="",
            api_base=api_base,
            query_model=settings.effective_query_model,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_API_BASE,
        )
        return {"rewritten_query": rewritten}


async def load_subtask_memory_node(
    state: AgentState,
    db: Any,
) -> dict:
    """No-op — historical memory removed. The checkpointer provides full conversation flow."""
    _ = state
    _ = db
    return {}


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

        writer = get_stream_writer()
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

        writer = get_stream_writer()
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

        writer = get_stream_writer()
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
            writer = get_stream_writer()
            writer({"event": "context", "docs": merged})

        return {
            "retrieved_docs": merged,
        }


# ---------------------------------------------------------------------------
# Node: collect_context
# ---------------------------------------------------------------------------

def collect_context_node(state: AgentState) -> dict:
    """Collect a subagent's retrieved context and add it to subtask_contexts.

    Records a ``context_source`` field so prepare_final_context can distinguish
    between different kinds of context:
    - ``"retrieval"``  — docs from vector/sparse/exact/neo4j search (agent_subgraph)
    - ``"file"``       — uploaded file content (file_context_subgraph)
    - ``"chat"``       — pure conversation follow-up (chat_subgraph)

    For file-context subtasks, the file_markdown is captured here so that
    prepare_final_context can merge it into the generation context.
    """
    with _agent_step("collect_context"):
        routing = state.get("subtask_routing", [])
        routing = routing[0] if routing else {}
        needs_file = routing.get("needs_file_content", False) or routing.get("needs_file_metadata", False)
        file_markdown = state.get("file_markdown")

        context_source = "retrieval"
        if needs_file:
            context_source = "file"
        elif not state.get("retrieved_docs") and not needs_file:
            # Chat subgraph: no docs, no file — just conversation context
            context_source = "chat"

        # Capture file content as a context entry so prepare_final_context
        # can merge it alongside retrieved docs.
        file_context = None
        if context_source == "file" and file_markdown:
            file_context = file_markdown[:3000]  # cap to prevent overflow

        return {
            "subtask_contexts": [{
                "question": state.get("original_query", ""),
                "rewritten_query": state.get("rewritten_query", ""),
                "context_source": context_source,  # NEW: type of context
                "file_context": file_context,       # NEW: captured file content
                "retrieved_docs": state.get("retrieved_docs", []),
                "retrieval_confidence": state.get("retrieval_confidence", 0.0),
                "leg_results": state.get("leg_results", {}),
                "failed_legs": state.get("failed_legs", []),
                "all_scored_docs": state.get("all_scored_docs", []),
                "historical_memory_docs": state.get("historical_memory_docs", []),
            }],
        }


# ---------------------------------------------------------------------------
# Node: prepare_final_context
# ---------------------------------------------------------------------------

def prepare_final_context_node(state: AgentState) -> dict:
    """Aggregate all subtask contexts into the final retrieval state.

    Deduplicates docs by content_hash across subtasks so the final LLM context
    contains no duplicate chunks. Tracks how many duplicates were removed.

    Context sources can come from different places (retrieval, file upload,
    chat-only follow-up). All sources are merged into a single document set
    for the generation node.
    """
    contexts = state.get("subtask_contexts", [])
    if not contexts:
        return {}

    all_docs: list[dict] = []
    all_scored: list[dict] = []
    all_memory_docs: list[dict] = []
    file_contents: list[str] = []
    confidences: list[float] = []

    seen_hashes: set[str] = set()
    doc_deduped = 0
    scored_deduped = 0
    memory_deduped = 0
    file_deduped = 0

    for ctx in contexts:
        context_source = ctx.get("context_source", "retrieval")

        # Collect file context content
        if ctx.get("file_context"):
            h = content_hash(ctx["file_context"])
            if h not in seen_hashes:
                seen_hashes.add(h)
                file_contents.append(ctx["file_context"])
            else:
                file_deduped += 1

        for doc in ctx.get("historical_memory_docs", []):
            h = doc.get("metadata", {}).get("content_hash") or content_hash(
                doc.get("page_content", "")
            )
            if h not in seen_hashes:
                seen_hashes.add(h)
                all_memory_docs.append(doc)
            else:
                memory_deduped += 1
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
        confidences.append(ctx.get("retrieval_confidence", 0.0))

    is_complex = len(contexts) > 1
    if is_complex and (doc_deduped or scored_deduped or memory_deduped or file_deduped):
        logger.info(
            "[DEDUP] complex query: %d subtasks, sources: "
            "retrieved=%d scored=%d memory=%d file=%d | "
            "final_retrieved=%d | final_scored=%d | final_memory=%d | files=%d",
            len(contexts), doc_deduped, scored_deduped, memory_deduped,
            file_deduped,
            len(all_docs), len(all_scored), len(all_memory_docs),
            len(file_contents),
        )

    with _agent_step("prepare_final_context"):
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        # Stream the final aggregated context so the UI can show citations before
        # generation starts.
        if all_docs:
            writer = get_stream_writer()
            writer({
                "event": "context",
                "docs": all_docs,
                "confidence": "high" if avg_conf > 0.7 else "medium" if avg_conf > 0.3 else "low",
                "score": int(avg_conf * 100),
                "synthesis_mode": is_complex,
            })

        return {
            "retrieved_docs": all_docs,
            "all_scored_docs": all_scored,
            "retrieval_confidence": avg_conf,
            "is_complex": is_complex,
            "historical_memory_docs": all_memory_docs,
            "file_contents": file_contents,  # NEW: captured file contexts from subtasks
        }


# ---------------------------------------------------------------------------
# Node: finalize_answer
# ---------------------------------------------------------------------------

def finalize_answer_node(state: AgentState) -> dict:
    """Promote the generated answer to final_answer."""
    with _agent_step("finalize_answer"):
        answer = state.get("answer", "")
        return {
            "final_answer": answer,
        }


async def save_memory_node(state: AgentState) -> dict:
    """Persist the completed turn to Redis long-term memory."""
    if not settings.MEMORY_ENABLED:
        return {}

    with _agent_step("save_memory"):
        answer = state.get("final_answer") or state.get("answer", "")
        query = state.get("original_query", "")
        user_id = state.get("user_id")
        chat_id = state.get("chat_id")

        if not answer or not query:
            return {}

        try:
            memory = await get_redis_memory()
            await memory.save_turn(
                query=query,
                answer=answer,
                user_id=user_id,
                chat_id=chat_id,
            )
        except Exception as exc:
            logger.warning("[SAVE_MEMORY] failed: %s", exc)

    return {}


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
