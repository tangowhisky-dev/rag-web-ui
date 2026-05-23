"""
LangGraph-based multi-agent RAG orchestration.

T01: Schema definition and interface contract for run_stream.
T02: rewrite_query + context_router nodes.
T03: extract_file_sections, kb_retrieval, grade_documents nodes.
T04: merge_context, generate_answer nodes + StateGraph assembly + full run_stream().
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncGenerator, List, Optional

from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class RAGGraphState(TypedDict):
    """Shared state passed between graph nodes."""

    query: str
    rewritten_query: str
    route: str                    # "file" | "kb" | "both"
    sources: List[str]            # ["kb", "file_current", "file_prior"]
    file_ids_needed: List[int]    # file IDs the router decided to use
    router_rationale: str         # LLM rationale for routing decision
    file_markdown: Optional[str]
    retrieved_docs: list          # raw docs from KB retrieval
    graded_docs: list             # docs that passed relevance grading
    merged_context: str           # final formatted context string
    answer: str
    agent_steps: list             # each: {"node": str, "latency_ms": float, "status": str}
    # run-time context injected by run_stream before graph execution
    knowledge_base_ids: List[int]
    recent_lc_history: list
    existing_summary: Optional[str]
    use_dense: bool
    use_sparse: bool
    use_exact: bool
    use_graph_rag: bool
    temperature: float
    model_name: Optional[str]
    display_query: Optional[str]


# ---------------------------------------------------------------------------
# SSE event type constants (used by chat_service to map run_stream events)
# ---------------------------------------------------------------------------

EVENT_AGENT_STEP    = "agent_step"      # graph node started/finished
EVENT_REWRITTEN     = "rewritten_query" # query after rewrite node
EVENT_CONTEXT       = "context"         # retrieved + graded docs + confidence
EVENT_TOKEN         = "token"           # streaming answer token
EVENT_DONE          = "done"            # final event, carries full_response + usage


# ---------------------------------------------------------------------------
# Node helpers
# ---------------------------------------------------------------------------

def _get_llm(model_name: Optional[str] = None, temperature: float = 0.0):
    """Return a ChatOpenAI instance (lazy import so tests can mock easily)."""
    from langchain_openai import ChatOpenAI  # noqa: PLC0415
    return ChatOpenAI(
        model=model_name or "gpt-4o-mini",
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Node: rewrite_query_node
# ---------------------------------------------------------------------------

async def rewrite_query_node(state: RAGGraphState) -> dict:
    """
    Rewrites the user query into a more retrieval-friendly form.

    Logs: [REWRITE] original=... rewritten=... latency_ms=...
    Updates: rewritten_query, agent_steps
    """
    t0 = time.monotonic()
    query = state["query"]
    model_name = state.get("model_name")
    temperature = state.get("temperature", 0.0)

    llm = _get_llm(model_name, temperature)

    system_prompt = (
        "You are a search query optimizer. "
        "Rewrite the user's question into a concise, keyword-rich retrieval query "
        "that maximises recall from a vector database. "
        "Return ONLY the rewritten query — no explanations, no quotes."
    )
    history = state.get("recent_lc_history") or []
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-4:]:  # last 2 turns for context
        messages.append(msg)
    messages.append({"role": "user", "content": query})

    response = await llm.ainvoke(messages)
    rewritten = response.content.strip() or query

    latency_ms = round((time.monotonic() - t0) * 1000, 2)
    logger.info("[REWRITE] original=%r rewritten=%r latency_ms=%.1f", query, rewritten, latency_ms)

    step = {"node": "rewrite_query", "latency_ms": latency_ms, "status": "done"}
    return {
        "rewritten_query": rewritten,
        "agent_steps": (state.get("agent_steps") or []) + [step],
    }


# ---------------------------------------------------------------------------
# Node: context_router_node
# ---------------------------------------------------------------------------

_ROUTER_SYSTEM_PROMPT = """\
You are a RAG context router. Given a user query and the available context sources,
decide which sources are needed to answer the question.

Available sources:
- "kb"           — knowledge base (vector store with embedded documents)
- "file_current" — the file currently open / being discussed in this chat turn
- "file_prior"   — files previously uploaded in earlier turns of this conversation

Respond with a JSON object:
{
  "sources": ["kb"],                 // list of sources needed, non-empty
  "rationale": "...",                // one-sentence reason
  "file_ids_needed": []              // integer file IDs if file_current or file_prior are chosen, else []
}

Rules:
- Always include at least one source.
- Include "kb" whenever general knowledge or knowledge-base content is useful.
- Include "file_current" only when the query asks about the current file/document.
- Include "file_prior" only when the query references earlier uploaded files by ID or content.
- file_ids_needed must be integer IDs, not strings.
"""


async def context_router_node(state: RAGGraphState) -> dict:
    """
    Decides which context sources to use: kb, file_current, file_prior.

    Logs: [ROUTER] sources=[...] rationale=... file_ids_needed=[...]
    Updates: sources, file_ids_needed, router_rationale, agent_steps
    """
    t0 = time.monotonic()
    query = state.get("rewritten_query") or state["query"]
    model_name = state.get("model_name")
    temperature = state.get("temperature", 0.0)

    llm = _get_llm(model_name, temperature)

    user_msg = f"Query: {query}"
    if state.get("file_markdown"):
        user_msg += "\n\n[A file is currently attached to this conversation.]"
    if state.get("existing_summary"):
        user_msg += "\n\n[There is a conversation summary from earlier turns.]"

    messages = [
        {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    # Request JSON output
    from langchain_openai import ChatOpenAI  # noqa: PLC0415 (already imported above but kept explicit)
    json_llm = _get_llm(model_name, 0.0)
    try:
        response = await json_llm.ainvoke(
            messages,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.content)
        sources: List[str] = parsed.get("sources") or ["kb"]
        rationale: str = parsed.get("rationale", "")
        file_ids_needed: List[int] = [int(x) for x in (parsed.get("file_ids_needed") or [])]
    except Exception as exc:  # noqa: BLE001
        logger.warning("[ROUTER] JSON parse error: %s — falling back to kb-only", exc)
        sources = ["kb"]
        rationale = "fallback due to parse error"
        file_ids_needed = []

    latency_ms = round((time.monotonic() - t0) * 1000, 2)
    logger.info(
        "[ROUTER] sources=%s rationale=%r file_ids_needed=%s latency_ms=%.1f",
        sources, rationale, file_ids_needed, latency_ms,
    )

    step = {"node": "context_router", "latency_ms": latency_ms, "status": "done",
            "sources": sources, "rationale": rationale}
    return {
        "sources": sources,
        "file_ids_needed": file_ids_needed,
        "router_rationale": rationale,
        "agent_steps": (state.get("agent_steps") or []) + [step],
    }


# ---------------------------------------------------------------------------
# Node: extract_file_sections_node
# ---------------------------------------------------------------------------

async def extract_file_sections_node(state: RAGGraphState) -> dict:
    """
    Extracts the most relevant sections from attached file markdown
    instead of dumping the entire document into context.

    Logs: [EXTRACT] chars_in=... sections=... chars_out=... latency_ms=...
    Updates: file_markdown (trimmed), agent_steps
    """
    t0 = time.monotonic()
    file_markdown: Optional[str] = state.get("file_markdown")

    if not file_markdown:
        latency_ms = round((time.monotonic() - t0) * 1000, 2)
        step = {"node": "extract_file_sections", "latency_ms": latency_ms, "status": "skipped"}
        return {"agent_steps": (state.get("agent_steps") or []) + [step]}

    query = state.get("rewritten_query") or state["query"]
    model_name = state.get("model_name")

    # Split into sections by markdown headings or double-newlines
    import re
    raw_sections = re.split(r"\n(?=#{1,3} )", file_markdown)
    if len(raw_sections) <= 1:
        # fallback: split by paragraph blocks
        raw_sections = [s.strip() for s in file_markdown.split("\n\n") if s.strip()]

    chars_in = len(file_markdown)

    # If the file is small enough, keep it all
    MAX_CHARS = 12_000
    if chars_in <= MAX_CHARS:
        latency_ms = round((time.monotonic() - t0) * 1000, 2)
        logger.info(
            "[EXTRACT] chars_in=%d sections=%d chars_out=%d latency_ms=%.1f (passthrough)",
            chars_in, len(raw_sections), chars_in, latency_ms,
        )
        step = {"node": "extract_file_sections", "latency_ms": latency_ms,
                "status": "passthrough", "sections": len(raw_sections)}
        return {"agent_steps": (state.get("agent_steps") or []) + [step]}

    # Use LLM to pick the top sections most relevant to the query
    llm = _get_llm(model_name, 0.0)

    # Build a numbered list of section previews (first 200 chars each)
    previews = "\n".join(
        f"[{i}] {s[:200].strip()}" for i, s in enumerate(raw_sections)
    )
    system_prompt = (
        "You are a document section selector. "
        "Given a query and a numbered list of document sections, "
        "return a JSON object: {\"indices\": [<list of section indices>]} "
        "selecting the 3-6 most relevant sections. "
        "Return ONLY the JSON object."
    )
    user_msg = f"Query: {query}\n\nSections:\n{previews}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    selected_sections = raw_sections  # fallback
    try:
        response = await llm.ainvoke(
            messages,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.content)
        indices: List[int] = [int(i) for i in (parsed.get("indices") or [])]
        if indices:
            selected_sections = [raw_sections[i] for i in indices if 0 <= i < len(raw_sections)]
    except Exception as exc:  # noqa: BLE001
        logger.warning("[EXTRACT] section selection failed: %s — keeping all sections", exc)

    trimmed = "\n\n".join(selected_sections)
    latency_ms = round((time.monotonic() - t0) * 1000, 2)
    logger.info(
        "[EXTRACT] chars_in=%d sections_total=%d sections_kept=%d chars_out=%d latency_ms=%.1f",
        chars_in, len(raw_sections), len(selected_sections), len(trimmed), latency_ms,
    )

    step = {"node": "extract_file_sections", "latency_ms": latency_ms, "status": "done",
            "sections_total": len(raw_sections), "sections_kept": len(selected_sections)}
    return {
        "file_markdown": trimmed,
        "agent_steps": (state.get("agent_steps") or []) + [step],
    }


# ---------------------------------------------------------------------------
# Node: kb_retrieval_node
# ---------------------------------------------------------------------------

async def kb_retrieval_node(state: RAGGraphState) -> dict:
    """
    Retrieves documents from the knowledge base using the existing
    multi-strategy retrieval utilities.

    Logs: [RETRIEVE] kb_ids=... docs_found=... latency_ms=...
    Updates: retrieved_docs, agent_steps
    """
    t0 = time.monotonic()
    sources: List[str] = state.get("sources") or ["kb"]

    if "kb" not in sources:
        latency_ms = round((time.monotonic() - t0) * 1000, 2)
        step = {"node": "kb_retrieval", "latency_ms": latency_ms, "status": "skipped"}
        return {
            "retrieved_docs": state.get("retrieved_docs") or [],
            "agent_steps": (state.get("agent_steps") or []) + [step],
        }

    query = state.get("rewritten_query") or state["query"]
    kb_ids: List[int] = state.get("knowledge_base_ids") or []
    db = state.get("_db")  # injected by run_stream

    use_dense: bool = state.get("use_dense", True)
    use_sparse: bool = state.get("use_sparse", True)
    use_exact: bool = state.get("use_exact", True)
    use_graph_rag: bool = state.get("use_graph_rag", False)

    docs: list = []
    try:
        from app.services.retrieval import retrieve_documents  # noqa: PLC0415
        docs = await retrieve_documents(
            query=query,
            knowledge_base_ids=kb_ids,
            db=db,
            use_dense=use_dense,
            use_sparse=use_sparse,
            use_exact=use_exact,
            use_graph_rag=use_graph_rag,
        )
    except ImportError:
        # retrieval module may not exist yet in test environments
        logger.warning("[RETRIEVE] retrieval module not available — returning empty docs")
    except Exception as exc:  # noqa: BLE001
        logger.error("[RETRIEVE] error: %s", exc)

    latency_ms = round((time.monotonic() - t0) * 1000, 2)
    logger.info(
        "[RETRIEVE] kb_ids=%s docs_found=%d latency_ms=%.1f",
        kb_ids, len(docs), latency_ms,
    )

    step = {"node": "kb_retrieval", "latency_ms": latency_ms,
            "status": "done", "docs_found": len(docs)}
    prior = state.get("retrieved_docs") or []
    return {
        "retrieved_docs": prior + docs,
        "agent_steps": (state.get("agent_steps") or []) + [step],
    }


# ---------------------------------------------------------------------------
# Node: grade_documents_node
# ---------------------------------------------------------------------------

_GRADE_SYSTEM_PROMPT = """\
You are a relevance grader. Given a query and a document excerpt, decide whether
the document is relevant to the query.

Respond with a JSON object:
{"relevant": true}  or  {"relevant": false}

Be strict: only mark relevant if the document directly helps answer the query.
"""


async def grade_documents_node(state: RAGGraphState) -> dict:
    """
    Grades each retrieved document for relevance, keeping only relevant ones.
    Triggers a rewrite-retry signal when too few docs pass.

    Logs: [GRADE] relevant=N irrelevant=M retry=bool latency_ms=...
    Updates: graded_docs, agent_steps
    """
    t0 = time.monotonic()
    docs: list = state.get("retrieved_docs") or []
    query = state.get("rewritten_query") or state["query"]
    model_name = state.get("model_name")

    if not docs:
        latency_ms = round((time.monotonic() - t0) * 1000, 2)
        step = {"node": "grade_documents", "latency_ms": latency_ms,
                "status": "skipped", "relevant": 0, "irrelevant": 0, "retry": False}
        return {
            "graded_docs": [],
            "agent_steps": (state.get("agent_steps") or []) + [step],
        }

    llm = _get_llm(model_name, 0.0)

    graded: list = []
    irrelevant_count = 0

    for doc in docs:
        # Extract text from various doc shapes (dict with "content", LangChain Document, or str)
        if isinstance(doc, dict):
            content = doc.get("content") or doc.get("page_content") or str(doc)
        elif hasattr(doc, "page_content"):
            content = doc.page_content
        else:
            content = str(doc)

        excerpt = content[:500]  # grade on first 500 chars for cost efficiency
        messages = [
            {"role": "system", "content": _GRADE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {query}\n\nDocument:\n{excerpt}"},
        ]
        try:
            response = await llm.ainvoke(
                messages,
                response_format={"type": "json_object"},
            )
            parsed = json.loads(response.content)
            is_relevant: bool = bool(parsed.get("relevant", True))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[GRADE] parse error for doc: %s — marking relevant by default", exc)
            is_relevant = True

        if is_relevant:
            graded.append(doc)
        else:
            irrelevant_count += 1

    # Retry signal: fewer than 2 relevant docs and at least one irrelevant doc was filtered
    retry = len(graded) < 2 and irrelevant_count > 0

    latency_ms = round((time.monotonic() - t0) * 1000, 2)
    logger.info(
        "[GRADE] relevant=%d irrelevant=%d retry=%s latency_ms=%.1f",
        len(graded), irrelevant_count, retry, latency_ms,
    )

    step = {"node": "grade_documents", "latency_ms": latency_ms, "status": "done",
            "relevant": len(graded), "irrelevant": irrelevant_count, "retry": retry}
    return {
        "graded_docs": graded,
        "agent_steps": (state.get("agent_steps") or []) + [step],
    }


# ---------------------------------------------------------------------------
# run_stream() — async generator interface contract
#
# Interface is stable from T01 onward.  Full graph wiring arrives in T04.
# ---------------------------------------------------------------------------

async def run_stream(
    query: str,
    file_markdown: Optional[str],
    db: Any,
    chat_id: int,
    knowledge_base_ids: List[int],
    recent_lc_history: list,
    existing_summary: Optional[str],
    use_dense: bool = True,
    use_sparse: bool = True,
    use_exact: bool = True,
    use_graph_rag: bool = False,
    temperature: float = 0.0,
    model_name: Optional[str] = None,
    display_query: Optional[str] = None,
) -> AsyncGenerator[dict, None]:
    """
    Async generator that runs the full RAG graph and streams events.

    Yield shapes (keyed by "event"):
      {"event": "agent_step",    "node": str, "latency_ms": float, "status": str}
      {"event": "rewritten_query","query": str}
      {"event": "context",       "docs": list, "confidence": str, "score": float,
                                 "suggestion": str, "failed_legs": list,
                                 "breakdown": dict, "query_classification": dict,
                                 "tool_trace": list, "synthesis_mode": bool}
      {"event": "token",         "content": str}
      {"event": "done",          "full_response": str,
                                 "usage": {"promptTokens": int, "completionTokens": int}}

    Full multi-node graph (rewrite → route → retrieve → grade →
    merge → generate) is wired in T02-T04.  This stub allows chat_service
    to compile and be tested against the interface contract.
    """
    start = time.monotonic()

    # Stub: emit one placeholder agent_step so the frontend has something to
    # render while T02-T04 implement the real nodes.
    yield {
        "event": EVENT_AGENT_STEP,
        "node": "placeholder",
        "latency_ms": round((time.monotonic() - start) * 1000, 2),
        "status": "pending",
        "message": "LangGraph graph not yet wired (T02-T04).",
    }

    # Stub: emit a pass-through rewritten_query
    yield {
        "event": EVENT_REWRITTEN,
        "query": display_query or query,
    }

    # Stub: emit empty context
    yield {
        "event": EVENT_CONTEXT,
        "docs": [],
        "confidence": "low",
        "score": 0.0,
        "suggestion": "Graph not yet wired.",
        "failed_legs": [],
        "breakdown": {},
        "query_classification": {"type": "SIMPLE", "confidence": 0.0, "latency_ms": 0, "fallback": True},
        "tool_trace": [],
        "synthesis_mode": False,
    }

    # Stub: emit a placeholder answer token
    stub_msg = "[Graph not yet wired — T02-T04 pending]"
    yield {"event": EVENT_TOKEN, "content": stub_msg}

    yield {
        "event": EVENT_DONE,
        "full_response": stub_msg,
        "usage": {"promptTokens": 0, "completionTokens": 0},
    }
