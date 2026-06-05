"""
LangGraph-based multi-agent RAG orchestration — Agentic Pipeline v2.

Pipeline flow:
  rewrite_query
    → context_router          (smart source routing: kb / file / both)
    → decompose_query         (split into 2-5 atomic sub-queries)
    → parallel_retrieval      (hybrid search per sub-query, reinforced dedup)
    → extract_file_sections   (select relevant file sections per sub-query)
    → draft_answer            (draft answer for grading — not final output)
    → grade_coverage          (LLM grades which sub-queries are covered)
    → [conditional_router]
        ├─ all covered          → generate_answer (final)
        ├─ uncovered, attempt=0 → widened_retrieval  → draft_answer → grade_coverage
        ├─ uncovered, attempt=1 → keyword_search_loop → draft_answer → grade_coverage
        └─ attempt >= 2         → generate_answer (partial / unable)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from pydantic import BaseModel
from typing_extensions import TypedDict

from app.core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class RAGGraphState(TypedDict):
    # ── Query lifecycle ────────────────────────────────────────────────────
    query: str
    rewritten_query: str
    sub_queries: List[str]             # from decompose_query

    # ── Routing (preserved from v1) ───────────────────────────────────────
    sources: List[str]                 # ["kb", "file_current", "file_prior", "chat_history"]
    chat_history_docs: list            # from chat_history_retrieval_node
    file_ids_needed: List[int]
    router_rationale: str

    # ── File ──────────────────────────────────────────────────────────────
    file_markdown: Optional[str]

    # ── Retrieval ─────────────────────────────────────────────────────────
    retrieved_docs: list               # accumulates across all retry attempts
    retrieval_attempt: int             # 0=first, 1=widened, 2=keyword
    keyword_iterations: list           # [{sub_query, iteration, keywords, results_found}]

    # ── Grading / coverage ────────────────────────────────────────────────
    draft_answer: str
    coverage_result: dict              # sub_query → "covered"|"partially_covered"|"not_covered"
    uncovered_sub_queries: List[str]

    # ── Final answer ──────────────────────────────────────────────────────
    merged_context: str
    answer: str
    _usage: dict

    # ── Observability ─────────────────────────────────────────────────────
    agent_steps: list

    # ── Run-time context injected by run_stream ───────────────────────────
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
    api_base: Optional[str]
    query_model: Optional[str]
    org_id: Optional[int]
    _db: Any


# ---------------------------------------------------------------------------
# SSE event type constants
# ---------------------------------------------------------------------------

EVENT_AGENT_STEP = "agent_step"
EVENT_REWRITTEN  = "rewritten_query"
EVENT_CONTEXT    = "context"
EVENT_TOKEN      = "token"
EVENT_DONE       = "done"


# ---------------------------------------------------------------------------
# Pydantic schemas for structured LLM calls
# ---------------------------------------------------------------------------

class _RouterOutput(BaseModel):
    sources: List[str]
    rationale: str
    file_ids_needed: List[int] = []

class _SectionOutput(BaseModel):
    indices: List[int]

class _SubQueriesOutput(BaseModel):
    sub_queries: List[str]

class _CoverageItem(BaseModel):
    sub_query: str
    status: str  # "covered" | "partially_covered" | "not_covered"

class _CoverageOutput(BaseModel):
    coverage: List[_CoverageItem]

class _KeywordsOutput(BaseModel):
    broad_keywords: List[str]
    narrow_keywords: List[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_llm(model_name: Optional[str] = None, temperature: float = 0.0, api_base: Optional[str] = None):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model_name or settings.OPENAI_MODEL,
        temperature=temperature,
        openai_api_base=api_base or settings.OPENAI_API_BASE,
        openai_api_key=settings.OPENAI_API_KEY,
        streaming=True,
    )


async def _invoke_structured(llm, messages: list, schema: type) -> str:
    """
    Invoke LLM with json_schema constrained output.
    Falls back to unconstrained invocation if the server rejects the format.
    strict=False intentional: local models often refuse strict mode.
    """
    json_schema_format = {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "strict": False,
            "schema": schema.model_json_schema(),
        },
    }
    try:
        resp = await llm.ainvoke(messages, response_format=json_schema_format)
    except Exception as exc:
        err = str(exc)
        if "response_format" in err or "json_schema" in err or "400" in err:
            resp = await llm.ainvoke(messages)
        else:
            raise
    return resp.content


def _content_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()


def _serialise_doc(doc: Any) -> dict:
    if isinstance(doc, dict):
        return doc
    if hasattr(doc, "page_content"):
        return {"page_content": doc.page_content, "metadata": dict(doc.metadata)}
    return {"page_content": str(doc), "metadata": {}}


def _dedup_and_reinforce(doc_lists: List[List[dict]]) -> List[dict]:
    """
    Merge multiple lists of serialised docs.
    A chunk found in N sub-query results has its score multiplied (reinforced).
    Returns sorted by reinforced score descending.
    """
    seen: Dict[str, dict] = {}
    for docs in doc_lists:
        for doc in docs:
            text = doc.get("page_content", "")
            h = _content_hash(text)
            meta = doc.get("metadata", {})
            score = float(meta.get("_rrf_score", meta.get("score", 0.001)))
            if h in seen:
                prev_meta = seen[h].get("metadata", {})
                prev_score = float(prev_meta.get("_reinforced_score", score))
                new_meta = dict(prev_meta)
                new_meta["_reinforced_score"] = prev_score + score
                new_meta["_retrieval_count"] = prev_meta.get("_retrieval_count", 1) + 1
                seen[h] = {"page_content": text, "metadata": new_meta}
            else:
                new_meta = dict(meta)
                new_meta["_reinforced_score"] = score
                new_meta["_retrieval_count"] = 1
                seen[h] = {"page_content": text, "metadata": new_meta}

    result = list(seen.values())
    result.sort(key=lambda d: d.get("metadata", {}).get("_reinforced_score", 0), reverse=True)
    return result


def _build_context_string(docs: List[dict], file_markdown: Optional[str] = None) -> str:
    """
    Build context string for the LLM.

    Chat history docs (metadata._source_type == 'chat_history') are placed first
    under a plain [Prior Answer] label with no number — the LLM is instructed
    not to cite them. KB chunks follow with sequential [KB-N] numbering so
    citations remain consistent.
    """
    parts: List[str] = []

    # ── Chat history docs first — no citation number ───────────────────────────
    for doc in docs:
        if doc.get("metadata", {}).get("_source_type") == "chat_history":
            content = doc.get("page_content", "").strip()
            parts.append(f"[Prior Answer]\n{content}")

    # ── KB chunks with sequential numbers for citations ─────────────────────
    kb_counter = 0
    for doc in docs:
        if doc.get("metadata", {}).get("_source_type") != "chat_history":
            kb_counter += 1
            content = doc.get("page_content", "").strip()
            meta = doc.get("metadata", {})
            source = meta.get("source") or meta.get("file_name", "")
            header = f"[KB-{kb_counter}]" + (f" ({source})" if source else "")
            parts.append(f"{header}\n{content}")

    if file_markdown and file_markdown.strip():
        parts.append(f"[FILE CONTENT]\n{file_markdown.strip()}")

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Node: rewrite_query_node  (preserved from v1)
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
    api_base = state.get("api_base")
    query_model_override = state.get("query_model")
    temperature = state.get("temperature", 0.0)

    # ── Abbreviation expansion (before LLM rewrite) ───────────────────────
    from app.services.query_expander import expand, load_org_abbreviations
    org_id = state.get("org_id")
    _db = state.get("_db")
    abbreviations = load_org_abbreviations(org_id, _db) if _db is not None else {}
    if abbreviations:
        expanded = expand(query, abbreviations)
        if expanded != query:
            logger.info("[EXPAND] org_id=%s substitutions=%d", org_id, len(abbreviations))
            query = expanded

    llm = _get_llm(model_name, temperature, api_base=api_base)

    system_prompt = (
        "You are a search query optimizer for a document retrieval system. "
        "Rewrite the user's question into a concise, keyword-rich retrieval query "
        "that maximises recall from a vector database. "
        "Use conversation history only to resolve pronouns or topic references — "
        "never to answer or evaluate the question. "
        "Return ONLY the rewritten query — no explanations, no quotes, no answers."
    )
    history = state.get("recent_lc_history") or []
    messages: list = [{"role": "system", "content": system_prompt}]
    for msg in history[-4:]:
        messages.append(msg)
    messages.append({"role": "user", "content": query})

    try:
        response = await llm.ainvoke(messages)
        rewritten = response.content.strip() or query
    except Exception as exc:
        logger.warning("[REWRITE] failed: %s — using original", exc)
        rewritten = query

    latency_ms = round((time.monotonic() - t0) * 1000, 2)
    logger.info("[REWRITE] original=%r rewritten=%r latency_ms=%.1f", query[:60], rewritten[:60], latency_ms)

    step = {
        "node": "rewrite_query", "latency_ms": latency_ms, "status": "done",
        "rewritten_query": rewritten,
    }
    return {
        "rewritten_query": rewritten,
        "agent_steps": (state.get("agent_steps") or []) + [step],
    }


# ---------------------------------------------------------------------------
# Node: context_router_node  (preserved from v1)
# ---------------------------------------------------------------------------

_ROUTER_SYSTEM_PROMPT = """\
You are a RAG context router. Given a user query and the available context sources,
decide which sources are needed to answer the question.

Available sources:
- "kb"           — knowledge base (vector store with embedded documents)
- "file_current" — the file currently open / being discussed in this chat turn
- "file_prior"   — files previously uploaded in earlier turns of this conversation
- "chat_history" — previous assistant responses in this conversation

Respond with a JSON object:
{
  "sources": ["kb"],
  "rationale": "...",
  "file_ids_needed": []
}

Rules:
- Always include at least one source.
- Include "kb" whenever general knowledge or knowledge-base content is useful.
- Include "file_current" only when the query asks about the current file/document.
- Include "file_prior" only when the query references earlier uploaded files by ID or content.
- Include "chat_history" when the query explicitly references a prior assistant response,
  asks to expand/elaborate/repeat something said earlier, or uses references like
  "your previous answer", "as you mentioned", "that example", "those points", "earlier".
- file_ids_needed must be integer IDs, not strings.
"""


async def context_router_node(state: RAGGraphState) -> dict:
    """
    Decides which context sources to use: kb, file_current, file_prior.

    Logs: [ROUTER] sources=[...] rationale=... file_ids_needed=[...]
    Updates: sources, file_ids_needed, router_rationale, agent_steps
    """
    t0 = time.monotonic()
    model_name = state.get("model_name")
    api_base = state.get("api_base")
    query_model_override = state.get("query_model")
    temperature = state.get("temperature", 0.0)
    json_llm = _get_llm(model_name, 0.0, api_base=api_base)
    # Router sees BOTH the original and rewritten query.
    # The rewriter strips conversational references ("your previous answer",
    # "as you mentioned") because they have no vector-search value — but those
    # references are exactly what the router needs to decide whether to include
    # chat_history as a source.
    original_query = state["query"]
    rewritten_query = state.get("rewritten_query") or original_query
    user_msg = f"Original query: {original_query}"
    if rewritten_query != original_query:
        user_msg += f"\nRewritten for retrieval: {rewritten_query}"
    if state.get("file_markdown"):
        user_msg += "\n\n[A file is currently attached to this conversation.]"
    if state.get("existing_summary"):
        user_msg += "\n\n[There is a conversation summary from earlier turns.]"
    history = state.get("recent_lc_history") or []
    from langchain_core.messages import AIMessage as _AIMsg
    prior_answers = [m for m in history if isinstance(m, _AIMsg) and m.content.strip()]
    if prior_answers:
        user_msg += f"\n\n[There are {len(prior_answers)} prior assistant response(s) in this conversation.]"

    messages = [
        {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    try:
        raw = await _invoke_structured(json_llm, messages, _RouterOutput)
        parsed = json.loads(raw)
        sources: List[str] = parsed.get("sources") or ["kb"]
        rationale: str = parsed.get("rationale", "")
        file_ids_needed: List[int] = [int(x) for x in (parsed.get("file_ids_needed") or [])]
    except Exception as exc:
        logger.warning("[ROUTER] JSON parse error: %s — falling back to kb-only", exc)
        sources = ["kb"]
        rationale = "fallback due to parse error"
        file_ids_needed = []

    latency_ms = round((time.monotonic() - t0) * 1000, 2)
    logger.info(
        "[ROUTER] sources=%s rationale=%r file_ids_needed=%s latency_ms=%.1f",
        sources, rationale, file_ids_needed, latency_ms,
    )

    step = {
        "node": "context_router", "latency_ms": latency_ms, "status": "done",
        "sources": sources, "rationale": rationale,
    }
    return {
        "sources": sources,
        "file_ids_needed": file_ids_needed,
        "router_rationale": rationale,
        "agent_steps": (state.get("agent_steps") or []) + [step],
    }


# ---------------------------------------------------------------------------
# Node: chat_history_retrieval_node
# ---------------------------------------------------------------------------

async def chat_history_retrieval_node(state: RAGGraphState) -> dict:
    """
    Scores prior assistant answers against the current query using the reranker.
    Only runs when context_router includes "chat_history" in sources.

    Relevant prior answers are stored in chat_history_docs with
    metadata._source_type="chat_history" so _build_context_string labels them
    [Prior Answer] (no citation number) and the generator knows not to cite them.

    Logs: [CHAT_HIST] candidates=... passed_threshold=... latency_ms=...
    Updates: chat_history_docs, agent_steps
    """
    t0 = time.monotonic()
    sources: List[str] = state.get("sources") or []

    if "chat_history" not in sources:
        step = {"node": "chat_history_retrieval", "status": "skipped", "latency_ms": 0}
        return {
            "chat_history_docs": [],
            "agent_steps": (state.get("agent_steps") or []) + [step],
        }

    # Score prior answers against the ORIGINAL query, not the rewritten one.
    # The rewriter strips conversational references, but the reranker needs the
    # full semantic intent to correctly judge whether a prior answer is relevant.
    query = state["query"]
    history = state.get("recent_lc_history") or []

    from langchain_core.messages import AIMessage
    from langchain_core.documents import Document as LangchainDocument

    # Extract assistant turns only
    assistant_turns = [
        (i, msg) for i, msg in enumerate(history)
        if isinstance(msg, AIMessage) and msg.content.strip()
    ]

    if not assistant_turns:
        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        step = {"node": "chat_history_retrieval", "status": "no_history", "latency_ms": latency_ms}
        return {
            "chat_history_docs": [],
            "agent_steps": (state.get("agent_steps") or []) + [step],
        }

    # Build LangchainDocument per prior answer.
    # Truncate at 1800 chars (~460 tokens) to stay within the reranker's
    # 512-token window (query ~50 tokens + passage ~460 tokens).
    # Answers front-load key content so tail truncation is safe.
    docs = [
        LangchainDocument(
            page_content=msg.content.strip()[:1800],
            metadata={"_source_type": "chat_history", "turn": idx, "source": "chat_history"},
        )
        for idx, msg in assistant_turns
    ]

    # Reranker-disabled fallback: include last 2 answers directly
    if not settings.RERANKER_ENABLED:
        result_docs = [_serialise_doc(d) for d in docs[-2:]]
        for d in result_docs:
            d["metadata"]["_reranker_score"] = 0.0
        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        step = {
            "node": "chat_history_retrieval", "status": "done_no_reranker",
            "latency_ms": latency_ms, "candidates": len(docs), "docs_found": len(result_docs),
        }
        return {
            "chat_history_docs": result_docs,
            "agent_steps": (state.get("agent_steps") or []) + [step],
        }

    # Score each prior answer against the query.
    # Threshold 0.0: empirically relevant answers score 9+, irrelevant score -11.
    # 0.0 cleanly separates them with no ambiguous middle ground.
    try:
        from app.services.reranker import rerank
        ranked = rerank(query=query, docs=docs, score_threshold=0.0)
    except Exception as exc:
        logger.warning("[CHAT_HIST] reranker failed: %s — skipping chat history", exc)
        ranked = []

    result_docs = [_serialise_doc(d) for d in ranked]

    latency_ms = round((time.monotonic() - t0) * 1000, 1)
    logger.info(
        "[CHAT_HIST] query=%r | candidates=%d | passed=%d | latency_ms=%.1f",
        query[:60], len(docs), len(result_docs), latency_ms,
    )

    step = {
        "node": "chat_history_retrieval", "status": "done",
        "latency_ms": latency_ms,
        "candidates": len(docs),
        "docs_found": len(result_docs),
        "scores": [round(d["metadata"].get("_reranker_score", 0), 3) for d in result_docs],
    }
    return {
        "chat_history_docs": result_docs,
        "agent_steps": (state.get("agent_steps") or []) + [step],
    }


# ---------------------------------------------------------------------------
# Node: decompose_query_node  ← NEW
# ---------------------------------------------------------------------------

_DECOMPOSE_SYSTEM = """\
You are a query decomposer. Break a complex question into 2–5 atomic sub-questions
that together fully cover the original question.

Rules:
- Simple, single-fact questions → produce exactly 1 sub-question (the question itself, unchanged).
- Each sub-question must be self-contained and independently answerable.
- Avoid overlap between sub-questions.
- Maximum 5 sub-questions.

Respond with JSON only: {"sub_queries": ["...", "..."]}
"""


async def decompose_query_node(state: RAGGraphState) -> dict:
    t0 = time.monotonic()
    query = state.get("rewritten_query") or state["query"]
    model_name = state.get("model_name")
    api_base = state.get("api_base")
    query_model_override = state.get("query_model")
    llm = _get_llm((model_name or query_model_override or settings.QUERY_MODEL), 0.0, api_base=api_base)

    messages = [
        {"role": "system", "content": _DECOMPOSE_SYSTEM},
        {"role": "user", "content": f"Question: {query}"},
    ]
    sub_queries = [query]
    try:
        raw = await _invoke_structured(llm, messages, _SubQueriesOutput)
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            parsed = json.loads(m.group(0))
            sqs = [q.strip() for q in (parsed.get("sub_queries") or []) if q.strip()]
            if sqs:
                sub_queries = sqs[:5]
    except Exception as exc:
        logger.warning("[DECOMPOSE] parse failed: %s — using single sub-query", exc)

    latency_ms = round((time.monotonic() - t0) * 1000, 1)
    logger.info("[DECOMPOSE] latency_ms=%.1f sub_queries=%d: %s",
                latency_ms, len(sub_queries), sub_queries)

    step = {
        "node": "decompose_query", "status": "done", "latency_ms": latency_ms,
        "sub_queries": sub_queries,
    }
    return {
        "sub_queries": sub_queries,
        "agent_steps": (state.get("agent_steps") or []) + [step],
    }


# ---------------------------------------------------------------------------
# Node: parallel_retrieval_node  ← replaces kb_retrieval_node
# ---------------------------------------------------------------------------

async def parallel_retrieval_node(state: RAGGraphState) -> dict:
    t0 = time.monotonic()
    sources: List[str] = state.get("sources") or ["kb"]

    if "kb" not in sources:
        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        step = {
            "node": "parallel_retrieval", "status": "skipped", "latency_ms": latency_ms,
            "docs_found": 0, "sub_queries_searched": 0,
        }
        return {
            "retrieved_docs": state.get("retrieved_docs") or [],
            "retrieval_attempt": 0,
            "agent_steps": (state.get("agent_steps") or []) + [step],
        }

    sub_queries: List[str] = state.get("sub_queries") or [state.get("rewritten_query") or state["query"]]
    kb_ids: List[int] = state.get("knowledge_base_ids") or []
    db = state.get("_db")
    use_dense = state.get("use_dense", True)
    use_sparse = state.get("use_sparse", True)
    use_exact = state.get("use_exact", True)
    use_graph_rag = state.get("use_graph_rag", False)

    # Get linked datastores for these KBs
    datastore_ids = []
    if kb_ids and db:
        from app.models.knowledge import KnowledgeBaseDataStore
        datastore_links = (
            db.query(KnowledgeBaseDataStore.data_store_id)
            .filter(KnowledgeBaseDataStore.knowledge_base_id.in_(kb_ids))
            .distinct()
            .all()
        )
        datastore_ids = [row.data_store_id for row in datastore_links]
        if datastore_ids:
            logger.info("[RETRIEVE] Found %d linked datastores for KBs %s", len(datastore_ids), kb_ids)

    from app.services.retrieval import hybrid_search_with_legs

    async def _retrieve_one(sq: str) -> List[dict]:
        try:
            result = await hybrid_search_with_legs(
                query=sq, kb_ids=kb_ids, db=db,
                use_dense=use_dense, use_sparse=use_sparse,
                use_exact=use_exact, use_graph_rag=use_graph_rag,
                datastore_ids=datastore_ids,
            )
            return [_serialise_doc(d) for d in result.get("docs", [])]
        except Exception as exc:
            logger.error("[PARALLEL_RETRIEVE] sub_query=%r error: %s", sq[:60], exc)
            return []

    results = await asyncio.gather(*[_retrieve_one(sq) for sq in sub_queries])
    merged = _dedup_and_reinforce(list(results))

    latency_ms = round((time.monotonic() - t0) * 1000, 1)
    logger.info("[PARALLEL_RETRIEVE] sub_queries=%d docs_merged=%d latency_ms=%.1f",
                len(sub_queries), len(merged), latency_ms)

    chunk_previews = []
    for doc in merged[:10]:
        meta = doc.get("metadata", {})
        text = doc.get("page_content", "")
        preview = text[:120].strip() + ("…" if len(text) > 120 else "")
        chunk_previews.append({
            "preview": preview,
            "source": meta.get("source") or meta.get("file_name", ""),
            "score": round(float(meta.get("_reinforced_score", 0)), 4),
            "retrieval_count": int(meta.get("_retrieval_count", 1)),
        })

    # Prepend chat history docs (already reranker-scored, no RRF scoring).
    # They go at the front so they're always in the [:20] context slice.
    # _build_context_string labels them [Prior Answer] (no citation number).
    chat_history_docs: list = state.get("chat_history_docs") or []
    if chat_history_docs:
        merged = chat_history_docs + merged
        logger.info("[PARALLEL_RETRIEVE] prepended %d chat_history_docs, total=%d",
                    len(chat_history_docs), len(merged))

    step = {
        "node": "parallel_retrieval", "status": "done", "latency_ms": latency_ms,
        "docs_found": len(merged), "sub_queries_searched": len(sub_queries),
        "chunks": chunk_previews,
    }
    return {
        "retrieved_docs": merged,
        "retrieval_attempt": 0,
        "agent_steps": (state.get("agent_steps") or []) + [step],
    }


# ---------------------------------------------------------------------------
# Node: extract_file_sections_node  (preserved from v1, adapted for sub_queries)
# ---------------------------------------------------------------------------

async def extract_file_sections_node(state: RAGGraphState) -> dict:
    """
    Extracts the most relevant sections from the attached file markdown.
    Uses the combined sub-queries for section selection (preserved v1 logic).

    Logs: [EXTRACT] chars_in=... sections=... chars_out=... latency_ms=...
    Updates: file_markdown (trimmed), agent_steps
    """
    t0 = time.monotonic()
    sources: List[str] = state.get("sources") or []
    file_markdown: Optional[str] = state.get("file_markdown")

    if not any(s.startswith("file") for s in sources) or not file_markdown:
        latency_ms = round((time.monotonic() - t0) * 1000, 2)
        step = {
            "node": "extract_file_sections", "latency_ms": latency_ms,
            "status": "skipped", "sections": 0,
        }
        return {"agent_steps": (state.get("agent_steps") or []) + [step]}

    # Combine sub-queries for richer section selection
    sub_queries: List[str] = state.get("sub_queries") or [state.get("rewritten_query") or state["query"]]
    query = " ".join(sub_queries[:3])
    model_name = state.get("model_name")
    api_base = state.get("api_base")
    query_model_override = state.get("query_model")
    llm = _get_llm(model_name, 0.0, api_base=api_base)

    # Split into sections by markdown headings or double-newlines
    raw_sections = re.split(r"\n(?=#{1,3} )", file_markdown)
    if len(raw_sections) <= 1:
        raw_sections = [s.strip() for s in file_markdown.split("\n\n") if s.strip()]

    chars_in = len(file_markdown)

    # If the file is small enough, keep it all (passthrough)
    MAX_CHARS = 12_000
    if chars_in <= MAX_CHARS:
        latency_ms = round((time.monotonic() - t0) * 1000, 2)
        logger.info(
            "[EXTRACT] chars_in=%d sections=%d chars_out=%d latency_ms=%.1f (passthrough)",
            chars_in, len(raw_sections), chars_in, latency_ms,
        )
        step = {
            "node": "extract_file_sections", "latency_ms": latency_ms,
            "status": "passthrough", "sections": len(raw_sections),
        }
        return {"agent_steps": (state.get("agent_steps") or []) + [step]}

    # Use LLM to pick the top sections most relevant to the combined query
    previews = "\n".join(
        f"[{i}] {s[:200].strip()}" for i, s in enumerate(raw_sections)
    )
    messages = [
        {"role": "system", "content": (
            "You are a document section selector. "
            "Given a query and a numbered list of document sections, "
            "return a JSON object: {\"indices\": [<list of 3-6 most relevant section indices>]}. "
            "Return ONLY the JSON object."
        )},
        {"role": "user", "content": f"Query: {query}\n\nSections:\n{previews}"},
    ]

    selected_sections = raw_sections  # fallback: keep all
    try:
        raw = await _invoke_structured(llm, messages, _SectionOutput)
        m = re.search(r'\{.*?\}', raw, re.DOTALL)
        if m:
            parsed = json.loads(m.group(0))
            indices: List[int] = [int(i) for i in (parsed.get("indices") or [])]
            if indices:
                selected_sections = [raw_sections[i] for i in indices if 0 <= i < len(raw_sections)]
    except Exception as exc:
        logger.warning("[EXTRACT] section selection failed: %s — keeping all sections", exc)

    trimmed = "\n\n".join(selected_sections)
    latency_ms = round((time.monotonic() - t0) * 1000, 2)
    logger.info(
        "[EXTRACT] chars_in=%d sections_total=%d sections_kept=%d chars_out=%d latency_ms=%.1f",
        chars_in, len(raw_sections), len(selected_sections), len(trimmed), latency_ms,
    )

    step = {
        "node": "extract_file_sections", "latency_ms": latency_ms, "status": "done",
        "sections_total": len(raw_sections), "sections_kept": len(selected_sections),
    }
    return {
        "file_markdown": trimmed,
        "agent_steps": (state.get("agent_steps") or []) + [step],
    }


# ---------------------------------------------------------------------------
# Node: draft_answer_node  ← NEW
# ---------------------------------------------------------------------------

_DRAFT_SYSTEM = """\
You are a research assistant. Using ONLY the provided context chunks, write a structured
draft answer with one section per sub-question.

Label each section: ### Sub-question N: <text>

If the context contains no information for a sub-question, write:
### Sub-question N: <text>
[NO INFORMATION FOUND]

Use inline citations [N](N) when you reference a knowledge base chunk [KB-N].
[Prior Answer] sections are previous conversation context — use them freely but do NOT cite them.
Keep each section concise (2-4 sentences). This draft is for internal quality grading only.
"""


async def draft_answer_node(state: RAGGraphState) -> dict:
    t0 = time.monotonic()
    sub_queries: List[str] = state.get("sub_queries") or [state.get("rewritten_query") or state["query"]]
    docs: list = state.get("retrieved_docs") or []
    file_markdown: Optional[str] = state.get("file_markdown")
    model_name = state.get("model_name")
    api_base = state.get("api_base")
    query_model_override = state.get("query_model")
    llm = _get_llm(model_name, 0.0, api_base=api_base)

    context = _build_context_string(docs[:20], file_markdown)
    sub_q_block = "\n".join(f"{i+1}. {sq}" for i, sq in enumerate(sub_queries))

    messages = [
        {"role": "system", "content": _DRAFT_SYSTEM},
        {"role": "user", "content": (
            f"Sub-questions to answer:\n{sub_q_block}\n\n"
            f"Context:\n{context or '[No context available]'}"
        )},
    ]

    draft = ""
    try:
        response = await llm.ainvoke(messages)
        draft = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        logger.error("[DRAFT] generation failed: %s", exc)
        draft = "[DRAFT FAILED]"

    latency_ms = round((time.monotonic() - t0) * 1000, 1)
    logger.info("[DRAFT] latency_ms=%.1f chars=%d", latency_ms, len(draft))

    step = {
        "node": "draft_answer", "status": "done", "latency_ms": latency_ms,
        "draft_chars": len(draft),
    }
    return {
        "draft_answer": draft,
        "agent_steps": (state.get("agent_steps") or []) + [step],
    }


# ---------------------------------------------------------------------------
# Node: grade_coverage_node  ← replaces grade_documents_node
# ---------------------------------------------------------------------------

_GRADE_COVERAGE_SYSTEM = """\
You are a coverage grader. Given a draft answer and the original sub-questions,
determine whether each sub-question is answered.

For each sub-question, assign one of:
  "covered"           — clearly and fully answered with supporting detail
  "partially_covered" — addressed but lacking depth or specifics
  "not_covered"       — no useful answer found or explicitly stated as not found

Respond with JSON only:
{"coverage": [{"sub_query": "...", "status": "covered|partially_covered|not_covered"}, ...]}
"""


async def grade_coverage_node(state: RAGGraphState) -> dict:
    t0 = time.monotonic()
    sub_queries: List[str] = state.get("sub_queries") or [state.get("rewritten_query") or state["query"]]
    draft: str = state.get("draft_answer") or ""
    model_name = state.get("model_name")
    api_base = state.get("api_base")
    query_model_override = state.get("query_model")
    llm = _get_llm((model_name or query_model_override or settings.QUERY_MODEL), 0.0, api_base=api_base)
    attempt = state.get("retrieval_attempt", 0)

    if not draft or draft == "[DRAFT FAILED]":
        coverage_result = {sq: "not_covered" for sq in sub_queries}
        uncovered = list(sub_queries)
        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        step = {
            "node": "grade_coverage", "status": "done", "latency_ms": latency_ms,
            "coverage": coverage_result, "coverage_lines": [f"✗ {sq}" for sq in uncovered],
            "uncovered_count": len(uncovered), "attempt": attempt,
        }
        return {
            "coverage_result": coverage_result,
            "uncovered_sub_queries": uncovered,
            "agent_steps": (state.get("agent_steps") or []) + [step],
        }

    sub_q_block = "\n".join(f"{i+1}. {sq}" for i, sq in enumerate(sub_queries))
    messages = [
        {"role": "system", "content": _GRADE_COVERAGE_SYSTEM},
        {"role": "user", "content": (
            f"Sub-questions:\n{sub_q_block}\n\n"
            f"Draft answer:\n{draft[:3000]}"
        )},
    ]

    coverage_result: dict = {}
    uncovered: List[str] = []
    try:
        raw = await _invoke_structured(llm, messages, _CoverageOutput)
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            parsed = json.loads(m.group(0))
            for item in (parsed.get("coverage") or []):
                sq = item.get("sub_query", "")
                status = item.get("status", "not_covered")
                coverage_result[sq] = status
                if status == "not_covered":
                    uncovered.append(sq)
    except Exception as exc:
        logger.warning("[GRADE_COVERAGE] parse failed: %s — marking all covered", exc)
        coverage_result = {sq: "covered" for sq in sub_queries}
        uncovered = []

    # Fill any sub-query the LLM didn't return
    for sq in sub_queries:
        if sq not in coverage_result:
            coverage_result[sq] = "not_covered"
            if sq not in uncovered:
                uncovered.append(sq)

    coverage_lines = []
    for sq, status in coverage_result.items():
        icon = "✓" if status == "covered" else ("~" if status == "partially_covered" else "✗")
        coverage_lines.append(f"{icon} {sq}")

    latency_ms = round((time.monotonic() - t0) * 1000, 1)
    logger.info(
        "[GRADE_COVERAGE] attempt=%d covered=%d not_covered=%d latency_ms=%.1f",
        attempt,
        sum(1 for s in coverage_result.values() if s == "covered"),
        len(uncovered), latency_ms,
    )

    step = {
        "node": "grade_coverage", "status": "done", "latency_ms": latency_ms,
        "coverage": coverage_result, "coverage_lines": coverage_lines,
        "uncovered_count": len(uncovered), "attempt": attempt,
    }
    return {
        "coverage_result": coverage_result,
        "uncovered_sub_queries": uncovered,
        "agent_steps": (state.get("agent_steps") or []) + [step],
    }


# ---------------------------------------------------------------------------
# Conditional router: post grade_coverage
# ---------------------------------------------------------------------------

def _route_after_grade(state: RAGGraphState) -> str:
    uncovered: List[str] = state.get("uncovered_sub_queries") or []
    attempt: int = state.get("retrieval_attempt", 0)

    if not uncovered:
        return "generate_answer"
    if attempt == 0:
        return "widened_retrieval"
    if attempt == 1:
        return "keyword_search_loop"
    return "generate_answer"


# ---------------------------------------------------------------------------
# Node: widened_retrieval_node  ← NEW (Retry 1)
# ---------------------------------------------------------------------------

async def widened_retrieval_node(state: RAGGraphState) -> dict:
    t0 = time.monotonic()
    uncovered: List[str] = state.get("uncovered_sub_queries") or []
    kb_ids: List[int] = state.get("knowledge_base_ids") or []
    db = state.get("_db")
    use_graph_rag = state.get("use_graph_rag", False)

    if not uncovered or not kb_ids:
        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        step = {"node": "widened_retrieval", "status": "skipped", "latency_ms": latency_ms}
        return {
            "retrieval_attempt": 1,
            "agent_steps": (state.get("agent_steps") or []) + [step],
        }

    logger.info("[WIDENED] re-retrieving for %d uncovered sub-queries (attempt 1)", len(uncovered))

    # Get linked datastores for these KBs
    datastore_ids = []
    if kb_ids and db:
        from app.models.knowledge import KnowledgeBaseDataStore
        datastore_links = (
            db.query(KnowledgeBaseDataStore.data_store_id)
            .filter(KnowledgeBaseDataStore.knowledge_base_id.in_(kb_ids))
            .distinct()
            .all()
        )
        datastore_ids = [row.data_store_id for row in datastore_links]
        if datastore_ids:
            logger.info("[WIDENED] Found %d linked datastores for KBs %s", len(datastore_ids), kb_ids)

    from app.services.retrieval import hybrid_search_with_legs

    async def _widened_one(sq: str) -> List[dict]:
        try:
            # All legs, no query_type preset (uses global top_k but pool is 4× so wider)
            result = await hybrid_search_with_legs(
                datastore_ids=datastore_ids,
                query=sq, kb_ids=kb_ids, db=db,
                use_dense=True, use_sparse=True, use_exact=True,
                use_graph_rag=use_graph_rag,
            )
            docs = [_serialise_doc(d) for d in result.get("docs", [])]
            # Apply reranker with very relaxed threshold (-5.0 vs default -2.0)
            # so weaker-matching chunks pass through
            if settings.RERANKER_ENABLED and docs:
                try:
                    from app.services.reranker import rerank as _rerank
                    lc_docs = [
                        type("_D", (), {
                            "page_content": d["page_content"],
                            "metadata": d.get("metadata", {}),
                        })()
                        for d in docs
                    ]
                    reranked = _rerank(query=sq, docs=lc_docs, score_threshold=-5.0)
                    docs = [_serialise_doc(d) for d in reranked]
                except Exception:
                    pass  # keep RRF order
            return docs
        except Exception as exc:
            logger.error("[WIDENED] error for %r: %s", sq[:60], exc)
            return []

    results = await asyncio.gather(*[_widened_one(sq) for sq in uncovered])
    new_docs = _dedup_and_reinforce(list(results))
    prior_docs: list = state.get("retrieved_docs") or []
    all_merged = _dedup_and_reinforce([prior_docs, new_docs])

    latency_ms = round((time.monotonic() - t0) * 1000, 1)
    logger.info("[WIDENED] new_docs=%d merged_total=%d latency_ms=%.1f",
                len(new_docs), len(all_merged), latency_ms)

    step = {
        "node": "widened_retrieval", "status": "done", "latency_ms": latency_ms,
        "uncovered_sub_queries": uncovered,
        "new_docs_found": len(new_docs),
        "total_docs": len(all_merged),
        "threshold_relaxed_to": -5.0,
    }
    return {
        "retrieved_docs": all_merged,
        "retrieval_attempt": 1,
        "agent_steps": (state.get("agent_steps") or []) + [step],
    }


# ---------------------------------------------------------------------------
# Node: keyword_search_loop_node  ← NEW (Retry 2)
# ---------------------------------------------------------------------------

_KEYWORD_EXTRACT_SYSTEM = """\
Extract search keywords from a question for a MySQL full-text keyword search.

Return two sets:
- broad_keywords:  3-4 general terms likely to appear in relevant documents
- narrow_keywords: 1-2 specific compound phrases that precisely identify the topic

Respond with JSON only:
{"broad_keywords": ["term1", "term2"], "narrow_keywords": ["compound phrase"]}
"""


async def keyword_search_loop_node(state: RAGGraphState) -> dict:
    t0 = time.monotonic()
    uncovered: List[str] = state.get("uncovered_sub_queries") or []
    kb_ids: List[int] = state.get("knowledge_base_ids") or []
    db = state.get("_db")
    model_name = state.get("model_name")
    api_base = state.get("api_base")
    query_model_override = state.get("query_model")

    if not uncovered or not kb_ids or not db:
        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        step = {"node": "keyword_search_loop", "status": "skipped", "latency_ms": latency_ms}
        return {
            "retrieval_attempt": 2,
            "agent_steps": (state.get("agent_steps") or []) + [step],
        }

    from app.services.retrieval import _exact_search  # type: ignore[attr-defined]
    llm = _get_llm((model_name or query_model_override or settings.QUERY_MODEL), 0.0, api_base=api_base)

    all_new_docs: List[dict] = []
    keyword_iterations: list = list(state.get("keyword_iterations") or [])

    # Cap: 3 uncovered sub-queries max, 2 iterations each = 6 keyword searches ceiling
    for sq in uncovered[:3]:
        messages = [
            {"role": "system", "content": _KEYWORD_EXTRACT_SYSTEM},
            {"role": "user", "content": f"Question: {sq}"},
        ]
        broad_kw: List[str] = []
        narrow_kw: List[str] = []
        try:
            raw = await _invoke_structured(llm, messages, _KeywordsOutput)
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                parsed = json.loads(m.group(0))
                broad_kw = [str(k) for k in (parsed.get("broad_keywords") or [])]
                narrow_kw = [str(k) for k in (parsed.get("narrow_keywords") or [])]
        except Exception as exc:
            logger.warning("[KEYWORD] keyword extraction failed for %r: %s", sq[:60], exc)
            broad_kw = sq.split()[:4]

        # Iteration 1: broad keyword search
        broad_query = " ".join(broad_kw)
        broad_candidates: dict = {}
        if broad_query.strip():
            try:
                broad_candidates = _exact_search(broad_query, kb_ids, db, 20)
            except Exception as exc:
                logger.warning("[KEYWORD] broad search failed: %s", exc)

        broad_docs = [_serialise_doc(c.doc) for c in broad_candidates.values()]
        keyword_iterations.append({
            "sub_query": sq, "iteration": "broad",
            "keywords": broad_kw, "results_found": len(broad_docs),
        })

        if not broad_docs and narrow_kw:
            # Iteration 2: narrow keyword search
            narrow_query = " ".join(narrow_kw)
            narrow_candidates: dict = {}
            try:
                narrow_candidates = _exact_search(narrow_query, kb_ids, db, 15)
            except Exception as exc:
                logger.warning("[KEYWORD] narrow search failed: %s", exc)

            narrow_docs = [_serialise_doc(c.doc) for c in narrow_candidates.values()]
            keyword_iterations.append({
                "sub_query": sq, "iteration": "narrow",
                "keywords": narrow_kw, "results_found": len(narrow_docs),
            })
            all_new_docs.extend(narrow_docs)
        else:
            all_new_docs.extend(broad_docs)

    prior_docs: list = state.get("retrieved_docs") or []
    all_merged = _dedup_and_reinforce([prior_docs, all_new_docs]) if all_new_docs else prior_docs

    latency_ms = round((time.monotonic() - t0) * 1000, 1)
    logger.info("[KEYWORD] iterations=%d new_docs=%d total=%d latency_ms=%.1f",
                len(keyword_iterations), len(all_new_docs), len(all_merged), latency_ms)

    step = {
        "node": "keyword_search_loop", "status": "done", "latency_ms": latency_ms,
        "keyword_iterations": keyword_iterations,
        "new_docs_found": len(all_new_docs),
        "total_docs": len(all_merged),
    }
    return {
        "retrieved_docs": all_merged,
        "keyword_iterations": keyword_iterations,
        "retrieval_attempt": 2,
        "agent_steps": (state.get("agent_steps") or []) + [step],
    }


# ---------------------------------------------------------------------------
# Node: generate_answer_node  (final, enhanced)
# ---------------------------------------------------------------------------

_FINAL_ANSWER_SYSTEM = """\
You are a helpful assistant. Answer the user's question using ONLY the provided context.

FORMATTING RULES:
- Use ### headers to divide multi-part answers (e.g., "### 1. Definition", "### 2. How It Works").
- Use numbered lists for sequential steps or algorithms.
- Use bullet points with **bold terms** for features, attributes, or comparisons.
- Use inline code for variable names, identifiers, and technical terms (e.g., `wait()`, `Available[j]`).
- For simple single-concept questions, plain prose is fine — do not force structure.

CITATION RULES — follow exactly:
- When you use information from a knowledge base chunk labelled [KB-N], cite it inline
  as a markdown link using ONLY the number: [N](N)
- Example: "Process scheduling [1](1) involves saving CPU state [2](2)."
- Do NOT write [KB-1] or [KB-N] in your answer — write [1](1) or [N](N).
- Only cite chunks you actually used.
- [Prior Answer] sections are previous conversation context — use them freely but do NOT cite them.

If some parts of the question could not be answered from available context, clearly state:
"I could not find sufficient information about: [specific topic]"

If no relevant context was found at all, clearly state that.
"""


async def generate_answer_node(state: RAGGraphState) -> dict:
    t0 = time.monotonic()
    docs: list = state.get("retrieved_docs") or []
    file_markdown: Optional[str] = state.get("file_markdown")
    sub_queries: List[str] = state.get("sub_queries") or [state.get("rewritten_query") or state["query"]]
    uncovered: List[str] = state.get("uncovered_sub_queries") or []
    coverage_result: dict = state.get("coverage_result") or {}
    model_name = state.get("model_name")
    api_base = state.get("api_base")
    query_model_override = state.get("query_model")
    temperature: float = state.get("temperature", 0.0)
    existing_summary: Optional[str] = state.get("existing_summary")
    query = state.get("rewritten_query") or state["query"]
    attempt = state.get("retrieval_attempt", 0)

    llm = _get_llm(model_name, temperature, api_base=api_base)
    context = _build_context_string(docs[:20], file_markdown)

    # Build coverage note for partial / unable answers
    coverage_note = ""
    if uncovered and attempt >= 2:
        coverage_note = (
            "\n\nNote: Despite exhaustive search (widened retrieval + keyword search), "
            "the following sub-questions could not be answered from available documents:\n"
            + "\n".join(f"  - {sq}" for sq in uncovered)
        )

    # Generator context: summary only — same policy as fast/thinking pipeline.
    # Raw prior answers pollute context and cause the LLM to treat its own
    # previous responses as user statements. The summary provides all necessary
    # prior context in a clean, condensed form.
    messages: list = [{"role": "system", "content": _FINAL_ANSWER_SYSTEM + coverage_note}]
    if existing_summary:
        messages.append({"role": "system", "content": f"[Conversation summary so far]\n{existing_summary}"})

    messages.append({
        "role": "user",
        "content": (
            f"Question: {query}\n\nContext:\n{context}"
            if context else
            f"Question: {query}\n\n[No relevant documents found in the knowledge base]"
        ),
    })

    # NOTE: streaming tokens are captured by run_stream via on_chat_model_stream events
    # generate_answer_node stores the completed answer via ainvoke as a fallback
    streamed_parts: list = []
    usage: dict = {"promptTokens": 0, "completionTokens": 0}
    try:
        async for chunk in llm.astream(messages):
            token = chunk.content or ""
            if token:
                streamed_parts.append(token)
        if hasattr(llm, "_last_usage"):
            usage = llm._last_usage  # type: ignore[attr-defined]
    except Exception as exc:
        logger.error("[GENERATE] failed: %s", exc)
        streamed_parts = ["I encountered an error generating the response. Please try again."]

    answer = re.sub(
        r'\[KB-(\d+)\]',
        lambda m: f'[{m.group(1)}]({m.group(1)})',
        "".join(streamed_parts),
    )
    answer = re.sub(
        r'\[(\d+)\](?!\()',
        lambda m: f'[{m.group(1)}]({m.group(1)})',
        answer,
    )

    latency_ms = round((time.monotonic() - t0) * 1000, 1)
    logger.info("[GENERATE] latency_ms=%.1f chars=%d model=%s partial=%s",
                latency_ms, len(answer), model_name, bool(uncovered and attempt >= 2))

    step = {
        "node": "generate_answer", "status": "done", "latency_ms": latency_ms,
        "usage": usage,
        "partial": bool(uncovered and attempt >= 2),
    }
    return {
        "answer": answer,
        "_usage": usage,
        "agent_steps": (state.get("agent_steps") or []) + [step],
    }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

_KNOWN_NODES = {
    "rewrite_query", "context_router", "chat_history_retrieval", "decompose_query",
    "parallel_retrieval", "extract_file_sections",
    "draft_answer", "grade_coverage",
    "widened_retrieval", "keyword_search_loop", "generate_answer",
}


def _build_rag_graph():
    from langgraph.graph import StateGraph, END

    builder = StateGraph(RAGGraphState)

    builder.add_node("rewrite_query",         rewrite_query_node)
    builder.add_node("context_router",        context_router_node)
    builder.add_node("chat_history_retrieval", chat_history_retrieval_node)
    builder.add_node("decompose_query",       decompose_query_node)
    builder.add_node("parallel_retrieval",    parallel_retrieval_node)
    builder.add_node("extract_file_sections", extract_file_sections_node)
    builder.add_node("draft_answer",          draft_answer_node)
    builder.add_node("grade_coverage",        grade_coverage_node)
    builder.add_node("widened_retrieval",     widened_retrieval_node)
    builder.add_node("keyword_search_loop",   keyword_search_loop_node)
    builder.add_node("generate_answer",       generate_answer_node)

    # Entry
    builder.set_entry_point("rewrite_query")

    # Linear spine
    builder.add_edge("rewrite_query",          "context_router")
    builder.add_edge("context_router",         "chat_history_retrieval")
    builder.add_edge("chat_history_retrieval", "decompose_query")
    builder.add_edge("decompose_query",       "parallel_retrieval")
    builder.add_edge("parallel_retrieval",    "extract_file_sections")
    builder.add_edge("extract_file_sections", "draft_answer")
    builder.add_edge("draft_answer",          "grade_coverage")

    # Conditional routing from grade_coverage
    builder.add_conditional_edges(
        "grade_coverage",
        _route_after_grade,
        {
            "generate_answer":     "generate_answer",
            "widened_retrieval":   "widened_retrieval",
            "keyword_search_loop": "keyword_search_loop",
        },
    )

    # Retry loops: each escalation feeds back into draft → grade
    builder.add_edge("widened_retrieval",   "draft_answer")
    builder.add_edge("keyword_search_loop", "draft_answer")

    builder.add_edge("generate_answer", END)

    return builder.compile()


_rag_graph = _build_rag_graph()


# ---------------------------------------------------------------------------
# run_stream() — public async generator interface
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
    api_base: Optional[str] = None,
    query_model: Optional[str] = None,
    org_id: Optional[int] = None,
) -> AsyncGenerator[dict, None]:
    """
    Async generator that runs the full agentic RAG graph and streams events.

    Yield shapes (keyed by "event"):
      {"event": "agent_step",      "node": str, "status": "active"|"done",
                                   "latency_ms": float|None, ...node-specific fields}
      {"event": "rewritten_query", "query": str}
      {"event": "context",         "docs": list, "confidence": str, "score": float, ...}
      {"event": "token",           "content": str}
      {"event": "answer_rewrite",  "content": str}   (if citations normalised post-stream)
      {"event": "done",            "full_response": str, "usage": dict}
    """
    initial_state: RAGGraphState = {
        "query": query,
        "rewritten_query": "",
        "sub_queries": [],
        "sources": [],
        "chat_history_docs": [],
        "file_ids_needed": [],
        "router_rationale": "",
        "file_markdown": file_markdown,
        "retrieved_docs": [],
        "retrieval_attempt": 0,
        "keyword_iterations": [],
        "draft_answer": "",
        "coverage_result": {},
        "uncovered_sub_queries": [],
        "merged_context": "",
        "answer": "",
        "_usage": {},
        "agent_steps": [],
        "knowledge_base_ids": knowledge_base_ids,
        "recent_lc_history": recent_lc_history,
        "existing_summary": existing_summary,
        "use_dense": use_dense,
        "use_sparse": use_sparse,
        "use_exact": use_exact,
        "use_graph_rag": use_graph_rag,
        "temperature": temperature,
        "model_name": model_name,
        "display_query": display_query,
        "api_base": api_base,
        "query_model": query_model,
        "org_id": org_id,
        "_db": db,
    }

    final_state: dict = dict(initial_state)
    emitted_active_nodes: set = set()
    emitted_done_nodes: set = set()
    answer_streaming_started = False
    streamed_answer_parts: list = []

    async for event in _rag_graph.astream_events(initial_state, version="v2"):
        ev_name = event.get("event", "")
        ev_data = event.get("data", {})
        metadata = event.get("metadata", {})
        langgraph_node = metadata.get("langgraph_node", "")

        # ── Emit "active" the moment a known node starts ──────────────────
        if ev_name == "on_chain_start" and langgraph_node in _KNOWN_NODES:
            if langgraph_node not in emitted_active_nodes:
                emitted_active_nodes.add(langgraph_node)
                yield {
                    "event": EVENT_AGENT_STEP,
                    "node": langgraph_node,
                    "status": "active",
                    "latency_ms": None,
                }
            continue

        # ── Real-time token streaming from generate_answer ────────────────
        if ev_name == "on_chat_model_stream" and langgraph_node == "generate_answer":
            chunk = ev_data.get("chunk")
            token = ""
            if chunk is not None:
                if hasattr(chunk, "content"):
                    token = chunk.content or ""
                elif isinstance(chunk, dict):
                    token = chunk.get("content", "")
            if token:
                answer_streaming_started = True
                streamed_answer_parts.append(token)
                yield {"event": EVENT_TOKEN, "content": token}
            continue

        # ── Node completion: emit "done" step events ──────────────────────
        if ev_name == "on_chain_end":
            output = ev_data.get("output")
            if isinstance(output, dict):
                final_state.update(output)

                new_steps: list = output.get("agent_steps") or []
                for step in new_steps:
                    step_node = step.get("node", "")
                    if step_node and step_node not in emitted_done_nodes:
                        emitted_done_nodes.add(step_node)
                        yield {"event": EVENT_AGENT_STEP, **{k: v for k, v in step.items()}}

                # Emit rewritten_query event once
                if output.get("rewritten_query") and "rewrite_done" not in emitted_done_nodes:
                    emitted_done_nodes.add("rewrite_done")
                    yield {
                        "event": EVENT_REWRITTEN,
                        "query": output["rewritten_query"],
                    }

    # ── Post-graph: citation normalisation ───────────────────────────────
    if answer_streaming_started and streamed_answer_parts:
        raw_answer = "".join(streamed_answer_parts)
        # Normalise [KB-N] citations the LLM emits instead of [N](N)
        normalised = re.sub(r'\[KB-(\d+)\]', lambda m: f'[{m.group(1)}]({m.group(1)})', raw_answer)
        normalised = re.sub(r'\[(\d+)\](?!\()', lambda m: f'[{m.group(1)}]({m.group(1)})', normalised)
        if normalised != raw_answer:
            yield {"event": "answer_rewrite", "content": normalised}
        final_state["answer"] = normalised

    # ── Context event (summary of all retrieval) ─────────────────────────
    # Exclude chat_history_docs — conversational context, not citable KB sources.
    all_docs = final_state.get("retrieved_docs") or []
    docs = [d for d in all_docs if d.get("metadata", {}).get("_source_type") != "chat_history"]
    answer = final_state.get("answer") or ""
    coverage_result = final_state.get("coverage_result") or {}
    covered_count = sum(1 for s in coverage_result.values() if s == "covered")
    total_count = len(coverage_result) or 1
    confidence_score = covered_count / total_count

    yield {
        "event": EVENT_CONTEXT,
        "docs": docs,
        "confidence": (
            "high" if confidence_score >= 0.8
            else ("medium" if confidence_score >= 0.4 else "low")
        ),
        "score": round(confidence_score, 2),
        "suggestion": "" if docs else "No relevant documents found.",
        "failed_legs": [],
        "breakdown": {
            "kb_docs": len(docs),
            "file_chars": len(final_state.get("file_markdown") or ""),
            "merged_chars": len(answer),
            "sources": final_state.get("sources") or [],
            "sub_queries": final_state.get("sub_queries") or [],
            "retrieval_attempts": final_state.get("retrieval_attempt", 0) + 1,
        },
        "query_classification": {
            "type": "AGENTIC",
            "confidence": confidence_score,
            "latency_ms": 0,
            "fallback": False,
        },
        "tool_trace": [s.get("node") for s in (final_state.get("agent_steps") or [])],
        "synthesis_mode": len((final_state.get("sources") or [])) > 1,
    }

    # Fallback: if streaming didn't fire, emit answer as single token
    if not answer_streaming_started and answer:
        yield {"event": EVENT_TOKEN, "content": answer}

    usage = final_state.get("_usage") or {"promptTokens": 0, "completionTokens": 0}
    yield {"event": EVENT_DONE, "full_response": answer, "usage": usage}
