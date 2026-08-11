"""
Built-in tools for agentic RAG: search_documents, extract_entities, summarize_chunks.

Imported by chat_service at startup to register tools in the global registry.
All tools return JSON-serializable dicts and swallow internal exceptions,
returning {"error": str} instead of raising.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from app.services.tool_registry import register_tool

logger = logging.getLogger(__name__)


# ── search_documents ──────────────────────────────────────────────────────────

@register_tool(
    name="search_documents",
    description=(
        "Search the knowledge base for documents relevant to a query. "
        "Returns the top matching chunks with their content and source metadata. "
        "Use this to retrieve factual information before answering."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to retrieve relevant documents for.",
            },
            "kb_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "List of knowledge base IDs to search within.",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of results to return (default 5).",
                "default": 5,
            },
        },
        "required": ["query", "kb_ids"],
        "additionalProperties": False,
    },
)
def search_documents(
    query: str,
    kb_ids: List[int],
    top_k: int = 5,
    org_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Search knowledge bases and return top-k chunks.

    Runs the async hybrid_search_with_legs in a separate thread so this sync
    function is safe to call from any context (including inside a running
    event loop, e.g. during tool-calling in chat_service).
    """
    from app.services.retrieval import hybrid_search_with_legs, get_effective_datastore_ids
    from app.db.session import SessionLocal
    import concurrent.futures

    def _run():
        import asyncio
        _db = SessionLocal()
        try:
            datastore_ids = get_effective_datastore_ids(kb_ids, None, _db)

            return asyncio.run(
                hybrid_search_with_legs(query=query, kb_ids=kb_ids, db=_db, datastore_ids=datastore_ids, org_id=org_id)
            )
        finally:
            _db.close()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(_run).result(timeout=30)
        docs = result.get("docs", [])
        return [
            {
                "content": doc.page_content[:1000],  # cap per chunk to limit token use
                "source": doc.metadata.get("source") or doc.metadata.get("file_name", "unknown"),
                "score": round(float(doc.metadata.get("score", 0.0)), 4),
                "chunk_index": doc.metadata.get("chunk_index"),
            }
            for doc in docs[:top_k]
        ]
    except Exception as exc:
        logger.warning("[TOOL] search_documents failed: %s", exc)
        return [{"error": str(exc)}]



# ── extract_entities ──────────────────────────────────────────────────────────

@register_tool(
    name="extract_entities",
    description=(
        "Extract named entities (people, organizations, places, products, events) "
        "from a piece of text. Returns a list of entity names and their types. "
        "Use this to identify key entities in retrieved chunks before further analysis."
    ),
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to extract named entities from.",
            },
        },
        "required": ["text"],
        "additionalProperties": False,
    },
)
def extract_entities(text: str) -> List[Dict[str, str]]:
    """Extract named entities from text using the GRAPHRAG_LLM model."""
    from app.services.graph.entity_extractor import extract_entities_from_query

    try:
        entities = extract_entities_from_query(text)
        return [{"name": e.name, "type": e.type} for e in entities]
    except Exception as exc:
        logger.warning("[TOOL] extract_entities failed: %s", exc)
        return [{"error": str(exc)}]


# ── summarize_chunks ──────────────────────────────────────────────────────────

@register_tool(
    name="summarize_chunks",
    description=(
        "Summarize a list of text chunks according to a specific instruction. "
        "Use this to synthesize information from multiple retrieved documents "
        "into a coherent answer or report."
    ),
    parameters={
        "type": "object",
        "properties": {
            "chunks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of text chunks to summarize.",
            },
            "instruction": {
                "type": "string",
                "description": "Summarization instruction, e.g. 'Summarize key themes' or 'Extract main findings'.",
            },
        },
        "required": ["chunks", "instruction"],
        "additionalProperties": False,
    },
)
def summarize_chunks(chunks: List[str], instruction: str) -> Dict[str, str]:
    """Summarize chunks via the chat LLM."""
    from openai import OpenAI as SyncOpenAI
    from app.core.config import settings

    if not chunks:
        return {"summary": "", "chunk_count": 0}

    combined = "\n\n---\n\n".join(chunks[:20])  # cap at 20 chunks
    prompt = (
        f"{instruction}\n\n"
        f"TEXT:\n{combined}"
    )

    try:
        client = SyncOpenAI(
            base_url=settings.OPENAI_API_BASE,
            api_key=settings.OPENAI_API_KEY,
        )
        resp = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise document summarizer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=1024,
        )
        summary = (resp.choices[0].message.content or "").strip()
        logger.info("[TOOL] summarize_chunks chunk_count=%d", len(chunks))
        return {"summary": summary, "chunk_count": len(chunks)}
    except Exception as exc:
        logger.warning("[TOOL] summarize_chunks failed: %s", exc)
        return {"error": str(exc)}


# ── synthesize_documents ──────────────────────────────────────────────────────

@register_tool(
    name="synthesize_documents",
    description=(
        "Gather broad document coverage for a topic by running multiple targeted search queries "
        "in parallel, deduplicating the results, and returning all unique chunks. "
        "Use this for synthesis tasks like 'summarize themes across all earnings calls' or "
        "'compare approaches across documents'. Prefer this over multiple search_documents calls."
    ),
    parameters={
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "The high-level topic being synthesized (used for logging).",
            },
            "sub_queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "List of targeted search queries covering different aspects of the topic. "
                    "E.g. ['Q4 revenue results', 'Q4 guidance outlook', 'Q4 risk factors']. "
                    "2–8 queries recommended."
                ),
            },
            "kb_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Knowledge base IDs to search within.",
            },
            "top_k_per_query": {
                "type": "integer",
                "description": "Results per sub-query (default 8). Total unique chunks ≤ sub_queries × top_k_per_query.",
                "default": 8,
            },
        },
        "required": ["topic", "sub_queries", "kb_ids"],
        "additionalProperties": False,
    },
)
def synthesize_documents(
    topic: str,
    sub_queries: List[str],
    kb_ids: List[int],
    top_k_per_query: int = 8,
    org_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Fan out multiple search queries in parallel, deduplicate results, return unique chunks.

    Deduplication key: content hash (first 200 chars). This handles the case where
    the same chunk appears in results for different sub-queries.
    """
    import hashlib
    import concurrent.futures
    from app.services.retrieval import hybrid_search_with_legs, get_effective_datastore_ids

    if not sub_queries:
        return {"topic": topic, "chunks": [], "total_unique": 0, "sub_queries_run": 0}

    def _run_all():
        import asyncio
        from app.db.session import SessionLocal
        _db = SessionLocal()
        try:
            datastore_ids = get_effective_datastore_ids(kb_ids, None, _db)

            async def _gather():
                tasks = [
                    hybrid_search_with_legs(query=q, kb_ids=kb_ids, db=_db, datastore_ids=datastore_ids, org_id=org_id)
                    for q in sub_queries
                ]
                return await asyncio.gather(*tasks, return_exceptions=True)

            return asyncio.run(_gather())
        finally:
            _db.close()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            results = pool.submit(_run_all).result(timeout=60)
    except Exception as exc:
        logger.warning("[SYNTHESIS] gather failed: %s", exc)
        return {"error": str(exc)}

    seen: set = set()
    chunks: List[Dict[str, Any]] = []

    for sub_result in results:
        if isinstance(sub_result, Exception):
            logger.warning("[SYNTHESIS] sub-query failed (skipped): %s", sub_result)
            continue
        for doc in sub_result.get("docs", []):
            # Deduplicate by content fingerprint
            key = hashlib.md5(doc.page_content[:200].encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            chunks.append({
                "content": doc.page_content[:1000],
                "source": doc.metadata.get("source") or doc.metadata.get("file_name", "unknown"),
                "score": round(float(doc.metadata.get("score", 0.0)), 4),
                "chunk_index": doc.metadata.get("chunk_index"),
            })

    # Sort by score descending so highest-relevance chunks appear first
    chunks.sort(key=lambda c: c["score"], reverse=True)

    logger.info(
        "[SYNTHESIS] topic=%r sub_queries=%d unique_chunks=%d",
        topic[:80], len(sub_queries), len(chunks),
    )
    return {
        "topic": topic,
        "chunks": chunks,
        "total_unique": len(chunks),
        "sub_queries_run": len(sub_queries),
    }

