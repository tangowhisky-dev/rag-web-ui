"""Shared helpers for search tools.

Extracted from the original monolithic retrieval tool so all search tools (keyword_search,
semantic_search) share the same filter resolution and
synonym expansion logic.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, List, Optional

from app.core.config import settings
from app.services.agentic_rag.tool_context import ToolContext

logger = logging.getLogger(__name__)


def _safe_writer():
    """Return the LangGraph stream writer if available, else None."""
    try:
        from langgraph.config import get_stream_writer
        return get_stream_writer()
    except (RuntimeError, KeyError, ImportError):
        return None


def _emit_progress(phase: str, message: str, **extra: Any) -> None:
    """Emit a progress event for the UI."""
    writer = _safe_writer()
    if writer:
        payload: dict[str, Any] = {"event": "progress", "phase": phase, "message": message}
        payload.update(extra)
        writer(payload)


def resolve_filter_to_doc_ids(
    db: Any,
    kb_ids: list[int],
    filters: dict | None,
) -> list[int] | None:
    """Translate metadata filters to a list of document_ids via MySQL.

    Returns None when no filters are provided (search all docs).
    Returns an empty list if filters match zero documents.
    """
    if not filters:
        return None

    from app.models.knowledge import Document
    from app.services.retrieval.retrieval import get_effective_datastore_ids
    from datetime import datetime as _dt
    from sqlalchemy import or_

    ds_ids = get_effective_datastore_ids(kb_ids, None, db)
    q = db.query(Document.id).filter(
        or_(
            Document.knowledge_base_id.in_(kb_ids),
            Document.data_store_id.in_(ds_ids) if ds_ids else False,
        )
    )

    if filters.get("title_contains"):
        q = q.filter(Document.title.ilike(f"%{filters['title_contains']}%"))
    if filters.get("file_name_contains"):
        q = q.filter(Document.file_name.ilike(f"%{filters['file_name_contains']}%"))
    if filters.get("content_type"):
        q = q.filter(Document.content_type == filters["content_type"])
    if filters.get("created_after"):
        try:
            after = _dt.fromisoformat(filters["created_after"])
            q = q.filter(Document.created_at >= after)
        except (ValueError, TypeError):
            pass
    if filters.get("created_before"):
        try:
            before = _dt.fromisoformat(filters["created_before"])
            q = q.filter(Document.created_at <= before)
        except (ValueError, TypeError):
            pass
    if filters.get("file_modified_after"):
        try:
            after = _dt.fromisoformat(filters["file_modified_after"])
            q = q.filter(Document.file_modified_at >= after)
        except (ValueError, TypeError):
            pass
    if filters.get("file_modified_before"):
        try:
            before = _dt.fromisoformat(filters["file_modified_before"])
            q = q.filter(Document.file_modified_at <= before)
        except (ValueError, TypeError):
            pass
    if filters.get("file_created_after"):
        try:
            after = _dt.fromisoformat(filters["file_created_after"])
            q = q.filter(Document.file_created_at >= after)
        except (ValueError, TypeError):
            pass
    if filters.get("file_created_before"):
        try:
            before = _dt.fromisoformat(filters["file_created_before"])
            q = q.filter(Document.file_created_at <= before)
        except (ValueError, TypeError):
            pass
    if filters.get("document_ids"):
        q = q.filter(Document.id.in_(filters["document_ids"]))

    return [r[0] for r in q.limit(200).all()]


async def expand_synonyms(query: str, ctx: ToolContext) -> tuple[str, list[str]]:
    """Expand query with spell-corrected + synonym variants via LLM.

    Uses the ``query`` LLM role. Cached in Redis
    (key: synonyms:{org_id}:{sha256(query)}).

    Returns (corrected_query, synonyms). corrected_query is the spell-corrected
    query (or original if no correction needed). synonyms is a list of
    alternative terms (may be empty).
    """
    from app.services.agentic_rag.llm_factory import build_chat_llm
    from app.services.settings_service import get_setting
    from app.services.agentic_rag.prompts import SYNONYM_EXPANSION_PROMPT

    n = get_setting(ctx.db, "SYNONYM_VARIANTS", ctx.org_id)
    cache_ttl = get_setting(ctx.db, "SYNONYM_CACHE_TTL", ctx.org_id)

    # Check Redis cache
    cache_key = f"synonyms:{ctx.org_id}:{hashlib.sha256(query.encode()).hexdigest()}"
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        try:
            cached = await r.get(cache_key)
            if cached:
                obj = json.loads(cached)
                return obj.get("corrected", query), obj.get("synonyms", [])
        finally:
            await r.aclose()
    except Exception:
        pass  # Redis unavailable — proceed without cache

    # Call LLM with query role
    try:
        tool_temp = get_setting(ctx.db, "TOOL_CALL_TEMPERATURE", ctx.org_id)
        llm = build_chat_llm(ctx.org_id, ctx.db, role="utility", temperature=tool_temp)
        prompt = SYNONYM_EXPANSION_PROMPT.format(n=n)
        resp = await llm.ainvoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ])
        raw = resp.content if isinstance(resp.content, str) else str(resp.content)
        import re as _re
        json_match = _re.search(r'\{[^{}]*\}', raw, _re.DOTALL)
        if not json_match:
            return query, []
        obj = json.loads(json_match.group())
        corrected = obj.get("corrected_query") or query
        synonyms = obj.get("queries") or []
        synonyms = [s for s in synonyms if s and s.lower() != query.lower() and s.lower() != corrected.lower()]
    except Exception as exc:
        logger.warning("[search_helpers] synonym expansion failed: %s", exc)
        return query, []

    # Cache result
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        try:
            await r.setex(cache_key, cache_ttl, json.dumps({"corrected": corrected, "synonyms": synonyms}))
        finally:
            await r.aclose()
    except Exception:
        pass

    logger.debug("[search_helpers] synonyms for %r: corrected=%r, synonyms=%s", query, corrected, synonyms)
    return corrected, synonyms


# ── Neighbor context injection ──────────────────────────────────────────────

_NEIGHBOR_TOP_N = 5
_NEIGHBOR_WINDOW = 1


def _fetch_neighbor_chunks(
    db: Any,
    doc_id: int,
    chunk_indices: list[int],
) -> list[dict]:
    """Fetch chunk_text and chunk_metadata for specific chunk indices from MySQL.

    Returns a list of dicts with keys: chunk_index, chunk_text, chunk_metadata.
    """
    if not chunk_indices:
        return []
    from sqlalchemy import text
    placeholders = ",".join(str(i) for i in chunk_indices)
    try:
        rows = db.execute(text(
            f"SELECT chunk_index, chunk_text, chunk_metadata "
            f"FROM document_chunks "
            f"WHERE document_id = :doc_id AND chunk_index IN ({placeholders})"
        ), {"doc_id": doc_id})
        result = []
        for row in rows:
            meta = row[2] if row[2] else {}
            if isinstance(meta, str):
                meta = json.loads(meta)
            result.append({
                "chunk_index": row[0],
                "chunk_text": row[1],
                "chunk_metadata": meta,
            })
        return result
    except Exception as exc:
        logger.warning("[neighbor] failed to fetch chunks for doc %s: %s", doc_id, exc)
        return []


def inject_neighbor_context(
    docs: list,
    db: Any,
    top_n: int = _NEIGHBOR_TOP_N,
    window: int = _NEIGHBOR_WINDOW,
) -> list:
    """Inject prev/next chunks for top-ranked evidence.

    For each of the top-N reranked docs, fetches the adjacent chunk(s) from
    the same document via MySQL. Injected neighbors are deduplicated against
    existing evidence. Returns the combined list (original docs + injected
    neighbors) in reranker order.

    Reordering by document/chunk_index and overlap pruning are handled
    downstream by ``group_docs_by_document()`` and ``_prune_contiguous_overlaps()``
    in the v2 think node — this function does not duplicate that work.

    Args:
        docs:  Reranked + elbow-truncated LangchainDocument list.
        db:    SQLAlchemy session for MySQL chunk lookups.
        top_n: How many of the top docs to inject neighbors for.
        window: How many chunks on each side to fetch (1 = prev+next).

    Returns:
        Combined list of LangchainDocument objects with neighbors appended.
    """
    if not docs or db is None:
        return docs

    from langchain_core.documents import Document as LangchainDocument

    # Collect existing (doc_id, chunk_index) pairs to avoid duplicate injection.
    existing: set[tuple[int, int]] = set()
    for doc in docs:
        meta = doc.metadata or {}
        did = meta.get("document_id")
        ci = meta.get("chunk_index")
        if did is not None and ci is not None:
            existing.add((did, ci))

    # For top-N docs, find neighbor chunk indices to fetch.
    to_fetch: dict[int, set[int]] = {}  # doc_id -> set of chunk_indices
    for doc in docs[:top_n]:
        meta = doc.metadata or {}
        did = meta.get("document_id")
        ci = meta.get("chunk_index")
        if did is None or ci is None:
            continue
        for offset in range(-window, window + 1):
            if offset == 0:
                continue
            neighbor_ci = ci + offset
            if (did, neighbor_ci) in existing:
                continue
            to_fetch.setdefault(did, set()).add(neighbor_ci)

    # Batch-fetch neighbor chunks from MySQL.
    neighbor_docs: list[LangchainDocument] = []
    for did, chunk_indices in to_fetch.items():
        rows = _fetch_neighbor_chunks(db, did, sorted(chunk_indices))
        for row in rows:
            ci = row["chunk_index"]
            if (did, ci) in existing:
                continue
            existing.add((did, ci))
            meta = dict(row.get("chunk_metadata") or {})
            # Carry over metadata from the parent doc if missing.
            meta.setdefault("document_id", did)
            meta["chunk_index"] = ci
            meta["_is_neighbor"] = True
            # Build citation_ref so the neighbor can be cited.
            meta.setdefault("citation_ref", {
                "document_id": did,
                "citation_kind": "chunk",
                "chunk_index": ci,
                "page": meta.get("page"),
                "quoted_text": row["chunk_text"][:200],
                "source_tool": "neighbor_context",
                "citation_id": "",
            })
            neighbor_docs.append(LangchainDocument(
                page_content=row["chunk_text"],
                metadata=meta,
            ))

    if not neighbor_docs:
        return docs

    return list(docs) + neighbor_docs
