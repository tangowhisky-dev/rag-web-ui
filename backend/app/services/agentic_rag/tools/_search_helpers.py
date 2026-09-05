"""Shared helpers for atomic search tools.

Extracted from the original monolithic retrieval tool so all search tools (search_exact,
search_sparse, search_dense) share the same filter resolution and
synonym expansion logic.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional

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
    from datetime import datetime as _dt
    from sqlalchemy import or_, and_

    q = db.query(Document.id).filter(
        or_(
            Document.knowledge_base_id.in_(kb_ids),
            and_(Document.knowledge_base_id.is_(None), Document.data_store_id.isnot(None)),
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
        llm = build_chat_llm(ctx.org_id, ctx.db, role="query", temperature=0.0)
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
