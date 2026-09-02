"""KB profiling — caches per-KB metadata for agent context.

Profiles are computed once per KB and cached in Redis. They provide
field availability, content types, date range, doc count, and avg
chunk length so that rewrite_query_node can suggest filters/sort
and plan/think prompts can include KB context without calling
kb_metadata as a tool.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional

from app.core.config import settings
from app.models.knowledge import Document, DocumentChunk, KnowledgeBase

logger = logging.getLogger(__name__)

PROFILE_TTL = 3600  # 1 hour default


def _cache_key(org_id: int, kb_id: int) -> str:
    return f"kb_profile:{org_id}:{kb_id}"


async def _redis_get(key: str) -> Optional[dict]:
    """Get a cached profile from Redis. Returns None if Redis is unavailable."""
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        try:
            val = await r.get(key)
            if val:
                return json.loads(val)
        finally:
            await r.aclose()
    except Exception as exc:
        logger.debug("[kb_profile] redis get failed for %s: %s", key, exc)
    return None


async def _redis_set(key: str, value: dict, ttl: int) -> None:
    """Cache a profile in Redis. Silently fails if Redis is unavailable."""
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        try:
            await r.setex(key, ttl, json.dumps(value))
        finally:
            await r.aclose()
    except Exception as exc:
        logger.debug("[kb_profile] redis set failed for %s: %s", key, exc)


def _compute_profile(org_id: int, kb_id: int, db: Any) -> dict:
    """Compute a KB profile from MySQL. Synchronous — called in load_context_node.

    Fast queries: doc count, distinct titles, content types, date range,
    avg chunk length. No Qdrant sampling — chunk_text length is in MySQL.
    """
    try:
        # Document-level metadata
        docs = db.query(Document).filter(
            Document.knowledge_base_id == kb_id
        ).all()

        if not docs:
            return {}

        doc_count = len(docs)
        titles = [d.title for d in docs if d.title][:20]
        content_types = list(set(d.content_type for d in docs if d.content_type))
        created_dates = [d.created_at for d in docs if d.created_at]
        date_range = {}
        if created_dates:
            date_range = {
                "min": min(created_dates).isoformat(),
                "max": max(created_dates).isoformat(),
            }

        # Chunk-level: avg chunk length
        chunks = db.query(DocumentChunk).filter(
            DocumentChunk.kb_id == kb_id
        ).limit(50).all()
        avg_chunk_len = 0
        if chunks:
            lengths = [len(c.chunk_text) for c in chunks if c.chunk_text]
            if lengths:
                avg_chunk_len = sum(lengths) // len(lengths)

        # Field availability: which rag_retrieve filter fields are usable
        has_titles = any(d.title for d in docs)
        has_file_names = any(d.file_name for d in docs)
        has_content_types = len(content_types) > 1  # only useful if there's variety
        has_dates = bool(date_range)

        fields = {}
        if has_titles:
            fields["title_contains"] = True
        if has_file_names:
            fields["file_name_contains"] = True
        if has_content_types:
            fields["content_type"] = True
        if has_dates:
            fields["created_after"] = True
            fields["created_before"] = True

        return {
            "kb_id": kb_id,
            "doc_count": doc_count,
            "titles": titles,
            "content_types": content_types,
            "date_range": date_range,
            "avg_chunk_len": avg_chunk_len,
            "fields": fields,
        }
    except Exception as exc:
        logger.warning("[kb_profile] compute failed for kb=%s: %s", kb_id, exc)
        return {}


async def profile_kb(org_id: int, kb_id: int, db: Any, ttl: int = PROFILE_TTL) -> dict:
    """Get a KB profile, computing + caching if needed."""
    key = _cache_key(org_id, kb_id)
    cached = await _redis_get(key)
    if cached:
        return cached
    profile = _compute_profile(org_id, kb_id, db)
    if profile:
        await _redis_set(key, profile, ttl)
    return profile


def merge_profiles(per_kb: list[dict]) -> dict:
    """Merge per-KB profiles into a single dict for AgentState.

    Union of fields, weighted average of chunk lengths, field availability
    tracked per-KB so intent extraction knows which fields exist where.
    """
    if not per_kb:
        return {}
    if len(per_kb) == 1:
        return per_kb[0]

    total_docs = sum(p.get("doc_count", 0) for p in per_kb)
    all_titles: list[str] = []
    all_content_types: set[str] = set()
    all_fields: dict[str, list[int]] = {}
    min_date = None
    max_date = None
    total_chunk_len = 0
    total_chunks_weighted = 0

    for p in per_kb:
        all_titles.extend(p.get("titles", []))
        all_content_types.update(p.get("content_types", []))
        for field in p.get("fields", {}):
            all_fields.setdefault(field, []).append(p.get("kb_id"))
        dr = p.get("date_range", {})
        if dr.get("min"):
            if min_date is None or dr["min"] < min_date:
                min_date = dr["min"]
        if dr.get("max"):
            if max_date is None or dr["max"] > max_date:
                max_date = dr["max"]
        dc = p.get("doc_count", 0)
        acl = p.get("avg_chunk_len", 0)
        if dc and acl:
            total_chunk_len += acl * dc
            total_chunks_weighted += dc

    avg_chunk = total_chunk_len // total_chunks_weighted if total_chunks_weighted else 0

    return {
        "doc_count": total_docs,
        "titles": all_titles[:20],
        "content_types": list(all_content_types),
        "date_range": {"min": min_date, "max": max_date} if min_date else {},
        "avg_chunk_len": avg_chunk,
        "fields": all_fields,  # {field_name: [kb_id, ...]}
    }


def format_profile_summary(profile: dict) -> str:
    """Format a KB profile into a compact text block for prompt injection."""
    if not profile or not profile.get("doc_count"):
        return ""

    parts = [f"[KB Profile]"]
    parts.append(f"Documents: {profile.get('doc_count', 0)}")
    if profile.get("content_types"):
        parts.append(f"Content types: {', '.join(profile['content_types'][:5])}")
    if profile.get("date_range", {}).get("min"):
        dr = profile["date_range"]
        parts.append(f"Date range: {dr['min'][:10]} to {dr['max'][:10]}")
    if profile.get("titles"):
        parts.append(f"Sample titles: {', '.join(profile['titles'][:5])}")
    fields = profile.get("fields", {})
    if fields:
        # fields may be {field: True} (single KB) or {field: [kb_ids]} (merged)
        field_names = []
        for f, v in fields.items():
            if v:
                field_names.append(f)
        if field_names:
            parts.append(f"Available filter fields: {', '.join(field_names)}")
    return "\n".join(parts)
