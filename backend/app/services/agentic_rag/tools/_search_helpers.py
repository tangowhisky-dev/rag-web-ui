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


_STATUS_ALIASES = {"obsolete": "superseded"}

_KNOWN_FILTER_KEYS = frozenset({
    "title_contains", "file_name_contains", "content_type",
    "created_after", "created_before",
    "file_modified_after", "file_modified_before",
    "file_created_after", "file_created_before",
    "document_ids",
    "document_status", "exclude_status",
    "effective_as_of",
    "effective_window_start", "effective_window_end",
    "effective_from_after", "effective_from_before",
    "effective_to_after", "effective_to_before",
    "version", "owner",
})


def _filter_value_list(value) -> list[str]:
    """Normalize a scalar-or-list filter value to a list of stripped strings."""
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple)) else [value]
    return [str(v).strip() for v in items if v is not None and str(v).strip()]


def _parse_filter_dt(value):
    """Parse an ISO date/datetime filter value to a naive-UTC datetime.

    Document date columns are naive DateTime — tz-aware input is normalized
    so comparisons don't raise, and 'Z' suffixes are accepted. Returns None
    for missing/unparseable values.
    """
    if value is None:
        return None
    from datetime import datetime as _dt, timezone as _tz
    try:
        dt = _dt.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return dt.astimezone(_tz.utc).replace(tzinfo=None) if dt.tzinfo else dt


def resolve_filter_to_doc_ids(
    db: Any,
    kb_ids: list[int],
    filters: dict | None,
) -> tuple[list[int] | None, dict]:
    """Translate metadata filters to a list of document_ids via MySQL.

    Returns (None, meta) when no recognized filter is provided — the search
    runs unfiltered. Returns ([], meta) when recognized filters match zero
    documents. meta carries ``total_matching`` and ``ignored_keys`` so the
    agent can distinguish "no documents matched" from "filter not applied"
    and detect keys/values that were silently ignored.
    """
    meta: dict = {"total_matching": None, "ignored_keys": []}
    if not filters:
        return None, meta

    from app.models.knowledge import Document
    from app.services.retrieval.retrieval import get_effective_datastore_ids
    from sqlalchemy import or_

    ignored = {k for k in filters if k not in _KNOWN_FILTER_KEYS}
    # Present-but-empty values on known keys are also ignored.
    ignored |= {
        k for k, v in filters.items()
        if k in _KNOWN_FILTER_KEYS and not v
    }
    applied = False

    ds_ids = get_effective_datastore_ids(kb_ids, None, db)
    q = db.query(Document.id).filter(
        or_(
            Document.knowledge_base_id.in_(kb_ids),
            Document.data_store_id.in_(ds_ids) if ds_ids else False,
        )
    )

    if filters.get("title_contains"):
        applied = True
        q = q.filter(Document.title.ilike(f"%{filters['title_contains']}%"))
    if filters.get("file_name_contains"):
        applied = True
        q = q.filter(Document.file_name.ilike(f"%{filters['file_name_contains']}%"))
    if filters.get("content_type"):
        applied = True
        q = q.filter(Document.content_type == filters["content_type"])
    if filters.get("version"):
        applied = True
        q = q.filter(Document.version == str(filters["version"]).strip())
    if filters.get("owner"):
        applied = True
        q = q.filter(Document.owner.ilike(f"%{filters['owner']}%"))
    if filters.get("document_ids"):
        # LLMs sometimes pass a scalar instead of a list — normalize, and
        # treat non-numeric values as ignored rather than erroring.
        raw_ids = _filter_value_list(filters["document_ids"])
        id_vals = [int(v) for v in raw_ids if str(v).lstrip("-").isdigit()]
        if id_vals:
            applied = True
            q = q.filter(Document.id.in_(id_vals))
        else:
            ignored.add("document_ids")

    # Plain date ranges. NULL effective_to rows never satisfy <=/>=
    # comparisons, so effective_to_* filters return only expiring documents.
    for key, col, is_after in (
        ("created_after", Document.created_at, True),
        ("created_before", Document.created_at, False),
        ("file_modified_after", Document.file_modified_at, True),
        ("file_modified_before", Document.file_modified_at, False),
        ("file_created_after", Document.file_created_at, True),
        ("file_created_before", Document.file_created_at, False),
        ("effective_from_after", Document.effective_from, True),
        ("effective_from_before", Document.effective_from, False),
        ("effective_to_after", Document.effective_to, True),
        ("effective_to_before", Document.effective_to, False),
    ):
        raw = filters.get(key)
        if raw is None:
            continue
        dt = _parse_filter_dt(raw)
        if dt is None:
            ignored.add(key)
            continue
        applied = True
        q = q.filter(col >= dt if is_after else col <= dt)

    # Lifecycle status — 'obsolete' is accepted as an alias for 'superseded'
    # because that is the term users/LLMs reach for naturally.
    if "document_status" in filters:
        statuses = [
            _STATUS_ALIASES.get(s.lower(), s.lower())
            for s in _filter_value_list(filters["document_status"])
        ]
        if statuses:
            applied = True
            q = q.filter(Document.document_status.in_(statuses))
    if "exclude_status" in filters:
        excl = [
            _STATUS_ALIASES.get(s.lower(), s.lower())
            for s in _filter_value_list(filters["exclude_status"])
        ]
        if excl:
            applied = True
            q = q.filter(Document.document_status.notin_(excl))

    # Point-in-time validity: "what was in force on date D".
    if "effective_as_of" in filters:
        as_of = _parse_filter_dt(filters["effective_as_of"])
        if as_of is None:
            ignored.add("effective_as_of")
        else:
            applied = True
            q = q.filter(Document.effective_from <= as_of)
            q = q.filter(or_(Document.effective_to.is_(None), Document.effective_to >= as_of))

    # Window overlap: "what was in force during [start, end]" — either bound
    # may be omitted.
    ws_raw = filters.get("effective_window_start")
    we_raw = filters.get("effective_window_end")
    if ws_raw is not None or we_raw is not None:
        ws = _parse_filter_dt(ws_raw)
        we = _parse_filter_dt(we_raw)
        if ws_raw is not None and ws is None:
            ignored.add("effective_window_start")
        if we_raw is not None and we is None:
            ignored.add("effective_window_end")
        if we is not None:
            applied = True
            q = q.filter(Document.effective_from <= we)
        if ws is not None:
            applied = True
            q = q.filter(or_(Document.effective_to.is_(None), Document.effective_to >= ws))

    meta["ignored_keys"] = sorted(ignored)
    if not applied:
        return None, meta

    # No cap: the caller passes doc_ids to Qdrant MatchAny / MySQL IN, both
    # of which handle thousands of ids. total_matching tells the agent how
    # selective its filter was.
    doc_ids = [r[0] for r in q.all()]
    meta["total_matching"] = len(doc_ids)
    return doc_ids, meta


def enrich_hits_with_authority(hits: list[dict], db: Any) -> list[dict]:
    """Stamp document lifecycle metadata onto each hit so the LLM can reason
    about authority — not just relevance.

    document_status / effective_from / effective_to / version are mutable
    post-ingestion (admins edit them on the documents table), so they are
    resolved live from MySQL here rather than stamped into Qdrant payloads
    or chunk_metadata at index time. Degrades gracefully: a DB failure
    returns the hits untagged.
    """
    if not hits or db is None:
        return hits
    doc_ids = {
        h.get("document_id") for h in hits
        if isinstance(h, dict) and isinstance(h.get("document_id"), int)
    }
    if not doc_ids:
        return hits
    try:
        from app.models.knowledge import Document
        rows = db.query(
            Document.id, Document.document_status,
            Document.effective_from, Document.effective_to,
            Document.version,
        ).filter(Document.id.in_(doc_ids)).all()
        by_id = {r.id: r for r in rows}
    except Exception as exc:
        logger.warning("[authority] doc metadata enrichment failed: %s", exc)
        return hits
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        row = by_id.get(hit.get("document_id"))
        if row is None:
            continue
        hit["document_status"] = row.document_status
        hit["effective_from"] = row.effective_from.isoformat() if row.effective_from else ""
        hit["effective_to"] = row.effective_to.isoformat() if row.effective_to else ""
        hit["version"] = row.version
    return hits


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
