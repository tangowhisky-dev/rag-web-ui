"""BM25 migration — backfills bm25/bm25_title sparse vectors on existing
Qdrant collections, in place (Qdrant ≥1.18 ``PUT /collections/{c}/vectors/{v}``).

No persistence: job progress lives in a module-level dict and is cleared on
restart. One job per collection at a time.
"""

import logging
import threading
import time
from typing import Dict, Optional

from qdrant_client.models import (
    Document,
    FieldCondition,
    Filter,
    HasVectorCondition,
    MatchValue,
    PointStruct,
    PointVectors,
)

from app.services.ingestion.document_qdrant import (
    _BM25_MODEL,
    _BM25_SPARSE_CONFIGS,
    _BM25_TITLE_VECTOR,
    _BM25_VECTOR,
    _ensure_bm25_vectors,
    _ensure_payload_indexes,
    _title_point_id,
)
from app.services.infrastructure import get_qdrant_client

logger = logging.getLogger(__name__)

_SCROLL_BATCH = 128
_UPDATE_BATCH = 64

_lock = threading.Lock()
# collection_name -> job state (in-memory only, per user requirement)
_jobs: Dict[str, dict] = {}

_CHUNK_FILTER = Filter(must_not=[
    FieldCondition(key="_title_point", match=MatchValue(value=True)),
])


def _chunk_points_missing_bm25(client, collection_name: str) -> int:
    """Count chunk points that lack the bm25 vector (title pseudo-points
    excluded — they never carry it)."""
    res = client.count(
        collection_name=collection_name,
        count_filter=Filter(must_not=[
            FieldCondition(key="_title_point", match=MatchValue(value=True)),
            HasVectorCondition(has_vector=_BM25_VECTOR),
        ]),
        exact=True,
    )
    return res.count


def collection_bm25_status(collection_name: str) -> dict:
    """Schema + backfill status for one collection."""
    client = get_qdrant_client()
    try:
        params = client.get_collection(collection_name).config.params
    except Exception:
        return {"collection": collection_name, "exists": False}
    sparse = set((params.sparse_vectors or {}).keys())
    missing = _BM25_SPARSE_CONFIGS.keys() - sparse
    out = {
        "collection": collection_name,
        "exists": True,
        "schema_ready": not missing,
        "missing_vectors": sorted(missing),
        "points_total": client.count(collection_name=collection_name, exact=True).count,
        "chunks_missing_bm25": None,
    }
    if not missing:
        out["chunks_missing_bm25"] = _chunk_points_missing_bm25(client, collection_name)
        out["needs_migration"] = out["chunks_missing_bm25"] > 0 or bool(
            client.count(
                collection_name=collection_name,
                count_filter=Filter(must=[FieldCondition(
                    key="_title_point", match=MatchValue(value=True))]),
                exact=True,
            ).count == 0
        )
    else:
        out["needs_migration"] = out["points_total"] > 0
    return out


def get_job(collection_name: str) -> Optional[dict]:
    with _lock:
        job = _jobs.get(collection_name)
        return dict(job) if job else None


def start_migration(collection_name: str) -> dict:
    """Spawn the backfill thread. Returns the job state; raises ValueError if
    a job is already running for this collection."""
    with _lock:
        job = _jobs.get(collection_name)
        if job and job.get("phase") not in ("done", "error"):
            raise ValueError("migration already running")
        job = {
            "collection": collection_name,
            "phase": "schema",
            "points_total": 0,
            "points_done": 0,
            "title_points": 0,
            "chunks_missing_bm25": None,
            "verified": False,
            "error": None,
            "started_at": time.time(),
            "finished_at": None,
        }
        _jobs[collection_name] = job
    threading.Thread(target=_run, args=(collection_name,), daemon=True).start()
    return dict(job)


def _run(collection_name: str) -> None:
    job = _jobs[collection_name]
    client = get_qdrant_client()
    try:
        # 1. Schema — add named sparse vectors + document_id payload index
        #    in place (no recreation).
        _ensure_bm25_vectors(client, collection_name)
        _ensure_payload_indexes(client, collection_name)

        # 2. Backfill chunk points: scroll payload+id only, add the bm25
        #    vector via update_vectors (server-side inference on chunk_text).
        job["points_total"] = client.count(
            collection_name=collection_name,
            count_filter=_CHUNK_FILTER,
            exact=True,
        ).count
        job["phase"] = "backfill"

        titles: Dict[int, str] = {}
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection_name,
                scroll_filter=_CHUNK_FILTER,
                limit=_SCROLL_BATCH,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            if not points:
                break
            batch = []
            for p in points:
                payload = p.payload or {}
                text = payload.get("chunk_text") or ""
                if not text:
                    job["points_done"] += 1
                    continue
                batch.append(PointVectors(
                    id=p.id,
                    vector={_BM25_VECTOR: Document(text=text, model=_BM25_MODEL)},
                ))
                doc_id = payload.get("document_id")
                title = payload.get("title") or ""
                if doc_id is not None and title:
                    titles.setdefault(int(doc_id), title)
            for i in range(0, len(batch), _UPDATE_BATCH):
                client.update_vectors(
                    collection_name=collection_name,
                    points=batch[i:i + _UPDATE_BATCH],
                )
            job["points_done"] += len(points)
            if offset is None:
                break

        # 3. Title pseudo-points — one upsert batch per document.
        job["phase"] = "titles"
        tpoints = [
            PointStruct(
                id=_title_point_id(doc_id),
                vector={_BM25_TITLE_VECTOR: Document(text=title, model=_BM25_MODEL)},
                payload={"document_id": doc_id, "_title_point": True, "title": title},
            )
            for doc_id, title in titles.items()
        ]
        for i in range(0, len(tpoints), _UPDATE_BATCH):
            client.upsert(collection_name=collection_name, points=tpoints[i:i + _UPDATE_BATCH])
        job["title_points"] = len(tpoints)

        # 4. Verify — every chunk point must carry a bm25 vector.
        job["phase"] = "verify"
        missing = _chunk_points_missing_bm25(client, collection_name)
        job["chunks_missing_bm25"] = missing
        job["verified"] = missing == 0
        job["phase"] = "done" if job["verified"] else "error"
        if not job["verified"]:
            job["error"] = f"{missing} chunk points still missing bm25 vector"
        else:
            logger.info("[bm25-migration] %s verified: %d chunks, %d title points",
                        collection_name, job["points_done"], job["title_points"])
    except Exception as e:
        logger.exception("[bm25-migration] %s failed", collection_name)
        job["phase"] = "error"
        job["error"] = str(e)
    finally:
        job["finished_at"] = time.time()
