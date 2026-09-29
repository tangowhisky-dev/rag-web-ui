"""Super-admin API for Qdrant BM25 migration of datastore and KB collections.

Endpoints:
    GET  /api/admin/migrations/bm25                    — status of every ds_*/kb_* collection
    POST /api/admin/migrations/bm25/{kind}/{id}        — start backfill job
    GET  /api/admin/migrations/bm25/{kind}/{id}/status — in-memory job progress

``kind`` is "datastores" (ds_{id}) or "knowledge-bases" (kb_{id}).

Progress is held in memory only (bm25_migration._jobs) — restarting the
backend clears it; the job itself is idempotent and safe to re-run.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.security import require_super_admin
from app.db.session import get_db
from app.models.datastore import DataStore
from app.models.knowledge import KnowledgeBase
from app.models.user import User
from app.services.datastore import bm25_migration

logger = logging.getLogger(__name__)

router = APIRouter()

_COLLECTION_PREFIX = {"datastores": "ds_", "knowledge-bases": "kb_"}


def _resolve(kind: str, entity_id: int, db: Session) -> tuple[str, str]:
    """Return (collection_name, display_name) or raise."""
    if kind == "datastores":
        entity = db.query(DataStore).filter(DataStore.id == entity_id).first()
    elif kind == "knowledge-bases":
        entity = db.query(KnowledgeBase).filter(KnowledgeBase.id == entity_id).first()
    else:
        raise HTTPException(status_code=404, detail=f"Unknown migration kind: {kind}")
    if entity is None:
        raise HTTPException(status_code=404, detail=f"{kind[:-1].replace('-', ' ').title()} not found")
    return f"{_COLLECTION_PREFIX[kind]}{entity_id}", entity.name


def _group_status(kind: str, db: Session) -> list[dict]:
    model = DataStore if kind == "datastores" else KnowledgeBase
    out = []
    for entity in db.query(model).order_by(model.id).all():
        coll = f"{_COLLECTION_PREFIX[kind]}{entity.id}"
        try:
            status = bm25_migration.collection_bm25_status(coll)
        except Exception as e:
            status = {"collection": coll, "exists": False, "error": str(e)}
        out.append({
            "id": entity.id,
            "name": entity.name,
            "job": bm25_migration.get_job(coll),
            **status,
        })
    return out


@router.get("/migrations/bm25")
def list_bm25_migration_status(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_super_admin),
):
    """BM25 migration status for all datastore + knowledge base collections."""
    return {
        "datastores": _group_status("datastores", db),
        "knowledge_bases": _group_status("knowledge-bases", db),
    }


@router.post("/migrations/bm25/{kind}/{entity_id}")
def start_bm25_migration(
    kind: str,
    entity_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_super_admin),
):
    """Start the in-place BM25 backfill for one collection."""
    coll, _ = _resolve(kind, entity_id, db)
    status = bm25_migration.collection_bm25_status(coll)
    if not status.get("exists"):
        raise HTTPException(status_code=404, detail=f"Qdrant collection {coll} not found")
    if status.get("schema_ready") and not status.get("needs_migration"):
        raise HTTPException(status_code=400, detail="Collection already has complete BM25 vectors")
    try:
        return bm25_migration.start_migration(coll)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/migrations/bm25/{kind}/{entity_id}/status")
def bm25_migration_job_status(
    kind: str,
    entity_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_super_admin),
):
    """Poll in-memory job progress for one collection."""
    coll, name = _resolve(kind, entity_id, db)
    job = bm25_migration.get_job(coll)
    try:
        status = bm25_migration.collection_bm25_status(coll)
    except Exception as e:
        status = {"collection": coll, "exists": False, "error": str(e)}
    return {"id": entity_id, "name": name, "job": job, **status}
