"""Shared deletion service for knowledge bases and datastores.

Cleans up resources across MySQL, Qdrant, Neo4j, and the filesystem in a
consistent order (Qdrant → Neo4j → DB → files).  Both the KB and DataStore
deletion endpoints delegate to this module so the logic lives in one place.

Pipeline types: ``"kb"`` or ``"ds"``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.knowledge import (
    Document,
    DocumentChunk,
    KnowledgeBase,
    ProcessingTask,
)
from app.services.ingestion import _chunk_id_to_point_id
from app.services.graph import (
    delete_graph_for_kb,
    purge_stale_graph_data,
)

logger = logging.getLogger(__name__)


# ── Cancel in-flight tasks before deletion ───────────────────────────────────


def _cancel_inflight_tasks(
    db: Session,
    scope: str,
    scope_id: int,
    document_ids: list[int] | None = None,
    wait_seconds: float = 5.0,
) -> int:
    """Signal cancellation for all in-flight ingestion/graph tasks.

    Sets Redis cancel flags and graph-build cancel events so running
    threads detect cancellation and abort.  Waits up to *wait_seconds*
    for tasks to transition out of "processing"/"pending" status.

    Returns the number of tasks that were signalled.
    """
    from app.services.infrastructure import set_cancel
    from app.services.ingestion.ingestion_dispatcher import (
        cancel_graph_build_for_document,
        cancel_graph_builds_for_datastore,
    )

    signalled = 0

    # 1. Signal cancellation via Redis (for OCR/conversion/embedding threads)
    set_cancel(scope, scope_id, reason="deletion")
    if document_ids:
        for doc_id in document_ids:
            set_cancel("doc", doc_id, reason="deletion")

    # 2. Cancel graph builds
    if scope == "ds":
        signalled += cancel_graph_builds_for_datastore(scope_id)
    elif document_ids:
        for doc_id in document_ids:
            tasks = (
                db.query(ProcessingTask)
                .filter(ProcessingTask.document_id == doc_id)
                .all()
            )
            for t in tasks:
                if cancel_graph_build_for_document(t.id):
                    signalled += 1

    # 3. Wait for tasks to observe cancellation
    if wait_seconds <= 0:
        return signalled

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        # Expire all cached objects so we see committed changes from
        # other sessions (ingestion threads use their own sessions).
        db.expire_all()
        still_running = (
            db.query(ProcessingTask)
            .filter(
                ProcessingTask.status.in_(["processing", "pending"]),
            )
        )
        if document_ids:
            still_running = still_running.filter(
                ProcessingTask.document_id.in_(document_ids)
            )
        count = still_running.count()
        if count == 0:
            break
        time.sleep(0.25)

    return signalled


def _get_qdrant() -> QdrantClient:
    """Return the shared Qdrant client singleton from infrastructure."""
    from app.services.infrastructure import get_qdrant_client
    return get_qdrant_client()


# ── Filesystem cleanup ────────────────────────────────────────────────────────


def _delete_kb_files(user_id: int, kb_id: int) -> None:
    """Remove the entire user_{user_id}/kb_{kb_id}/ directory tree."""
    import shutil
    kb_dir = Path(settings.UPLOAD_DIR) / "user_" / str(user_id) / f"kb_{kb_id}"
    if kb_dir.exists():
        shutil.rmtree(kb_dir)
        logger.debug("DeletionService: deleted KB directory %s", kb_dir)
    else:
        logger.debug("DeletionService: KB directory not found (skip): %s", kb_dir)


# ── Qdrant cleanup ───────────────────────────────────────────────────────────


def _delete_qdrant_for_kb(kb_id: int) -> None:
    """Delete the Qdrant collection for a KB (direct uploads only)."""
    collection_name = f"kb_{kb_id}"
    try:
        qdrant = _get_qdrant()
        logger.info(
            "[QDRANT-DELETE] dropping collection=%s cause=kb_deletion kb_id=%d",
            collection_name, kb_id,
        )
        qdrant.delete_collection(collection_name)
        logger.info("[QDRANT-DELETE] dropped collection=%s", collection_name)
    except Exception as e:
        logger.warning("DeletionService: Qdrant delete failed for kb_%d: %s", kb_id, e)


def _delete_qdrant_for_ds(db: Session, datastore_id: int) -> None:
    """Delete Qdrant points + collection for a DataStore."""
    collection_name = f"ds_{datastore_id}"
    try:
        qdrant = _get_qdrant()

        # Delete individual points first
        doc_ids = [
            d.id for d in db.query(Document)
            .filter(
                Document.data_store_id == datastore_id,
                Document.data_store_id.isnot(None),
            )
            .all()
        ]
        chunk_ids = [
            cid[0] for cid in db.query(DocumentChunk.id)
            .filter(DocumentChunk.document_id.in_(doc_ids))
            .all()
        ]
        if chunk_ids:
            point_ids = [_chunk_id_to_point_id(cid) for cid in chunk_ids]
            qdrant.delete(
                collection_name=collection_name,
                points_selector=PointIdsList(points=point_ids),
            )
            logger.info(
                "DeletionService: deleted %d Qdrant points from %s",
                len(point_ids), collection_name,
            )

        # Delete the collection entirely
        try:
            collections = [c.name for c in qdrant.get_collections().collections]
            if collection_name in collections:
                logger.info(
                    "[QDRANT-DELETE] dropping collection=%s cause=datastore_deletion datastore_id=%d",
                    collection_name, datastore_id,
                )
                qdrant.delete_collection(collection_name)
                logger.info("[QDRANT-DELETE] dropped collection=%s", collection_name)
        except Exception as e:
            logger.warning("DeletionService: Qdrant collection delete failed: %s", e)
    except Exception as e:
        logger.warning("DeletionService: Qdrant delete failed for ds_%d: %s", datastore_id, e)


# ── Neo4j cleanup ────────────────────────────────────────────────────────────


def _delete_neo4j_for_kb(db: Session, kb_id: int) -> None:
    """Delete Neo4j nodes for a KB (direct uploads only)."""
    try:
        remaining_kb_ids = [
            rid[0] for rid in db.query(KnowledgeBase.id)
            .filter(KnowledgeBase.id != kb_id)
            .all()
        ]
        delete_graph_for_kb(kb_id=kb_id)
        purge_stale_graph_data(active_kb_ids=remaining_kb_ids)
        logger.debug("DeletionService: cleaned Neo4j graph for kb_%d", kb_id)
    except Exception as e:
        logger.warning("DeletionService: Neo4j delete failed for kb_%d: %s", kb_id, e)


def _delete_neo4j_for_ds(datastore_id: int) -> None:
    """Delete Neo4j Chunk and Entity nodes for a DataStore."""
    try:
        from app.services.graph import _get_driver
        if settings.NEO4J_URI:
            driver = _get_driver()
            ds_id_str = str(datastore_id)
            with driver.session() as session:
                # 1. Delete Chunk nodes for this datastore
                session.run(
                    """
                    MATCH (c:Chunk {data_store_id: $data_store_id})
                    DETACH DELETE c
                    """,
                    data_store_id=ds_id_str,
                )
                # 2. Delete Entity nodes scoped to this datastore.
                # Entities carry data_store_id as a property (set during
                # extraction). This catches entities written by graph builds
                # that completed after the delete started.
                session.run(
                    """
                    MATCH (e)
                    WHERE (e:__KGBuilder__ OR e:Entity OR e:__Entity__)
                      AND e.data_store_id = $data_store_id
                    DETACH DELETE e
                    """,
                    data_store_id=ds_id_str,
                )
                # 3. Clean up any remaining orphaned Entity nodes (no
                # FROM_CHUNK edges remain). This catches entities whose
                # data_store_id property wasn't set (older code path).
                session.run(
                    """
                    MATCH (e)
                    WHERE (e:__KGBuilder__ OR e:Entity OR e:__Entity__)
                      AND NOT EXISTS { MATCH (e)-[:FROM_CHUNK]->() }
                    DETACH DELETE e
                    """,
                )
            logger.debug("DeletionService: cleaned Neo4j Chunk and Entity nodes for ds_%d", datastore_id)
    except Exception as e:
        logger.warning("DeletionService: Neo4j delete failed for ds_%d: %s", datastore_id, e)


# ── Public API ────────────────────────────────────────────────────────────────


def delete_kb(
    db: Session,
    kb_id: int,
    user_id: int,
) -> dict:
    """Delete a knowledge base and all its resources.

    Direct uploads (data_store_id=NULL): files + Qdrant + Neo4j + DB deleted.
    DataStore docs (data_store_id!=NULL): only the KB link is removed.
    """
    kb = (
        db.query(KnowledgeBase)
        .filter(KnowledgeBase.id == kb_id, KnowledgeBase.user_id == user_id)
        .first()
    )
    if not kb:
        return {"message": "Knowledge base not found"}, 404

    # Categorize documents
    direct_docs = [doc for doc in kb.documents if doc.data_store_id is None]
    datastore_docs = [doc for doc in kb.documents if doc.data_store_id is not None]
    cleanup_errors = []

    # ── 0. Cancel in-flight ingestion/graph tasks ────────────────────────
    all_doc_ids = [d.id for d in kb.documents]
    if all_doc_ids:
        cancelled = _cancel_inflight_tasks(
            db, "kb", kb_id, document_ids=all_doc_ids, wait_seconds=5.0,
        )
        if cancelled:
            logger.info(
                "KB %d deletion: cancelled %d in-flight tasks before cleanup",
                kb_id, cancelled,
            )

    # ── 1. Filesystem cleanup (direct uploads only) ──────────────────────
    if direct_docs:
        try:
            _delete_kb_files(user_id, kb_id)
        except Exception as e:
            cleanup_errors.append(f"Failed to clean up storage files: {e}")

    # ── 2. DB cleanup ────────────────────────────────────────────────────
    # Delete the KB record first. If this commit fails, vector/graph data
    # is still intact and the user can retry. If it succeeds but
    # Qdrant/Neo4j cleanup fails below, the orphaned data is invisible to
    # users (KB no longer in their scope) and reconciliation will clean it
    # up on next startup.
    db.delete(kb)
    db.commit()

    # ── 3. Qdrant cleanup (direct uploads only) ──────────────────────────
    if direct_docs:
        _delete_qdrant_for_kb(kb_id)

    # ── 4. Neo4j cleanup (direct uploads only) ───────────────────────────
    if direct_docs:
        _delete_neo4j_for_kb(db, kb_id)

    # ── 5. Response ──────────────────────────────────────────────────────
    logger.info(
        "KB %d deleted: %d direct uploads removed, %d DataStore links severed",
        kb_id, len(direct_docs), len(datastore_docs),
    )

    if cleanup_errors:
        return {
            "message": "Knowledge base deleted with cleanup warnings",
            "warnings": cleanup_errors,
        }, 200

    return {"message": "Knowledge base and all associated resources deleted successfully"}, 200


def delete_datastore(
    db: Session,
    datastore_id: int,
) -> dict:
    """Delete a datastore and all its associated data.

    Files remain on disk (only DB, Qdrant, Neo4j records are removed).
    """
    from app.models.datastore import DataStore, OrganizationDataStore
    from app.models.knowledge import KnowledgeBaseDataStore

    ds = (
        db.query(DataStore)
        .filter(DataStore.id == datastore_id)
        .first()
    )
    if not ds:
        return {"message": "Datastore not found"}, 404

    # Check org assignments
    assigned = (
        db.query(OrganizationDataStore)
        .filter(OrganizationDataStore.data_store_id == datastore_id)
        .count()
    )
    if assigned > 0:
        return {
            "message": "Cannot delete datastore — it is assigned to one or more organisations",
        }, 409

    # Get all documents in this DataStore before deletion
    datastore_docs = (
        db.query(Document)
        .filter(Document.data_store_id == datastore_id)
        .all()
    )
    logger.info(
        "Datastore preparing to delete id=%d with %d documents",
        datastore_id, len(datastore_docs),
    )

    datastore_docs = [d for d in datastore_docs if d.data_store_id is not None]

    # Get all document IDs for this datastore (needed for Qdrant/Neo4j cleanup)
    doc_ids = [d.id for d in datastore_docs]

    # ── 0. Cancel in-flight scan/ingestion/graph tasks ───────────────────
    if doc_ids:
        cancelled = _cancel_inflight_tasks(
            db, "ds", datastore_id, document_ids=doc_ids, wait_seconds=5.0,
        )
        if cancelled:
            logger.info(
                "Datastore %d deletion: cancelled %d in-flight tasks before cleanup",
                datastore_id, cancelled,
            )

    # ── 1. DB cleanup ────────────────────────────────────────────────────
    # Delete DB records first. If this commit fails, vector/graph data is
    # still intact and the user can retry. If it succeeds but Qdrant/Neo4j
    # cleanup fails below, orphaned data is invisible to users (DataStore no
    # longer in their scope) and reconciliation will clean it up on startup.
    db.query(KnowledgeBaseDataStore).filter(
        KnowledgeBaseDataStore.data_store_id == datastore_id
    ).delete(synchronize_session=False)

    db.query(OrganizationDataStore).filter(
        OrganizationDataStore.data_store_id == datastore_id
    ).delete(synchronize_session=False)

    # Bulk-delete child rows first (works on both MySQL and SQLite where
    # FK CASCADE may not be enforced), then bulk-delete documents.
    if doc_ids:
        db.query(DocumentChunk).filter(
            DocumentChunk.document_id.in_(doc_ids)
        ).delete(synchronize_session=False)
        db.query(ProcessingTask).filter(
            ProcessingTask.document_id.in_(doc_ids)
        ).delete(synchronize_session=False)
        db.query(Document).filter(
            Document.data_store_id == datastore_id
        ).delete(synchronize_session=False)
    db.commit()

    # Capture datastore info before the session state becomes stale.
    ds_id_log = ds.id
    ds_name_log = ds.name

    # Bulk-deleting child rows leaves the ORM's relationship state stale.
    # Use a raw SQL DELETE for the datastore row itself to avoid triggering
    # ORM cascades on stale state, then expunge the ORM instance so the
    # session doesn't try to track or refresh it.
    from sqlalchemy import text as _text
    db.execute(_text("DELETE FROM data_stores WHERE id = :id"), {"id": datastore_id})
    db.commit()
    db.expunge(ds)

    # ── 2. Qdrant cleanup ────────────────────────────────────────────────
    if doc_ids:
        _delete_qdrant_for_ds(db, datastore_id)

    # ── 3. Neo4j cleanup ─────────────────────────────────────────────────
    _delete_neo4j_for_ds(datastore_id)
    logger.info("Datastore deleted id=%d name=%s", ds_id_log, ds_name_log)

    return {"message": "Datastore and all associated data deleted successfully"}, 204
