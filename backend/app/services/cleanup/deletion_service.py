"""Shared deletion service for knowledge bases and datastores.

Cleans up resources across MySQL, Qdrant, Neo4j, and the filesystem in a
consistent order (Qdrant → Neo4j → DB → files).  Both the KB and DataStore
deletion endpoints delegate to this module so the logic lives in one place.

Pipeline types: ``"kb"`` or ``"ds"``.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.knowledge import (
    Document,
    DocumentChunk,
    KnowledgeBase,
)
from app.services.ingestion import _chunk_id_to_point_id
from app.services.graph import (
    delete_graph_for_kb,
    purge_stale_graph_data,
)

logger = logging.getLogger(__name__)


def _get_qdrant() -> QdrantClient:
    """Lazy singleton for Qdrant client."""
    return QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)


# ── Filesystem cleanup ────────────────────────────────────────────────────────


def _delete_kb_files(user_id: int, kb_id: int) -> None:
    """Remove the entire user_{user_id}/kb_{kb_id}/ directory tree."""
    import shutil
    kb_dir = Path(settings.UPLOAD_DIR) / "user_" / str(user_id) / f"kb_{kb_id}"
    if kb_dir.exists():
        shutil.rmtree(kb_dir)
        logger.info("DeletionService: deleted KB directory %s", kb_dir)
    else:
        logger.info("DeletionService: KB directory not found (skip): %s", kb_dir)


# ── Qdrant cleanup ───────────────────────────────────────────────────────────


def _delete_qdrant_for_kb(kb_id: int) -> None:
    """Delete the Qdrant collection for a KB (direct uploads only)."""
    collection_name = f"kb_{kb_id}"
    try:
        qdrant = _get_qdrant()
        qdrant.delete_collection(collection_name)
        logger.info("DeletionService: deleted Qdrant collection %s", collection_name)
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
                qdrant.delete_collection(collection_name)
                logger.info("DeletionService: deleted Qdrant collection %s", collection_name)
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
        asyncio.get_event_loop().run_in_executor(
            None, lambda: delete_graph_for_kb(kb_id=kb_id)
        )
        asyncio.get_event_loop().run_in_executor(
            None, lambda: purge_stale_graph_data(active_kb_ids=remaining_kb_ids)
        )
        logger.info("DeletionService: cleaned Neo4j graph for kb_%d", kb_id)
    except Exception as e:
        logger.warning("DeletionService: Neo4j delete failed for kb_%d: %s", kb_id, e)


def _delete_neo4j_for_ds(datastore_id: int) -> None:
    """Delete Neo4j Chunk nodes for a DataStore, and orphaned Entity nodes."""
    try:
        from app.services.graph import _get_driver
        if graph_settings.NEO4J_URI:
            driver = _get_driver()
            with driver.session() as session:
                # 1. Delete Chunk nodes for this datastore
                session.run(
                    """
                    MATCH (c:Chunk {data_store_id: $data_store_id})
                    DETACH DELETE c
                    """,
                    data_store_id=str(datastore_id),
                )
                # 2. Clean up orphaned Entity nodes (no FROM_CHUNK edges remain)
                session.run(
                    """
                    MATCH (e)
                    WHERE (e:__KGBuilder__ OR e:Entity OR e:__Entity__)
                      AND NOT EXISTS { MATCH (e)-[:FROM_CHUNK]->() }
                    DETACH DELETE e
                    """,
                )
            logger.info("DeletionService: cleaned Neo4j Chunk and Entity nodes for ds_%d", datastore_id)
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
        .filter(KnowledgeBase.id == kb_id)
        .first()
    )
    if not kb:
        return {"message": "Knowledge base not found"}, 404

    # Categorize documents
    direct_docs = [doc for doc in kb.documents if doc.data_store_id is None]
    datastore_docs = [doc for doc in kb.documents if doc.data_store_id is not None]
    cleanup_errors = []

    # ── 1. Filesystem cleanup (direct uploads only) ──────────────────────
    if direct_docs:
        try:
            _delete_kb_files(user_id, kb_id)
        except Exception as e:
            cleanup_errors.append(f"Failed to clean up storage files: {e}")

    # ── 2. Qdrant cleanup (direct uploads only) ──────────────────────────
    if direct_docs:
        _delete_qdrant_for_kb(kb_id)

    # ── 3. Neo4j cleanup (direct uploads only) ───────────────────────────
    if direct_docs:
        _delete_neo4j_for_kb(db, kb_id)

    # ── 4. DB cleanup ────────────────────────────────────────────────────
    # Event listener handles conditional document deletion (direct uploads
    # are deleted; DataStore links are severed).
    db.delete(kb)
    db.commit()

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

    # ── 1. Qdrant cleanup ────────────────────────────────────────────────
    if datastore_docs:
        _delete_qdrant_for_ds(db, datastore_id)

    # ── 2. Neo4j cleanup ─────────────────────────────────────────────────
    _delete_neo4j_for_ds(datastore_id)

    # ── 3. DB cleanup ────────────────────────────────────────────────────
    # Delete junction records before DataStore (ORM cascade won't handle NOT
    # NULL FK constraints cleanly).
    db.query(KnowledgeBaseDataStore).filter(
        KnowledgeBaseDataStore.data_store_id == datastore_id
    ).delete(synchronize_session=False)

    db.query(OrganizationDataStore).filter(
        OrganizationDataStore.data_store_id == datastore_id
    ).delete(synchronize_session=False)

    # Delete documents explicitly (so we can log), then delete DataStore
    # — CASCADE handles chunks/tasks automatically.
    for doc in datastore_docs:
        db.delete(doc)
    db.commit()

    db.delete(ds)
    db.commit()
    logger.info("Datastore deleted id=%d name=%s", ds.id, ds.name)

    return {"message": "Datastore and all associated data deleted successfully"}, 204
