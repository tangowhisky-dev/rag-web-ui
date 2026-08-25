"""Startup reconciliation — cross-store consistency sweep.

Compares Qdrant, Neo4j, and MySQL state on app startup and removes
orphaned data left by failed transactions, crashed workers, or prior
code paths that skipped cleanup.

Three passes, in order:
1. MySQL: delete orphan chunks/tasks whose document was already removed.
2. Qdrant: drop collections for deleted KBs/DataStores; delete orphan
   points within active collections (chunk_id not in MySQL).
3. Neo4j: purge stale Chunk/Entity nodes for KBs/DataStores no longer
   in MySQL.

All passes are resilient — a failure in one store does not block the
others.  Each pass logs a summary of what was cleaned.
"""
from __future__ import annotations

import logging
from typing import List

from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.knowledge import (
    KnowledgeBase,
    Document,
    DocumentChunk,
    ProcessingTask,
)
from app.models.datastore import DataStore

logger = logging.getLogger(__name__)


def run_reconciliation() -> dict:
    """Run all reconciliation passes and return a summary dict.

    Safe to call on every startup.  Each pass is independently try/excepted.
    """
    summary: dict = {
        "mysql": {"orphan_chunks": 0, "orphan_tasks": 0},
        "qdrant": {"dropped_collections": 0, "orphan_points": 0},
        "neo4j": {"purged_kbs": 0, "purged_datastores": 0},
    }

    db = SessionLocal()
    try:
        active_kb_ids = [kb.id for kb in db.query(KnowledgeBase.id).all()]
        active_ds_ids = [ds.id for ds in db.query(DataStore.id).all()]
        logger.info("[RECONCILE] active_kb_ids=%s active_ds_ids=%s", active_kb_ids, active_ds_ids)
    except Exception as e:
        logger.warning("[RECONCILE] Could not query active KB/DataStore IDs: %s", e)
        db.close()
        return summary
    finally:
        try:
            db.close()
        except Exception:
            pass

    # ── 1. MySQL orphan cleanup ──────────────────────────────────────
    try:
        _reconcile_mysql(summary)
    except Exception as e:
        logger.warning("[RECONCILE] MySQL pass failed: %s", e, exc_info=True)

    # ── 2. Qdrant reconciliation ─────────────────────────────────────
    try:
        _reconcile_qdrant(summary, active_kb_ids, active_ds_ids)
    except Exception as e:
        logger.warning("[RECONCILE] Qdrant pass failed: %s", e, exc_info=True)

    # ── 3. Neo4j reconciliation ──────────────────────────────────────
    try:
        _reconcile_neo4j(summary, active_kb_ids, active_ds_ids)
    except Exception as e:
        logger.warning("[RECONCILE] Neo4j pass failed: %s", e, exc_info=True)

    logger.info("[RECONCILE] complete: %s", summary)
    return summary


# ── MySQL ────────────────────────────────────────────────────────────


def _reconcile_mysql(summary: dict) -> None:
    """Delete orphan DocumentChunk and ProcessingTask rows whose document
    no longer exists in the documents table.

    These arise when a Document is deleted but the chunk/task rows survive
    (e.g. a failed cascade or a manual DB delete).
    """
    db = SessionLocal()
    try:
        # Orphan chunks: document_id not in documents
        orphan_chunk_ids = [
            row[0] for row in db.query(DocumentChunk.id)
            .outerjoin(Document, DocumentChunk.document_id == Document.id)
            .filter(Document.id.is_(None))
            .all()
        ]
        if orphan_chunk_ids:
            db.query(DocumentChunk).filter(
                DocumentChunk.id.in_(orphan_chunk_ids)
            ).delete(synchronize_session=False)
            db.commit()
            summary["mysql"]["orphan_chunks"] = len(orphan_chunk_ids)
            logger.info("[RECONCILE] MySQL: deleted %d orphan chunk rows", len(orphan_chunk_ids))

        # Orphan tasks: document_id not null and not in documents
        orphan_task_ids = [
            row[0] for row in db.query(ProcessingTask.id)
            .outerjoin(Document, ProcessingTask.document_id == Document.id)
            .filter(
                Document.id.is_(None),
                ProcessingTask.document_id.isnot(None),
            )
            .all()
        ]
        if orphan_task_ids:
            db.query(ProcessingTask).filter(
                ProcessingTask.id.in_(orphan_task_ids)
            ).delete(synchronize_session=False)
            db.commit()
            summary["mysql"]["orphan_tasks"] = len(orphan_task_ids)
            logger.info("[RECONCILE] MySQL: deleted %d orphan task rows", len(orphan_task_ids))
    finally:
        db.close()


# ── Qdrant ───────────────────────────────────────────────────────────


def _reconcile_qdrant(summary: dict, active_kb_ids: List[int], active_ds_ids: List[int]) -> None:
    """Drop Qdrant collections for deleted KBs/DataStores and delete
    orphaned points within active collections."""
    from app.services.infrastructure.utils import get_qdrant_client
    from app.services.ingestion import _chunk_id_to_point_id

    qdrant = get_qdrant_client()

    # Get all existing collections
    try:
        collections = [c.name for c in qdrant.get_collections().collections]
    except Exception as e:
        logger.warning("[RECONCILE] Qdrant: could not list collections: %s", e)
        return

    logger.info("[RECONCILE] Qdrant: collections=%s", collections)

    # ── Drop stale kb_ collections ────────────────────────────────
    active_kb_names = {f"kb_{kid}" for kid in active_kb_ids}
    stale_kb_collections = [c for c in collections if c.startswith("kb_") and c not in active_kb_names]
    logger.info("[RECONCILE] Qdrant: active_kb_names=%s stale_kb_collections=%s", active_kb_names, stale_kb_collections)
    for cname in stale_kb_collections:
        try:
            logger.info("[RECONCILE] Qdrant: dropping stale collection %s", cname)
            qdrant.delete_collection(cname)
            logger.info("[RECONCILE] Qdrant: dropped stale collection %s", cname)
        except Exception as e:
            logger.warning("[RECONCILE] Qdrant: failed to drop %s: %s", cname, e)
    summary["qdrant"]["dropped_collections"] += len(stale_kb_collections)

    # ── Drop stale ds_ collections ────────────────────────────────
    active_ds_names = {f"ds_{did}" for did in active_ds_ids}
    stale_ds_collections = [c for c in collections if c.startswith("ds_") and c not in active_ds_names]
    logger.info("[RECONCILE] Qdrant: active_ds_names=%s stale_ds_collections=%s", active_ds_names, stale_ds_collections)
    for cname in stale_ds_collections:
        try:
            logger.info("[RECONCILE] Qdrant: dropping stale collection %s", cname)
            qdrant.delete_collection(cname)
            logger.info("[RECONCILE] Qdrant: dropped stale collection %s", cname)
        except Exception as e:
            logger.warning("[RECONCILE] Qdrant: failed to drop %s: %s", cname, e)
    summary["qdrant"]["dropped_collections"] += len(stale_ds_collections)

    # ── Delete orphan points within active collections ────────────
    # For each active kb_ collection, check if Qdrant points have
    # corresponding DocumentChunk rows in MySQL.  Delete orphans.
    db = SessionLocal()
    try:
        for kid in active_kb_ids:
            cname = f"kb_{kid}"
            if cname not in collections:
                continue
            _delete_orphan_points(qdrant, db, cname, kb_id=kid, summary=summary)

        for did in active_ds_ids:
            cname = f"ds_{did}"
            if cname not in collections:
                continue
            _delete_orphan_points(qdrant, db, cname, data_store_id=did, summary=summary)
    finally:
        db.close()


def _delete_orphan_points(
    qdrant,
    db: Session,
    collection_name: str,
    summary: dict,
    kb_id: int | None = None,
    data_store_id: int | None = None,
) -> None:
    """Delete Qdrant points whose chunk_id has no matching DocumentChunk row."""
    from qdrant_client.models import PointIdsList, ScrollRequest
    from app.services.ingestion import _chunk_id_to_point_id

    try:
        # Get all point IDs from Qdrant via scroll
        all_point_ids: list[str] = []
        offset = None
        while True:
            results, offset = qdrant.scroll(
                collection_name=collection_name,
                limit=500,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            if not results:
                break
            all_point_ids.extend(str(p.id) for p in results)
            if offset is None:
                break

        if not all_point_ids:
            return

        # Get all chunk IDs from MySQL for this collection's scope
        query = db.query(DocumentChunk.id)
        if kb_id is not None:
            query = query.filter(DocumentChunk.kb_id == kb_id)
        elif data_store_id is not None:
            query = query.filter(DocumentChunk.data_store_id == data_store_id)

        mysql_chunk_ids = {str(_chunk_id_to_point_id(row[0])) for row in query.all()}

        # Orphan points: in Qdrant but not in MySQL
        orphan_ids = [pid for pid in all_point_ids if pid not in mysql_chunk_ids]
        if orphan_ids:
            qdrant.delete(
                collection_name=collection_name,
                points_selector=PointIdsList(points=orphan_ids),
            )
            summary["qdrant"]["orphan_points"] += len(orphan_ids)
            logger.info(
                "[RECONCILE] Qdrant: deleted %d orphan points from %s",
                len(orphan_ids), collection_name,
            )
    except Exception as e:
        logger.warning("[RECONCILE] Qdrant: orphan point scan failed for %s: %s", collection_name, e)


# ── Neo4j ────────────────────────────────────────────────────────────


def _reconcile_neo4j(summary: dict, active_kb_ids: List[int], active_ds_ids: List[int]) -> None:
    """Purge stale Neo4j Chunk/Entity nodes for deleted KBs and DataStores."""
    from app.core.config import settings

    if not settings.NEO4J_URI:
        return

    from app.services.graph import purge_stale_graph_data, _get_driver

    # ── KB-level purge (existing function) ────────────────────────
    try:
        purge_stale_graph_data(active_kb_ids=active_kb_ids)
        # purge_stale_graph_data logs its own summary, but we can't easily
        # get the count back.  Mark as run.
        summary["neo4j"]["purged_kbs"] = -1  # sentinel: ran but count unknown
    except Exception as e:
        logger.warning("[RECONCILE] Neo4j: purge_stale_graph_data failed: %s", e)

    # ── DataStore-level purge ─────────────────────────────────────
    # purge_stale_graph_data only handles KB-scoped chunks (data_store_id IS NULL).
    # We also need to sweep Chunk nodes whose data_store_id is no longer active.
    try:
        driver = _get_driver()
        active_ds_str = [str(did) for did in active_ds_ids]

        with driver.session() as session:
            # Find stale data_store_ids in Neo4j
            try:
                stale_rec = session.run(
                    """
                    MATCH (c:Chunk)
                    WHERE c.data_store_id IS NOT NULL
                      AND NOT c.data_store_id IN $active_ids
                    RETURN DISTINCT c.data_store_id AS stale_ds_id
                    """,
                    active_ids=active_ds_str,
                )
                stale_ds_ids = [r["stale_ds_id"] for r in stale_rec if r["stale_ds_id"] is not None]
            except Exception:
                stale_ds_ids = []

        if stale_ds_ids:
            logger.info("[RECONCILE] Neo4j: found stale data_store_ids: %s", stale_ds_ids)
            total_chunks = 0
            for stale_ds_id in stale_ds_ids:
                # Delete Chunk nodes in batches
                while True:
                    with driver.session() as session:
                        rec = session.run(
                            """
                            MATCH (c:Chunk {data_store_id: $ds_id})
                            WITH c LIMIT 100
                            DETACH DELETE c
                            RETURN count(c) AS n
                            """,
                            ds_id=stale_ds_id,
                        ).single()
                        n = rec["n"] if rec else 0
                        total_chunks += n
                        if n == 0:
                            break

                # Delete ALL entities scoped to this stale datastore.
                # Entities carry data_store_id as a property. We delete all
                # entities for the stale datastore, not just orphaned ones,
                # because graph builds that completed after the datastore
                # was deleted may have written entities WITH FROM_CHUNK
                # edges to newly written (and also stale) chunk nodes.
                with driver.session() as session:
                    rec = session.run(
                        """
                        MATCH (e)
                        WHERE (e:__KGBuilder__ OR e:Entity OR e:__Entity__)
                          AND e.data_store_id = $ds_id
                        DETACH DELETE e
                        RETURN count(e) AS n
                        """,
                        ds_id=stale_ds_id,
                    ).single()
                    if rec and rec["n"]:
                        logger.info(
                            "[RECONCILE] Neo4j: cleaned %d entities for stale ds_%s",
                            rec["n"], stale_ds_id,
                        )

            summary["neo4j"]["purged_datastores"] = len(stale_ds_ids)
            logger.info("[RECONCILE] Neo4j: purged %d chunks across %d stale datastores", total_chunks, len(stale_ds_ids))

    except Exception as e:
        logger.warning("[RECONCILE] Neo4j: datastore purge failed: %s", e)
