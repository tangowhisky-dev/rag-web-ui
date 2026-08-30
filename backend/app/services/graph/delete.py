"""Graph deletion and stale data purge.

Removes Chunk nodes, Entity nodes, and inter-entity relationships
from Neo4j when documents or knowledge bases are deleted, and
sweeps historical debris left by prior code paths that skipped
cleanup.

All deletion functions run regardless of GRAPHRAG_ENABLED — data
may exist from a prior ingest run when the flag was on.

  delete_graph_for_document()         — remove chunks + orphaned entities
  delete_graph_for_kb()               — remove all nodes for a deleted KB
  _batch_delete_chunks()              — batched chunk deletion (memory-safe)
  _batch_delete_orphaned_entities()   — batched entity cleanup
  purge_stale_graph_data()            — sweep KBs no longer in MySQL
"""

import logging
from typing import Optional

import neo4j

from app.core.config import settings

from .setup import _get_driver

logger = logging.getLogger(__name__)


def delete_graph_for_document(
    kb_id: Optional[int],
    document_id: int,
    data_store_id: Optional[int] = None,
) -> None:
    """
    Remove all Neo4j Chunk nodes for a deleted document, and clean up
    any Entity nodes that no longer have any Chunk connections.

    NOT gated on GRAPHRAG_ENABLED — data may exist from a prior ingest
    run when the flag was on.
    """
    if not settings.NEO4J_URI:
        return

    driver = _get_driver()
    with driver.session() as session:
        rec = session.run(
            """
            MATCH (c:Chunk {document_id: $doc_id})
            DETACH DELETE c
            RETURN count(c) AS deleted
            """,
            doc_id=str(document_id),
        ).single()
        logger.info(
            "GraphService: deleted %d Chunk nodes for doc %d",
            rec["deleted"] if rec else 0, document_id,
        )

        # Clean up orphaned entity nodes. With per-KB/datastore scoping,
        # entities from other KBs are separate nodes and won't be affected.
        # Only delete entities that have no remaining FROM_CHUNK edges.
        # Scope the cleanup to avoid deleting orphaned entities from other KBs.
        rec = session.run(
            """
            MATCH (e)
            WHERE (e:__KGBuilder__ OR e:Entity OR e:__Entity__)
              AND NOT EXISTS { MATCH (e)-[:FROM_CHUNK]->() }
              AND (
                ($kb_id IS NOT NULL AND e.kb_id = $kb_id)
                OR ($ds_id IS NOT NULL AND e.data_store_id = $ds_id)
                OR (e.kb_id IS NULL AND e.data_store_id IS NULL)
              )
            DETACH DELETE e
            RETURN count(e) AS cleaned
            """,
            kb_id=str(kb_id) if kb_id is not None else None,
            ds_id=str(data_store_id) if data_store_id is not None else None,
        ).single()
        logger.info(
            "GraphService: cleaned %d orphaned entity nodes after doc %d deletion",
            rec["cleaned"] if rec else 0, document_id,
        )


def delete_graph_for_kb(kb_id: int) -> None:
    """
    Remove all Neo4j nodes for an entire deleted knowledge base.

    NOT gated on GRAPHRAG_ENABLED — same reasoning as delete_graph_for_document.
    """
    if not settings.NEO4J_URI:
        return

    driver = _get_driver()
    with driver.session() as session:
        # 1. Delete all inter-entity relationships stamped with this KB's id.
        #    The ReLiK pipeline writes these as MERGE (a)-[:REL {kb_id: ...}]->(b).
        #    Must run before the chunk DETACH DELETE while entity nodes still exist.
        rec = session.run(
            """
            MATCH ()-[r {kb_id: $kb_id}]->()
            DELETE r
            RETURN count(r) AS deleted_rels
            """,
            kb_id=str(kb_id),
        ).single()
        logger.info(
            "GraphService: deleted %d inter-entity relationships for kb_%d",
            rec["deleted_rels"] if rec else 0, kb_id,
        )

        # 2. Delete Chunk nodes for direct uploads only (data_store_id IS NULL).
        #    DataStore document chunks are preserved — they belong to the datastore, not the KB.
        rec = session.run(
            """
            MATCH (c:Chunk {kb_id: $kb_id})
            WHERE c.data_store_id IS NULL
            DETACH DELETE c
            RETURN count(c) AS deleted
            """,
            kb_id=str(kb_id),
        ).single()
        logger.info(
            "GraphService: deleted %d Chunk nodes (direct uploads only) for kb_%d",
            rec["deleted"] if rec else 0, kb_id,
        )

        # 3. Sweep entity nodes scoped to this KB that have no remaining
        #    FROM_CHUNK edges. With per-KB scoping, entities from other KBs
        #    are separate nodes and won't be affected.
        #    Also covers legacy entities (no kb_id property) that were orphaned.
        rec = session.run(
            """
            MATCH (e)
            WHERE (e:__KGBuilder__ OR e:Entity OR e:__Entity__)
              AND (e.kb_id = $kb_id OR e.kb_id IS NULL)
              AND NOT EXISTS { MATCH (e)-[:FROM_CHUNK]->() }
            DETACH DELETE e
            RETURN count(e) AS cleaned
            """,
            kb_id=str(kb_id),
        ).single()
        logger.info(
            "GraphService: cleaned %d orphaned entity nodes after kb_%d deletion",
            rec["cleaned"] if rec else 0, kb_id,
        )


def _batch_delete_chunks(driver: neo4j.Driver, stale_id: str) -> int:
    total = 0
    while True:
        with driver.session() as session:
            rec = session.run(
                """
                MATCH (c:Chunk {kb_id: $kb_id})
                WITH c LIMIT 100
                DETACH DELETE c
                RETURN count(c) AS n
                """,
                kb_id=stale_id,
            ).single()
            n = rec["n"] if rec else 0
            total += n
            if n == 0:
                break
    return total


def _batch_delete_orphaned_entities(driver: neo4j.Driver, stale_id: str) -> int:
    total = 0
    while True:
        with driver.session() as session:
            rec = session.run(
                """
                MATCH (e)
                WHERE (e:__KGBuilder__ OR e:Entity OR e:__Entity__)
                  AND (e.kb_id = $kb_id OR e.kb_id IS NULL)
                  AND NOT EXISTS { MATCH (e)-[:FROM_CHUNK]->() }
                WITH e LIMIT 500
                DETACH DELETE e
                RETURN count(e) AS n
                """,
                kb_id=stale_id,
            ).single()
            n = rec["n"] if rec else 0
            total += n
            if n == 0:
                break
    return total


def purge_stale_graph_data(active_kb_ids: list[int]) -> None:
    """
    Delete any Chunk nodes (and their dependent entities) whose kb_id is not
    in active_kb_ids.  Call this after every KB deletion to sweep historical
    debris left by prior code paths that skipped Neo4j cleanup.
    """
    if not settings.NEO4J_URI:
        return

    driver = _get_driver()
    active_str = [str(i) for i in active_kb_ids]

    with driver.session() as session:
        # Find kb_ids present in Neo4j that are no longer in MySQL.
        # Only consider direct uploads (data_store_id IS NULL) — DataStore docs persist.
        # Wrap in a try-catch to handle case when Chunk label doesn't exist.
        try:
            stale_rec = session.run(
                """
                MATCH (c:Chunk)
                WHERE c.data_store_id IS NULL
                  AND NOT c.kb_id IN $active_ids
                RETURN DISTINCT c.kb_id AS stale_kb_id
                """,
                active_ids=active_str,
            )
            stale_ids = [r["stale_kb_id"] for r in stale_rec if r["stale_kb_id"] is not None]
        except Exception:
            # Chunk label doesn't exist or other error - no stale data to purge
            stale_ids = []

    if not stale_ids:
        return

    logger.info("GraphService: found stale kb_ids in Neo4j not in MySQL: %s", stale_ids)

    for stale_id in stale_ids:
        logger.info("GraphService: purging stale kb_%s in batches", stale_id)

        # Inter-entity rels stamped with this kb_id (usually small, single pass ok)
        with driver.session() as session:
            r1 = session.run(
                "MATCH ()-[r {kb_id: $kb_id}]->() DELETE r RETURN count(r) AS n",
                kb_id=stale_id,
            ).single()
            logger.info("GraphService: purged %d inter-entity rels for stale kb_%s", r1["n"] if r1 else 0, stale_id)

        # Chunk nodes in batches — each chunk can have hundreds of FROM_CHUNK
        # relationships, so deleting all at once blows the transaction memory limit.
        total_chunks = _batch_delete_chunks(driver, stale_id)
        logger.info("GraphService: purged %d chunks for stale kb_%s", total_chunks, stale_id)

        # Entity nodes scoped to this KB now orphaned — batch delete
        total_entities = _batch_delete_orphaned_entities(driver, stale_id)
        logger.info("GraphService: purged %d orphaned entities for stale kb_%s", total_entities, stale_id)
