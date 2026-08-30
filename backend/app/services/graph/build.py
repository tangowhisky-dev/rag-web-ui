"""Graph ingestion — build entity/relationship graph for a document.

Writes Chunk nodes to Neo4j (keyed by qdrant_point_id) and runs the
LLM extraction pipeline to produce Entity nodes + typed relationship
edges, linked back to their source chunks via FROM_CHUNK edges.

  build_graph_for_document()  — main entry point for ingestion
  _extract_seen_point_ids()   — helper used by graph expansion

Chunk nodes carry:
  qdrant_point_id    — primary cross-reference key (UUID string)
  qdrant_collection  — which Qdrant collection (kb_<kb_id> or ds_<data_store_id>)
  document_id        — for deletion queries
  chunk_index        — ordinal position within the document
  kb_id              — knowledge base
  data_store_id      — datastore (optional)
  file_name          — human-readable source
"""

import asyncio
import logging
from typing import Optional

from sqlalchemy.orm import Session

from langchain_core.documents import Document as LangchainDocument

from app.services.settings_service import get_setting

from .setup import _get_driver, _chunk_id_to_point_id
from .extraction import _extract_with_llm

logger = logging.getLogger(__name__)


async def build_graph_for_document(
    kb_id: Optional[int],
    document_id: int,
    file_name: str,
    chunks: list[str],
    chunk_ids: list[str],
    data_store_id: Optional[int] = None,
    pt=None,            # optional ProgressTimeout for periodic pings
    db: Optional[Session] = None,
    org_id: Optional[int] = None,
    task_id: Optional[int] = None,
    cancel_event=None,  # optional threading.Event — abort if set
) -> int:
    """Extract entity/relationship graph from document chunks and store in Neo4j.

    Returns the number of extraction batches skipped due to cancellation/pause.

    chunk_ids are the SHA-256 hex strings from document_processor — we convert
    them to the actual Qdrant point UUIDs here via the same deterministic
    uuid5 transform, so qdrant_point_id in Neo4j always matches the Qdrant
    point ID exactly.

    Chunk nodes carry:
      qdrant_point_id    — primary cross-reference key (UUID string)
      qdrant_collection  — which Qdrant collection (kb_<kb_id> or ds_<data_store_id>)
      document_id        — for deletion queries
      chunk_index        — ordinal position within the document
      kb_id              — knowledge base
      data_store_id      — datastore (optional)
      file_name          — human-readable source
    """
    if not get_setting(db, "GRAPHRAG_ENABLED", None):
        return

    # Resolve app-level ingestion settings from the settings service
    from app.db.session import SessionLocal
    _db = SessionLocal()
    try:
        graphrag_enabled = get_setting(_db, "GRAPHRAG_ENABLED", None)
        max_chunks = get_setting(_db, "GRAPHRAG_MAX_CHUNKS", None)
        neo4j_llm_context = get_setting(_db, "NEO4J_LLM_CONTEXT", None)
    finally:
        _db.close()

    if not graphrag_enabled:
        return

    driver = _get_driver()
    loop = asyncio.get_event_loop()

    # Convert SHA-256 chunk IDs → the actual UUIDs Qdrant uses as point IDs
    qdrant_point_ids = [_chunk_id_to_point_id(cid) for cid in chunk_ids]
    # Determine collection based on document source
    if data_store_id:
        collection_name = f"ds_{data_store_id}"
    else:
        collection_name = f"kb_{kb_id}"

    logger.info(
        "GraphService[llm]: writing %d Chunk nodes for doc %d (kb=%s, ds=%s)",
        len(chunks), document_id, kb_id, data_store_id,
    )

    # Write Chunk nodes — sync Neo4j I/O, must run in executor to avoid
    # blocking the event loop and causing ECONNRESET on concurrent requests.
    def _write_chunk_nodes():
        with driver.session() as session:
            for idx, (text, point_id) in enumerate(zip(chunks, qdrant_point_ids)):
                session.run(
                    """
                    MERGE (c:Chunk {qdrant_point_id: $point_id})
                    SET c.qdrant_collection = $collection,
                        c.document_id       = $document_id,
                        c.chunk_index       = $chunk_index,
                        c.kb_id             = $kb_id,
                        c.data_store_id     = $data_store_id,
                        c.file_name         = $file_name
                    """,
                    point_id=point_id,
                    collection=collection_name,
                    document_id=str(document_id),
                    chunk_index=idx,
                    kb_id=str(kb_id),
                    data_store_id=str(data_store_id) if data_store_id else None,
                    file_name=file_name,
                )

    await loop.run_in_executor(None, _write_chunk_nodes)

    # Check if cancelled before starting expensive LLM extraction.
    if cancel_event is not None and cancel_event.is_set():
        logger.info("GraphService[llm]: doc %d — cancelled before extraction", document_id)
        return 0

    # Extract entities and relationships.  Concurrency is controlled by the
    # global LLM semaphore inside _extract_with_llm, which caps total
    # concurrent LLM calls across all documents at 4.
    total_entities, total_relations, skipped_batches = await _extract_with_llm(
        document_id=document_id,
        file_name=file_name,
        chunks=chunks,
        qdrant_point_ids=qdrant_point_ids,
        pt=pt,
        max_chunks=max_chunks,
        neo4j_llm_context=neo4j_llm_context,
        kb_id=kb_id,
        data_store_id=data_store_id,
        task_id=task_id,
        cancel_event=cancel_event,
    )

    logger.info(
        "GraphService[llm]: doc %d — %d entities, %d relations written to Neo4j (skipped_batches=%d)",
        document_id, total_entities, total_relations, skipped_batches,
    )
    return skipped_batches


def _extract_seen_point_ids(docs: list[LangchainDocument]) -> set[str]:
    """Extract Qdrant point UUIDs from the retrieved docs' metadata."""
    seen = set()
    for doc in docs:
        pid = doc.metadata.get("qdrant_point_id")
        if pid:
            seen.add(pid)
    return seen
