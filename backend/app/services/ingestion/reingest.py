"""Re-ingest primitive for the markdown editor.

``reset_document_for_reingest`` deletes chunks (MySQL + Qdrant), graph
nodes (Neo4j), and the old ProcessingTask — but keeps the Document
record, its ``converted_markdown``, ``is_selected`` flag, and the
datastore manifest entry.  This is distinct from ``delete_document_data``
which unselects the document and removes the manifest.
"""
import logging
from typing import Optional

from qdrant_client.models import PointIdsList
from sqlalchemy import and_
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.knowledge import Document, DocumentChunk, ProcessingTask
from app.services.ingestion import _chunk_id_to_point_id
from app.services.infrastructure import get_qdrant_client

logger = logging.getLogger(__name__)


def reset_document_for_reingest(
    db: Session,
    document_id: int,
    data_store_id: Optional[int] = None,
    kb_id: Optional[int] = None,
) -> dict:
    """Delete chunks, vectors, graph, and old task for a document.

    Keeps:
    - The Document record (including converted_markdown)
    - is_selected flag
    - DataStoreFileManifest entry
    - The source file on disk

    Returns a dict with deletion counts.
    """
    document = db.get(Document, document_id)
    if not document:
        return {"deleted": False, "reason": "document not found"}

    # Use the document's own scope if not provided
    if data_store_id is None:
        data_store_id = document.data_store_id
    if kb_id is None:
        kb_id = document.knowledge_base_id

    # 1. Collect old chunk IDs
    if data_store_id is not None:
        scope_filter = DocumentChunk.data_store_id == data_store_id
    else:
        scope_filter = and_(
            DocumentChunk.kb_id == kb_id,
            DocumentChunk.data_store_id.is_(None),
        )
    chunk_ids = [
        row[0] for row in
        db.query(DocumentChunk.id)
        .filter(DocumentChunk.document_id == document_id, scope_filter)
        .all()
    ]

    # 2. Delete Qdrant points
    qdrant_deleted = 0
    if chunk_ids:
        collection_name = f"ds_{data_store_id}" if data_store_id else f"kb_{kb_id}"
        try:
            point_ids = [_chunk_id_to_point_id(cid) for cid in chunk_ids]
            get_qdrant_client().delete(
                collection_name=collection_name,
                points_selector=PointIdsList(points=point_ids),
            )
            qdrant_deleted = len(point_ids)
        except Exception as e:
            logger.warning("[REINGEST] Qdrant delete failed for doc %d: %s", document_id, e)

    # 3. Delete DocumentChunk rows
    if data_store_id is not None:
        scope_filter = DocumentChunk.data_store_id == data_store_id
    else:
        scope_filter = and_(
            DocumentChunk.kb_id == kb_id,
            DocumentChunk.data_store_id.is_(None),
        )
    db.query(DocumentChunk).filter(
        DocumentChunk.document_id == document_id, scope_filter,
    ).delete(synchronize_session="fetch")

    # 4. Delete old ProcessingTask(s) for this document
    tasks_deleted = db.query(ProcessingTask).filter(
        ProcessingTask.document_id == document_id,
    ).delete(synchronize_session="fetch")

    # 5. Delete Neo4j graph
    graph_deleted = 0
    try:
        from app.services.graph import delete_graph_for_document
        delete_graph_for_document(
            kb_id=kb_id,
            document_id=document_id,
            data_store_id=data_store_id,
        )
        graph_deleted = 1
    except Exception as e:
        logger.warning("[REINGEST] Neo4j cleanup failed for doc %d: %s", document_id, e)

    db.commit()

    logger.debug(
        "[REINGEST] doc_id=%d chunks=%d qdrant=%d tasks=%d graph=%d",
        document_id, len(chunk_ids), qdrant_deleted, tasks_deleted, graph_deleted,
    )

    return {
        "deleted": True,
        "document_id": document_id,
        "chunks_deleted": len(chunk_ids),
        "qdrant_points_deleted": qdrant_deleted,
        "tasks_deleted": tasks_deleted,
        "graph_nodes_deleted": graph_deleted,
    }
