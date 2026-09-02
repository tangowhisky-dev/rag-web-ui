"""Shared helpers for the knowledge-base API sub-package.

Provides filtering and utility functions used across the KB, document,
and datastore-linking endpoints:

- ``_get_user_org_ids``    - org hierarchy IDs for RBAC scoping
- ``_kb_owner_filter``     - SQLAlchemy clause scoping KB access by user
- ``_file_chunks``         - async generator for streaming UploadFile
- ``_get_chunk_scope_filter`` - SQLAlchemy filters for chunk queries
- ``_delete_qdrant_points``   - delete chunk vectors from Qdrant
"""

import logging
from typing import List

from fastapi import UploadFile
from qdrant_client import QdrantClient
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import get_org_descendants
from app.models.knowledge import KnowledgeBase, Document, DocumentChunk

logger = logging.getLogger(__name__)


def _get_user_org_ids(db: Session, org_id: int) -> List[int]:
    """Get all org IDs in the user's hierarchy (user's org + all descendant orgs).

    Delegates to the shared get_org_descendants helper in app.core.security.
    """
    return get_org_descendants(db, org_id)


def _kb_owner_filter(current_user):
    """Return SQLAlchemy filter clause scoping KB access.
    Users can only access KBs they personally own (user_id).
    """
    return KnowledgeBase.user_id == current_user.id


async def _file_chunks(file: UploadFile, chunk_size: int = 1024 * 1024):
    """Yield chunks from an UploadFile without loading it all into memory."""
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        yield chunk


def _get_chunk_scope_filter(document: Document, kb_id: int) -> list:
    """Return a list of SQLAlchemy filter conditions for chunk queries.

    DataStore documents have kb_id=NULL and data_store_id set, so we
    must use the correct scope filter to find their chunks.
    """
    if document.data_store_id is not None:
        return [DocumentChunk.data_store_id == document.data_store_id]
    return [
        DocumentChunk.kb_id == kb_id,
        DocumentChunk.data_store_id.is_(None),
    ]


def _delete_qdrant_points(
    document: Document,
    chunk_ids: list[int],
    kb_id: int,
    cleanup_warnings: list[str],
) -> None:
    """Delete chunk vectors from Qdrant for the given document."""
    from app.services.ingestion import _chunk_id_to_point_id
    from qdrant_client.models import PointIdsList

    if not chunk_ids:
        return
    try:
        qdrant = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        if document.data_store_id is not None:
            collection_name = f"ds_{document.data_store_id}"
        else:
            collection_name = f"kb_{kb_id}"
        existing = {c.name for c in qdrant.get_collections().collections}
        if collection_name not in existing:
            logger.debug(f"Qdrant collection {collection_name} does not exist — skipping point deletion for document {document.id}")
        else:
            point_ids = [_chunk_id_to_point_id(cid) for cid in chunk_ids]
            qdrant.delete(
                collection_name=collection_name,
                points_selector=PointIdsList(points=point_ids),
            )
            logger.debug(f"Deleted {len(point_ids)} Qdrant points for document {document.id}")
    except Exception as e:
        cleanup_warnings.append(f"Qdrant cleanup warning: {str(e)}")
        logger.error(f"Failed to delete Qdrant points for document {document.id}: {e}")
