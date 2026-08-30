"""Document endpoints for knowledge bases.

Endpoints:
    GET    /documents/{doc_id}                  - get document by ID (KB or datastore)
    GET    /documents/{doc_id}/download         - download document by ID
    DELETE /{kb_id}/documents/{doc_id}          - delete a document
    POST   /{kb_id}/documents/{doc_id}/retry    - retry failed ingestion
    GET    /{kb_id}/documents/{doc_id}          - get document details
    GET    /{kb_id}/documents/{doc_id}/download - download document

Includes ``_check_document_access``, a shared access-control helper
used by the static-path document routes (``/documents/{doc_id}``).
"""

import os
import asyncio
import logging
from typing import Any, Optional

from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.user import User
from app.core.security import get_current_user
from app.models.knowledge import (
    KnowledgeBase,
    Document,
    ProcessingTask,
    DocumentChunk,
    KnowledgeBaseDataStore,
)
from app.models.datastore import DataStore, OrganizationDataStore
from app.schemas.knowledge import DocumentResponse
from app.core.config import settings
from app.core.storage import delete_file

from app.api.api_v1.knowledge_base import router
from app.api.api_v1.knowledge_base.helpers import (
    _kb_owner_filter,
    _get_user_org_ids,
    _get_chunk_scope_filter,
    _delete_qdrant_points,
)
from app.api.api_v1.knowledge_base.ingestion import _process_and_graph

logger = logging.getLogger(__name__)


# ── Static-path GET routes (must be declared before /{kb_id}) ──────────────

@router.get("/documents/{doc_id}")
async def get_document_by_id(
    *,
    db: Session = Depends(get_db),
    doc_id: int,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Get document details by document ID alone (works for KB and data store docs).
    Used by citation popups that only have document_id from citation metadata.
    """
    document = db.query(Document).filter(Document.id == doc_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    if not _check_document_access(db, document, current_user):
        raise HTTPException(status_code=404, detail="Document not found")

    # Build a response that includes the parent name (KB or data store)
    parent_name = None
    if document.knowledge_base_id is not None:
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == document.knowledge_base_id).first()
        parent_name = kb.name if kb else None
    elif document.data_store_id is not None:
        from app.models.datastore import DataStore
        ds = db.query(DataStore).filter(DataStore.id == document.data_store_id).first()
        parent_name = ds.name if ds else None

    return {
        "id": document.id,
        "file_name": document.file_name,
        "title": document.title,
        "file_path": document.file_path,
        "file_size": document.file_size,
        "content_type": document.content_type,
        "knowledge_base_id": document.knowledge_base_id,
        "data_store_id": document.data_store_id,
        "parent_name": parent_name,
    }


@router.get("/documents/{doc_id}/download")
def download_document_by_id(
    doc_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Download a document by ID alone (works for KB and data store docs)."""
    document = db.query(Document).filter(Document.id == doc_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    if not _check_document_access(db, document, current_user):
        raise HTTPException(status_code=404, detail="Document not found")

    if not document.file_path:
        raise HTTPException(status_code=404, detail="File no longer available on disk")

    file_path = document.file_path
    if not os.path.isabs(file_path):
        file_path = os.path.join(settings.UPLOAD_DIR, file_path)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File no longer available on disk")

    from fastapi.responses import FileResponse
    return FileResponse(
        path=file_path,
        filename=document.file_name,
        media_type=document.content_type or "application/octet-stream",
    )


@router.delete("/{kb_id}/documents/{doc_id}")
async def delete_document(
    *,
    db: Session = Depends(get_db),
    kb_id: int,
    doc_id: int,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Delete a single document and all its associated data:
    - Physical file from local storage
    - All chunk vectors from Qdrant
    - All chunk records from MySQL
    - Processing task records from MySQL
    - The document record itself from MySQL
    """
    # Verify the KB belongs to this user/org
    kb = (
        db.query(KnowledgeBase)
        .filter(
            KnowledgeBase.id == kb_id,
            _kb_owner_filter(current_user)
        )
        .first()
    )
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    document = (
        db.query(Document)
        .filter(
            Document.id == doc_id,
            Document.knowledge_base_id == kb_id
        )
        .first()
    )
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    cleanup_warnings = []

    try:
        # 1. Collect chunk IDs before deleting them (needed for Qdrant point IDs).
        scope_filter = _get_chunk_scope_filter(document, kb_id)
        chunk_ids = [
            c.id for c in
            db.query(DocumentChunk.id)
            .filter(
                DocumentChunk.document_id == doc_id,
                *scope_filter
            )
            .all()
        ]

        # 2. Delete vectors from Qdrant
        _delete_qdrant_points(document, chunk_ids, kb_id, cleanup_warnings)

        # 3. Delete chunk rows from MySQL (explicit, don't rely on cascade here
        #    since we already fetched the IDs and want the delete to be transactional)
        db.query(DocumentChunk).filter(
            DocumentChunk.document_id == doc_id,
            *scope_filter
        ).delete(synchronize_session=False)

        # 4. Delete processing task records for this document
        db.query(ProcessingTask).filter(
            ProcessingTask.document_id == doc_id
        ).delete(synchronize_session=False)

        # 5. Delete the physical file from local storage
        try:
            delete_file(document.file_path)
            logger.info(f"Deleted file from storage: {document.file_path}")
        except Exception as e:
            cleanup_warnings.append(f"File storage cleanup warning: {str(e)}")
            logger.error(f"Failed to delete file {document.file_path}: {e}")

        # 5b. Delete Neo4j graph nodes for this document
        try:
            from app.services.graph import delete_graph_for_document
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: delete_graph_for_document(
                    kb_id=kb_id if document.data_store_id is None else None,
                    document_id=doc_id,
                    data_store_id=document.data_store_id,
                ),
            )
            logger.info(f"Deleted Neo4j graph nodes for document {doc_id}")
        except Exception as e:
            cleanup_warnings.append(f"Neo4j graph cleanup warning: {str(e)}")
            logger.error(f"Failed to delete Neo4j nodes for document {doc_id}: {e}")

        # 6. Delete the document record itself
        db.delete(document)
        db.commit()

        logger.info(f"Document {doc_id} deleted from KB {kb_id}")

        response = {"message": f"Document '{document.file_name}' deleted successfully"}
        if cleanup_warnings:
            response["warnings"] = cleanup_warnings
        return response

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to delete document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@router.post("/{kb_id}/documents/{doc_id}/retry", status_code=202)
async def retry_document_ingestion(
    *,
    db: Session = Depends(get_db),
    kb_id: int,
    doc_id: int,
    enable_ocr: Optional[bool] = None,
    current_user: User = Depends(get_current_user)
) -> Any:
    """Retry ingestion for a failed document.

    Resets the failed ProcessingTask to 'pending' and re-queues the
    background ingestion pipeline. Only works for KB documents (not
    DataStore documents — those are retried via scan/recovery).
    """
    # Verify KB ownership
    kb = (
        db.query(KnowledgeBase)
        .filter(
            KnowledgeBase.id == kb_id,
            _kb_owner_filter(current_user)
        )
        .first()
    )
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    document = (
        db.query(Document)
        .filter(
            Document.id == doc_id,
            Document.knowledge_base_id == kb_id
        )
        .first()
    )
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Find the failed task
    task = (
        db.query(ProcessingTask)
        .filter(
            ProcessingTask.document_id == doc_id,
            ProcessingTask.status == "failed"
        )
        .first()
    )
    if not task:
        raise HTTPException(
            status_code=409,
            detail=f"No failed task found for document {doc_id} (current status may already be processing or completed)"
        )

    # Verify the file still exists
    if not document.file_path or not os.path.isfile(document.file_path):
        raise HTTPException(
            status_code=410,
            detail=f"Source file no longer exists: {document.file_path}"
        )

    # Reset task to pending
    task.status = "pending"
    task.error_message = None
    task.progress = 0
    task.progress_message = "Queued for retry"
    db.commit()

    # Re-queue the background ingestion
    asyncio.create_task(
        _process_and_graph(
            document.file_path,
            document.file_name,
            kb_id,
            task.id,
            None,
            current_user.id,
            enable_ocr=enable_ocr,
        )
    )

    logger.info(f"Retry queued for document {doc_id} in KB {kb_id} (task {task.id})")
    return {"message": "Retry queued", "task_id": task.id, "document_id": doc_id}


@router.get("/{kb_id}/documents/{doc_id}", response_model=DocumentResponse)
async def get_document(
    *,
    db: Session = Depends(get_db),
    kb_id: int,
    doc_id: int,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Get document details by ID.
    """
    document = (
        db.query(Document)
        .join(KnowledgeBase)
        .filter(
            Document.id == doc_id,
            Document.knowledge_base_id == kb_id,
            _kb_owner_filter(current_user)
        )
        .first()
    )

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    return document


def _check_document_access(db: Session, document: Document, current_user: User) -> bool:
    """Check if the current user can access this document.

    - KB documents: user must own the KB.
    - Data store documents: user must own a KB linked to the data store,
      AND the data store must still be assigned to the user's org.
    """
    if document.knowledge_base_id is not None:
        kb = db.query(KnowledgeBase).filter(
            KnowledgeBase.id == document.knowledge_base_id,
            _kb_owner_filter(current_user),
        ).first()
        return kb is not None

    if document.data_store_id is not None:
        # Check if user owns any KB linked to this data store
        linked_kb = (
            db.query(KnowledgeBase)
            .join(KnowledgeBaseDataStore, KnowledgeBaseDataStore.knowledge_base_id == KnowledgeBase.id)
            .filter(
                KnowledgeBaseDataStore.data_store_id == document.data_store_id,
                _kb_owner_filter(current_user),
            )
            .first()
        )
        if linked_kb is None:
            return False

        # Verify the datastore is still assigned to the user's org
        if current_user.org_id:
            user_org_ids = _get_user_org_ids(db, current_user.org_id)
            org_link = (
                db.query(OrganizationDataStore)
                .filter(
                    OrganizationDataStore.data_store_id == document.data_store_id,
                    OrganizationDataStore.org_id.in_(user_org_ids),
                    OrganizationDataStore.is_active == True,
                )
                .first()
            )
            return org_link is not None

        return False

    return False


@router.get("/{kb_id}/documents/{doc_id}/download")
def download_document(
    kb_id: int,
    doc_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Download the original uploaded document file."""
    document = (
        db.query(Document)
        .join(KnowledgeBase)
        .filter(
            Document.id == doc_id,
            Document.knowledge_base_id == kb_id,
            _kb_owner_filter(current_user),
        )
        .first()
    )
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    if not document.file_path:
        raise HTTPException(status_code=404, detail="File no longer available on disk")

    # file_path is stored relative to UPLOAD_DIR.
    file_path = document.file_path
    if not os.path.isabs(file_path):
        file_path = os.path.join(settings.UPLOAD_DIR, file_path)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File no longer available on disk")

    from fastapi.responses import FileResponse
    return FileResponse(
        path=file_path,
        filename=document.file_name,
        media_type=document.content_type or "application/octet-stream",
    )
