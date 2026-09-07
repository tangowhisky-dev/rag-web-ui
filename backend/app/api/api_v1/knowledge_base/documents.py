"""Document endpoints for knowledge bases.

Endpoints:
    GET    /documents/{doc_id}                  - get document by ID (KB or datastore)
    GET    /documents/{doc_id}/download         - download document by ID
    DELETE /{kb_id}/documents/{doc_id}          - delete a document
    POST   /{kb_id}/documents/{doc_id}/retry    - retry failed ingestion
    GET    /{kb_id}/documents/{doc_id}          - get document details
    GET    /{kb_id}/documents/{doc_id}/download - download document
    GET    /{kb_id}/documents/{doc_id}/markdown - get converted markdown
    PUT    /{kb_id}/documents/{doc_id}/markdown - save edited markdown + re-ingest
    GET    /{kb_id}/documents/{doc_id}/ingest-status - current ingestion/graph status
    POST   /{kb_id}/documents/{doc_id}/graph-pause  - pause graph ingestion for a file
    POST   /{kb_id}/documents/{doc_id}/graph-resume - resume graph ingestion for a file
    POST   /{kb_id}/documents/{doc_id}/graph-build  - start graph ingestion for a file

Includes ``_check_document_access``, a shared access-control helper
used by the static-path document routes (``/documents/{doc_id}``).
"""

import os
import asyncio
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

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
            logger.debug(f"Deleted file from storage: {document.file_path}")
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
            logger.debug(f"Deleted Neo4j graph nodes for document {doc_id}")
        except Exception as e:
            cleanup_warnings.append(f"Neo4j graph cleanup warning: {str(e)}")
            logger.error(f"Failed to delete Neo4j nodes for document {doc_id}: {e}")

        # 6. Delete the document record itself
        db.delete(document)
        db.commit()

        logger.debug(f"Document {doc_id} deleted from KB {kb_id}")

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

    logger.debug(f"Retry queued for document {doc_id} in KB {kb_id} (task {task.id})")
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


# ── Markdown editor + graph per-file controls ────────────────────────────────


class UpdateMarkdownRequest(BaseModel):
    markdown: str = Field(..., min_length=1)
    lock_version: int


def _verify_kb_document(db: Session, kb_id: int, doc_id: int, current_user: User):
    """Verify KB ownership and return the document, or raise 404."""
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
    return document


@router.get("/{kb_id}/documents/{doc_id}/markdown")
def get_kb_document_markdown(
    kb_id: int,
    doc_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get the converted markdown for a KB document (editor source).

    Available as soon as conversion_status is 'completed', even if graph
    ingestion is still running.
    """
    doc = _verify_kb_document(db, kb_id, doc_id, current_user)

    if doc.conversion_status == "processing":
        raise HTTPException(status_code=409, detail="Conversion in progress")
    if doc.conversion_status == "pending":
        raise HTTPException(status_code=409, detail="Conversion pending")
    if not doc.converted_markdown:
        if doc.conversion_status == "error":
            raise HTTPException(
                status_code=422,
                detail=f"Conversion failed: {doc.conversion_error or 'unknown error'}",
            )
        raise HTTPException(
            status_code=409,
            detail="Markdown not available — run re-convert first",
        )

    return {
        "document_id": doc.id,
        "markdown": doc.converted_markdown,
        "conversion_status": doc.conversion_status,
        "lock_version": doc.lock_version,
        "title": doc.title,
    }


@router.put("/{kb_id}/documents/{doc_id}/markdown")
async def update_kb_document_markdown(
    kb_id: int,
    doc_id: int,
    body: UpdateMarkdownRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Save edited markdown for a KB document and trigger immediate re-ingestion.

    1. Cancel any in-flight graph build for this document.
    2. Delete old chunks (MySQL + Qdrant) and graph nodes (Neo4j).
    3. Persist new markdown + bump lock version.
    4. Create a fresh ProcessingTask and re-ingest using the saved markdown
       (skip_conversion=True).
    5. Start fresh graph build if GRAPHRAG_ENABLED.
    """
    doc = _verify_kb_document(db, kb_id, doc_id, current_user)

    # Optimistic lock check
    if doc.lock_version != body.lock_version:
        raise HTTPException(
            status_code=409,
            detail=f"Document was modified by another editor. Expected lock_version={doc.lock_version}, got {body.lock_version}.",
        )

    # Check conversion is done
    if not doc.converted_markdown and doc.conversion_status != "completed":
        raise HTTPException(
            status_code=409,
            detail="Document has not been converted yet — run re-convert first",
        )

    # 1. Cancel any in-flight graph build for this document
    from app.services.ingestion.ingestion_dispatcher import cancel_graph_build_for_document
    latest_task = (
        db.query(ProcessingTask)
        .filter(ProcessingTask.document_id == doc_id)
        .order_by(ProcessingTask.id.desc())
        .first()
    )
    if latest_task:
        cancel_graph_build_for_document(latest_task.id)

    # 2. Delete old chunks + Qdrant points + graph nodes + old tasks
    from app.services.ingestion.reingest import reset_document_for_reingest
    reset_document_for_reingest(db, doc_id, data_store_id=None, kb_id=kb_id)

    # 3. Persist new markdown + bump lock version
    doc.converted_markdown = body.markdown
    doc.conversion_status = "completed"
    doc.lock_version = doc.lock_version + 1
    doc.file_edited_at = datetime.now(timezone.utc)

    # 4. Create a fresh ProcessingTask and re-ingest
    new_task = ProcessingTask(
        knowledge_base_id=kb_id,
        document_id=doc_id,
        status="pending",
        progress=0,
        progress_message="Queued for re-ingestion",
    )
    db.add(new_task)
    db.commit()
    db.refresh(new_task)

    logger.debug(
        "[EDITOR] kb_markdown_saved doc_id=%s kb_id=%s task_id=%s — re-ingesting",
        doc_id, kb_id, new_task.id,
    )

    # 5. Trigger re-ingestion in a background thread
    _file_path = doc.file_path
    _file_name = doc.file_name
    _task_id = new_task.id
    _user_id = current_user.id
    _file_hash = doc.file_hash
    _file_size = doc.file_size
    _content_type = doc.content_type

    def _reingest():
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                _process_and_graph(
                    _file_path,
                    _file_name,
                    kb_id,
                    _task_id,
                    None,
                    _user_id,
                    document_id=doc_id,
                    enable_graph=True,
                    file_hash=_file_hash,
                    file_size=_file_size,
                    content_type=_content_type,
                )
            )
        except Exception as e:
            logger.error("[EDITOR] kb_reingest_failed doc_id=%s: %s", doc_id, e, exc_info=True)
        finally:
            loop.close()

    t = threading.Thread(target=_reingest, name=f"kb-reingest-{doc_id}", daemon=True)
    t.start()

    return JSONResponse(
        status_code=202,
        content={
            "document_id": doc_id,
            "lock_version": doc.lock_version,
            "task_id": new_task.id,
            "message": "Markdown saved. Re-ingesting with fresh chunks and graph.",
        },
    )


@router.get("/{kb_id}/documents/{doc_id}/ingest-status")
def get_kb_ingest_status(
    kb_id: int,
    doc_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get the current ingestion/conversion/graph status for a KB document."""
    doc = _verify_kb_document(db, kb_id, doc_id, current_user)

    latest_task = (
        db.query(ProcessingTask)
        .filter(ProcessingTask.document_id == doc_id)
        .order_by(ProcessingTask.id.desc())
        .first()
    )

    chunk_count = db.query(DocumentChunk).filter(
        DocumentChunk.document_id == doc_id,
    ).count()

    return {
        "document_id": doc.id,
        "conversion_status": doc.conversion_status,
        "conversion_error": doc.conversion_error,
        "ingest_status": latest_task.status if latest_task else None,
        "ingest_progress": latest_task.progress if latest_task else 0,
        "ingest_message": latest_task.progress_message if latest_task else None,
        "ingest_error": latest_task.error_message if latest_task else None,
        "graph_status": latest_task.graph_status if latest_task else None,
        "graph_error": latest_task.graph_error if latest_task else None,
        "chunk_count": chunk_count,
        "lock_version": doc.lock_version,
    }


@router.post("/{kb_id}/documents/{doc_id}/graph-pause")
def pause_kb_graph_ingestion(
    kb_id: int,
    doc_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Pause/cancel graph ingestion for a single KB document."""
    doc = _verify_kb_document(db, kb_id, doc_id, current_user)

    latest_task = (
        db.query(ProcessingTask)
        .filter(ProcessingTask.document_id == doc_id)
        .order_by(ProcessingTask.id.desc())
        .first()
    )
    if not latest_task:
        raise HTTPException(status_code=409, detail="No processing task found for this document")

    from app.services.ingestion.ingestion_dispatcher import cancel_graph_build_for_document
    cancelled = cancel_graph_build_for_document(latest_task.id)

    return {
        "document_id": doc_id,
        "graph_status": latest_task.graph_status,
        "cancelled": cancelled,
        "message": "Graph ingestion paused" if cancelled else "No active graph build to pause",
    }


@router.post("/{kb_id}/documents/{doc_id}/graph-resume")
def resume_kb_graph_ingestion(
    kb_id: int,
    doc_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Resume graph ingestion for a single KB document.

    Resets graph_status from 'failed' (cancelled) back to 'pending' and
    starts a new graph build thread.
    """
    doc = _verify_kb_document(db, kb_id, doc_id, current_user)

    latest_task = (
        db.query(ProcessingTask)
        .filter(ProcessingTask.document_id == doc_id)
        .order_by(ProcessingTask.id.desc())
        .first()
    )
    if not latest_task:
        raise HTTPException(status_code=409, detail="No processing task found for this document")

    if latest_task.status != "completed":
        raise HTTPException(
            status_code=409,
            detail="Ingestion must be completed before graph build can run",
        )

    if latest_task.graph_status not in ("failed", None):
        if latest_task.graph_status == "completed":
            raise HTTPException(status_code=409, detail="Graph ingestion already completed")
        if latest_task.graph_status == "pending":
            # "pending" might mean a thread is running, OR it might be a
            # stale "pending" left by a cancelled thread (the cancel handler
            # sets "failed" but the thread's exit path overwrites it back to
            # "pending").  Check if there's actually an active thread.
            from app.services.ingestion.ingestion_dispatcher import (
                _active_graph_builds, _active_graph_lock,
            )
            with _active_graph_lock:
                is_active = latest_task.id in _active_graph_builds
            if is_active:
                raise HTTPException(status_code=409, detail="Graph ingestion already running")
            # Stale "pending" — treat as resumable
            logger.debug(
                "[GRAPH-RESUME] doc_id=%s task_id=%s — stale pending, no active thread",
                doc_id, latest_task.id,
            )

    # Validate chunks BEFORE committing graph_status=pending so a failure
    # doesn't leave a stuck "pending" state with no running thread.
    from app.services.ingestion.document_processor import GraphBuildRequest
    chunks = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == doc_id)
        .all()
    )
    if not chunks:
        raise HTTPException(
            status_code=409,
            detail="No chunks found — ingest the document first",
        )

    # Reset to pending and start graph build
    latest_task.graph_status = "pending"
    latest_task.graph_error = None
    db.commit()

    graph_req = GraphBuildRequest(
        document_id=doc_id,
        file_name=doc.file_name,
        chunks=[c.chunk_text for c in chunks],
        chunk_ids=[c.id for c in chunks],
        kb_id=kb_id,
        data_store_id=None,
        task_id=latest_task.id,
    )

    from app.services.ingestion.ingestion_dispatcher import _start_graph_build_thread
    try:
        _start_graph_build_thread(graph_req)
    except Exception as e:
        latest_task.graph_status = "failed"
        latest_task.graph_error = f"Failed to start graph build: {e}"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Failed to start graph build: {e}")

    return {
        "document_id": doc_id,
        "graph_status": "pending",
        "message": "Graph ingestion resumed",
    }


@router.post("/{kb_id}/documents/{doc_id}/graph-build")
def start_kb_graph_build(
    kb_id: int,
    doc_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Start graph ingestion for a KB document that doesn't have it yet.

    This is for files where graph ingestion was disabled during upload
    but the user later wants to enable it.
    """
    doc = _verify_kb_document(db, kb_id, doc_id, current_user)

    latest_task = (
        db.query(ProcessingTask)
        .filter(ProcessingTask.document_id == doc_id)
        .order_by(ProcessingTask.id.desc())
        .first()
    )
    if not latest_task:
        raise HTTPException(status_code=409, detail="No processing task found for this document")

    if latest_task.status != "completed":
        raise HTTPException(
            status_code=409,
            detail="Ingestion must be completed before graph build can run",
        )

    if latest_task.graph_status == "completed":
        raise HTTPException(status_code=409, detail="Graph ingestion already completed")

    if latest_task.graph_status == "pending":
        raise HTTPException(status_code=409, detail="Graph ingestion already running")

    # Validate chunks BEFORE committing graph_status=pending so a failure
    # doesn't leave a stuck "pending" state with no running thread.
    from app.services.ingestion.document_processor import GraphBuildRequest
    chunks = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == doc_id)
        .all()
    )
    if not chunks:
        raise HTTPException(
            status_code=409,
            detail="No chunks found — ingest the document first",
        )

    # Set to pending and start
    latest_task.graph_status = "pending"
    latest_task.graph_error = None
    db.commit()

    graph_req = GraphBuildRequest(
        document_id=doc_id,
        file_name=doc.file_name,
        chunks=[c.chunk_text for c in chunks],
        chunk_ids=[c.id for c in chunks],
        kb_id=kb_id,
        data_store_id=None,
        task_id=latest_task.id,
    )

    from app.services.ingestion.ingestion_dispatcher import _start_graph_build_thread
    try:
        _start_graph_build_thread(graph_req)
    except Exception as e:
        # Reset graph_status so the user can retry
        latest_task.graph_status = "failed"
        latest_task.graph_error = f"Failed to start graph build: {e}"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Failed to start graph build: {e}")

    return {
        "document_id": doc_id,
        "graph_status": "pending",
        "message": "Graph ingestion started",
    }
