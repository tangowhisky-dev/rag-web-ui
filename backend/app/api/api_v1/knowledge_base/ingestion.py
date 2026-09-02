"""Document ingestion endpoints for knowledge bases.

Endpoints:
    POST /{kb_id}/documents/upload   - stream-upload multiple documents to storage
    POST /{kb_id}/documents/preview  - preview chunk boundaries for documents
    POST /{kb_id}/documents/process  - queue background processing for uploads
    POST /cleanup                     - remove expired temp upload files
    GET  /{kb_id}/documents/tasks    - poll processing task statuses

Also provides ``add_processing_tasks_to_queue`` and ``_process_and_graph``,
internal helpers used by the process endpoint and by the retry endpoint
in ``documents.py``.
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, UploadFile, BackgroundTasks, Query
from sqlalchemy.orm import Session, selectinload
from sqlalchemy.exc import IntegrityError as SAIntegrityError

from app.db.session import get_db
from app.models.user import User
from app.core.security import get_current_user
from app.models.knowledge import (
    KnowledgeBase,
    Document,
    ProcessingTask,
    DocumentUpload,
)
from app.schemas.knowledge import PreviewRequest
from app.services.ingestion import (
    process_document_background,
    preview_document,
    PreviewResult,
)
from app.services.ingestion import SUPPORTED_EXTENSIONS
from app.core.storage import save_file_stream, delete_file

from app.api.api_v1.knowledge_base import router
from app.api.api_v1.knowledge_base.helpers import _kb_owner_filter, _file_chunks

logger = logging.getLogger(__name__)


# Batch upload documents
@router.post("/{kb_id}/documents/upload")
async def upload_kb_documents(
    kb_id: int,
    files: List[UploadFile],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Upload multiple documents to local storage.
    """
    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == kb_id,
        _kb_owner_filter(current_user)
    ).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    results = []
    for file in files:
        # Reject unsupported extensions early with a clear error
        _, ext = os.path.splitext(file.filename or "")
        if ext.lower() not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        # 1. Stream file to disk, computing hash and size as we go.
        #    This avoids loading the entire file into memory.
        temp_path = f"user_{current_user.id}/kb_{kb_id}/temp/{file.filename}"
        try:
            file_hash, file_size = await save_file_stream(temp_path, _file_chunks(file))
        except Exception as e:
            logger.error(f"Failed to save file to storage: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to upload file")

        # 2. 检查是否存在完全相同的文件（名称和hash都相同）
        existing_document = db.query(Document).filter(
            Document.file_name == file.filename,
            Document.file_hash == file_hash,
            Document.knowledge_base_id == kb_id
        ).first()

        if existing_document:
            # 完全相同的文件，直接返回 — clean up the temp file we just wrote.
            delete_file(temp_path)
            results.append({
                "document_id": existing_document.id,
                "file_name": existing_document.file_name,
                "status": "exists",
                "message": "文件已存在且已处理完成",
                "skip_processing": True
            })
            continue

        # 3. 创建上传记录
        upload = DocumentUpload(
            knowledge_base_id=kb_id,
            file_name=file.filename,
            file_hash=file_hash,
            file_size=file_size,
            content_type=file.content_type,
            temp_path=temp_path
        )
        db.add(upload)
        try:
            db.commit()
            db.refresh(upload)
        except SAIntegrityError:
            db.rollback()
            # Another concurrent request already created this upload — fetch it
            existing_upload = db.query(DocumentUpload).filter(
                DocumentUpload.knowledge_base_id == kb_id,
                DocumentUpload.file_name == file.filename,
                DocumentUpload.file_hash == file_hash,
            ).first()
            if existing_upload:
                results.append({
                    "upload_id": existing_upload.id,
                    "file_name": file.filename,
                    "temp_path": existing_upload.temp_path,
                    "status": "pending",
                    "skip_processing": False
                })
                continue
            raise

        results.append({
            "upload_id": upload.id,
            "file_name": file.filename,
            "temp_path": temp_path,
            "status": "pending",
            "skip_processing": False
        })

    return results

@router.post("/{kb_id}/documents/preview")
async def preview_kb_documents(
    kb_id: int,
    preview_request: PreviewRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[int, PreviewResult]:
    """
    Preview multiple documents' chunks.
    """
    results = {}
    for doc_id in preview_request.document_ids:
        document = db.query(Document).join(KnowledgeBase).filter(
            Document.id == doc_id,
            Document.knowledge_base_id == kb_id,
            _kb_owner_filter(current_user)
        ).first()

        if document:
            file_path = document.file_path
        else:
            upload = db.query(DocumentUpload).join(KnowledgeBase).filter(
                DocumentUpload.id == doc_id,
                DocumentUpload.knowledge_base_id == kb_id,
                _kb_owner_filter(current_user)
            ).first()

            if not upload:
                raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

            file_path = upload.temp_path

        preview = await preview_document(
            file_path,
            chunk_size=preview_request.chunk_size,
            chunk_overlap=preview_request.chunk_overlap
        )
        results[doc_id] = preview

    return results

@router.post("/{kb_id}/documents/process")
async def process_kb_documents(
    kb_id: int,
    upload_results: List[dict],
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Process multiple documents asynchronously.
    """
    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == kb_id,
        _kb_owner_filter(current_user)
    ).first()

    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    task_info = []
    upload_ids = []

    # Build a map of upload_id -> enable_ocr from the client request,
    # deduplicating upload_ids (preserves first occurrence's OCR setting).
    enable_ocr_map: Dict[int, Any] = {}
    for result in upload_results:
        if result.get("skip_processing"):
            continue
        uid = result["upload_id"]
        if uid not in enable_ocr_map:
            enable_ocr_map[uid] = result.get("enable_ocr")
            upload_ids.append(uid)

    if not upload_ids:
        return {"tasks": []}

    # Lock DocumentUpload rows to prevent concurrent process calls
    uploads = db.query(DocumentUpload).filter(
        DocumentUpload.id.in_(upload_ids),
        DocumentUpload.knowledge_base_id == kb_id
    ).with_for_update().all()
    uploads_dict = {upload.id: upload for upload in uploads}
    if len(uploads_dict) != len(set(upload_ids)):
        raise HTTPException(status_code=400, detail="One or more upload IDs are invalid")

    all_tasks = []
    for upload_id in upload_ids:
        upload = uploads_dict.get(upload_id)
        if not upload:
            continue

        # Skip if a non-failed task already exists for this upload
        existing_task = (
            db.query(ProcessingTask)
            .filter(
                ProcessingTask.document_upload_id == upload_id,
                ProcessingTask.status.in_(["pending", "processing"]),
            )
            .first()
        )
        if existing_task:
            task_info.append({
                "upload_id": upload_id,
                "task_id": existing_task.id,
            })
            continue

        task = ProcessingTask(
            document_upload_id=upload_id,
            knowledge_base_id=kb_id,
            status="pending"
        )
        all_tasks.append(task)

    db.add_all(all_tasks)
    db.commit()

    for task in all_tasks:
        db.refresh(task)

    task_data = []
    new_task_info = []
    for task in all_tasks:
        upload_id = task.document_upload_id
        upload = uploads_dict.get(upload_id)

        new_task_info.append({
            "upload_id": upload_id,
            "task_id": task.id
        })

        if upload:
            task_data.append({
                "task_id": task.id,
                "upload_id": upload_id,
                "temp_path": upload.temp_path,
                "file_name": upload.file_name,
                "enable_ocr": enable_ocr_map.get(upload_id),
            })

    task_info.extend(new_task_info)

    background_tasks.add_task(
        add_processing_tasks_to_queue,
        task_data,
        kb_id,
        current_user.id
    )

    return {"tasks": task_info}

async def add_processing_tasks_to_queue(task_data, kb_id, user_id):
    """Helper function to add document processing tasks to the queue without blocking the main response."""
    for data in task_data:
        asyncio.create_task(
            _process_and_graph(
                data["temp_path"],
                data["file_name"],
                kb_id,
                data["task_id"],
                None,
                user_id,
                enable_ocr=data.get("enable_ocr"),
            )
        )
    logger.debug(f"Added {len(task_data)} document processing tasks to queue")


async def _process_and_graph(
    temp_path: str,
    file_name: str,
    kb_id: int,
    task_id: int,
    db_session,
    user_id: int,
    enable_ocr: Optional[bool] = None,
    document_id: Optional[int] = None,
) -> None:
    """Run ingestion, then fire graph build as a background task if needed."""
    from app.services.ingestion.ingestion_dispatcher import _start_graph_build_thread
    graph_req = await process_document_background(
        temp_path,
        file_name,
        kb_id,
        task_id,
        db_session,
        user_id,
        enable_ocr=enable_ocr,
        document_id=document_id,
    )
    if graph_req is not None:
        _start_graph_build_thread(graph_req)

@router.post("/cleanup")
async def cleanup_temp_files(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Clean up expired temporary files for the current user's knowledge bases only.
    """
    expired_time = datetime.now(timezone.utc) - timedelta(hours=24)
    user_kb_ids = (
        db.query(KnowledgeBase.id)
        .filter(KnowledgeBase.user_id == current_user.id)
        .subquery()
    )
    expired_uploads = (
        db.query(DocumentUpload)
        .filter(
            DocumentUpload.created_at < expired_time,
            DocumentUpload.knowledge_base_id.in_(user_kb_ids),
        )
        .all()
    )

    for upload in expired_uploads:
        try:
            delete_file(upload.temp_path)
        except Exception as e:
            logger.error(f"Failed to delete temp file {upload.temp_path}: {str(e)}")

        db.delete(upload)

    db.commit()

    return {"message": f"Cleaned up {len(expired_uploads)} expired uploads"}

@router.get("/{kb_id}/documents/tasks")
async def get_processing_tasks(
    kb_id: int,
    task_ids: str = Query(..., description="Comma-separated list of task IDs to check status for"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get status of multiple processing tasks.
    """
    task_id_list = [int(id.strip()) for id in task_ids.split(",")]

    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == kb_id,
        _kb_owner_filter(current_user)
    ).first()

    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    tasks = (
        db.query(ProcessingTask)
        .options(
            selectinload(ProcessingTask.document_upload)
        )
        .filter(
            ProcessingTask.id.in_(task_id_list),
            ProcessingTask.knowledge_base_id == kb_id
        )
        .all()
    )

    from app.services.graph import get_graph_batch_progress

    result = {}
    for task in tasks:
        batch_progress = get_graph_batch_progress(task.id)
        if batch_progress:
            completed, total = batch_progress
            graph_progress = int(completed / total * 100) if total > 0 else 0
            graph_progress_message = f"Building knowledge graph ({completed}/{total} batches)"
        else:
            graph_progress = None
            graph_progress_message = None
        result[task.id] = {
            "document_id": task.document_id,
            "status": task.status,
            "error_message": task.error_message,
            "progress": task.progress or 0,
            "progress_message": task.progress_message or "",
            "upload_id": task.document_upload_id,
            "file_name": task.document_upload.file_name if task.document_upload else None,
            "graph_status": task.graph_status,
            "graph_error": task.graph_error,
            "graph_progress": graph_progress,
            "graph_progress_message": graph_progress_message,
        }
    return result
