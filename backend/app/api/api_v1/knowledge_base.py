import hashlib
import os
from typing import List, Any, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError as SAIntegrityError
from qdrant_client import QdrantClient
import logging
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel
from sqlalchemy.orm import selectinload
import time
import asyncio

from app.db.session import get_db
from app.models.user import User
from app.core.security import get_current_user, get_org_descendants
from app.models.knowledge import KnowledgeBase, Document, ProcessingTask, DocumentChunk, DocumentUpload, KnowledgeBaseDataStore
from app.models.datastore import DataStore, OrganizationDataStore
from app.models.organisation import Organisation


def _get_user_org_ids(db: Session, org_id: int) -> List[int]:
    """Get all org IDs in the user's hierarchy (user's org + all descendant orgs).

    Delegates to the shared get_org_descendants helper in app.core.security.
    """
    return get_org_descendants(db, org_id)

from app.schemas.knowledge import (
    KnowledgeBaseCreate,
    KnowledgeBaseResponse,
    KnowledgeBaseUpdate,
    DocumentResponse,
    PreviewRequest,
    DataStoreInfo
)
from app.services.ingestion import (
    process_document_background,
    upload_document,
    preview_document,
    PreviewResult,
)
from app.services.ingestion import SUPPORTED_EXTENSIONS
from app.core.config import settings
from app.core.storage import save_file, save_file_stream, delete_file


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


router = APIRouter()

logger = logging.getLogger(__name__)

@router.post("", response_model=KnowledgeBaseResponse)
def create_knowledge_base(
    *,
    db: Session = Depends(get_db),
    kb_in: KnowledgeBaseCreate,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Create new knowledge base.
    """
    kb = KnowledgeBase(
        name=kb_in.name,
        description=kb_in.description,
        user_id=current_user.id
    )
    kb.org_id = current_user.org_id
    db.add(kb)
    db.commit()
    db.refresh(kb)
    logger.info(f"Knowledge base created: {kb.name} for user {current_user.id}")
    return kb

@router.get("", response_model=List[KnowledgeBaseResponse])
def get_knowledge_bases(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    skip: int = 0,
    limit: int = 100
) -> Any:
    """
    Retrieve knowledge bases with linked data sources.
    """
    knowledge_bases = (
        db.query(KnowledgeBase)
        .filter(_kb_owner_filter(current_user))
        .options(selectinload(KnowledgeBase.data_sources))
        .offset(skip)
        .limit(limit)
        .all()
    )

    # Batch-fetch all linked datastores in one query instead of one per KB.
    all_ds_ids = set()
    for kb in knowledge_bases:
        all_ds_ids.update(link.data_store_id for link in kb.data_sources)

    all_datastores: dict[int, DataStore] = {}
    if all_ds_ids:
        all_datastores = {
            ds.id: ds for ds in db.query(DataStore).filter(
                DataStore.id.in_(all_ds_ids),
                DataStore.is_active == True,
            ).all()
        }

    # Batch-count documents per datastore (avoids N+1)
    from sqlalchemy import func as sa_func
    doc_counts: dict[int, int] = {}
    if all_ds_ids:
        count_rows = (
            db.query(Document.data_store_id, sa_func.count(Document.id))
            .filter(Document.data_store_id.in_(all_ds_ids))
            .group_by(Document.data_store_id)
            .all()
        )
        doc_counts = {row[0]: int(row[1]) for row in count_rows}

    # Add data_sources to each response (only explicitly linked ones)
    result = []
    for kb in knowledge_bases:
        linked_ds_ids = [link.data_store_id for link in kb.data_sources]
        linked_datastores = [all_datastores[did] for did in linked_ds_ids if did in all_datastores]
        data_sources = [
            DataStoreInfo(
                id=ds.id,
                name=ds.name,
                folder_path=ds.folder_path,
                auto_process_enabled=bool(ds.auto_process_enabled),
                document_count=doc_counts.get(ds.id, 0),
            )
            for ds in linked_datastores
        ]

        kb_dict = {
            "id": kb.id,
            "name": kb.name,
            "description": kb.description,
            "user_id": kb.user_id,
            "created_at": kb.created_at,
            "updated_at": kb.updated_at,
            "documents": kb.documents or [],
            "data_sources": data_sources,
            "data_source_count": len(linked_datastores),
        }
        result.append(KnowledgeBaseResponse(**kb_dict))

    return result


@router.get("/ocr-availability")
async def get_ocr_availability(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, bool]:
    """Check whether OCR is available (VISION_MODEL configured)."""
    from app.services.settings_service import get_setting
    vision_model = get_setting(db, "VISION_MODEL", None)
    return {"ocr_available": bool(vision_model)}


# ── Static-path GET routes (must be declared before /{kb_id}) ──────────────

@router.get("/available-datastores")
def list_available_datastores(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """List datastores assigned to the current user's org hierarchy.

    Returns datastores that the user can link to their knowledge bases.
    Org-level assignment makes a datastore visible for linking; it does NOT
    make it queryable — the user must explicitly link it to a KB first.
    """
    if not current_user.org_id:
        return []

    user_org_ids = _get_user_org_ids(db, current_user.org_id)

    datastores = (
        db.query(DataStore)
        .join(OrganizationDataStore)
        .filter(
            OrganizationDataStore.org_id.in_(user_org_ids),
            OrganizationDataStore.is_active == True,
            DataStore.is_active == True,
        )
        .distinct()
        .order_by(DataStore.id)
        .all()
    )

    # Batch-fetch all org links for these datastores in one query with
    # eager-loaded organisations (avoids N+1 per datastore + lazy loads).
    all_ds_ids = [ds.id for ds in datastores]
    all_links = (
        db.query(OrganizationDataStore)
        .options(selectinload(OrganizationDataStore.organisation))
        .filter(
            OrganizationDataStore.data_store_id.in_(all_ds_ids),
            OrganizationDataStore.is_active == True,
        )
        .all()
    ) if all_ds_ids else []

    links_by_ds: dict[int, list[OrganizationDataStore]] = {}
    for link in all_links:
        links_by_ds.setdefault(link.data_store_id, []).append(link)

    result = []
    for ds in datastores:
        links = links_by_ds.get(ds.id, [])
        result.append({
            "id": ds.id,
            "name": ds.name,
            "description": ds.description,
            "folder_path": ds.folder_path,
            "assigned_orgs": [
                {"id": link.organisation.id, "name": link.organisation.name}
                for link in links
            ],
        })
    return result


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


# ── Parameterised routes ────────────────────────────────────────────────────

@router.get("/{kb_id}", response_model=KnowledgeBaseResponse)
def get_knowledge_base(
    *,
    db: Session = Depends(get_db),
    kb_id: int,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Get knowledge base by ID with linked data sources.
    """
    from sqlalchemy.orm import joinedload
    
    kb = (
        db.query(KnowledgeBase)
        .options(
            joinedload(KnowledgeBase.documents)
            .joinedload(Document.processing_tasks)
        )
        .filter(
            KnowledgeBase.id == kb_id,
            _kb_owner_filter(current_user)
        )
        .first()
    )

    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    # Get only explicitly linked datastores
    linked_ds_ids = [link.data_store_id for link in kb.data_sources]
    linked_datastores = db.query(DataStore).filter(
        DataStore.id.in_(linked_ds_ids),
        DataStore.is_active == True
    ).all()

    # Batch-count documents per datastore (avoids N+1)
    from sqlalchemy import func as sa_func
    doc_counts: dict[int, int] = {}
    if linked_ds_ids:
        count_rows = (
            db.query(Document.data_store_id, sa_func.count(Document.id))
            .filter(Document.data_store_id.in_(linked_ds_ids))
            .group_by(Document.data_store_id)
            .all()
        )
        doc_counts = {row[0]: int(row[1]) for row in count_rows}

    data_sources = [
        DataStoreInfo(
            id=ds.id,
            name=ds.name,
            folder_path=ds.folder_path,
            auto_process_enabled=bool(ds.auto_process_enabled),
            document_count=doc_counts.get(ds.id, 0),
        )
        for ds in linked_datastores
    ]
    
    # Build response with data_sources
    kb_dict = {
        "id": kb.id,
        "name": kb.name,
        "description": kb.description,
        "user_id": kb.user_id,
        "created_at": kb.created_at,
        "updated_at": kb.updated_at,
        "documents": kb.documents or [],
        "data_sources": data_sources,
        "data_source_count": len(linked_datastores),
    }
    return KnowledgeBaseResponse(**kb_dict)

@router.put("/{kb_id}", response_model=KnowledgeBaseResponse)
def update_knowledge_base(
    *,
    db: Session = Depends(get_db),
    kb_id: int,
    kb_in: KnowledgeBaseUpdate,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Update knowledge base.
    """
    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == kb_id,
        _kb_owner_filter(current_user)
    ).first()
    
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    for field, value in kb_in.model_dump(exclude_unset=True).items():
        setattr(kb, field, value)

    db.add(kb)
    db.commit()
    db.refresh(kb)
    logger.info(f"Knowledge base updated: {kb.name} for user {current_user.id}")
    return kb

@router.delete("/{kb_id}")
async def delete_knowledge_base(
    *,
    db: Session = Depends(get_db),
    kb_id: int,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Delete knowledge base and all associated resources.
    
    Document handling:
    - Direct uploads (data_store_id=NULL): Deleted with KB (files, vectors, chunks, Neo4j)
    - DataStore docs (data_store_id!=NULL): KB link only is removed; documents persist
    """
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
    
    from app.services.cleanup import delete_kb as _delete_kb
    result, status = _delete_kb(db, kb_id, current_user.id)
    return JSONResponse(status_code=status, content=result)


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
    logger.info(f"Added {len(task_data)} document processing tasks to queue")


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
    from app.services.ingestion import _chunk_id_to_point_id
    from qdrant_client.models import PointIdsList

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
        #    DataStore documents have kb_id=NULL and data_store_id set, so we
        #    must use the correct scope filter to find their chunks.
        if document.data_store_id is not None:
            scope_filter = DocumentChunk.data_store_id == document.data_store_id
        else:
            scope_filter = (
                DocumentChunk.kb_id == kb_id,
                DocumentChunk.data_store_id.is_(None),
            )
        chunk_ids = [
            c.id for c in
            db.query(DocumentChunk.id)
            .filter(
                DocumentChunk.document_id == doc_id,
                *scope_filter if isinstance(scope_filter, tuple) else [scope_filter]
            )
            .all()
        ]

        # 2. Delete vectors from Qdrant
        if chunk_ids:
            try:
                qdrant = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
                if document.data_store_id is not None:
                    collection_name = f"ds_{document.data_store_id}"
                else:
                    collection_name = f"kb_{kb_id}"
                # Check if collection exists — it may not if ingestion failed
                # before the Qdrant upsert step, or if it was never created.
                existing = {c.name for c in qdrant.get_collections().collections}
                if collection_name not in existing:
                    logger.info(f"Qdrant collection {collection_name} does not exist — skipping point deletion for document {doc_id}")
                else:
                    point_ids = [_chunk_id_to_point_id(cid) for cid in chunk_ids]
                    qdrant.delete(
                        collection_name=collection_name,
                        points_selector=PointIdsList(points=point_ids),
                    )
                    logger.info(f"Deleted {len(point_ids)} Qdrant points for document {doc_id}")
            except Exception as e:
                cleanup_warnings.append(f"Qdrant cleanup warning: {str(e)}")
                logger.error(f"Failed to delete Qdrant points for document {doc_id}: {e}")

        # 3. Delete chunk rows from MySQL (explicit, don't rely on cascade here
        #    since we already fetched the IDs and want the delete to be transactional)
        db.query(DocumentChunk).filter(
            DocumentChunk.document_id == doc_id,
            *scope_filter if isinstance(scope_filter, tuple) else [scope_filter]
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


# ────────────────────────────────────────────────────────────────────────────
# Data Source Linking Endpoints
# ────────────────────────────────────────────────────────────────────────────

class LinkDataSourceRequest(BaseModel):
    data_store_id: int

@router.post("/{kb_id}/link-datastore")
def link_datastore_to_kb(
    *,
    db: Session = Depends(get_db),
    kb_id: int,
    request: LinkDataSourceRequest,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Link a datastore to a knowledge base.
    This creates an implicit relationship - documents from the datastore
    that match the KB's folder structure will be automatically ingested.
    """
    from app.models.datastore import DataStore, OrganizationDataStore
    
    # Verify KB ownership
    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == kb_id,
        _kb_owner_filter(current_user)
    ).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    # Verify datastore exists and is assigned to user's org or any descendant org
    user_org_ids = _get_user_org_ids(db, current_user.org_id)

    ds = db.query(DataStore).join(OrganizationDataStore).filter(
        DataStore.id == request.data_store_id,
        OrganizationDataStore.org_id.in_(user_org_ids),
        OrganizationDataStore.is_active == True,
        DataStore.is_active == True,
    ).first()
    if not ds:
        raise HTTPException(
            status_code=404,
            detail="Data source not found or not assigned to your organisation"
        )
    
    # Check if already linked
    existing = db.query(KnowledgeBaseDataStore).filter(
        KnowledgeBaseDataStore.knowledge_base_id == kb_id,
        KnowledgeBaseDataStore.data_store_id == request.data_store_id
    ).first()
    
    if existing:
        return {"message": f"Data source '{ds.name}' already linked to knowledge base '{kb.name}'"}
    
    # Create the junction record
    link = KnowledgeBaseDataStore(
        knowledge_base_id=kb_id,
        data_store_id=request.data_store_id
    )
    db.add(link)
    db.commit()
    
    logger.info("Data source '%s' linked to knowledge base '%s' (kb_id=%d)", ds.name, kb.name, kb_id)
    return {"message": f"Data source '{ds.name}' linked to knowledge base '{kb.name}'"}

@router.delete("/{kb_id}/unlink-datastore/{data_store_id}")
def unlink_datastore_from_kb(
    *,
    db: Session = Depends(get_db),
    kb_id: int,
    data_store_id: int,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Unlink a datastore from a knowledge base.
    """
    # Verify KB ownership
    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == kb_id,
        _kb_owner_filter(current_user)
    ).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    # Delete the junction record
    link = db.query(KnowledgeBaseDataStore).filter(
        KnowledgeBaseDataStore.knowledge_base_id == kb_id,
        KnowledgeBaseDataStore.data_store_id == data_store_id
    ).first()
    
    if not link:
        raise HTTPException(status_code=404, detail="Data source not linked to this knowledge base")
    
    db.delete(link)
    db.commit()
    
    ds = db.query(DataStore).filter(DataStore.id == data_store_id).first()
    logger.info("Data source '%s' unlinked from knowledge base '%s' (kb_id=%d)", ds.name if ds else data_store_id, kb.name, kb_id)
    return {"message": f"Data source unlinked from knowledge base '{kb.name}'"}
