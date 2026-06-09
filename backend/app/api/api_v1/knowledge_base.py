import hashlib
import os
from typing import List, Any, Dict
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from qdrant_client import QdrantClient
from sqlalchemy import text
import logging
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel
from sqlalchemy.orm import selectinload
import time
import asyncio

from app.db.session import get_db
from app.models.user import User
from app.core.security import get_current_user
from app.models.knowledge import KnowledgeBase, Document, ProcessingTask, DocumentChunk, DocumentUpload, KnowledgeBaseDataStore
from app.models.datastore import DataStore, OrganizationDataStore
from app.models.organisation import Organisation


def _get_user_org_ids(db: Session, org_id: int) -> List[int]:
    """Get all org IDs in the user's hierarchy (user's org + all descendant orgs).
    
    Users can access data from their org and any child orgs.
    """
    org_ids = [org_id]
    orgs_to_check = [org_id]
    
    for _ in range(100):  # Safety limit to prevent infinite loops
        if not orgs_to_check:
            break
        parent_id = orgs_to_check.pop(0)
        children = db.query(Organisation).filter(
            Organisation.parent_id == parent_id,
            Organisation.id != parent_id  # Exclude self-referencing root
        ).all()
        for child in children:
            if child.id not in org_ids:
                org_ids.append(child.id)
                orgs_to_check.append(child.id)
    
    return org_ids
from app.schemas.knowledge import (
    KnowledgeBaseCreate,
    KnowledgeBaseResponse,
    KnowledgeBaseUpdate,
    DocumentResponse,
    PreviewRequest,
    DataStoreInfo
)
from app.services.document_processor import process_document_background, upload_document, preview_document, PreviewResult, SUPPORTED_EXTENSIONS
from app.core.config import settings
from app.core.storage import save_file, delete_file


def _kb_owner_filter(current_user):
    """Return SQLAlchemy filter clause scoping KB access.
    Users can only access KBs they personally own (user_id).
    """
    return KnowledgeBase.user_id == current_user.id


router = APIRouter()

logger = logging.getLogger(__name__)

class TestRetrievalRequest(BaseModel):
    query: str
    kb_id: int
    top_k: int

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
    # Get datastores assigned to user's org and all descendant orgs
    org_datastores = []
    if current_user.org_id:
        user_org_ids = _get_user_org_ids(db, current_user.org_id)
        org_datastores = (
            db.query(DataStore)
            .join(OrganizationDataStore)
            .filter(
                OrganizationDataStore.org_id.in_(user_org_ids),
                OrganizationDataStore.is_active == True,
                DataStore.is_active == True,
            )
            .all()
        )
    
    knowledge_bases = (
        db.query(KnowledgeBase)
        .filter(_kb_owner_filter(current_user))
        .options(selectinload(KnowledgeBase.data_sources))
        .offset(skip)
        .limit(limit)
        .all()
    )
    
    # Add data_sources to each response (only explicitly linked ones)
    result = []
    for kb in knowledge_bases:
        linked_ds_ids = [link.data_store_id for link in kb.data_sources]
        linked_datastores = db.query(DataStore).filter(
            DataStore.id.in_(linked_ds_ids),
            DataStore.is_active == True
        ).all()
        data_sources = [DataStoreInfo(id=ds.id, name=ds.name, folder_path=ds.folder_path) for ds in linked_datastores]
        
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
    
    # Get datastores assigned to user's org and all descendant orgs
    org_datastores = []
    if current_user.org_id:
        user_org_ids = _get_user_org_ids(db, current_user.org_id)
        org_datastores = (
            db.query(DataStore)
            .join(OrganizationDataStore)
            .filter(
                OrganizationDataStore.org_id.in_(user_org_ids),
                OrganizationDataStore.is_active == True,
                DataStore.is_active == True,
            )
            .all()
        )
    
    # Get only explicitly linked datastores
    linked_ds_ids = [link.data_store_id for link in kb.data_sources]
    linked_datastores = db.query(DataStore).filter(
        DataStore.id.in_(linked_ds_ids),
        DataStore.is_active == True
    ).all()
    data_sources = [DataStoreInfo(id=ds.id, name=ds.name, folder_path=ds.folder_path) for ds in linked_datastores]
    
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

    for field, value in kb_in.dict(exclude_unset=True).items():
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
    
    from app.services.deletion_service import delete_kb as _delete_kb
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

        # 1. 计算文件 hash
        file_content = await file.read()
        file_hash = hashlib.sha256(file_content).hexdigest()
        
        # 2. 检查是否存在完全相同的文件（名称和hash都相同）
        existing_document = db.query(Document).filter(
            Document.file_name == file.filename,
            Document.file_hash == file_hash,
            Document.knowledge_base_id == kb_id
        ).first()
        
        if existing_document:
            # 完全相同的文件，直接返回
            results.append({
                "document_id": existing_document.id,
                "file_name": existing_document.file_name,
                "status": "exists",
                "message": "文件已存在且已处理完成",
                "skip_processing": True
            })
            continue
        
        # 3. Save to temp directory
        temp_path = f"user_{current_user.id}/kb_{kb_id}/temp/{file.filename}"
        try:
            file_size = len(file_content)
            save_file(temp_path, file_content)
        except Exception as e:
            logger.error(f"Failed to save file to storage: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to upload file")
        
        # 4. 创建上传记录
        upload = DocumentUpload(
            knowledge_base_id=kb_id,
            file_name=file.filename,
            file_hash=file_hash,
            file_size=len(file_content),
            content_type=file.content_type,
            temp_path=temp_path
        )
        db.add(upload)
        db.commit()
        db.refresh(upload)
        
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
    start_time = time.time()
    
    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == kb_id,
        _kb_owner_filter(current_user)
    ).first()
    
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    task_info = []
    upload_ids = []
    
    for result in upload_results:
        if result.get("skip_processing"):
            continue
        upload_ids.append(result["upload_id"])
    
    if not upload_ids:
        return {"tasks": []}
    
    uploads = db.query(DocumentUpload).filter(
        DocumentUpload.id.in_(upload_ids),
        DocumentUpload.knowledge_base_id == kb_id
    ).all()
    uploads_dict = {upload.id: upload for upload in uploads}
    if len(uploads_dict) != len(set(upload_ids)):
        raise HTTPException(status_code=400, detail="One or more upload IDs are invalid")
    
    all_tasks = []
    for upload_id in upload_ids:
        upload = uploads_dict.get(upload_id)
        if not upload:
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
    for i, upload_id in enumerate(upload_ids):
        if i < len(all_tasks):
            task = all_tasks[i]
            upload = uploads_dict.get(upload_id)
            
            task_info.append({
                "upload_id": upload_id,
                "task_id": task.id
            })
            
            if upload:
                task_data.append({
                    "task_id": task.id,
                    "upload_id": upload_id,
                    "temp_path": upload.temp_path,
                    "file_name": upload.file_name,
                    "enable_ocr": result.get("enable_ocr"),  # None|True|False
                })
    
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
            process_document_background(
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

@router.post("/cleanup")
async def cleanup_temp_files(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Clean up expired temporary files.
    """
    expired_time = datetime.now(timezone.utc) - timedelta(hours=24)
    expired_uploads = db.query(DocumentUpload).filter(
        DocumentUpload.created_at < expired_time
    ).all()
    
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
    
    return {
        task.id: {
            "document_id": task.document_id,
            "status": task.status,
            "error_message": task.error_message,
            "progress": task.progress or 0,
            "progress_message": task.progress_message or "",
            "upload_id": task.document_upload_id,
            "file_name": task.document_upload.file_name if task.document_upload else None
        }
        for task in tasks
    }

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
    from app.services.document_processor import _chunk_id_to_point_id
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
        # 1. Collect chunk IDs before deleting them (needed for Qdrant point IDs)
        chunk_ids = [
            c.id for c in
            db.query(DocumentChunk.id)
            .filter(
                DocumentChunk.document_id == doc_id,
                DocumentChunk.kb_id == kb_id
            )
            .all()
        ]

        # 2. Delete vectors from Qdrant
        if chunk_ids:
            try:
                qdrant = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
                point_ids = [_chunk_id_to_point_id(cid) for cid in chunk_ids]
                qdrant.delete(
                    collection_name=f"kb_{kb_id}",
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
            DocumentChunk.kb_id == kb_id
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
            from app.services.graph_service import delete_graph_for_document
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: delete_graph_for_document(kb_id=kb_id, document_id=doc_id)
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

@router.post("/test-retrieval")
async def test_retrieval(
    request: TestRetrievalRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Test retrieval quality for a given query against a knowledge base.
    """
    try:
        kb = db.query(KnowledgeBase).filter(
            KnowledgeBase.id == request.kb_id,
            _kb_owner_filter(current_user)
        ).first()
        
        if not kb:
            raise HTTPException(
                status_code=404,
                detail=f"Knowledge base {request.kb_id} not found",
            )
        
        from app.services.retrieval import hybrid_search
        docs = await hybrid_search(
            query=request.query,
            kb_ids=[request.kb_id],
            db=db,
        )
        response = [
            {"content": doc.page_content, "metadata": doc.metadata, "score": 0.0}
            for doc in docs[: request.top_k]
        ]
        return {"results": response}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    # Get all org IDs in the user's hierarchy (user's org + all descendants)
    user_org_ids = [current_user.org_id]
    
    # Walk down the hierarchy to find all child orgs
    orgs_to_check = [current_user.org_id]
    for _ in range(100):  # Safety limit to prevent infinite loops
        if not orgs_to_check:
            break
        parent_id = orgs_to_check.pop(0)
        children = db.query(Organisation).filter(
            Organisation.parent_id == parent_id,
            Organisation.id != parent_id  # Exclude self-referencing root
        ).all()
        for child in children:
            if child.id not in user_org_ids:
                user_org_ids.append(child.id)
                orgs_to_check.append(child.id)
    
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
