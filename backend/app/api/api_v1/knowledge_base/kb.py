"""Knowledge-base CRUD endpoints (create, list, get, update, delete).

Also includes the OCR-availability check endpoint.  These are the
"static-path" KB routes — routes whose paths do not contain ``{kb_id}``
(except for the parameterised CRUD routes on ``/{kb_id}`` itself).
"""

import logging
from typing import List, Any, Dict

from fastapi import Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session, selectinload

from app.db.session import get_db
from app.models.user import User
from app.core.security import get_current_user
from app.models.knowledge import KnowledgeBase, Document
from app.models.datastore import DataStore
from app.schemas.knowledge import (
    KnowledgeBaseCreate,
    KnowledgeBaseResponse,
    KnowledgeBaseUpdate,
    DataStoreInfo,
)

from app.api.api_v1.knowledge_base import router
from app.api.api_v1.knowledge_base.helpers import _kb_owner_filter

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
    logger.debug(f"Knowledge base created: {kb.name} for user {current_user.id}")
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

    # Batch-count processed documents per datastore (documents with chunks
    # are available for retrieval; unprocessed docs don't count).
    from sqlalchemy import func as sa_func
    doc_counts: dict[int, int] = {}
    if all_ds_ids:
        count_rows = (
            db.query(Document.data_store_id, sa_func.count(Document.id))
            .filter(
                Document.data_store_id.in_(all_ds_ids),
                Document.chunks.any(),
            )
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

    # Batch-count processed documents per datastore (documents with chunks
    # are available for retrieval; unprocessed docs don't count).
    from sqlalchemy import func as sa_func
    doc_counts: dict[int, int] = {}
    if linked_ds_ids:
        count_rows = (
            db.query(Document.data_store_id, sa_func.count(Document.id))
            .filter(
                Document.data_store_id.in_(linked_ds_ids),
                Document.chunks.any(),
            )
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
    logger.debug(f"Knowledge base updated: {kb.name} for user {current_user.id}")
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
