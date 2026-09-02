"""Datastore linking endpoints for knowledge bases.

Endpoints:
    GET    /available-datastores               - list datastores linkable to KBs
    POST   /{kb_id}/link-datastore             - link a datastore to a KB
    DELETE /{kb_id}/unlink-datastore/{ds_id}   - unlink a datastore from a KB

Org-level assignment makes a datastore visible for linking; it does NOT
make it queryable — the user must explicitly link it to a KB first.
"""

import logging
from typing import Any

from fastapi import Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session, selectinload

from app.db.session import get_db
from app.models.user import User
from app.core.security import get_current_user
from app.models.knowledge import KnowledgeBase, KnowledgeBaseDataStore
from app.models.datastore import DataStore, OrganizationDataStore
from app.models.organisation import Organisation

from app.api.api_v1.knowledge_base import router
from app.api.api_v1.knowledge_base.helpers import _kb_owner_filter, _get_user_org_ids

logger = logging.getLogger(__name__)


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

    logger.debug("Data source '%s' linked to knowledge base '%s' (kb_id=%d)", ds.name, kb.name, kb_id)
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
    logger.debug("Data source '%s' unlinked from knowledge base '%s' (kb_id=%d)", ds.name if ds else data_store_id, kb.name, kb_id)
    return {"message": f"Data source unlinked from knowledge base '{kb.name}'"}
