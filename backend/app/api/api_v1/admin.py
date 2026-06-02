import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.security import require_admin
from app.db.session import get_db
from app.models.organisation import Organisation
from app.schemas.organisation import OrgCreate, OrgUpdate, OrgResponse

org_router = APIRouter()


@org_router.get("/orgs", response_model=List[OrgResponse])
def list_orgs(
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    return db.query(Organisation).order_by(Organisation.id).all()


@org_router.post("/orgs", response_model=OrgResponse, status_code=201)
def create_org(
    payload: OrgCreate,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    if payload.parent_id is not None:
        parent = db.query(Organisation).filter(Organisation.id == payload.parent_id).first()
        if parent is None:
            raise HTTPException(status_code=404, detail="Parent org not found")

    org = Organisation(name=payload.name, parent_id=payload.parent_id)
    db.add(org)
    db.flush()  # get org.id before computing path

    if payload.parent_id is not None:
        parent = db.query(Organisation).filter(Organisation.id == payload.parent_id).first()
        org.path = f"{parent.path}/{org.id}"
    else:
        org.path = f"/{org.id}"

    db.commit()
    db.refresh(org)
    logging.info(f"[ADMIN] org_created id={org.id} name={org.name}")
    return org


@org_router.patch("/orgs/{org_id}", response_model=OrgResponse)
def update_org(
    org_id: int,
    payload: OrgUpdate,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    org = db.query(Organisation).filter(Organisation.id == org_id).first()
    if org is None:
        raise HTTPException(status_code=404, detail="Org not found")

    if payload.name is not None:
        org.name = payload.name

    if payload.parent_id is not None:
        parent = db.query(Organisation).filter(Organisation.id == payload.parent_id).first()
        if parent is None:
            raise HTTPException(status_code=404, detail="Parent org not found")
        org.parent_id = payload.parent_id
        org.path = f"{parent.path}/{org.id}"
    elif payload.parent_id is None and "parent_id" in payload.model_fields_set:
        # Explicitly set to None → promote to root
        org.parent_id = None
        org.path = f"/{org.id}"

    db.commit()
    db.refresh(org)
    return org


@org_router.delete("/orgs/{org_id}", status_code=204)
def delete_org(
    org_id: int,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    org = db.query(Organisation).filter(Organisation.id == org_id).first()
    if org is None:
        raise HTTPException(status_code=404, detail="Org not found")

    children = db.query(Organisation).filter(Organisation.parent_id == org_id).count()
    if children:
        raise HTTPException(status_code=409, detail="Org has child organisations")

    assigned_users = org.users
    if assigned_users:
        raise HTTPException(status_code=409, detail="Org has assigned users")

    db.delete(org)
    db.commit()
    logging.info(f"[ADMIN] org_deleted id={org_id} name={org.name}")
