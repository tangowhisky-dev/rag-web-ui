import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.security import require_admin
from app.db.session import get_db
from app.models.organisation import Organisation
from app.schemas.organisation import OrgCreate, OrgUpdate, OrgResponse
from app.schemas.user import UserAdminCreate, UserAdminUpdate, UserResponse
from app.models.user import User, UserRole
from app.core.security import get_password_hash

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


# ---------------------------------------------------------------------------
# User admin router
# ---------------------------------------------------------------------------

user_router = APIRouter()


@user_router.get("/users", response_model=List[UserResponse])
def list_users(
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    return db.query(User).order_by(User.id).all()


@user_router.post("/users", response_model=UserResponse, status_code=201)
def create_user(
    payload: UserAdminCreate,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    if payload.org_id is not None:
        org = db.query(Organisation).filter(Organisation.id == payload.org_id).first()
        if org is None:
            raise HTTPException(status_code=404, detail="Org not found")

    try:
        role_enum = UserRole(payload.role)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid role: {payload.role}")

    user = User(
        username=payload.username,
        email=payload.email,
        hashed_password=get_password_hash(payload.password),
        role=role_enum,
        org_id=payload.org_id,
        is_active=payload.is_active,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logging.info(f"[ADMIN] user_created id={user.id} role={user.role}")
    return user


@user_router.patch("/users/{user_id}", response_model=UserResponse)
def update_user(
    user_id: int,
    payload: UserAdminUpdate,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if payload.role is not None:
        try:
            user.role = UserRole(payload.role)
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Invalid role: {payload.role}")

    if payload.org_id is not None:
        org = db.query(Organisation).filter(Organisation.id == payload.org_id).first()
        if org is None:
            raise HTTPException(status_code=404, detail="Org not found")
        user.org_id = payload.org_id
    elif "org_id" in payload.model_fields_set:
        user.org_id = None

    if payload.is_active is not None:
        user.is_active = payload.is_active

    db.commit()
    db.refresh(user)
    return user


@user_router.delete("/users/{user_id}", status_code=204)
def deactivate_user(
    user_id: int,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    user.is_active = False
    db.commit()
    logging.info(f"[ADMIN] user_deactivated id={user_id}")
