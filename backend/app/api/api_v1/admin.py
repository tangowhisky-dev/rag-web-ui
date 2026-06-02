import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.security import require_admin
from app.db.session import get_db
from app.models.organisation import Organisation
from app.models.org_llm_config import OrgLLMConfig
from app.models.organisation import OrgAbbreviation
from app.schemas.organisation import OrgCreate, OrgUpdate, OrgResponse, OrgLLMConfigUpdate, OrgLLMConfigResponse
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
    if db.query(Organisation).filter(Organisation.name == payload.name).first():
        raise HTTPException(status_code=400, detail="Org name already exists")

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
        if payload.name != org.name and db.query(Organisation).filter(
            Organisation.name == payload.name, Organisation.id != org_id
        ).first():
            raise HTTPException(status_code=400, detail="Org name already exists")
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


@org_router.get("/orgs/{org_id}/llm-config", response_model=OrgLLMConfigResponse)
def get_org_llm_config(
    org_id: int,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    config = db.query(OrgLLMConfig).filter(OrgLLMConfig.org_id == org_id).first()
    if config is None:
        raise HTTPException(status_code=404, detail="LLM config not found for this org")
    return config


@org_router.put("/orgs/{org_id}/llm-config", response_model=OrgLLMConfigResponse)
def upsert_org_llm_config(
    org_id: int,
    payload: OrgLLMConfigUpdate,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    org = db.query(Organisation).filter(Organisation.id == org_id).first()
    if org is None:
        raise HTTPException(status_code=404, detail="Org not found")

    config = db.query(OrgLLMConfig).filter(OrgLLMConfig.org_id == org_id).first()
    if config is None:
        config = OrgLLMConfig(org_id=org_id)
        db.add(config)

    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(config, field, value)

    db.commit()
    db.refresh(config)
    logging.info("[ADMIN] org_llm_config_updated id=%s", org_id)
    return config


# ---------------------------------------------------------------------------
# Org abbreviation endpoints
# ---------------------------------------------------------------------------

from pydantic import BaseModel as _BaseModel


class AbbreviationCreate(_BaseModel):
    short: str
    expansion: str


class AbbreviationOut(_BaseModel):
    id: int
    org_id: int
    short: str
    expansion: str

    model_config = {"from_attributes": True}


@org_router.post("/orgs/{org_id}/abbreviations", response_model=AbbreviationOut, status_code=201)
def create_abbreviation(
    org_id: int,
    payload: AbbreviationCreate,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    org = db.query(Organisation).filter(Organisation.id == org_id).first()
    if org is None:
        raise HTTPException(status_code=404, detail="Org not found")
    existing = (
        db.query(OrgAbbreviation)
        .filter(OrgAbbreviation.org_id == org_id, OrgAbbreviation.short == payload.short)
        .first()
    )
    if existing:
        raise HTTPException(status_code=409, detail="Abbreviation already exists for this org")
    abbrev = OrgAbbreviation(org_id=org_id, short=payload.short, expansion=payload.expansion)
    db.add(abbrev)
    db.commit()
    db.refresh(abbrev)
    logging.info("[ADMIN] abbreviation_created org_id=%s short=%s", org_id, payload.short)
    return abbrev


@org_router.get("/orgs/{org_id}/abbreviations", response_model=List[AbbreviationOut])
def list_abbreviations(
    org_id: int,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    org = db.query(Organisation).filter(Organisation.id == org_id).first()
    if org is None:
        raise HTTPException(status_code=404, detail="Org not found")
    return db.query(OrgAbbreviation).filter(OrgAbbreviation.org_id == org_id).all()


@org_router.delete("/orgs/{org_id}/abbreviations/{abbrev_id}", status_code=204)
def delete_abbreviation(
    org_id: int,
    abbrev_id: int,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    abbrev = (
        db.query(OrgAbbreviation)
        .filter(OrgAbbreviation.id == abbrev_id, OrgAbbreviation.org_id == org_id)
        .first()
    )
    if abbrev is None:
        raise HTTPException(status_code=404, detail="Abbreviation not found")
    db.delete(abbrev)
    db.commit()
    logging.info("[ADMIN] abbreviation_deleted id=%s org_id=%s", abbrev_id, org_id)


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

    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    if db.query(User).filter(User.username == payload.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")

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
