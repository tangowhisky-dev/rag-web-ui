import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

from app.core.security import require_admin
from app.db.session import get_db
from app.models.organisation import Organisation
from app.models.org_llm_config import OrgLLMConfig
from app.models.organisation import OrgAbbreviation
from app.models.datastore import DataStore
from app.schemas.organisation import OrgCreate, OrgUpdate, OrgResponse, OrgLLMConfigUpdate, OrgLLMConfigResponse, OrgIngestionStatusResponse
from app.models.knowledge import KnowledgeBase, ProcessingTask
from app.schemas.user import UserAdminCreate, UserAdminUpdate, UserResponse, UserDeleteResponse, PasswordChange
from app.models.user import User, UserRole
from app.core.security import get_password_hash
from app.services.agentic_rag.redis_memory import delete_user_redis_sync

org_router = APIRouter()


@org_router.get("/orgs", response_model=List[OrgResponse])
def list_orgs(
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    orgs = db.query(Organisation).order_by(Organisation.id).all()
    # Attach user_count and level to each org response
    user_counts = {
        o.id: db.query(User).filter(User.org_id == o.id).count()
        for o in orgs
    }
    result = []
    for org in orgs:
        resp = OrgResponse.model_validate(org)
        resp.user_count = user_counts.get(org.id, 0)
        # level = number of segments in path minus 1 (root = level 0)
        if org.path:
            resp.level = max(0, len([p for p in org.path.split("/") if p]) - 1)
        else:
            resp.level = 0
        result.append(resp)
    # Sort hierarchically: by path (depth-first), then alphabetically within same level
    result.sort(key=lambda o: (o.path or "", o.name))
    return result


@org_router.post("/orgs", response_model=OrgResponse, status_code=201)
def create_org(
    payload: OrgCreate,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    if db.query(Organisation).filter(Organisation.name == payload.name).first():
        raise HTTPException(status_code=400, detail="Org name already exists")

    # parent_id is now required — every org must have a parent
    parent = db.query(Organisation).filter(Organisation.id == payload.parent_id).first()
    if parent is None:
        raise HTTPException(status_code=404, detail="Parent org not found")

    org = Organisation(name=payload.name, parent_id=payload.parent_id)
    db.add(org)
    db.flush()  # get org.id before computing path
    # Guard against self-referencing root org
    if parent.id == org.id:
        org.path = f"/{org.id}"
    else:
        org.path = f"{parent.path}/{org.id}"

    db.commit()
    db.refresh(org)
    logger.info("[ADMIN] org_created id=%s name=%s", org.id, org.name)
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

    if payload.remove_parent:
        raise HTTPException(status_code=400, detail="Organisation must always have a parent")

    if payload.parent_id is not None:
        parent = db.query(Organisation).filter(Organisation.id == payload.parent_id).first()
        if parent is None:
            raise HTTPException(status_code=404, detail="Parent org not found")
        # Prevent setting a descendant as parent (would create a cycle)
        if org.path and parent.path and org.id != parent.id:
            # parent must not be a descendant of org
            # i.e., org's path must not be a prefix of parent's path
            if parent.path.startswith(org.path + "/"):
                raise HTTPException(
                    status_code=400,
                    detail="Cannot set a child or descendant organisation as parent",
                )
        org.parent_id = payload.parent_id
        # Guard against self-referencing root org
        if parent.id == org.id:
            org.path = f"/{org.id}"
        else:
            org.path = f"{parent.path}/{org.id}"

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
    logger.info("[ADMIN] org_deleted id=%s name=%s", org_id, org.name)


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
    logger.info("[ADMIN] org_llm_config_updated id=%s", org_id)
    return config


# ---------------------------------------------------------------------------
# Org ingestion status endpoint
# ---------------------------------------------------------------------------


@org_router.get("/orgs/{org_id}/ingestion-status", response_model=OrgIngestionStatusResponse)
def get_org_ingestion_status(
    org_id: int,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    org = db.query(Organisation).filter(Organisation.id == org_id).first()
    if org is None:
        raise HTTPException(status_code=404, detail="Org not found")

    # Aggregate ProcessingTask rows across all KBs belonging to this org
    tasks = (
        db.query(ProcessingTask)
        .join(KnowledgeBase, ProcessingTask.knowledge_base_id == KnowledgeBase.id)
        .filter(KnowledgeBase.org_id == org_id)
        .all()
    )

    pending_docs = sum(1 for t in tasks if t.status == "pending")
    processing_docs = sum(1 for t in tasks if t.status == "processing")
    completed_docs = sum(1 for t in tasks if t.status == "completed")
    failed_docs = sum(1 for t in tasks if t.status == "failed")
    total_docs = len(tasks)

    if processing_docs > 0:
        status = "running"
    elif failed_docs > 0:
        status = "failed"
    elif total_docs > 0 and completed_docs == total_docs:
        status = "completed"
    else:
        status = "idle"

    # last_run_at = max updated_at among completed or failed tasks
    terminal_tasks = [t for t in tasks if t.status in ("completed", "failed")]
    last_run_at = max((t.updated_at for t in terminal_tasks), default=None)

    logger.info(
        "[ADMIN] org_ingestion_status_fetched org_id=%s status=%s total_docs=%s",
        org_id, status, total_docs,
    )
    return OrgIngestionStatusResponse(
        org_id=org_id,
        status=status,
        total_docs=total_docs,
        pending_docs=pending_docs,
        processing_docs=processing_docs,
        completed_docs=completed_docs,
        failed_docs=failed_docs,
        last_run_at=last_run_at,
    )


# ---------------------------------------------------------------------------
# Org abbreviation endpoints
# ---------------------------------------------------------------------------

from pydantic import BaseModel as _BaseModel


class AdminCountsResponse(_BaseModel):
    organizations: int
    users: int
    data_sources: int


@org_router.get("/counts", response_model=AdminCountsResponse)
def get_admin_counts(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    # Organisations: all admins see all orgs (consistent with list_orgs endpoint)
    org_count = db.query(Organisation).count()

    # Users: super admins see all, regular admins see their org's users
    # (consistent with list_users endpoint visibility rules)
    if current_user.role == UserRole.super_admin:
        user_count = db.query(User).count()
    else:
        user_count = db.query(User).filter(User.org_id == current_user.org_id).count()

    # Data sources: all admins see all data stores
    ds_count = db.query(DataStore).count()

    return AdminCountsResponse(
        organizations=org_count, users=user_count, data_sources=ds_count,
    )


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
    logger.info("[ADMIN] abbreviation_created org_id=%s short=%s", org_id, payload.short)
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
    logger.info("[ADMIN] abbreviation_deleted id=%s org_id=%s", abbrev_id, org_id)


# ---------------------------------------------------------------------------
# User admin router
# ---------------------------------------------------------------------------

user_router = APIRouter()


@user_router.get("/users", response_model=List[UserResponse])
def list_users(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    if current_user.role == UserRole.super_admin:
        return db.query(User).order_by(User.id).all()
    return db.query(User).filter(User.org_id == current_user.org_id).order_by(User.id).all()


@user_router.post("/users", response_model=UserResponse, status_code=201)
def create_user(
    payload: UserAdminCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    # Only super_admin can create admin or super_admin users
    if current_user.role == UserRole.admin and payload.role not in ("user",):
        raise HTTPException(
            status_code=403,
            detail="Only super admin can create users with admin or super admin role",
        )

    if not payload.org_id:
        raise HTTPException(status_code=422, detail="User must belong to an organisation")
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
    logger.info("[ADMIN] user_created id=%s role=%s", user.id, user.role)
    return user


@user_router.patch("/users/{user_id}", response_model=UserResponse)
def update_user(
    user_id: int,
    payload: UserAdminUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if payload.role is not None:
        # Only super_admin can promote a user to admin or super_admin role
        if current_user.role == UserRole.admin and payload.role not in ("user",):
            raise HTTPException(
                status_code=403,
                detail="Only super admin can promote users to admin or super admin role",
            )
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
        raise HTTPException(status_code=422, detail="User must belong to an organisation")

    if payload.is_active is not None:
        user.is_active = payload.is_active

    db.commit()
    db.refresh(user)
    return user


@user_router.post("/users/{user_id}/change-password", status_code=200)
def change_user_password(
    user_id: int,
    payload: PasswordChange,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Change a user's password. Only super_admin can change other users' passwords."""
    if current_user.role != UserRole.super_admin:
        raise HTTPException(
            status_code=403,
            detail="Only super admin can change user passwords",
        )

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    user.hashed_password = get_password_hash(payload.new_password)
    db.commit()
    db.refresh(user)
    logger.info("[ADMIN] password_changed user_id=%s username=%s", user_id, user.username)
    return {"message": "Password changed successfully"}


@user_router.delete("/users/{user_id}", status_code=200, response_model=UserDeleteResponse)
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Permanently delete a user. The DB FK ON DELETE CASCADE removes all
    knowledge_bases, chats, folders (and their children) automatically.
    Only super_admins can permanently delete users."""
    if current_user.role != UserRole.super_admin:
        raise HTTPException(
            status_code=403,
            detail="Only super admin can permanently delete users",
        )

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    username = user.username
    chat_ids = [chat.id for chat in user.chats]
    db.delete(user)
    db.commit()
    # Clean up Redis checkpoints and long-term memory for this user and their chats
    delete_user_redis_sync(user_id, chat_ids)
    logger.info("[ADMIN] user_deleted id=%s username=%s", user_id, username)
    return UserDeleteResponse(id=user.id, username=username, email=user.email)
