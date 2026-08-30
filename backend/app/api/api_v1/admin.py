import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

from app.core.security import get_admin_org_ids, require_admin, require_super_admin
from app.db.session import get_db
from app.models.organisation import Organisation
from app.models.datastore import DataStore, OrganizationDataStore
from app.schemas.organisation import OrgCreate, OrgUpdate, OrgResponse, OrgIngestionStatusResponse
from app.models.knowledge import KnowledgeBase, ProcessingTask
from app.schemas.user import UserAdminCreate, UserAdminUpdate, UserResponse, UserDeleteResponse, PasswordChange, AdminPasswordChange
from app.models.user import User, UserRole
from app.core.security import get_password_hash
from app.services.agentic_rag.redis_memory import delete_user_redis_sync

org_router = APIRouter()


def _build_org_name_lookup(db: Session, orgs) -> dict:
    all_org_ids_in_paths = set()
    for o in orgs:
        if o.path:
            for part in o.path.split("/"):
                if part:
                    all_org_ids_in_paths.add(int(part))
    ancestor_ids = all_org_ids_in_paths - {o.id for o in orgs}
    ancestor_names = {
        a.id: a.name
        for a in db.query(Organisation).filter(Organisation.id.in_(ancestor_ids)).all()
    } if ancestor_ids else {}
    name_lookup = {o.id: o.name for o in orgs}
    name_lookup.update(ancestor_names)
    return name_lookup


def _org_response_with_hierarchy(org, user_count: int, name_lookup: dict) -> OrgResponse:
    resp = OrgResponse.model_validate(org)
    resp.user_count = user_count
    if org.path:
        parts = [p for p in org.path.split("/") if p]
        resp.level = max(0, len(parts) - 1)
        resp.hierarchy_name = " → ".join(
            name_lookup.get(int(p), f"#{p}") for p in parts
        )
    else:
        resp.level = 0
        resp.hierarchy_name = org.name
    return resp


@org_router.get("/orgs", response_model=List[OrgResponse])
def list_orgs(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    admin_org_ids = get_admin_org_ids(db, current_user)
    query = db.query(Organisation)
    if admin_org_ids is not None:
        query = query.filter(Organisation.id.in_(admin_org_ids))
    orgs = query.order_by(Organisation.id).all()
    user_counts = {
        o.id: db.query(User).filter(User.org_id == o.id).count()
        for o in orgs
    }
    name_lookup = _build_org_name_lookup(db, orgs)
    result = [
        _org_response_with_hierarchy(org, user_counts.get(org.id, 0), name_lookup)
        for org in orgs
    ]
    result.sort(key=lambda o: (o.path or "", o.name))
    return result


@org_router.post("/orgs", response_model=OrgResponse, status_code=201)
def create_org(
    payload: OrgCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    if db.query(Organisation).filter(Organisation.name == payload.name).first():
        raise HTTPException(status_code=400, detail="Org name already exists")

    # parent_id must be within the admin's scope
    admin_org_ids = get_admin_org_ids(db, current_user)
    if admin_org_ids is not None and payload.parent_id not in admin_org_ids:
        raise HTTPException(status_code=403, detail="Parent org is outside your organisation scope")

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


def _validate_admin_edit_restriction(db: Session, current_user: User, org: Organisation):
    if current_user.role != UserRole.admin or current_user.org_id is None:
        return
    if org.id == current_user.org_id:
        raise HTTPException(
            status_code=403,
            detail="You cannot edit your own organisation. Only super admin can do that.",
        )
    # Check if the org is an ancestor of the admin's org
    if org.path and current_user.org_id:
        admin_org = db.query(Organisation).filter(Organisation.id == current_user.org_id).first()
        if admin_org and admin_org.path and admin_org.path.startswith(org.path + "/"):
            raise HTTPException(
                status_code=403,
                detail="You cannot edit a parent organisation. Only super admin can do that.",
            )


def _update_org_name(db: Session, org: Organisation, org_id: int, name: str):
    if name != org.name and db.query(Organisation).filter(
        Organisation.name == name, Organisation.id != org_id
    ).first():
        raise HTTPException(status_code=400, detail="Org name already exists")
    org.name = name


def _update_org_parent(db: Session, org: Organisation, parent_id: int, admin_org_ids):
    if admin_org_ids is not None and parent_id not in admin_org_ids:
        raise HTTPException(status_code=403, detail="Parent org is outside your organisation scope")
    parent = db.query(Organisation).filter(Organisation.id == parent_id).first()
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
    org.parent_id = parent_id
    # Guard against self-referencing root org
    if parent.id == org.id:
        org.path = f"/{org.id}"
    else:
        org.path = f"{parent.path}/{org.id}"


@org_router.patch("/orgs/{org_id}", response_model=OrgResponse)
def update_org(
    org_id: int,
    payload: OrgUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    org = db.query(Organisation).filter(Organisation.id == org_id).first()
    if org is None:
        raise HTTPException(status_code=404, detail="Org not found")

    admin_org_ids = get_admin_org_ids(db, current_user)
    if admin_org_ids is not None and org.id not in admin_org_ids:
        raise HTTPException(status_code=403, detail="Org is outside your organisation scope")

    # Org admins cannot edit their own org or ancestors — only descendants
    _validate_admin_edit_restriction(db, current_user, org)

    if payload.name is not None:
        _update_org_name(db, org, org_id, payload.name)

    if payload.remove_parent:
        raise HTTPException(status_code=400, detail="Organisation must always have a parent")

    if payload.parent_id is not None:
        _update_org_parent(db, org, payload.parent_id, admin_org_ids)

    db.commit()
    db.refresh(org)
    return org


@org_router.delete("/orgs/{org_id}", status_code=204)
def delete_org(
    org_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    org = db.query(Organisation).filter(Organisation.id == org_id).first()
    if org is None:
        raise HTTPException(status_code=404, detail="Org not found")

    admin_org_ids = get_admin_org_ids(db, current_user)
    if admin_org_ids is not None and org.id not in admin_org_ids:
        raise HTTPException(status_code=403, detail="Org is outside your organisation scope")

    # Org admins cannot delete their own org or ancestors — only descendants
    if current_user.role == UserRole.admin and current_user.org_id is not None:
        if org.id == current_user.org_id:
            raise HTTPException(
                status_code=403,
                detail="You cannot delete your own organisation. Only super admin can do that.",
            )
        if org.path and current_user.org_id:
            admin_org = db.query(Organisation).filter(Organisation.id == current_user.org_id).first()
            if admin_org and admin_org.path and admin_org.path.startswith(org.path + "/"):
                raise HTTPException(
                    status_code=403,
                    detail="You cannot delete a parent organisation. Only super admin can do that.",
                )

    children = db.query(Organisation).filter(Organisation.parent_id == org_id).count()
    if children:
        raise HTTPException(status_code=409, detail="Org has child organisations")

    assigned_users = org.users
    if assigned_users:
        raise HTTPException(status_code=409, detail="Org has assigned users")

    db.delete(org)
    db.commit()
    logger.info("[ADMIN] org_deleted id=%s name=%s", org_id, org.name)


# ---------------------------------------------------------------------------
# Org ingestion status endpoint
# ---------------------------------------------------------------------------


def _count_tasks_by_status(tasks) -> tuple:
    counts = {"pending": 0, "processing": 0, "completed": 0, "failed": 0}
    for t in tasks:
        if t.status in counts:
            counts[t.status] += 1
    return counts["pending"], counts["processing"], counts["completed"], counts["failed"]


def _ingestion_status_from_counts(processing_docs: int, failed_docs: int, completed_docs: int, total_docs: int) -> str:
    if processing_docs > 0:
        return "running"
    if failed_docs > 0:
        return "failed"
    if total_docs > 0 and completed_docs == total_docs:
        return "completed"
    return "idle"


@org_router.get("/orgs/{org_id}/ingestion-status", response_model=OrgIngestionStatusResponse)
def get_org_ingestion_status(
    org_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    org = db.query(Organisation).filter(Organisation.id == org_id).first()
    if org is None:
        raise HTTPException(status_code=404, detail="Org not found")

    admin_org_ids = get_admin_org_ids(db, current_user)
    if admin_org_ids is not None and org_id not in admin_org_ids:
        raise HTTPException(status_code=403, detail="Org is outside your organisation scope")

    tasks = (
        db.query(ProcessingTask)
        .join(KnowledgeBase, ProcessingTask.knowledge_base_id == KnowledgeBase.id)
        .filter(KnowledgeBase.org_id == org_id)
        .all()
    )

    pending_docs, processing_docs, completed_docs, failed_docs = _count_tasks_by_status(tasks)
    total_docs = len(tasks)
    status = _ingestion_status_from_counts(processing_docs, failed_docs, completed_docs, total_docs)

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
# Org admin endpoints
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
    admin_org_ids = get_admin_org_ids(db, current_user)

    if admin_org_ids is None:
        org_count = db.query(Organisation).count()
        user_count = db.query(User).count()
        ds_count = db.query(DataStore).count()
    else:
        org_count = db.query(Organisation).filter(Organisation.id.in_(admin_org_ids)).count()
        user_count = db.query(User).filter(User.org_id.in_(admin_org_ids)).count()
        ds_count = (
            db.query(DataStore)
            .join(OrganizationDataStore)
            .filter(OrganizationDataStore.org_id.in_(admin_org_ids))
            .distinct()
            .count()
        )

    return AdminCountsResponse(
        organizations=org_count, users=user_count, data_sources=ds_count,
    )


# ---------------------------------------------------------------------------
# User admin router
# ---------------------------------------------------------------------------

user_router = APIRouter()


@user_router.get("/users", response_model=List[UserResponse])
def list_users(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    admin_org_ids = get_admin_org_ids(db, current_user)
    if admin_org_ids is None:
        return db.query(User).order_by(User.id).all()
    return db.query(User).filter(User.org_id.in_(admin_org_ids)).order_by(User.id).all()


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

    admin_org_ids = get_admin_org_ids(db, current_user)
    if not payload.org_id:
        raise HTTPException(status_code=422, detail="User must belong to an organisation")
    if admin_org_ids is not None and payload.org_id not in admin_org_ids:
        raise HTTPException(status_code=403, detail="Cannot create users outside your organisation scope")
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


def _apply_user_role_update(current_user: User, payload: UserAdminUpdate, user: User):
    if current_user.role == UserRole.admin and payload.role not in ("user",):
        raise HTTPException(
            status_code=403,
            detail="Only super admin can promote users to admin or super admin role",
        )
    try:
        user.role = UserRole(payload.role)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid role: {payload.role}")


def _apply_user_org_update(db: Session, payload: UserAdminUpdate, admin_org_ids, user: User):
    if admin_org_ids is not None and payload.org_id not in admin_org_ids:
        raise HTTPException(status_code=403, detail="Target org is outside your organisation scope")
    org = db.query(Organisation).filter(Organisation.id == payload.org_id).first()
    if org is None:
        raise HTTPException(status_code=404, detail="Org not found")
    user.org_id = payload.org_id


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

    admin_org_ids = get_admin_org_ids(db, current_user)
    if admin_org_ids is not None and user.org_id not in admin_org_ids:
        raise HTTPException(status_code=403, detail="User is outside your organisation scope")

    if current_user.role == UserRole.admin and user.role != UserRole.user:
        raise HTTPException(
            status_code=403,
            detail="Org admins can only manage normal users",
        )

    if payload.role is not None:
        _apply_user_role_update(current_user, payload, user)

    if payload.org_id is not None:
        _apply_user_org_update(db, payload, admin_org_ids, user)
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
    payload: AdminPasswordChange,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Change a user's password.

    super_admin: can change any user's password.
    org admin: can change passwords of normal users in their org scope.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if current_user.role != UserRole.super_admin:
        # Org admins can only change passwords of normal users in their scope
        if user.role != UserRole.user:
            raise HTTPException(
                status_code=403,
                detail="Org admins can only manage normal users",
            )
        admin_org_ids = get_admin_org_ids(db, current_user)
        if admin_org_ids is not None and user.org_id not in admin_org_ids:
            raise HTTPException(
                status_code=403,
                detail="User is outside your organisation scope",
            )

    user.hashed_password = get_password_hash(payload.new_password)
    user.token_version += 1
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
