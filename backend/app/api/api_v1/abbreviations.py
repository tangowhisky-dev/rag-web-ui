"""Admin API endpoints for abbreviation list management.

Super admins can upload universal lists (org_id=NULL) and org-specific lists.
Org admins can upload org-specific lists for their own org only.
Both can enable/disable, update, and delete lists within their scope.
"""
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.security import get_admin_org_ids, require_admin, require_super_admin
from app.db.session import get_db
from app.models.abbreviation import Abbreviation, AbbreviationList
from app.models.organisation import Organisation
from app.models.user import User
from app.services.abbreviation_service import _invalidate_cache, parse_csv_content

logger = logging.getLogger(__name__)

router = APIRouter()


# ─── Schemas ───────────────────────────────────────────────────────────────

class AbbreviationListOut(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    org_id: Optional[int] = None
    org_name: Optional[str] = None
    is_enabled: bool
    row_count: int
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}


class AbbreviationListUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    is_enabled: Optional[bool] = None


class AbbreviationOut(BaseModel):
    id: int
    list_id: int
    abbreviation: str
    expanded_form: str
    category: Optional[str] = None

    model_config = {"from_attributes": True}


class AbbreviationPaginated(BaseModel):
    items: List[AbbreviationOut]
    total: int
    page: int
    size: int


class UploadResponse(BaseModel):
    id: int
    name: str
    row_count: int
    message: str


# ─── Helpers ────────────────────────────────────────────────────────────────

def _serialize_list(lst: AbbreviationList, db: Session) -> AbbreviationListOut:
    org_name = None
    if lst.org_id is not None:
        org = db.query(Organisation).filter(Organisation.id == lst.org_id).first()
        org_name = org.name if org else None
    return AbbreviationListOut(
        id=lst.id,
        name=lst.name,
        description=lst.description,
        org_id=lst.org_id,
        org_name=org_name,
        is_enabled=lst.is_enabled,
        row_count=lst.row_count,
        created_at=lst.created_at.isoformat() if lst.created_at else "",
        updated_at=lst.updated_at.isoformat() if lst.updated_at else "",
    )


def _get_user_org_id(db: Session, current_user: User) -> Optional[int]:
    """Get the org_id for an admin user. Returns None for super_admin."""
    if current_user.role == "super_admin":
        return None
    admin_org_ids = get_admin_org_ids(db, current_user)
    if admin_org_ids is not None and len(admin_org_ids) > 0:
        return admin_org_ids[0]
    return current_user.org_id


def _check_list_access(lst: AbbreviationList, current_user: User, admin_org_ids: Optional[List[int]]) -> None:
    """Verify the user can access this list."""
    if current_user.role == "super_admin":
        return
    if lst.org_id is None:
        # Universal list — admin can view but not modify
        return
    if admin_org_ids is not None and lst.org_id not in admin_org_ids:
        raise HTTPException(status_code=403, detail="List is outside your organisation scope")


def _check_list_modify_access(lst: AbbreviationList, current_user: User, admin_org_ids: Optional[List[int]]) -> None:
    """Verify the user can modify/delete this list. Universal lists are super_admin only."""
    if current_user.role == "super_admin":
        return
    if lst.org_id is None:
        raise HTTPException(status_code=403, detail="Only super admins can modify universal lists")
    if admin_org_ids is not None and lst.org_id not in admin_org_ids:
        raise HTTPException(status_code=403, detail="List is outside your organisation scope")


# ─── Endpoints ──────────────────────────────────────────────────────────────

@router.get("/abbreviation-lists", response_model=List[AbbreviationListOut])
def list_abbreviation_lists(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """List all abbreviation lists visible to the caller."""
    admin_org_ids = get_admin_org_ids(db, current_user)
    query = db.query(AbbreviationList)
    if current_user.role != "super_admin" and admin_org_ids is not None:
        query = query.filter(
            (AbbreviationList.org_id.is_(None)) |
            (AbbreviationList.org_id.in_(admin_org_ids))
        )
    lists = query.order_by(AbbreviationList.created_at.desc()).all()
    return [_serialize_list(lst, db) for lst in lists]


@router.get("/abbreviation-lists/{list_id}", response_model=AbbreviationListOut)
def get_abbreviation_list(
    list_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Get a single abbreviation list."""
    admin_org_ids = get_admin_org_ids(db, current_user)
    lst = db.query(AbbreviationList).filter(AbbreviationList.id == list_id).first()
    if lst is None:
        raise HTTPException(status_code=404, detail="List not found")
    _check_list_access(lst, current_user, admin_org_ids)
    return _serialize_list(lst, db)


@router.post("/abbreviation-lists/upload", response_model=UploadResponse)
async def upload_abbreviation_list(
    file: UploadFile = File(...),
    name: str = Query(..., description="List name"),
    description: str = Query("", description="Optional description"),
    scope: str = Query("org", description="universal or org"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Upload a CSV file as a new abbreviation list.

    CSV format: abbreviation,expanded_form,category
    If a list with the same name and scope already exists, it is replaced.
    """
    admin_org_ids = get_admin_org_ids(db, current_user)

    # Determine org_id based on scope
    if scope == "universal":
        if current_user.role != "super_admin":
            raise HTTPException(status_code=403, detail="Only super admins can upload universal lists")
        org_id = None
    else:
        org_id = _get_user_org_id(db, current_user)
        if org_id is None:
            raise HTTPException(status_code=400, detail="No organisation associated with your account")
        if admin_org_ids is not None and org_id not in admin_org_ids:
            raise HTTPException(status_code=403, detail="Org is outside your scope")

    # Read and parse CSV
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    rows = parse_csv_content(content)
    if not rows:
        raise HTTPException(status_code=400, detail="No valid abbreviation rows found in CSV")

    # Check for existing list with same name+scope (upsert)
    existing = (
        db.query(AbbreviationList)
        .filter(AbbreviationList.name == name, AbbreviationList.org_id == org_id)
        .first()
    )
    if existing:
        # Replace: delete old abbreviations and update list
        db.query(Abbreviation).filter(Abbreviation.list_id == existing.id).delete()
        lst = existing
        lst.description = description or lst.description
        lst.row_count = len(rows)
        lst.is_enabled = True
    else:
        lst = AbbreviationList(
            name=name,
            description=description or None,
            org_id=org_id,
            is_enabled=True,
            row_count=len(rows),
            created_by=current_user.id,
        )
        db.add(lst)
        db.flush()

    # Insert abbreviation rows
    abbr_rows = [
        Abbreviation(
            list_id=lst.id,
            abbreviation=r["abbreviation"],
            expanded_form=r["expanded_form"],
            category=r["category"],
        )
        for r in rows
    ]
    db.add_all(abbr_rows)
    db.commit()
    db.refresh(lst)

    _invalidate_cache(org_id)
    logger.info("[ABBREV] list uploaded id=%s name=%s rows=%d scope=%s", lst.id, name, len(rows), scope)
    return UploadResponse(id=lst.id, name=lst.name, row_count=len(rows), message="Uploaded successfully")


@router.put("/abbreviation-lists/{list_id}", response_model=AbbreviationListOut)
def update_abbreviation_list(
    list_id: int,
    payload: AbbreviationListUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Update list metadata (name, description, is_enabled). No CSV re-upload."""
    admin_org_ids = get_admin_org_ids(db, current_user)
    lst = db.query(AbbreviationList).filter(AbbreviationList.id == list_id).first()
    if lst is None:
        raise HTTPException(status_code=404, detail="List not found")
    _check_list_modify_access(lst, current_user, admin_org_ids)

    if payload.name is not None:
        lst.name = payload.name
    if payload.description is not None:
        lst.description = payload.description
    if payload.is_enabled is not None:
        lst.is_enabled = payload.is_enabled
    db.commit()
    db.refresh(lst)
    _invalidate_cache(lst.org_id)
    return _serialize_list(lst, db)


@router.delete("/abbreviation-lists/{list_id}", status_code=204)
def delete_abbreviation_list(
    list_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Delete an abbreviation list and all its rows."""
    admin_org_ids = get_admin_org_ids(db, current_user)
    lst = db.query(AbbreviationList).filter(AbbreviationList.id == list_id).first()
    if lst is None:
        raise HTTPException(status_code=404, detail="List not found")
    _check_list_modify_access(lst, current_user, admin_org_ids)
    org_id = lst.org_id
    db.delete(lst)
    db.commit()
    _invalidate_cache(org_id)
    logger.info("[ABBREV] list deleted id=%s", list_id)


@router.get("/abbreviation-lists/{list_id}/abbreviations", response_model=AbbreviationPaginated)
def list_abbreviations_in_list(
    list_id: int,
    search: str = Query("", description="Filter by abbreviation or expanded_form"),
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Paginated abbreviation listing for a specific list."""
    admin_org_ids = get_admin_org_ids(db, current_user)
    lst = db.query(AbbreviationList).filter(AbbreviationList.id == list_id).first()
    if lst is None:
        raise HTTPException(status_code=404, detail="List not found")
    _check_list_access(lst, current_user, admin_org_ids)

    query = db.query(Abbreviation).filter(Abbreviation.list_id == list_id)
    if search:
        like = f"%{search}%"
        query = query.filter(
            (Abbreviation.abbreviation.ilike(like)) |
            (Abbreviation.expanded_form.ilike(like))
        )
    total = query.count()
    items = query.order_by(Abbreviation.abbreviation).offset((page - 1) * size).limit(size).all()
    return AbbreviationPaginated(items=items, total=total, page=page, size=size)
