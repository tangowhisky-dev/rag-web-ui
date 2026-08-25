"""Admin API endpoints for DataStore CRUD management.

Endpoints:
    GET    /api/admin/datastores              — list all datastores
    POST   /api/admin/datastores              — create a datastore
    GET    /api/admin/datastores/{id}         — get datastore details
    PATCH  /api/admin/datastores/{id}         — update a datastore
    DELETE /api/admin/datastores/{id}         — delete a datastore
    POST   /api/admin/datastores/{id}/assign  — assign datastore to orgs
    DELETE /api/admin/datastores/{id}/assign  — unassign datastore from orgs
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict
from sqlalchemy.orm import Session

from app.core.security import get_admin_org_ids, require_admin, require_super_admin
from app.db.session import get_db
from app.models.datastore import DataStore, OrganizationDataStore
from app.models.organisation import Organisation
from app.models.user import User, UserRole
from app.services.datastore_watcher import DataStoreWatcher

logger = logging.getLogger(__name__)

router = APIRouter()


def _validate_folder_path(folder_path: str) -> str:
    """Validate that a folder path exists and is located under /app/data."""
    abs_path = os.path.abspath(folder_path)
    data_root = "/app/data"
    real_path = os.path.realpath(abs_path)
    real_root = os.path.realpath(data_root)

    if not os.path.isdir(real_path):
        if os.path.isdir(real_root):
            raise HTTPException(
                status_code=400,
                detail=f"Directory does not exist: {abs_path}. Valid data sources must be subdirectories of {data_root}.",
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Directory does not exist: {abs_path}. Ensure the /app/data volume is mounted and the folder exists.",
            )

    if real_path != real_root and not real_path.startswith(real_root + os.sep):
        raise HTTPException(
            status_code=400,
            detail=f"Directory must be under {data_root}: {abs_path}",
        )

    return abs_path


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class DataStoreCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    folder_path: str = Field(..., min_length=1, max_length=768)
    scan_pattern: str = Field(default="*")
    auto_scan_enabled: bool = False
    auto_scan_interval_minutes: int = Field(default=60, ge=1, le=1440)


class DataStoreUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    description: Optional[str] = None
    folder_path: Optional[str] = Field(default=None, min_length=1, max_length=768)
    scan_pattern: Optional[str] = None
    is_active: Optional[bool] = None
    auto_scan_enabled: Optional[bool] = None
    auto_scan_interval_minutes: Optional[int] = Field(default=None, ge=1, le=1440)


class DataStoreResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    folder_path: str
    scan_pattern: str
    is_active: bool
    auto_scan_enabled: bool
    auto_scan_interval_minutes: int
    last_scan_at: Optional[str] = None
    last_scan_status: str
    last_scan_error: Optional[str] = None
    last_scan_total_files: int
    last_scan_processed: int
    last_scan_new: int = 0
    last_scan_modified: int = 0
    last_scan_skipped: int = 0
    last_scan_errors: int = 0
    assigned_orgs: List[dict] = []
    created_at: datetime
    updated_at: datetime
    # Real-time scan progress (populated when a manual scan is running)
    scan_progress: Optional[dict] = None
    # Pending changes detected but not yet processed (event-driven queue)
    pending_changes: int = 0
    # Timestamp of the last successful recovery scan
    last_recovered_at: Optional[str] = None
    # Whether changes are currently being processed (event-driven ingestion)
    processing: bool = False
    # Aggregated graph build status across all documents in this datastore
    graph_summary: Optional[dict] = None

    model_config = ConfigDict(from_attributes=True)


class AssignRequest(BaseModel):
    org_ids: List[int]
    # When true, an empty org_ids list removes all in-scope assignments.
    # Without this flag, empty org_ids is rejected to prevent accidental
    # mass-unassignment from a misclicked empty payload.
    force_clear: bool = False


class DataStoreStatusResponse(BaseModel):
    datastore_id: int
    name: str
    folder_path: str
    last_scan_at: Optional[str] = None
    last_scan_status: str
    last_scan_error: Optional[str] = None
    last_scan_total_files: int
    last_scan_processed: int
    last_scan_new: int = 0
    last_scan_modified: int = 0
    last_scan_skipped: int = 0
    last_scan_errors: int = 0
    # Pending changes detected but not yet processed (event-driven queue)
    pending_changes: int = 0
    # Whether changes are currently being processed (event-driven ingestion)
    processing: bool = False
    # Real-time scan progress (populated when a manual scan is running)
    scan_progress: Optional[dict] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_datastore_or_404(db: Session, datastore_id: int) -> DataStore:
    ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
    if ds is None:
        raise HTTPException(status_code=404, detail="DataStore not found")
    return ds


def _datastore_in_scope(db: Session, datastore_id: int, admin_org_ids: Optional[List[int]]) -> bool:
    """Return True if the datastore is assigned to an org in the admin's scope."""
    if admin_org_ids is None:
        return True
    return (
        db.query(OrganizationDataStore)
        .filter(
            OrganizationDataStore.data_store_id == datastore_id,
            OrganizationDataStore.org_id.in_(admin_org_ids),
            OrganizationDataStore.is_active == True,
        )
        .first()
        is not None
    )


def _get_watcher() -> DataStoreWatcher:
    """Access the module-level watcher_service from main.py.
    
    This is the single source of truth for _get_watcher.  Both this module
    (CRUD endpoints) and datastore_scan.py (scan endpoints) import from
    here so that tests which patch 'app.api.api_v1.datastores._get_watcher'
    can reach all endpoints.
    """
    from app.main import watcher_service

    if watcher_service is None:
        raise HTTPException(
            status_code=503,
            detail="DataStoreWatcher is not initialized (WATCHER_ENABLED=false?)",
        )
    return watcher_service


def _utc_iso(dt) -> Optional[str]:
    """Serialize a naive-UTC datetime to ISO 8601 with Z suffix."""
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _serialize_ds(ds: DataStore) -> dict:
    """Serialize a DataStore to a response dict."""
    return {
        "id": ds.id,
        "name": ds.name,
        "description": ds.description,
        "folder_path": ds.folder_path,
        "scan_pattern": ds.scan_pattern,
        "is_active": ds.is_active,
        "auto_scan_enabled": ds.auto_scan_enabled,
        "auto_scan_interval_minutes": ds.auto_scan_interval_minutes,
        "last_scan_at": _utc_iso(ds.last_scan_at),
        "last_scan_status": ds.last_scan_status,
        "last_scan_error": ds.last_scan_error,
        "last_scan_total_files": ds.last_scan_total_files or 0,
        "last_scan_processed": ds.last_scan_processed or 0,
        "last_scan_new": ds.last_scan_new or 0,
        "last_scan_modified": ds.last_scan_modified or 0,
        "last_scan_skipped": ds.last_scan_skipped or 0,
        "last_scan_errors": ds.last_scan_errors or 0,
        "last_event_processed": getattr(ds, "last_event_processed", 0) or 0,
        "last_event_at": _utc_iso(getattr(ds, "last_event_at", None)),
        "created_at": _utc_iso(ds.created_at),
        "updated_at": _utc_iso(ds.updated_at),
        "last_recovered_at": _utc_iso(ds.last_recovered_at),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


class DataStoreListResponse(BaseModel):
    items: List[DataStoreResponse]
    total: int
    skip: int
    limit: int


@router.get("/datastores", response_model=DataStoreListResponse)
def list_datastores(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """List datastores visible to the current admin's organisation scope.

    Paginated — use ``skip`` and ``limit`` query params. Default page
    size is 50, max 200.
    """
    limit = min(max(limit, 1), 200)
    skip = max(skip, 0)

    admin_org_ids = get_admin_org_ids(db, current_user)
    query = db.query(DataStore)
    if admin_org_ids is not None:
        query = (
            query
            .join(OrganizationDataStore)
            .filter(
                OrganizationDataStore.org_id.in_(admin_org_ids),
                OrganizationDataStore.is_active == True,
            )
            .distinct()
        )
    total = query.count()
    datastores = query.order_by(DataStore.id).offset(skip).limit(limit).all()

    # Batch-fetch org assignments for all datastores in one query (avoids N+1)
    ds_ids = [ds.id for ds in datastores]
    all_links = (
        db.query(OrganizationDataStore)
        .join(Organisation)
        .filter(
            OrganizationDataStore.data_store_id.in_(ds_ids),
            OrganizationDataStore.is_active == True,
        )
        .all()
        if ds_ids
        else []
    )
    orgs_by_ds: dict[int, list[dict]] = {}
    for link in all_links:
        orgs_by_ds.setdefault(link.data_store_id, []).append(
            {"id": link.organisation.id, "name": link.organisation.name}
        )

    # Batch-fetch graph status counts per datastore (avoids N+1)
    from app.models.knowledge import ProcessingTask
    from sqlalchemy import func, case
    graph_counts: dict[int, dict[str, int]] = {}
    if ds_ids:
        rows = (
            db.query(
                ProcessingTask.data_store_id,
                func.count().label("total"),
                func.sum(case(
                    (ProcessingTask.graph_status == "pending", 1), else_=0,
                )).label("pending"),
                func.sum(case(
                    (ProcessingTask.graph_status == "completed", 1), else_=0,
                )).label("completed"),
                func.sum(case(
                    (ProcessingTask.graph_status == "failed", 1), else_=0,
                )).label("failed"),
            )
            .filter(ProcessingTask.data_store_id.in_(ds_ids))
            .group_by(ProcessingTask.data_store_id)
            .all()
        )
        for r in rows:
            total = int(r.total or 0)
            pending = int(r.pending or 0)
            completed = int(r.completed or 0)
            failed = int(r.failed or 0)
            if pending > 0:
                status = "running"
            elif failed > 0 and completed < total:
                status = "failed"
            elif completed == total and total > 0:
                status = "completed"
            else:
                status = "idle"
            graph_counts[r.data_store_id] = {
                "total": total,
                "pending": pending,
                "completed": completed,
                "failed": failed,
                "status": status,
            }

    result = []
    for ds in datastores:
        resp = _serialize_ds(ds)
        resp["assigned_orgs"] = orgs_by_ds.get(ds.id, [])
        # Include real-time scan progress if a scan is running
        try:
            watcher = _get_watcher()
            status = watcher.get_status()
            resp["pending_changes"] = 0
            resp["processing"] = False
            for ds_status in status.get("datastores", []):
                if ds_status.get("datastore_id") == ds.id:
                    resp["pending_changes"] = ds_status.get("pending_changes", 0)
                    resp["processing"] = ds_status.get("processing", False)
                    break
            for scan in status.get("active_scans", []):
                if scan.get("datastore_id") == ds.id:
                    resp["scan_progress"] = {
                        "total_files": scan.get("total", 0),
                        "processed_files": scan.get("processed", 0),
                        "status": scan.get("status", "idle"),
                        "new_files": scan.get("new", 0),
                        "skipped_files": scan.get("skipped", 0),
                        "error_files": scan.get("error_count", 0),
                    }
                    break
        except HTTPException:
            pass
        resp["graph_summary"] = graph_counts.get(ds.id)
        result.append(DataStoreResponse(**resp))
    return DataStoreListResponse(items=result, total=total, skip=skip, limit=limit)


@router.post("/datastores", response_model=DataStoreResponse, status_code=201)
def create_datastore(
    payload: DataStoreCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_super_admin),
):
    """Create a new datastore. Non-super-admins are auto-assigned to their own org."""
    # Validate folder exists and is under /app/data
    abs_path = _validate_folder_path(payload.folder_path)

    # Check for duplicate path
    existing = (
        db.query(DataStore)
        .filter(DataStore.folder_path == abs_path)
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"DataStore with this path already exists (id={existing.id})",
        )

    # Count files on creation (pre-scan heuristic — no ingestion yet)
    def count_files_in_folder(folder_path: str, scan_pattern: str = "*") -> int:
        try:
            path = Path(folder_path)
            if not path.exists():
                return 0
            patterns = [p.strip() for p in scan_pattern.split(",")]
            all_files = set()
            for pattern in patterns:
                if "*" in pattern:
                    matched = list(path.rglob(pattern))
                else:
                    matched = list(path.glob(pattern))
                all_files.update(f for f in matched if f.is_file() and not f.name.startswith("."))
            return len(all_files)
        except Exception:
            return 0

    file_count = count_files_in_folder(abs_path, payload.scan_pattern)

    ds = DataStore(
        name=payload.name,
        description=payload.description,
        folder_path=abs_path,
        scan_pattern=payload.scan_pattern,
        auto_scan_enabled=payload.auto_scan_enabled,
        auto_scan_interval_minutes=payload.auto_scan_interval_minutes,
        last_scan_total_files=file_count,
        last_scan_status="never",
        last_scan_processed=0,
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)
    logger.info(
        "[DATASTORE] created id=%d name=%s path=%s file_count=%d",
        ds.id, ds.name, ds.folder_path, file_count,
    )

    # Auto-assign non-super-admin created datastores to the admin's own org
    assigned_org_ids = []
    if current_user.role != UserRole.super_admin and current_user.org_id is not None:
        link = OrganizationDataStore(
            org_id=current_user.org_id,
            data_store_id=ds.id,
            is_active=True,
        )
        db.add(link)
        db.commit()
        assigned_org_ids = [{"id": current_user.org_id, "name": current_user.organisation.name if current_user.organisation else None}]

    resp = _serialize_ds(ds)
    resp["assigned_orgs"] = assigned_org_ids

    # Sync watcher so auto_scan-enabled datastores start watching immediately
    if ds.auto_scan_enabled:
        try:
            watcher = _get_watcher()
            watcher.sync_watchers_with_database()
        except Exception:
            logger.warning("[DATASTORE] failed_to_sync_watchers_on_create id=%d", ds.id)

    return DataStoreResponse(**resp)


@router.get("/datastores/{datastore_id}", response_model=DataStoreResponse)
def get_datastore(
    datastore_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Get datastore details."""
    admin_org_ids = get_admin_org_ids(db, current_user)
    ds = _get_datastore_or_404(db, datastore_id)
    if not _datastore_in_scope(db, datastore_id, admin_org_ids):
        raise HTTPException(status_code=404, detail="DataStore not found")

    resp = _serialize_ds(ds)
    links = (
        db.query(OrganizationDataStore)
        .join(Organisation)
        .filter(
            OrganizationDataStore.data_store_id == ds.id,
            OrganizationDataStore.is_active == True,
        )
        .all()
    )
    if admin_org_ids is not None:
        links = [link for link in links if link.organisation.id in admin_org_ids]
    resp["assigned_orgs"] = [
        {"id": link.organisation.id, "name": link.organisation.name}
        for link in links
    ]
    # Graph summary for single datastore
    from app.models.knowledge import ProcessingTask
    from sqlalchemy import func, case
    row = (
        db.query(
            func.count().label("total"),
            func.sum(case(
                (ProcessingTask.graph_status == "pending", 1), else_=0,
            )).label("pending"),
            func.sum(case(
                (ProcessingTask.graph_status == "completed", 1), else_=0,
            )).label("completed"),
            func.sum(case(
                (ProcessingTask.graph_status == "failed", 1), else_=0,
            )).label("failed"),
        )
        .filter(ProcessingTask.data_store_id == ds.id)
        .first()
    )
    if row and (row.total or 0) > 0:
        total = int(row.total or 0)
        pending = int(row.pending or 0)
        completed = int(row.completed or 0)
        failed = int(row.failed or 0)
        if pending > 0:
            status = "running"
        elif failed > 0 and completed < total:
            status = "failed"
        elif completed == total and total > 0:
            status = "completed"
        else:
            status = "idle"
        resp["graph_summary"] = {
            "total": total, "pending": pending,
            "completed": completed, "failed": failed, "status": status,
        }
    return DataStoreResponse(**resp)


@router.patch("/datastores/{datastore_id}", response_model=DataStoreResponse)
def update_datastore(
    datastore_id: int,
    payload: DataStoreUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_super_admin),
):
    """Update a datastore."""
    admin_org_ids = get_admin_org_ids(db, current_user)
    ds = _get_datastore_or_404(db, datastore_id)
    if not _datastore_in_scope(db, datastore_id, admin_org_ids):
        raise HTTPException(status_code=404, detail="DataStore not found")

    if payload.folder_path is not None:
        abs_path = _validate_folder_path(payload.folder_path)

        # Check for duplicate path (excluding current)
        existing = (
            db.query(DataStore)
            .filter(
                DataStore.folder_path == abs_path,
                DataStore.id != datastore_id,
            )
            .first()
        )
        if existing:
            raise HTTPException(
                status_code=400,
                detail=f"DataStore with this path already exists (id={existing.id})",
            )
        ds.folder_path = abs_path

    if payload.name is not None:
        ds.name = payload.name
    if payload.description is not None:
        ds.description = payload.description
    if payload.scan_pattern is not None:
        ds.scan_pattern = payload.scan_pattern
    if payload.is_active is not None:
        ds.is_active = payload.is_active
    if payload.auto_scan_enabled is not None:
        ds.auto_scan_enabled = payload.auto_scan_enabled
    if payload.auto_scan_interval_minutes is not None:
        ds.auto_scan_interval_minutes = payload.auto_scan_interval_minutes

    db.commit()

    # Immediately sync watchers when auto-scan settings change
    if payload.auto_scan_enabled is not None or payload.auto_scan_interval_minutes is not None:
        try:
            watcher = _get_watcher()
            if watcher.is_running:
                watcher.sync_watchers_with_database()
        except Exception as e:
            logger.warning(
                "[DATASTORE] failed_to_sync_watchers_on_update id=%d: %s",
                ds.id, e,
            )
    db.refresh(ds)
    logger.info("[DATASTORE] updated id=%d", ds.id)
    # Get assigned orgs for this datastore (filtered to the admin's scope)
    links = (
        db.query(OrganizationDataStore)
        .join(Organisation)
        .filter(
            OrganizationDataStore.data_store_id == ds.id,
            OrganizationDataStore.is_active == True,
        )
        .all()
    )
    if admin_org_ids is not None:
        links = [link for link in links if link.organisation.id in admin_org_ids]
    resp = _serialize_ds(ds)
    resp["assigned_orgs"] = [
        {
            "id": link.organisation.id,
            "name": link.organisation.name,
        }
        for link in links
    ]
    return DataStoreResponse(**resp)


@router.delete("/datastores/{datastore_id}", status_code=204)
def delete_datastore(
    datastore_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_super_admin),
):
    """
    Delete a datastore and all its associated data.

    Note: Actual files in the DataStore folder are NOT deleted from disk.
    Only database records (DataStore, Documents, Chunks, Vectors, Graph data) are removed.
    """
    admin_org_ids = get_admin_org_ids(db, current_user)
    ds = _get_datastore_or_404(db, datastore_id)
    if not _datastore_in_scope(db, datastore_id, admin_org_ids):
        raise HTTPException(status_code=404, detail="DataStore not found")

    from app.services.cleanup import delete_datastore as _delete_ds
    result, status = _delete_ds(db, datastore_id)
    # Return 204 No Content for success (maintains backward compatibility)
    if status == 204:
        return Response(status_code=204)
    return JSONResponse(status_code=status, content=result)


@router.post("/datastores/{datastore_id}/assign")
def assign_datastore_to_orgs(
    datastore_id: int,
    payload: AssignRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_super_admin),
):
    """Assign a datastore to one or more organisations within the admin's scope."""
    admin_org_ids = get_admin_org_ids(db, current_user)
    _get_datastore_or_404(db, datastore_id)

    if not payload.org_ids:
        if not payload.force_clear:
            raise HTTPException(
                status_code=400,
                detail="Empty org_ids would remove all assignments. "
                       "Set force_clear=true to confirm.",
            )
        # Remove only assignments within the admin's scope
        deleted = (
            db.query(OrganizationDataStore)
            .filter(
                OrganizationDataStore.data_store_id == datastore_id,
                OrganizationDataStore.org_id.in_(admin_org_ids or []),
            )
            .delete(synchronize_session=False)
        )
        logger.info(
            "[DATASTORE] removed %d assignments in scope for id=%d",
            deleted, datastore_id,
        )
        db.commit()
        return

    for org_id in payload.org_ids:
        if admin_org_ids is not None and org_id not in admin_org_ids:
            raise HTTPException(
                status_code=403,
                detail=f"Organisation outside your scope (id={org_id})",
            )
        org = db.query(Organisation).filter(Organisation.id == org_id).first()
        if org is None:
            raise HTTPException(
                status_code=404,
                detail=f"Organisation not found (id={org_id})",
            )

        # Check for duplicate assignment
        existing = (
            db.query(OrganizationDataStore)
            .filter(
                OrganizationDataStore.data_store_id == datastore_id,
                OrganizationDataStore.org_id == org_id,
            )
            .first()
        )
        if existing:
            continue  # Skip duplicates silently

        link = OrganizationDataStore(
            org_id=org_id,
            data_store_id=datastore_id,
            is_active=True,
        )
        db.add(link)

    db.commit()
    logger.info(
        "[DATASTORE] assigned id=%d to orgs=%s",
        datastore_id,
        payload.org_ids,
    )


@router.delete("/datastores/{datastore_id}/assign")
def unassign_datastore_from_orgs(
    datastore_id: int,
    payload: AssignRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_super_admin),
):
    """Unassign a datastore from orgs within the admin's scope."""
    admin_org_ids = get_admin_org_ids(db, current_user)
    _get_datastore_or_404(db, datastore_id)

    if not payload.org_ids:
        raise HTTPException(
            status_code=400,
            detail="org_ids required — cannot unassign without specifying "
                   "which organisations to remove.",
        )

    for org_id in payload.org_ids:
        if admin_org_ids is not None and org_id not in admin_org_ids:
            raise HTTPException(
                status_code=403,
                detail=f"Organisation outside your scope (id={org_id})",
            )
        link = (
            db.query(OrganizationDataStore)
            .filter(
                OrganizationDataStore.data_store_id == datastore_id,
                OrganizationDataStore.org_id == org_id,
            )
            .first()
        )
        if link:
            db.delete(link)

    db.commit()


@router.get("/datastores/{datastore_id}/status", response_model=DataStoreStatusResponse)
def get_datastore_status(
    datastore_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Get datastore scan status."""
    admin_org_ids = get_admin_org_ids(db, current_user)
    ds = _get_datastore_or_404(db, datastore_id)
    if not _datastore_in_scope(db, datastore_id, admin_org_ids):
        raise HTTPException(status_code=404, detail="DataStore not found")
    resp = _serialize_ds(ds)
    resp["datastore_id"] = resp.pop("id")
    resp["pending_changes"] = 0
    resp["processing"] = False

    # Check if watcher has pending changes for this datastore
    try:
        watcher = _get_watcher()
        status = watcher.get_status()
        for ds_status in status.get("datastores", []):
            if ds_status.get("datastore_id") == datastore_id:
                resp["pending_changes"] = ds_status.get("pending_changes", 0)
                resp["processing"] = ds_status.get("processing", False)
                break

        # Include real-time scan progress
        for scan in status.get("active_scans", []):
            if scan.get("datastore_id") == datastore_id:
                resp["scan_progress"] = {
                    "total_files": scan.get("total", 0),
                    "processed_files": scan.get("processed", 0),
                    "status": scan.get("status", "idle"),
                    "new_files": scan.get("new", 0),
                    "skipped_files": scan.get("skipped", 0),
                    "error_files": scan.get("error_count", 0),
                }
                break
    except HTTPException:
        pass

    return DataStoreStatusResponse(**resp)
