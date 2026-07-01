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
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict
from sqlalchemy.orm import Session

from app.db.session import SessionLocal as _SessionLocal

from app.core.security import require_admin
from app.db.session import get_db
from app.models.datastore import DataStore, OrganizationDataStore
from app.models.knowledge import Document, DocumentChunk, ProcessingTask, KnowledgeBaseDataStore
from app.models.organisation import Organisation
from app.services.datastore_watcher import DataStoreWatcher
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class DataStoreCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    folder_path: str = Field(..., min_length=1, max_length=1024)
    scan_pattern: str = Field(default="*")
    auto_scan_enabled: bool = False
    auto_scan_interval_minutes: int = Field(default=60, ge=1, le=1440)


class DataStoreUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    folder_path: Optional[str] = None
    scan_pattern: Optional[str] = None
    is_active: Optional[bool] = None
    auto_scan_enabled: Optional[bool] = None
    auto_scan_interval_minutes: Optional[int] = None


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

    model_config = ConfigDict(from_attributes=True)


class AssignRequest(BaseModel):
    org_ids: List[int]


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
        "last_scan_at": ds.last_scan_at.isoformat() if ds.last_scan_at else None,
        "last_scan_status": ds.last_scan_status,
        "last_scan_error": ds.last_scan_error,
        "last_scan_total_files": ds.last_scan_total_files or 0,
        "last_scan_processed": ds.last_scan_processed or 0,
        "last_scan_new": ds.last_scan_new or 0,
        "last_scan_modified": ds.last_scan_modified or 0,
        "last_scan_skipped": ds.last_scan_skipped or 0,
        "last_scan_errors": ds.last_scan_errors or 0,
        "created_at": ds.created_at.isoformat() if ds.created_at else None,
        "updated_at": ds.updated_at.isoformat() if ds.updated_at else None,
        "last_recovered_at": ds.last_recovered_at.isoformat() if ds.last_recovered_at else None,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/datastores", response_model=List[DataStoreResponse])
def list_datastores(
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    """List all datastores with their assigned organisations."""
    datastores = db.query(DataStore).order_by(DataStore.id).all()
    result = []
    for ds in datastores:
        resp = _serialize_ds(ds)
        # Get assigned orgs
        links = (
            db.query(OrganizationDataStore)
            .join(Organisation)
            .filter(
                OrganizationDataStore.data_store_id == ds.id,
                OrganizationDataStore.is_active == True,
            )
            .all()
        )
        resp["assigned_orgs"] = [
            {
                "id": link.organisation.id,
                "name": link.organisation.name,
            }
            for link in links
        ]
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
        result.append(DataStoreResponse(**resp))
    return result


@router.post("/datastores", response_model=DataStoreResponse, status_code=201)
def create_datastore(
    payload: DataStoreCreate,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    """Create a new datastore."""
    # Validate folder exists
    abs_path = os.path.abspath(payload.folder_path)
    if not os.path.isdir(abs_path):
        # Provide helpful guidance about valid paths
        valid_base = "/app/data"
        if os.path.isdir(valid_base):
            raise HTTPException(
                status_code=400,
                detail=f"Directory does not exist: {abs_path}. Valid data sources must be subdirectories of {valid_base}. Create the folder in the Docker container first.",
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Directory does not exist: {abs_path}. Ensure the /app/data volume is mounted and the folder exists.",
            )

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
    resp = _serialize_ds(ds)
    resp["assigned_orgs"] = []
    return DataStoreResponse(**resp)


@router.get("/datastores/{datastore_id}", response_model=DataStoreResponse)
def get_datastore(
    datastore_id: int,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    """Get datastore details."""
    ds = _get_datastore_or_404(db, datastore_id)
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
    resp["assigned_orgs"] = [
        {"id": link.organisation.id, "name": link.organisation.name}
        for link in links
    ]
    return DataStoreResponse(**resp)


@router.patch("/datastores/{datastore_id}", response_model=DataStoreResponse)
def update_datastore(
    datastore_id: int,
    payload: DataStoreUpdate,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    """Update a datastore."""
    ds = _get_datastore_or_404(db, datastore_id)

    if payload.folder_path is not None:
        abs_path = os.path.abspath(payload.folder_path)
        if not os.path.isdir(abs_path):
            valid_base = "/app/data"
            if os.path.isdir(valid_base):
                raise HTTPException(
                    status_code=400,
                    detail=f"Directory does not exist: {abs_path}. Valid data sources must be subdirectories of {valid_base}. Create the folder in the Docker container first.",
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Directory does not exist: {abs_path}. Ensure the /app/data volume is mounted and the folder exists.",
                )
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
    # Get assigned orgs for this datastore
    links = (
        db.query(OrganizationDataStore)
        .join(Organisation)
        .filter(
            OrganizationDataStore.data_store_id == ds.id,
            OrganizationDataStore.is_active == True,
        )
        .all()
    )
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
    _: object = Depends(require_admin),
):
    """
    Delete a datastore and all its associated data.
    
    Note: Actual files in the DataStore folder are NOT deleted from disk.
    Only database records (DataStore, Documents, Chunks, Vectors, Graph data) are removed.
    """
    from app.services.deletion_service import delete_datastore as _delete_ds
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
    _: object = Depends(require_admin),
):
    """Assign a datastore to one or more organisations. Empty org_ids removes all assignments."""
    ds = _get_datastore_or_404(db, datastore_id)

    if not payload.org_ids:
        # Remove all existing assignments
        deleted = (
            db.query(OrganizationDataStore)
            .filter(OrganizationDataStore.data_store_id == datastore_id)
            .delete(synchronize_session=False)
        )
        logger.info(
            "[DATASTORE] removed %d existing assignments for id=%d (org_ids empty)",
            deleted, datastore_id,
        )
        db.commit()
        return

    for org_id in payload.org_ids:
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
    _: object = Depends(require_admin),
):
    """Unassign a datastore from one or more organisations. If org_ids is empty, unassign from all orgs."""
    _get_datastore_or_404(db, datastore_id)

    if not payload.org_ids:
        # Empty list = unassign from ALL orgs
        deleted = (
            db.query(OrganizationDataStore)
            .filter(OrganizationDataStore.data_store_id == datastore_id)
            .delete(synchronize_session=False)
        )
        logger.info(
            "[DATASTORE] unassigned id=%d from all orgs (%d removed)",
            datastore_id, deleted,
        )
    else:
        for org_id in payload.org_ids:
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
    _: object = Depends(require_admin),
):
    """Get datastore scan status."""
    ds = _get_datastore_or_404(db, datastore_id)
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
