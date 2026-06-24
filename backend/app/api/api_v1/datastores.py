"""Admin API endpoints for DataStore management.

Endpoints:
    GET    /api/admin/datastores              — list all datastores
    POST   /api/admin/datastores              — create a datastore
    GET    /api/admin/datastores/{id}         — get datastore details
    PATCH  /api/admin/datastores/{id}         — update a datastore
    DELETE /api/admin/datastores/{id}         — delete a datastore
    POST   /api/admin/datastores/{id}/assign  — assign datastore to orgs
    DELETE /api/admin/datastores/{id}/assign  — unassign datastore from orgs
    GET    /api/admin/datastores/{id}/status  — get datastore scan status
    POST   /api/admin/datastores/{id}/scan    — trigger manual scan
"""

import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse

# Helper function to count files in folder
def count_files_in_folder(folder_path: str, scan_pattern: str = "*") -> int:
    """Count files matching pattern in folder."""
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


class ScanResultResponse(BaseModel):
    scanned: int
    new: int
    modified: int
    skipped: int
    errors: int


class ScanProgressResponse(BaseModel):
    datastore_id: int
    datastore_name: str
    scan_id: int | None
    total_files: int
    processed_files: int
    new_files: int
    modified_files: int
    skipped_files: int
    error_files: int
    status: str  # running, completed, error, idle, cancelled
    last_scan_at: Optional[str] = None
    error_message: Optional[str] = None


class ScanStatusResponse(BaseModel):
    running: bool
    active_scans: int
    datastores: List[dict]


class RecoveryStatusResponse(BaseModel):
    """Recovery status for a single datastore."""
    id: int
    name: str
    recovery_status: str  # idle / running / complete / error
    scan_id: Optional[int] = None
    total_files: int = 0
    processed_files: int = 0
    new_files: int = 0
    modified_files: int = 0
    deleted_files: int = 0
    started_at: Optional[str] = None
    error_message: Optional[str] = None
    last_recovered_at: Optional[str] = None


class RecoverResponse(BaseModel):
    """Response for manual recovery trigger."""
    status: str
    scan_id: int


class ManualRecoverRequest(BaseModel):
    """Request body for manual recovery (placeholder — no body params yet)."""
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_datastore_or_404(db: Session, datastore_id: int) -> DataStore:
    ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
    if ds is None:
        raise HTTPException(status_code=404, detail="DataStore not found")
    return ds


def _get_watcher() -> DataStoreWatcher:
    """Access the module-level watcher_service from main.py."""
    from app.main import watcher_service

    if watcher_service is None:
        raise HTTPException(
            status_code=503,
            detail="DataStoreWatcher is not initialized (WATCHER_ENABLED=false?)",
        )
    return watcher_service


def _get_startup_recovery():
    """Access the module-level startup_recovery from main.py."""
    from app.main import startup_recovery

    if startup_recovery is None:
        return None
    return startup_recovery


def _map_recovery_status(scan: Dict[str, Any]) -> RecoveryStatusResponse:
    """Map a recovery scan dict to RecoveryStatusResponse."""
    return RecoveryStatusResponse(
        id=scan.get("datastore_id", 0),
        name=scan.get("datastore_name", ""),
        recovery_status=scan.get("status", "idle"),
        scan_id=scan.get("scan_id"),
        total_files=scan.get("total_files", 0),
        processed_files=scan.get("processed_files", 0),
        new_files=scan.get("new_files", 0),
        modified_files=scan.get("modified_files", 0),
        deleted_files=scan.get("deleted_files", 0),
        started_at=scan.get("started_at"),
        error_message=scan.get("error_message"),
        last_recovered_at=scan.get("last_recovered_at"),
    )


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


@router.get("/datastores/{datastore_id}/scan-progress", response_model=ScanProgressResponse)
def get_datastore_scan_progress(
    datastore_id: int,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    """Get scan progress for a specific datastore."""
    ds = _get_datastore_or_404(db, datastore_id)

    try:
        watcher = _get_watcher()
        status = watcher.get_status()
        active_scans = status.get("active_scans", [])

        # Find the active scan for this datastore
        scan_info = None
        for scan in active_scans:
            if scan.get("datastore_id") == datastore_id:
                scan_info = scan
                break

        # If no active scan, check DB for last scan status
        if not scan_info:
            return ScanProgressResponse(
                datastore_id=ds.id,
                datastore_name=ds.name,
                scan_id=None,
                total_files=ds.last_scan_total_files or 0,
                processed_files=ds.last_scan_processed or 0,
                new_files=ds.last_scan_new or 0,
                modified_files=ds.last_scan_modified or 0,
                skipped_files=ds.last_scan_skipped or 0,
                error_files=ds.last_scan_errors or 0,
                status=ds.last_scan_status if ds.last_scan_status != "running" else "idle",
                last_scan_at=ds.last_scan_at.isoformat() if ds.last_scan_at else None,
                error_message=ds.last_scan_error,
            )

        return ScanProgressResponse(
            datastore_id=ds.id,
            datastore_name=ds.name,
            scan_id=active_scans.index(scan_info) + 1 if scan_info in active_scans else None,
            total_files=scan_info.get("total", 0),
            processed_files=scan_info.get("processed", 0),
            new_files=scan_info.get("new", 0),
            modified_files=scan_info.get("modified", 0),
            skipped_files=scan_info.get("skipped", 0),
            error_files=scan_info.get("error_count", 0),
            status=scan_info.get("status", "idle"),
            last_scan_at=ds.last_scan_at.isoformat() if ds.last_scan_at else None,
            error_message=scan_info.get("error_message"),
        )
    except HTTPException:
        return ScanProgressResponse(
            datastore_id=ds.id,
            datastore_name=ds.name,
            scan_id=None,
            total_files=ds.last_scan_total_files or 0,
            processed_files=ds.last_scan_processed or 0,
            new_files=ds.last_scan_new or 0,
            modified_files=ds.last_scan_modified or 0,
            skipped_files=ds.last_scan_skipped or 0,
            error_files=ds.last_scan_errors or 0,
            status=ds.last_scan_status if ds.last_scan_status != "running" else "idle",
            last_scan_at=ds.last_scan_at.isoformat() if ds.last_scan_at else None,
            error_message=ds.last_scan_error,
        )


@router.get("/datastores/{datastore_id}/scan-progress-stream")
def scan_progress_stream(
    datastore_id: int,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    """SSE endpoint — streams scan progress for a datastore as real-time events.

    Yields JSON-encoded progress events while the scan is running.
    Connection closes automatically when the scan completes, errors,
    or is cancelled. Designed to replace polling with server-driven
    push for smooth, real-time progress updates.
    """
    import asyncio
    import json
    import time

    # Validate datastore exists outside the generator (db session may close
    # after the endpoint returns, before the generator starts)
    ds = _get_datastore_or_404(db, datastore_id)

    async def event_stream():

        try:
            watcher = _get_watcher()
        except HTTPException:
            yield 'data: {"status": "error", "message": "Watcher not available"}\n\n'
            return

        # Wait for scan to appear in active_scans (up to 5 seconds).
        # Iterate in reverse insertion order (Python 3.7+) so that the
        # SSE endpoint always finds the most recently started scan for
        # the datastore — not a stale completed scan from a previous run.
        start_time = time.monotonic()
        while time.monotonic() - start_time < 5:
            scan = None
            for scan_id in reversed(watcher._active_scans.keys()):
                scan = watcher._active_scans[scan_id]
                if scan.get("datastore_id") == datastore_id:
                    break
            if scan is not None:
                logger.info(
                    "[SSE] found scan for datastore_id=%d scan_id=%d status=%s",
                    datastore_id, scan_id, scan.get("status"),
                )
                break
            logger.debug(
                "[SSE] waiting for scan datastore_id=%d active_scans=%s",
                datastore_id, list(watcher._active_scans.keys()),
            )
            yield 'data: {"status": "waiting", "message": "Scan starting..."}\n\n'
            await asyncio.sleep(0.5)
        else:
            logger.info(
                "[SSE] scan_not_found_for_datastore datastore_id=%d",
                datastore_id,
            )
            yield 'data: {"status": "error", "message": "Scan not found"}\n\n'
            return

        # Stream progress updates.
        # Always emit at least one event on entry (even if nothing changed
        # yet) so the client gets a valid initial state.  Then only emit
        # subsequent events when something actually changes — this avoids
        # flooding the SSE connection with redundant events for scans that
        # complete quickly.

        # First emission — always emit the current state so the client
        # never sees undefined values, even for scans that finish before
        # any progress events fire.
        scan = None
        for sid in reversed(watcher._active_scans.keys()):
            scan = watcher._active_scans[sid]
            if scan.get("datastore_id") == datastore_id:
                break
        if scan:
            initial_event = {
                "total_files": scan.get("total", 0),
                "processed_files": scan.get("processed", 0),
                "scanned": scan.get("processed", 0),
                "status": scan.get("status", "running"),
                "new_files": scan.get("new", 0),
                "modified_files": scan.get("modified", 0),
                "skipped_files": scan.get("skipped", 0),
                "error_files": scan.get("error_count", 0),
            }
            if scan.get("error_message"):
                initial_event["error_message"] = scan["error_message"]
            logger.info(
                "[SSE] emitting_initial_event datastore_id=%d event=%s",
                datastore_id, json.dumps(initial_event),
            )
            yield f"data: {json.dumps(initial_event)}\n\n"

        # Subsequent emissions — only when values change
        last_scanned = -1
        last_total = -1
        last_new = -1
        last_modified = -1
        last_skipped = -1
        last_error_count = -1
        last_error_message = None
        last_status = None

        while True:
            scan = None
            for scan_id in reversed(watcher._active_scans.keys()):
                scan = watcher._active_scans[scan_id]
                if scan.get("datastore_id") == datastore_id:
                    break
            if scan is None:
                break  # Scan gone, close connection silently

            scan_status = scan.get("status")
            # _active_scans fields:
            #   processed = total files processed so far (same as summary["scanned"])
            #   total     = total files in folder
            #   status    = running / completed / error / cancelled
            #   new, modified, skipped, error_count = tracked during scan
            #   error_message = string error message from _complete_scan

            current_scanned = scan.get("processed", 0)
            current_total = scan.get("total", 0)
            current_status = scan_status
            current_new = scan.get("new", 0)
            current_modified = scan.get("modified", 0)
            current_skipped = scan.get("skipped", 0)
            current_error_count = scan.get("error_count", 0)
            current_error_message = scan.get("error_message")  # string error from _complete_scan

            # Only emit if something changed since last event
            if (
                current_scanned != last_scanned
                or current_total != last_total
                or current_status != last_status
                or current_error_count != last_error_count
                or current_error_message != last_error_message
                or current_skipped != last_skipped
                or current_modified != last_modified
                or current_new != last_new
            ):
                event = {
                    "total_files": current_total,
                    "processed_files": current_scanned,
                    "scanned": current_scanned,
                    "status": current_status,
                    "new_files": current_new,
                    "modified_files": current_modified,
                    "skipped_files": current_skipped,
                    "error_files": current_error_count,
                }
                # Include error_message only when there's an error
                if current_error_message:
                    event["error_message"] = current_error_message
                logger.info(
                    "[SSE] emitting_event datastore_id=%d event=%s",
                    datastore_id, json.dumps(event),
                )
                yield f"data: {json.dumps(event)}\n\n"
                last_scanned = current_scanned
                last_total = current_total
                last_new = current_new
                last_modified = current_modified
                last_skipped = current_skipped
                last_error_count = current_error_count
                last_error_message = current_error_message
                last_status = current_status

            # If scan is done, stop streaming
            if current_status in ("completed", "error", "cancelled"):
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )


@router.get("/datastores/scan-status", response_model=ScanStatusResponse)
def get_datastores_scan_status(
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    """Get scan status for all datastores."""
    try:
        watcher = _get_watcher()
        status = watcher.get_status()
        return ScanStatusResponse(
            running=status.get("running", False),
            active_scans=len(status.get("active_scans", [])),
            datastores=status.get("datastores", []),
        )
    except HTTPException:
        return ScanStatusResponse(
            running=False,
            active_scans=0,
            datastores=[],
        )


@router.post("/datastores/{datastore_id}/stop-scan")
def stop_datastore_scan(
    datastore_id: int,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    """Cancel a running scan on a datastore."""
    ds = _get_datastore_or_404(db, datastore_id)

    try:
        watcher = _get_watcher()
        cancelled = watcher._cancel_scan(datastore_id)
        if cancelled:
            return {"message": "Scan cancelled", "datastore_id": datastore_id}
        else:
            return {"message": f"No running scan to cancel (status: {ds.last_scan_status})", "status": ds.last_scan_status}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop scan: {str(e)}")


# ---------------------------------------------------------------------------
# Recovery status endpoints
# ---------------------------------------------------------------------------


@router.get("/datastores/recovery-status", response_model=List[RecoveryStatusResponse])
def get_all_recovery_status(
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    """List recovery status for all datastores.

    Returns a list of recovery status dicts (sorted by scan_id).
    Empty list when no recovery is in progress or the service is disabled.
    """
    recovery = _get_startup_recovery()
    if recovery is None:
        return []

    scans = recovery.get_all_status()
    return [_map_recovery_status(s) for s in scans]


@router.get("/datastores/{datastore_id}/recovery-status", response_model=RecoveryStatusResponse)
def get_datastore_recovery_status(
    datastore_id: int,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    """Get recovery status for a specific datastore.

    Returns 404 if the datastore doesn't exist. Returns 503 when
    the recovery service is not initialized.
    """
    ds = _get_datastore_or_404(db, datastore_id)

    recovery = _get_startup_recovery()
    if recovery is None:
        raise HTTPException(
            status_code=503,
            detail="StartupRecoveryService is not initialized",
        )

    scan = recovery.get_status(datastore_id)
    return _map_recovery_status(scan)


@router.get("/datastores/{datastore_id}/recovery-stream")
def recovery_status_stream(
    datastore_id: int,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    """SSE endpoint — streams recovery progress for a datastore.

    Mirrors ``scan_progress_stream()`` but reads from the recovery
    service's ``_active_scans`` instead of the watcher's.  Connection
    closes automatically when the scan completes, errors, or times out.
    """
    import asyncio
    import json
    import time

    ds = _get_datastore_or_404(db, datastore_id)

    async def event_stream():
        recovery = _get_startup_recovery()
        if recovery is None:
            yield 'data: {"status": "error", "message": "Recovery service not available"}\n\n'
            return

        # Wait for scan to appear in active_scans (up to 5 seconds).
        start_time = time.monotonic()
        while time.monotonic() - start_time < 5:
            scan = None
            for sid in reversed(recovery._active_scans.keys()):
                scan = recovery._active_scans[sid]
                if scan.get("datastore_id") == datastore_id:
                    break
            if scan is not None:
                logger.info(
                    "[SSE] found recovery scan for datastore_id=%d scan_id=%d status=%s",
                    datastore_id, sid, scan.get("status"),
                )
                break
            logger.debug(
                "[SSE] waiting for recovery scan datastore_id=%d active_scans=%s",
                datastore_id, list(recovery._active_scans.keys()),
            )
            yield 'data: {"status": "waiting", "message": "Recovery scan starting..."}\n\n'
            await asyncio.sleep(0.5)
        else:
            logger.info(
                "[SSE] recovery_scan_not_found_for_datastore datastore_id=%d",
                datastore_id,
            )
            yield 'data: {"status": "error", "message": "Recovery scan not found"}\n\n'
            return

        # First emission — always emit the current state so the client
        # never sees undefined values.
        scan = None
        for sid in reversed(recovery._active_scans.keys()):
            scan = recovery._active_scans[sid]
            if scan.get("datastore_id") == datastore_id:
                break
        if scan:
            initial_event = {
                "total_files": scan.get("total_files", 0),
                "processed_files": scan.get("processed_files", 0),
                "status": scan.get("status", "running"),
                "new_files": scan.get("new_files", 0),
                "modified_files": scan.get("modified_files", 0),
                "deleted_files": scan.get("deleted_files", 0),
            }
            if scan.get("error_message"):
                initial_event["error_message"] = scan["error_message"]
            logger.info(
                "[SSE] emitting_recovery_initial_event datastore_id=%d event=%s",
                datastore_id, json.dumps(initial_event),
            )
            yield f"data: {json.dumps(initial_event)}\n\n"

        # Subsequent emissions — only when values change
        last_processed = -1
        last_total = -1
        last_new = -1
        last_modified = -1
        last_deleted = -1
        last_error_message = None
        last_status = None

        while True:
            scan = None
            for sid in reversed(recovery._active_scans.keys()):
                scan = recovery._active_scans[sid]
                if scan.get("datastore_id") == datastore_id:
                    break
            if scan is None:
                break  # Scan gone, close connection silently

            current_processed = scan.get("processed_files", 0)
            current_total = scan.get("total_files", 0)
            current_status = scan.get("status")
            current_new = scan.get("new_files", 0)
            current_modified = scan.get("modified_files", 0)
            current_deleted = scan.get("deleted_files", 0)
            current_error_message = scan.get("error_message")

            # Only emit if something changed
            if (
                current_processed != last_processed
                or current_total != last_total
                or current_status != last_status
                or current_new != last_new
                or current_modified != last_modified
                or current_deleted != last_deleted
                or current_error_message != last_error_message
            ):
                event = {
                    "total_files": current_total,
                    "processed_files": current_processed,
                    "status": current_status,
                    "new_files": current_new,
                    "modified_files": current_modified,
                    "deleted_files": current_deleted,
                }
                if current_error_message:
                    event["error_message"] = current_error_message
                logger.info(
                    "[SSE] emitting_recovery_event datastore_id=%d event=%s",
                    datastore_id, json.dumps(event),
                )
                yield f"data: {json.dumps(event)}\n\n"
                last_processed = current_processed
                last_total = current_total
                last_new = current_new
                last_modified = current_modified
                last_deleted = current_deleted
                last_error_message = current_error_message
                last_status = current_status

            # If scan is done, stop streaming
            if current_status in ("complete", "error"):
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )


@router.post("/datastores/{datastore_id}/scan", status_code=202)
async def trigger_datastore_scan(
    datastore_id: int,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    """Asynchronously trigger a scan of a specific datastore.
    
    Returns 202 Accepted immediately with a scan_id. Progress is tracked
    via the SSE endpoint (scan-progress-stream) or the polling endpoint
    (scan-progress). The scan runs in the background and updates the
    datastore status when complete.
    """
    ds = _get_datastore_or_404(db, datastore_id)

    if not ds.folder_path or not os.path.isdir(ds.folder_path):
        raise HTTPException(
            status_code=400,
            detail=f"DataStore folder does not exist: {ds.folder_path}",
        )

    # Check if a scan is already running for this datastore
    try:
        watcher = _get_watcher()
        logger.info(
            "[DATASTORE] scan_check datastore_id=%d active_scans=%s",
            datastore_id, list(watcher._active_scans.items()),
        )
        for scan in watcher._active_scans.values():
            if scan.get("datastore_id") == datastore_id and scan.get("status") == "running":
                raise HTTPException(
                    status_code=409,
                    detail="A scan is already running for this datastore",
                )
    except HTTPException:
        raise
    except Exception:
        pass

    # Clean up any stale scans from previous runs — this prevents the SSE
    # endpoint from finding them if it connects before the new scan thread
    # has started. The 409 guard above guarantees no "running" scans exist,
    # so any remaining entries are from completed/error/cancelled scans.
    # NOTE: we collect the stale scan ID first and pop AFTER the loop, because
    # calling pop() while iterating over items() raises RuntimeError.
    stale_scan_id = None
    for sid, info in watcher._active_scans.items():
        if info.get("datastore_id") == datastore_id:
            stale_scan_id = sid
            break
    if stale_scan_id is not None:
        watcher._active_scans.pop(stale_scan_id, None)
        logger.info(
            "[DATASTORE] cleanup_stale_scan scan_id=%d datastore_id=%d",
            stale_scan_id, datastore_id,
        )
    else:
        logger.info(
            "[DATASTORE] no_stale_scan_for_datastore datastore_id=%d active_scans=%s",
            datastore_id, list(watcher._active_scans.keys()),
        )

    # Also check the DB record — the scan thread may have crashed
    # and left the DB in a "running" state even though _active_scans
    # doesn't have the entry.
    try:
        ds_local = _get_datastore_or_404(db, datastore_id)
        if ds_local.last_scan_status == "running":
            raise HTTPException(
                status_code=409,
                detail="A scan is already running for this datastore",
            )
    except HTTPException:
        raise
    except Exception:
        pass

    # Start scan in background thread
    db.close()

    # Initialize the scan BEFORE starting the thread so the SSE endpoint
    # can always find it — even if the scan completes before the thread
    # begins file processing. This prevents a race condition where the
    # SSE endpoint connects and finds a stale scan entry (or none at all).
    try:
        watcher_local = _get_watcher()
        scan_id = watcher_local._init_scan(datastore_id)
    except Exception as e:
        logger.warning("[DATASTORE] failed_to_init_scan datastore_id=%d: %s", datastore_id, e)
        scan_id = -1

    def run_scan():
        # Need a fresh DB session inside the thread
        db_session: Session = _SessionLocal()
        try:
            ds_local = db_session.query(DataStore).filter(DataStore.id == datastore_id).first()
            if not ds_local or not ds_local.folder_path or not os.path.isdir(ds_local.folder_path):
                logger.warning("[DATASTORE] scan cancelled — folder missing datastore_id=%d", datastore_id)
                return

            # Check if _init_scan was already called from the POST handler.
            # If so, skip re-initialization to avoid creating a duplicate
            # scan entry and losing the progress state from the SSE endpoint.
            watcher_local = _get_watcher()
            scan_already_initialized = False
            for sid, info in watcher_local._active_scans.items():
                if info.get("datastore_id") == datastore_id:
                    scan_already_initialized = True
                    break

            latest_file_count = count_files_in_folder(
                ds_local.folder_path, ds_local.scan_pattern
            )

            if not scan_already_initialized:
                ds_local.last_scan_total_files = latest_file_count
                ds_local.last_scan_at = datetime.now(timezone.utc)
                ds_local.last_scan_status = "running"
                ds_local.last_scan_error = None
                db_session.commit()

            result = watcher_local.scan_single_datastore(datastore_id)

            # Update datastore status with scan results
            ds_local.last_scan_at = datetime.now(timezone.utc)
            ds_local.last_scan_status = "completed" if result.get("errors", 0) == 0 else "error"
            ds_local.last_scan_total_files = latest_file_count
            ds_local.last_scan_processed = result.get("scanned", 0)
            ds_local.last_scan_new = result.get("new", 0)
            ds_local.last_scan_modified = result.get("modified", 0)
            ds_local.last_scan_skipped = result.get("skipped", 0)
            ds_local.last_scan_errors = result.get("errors", 0)
            if result.get("errors", 0) > 0:
                ds_local.last_scan_error = f"{result['errors']} errors during scan"
            else:
                ds_local.last_scan_error = None

            db_session.commit()
            logger.info(
                "[DATASTORE] scan_complete id=%d scanned=%d new=%d modified=%d skipped=%d errors=%d",
                datastore_id,
                result.get("scanned", 0),
                result.get("new", 0),
                result.get("modified", 0),
                result.get("skipped", 0),
                result.get("errors", 0),
            )
        except Exception as e:
            logger.error(
                "[DATASTORE] scan_error id=%d: %s", datastore_id, e, exc_info=True
            )
            try:
                db_local = _SessionLocal()
                ds_err = db_local.query(DataStore).filter(DataStore.id == datastore_id).first()
                if ds_err:
                    ds_err.last_scan_status = "error"
                    ds_err.last_scan_error = str(e)
                    db_local.commit()
                db_local.close()
            except Exception:
                pass
        finally:
            db_session.close()

    thread = threading.Thread(
        target=run_scan,
        name=f"scan-datastore-{datastore_id}",
        daemon=True,
    )
    thread.start()

    # Return scan_id from the watcher's active_scans — it will be
    # assigned by scan_single_datastore -> _init_scan when the thread runs
    return JSONResponse(
        status_code=202,
        content={"message": "Scan started", "datastore_id": datastore_id},
    )


class FlushResultResponse(BaseModel):
    """Response for flush endpoint — process pending changes for a datastore."""
    datastore_id: int
    pending_processed: int
    processing: bool = False


@router.post("/datastores/{datastore_id}/flush", response_model=FlushResultResponse)
def flush_datastore_changes(
    datastore_id: int,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    """Flush and process pending changes for a specific datastore.

    Processes all queued file changes (from event-driven detection or manual
    flush triggers) immediately. Returns the number of changes processed
    and whether the datastore is still processing.
    """
    ds = _get_datastore_or_404(db, datastore_id)

    if not ds.folder_path or not os.path.isdir(ds.folder_path):
        raise HTTPException(
            status_code=400,
            detail=f"DataStore folder does not exist: {ds.folder_path}",
        )

    # Check if there are pending changes
    try:
        watcher = _get_watcher()
        status = watcher.get_status()
        ds_status = None
        for ds_s in status.get("datastores", []):
            if ds_s.get("datastore_id") == datastore_id:
                ds_status = ds_s
                break

        if not ds_status:
            return FlushResultResponse(
                datastore_id=datastore_id,
                pending_processed=0,
                processing=False,
            )

        pending_count = ds_status.get("pending_changes", 0)
        was_processing = ds_status.get("processing", False)

        # Process pending changes if any
        if pending_count > 0:
            watcher._handler._process_pending_changes(datastore_id)

        # Get updated status after processing
        updated_status = watcher.get_status()
        updated_ds_status = None
        for ds_s in updated_status.get("datastores", []):
            if ds_s.get("datastore_id") == datastore_id:
                updated_ds_status = ds_s
                break

        return FlushResultResponse(
            datastore_id=datastore_id,
            pending_processed=pending_count,
            processing=updated_ds_status.get("processing", False) if updated_ds_status else was_processing,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to flush changes: {str(e)}")


@router.post("/datastores/{datastore_id}/recover", status_code=202)
def trigger_datastore_recovery(
    datastore_id: int,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    """Asynchronously trigger a recovery scan of a specific datastore.

    Returns 202 Accepted immediately with a scan_id. Progress is tracked
    via the recovery-status-stream SSE endpoint or the polling endpoint
    (recovery-status). The recovery runs in the background and sets
    last_recovered_at when complete.
    """
    ds = _get_datastore_or_404(db, datastore_id)

    if not ds.folder_path or not os.path.isdir(ds.folder_path):
        raise HTTPException(
            status_code=400,
            detail=f"DataStore folder does not exist: {ds.folder_path}",
        )

    recovery = _get_startup_recovery()
    if recovery is None:
        raise HTTPException(
            status_code=503,
            detail="StartupRecoveryService is not initialized",
        )

    # Check if a recovery scan is already running for this datastore
    for scan in recovery._active_scans.values():
        if scan.get("datastore_id") == datastore_id and scan.get("status") == "running":
            raise HTTPException(
                status_code=409,
                detail="A recovery scan is already running for this datastore",
            )

    # Clean up any stale recovery entries from previous runs
    stale_scan_id = None
    for sid, info in recovery._active_scans.items():
        if info.get("datastore_id") == datastore_id:
            stale_scan_id = sid
            break
    if stale_scan_id is not None:
        recovery._active_scans.pop(stale_scan_id, None)
        logger.info(
            "[RECOVERY] cleanup_stale_recovery scan_id=%d datastore_id=%d",
            stale_scan_id, datastore_id,
        )

    # Generate a new scan_id and register the scan in active_scans
    scan_id = recovery._next_scan_id()
    recovery._active_scans[scan_id] = {
        "datastore_id": datastore_id,
        "datastore_name": ds.name,
        "status": "running",
        "scan_id": scan_id,
        "total_files": 0,
        "processed_files": 0,
        "new_files": 0,
        "modified_files": 0,
        "deleted_files": 0,
        "error_message": None,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    logger.info(
        "[RECOVERY] recover_triggered datastore_id=%d scan_id=%d",
        datastore_id, scan_id,
    )

    # Submit the discovery pipeline worker for this datastore
    recovery.executor.submit(recovery._discovery_pipeline_worker, ds.id, scan_id)

    return JSONResponse(
        status_code=202,
        content={"status": "accepted", "scan_id": scan_id},
    )
