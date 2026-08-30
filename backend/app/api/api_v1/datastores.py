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
    auto_process_enabled: bool = False
    auto_process_interval_minutes: int = Field(default=60, ge=1, le=1440)
    select_all_files: bool = Field(default=False, description="Select all files for immediate processing on creation")


class DataStoreUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    description: Optional[str] = None
    folder_path: Optional[str] = Field(default=None, min_length=1, max_length=768)
    scan_pattern: Optional[str] = None
    is_active: Optional[bool] = None
    auto_process_enabled: Optional[bool] = None
    auto_process_interval_minutes: Optional[int] = Field(default=None, ge=1, le=1440)


class DataStoreResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    folder_path: str
    scan_pattern: str
    is_active: bool
    auto_process_enabled: bool
    auto_process_interval_minutes: int
    last_scan_at: Optional[str] = None
    last_scan_status: str
    last_scan_error: Optional[str] = None
    last_scan_total_files: int
    last_scan_processed: int
    last_scan_new: int = 0
    last_scan_modified: int = 0
    last_scan_skipped: int = 0
    last_scan_errors: int = 0
    selected_files: int = 0
    processed_files: int = 0
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
    # Whether Neo4j graph ingestion is paused for this datastore
    graph_ingestion_paused: bool = False

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
        "auto_process_enabled": ds.auto_process_enabled,
        "auto_process_interval_minutes": ds.auto_process_interval_minutes,
        "last_scan_at": _utc_iso(ds.last_scan_at),
        "last_scan_status": ds.last_scan_status,
        "last_scan_error": ds.last_scan_error,
        "last_scan_total_files": ds.last_scan_total_files or 0,
        "last_scan_processed": ds.last_scan_processed or 0,
        "last_scan_new": ds.last_scan_new or 0,
        "last_scan_modified": ds.last_scan_modified or 0,
        "last_scan_skipped": ds.last_scan_skipped or 0,
        "last_scan_errors": ds.last_scan_errors or 0,
        "selected_files": 0,  # populated by list endpoint
        "processed_files": 0,  # populated by list endpoint
        "last_event_processed": getattr(ds, "last_event_processed", 0) or 0,
        "last_event_at": _utc_iso(getattr(ds, "last_event_at", None)),
        "created_at": _utc_iso(ds.created_at),
        "updated_at": _utc_iso(ds.updated_at),
        "last_recovered_at": _utc_iso(ds.last_recovered_at),
    }


def _build_datastore_query(db: Session, admin_org_ids: Optional[List[int]]):
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
    return query


def _fetch_org_assignments(db: Session, ds_ids: list[int]) -> dict[int, list[dict]]:
    if not ds_ids:
        return {}
    all_links = (
        db.query(OrganizationDataStore)
        .join(Organisation)
        .filter(
            OrganizationDataStore.data_store_id.in_(ds_ids),
            OrganizationDataStore.is_active == True,
        )
        .all()
    )
    orgs_by_ds: dict[int, list[dict]] = {}
    for link in all_links:
        orgs_by_ds.setdefault(link.data_store_id, []).append(
            {"id": link.organisation.id, "name": link.organisation.name}
        )
    return orgs_by_ds


def _fetch_document_counts(
    db: Session, ds_ids: list[int]
) -> tuple[dict[int, int], dict[int, int]]:
    if not ds_ids:
        return {}, {}
    from app.models.knowledge import Document
    from sqlalchemy import func

    selected_rows = (
        db.query(Document.data_store_id, func.count(Document.id))
        .filter(
            Document.data_store_id.in_(ds_ids),
            Document.is_selected == True,
        )
        .group_by(Document.data_store_id)
        .all()
    )
    selected_counts = {r[0]: r[1] for r in selected_rows}

    processed_rows = (
        db.query(Document.data_store_id, func.count(Document.id))
        .filter(
            Document.data_store_id.in_(ds_ids),
            Document.chunks.any(),
        )
        .group_by(Document.data_store_id)
        .all()
    )
    processed_counts = {r[0]: r[1] for r in processed_rows}
    return selected_counts, processed_counts


def _graph_status_from_counts(total: int, pending: int, completed: int, failed: int) -> str:
    if pending > 0:
        return "running"
    if failed > 0 and completed < total:
        return "failed"
    if completed == total and total > 0:
        return "completed"
    return "idle"


def _fetch_graph_counts(db: Session, ds_ids: list[int]) -> dict[int, dict[str, int]]:
    if not ds_ids:
        return {}
    from app.models.knowledge import ProcessingTask
    from sqlalchemy import func, case

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
    graph_counts: dict[int, dict[str, int]] = {}
    for r in rows:
        total = int(r.total or 0)
        pending = int(r.pending or 0)
        completed = int(r.completed or 0)
        failed = int(r.failed or 0)
        graph_counts[r.data_store_id] = {
            "total": total,
            "pending": pending,
            "completed": completed,
            "failed": failed,
            "status": _graph_status_from_counts(total, pending, completed, failed),
        }
    return graph_counts


def _apply_watcher_status(ds_id: int, resp: dict) -> None:
    try:
        watcher = _get_watcher()
        status = watcher.get_status()
        resp["pending_changes"] = 0
        resp["processing"] = False
        for ds_status in status.get("datastores", []):
            if ds_status.get("datastore_id") == ds_id:
                resp["pending_changes"] = ds_status.get("pending_changes", 0)
                resp["processing"] = ds_status.get("processing", False)
                break
        for scan in status.get("active_scans", []):
            if scan.get("datastore_id") == ds_id:
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


def _fetch_assigned_orgs(db: Session, ds_id: int, admin_org_ids: Optional[List[int]]) -> list[dict]:
    links = (
        db.query(OrganizationDataStore)
        .join(Organisation)
        .filter(
            OrganizationDataStore.data_store_id == ds_id,
            OrganizationDataStore.is_active == True,
        )
        .all()
    )
    if admin_org_ids is not None:
        links = [link for link in links if link.organisation.id in admin_org_ids]
    return [
        {"id": link.organisation.id, "name": link.organisation.name}
        for link in links
    ]


def _compute_graph_summary_for_ds(db: Session, ds_id: int) -> Optional[dict]:
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
        .filter(ProcessingTask.data_store_id == ds_id)
        .first()
    )
    if row and (row.total or 0) > 0:
        total = int(row.total or 0)
        pending = int(row.pending or 0)
        completed = int(row.completed or 0)
        failed = int(row.failed or 0)
        status = _graph_status_from_counts(total, pending, completed, failed)
        return {
            "total": total, "pending": pending,
            "completed": completed, "failed": failed, "status": status,
        }
    return None


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
    query = _build_datastore_query(db, admin_org_ids)
    total = query.count()
    datastores = query.order_by(DataStore.id).offset(skip).limit(limit).all()

    ds_ids = [ds.id for ds in datastores]
    orgs_by_ds = _fetch_org_assignments(db, ds_ids)
    selected_counts, processed_counts = _fetch_document_counts(db, ds_ids)
    graph_counts = _fetch_graph_counts(db, ds_ids)

    result = []
    for ds in datastores:
        resp = _serialize_ds(ds)
        resp["assigned_orgs"] = orgs_by_ds.get(ds.id, [])
        _apply_watcher_status(ds.id, resp)
        resp["graph_summary"] = graph_counts.get(ds.id)
        resp["graph_ingestion_paused"] = bool(getattr(ds, 'graph_ingestion_paused', False))
        resp["selected_files"] = selected_counts.get(ds.id, 0)
        resp["processed_files"] = processed_counts.get(ds.id, 0)
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

    # Walk the folder and create Document records for all supported files.
    # is_selected is set based on select_all_files flag or auto_process_enabled.
    from app.services.ingestion.document_converter import SUPPORTED_EXTENSIONS, CONTENT_TYPE_MAP
    from app.models.knowledge import Document

    def walk_and_create_documents(folder_path: str, scan_pattern: str, datastore_id: int, is_selected: bool) -> int:
        import fnmatch as _fnmatch
        try:
            path = Path(folder_path)
            if not path.exists():
                return 0
            patterns = [p.strip() for p in scan_pattern.split(",")]
            count = 0
            for root, _dirs, filenames in os.walk(folder_path):
                for fname in filenames:
                    if fname.startswith(".") or fname.startswith("~$") or fname.startswith(".~"):
                        continue
                    ext = os.path.splitext(fname)[1].lower()
                    if ext not in SUPPORTED_EXTENSIONS:
                        continue
                    # Check scan pattern
                    matched = any(_fnmatch.fnmatch(fname, p) for p in patterns) if patterns else True
                    if not matched:
                        continue
                    fp = os.path.join(root, fname)
                    try:
                        st = os.stat(fp)
                        size = st.st_size
                    except OSError:
                        size = 0
                    doc = Document(
                        knowledge_base_id=None,
                        data_store_id=datastore_id,
                        file_path=fp,
                        file_name=fname,
                        file_size=size,
                        content_type=CONTENT_TYPE_MAP.get(ext, "application/octet-stream"),
                        is_selected=is_selected,
                    )
                    db.add(doc)
                    count += 1
            db.commit()
            return count
        except Exception as e:
            logger.warning("[DATASTORE] walk_failed path=%s error=%s", folder_path, e)
            return 0

    # Determine initial selection state: select_all_files takes precedence,
    # otherwise auto_process_enabled determines the default.
    initial_selected = payload.select_all_files

    ds = DataStore(
        name=payload.name,
        description=payload.description,
        folder_path=abs_path,
        scan_pattern=payload.scan_pattern,
        auto_process_enabled=payload.auto_process_enabled,
        auto_process_interval_minutes=payload.auto_process_interval_minutes,
        last_scan_total_files=0,
        last_scan_status="never",
        last_scan_processed=0,
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)

    # Create Document records for all supported files in the folder
    file_count = walk_and_create_documents(abs_path, payload.scan_pattern, ds.id, initial_selected)
    ds.last_scan_total_files = file_count
    db.commit()

    logger.info(
        "[DATASTORE] created id=%d name=%s path=%s file_count=%d selected=%s",
        ds.id, ds.name, ds.folder_path, file_count, initial_selected,
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
    if ds.auto_process_enabled:
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
    resp["assigned_orgs"] = _fetch_assigned_orgs(db, ds.id, admin_org_ids)
    resp["graph_summary"] = _compute_graph_summary_for_ds(db, ds.id)
    return DataStoreResponse(**resp)


def _apply_datastore_field_updates(ds: DataStore, payload: DataStoreUpdate) -> None:
    """Apply non-folder_path field updates from payload to the datastore."""
    if payload.name is not None:
        ds.name = payload.name
    if payload.description is not None:
        ds.description = payload.description
    if payload.scan_pattern is not None:
        ds.scan_pattern = payload.scan_pattern
    if payload.is_active is not None:
        ds.is_active = payload.is_active
    if payload.auto_process_enabled is not None:
        ds.auto_process_enabled = payload.auto_process_enabled
    if payload.auto_process_interval_minutes is not None:
        ds.auto_process_interval_minutes = payload.auto_process_interval_minutes


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

    _apply_datastore_field_updates(ds, payload)

    db.commit()

    # Immediately sync watchers when auto-scan settings change
    if payload.auto_process_enabled is not None or payload.auto_process_interval_minutes is not None:
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
    resp = _serialize_ds(ds)
    resp["assigned_orgs"] = _fetch_assigned_orgs(db, ds.id, admin_org_ids)
    return DataStoreResponse(**resp)


@router.delete("/datastores/{datastore_id}", status_code=204)
def delete_datastore(
    datastore_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_super_admin),
):
    """
    Delete a datastore and all its associated data.

    Stops all in-flight ingestion and graph builds before deleting to
    prevent race conditions with background Neo4j writes.

    Note: Actual files in the DataStore folder are NOT deleted from disk.
    Only database records (DataStore, Documents, Chunks, Vectors, Graph data) are removed.
    """
    admin_org_ids = get_admin_org_ids(db, current_user)
    ds = _get_datastore_or_404(db, datastore_id)
    if not _datastore_in_scope(db, datastore_id, admin_org_ids):
        raise HTTPException(status_code=404, detail="DataStore not found")

    # Stop all ingestion and graph builds before deleting.
    # Neo4j graph builds run in background daemon threads and can outlive
    # the scan that started them. Cancelling prevents writes to Neo4j
    # while the deletion service is trying to clean up graph data.
    try:
        watcher = _get_watcher()
        if watcher and watcher.is_running:
            watcher._cancel_scan(datastore_id)
            logger.info("[DATASTORE] stopped scan before delete id=%d", datastore_id)
    except Exception:
        logger.warning("[DATASTORE] failed to stop scan before delete id=%d", datastore_id)

    try:
        from app.services.ingestion.ingestion_dispatcher import cancel_graph_builds_for_datastore
        cancelled_graphs = cancel_graph_builds_for_datastore(datastore_id)
        if cancelled_graphs:
            logger.info("[DATASTORE] cancelled %d graph builds before delete id=%d", cancelled_graphs, datastore_id)
    except Exception:
        logger.warning("[DATASTORE] failed to cancel graph builds before delete id=%d", datastore_id)

    # Remove the datastore from the watcher so no new file events are processed
    try:
        watcher = _get_watcher()
        if watcher and watcher.is_running:
            watcher.remove_datastore(datastore_id)
    except Exception:
        pass

    # Wait for in-flight ingestion threads to finish.  Cancellation signals
    # stop *new* work, but threads already past the check point (e.g. doing
    # OCR) will continue.  Waiting prevents them from recreating Qdrant
    # collections or Neo4j nodes after the delete cleans them up.
    try:
        from app.services.ingestion.ingestion_dispatcher import wait_for_ingestions
        remaining = wait_for_ingestions(datastore_id, timeout=30.0)
        if remaining > 0:
            logger.warning(
                "[DATASTORE] %d ingestions still running after 30s wait — proceeding with delete id=%d",
                remaining, datastore_id,
            )
        elif remaining == 0:
            logger.info("[DATASTORE] all ingestions finished before delete id=%d", datastore_id)
    except Exception:
        logger.warning("[DATASTORE] failed to wait for ingestions before delete id=%d", datastore_id)

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
        # Remove only assignments within the admin's scope.
        # super_admin (admin_org_ids=None) has access to all orgs,
        # so don't filter by org_id in that case.
        q = db.query(OrganizationDataStore).filter(
            OrganizationDataStore.data_store_id == datastore_id,
        )
        if admin_org_ids is not None:
            q = q.filter(OrganizationDataStore.org_id.in_(admin_org_ids))
        deleted = q.delete(synchronize_session=False)
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


# ---------------------------------------------------------------------------
# Per-document management endpoints
# ---------------------------------------------------------------------------

@router.get("/datastores/{datastore_id}/browse")
def browse_datastore(
    datastore_id: int,
    path: str = "",
    sort: str = "name",
    page: int = 0,
    page_size: int = 100,
    search: str = "",
    include_unsupported: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Browse datastore contents as a file tree.

    Returns immediate children (folders + files) of the given path
    within the datastore, with ingestion state for each file.
    """
    admin_org_ids = get_admin_org_ids(db, current_user)
    _get_datastore_or_404(db, datastore_id)
    if not _datastore_in_scope(db, datastore_id, admin_org_ids):
        raise HTTPException(status_code=404, detail="DataStore not found")

    from app.services.datastore.document_management import get_folder_contents
    result = get_folder_contents(
        db, datastore_id,
        relative_path=path,
        sort=sort,
        page=min(max(page, 0), 10000),
        page_size=min(max(page_size, 1), 500),
        search=search,
        include_unsupported=include_unsupported,
    )

    if "error" in result:
        if result["error"] == "datastore_not_found":
            raise HTTPException(status_code=404, detail="DataStore not found")
        if result["error"] == "path_outside_datastore":
            raise HTTPException(status_code=400, detail="Path is outside the datastore")
        if result["error"] == "folder_not_found":
            raise HTTPException(status_code=404, detail="Folder not found")
        raise HTTPException(status_code=500, detail=result.get("detail", "Scan failed"))

    return result


@router.get("/datastores/{datastore_id}/folder-files")
def list_folder_files(
    datastore_id: int,
    path: str = "",
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """List all files recursively under a folder, with selection state.

    Used by the frontend when a folder checkbox is toggled.
    """
    admin_org_ids = get_admin_org_ids(db, current_user)
    _get_datastore_or_404(db, datastore_id)
    if not _datastore_in_scope(db, datastore_id, admin_org_ids):
        raise HTTPException(status_code=404, detail="DataStore not found")

    from app.services.datastore.document_management import list_folder_files as _list
    result = _list(db, datastore_id, relative_path=path)
    if "error" in result:
        if result["error"] == "datastore_not_found":
            raise HTTPException(status_code=404, detail="DataStore not found")
        if result["error"] == "path_outside_datastore":
            raise HTTPException(status_code=400, detail="Path is outside the datastore")
        if result["error"] == "folder_not_found":
            raise HTTPException(status_code=404, detail="Folder not found")
        raise HTTPException(status_code=500, detail=result.get("detail", "Failed"))
    return result


class SaveSelectionRequest(BaseModel):
    select: List[str] = Field(default_factory=list, description="Absolute file or folder paths to select for ingestion")
    unselect: List[str] = Field(default_factory=list, description="Absolute file or folder paths to unselect (deletes ingested data)")


@router.post("/datastores/{datastore_id}/save-selection")
def save_selection(
    datastore_id: int,
    body: SaveSelectionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Save document selection changes.

    Paths in *unselect* have their ingested data (Qdrant, MySQL chunks,
    Neo4j nodes) deleted and is_selected set to false. Files on disk
    are never deleted.  Folder paths are expanded to all contained files.

    Paths in *select* have is_selected set to true (or a Document
    record created if none exists). They will be ingested on the next
    scan.  Folder paths are expanded to all contained files.
    """
    admin_org_ids = get_admin_org_ids(db, current_user)
    _get_datastore_or_404(db, datastore_id)
    if not _datastore_in_scope(db, datastore_id, admin_org_ids):
        raise HTTPException(status_code=404, detail="DataStore not found")

    from app.services.datastore.document_management import (
        unselect_documents, select_documents, expand_folder_paths,
    )

    # Expand any folder paths to their contained file paths
    select_paths = expand_folder_paths(db, datastore_id, body.select) if body.select else []
    unselect_paths = expand_folder_paths(db, datastore_id, body.unselect) if body.unselect else []

    unselect_result = {"unselected": 0, "deleted_chunks": 0, "deleted_qdrant_points": 0, "deleted_graph_nodes": 0, "errors": []}
    select_result = {"selected": 0, "created": 0, "errors": []}

    if unselect_paths:
        unselect_result = unselect_documents(datastore_id, unselect_paths)

    if select_paths:
        select_result = select_documents(datastore_id, select_paths)

    return {
        "unselect": unselect_result,
        "select": select_result,
    }


class SelectFolderRequest(BaseModel):
    path: str = Field(..., description="Relative folder path within the datastore")
    selected: bool = Field(..., description="True to select, False to unselect (deletes data)")
    recursive: bool = True


@router.post("/datastores/{datastore_id}/select-folder")
def select_folder(
    datastore_id: int,
    body: SelectFolderRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Select or unselect all files in a folder.

    When unselecting, deletes ingested data (Qdrant, MySQL, Neo4j)
    for all documents under the folder. Files on disk are untouched.
    """
    admin_org_ids = get_admin_org_ids(db, current_user)
    ds = _get_datastore_or_404(db, datastore_id)
    if not _datastore_in_scope(db, datastore_id, admin_org_ids):
        raise HTTPException(status_code=404, detail="DataStore not found")

    # Resolve the absolute folder path
    folder_abs = os.path.normpath(os.path.join(ds.folder_path, body.path)) if body.path else ds.folder_path
    if not folder_abs.startswith(ds.folder_path):
        raise HTTPException(status_code=400, detail="Path is outside the datastore")
    if not os.path.isdir(folder_abs):
        raise HTTPException(status_code=404, detail="Folder not found")

    from app.services.datastore.document_management import select_folder as _select_folder
    result = _select_folder(datastore_id, folder_abs, body.selected, body.recursive)
    return result


# ---------------------------------------------------------------------------
# Markdown editor endpoints (3-phase pipeline)
# ---------------------------------------------------------------------------

def _get_document_or_404(db: Session, document_id: int) -> "Document":
    from app.models.knowledge import Document
    doc = db.query(Document).filter(Document.id == document_id).first()
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


def _verify_document_in_datastore(db: Session, datastore_id: int, document_id: int):
    """Ensure the document belongs to the given datastore."""
    doc = _get_document_or_404(db, document_id)
    if doc.data_store_id != datastore_id:
        raise HTTPException(status_code=404, detail="Document not found in this datastore")
    return doc


@router.get("/datastores/{datastore_id}/documents/{document_id}/markdown")
def get_document_markdown(
    datastore_id: int,
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Get the converted markdown for a document (editor source)."""
    admin_org_ids = get_admin_org_ids(db, current_user)
    _get_datastore_or_404(db, datastore_id)
    if not _datastore_in_scope(db, datastore_id, admin_org_ids):
        raise HTTPException(status_code=404, detail="DataStore not found")

    doc = _verify_document_in_datastore(db, datastore_id, document_id)

    if doc.conversion_status == "processing":
        raise HTTPException(status_code=409, detail="Conversion in progress")
    if doc.conversion_status == "pending":
        raise HTTPException(status_code=409, detail="Conversion pending")
    if not doc.converted_markdown:
        if doc.conversion_status == "error":
            raise HTTPException(
                status_code=422,
                detail=f"Conversion failed: {doc.conversion_error or 'unknown error'}",
            )
        raise HTTPException(
            status_code=409,
            detail="Markdown not available — run re-convert first",
        )

    return {
        "document_id": doc.id,
        "markdown": doc.converted_markdown,
        "conversion_status": doc.conversion_status,
        "lock_version": doc.lock_version,
        "title": doc.title,
    }


class UpdateMarkdownRequest(BaseModel):
    markdown: str = Field(..., min_length=1, description="Edited markdown content")
    lock_version: int = Field(..., description="Optimistic lock version from GET")


@router.put("/datastores/{datastore_id}/documents/{document_id}/markdown")
def update_document_markdown(
    datastore_id: int,
    document_id: int,
    body: UpdateMarkdownRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Save edited markdown and earmark for reprocessing.

    1. Validate non-empty markdown.
    2. Optimistic lock check.
    3. Persist new markdown.
    4. Set needs_reprocess=True so the next scan re-ingests using
       the saved markdown (without re-converting the source file).
    Returns 202 Accepted.
    """
    admin_org_ids = get_admin_org_ids(db, current_user)
    _get_datastore_or_404(db, datastore_id)
    if not _datastore_in_scope(db, datastore_id, admin_org_ids):
        raise HTTPException(status_code=404, detail="DataStore not found")

    doc = _verify_document_in_datastore(db, datastore_id, document_id)

    # Optimistic lock check
    if doc.lock_version != body.lock_version:
        raise HTTPException(
            status_code=409,
            detail=f"Document was modified by another editor. Expected lock_version={doc.lock_version}, got {body.lock_version}.",
        )

    # Check conversion is done
    if not doc.converted_markdown and doc.conversion_status != "completed":
        raise HTTPException(
            status_code=409,
            detail="Document has not been converted yet — run re-convert first",
        )

    # Persist new markdown + bump lock version + earmark for reprocessing
    doc.converted_markdown = body.markdown
    doc.conversion_status = "completed"
    doc.lock_version = doc.lock_version + 1
    doc.needs_reprocess = True
    db.commit()

    logger.info(
        "[EDITOR] markdown_saved doc_id=%s datastore_id=%s — earmarked for reprocessing",
        document_id, datastore_id,
    )

    return JSONResponse(
        status_code=202,
        content={
            "document_id": document_id,
            "lock_version": doc.lock_version,
            "needs_reprocess": True,
            "message": "Markdown saved. File will be re-ingested on next process cycle.",
        },
    )


@router.post("/datastores/{datastore_id}/documents/{document_id}/reconvert")
def reconvert_document(
    datastore_id: int,
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Re-run conversion from the source file. Overwrites current markdown."""
    admin_org_ids = get_admin_org_ids(db, current_user)
    _get_datastore_or_404(db, datastore_id)
    if not _datastore_in_scope(db, datastore_id, admin_org_ids):
        raise HTTPException(status_code=404, detail="DataStore not found")

    doc = _verify_document_in_datastore(db, datastore_id, document_id)

    if not os.path.isfile(doc.file_path):
        raise HTTPException(status_code=404, detail="Source file not found on disk")

    # Capture plain strings before starting the thread — the SQLAlchemy
    # object will be detached once the request session closes.
    _file_path = doc.file_path
    _file_name = doc.file_name

    import threading

    def _do_reconvert():
        import asyncio
        from app.db.session import SessionLocal
        from app.services.ingestion.document_processor import convert_document
        from app.services.infrastructure.progress_timeout import ProgressTimeout

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            rdb = SessionLocal()
            try:
                def _on_timeout():
                    try:
                        from app.models.knowledge import Document as _Doc
                        d = rdb.query(_Doc).filter(_Doc.id == document_id).first()
                        if d:
                            d.conversion_status = "error"
                            d.conversion_error = "Conversion timeout — no activity for 600s"
                            rdb.commit()
                    except Exception:
                        rdb.rollback()

                async def _run_with_timeout():
                    async with ProgressTimeout(
                        silence_seconds=600,
                        on_timeout=_on_timeout,
                    ):
                        return await convert_document(
                            document_id=document_id,
                            file_path=_file_path,
                            file_name=_file_name,
                            db=rdb,
                        )

                try:
                    loop.run_until_complete(_run_with_timeout())
                    # Mark for reprocessing — the markdown was regenerated
                    # and needs to be re-ingested on next process cycle.
                    from app.models.knowledge import Document as _Doc
                    d = rdb.query(_Doc).filter(_Doc.id == document_id).first()
                    if d:
                        d.needs_reprocess = True
                        rdb.commit()
                except Exception as e:
                    logger.error("reconvert_failed document_id=%s: %s", document_id, e)
                    try:
                        from app.models.knowledge import Document as _Doc
                        d = rdb.query(_Doc).filter(_Doc.id == document_id).first()
                        if d:
                            d.conversion_status = "error"
                            d.conversion_error = str(e)[:500]
                            rdb.commit()
                    except Exception:
                        rdb.rollback()
            finally:
                rdb.close()
        finally:
            loop.close()

    doc.conversion_status = "pending"
    db.commit()

    t = threading.Thread(target=_do_reconvert, name=f"reconvert-{document_id}", daemon=True)
    t.start()

    return JSONResponse(
        status_code=202,
        content={
            "document_id": document_id,
            "conversion_status": "pending",
            "message": "Re-convert queued",
        },
    )


@router.get("/datastores/{datastore_id}/documents/{document_id}/ingest-status")
def get_ingest_status(
    datastore_id: int,
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Get the current ingestion/conversion/graph status for a document."""
    admin_org_ids = get_admin_org_ids(db, current_user)
    _get_datastore_or_404(db, datastore_id)
    if not _datastore_in_scope(db, datastore_id, admin_org_ids):
        raise HTTPException(status_code=404, detail="DataStore not found")

    doc = _verify_document_in_datastore(db, datastore_id, document_id)

    # Get latest task
    from app.models.knowledge import ProcessingTask
    latest_task = (
        db.query(ProcessingTask)
        .filter(ProcessingTask.document_id == document_id)
        .order_by(ProcessingTask.id.desc())
        .first()
    )

    from app.models.knowledge import DocumentChunk
    chunk_count = db.query(DocumentChunk).filter(
        DocumentChunk.document_id == document_id,
        DocumentChunk.data_store_id == datastore_id,
    ).count()

    return {
        "document_id": doc.id,
        "conversion_status": doc.conversion_status,
        "conversion_error": doc.conversion_error,
        "ingest_status": latest_task.status if latest_task else None,
        "ingest_progress": latest_task.progress if latest_task else 0,
        "ingest_message": latest_task.progress_message if latest_task else None,
        "ingest_error": latest_task.error_message if latest_task else None,
        "graph_status": latest_task.graph_status if latest_task else None,
        "graph_error": latest_task.graph_error if latest_task else None,
        "chunk_count": chunk_count,
        "lock_version": doc.lock_version,
    }
