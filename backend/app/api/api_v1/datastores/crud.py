"""CRUD endpoints for DataStore management (list, create, get, update, delete).

Also defines ``_get_datastore_or_404``, the shared helper that every
other submodule (assignments, status, folders, documents) uses to
fetch a datastore or raise 404.

``_get_watcher`` and ``_validate_folder_path`` are lazily imported
from the package ``__init__`` / ``schemas`` so that tests which
patch ``app.api.api_v1.datastores._get_watcher`` or
``app.api.api_v1.datastores._validate_folder_path`` reach these
endpoints.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, HTTPException, Response
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.core.security import get_admin_org_ids, require_admin, require_super_admin
from app.db.session import get_db
from app.models.datastore import DataStore, OrganizationDataStore
from app.models.user import User, UserRole

from app.api.api_v1.datastores import router, _datastore_in_scope
from app.api.api_v1.datastores.schemas import (
    DataStoreCreate,
    DataStoreUpdate,
    DataStoreResponse,
    DataStoreListResponse,
    _serialize_ds,
)
from app.api.api_v1.datastores.queries import (
    _build_datastore_query,
    _fetch_org_assignments,
    _fetch_document_counts,
    _fetch_graph_counts,
    _apply_watcher_status,
    _fetch_assigned_orgs,
    _compute_graph_summary_for_ds,
)

logger = logging.getLogger(__name__)


def _get_watcher():
    """Lazy wrapper so monkeypatch on ``app.api.api_v1.datastores._get_watcher`` propagates."""
    from app.api.api_v1.datastores import _get_watcher as _impl
    return _impl()


def _get_datastore_or_404(db: Session, datastore_id: int) -> DataStore:
    ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
    if ds is None:
        raise HTTPException(status_code=404, detail="DataStore not found")
    return ds


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
    from app.api.api_v1.datastores import _validate_folder_path
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
        from app.api.api_v1.datastores import _validate_folder_path
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
