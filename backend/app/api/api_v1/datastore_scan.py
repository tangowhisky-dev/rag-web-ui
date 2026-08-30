"""Admin API endpoints for DataStore scan management.

Endpoints:
    GET    /api/admin/datastores/{id}/scan-progress      — scan progress
    GET    /api/admin/datastores/{id}/scan-progress-stream — SSE scan progress
    GET    /api/admin/datastores/scan-status              — all scan status
    POST   /api/admin/datastores/{id}/stop-scan           — stop scan
    POST   /api/admin/datastores/{id}/scan                — trigger scan
    POST   /api/admin/datastores/{id}/flush               — flush pending changes
    POST   /api/admin/datastores/{id}/graph-pause         — pause graph ingestion
    POST   /api/admin/datastores/{id}/graph-resume        — resume graph ingestion
"""

import asyncio
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.session import SessionLocal as _SessionLocal
from app.core.security import require_admin, require_super_admin, get_admin_org_ids
from app.db.session import get_db
from app.models.datastore import DataStore
from app.models.user import User
from app.services.datastore_watcher import DataStoreWatcher

logger = logging.getLogger(__name__)

router = APIRouter()


def _check_datastore_scope(db: Session, datastore_id: int, current_user: User):
    """Raise 403 if the datastore is not in the admin's org scope."""
    from app.api.api_v1.datastores import _datastore_in_scope
    admin_org_ids = get_admin_org_ids(db, current_user)
    if not _datastore_in_scope(db, datastore_id, admin_org_ids):
        raise HTTPException(status_code=403, detail="DataStore not in your organisation scope")


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


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
    status: str  # running, completed, error, idle, cancelled, paused
    last_scan_at: Optional[str] = None
    error_message: Optional[str] = None


class ScanStatusResponse(BaseModel):
    running: bool
    active_scans: int
    datastores: list


class FlushResultResponse(BaseModel):
    """Response for flush endpoint — process pending changes for a datastore."""
    datastore_id: int
    pending_processed: int
    processing: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Import _get_watcher from datastores module at call time (lazy) so that
# tests which patch 'app.api.api_v1.datastores._get_watcher' can reach
# the scan endpoints. This avoids circular imports (datastores imports
# from datastore_scan via the router include).
def _get_watcher() -> DataStoreWatcher:
    from app.api.api_v1.datastores import _get_watcher as _impl
    return _impl()


def _count_files_in_folder(folder_path: str, scan_pattern: str = "*") -> int:
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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _find_active_scan(active_scans, datastore_id):
    for scan in active_scans:
        if scan.get("datastore_id") == datastore_id:
            return scan
    return None


def _scan_progress_from_db(ds):
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
        last_scan_at=ds.last_scan_at.strftime("%Y-%m-%dT%H:%M:%SZ") if ds.last_scan_at else None,
        error_message=ds.last_scan_error,
    )


def _scan_progress_from_active(ds, scan_info, active_scans):
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
        last_scan_at=ds.last_scan_at.strftime("%Y-%m-%dT%H:%M:%SZ") if ds.last_scan_at else None,
        error_message=scan_info.get("error_message"),
    )


@router.get("/datastores/{datastore_id}/scan-progress", response_model=ScanProgressResponse)
def get_datastore_scan_progress(
    datastore_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Get scan progress for a specific datastore."""
    ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
    if ds is None:
        raise HTTPException(status_code=404, detail="DataStore not found")
    _check_datastore_scope(db, datastore_id, current_user)

    try:
        watcher = _get_watcher()
        status = watcher.get_status()
        active_scans = status.get("active_scans", [])
        scan_info = _find_active_scan(active_scans, datastore_id)
        if not scan_info:
            return _scan_progress_from_db(ds)
        return _scan_progress_from_active(ds, scan_info, active_scans)
    except HTTPException:
        return _scan_progress_from_db(ds)


def _find_scan_for_datastore(watcher, datastore_id):
    scan = None
    scan_id = None
    for scan_id in reversed(watcher._active_scans.keys()):
        scan = watcher._active_scans[scan_id]
        if scan.get("datastore_id") == datastore_id:
            break
    return scan_id, scan


def _build_scan_event(scan, default_status="running"):
    event = {
        "total_files": scan.get("total", 0),
        "total_files_on_disk": scan.get("total_files_on_disk", 0),
        "processed_files": scan.get("processed", 0),
        "scanned": scan.get("processed", 0),
        "status": scan.get("status", default_status),
        "new_files": scan.get("new", 0),
        "modified_files": scan.get("modified", 0),
        "skipped_files": scan.get("skipped", 0),
        "error_files": scan.get("error_count", 0),
    }
    if scan.get("error_message"):
        event["error_message"] = scan["error_message"]
    return event


@router.get("/datastores/{datastore_id}/scan-progress-stream")
def scan_progress_stream(
    datastore_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """SSE endpoint — streams scan progress for a datastore as real-time events.

    Yields JSON-encoded progress events while the scan is running.
    Connection closes automatically when the scan completes, errors,
    or is cancelled. Designed to replace polling with server-driven
    push for smooth, real-time progress updates.
    """
    # Validate datastore exists outside the generator (db session may close
    # after the endpoint returns, before the generator starts)
    ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
    if ds is None:
        raise HTTPException(status_code=404, detail="DataStore not found")
    _check_datastore_scope(db, datastore_id, current_user)

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
            if await request.is_disconnected():
                logger.info("[SSE] client disconnected (wait phase) datastore_id=%d", datastore_id)
                return
            scan_id, scan = _find_scan_for_datastore(watcher, datastore_id)
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
        _, scan = _find_scan_for_datastore(watcher, datastore_id)
        if scan:
            initial_event = _build_scan_event(scan)
            logger.info(
                "[SSE] emitting_initial_event datastore_id=%d event=%s",
                datastore_id, json.dumps(initial_event),
            )
            yield f"data: {json.dumps(initial_event)}\n\n"

        # Subsequent emissions — only when values change
        last_state = (-1, -1, None, -1, None, -1, -1, -1)

        while True:
            if await request.is_disconnected():
                logger.info("[SSE] client disconnected datastore_id=%d", datastore_id)
                break

            _, scan = _find_scan_for_datastore(watcher, datastore_id)
            if scan is None:
                break  # Scan gone, close connection silently

            current_status = scan.get("status")
            current_state = (
                scan.get("processed", 0),
                scan.get("total", 0),
                current_status,
                scan.get("error_count", 0),
                scan.get("error_message"),
                scan.get("skipped", 0),
                scan.get("modified", 0),
                scan.get("new", 0),
            )

            # Only emit if something changed since last event
            if current_state != last_state:
                event = _build_scan_event(scan, default_status=None)
                logger.info(
                    "[SSE] emitting_event datastore_id=%d event=%s",
                    datastore_id, json.dumps(event),
                )
                yield f"data: {json.dumps(event)}\n\n"
                last_state = current_state

            # If scan is done, stop streaming
            if current_status in ("completed", "error", "cancelled", "paused", "idle"):
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
    current_user: User = Depends(require_admin),
):
    """Get scan status for all datastores in the admin's org scope."""
    admin_org_ids = get_admin_org_ids(db, current_user)
    try:
        watcher = _get_watcher()
        status = watcher.get_status()
        # Filter datastores to admin's org scope
        if admin_org_ids is not None:
            from app.api.api_v1.datastores import _datastore_in_scope
            scoped_ds_ids = [
                ds_id for ds_id in status.get("datastores", [])
                if isinstance(ds_id, dict) and _datastore_in_scope(db, ds_id.get("datastore_id"), admin_org_ids)
            ]
            status["datastores"] = scoped_ds_ids
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
    pause: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_super_admin),
):
    """Cancel or pause a running scan on a datastore.

    When *pause* is True, sets the scan status to "paused" so it can be
    resumed by calling the scan endpoint again. Already-processed files
    are skipped on resume (idempotent scan design).
    """
    ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
    if ds is None:
        raise HTTPException(status_code=404, detail="DataStore not found")
    _check_datastore_scope(db, datastore_id, current_user)

    try:
        watcher = _get_watcher()
        cancelled = watcher._cancel_scan(datastore_id, pause=pause)
        if cancelled:
            msg = "Scan paused" if pause else "Scan cancelled"
            return {"message": msg, "datastore_id": datastore_id}
        else:
            return {"message": f"No running scan to cancel (status: {ds.last_scan_status})", "status": ds.last_scan_status}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop scan: {str(e)}")


@router.post("/datastores/{datastore_id}/graph-pause")
def pause_graph_ingestion(
    datastore_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_super_admin),
):
    """Pause Neo4j graph ingestion for a datastore.

    Signals in-flight graph builds to stop and prevents new ones from
    starting.  Qdrant ingestion is unaffected.  Pending graph builds
    remain in graph_status='pending' and can be resumed with graph-resume.
    """
    ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
    if ds is None:
        raise HTTPException(status_code=404, detail="DataStore not found")
    _check_datastore_scope(db, datastore_id, current_user)

    ds.graph_ingestion_paused = True
    db.commit()

    # Signal in-flight graph builds to stop
    from app.services.ingestion.ingestion_dispatcher import cancel_graph_builds_for_datastore
    cancelled = cancel_graph_builds_for_datastore(datastore_id)

    logger.info(
        "[DATASTORE] graph_ingestion_paused id=%d cancelled_graph_builds=%d",
        datastore_id, cancelled,
    )
    return {
        "message": "Graph ingestion paused",
        "datastore_id": datastore_id,
        "cancelled_graph_builds": cancelled,
    }


@router.post("/datastores/{datastore_id}/graph-resume")
def resume_graph_ingestion(
    datastore_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_super_admin),
):
    """Resume Neo4j graph ingestion for a datastore.

    Clears the pause flag and starts all pending graph builds for this
    datastore.
    """
    ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
    if ds is None:
        raise HTTPException(status_code=404, detail="DataStore not found")
    _check_datastore_scope(db, datastore_id, current_user)

    ds.graph_ingestion_paused = False
    db.commit()

    # Reset graph_status from 'failed' (cancelled by pause) back to 'pending'
    # so they get picked up by the retry
    from app.models.knowledge import ProcessingTask
    tasks = (
        db.query(ProcessingTask)
        .filter(
            ProcessingTask.data_store_id == datastore_id,
            ProcessingTask.status == "completed",
            ProcessingTask.graph_status == "failed",
        )
        .all()
    )
    reset_count = 0
    for t in tasks:
        if t.graph_error and "Cancelled" in t.graph_error:
            t.graph_status = "pending"
            t.graph_error = None
            reset_count += 1
    if reset_count:
        db.commit()

    # Start pending graph builds
    from app.services.discovery.startup_recovery_service import StartupRecoveryService
    recovery = StartupRecoveryService()
    recovery._retry_pending_graph_builds(datastore_id)

    logger.info(
        "[DATASTORE] graph_ingestion_resumed id=%d reset_failed=%d",
        datastore_id, reset_count,
    )
    return {
        "message": "Graph ingestion resumed",
        "datastore_id": datastore_id,
        "reset_failed_graph_builds": reset_count,
    }


def _check_watcher_scan_running(watcher, datastore_id: int) -> None:
    """Raise 409 if a scan is already running in the watcher's active scans."""
    try:
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
        logger.warning(
            "[DATASTORE] scan_check error for datastore_id=%d",
            datastore_id,
        )


def _cleanup_stale_scans_for_datastore(watcher, datastore_id: int) -> None:
    """Remove any stale scan entries for a datastore from the watcher."""
    stale_scan_id = None
    with watcher._active_scans_lock:
        for sid, info in watcher._active_scans.items():
            if info.get("datastore_id") == datastore_id:
                stale_scan_id = sid
                break
        if stale_scan_id is not None:
            watcher._active_scans.pop(stale_scan_id, None)
    if stale_scan_id is not None:
        logger.info(
            "[DATASTORE] cleanup_stale_scan scan_id=%d datastore_id=%d",
            stale_scan_id, datastore_id,
        )
    else:
        logger.info(
            "[DATASTORE] no_stale_scan_for_datastore datastore_id=%d active_scans=%s",
            datastore_id, list(watcher._active_scans.keys()),
        )


def _check_db_scan_running(db: Session, datastore_id: int) -> None:
    """Raise 409 if the DB record shows a scan is already running."""
    try:
        ds_local = db.query(DataStore).filter(DataStore.id == datastore_id).first()
        if ds_local and ds_local.last_scan_status == "running":
            raise HTTPException(
                status_code=409,
                detail="A scan is already running for this datastore",
            )
    except HTTPException:
        raise
    except Exception:
        pass


def _apply_scan_results_to_db(
    ds_local, result: dict, latest_file_count: int, db_session: Session,
) -> None:
    """Update datastore DB record with scan results."""
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


def _handle_scan_thread_error(datastore_id: int, e: Exception) -> None:
    """Persist scan error status to DB when the scan thread raises."""
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


@router.post("/datastores/{datastore_id}/scan", status_code=202)
async def trigger_datastore_scan(
    datastore_id: int,
    force_full_hash: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_super_admin),
):
    """Asynchronously trigger a scan of a specific datastore.

    Returns 202 Accepted immediately with a scan_id. Progress is tracked
    via the SSE endpoint (scan-progress-stream) or the polling endpoint
    (scan-progress). The scan runs in the background and updates the
    datastore status when complete.

    Query params:
        force_full_hash: When true, hash every file instead of using
            stat-first incremental comparison. Use for periodic
            safety-net scans.
    """
    ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
    if ds is None:
        raise HTTPException(status_code=404, detail="DataStore not found")
    _check_datastore_scope(db, datastore_id, current_user)

    if not ds.folder_path or not os.path.isdir(ds.folder_path):
        raise HTTPException(
            status_code=400,
            detail=f"DataStore folder does not exist: {ds.folder_path}",
        )

    # Check if a scan is already running for this datastore
    watcher = None
    try:
        watcher = _get_watcher()
    except HTTPException:
        raise
    except Exception:
        logger.warning(
            "[DATASTORE] scan_check error for datastore_id=%d",
            datastore_id,
        )

    if watcher is not None:
        _check_watcher_scan_running(watcher, datastore_id)
        _cleanup_stale_scans_for_datastore(watcher, datastore_id)

    # Also check the DB record
    _check_db_scan_running(db, datastore_id)

    # Start scan in background thread
    db.close()

    # Initialize the scan BEFORE starting the thread
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
            watcher_local = _get_watcher()
            scan_already_initialized = False
            for sid, info in watcher_local._active_scans.items():
                if info.get("datastore_id") == datastore_id:
                    scan_already_initialized = True
                    break

            latest_file_count = _count_files_in_folder(
                ds_local.folder_path, ds_local.scan_pattern
            )

            if not scan_already_initialized:
                ds_local.last_scan_total_files = latest_file_count
                ds_local.last_scan_at = datetime.now(timezone.utc)
                ds_local.last_scan_status = "running"
                ds_local.last_scan_error = None
                db_session.commit()

            result = watcher_local.scan_single_datastore(
                datastore_id, force_full_hash=force_full_hash,
            )

            # Check if the scan was paused or cancelled while running.
            # If so, do NOT overwrite the status — _cancel_scan already set
            # it to "paused" or "idle". Overwriting would clobber the pause
            # state and cause recovery to auto-resume on restart.
            #
            # _complete_scan (inside scan_single_datastore) uses a different
            # session, so ds_local may have a stale identity map. Expire it
            # to force a fresh DB read.
            db_session.expire_all()
            ds_fresh = db_session.query(DataStore).filter(DataStore.id == datastore_id).first()
            current_status = ds_fresh.last_scan_status if ds_fresh else "unknown"
            if current_status in ("paused", "idle"):
                logger.info(
                    "[DATASTORE] scan_thread_exit id=%d status=%s — preserving status, not overwriting",
                    datastore_id, current_status,
                )
            else:
                _apply_scan_results_to_db(ds_local, result, latest_file_count, db_session)
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
            _handle_scan_thread_error(datastore_id, e)
        finally:
            db_session.close()

    thread = threading.Thread(
        target=run_scan,
        name=f"scan-datastore-{datastore_id}",
        daemon=True,
    )
    thread.start()

    return JSONResponse(
        status_code=202,
        content={"message": "Scan started", "datastore_id": datastore_id},
    )


@router.post("/datastores/{datastore_id}/flush", response_model=FlushResultResponse)
def flush_datastore_changes(
    datastore_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_super_admin),
):
    """Flush and process pending changes for a specific datastore.

    Processes all queued file changes (from event-driven detection or manual
    flush triggers) immediately. Returns the number of changes processed
    and whether the datastore is still processing.
    """
    ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
    if ds is None:
        raise HTTPException(status_code=404, detail="DataStore not found")
    _check_datastore_scope(db, datastore_id, current_user)

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
