"""Admin API endpoints for DataStore recovery management.

Endpoints:
    GET    /api/admin/datastores/recovery-status        — all recovery status
    GET    /api/admin/datastores/{id}/recovery-status   — specific recovery status
    GET    /api/admin/datastores/{id}/recovery-stream   — SSE recovery stream
    POST   /api/admin/datastores/{id}/recover           — trigger recovery
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.security import require_admin
from app.db.session import get_db
from app.models.datastore import DataStore

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Endpoints
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
    ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
    if ds is None:
        raise HTTPException(status_code=404, detail="DataStore not found")

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
    # Validate datastore exists outside the generator
    ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
    if ds is None:
        raise HTTPException(status_code=404, detail="DataStore not found")

    async def event_stream():
        recovery = _get_startup_recovery()
        if recovery is None:
            yield 'data: {"status": "error", "message": "Recovery service not available"}\n\n'
            return

        # Wait for scan to appear in active_scans (up to 5 seconds).
        start_time = asyncio.get_event_loop().time() if hasattr(asyncio.get_event_loop(), 'time') else 0
        start_time = __import__('time').monotonic()
        while __import__('time').monotonic() - start_time < 5:
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
    ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
    if ds is None:
        raise HTTPException(status_code=404, detail="DataStore not found")

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
