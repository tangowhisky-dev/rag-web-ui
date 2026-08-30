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
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.security import require_admin, get_admin_org_ids
from app.db.session import get_db
from app.models.datastore import DataStore
from app.models.user import User

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
    """Access the startup recovery service stored in main._services."""
    from app.main import _services

    recovery = _services.get("recovery")
    if recovery is None:
        return None
    return recovery


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


def _find_recovery_scan(recovery, datastore_id: int):
    scan = None
    sid = None
    for sid in reversed(recovery._active_scans.keys()):
        scan = recovery._active_scans[sid]
        if scan.get("datastore_id") == datastore_id:
            break
    return scan, sid


def _build_recovery_event(scan: Dict[str, Any], status_default="running") -> Dict[str, Any]:
    event = {
        "total_files": scan.get("total_files", 0),
        "processed_files": scan.get("processed_files", 0),
        "status": scan.get("status", status_default),
        "new_files": scan.get("new_files", 0),
        "modified_files": scan.get("modified_files", 0),
        "deleted_files": scan.get("deleted_files", 0),
    }
    if scan.get("error_message"):
        event["error_message"] = scan["error_message"]
    return event


def _recovery_scan_state(scan: Dict[str, Any]) -> tuple:
    return (
        scan.get("processed_files", 0),
        scan.get("total_files", 0),
        scan.get("status"),
        scan.get("new_files", 0),
        scan.get("modified_files", 0),
        scan.get("deleted_files", 0),
        scan.get("error_message"),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/datastores/recovery-status", response_model=List[RecoveryStatusResponse])
def get_all_recovery_status(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """List recovery status for all datastores in the admin's org scope.

    Returns a list of recovery status dicts (sorted by scan_id).
    Empty list when no recovery is in progress or the service is disabled.
    """
    recovery = _get_startup_recovery()
    if recovery is None:
        return []

    admin_org_ids = get_admin_org_ids(db, current_user)
    from app.api.api_v1.datastores import _datastore_in_scope
    scans = recovery.get_all_status()
    if admin_org_ids is not None:
        scans = [s for s in scans if _datastore_in_scope(db, s.get("datastore_id", 0), admin_org_ids)]
    return [_map_recovery_status(s) for s in scans]


@router.get("/datastores/{datastore_id}/recovery-status", response_model=RecoveryStatusResponse)
def get_datastore_recovery_status(
    datastore_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Get recovery status for a specific datastore.

    Returns 404 if the datastore doesn't exist. Returns 503 when
    the recovery service is not initialized.
    """
    ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
    if ds is None:
        raise HTTPException(status_code=404, detail="DataStore not found")
    _check_datastore_scope(db, datastore_id, current_user)

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
    current_user: User = Depends(require_admin),
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
    _check_datastore_scope(db, datastore_id, current_user)

    async def event_stream():
        recovery = _get_startup_recovery()
        if recovery is None:
            yield 'data: {"status": "error", "message": "Recovery service not available"}\n\n'
            return

        # Wait for scan to appear in active_scans (up to 5 seconds).
        start_time = asyncio.get_event_loop().time() if hasattr(asyncio.get_event_loop(), 'time') else 0
        start_time = __import__('time').monotonic()
        while __import__('time').monotonic() - start_time < 5:
            scan, sid = _find_recovery_scan(recovery, datastore_id)
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
        scan, _ = _find_recovery_scan(recovery, datastore_id)
        if scan:
            initial_event = _build_recovery_event(scan)
            logger.info(
                "[SSE] emitting_recovery_initial_event datastore_id=%d event=%s",
                datastore_id, json.dumps(initial_event),
            )
            yield f"data: {json.dumps(initial_event)}\n\n"

        # Subsequent emissions — only when values change
        last_state = None

        while True:
            scan, _ = _find_recovery_scan(recovery, datastore_id)
            if scan is None:
                break  # Scan gone, close connection silently

            current_state = _recovery_scan_state(scan)

            if current_state != last_state:
                event = _build_recovery_event(scan, status_default=None)
                logger.info(
                    "[SSE] emitting_recovery_event datastore_id=%d event=%s",
                    datastore_id, json.dumps(event),
                )
                yield f"data: {json.dumps(event)}\n\n"
                last_state = current_state

            # If scan is done, stop streaming
            if current_state[2] in ("complete", "error"):
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
