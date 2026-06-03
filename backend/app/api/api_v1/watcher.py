"""Admin API endpoints for the local folder file watcher.

Endpoints:
    POST   /api/admin/orgs/{org_id}/watch-dir       — set watch directory
    DELETE /api/admin/orgs/{org_id}/watch-dir       — remove watch directory
    GET    /api/admin/orgs/{org_id}/watcher-status   — get watcher status
    GET    /api/admin/watcher-status-all             — bulk watcher status for all orgs
    POST   /api/admin/orgs/{org_id}/watcher-trigger  — manually trigger a scan
"""

import logging
import os

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.security import require_admin
from app.db.session import get_db
from app.models.organisation import Organisation
from app.services.watcher_service import WatcherService

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class WatchDirRequest(BaseModel):
    watch_dir: str = Field(..., description="Absolute path to the directory to watch")


class WatcherStatusResponse(BaseModel):
    org_id: int
    watch_dir: str | None
    status: str  # "watching" | "stopped" | "not_configured"
    last_scan_at: float | None = None
    files_scanned: int = 0


class ScanResultResponse(BaseModel):
    scanned: int
    new: int
    skipped: int
    errors: int


class SMBShareStatus(BaseModel):
    host: str
    share: str
    connected: bool
    last_scan_at: float | None = None
    last_error: str | None = None


class BulkWatcherStatusEntry(BaseModel):
    org_id: int
    name: str
    watch_dir: str | None
    status: str  # "watching" | "stopped" | "not_configured"
    last_scan_at: float | None = None
    files_scanned: int = 0
    smb_watches: list[SMBShareStatus] = []


class BulkWatcherStatusResponse(BaseModel):
    orgs: list[BulkWatcherStatusEntry]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_org_or_404(db: Session, org_id: int) -> Organisation:
    org = db.query(Organisation).filter(Organisation.id == org_id).first()
    if org is None:
        raise HTTPException(status_code=404, detail="Organisation not found")
    return org


def _get_watcher() -> WatcherService:
    """Access the module-level watcher_service from main.py.

    This is intentionally lazy so the router can be imported even before
    the FastAPI app starts up (e.g. for dependency injection tests).
    """
    # Import here to avoid circular imports
    from app.main import watcher_service  # type: ignore[name-defined]

    if watcher_service is None:
        raise HTTPException(
            status_code=503,
            detail="WatcherService is not initialized (WATCHER_ENABLED=false?)",
        )
    return watcher_service


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/orgs/{org_id}/watch-dir",
    response_model=WatcherStatusResponse,
    status_code=200,
)
def set_watch_dir(
    org_id: int,
    payload: WatchDirRequest,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
) -> WatcherStatusResponse:
    """Set (or update) the watch directory for an organisation."""
    org = _get_org_or_404(db, org_id)

    watch_dir = os.path.abspath(payload.watch_dir)

    if not os.path.isdir(watch_dir):
        raise HTTPException(status_code=400, detail=f"Directory does not exist: {watch_dir}")

    org.watch_dir = watch_dir
    db.commit()
    db.refresh(org)

    # Restart watching to pick up the new directory
    watcher = _get_watcher()
    watcher.add_watch(org_id, watch_dir)

    logger.info(
        "[WATCHER] watch_dir_set org_id=%s dir=%s",
        org_id, watch_dir,
    )

    return WatcherStatusResponse(
        org_id=org_id,
        watch_dir=watch_dir,
        status="watching",
    )


@router.delete(
    "/orgs/{org_id}/watch-dir",
    response_model=WatcherStatusResponse,
)
def remove_watch_dir(
    org_id: int,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
) -> WatcherStatusResponse:
    """Remove the watch directory for an organisation and stop watching it."""
    org = _get_org_or_404(db, org_id)

    if not org.watch_dir:
        return WatcherStatusResponse(
            org_id=org_id,
            watch_dir=None,
            status="not_configured",
        )

    removed_dir = org.watch_dir
    org.watch_dir = None
    db.commit()

    # Stop watching the removed directory
    try:
        watcher = _get_watcher()
        watcher.remove_watch(org_id)
    except HTTPException:
        # Watcher not initialized — skip gracefully
        pass

    logger.info(
        "[WATCHER] watch_dir_removed org_id=%s dir=%s",
        org_id, removed_dir,
    )

    return WatcherStatusResponse(
        org_id=org_id,
        watch_dir=None,
        status="stopped",
    )


@router.get(
    "/orgs/{org_id}/watcher-status",
    response_model=WatcherStatusResponse,
)
def get_watcher_status(
    org_id: int,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
) -> WatcherStatusResponse:
    """Get the current watcher status for an organisation."""
    org = _get_org_or_404(db, org_id)

    if not org.watch_dir:
        return WatcherStatusResponse(
            org_id=org_id,
            watch_dir=None,
            status="not_configured",
        )

    try:
        watcher = _get_watcher()
        status = watcher.get_status()
        return WatcherStatusResponse(
            org_id=org_id,
            watch_dir=org.watch_dir,
            status="watching" if status.get("running") else "stopped",
            last_scan_at=status.get("last_scan_at"),
            files_scanned=status.get("files_scanned", 0),
        )
    except HTTPException:
        # Watcher not initialized
        return WatcherStatusResponse(
            org_id=org_id,
            watch_dir=org.watch_dir,
            status="stopped",
        )


@router.get(
    "/watcher-status-all",
    response_model=BulkWatcherStatusResponse,
)
def get_all_watcher_status(
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
) -> BulkWatcherStatusResponse:
    """Get watcher status for all organisations in a single call."""
    orgs = db.query(Organisation).order_by(Organisation.id).all()

    entries: list[BulkWatcherStatusEntry] = []
    for org in orgs:
        if not org.watch_dir:
            entries.append(
                BulkWatcherStatusEntry(
                    org_id=org.id,
                    name=org.name,
                    watch_dir=None,
                    status="not_configured",
                )
            )
            continue

        try:
            watcher = _get_watcher()
            status = watcher.get_status()
            smb_watches: list[SMBShareStatus] = []
            if org.smb_host and org.smb_share:
                # Attach SMB share status from the global smb_watches list
                for sw in status.get("smb_watches", []):
                    if sw.get("host") == org.smb_host and sw.get("share") == org.smb_share:
                        smb_watches.append(
                            SMBShareStatus(
                                host=sw["host"],
                                share=sw["share"],
                                connected=sw.get("connected", False),
                                last_scan_at=sw.get("last_scan_at"),
                                last_error=sw.get("last_error"),
                            )
                        )
            entries.append(
                BulkWatcherStatusEntry(
                    org_id=org.id,
                    name=org.name,
                    watch_dir=org.watch_dir,
                    status="watching" if status.get("running") else "stopped",
                    last_scan_at=status.get("last_scan_at"),
                    files_scanned=status.get("files_scanned", 0),
                    smb_watches=smb_watches,
                )
            )
        except HTTPException:
            # Watcher not initialized
            entries.append(
                BulkWatcherStatusEntry(
                    org_id=org.id,
                    name=org.name,
                    watch_dir=org.watch_dir,
                    status="stopped",
                    smb_watches=[],
                )
            )

    logger.info(
        "[WATCHER] bulk_status_all orgs=%d",
        len(entries),
    )

    return BulkWatcherStatusResponse(orgs=entries)


@router.post(
    "/orgs/{org_id}/watcher-trigger",
    response_model=ScanResultResponse,
)
def trigger_watcher_scan(
    org_id: int,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
) -> ScanResultResponse:
    """Manually trigger a scan of the watched directory."""
    org = _get_org_or_404(db, org_id)

    if not org.watch_dir:
        raise HTTPException(
            status_code=400,
            detail=f"No watch directory configured for org {org_id}",
        )

    watcher = _get_watcher()
    result = watcher.scan()

    logger.info(
        "[WATCHER] api_trigger org_id=%s scanned=%d new=%d skipped=%d errors=%d",
        org_id, result["scanned"], result["new"], result["skipped"], result["errors"],
    )

    return ScanResultResponse(**result)
