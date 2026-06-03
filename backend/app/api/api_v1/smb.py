"""Admin API endpoints for SMB share management.

Endpoints:
    POST   /api/admin/orgs/{org_id}/smb-config       — save SMB config, encrypt password
    DELETE /api/admin/orgs/{org_id}/smb-config       — clear SMB config, remove watcher
    POST   /api/admin/orgs/{org_id}/smb-test-connection — test SMB connection without saving
    POST   /api/admin/orgs/{org_id}/smb-scan          — manually trigger a scan
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.security import require_admin
from app.db.session import get_db
from app.models.organisation import Organisation
from app.schemas.smb import (
    SMBConfigRequest,
    SMBConfigResponse,
    SMBScanResponse,
    SMBTestConnectionResponse,
)
from app.services.watcher_service import WatcherService

logger = logging.getLogger(__name__)

router = APIRouter()


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
    from app.main import watcher_service  # type: ignore[name-defined]

    if watcher_service is None:
        raise HTTPException(
            status_code=503,
            detail="WatcherService is not initialized (WATCHER_ENABLED=false?)",
        )
    return watcher_service


def _get_smb_auth():
    """Lazy-import SMBAuth to avoid hard dependency at import time."""
    from app.core.config import settings
    from app.services.smb_auth import get_smb_auth

    return get_smb_auth(settings.SMB_MASTER_KEY)


def _find_smb_watcher(watcher: WatcherService, host: str, share: str):
    """Find an SMBShareWatcher instance by host and share."""
    for w in watcher._smb_watches:
        if w.host == host and w.share == share:
            return w
    return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/orgs/{org_id}/smb-config",
    response_model=SMBConfigResponse,
    status_code=200,
)
def configure_smb_share(
    org_id: int,
    payload: SMBConfigRequest,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
) -> SMBConfigResponse:
    """Save SMB share configuration, encrypt the password, and restart the watcher."""
    org = _get_org_or_404(db, org_id)

    # Encrypt password
    auth = _get_smb_auth()
    encrypted_pw = auth.encrypt(payload.password)

    # Save to org
    org.smb_host = payload.host
    org.smb_share = payload.share
    org.smb_username = payload.username
    org.smb_password_encrypted = encrypted_pw
    org.smb_domain = payload.domain
    db.commit()
    db.refresh(org)

    # Restart SMB watcher to pick up new config
    watcher = _get_watcher()

    # Remove existing watcher for this host/share if present
    existing = _find_smb_watcher(watcher, payload.host, payload.share)
    if existing:
        watcher._smb_watches.remove(existing)

    # Import here to avoid circular imports
    from app.services.smb_watcher import SMBShareWatcher

    new_watcher = SMBShareWatcher(
        host=payload.host,
        share=payload.share,
        username=payload.username,
        password=auth.decrypt(encrypted_pw),
        domain=payload.domain,
        kb_id=None,
    )
    watcher._smb_watches.append(new_watcher)

    logger.info(
        "[SMB] config_saved org_id=%s host=%s share=%s",
        org_id, payload.host, payload.share,
    )

    return SMBConfigResponse(
        org_id=org_id,
        smb_host=payload.host,
        smb_share=payload.share,
        smb_username=payload.username,
        smb_domain=payload.domain,
        status="configured",
    )


@router.delete(
    "/orgs/{org_id}/smb-config",
    response_model=SMBConfigResponse,
)
def remove_smb_config(
    org_id: int,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
) -> SMBConfigResponse:
    """Clear SMB share configuration and remove the watcher."""
    org = _get_org_or_404(db, org_id)

    if not org.smb_host or not org.smb_share:
        return SMBConfigResponse(
            org_id=org_id,
            smb_host=None,
            smb_share=None,
            smb_username=None,
            smb_domain=None,
            status="not_configured",
        )

    host = org.smb_host
    share = org.smb_share
    org.smb_host = None
    org.smb_share = None
    org.smb_username = None
    org.smb_password_encrypted = None
    org.smb_domain = None
    db.commit()

    # Remove the watcher
    try:
        watcher = _get_watcher()
        removed = _find_smb_watcher(watcher, host, share)
        if removed:
            watcher._smb_watches.remove(removed)
            logger.info(
                "[SMB] config_removed org_id=%s host=%s share=%s",
                org_id, host, share,
            )
    except HTTPException:
        pass

    return SMBConfigResponse(
        org_id=org_id,
        smb_host=None,
        smb_share=None,
        smb_username=None,
        smb_domain=None,
        status="not_configured",
    )


@router.post(
    "/orgs/{org_id}/smb-test-connection",
    response_model=SMBTestConnectionResponse,
)
def test_smb_connection(
    org_id: int,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
) -> SMBTestConnectionResponse:
    """Test SMB connection using the saved config — does not save anything."""
    org = _get_org_or_404(db, org_id)

    if not org.smb_host or not org.smb_share:
        return SMBTestConnectionResponse(
            connected=False,
            share_accessible=False,
            error="No SMB configuration found for this organisation",
        )

    auth = _get_smb_auth()
    decrypted_pw = auth.decrypt(org.smb_password_encrypted or "")

    from app.services.smb_watcher import SMBShareWatcher

    test_watcher = SMBShareWatcher(
        host=org.smb_host,
        share=org.smb_share,
        username=org.smb_username or "",
        password=decrypted_pw,
        domain=org.smb_domain,
    )

    connected, error = test_watcher.test_connection()

    if connected:
        logger.info(
            "[SMB] test_connection_ok org_id=%s host=%s share=%s",
            org_id, org.smb_host, org.smb_share,
        )
        return SMBTestConnectionResponse(
            connected=True,
            share_accessible=True,
        )
    else:
        logger.warning(
            "[SMB] test_connection_failed org_id=%s host=%s share=%s error=%s",
            org_id, org.smb_host, org.smb_share, error,
        )
        return SMBTestConnectionResponse(
            connected=False,
            share_accessible=False,
            error=error,
        )


@router.post(
    "/orgs/{org_id}/smb-scan",
    response_model=SMBScanResponse,
)
def trigger_smb_scan(
    org_id: int,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
) -> SMBScanResponse:
    """Manually trigger a scan of the configured SMB share."""
    org = _get_org_or_404(db, org_id)

    if not org.smb_host or not org.smb_share:
        raise HTTPException(
            status_code=400,
            detail=f"No SMB share configured for org {org_id}",
        )

    watcher = _get_watcher()
    auth = _get_smb_auth()

    # Find existing watcher or create a temporary one
    smb_watcher = _find_smb_watcher(watcher, org.smb_host, org.smb_share)
    if smb_watcher is None:
        decrypted_pw = auth.decrypt(org.smb_password_encrypted or "")
        from app.services.smb_watcher import SMBShareWatcher

        smb_watcher = SMBShareWatcher(
            host=org.smb_host,
            share=org.smb_share,
            username=org.smb_username or "",
            password=decrypted_pw,
            domain=org.smb_domain,
        )

    result = smb_watcher.scan()

    logger.info(
        "[SMB] api_scan org_id=%s scanned=%d new=%d skipped=%d errors=%d",
        org_id, result["scanned"], result["new"], result["skipped"], result["errors"],
    )

    return SMBScanResponse(**result)
