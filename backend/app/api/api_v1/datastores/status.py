"""DataStore scan-status endpoint.

GET /datastores/{id}/status — returns the current scan, ingestion,
and watcher status for a single datastore.

``_get_watcher`` is lazily imported from the package ``__init__``
so that tests which patch ``app.api.api_v1.datastores._get_watcher``
reach this endpoint.
"""

from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.security import get_admin_org_ids, require_admin
from app.db.session import get_db
from app.models.user import User

from app.api.api_v1.datastores import router, _datastore_in_scope
from app.api.api_v1.datastores.schemas import DataStoreStatusResponse, _serialize_ds
from app.api.api_v1.datastores.crud import _get_datastore_or_404


def _get_watcher():
    """Lazy wrapper so monkeypatch on ``app.api.api_v1.datastores._get_watcher`` propagates."""
    from app.api.api_v1.datastores import _get_watcher as _impl
    return _impl()


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
