"""Admin API endpoints for DataStore CRUD management.

Endpoints:
    GET    /api/admin/datastores              — list all datastores
    POST   /api/admin/datastores              — create a datastore
    GET    /api/admin/datastores/{id}         — get datastore details
    PATCH  /api/admin/datastores/{id}         — update a datastore
    DELETE /api/admin/datastores/{id}         — delete a datastore
    POST   /api/admin/datastores/{id}/assign  — assign datastore to orgs
    DELETE /api/admin/datastores/{id}/assign  — unassign datastore from orgs

This package was split from a single ``datastores.py`` module.  The
``router``, ``_datastore_in_scope``, and ``_get_watcher`` are defined
here so that external modules (``datastore_scan.py``,
``datastore_recovery.py``) and tests that patch
``app.api.api_v1.datastores._get_watcher`` continue to work without
changes.  Schema and serialization helpers are re-exported from
``schemas`` for the same reason.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session

from app.models.datastore import DataStore, OrganizationDataStore
from app.services.datastore_watcher import DataStoreWatcher

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Shared helpers (kept here so external patch targets stay stable)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Re-exports (schemas + serialization) so existing import paths work
# ---------------------------------------------------------------------------

from app.api.api_v1.datastores.schemas import (  # noqa: E402
    _validate_folder_path,
    _utc_iso,
    _serialize_ds,
    DataStoreCreate,
    DataStoreUpdate,
    DataStoreResponse,
    AssignRequest,
    DataStoreStatusResponse,
    DataStoreListResponse,
    SaveSelectionRequest,
    SelectFolderRequest,
    UpdateMarkdownRequest,
)

# ---------------------------------------------------------------------------
# Import submodules to register their routes on ``router``
# ---------------------------------------------------------------------------

from app.api.api_v1.datastores import crud  # noqa: E402, F401
from app.api.api_v1.datastores import assignments  # noqa: E402, F401
from app.api.api_v1.datastores import status  # noqa: E402, F401
from app.api.api_v1.datastores import folders  # noqa: E402, F401
from app.api.api_v1.datastores import documents  # noqa: E402, F401
