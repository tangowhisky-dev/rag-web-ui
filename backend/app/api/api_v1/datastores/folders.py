"""Folder-browsing and file-selection endpoints for datastores.

GET  /datastores/{id}/browse          — file tree with ingestion state
GET  /datastores/{id}/folder-files    — recursive file list with selection state
POST /datastores/{id}/save-selection  — select/unselect files (deletes ingested data on unselect)
POST /datastores/{id}/select-folder   — select/unselect all files in a folder

All endpoints are admin-only and enforce organisation scope.
"""

import logging
import os

from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.security import get_admin_org_ids, require_admin
from app.db.session import get_db
from app.models.user import User

from app.api.api_v1.datastores import router, _datastore_in_scope
from app.api.api_v1.datastores.schemas import SaveSelectionRequest, SelectFolderRequest
from app.api.api_v1.datastores.crud import _get_datastore_or_404

logger = logging.getLogger(__name__)


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


@router.post("/datastores/{datastore_id}/save-selection")
def save_selection(
    datastore_id: int,
    body: SaveSelectionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Save document selection changes.

    Paths in *unselect* have their ingested data (Qdrant, MySQL chunks,
    Neo4j nodes) deleted immediately and is_selected set to false.
    Files on disk are never deleted.  Folder paths are expanded to all
    contained files.

    Paths in *select* have is_selected set to true (or a Document
    record created if none exists).  Ingestion is NOT triggered here —
    selected files are processed later:
      - Manual datastores: on the next manual scan (Process button).
      - Auto-process datastores: on the next interval tick, which
        detects orphan selected documents (is_selected=True, no chunks,
        no task) and ingests them.
    """
    admin_org_ids = get_admin_org_ids(db, current_user)
    ds = _get_datastore_or_404(db, datastore_id)
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
