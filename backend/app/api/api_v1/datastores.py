"""Admin API endpoints for DataStore management.

Endpoints:
    GET    /api/admin/datastores              — list all datastores
    POST   /api/admin/datastores              — create a datastore
    GET    /api/admin/datastores/{id}         — get datastore details
    PATCH  /api/admin/datastores/{id}         — update a datastore
    DELETE /api/admin/datastores/{id}         — delete a datastore
    POST   /api/admin/datastores/{id}/assign  — assign datastore to orgs
    DELETE /api/admin/datastores/{id}/assign  — unassign datastore from orgs
    GET    /api/admin/datastores/{id}/status  — get datastore scan status
    POST   /api/admin/datastores/{id}/scan    — trigger manual scan
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException

# Helper function to count files in folder
def count_files_in_folder(folder_path: str, scan_pattern: str = "*") -> int:
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
            all_files.update(f for f in matched if f.is_file())
        
        return len(all_files)
    except Exception:
        return 0
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict
from sqlalchemy.orm import Session

from app.core.security import require_admin
from app.db.session import get_db
from app.models.datastore import DataStore, OrganizationDataStore
from app.models.organisation import Organisation
from app.services.datastore_watcher import DataStoreWatcher

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class DataStoreCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    folder_path: str = Field(..., min_length=1, max_length=1024)
    scan_pattern: str = Field(default="*")
    auto_scan_enabled: bool = False
    auto_scan_interval_minutes: int = Field(default=60, ge=1, le=1440)


class DataStoreUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    folder_path: Optional[str] = None
    scan_pattern: Optional[str] = None
    is_active: Optional[bool] = None
    auto_scan_enabled: Optional[bool] = None
    auto_scan_interval_minutes: Optional[int] = None


class DataStoreResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    folder_path: str
    scan_pattern: str
    is_active: bool
    auto_scan_enabled: bool
    auto_scan_interval_minutes: int
    last_scan_at: Optional[str] = None
    last_scan_status: str
    last_scan_error: Optional[str] = None
    last_scan_total_files: int
    last_scan_processed: int
    assigned_orgs: List[dict] = []
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AssignRequest(BaseModel):
    org_ids: List[int]


class DataStoreStatusResponse(BaseModel):
    datastore_id: int
    name: str
    folder_path: str
    last_scan_at: Optional[str] = None
    last_scan_status: str
    last_scan_error: Optional[str] = None
    last_scan_total_files: int
    last_scan_processed: int
    pending_changes: int = 0


class ScanResultResponse(BaseModel):
    scanned: int
    new: int
    skipped: int
    errors: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_datastore_or_404(db: Session, datastore_id: int) -> DataStore:
    ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
    if ds is None:
        raise HTTPException(status_code=404, detail="DataStore not found")
    return ds


def _get_watcher() -> DataStoreWatcher:
    """Access the module-level watcher_service from main.py."""
    from app.main import watcher_service

    if watcher_service is None:
        raise HTTPException(
            status_code=503,
            detail="DataStoreWatcher is not initialized (WATCHER_ENABLED=false?)",
        )
    return watcher_service


def _serialize_ds(ds: DataStore) -> dict:
    """Serialize a DataStore to a response dict."""
    return {
        "id": ds.id,
        "name": ds.name,
        "description": ds.description,
        "folder_path": ds.folder_path,
        "scan_pattern": ds.scan_pattern,
        "is_active": ds.is_active,
        "auto_scan_enabled": ds.auto_scan_enabled,
        "auto_scan_interval_minutes": ds.auto_scan_interval_minutes,
        "last_scan_at": ds.last_scan_at.isoformat() if ds.last_scan_at else None,
        "last_scan_status": ds.last_scan_status,
        "last_scan_error": ds.last_scan_error,
        "last_scan_total_files": ds.last_scan_total_files or 0,
        "last_scan_processed": ds.last_scan_processed or 0,
        "created_at": ds.created_at.isoformat() if ds.created_at else None,
        "updated_at": ds.updated_at.isoformat() if ds.updated_at else None,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/datastores", response_model=List[DataStoreResponse])
def list_datastores(
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    """List all datastores with their assigned organisations."""
    datastores = db.query(DataStore).order_by(DataStore.id).all()
    result = []
    for ds in datastores:
        resp = _serialize_ds(ds)
        # Get assigned orgs
        links = (
            db.query(OrganizationDataStore)
            .join(Organisation)
            .filter(
                OrganizationDataStore.data_store_id == ds.id,
                OrganizationDataStore.is_active == True,
            )
            .all()
        )
        resp["assigned_orgs"] = [
            {
                "id": link.organisation.id,
                "name": link.organisation.name,
            }
            for link in links
        ]
        result.append(DataStoreResponse(**resp))
    return result


@router.post("/datastores", response_model=DataStoreResponse, status_code=201)
def create_datastore(
    payload: DataStoreCreate,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    """Create a new datastore."""
    # Validate folder exists
    abs_path = os.path.abspath(payload.folder_path)
    if not os.path.isdir(abs_path):
        # Provide helpful guidance about valid paths
        valid_base = "/app/data"
        if os.path.isdir(valid_base):
            raise HTTPException(
                status_code=400,
                detail=f"Directory does not exist: {abs_path}. Valid data sources must be subdirectories of {valid_base}. Create the folder in the Docker container first.",
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Directory does not exist: {abs_path}. Ensure the /app/data volume is mounted and the folder exists.",
            )

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

    # Count files on creation
    file_count = count_files_in_folder(abs_path, payload.scan_pattern)

    ds = DataStore(
        name=payload.name,
        description=payload.description,
        folder_path=abs_path,
        scan_pattern=payload.scan_pattern,
        auto_scan_enabled=payload.auto_scan_enabled,
        auto_scan_interval_minutes=payload.auto_scan_interval_minutes,
        last_scan_total_files=file_count,
        last_scan_at=datetime.now(timezone.utc),
        last_scan_status="completed",
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)
    logger.info(
        "[DATASTORE] created id=%d name=%s path=%s file_count=%d",
        ds.id, ds.name, ds.folder_path, file_count,
    )
    return ds


@router.get("/datastores/{datastore_id}", response_model=DataStoreResponse)
def get_datastore(
    datastore_id: int,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    """Get datastore details."""
    ds = _get_datastore_or_404(db, datastore_id)
    resp = _serialize_ds(ds)
    links = (
        db.query(OrganizationDataStore)
        .join(Organisation)
        .filter(
            OrganizationDataStore.data_store_id == ds.id,
            OrganizationDataStore.is_active == True,
        )
        .all()
    )
    resp["assigned_orgs"] = [
        {"id": link.organisation.id, "name": link.organisation.name}
        for link in links
    ]
    return DataStoreResponse(**resp)


@router.patch("/datastores/{datastore_id}", response_model=DataStoreResponse)
def update_datastore(
    datastore_id: int,
    payload: DataStoreUpdate,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    """Update a datastore."""
    ds = _get_datastore_or_404(db, datastore_id)

    if payload.folder_path is not None:
        abs_path = os.path.abspath(payload.folder_path)
        if not os.path.isdir(abs_path):
            valid_base = "/app/data"
            if os.path.isdir(valid_base):
                raise HTTPException(
                    status_code=400,
                    detail=f"Directory does not exist: {abs_path}. Valid data sources must be subdirectories of {valid_base}. Create the folder in the Docker container first.",
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Directory does not exist: {abs_path}. Ensure the /app/data volume is mounted and the folder exists.",
                )
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

    if payload.name is not None:
        ds.name = payload.name
    if payload.description is not None:
        ds.description = payload.description
    if payload.scan_pattern is not None:
        ds.scan_pattern = payload.scan_pattern
    if payload.is_active is not None:
        ds.is_active = payload.is_active
    if payload.auto_scan_enabled is not None:
        ds.auto_scan_enabled = payload.auto_scan_enabled
    if payload.auto_scan_interval_minutes is not None:
        ds.auto_scan_interval_minutes = payload.auto_scan_interval_minutes

    db.commit()
    db.refresh(ds)
    logger.info("[DATASTORE] updated id=%d", ds.id)
    return ds


@router.delete("/datastores/{datastore_id}", status_code=204)
def delete_datastore(
    datastore_id: int,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    """
    Delete a datastore and all its associated data.
    
    Note: Actual files in the DataStore folder are NOT deleted from disk.
    Only database records (DataStore, Documents, Chunks, Vectors, Graph data) are removed.
    """
    ds = _get_datastore_or_404(db, datastore_id)

    # Check if any orgs are assigned
    assigned = (
        db.query(OrganizationDataStore)
        .filter(OrganizationDataStore.data_store_id == datastore_id)
        .count()
    )
    if assigned > 0:
        raise HTTPException(
            status_code=409,
            detail="Cannot delete datastore — it is assigned to one or more organisations",
        )
    
    # Get all documents in this DataStore before deletion
    datastore_docs = db.query(Document).filter(Document.data_store_id == datastore_id).all()
    logger.info(f"[DATASTORE] preparing to delete datastore_id={datastore_id} with {len(datastore_docs)} documents")
    
    # Clean up Qdrant vectors: delete the ds_{datastore_id} collection entirely
    try:
        from qdrant_client import QdrantClient
        from app.core.config import settings
        
        qdrant = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        collection_name = f"ds_{datastore_id}"
        
        # Check if collection exists
        collections = [c.name for c in qdrant.get_collections().collections]
        if collection_name in collections:
            qdrant.delete_collection(collection_name)
            logger.info(f"[DATASTORE] deleted Qdrant collection {collection_name}")
        else:
            logger.info(f"[DATASTORE] collection {collection_name} does not exist, skipping")
    except Exception as e:
        logger.warning(f"[DATASTORE] Qdrant cleanup failed for {collection_name}: {e}")
    
    # Clean up Neo4j graph data for all documents in this DataStore
    # Delete all Chunk nodes with data_store_id = this datastore
    try:
        from app.services.graph_service import _get_driver, settings as graph_settings
        if graph_settings.NEO4J_URI:
            driver = _get_driver()
            with driver.session() as session:
                session.run(
                    """
                    MATCH (c:Chunk {data_store_id: $data_store_id})
                    DETACH DELETE c
                    """,
                    data_store_id=str(datastore_id),
                )
            logger.info(f"[DATASTORE] cleaned up Neo4j Chunk nodes for datastore_id={datastore_id}")
    except Exception as e:
        logger.warning(f"[DATASTORE] Neo4j cleanup failed for datastore_id={datastore_id}: {e}")

    # Delete DB records (CASCADE will handle documents, chunks, tasks)
    db.delete(ds)
    db.commit()
    logger.info("[DATASTORE] deleted id=%d name=%s", ds.id, ds.name)


@router.post("/datastores/{datastore_id}/assign")
def assign_datastore_to_orgs(
    datastore_id: int,
    payload: AssignRequest,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    """Assign a datastore to one or more organisations."""
    ds = _get_datastore_or_404(db, datastore_id)

    for org_id in payload.org_ids:
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
    _: object = Depends(require_admin),
):
    """Unassign a datastore from one or more organisations."""
    _get_datastore_or_404(db, datastore_id)

    for org_id in payload.org_ids:
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
    logger.info(
        "[DATASTORE] unassigned id=%d from orgs=%s",
        datastore_id,
        payload.org_ids,
    )


@router.get("/datastores/{datastore_id}/status", response_model=DataStoreStatusResponse)
def get_datastore_status(
    datastore_id: int,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    """Get datastore scan status."""
    ds = _get_datastore_or_404(db, datastore_id)
    resp = _serialize_ds(ds)
    resp["pending_changes"] = 0

    # Check if watcher has pending changes for this datastore
    try:
        watcher = _get_watcher()
        status = watcher.get_status()
        for ds_status in status.get("datastores", []):
            if ds_status.get("datastore_id") == datastore_id:
                resp["pending_changes"] = ds_status.get("pending_changes", 0)
                break
    except HTTPException:
        pass

    return DataStoreStatusResponse(**resp)


@router.post("/datastores/{datastore_id}/scan", response_model=ScanResultResponse)
def trigger_datastore_scan(
    datastore_id: int,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
):
    """Manually trigger a scan of a specific datastore.
    
    Returns scan results with file counts. The watcher processes files
    in the background and updates progress.
    """
    ds = _get_datastore_or_404(db, datastore_id)

    if not ds.folder_path or not os.path.isdir(ds.folder_path):
        raise HTTPException(
            status_code=400,
            detail=f"DataStore folder does not exist: {ds.folder_path}",
        )

    # Update status to "running"
    ds.last_scan_status = "running"
    ds.last_scan_error = None
    db.commit()

    watcher = _get_watcher()
    result = watcher.scan_single_datastore(datastore_id)

    # Update datastore status
    ds.last_scan_at = datetime.now(timezone.utc)
    ds.last_scan_status = "completed" if result["errors"] == 0 else "error"
    ds.last_scan_total_files = result["scanned"]
    ds.last_scan_processed = result["new"]
    if result["errors"] > 0:
        ds.last_scan_error = f"{result['errors']} errors during scan"
    else:
        ds.last_scan_error = None

    db.commit()
    logger.info(
        "[DATASTORE] scan_complete id=%d scanned=%d new=%d skipped=%d errors=%d",
        datastore_id,
        result["scanned"],
        result["new"],
        result["skipped"],
        result["errors"],
    )
    return ScanResultResponse(**result)
