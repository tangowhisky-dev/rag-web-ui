"""Pydantic request/response schemas and serialization helpers for DataStore endpoints.

Defines every model used by the datastore CRUD, assignment, status,
folder-browsing, and document-editing endpoints, plus two shared
helpers:

* ``_validate_folder_path`` — ensures a folder exists under ``/app/data``.
* ``_serialize_ds`` / ``_utc_iso`` — convert a ``DataStore`` ORM object
  into the plain dict consumed by ``DataStoreResponse``.
"""

import os
from datetime import datetime
from typing import List, Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict

from app.models.datastore import DataStore


# ---------------------------------------------------------------------------
# Folder-path validation
# ---------------------------------------------------------------------------


def _validate_folder_path(folder_path: str) -> str:
    """Validate that a folder path exists and is located under /app/data."""
    abs_path = os.path.abspath(folder_path)
    data_root = "/app/data"
    real_path = os.path.realpath(abs_path)
    real_root = os.path.realpath(data_root)

    if not os.path.isdir(real_path):
        if os.path.isdir(real_root):
            raise HTTPException(
                status_code=400,
                detail=f"Directory does not exist: {abs_path}. Valid data sources must be subdirectories of {data_root}.",
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Directory does not exist: {abs_path}. Ensure the /app/data volume is mounted and the folder exists.",
            )

    if real_path != real_root and not real_path.startswith(real_root + os.sep):
        raise HTTPException(
            status_code=400,
            detail=f"Directory must be under {data_root}: {abs_path}",
        )

    return abs_path


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class DataStoreCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    folder_path: str = Field(..., min_length=1, max_length=768)
    scan_pattern: str = Field(default="*")
    auto_process_enabled: bool = False
    auto_process_interval_minutes: int = Field(default=60, ge=1, le=1440)
    select_all_files: bool = Field(default=False, description="Select all files for immediate processing on creation")


class DataStoreUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    description: Optional[str] = None
    folder_path: Optional[str] = Field(default=None, min_length=1, max_length=768)
    scan_pattern: Optional[str] = None
    is_active: Optional[bool] = None
    auto_process_enabled: Optional[bool] = None
    auto_process_interval_minutes: Optional[int] = Field(default=None, ge=1, le=1440)


class DataStoreResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    folder_path: str
    scan_pattern: str
    is_active: bool
    auto_process_enabled: bool
    auto_process_interval_minutes: int
    last_scan_at: Optional[str] = None
    last_scan_status: str
    last_scan_error: Optional[str] = None
    last_scan_total_files: int
    last_scan_processed: int
    last_scan_new: int = 0
    last_scan_modified: int = 0
    last_scan_skipped: int = 0
    last_scan_errors: int = 0
    selected_files: int = 0
    processed_files: int = 0
    assigned_orgs: List[dict] = []
    created_at: datetime
    updated_at: datetime
    # Real-time scan progress (populated when a manual scan is running)
    scan_progress: Optional[dict] = None
    # Pending changes detected but not yet processed (event-driven queue)
    pending_changes: int = 0
    # Selected files waiting for ingestion (no chunks, no task yet)
    pending_ingestion: int = 0
    # Timestamp of the last successful recovery scan
    last_recovered_at: Optional[str] = None
    # Whether changes are currently being processed (event-driven ingestion)
    processing: bool = False
    # Aggregated graph build status across all documents in this datastore
    graph_summary: Optional[dict] = None
    # Whether Neo4j graph ingestion is paused for this datastore
    graph_ingestion_paused: bool = False

    model_config = ConfigDict(from_attributes=True)


class AssignRequest(BaseModel):
    org_ids: List[int]
    # When true, an empty org_ids list removes all in-scope assignments.
    # Without this flag, empty org_ids is rejected to prevent accidental
    # mass-unassignment from a misclicked empty payload.
    force_clear: bool = False


class DataStoreStatusResponse(BaseModel):
    datastore_id: int
    name: str
    folder_path: str
    last_scan_at: Optional[str] = None
    last_scan_status: str
    last_scan_error: Optional[str] = None
    last_scan_total_files: int
    last_scan_processed: int
    last_scan_new: int = 0
    last_scan_modified: int = 0
    last_scan_skipped: int = 0
    last_scan_errors: int = 0
    # Pending changes detected but not yet processed (event-driven queue)
    pending_changes: int = 0
    # Whether changes are currently being processed (event-driven ingestion)
    processing: bool = False
    # Real-time scan progress (populated when a manual scan is running)
    scan_progress: Optional[dict] = None


class DataStoreListResponse(BaseModel):
    items: List[DataStoreResponse]
    total: int
    skip: int
    limit: int


class SaveSelectionRequest(BaseModel):
    select: List[str] = Field(default_factory=list, description="Absolute file or folder paths to select for ingestion")
    unselect: List[str] = Field(default_factory=list, description="Absolute file or folder paths to unselect (deletes ingested data)")


class SelectFolderRequest(BaseModel):
    path: str = Field(..., description="Relative folder path within the datastore")
    selected: bool = Field(..., description="True to select, False to unselect (deletes data)")
    recursive: bool = True


class UpdateMarkdownRequest(BaseModel):
    markdown: str = Field(..., min_length=1, description="Edited markdown content")
    lock_version: int = Field(..., description="Optimistic lock version from GET")


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _utc_iso(dt) -> Optional[str]:
    """Serialize a naive-UTC datetime to ISO 8601 with Z suffix."""
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _serialize_ds(ds: DataStore) -> dict:
    """Serialize a DataStore to a response dict."""
    return {
        "id": ds.id,
        "name": ds.name,
        "description": ds.description,
        "folder_path": ds.folder_path,
        "scan_pattern": ds.scan_pattern,
        "is_active": ds.is_active,
        "auto_process_enabled": ds.auto_process_enabled,
        "auto_process_interval_minutes": ds.auto_process_interval_minutes,
        "last_scan_at": _utc_iso(ds.last_scan_at),
        "last_scan_status": ds.last_scan_status,
        "last_scan_error": ds.last_scan_error,
        "last_scan_total_files": ds.last_scan_total_files or 0,
        "last_scan_processed": ds.last_scan_processed or 0,
        "last_scan_new": ds.last_scan_new or 0,
        "last_scan_modified": ds.last_scan_modified or 0,
        "last_scan_skipped": ds.last_scan_skipped or 0,
        "last_scan_errors": ds.last_scan_errors or 0,
        "selected_files": 0,  # populated by list endpoint
        "processed_files": 0,  # populated by list endpoint
        "last_event_processed": getattr(ds, "last_event_processed", 0) or 0,
        "last_event_at": _utc_iso(getattr(ds, "last_event_at", None)),
        "created_at": _utc_iso(ds.created_at),
        "updated_at": _utc_iso(ds.updated_at),
        "last_recovered_at": _utc_iso(ds.last_recovered_at),
    }
