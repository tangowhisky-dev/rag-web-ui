"""Reusable query and aggregation helpers for DataStore list/detail endpoints.

Centralises the batch-query logic that the list endpoint needs
(org assignments, document counts, graph-build counts, watcher
status) so the endpoint functions stay readable.  ``_get_watcher``
is lazily imported from the package ``__init__`` so that tests
which patch ``app.api.api_v1.datastores._get_watcher`` reach
these helpers too.
"""

from typing import List, Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.datastore import DataStore, OrganizationDataStore
from app.models.organisation import Organisation


def _get_watcher():
    """Lazy wrapper so monkeypatch on ``app.api.api_v1.datastores._get_watcher`` propagates."""
    from app.api.api_v1.datastores import _get_watcher as _impl
    return _impl()


def _build_datastore_query(db: Session, admin_org_ids: Optional[List[int]]):
    query = db.query(DataStore)
    if admin_org_ids is not None:
        query = (
            query
            .join(OrganizationDataStore)
            .filter(
                OrganizationDataStore.org_id.in_(admin_org_ids),
                OrganizationDataStore.is_active == True,
            )
            .distinct()
        )
    return query


def _fetch_org_assignments(db: Session, ds_ids: list[int]) -> dict[int, list[dict]]:
    if not ds_ids:
        return {}
    all_links = (
        db.query(OrganizationDataStore)
        .join(Organisation)
        .filter(
            OrganizationDataStore.data_store_id.in_(ds_ids),
            OrganizationDataStore.is_active == True,
        )
        .all()
    )
    orgs_by_ds: dict[int, list[dict]] = {}
    for link in all_links:
        orgs_by_ds.setdefault(link.data_store_id, []).append(
            {"id": link.organisation.id, "name": link.organisation.name}
        )
    return orgs_by_ds


def _fetch_document_counts(
    db: Session, ds_ids: list[int]
) -> tuple[dict[int, int], dict[int, int]]:
    if not ds_ids:
        return {}, {}
    from app.models.knowledge import Document
    from sqlalchemy import func

    selected_rows = (
        db.query(Document.data_store_id, func.count(Document.id))
        .filter(
            Document.data_store_id.in_(ds_ids),
            Document.is_selected == True,
        )
        .group_by(Document.data_store_id)
        .all()
    )
    selected_counts = {r[0]: r[1] for r in selected_rows}

    processed_rows = (
        db.query(Document.data_store_id, func.count(Document.id))
        .filter(
            Document.data_store_id.in_(ds_ids),
            Document.chunks.any(),
        )
        .group_by(Document.data_store_id)
        .all()
    )
    processed_counts = {r[0]: r[1] for r in processed_rows}
    return selected_counts, processed_counts


def _fetch_pending_ingestion_counts(
    db: Session, ds_ids: list[int]
) -> dict[int, int]:
    """Count selected documents with no chunks and no ProcessingTask.

    These are files selected via the datastore browser that are waiting
    for ingestion — either on the next manual scan or the next auto-process
    interval tick.
    """
    if not ds_ids:
        return {}
    from app.models.knowledge import Document, ProcessingTask
    from sqlalchemy import func

    rows = (
        db.query(
            Document.data_store_id,
            func.count(Document.id).label("pending"),
        )
        .outerjoin(ProcessingTask, ProcessingTask.document_id == Document.id)
        .filter(
            Document.data_store_id.in_(ds_ids),
            Document.is_selected == True,  # noqa: E712
            ProcessingTask.id.is_(None),
            ~Document.chunks.any(),
        )
        .group_by(Document.data_store_id)
        .all()
    )
    return {r[0]: int(r[1]) for r in rows}


def _graph_status_from_counts(total: int, pending: int, completed: int, failed: int) -> str:
    if pending > 0:
        return "running"
    if failed > 0 and completed < total:
        return "failed"
    if completed == total and total > 0:
        return "completed"
    return "idle"


def _fetch_graph_counts(db: Session, ds_ids: list[int]) -> dict[int, dict[str, int]]:
    if not ds_ids:
        return {}
    from app.models.knowledge import ProcessingTask
    from sqlalchemy import func, case

    rows = (
        db.query(
            ProcessingTask.data_store_id,
            func.count().label("total"),
            func.sum(case(
                (ProcessingTask.graph_status == "pending", 1), else_=0,
            )).label("pending"),
            func.sum(case(
                (ProcessingTask.graph_status == "completed", 1), else_=0,
            )).label("completed"),
            func.sum(case(
                (ProcessingTask.graph_status == "failed", 1), else_=0,
            )).label("failed"),
        )
        .filter(ProcessingTask.data_store_id.in_(ds_ids))
        .group_by(ProcessingTask.data_store_id)
        .all()
    )
    graph_counts: dict[int, dict[str, int]] = {}
    for r in rows:
        total = int(r.total or 0)
        pending = int(r.pending or 0)
        completed = int(r.completed or 0)
        failed = int(r.failed or 0)
        graph_counts[r.data_store_id] = {
            "total": total,
            "pending": pending,
            "completed": completed,
            "failed": failed,
            "status": _graph_status_from_counts(total, pending, completed, failed),
        }
    return graph_counts


def _apply_watcher_status(ds_id: int, resp: dict) -> None:
    try:
        watcher = _get_watcher()
        status = watcher.get_status()
        resp["pending_changes"] = 0
        resp["processing"] = False
        for ds_status in status.get("datastores", []):
            if ds_status.get("datastore_id") == ds_id:
                resp["pending_changes"] = ds_status.get("pending_changes", 0)
                resp["processing"] = ds_status.get("processing", False)
                break
        for scan in status.get("active_scans", []):
            if scan.get("datastore_id") == ds_id:
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


def _fetch_assigned_orgs(db: Session, ds_id: int, admin_org_ids: Optional[List[int]]) -> list[dict]:
    links = (
        db.query(OrganizationDataStore)
        .join(Organisation)
        .filter(
            OrganizationDataStore.data_store_id == ds_id,
            OrganizationDataStore.is_active == True,
        )
        .all()
    )
    if admin_org_ids is not None:
        links = [link for link in links if link.organisation.id in admin_org_ids]
    return [
        {"id": link.organisation.id, "name": link.organisation.name}
        for link in links
    ]


def _compute_graph_summary_for_ds(db: Session, ds_id: int) -> Optional[dict]:
    from app.models.knowledge import ProcessingTask
    from sqlalchemy import func, case
    row = (
        db.query(
            func.count().label("total"),
            func.sum(case(
                (ProcessingTask.graph_status == "pending", 1), else_=0,
            )).label("pending"),
            func.sum(case(
                (ProcessingTask.graph_status == "completed", 1), else_=0,
            )).label("completed"),
            func.sum(case(
                (ProcessingTask.graph_status == "failed", 1), else_=0,
            )).label("failed"),
        )
        .filter(ProcessingTask.data_store_id == ds_id)
        .first()
    )
    if row and (row.total or 0) > 0:
        total = int(row.total or 0)
        pending = int(row.pending or 0)
        completed = int(row.completed or 0)
        failed = int(row.failed or 0)
        status = _graph_status_from_counts(total, pending, completed, failed)
        return {
            "total": total, "pending": pending,
            "completed": completed, "failed": failed, "status": status,
        }
    return None
