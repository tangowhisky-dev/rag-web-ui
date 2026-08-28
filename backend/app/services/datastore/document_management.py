"""document_management.py — Per-document management for datastores.

Provides:
  - Folder browsing (mini file-browser backed by os.scandir + Document state)
  - Select/unselect documents for ingestion
  - Delete ingested data (MySQL + Qdrant + Neo4j) when unselecting

The file on disk is NEVER deleted — only ingested data is removed.
Unselecting a document that has chunks deletes its Qdrant points,
DocumentChunk rows, ProcessingTask rows, Neo4j Chunk nodes, and
DataStoreFileManifest entry, then sets is_selected=false.
"""
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import func, text
from sqlalchemy.orm import Session
from qdrant_client.models import PointIdsList
from qdrant_client.http.exceptions import UnexpectedResponse

from app.db.session import SessionLocal
from app.models.datastore import DataStore, DataStoreFileManifest
from app.models.knowledge import Document, DocumentChunk, ProcessingTask
from app.services.ingestion.document_converter import SUPPORTED_EXTENSIONS, CONTENT_TYPE_MAP
from app.services.ingestion import _chunk_id_to_point_id
from app.services.infrastructure import get_qdrant_client

logger = logging.getLogger(__name__)

# Batch size for bulk deletions — keeps DB transactions and Qdrant
# requests bounded.
_BATCH_SIZE = 50


# ---------------------------------------------------------------------------
# File-system helpers
# ---------------------------------------------------------------------------

def _should_skip_name(name: str) -> bool:
    """Return True for hidden/temp/lock files that should not be shown."""
    if name.startswith(".") or name.startswith("~$") or name.startswith(".~"):
        return True
    ext = os.path.splitext(name)[1].lower()
    if ext in (".tmp", ".swp", ".swo", ".bak", ".lock"):
        return True
    return False


def _relative_path(absolute_path: str, root: str) -> str:
    """Return *absolute_path* relative to *root*, using forward slashes."""
    rel = os.path.relpath(absolute_path, root)
    return rel.replace(os.sep, "/")


# ---------------------------------------------------------------------------
# Folder browsing
# ---------------------------------------------------------------------------

def get_folder_contents(
    db: Session,
    datastore_id: int,
    relative_path: str = "",
    sort: str = "name",
    page: int = 0,
    page_size: int = 100,
    search: str = "",
    include_unsupported: bool = False,
) -> dict[str, Any]:
    """List immediate children of a folder within a datastore.

    Returns folders and files with their ingestion state.  Folders get
    aggregate counts from the manifest; files get Document state.
    """
    ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
    if not ds:
        return {"error": "datastore_not_found"}

    root = ds.folder_path
    target = os.path.normpath(os.path.join(root, relative_path)) if relative_path else root

    # Security: ensure target is within root
    if not target.startswith(root):
        return {"error": "path_outside_datastore"}

    if not os.path.isdir(target):
        return {"error": "folder_not_found"}

    # ── Scan directory ────────────────────────────────────────────────
    entries = []
    try:
        for entry in os.scandir(target):
            if _should_skip_name(entry.name):
                continue
            entries.append(entry)
    except OSError as e:
        logger.warning("[BROWSE] scandir failed for %s: %s", target, e)
        return {"error": "scan_failed", "detail": str(e)}

    # ── Split into folders and files ──────────────────────────────────
    folders = []
    files = []
    for entry in entries:
        try:
            is_dir = entry.is_dir(follow_symlinks=False)
        except OSError:
            continue
        if is_dir:
            folders.append(entry)
        else:
            ext = os.path.splitext(entry.name)[1].lower()
            if not include_unsupported and ext not in SUPPORTED_EXTENSIONS:
                continue
            # Check scan pattern
            if not _matches_scan_pattern(entry.name, ds.scan_pattern):
                continue
            files.append(entry)

    # ── Apply search filter ───────────────────────────────────────────
    if search:
        search_lower = search.lower()
        folders = [e for e in folders if search_lower in e.name.lower()]
        files = [e for e in files if search_lower in e.name.lower()]

    # ── Batch-query Document state for all files in this folder ───────
    file_abs_paths = [os.path.join(target, e.name) for e in files]
    doc_map: dict[str, Document] = {}
    if file_abs_paths:
        docs = (
            db.query(Document)
            .filter(
                Document.data_store_id == datastore_id,
                Document.file_path.in_(file_abs_paths),
            )
            .all()
        )
        doc_map = {d.file_path: d for d in docs}

    # ── Batch-query chunk counts per document ─────────────────────────
    doc_ids = [d.id for d in doc_map.values()]
    chunk_counts: dict[int, int] = {}
    task_statuses: dict[int, tuple[str, Optional[str]]] = {}  # doc_id -> (status, error)
    if doc_ids:
        # Chunk counts
        rows = (
            db.query(DocumentChunk.document_id, func.count(DocumentChunk.id))
            .filter(DocumentChunk.document_id.in_(doc_ids))
            .group_by(DocumentChunk.document_id)
            .all()
        )
        chunk_counts = {r[0]: r[1] for r in rows}

        # Latest task status + graph_status per document
        rows = (
            db.query(
                ProcessingTask.document_id,
                ProcessingTask.status,
                ProcessingTask.error_message,
                ProcessingTask.graph_status,
            )
            .filter(ProcessingTask.document_id.in_(doc_ids))
            .order_by(ProcessingTask.id.desc())
            .all()
        )
        for r in rows:
            if r[0] not in task_statuses:  # first = latest due to desc order
                task_statuses[r[0]] = (r[1], r[2], r[3])

    # ── Build file items ──────────────────────────────────────────────
    file_items = []
    for entry in files:
        abs_path = os.path.join(target, entry.name)
        doc = doc_map.get(abs_path)
        ext = os.path.splitext(entry.name)[1].lower()
        try:
            st = entry.stat(follow_symlinks=False)
            size = st.st_size
            mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
        except OSError:
            size = 0
            mtime = None

        if doc:
            task_info = task_statuses.get(doc.id, ("not_ingested", None, None))
            status, error_msg, graph_status = task_info
            file_items.append({
                "type": "file",
                "name": entry.name,
                "path": _relative_path(abs_path, root),
                "absolute_path": abs_path,
                "size": size,
                "content_type": CONTENT_TYPE_MAP.get(ext, "application/octet-stream"),
                "modified_at": mtime,
                "document_id": doc.id,
                "is_selected": doc.is_selected,
                "status": status or "not_ingested",
                "chunk_count": chunk_counts.get(doc.id, 0),
                "graph_status": graph_status,
                "conversion_status": doc.conversion_status,
                "title": doc.title,
                "error_message": error_msg,
            })
        else:
            file_items.append({
                "type": "file",
                "name": entry.name,
                "path": _relative_path(abs_path, root),
                "absolute_path": abs_path,
                "size": size,
                "content_type": CONTENT_TYPE_MAP.get(ext, "application/octet-stream"),
                "modified_at": mtime,
                "document_id": None,
                "is_selected": False,
                "status": "not_ingested",
                "chunk_count": 0,
                "graph_status": None,
                "conversion_status": None,
                "title": None,
                "error_message": None,
            })

    # ── Build folder items with aggregate counts ──────────────────────
    folder_items = []
    for entry in folders:
        folder_abs = os.path.join(target, entry.name)
        folder_rel = _relative_path(folder_abs, root)
        # Count files in manifest under this folder prefix
        prefix = folder_abs + os.sep
        manifest_count = (
            db.query(func.count(DataStoreFileManifest.id))
            .filter(
                DataStoreFileManifest.datastore_id == datastore_id,
                DataStoreFileManifest.file_path.like(prefix + "%"),
            )
            .scalar()
        ) or 0

        # Count ingested documents under this folder
        ingested_count = (
            db.query(func.count(Document.id))
            .filter(
                Document.data_store_id == datastore_id,
                Document.file_path.like(prefix + "%"),
                Document.chunks.any(),
            )
            .scalar()
        ) or 0

        # Count selected documents
        selected_count = (
            db.query(func.count(Document.id))
            .filter(
                Document.data_store_id == datastore_id,
                Document.file_path.like(prefix + "%"),
                Document.is_selected == True,
            )
            .scalar()
        ) or 0

        folder_items.append({
            "type": "folder",
            "name": entry.name,
            "path": folder_rel,
            "file_count": manifest_count,
            "ingested_count": ingested_count,
            "selected_count": selected_count,
        })

    # ── Sort ──────────────────────────────────────────────────────────
    reverse = False
    sort_field = sort.lstrip("-")
    if sort.startswith("-"):
        reverse = True

    def _folder_sort_key(f):
        return f["name"].lower()

    def _file_sort_key(f):
        if sort_field == "size":
            return f["size"]
        if sort_field == "modified":
            return f["modified_at"] or ""
        if sort_field == "status":
            return f["status"]
        return f["name"].lower()

    folder_items.sort(key=_folder_sort_key, reverse=reverse)
    file_items.sort(key=_file_sort_key, reverse=reverse)

    # ── Pagination (applied to files only; folders always shown) ──────
    total_files = len(file_items)
    total_folders = len(folder_items)
    start = page * page_size
    end = start + page_size
    paged_files = file_items[start:end]

    items = folder_items + paged_files

    # ── Breadcrumbs ───────────────────────────────────────────────────
    breadcrumbs = [{"name": "Root", "path": ""}]
    if relative_path:
        parts = relative_path.strip("/").split("/")
        cumulative = ""
        for part in parts:
            cumulative = cumulative + "/" + part if cumulative else part
            breadcrumbs.append({"name": part, "path": cumulative})

    # ── Datastore-level stats ─────────────────────────────────────────
    stats = _get_datastore_stats(db, datastore_id)

    return {
        "datastore_id": datastore_id,
        "datastore_name": ds.name,
        "folder_path": root,
        "current_path": relative_path,
        "breadcrumbs": breadcrumbs,
        "items": items,
        "total": total_files + total_folders,
        "total_files": total_files,
        "total_folders": total_folders,
        "page": page,
        "page_size": page_size,
        "stats": stats,
    }


def list_folder_files(
    db: Session,
    datastore_id: int,
    relative_path: str = "",
) -> dict[str, Any]:
    """List all files recursively under a folder, with selection state.

    Used by the frontend when a folder checkbox is toggled — it needs
    the full set of file paths under the folder to update the dirty map.
    """
    ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
    if not ds:
        return {"error": "datastore_not_found"}

    root = ds.folder_path
    target = os.path.normpath(os.path.join(root, relative_path)) if relative_path else root
    if not target.startswith(root):
        return {"error": "path_outside_datastore"}
    if not os.path.isdir(target):
        return {"error": "folder_not_found"}

    prefix = target + os.sep

    # Query manifest for all files under this folder
    rows = (
        db.query(DataStoreFileManifest.file_path)
        .filter(
            DataStoreFileManifest.datastore_id == datastore_id,
            DataStoreFileManifest.file_path.like(prefix + "%"),
        )
        .all()
    )
    manifest_paths = {r[0] for r in rows}

    # Also walk the filesystem in case manifest is incomplete (new files)
    fs_paths: set[str] = set()
    for dirpath, _dirs, filenames in os.walk(target):
        for fname in filenames:
            if _should_skip_name(fname):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            if not _matches_scan_pattern(fname, ds.scan_pattern):
                continue
            fs_paths.add(os.path.join(dirpath, fname))

    all_paths = manifest_paths | fs_paths

    # Get selection state from Documents
    doc_map: dict[str, bool] = {}
    if all_paths:
        docs = (
            db.query(Document.file_path, Document.is_selected)
            .filter(
                Document.data_store_id == datastore_id,
                Document.file_path.in_(list(all_paths)),
            )
            .all()
        )
        doc_map = {d[0]: d[1] for d in docs}

    files = []
    for fp in sorted(all_paths):
        files.append({
            "path": _relative_path(fp, root),
            "absolute_path": fp,
            "is_selected": doc_map.get(fp, False),
        })

    return {"files": files}


def expand_folder_paths(
    db: Session,
    datastore_id: int,
    paths: list[str],
) -> list[str]:
    """Expand any folder paths to their contained file paths.

    Given a mix of file and folder absolute paths, returns a flat list
    of file paths.  Folder paths are expanded recursively using the
    manifest + filesystem walk.
    """
    result: list[str] = []
    ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
    if not ds:
        return paths  # let downstream handle the error

    root = ds.folder_path
    for p in paths:
        # Normalise path
        norm = os.path.normpath(p)
        if os.path.isdir(norm):
            prefix = norm + os.sep
            # Manifest paths
            rows = (
                db.query(DataStoreFileManifest.file_path)
                .filter(
                    DataStoreFileManifest.datastore_id == datastore_id,
                    DataStoreFileManifest.file_path.like(prefix + "%"),
                )
                .all()
            )
            for r in rows:
                result.append(r[0])
            # Filesystem walk for new files not yet in manifest
            for dirpath, _dirs, filenames in os.walk(norm):
                for fname in filenames:
                    if _should_skip_name(fname):
                        continue
                    ext = os.path.splitext(fname)[1].lower()
                    if ext not in SUPPORTED_EXTENSIONS:
                        continue
                    if not _matches_scan_pattern(fname, ds.scan_pattern):
                        continue
                    fp = os.path.join(dirpath, fname)
                    if fp not in result:
                        result.append(fp)
        else:
            result.append(p)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for p in result:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _matches_scan_pattern(filename: str, pattern: str) -> bool:
    """Check if filename matches the datastore's scan pattern."""
    import fnmatch
    if not pattern or pattern == "*":
        return True
    # Pattern can be comma-separated
    patterns = [p.strip() for p in pattern.split(",")]
    for p in patterns:
        if fnmatch.fnmatch(filename, p):
            return True
    return False


def _get_datastore_stats(db: Session, datastore_id: int) -> dict:
    """Compute aggregate stats for a datastore."""
    total = db.query(func.count(Document.id)).filter(
        Document.data_store_id == datastore_id
    ).scalar() or 0

    selected = db.query(func.count(Document.id)).filter(
        Document.data_store_id == datastore_id,
        Document.is_selected == True,
    ).scalar() or 0

    unselected = total - selected

    # Count documents with chunks (ingested)
    ingested = db.query(func.count(Document.id)).filter(
        Document.data_store_id == datastore_id,
        Document.chunks.any(),
    ).scalar() or 0

    # Count by task status
    status_counts = {}
    rows = (
        db.query(ProcessingTask.status, func.count(ProcessingTask.id))
        .join(Document, ProcessingTask.document_id == Document.id)
        .filter(Document.data_store_id == datastore_id)
        .group_by(ProcessingTask.status)
        .all()
    )
    for r in rows:
        status_counts[r[0]] = r[1]

    return {
        "total_documents": total,
        "selected": selected,
        "unselected": unselected,
        "ingested": ingested,
        "completed": status_counts.get("completed", 0),
        "failed": status_counts.get("failed", 0),
        "processing": status_counts.get("processing", 0),
        "pending": status_counts.get("pending", 0),
    }


# ---------------------------------------------------------------------------
# Delete ingested data for a single document
# ---------------------------------------------------------------------------

def delete_document_data(db: Session, document_id: int, datastore_id: int) -> dict:
    """Delete all ingested data for a document. File on disk is untouched.

    Deletes: Qdrant points, DocumentChunk rows, ProcessingTask rows,
    DataStoreFileManifest entry, Neo4j Chunk + orphaned Entity nodes.
    Sets is_selected=false on the Document record (keeps the record so
    next scan knows to skip it).

    Returns a summary dict with counts of what was deleted.
    """
    doc = db.query(Document).filter(
        Document.id == document_id,
        Document.data_store_id == datastore_id,
    ).first()
    if not doc:
        return {"deleted": False, "reason": "document_not_found"}

    # Capture chunk IDs before DB deletion
    chunk_ids = [
        cid[0] for cid in db.query(DocumentChunk.id).filter(
            DocumentChunk.document_id == doc.id
        ).all()
    ]
    doc_id = doc.id
    file_path = doc.file_path

    # DB cleanup
    db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).delete()
    db.query(ProcessingTask).filter(ProcessingTask.document_id == doc.id).delete()

    # Delete manifest entry so file is re-discovered on next scan if re-selected
    db.query(DataStoreFileManifest).filter(
        DataStoreFileManifest.datastore_id == datastore_id,
        DataStoreFileManifest.file_path == file_path,
    ).delete(synchronize_session=False)

    # Set is_selected=false (keep the Document record)
    doc.is_selected = False
    db.commit()

    # Qdrant cleanup (after DB commit)
    qdrant_deleted = 0
    if chunk_ids:
        try:
            point_ids = [_chunk_id_to_point_id(cid) for cid in chunk_ids]
            get_qdrant_client().delete(
                collection_name=f"ds_{datastore_id}",
                points_selector=PointIdsList(points=point_ids),
            )
            qdrant_deleted = len(point_ids)
        except UnexpectedResponse as e:
            if "404" not in str(e):
                logger.warning("[DOC_MGMT] Qdrant delete failed for doc %d: %s", doc_id, e)
        except Exception as e:
            logger.warning("[DOC_MGMT] Qdrant delete failed for doc %d: %s", doc_id, e)

    # Neo4j cleanup
    graph_deleted = 0
    try:
        from app.services.graph import delete_graph_for_document
        delete_graph_for_document(kb_id=None, document_id=doc_id, data_store_id=datastore_id)
        graph_deleted = 1
    except Exception as e:
        logger.warning("[DOC_MGMT] Neo4j cleanup failed for doc %d: %s", doc_id, e)

    logger.info(
        "[DOC_MGMT] document_data_deleted doc_id=%d chunks=%d qdrant=%d graph=%d",
        doc_id, len(chunk_ids), qdrant_deleted, graph_deleted,
    )

    return {
        "deleted": True,
        "document_id": doc_id,
        "chunks_deleted": len(chunk_ids),
        "qdrant_points_deleted": qdrant_deleted,
        "graph_nodes_deleted": graph_deleted,
    }


# ---------------------------------------------------------------------------
# Batch unselect (delete data) and select
# ---------------------------------------------------------------------------

def unselect_documents(
    datastore_id: int,
    file_paths: list[str],
) -> dict:
    """Unselect documents and delete their ingested data.

    Args:
        datastore_id: The datastore ID.
        file_paths: Absolute file paths to unselect.

    Returns:
        Summary dict with counts.
    """
    total_deleted = 0
    total_chunks = 0
    total_qdrant = 0
    total_graph = 0
    errors = []

    for i in range(0, len(file_paths), _BATCH_SIZE):
        batch = file_paths[i:i + _BATCH_SIZE]
        db = SessionLocal()
        try:
            for fp in batch:
                doc = db.query(Document).filter(
                    Document.file_path == fp,
                    Document.data_store_id == datastore_id,
                ).first()
                if not doc:
                    continue
                if not doc.is_selected:
                    continue  # Already unselected, no-op

                result = delete_document_data(db, doc.id, datastore_id)
                if result.get("deleted"):
                    total_deleted += 1
                    total_chunks += result.get("chunks_deleted", 0)
                    total_qdrant += result.get("qdrant_points_deleted", 0)
                    total_graph += result.get("graph_nodes_deleted", 0)
                else:
                    errors.append({"file_path": fp, "reason": result.get("reason")})
        except Exception as e:
            logger.error("[DOC_MGMT] unselect batch error: %s", e)
            errors.append({"batch": i, "error": str(e)})
        finally:
            db.close()

    return {
        "unselected": total_deleted,
        "deleted_chunks": total_chunks,
        "deleted_qdrant_points": total_qdrant,
        "deleted_graph_nodes": total_graph,
        "errors": errors,
    }


def select_documents(
    datastore_id: int,
    file_paths: list[str],
) -> dict:
    """Select documents for ingestion.

    For files with existing Document records: set is_selected=true.
    For files without a Document record: create one with is_selected=true
    (will be ingested on next scan).

    Returns summary dict.
    """
    total_selected = 0
    total_created = 0
    errors = []

    db = SessionLocal()
    try:
        ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
        if not ds:
            return {"error": "datastore_not_found"}

        for fp in file_paths:
            try:
                doc = db.query(Document).filter(
                    Document.file_path == fp,
                    Document.data_store_id == datastore_id,
                ).first()

                if doc:
                    if not doc.is_selected:
                        doc.is_selected = True
                        total_selected += 1
                else:
                    # Create a minimal Document record
                    fname = os.path.basename(fp)
                    ext = os.path.splitext(fname)[1].lower()
                    try:
                        st = os.stat(fp)
                        size = st.st_size
                    except OSError:
                        size = 0

                    doc = Document(
                        knowledge_base_id=None,
                        data_store_id=datastore_id,
                        file_path=fp,
                        file_name=fname,
                        file_size=size,
                        content_type=CONTENT_TYPE_MAP.get(ext, "application/octet-stream"),
                        file_hash=None,
                        is_selected=True,
                    )
                    db.add(doc)
                    total_created += 1
            except Exception as e:
                logger.error("[DOC_MGMT] select error for %s: %s", fp, e)
                errors.append({"file_path": fp, "error": str(e)})

        db.commit()
    finally:
        db.close()

    return {
        "selected": total_selected,
        "created": total_created,
        "errors": errors,
    }


def select_folder(
    datastore_id: int,
    folder_path: str,
    selected: bool,
    recursive: bool = True,
) -> dict:
    """Select or unselect all files in a folder.

    Args:
        datastore_id: The datastore ID.
        folder_path: Absolute path to the folder.
        selected: True to select, False to unselect (deletes data).
        recursive: If True, include all subdirectories.

    Returns summary dict.
    """
    prefix = folder_path
    if not prefix.endswith(os.sep):
        prefix += os.sep

    db = SessionLocal()
    try:
        # Find all Document file_paths under this folder
        query = db.query(Document.file_path).filter(
            Document.data_store_id == datastore_id,
            Document.file_path.like(prefix + "%"),
        )
        if not recursive:
            # Only immediate children — no further path separators
            query = query.filter(
                ~Document.file_path.like(prefix + "%" + os.sep + "%")
            )

        paths = [r[0] for r in query.all()]
    finally:
        db.close()

    if selected:
        return select_documents(datastore_id, paths)
    else:
        return unselect_documents(datastore_id, paths)
