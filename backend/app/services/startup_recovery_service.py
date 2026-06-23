"""Startup recovery service — background ingestion on app start.

Runs discovery on every active DataStore, queues new/modified files for
ingestion, and cleans up orphaned records for deleted files.  All work
is non-blocking: the FastAPI app is fully functional before recovery
completes.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from app.db.session import SessionLocal
from app.models.datastore import DataStore, DataStoreFileManifest
from app.models.knowledge import Document, DocumentChunk, ProcessingTask

logger = logging.getLogger(__name__)


class StartupRecoveryService:
    """Orchestrates background recovery for all active DataStores.

    On app start, one background thread is spawned per active datastore.
    Each thread runs a discovery pipeline that queues new/modified files
    for ingestion and deletes orphaned records for files that no longer
    exist on disk.
    """

    def __init__(self) -> None:
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=4)
        self._running: bool = True
        self._scan_id_counter: int = 0
        self._scan_id_lock = threading.Lock()
        # scan_id -> status dict
        self._active_scans: Dict[int, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch one background recovery thread per active DataStore.

        Called from ``startup_event()`` after the database is ready.
        If the migration hasn't run yet (``tables_not_found``), logs a
        warning and skips gracefully.
        """
        logger.info("[RECOVERY] StartupRecoveryService.start() invoked")
        db: Session = SessionLocal()
        try:
            active = db.query(DataStore).filter(DataStore.is_active == True).all()  # noqa: E712
        except Exception as e:
            logger.warning("[RECOVERY] Could not query DataStores (migration may not be applied): %s", e)
            return
        finally:
            db.close()

        if not active:
            logger.info("[RECOVERY] No active DataStores found — skipping recovery")
            return

        for ds in active:
            logger.info("[RECOVERY] recovery_start datastore_id=%s name=%s", ds.id, ds.name)
            scan_id = self._next_scan_id()
            self._active_scans[scan_id] = {
                "datastore_id": ds.id,
                "datastore_name": ds.name,
                "status": "running",
                "scan_id": scan_id,
                "total_files": 0,
                "processed_files": 0,
                "new_files": 0,
                "modified_files": 0,
                "deleted_files": 0,
                "error_message": None,
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            self.executor.submit(self._discovery_pipeline_worker, ds.id, scan_id)

    def stop(self) -> None:
        """Signal shutdown and stop accepting new work."""
        logger.info("[RECOVERY] stop() invoked — setting _running=False")
        self._running = False
        self.executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def _get_status(self, datastore_id: int) -> Dict[str, Any]:
        """Return the current recovery status for a DataStore.

        Returns ``{"status": "idle"}`` when no scan is in progress.
        """
        return self.get_status(datastore_id)

    def get_status(self, datastore_id: int) -> Dict[str, Any]:
        """Return the current recovery status for a DataStore.

        Returns ``{"status": "idle"}`` when no scan is in progress.
        """
        for scan in self._active_scans.values():
            if scan.get("datastore_id") == datastore_id:
                return scan
        return {"status": "idle"}

    def get_all_status(self) -> List[Dict[str, Any]]:
        """Return recovery status for all active datastores.

        Returns a list of status dicts (sorted by scan_id) plus
        idle entries for datastores that have no scan in progress.
        """
        results: List[Dict[str, Any]] = []
        for scan in sorted(self._active_scans.values(), key=lambda s: s.get("scan_id", 0)):
            results.append(dict(scan))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_scan_id(self) -> int:
        with self._scan_id_lock:
            self._scan_id_counter += 1
            return self._scan_id_counter

    def _discovery_pipeline_worker(self, datastore_id: int, scan_id: int) -> None:
        """Run the discovery pipeline in a background thread."""
        try:
            from app.services.discovery_engine import discover_datastore  # noqa: T100

            db: Session = SessionLocal()
            try:
                result: DiscoveryResult = discover_datastore(datastore_id, db)
            except Exception as e:
                logger.error("[RECOVERY] discovery_pipeline error for datastore_id=%s scan_id=%s: %s", datastore_id, scan_id, e, exc_info=True)
                self._active_scans[scan_id]["status"] = "error"
                self._active_scans[scan_id]["error_message"] = str(e)
                self._active_scans[scan_id]["processed_files"] = 0
                return
            finally:
                db.close()

            logger.info(
                "[RECOVERY] discovery_complete datastore_id=%s scan_id=%s new=%d modified=%d deleted=%d total=%d elapsed_ms=%.1f",
                datastore_id, scan_id,
                len(result.new_files), len(result.modified_files),
                len(result.deleted_files), result.total_files_discovered,
                result.elapsed_ms,
            )

            # Update status with discovery counts
            self._active_scans[scan_id]["total_files"] = result.total_files_discovered
            self._active_scans[scan_id]["new_files"] = len(result.new_files)
            self._active_scans[scan_id]["modified_files"] = len(result.modified_files)
            self._active_scans[scan_id]["deleted_files"] = len(result.deleted_files)

            # Process new files
            for fmeta in result.new_files:
                if not self._running:
                    break
                self.process_new_file(fmeta["file_path"], datastore_id)

            # Process modified files
            for fmeta in result.modified_files:
                if not self._running:
                    break
                self.process_new_file(fmeta["file_path"], datastore_id)

            # Delete orphaned records for deleted files
            for fmeta in result.deleted_files:
                if not self._running:
                    break
                self._handle_deletion_records(fmeta["file_path"], datastore_id)

            # Mark complete
            self._active_scans[scan_id]["status"] = "complete"
            self._active_scans[scan_id]["processed_files"] = (
                len(result.new_files) + len(result.modified_files) + len(result.deleted_files)
            )
            logger.info(
                "[RECOVERY] recovery_complete datastore_id=%s scan_id=%s total_files=%d processed=%d",
                datastore_id, scan_id,
                len(result.new_files) + len(result.modified_files),
                len(result.new_files) + len(result.modified_files) + len(result.deleted_files),
            )

        except Exception as e:
            logger.error("[RECOVERY] recovery_error datastore_id=%s scan_id=%s: %s", datastore_id, scan_id, e, exc_info=True)
            self._active_scans[scan_id]["status"] = "error"
            self._active_scans[scan_id]["error_message"] = str(e)

    # ------------------------------------------------------------------
    # New / Modified file ingestion
    # ------------------------------------------------------------------

    def process_new_file(self, file_path: str, datastore_id: int) -> None:
        """Queue a file for ingestion via the background processor.

        Creates (or reuses) a Document record, creates a ProcessingTask,
        and submits the file to ``process_document_background``.
        """
        file_name = os.path.basename(file_path)
        db: Session = SessionLocal()
        try:
            # Check if document already exists
            doc = (
                db.query(Document)
                .filter(Document.file_path == file_path, Document.data_store_id == datastore_id)
                .first()
            )

            # Compute hash and size
            try:
                file_hash = _sha256(file_path)
                file_size = os.path.getsize(file_path)
            except OSError as e:
                logger.warning("[RECOVERY] Cannot read file during ingestion queue: %s", e)
                return

            if doc:
                # File exists — update metadata and re-queue for ingestion
                doc.file_hash = file_hash
                doc.file_size = file_size
                doc.updated_at = datetime.now(timezone.utc)
                logger.info(
                    "[RECOVERY] ingestion_queued datastore_id=%s scan_id=%d file_path=%s doc_id=%s (modified)",
                    datastore_id, self._active_scans.get(0, {}).get("scan_id"), file_path, doc.id,
                )
            else:
                # Create new Document record
                doc = Document(
                    file_path=file_path,
                    file_name=file_name,
                    file_size=file_size,
                    content_type=_guess_content_type(file_name),
                    file_hash=file_hash,
                    data_store_id=datastore_id,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )
                db.add(doc)
                db.flush()
                logger.info(
                    "[RECOVERY] ingestion_queued datastore_id=%s file_path=%s doc_id=%s (new)",
                    datastore_id, file_path, doc.id,
                )

            # Create ProcessingTask
            task = ProcessingTask(
                document_id=doc.id,
                data_store_id=datastore_id,
                status="pending",
                progress=0,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            db.add(task)
            db.commit()

            # Submit to background processor (async)
            self._submit_ingestion(file_path, file_name, datastore_id, doc, file_hash, file_size, task.id)

        except Exception as e:
            logger.warning("[RECOVERY] Failed to queue file for ingestion: %s", e, exc_info=True)
            try:
                db.rollback()
            except Exception:
                pass
        finally:
            db.close()

    def _submit_ingestion(
        self,
        file_path: str,
        file_name: str,
        datastore_id: int,
        doc: Document,
        file_hash: str,
        file_size: int,
        task_id: int,
    ) -> None:
        """Submit a file to the async background processor."""
        loop = asyncio.new_event_loop()
        try:
            from app.services.document_processor import process_document_background  # noqa: T100

            coro = process_document_background(
                temp_path=file_path,
                file_name=file_name,
                data_store_id=datastore_id,
                file_path=file_path,
                file_hash=file_hash,
                file_size=file_size,
                document_id=doc.id,
                task_id=task_id,
                kb_id=None,  # DataStore file, not KB
            )
            loop.run_until_complete(coro)
        except Exception as e:
            logger.warning("[RECOVERY] Background ingestion submit failed: %s", e)
        finally:
            loop.close()

    # ------------------------------------------------------------------
    # Deleted file cleanup
    # ------------------------------------------------------------------

    def _handle_deletion_records(self, file_path: str, datastore_id: int) -> None:
        """Delete orphaned records for a file that no longer exists on disk.

        Mirrors ``DataStoreWatcher._handle_deletion`` for DataStore files
        exactly: delete Document, DocumentChunk, ProcessingTask, Qdrant
        vectors, Neo4j graph, and manifest entry.  Silently skips if the
        Document is already gone (no orphaned records to clean).
        """
        logger.info("[RECOVERY] deletion_start datastore_id=%s file_path=%s", datastore_id, file_path)
        db: Session = SessionLocal()
        try:
            doc = (
                db.query(Document)
                .filter(Document.file_path == file_path, Document.data_store_id == datastore_id)
                .first()
            )
            if not doc:
                logger.info(
                    "[RECOVERY] deletion_skip datastore_id=%s file_path=%s reason=doc_not_found",
                    datastore_id, file_path,
                )
                return

            # Delete Qdrant vectors first (before DB, so DB rollback doesn't orphan vectors)
            try:
                from qdrant_client import models
                from app.services.document_processor import _chunk_id_to_point_id, _get_qdrant_client  # noqa: T100

                chunk_ids = [
                    cid[0] for cid in db.query(DocumentChunk.id)
                    .filter(DocumentChunk.document_id == doc.id)
                    .all()
                ]
                if chunk_ids:
                    point_ids = [_chunk_id_to_point_id(cid) for cid in chunk_ids]
                    _get_qdrant_client().delete(
                        collection_name=f"ds_{datastore_id}",
                        points_selector=models.PointIdsList(points=point_ids),
                    )
                logger.info("[RECOVERY] deletion_done datastore_id=%s doc_id=%s reason=qdrant_vectors_deleted", datastore_id, doc.id)
            except Exception as e:
                logger.warning("[RECOVERY] Qdrant delete failed for doc_id=%s: %s", doc.id, e)

            # Delete DB records
            db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).delete()
            db.query(ProcessingTask).filter(ProcessingTask.document_id == doc.id).delete()

            # Clean up Neo4j graph nodes for this document
            try:
                from app.services.graph_service import delete_graph_for_document  # noqa: T100
                delete_graph_for_document(kb_id=None, document_id=doc.id)
                logger.info("[RECOVERY] deletion_done datastore_id=%s doc_id=%s reason=neo4j_cleanup", datastore_id, doc.id)
            except Exception as e:
                logger.warning("[RECOVERY] Neo4j cleanup failed for doc_id=%s: %s", doc.id, e)

            # Delete the Document record
            db.delete(doc)
            logger.info("[RECOVERY] document_deleted datastore_id=%s doc_id=%s file_path=%s", datastore_id, doc.id, file_path)

            # Delete DataStoreFileManifest entry
            manifest = (
                db.query(DataStoreFileManifest)
                .filter(DataStoreFileManifest.datastore_id == datastore_id, DataStoreFileManifest.file_path == file_path)
                .first()
            )
            if manifest:
                db.delete(manifest)
                logger.info("[RECOVERY] manifest_deleted datastore_id=%s file_path=%s", datastore_id, file_path)

            db.commit()

        except Exception as e:
            logger.warning("[RECOVERY] Deletion failed for datastore_id=%s file_path=%s: %s", datastore_id, file_path, e, exc_info=True)
            try:
                db.rollback()
            except Exception:
                pass
        finally:
            db.close()

    def process_deleted_file(self, file_path: str, datastore_id: int) -> None:
        """Delete all records for a file that was deleted from disk.

        Convenience wrapper that delegates to ``_handle_deletion_records``.
        Kept for API symmetry with ``process_new_file``.
        """
        self._handle_deletion_records(file_path, datastore_id)


# ------------------------------------------------------------------
# Module-level utilities
# ------------------------------------------------------------------

def _sha256(file_path: str) -> str:
    """Compute the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    except OSError:
        h = hashlib.sha256(b"")  # file may be unreadable during recovery
    return h.hexdigest()


def _guess_content_type(file_name: str) -> str:
    """Guess the MIME content type from the file extension."""
    import mimetypes
    ct, _ = mimetypes.guess_type(file_name)
    return ct or "application/octet-stream"


# ------------------------------------------------------------------
# Type imports for type hints
# ------------------------------------------------------------------
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — imported for type hints only
    from sqlalchemy.orm import Session
    from app.services.discovery_engine import DiscoveryResult  # noqa: F401
