"""DataStore Watcher Service — per-datastore file watching with batch processing.

Architecture:
- Single ``PollingObserver`` instance for all datastores (efficient resource usage)
- One ``DatastoreFileEventHandler`` per datastore (per-folder event handler)
- Each handler queues file events and processes them in batches at a configurable interval
- Periodic DB sync (every 5 min) to add/remove watchers based on datastore configuration

File path convention inside a watched directory:
    kb_{kb_id}/{file_name}

The service parses the path to determine which knowledge base owns the file,
then routes it into the existing ``process_document_background()`` pipeline.

Progress tracking:
- Each scan is assigned an integer scan_id
- The scan_id is stored on the ProcessingTask so it can be matched to a running scan
- When a scan starts, the datastore is updated with scan_id and progress=0
- As files are processed, progress is updated
- When the scan ends, the scan_id is cleared

Cancellation:
- A running scan can be stopped by setting scan_id=None on the datastore
- The scan checks this between files and stops early if cancelled
"""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import logging
import os
import re
import threading
import time as time_module
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session
from qdrant_client.models import PointIdsList

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.datastore import DataStore, OrganizationDataStore
from app.models.knowledge import Document, DocumentUpload, ProcessingTask, DocumentChunk
from app.models.knowledge import KnowledgeBase
from app.services.document_processor import (
    SUPPORTED_EXTENSIONS,
    process_document_background,
    _get_qdrant_client,
    _chunk_id_to_point_id,
)

logger = logging.getLogger(__name__)


class _Debouncer:
    """Coalesces rapid repeated events for the same path into a single handling."""

    def __init__(self, delay: float = 1.0) -> None:
        self._delay = delay
        self._lock = threading.Lock()
        self._last_event: Dict[str, float] = {}
        self._last_type: Dict[str, str] = {}

    def touch(self, path: str, event_type: str) -> Optional[str]:
        """Record an event. Returns the coalesced event_type if this path should
        be processed now, or None if another event is expected soon."""
        now = time_module.monotonic()
        with self._lock:
            prev_time = self._last_event.get(path)
            prev_type = self._last_type.get(path)
            self._last_event[path] = now
            self._last_type[path] = event_type
            if prev_time is not None and (now - prev_time) < self._delay:
                return None
            return event_type


class DatastoreFileEventHandler:
    """Per-datastore event handler with custom configuration.

    Features:
    - Debouncing per-file (prevents duplicate events from editors)
    - Batch processing with configurable intervals
    - org_id and datastore_id tracking for each event
    """

    def __init__(
        self,
        datastore_id: int,
        org_id: int,
        callback,
        debounce_ms: int = 1000,
        min_interval_seconds: int = 300,
        debouncer: Optional[_Debouncer] = None,
    ) -> None:
        self.datastore_id = datastore_id
        self.org_id = org_id
        self.callback = callback
        self.debounce_ms = debounce_ms / 1000.0
        self.min_interval_seconds = min_interval_seconds
        self.debouncer = debouncer
        self.last_call: Dict[str, float] = {}
        self.last_processing_time: float = 0
        self.pending_changes: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def _should_process(self, src_path: str) -> bool:
        now = time_module.time()
        with self._lock:
            if now - self.last_call.get(src_path, 0) < self.debounce_ms:
                return False
            self.last_call[src_path] = now
            return True

        def _queue_change(self, event, event_type: str) -> None:
            with self._lock:
                self.pending_changes.append(
                    {
                        "datastore_id": self.datastore_id,
                        "org_id": self.org_id,
                        "path": event.src_path,
                        "event_type": event_type,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
            pending_count = len(self.pending_changes)
            time_since_last = time_module.time() - self.last_processing_time
            if time_since_last >= self.min_interval_seconds:
                logger.info(
                    "[WATCHER] batch_trigger datastore_id=%d path=%s reason=immediate (time_since_last=%.1fs >= interval=%.0fs, pending=%d)",
                    self.datastore_id, event.src_path, time_since_last, self.min_interval_seconds, pending_count,
                )
                self._process_batch()
            else:
                logger.info(
                    "[WATCHER] batch_deferred datastore_id=%d path=%s reason=within_interval (time_since_last=%.1fs < interval=%.0fs, pending=%d, wait=%.1fs)",
                    self.datastore_id, event.src_path, time_since_last, self.min_interval_seconds, pending_count, self.min_interval_seconds - time_since_last,
                )

    def _handle_deletion(
        self,
        event_path: str,
        org_id: Optional[int],
        datastore_id: Optional[int] = None,
    ) -> None:
        """Handle file deletion - remove Document records and Qdrant vectors.

        For DataStore files: delete the document for this datastore and its Qdrant vectors.
        For KB files: delete from all KBs for the org and their Qdrant vectors.
        """
        logger.info(
            "[WATCHER] file_deleted path=%s org_id=%s datastore_id=%s",
            event_path,
            org_id,
            datastore_id,
        )

        db: Session = SessionLocal()
        try:
            # DataStore deletion: delete the document for this datastore
            if datastore_id is not None:
                doc = (
                    db.query(Document)
                    .filter(
                        Document.file_path == event_path,
                        Document.data_store_id == datastore_id,
                    )
                    .first()
                )
                if doc:
                    # Delete Qdrant vectors first (before DB, so DB rollback doesn't orphan vectors)
                    try:
                        chunk_ids = [
                            cid[0] for cid in db.query(DocumentChunk.id).filter(
                                DocumentChunk.document_id == doc.id
                            ).all()
                        ]
                        if chunk_ids:
                            point_ids = [_chunk_id_to_point_id(cid) for cid in chunk_ids]
                            _get_qdrant_client().delete(
                                collection_name=f"ds_{datastore_id}",
                                points_selector=PointIdsList(points=point_ids),
                            )
                    except Exception as e:
                        logger.warning(
                            "[WATCHER] Qdrant delete failed for document_id=%s: %s",
                            doc.id, e,
                        )

                    db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).delete()
                    db.query(ProcessingTask).filter(ProcessingTask.document_id == doc.id).delete()

                    # Clean up Neo4j graph nodes for this document
                    try:
                        from app.services.graph_service import delete_graph_for_document
                        delete_graph_for_document(kb_id=None, document_id=doc.id)
                        logger.info(
                            "[WATCHER] Neo4j cleanup done for document_id=%s",
                            doc.id,
                        )
                    except Exception as e:
                        logger.warning(
                            "[WATCHER] Neo4j cleanup failed for document_id=%s: %s",
                            doc.id, e,
                        )

                    db.delete(doc)
                    logger.info(
                        "[WATCHER] document_deleted path=%s datastore_id=%s doc_id=%s",
                        event_path,
                        datastore_id,
                        doc.id,
                    )
                return

            # KB deletion: delete from all KBs for the org
            if org_id is not None:
                kb_list = (
                    db.query(KnowledgeBase)
                    .filter(KnowledgeBase.org_id == org_id)
                    .values("id")
                )
                kb_list = [kb[0] for kb in kb_list]

                for kb_id in kb_list:
                    doc = (
                        db.query(Document)
                        .filter(
                            Document.file_path == event_path,
                            Document.knowledge_base_id == kb_id,
                        )
                        .first()
                    )

                    if doc:
                        # Delete Qdrant vectors first
                        try:
                            chunk_ids = [
                                cid[0] for cid in db.query(DocumentChunk.id).filter(
                                    DocumentChunk.document_id == doc.id
                                ).all()
                            ]
                            if chunk_ids:
                                point_ids = [_chunk_id_to_point_id(cid) for cid in chunk_ids]
                                _get_qdrant_client().delete(
                                    collection_name=f"kb_{kb_id}",
                                    points_selector=PointIdsList(points=point_ids),
                                )
                        except Exception as e:
                            logger.warning(
                                "[WATCHER] Qdrant delete failed for document_id=%s: %s",
                                doc.id, e,
                            )

                        db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).delete()
                        db.query(ProcessingTask).filter(ProcessingTask.document_id == doc.id).delete()

                        # Clean up Neo4j graph nodes for this document
                        try:
                            from app.services.graph_service import delete_graph_for_document
                            delete_graph_for_document(kb_id=kb_id, document_id=doc.id)
                            logger.info(
                                "[WATCHER] Neo4j cleanup done for kb_id=%s doc_id=%s",
                                kb_id, doc.id,
                            )
                        except Exception as e:
                            logger.warning(
                                "[WATCHER] Neo4j cleanup failed for kb_id=%s doc_id=%s: %s",
                                kb_id, doc.id, e,
                            )

                        db.delete(doc)
                        logger.info(
                            "[WATCHER] document_deleted path=%s kb_id=%s doc_id=%s",
                            event_path,
                            kb_id,
                            doc.id,
                        )

            db.commit()
        finally:
            db.close()

    def _process_batch(self) -> None:
        with self._lock:
            if not self.pending_changes:
                return
            changes_to_process = self.pending_changes.copy()
            self.pending_changes.clear()
            self.last_processing_time = time_module.time()

        logger.info(
            "[WATCHER] batch_process datastore_id=%d pending=%d",
            self.datastore_id, len(changes_to_process),
        )
        try:
            self.callback(self.datastore_id, self.org_id, changes_to_process)
        except Exception as e:
            logger.error(
                "[WATCHER] batch_process_error datastore_id=%s org_id=%s: %s",
                self.datastore_id,
                self.org_id,
                e,
                exc_info=True,
            )

    def force_process_pending(self) -> None:
        changes_to_process: List[Dict[str, Any]] = []
        with self._lock:
            if self.pending_changes:
                changes_to_process = self.pending_changes.copy()
                self.pending_changes.clear()
        try:
            if changes_to_process:
                self.callback(self.datastore_id, self.org_id, changes_to_process)
        except Exception as e:
            logger.error(
                "[WATCHER] force_process_error datastore_id=%s: %s",
                self.datastore_id,
                e,
                exc_info=True,
            )

    def on_created(self, event) -> None:
        if event.is_directory:
            return
        if not self._should_process(event.src_path):
            logger.debug(
                "[WATCHER] event_debounced datastore_id=%d path=%s reason=short_gap",
                self.datastore_id, event.src_path,
            )
            return
        logger.info(
            "[WATCHER] event_detected datastore_id=%d path=%s event=created",
            self.datastore_id, event.src_path,
        )
        self._dispatch(event.src_path, "created")

    def on_modified(self, event) -> None:
        if event.is_directory:
            return
        if not self._should_process(event.src_path):
            logger.debug(
                "[WATCHER] event_debounced datastore_id=%d path=%s reason=short_gap",
                self.datastore_id, event.src_path,
            )
            return
        logger.info(
            "[WATCHER] event_detected datastore_id=%d path=%s event=modified",
            self.datastore_id, event.src_path,
        )
        self._dispatch(event.src_path, "modified")

    def on_deleted(self, event) -> None:
        if event.is_directory:
            return
        if not self._should_process(event.src_path):
            logger.debug(
                "[WATCHER] event_debounced datastore_id=%d path=%s reason=short_gap",
                self.datastore_id, event.src_path,
            )
            return
        logger.info(
            "[WATCHER] event_detected datastore_id=%d path=%s event=deleted",
            self.datastore_id, event.src_path,
        )
        self._dispatch(event.src_path, "deleted")

    def on_moved(self, event) -> None:
        if event.is_directory:
            return
        if not self._should_process(event.dest_path):
            logger.debug(
                "[WATCHER] event_debounced datastore_id=%d path=%s reason=short_gap",
                self.datastore_id, event.dest_path,
            )
            return
        logger.info(
            "[WATCHER] event_detected datastore_id=%d path=%s event=moved from=%s",
            self.datastore_id, event.dest_path, event.src_path,
        )
        self._dispatch(event.dest_path, "moved")

    def _dispatch(self, path: str, event_type: str) -> None:
        """Apply debouncer then queue the change."""
        if self.debouncer is not None:
            coalesced = self.debouncer.touch(path, event_type)
            if coalesced is None:
                logger.debug(
                    "[WATCHER] event_coalesced datastore_id=%d path=%s",
                    self.datastore_id, path,
                )
                return  # debounced

        # Delay processing by 1 second to allow file write to complete
        logger.info(
            "[WATCHER] event_queued datastore_id=%d path=%s event=%s reason=write_complete_delay",
            self.datastore_id, path, event_type,
        )
        import threading as _threading
        import time as _time

        def _delayed_dispatch(p, et):
            _time.sleep(1.0)
            # Create a simple event-like object
            event = type('_Event', (), {'src_path': p})()
            self._queue_change(event, et)

        _threading.Thread(target=_delayed_dispatch, args=(path, event_type), daemon=True).start()


class DataStoreWatcher:
    """Watches multiple datastore folders with per-datastore configuration.

    Features:
    - Single Observer instance for all datastores (efficient)
    - Dynamic add/remove based on database configuration
    - Per-datastore processing intervals
    - Automatic sync with database every 5 minutes
    - Progress tracking for scans
    - Cancellation support for in-progress scans
    """

    def __init__(self) -> None:
        self._observer = None
        self._lock = threading.Lock()
        self._running = False
        self._executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="watcher"
        )
        self._debouncer = _Debouncer(delay=1.0)
        self.datastore_handlers: Dict[int, DatastoreFileEventHandler] = {}
        self.watched_paths: Dict[int, Path] = {}
        self._last_scan_at: Optional[float] = None
        self._files_scanned: int = 0

        # Progress tracking: scan_id -> {datastore_id, total, processed, status, error}
        self._active_scans: Dict[int, Dict[str, Any]] = {}
        self._scan_id_counter: int = 0
        self._scan_id_lock = threading.Lock()
        # Futures tracking: scan_id -> [Future, ...] for waiting on ingestion tasks
        self._scan_futures: Dict[int, List[Future]] = {}
        self._scan_futures_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Scan ID management (thread-safe)
    # ------------------------------------------------------------------

    def _next_scan_id(self) -> int:
        with self._scan_id_lock:
            self._scan_id_counter += 1
            return self._scan_id_counter

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the observer and begin watching all configured datastores."""
        with self._lock:
            if self._running:
                logger.warning("[WATCHER] already running, ignoring start()")
                return
            self._running = True

        try:
            from watchdog.observers.polling import PollingObserver

            self._observer = PollingObserver()
            self._observer.start()
            logger.info("[WATCHER] observer started (PollingObserver)")
        except Exception as e:
            logger.error("[WATCHER] failed to start observer: %s", e)
            self._running = False
            raise

        self._sync_watchers_with_database()

        # Start periodic sync thread
        sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        sync_thread.start()

        logger.info("[WATCHER] service started")

    def stop(self) -> None:
        """Stop the observer and shut down the thread pool."""
        with self._lock:
            if not self._running:
                return
            self._running = False

        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(timeout=10)
                logger.info("[WATCHER] observer stopped")
            except Exception as e:
                logger.warning("[WATCHER] error stopping observer: %s", e)

        # Process all pending changes
        for handler in self.datastore_handlers.values():
            handler.force_process_pending()

        self._executor.shutdown(wait=False, cancel_futures=True)
        logger.info("[WATCHER] service stopped")

    @property
    def is_running(self) -> bool:
        """Check if the watcher service is running."""
        return self._running

    def sync_watchers_with_database(self) -> None:
        """Sync watchers with database configuration.

        Public method — called when datastore settings change via API.
        """
        self._sync_watchers_with_database()

    def add_datastore(self, datastore_id: int, org_id: int, folder_path: str, interval_minutes: int = 60) -> None:
        """Start watching a specific datastore folder."""
        abs_path = Path(folder_path).resolve()
        if not abs_path.exists() or not abs_path.is_dir():
            logger.warning(
                "[WATCHER] add_datastore_path_not_found datastore_id=%s path=%s",
                datastore_id,
                folder_path,
            )
            return

        # Skip if already watching
        if datastore_id in self.datastore_handlers:
            logger.info(
                "[WATCHER] add_datastore_already_watching datastore_id=%s",
                datastore_id,
            )
            return

        handler = DatastoreFileEventHandler(
            datastore_id=datastore_id,
            org_id=org_id,
            callback=self._on_changes,
            debounce_ms=1000,
            min_interval_seconds=interval_minutes * 60,
            debouncer=self._debouncer,
        )

        try:
            self._observer.schedule(handler, str(abs_path), recursive=True)
            self.datastore_handlers[datastore_id] = handler
            self.watched_paths[datastore_id] = abs_path
            logger.info(
                "[WATCHER] datastore_added datastore_id=%s path=%s interval=%dm",
                datastore_id,
                folder_path,
                interval_minutes,
            )
        except Exception as e:
            logger.error(
                "[WATCHER] add_datastore_failed datastore_id=%s: %s",
                datastore_id,
                e,
                exc_info=True,
            )

    def remove_datastore(self, datastore_id: int) -> None:
        """Stop watching a datastore."""
        if datastore_id not in self.datastore_handlers:
            return

        handler = self.datastore_handlers[datastore_id]
        del self.datastore_handlers[datastore_id]
        if datastore_id in self.watched_paths:
            del self.watched_paths[datastore_id]

        logger.info("[WATCHER] datastore_removed datastore_id=%s", datastore_id)

    # ------------------------------------------------------------------
    # Scan progress helpers
    # ------------------------------------------------------------------

    def _init_scan(self, datastore_id: int) -> int:
        """Initialize a scan on a datastore. Returns the scan_id."""
        db: Session = SessionLocal()
        try:
            ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
            if not ds:
                return -1

            scan_id = self._next_scan_id()

            # Record scan status
            ds.last_scan_status = "running"
            ds.last_scan_at = datetime.now(timezone.utc)
            ds.last_scan_error = None

            # Count total files to scan
            total_files = self._count_files_in_folder(ds.folder_path, ds.scan_pattern)
            ds.last_scan_total_files = total_files
            ds.last_scan_processed = 0

            db.commit()

            # Track in memory
            self._active_scans[scan_id] = {
                "datastore_id": datastore_id,
                "total": total_files,
                "processed": 0,
                "status": "running",
                "error": None,
            }
            # Initialize futures list for this scan
            with self._scan_futures_lock:
                self._scan_futures[scan_id] = []

            logger.info(
                "[WATCHER] scan_init scan_id=%d datastore_id=%d total_files=%d",
                scan_id,
                datastore_id,
                total_files,
            )
            return scan_id
        finally:
            db.close()

    def _update_scan_progress(self, datastore_id: int, processed: int, error: Optional[str] = None) -> None:
        """Update scan progress in memory and DB."""
        db: Session = SessionLocal()
        try:
            ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
            if not ds:
                return

            # Find this datastore in active scans
            for sid, info in self._active_scans.items():
                if info["datastore_id"] == datastore_id:
                    info["processed"] = processed
                    if error:
                        info["status"] = "error"
                        info["error"] = error
                    break

            # Update DB
            ds.last_scan_processed = processed
            ds.last_scan_total_files = ds.last_scan_total_files or 0

            if error:
                ds.last_scan_status = "error"
                ds.last_scan_error = error
            else:
                ds.last_scan_status = "running"

            db.commit()
        finally:
            db.close()

    def _complete_scan(self, datastore_id: int, success: bool, error: Optional[str] = None) -> None:
        """Mark a scan as completed."""
        db: Session = SessionLocal()
        try:
            ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
            if not ds:
                return

            # Find this datastore in active scans
            for sid, info in self._active_scans.items():
                if info["datastore_id"] == datastore_id:
                    info["status"] = "completed" if success else "error"
                    info["error"] = error
                    break

            # Update DB
            if success:
                ds.last_scan_status = "completed"
                ds.last_scan_error = None
            else:
                ds.last_scan_status = "error"
                ds.last_scan_error = error

            db.commit()
        finally:
            db.close()

    def _refresh_file_count(self, datastore_id: int) -> None:
        """Refresh last_scan_total_files from the filesystem.

        Called after file changes are detected so the UI always shows
        the latest file count even before a manual scan runs.
        """
        db: Session = SessionLocal()
        try:
            ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
            if not ds or not ds.folder_path:
                return

            count = self._count_files_in_folder(ds.folder_path, ds.scan_pattern)
            ds.last_scan_total_files = count
            db.commit()
        finally:
            db.close()

    def _cancel_scan(self, datastore_id: int) -> bool:
        """Cancel a running scan on a datastore. Returns True if cancelled."""
        db: Session = SessionLocal()
        try:
            ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
            if not ds or ds.last_scan_status != "running":
                return False

            ds.last_scan_status = "idle"
            ds.last_scan_error = "Scan cancelled by admin"
            db.commit()

            # Remove from active scans
            for sid, info in self._active_scans.items():
                if info["datastore_id"] == datastore_id:
                    info["status"] = "cancelled"
                    break

            logger.info("[WATCHER] scan_cancelled datastore_id=%d", datastore_id)
            return True
        finally:
            db.close()

    def _is_scan_cancelled(self, datastore_id: int) -> bool:
        """Check if a scan is cancelled (should stop processing)."""
        db: Session = SessionLocal()
        try:
            ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
            if not ds:
                return True
            return ds.last_scan_status != "running"
        finally:
            db.close()

    def _count_files_in_folder(self, folder_path: str, scan_pattern: str = "*") -> int:
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
                all_files.update(f for f in matched if f.is_file() and not f.name.startswith("."))

            return len(all_files)
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Scan endpoints
    # ------------------------------------------------------------------

    def scan_single_datastore(self, datastore_id: int) -> Dict[str, Any]:
        """Manually scan a specific datastore for new/modified files.

        Args:
            datastore_id: ID of the datastore to scan

        Returns a summary dict with counts of scanned, new, and skipped files.
        """
        summary: Dict[str, Any] = {"scanned": 0, "new": 0, "modified": 0, "skipped": 0, "errors": 0}

        if not self._running or self._observer is None:
            logger.warning("[WATCHER] scan attempted but service is not running")
            return summary

        db: Session = SessionLocal()
        try:
            ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
            if not ds or not ds.is_active or not ds.folder_path or not os.path.isdir(ds.folder_path):
                logger.warning("[WATCHER] scan skipped invalid datastore id=%d", datastore_id)
                return summary

            # Initialize scan tracking
            scan_id = self._init_scan(datastore_id)
            if scan_id <= 0:
                return summary

            # Count total files first
            total_files = self._count_files_in_folder(ds.folder_path, ds.scan_pattern)

            # Collect futures from ingestion tasks
            ingestion_futures: List[Future] = []

            # Walk all files in the folder
            for root, _dirs, files in os.walk(ds.folder_path):
                # Check for cancellation
                if self._is_scan_cancelled(datastore_id):
                    logger.info("[WATCHER] scan_cancelled mid-scan datastore_id=%d", datastore_id)
                    self._complete_scan(datastore_id, False, "Scan cancelled by admin")
                    summary["errors"] = 1
                    return summary

                for fname in files:
                    fpath = os.path.join(root, fname)

                    try:
                        # Check if file matches scan_pattern
                        if not self._matches_pattern(fpath, ds.scan_pattern):
                            summary["skipped"] += 1
                            continue

                        summary["scanned"] += 1

                        # Process file for this datastore (no KB knowledge needed)
                        future = self._handle_file_in_scan(fpath, datastore_id, scan_id)
                        if future is not None:
                            ingestion_futures.append(future)

                        # Update progress
                        self._update_scan_progress(datastore_id, summary["scanned"])

                    except Exception as e:
                        logger.error(
                            "[WATCHER] scan error for %s: %s", fpath, e
                        )
                        summary["errors"] += 1

            # Wait for all ingestion tasks to complete before marking scan done
            if ingestion_futures:
                logger.info(
                    "[WATCHER] waiting_for_ingestion scan_id=%d datastore_id=%d tasks=%d",
                    scan_id, datastore_id, len(ingestion_futures),
                )
                for future in ingestion_futures:
                    try:
                        future.result(timeout=3600)  # up to 1 hour per task
                    except Exception as e:
                        logger.error(
                            "[WATCHER] ingestion_task_failed scan_id=%d: %s",
                            scan_id, e,
                        )
                        summary["errors"] += 1

            # Clean up futures tracking
            with self._scan_futures_lock:
                self._scan_futures.pop(scan_id, None)

            # Mark scan complete — success only if no ingestion errors
            scan_success = summary["errors"] == 0
            scan_error = f"{summary['errors']} file(s) failed ingestion" if not scan_success else None
            self._complete_scan(datastore_id, scan_success, error=scan_error)
            logger.info(
                "[WATCHER] scan_complete scan_id=%d datastore_id=%d scanned=%d new=%d modified=%d skipped=%d errors=%d",
                scan_id,
                datastore_id,
                summary["scanned"],
                summary["new"],
                summary["modified"],
                summary["skipped"],
                summary["errors"],
            )
            return summary
        finally:
            db.close()

    def _matches_pattern(self, filepath: str, pattern: str = "*") -> bool:
        """Check if a filepath matches the scan pattern."""
        fname = os.path.basename(filepath)
        # Exclude hidden files regardless of pattern
        if fname.startswith("."):
            return False
        if pattern == "*":
            return True

        patterns = [p.strip() for p in pattern.split(",")]
        for pat in patterns:
            if "*" in pat:
                # Use fnmatch for glob patterns
                if fnmatch.fnmatch(fname, pat):
                    return True
            else:
                # Exact match
                if fname == pat:
                    return True
        return False

    def _handle_file_in_scan(
        self,
        event_path: str,
        datastore_id: int,
        scan_id: int,
    ) -> Optional[Future]:
        """Handle a file during scan. Creates or updates Document records and triggers ingestion.
        
        Returns the ingestion Future so the caller can wait for completion.
        DataStore files are processed independently — no KB knowledge needed.
        """
        db: Session = SessionLocal()
        try:

            # Compute hash
            file_hash = self._compute_hash(event_path)
            if not file_hash:
                return

            # Check if document already exists for this file path
            existing = (
                db.query(Document)
                .filter(
                    Document.file_path == event_path,
                    Document.data_store_id == datastore_id,
                )
                .first()
            )

            if existing:
                # Document exists - check if hash changed (file modified)
                if existing.file_hash == file_hash:
                    # File unchanged - but check if chunks exist (ingestion may have failed)
                    chunk_count = db.query(DocumentChunk).filter(
                        DocumentChunk.document_id == existing.id
                    ).count()
                    if chunk_count > 0:
                        # File unchanged and chunks exist - skip
                        return
                    else:
                        # File unchanged but no chunks - re-ingest (ingestion likely failed)
                        logger.info(
                            "[WATCHER] re_ingest_no_chunks path=%s doc_id=%s datastore_id=%s",
                            event_path,
                            existing.id,
                            datastore_id,
                        )
                        future = self._update_document(
                            existing.id, event_path, file_hash, datastore_id, scan_id
                        )
                        return future
                else:
                    # File was modified - trigger re-ingestion
                    future = self._update_document(
                        existing.id, event_path, file_hash, datastore_id, scan_id
                    )
                    return future

            # File is new - trigger ingestion
            future = self._ingest_file(
                event_path, datastore_id, scan_id, file_hash=file_hash
            )
            return future
        finally:
            db.close()

    def _ingest_file(
        self,
        event_path: str,
        datastore_id: int,
        scan_id: int,
        file_hash: Optional[str] = None,
    ) -> Optional[Future]:
        """Create Document + ProcessingTask records and enqueue background processing.
        
        Returns the Future so the caller can wait for completion.
        DataStore files are processed independently — no KB knowledge needed.
        """
        fname = os.path.basename(event_path)
        _, ext = os.path.splitext(fname)
        ext = ext.lower()

        # Skip non-supported extensions
        if ext not in SUPPORTED_EXTENSIONS:
            logger.debug(
                "[WATCHER] file_detected path=%s ext=%s action=skip reason=unsupported_ext",
                event_path,
                ext,
            )
            return

        # Skip hidden/system files
        if fname.startswith("."):
            logger.debug(
                "[WATCHER] file_detected path=%s action=skip reason=hidden_file",
                event_path,
            )
            return

        # Check if file exists
        if not os.path.exists(event_path):
            logger.debug(
                "[WATCHER] file_not_exists path=%s action=skip",
                event_path,
            )
            return

        try:
            file_size = os.path.getsize(event_path)
        except OSError:
            file_size = 0

        from app.services.document_processor import CONTENT_TYPE_MAP

        content_type = CONTENT_TYPE_MAP.get(ext, "application/octet-stream")

        # Use pre-computed hash if available (avoids reading the file twice)
        if file_hash is None:
            file_hash = self._compute_hash(event_path)
        if not file_hash:
            logger.warning(
                "[WATCHER] failed_to_compute_hash path=%s action=skip",
                event_path,
            )
            return

        logger.info(
            "[WATCHER] ingestion_start datastore_id=%d path=%s doc_id=NEW",
            datastore_id, event_path,
        )
        db: Session = SessionLocal()
        try:
            # Create Document record (in-place, no copy to uploads)
            doc = Document(
                knowledge_base_id=None,
                data_store_id=datastore_id,
                file_path=event_path,
                file_name=fname,
                file_hash=file_hash,
                file_size=file_size,
                content_type=content_type,
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)

            # Create ProcessingTask record
            task = ProcessingTask(
                knowledge_base_id=None,
                data_store_id=datastore_id,
                document_id=doc.id,
                status="pending",
                progress=0,
                progress_message="Queued by watcher scan",
            )
            db.add(task)
            db.commit()
            db.refresh(task)

            logger.info(
                "[WATCHER] ingestion_started path=%s datastore_id=%s doc_id=%s task_id=%s",
                event_path,
                datastore_id,
                doc.id,
                task.id,
            )

            # Enqueue background processing
            loop = asyncio.new_event_loop()
            future = self._executor.submit(
                self._run_ingestion,
                event_path,
                fname,
                None,  # kb_id — DataStore doesn't use KBs
                task.id,
                doc.id,
                datastore_id,
                None,  # db — _run_ingestion creates its own session
                loop,
            )
            future.add_done_callback(
                lambda f: self._on_ingestion_done(f, task.id, event_path)
            )

            # Track future for scan completion
            if scan_id > 0:
                with self._scan_futures_lock:
                    if scan_id in self._scan_futures:
                        self._scan_futures[scan_id].append(future)

            # Count as "new"
            for sid, info in self._active_scans.items():
                if info["datastore_id"] == datastore_id:
                    info["new"] = info.get("new", 0) + 1
                    break

            return future

        except Exception as e:
            logger.error(
                "[WATCHER] failed_to_create_ingestion_records: %s",
                e,
                exc_info=True,
            )
            try:
                db.rollback()
            except Exception:
                pass
        finally:
            db.close()

    def _update_document(
        self,
        document_id: int,
        event_path: str,
        file_hash: str,
        datastore_id: int,
        scan_id: int,
    ) -> Optional[Future]:
        """Update an existing document when file content changes.
        
        Returns the Future so the caller can wait for completion.
        DataStore documents don't use KBs — kb_id is always None.
        """
        fname = os.path.basename(event_path)
        _, ext = os.path.splitext(fname)
        ext = ext.lower()

       # Skip non-supported extensions
        if ext not in SUPPORTED_EXTENSIONS:
            return

        if not os.path.exists(event_path):
            return

        try:
            file_size = os.path.getsize(event_path)
        except OSError:
            file_size = 0

        from app.services.document_processor import CONTENT_TYPE_MAP

        content_type = CONTENT_TYPE_MAP.get(ext, "application/octet-stream")

        logger.info(
            "[WATCHER] ingestion_update_start datastore_id=%d path=%s doc_id=%d",
            datastore_id, event_path, document_id,
        )
        db: Session = SessionLocal()
        try:
            doc = db.query(Document).filter(Document.id == document_id).first()
            if not doc:
                return

            # Update document metadata
            doc.file_hash = file_hash
            doc.file_size = file_size
            doc.content_type = content_type
            db.commit()

            # Reset or create processing task
            task = (
                db.query(ProcessingTask)
                .filter(ProcessingTask.document_id == document_id)
                .first()
            )
            if task:
                task.status = "pending"
                task.progress = 0
                task.progress_message = "Re-queued by watcher scan"
            else:
                # Re-ingest with no chunks — create a new task
                task = ProcessingTask(
                    knowledge_base_id=None,
                    data_store_id=datastore_id,
                    document_id=document_id,
                    status="pending",
                    progress=0,
                    progress_message="Re-queued by watcher scan",
                )
                db.add(task)
            db.commit()

            logger.info(
                "[WATCHER] update_started doc_id=%s path=%s",
                document_id,
                event_path,
            )

            # Enqueue background re-processing
            # Note: pass db=None — _run_ingestion will create its own session
            loop = asyncio.new_event_loop()
            future = self._executor.submit(
                self._run_ingestion,
                event_path,
                fname,
                None,  # kb_id — DataStore doesn't use KBs
                task.id,  # task.id is guaranteed to be set (created above if missing)
                document_id,
                datastore_id,
                None,  # db — _run_ingestion creates its own session
                loop,
            )
            future.add_done_callback(
                lambda f: self._on_ingestion_done(f, task.id, event_path)
            )

            # Track future for scan completion
            if scan_id > 0:
                with self._scan_futures_lock:
                    if scan_id in self._scan_futures:
                        self._scan_futures[scan_id].append(future)

            # Count as "modified"
            for sid, info in self._active_scans.items():
                if info["datastore_id"] == datastore_id:
                    info["modified"] = info.get("modified", 0) + 1
                    break

            return future

        except Exception as e:
            logger.error(
                "[WATCHER] update_error doc_id=%s path=%s: %s",
                document_id,
                event_path,
                e,
                exc_info=True,
            )
            try:
                db.rollback()
            except Exception:
                pass
        finally:
            db.close()

    def scan(self) -> Dict[str, Any]:
        """Manually scan all watched datastores for new/modified files.

        Returns a summary dict with counts of scanned, new, and skipped files.
        """
        summary: Dict[str, int] = {"scanned": 0, "new": 0, "modified": 0, "skipped": 0, "errors": 0}

        if not self._running or self._observer is None:
            logger.warning("[WATCHER] scan attempted but service is not running")
            return summary

        # Get all active datastores with auto_scan_enabled
        db: Session = SessionLocal()
        try:
            datastores = (
                db.query(DataStore)
                .filter(DataStore.is_active == True)
                .all()
            )
            for ds in datastores:
                if not ds.folder_path or not os.path.isdir(ds.folder_path):
                    continue

                # Get org_id from first assignment
                assignment = (
                    db.query(OrganizationDataStore)
                    .filter(OrganizationDataStore.data_store_id == ds.id)
                    .first()
                )
                org_id = assignment.org_id if assignment else None

                for root, _dirs, files in os.walk(ds.folder_path):
                    for fname in files:
                        fpath = os.path.join(root, fname)
                        try:
                            # Check if file matches scan_pattern
                            if not self._matches_pattern(fpath, ds.scan_pattern):
                                summary["skipped"] += 1
                                continue
                            summary["scanned"] += 1
                            self._handle_file(fpath, ds.id, "created")
                        except Exception as e:
                            logger.error(
                                "[WATCHER] scan error for %s: %s", fpath, e
                            )
                            summary["errors"] += 1
        finally:
            db.close()

        self._last_scan_at = time_module.time()
        logger.info(
            "[WATCHER] scan_complete scanned=%d new=%d skipped=%d errors=%d",
            summary["scanned"],
            summary["new"],
            summary["skipped"],
            summary["errors"],
        )
        return summary

    def get_status(self) -> Dict[str, Any]:
        """Return current watcher state for the admin status endpoint."""
        return {
            "running": self._running,
            "last_scan_at": self._last_scan_at,
            "files_scanned": self._files_scanned,
            "active_scans": list(self._active_scans.values()),
            "datastores": [
                {
                    "datastore_id": ds_id,
                    "path": str(self.watched_paths.get(ds_id, "unknown")),
                    "pending_changes": len(handler.pending_changes),
                    "min_interval_seconds": handler.min_interval_seconds,
                    "last_processing_time": handler.last_processing_time,
                }
                for ds_id, handler in self.datastore_handlers.items()
            ],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sync_watchers_with_database(self) -> None:
        """Sync watchers with database configuration."""
        db: Session = SessionLocal()
        try:
            # Get all active datastore-organization assignments
            assignments = (
                db.query(OrganizationDataStore)
                .join(DataStore)
                .filter(
                    OrganizationDataStore.is_active == True,
                    DataStore.is_active == True,
                    DataStore.auto_scan_enabled == True,
                )
                .all()
            )

            # Group by datastore_id (org_id comes from assignment)
            datastore_ids = set()
            for assignment in assignments:
                ds_id = assignment.data_store_id
                org_id = assignment.org_id
                datastore_ids.add(ds_id)

                ds = assignment.data_store
                if ds_id not in self.datastore_handlers:
                    self.add_datastore(
                        ds_id,
                        org_id,
                        ds.folder_path,
                        ds.auto_scan_interval_minutes or 60,
                    )

            # Remove watchers for disabled/unassigned datastores
            current_ids = set(self.datastore_handlers.keys())
            to_remove = current_ids - datastore_ids
            for ds_id in to_remove:
                self.remove_datastore(ds_id)

        finally:
            db.close()

    def _sync_loop(self) -> None:
        """Periodic sync thread — checks every 5 minutes."""
        while self._running:
            time_module.sleep(300)  # 5 minutes
            if self._running:
                self._sync_watchers_with_database()

    def _on_changes(self, datastore_id: int, org_id: int, changes: List[Dict[str, Any]]) -> None:
        """Callback when batch of changes is ready to process.

        Args:
            datastore_id: ID of the datastore with changes
            org_id: Organization ID this datastore belongs to
            changes: List of change events with path and event_type
        """
        if not changes:
            return

        logger.info(
            "[WATCHER] batch_ready datastore_id=%d org_id=%d changes=%d",
            datastore_id,
            org_id,
            len(changes),
        )

        # Process each change
        for change in changes:
            fpath = change["path"]
            event_type = change.get("event_type", "modified")
            try:
                self._handle_file(fpath, datastore_id, event_type)
            except Exception as e:
                logger.error(
                    "[WATCHER] handle_file_error path=%s event=%s: %s", fpath, event_type, e, exc_info=True
                )

        # Refresh file count so UI reflects latest state
        self._refresh_file_count(datastore_id)

    def _matches_pattern(self, filepath: str, scan_pattern: str = "*") -> bool:
        """Check if a filepath matches the scan pattern."""
        import fnmatch as _fnmatch

        fname = os.path.basename(filepath)
        # Exclude hidden files regardless of pattern
        if fname.startswith("."):
            return False
        if scan_pattern == "*":
            return True

        patterns = [p.strip() for p in scan_pattern.split(",")]
        for pat in patterns:
            if "*" in pat:
                # Use fnmatch for glob patterns
                if _fnmatch.fnmatch(fname, pat):
                    return True
            else:
                # Exact match
                if fname == pat:
                    return True
        return False

    def _handle_file(
        self,
        event_path: str,
        datastore_id: int,
        event_type: str = "modified"
    ) -> Optional[Future]:
        """Core logic: handle file events (created, modified, deleted).

        Files are processed in-place without copying. DataStore documents
        are independent of KnowledgeBases.

        Args:
            event_path: Full path to the file
            datastore_id: ID of the datastore containing the file
            event_type: One of 'created', 'modified', 'deleted'
        """
        # Handle deletion differently - no need to hash or check extensions
        if event_type == "deleted":
            self._handle_deletion(event_path, None, datastore_id)
            return

        fname = os.path.basename(event_path)
        _, ext = os.path.splitext(fname)
        ext = ext.lower()

        # Skip non-supported extensions
        if ext not in SUPPORTED_EXTENSIONS:
            logger.debug(
                "[WATCHER] file_detected path=%s ext=%s action=skip reason=unsupported_ext",
                event_path,
                ext,
            )
            return

        # Skip hidden/system files
        if fname.startswith("."):
            logger.debug(
                "[WATCHER] file_detected path=%s action=skip reason=hidden_file",
                event_path,
            )
            return

        # Check if file exists (for modified events, file might have been deleted)
        if not os.path.exists(event_path):
            logger.debug(
                "[WATCHER] file_not_exists path=%s action=skip",
                event_path,
            )
            return

        # Get datastore scan_pattern
        db: Session = SessionLocal()
        try:
            ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
            if not ds:
                return
            scan_pattern = ds.scan_pattern or "*"

            # Check scan pattern
            if not self._matches_pattern(event_path, scan_pattern):
                logger.debug(
                    "[WATCHER] file_detected path=%s action=skip reason=pattern_mismatch",
                    event_path,
                )
                return

            # Log the file that is about to be processed
            logger.info(
                "[WATCHER] file_processing datastore_id=%d path=%s event=%s",
                datastore_id, event_path, event_type,
            )
        finally:
            db.close()

        # Compute SHA-256 hash
        file_hash = self._compute_hash(event_path)
        hash_prefix = file_hash[:8] if file_hash else "none"
        file_size = os.path.getsize(event_path)

        # DataStore: process file independently (no KB knowledge needed)
        db: Session = SessionLocal()
        try:
            # Check if document already exists for this file path
            existing = (
                db.query(Document)
                .filter(
                    Document.file_path == event_path,
                    Document.data_store_id == datastore_id,
                )
                .first()
            )

            if existing:
                # Document exists - check if hash changed (file modified)
                if existing.file_hash == file_hash:
                    logger.info(
                        "[WATCHER] no_change path=%s hash=%s datastore_id=%s doc_id=%s",
                        event_path,
                        hash_prefix,
                        datastore_id,
                        existing.id,
                    )
                    return
                else:
                    # File was modified - trigger re-ingestion
                    logger.info(
                        "[WATCHER] file_modified path=%s old_hash=%s new_hash=%s datastore_id=%s doc_id=%s",
                        event_path,
                        existing.file_hash[:8],
                        hash_prefix,
                        datastore_id,
                        existing.id,
                    )
                    # Update existing document
                    return self._update_document(
                        existing.id, event_path, file_hash, datastore_id, scan_id=0
                    )

            # File is new — trigger ingestion in-place
            return self._ingest_file(
                event_path, datastore_id, scan_id=0, file_hash=file_hash
            )
        finally:
            db.close()

    def _update_document_inplace(
        self,
        document_id: int,
        event_path: str,
        fname: str,
        file_hash: str,
        file_size: int,
        ext: str,
        datastore_id: int,
    ) -> None:
        """Update an existing document when file content changes.

        DEPRECATED: Use _update_document instead. This method is kept for
        backward compatibility but delegates to _update_document.
        """
        logger.info(
            "[WATCHER] update_start doc_id=%s path=%s",
            document_id,
            event_path,
        )

        # Trigger background re-processing
        try:
            self._update_document(
                document_id, event_path, file_hash, datastore_id, scan_id=0
            )
        except Exception as e:
            logger.error(
                "[WATCHER] update_error doc_id=%s path=%s: %s",
                document_id,
                event_path,
                e,
                exc_info=True,
            )

    def _trigger_ingestion(
        self,
        event_path: str,
        file_name: str,
        datastore_id: int,
        file_hash: str,
    ) -> None:
        """Create Document + ProcessingTask records and enqueue background processing.

        Files are processed in-place - NOT copied to uploads folder.

        DEPRECATED: Use _ingest_file instead. This method is kept for
        backward compatibility but delegates to _ingest_file.
        """
        # Delegate to _ingest_file
        self._ingest_file(event_path, datastore_id, scan_id=0, file_hash=file_hash)

    def _run_ingestion(
        self,
        file_path: str,
        file_name: str,
        kb_id: Optional[int],
        task_id: int,
        document_id: int,
        data_store_id: Optional[int],
        db: Session,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Run the async ingestion pipeline in a dedicated event loop (threaded).

        Files are processed in-place - NOT copied to uploads folder.

        For DataStore files: kb_id is None, data_store_id is set.
        For KB files: kb_id is set, data_store_id is None.

        IMPORTANT: After ingestion completes, this updates the ProcessingTask
        status to 'completed' or 'failed' and updates the datastore progress.
        """
        try:
            async def _do() -> None:
                await process_document_background(
                    temp_path=file_path,  # Original path, not temp
                    file_name=file_name,
                    kb_id=kb_id,
                    task_id=task_id,
                    document_id=document_id,
                    data_store_id=data_store_id,
                    db=None,  # process_document_background creates its own session
                )

            asyncio.set_event_loop(loop)
            loop.run_until_complete(_do())

            # Mark task as completed (use a fresh session since the old one may be closed)
            try:
                fresh_db = SessionLocal()
                try:
                    db_task = fresh_db.query(ProcessingTask).filter(ProcessingTask.id == task_id).first()
                    if db_task:
                        db_task.status = "completed"
                        db_task.progress = 100
                        db_task.progress_message = "Ingestion completed"
                        fresh_db.commit()
                finally:
                    fresh_db.close()
            except Exception:
                pass

            logger.info(
                "[WATCHER] ingestion_completed task_id=%s path=%s",
                task_id,
                file_path,
            )
        except Exception as e:
            logger.error(
                "[WATCHER] ingestion_failed task_id=%s error=%s",
                task_id,
                e,
                exc_info=True,
            )
            # Mark task as failed (use a fresh session)
            try:
                fresh_db = SessionLocal()
                try:
                    db_task = fresh_db.query(ProcessingTask).filter(ProcessingTask.id == task_id).first()
                    if db_task:
                        db_task.status = "failed"
                        db_task.progress = 0
                        db_task.progress_message = f"Ingestion failed: {str(e)}"
                        fresh_db.commit()
                finally:
                    fresh_db.close()
            except Exception:
                pass
            # Re-raise so future.result() will surface the failure to the scan waiter
            raise
        finally:
            loop.close()

    def _on_ingestion_done(self, future, task_id: int, event_path: str) -> None:
        """Callback after ingestion completes (success or failure).

        DEPRECATED: The _run_ingestion method now handles task status updates
        directly. This method is kept for backward compatibility.
        """
        exc = future.exception()
        if exc:
            logger.error(
                "[WATCHER] ingestion_future_error task_id=%s: %s",
                task_id,
                exc,
            )
        else:
            logger.info(
                "[WATCHER] ingestion_completed task_id=%s path=%s",
                task_id,
                event_path,
            )

    @staticmethod
    def _compute_hash(path: str) -> str:
        """Compute SHA-256 hash of a file."""
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()
        except OSError:
            return ""
