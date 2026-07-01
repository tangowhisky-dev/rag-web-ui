"""DatastoreFileEventHandler — file system event handling for datastore watches.

Contains:
- _Debouncer: coalesces rapid repeated events for the same path
- _SyntheticEvent: synthetic file event for delayed dispatch
- DatastoreFileEventHandler: global handler that resolves datastore from event path
"""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import logging
import os
import threading
import time as time_module
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    _chunk_id_to_point_id,
)
from app.services.utils import get_qdrant_client

from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)


# Root folder watched by the observer (all datastores live under this)
WATCH_ROOT = "/app/data"


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


class _SyntheticEvent:
    """A synthetic file event created for delayed dispatch after write-completion delay.

    Watchdog's observer fires events immediately, but we want to delay processing
    by ~1 second to let the file write complete (especially for editors like VS Code
    that write via temp files and renames). This class provides a minimal event-like
    object with the attributes needed by the handler (src_path, is_directory, event_type, etc.).
    """

    def __init__(self, src_path: str, is_directory: bool = False, event_type: str = "modified") -> None:
        self.src_path = src_path
        self.is_directory = is_directory
        self.src_dir = os.path.dirname(src_path)
        self.dest_path = ""
        self.dest_dir = ""
        self.cookie = 0
        self.name = os.path.basename(src_path)
        self.dir = is_directory
        self.event_type = event_type


class DatastoreFileEventHandler(FileSystemEventHandler):
    """Global event handler that resolves datastore from event path.

    Maintains a mapping of datastore_id -> (org_id, folder_path, min_interval_seconds).
    When an event fires, it resolves the datastore by checking which datastore's
    folder_path contains the event path.

    Features:
    - Debouncing per-file (prevents duplicate events from editors)
    - Per-datastore batch processing with configurable intervals (default: 5 min)
    - org_id and datastore_id tracking for each event
    - File processing (ingestion/update) via _handle_file
    """

    def __init__(
        self,
        callback,
        executor,
        debounce_ms: int = 1000,
        default_min_interval_seconds: int = 300,  # 5 minutes
        debouncer: Optional[_Debouncer] = None,
        progress_lock: Optional[threading.Lock] = None,
    ) -> None:
        self.callback = callback
        self._executor = executor
        self.debounce_ms = debounce_ms / 1000.0
        self.default_min_interval_seconds = default_min_interval_seconds
        self.debouncer = debouncer

        # datastore_id -> (org_id, folder_path, min_interval_seconds)
        self.folder_paths: Dict[int, Tuple[int, str, int]] = {}

        # datastore_id -> list of pending changes
        self.pending_changes: Dict[int, List[Dict[str, Any]]] = {}

        # datastore_id -> threading.Timer for batch processing
        self._batch_timers: Dict[int, threading.Timer] = {}

        # Per-file debouncing: file_path -> last call timestamp (monotonic)
        self._last_call: Dict[str, float] = {}

        # Processing state flag — per-datastore set to allow independent
        # processing across different datastores (a slow ingestion for one
        # datastore doesn't block others).
        self._processing: set[int] = set()

        self._lock = threading.Lock()

        # Lock for scan progress updates — prevents race between event-driven
        # ingestion and manual scan when both update last_scan_processed
        # simultaneously. Shared with DataStoreWatcher to avoid double-counting.
        self._progress_lock = progress_lock if progress_lock is not None else threading.Lock()

    # ------------------------------------------------------------------
    # Folder management
    # ------------------------------------------------------------------

    def add_folder(self, datastore_id: int, org_id: int, folder_path: str, min_interval_seconds: int = 300) -> None:
        """Register a datastore folder path for monitoring."""
        with self._lock:
            self.folder_paths[datastore_id] = (org_id, folder_path, min_interval_seconds)
        logger.info(
            "[WATCHER] handler_folder_added datastore_id=%d path=%s",
            datastore_id, folder_path,
        )

    def remove_folder(self, datastore_id: int) -> None:
        """Unregister a datastore folder path and flush pending changes."""
        with self._lock:
            if datastore_id in self._batch_timers:
                timer = self._batch_timers.pop(datastore_id)
                timer.cancel()
            if datastore_id in self.pending_changes and self.pending_changes[datastore_id]:
                # Flush pending changes before removing the folder
                changes = self.pending_changes.pop(datastore_id)
                org_id = self.folder_paths.get(datastore_id, (None,))[0]
                logger.info(
                    "[WATCHER] handler_folder_removed datastore_id=%d flushing %d pending changes",
                    datastore_id, len(changes),
                )
                try:
                    self.callback(datastore_id, org_id, changes)
                except Exception as e:
                    logger.error(
                        "[WATCHER] handler_flush_on_remove datastore_id=%d: %s",
                        datastore_id, e,
                    )
            if datastore_id in self.folder_paths:
                del self.folder_paths[datastore_id]
        logger.info("[WATCHER] handler_folder_removed datastore_id=%s", datastore_id)

    # ------------------------------------------------------------------
    # Datastore resolution from event path
    # ------------------------------------------------------------------

    def _resolve_datastore(self, event_path: str) -> Optional[int]:
        """Find which datastore's folder_path contains the event path.

        Returns datastore_id or None if no match found.
        Sorts folder paths by length (descending) so longer paths match first.
        This prevents a datastore with folder_path /app/data/reports from
        incorrectly matching files in /app/data/reports/2024.
        """
        sorted_ids = sorted(
            self.folder_paths.keys(),
            key=lambda ds_id: len(str(self.folder_paths[ds_id][1])),
            reverse=True,
        )
        for ds_id in sorted_ids:
            _, folder_path, _ = self.folder_paths[ds_id]
            folder_path_str = str(folder_path)
            if event_path.startswith(folder_path_str + '/') or event_path == folder_path_str:
                return ds_id
        return None

    # ------------------------------------------------------------------
    # Per-file debouncing
    # ------------------------------------------------------------------

    def _should_process(self, src_path: str) -> bool:
        """Check if this file should be processed based on the debouncing window.

        Does NOT update _last_call — that happens in _after_process to ensure
        only processed events reset the debounce timer. Events rejected by
        _should_process must not prevent future events from being processed.
        """
        now = time_module.monotonic()
        with self._lock:
            last = self._last_call.get(src_path)
            if last is not None and (now - last) < self.debounce_ms:
                return False
            return True

    def _after_process(self, src_path: str) -> None:
        """Record that a file was processed. Must be called after _should_process
        returns True and the event has been queued for processing.

        This ensures only processed events (not rejected ones) reset the debounce timer.
        """
        now = time_module.monotonic()
        with self._lock:
            self._last_call[src_path] = now

    # ------------------------------------------------------------------
    # Batch timer management
    # ------------------------------------------------------------------

    def _start_batch_timer(self, datastore_id: int) -> None:
        """Start a per-datastore batch timer. If one already exists, do nothing."""
        with self._lock:
            if datastore_id in self._batch_timers:
                return
            org_id, _, min_interval = self.folder_paths.get(datastore_id, (None, None, self.default_min_interval_seconds))
            interval = min_interval if min_interval is not None else self.default_min_interval_seconds
            timer = threading.Timer(interval, self._flush_batch, args=(datastore_id,))
            timer.daemon = True
            self._batch_timers[datastore_id] = timer
            timer.start()
            logger.info(
                "[WATCHER] batch_timer_started datastore_id=%d interval=%ds",
                datastore_id, interval,
            )

    def _stop_batch_timer(self, datastore_id: int) -> None:
        """Stop a per-datastore batch timer."""
        with self._lock:
            timer = self._batch_timers.pop(datastore_id, None)
            if timer:
                timer.cancel()

    # ------------------------------------------------------------------
    # Event queueing
    # ------------------------------------------------------------------

    def _queue_change(self, datastore_id: int, event, event_type: str) -> None:
        """Queue a change for immediate processing.

        For event-driven processing: process immediately after the change is queued.
        For flush (manual): process immediately — no batch timer.
        The batch timer was removed to avoid 5-minute delays in event-driven processing.
        """
        change = {
            "datastore_id": datastore_id,
            "org_id": self.folder_paths.get(datastore_id, (None,))[0],
            "path": event.src_path,
            "event_type": event.event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with self._lock:
            self.pending_changes.setdefault(datastore_id, []).append(change)

    def _process_pending_changes(self, datastore_id: int) -> None:
        """Process all pending changes for a datastore immediately.

        Used by: event-driven processing (after _queue_change) and flush endpoint.
        Sets _processing flag and processes all pending changes.
        """
        with self._lock:
            if datastore_id not in self.pending_changes or not self.pending_changes[datastore_id]:
                return
            if datastore_id in self._processing:
                return  # Already processing for this datastore
            self._processing.add(datastore_id)
            changes = self.pending_changes.pop(datastore_id)
            org_id = self.folder_paths.get(datastore_id, (None,))[0]
            logger.info(
                "[WATCHER] processing_pending datastore_id=%d changes=%d",
                datastore_id, len(changes),
            )

        # Process outside the lock to avoid holding it during ingestion
        try:
            self._on_changes(datastore_id, org_id, changes)
        except Exception as e:
            logger.error(
                "[WATCHER] process_pending_error datastore_id=%d: %s",
                datastore_id, e, exc_info=True,
            )
        finally:
            with self._lock:
                self._processing.discard(datastore_id)

    def _flush_batch(self, datastore_id: int) -> None:
        """Process all pending changes for a datastore and stop its timer."""
        with self._lock:
            changes = self.pending_changes.pop(datastore_id, [])
            self._stop_batch_timer(datastore_id)

        if changes:
            org_id = self.folder_paths.get(datastore_id, (None,))[0]
            logger.info(
                "[WATCHER] batch_flush datastore_id=%d pending=%d",
                datastore_id, len(changes),
            )
            try:
                self.callback(datastore_id, org_id, changes)
            except Exception as e:
                logger.error(
                    "[WATCHER] batch_flush_error datastore_id=%d: %s",
                    datastore_id, e,
                    exc_info=True,
                )

    def force_process_pending(self, datastore_id: int) -> None:
        """Force process pending changes for a datastore (used on shutdown)."""
        with self._lock:
            changes = self.pending_changes.pop(datastore_id, [])
            self._stop_batch_timer(datastore_id)

        if changes:
            org_id = self.folder_paths.get(datastore_id, (None,))[0]
            logger.info(
                "[WATCHER] force_process datastore_id=%d pending=%d",
                datastore_id, len(changes),
            )
            try:
                self.callback(datastore_id, org_id, changes)
            except Exception as e:
                logger.error(
                    "[WATCHER] force_process_error datastore_id=%d: %s",
                    datastore_id, e,
                    exc_info=True,
                )

    # ------------------------------------------------------------------
    # Watchdog event handlers
    # ------------------------------------------------------------------

    def on_created(self, event) -> None:
        if event.is_directory:
            return
        datastore_id = self._resolve_datastore(event.src_path)
        if datastore_id is None:
            return
        if not self._should_process(event.src_path):
            logger.debug(
                "[WATCHER] event_debounced path=%s datastore_id=%d reason=short_gap",
                event.src_path, datastore_id,
            )
            return
        self._after_process(event.src_path)
        logger.info(
            "[WATCHER] event_detected path=%s datastore_id=%d event=created",
            event.src_path, datastore_id,
        )
        self._dispatch(event.src_path, "created")

    def on_modified(self, event) -> None:
        if event.is_directory:
            return
        datastore_id = self._resolve_datastore(event.src_path)
        if datastore_id is None:
            return
        if not self._should_process(event.src_path):
            logger.debug(
                "[WATCHER] event_debounced path=%s datastore_id=%d reason=short_gap",
                event.src_path, datastore_id,
            )
            return
        self._after_process(event.src_path)
        logger.info(
            "[WATCHER] event_detected path=%s datastore_id=%d event=modified",
            event.src_path, datastore_id,
        )
        self._dispatch(event.src_path, "modified")

    def on_deleted(self, event) -> None:
        if event.is_directory:
            return
        datastore_id = self._resolve_datastore(event.src_path)
        if datastore_id is None:
            return
        if not self._should_process(event.src_path):
            logger.debug(
                "[WATCHER] event_debounced path=%s datastore_id=%d reason=short_gap",
                event.src_path, datastore_id,
            )
            return
        self._after_process(event.src_path)
        logger.info(
            "[WATCHER] event_detected path=%s datastore_id=%d event=deleted",
            event.src_path, datastore_id,
        )
        self._dispatch(event.src_path, "deleted")

    def on_moved(self, event) -> None:
        if event.is_directory:
            return

        # Resolve datastore for the destination path (file moved INTO a datastore)
        dest_datastore_id = self._resolve_datastore(event.dest_path)
        # Resolve datastore for the source path (file moved OUT of a datastore)
        src_datastore_id = self._resolve_datastore(event.src_path)

        # Handle source path: if the file was moved OUT of a datastore,
        # treat it as a deletion to clean up the old document record.
        if src_datastore_id is not None:
            if self._should_process(event.src_path):
                self._after_process(event.src_path)
                logger.info(
                    "[WATCHER] event_detected path=%s datastore_id=%d event=moved_out",
                    event.src_path, src_datastore_id,
                )
                # Dispatch as "deleted" so the old document is cleaned up
                self._dispatch(event.src_path, "deleted")

        # Handle destination path: if the file was moved INTO a datastore,
        # treat it as a "created" event to start ingestion.
        if dest_datastore_id is not None:
            if self._should_process(event.dest_path):
                self._after_process(event.dest_path)
                logger.info(
                    "[WATCHER] event_detected path=%s datastore_id=%d event=moved_in from=%s",
                    event.dest_path, dest_datastore_id, event.src_path,
                )
                # Dispatch as "created" so the file gets ingested as new
                self._dispatch(event.dest_path, "created")
            else:
                logger.debug(
                    "[WATCHER] event_debounced path=%s datastore_id=%d reason=short_gap",
                    event.dest_path, dest_datastore_id,
                )
        else:
            # File was moved to a path that doesn't belong to any datastore
            logger.info(
                "[WATCHER] event_moved_out_of_watch path=%s datastore_id=%s reason=not_watched",
                event.dest_path, src_datastore_id,
            )

    def _dispatch(self, path: str, event_type: str) -> None:
        """Apply debouncer then queue the change."""
        if self.debouncer is not None:
            coalesced = self.debouncer.touch(path, event_type)
            if coalesced is None:
                logger.debug(
                    "[WATCHER] event_coalesced path=%s",
                    path,
                )
                return  # debounced

        # Delay processing by 1 second to allow file write to complete
        logger.info(
            "[WATCHER] event_queued path=%s event=%s reason=write_complete_delay",
            path, event_type,
        )
        import threading as _threading
        import time as _time

        def _delayed_dispatch(p, et):
            _time.sleep(1.0)
            # Create a synthetic event with is_directory and event_type attributes
            event = _SyntheticEvent(src_path=p, is_directory=False, event_type=et)
            ds_id = self._resolve_datastore(p)
            if ds_id is not None:
                self._queue_change(ds_id, event, et)
                self._process_pending_changes(ds_id)

        _threading.Thread(target=_delayed_dispatch, args=(path, event_type), daemon=True).start()

    # ------------------------------------------------------------------
    # Watchdog dispatch — called by observer to route events to on_* methods
    # ------------------------------------------------------------------

    def dispatch(self, event) -> None:
        """Called by the watchdog observer to dispatch events to on_* methods.
        Overrides BaseEventHandler.dispatch — the on_* methods handle datastore resolution.
        """
        if event.is_directory:
            return
        super().dispatch(event)

    # ------------------------------------------------------------------
    # File processing (called by _on_changes callback)
    # ------------------------------------------------------------------

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
            self._handle_deletion(event_path, datastore_id)
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
                    # Update existing document — use scan_id=0 for event-driven processing
                    return self._update_document(
                        existing.id, event_path, file_hash, datastore_id, scan_id=0
                    )

            # File is new — trigger ingestion in-place
            return self._ingest_file(
                event_path, datastore_id, scan_id=0, file_hash=file_hash
            )
        finally:
            db.close()

    def _matches_pattern(self, filepath: str, scan_pattern: str = "*") -> bool:
        """Check if a filepath matches the scan pattern.

        Hidden files (basename starting with '.') are always excluded —
        this is intentional design. Hidden files are typically config,
        lock, or temporary files (e.g., .env, .DS_Store, .gitignore) and
        are not meant to be ingested as documents regardless of the
        scan_pattern setting.
        """
        fname = os.path.basename(filepath)
        # Exclude hidden files regardless of pattern — intentional design.
        # Hidden files are typically config/lock/temp files, not documents.
        if fname.startswith("."):
            return False
        if scan_pattern == "*":
            return True

        patterns = [p.strip() for p in scan_pattern.split(",")]
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

    def _compute_hash(self, path: str) -> str:
        """Compute SHA-256 hash of a file."""
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()
        except OSError:
            return ""

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
                None,  # kb_id (DataStore files have no KB)
                task.id,
                doc.id,
                datastore_id,
                None,
                loop,
            )
            future.add_done_callback(
                lambda f: self._on_ingestion_done(f, task.id, event_path)
            )
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
            loop = asyncio.new_event_loop()
            future = self._executor.submit(
                self._run_ingestion,
                event_path,
                fname,
                None,  # kb_id (DataStore files have no KB)
                task.id,
                document_id,
                datastore_id,
                None,
                loop,
            )
            future.add_done_callback(
                lambda f: self._on_ingestion_done(f, task.id, event_path)
            )
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

    # ------------------------------------------------------------------
    # Deletion handling
    # ------------------------------------------------------------------

    def _handle_deletion(
        self,
        event_path: str,
        datastore_id: Optional[int],
    ) -> None:
        """Handle file deletion - remove Document records and Qdrant vectors.

        For DataStore files: delete the document for this datastore and its Qdrant vectors.
        For KB files: delete from all KBs for the org and their Qdrant vectors.
        """
        logger.info(
            "[WATCHER] file_deleted path=%s datastore_id=%s",
            event_path,
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
                            get_qdrant_client().delete(
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

            # KB deletion: query by org_id from handler mapping
            org_id = self.folder_paths.get(datastore_id, (None,))[0] if datastore_id else None
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
                                get_qdrant_client().delete(
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

    # ------------------------------------------------------------------
    # File count refresh
    # ------------------------------------------------------------------

    def _refresh_file_count(self, datastore_id: int) -> None:
        """Refresh last_scan_total_files from the filesystem."""
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

    def _update_scan_progress(self, datastore_id: int, processed: int) -> None:
        """Update last_scan_processed so UI reflects ingestion progress.

        Called after event-driven ingestion completes. Uses SQL-level atomic
        increment (UPDATE ... SET col = col + :val) so concurrent threads
        cannot lose a counter increment due to a read-modify-write race.
        """
        db: Session = SessionLocal()
        try:
            from sqlalchemy import update

            db.execute(
                update(DataStore)
                .where(DataStore.id == datastore_id)
                .values(last_scan_processed=DataStore.last_scan_processed + processed)
            )
            db.commit()
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
    # Async ingestion runner
    # ------------------------------------------------------------------

    def _run_ingestion(
        self,
        file_path: str,
        file_name: str,
        kb_id: Optional[int],
        task_id: int,
        document_id: int,
        data_store_id: Optional[int],
        db,  # Session — kept for API compatibility with DataStoreWatcher
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Run the async ingestion pipeline in a dedicated event loop (threaded).

        IMPORTANT: After ingestion completes, this updates the ProcessingTask
        status to 'completed' or 'failed' and updates the datastore progress.
        """
        try:
            async def _do() -> None:
                await process_document_background(
                    temp_path=file_path,
                    file_name=file_name,
                    kb_id=kb_id,
                    task_id=task_id,
                    document_id=document_id,
                    data_store_id=data_store_id,
                    db=None,
                )

            asyncio.set_event_loop(loop)
            if not loop.is_running():
                loop.run_until_complete(_do())
            else:
                loop.close()
                logger.warning(
                    "[WATCHER] loop.already_running task_id=%s, closing and re-creating",
                    task_id,
                )
                return

            # Mark task as completed
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
            raise
        finally:
            loop.close()

    def _on_ingestion_done(self, future, task_id: int, event_path: str) -> None:
        """Callback after ingestion completes (success or failure)."""
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

    # ------------------------------------------------------------------
    # Batch callback handler
    # ------------------------------------------------------------------

    def _on_changes(self, datastore_id: int, org_id: int, changes: List[Dict[str, Any]]) -> None:
        """Callback when batch of changes is ready to process.

        Processes all changes and waits for ingestion Futures before
        updating scan progress, so the UI reflects actual completed
        ingestion rather than queued work.

        Args:
            datastore_id: ID of the datastore with changes
            org_id: Organization ID this datastore belongs to
            changes: List of change events with path and event_type
        """
        if not changes:
            return

        logger.info(
            "[WATCHER] batch_ready datastore_id=%d org_id=%s changes=%d",
            datastore_id,
            org_id,
            len(changes),
        )

        # Process each change
        changes_processed = 0
        for change in changes:
            fpath = change["path"]
            event_type = change.get("event_type", "modified")
            try:
                self._handle_file(fpath, datastore_id, event_type)
                changes_processed += 1
            except Exception as e:
                logger.error(
                    "[WATCHER] handle_file_error path=%s event=%s: %s", fpath, event_type, e, exc_info=True
                )

        # Update last_scan_processed so UI doesn't show stale 0
        self._update_scan_progress(datastore_id, changes_processed)
        # Refresh file count so UI reflects latest state
        self._refresh_file_count(datastore_id)
