"""DataStore Watcher Service — global file watching with per-datastore batching.

Architecture:
- Single ``watchdog.Observer`` instance with ``recursive=True`` watching the root
  folder (/app/data/). On Linux this uses ``InotifyObserver`` (instant event
  delivery via inotify), on macOS ``FSEventsObserver``, on Windows ``ReadDirectoryChangesW``.
- Single ``DatastoreFileEventHandler`` that resolves datastore from event path
- Handler maintains a mapping of datastore_id -> (org_id, folder_path, min_interval)
- When an event fires, handler resolves datastore by checking folder_path prefix
- Per-datastore batch timers fire at configurable intervals
- Progress tracking for manual scans

File path convention inside a watched directory:
    kb_{kb_id}/{file_name}

The service parses the path to determine which knowledge base owns the file,
then routes it into the existing ``process_document_background()`` pipeline.

Progress tracking:
- Each scan is assigned an integer scan_id
- The scan_id is stored on the ProcessingTask so it can be matched to a running scan
- When a scan starts, the datastore is updated with scan_id and progress=0
- As files are processed, progress is updated
- When a scan ends, the scan_id is cleared

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
    _get_qdrant_client,
    _chunk_id_to_point_id,
)

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

        Called after event-driven ingestion completes. Uses += to accumulate
        the batch count. Protected by _progress_lock to prevent race with
        the scan thread's = assignment.
        """
        db: Session = SessionLocal()
        try:
            ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
            if not ds:
                return

            with self._progress_lock:
                ds.last_scan_processed += processed
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

        # Process each change and update last_scan_processed so UI
        # reflects ingestion progress, not just total file count.
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


class DataStoreWatcher:
    """Watches datastore folders with global handler and per-datastore batching.

    Architecture:
    - One PollingObserver watches the root folder (/app/data/)
    - One DatastoreFileEventHandler resolves datastore from event path
    - Per-datastore batch timers fire at configurable intervals

    Features:
    - Single Observer instance (efficient resource usage)
    - Dynamic add/remove based on database configuration
    - Per-datastore processing intervals
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

        # Shared lock for scan progress updates — prevents race between
        # event-driven ingestion (handler) and manual scan (this class)
        # when both update last_scan_processed simultaneously.
        self._progress_lock = threading.Lock()

        self._handler = DatastoreFileEventHandler(
            callback=self._on_changes,
            executor=self._executor,
            debounce_ms=1000,
            default_min_interval_seconds=300,  # 5 minutes
            debouncer=self._debouncer,
            progress_lock=self._progress_lock,
        )
        # datastore_id -> folder_path for status reporting
        self._datastore_paths: Dict[int, str] = {}
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
        """Start the observer and begin watching all configured datastores.

        Uses watchdog.Observer with recursive=True which auto-selects the
        platform-native observer:
        - Linux: InotifyObserver (instant event delivery via inotify)
        - macOS: FSEventsObserver (instant event delivery)
        - Windows: ReadDirectoryChangesWObserver (instant event delivery)

        Falls back to PollingObserver when the platform-native observer is
        unavailable (e.g., Docker volume incompatibility on Linux).
        """
        with self._lock:
            if self._running:
                logger.warning("[WATCHER] already running, ignoring start()")
                return
            self._running = True

        try:
            from watchdog.observers import Observer

            # Force PollingObserver when inotify is disabled (e.g., Docker Desktop
            # on macOS where inotify doesn't properly propagate events from
            # host bind mounts into the container).
            if not settings.WATCHER_USE_INOTIFY:
                from watchdog.observers.polling import PollingObserver

                self._observer = PollingObserver(timeout=settings.WATCH_POLL_INTERVAL)
                self._observer.start()
                logger.info(
                    "[WATCHER] observer started (PollingObserver with "
                    "recursive=True, WATCH_POLL_INTERVAL=%ds, "
                    "WATCHER_USE_INOTIFY=%s)",
                    settings.WATCH_POLL_INTERVAL,
                    settings.WATCHER_USE_INOTIFY,
                )
            else:
                self._observer = Observer(timeout=settings.WATCH_POLL_INTERVAL)
                self._observer.start()
                logger.info(
                    "[WATCHER] observer started (Observer with recursive=True, "
                    "WATCHER_USE_INOTIFY=%s)",
                    settings.WATCHER_USE_INOTIFY,
                )
        except (ImportError, OSError) as e:
            # Fallback to PollingObserver if native observer is unavailable
            from watchdog.observers.polling import PollingObserver

            self._observer = PollingObserver(timeout=settings.WATCH_POLL_INTERVAL)
            self._observer.start()
            logger.warning(
                "[WATCHER] native observer unavailable, "
                "falling back to PollingObserver (WATCH_POLL_INTERVAL=%ds): %s",
                settings.WATCH_POLL_INTERVAL,
                e,
            )

        # Register the observer on the root folder with recursive=True
        # This tells the observer to watch subdirectories as well
        self._observer.schedule(self._handler, WATCH_ROOT, recursive=True)
        logger.info("[WATCHER] observer registered on root=%s (recursive=True)", WATCH_ROOT)

        self._sync_watchers_with_database()

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
            except Exception as e:
                logger.warning("[WATCHER] error stopping observer: %s", e)

        # Process all pending changes for all datastores
        for datastore_id in list(self._handler.pending_changes.keys()):
            self._handler.force_process_pending(datastore_id)

        self._executor.shutdown(wait=False, cancel_futures=True)
        logger.info("[WATCHER] service stopped")

    @property
    def is_running(self) -> bool:
        """Check if the watcher service is running."""
        return self._running

    # ------------------------------------------------------------------
    # Watcher sync (called on startup and on datastore add/remove)
    # ------------------------------------------------------------------

    def sync_watchers_with_database(self) -> None:
        """Sync watchers with database configuration.

        For each active datastore, adds it to the handler.
        Removes any datastore that is no longer active.
        """
        self._sync_watchers_with_database()

    def add_datastore(self, datastore_id: int, org_id: int, folder_path: str, interval_minutes: int = 60) -> None:
        """Register a datastore in the handler's path mapping."""
        abs_path = Path(folder_path).resolve()
        if not abs_path.exists() or not abs_path.is_dir():
            logger.warning(
                "[WATCHER] add_datastore_path_not_found datastore_id=%s path=%s",
                datastore_id,
                folder_path,
            )
            return

        # Skip if already watching
        if datastore_id in self._datastore_paths:
            logger.info(
                "[WATCHER] add_datastore_already_watching datastore_id=%s",
                datastore_id,
            )
            return

        # Register the datastore in the handler's folder_paths map
        self._handler.add_folder(datastore_id, org_id, abs_path, interval_minutes * 60)
        self._datastore_paths[datastore_id] = str(abs_path)

        logger.info(
            "[WATCHER] datastore_added datastore_id=%s path=%s interval=%dm",
            datastore_id,
            folder_path,
            interval_minutes,
        )

    def remove_datastore(self, datastore_id: int) -> None:
        """Unregister a datastore and flush pending changes."""
        if datastore_id not in self._datastore_paths:
            return

        self._datastore_paths.pop(datastore_id)

        # Remove from handler (flushes pending changes)
        self._handler.remove_folder(datastore_id)

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

            # Check if this scan was already initialized from the POST
            # handler (which calls _init_scan before starting the thread
            # to ensure the SSE endpoint can always find it). If so, skip
            # re-initialization to avoid creating a duplicate scan entry.
            existing_scan = None
            for sid, info in self._active_scans.items():
                if info["datastore_id"] == datastore_id:
                    existing_scan = (sid, info)
                    break

            if existing_scan:
                # Scan already initialized from POST handler — return the
                # existing scan_id without re-adding it. This avoids the race
                # condition where the scan thread re-init creates a new entry
                # and the old entry (which SSE might be reading from) is lost.
                logger.info(
                    "[WATCHER] scan_already_initialized from POST handler scan_id=%d datastore_id=%d",
                    existing_scan[0], datastore_id,
                )
                return existing_scan[0]  # Return existing scan_id

            # Check if another thread (e.g., POST handler or scan thread) has
            # already initialized this scan. This can happen when both the
            # POST handler and the scan thread call _init_scan concurrently.
            # If a scan exists for this datastore, return the existing scan_id
            # without re-adding it. This prevents duplicate scan entries in
            # _active_scans and avoids losing progress state that the SSE
            # endpoint may be reading.
            existing_scan_id = None
            for sid, info in self._active_scans.items():
                if info["datastore_id"] == datastore_id:
                    existing_scan_id = sid
                    break

            if existing_scan_id is not None:
                # Another thread already initialized this scan — return the
                # existing scan_id so the caller doesn't try to re-init.
                logger.info(
                    "[WATCHER] scan_already_initialized by other thread scan_id=%d datastore_id=%d",
                    existing_scan_id, datastore_id,
                )
                return existing_scan_id

            # Clean up any stale scans from previous runs — the SSE
            # endpoint finds the most recent scan for a datastore by
            # iterating _active_scans in reverse insertion order. If the
            # SSE endpoint connects before the new scan has been added,
            # it would find the old scan and emit stale data.
            stale_scan_id = None
            for sid, info in self._active_scans.items():
                if info["datastore_id"] == datastore_id:
                    stale_scan_id = sid
                    break
            if stale_scan_id is not None:
                self._active_scans.pop(stale_scan_id, None)
                logger.info(
                    "[WATCHER] cleanup_stale_scan_in_init scan_id=%d datastore_id=%d",
                    stale_scan_id, datastore_id,
                )
            else:
                logger.debug(
                    "[WATCHER] no_stale_scan_in_init datastore_id=%d active_scans=%s",
                    datastore_id, list(self._active_scans.keys()),
                )

            # Record scan status
            ds.last_scan_status = "running"
            ds.last_scan_at = datetime.now(timezone.utc)
            ds.last_scan_error = None

            # Count total files to scan
            total_files = self._count_files_in_folder(ds.folder_path, ds.scan_pattern)
            ds.last_scan_total_files = total_files
            ds.last_scan_processed = 0

            db.commit()

            # Track in memory — the SSE endpoint always finds the most
            # recently added scan for a given datastore by iterating
            # _active_scans in reverse insertion order (Python 3.7+).
            self._active_scans[scan_id] = {
                "datastore_id": datastore_id,
                "total": total_files,
                "processed": 0,
                "status": "running",
                "error_count": 0,
                "new": 0,
                "modified": 0,
                "skipped": 0,
                "error_message": None,  # string error message from _complete_scan
            }
            logger.info(
                "[WATCHER] scan_init scan_id=%d datastore_id=%d total_files=%d status=running",
                scan_id, datastore_id, total_files,
            )
            # Initialize futures list for this scan
            with self._scan_futures_lock:
                self._scan_futures[scan_id] = []

            logger.info(
                "[WATCHER] added_scan_to_active_scans scan_id=%d datastore_id=%d",
                scan_id, datastore_id,
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
                        info["error_count"] = info.get("error_count", 0)
                        info["error_message"] = error
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

            # Find this datastore in active scans and update its status
            # so the SSE endpoint can detect completion. Do NOT remove the
            # entry — the SSE endpoint may still be reading from it.
            for sid, info in self._active_scans.items():
                if info["datastore_id"] == datastore_id:
                    info["status"] = "completed" if success else "error"
                    info["error_count"] = info.get("error_count", 0)
                    info["error_message"] = error
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

            # Find this datastore in active scans — do NOT remove it. The
            # SSE endpoint may still be reading from _active_scans and needs
            # to find the cancelled entry to emit the final status event.
            # Stale scans are cleaned up in _init_scan before adding a new scan.
            for sid, info in self._active_scans.items():
                if info["datastore_id"] == datastore_id:
                    info["status"] = "cancelled"
                    info["error_message"] = "Scan cancelled by admin"
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
                            # Update _active_scans for SSE
                            for sid, scan_info in self._active_scans.items():
                                if scan_info["datastore_id"] == datastore_id:
                                    scan_info["skipped"] = summary["skipped"]
                                    scan_info["modified"] = summary["modified"]
                                    break
                            continue

                        summary["scanned"] += 1

                        # Check if file is new or modified (without re-doing the hash)
                        file_is_modified = False
                        db2 = SessionLocal()
                        try:
                            existing = (
                                db2.query(Document)
                                .filter(
                                    Document.file_path == fpath,
                                    Document.data_store_id == datastore_id,
                                )
                                .first()
                            )
                            if existing:
                                file_is_modified = True
                        finally:
                            db2.close()

                        if file_is_modified:
                            summary["modified"] += 1
                        else:
                            summary["new"] += 1

                        # Process file for this datastore (no KB knowledge needed)
                        future = self._handle_file_in_scan(fpath, datastore_id, scan_id)
                        if future is not None:
                            ingestion_futures.append(future)

                        # Update progress in memory for SSE
                        self._update_scan_progress(datastore_id, summary["scanned"])
                        # Update new/modified/skipped/error counts in _active_scans for SSE
                        for sid, scan_info in self._active_scans.items():
                            if scan_info["datastore_id"] == datastore_id:
                                scan_info["new"] = summary["new"]
                                scan_info["modified"] = summary["modified"]
                                scan_info["skipped"] = summary["skipped"]
                                scan_info["error_count"] = summary["errors"]
                                break

                    except Exception as e:
                        logger.error(
                            "[WATCHER] scan error for %s: %s", fpath, e
                        )
                        summary["errors"] += 1
                        # Update _active_scans for SSE
                        for sid, scan_info in self._active_scans.items():
                            if scan_info["datastore_id"] == datastore_id:
                                scan_info["error_count"] = summary["errors"]
                                break

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
            # Compute hash (delegate to handler which has the method)
            file_hash = self._handler._compute_hash(event_path)
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
                        future = self._update_document_in_scan(
                            existing.id, event_path, file_hash, datastore_id, scan_id
                        )
                        return future
                else:
                    # File was modified - trigger re-ingestion
                    future = self._update_document_in_scan(
                        existing.id, event_path, file_hash, datastore_id, scan_id
                    )
                    return future

            # File is new - trigger ingestion
            future = self._ingest_file_in_scan(
                event_path, datastore_id, scan_id, file_hash=file_hash
            )
            return future
        finally:
            db.close()

    def _ingest_file_in_scan(
        self,
        event_path: str,
        datastore_id: int,
        scan_id: int,
        file_hash: Optional[str] = None,
    ) -> Optional[Future]:
        """Create Document + ProcessingTask records and enqueue background processing for scans."""
        fname = os.path.basename(event_path)
        _, ext = os.path.splitext(fname)
        ext = ext.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            return

        if fname.startswith("."):
            return

        if not os.path.exists(event_path):
            return

        try:
            file_size = os.path.getsize(event_path)
        except OSError:
            file_size = 0

        from app.services.document_processor import CONTENT_TYPE_MAP
        content_type = CONTENT_TYPE_MAP.get(ext, "application/octet-stream")

        if file_hash is None:
            file_hash = self._handler._compute_hash(event_path)
        if not file_hash:
            return

        db: Session = SessionLocal()
        try:
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

            # Enqueue background processing
            loop = asyncio.new_event_loop()
            future = self._executor.submit(
                self._run_ingestion,
                event_path,
                fname,
                None,  # kb_id
                task.id,
                doc.id,
                datastore_id,
                None,
                loop,
            )
            future.add_done_callback(
                lambda f: self._on_ingestion_done(f, task.id, event_path)
            )

            if scan_id > 0:
                with self._scan_futures_lock:
                    if scan_id in self._scan_futures:
                        self._scan_futures[scan_id].append(future)

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

    def _update_document_in_scan(
        self,
        document_id: int,
        event_path: str,
        file_hash: str,
        datastore_id: int,
        scan_id: int,
    ) -> Optional[Future]:
        """Update an existing document when file content changes during a scan."""
        fname = os.path.basename(event_path)
        _, ext = os.path.splitext(fname)
        ext = ext.lower()

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

        db: Session = SessionLocal()
        try:
            doc = db.query(Document).filter(Document.id == document_id).first()
            if not doc:
                return

            doc.file_hash = file_hash
            doc.file_size = file_size
            doc.content_type = content_type
            db.commit()

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

            # Enqueue background re-processing
            loop = asyncio.new_event_loop()
            future = self._executor.submit(
                self._run_ingestion,
                event_path,
                fname,
                None,
                task.id,
                document_id,
                datastore_id,
                None,
                loop,
            )
            future.add_done_callback(
                lambda f: self._on_ingestion_done(f, task.id, event_path)
            )

            if scan_id > 0:
                with self._scan_futures_lock:
                    if scan_id in self._scan_futures:
                        self._scan_futures[scan_id].append(future)

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
        """Manually scan all watched datastores for new/modified files."""
        summary: Dict[str, int] = {"scanned": 0, "new": 0, "modified": 0, "skipped": 0, "errors": 0}

        if not self._running or self._observer is None:
            logger.warning("[WATCHER] scan attempted but service is not running")
            return summary

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

                for root, _dirs, files in os.walk(ds.folder_path):
                    for fname in files:
                        fpath = os.path.join(root, fname)
                        try:
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
        with self._handler._lock:
            processing = self._handler._processing
        return {
            "running": self._running,
            "last_scan_at": self._last_scan_at,
            "files_scanned": self._files_scanned,
            "active_scans": list(self._active_scans.values()),
            "datastores": [
                {
                    "datastore_id": ds_id,
                    "path": self._datastore_paths.get(ds_id, "unknown"),
                    "pending_changes": len(self._handler.pending_changes.get(ds_id, [])),
                    "min_interval_seconds": self._handler.folder_paths.get(ds_id, (None, None, 300))[2],
                    "processing": ds_id in self._handler._processing if ds_id in self._datastore_paths else False,
                }
                for ds_id in self._handler.folder_paths
            ],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sync_watchers_with_database(self) -> None:
        """Sync watchers with database configuration.

        Registers all active, auto_scan_enabled datastores — regardless of
        whether they have org assignments. Unassigned datastores are watched
        with org_id=None; file processing works identically since org_id is
        only used for logging and KB-deletion cleanup (which already guards
        on org_id is not None)."""
        db: Session = SessionLocal()
        try:
            # Build org_id lookup from assignments (unassigned → None)
            assignment_map: Dict[int, int] = {}
            for a in (
                db.query(OrganizationDataStore)
                .filter(OrganizationDataStore.is_active == True)
                .all()
            ):
                assignment_map[a.data_store_id] = a.org_id

            # Query all active datastores with auto-scan enabled
            datastores = (
                db.query(DataStore)
                .filter(
                    DataStore.is_active == True,
                    DataStore.auto_scan_enabled == True,
                )
                .all()
            )

            datastore_ids = set()
            for ds in datastores:
                ds_id = ds.id
                datastore_ids.add(ds_id)
                org_id = assignment_map.get(ds_id)  # None for unassigned
                interval = ds.auto_scan_interval_minutes or 60
                self.add_datastore(
                    ds_id,
                    org_id,
                    ds.folder_path,
                    interval,
                )

            current_ids = set(self._datastore_paths.keys())
            to_remove = current_ids - datastore_ids
            for ds_id in to_remove:
                self.remove_datastore(ds_id)

        finally:
            db.close()

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
            "[WATCHER] batch_ready datastore_id=%d org_id=%s changes=%d",
            datastore_id,
            org_id,
            len(changes),
        )

        # Process each change and update last_scan_processed so UI
        # reflects ingestion progress, not just total file count.
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

        # Update last_scan_processed so UI reflects latest state.
        # Delegate to handler's _update_scan_progress (+=) — the handler
        # method handles accumulation and is protected by _progress_lock.
        # The handler also calls _refresh_file_count internally, so we don't
        # need to call it here — that would duplicate the refresh.

    def _handle_file(
        self,
        event_path: str,
        datastore_id: int,
        event_type: str = "modified"
    ) -> Optional[Future]:
        """Core logic: handle file events (created, modified, deleted).

        This is the entry point for event-driven ingestion. It delegates
        to the handler's methods for file processing.
        """
        # Delegate to handler's _handle_file which handles all the logic
        return self._handler._handle_file(event_path, datastore_id, event_type)

    def _update_scan_progress(self, datastore_id: int, processed: int) -> None:
        """Set last_scan_processed to the current processed count.

        Called during a scan after each file is processed. The `processed`
        parameter is the cumulative count (summary["scanned"]), so we set
        it directly with = instead of += to avoid double-counting.

        Protected by _progress_lock to prevent race with event-driven
        ingestion's += update.
        """
        db: Session = SessionLocal()
        try:
            ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
            if not ds:
                return

            with self._progress_lock:
                ds.last_scan_processed = processed
            db.commit()
        finally:
            db.close()

    # NOTE: _refresh_file_count was removed — it's handled by the handler's
    # _on_changes() which already calls it after event-driven ingestion.
    # The manual scan sets it via _count_files_in_folder at scan start and
    # end, and the scan's _update_scan_progress sets last_scan_processed.

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
