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
from sqlalchemy.exc import IntegrityError
from qdrant_client.models import PointIdsList
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.datastore import DataStore, DataStoreFileManifest, OrganizationDataStore
from app.models.knowledge import Document, DocumentUpload, ProcessingTask, DocumentChunk
from app.models.knowledge import KnowledgeBase
from app.services.ingestion import (
    SUPPORTED_EXTENSIONS,
    process_document_background,
    _chunk_id_to_point_id,
)
from app.services.infrastructure import get_qdrant_client

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

        # Per-file debouncing: file_path -> last call timestamp (monotonic)
        self._last_call: Dict[str, float] = {}

        # Track delayed-dispatch threads so we can wait for them on shutdown.
        self._delayed_threads: List[threading.Thread] = []
        self._delayed_threads_lock = threading.Lock()

        # Processing state flag — per-datastore set to allow independent
        # processing across different datastores (a slow ingestion for one
        # datastore doesn't block others).
        self._processing: set[int] = set()

        # Per-datastore batch timers for auto_process_enabled datastores.
        # When auto_process is enabled, file events are queued but NOT processed
        # immediately — they accumulate until the timer fires, then the batch
        # is processed. This matches the UI description: "File changes are
        # automatically processed every N minutes."
        # When auto_process is disabled (min_interval_seconds <= 0), events are
        # processed immediately (the original behavior).
        self._batch_timers: Dict[int, threading.Timer] = {}
        self._batch_timers_lock = threading.Lock()

        self._lock = threading.Lock()

        # Lock for scan progress updates — prevents race between event-driven
        # ingestion and manual scan when both update last_scan_processed
        # simultaneously. Shared with DataStoreWatcher to avoid double-counting.
        self._progress_lock = progress_lock if progress_lock is not None else threading.Lock()

    # ------------------------------------------------------------------
    # Folder management
    # ------------------------------------------------------------------

    def add_folder(self, datastore_id: int, org_id: int, folder_path: str, min_interval_seconds: int = 300) -> None:
        """Register a datastore folder path for monitoring.

        When min_interval_seconds > 0, file events are batched and processed
        every min_interval_seconds (auto_process mode). When <= 0, events are
        processed immediately.
        """
        with self._lock:
            self.folder_paths[datastore_id] = (org_id, folder_path, min_interval_seconds)
        # Cancel any existing batch timer for this datastore
        self._cancel_batch_timer(datastore_id)
        logger.info(
            "[WATCHER] handler_folder_added datastore_id=%d path=%s",
            datastore_id, folder_path,
        )

    def remove_folder(self, datastore_id: int) -> None:
        """Unregister a datastore folder path and flush pending changes."""
        self._cancel_batch_timer(datastore_id)
        with self._lock:
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

    def _cancel_batch_timer(self, datastore_id: int) -> None:
        """Cancel the batch timer for a datastore if one is running."""
        with self._batch_timers_lock:
            timer = self._batch_timers.pop(datastore_id, None)
        if timer is not None:
            timer.cancel()

    def _schedule_batch_timer(self, datastore_id: int) -> None:
        """Schedule (or reschedule) the batch processing timer for a datastore.

        If the datastore has min_interval_seconds > 0 (auto_process enabled),
        a timer is set to fire after that interval, at which point all
        accumulated pending_changes are processed as a batch.

        If min_interval_seconds <= 0, events are processed immediately
        and no timer is scheduled.
        """
        entry = self.folder_paths.get(datastore_id)
        if entry is None:
            return
        _, _, min_interval = entry
        if min_interval <= 0:
            return  # immediate mode — no timer needed

        self._cancel_batch_timer(datastore_id)

        def _fire():
            # Process accumulated changes when the timer fires
            self._process_pending_changes(datastore_id)
            # Reschedule if there are still pending changes or the datastore
            # is still being watched
            if datastore_id in self.folder_paths:
                self._schedule_batch_timer(datastore_id)

        timer = threading.Timer(min_interval, _fire)
        timer.daemon = True
        timer.name = f"batch-timer-{datastore_id}"
        with self._batch_timers_lock:
            self._batch_timers[datastore_id] = timer
        timer.start()

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
    # Event queueing
    # ------------------------------------------------------------------

    def _queue_change(self, datastore_id: int, event, event_type: str) -> None:
        """Queue a change for immediate processing.

        For event-driven processing: process immediately after the change is queued.
        For flush (manual): process immediately — no batch timer.
        The batch timer was removed to avoid 5-minute delays in event-driven processing.

        Deduplicates by file path: if a change for the same path is already
        queued, it is replaced with the latest event. This prevents processing
        the same file multiple times when it is saved repeatedly while a batch
        is being processed.
        """
        change = {
            "datastore_id": datastore_id,
            "org_id": self.folder_paths.get(datastore_id, (None,))[0],
            "path": event.src_path,
            "event_type": event.event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with self._lock:
            queue = self.pending_changes.setdefault(datastore_id, [])
            # Deduplicate by file path — replace any existing entry for the
            # same path with the latest event. This avoids ingesting the same
            # file N times when it is saved N times while a batch is running.
            for i, existing in enumerate(queue):
                if existing["path"] == change["path"]:
                    queue[i] = change
                    return
            queue.append(change)

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

    def force_process_pending(self, datastore_id: int) -> None:
        """Force process pending changes for a datastore (used on shutdown)."""
        self._cancel_batch_timer(datastore_id)
        with self._lock:
            changes = self.pending_changes.pop(datastore_id, [])

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
            try:
                ds_id = self._resolve_datastore(p)
                if ds_id is not None:
                    entry = self.folder_paths.get(ds_id)
                    min_interval = entry[2] if entry else 0

                    # Track-only mode (min_interval == -1): manual-scan
                    # datastores. Update file counts but don't ingest.
                    if min_interval == -1:
                        self._update_file_count(ds_id, p, et)
                        return

                    # Ingest mode: queue the change for processing
                    event = _SyntheticEvent(src_path=p, is_directory=False, event_type=et)
                    self._queue_change(ds_id, event, et)
                    if min_interval > 0:
                        self._schedule_batch_timer(ds_id)
                    else:
                        self._process_pending_changes(ds_id)
            except Exception as e:
                logger.warning("[WATCHER] delayed_dispatch error path=%s: %s", p, e)
            finally:
                with self._delayed_threads_lock:
                    try:
                        self._delayed_threads.remove(threading.current_thread())
                    except ValueError:
                        pass

        t = _threading.Thread(target=_delayed_dispatch, args=(path, event_type), daemon=True)
        with self._delayed_threads_lock:
            # Prune finished threads to prevent unbounded list growth
            self._delayed_threads = [th for th in self._delayed_threads if th.is_alive()]
            self._delayed_threads.append(t)
        t.start()

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

    def _should_skip_file(self, fname: str, ext: str, event_path: str) -> bool:
        if ext not in SUPPORTED_EXTENSIONS:
            logger.debug(
                "[WATCHER] file_detected path=%s ext=%s action=skip reason=unsupported_ext",
                event_path,
                ext,
            )
            return True
        if fname.startswith(".") or fname.startswith("~$") or fname.startswith(".~"):
            logger.debug(
                "[WATCHER] file_detected path=%s action=skip reason=hidden_or_temp",
                event_path,
            )
            return True
        if ext in (".tmp", ".swp", ".swo", ".bak", ".lock"):
            logger.debug(
                "[WATCHER] file_detected path=%s ext=%s action=skip reason=temp_ext",
                event_path,
                ext,
            )
            return True
        if not os.path.exists(event_path):
            logger.debug(
                "[WATCHER] file_not_exists path=%s action=skip",
                event_path,
            )
            return True
        return False

    def _get_scan_pattern(
        self, datastore_id: int, event_path: str, event_type: str,
    ) -> Optional[str]:
        db: Session = SessionLocal()
        try:
            ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
            if not ds:
                return None
            scan_pattern = ds.scan_pattern or "*"
            if not self._matches_pattern(event_path, scan_pattern):
                logger.debug(
                    "[WATCHER] file_detected path=%s action=skip reason=pattern_mismatch",
                    event_path,
                )
                return None
            logger.info(
                "[WATCHER] file_processing datastore_id=%d path=%s event=%s",
                datastore_id, event_path, event_type,
            )
            return scan_pattern
        finally:
            db.close()

    def _handle_existing_document(
        self,
        db: Session,
        existing: Document,
        event_path: str,
        file_hash: str,
        hash_prefix: str,
        datastore_id: int,
    ) -> Optional[Future]:
        if not existing.is_selected:
            logger.info(
                "[WATCHER] file_unselected path=%s doc_id=%s — skipping",
                event_path, existing.id,
            )
            return None

        if existing.file_hash == file_hash:
            chunk_count = db.query(DocumentChunk).filter(
                DocumentChunk.document_id == existing.id
            ).count()
            if chunk_count > 0:
                logger.info(
                    "[WATCHER] no_change path=%s hash=%s datastore_id=%s doc_id=%s",
                    event_path,
                    hash_prefix,
                    datastore_id,
                    existing.id,
                )
                return None
            else:
                logger.info(
                    "[WATCHER] re_ingest_no_chunks path=%s doc_id=%s datastore_id=%s",
                    event_path,
                    existing.id,
                    datastore_id,
                )
                return self._update_document(
                    existing.id, event_path, file_hash, datastore_id, scan_id=0
                )
        else:
            logger.info(
                "[WATCHER] file_modified path=%s old_hash=%s new_hash=%s datastore_id=%s doc_id=%s",
                event_path,
                (existing.file_hash or "none")[:8],
                hash_prefix,
                datastore_id,
                existing.id,
            )
            return self._update_document(
                existing.id, event_path, file_hash, datastore_id, scan_id=0
            )

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
        if event_type == "deleted":
            self._handle_deletion(event_path, datastore_id)
            return

        fname = os.path.basename(event_path)
        _, ext = os.path.splitext(fname)
        ext = ext.lower()

        if self._should_skip_file(fname, ext, event_path):
            return

        scan_pattern = self._get_scan_pattern(datastore_id, event_path, event_type)
        if scan_pattern is None:
            return

        file_hash = self._compute_hash(event_path)
        hash_prefix = file_hash[:8] if file_hash else "none"
        file_size = os.path.getsize(event_path)

        db: Session = SessionLocal()
        from app.services.datastore_watcher.utils import acquire_file_lock, release_file_lock
        if not acquire_file_lock(db, datastore_id, event_path):
            logger.info("[WATCHER] file_locked path=%s — skipping (another process holds lock)", event_path)
            db.close()
            return
        try:
            existing = (
                db.query(Document)
                .filter(
                    Document.file_path == event_path,
                    Document.data_store_id == datastore_id,
                )
                .first()
            )

            if existing:
                return self._handle_existing_document(
                    db, existing, event_path, file_hash, hash_prefix, datastore_id,
                )

            return self._ingest_file(
                event_path, datastore_id, scan_id=0, file_hash=file_hash
            )
        finally:
            release_file_lock(db, datastore_id, event_path)
            db.close()

    def _matches_pattern(self, filepath: str, scan_pattern: str = "*") -> bool:
        """Check if a filepath matches the scan pattern. Delegates to shared utility."""
        from app.services.datastore_watcher.utils import matches_pattern
        return matches_pattern(filepath, scan_pattern)

    def _compute_hash(self, path: str) -> str:
        """Compute SHA-256 hash of a file, aborting if the file changes mid-read."""
        try:
            size_before = os.path.getsize(path)
        except OSError:
            return ""

        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
        except OSError:
            return ""

        try:
            size_after = os.path.getsize(path)
        except OSError:
            return ""

        if size_before != size_after:
            logger.warning("[WATCHER] file size changed during hashing: %s", path)
            return ""

        return h.hexdigest()

    def _upsert_manifest(
        self,
        db: Session,
        datastore_id: int,
        event_path: str,
        file_hash: str,
        file_size: int,
    ) -> None:
        """Create or update the DataStoreFileManifest row for a file."""
        now = datetime.now(timezone.utc)
        manifest = (
            db.query(DataStoreFileManifest)
            .filter(
                DataStoreFileManifest.datastore_id == datastore_id,
                DataStoreFileManifest.file_path == event_path,
            )
            .first()
        )
        if manifest:
            manifest.file_hash = file_hash
            manifest.file_size = file_size
            manifest.updated_at = now
        else:
            db.add(
                DataStoreFileManifest(
                    datastore_id=datastore_id,
                    file_path=event_path,
                    file_hash=file_hash,
                    file_size=file_size,
                    discovered_at=now,
                    updated_at=now,
                )
            )
        db.commit()

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

        # Skip hidden/system files and temp/lock files
        if fname.startswith(".") or fname.startswith("~$") or fname.startswith(".~"):
            logger.debug(
                "[WATCHER] file_detected path=%s action=skip reason=hidden_or_temp",
                event_path,
            )
            return
        if ext in (".tmp", ".swp", ".swo", ".bak", ".lock"):
            logger.debug(
                "[WATCHER] file_detected path=%s ext=%s action=skip reason=temp_ext",
                event_path,
                ext,
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

        from app.services.ingestion.document_processor import CONTENT_TYPE_MAP

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
            except IntegrityError:
                db.rollback()
                logger.warning(
                    "[WATCHER] duplicate_document path=%s datastore_id=%s action=skip",
                    event_path,
                    datastore_id,
                )
                return

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

            # Keep manifest in sync so discovery does not re-process this file
            self._upsert_manifest(db, datastore_id, event_path, file_hash, file_size)

            logger.info(
                "[WATCHER] ingestion_started path=%s datastore_id=%s doc_id=%s task_id=%s",
                event_path,
                datastore_id,
                doc.id,
                task.id,
            )

            # Enqueue background processing
            future = self._executor.submit(
                self._run_ingestion,
                event_path,
                fname,
                None,  # kb_id (DataStore files have no KB)
                task.id,
                doc.id,
                datastore_id,
                None,
                file_hash=file_hash,
                file_size=file_size,
                content_type=content_type,
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

        from app.services.ingestion.document_processor import CONTENT_TYPE_MAP

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

            # Keep manifest in sync with the new hash
            self._upsert_manifest(db, datastore_id, event_path, file_hash, file_size)

            logger.info(
                "[WATCHER] update_started doc_id=%s path=%s",
                document_id,
                event_path,
            )

            # Enqueue background re-processing
            future = self._executor.submit(
                self._run_ingestion,
                event_path,
                fname,
                None,  # kb_id (DataStore files have no KB)
                task.id,
                document_id,
                datastore_id,
                None,
                file_hash=file_hash,
                file_size=file_size,
                content_type=content_type,
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

    def _delete_qdrant_vectors(self, collection_name: str, doc_id: int, chunk_ids: list) -> None:
        """Delete Qdrant vectors for a document, handling missing collections."""
        if not chunk_ids:
            return
        try:
            point_ids = [_chunk_id_to_point_id(cid) for cid in chunk_ids]
            get_qdrant_client().delete(
                collection_name=collection_name,
                points_selector=PointIdsList(points=point_ids),
            )
        except UnexpectedResponse as e:
            if "404" in str(e):
                logger.info(
                    "[WATCHER] Qdrant vectors already gone for document_id=%s",
                    doc_id,
                )
            else:
                logger.warning(
                    "[WATCHER] Qdrant delete failed for document_id=%s: %s",
                    doc_id, e,
                )
        except Exception as e:
            logger.warning(
                "[WATCHER] Qdrant delete failed for document_id=%s: %s",
                doc_id, e,
            )

    def _handle_datastore_deletion(
        self,
        db: Session,
        event_path: str,
        datastore_id: int,
    ) -> None:
        """Handle deletion for a DataStore document."""
        # DataStore deletion: delete the document for this datastore
        doc = (
            db.query(Document)
            .filter(
                Document.file_path == event_path,
                Document.data_store_id == datastore_id,
            )
            .first()
        )
        if doc:
            # Capture IDs before DB deletion — needed for Qdrant/Neo4j
            # cleanup after the DB commit.
            doc_id = doc.id
            chunk_ids = [
                cid[0] for cid in db.query(DocumentChunk.id).filter(
                    DocumentChunk.document_id == doc.id
                ).all()
            ]

            # DB cleanup first. If this commit fails, vector/graph data
            # is still intact and the next scan retries. If it succeeds
            # but Qdrant/Neo4j cleanup fails below, orphaned data is
            # invisible (document gone from DB) and reconciliation
            # cleans it up on next startup.
            db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).delete()
            db.query(ProcessingTask).filter(ProcessingTask.document_id == doc.id).delete()
            db.delete(doc)
            logger.info(
                "[WATCHER] document_deleted path=%s datastore_id=%s doc_id=%s",
                event_path,
                datastore_id,
                doc_id,
            )

        # Remove manifest entry for the deleted file (or if it was never ingested)
        db.query(DataStoreFileManifest).filter(
            DataStoreFileManifest.datastore_id == datastore_id,
            DataStoreFileManifest.file_path == event_path,
        ).delete(synchronize_session=False)

        db.commit()

        # Qdrant cleanup (after DB commit, using captured IDs)
        if doc and chunk_ids:
            self._delete_qdrant_vectors(f"ds_{datastore_id}", doc_id, chunk_ids)

        # Neo4j cleanup (after DB commit, using captured doc_id)
        if doc:
            try:
                from app.services.graph import delete_graph_for_document
                delete_graph_for_document(kb_id=None, document_id=doc_id, data_store_id=datastore_id)
                logger.info(
                    "[WATCHER] Neo4j cleanup done for document_id=%s",
                    doc_id,
                )
            except Exception as e:
                logger.warning(
                    "[WATCHER] Neo4j cleanup failed for document_id=%s: %s",
                    doc_id, e,
                )

    def _handle_kb_deletion(
        self,
        db: Session,
        event_path: str,
        datastore_id: Optional[int],
    ) -> None:
        """Handle deletion for KB documents."""
        # KB deletion: query by org_id from handler mapping
        org_id = self.folder_paths.get(datastore_id, (None,))[0] if datastore_id else None
        if org_id is not None:
            kb_list = (
                db.query(KnowledgeBase)
                .filter(KnowledgeBase.org_id == org_id)
                .values("id")
            )
            kb_list = [kb[0] for kb in kb_list]

            # Capture (kb_id, doc_id, chunk_ids) for all affected docs
            # before DB deletion — needed for Qdrant/Neo4j cleanup after.
            cleanup_targets = []
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
                    chunk_ids = [
                        cid[0] for cid in db.query(DocumentChunk.id).filter(
                            DocumentChunk.document_id == doc.id
                        ).all()
                    ]
                    cleanup_targets.append((kb_id, doc.id, chunk_ids))

                    db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).delete()
                    db.query(ProcessingTask).filter(ProcessingTask.document_id == doc.id).delete()
                    db.delete(doc)
                    logger.info(
                        "[WATCHER] document_deleted path=%s kb_id=%s doc_id=%s",
                        event_path,
                        kb_id,
                        doc.id,
                    )

            db.commit()

            # Qdrant + Neo4j cleanup after DB commit, using captured IDs
            for kb_id, doc_id, chunk_ids in cleanup_targets:
                self._delete_qdrant_vectors(f"kb_{kb_id}", doc_id, chunk_ids)

                try:
                    from app.services.graph import delete_graph_for_document
                    delete_graph_for_document(kb_id=kb_id, document_id=doc_id)
                    logger.info(
                        "[WATCHER] Neo4j cleanup done for kb_id=%s doc_id=%s",
                        kb_id, doc_id,
                    )
                except Exception as e:
                    logger.warning(
                        "[WATCHER] Neo4j cleanup failed for kb_id=%s doc_id=%s: %s",
                        kb_id, doc_id, e,
                    )

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
            if datastore_id is not None:
                self._handle_datastore_deletion(db, event_path, datastore_id)
                return

            self._handle_kb_deletion(db, event_path, datastore_id)
        finally:
            db.close()

    # ------------------------------------------------------------------
    # File count refresh
    # ------------------------------------------------------------------

    def _update_file_count(self, datastore_id: int, file_path: str, event_type: str) -> None:
        """Incrementally update last_scan_total_files for a track-only datastore.

        Called when a file event fires for a manual-scan datastore. Instead of
        ingesting the file, just adjusts the file count so the UI always shows
        the current number of files in the folder.

        Uses the same filtering as _handle_file: only counts files with
        supported extensions and skips hidden/temp files.
        """
        from app.services.ingestion.document_converter import SUPPORTED_EXTENSIONS
        fname = os.path.basename(file_path)
        _, ext = os.path.splitext(fname)
        ext = ext.lower()

        # Same filtering as _handle_file — don't count unsupported/temp files
        if ext not in SUPPORTED_EXTENSIONS:
            return
        if fname.startswith(".") or fname.startswith("~$") or fname.startswith(".~"):
            return
        if ext in (".tmp", ".swp", ".swo", ".bak", ".lock"):
            return

        delta = 0
        if event_type in ("created", "moved"):
            if os.path.exists(file_path):
                delta = 1
        elif event_type in ("deleted", "moved_from"):
            delta = -1

        if delta == 0:
            return

        db: Session = SessionLocal()
        try:
            from sqlalchemy import text
            db.execute(
                text("UPDATE data_stores SET last_scan_total_files = GREATEST(last_scan_total_files + :delta, 0) WHERE id = :ds_id"),
                {"delta": delta, "ds_id": datastore_id},
            )
            db.commit()
            logger.debug(
                "[WATCHER] file_count_updated datastore_id=%d delta=%d event=%s",
                datastore_id, delta, event_type,
            )
        except Exception as e:
            logger.warning("[WATCHER] file_count_update_failed datastore_id=%d: %s", datastore_id, e)
            db.rollback()
        finally:
            db.close()

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
        """Update last_event_processed so UI reflects event-driven ingestion progress.

        Called after event-driven ingestion completes. Uses SQL-level atomic
        increment (UPDATE ... SET col = col + :val) so concurrent threads
        cannot lose a counter increment due to a read-modify-write race.

        This is separate from manual scan progress (last_scan_processed) so
        the UI can distinguish event-driven vs manual scan activity.
        """
        db: Session = SessionLocal()
        try:
            from sqlalchemy import update
            from datetime import datetime, timezone

            db.execute(
                update(DataStore)
                .where(DataStore.id == datastore_id)
                .values(
                    last_event_processed=DataStore.last_event_processed + processed,
                    last_event_at=datetime.now(timezone.utc),
                )
            )
            db.commit()
        finally:
            db.close()

    def _count_files_in_folder(self, folder_path: str, scan_pattern: str = "*") -> int:
        """Count files matching pattern in folder. Delegates to shared utility."""
        from app.services.datastore_watcher.utils import count_files_in_folder
        return count_files_in_folder(folder_path, scan_pattern)

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
        file_hash: Optional[str] = None,
        file_size: Optional[int] = None,
        content_type: Optional[str] = None,
    ) -> None:
        """Run the async ingestion pipeline in a dedicated event loop (threaded).

        Delegates to the centralized ingestion dispatcher.
        """
        from app.services.ingestion.ingestion_dispatcher import run_ingestion_in_thread
        run_ingestion_in_thread(
            file_path=file_path,
            file_name=file_name,
            task_id=task_id,
            document_id=document_id,
            kb_id=kb_id,
            data_store_id=data_store_id,
            file_hash=file_hash,
            file_size=file_size,
            content_type=content_type,
        )

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

        # Process each change and collect ingestion futures
        changes_processed = 0
        futures: List[Tuple[Future, str]] = []
        for change in changes:
            fpath = change["path"]
            event_type = change.get("event_type", "modified")
            try:
                future = self._handle_file(fpath, datastore_id, event_type)
                if future is not None:
                    futures.append((future, fpath))
                changes_processed += 1
            except Exception as e:
                logger.error(
                    "[WATCHER] handle_file_error path=%s event=%s: %s", fpath, event_type, e, exc_info=True
                )

        # Wait for event-driven ingestion to finish before updating progress
        for future, fpath in futures:
            try:
                future.result(timeout=3600)
            except Exception as e:
                logger.error(
                    "[WATCHER] ingestion_future_error path=%s: %s", fpath, e, exc_info=True
                )

        # Update last_event_processed so UI doesn't show stale 0
        self._update_scan_progress(datastore_id, changes_processed)
        # Refresh file count so UI reflects latest state
        self._refresh_file_count(datastore_id)
