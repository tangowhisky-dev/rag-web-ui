"""DataStoreWatcher — lifecycle, scan orchestration, and recovery management.

Contains:
- DataStoreWatcher: manages the watchdog observer, scan lifecycle, and progress tracking
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import os
import threading
import time as time_module
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import update
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.datastore import DataStore, OrganizationDataStore
from app.models.knowledge import Document, DocumentUpload, ProcessingTask, DocumentChunk
from app.models.knowledge import KnowledgeBase
from app.services.datastore_watcher.handler import (
    DatastoreFileEventHandler,
    _Debouncer,
    _SyntheticEvent,
)
from app.services.discovery import discover_datastore
from app.services.ingestion import (
    SUPPORTED_EXTENSIONS,
    process_document_background,
    _chunk_id_to_point_id,
)

logger = logging.getLogger(__name__)


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
        self._health_thread = None
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
        self._datastore_paths_lock = threading.Lock()
        self._last_scan_at: Optional[float] = None
        self._files_scanned: int = 0

        # Progress tracking: scan_id -> {datastore_id, total, processed, status, error}
        self._active_scans: Dict[int, Dict[str, Any]] = {}
        self._active_scans_lock = threading.Lock()
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
            from app.core.settings_registry import get_def
            poll_interval = get_def("WATCH_POLL_INTERVAL").default

            # Force PollingObserver when inotify is disabled (e.g., Docker Desktop
            # on macOS where inotify doesn't properly propagate events from
            # host bind mounts into the container).
            if not settings.WATCHER_USE_INOTIFY:
                from watchdog.observers.polling import PollingObserver

                self._observer = PollingObserver(timeout=poll_interval)
                self._observer.start()
                logger.info(
                    "[WATCHER] observer started (PollingObserver with "
                    "recursive=True, WATCH_POLL_INTERVAL=%ds, "
                    "WATCHER_USE_INOTIFY=%s)",
                    poll_interval,
                    settings.WATCHER_USE_INOTIFY,
                )
            else:
                self._observer = Observer(timeout=poll_interval)
                self._observer.start()
                logger.info(
                    "[WATCHER] observer started (Observer with recursive=True, "
                    "WATCHER_USE_INOTIFY=%s)",
                    settings.WATCHER_USE_INOTIFY,
                )
        except (ImportError, OSError) as e:
            # Fallback to PollingObserver if native observer is unavailable
            from watchdog.observers.polling import PollingObserver
            from app.core.settings_registry import get_def
            poll_interval = get_def("WATCH_POLL_INTERVAL").default

            self._observer = PollingObserver(timeout=poll_interval)
            self._observer.start()
            logger.warning(
                "[WATCHER] native observer unavailable, "
                "falling back to PollingObserver (WATCH_POLL_INTERVAL=%ds): %s",
                poll_interval,
                e,
            )

        # Register the observer on the root folder with recursive=True
        # This tells the observer to watch subdirectories as well
        import os as _os
        if not _os.path.isdir("/app/data"):
            logger.warning(
                "[WATCHER] root directory /app/data does not exist — "
                "creating it so the observer can be scheduled"
            )
            try:
                _os.makedirs("/app/data", exist_ok=True)
            except OSError as e:
                logger.error("[WATCHER] failed to create /app/data: %s — watcher will not detect file changes", e)
                return
        self._observer.schedule(self._handler, "/app/data", recursive=True)
        logger.info("[WATCHER] observer registered on root=/app/data (recursive=True)")

        self._sync_watchers_with_database()

        # Start a lightweight health-check thread that restarts the
        # observer if it dies unexpectedly (e.g. inotify exhaustion,
        # too many open files, OS-level resource limits).
        self._health_thread = threading.Thread(
            target=self._health_check_loop,
            name="watcher-health",
            daemon=True,
        )
        self._health_thread.start()

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

    def _health_check_loop(self) -> None:
        """Monitor the observer thread and restart it if it dies.

        Also runs periodic cleanup of stale completed scans from
        _active_scans to prevent unbounded memory growth.
        """
        import time as _time
        while True:
            _time.sleep(30)
            with self._lock:
                if not self._running:
                    return
                observer = self._observer

            # Clean up stale completed scans (5-minute TTL)
            self._cleanup_stale_scans()

            if observer is None:
                continue
            # watchdog observers set ._stopped_event when they finish
            if getattr(observer, "is_alive", lambda: False)():
                continue
            logger.warning("[WATCHER] observer thread died — attempting restart")
            try:
                from watchdog.observers import Observer
                from watchdog.observers.polling import PollingObserver
                from app.core.settings_registry import get_def
                poll_interval = get_def("WATCH_POLL_INTERVAL").default
                if settings.WATCHER_USE_INOTIFY:
                    new_observer = Observer(timeout=poll_interval)
                else:
                    new_observer = PollingObserver(timeout=poll_interval)
                new_observer.start()
                new_observer.schedule(self._handler, "/app/data", recursive=True)
                with self._lock:
                    self._observer = new_observer
                logger.info("[WATCHER] observer restarted successfully")

                # Trigger a scan for all active datastores to catch any
                # file changes that happened while the observer was dead.
                # The discovery engine's manifest comparison skips unchanged
                # files, so this is cheap when nothing changed.
                self._trigger_post_restart_scan()
            except Exception as e:
                logger.error("[WATCHER] observer restart failed: %s — will retry in 30s", e)

    def _trigger_post_restart_scan(self) -> None:
        """Scan all active datastores after observer restart.

        Files changed while the observer was dead (up to 30s) are missed
        by the event system.  A discovery scan catches them by comparing
        the filesystem against the manifest.  Runs in a daemon thread so
        it doesn't block the health-check loop.
        """
        import threading as _threading

        def _scan_all() -> None:
            db: Session = SessionLocal()
            try:
                ds_ids = [
                    row[0]
                    for row in db.query(DataStore.id)
                    .filter(DataStore.is_active == True)
                    .all()
                ]
            finally:
                db.close()

            if not ds_ids:
                return

            logger.info(
                "[WATCHER] post_restart_scan_start datastore_ids=%s",
                ds_ids,
            )
            for ds_id in ds_ids:
                try:
                    self.scan_single_datastore(ds_id)
                except Exception as e:
                    logger.warning(
                        "[WATCHER] post_restart_scan_failed datastore_id=%d: %s",
                        ds_id, e,
                    )
            logger.info("[WATCHER] post_restart_scan_complete")

        t = _threading.Thread(target=_scan_all, name="post-restart-scan", daemon=True)
        t.start()

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

        # If already watching, still update the handler's folder_paths so
        # org_id stays current after org reassignment.
        with self._datastore_paths_lock:
            if datastore_id in self._datastore_paths:
                logger.info(
                    "[WATCHER] add_datastore_already_watching datastore_id=%s — updating org_id",
                    datastore_id,
                )
                self._handler.add_folder(datastore_id, org_id, abs_path, interval_minutes * 60)
                return

        # Register the datastore in the handler's folder_paths map
        self._handler.add_folder(datastore_id, org_id, abs_path, interval_minutes * 60)
        with self._datastore_paths_lock:
            self._datastore_paths[datastore_id] = str(abs_path)

        logger.info(
            "[WATCHER] datastore_added datastore_id=%s path=%s interval=%dm",
            datastore_id,
            folder_path,
            interval_minutes,
        )

    def remove_datastore(self, datastore_id: int) -> None:
        """Unregister a datastore and flush pending changes."""
        with self._datastore_paths_lock:
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

            # Atomic check-then-insert: ensure no duplicate scan entries
            # are created when the POST handler and scan thread race.
            with self._active_scans_lock:
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
            with self._active_scans_lock:
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
        """Update scan progress in memory and DB atomically."""
        db: Session = SessionLocal()
        try:
            ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
            if not ds:
                return

            # Find this datastore in active scans
            with self._active_scans_lock:
                for sid, info in self._active_scans.items():
                    if info["datastore_id"] == datastore_id:
                        info["processed"] = info.get("processed", 0) + processed
                        if error:
                            info["status"] = "error"
                            info["error_count"] = info.get("error_count", 0)
                            info["error_message"] = error
                        break

            # Update DB atomically (same pattern as handler._update_scan_progress)
            db.execute(
                update(DataStore)
                .where(DataStore.id == datastore_id)
                .values(last_scan_processed=DataStore.last_scan_processed + processed)
            )
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
            with self._active_scans_lock:
                for sid, info in self._active_scans.items():
                    if info["datastore_id"] == datastore_id:
                        info["status"] = "completed" if success else "error"
                        info["error_count"] = info.get("error_count", 0)
                        info["error_message"] = error
                        info["_completed_at"] = time_module.time()
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

        self._cleanup_stale_scans()

    def _cancel_scan(self, datastore_id: int) -> bool:
        """Cancel a running scan on a datastore. Returns True if cancelled.

        Cancels:
        - Prevents new files from being submitted (status → cancelled)
        - Cancels all in-flight ingestion futures for this scan
        - Cancels all pending and in-flight graph builds for this datastore
        """
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
            cancelled_scan_ids: list[int] = []
            with self._active_scans_lock:
                for sid, info in self._active_scans.items():
                    if info["datastore_id"] == datastore_id:
                        info["status"] = "cancelled"
                        info["error_message"] = "Scan cancelled by admin"
                        info["_completed_at"] = time_module.time()
                        cancelled_scan_ids.append(sid)
                        break

            # Cancel all in-flight ingestion futures for this datastore's scans.
            cancelled_futures = 0
            for scan_id in cancelled_scan_ids:
                with self._scan_futures_lock:
                    futures = self._scan_futures.get(scan_id, [])
                    for f in futures:
                        if not f.done():
                            f.cancel()
                            cancelled_futures += 1

            # Cancel all pending and in-flight graph builds for this datastore.
            from app.services.ingestion.ingestion_dispatcher import cancel_graph_builds_for_datastore
            cancelled_graphs = cancel_graph_builds_for_datastore(datastore_id)

            logger.info(
                "[WATCHER] scan_cancelled datastore_id=%d futures=%d graph_builds=%d",
                datastore_id, cancelled_futures, cancelled_graphs,
            )
            return True
        finally:
            db.close()

        # Unreachable — return above exits before this line.
        # _cleanup_stale_scans is now called from the health-check loop.

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

    def _cleanup_stale_scans(self, max_age_seconds: int = 300) -> None:
        """Remove completed/error/cancelled scans older than max_age_seconds.

        Prevents _active_scans from growing unbounded. The SSE endpoint
        reads final state immediately after completion, so a 5-minute TTL
        is safe — by then the client has either received the final event
        or timed out.
        """
        now = time_module.time()
        with self._active_scans_lock:
            stale_ids = [
                sid for sid, info in self._active_scans.items()
                if info.get("status") in ("completed", "error", "cancelled")
                and info.get("_completed_at") is not None
                and (now - info["_completed_at"]) > max_age_seconds
            ]
            for sid in stale_ids:
                self._active_scans.pop(sid, None)
                with self._scan_futures_lock:
                    self._scan_futures.pop(sid, None)
        if stale_ids:
            logger.info("[WATCHER] cleanup_stale_scans removed=%d", len(stale_ids))

    def _count_files_in_folder(self, folder_path: str, scan_pattern: str = "*") -> int:
        """Count files matching pattern in folder. Delegates to shared utility."""
        from app.services.datastore_watcher.utils import count_files_in_folder
        return count_files_in_folder(folder_path, scan_pattern)

    # ------------------------------------------------------------------
    # Scan endpoints
    # ------------------------------------------------------------------

    def scan_single_datastore(
        self,
        datastore_id: int,
        force_full_hash: bool = False,
    ) -> Dict[str, Any]:
        """Manually scan a specific datastore for new/modified files.

        Args:
            datastore_id: ID of the datastore to scan
            force_full_hash: When True, hash every file instead of using
                stat-first incremental comparison.  Use for periodic
                safety-net scans.

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

            # Use the discovery engine for accurate new/modified/deleted classification
            try:
                result = discover_datastore(datastore_id, force_full_hash=force_full_hash)
            except Exception as e:
                logger.error(
                    "[WATCHER] discovery_failed datastore_id=%d: %s", datastore_id, e,
                    exc_info=True,
                )
                self._complete_scan(datastore_id, False, str(e))
                summary["errors"] = 1
                return summary

            summary["scanned"] = result.total_files_discovered
            summary["skipped"] = result.skipped_files
            summary["new"] = len(result.new_files)
            summary["modified"] = len(result.modified_files)
            summary["deleted"] = len(result.deleted_files)

            # Reflect discovery counts in SSE state
            with self._active_scans_lock:
                for sid, scan_info in self._active_scans.items():
                    if scan_info["datastore_id"] == datastore_id:
                        scan_info["new"] = summary["new"]
                        scan_info["modified"] = summary["modified"]
                        scan_info["skipped"] = summary["skipped"]
                        scan_info["deleted"] = summary["deleted"]
                        scan_info["error_count"] = summary["errors"]
                        scan_info["total"] = total_files
                        break

            # Collect futures from ingestion tasks
            ingestion_futures: List[Future] = []

            # Process new/modified files
            files_to_process = result.new_files + result.modified_files
            for idx, fmeta in enumerate(files_to_process):
                if self._is_scan_cancelled(datastore_id):
                    logger.info("[WATCHER] scan_cancelled mid-scan datastore_id=%d", datastore_id)
                    self._complete_scan(datastore_id, False, "Scan cancelled by admin")
                    summary["errors"] = 1
                    return summary

                fpath = fmeta["file_path"]
                try:
                    future = self._handle_file_in_scan(
                        fpath, datastore_id, scan_id,
                        file_hash=fmeta.get("file_hash"),
                    )
                    if future is not None:
                        ingestion_futures.append(future)
                        # Progress is incremented when the ingestion future
                        # completes (via _on_scan_ingestion_done callback),
                        # not when it's submitted. This gives the UI real-time
                        # progress that reflects actual completion.
                    else:
                        # File was skipped (unsupported extension, duplicate,
                        # or already ingested) — count it as processed now.
                        self._update_scan_progress(datastore_id, 1)

                    with self._active_scans_lock:
                        for sid, scan_info in self._active_scans.items():
                            if scan_info["datastore_id"] == datastore_id:
                                scan_info["new"] = summary["new"]
                                scan_info["modified"] = summary["modified"]
                                scan_info["skipped"] = summary["skipped"]
                                scan_info["error_count"] = summary["errors"]
                                break

                except Exception as e:
                    logger.error("[WATCHER] scan error for %s: %s", fpath, e)
                    summary["errors"] += 1
                    with self._active_scans_lock:
                        for sid, scan_info in self._active_scans.items():
                            if scan_info["datastore_id"] == datastore_id:
                                scan_info["error_count"] = summary["errors"]
                                break

            # Process deleted files (files on disk that no longer exist)
            for fmeta in result.deleted_files:
                fpath = fmeta["file_path"]
                try:
                    self._handler._handle_deletion(fpath, datastore_id)
                except Exception as e:
                    logger.error("[WATCHER] deletion error for %s: %s", fpath, e)
                    summary["errors"] += 1
                    with self._active_scans_lock:
                        for sid, scan_info in self._active_scans.items():
                            if scan_info["datastore_id"] == datastore_id:
                                scan_info["error_count"] = summary["errors"]
                                break

            # Wait for all ingestion tasks to complete before marking scan done.
            # Each task covers: parse → embed → Qdrant upsert.  Graph build runs
            # in a separate daemon thread and does not block the scan.  10 minutes
            # per file covers large PDFs with OCR; a timeout is a real hang (API
            # down, DB locked), not a slow graph build.
            if ingestion_futures:
                logger.info(
                    "[WATCHER] waiting_for_ingestion scan_id=%d datastore_id=%d tasks=%d",
                    scan_id, datastore_id, len(ingestion_futures),
                )
                for future in ingestion_futures:
                    try:
                        future.result(timeout=600)  # 10 minutes per task
                    except TimeoutError:
                        logger.error(
                            "[WATCHER] ingestion_task_timeout scan_id=%d — cancelling future",
                            scan_id,
                        )
                        future.cancel()
                        summary["errors"] += 1
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
        """Check if a filepath matches the scan pattern. Delegates to shared utility."""
        from app.services.datastore_watcher.utils import matches_pattern
        return matches_pattern(filepath, pattern)

    def _handle_file_in_scan(
        self,
        event_path: str,
        datastore_id: int,
        scan_id: int,
        file_hash: Optional[str] = None,
    ) -> Optional[Future]:
        """Handle a file during scan. Creates or updates Document records and triggers ingestion.

        Returns the ingestion Future so the caller can wait for completion.
        DataStore files are processed independently — no KB knowledge needed.

        Args:
            file_hash: SHA-256 from the discovery engine.  If provided,
                skips re-hashing the file (the discovery engine already
                computed it moments ago).  Falls back to hashing if None
                (e.g. event-driven path where discovery didn't run).
        """
        db: Session = SessionLocal()
        # Acquire per-file advisory lock to prevent race with event-driven processing
        from app.services.datastore_watcher.utils import acquire_file_lock, release_file_lock
        if not acquire_file_lock(db, datastore_id, event_path):
            logger.info("[WATCHER] file_locked path=%s — skipping (another process holds lock)", event_path)
            db.close()
            return
        try:
            # Use the hash from discovery if available; otherwise compute it.
            # Discovery hashed the file moments ago during the walk phase,
            # so re-hashing would waste I/O on large files.
            if not file_hash:
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
            release_file_lock(db, datastore_id, event_path)
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

        from app.services.ingestion.document_processor import CONTENT_TYPE_MAP
        content_type = CONTENT_TYPE_MAP.get(ext, "application/octet-stream")

        if file_hash is None:
            file_hash = self._handler._compute_hash(event_path)
        if not file_hash:
            return

        db: Session = SessionLocal()
        try:
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
            future = self._executor.submit(
                self._run_ingestion,
                event_path,
                fname,
                None,  # kb_id
                task.id,
                doc.id,
                datastore_id,
                None,
                file_hash=file_hash,
                file_size=file_size,
                content_type=content_type,
            )
            future.add_done_callback(
                lambda f, ds=datastore_id: self._on_scan_ingestion_done(f, task.id, event_path, ds)
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

        from app.services.ingestion.document_processor import CONTENT_TYPE_MAP
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
            future = self._executor.submit(
                self._run_ingestion,
                event_path,
                fname,
                None,
                task.id,
                document_id,
                datastore_id,
                None,
                file_hash=file_hash,
                file_size=file_size,
                content_type=content_type,
            )
            future.add_done_callback(
                lambda f, ds=datastore_id: self._on_scan_ingestion_done(f, task.id, event_path, ds)
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

    def get_status(self) -> Dict[str, Any]:
        """Return current watcher state for the admin status endpoint."""
        with self._handler._lock:
            processing = self._handler._processing
        with self._active_scans_lock:
            active_scans = list(self._active_scans.values())
        with self._datastore_paths_lock:
            datastore_paths = dict(self._datastore_paths)
        return {
            "running": self._running,
            "last_scan_at": self._last_scan_at,
            "files_scanned": self._files_scanned,
            "active_scans": active_scans,
            "datastores": [
                {
                    "datastore_id": ds_id,
                    "path": datastore_paths.get(ds_id, "unknown"),
                    "pending_changes": len(self._handler.pending_changes.get(ds_id, [])),
                    "min_interval_seconds": self._handler.folder_paths.get(ds_id, (None, None, 300))[2],
                    "processing": ds_id in processing if ds_id in datastore_paths else False,
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

            with self._datastore_paths_lock:
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

        # Process each change and update last_event_processed so UI
        # reflects event-driven ingestion progress.
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

        # Update last_event_processed so UI reflects latest state.
        # The handler's _update_scan_progress increments last_event_processed
        # atomically and is protected by _progress_lock.
        if changes_processed > 0:
            self._handler._update_scan_progress(datastore_id, changes_processed)

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
        """Increment last_scan_processed by the given delta.

        Called during a scan after each file is processed. Uses SQL-level
        atomic increment (UPDATE ... SET col = col + :val) so concurrent
        event-driven ingestion cannot lose a counter increment.

        Also updates the in-memory active scan so the polling endpoint sees
        progress immediately.
        """
        db: Session = SessionLocal()
        try:
            with self._progress_lock:
                db.execute(
                    update(DataStore)
                    .where(DataStore.id == datastore_id)
                    .values(last_scan_processed=DataStore.last_scan_processed + processed)
                )

                with self._active_scans_lock:
                    for sid, info in self._active_scans.items():
                        if info["datastore_id"] == datastore_id:
                            info["processed"] = info.get("processed", 0) + processed
                            break

                db.commit()

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

    def _on_scan_ingestion_done(
        self, future, task_id: int, event_path: str, datastore_id: int,
    ) -> None:
        """Callback after scan-submitted ingestion completes.

        Delegates to _on_ingestion_done for logging, then increments the
        scan's processed counter so the UI progress reflects actual
        completion rather than mere submission.
        """
        self._on_ingestion_done(future, task_id, event_path)
        # Only increment progress if the scan is still running.
        # If cancelled, _cancel_scan already set status to "cancelled" and
        # we don't want to muddy the progress counter.
        if not self._is_scan_cancelled(datastore_id):
            self._update_scan_progress(datastore_id, 1)
