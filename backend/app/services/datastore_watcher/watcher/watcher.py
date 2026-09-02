"""DataStoreWatcher — composed watcher class with lifecycle and sync methods.

Combines all watcher mixins into a single ``DataStoreWatcher`` class that
manages the watchdog observer, scan lifecycle, and progress tracking.

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

Mixin composition (MRO order):
1. LifecycleMixin — health check, post-restart scan, thread management
2. ScanMixin — scan init/cancel/progress, single-datastore scan, discovery
3. IngestionMixin — requeue, file ingestion in scan, ingestion tracking
4. ChangesMixin — batch callback, file delegation, ingestion done callbacks

This module also re-exports ``SessionLocal`` for backward compatibility
with tests that import it from ``app.services.datastore_watcher.watcher``.
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.datastore import DataStore, OrganizationDataStore
from app.services.datastore_watcher.handler import (
    DatastoreFileEventHandler,
    _Debouncer,
)
from app.services.datastore_watcher.watcher.lifecycle import LifecycleMixin
from app.services.datastore_watcher.watcher.scan import ScanMixin
from app.services.datastore_watcher.watcher.ingestion import IngestionMixin
from app.services.datastore_watcher.watcher.changes import ChangesMixin

logger = logging.getLogger(__name__)


class DataStoreWatcher(
    LifecycleMixin,
    ScanMixin,
    IngestionMixin,
    ChangesMixin,
):
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
        # Read INGESTION_CONCURRENCY from settings (default 16).
        # reload="restart" — changing the setting requires a backend restart.
        max_workers = self._read_ingestion_concurrency()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="watcher"
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

    @staticmethod
    def _read_ingestion_concurrency() -> int:
        """Read INGESTION_CONCURRENCY from the settings table.

        Falls back to 16 if the setting is missing or the DB is not yet
        available (e.g. during early startup before migrations).
        """
        try:
            from app.services.settings_service import get_setting
            db = SessionLocal()
            try:
                val = get_setting(db, "INGESTION_CONCURRENCY", None)
                if val and isinstance(val, int) and 1 <= val <= 32:
                    return val
            finally:
                db.close()
        except Exception:
            pass
        return 8

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
                logger.debug(
                    "[WATCHER] observer started (PollingObserver with "
                    "recursive=True, WATCH_POLL_INTERVAL=%ds, "
                    "WATCHER_USE_INOTIFY=%s)",
                    poll_interval,
                    settings.WATCHER_USE_INOTIFY,
                )
            else:
                self._observer = Observer(timeout=poll_interval)
                self._observer.start()
                logger.debug(
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
        logger.debug("[WATCHER] observer registered on root=/app/data (recursive=True)")

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

        logger.debug("[WATCHER] service started")

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
        logger.debug("[WATCHER] service stopped")

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
                logger.debug(
                    "[WATCHER] add_datastore_already_watching datastore_id=%s — updating org_id",
                    datastore_id,
                )
                self._handler.add_folder(datastore_id, org_id, abs_path, interval_minutes * 60)
                return

        # Register the datastore in the handler's folder_paths map
        self._handler.add_folder(datastore_id, org_id, abs_path, interval_minutes * 60)
        with self._datastore_paths_lock:
            self._datastore_paths[datastore_id] = str(abs_path)

        logger.debug(
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

        logger.debug("[WATCHER] datastore_removed datastore_id=%s", datastore_id)

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

        Registers all active, auto_process_enabled datastores — regardless of
        whether they have org assignments. Unassigned datastores are watched
        with org_id=None; file processing works identically since org_id is
        only used for logging and KB-deletion cleanup (which already guards
        on org_id is not None)."""
        db = SessionLocal()
        try:
            # Build org_id lookup from assignments (unassigned → None)
            assignment_map: Dict[int, int] = {}
            for a in (
                db.query(OrganizationDataStore)
                .filter(OrganizationDataStore.is_active == True)
                .all()
            ):
                assignment_map[a.data_store_id] = a.org_id

            # Query all active datastores — register ALL of them with the
            # handler so file events are resolved for every datastore.
            # auto_process_enabled=True: interval in minutes (batch processing)
            # auto_process_enabled=False: interval=-1 (track-only mode —
            #   update file counts but don't ingest)
            datastores = (
                db.query(DataStore)
                .filter(DataStore.is_active == True)
                .all()
            )

            datastore_ids = set()
            for ds in datastores:
                ds_id = ds.id
                datastore_ids.add(ds_id)
                org_id = assignment_map.get(ds_id)  # None for unassigned
                if ds.auto_process_enabled:
                    interval = ds.auto_process_interval_minutes or 60
                else:
                    interval = -1  # track-only mode
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
