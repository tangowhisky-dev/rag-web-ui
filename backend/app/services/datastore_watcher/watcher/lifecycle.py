"""Lifecycle management — health checks, post-restart scans, and thread management.

Provides ``LifecycleMixin`` for ``DataStoreWatcher``. Runs a background
health-check thread that monitors the watchdog observer and restarts it
if it dies unexpectedly (inotify exhaustion, too many open files, etc.).
After a restart, triggers a discovery scan for all active datastores to
catch file changes that happened while the observer was dead.

Also performs periodic cleanup of stale completed scans from
``_active_scans`` to prevent unbounded memory growth.

Methods:
- _health_check_loop: monitor observer thread, restart on death, clean stale scans
- _trigger_post_restart_scan: scan all active datastores after observer restart
"""

from __future__ import annotations

import logging
import threading
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.datastore import DataStore

logger = logging.getLogger(__name__)


class LifecycleMixin:
    """Health check, post-restart scan, and thread management."""

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
                logger.debug("[WATCHER] observer restarted successfully")

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
            logger.debug("[WATCHER] post_restart_scan_complete")

        t = _threading.Thread(target=_scan_all, name="post-restart-scan", daemon=True)
        t.start()
