"""Folder management — datastore registration and batch timer scheduling.

Provides ``FolderMixin`` for ``DatastoreFileEventHandler``. Handles the
datastore_id → (org_id, folder_path, min_interval) mapping that the
handler uses to resolve events, plus the per-datastore batch timers
that coalesce events in auto-process mode.

Methods:
- add_folder / remove_folder: register/unregister a datastore for monitoring
- _cancel_batch_timer / _schedule_batch_timer: per-datastore timer lifecycle
- _resolve_datastore: find which datastore owns an event path (longest-prefix match)
- _count_files_in_folder: delegate to shared utility for file counting
- _update_file_count: incremental file-count adjustment for track-only datastores
- _refresh_file_count: full filesystem recount for a datastore
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, Optional, Tuple

from app.db.session import SessionLocal
from app.models.datastore import DataStore

logger = logging.getLogger(__name__)


class FolderMixin:
    """Folder registration, batch timers, and datastore resolution."""

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
        # Schedule the batch timer immediately so that orphan selected
        # documents (e.g. files selected via the datastore browser) are
        # picked up on the next interval tick even if no filesystem
        # events arrive.  Without this, the timer only starts when the
        # first filesystem event triggers _dispatch → _schedule_batch_timer.
        if min_interval_seconds > 0:
            self._schedule_batch_timer(datastore_id)
        logger.debug(
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
                logger.debug(
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
        logger.debug("[WATCHER] handler_folder_removed datastore_id=%s", datastore_id)

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
            # Process accumulated filesystem changes when the timer fires
            self._process_pending_changes(datastore_id)
            # Also pick up orphan selected documents — files that were
            # selected via the datastore browser (save-selection) but
            # have no ProcessingTask and no chunks.  The watcher only
            # processes filesystem events, so without this check these
            # files would never be ingested on auto-process datastores.
            self._process_orphan_selected(datastore_id)
            # Reschedule for the next interval
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

        db = SessionLocal()
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
        db = SessionLocal()
        try:
            ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
            if not ds or not ds.folder_path:
                return

            count = self._count_files_in_folder(ds.folder_path, ds.scan_pattern)
            ds.last_scan_total_files = count
            db.commit()
        finally:
            db.close()

    def _count_files_in_folder(self, folder_path: str, scan_pattern: str = "*") -> int:
        """Count files matching pattern in folder. Delegates to shared utility."""
        from app.services.datastore_watcher.utils import count_files_in_folder
        return count_files_in_folder(folder_path, scan_pattern)
