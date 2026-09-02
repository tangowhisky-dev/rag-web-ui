"""Event dispatch — watchdog callbacks and delayed-dispatch logic.

Provides ``DispatchMixin`` for ``DatastoreFileEventHandler``. Receives
raw watchdog file-system events (created, modified, deleted, moved),
resolves the owning datastore, applies per-file debouncing, then
queues the change for processing after a 1-second write-completion delay.

Methods:
- on_created / on_modified / on_deleted / on_moved: watchdog event callbacks
- dispatch: observer entry point (overrides FileSystemEventHandler.dispatch)
- _dispatch: applies debouncer, spawns delayed-dispatch thread
- _should_process / _after_process: per-file debounce window management
"""

from __future__ import annotations

import logging
import os
import threading
import time as time_module
from typing import Optional

logger = logging.getLogger(__name__)


class DispatchMixin:
    """Watchdog event callbacks and event dispatch logic."""

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
        logger.debug(
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
        logger.debug(
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
        logger.debug(
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
                logger.debug(
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
                logger.debug(
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
            logger.debug(
                "[WATCHER] event_moved_out_of_watch path=%s datastore_id=%s reason=not_watched",
                event.dest_path, src_datastore_id,
            )

    def _dispatch(self, path: str, event_type: str) -> None:
        """Apply debouncer then queue the change."""
        # Filter hidden/temp/unsupported files before queueing so they
        # don't inflate the pending_changes count shown in the UI.
        # Deletions are never filtered — we need to clean up DB records
        # even for files that would have been skipped on ingest.
        if event_type != "deleted":
            fname = os.path.basename(path)
            _, ext = os.path.splitext(fname)
            ext = ext.lower()
            if self._should_skip_file(fname, ext, path):
                return

        if self.debouncer is not None:
            coalesced = self.debouncer.touch(path, event_type)
            if coalesced is None:
                logger.debug(
                    "[WATCHER] event_coalesced path=%s",
                    path,
                )
                return  # debounced

        # Delay processing by 1 second to allow file write to complete
        logger.debug(
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
                    from app.services.datastore_watcher.handler.handler import _SyntheticEvent
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
