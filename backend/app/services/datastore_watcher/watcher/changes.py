"""Change callbacks — batch processing, file delegation, and ingestion completion tracking.

Provides ``ChangesMixin`` for ``DataStoreWatcher``. Contains the callback
methods that the handler invokes when a batch of file changes is ready
for processing, plus the ingestion-completion callbacks that update scan
progress as each background ingestion job finishes.

Methods:
- _on_changes: batch callback — process each change via handler, update progress
- _handle_file: delegate to handler's _handle_file for event-driven ingestion
- _on_ingestion_done: log ingestion future completion/failure
- _on_scan_ingestion_done: log completion + increment scan progress counter
"""

from __future__ import annotations

import logging
from concurrent.futures import Future
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ChangesMixin:
    """Batch change callback, file delegation, and ingestion done callbacks."""

    def _on_changes(self, datastore_id: int, org_id: int, changes: List[Dict[str, Any]]) -> None:
        """Callback when batch of changes is ready to process.

        Args:
            datastore_id: ID of the datastore with changes
            org_id: Organization ID this datastore belongs to
            changes: List of change events with path and event_type
        """
        if not changes:
            return

        logger.debug(
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
            logger.debug(
                "[WATCHER] ingestion_completed task_id=%s path=%s",
                task_id,
                event_path,
            )

    def _on_scan_ingestion_done(
        self, future, task_id: int, event_path: str, datastore_id: int,
    ) -> None:
        """Callback after scan-submitted ingestion completes.

        Delegates to _on_ingestion_done for logging, then increments the
        scan's processed counter — but only for successful completions.
        Failed/timed-out futures are counted by _wait_for_ingestion to
        avoid double-counting.
        """
        self._on_ingestion_done(future, task_id, event_path)
        if future.exception() is not None:
            # Failed — _wait_for_ingestion will increment the counter
            return
        if not self._is_scan_cancelled(datastore_id):
            self._update_scan_progress(datastore_id, 1)
