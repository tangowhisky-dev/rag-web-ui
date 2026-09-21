"""Pending change queue and batch callback — event-driven processing pipeline.

Provides ``ChangesMixin`` for ``DatastoreFileEventHandler``. Manages the
per-datastore pending-changes queue, processes batches when timers fire
or flush is requested, and tracks ingestion completion for event-driven
progress updates.

Methods:
- _queue_change: add a change to the pending queue (deduplicates by path)
- _process_pending_changes: drain the queue and invoke the batch callback
- force_process_pending: flush pending changes (used on shutdown)
- _on_changes: batch callback — process each change, wait for futures, update progress
- _on_ingestion_done: log ingestion future completion/failure
- _update_scan_progress: atomically increment last_event_processed for UI progress
"""

from __future__ import annotations

import logging
from concurrent.futures import Future
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from app.db.session import SessionLocal
from app.models.datastore import DataStore

logger = logging.getLogger(__name__)


class ChangesMixin:
    """Pending change queue, batch callback, and ingestion tracking."""

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
            logger.debug(
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
            logger.debug(
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

        logger.debug(
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

    def _update_scan_progress(self, datastore_id: int, processed: int) -> None:
        """Update last_event_processed so UI reflects event-driven ingestion progress.

        Called after event-driven ingestion completes. Uses SQL-level atomic
        increment (UPDATE ... SET col = col + :val) so concurrent threads
        cannot lose a counter increment due to a read-modify-write race.

        This is separate from manual scan progress (last_scan_processed) so
        the UI can distinguish event-driven vs manual scan activity.
        """
        db = SessionLocal()
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

    def _process_orphan_selected(self, datastore_id: int) -> None:
        """Find and ingest selected documents that need ingestion on an
        auto-process datastore.

        Picks up two categories:
        1. Selected documents with no ProcessingTask and no chunks — files
           selected via the datastore browser that were never ingested.
        2. Selected documents with a "pending" task, no chunks, and no
           Redis claim — queued work whose executor future was lost (scan
           cancelled, app restart, or an admin's retry-failed flip).  The
           claim registry answers "is anyone holding this task" exactly,
           so no age heuristics are needed.  "failed" tasks are deliberately
           excluded — they are terminal verdicts retried only via
           retry-failed.

        Called by the batch timer on each interval tick.
        """
        from app.models.knowledge import Document, ProcessingTask
        from app.services.infrastructure.ingest_claims import get_claimed_task_ids
        from sqlalchemy.orm import Session as _Session

        db: _Session = SessionLocal()
        try:
            # Category 1: no ProcessingTask at all, no chunks
            orphans = (
                db.query(Document)
                .outerjoin(ProcessingTask, ProcessingTask.document_id == Document.id)
                .filter(
                    Document.data_store_id == datastore_id,
                    Document.is_selected == True,  # noqa: E712
                    ProcessingTask.id.is_(None),
                    ~Document.chunks.any(),
                )
                .all()
            )

            # Category 2: pending task with no chunks and no live claim —
            # the task's executor future is gone, so it's safe to re-submit.
            pending_rows = (
                db.query(Document, ProcessingTask)
                .join(ProcessingTask, ProcessingTask.document_id == Document.id)
                .filter(
                    Document.data_store_id == datastore_id,
                    Document.is_selected == True,  # noqa: E712
                    ProcessingTask.status == "pending",
                    ~Document.chunks.any(),
                )
                .all()
            )
            claimed = get_claimed_task_ids(t.id for _, t in pending_rows)
            pending_docs = [doc for doc, t in pending_rows if t.id not in claimed]
        finally:
            db.close()

        # Deduplicate by document id
        seen_ids: set = set()
        to_process = []
        for doc in [*orphans, *pending_docs]:
            if doc.id not in seen_ids:
                seen_ids.add(doc.id)
                to_process.append(doc)

        if not to_process:
            return

        logger.debug(
            "[WATCHER] orphan_selected_found datastore_id=%d orphans=%d pending=%d",
            datastore_id, len(orphans), len(pending_docs),
        )

        futures = []
        for doc in to_process:
            try:
                future = self._handle_file(doc.file_path, datastore_id, "modified")
                if future is not None:
                    futures.append(future)
            except Exception as e:
                logger.error(
                    "[WATCHER] orphan_ingest_error doc_id=%s path=%s: %s",
                    doc.id, doc.file_path, e,
                )

        # Don't block the batch timer thread waiting for ingestion to
        # complete.  Futures run in the executor's thread pool; the UI
        # polls for processing/pending_ingestion status every 3 seconds.
        # Blocking here prevents the next interval tick from firing and
        # delays processing of new filesystem events.  Orphaned pending
        # tasks are retried on the next tick via the query above.
        if futures:
            logger.debug(
                "[WATCHER] orphan_ingestion_submitted datastore_id=%d count=%d",
                datastore_id, len(futures),
            )
