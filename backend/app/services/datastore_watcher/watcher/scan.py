"""Scan orchestration — init, cancel, progress tracking, and discovery updates.

Provides ``ScanMixin`` for ``DataStoreWatcher``. Manages the full lifecycle
of a manual datastore scan: initializing scan state, tracking progress,
cancelling/pausing, running the discovery engine, processing new/modified
files, handling deletions, and waiting for ingestion completion.

Scan progress is tracked both in-memory (``_active_scans`` dict for SSE)
and in the database (``last_scan_processed``, ``last_scan_total_files``).
A shared ``_progress_lock`` prevents races between event-driven ingestion
and manual scans.

Methods:
- _next_scan_id: thread-safe scan ID counter
- _init_scan: initialize scan state in DB and memory
- _update_scan_progress: atomically increment last_scan_processed
- _complete_scan: mark scan as completed/paused/cancelled
- _cancel_scan: stop a running scan (pause or full cancel)
- _is_scan_cancelled: check if scan should stop processing
- _cleanup_stale_scans: remove old completed scans from memory
- _count_files_in_folder: delegate to shared utility
- _run_discovery_and_update_sse: run discovery engine, update SSE state
- scan_single_datastore: full manual scan of one datastore
- _ingest_new_and_modified: process new/modified files during scan
- _process_deletions: handle deleted files during scan
- _wait_for_ingestion: wait for all ingestion futures to complete
- _matches_pattern: delegate to shared pattern-matching utility
"""

from __future__ import annotations

import logging
import os
import time as time_module
from concurrent.futures import Future
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import update, func
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.datastore import DataStore
from app.models.knowledge import Document, ProcessingTask, DocumentChunk
from app.services.discovery import discover_datastore

logger = logging.getLogger(__name__)


class ScanMixin:
    """Scan init/cancel/progress, single-datastore scan, and discovery update."""

    # ------------------------------------------------------------------
    # Scan ID management (thread-safe)
    # ------------------------------------------------------------------

    def _next_scan_id(self) -> int:
        with self._scan_id_lock:
            self._scan_id_counter += 1
            return self._scan_id_counter

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

            # Capture previous status before overwriting — needed to
            # distinguish a resume (paused) from a fresh scan.
            previous_status = ds.last_scan_status

            # Record scan status
            ds.last_scan_status = "running"
            ds.last_scan_at = datetime.now(timezone.utc)
            ds.last_scan_error = None
            # Clear graph ingestion pause flag — starting a scan means
            # graph builds should run too. Pause sets this flag; resume
            # (which calls this scan endpoint) clears it.
            if ds.graph_ingestion_paused:
                ds.graph_ingestion_paused = False
                logger.info(
                    "[WATCHER] graph_ingestion_resumed datastore_id=%d (cleared by scan init)",
                    datastore_id,
                )

            # Count selected files — the scan only processes selected files,
            # so the progress denominator is the number of selected documents
            # that need work (new, modified, or needs_reprocess).
            # We also count files on disk for the total_files display.
            total_files_on_disk = self._count_files_in_folder(ds.folder_path, ds.scan_pattern)
            selected_count = (
                db.query(func.count(Document.id))
                .filter(
                    Document.data_store_id == datastore_id,
                    Document.is_selected == True,
                )
                .scalar()
            ) or 0
            ds.last_scan_total_files = total_files_on_disk

            # On resume from pause, start the progress counter at the number
            # of already-completed selected documents.  Without this, the UI
            # shows 0/16 even though 4 files were fully ingested before the
            # pause — the actual ingestion correctly skips them (discovery
            # manifest comparison), but the counter makes it look like a
            # restart from scratch.
            if previous_status == "paused":
                completed_count = (
                    db.query(func.count(Document.id))
                    .filter(
                        Document.data_store_id == datastore_id,
                        Document.is_selected == True,  # noqa: E712
                        Document.chunks.any(),
                    )
                    .scalar()
                ) or 0
                ds.last_scan_processed = completed_count
                logger.info(
                    "[WATCHER] scan_resume scan_id=%d datastore_id=%d completed_before=%d total=%d",
                    scan_id, datastore_id, completed_count, selected_count,
                )
            else:
                completed_count = 0
                ds.last_scan_processed = 0

            db.commit()

            # Track in memory — the SSE endpoint always finds the most
            # recently added scan for a given datastore by iterating
            # _active_scans in reverse insertion order (Python 3.7+).
            # "total" is the number of selected files (progress denominator).
            # "total_files_on_disk" is the total files in the folder (for display).
            with self._active_scans_lock:
                self._active_scans[scan_id] = {
                    "datastore_id": datastore_id,
                    "total": selected_count,
                    "total_files_on_disk": total_files_on_disk,
                    "processed": completed_count,
                    "status": "running",
                    "error_count": 0,
                    "new": 0,
                    "modified": 0,
                    "skipped": 0,
                    "error_message": None,  # string error message from _complete_scan
                }
            logger.info(
                "[WATCHER] scan_init scan_id=%d datastore_id=%d selected_files=%d total_on_disk=%d status=running",
                scan_id, datastore_id, selected_count, total_files_on_disk,
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

    def _complete_scan(self, datastore_id: int, success: bool, error: Optional[str] = None) -> None:
        """Mark a scan as completed.

        If the scan was paused or cancelled (status already changed by
        _cancel_scan), preserve that status instead of overwriting it.
        """
        db: Session = SessionLocal()
        try:
            ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
            if not ds:
                return

            # If the scan was paused or stopped, _cancel_scan already set
            # the appropriate status. Do NOT overwrite it.
            current_status = ds.last_scan_status
            if current_status in ("paused", "idle"):
                logger.info(
                    "[WATCHER] complete_scan_skip datastore_id=%d status=%s — preserving",
                    datastore_id, current_status,
                )
                # Still update _active_scans so SSE can close
                with self._active_scans_lock:
                    for sid, info in self._active_scans.items():
                        if info["datastore_id"] == datastore_id:
                            info["status"] = current_status
                            info["_completed_at"] = time_module.time()
                            break
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

    def _cancel_scan(self, datastore_id: int, pause: bool = False) -> bool:
        """Cancel a running scan on a datastore. Returns True if cancelled.

        When *pause* is False (default): sets status to "idle" — the scan
        is fully stopped and must be re-triggered from scratch.

        When *pause* is True: sets status to "paused" — the scan stops
        but can be resumed by calling the scan endpoint again. The scan
        is idempotent: already-processed files (with manifest entries and
        existing chunks) are skipped on resume.

        Cancels:
        - Prevents new files from being submitted (status → cancelled/paused)
        - Cancels all in-flight ingestion futures for this scan
        - Cancels all pending and in-flight graph builds for this datastore
        """
        db: Session = SessionLocal()
        try:
            ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
            if not ds or ds.last_scan_status != "running":
                return False

            if pause:
                ds.last_scan_status = "paused"
                ds.last_scan_error = "Scan paused by admin"
                # Pause graph ingestion too — otherwise recovery's
                # _retry_pending_graph_builds will immediately re-queue
                # the graph builds we just cancelled, causing LLM calls
                # to continue while the user expects everything stopped.
                ds.graph_ingestion_paused = True
            else:
                ds.last_scan_status = "idle"
                ds.last_scan_error = "Scan cancelled by admin"
            db.commit()

            # Find this datastore in active scans — do NOT remove it. The
            # SSE endpoint may still be reading from _active_scans and needs
            # to find the cancelled entry to emit the final status event.
            # Stale scans are cleaned up in _init_scan before adding a new scan.
            cancelled_scan_ids: list[int] = []
            scan_status = "paused" if pause else "cancelled"
            scan_msg = "Scan paused by admin" if pause else "Scan cancelled by admin"
            with self._active_scans_lock:
                for sid, info in self._active_scans.items():
                    if info["datastore_id"] == datastore_id:
                        info["status"] = scan_status
                        info["error_message"] = scan_msg
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
                "[WATCHER] scan_%s datastore_id=%d futures=%d graph_builds=%d",
                "paused" if pause else "cancelled",
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

    def _run_discovery_and_update_sse(
        self,
        datastore_id: int,
        force_full_hash: bool,
        summary: Dict[str, Any],
    ) -> Optional[Any]:
        """Run discovery engine and update SSE state with counts.

        Returns the discovery result, or None if discovery failed
        (in which case summary['errors'] is set and the scan is marked
        as failed via _complete_scan).
        """
        try:
            result = discover_datastore(datastore_id, force_full_hash=force_full_hash)
        except Exception as e:
            logger.error(
                "[WATCHER] discovery_failed datastore_id=%d: %s", datastore_id, e,
                exc_info=True,
            )
            self._complete_scan(datastore_id, False, str(e))
            summary["errors"] = 1
            return None

        summary["scanned"] = result.total_files_discovered
        summary["skipped"] = result.skipped_files
        summary["new"] = len(result.new_files)
        summary["modified"] = len(result.modified_files)
        summary["deleted"] = len(result.deleted_files)

        with self._active_scans_lock:
            for sid, scan_info in self._active_scans.items():
                if scan_info["datastore_id"] == datastore_id:
                    scan_info["new"] = summary["new"]
                    scan_info["modified"] = summary["modified"]
                    scan_info["skipped"] = summary["skipped"]
                    scan_info["deleted"] = summary["deleted"]
                    scan_info["error_count"] = summary["errors"]
                    break

        return result

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
            result = self._run_discovery_and_update_sse(
                datastore_id, force_full_hash, summary,
            )
            if result is None:
                return summary

            # Collect futures from ingestion tasks
            ingestion_futures: List[Future] = []

            # Process new/modified files
            files_to_process = result.new_files + result.modified_files
            if self._ingest_new_and_modified(
                datastore_id, files_to_process, scan_id, summary, ingestion_futures,
            ):
                return summary

            # Expire all cached objects so we see the latest DB state.
            # The db session has been open since the start of the scan and
            # may have stale data from before the discovery/ingestion phase.
            db.expire_all()

            seen_paths = {fmeta["file_path"] for fmeta in files_to_process}

            # Re-queue documents with pending or failed tasks that were not
            # in the new/modified set.  This handles the pause/resume case:
            # after a pause, Document records exist but their ProcessingTasks
            # may be stuck in "pending" (never started) or "failed" (interrupted).
            # Without this, a resume scan finds 0 new files and exits immediately,
            # leaving those tasks orphaned.
            stuck_docs = (
                db.query(Document, ProcessingTask)
                .join(ProcessingTask, ProcessingTask.document_id == Document.id)
                .filter(
                    Document.data_store_id == datastore_id,
                    Document.is_selected == True,  # noqa: E712
                    ProcessingTask.status.in_(("pending", "failed", "processing")),
                )
                .all()
            )
            self._requeue_stuck_documents(
                db, datastore_id, stuck_docs, seen_paths, scan_id, summary, ingestion_futures,
            )

            # Re-queue selected documents that have NO ProcessingTask and NO
            # chunks.  This happens when a file was previously unselected
            # (skipped during a prior scan) and is now selected.  The discovery
            # engine classifies it as "unchanged" (manifest entry exists), so
            # it's not in new/modified files.  But it has no chunks and no
            # task, so it would be missed without this check.
            orphan_selected = (
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
            self._requeue_orphan_documents(
                db, datastore_id, orphan_selected, seen_paths, scan_id, summary, ingestion_futures,
            )

            # Re-queue selected documents with needs_reprocess=True.
            # This happens when an admin edited the markdown in the editor
            # and saved it.  The file itself is unchanged (same hash), so
            # discovery classifies it as "unchanged" and it's not in
            # new/modified files.  But the markdown was edited and needs
            # to be re-ingested using the saved markdown (skip_conversion=True).
            reprocess_docs = (
                db.query(Document)
                .filter(
                    Document.data_store_id == datastore_id,
                    Document.is_selected == True,  # noqa: E712
                    Document.needs_reprocess == True,  # noqa: E712
                )
                .all()
            )
            self._requeue_reprocess_documents(
                db, datastore_id, reprocess_docs, seen_paths, scan_id, summary, ingestion_futures,
            )

            # Process deleted files (files on disk that no longer exist)
            self._process_deletions(datastore_id, result.deleted_files, scan_id, summary)

            # Wait for all ingestion tasks to complete before marking scan done.
            # Each task covers: parse → embed → Qdrant upsert.  Graph build runs
            # in a separate daemon thread and does not block the scan.  10 minutes
            # per file covers large PDFs with OCR; a timeout is a real hang (API
            # down, DB locked), not a slow graph build.
            self._wait_for_ingestion(ingestion_futures, scan_id, datastore_id, summary)

            # Clean up futures tracking
            with self._scan_futures_lock:
                self._scan_futures.pop(scan_id, None)

            # Retry pending/failed graph builds for documents that completed
            # ingestion but whose graph extraction was never started or was
            # cancelled (e.g. by a pause).  Without this, a resume scan finds
            # 0 new files and exits immediately, leaving graph builds orphaned
            # at graph_status="pending" forever.  Graph builds run in daemon
            # threads and don't block scan completion.
            if not self._is_scan_cancelled(datastore_id):
                try:
                    # Reset "failed" graph builds that were cancelled by a
                    # pause back to "pending" so the retry picks them up.
                    # _check_graph_build_eligible in the dispatcher skips
                    # "failed" tasks, so we must reset them first.
                    graph_db = SessionLocal()
                    try:
                        failed_tasks = (
                            graph_db.query(ProcessingTask)
                            .filter(
                                ProcessingTask.data_store_id == datastore_id,
                                ProcessingTask.status == "completed",
                                ProcessingTask.graph_status == "failed",
                            )
                            .all()
                        )
                        reset_count = 0
                        for t in failed_tasks:
                            if t.graph_error and "Cancelled" in t.graph_error:
                                t.graph_status = "pending"
                                t.graph_error = None
                                reset_count += 1
                        if reset_count:
                            graph_db.commit()
                            logger.info(
                                "[WATCHER] graph_reset_failed scan_id=%d datastore_id=%d reset=%d",
                                scan_id, datastore_id, reset_count,
                            )
                    finally:
                        graph_db.close()

                    from app.services.discovery.startup_recovery_service import StartupRecoveryService
                    StartupRecoveryService()._retry_pending_graph_builds(datastore_id)
                except Exception as e:
                    logger.warning(
                        "[WATCHER] graph_retry_failed scan_id=%d datastore_id=%d: %s",
                        scan_id, datastore_id, e,
                    )

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

    def _ingest_new_and_modified(
        self,
        datastore_id: int,
        files_to_process: List[Dict[str, Any]],
        scan_id: int,
        summary: Dict[str, Any],
        ingestion_futures: List[Future],
    ) -> bool:
        """Process new/modified files during a scan.

        Returns True if the scan was cancelled mid-phase (caller should
        return immediately), False otherwise.
        """
        # Pre-filter: only process selected files (or new files on
        # auto-process datastores).  Without this, the scan iterates
        # over every file on disk — including the 7700 unselected ones —
        # calling _handle_file_in_scan which skips them, but still
        # incrementing the progress counter and spamming the log with
        # "file_unselected" messages.  This makes the progress counter
        # exceed the total (e.g. 3616/372) and delays actual ingestion
        # of the selected files.
        db = SessionLocal()
        try:
            ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
            auto_select = ds.auto_process_enabled if ds else False

            # Build a set of selected file paths for fast lookup.
            selected_paths = set(
                row[0] for row in
                db.query(Document.file_path)
                .filter(
                    Document.data_store_id == datastore_id,
                    Document.is_selected == True,  # noqa: E712
                )
                .all()
            )
        finally:
            db.close()

        filtered = []
        skipped_unselected = 0
        for fmeta in files_to_process:
            fpath = fmeta["file_path"]
            if fpath in selected_paths:
                filtered.append(fmeta)
            elif fpath not in selected_paths and auto_select:
                # New file on auto-process datastore — no Document record
                # yet, but auto_select means it should be ingested.
                filtered.append(fmeta)
            else:
                skipped_unselected += 1

        if skipped_unselected > 0:
            logger.info(
                "[WATCHER] scan_filtered_unselected datastore_id=%d skipped=%d selected_to_process=%d",
                datastore_id, skipped_unselected, len(filtered),
            )

        for fmeta in filtered:
            if self._is_scan_cancelled(datastore_id):
                logger.info("[WATCHER] scan_cancelled mid-scan datastore_id=%d", datastore_id)
                self._complete_scan(datastore_id, False, "Scan cancelled by admin")
                summary["errors"] = 1
                return True

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

        return False

    def _process_deletions(
        self,
        datastore_id: int,
        deleted_files: List[Dict[str, Any]],
        scan_id: int,
        summary: Dict[str, Any],
    ) -> None:
        """Process deleted files (files on disk that no longer exist)."""
        for fmeta in deleted_files:
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

    def _wait_for_ingestion(
        self,
        ingestion_futures: List[Future],
        scan_id: int,
        datastore_id: int,
        summary: Dict[str, Any],
    ) -> None:
        """Wait for all ingestion tasks to complete before marking scan done.

        Each task covers: parse → embed → Qdrant upsert.  Graph build runs
        in a separate daemon thread and does not block the scan.

        Uses a per-future timeout of 10 minutes (enough for any single file
        including OCR) with a total cap of 4 hours (prevents infinite
        blocking if the worker pool is dead).  When a future times out or
        fails, the progress counter is still incremented so the UI doesn't
        freeze, and the task is marked as failed in the DB so the next
        scan can re-queue it.
        """
        if not ingestion_futures:
            return

        logger.info(
            "[WATCHER] waiting_for_ingestion scan_id=%d datastore_id=%d tasks=%d",
            scan_id, datastore_id, len(ingestion_futures),
        )

        per_future_timeout = 600  # 10 min per file (covers large PDFs + OCR)
        total_deadline = time_module.monotonic() + 14400  # 4 hour total cap

        for future in ingestion_futures:
            total_remaining = total_deadline - time_module.monotonic()
            if total_remaining <= 0:
                # Total cap exhausted — mark remaining futures as errors
                logger.error(
                    "[WATCHER] ingestion_total_cap_exhausted scan_id=%d — marking remaining %d tasks as failed",
                    scan_id, len(ingestion_futures) - ingestion_futures.index(future),
                )
                future.cancel()
                self._mark_task_failed_for_future(future, datastore_id)
                summary["errors"] += 1
                self._update_scan_progress(datastore_id, 1)
                continue

            # Use the smaller of per-future timeout and total remaining
            timeout = min(per_future_timeout, total_remaining)
            try:
                future.result(timeout=timeout)
                # Success — progress was already incremented by the
                # _on_scan_ingestion_done callback.
            except TimeoutError:
                logger.error(
                    "[WATCHER] ingestion_task_timeout scan_id=%d — cancelling future",
                    scan_id,
                )
                future.cancel()
                self._mark_task_failed_for_future(future, datastore_id)
                summary["errors"] += 1
                self._update_scan_progress(datastore_id, 1)
            except Exception as e:
                logger.error(
                    "[WATCHER] ingestion_task_failed scan_id=%d: %s",
                    scan_id, e,
                )
                self._mark_task_failed_for_future(future, datastore_id)
                summary["errors"] += 1
                self._update_scan_progress(datastore_id, 1)

    def _mark_task_failed_for_future(self, future: Future, datastore_id: int) -> None:
        """Best-effort: mark the ProcessingTask for a failed/timed-out future as 'failed'.

        The future itself doesn't carry the task_id, but the task was
        submitted via _submit_ingestion which stores task_id in the
        future's context.  We query the DB for processing tasks on this
        datastore that are still in 'processing' state and mark them
        failed — the next scan will re-queue them via _requeue_stuck_documents.
        """
        db = SessionLocal()
        try:
            stuck = (
                db.query(ProcessingTask)
                .filter(
                    ProcessingTask.data_store_id == datastore_id,
                    ProcessingTask.status == "processing",
                )
                .all()
            )
            for t in stuck:
                t.status = "failed"
                t.error_message = "Ingestion timed out or worker died"
            if stuck:
                db.commit()
                logger.info(
                    "[WATCHER] marked_stuck_tasks_failed datastore_id=%d count=%d",
                    datastore_id, len(stuck),
                )
        except Exception as e:
            logger.warning("[WATCHER] mark_stuck_failed error: %s", e)
        finally:
            db.close()

    def _matches_pattern(self, filepath: str, pattern: str = "*") -> bool:
        """Check if a filepath matches the scan pattern. Delegates to shared utility."""
        from app.services.datastore_watcher.utils import matches_pattern
        return matches_pattern(filepath, pattern)

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
