"""Startup recovery service — background ingestion on app start.

Runs discovery on every active DataStore, queues new/modified files for
ingestion, and cleans up orphaned records for deleted files.  All work
is non-blocking: the FastAPI app is fully functional before recovery
completes.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from app.db.session import SessionLocal
from app.models.datastore import DataStore, DataStoreFileManifest
from app.models.knowledge import Document, DocumentChunk, ProcessingTask

logger = logging.getLogger(__name__)


class StartupRecoveryService:
    """Orchestrates background recovery for all active DataStores.

    On app start, one background thread is spawned per active datastore.
    Each thread runs a discovery pipeline that queues new/modified files
    for ingestion and deletes orphaned records for files that no longer
    exist on disk.
    """

    def __init__(self) -> None:
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=4)
        self._running: bool = True
        self._scan_id_counter: int = 0
        self._scan_id_lock = threading.Lock()
        # scan_id -> status dict
        self._active_scans: Dict[int, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch background recovery for active DataStores on startup.

        Recovery logic per datastore:
        - auto_process_enabled=True: always run full discovery + ingestion.
          The watcher was supposed to be processing files continuously; any
          downtime means missed events. Recovery fills that gap.
        - auto_process_enabled=False + last_scan_status="running": a manual
          scan was interrupted by the restart. Run discovery + ingestion to
          resume the interrupted work.
        - auto_process_enabled=False + no interrupted scan: skip discovery
          entirely. The user hasn't asked for processing — respect that.
          But still run _retry_pending_graph_builds to handle graph builds
          left pending/failed from a previous completed scan.

        Called from ``startup_event()`` after the database is ready.
        """
        logger.info("[RECOVERY] StartupRecoveryService.start() invoked")
        db: Session = SessionLocal()
        try:
            active = db.query(DataStore).filter(DataStore.is_active == True).all()  # noqa: E712

            # Do NOT reset "running" status to "completed" — it's the signal
            # that a scan was interrupted. The old code destroyed this evidence.
        except Exception as e:
            logger.warning("[RECOVERY] Could not query DataStores (migration may not be applied): %s", e)
            return
        finally:
            db.close()

        if not active:
            logger.info("[RECOVERY] No active DataStores found — skipping recovery")
            return

        for ds in active:
            should_discover = False
            reason = ""

            if ds.auto_process_enabled:
                should_discover = True
                reason = "auto_process_enabled"
            elif ds.last_scan_status == "running":
                should_discover = True
                reason = "interrupted_scan (last_scan_status=running)"
            else:
                # Check for interrupted ingestion tasks (pending/processing)
                task_db: Session = SessionLocal()
                try:
                    interrupted = (
                        task_db.query(ProcessingTask)
                        .filter(
                            ProcessingTask.data_store_id == ds.id,
                            ProcessingTask.status.in_(["pending", "processing"]),
                        )
                        .count()
                    )
                    if interrupted > 0:
                        should_discover = True
                        reason = f"interrupted_tasks ({interrupted} pending/processing)"
                finally:
                    task_db.close()

            if should_discover:
                logger.info(
                    "[RECOVERY] recovery_start datastore_id=%s name=%s reason=%s",
                    ds.id, ds.name, reason,
                )
                scan_id = self._next_scan_id()
                self._active_scans[scan_id] = {
                    "datastore_id": ds.id,
                    "datastore_name": ds.name,
                    "status": "running",
                    "scan_id": scan_id,
                    "total_files": 0,
                    "processed_files": 0,
                    "new_files": 0,
                    "modified_files": 0,
                    "deleted_files": 0,
                    "error_message": None,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                }
                self.executor.submit(self._discovery_pipeline_worker, ds.id, scan_id)
            else:
                # No discovery needed, but still retry pending graph builds
                # from any previously completed scan.
                logger.info(
                    "[RECOVERY] skip_discovery datastore_id=%s name=%s — no interrupted work, checking graph builds only",
                    ds.id, ds.name,
                )
                self.executor.submit(self._graph_only_worker, ds.id)

    def stop(self) -> None:
        """Signal shutdown and stop accepting new work."""
        logger.info("[RECOVERY] stop() invoked — setting _running=False")
        self._running = False
        self.executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def _get_status(self, datastore_id: int) -> Dict[str, Any]:
        """Return the current recovery status for a DataStore.

        Returns ``{"status": "idle"}`` when no scan is in progress.
        """
        return self.get_status(datastore_id)

    def get_status(self, datastore_id: int) -> Dict[str, Any]:
        """Return the current recovery status for a DataStore.

        Returns ``{"status": "idle"}`` when no scan is in progress.
        """
        for scan in self._active_scans.values():
            if scan.get("datastore_id") == datastore_id:
                return scan
        return {"status": "idle"}

    def get_all_status(self) -> List[Dict[str, Any]]:
        """Return recovery status for all active datastores.

        Returns a list of status dicts (sorted by scan_id) plus
        idle entries for datastores that have no scan in progress.
        """
        results: List[Dict[str, Any]] = []
        for scan in sorted(self._active_scans.values(), key=lambda s: s.get("scan_id", 0)):
            results.append(dict(scan))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_scan_id(self) -> int:
        with self._scan_id_lock:
            self._scan_id_counter += 1
            return self._scan_id_counter

    def _graph_only_worker(self, datastore_id: int) -> None:
        """Retry pending/failed graph builds without running discovery.

        Used for manual-scan datastores that have no interrupted scan but
        may have graph builds left pending from a previous completed scan.
        """
        try:
            self._retry_pending_graph_builds(datastore_id)
        except Exception as e:
            logger.warning(
                "[RECOVERY] graph_retry_failed datastore_id=%s: %s",
                datastore_id, e,
            )

    def _update_datastore_scan_fields(
        self, datastore_id: int, total_files: int | None = None,
        processed: int | None = None, status: str | None = None,
    ) -> None:
        """Update DataStore last_scan_* fields so the UI Status column
        reflects recovery progress in real time."""
        db: Session = SessionLocal()
        try:
            ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
            if not ds:
                return
            if total_files is not None:
                ds.last_scan_total_files = total_files
            if processed is not None:
                ds.last_scan_processed = processed
            if status is not None:
                ds.last_scan_status = status
                # Set last_scan_at when recovery starts so the UI doesn't
                # show "Completed never" — last_scan_at was null because
                # recovery never set it (only manual scans did).
                if status == "running" and ds.last_scan_at is None:
                    ds.last_scan_at = datetime.now(timezone.utc)
            db.commit()
        except Exception as e:
            logger.warning("[RECOVERY] Failed to update scan fields: %s", e)
            db.rollback()
        finally:
            db.close()

    def _increment_progress(self, scan_id: int) -> None:
        """Atomically increment processed_files for a scan."""
        current = self._active_scans[scan_id].get("processed_files", 0)
        self._active_scans[scan_id]["processed_files"] = current + 1

    def _discovery_pipeline_worker(self, datastore_id: int, scan_id: int) -> None:
        """Run the discovery pipeline in a background thread.

        Phases:
        1. Discovery — classify files as new/modified/deleted/unchanged.
        2. Queue ingestion for new + modified files (non-blocking).
        3. Handle deletions synchronously.
        4. Wait for all ingestion futures to complete, updating progress
           incrementally as each future resolves.
        5. Mark recovery complete only after all ingestion is done.
        """
        from concurrent.futures import Future
        from app.services.discovery import discover_datastore  # noqa: T100

        try:
            db: Session = SessionLocal()
            try:
                result: DiscoveryResult = discover_datastore(datastore_id)
            except Exception as e:
                logger.error("[RECOVERY] discovery_pipeline error for datastore_id=%s scan_id=%s: %s", datastore_id, scan_id, e, exc_info=True)
                self._active_scans[scan_id]["status"] = "error"
                self._active_scans[scan_id]["error_message"] = str(e)
                self._active_scans[scan_id]["processed_files"] = 0
                return
            finally:
                db.close()

            logger.info(
                "[RECOVERY] discovery_complete datastore_id=%s scan_id=%s new=%d modified=%d deleted=%d total=%d elapsed_ms=%.1f",
                datastore_id, scan_id,
                len(result.new_files), len(result.modified_files),
                len(result.deleted_files), result.total_files_discovered,
                result.elapsed_ms,
            )

            # Update status with discovery counts
            total_to_process = len(result.new_files) + len(result.modified_files)
            self._active_scans[scan_id]["total_files"] = total_to_process
            self._active_scans[scan_id]["new_files"] = len(result.new_files)
            self._active_scans[scan_id]["modified_files"] = len(result.modified_files)
            self._active_scans[scan_id]["deleted_files"] = len(result.deleted_files)

            # Update DataStore scan fields so the Status column reflects
            # recovery activity, not stale creation-time defaults.
            self._update_datastore_scan_fields(
                datastore_id,
                total_files=total_to_process,
                processed=0,
                status="running",
            )

            # Phase 2: Queue ingestion for new + modified files.
            # Collect futures so we can wait for actual completion.
            ingestion_futures: list[tuple[Future, str]] = []

            for fmeta in result.new_files:
                if not self._running:
                    break
                future = self.process_new_file(fmeta["file_path"], datastore_id)
                if future is not None:
                    ingestion_futures.append((future, fmeta["file_path"]))

            for fmeta in result.modified_files:
                if not self._running:
                    break
                future = self.process_new_file(fmeta["file_path"], datastore_id)
                if future is not None:
                    ingestion_futures.append((future, fmeta["file_path"]))

            # Phase 3: Handle deletions synchronously (fast — no embedding)
            for fmeta in result.deleted_files:
                if not self._running:
                    break
                self._handle_deletion_records(fmeta["file_path"], datastore_id)
                self._increment_progress(scan_id)

            # Phase 4: Wait for all ingestion futures to complete.
            # Update progress incrementally as each future resolves.
            completed_count = len(result.deleted_files)
            self._active_scans[scan_id]["processed_files"] = completed_count

            for future, fpath in ingestion_futures:
                if not self._running:
                    break
                try:
                    future.result(timeout=600)  # 10 min per file
                except Exception as e:
                    logger.error(
                        "[RECOVERY] ingestion_failed scan_id=%s path=%s: %s",
                        scan_id, fpath, e,
                    )
                completed_count += 1
                self._active_scans[scan_id]["processed_files"] = completed_count
                # Update DataStore last_scan_processed so the Status column
                # shows live progress.
                self._update_datastore_scan_fields(
                    datastore_id,
                    processed=completed_count,
                    status="running",
                )

            # Phase 5: Mark complete only after all ingestion is done.
            self._active_scans[scan_id]["status"] = "complete"
            logger.info(
                "[RECOVERY] recovery_complete datastore_id=%s scan_id=%s total=%d processed=%d",
                datastore_id, scan_id, total_to_process, completed_count,
            )

            # Set last_recovered_at timestamp on the DataStore
            recovered_at = datetime.now(timezone.utc)
            db2: Session = SessionLocal()
            try:
                ds_record = db2.query(DataStore).filter(DataStore.id == datastore_id).first()
                if ds_record:
                    ds_record.last_recovered_at = recovered_at
                    ds_record.last_scan_processed = completed_count
                    ds_record.last_scan_status = "completed"
                    ds_record.last_scan_at = recovered_at
                    db2.commit()
                    logger.info(
                        "[RECOVERY] recovery_timestamp_set datastore_id=%s last_recovered_at=%s",
                        datastore_id, ds_record.last_recovered_at.isoformat(),
                    )
            except Exception as e:
                logger.warning(
                    "[RECOVERY] Failed to set last_recovered_at datastore_id=%s: %s",
                    datastore_id, e,
                )
            finally:
                db2.close()

            # Store in scan dict so the API can return it
            self._active_scans[scan_id]["last_recovered_at"] = recovered_at.isoformat()

            # Retry graph builds that were left pending or failed from a
            # previous run.  This runs after discovery so new/modified file
            # ingestion is queued first.  Graph retry is best-effort —
            # failures are logged but don't fail the recovery.
            try:
                self._retry_pending_graph_builds(datastore_id)
            except Exception as e:
                logger.warning(
                    "[RECOVERY] graph_retry_failed datastore_id=%s: %s",
                    datastore_id, e,
                )

        except Exception as e:
            logger.error("[RECOVERY] recovery_error datastore_id=%s scan_id=%s: %s", datastore_id, scan_id, e, exc_info=True)
            self._active_scans[scan_id]["status"] = "error"
            self._active_scans[scan_id]["error_message"] = str(e)
            self._update_datastore_scan_fields(datastore_id, status="error")
            # Set last_recovered_at even on error so the UI shows the recovery
            # attempt timestamp rather than being indefinitely empty.
            try:
                db3: Session = SessionLocal()
                try:
                    ds_record = db3.query(DataStore).filter(DataStore.id == datastore_id).first()
                    if ds_record:
                        recovered_at = datetime.now(timezone.utc)
                        ds_record.last_recovered_at = recovered_at
                        self._active_scans[scan_id]["last_recovered_at"] = recovered_at.isoformat()
                        db3.commit()
                        logger.info(
                            "[RECOVERY] recovery_timestamp_set datastore_id=%s last_recovered_at=%s reason=error",
                            datastore_id, ds_record.last_recovered_at.isoformat(),
                        )
                except Exception:
                    db3.rollback()
                finally:
                    db3.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # New / Modified file ingestion
    # ------------------------------------------------------------------

    def process_new_file(self, file_path: str, datastore_id: int) -> Optional[Any]:
        """Queue a file for ingestion via the background processor.

        Returns the Future for the ingestion task, or None if the file was
        skipped (already handled, unreadable, or error).
        """
        file_name = os.path.basename(file_path)
        db: Session = SessionLocal()
        try:
            # Check if document already exists
            doc = (
                db.query(Document)
                .filter(Document.file_path == file_path, Document.data_store_id == datastore_id)
                .first()
            )

            # Compute hash and size
            try:
                file_hash = _sha256(file_path)
                file_size = os.path.getsize(file_path)
            except OSError as e:
                logger.warning("[RECOVERY] Cannot read file during ingestion queue: %s", e)
                return None

            if not file_hash:
                logger.warning("[RECOVERY] Cannot hash file, skipping: %s", file_path)
                return None

            task_id: int | None = None  # set by one of the branches below

            if doc:
                # Look for ANY non-completed task (including failed ones).
                # If a failed task exists, reuse it by resetting its state
                # rather than creating a duplicate.
                existing_task = (
                    db.query(ProcessingTask)
                    .filter(
                        ProcessingTask.document_id == doc.id,
                        ProcessingTask.status != "completed",
                    )
                    .first()
                )
                if existing_task:
                    if existing_task.status == "failed":
                        # Reuse the failed task — reset for re-ingestion
                        # Also update document metadata to current values.
                        doc.file_hash = file_hash
                        doc.file_size = file_size
                        doc.updated_at = datetime.now(timezone.utc)
                        existing_task.status = "pending"
                        existing_task.progress = 0
                        existing_task.progress_message = None
                        existing_task.error_message = None
                        existing_task.updated_at = datetime.now(timezone.utc)
                        task_id = int(existing_task.id)
                        logger.info(
                            "[RECOVERY] reused_task task_id=%s doc_id=%s",
                            existing_task.id, doc.id,
                        )
                    else:
                        # Has an active task (pending/processing) — skip
                        logger.info(
                            "[RECOVERY] skip_file_already_handled datastore_id=%s file_path=%s doc_id=%s task_id=%s",
                            datastore_id, file_path, doc.id, existing_task.id,
                        )
                        return None
                else:
                    # No existing task — update document metadata and create task
                    doc.file_hash = file_hash
                    doc.file_size = file_size
                    doc.updated_at = datetime.now(timezone.utc)
                    db.flush()
                    task_id = None  # signal below to create new task
                    logger.info(
                        "[RECOVERY] ingestion_queued datastore_id=%s file_path=%s doc_id=%s (new)",
                        datastore_id, file_path, doc.id,
                    )
            else:
                # Brand-new file — no Document record at all
                doc = Document(
                    file_path=file_path,
                    file_name=file_name,
                    file_size=file_size,
                    content_type=_guess_content_type(file_name),
                    file_hash=file_hash,
                    data_store_id=datastore_id,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )
                db.add(doc)
                db.flush()
                task_id = None
                logger.info(
                    "[RECOVERY] ingestion_queued datastore_id=%s file_path=%s doc_id=%s (new)",
                    datastore_id, file_path, doc.id,
                )

            # Create ProcessingTask (only if task_id not already set from a
            # reused failed task).
            if task_id is None:
                task = ProcessingTask(
                    document_id=doc.id,
                    data_store_id=datastore_id,
                    status="pending",
                    progress=0,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )
                db.add(task)
                db.commit()
                task_id = task.id

            # Submit to background processor (async)
            future = self._submit_ingestion(file_path, file_name, datastore_id, doc, file_hash, file_size, task_id)
            return future

        except Exception as e:
            logger.warning("[RECOVERY] Failed to queue file for ingestion: %s", e, exc_info=True)
            try:
                db.rollback()
            except Exception:
                pass
            return None
        finally:
            db.close()

    def _submit_ingestion(
        self,
        file_path: str,
        file_name: str,
        datastore_id: int,
        doc: Document,
        file_hash: str,
        file_size: int,
        task_id: int,
    ) -> Any:
        """Submit a file to the async background processor (non-blocking).

        Returns the Future so callers can wait for completion.
        """
        future = self.executor.submit(
            self._run_ingestion,
            file_path,
            file_name,
            datastore_id,
            doc.id,
            task_id,
            file_hash,
            file_size,
            doc.content_type,
        )
        future.add_done_callback(
            lambda f: self._on_ingestion_done(f, task_id, file_path)
        )
        return future

    def _run_ingestion(
        self,
        file_path: str,
        file_name: str,
        datastore_id: int,
        document_id: int,
        task_id: int,
        file_hash: Optional[str] = None,
        file_size: Optional[int] = None,
        content_type: Optional[str] = None,
    ) -> None:
        """Run the async ingestion pipeline in a threadpool worker.

        Delegates to the centralized ingestion dispatcher.
        """
        from app.services.ingestion.ingestion_dispatcher import run_ingestion_in_thread
        run_ingestion_in_thread(
            file_path=file_path,
            file_name=file_name,
            task_id=task_id,
            document_id=document_id,
            data_store_id=datastore_id,
            file_hash=file_hash,
            file_size=file_size,
            content_type=content_type,
        )

    def _on_ingestion_done(self, future, task_id: int, file_path: str) -> None:
        """Callback after recovery ingestion completes (success or failure)."""
        exc = future.exception()
        if exc:
            logger.error(
                "[RECOVERY] ingestion_future_error task_id=%s: %s",
                task_id,
                exc,
            )
        else:
            logger.info(
                "[RECOVERY] ingestion_completed task_id=%s path=%s",
                task_id,
                file_path,
            )

    # ------------------------------------------------------------------
    # Deleted file cleanup
    # ------------------------------------------------------------------

    def _handle_deletion_records(self, file_path: str, datastore_id: int) -> None:
        """Delete orphaned records for a file that no longer exists on disk.

        Mirrors ``DataStoreWatcher._handle_deletion`` for DataStore files
        exactly: delete Document, DocumentChunk, ProcessingTask, Qdrant
        vectors, Neo4j graph, and manifest entry.  Silently skips if the
        Document is already gone (no orphaned records to clean).
        """
        logger.info("[RECOVERY] deletion_start datastore_id=%s file_path=%s", datastore_id, file_path)
        db: Session = SessionLocal()
        try:
            doc = (
                db.query(Document)
                .filter(Document.file_path == file_path, Document.data_store_id == datastore_id)
                .first()
            )
            if not doc:
                logger.info(
                    "[RECOVERY] deletion_skip datastore_id=%s file_path=%s reason=doc_not_found",
                    datastore_id, file_path,
                )
                return

            # Capture IDs before DB deletion — needed for Qdrant/Neo4j
            # cleanup after the DB commit.
            doc_id = doc.id
            chunk_ids = [
                cid[0] for cid in db.query(DocumentChunk.id)
                .filter(DocumentChunk.document_id == doc.id)
                .all()
            ]

            # DB cleanup first. If this commit fails, vector/graph data is
            # still intact and the next scan retries. If it succeeds but
            # Qdrant/Neo4j cleanup fails below, orphaned data is invisible
            # (document gone from DB) and reconciliation cleans it up on
            # next startup.
            db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).delete(synchronize_session=False)
            db.query(ProcessingTask).filter(ProcessingTask.document_id == doc.id).delete(synchronize_session=False)
            db.query(Document).filter(Document.id == doc.id).delete(synchronize_session=False)
            logger.info("[RECOVERY] document_deleted datastore_id=%s doc_id=%s file_path=%s", datastore_id, doc_id, file_path)

            # Delete DataStoreFileManifest entry
            manifest = (
                db.query(DataStoreFileManifest)
                .filter(DataStoreFileManifest.datastore_id == datastore_id, DataStoreFileManifest.file_path == file_path)
                .first()
            )
            if manifest:
                db.delete(manifest)
                logger.info("[RECOVERY] manifest_deleted datastore_id=%s file_path=%s", datastore_id, file_path)

            db.commit()

            # Qdrant cleanup (after DB commit, using captured IDs)
            try:
                from qdrant_client import models
                from app.services.ingestion import _chunk_id_to_point_id  # noqa: T100
                from app.services.infrastructure import get_qdrant_client  # noqa: T100

                if chunk_ids:
                    point_ids = [_chunk_id_to_point_id(cid) for cid in chunk_ids]
                    get_qdrant_client().delete(
                        collection_name=f"ds_{datastore_id}",
                        points_selector=models.PointIdsList(points=point_ids),
                    )
                logger.info("[RECOVERY] deletion_done datastore_id=%s doc_id=%s reason=qdrant_vectors_deleted", datastore_id, doc_id)
            except Exception as e:
                logger.warning("[RECOVERY] Qdrant delete failed for doc_id=%s: %s", doc_id, e)

            # Neo4j cleanup (after DB commit, using captured doc_id)
            try:
                from app.services.graph import delete_graph_for_document  # noqa: T100
                delete_graph_for_document(kb_id=None, document_id=doc_id, data_store_id=datastore_id)
                logger.info("[RECOVERY] deletion_done datastore_id=%s doc_id=%s reason=neo4j_cleanup", datastore_id, doc_id)
            except Exception as e:
                logger.warning("[RECOVERY] Neo4j cleanup failed for doc_id=%s: %s", doc_id, e)

        except Exception as e:
            logger.warning("[RECOVERY] Deletion failed for datastore_id=%s file_path=%s: %s", datastore_id, file_path, e, exc_info=True)
            try:
                db.rollback()
            except Exception:
                pass
        finally:
            db.close()

    def process_deleted_file(self, file_path: str, datastore_id: int) -> None:
        """Delete all records for a file that was deleted from disk.

        Convenience wrapper that delegates to ``_handle_deletion_records``.
        Kept for API symmetry with ``process_new_file``.
        """
        self._handle_deletion_records(file_path, datastore_id)

    # ------------------------------------------------------------------
    # Graph build retry
    # ------------------------------------------------------------------

    def _retry_pending_graph_builds(self, datastore_id: int) -> None:
        """Find tasks with completed ingestion but pending/failed graph build
        and re-run the graph build in background threads.

        This handles the case where a graph build was interrupted (app crash,
        LLM API down) after the document was successfully ingested to Qdrant.
        """
        db: Session = SessionLocal()
        try:
            tasks = (
                db.query(ProcessingTask)
                .filter(
                    ProcessingTask.data_store_id == datastore_id,
                    ProcessingTask.status == "completed",
                    ProcessingTask.graph_status.in_(["pending", "failed"]),
                )
                .all()
            )
        finally:
            db.close()

        if not tasks:
            return

        # Limit retries to 3 per task to avoid infinite retry loops on
        # permanently failing graph builds (e.g. LLM API down, bad config).
        # The retry count is encoded as a prefix in graph_error: "[retry:N] ...".
        MAX_GRAPH_RETRIES = 3
        retryable = []
        for t in tasks:
            retries = 0
            if t.graph_error and t.graph_error.startswith("[retry:"):
                try:
                    retries = int(t.graph_error.split("]")[0].split(":")[1])
                except (ValueError, IndexError):
                    pass
            if retries < MAX_GRAPH_RETRIES:
                retryable.append(t)
            else:
                logger.warning(
                    "[RECOVERY] graph_retry_skip task_id=%s — exceeded max retries (%d)",
                    t.id, MAX_GRAPH_RETRIES,
                )

        if not retryable:
            return

        logger.info(
            "[RECOVERY] graph_retry_start datastore_id=%s count=%d (skipped=%d)",
            datastore_id, len(retryable), len(tasks) - len(retryable),
        )

        from app.services.ingestion.ingestion_dispatcher import (
            _start_graph_build_thread,
        )
        from app.services.ingestion.document_processor import GraphBuildRequest

        for task in retryable:
            # Fetch chunks for this document to rebuild the graph
            chunk_db: Session = SessionLocal()
            try:
                doc = chunk_db.query(Document).filter(
                    Document.id == task.document_id
                ).first()
                if not doc:
                    logger.warning(
                        "[RECOVERY] graph_retry_skip task_id=%s — document %s not found",
                        task.id, task.document_id,
                    )
                    continue

                chunks = chunk_db.query(DocumentChunk).filter(
                    DocumentChunk.document_id == doc.id
                ).order_by(DocumentChunk.chunk_index).all()
                if not chunks:
                    logger.warning(
                        "[RECOVERY] graph_retry_skip task_id=%s — no chunks for doc %s",
                        task.id, doc.id,
                    )
                    continue

                req = GraphBuildRequest(
                    document_id=doc.id,
                    file_name=doc.file_name,
                    chunks=[c.chunk_text for c in chunks],
                    chunk_ids=[c.id for c in chunks],
                    kb_id=None,
                    data_store_id=datastore_id,
                    task_id=task.id,
                )
                _start_graph_build_thread(req)
                logger.info(
                    "[RECOVERY] graph_retry_queued task_id=%s doc_id=%s",
                    task.id, doc.id,
                )
            finally:
                chunk_db.close()


# ------------------------------------------------------------------
# Module-level utilities
# ------------------------------------------------------------------

def _sha256(file_path: str) -> str:
    """Compute the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    except OSError:
        return ""  # file may be unreadable during recovery
    return h.hexdigest()


def _guess_content_type(file_name: str) -> str:
    """Guess the MIME content type from the file extension."""
    import mimetypes
    ct, _ = mimetypes.guess_type(file_name)
    return ct or "application/octet-stream"


# ------------------------------------------------------------------
# Type imports for type hints
# ------------------------------------------------------------------
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — imported for type hints only
    from sqlalchemy.orm import Session
    from app.services.discovery import DiscoveryResult  # noqa: F401
