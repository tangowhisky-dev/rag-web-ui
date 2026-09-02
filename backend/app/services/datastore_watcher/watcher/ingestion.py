"""Scan ingestion — requeue logic, file ingestion during scans, and ingestion tracking.

Provides ``IngestionMixin`` for ``DataStoreWatcher``. Handles the ingestion
side of manual scans: processing new/modified files found by the discovery
engine, re-queuing stuck/orphan/needs-reprocess documents after pause/resume,
creating Document + ProcessingTask records, and submitting background
ingestion jobs to the thread pool with future tracking.

Methods:
- _requeue_stuck_documents: re-queue docs with pending/failed/processing tasks
- _requeue_orphan_documents: re-queue selected docs with no task and no chunks
- _requeue_reprocess_documents: re-queue docs with needs_reprocess=True
- _handle_file_in_scan: handle a single file during scan (create/update Document)
- _validate_file_for_scan: check extension, hidden/temp, existence
- _submit_and_track_ingestion: submit to executor, track future for scan_id
- _ingest_file_in_scan: create Document + ProcessingTask for new files in scan
- _update_document_in_scan: update existing Document during scan, reset task
- _upsert_manifest_with_mtime: create/update manifest with mtime for stat-first comparison
- _run_ingestion: run async ingestion pipeline in a threaded event loop
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import Future
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.db.session import SessionLocal
from app.models.datastore import DataStore, DataStoreFileManifest
from app.models.knowledge import Document, ProcessingTask, DocumentChunk
from app.services.ingestion import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


class IngestionMixin:
    """Requeue, file ingestion in scan, and ingestion tracking."""

    def _requeue_stuck_documents(
        self,
        db: Session,
        datastore_id: int,
        stuck_docs: List[Any],
        seen_paths: set,
        scan_id: int,
        summary: Dict[str, Any],
        ingestion_futures: List[Future],
    ) -> None:
        """Re-queue documents with pending/failed/processing tasks."""
        requeued = 0
        for doc, task in stuck_docs:
            if doc.file_path in seen_paths:
                continue  # already handled above
            if self._is_scan_cancelled(datastore_id):
                break
            # Check if chunks already exist (task may have completed
            # before the pause took effect).  If chunks exist, mark
            # the task as completed and skip re-ingestion.
            chunk_count = (
                db.query(DocumentChunk)
                .filter(DocumentChunk.document_id == doc.id)
                .count()
            )
            if chunk_count > 0:
                task.status = "completed"
                task.progress = 100
                db.commit()
                continue
            # No chunks — re-ingest using _update_document_in_scan
            # which reuses the existing Document, resets the task,
            # and submits to the executor.
            try:
                future = self._update_document_in_scan(
                    doc.id, doc.file_path, doc.file_hash or "",
                    datastore_id, scan_id,
                )
                if future is not None:
                    ingestion_futures.append(future)
                    requeued += 1
            except Exception as e:
                logger.error("[WATCHER] requeue error for %s: %s", doc.file_path, e)
                summary["errors"] += 1
        if requeued:
            logger.debug(
                "[WATCHER] requeued_stuck_tasks datastore_id=%d count=%d",
                datastore_id, requeued,
            )

    def _requeue_orphan_documents(
        self,
        db: Session,
        datastore_id: int,
        orphan_selected: List[Document],
        seen_paths: set,
        scan_id: int,
        summary: Dict[str, Any],
        ingestion_futures: List[Future],
    ) -> None:
        """Re-queue selected documents with no ProcessingTask and no chunks."""
        orphan_queued = 0
        for doc in orphan_selected:
            if doc.file_path in seen_paths:
                continue  # already handled above
            if self._is_scan_cancelled(datastore_id):
                break
            try:
                future = self._update_document_in_scan(
                    doc.id, doc.file_path, doc.file_hash or "",
                    datastore_id, scan_id,
                )
                if future is not None:
                    ingestion_futures.append(future)
                    orphan_queued += 1
            except Exception as e:
                logger.error("[WATCHER] orphan_queue error for %s: %s", doc.file_path, e)
                summary["errors"] += 1
        if orphan_queued:
            logger.debug(
                "[WATCHER] queued_orphan_selected datastore_id=%d count=%d",
                datastore_id, orphan_queued,
            )

    def _requeue_reprocess_documents(
        self,
        db: Session,
        datastore_id: int,
        reprocess_docs: List[Document],
        seen_paths: set,
        scan_id: int,
        summary: Dict[str, Any],
        ingestion_futures: List[Future],
    ) -> None:
        """Re-queue selected documents with needs_reprocess=True."""
        reprocess_queued = 0
        for doc in reprocess_docs:
            if doc.file_path in seen_paths:
                continue  # already handled above
            if self._is_scan_cancelled(datastore_id):
                break
            try:
                # Clear flag before re-ingesting
                doc.needs_reprocess = False
                db.commit()
                # Re-ingest using existing markdown (skip conversion)
                future = self._update_document_in_scan(
                    doc.id, doc.file_path, doc.file_hash or "",
                    datastore_id, scan_id,
                    skip_conversion=True,
                )
                if future is not None:
                    ingestion_futures.append(future)
                    reprocess_queued += 1
            except Exception as e:
                logger.error("[WATCHER] reprocess_queue error for %s: %s", doc.file_path, e)
                summary["errors"] += 1
        if reprocess_queued:
            logger.debug(
                "[WATCHER] queued_needs_reprocess datastore_id=%d count=%d",
                datastore_id, reprocess_queued,
            )

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
            logger.debug("[WATCHER] file_locked path=%s — skipping (another process holds lock)", event_path)
            db.close()
            return
        try:
            # Use the hash from discovery if available; otherwise compute it.
            # Discovery hashed the file moments ago during the walk phase,
            # so re-hashing would waste I/O on large files.
            # On first scans (empty manifest), discovery skips hashing and
            # passes file_hash=None — we hash here and write the manifest
            # entry so the next scan can use stat-first comparison.
            wrote_manifest = False
            if not file_hash:
                file_hash = self._handler._compute_hash(event_path)
                if not file_hash:
                    return
                # Write manifest entry with mtime so future scans skip this file.
                try:
                    st = os.stat(event_path)
                    file_size = st.st_size
                    file_mtime = st.st_mtime_ns
                except OSError:
                    file_size = 0
                    file_mtime = None
                self._upsert_manifest_with_mtime(
                    db, datastore_id, event_path, file_hash, file_size, file_mtime,
                )
                wrote_manifest = True

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
                # Skip if document was explicitly unselected by an admin
                if not existing.is_selected:
                    logger.debug(
                        "[WATCHER] file_unselected path=%s doc_id=%s — skipping",
                        event_path, existing.id,
                    )
                    return

                # Document exists - check if hash changed (file modified)
                if existing.file_hash == file_hash:
                    # File unchanged - check if re-ingest was requested (markdown edited)
                    if existing.needs_reprocess:
                        logger.debug(
                            "[WATCHER] re_ingest_needs_reprocess path=%s doc_id=%s datastore_id=%s",
                            event_path, existing.id, datastore_id,
                        )
                        # Clear flag before re-ingesting
                        existing.needs_reprocess = False
                        db.commit()
                        # Re-ingest using existing markdown (skip conversion)
                        future = self._update_document_in_scan(
                            existing.id, event_path, file_hash, datastore_id, scan_id,
                            skip_conversion=True,
                        )
                        return future

                    # File unchanged - check if chunks exist (ingestion may have failed)
                    chunk_count = db.query(DocumentChunk).filter(
                        DocumentChunk.document_id == existing.id
                    ).count()
                    if chunk_count > 0:
                        # File unchanged and chunks exist - skip
                        return
                    else:
                        # File unchanged but no chunks - re-ingest (ingestion likely failed)
                        logger.debug(
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
                    # File was modified - trigger re-ingestion (will re-convert)
                    future = self._update_document_in_scan(
                        existing.id, event_path, file_hash, datastore_id, scan_id
                    )
                    return future

            # File is new - check datastore's auto_process_enabled to
            # determine if it should be auto-selected for ingestion.
            ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
            auto_select = ds.auto_process_enabled if ds else False

            future = self._ingest_file_in_scan(
                event_path, datastore_id, scan_id, file_hash=file_hash,
                is_selected=auto_select,
            )
            return future
        finally:
            release_file_lock(db, datastore_id, event_path)
            db.close()

    def _validate_file_for_scan(
        self,
        event_path: str,
    ) -> Optional[tuple]:
        """Validate a file for scan ingestion.

        Returns (fname, ext, file_size, content_type) or None if the file
        should be skipped (unsupported extension, hidden/temp file, missing).
        """
        fname = os.path.basename(event_path)
        _, ext = os.path.splitext(fname)
        ext = ext.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            return None

        if fname.startswith(".") or fname.startswith("~$") or fname.startswith(".~"):
            return None
        if ext in (".tmp", ".swp", ".swo", ".bak", ".lock"):
            return None

        if not os.path.exists(event_path):
            return None

        try:
            file_size = os.path.getsize(event_path)
        except OSError:
            file_size = 0

        from app.services.ingestion.document_processor import CONTENT_TYPE_MAP
        content_type = CONTENT_TYPE_MAP.get(ext, "application/octet-stream")

        return (fname, ext, file_size, content_type)

    def _submit_and_track_ingestion(
        self,
        event_path: str,
        fname: str,
        task_id: int,
        document_id: int,
        datastore_id: int,
        scan_id: int,
        file_hash: str,
        file_size: int,
        content_type: str,
        skip_conversion: bool = False,
    ) -> Future:
        """Submit ingestion to executor and track the future for scan_id."""
        future = self._executor.submit(
            self._run_ingestion,
            event_path,
            fname,
            None,
            task_id,
            document_id,
            datastore_id,
            None,
            file_hash=file_hash,
            file_size=file_size,
            content_type=content_type,
            skip_conversion=skip_conversion,
        )
        future.add_done_callback(
            lambda f, ds=datastore_id: self._on_scan_ingestion_done(f, task_id, event_path, ds)
        )

        if scan_id > 0:
            with self._scan_futures_lock:
                if scan_id in self._scan_futures:
                    self._scan_futures[scan_id].append(future)

        return future

    def _ingest_file_in_scan(
        self,
        event_path: str,
        datastore_id: int,
        scan_id: int,
        file_hash: Optional[str] = None,
        is_selected: bool = False,
    ) -> Optional[Future]:
        """Create Document + ProcessingTask records and enqueue background processing for scans."""
        validated = self._validate_file_for_scan(event_path)
        if validated is None:
            return
        fname, ext, file_size, content_type = validated

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
                    is_selected=is_selected,
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

            return self._submit_and_track_ingestion(
                event_path, fname, task.id, doc.id, datastore_id, scan_id,
                file_hash, file_size, content_type,
            )
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
        skip_conversion: bool = False,
    ) -> Optional[Future]:
        """Update an existing document when file content changes during a scan.

        When skip_conversion=True, re-ingests using the existing
        converted_markdown instead of re-converting the source file.
        """
        validated = self._validate_file_for_scan(event_path)
        if validated is None:
            return
        fname, ext, file_size, content_type = validated

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

            return self._submit_and_track_ingestion(
                event_path, fname, task.id, document_id, datastore_id, scan_id,
                file_hash, file_size, content_type,
                skip_conversion=skip_conversion,
            )
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

    def _upsert_manifest_with_mtime(
        self,
        db: Session,
        datastore_id: int,
        file_path: str,
        file_hash: str,
        file_size: int,
        file_mtime: Optional[int],
    ) -> None:
        """Create or update a DataStoreFileManifest entry with mtime.

        Unlike the handler's _upsert_manifest, this stores file_mtime so
        the next discovery scan can use stat-first comparison and skip
        hashing this file.
        """
        now = datetime.now(timezone.utc)
        manifest = (
            db.query(DataStoreFileManifest)
            .filter(
                DataStoreFileManifest.datastore_id == datastore_id,
                DataStoreFileManifest.file_path == file_path,
            )
            .first()
        )
        if manifest:
            manifest.file_hash = file_hash
            manifest.file_size = file_size
            manifest.file_mtime = file_mtime
            manifest.updated_at = now
        else:
            db.add(
                DataStoreFileManifest(
                    datastore_id=datastore_id,
                    file_path=file_path,
                    file_hash=file_hash,
                    file_size=file_size,
                    file_mtime=file_mtime,
                    discovered_at=now,
                    updated_at=now,
                )
            )
        db.commit()

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
        skip_conversion: bool = False,
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
            skip_conversion=skip_conversion,
        )
