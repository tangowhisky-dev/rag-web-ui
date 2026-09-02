"""File ingestion — new/modified file handling, manifest, and ingestion pipeline.

Provides ``IngestMixin`` for ``DatastoreFileEventHandler``. Handles the
core file-processing logic for event-driven ingestion: validates files
against extension/pattern filters, computes hashes, creates or updates
Document + ProcessingTask records, upserts the file manifest, and
submits background ingestion jobs to the thread pool.

Methods:
- _should_skip_file: filter unsupported extensions, hidden/temp files
- _get_scan_pattern: check file against datastore scan_pattern
- _handle_existing_document: decide re-ingest vs skip for known files
- _handle_file: entry point for created/modified/deleted event processing
- _matches_pattern: delegate to shared pattern-matching utility
- _compute_hash: SHA-256 with mid-read size-change detection
- _upsert_manifest: create/update DataStoreFileManifest row
- _ingest_file: create Document + ProcessingTask, enqueue ingestion
- _update_document: update existing Document metadata, reset task, re-ingest
- _run_ingestion: run async ingestion pipeline in a threaded event loop
"""

from __future__ import annotations

import hashlib
import logging
import os
from concurrent.futures import Future
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.db.session import SessionLocal
from app.models.datastore import DataStore, DataStoreFileManifest
from app.models.knowledge import Document, ProcessingTask, DocumentChunk
from app.services.ingestion import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


class IngestMixin:
    """New/modified file handling, manifest management, and ingestion."""

    # ------------------------------------------------------------------
    # File processing (called by _on_changes callback)
    # ------------------------------------------------------------------

    def _should_skip_file(self, fname: str, ext: str, event_path: str) -> bool:
        if ext not in SUPPORTED_EXTENSIONS:
            logger.debug(
                "[WATCHER] file_detected path=%s ext=%s action=skip reason=unsupported_ext",
                event_path,
                ext,
            )
            return True
        if fname.startswith(".") or fname.startswith("~$") or fname.startswith(".~"):
            logger.debug(
                "[WATCHER] file_detected path=%s action=skip reason=hidden_or_temp",
                event_path,
            )
            return True
        if ext in (".tmp", ".swp", ".swo", ".bak", ".lock"):
            logger.debug(
                "[WATCHER] file_detected path=%s ext=%s action=skip reason=temp_ext",
                event_path,
                ext,
            )
            return True
        if not os.path.exists(event_path):
            logger.debug(
                "[WATCHER] file_not_exists path=%s action=skip",
                event_path,
            )
            return True
        return False

    def _get_scan_pattern(
        self, datastore_id: int, event_path: str, event_type: str,
    ) -> Optional[str]:
        db: Session = SessionLocal()
        try:
            ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
            if not ds:
                return None
            scan_pattern = ds.scan_pattern or "*"
            if not self._matches_pattern(event_path, scan_pattern):
                logger.debug(
                    "[WATCHER] file_detected path=%s action=skip reason=pattern_mismatch",
                    event_path,
                )
                return None
            logger.debug(
                "[WATCHER] file_processing datastore_id=%d path=%s event=%s",
                datastore_id, event_path, event_type,
            )
            return scan_pattern
        finally:
            db.close()

    def _handle_existing_document(
        self,
        db: Session,
        existing: Document,
        event_path: str,
        file_hash: str,
        hash_prefix: str,
        datastore_id: int,
    ) -> Optional[Future]:
        if not existing.is_selected:
            logger.debug(
                "[WATCHER] file_unselected path=%s doc_id=%s — skipping",
                event_path, existing.id,
            )
            return None

        if existing.file_hash == file_hash:
            chunk_count = db.query(DocumentChunk).filter(
                DocumentChunk.document_id == existing.id
            ).count()
            if chunk_count > 0:
                logger.debug(
                    "[WATCHER] no_change path=%s hash=%s datastore_id=%s doc_id=%s",
                    event_path,
                    hash_prefix,
                    datastore_id,
                    existing.id,
                )
                return None
            else:
                logger.debug(
                    "[WATCHER] re_ingest_no_chunks path=%s doc_id=%s datastore_id=%s",
                    event_path,
                    existing.id,
                    datastore_id,
                )
                return self._update_document(
                    existing.id, event_path, file_hash, datastore_id, scan_id=0
                )
        else:
            logger.debug(
                "[WATCHER] file_modified path=%s old_hash=%s new_hash=%s datastore_id=%s doc_id=%s",
                event_path,
                (existing.file_hash or "none")[:8],
                hash_prefix,
                datastore_id,
                existing.id,
            )
            return self._update_document(
                existing.id, event_path, file_hash, datastore_id, scan_id=0
            )

    def _handle_file(
        self,
        event_path: str,
        datastore_id: int,
        event_type: str = "modified"
    ) -> Optional[Future]:
        """Core logic: handle file events (created, modified, deleted).

        Files are processed in-place without copying. DataStore documents
        are independent of KnowledgeBases.

        Args:
            event_path: Full path to the file
            datastore_id: ID of the datastore containing the file
            event_type: One of 'created', 'modified', 'deleted'
        """
        if event_type == "deleted":
            self._handle_deletion(event_path, datastore_id)
            return

        fname = os.path.basename(event_path)
        _, ext = os.path.splitext(fname)
        ext = ext.lower()

        if self._should_skip_file(fname, ext, event_path):
            return

        scan_pattern = self._get_scan_pattern(datastore_id, event_path, event_type)
        if scan_pattern is None:
            return

        file_hash = self._compute_hash(event_path)
        hash_prefix = file_hash[:8] if file_hash else "none"
        file_size = os.path.getsize(event_path)

        db: Session = SessionLocal()
        from app.services.datastore_watcher.utils import acquire_file_lock, release_file_lock
        if not acquire_file_lock(db, datastore_id, event_path):
            logger.debug("[WATCHER] file_locked path=%s — skipping (another process holds lock)", event_path)
            db.close()
            return
        try:
            existing = (
                db.query(Document)
                .filter(
                    Document.file_path == event_path,
                    Document.data_store_id == datastore_id,
                )
                .first()
            )

            if existing:
                return self._handle_existing_document(
                    db, existing, event_path, file_hash, hash_prefix, datastore_id,
                )

            return self._ingest_file(
                event_path, datastore_id, scan_id=0, file_hash=file_hash
            )
        finally:
            release_file_lock(db, datastore_id, event_path)
            db.close()

    def _matches_pattern(self, filepath: str, scan_pattern: str = "*") -> bool:
        """Check if a filepath matches the scan pattern. Delegates to shared utility."""
        from app.services.datastore_watcher.utils import matches_pattern
        return matches_pattern(filepath, scan_pattern)

    def _compute_hash(self, path: str) -> str:
        """Compute SHA-256 hash of a file, aborting if the file changes mid-read."""
        try:
            size_before = os.path.getsize(path)
        except OSError:
            return ""

        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
        except OSError:
            return ""

        try:
            size_after = os.path.getsize(path)
        except OSError:
            return ""

        if size_before != size_after:
            logger.warning("[WATCHER] file size changed during hashing: %s", path)
            return ""

        return h.hexdigest()

    def _upsert_manifest(
        self,
        db: Session,
        datastore_id: int,
        event_path: str,
        file_hash: str,
        file_size: int,
    ) -> None:
        """Create or update the DataStoreFileManifest row for a file."""
        now = datetime.now(timezone.utc)
        manifest = (
            db.query(DataStoreFileManifest)
            .filter(
                DataStoreFileManifest.datastore_id == datastore_id,
                DataStoreFileManifest.file_path == event_path,
            )
            .first()
        )
        if manifest:
            manifest.file_hash = file_hash
            manifest.file_size = file_size
            manifest.updated_at = now
        else:
            db.add(
                DataStoreFileManifest(
                    datastore_id=datastore_id,
                    file_path=event_path,
                    file_hash=file_hash,
                    file_size=file_size,
                    discovered_at=now,
                    updated_at=now,
                )
            )
        db.commit()

    def _ingest_file(
        self,
        event_path: str,
        datastore_id: int,
        scan_id: int,
        file_hash: Optional[str] = None,
    ) -> Optional[Future]:
        """Create Document + ProcessingTask records and enqueue background processing.

        Returns the Future so the caller can wait for completion.
        DataStore files are processed independently — no KB knowledge needed.
        """
        fname = os.path.basename(event_path)
        _, ext = os.path.splitext(fname)
        ext = ext.lower()

        # Skip non-supported extensions
        if ext not in SUPPORTED_EXTENSIONS:
            logger.debug(
                "[WATCHER] file_detected path=%s ext=%s action=skip reason=unsupported_ext",
                event_path,
                ext,
            )
            return

        # Skip hidden/system files and temp/lock files
        if fname.startswith(".") or fname.startswith("~$") or fname.startswith(".~"):
            logger.debug(
                "[WATCHER] file_detected path=%s action=skip reason=hidden_or_temp",
                event_path,
            )
            return
        if ext in (".tmp", ".swp", ".swo", ".bak", ".lock"):
            logger.debug(
                "[WATCHER] file_detected path=%s ext=%s action=skip reason=temp_ext",
                event_path,
                ext,
            )
            return

        # Check if file exists
        if not os.path.exists(event_path):
            logger.debug(
                "[WATCHER] file_not_exists path=%s action=skip",
                event_path,
            )
            return

        try:
            file_size = os.path.getsize(event_path)
        except OSError:
            file_size = 0

        from app.services.ingestion.document_processor import CONTENT_TYPE_MAP

        content_type = CONTENT_TYPE_MAP.get(ext, "application/octet-stream")

        # Use pre-computed hash if available (avoids reading the file twice)
        if file_hash is None:
            file_hash = self._compute_hash(event_path)
        if not file_hash:
            logger.warning(
                "[WATCHER] failed_to_compute_hash path=%s action=skip",
                event_path,
            )
            return

        logger.debug(
            "[WATCHER] ingestion_start datastore_id=%d path=%s doc_id=NEW",
            datastore_id, event_path,
        )
        db: Session = SessionLocal()
        try:
            # Check auto_process_enabled to determine is_selected.
            # Matches the scan path (_handle_file_in_scan) which sets
            # is_selected=auto_select for new files.  Without this,
            # event-detected files on auto-process datastores are
            # ingested but never marked as selected, so the UI shows
            # a stale selected count (e.g. 16/17 instead of 17/17).
            ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
            auto_select = ds.auto_process_enabled if ds else False

            # Create Document record (in-place, no copy to uploads)
            try:
                doc = Document(
                    knowledge_base_id=None,
                    data_store_id=datastore_id,
                    file_path=event_path,
                    file_name=fname,
                    file_hash=file_hash,
                    file_size=file_size,
                    content_type=content_type,
                    is_selected=auto_select,
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

            # Create ProcessingTask record
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

            # Keep manifest in sync so discovery does not re-process this file
            self._upsert_manifest(db, datastore_id, event_path, file_hash, file_size)

            logger.debug(
                "[WATCHER] ingestion_started path=%s datastore_id=%s doc_id=%s task_id=%s",
                event_path,
                datastore_id,
                doc.id,
                task.id,
            )

            # Enqueue background processing
            future = self._executor.submit(
                self._run_ingestion,
                event_path,
                fname,
                None,  # kb_id (DataStore files have no KB)
                task.id,
                doc.id,
                datastore_id,
                None,
                file_hash=file_hash,
                file_size=file_size,
                content_type=content_type,
            )
            future.add_done_callback(
                lambda f: self._on_ingestion_done(f, task.id, event_path)
            )
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

    def _update_document(
        self,
        document_id: int,
        event_path: str,
        file_hash: str,
        datastore_id: int,
        scan_id: int,
    ) -> Optional[Future]:
        """Update an existing document when file content changes.

        Returns the Future so the caller can wait for completion.
        DataStore documents don't use KBs — kb_id is always None.
        """
        fname = os.path.basename(event_path)
        _, ext = os.path.splitext(fname)
        ext = ext.lower()

        # Skip non-supported extensions
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

        logger.debug(
            "[WATCHER] ingestion_update_start datastore_id=%d path=%s doc_id=%d",
            datastore_id, event_path, document_id,
        )
        db: Session = SessionLocal()
        try:
            doc = db.query(Document).filter(Document.id == document_id).first()
            if not doc:
                return

            # Update document metadata
            doc.file_hash = file_hash
            doc.file_size = file_size
            doc.content_type = content_type
            db.commit()

            # Reset or create processing task
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
                # Re-ingest with no chunks — create a new task
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

            # Keep manifest in sync with the new hash
            self._upsert_manifest(db, datastore_id, event_path, file_hash, file_size)

            logger.debug(
                "[WATCHER] update_started doc_id=%s path=%s",
                document_id,
                event_path,
            )

            # Enqueue background re-processing
            future = self._executor.submit(
                self._run_ingestion,
                event_path,
                fname,
                None,  # kb_id (DataStore files have no KB)
                task.id,
                document_id,
                datastore_id,
                None,
                file_hash=file_hash,
                file_size=file_size,
                content_type=content_type,
            )
            future.add_done_callback(
                lambda f: self._on_ingestion_done(f, task.id, event_path)
            )
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

    # ------------------------------------------------------------------
    # Async ingestion runner
    # ------------------------------------------------------------------

    def _run_ingestion(
        self,
        file_path: str,
        file_name: str,
        kb_id: Optional[int],
        task_id: int,
        document_id: int,
        data_store_id: Optional[int],
        db,  # Session — kept for API compatibility with DataStoreWatcher
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
