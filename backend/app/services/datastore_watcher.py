"""DataStore Watcher Service — per-datastore file watching with batch processing.

Architecture:
- Single ``PollingObserver`` instance for all datastores (efficient resource usage)
- One ``DatastoreFileEventHandler`` per datastore (per-folder event handler)
- Each handler queues file events and processes them in batches at a configurable interval
- Periodic DB sync (every 5 min) to add/remove watchers based on datastore configuration

File path convention inside a watched directory:
    kb_{kb_id}/{file_name}

The service parses the path to determine which knowledge base owns the file,
then routes it into the existing ``process_document_background()`` pipeline.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import threading
import time as time_module
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.datastore import DataStore, OrganizationDataStore
from app.models.knowledge import Document, DocumentUpload, ProcessingTask
from app.models.knowledge import KnowledgeBase
from app.services.document_processor import (
    SUPPORTED_EXTENSIONS,
    process_document_background,
)

logger = logging.getLogger(__name__)


class _Debouncer:
    """Coalesces rapid repeated events for the same path into a single handling."""

    def __init__(self, delay: float = 1.0) -> None:
        self._delay = delay
        self._lock = threading.Lock()
        self._last_event: Dict[str, float] = {}
        self._last_type: Dict[str, str] = {}

    def touch(self, path: str, event_type: str) -> Optional[str]:
        """Record an event. Returns the coalesced event_type if this path should
        be processed now, or None if another event is expected soon."""
        now = time_module.monotonic()
        with self._lock:
            prev_time = self._last_event.get(path)
            prev_type = self._last_type.get(path)
            self._last_event[path] = now
            self._last_type[path] = event_type
            if prev_time is not None and (now - prev_time) < self._delay:
                return None
            return event_type


class DatastoreFileEventHandler:
    """Per-datastore event handler with custom configuration.

    Features:
    - Debouncing per-file (prevents duplicate events from editors)
    - Batch processing with configurable intervals
    - org_id and datastore_id tracking for each event
    """

    def __init__(
        self,
        datastore_id: int,
        org_id: int,
        callback,
        debounce_ms: int = 1000,
        min_interval_seconds: int = 300,
    ) -> None:
        self.datastore_id = datastore_id
        self.org_id = org_id
        self.callback = callback
        self.debounce_ms = debounce_ms / 1000.0
        self.min_interval_seconds = min_interval_seconds
        self.last_call: Dict[str, float] = {}
        self.last_processing_time: float = 0
        self.pending_changes: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def _should_process(self, src_path: str) -> bool:
        now = time_module.time()
        with self._lock:
            if now - self.last_call.get(src_path, 0) < self.debounce_ms:
                return False
            self.last_call[src_path] = now
            return True

    def _queue_change(self, event, event_type: str) -> None:
        with self._lock:
            self.pending_changes.append(
                {
                    "datastore_id": self.datastore_id,
                    "org_id": self.org_id,
                    "path": event.src_path,
                    "event_type": event_type,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            # Check if it's time to process batch
            if time_module.time() - self.last_processing_time >= self.min_interval_seconds:
                self._process_batch()

    def _handle_deletion(self, event_path: str, org_id: int, datastore_id: Optional[int] = None) -> None:
        """Handle file deletion - remove Document records from all linked KBs."""
        logger.info(
            "[WATCHER] file_deleted path=%s org_id=%s datastore_id=%s",
            event_path,
            org_id,
            datastore_id,
        )
        
        db: Session = SessionLocal()
        try:
            from app.models.knowledge import KnowledgeBase, Document, DocumentChunk, ProcessingTask, KnowledgeBaseDataStore
            
            # Determine which KBs to delete from
            if datastore_id is not None:
                # Get KBs explicitly linked to this DataStore
                linked_kb_ids = (
                    db.query(KnowledgeBaseDataStore.knowledge_base_id)
                    .filter(KnowledgeBaseDataStore.data_store_id == datastore_id)
                    .all()
                )
                kb_list = [kb_id[0] for kb_id in linked_kb_ids]
                logger.info(
                    "[WATCHER] deletion_linked_kb_count path=%s datastore_id=%s count=%d",
                    event_path,
                    datastore_id,
                    len(kb_list),
                )
            else:
                # Fallback: get all KBs for the org (for backward compatibility)
                kb_list = (
                    db.query(KnowledgeBase)
                    .filter(KnowledgeBase.org_id == org_id)
                    .values('id')
                )
                kb_list = [kb[0] for kb in kb_list]
            
            for kb_id in kb_list:
                # Find and delete document for this file path
                doc = (
                    db.query(Document)
                    .filter(
                        Document.file_path == event_path,
                        Document.knowledge_base_id == kb_id,
                    )
                    .first()
                )
                
                if doc:
                    # Delete associated chunks
                    db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).delete()
                    # Delete associated processing tasks
                    db.query(ProcessingTask).filter(ProcessingTask.document_id == doc.id).delete()
                    # Delete the document
                    db.delete(doc)
                    logger.info(
                        "[WATCHER] document_deleted path=%s kb_id=%s doc_id=%s",
                        event_path,
                        kb_id,
                        doc.id,
                    )
            
            db.commit()
        finally:
            db.close()

    def _process_batch(self) -> None:
        with self._lock:
            if not self.pending_changes:
                return
            changes_to_process = self.pending_changes.copy()
            self.pending_changes.clear()
            self.last_processing_time = time_module.time()

        try:
            self.callback(self.datastore_id, self.org_id, changes_to_process)
        except Exception as e:
            logger.error(
                "[WATCHER] batch_process_error datastore_id=%s org_id=%s: %s",
                self.datastore_id,
                self.org_id,
                e,
                exc_info=True,
            )

    def force_process_pending(self) -> None:
        changes_to_process: List[Dict[str, Any]] = []
        with self._lock:
            if self.pending_changes:
                changes_to_process = self.pending_changes.copy()
                self.pending_changes.clear()
        try:
            if changes_to_process:
                self.callback(self.datastore_id, self.org_id, changes_to_process)
        except Exception as e:
            logger.error(
                "[WATCHER] force_process_error datastore_id=%s: %s",
                self.datastore_id,
                e,
                exc_info=True,
            )

    def on_created(self, event) -> None:
        if event.is_directory:
            return
        if not self._should_process(event.src_path):
            return
        self._queue_change(event, "created")

    def on_modified(self, event) -> None:
        if event.is_directory:
            return
        if not self._should_process(event.src_path):
            return
        self._queue_change(event, "modified")

    def on_deleted(self, event) -> None:
        if event.is_directory:
            return
        if not self._should_process(event.src_path):
            return
        self._queue_change(event, "deleted")


class DataStoreWatcher:
    """Watches multiple datastore folders with per-datastore configuration.

    Features:
    - Single Observer instance for all datastores (efficient)
    - Dynamic add/remove based on database configuration
    - Per-datastore processing intervals
    - Automatic sync with database every 5 minutes
    """

    def __init__(self) -> None:
        self._observer = None
        self._lock = threading.Lock()
        self._running = False
        self._executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="watcher"
        )
        self._debouncer = _Debouncer(delay=1.0)
        self.datastore_handlers: Dict[int, DatastoreFileEventHandler] = {}
        self.watched_paths: Dict[int, Path] = {}
        self._last_scan_at: Optional[float] = None
        self._files_scanned: int = 0

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the observer and begin watching all configured datastores."""
        with self._lock:
            if self._running:
                logger.warning("[WATCHER] already running, ignoring start()")
                return
            self._running = True

        try:
            from watchdog.observers.polling import PollingObserver

            self._observer = PollingObserver()
            self._observer.start()
            logger.info("[WATCHER] observer started (PollingObserver)")
        except Exception as e:
            logger.error("[WATCHER] failed to start observer: %s", e)
            self._running = False
            raise

        self._sync_watchers_with_database()

        # Start periodic sync thread
        sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        sync_thread.start()

        logger.info("[WATCHER] service started")

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
                logger.info("[WATCHER] observer stopped")
            except Exception as e:
                logger.warning("[WATCHER] error stopping observer: %s", e)

        # Process all pending changes
        for handler in self.datastore_handlers.values():
            handler.force_process_pending()

        self._executor.shutdown(wait=False, cancel_futures=True)
        logger.info("[WATCHER] service stopped")

    def add_datastore(self, datastore_id: int, org_id: int, folder_path: str, interval_minutes: int = 60) -> None:
        """Start watching a specific datastore folder."""
        abs_path = Path(folder_path).resolve()
        if not abs_path.exists() or not abs_path.is_dir():
            logger.warning(
                "[WATCHER] add_datastore_path_not_found datastore_id=%s path=%s",
                datastore_id,
                folder_path,
            )
            return

        # Skip if already watching
        if datastore_id in self.datastore_handlers:
            logger.info(
                "[WATCHER] add_datastore_already_watching datastore_id=%s",
                datastore_id,
            )
            return

        handler = DatastoreFileEventHandler(
            datastore_id=datastore_id,
            org_id=org_id,
            callback=self._on_changes,
            debounce_ms=1000,
            min_interval_seconds=interval_minutes * 60,
        )

        try:
            self._observer.schedule(handler, str(abs_path), recursive=True)
            self.datastore_handlers[datastore_id] = handler
            self.watched_paths[datastore_id] = abs_path
            logger.info(
                "[WATCHER] datastore_added datastore_id=%s path=%s interval=%dm",
                datastore_id,
                folder_path,
                interval_minutes,
            )
        except Exception as e:
            logger.error(
                "[WATCHER] add_datastore_failed datastore_id=%s: %s",
                datastore_id,
                e,
                exc_info=True,
            )

    def remove_datastore(self, datastore_id: int) -> None:
        """Stop watching a datastore."""
        if datastore_id not in self.datastore_handlers:
            return

        handler = self.datastore_handlers[datastore_id]
        del self.datastore_handlers[datastore_id]
        if datastore_id in self.watched_paths:
            del self.watched_paths[datastore_id]

        logger.info("[WATCHER] datastore_removed datastore_id=%s", datastore_id)

    def scan_single_datastore(self, datastore_id: int) -> Dict[str, int]:
        """Manually scan a specific datastore for new/modified files.

        Args:
            datastore_id: ID of the datastore to scan

        Returns a summary dict with counts of scanned, new, and skipped files.
        """
        summary: Dict[str, int] = {"scanned": 0, "new": 0, "skipped": 0, "errors": 0}

        if not self._running or self._observer is None:
            logger.warning("[WATCHER] scan attempted but service is not running")
            return summary

        db: Session = SessionLocal()
        try:
            ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
            if not ds or not ds.is_active or not ds.folder_path or not os.path.isdir(ds.folder_path):
                logger.warning("[WATCHER] scan skipped invalid datastore id=%d", datastore_id)
                return summary

            # Get org_id from first assignment
            assignment = (
                db.query(OrganizationDataStore)
                .filter(OrganizationDataStore.data_store_id == datastore_id)
                .first()
            )
            org_id = assignment.org_id if assignment else None

            for root, _dirs, files in os.walk(ds.folder_path):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    summary["scanned"] += 1
                    try:
                        if org_id:
                            self._handle_file(fpath, datastore_id, org_id, "created")
                        else:
                            summary["skipped"] += 1
                    except Exception as e:
                        logger.error(
                            "[WATCHER] scan error for %s: %s", fpath, e
                        )
                        summary["errors"] += 1

            return summary
        finally:
            db.close()

    def scan(self) -> Dict[str, int]:
        """Manually scan all watched datastores for new/modified files.

        Returns a summary dict with counts of scanned, new, and skipped files.
        """
        summary: Dict[str, int] = {"scanned": 0, "new": 0, "skipped": 0, "errors": 0}

        if not self._running or self._observer is None:
            logger.warning("[WATCHER] scan attempted but service is not running")
            return summary

        # Get all active datastores with auto_scan_enabled
        db: Session = SessionLocal()
        try:
            datastores = (
                db.query(DataStore)
                .filter(DataStore.is_active == True)
                .all()
            )
            for ds in datastores:
                if not ds.folder_path or not os.path.isdir(ds.folder_path):
                    continue
                for root, _dirs, files in os.walk(ds.folder_path):
                    for fname in files:
                        fpath = os.path.join(root, fname)
                        summary["scanned"] += 1
                        try:
                            # Get org_id from first assignment
                            assignment = (
                                db.query(OrganizationDataStore)
                                .filter(OrganizationDataStore.data_store_id == ds.id)
                                .first()
                            )
                            org_id = assignment.org_id if assignment else None
                            if org_id:
                                self._handle_file(fpath, ds.id, org_id, "created")
                            else:
                                summary["skipped"] += 1
                        except Exception as e:
                            logger.error(
                                "[WATCHER] scan error for %s: %s", fpath, e
                            )
                            summary["errors"] += 1
        finally:
            db.close()

        self._last_scan_at = time_module.time()
        logger.info(
            "[WATCHER] scan_complete scanned=%d new=%d skipped=%d errors=%d",
            summary["scanned"],
            summary["new"],
            summary["skipped"],
            summary["errors"],
        )
        return summary

    def get_status(self) -> Dict[str, Any]:
        """Return current watcher state for the admin status endpoint."""
        return {
            "running": self._running,
            "last_scan_at": self._last_scan_at,
            "files_scanned": self._files_scanned,
            "datastores": [
                {
                    "datastore_id": ds_id,
                    "path": str(self.watched_paths.get(ds_id, "unknown")),
                    "pending_changes": len(handler.pending_changes),
                    "min_interval_seconds": handler.min_interval_seconds,
                    "last_processing_time": handler.last_processing_time,
                }
                for ds_id, handler in self.datastore_handlers.items()
            ],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sync_watchers_with_database(self) -> None:
        """Sync watchers with database configuration."""
        db: Session = SessionLocal()
        try:
            # Get all active datastore-organization assignments
            assignments = (
                db.query(OrganizationDataStore)
                .join(DataStore)
                .filter(
                    OrganizationDataStore.is_active == True,
                    DataStore.is_active == True,
                    DataStore.auto_scan_enabled == True,
                )
                .all()
            )

            # Group by datastore_id (org_id comes from assignment)
            datastore_ids = set()
            for assignment in assignments:
                ds_id = assignment.data_store_id
                org_id = assignment.org_id
                datastore_ids.add(ds_id)

                ds = assignment.data_store
                if ds_id not in self.datastore_handlers:
                    self.add_datastore(
                        ds_id,
                        org_id,
                        ds.folder_path,
                        ds.auto_scan_interval_minutes or 60,
                    )

            # Remove watchers for disabled/unassigned datastores
            current_ids = set(self.datastore_handlers.keys())
            to_remove = current_ids - datastore_ids
            for ds_id in to_remove:
                self.remove_datastore(ds_id)

        finally:
            db.close()

    def _sync_loop(self) -> None:
        """Periodic sync thread — checks every 5 minutes."""
        while self._running:
            time_module.sleep(300)  # 5 minutes
            if self._running:
                self._sync_watchers_with_database()

    def _on_changes(self, datastore_id: int, org_id: int, changes: List[Dict[str, Any]]) -> None:
        """Callback when batch of changes is ready to process.
        
        Args:
            datastore_id: ID of the datastore with changes
            org_id: Organization ID this datastore belongs to  
            changes: List of change events with path and event_type
        """
        if not changes:
            return

        logger.info(
            "[WATCHER] batch_ready datastore_id=%d org_id=%d changes=%d",
            datastore_id,
            org_id,
            len(changes),
        )

        # Process each change
        for change in changes:
            fpath = change["path"]
            event_type = change.get("event_type", "modified")
            try:
                self._handle_file(fpath, datastore_id, org_id, event_type)
            except Exception as e:
                logger.error(
                    "[WATCHER] handle_file_error path=%s event=%s: %s", fpath, event_type, e, exc_info=True
                )

    def _handle_file(
        self, 
        event_path: str, 
        datastore_id: int, 
        org_id: int,
        event_type: str = "modified"
    ) -> None:
        """Core logic: handle file events (created, modified, deleted).
        
        Files are processed in-place without copying. The file is linked to the
        KnowledgeBase(s) that the org has access to.
        
        Args:
            event_path: Full path to the file
            datastore_id: ID of the datastore containing the file
            org_id: Organization ID that owns this datastore
            event_type: One of 'created', 'modified', 'deleted'
        """
        # Handle deletion differently - no need to hash or check extensions
        if event_type == "deleted":
            self._handle_deletion(event_path, org_id, datastore_id)
            return
        
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

        # Skip hidden/system files
        if fname.startswith("."):
            logger.debug(
                "[WATCHER] file_detected path=%s action=skip reason=hidden_file",
                event_path,
            )
            return

        # Check if file exists (for modified events, file might have been deleted)
        if not os.path.exists(event_path):
            logger.debug(
                "[WATCHER] file_not_exists path=%s action=skip",
                event_path,
            )
            return

        # Compute SHA-256 hash
        file_hash = self._compute_hash(event_path)
        hash_prefix = file_hash[:8] if file_hash else "none"
        file_size = os.path.getsize(event_path)

        # Get KBs explicitly linked to this DataStore via junction table
        db: Session = SessionLocal()
        try:
            from app.models.knowledge import KnowledgeBase, KnowledgeBaseDataStore
            
            # Get all KBs linked to this DataStore
            linked_kb_ids = (
                db.query(KnowledgeBaseDataStore.knowledge_base_id)
                .filter(KnowledgeBaseDataStore.data_store_id == datastore_id)
                .all()
            )
            
            if not linked_kb_ids:
                logger.info(
                    "[WATCHER] no_linked_kb path=%s datastore_id=%s",
                    event_path,
                    datastore_id,
                )
                return
            
            kb_list = [kb_id[0] for kb_id in linked_kb_ids]
            logger.info(
                "[WATCHER] linked_kb_count path=%s datastore_id=%s count=%d",
                event_path,
                datastore_id,
                len(kb_list),
            )

            # Process file for each linked KB
            for kb_id in kb_list:
                # Check if document already exists for this file path
                existing = (
                    db.query(Document)
                    .filter(
                        Document.file_path == event_path,
                        Document.knowledge_base_id == kb_id,
                    )
                    .first()
                )
                
                if existing:
                    # Document exists - check if hash changed (file modified)
                    if existing.file_hash == file_hash:
                        logger.info(
                            "[WATCHER] no_change path=%s hash=%s kb_id=%s doc_id=%s",
                            event_path,
                            hash_prefix,
                            kb_id,
                            existing.id,
                        )
                        continue
                    else:
                        # File was modified - re-ingest
                        logger.info(
                            "[WATCHER] file_modified path=%s old_hash=%s new_hash=%s kb_id=%s doc_id=%s",
                            event_path,
                            existing.file_hash[:8],
                            hash_prefix,
                            kb_id,
                            existing.id,
                        )
                        # Update existing document
                        self._update_document_inplace(
                            existing.id, event_path, fname, file_hash, file_size, ext, kb_id, datastore_id
                        )
                        continue

                # File is new for this KB — trigger ingestion in-place
                self._ingest_file_inplace(
                    event_path, fname, file_hash, file_size, ext, kb_id, datastore_id
                )
        finally:
            db.close()

    def _update_document_inplace(
        self,
        document_id: int,
        event_path: str,
        fname: str,
        file_hash: str,
        file_size: int,
        ext: str,
        kb_id: int,
        datastore_id: int,
    ) -> None:
        """Update an existing document when file content changes."""
        logger.info(
            "[WATCHER] update_start doc_id=%s path=%s",
            document_id,
            event_path,
        )
        
        # Trigger background re-processing
        try:
            process_document_background(
                kb_id=kb_id,
                file_path=event_path,
                file_name=fname,
                file_hash=file_hash,
                file_size=file_size,
                content_type=ext.lstrip("."),
                data_store_id=datastore_id,
                document_id=document_id,  # Update existing document
            )
            logger.info(
                "[WATCHER] update_queued doc_id=%s path=%s",
                document_id,
                event_path,
            )
        except Exception as e:
            logger.error(
                "[WATCHER] update_error doc_id=%s path=%s: %s",
                document_id,
                event_path,
                e,
                exc_info=True,
            )
        finally:
            db.close()
        self._trigger_ingestion(event_path, fname, resolved_kb_id, datastore_id, file_hash)

    def _trigger_ingestion(
        self,
        event_path: str,
        file_name: str,
        kb_id: int,
        datastore_id: int,
        file_hash: str,
    ) -> None:
        """Create Document + ProcessingTask records and enqueue background processing.
        
        Files are processed in-place - NOT copied to uploads folder.
        """
        logger.info(
            "[WATCHER] ingestion_started path=%s kb_id=%s datastore_id=%s",
            event_path,
            kb_id,
            datastore_id,
        )

        try:
            file_size = os.path.getsize(event_path)
        except OSError:
            file_size = 0

        _, ext = os.path.splitext(file_name)
        from app.services.document_processor import CONTENT_TYPE_MAP

        content_type = CONTENT_TYPE_MAP.get(ext, "application/octet-stream")

        db: Session = SessionLocal()
        try:
            # Create Document record (in-place, no copy to uploads)
            # Use temp_path as the actual file path since we're not copying
            doc = Document(
                knowledge_base_id=kb_id,
                data_store_id=datastore_id,
                file_path=event_path,  # Original path
                file_name=file_name,
                file_hash=file_hash,
                file_size=file_size,
                content_type=content_type,
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)

            # Create ProcessingTask record
            task = ProcessingTask(
                knowledge_base_id=kb_id,
                document_id=doc.id,
                status="pending",
                progress=0,
                progress_message="Queued by watcher",
            )
            db.add(task)
            db.commit()
            db.refresh(task)

            logger.info(
                "[WATCHER] records_created doc_id=%s task_id=%s",
                doc.id,
                task.id,
            )

            # Enqueue background processing
            loop = asyncio.new_event_loop()
            future = self._executor.submit(
                self._run_ingestion,
                event_path,
                file_name,
                kb_id,
                task.id,
                doc.id,
                datastore_id,
                db,
                loop,
            )
            future.add_done_callback(
                lambda f: self._on_ingestion_done(f, task.id, event_path)
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

    def _run_ingestion(
        self,
        file_path: str,
        file_name: str,
        kb_id: int,
        task_id: int,
        document_id: int,
        data_store_id: int,
        db: Session,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Run the async ingestion pipeline in a dedicated event loop (threaded).
        
        Files are processed in-place - NOT copied to uploads folder.
        """
        try:
            async def _do() -> None:
                await process_document_background(
                    temp_path=file_path,  # Original path, not temp
                    file_name=file_name,
                    kb_id=kb_id,
                    task_id=task_id,
                    document_id=document_id,
                    data_store_id=data_store_id,
                    db=db,
                )

            asyncio.set_event_loop(loop)
            loop.run_until_complete(_do())
        except Exception as e:
            logger.error(
                "[WATCHER] ingestion_failed task_id=%s error=%s",
                task_id,
                e,
                exc_info=True,
            )
        finally:
            loop.close()

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
            logger.info(
                "[WATCHER] ingestion_completed task_id=%s path=%s",
                task_id,
                event_path,
            )

    @staticmethod
    def _compute_hash(path: str) -> str:
        """Compute SHA-256 hash of a file."""
