"""WatcherService — monitors local directories for new/modified files and triggers ingestion.

Uses watchdog's PollingObserver to detect file-system events, computes SHA-256 hashes
for deduplication, and routes new files into the existing ``process_document_background()``
pipeline via a ThreadPoolExecutor so the observer thread is never blocked.

File path convention inside a watched directory:
    user_{user_id}/kb_{kb_id}/{file_name}

The service matches the event path against each organisation's ``watch_dir`` to
determine the owning org, then parses the path segments for user_id and kb_id.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.knowledge import Document, DocumentUpload, ProcessingTask
from app.models.organisation import Organisation
from app.services.document_processor import (
    SUPPORTED_EXTENSIONS,
    process_document_background,
)

logger = logging.getLogger(__name__)

# Regex to parse user_{uid}/kb_{kid}/ path segments
_PATH_RE = re.compile(r"user_(\d+)/kb_(\d+)/(.+)")


class _Debouncer:
    """Coalesces rapid repeated events for the same path into a single handling."""

    def __init__(self, delay: float = 1.0) -> None:
        self._delay = delay
        self._lock = threading.Lock()
        # path -> timestamp of last event
        self._last_event: dict[str, float] = {}
        # path -> event-type string
        self._last_type: dict[str, str] = {}

    def touch(self, path: str, event_type: str) -> Optional[str]:
        """Record an event.  Returns the coalesced event_type if this path should
        be processed now, or None if another event is expected soon."""
        now = time.monotonic()
        with self._lock:
            prev_time = self._last_event.get(path)
            prev_type = self._last_type.get(path)
            self._last_event[path] = now
            self._last_type[path] = event_type
            if prev_time is not None and (now - prev_time) < self._delay:
                # Too soon — debounce
                return None
            return event_type


class WatcherService:
    """File-system watcher that triggers document ingestion for new/modified files.

    Thread-safe via ``threading.Lock`` for internal state mutations.
    Creates a fresh ``SessionLocal()`` per DB operation for SQLAlchemy thread safety.
    """

    def __init__(self) -> None:
        self._observer = None  # type: Optional[object]
        self._lock = threading.Lock()
        self._running = False
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="watcher")
        self._debouncer = _Debouncer(delay=1.0)
        self._last_scan_at: Optional[float] = None
        self._files_scanned: int = 0

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the observer and begin watching all configured directories.

        Scans each organisation with a non-null ``watch_dir`` and registers it
        with the watchdog observer.  Skips directories that do not exist on disk.
        """
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

        self._watch_all_dirs()
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

        self._executor.shutdown(wait=False, cancel_futures=True)
        logger.info("[WATCHER] service stopped")

    def add_watch(self, org_id: int, watch_dir: str) -> None:
        """Start watching a specific directory for an organisation."""
        abs_dir = os.path.abspath(watch_dir)
        if not os.path.isdir(abs_dir):
            logger.warning("[WATCHER] add_watch dir_not_found dir=%s", abs_dir)
            return

        # Unregister any existing watch for this org first
        self.remove_watch(org_id)

        if not self._running or self._observer is None:
            logger.warning("[WATCHER] add_watch service_not_running org_id=%s", org_id)
            return

        handler = _WatcherHandler(
            service=self,
            org_id=org_id,
            watch_dir=abs_dir,
        )
        try:
            self._observer.schedule(handler, abs_dir, recursive=True)  # type: ignore[union-attr]
            logger.info("[WATCHER] watch_added org_id=%s dir=%s", org_id, abs_dir)
        except Exception as e:
            logger.error("[WATCHER] watch_add_failed org_id=%s dir=%s: %s", org_id, abs_dir, e)

    def remove_watch(self, org_id: int) -> None:
        """Stop watching the directory for a specific organisation."""
        if not self._observer:
            return

        db: Session = SessionLocal()
        try:
            org = (
                db.query(Organisation)
                .filter(Organisation.id == org_id)
                .first()
            )
            if org and org.watch_dir:
                abs_dir = os.path.abspath(org.watch_dir)
                # Find and unschedule the handler for this org
                if hasattr(self._observer, 'handlers'):
                    for handler in self._observer.handlers:  # type: ignore[attr-defined]
                        if getattr(handler, '_watch_dir', '').rstrip('/') == abs_dir.rstrip('/'):
                            self._observer.unschedule(handler)  # type: ignore[union-attr]
                            logger.info("[WATCHER] watch_removed org_id=%s dir=%s", org_id, abs_dir)
                            return
        finally:
            db.close()

    def scan(self) -> dict:
        """Manually scan all watched directories for new/modified files.

        Returns a summary dict with counts of scanned, new, and skipped files.
        """
        from watchdog.observers.api import BaseObserver

        summary = {"scanned": 0, "new": 0, "skipped": 0, "errors": 0}

        if not self._running or self._observer is None:
            logger.warning("[WATCHER] scan attempted but service is not running")
            return summary

        orgs = self._get_orgs_with_watch_dirs()
        for org in orgs:
            watch_dir = org.watch_dir
            if not watch_dir or not os.path.isdir(watch_dir):
                continue

            for root, _dirs, files in os.walk(watch_dir):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    summary["scanned"] += 1
                    try:
                        self._handle_file(fpath, str(org.id), None)
                    except Exception as e:
                        logger.error("[WATCHER] scan error for %s: %s", fpath, e)
                        summary["errors"] += 1

        self._last_scan_at = time.time()
        logger.info(
            "[WATCHER] scan_complete scanned=%d new=%d skipped=%d errors=%d",
            summary["scanned"], summary["new"], summary["skipped"], summary["errors"],
        )
        return summary

    # ------------------------------------------------------------------
    # State query (for admin API)
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return current watcher state for the admin status endpoint."""
        return {
            "running": self._running,
            "last_scan_at": self._last_scan_at,
            "files_scanned": self._files_scanned,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _watch_all_dirs(self) -> None:
        """Register every org with a watch_dir on the observer."""
        orgs = self._get_orgs_with_watch_dirs()
        seen = set()
        for org in orgs:
            watch_dir = org.watch_dir
            if not watch_dir or not os.path.isdir(watch_dir):
                continue
            abs_dir = os.path.abspath(watch_dir)
            if abs_dir in seen:
                continue
            seen.add(abs_dir)
            handler = _WatcherHandler(
                service=self,
                org_id=int(org.id),
                watch_dir=abs_dir,
            )
            try:
                self._observer.schedule(  # type: ignore[union-attr]
                    handler, abs_dir, recursive=True
                )
                logger.info(
                    "[WATCHER] watching org_id=%s dir=%s", org.id, abs_dir
                )
            except Exception as e:
                logger.error(
                    "[WATCHER] failed to watch dir %s: %s", abs_dir, e
                )

    def _get_orgs_with_watch_dirs(self) -> list:
        """Return organisations that have a non-null watch_dir."""
        db: Session = SessionLocal()
        try:
            return (
                db.query(Organisation)
                .filter(Organisation.watch_dir.isnot(None))
                .all()
            )
        finally:
            db.close()

    def _handle_file(self, event_path: str, org_id: str, kb_id: Optional[str]) -> None:
        """Core logic: hash the file, check dedup, decide ingest or skip.

        Parses the file path to extract user_id and kb_id from the
        ``user_{uid}/kb_{kid}/`` convention.  If kb_id cannot be parsed,
        falls back to the provided *kb_id* argument (used by manual scan).
        """
        fname = os.path.basename(event_path)
        _, ext = os.path.splitext(fname)
        ext = ext.lower()

        # Skip non-supported extensions
        if ext not in SUPPORTED_EXTENSIONS:
            logger.debug(
                "[WATCHER] file_detected path=%s ext=%s action=skip reason=unsupported_ext",
                event_path, ext,
            )
            return

        # Skip hidden/system files
        if fname.startswith("."):
            logger.debug(
                "[WATCHER] file_detected path=%s action=skip reason=hidden_file",
                event_path,
            )
            return

        # Compute SHA-256 hash
        file_hash = self._compute_hash(event_path)
        hash_prefix = file_hash[:8] if file_hash else "none"

        # Parse path for user_id and kb_id
        user_id, resolved_kb_id = self._parse_path(event_path, org_id, kb_id)
        if resolved_kb_id is None:
            logger.warning(
                "[WATCHER] file_detected path=%s action=skip reason=unparseable_path",
                event_path,
            )
            return

        # Dedup check
        db: Session = SessionLocal()
        try:
            existing = (
                db.query(Document)
                .filter(
                    Document.file_hash == file_hash,
                    Document.knowledge_base_id == resolved_kb_id,
                )
                .first()
            )
            if existing:
                logger.info(
                    "[WATCHER] dedup_skip path=%s hash=%s kb_id=%s existing_doc_id=%s",
                    event_path, hash_prefix, resolved_kb_id, existing.id,
                )
                return
        finally:
            db.close()

        # File is new — trigger ingestion
        self._trigger_ingestion(event_path, fname, resolved_kb_id, org_id, user_id, file_hash)

    def _trigger_ingestion(
        self,
        event_path: str,
        file_name: str,
        kb_id: int,
        org_id: str,
        user_id: int,
        file_hash: str,
    ) -> None:
        """Create DocumentUpload + ProcessingTask records and enqueue background processing."""
        logger.info(
            "[WATCHER] ingestion_started path=%s kb_id=%s org_id=%s",
            event_path, kb_id, org_id,
        )

        # Get file size
        try:
            file_size = os.path.getsize(event_path)
        except OSError:
            file_size = 0

        # Determine content type
        _, ext = os.path.splitext(file_name)
        from app.services.document_processor import CONTENT_TYPE_MAP
        content_type = CONTENT_TYPE_MAP.get(ext, "application/octet-stream")

        # Create temporary path reference (the file is already in place)
        temp_path = event_path

        db: Session = SessionLocal()
        try:
            # Create DocumentUpload record
            upload = DocumentUpload(
                knowledge_base_id=kb_id,
                file_name=file_name,
                file_hash=file_hash,
                file_size=file_size,
                content_type=content_type,
                temp_path=temp_path,
                status="pending",
            )
            db.add(upload)
            db.commit()
            db.refresh(upload)

            # Create ProcessingTask record
            task = ProcessingTask(
                knowledge_base_id=kb_id,
                document_upload_id=upload.id,
                status="pending",
                progress=0,
                progress_message="Queued by watcher",
            )
            db.add(task)
            db.commit()
            db.refresh(task)

            logger.info(
                "[WATCHER] records_created upload_id=%s task_id=%s",
                upload.id, task.id,
            )

            # Enqueue background processing in a thread pool
            loop = asyncio.new_event_loop()
            future = self._executor.submit(
                self._run_ingestion,
                temp_path,
                file_name,
                kb_id,
                task.id,
                upload.id,
                user_id,
                db,
                loop,
            )
            future.add_done_callback(
                lambda f: self._on_ingestion_done(f, task.id, event_path)
            )

        except Exception as e:
            logger.error(
                "[WATCHER] failed to create ingestion records: %s", e, exc_info=True
            )
            try:
                db.rollback()
            except Exception:
                pass
        finally:
            db.close()

    def _run_ingestion(
        self,
        temp_path: str,
        file_name: str,
        kb_id: int,
        task_id: int,
        upload_id: int,
        user_id: int,
        db: Session,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Run the async ingestion pipeline in a dedicated event loop (threaded)."""
        try:
            async def _do() -> None:
                # Pass the existing session to process_document_background so it
                # doesn't close it — we manage the session lifecycle here.
                await process_document_background(
                    temp_path=temp_path,
                    file_name=file_name,
                    kb_id=kb_id,
                    task_id=task_id,
                    db=db,
                    user_id=user_id,
                )

            asyncio.set_event_loop(loop)
            loop.run_until_complete(_do())
        except Exception as e:
            logger.error(
                "[WATCHER] ingestion failed task_id=%s error=%s", task_id, e, exc_info=True
            )
        finally:
            loop.close()

    def _on_ingestion_done(self, future, task_id: int, event_path: str) -> None:
        """Callback after ingestion completes (success or failure)."""
        exc = future.exception()
        if exc:
            logger.error("[WATCHER] ingestion_future_error task_id=%s: %s", task_id, exc)
        else:
            logger.info("[WATCHER] ingestion_completed task_id=%s path=%s", task_id, event_path)

    @staticmethod
    def _compute_hash(path: str) -> str:
        """Compute SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except OSError as e:
            logger.warning("[WATCHER] hash_error path=%s: %s", path, e)
            return ""

    @staticmethod
    def _parse_path(
        event_path: str, org_id: str, kb_id: Optional[str]
    ) -> tuple[int, Optional[int]]:
        """Parse user_id and kb_id from the file path.

        Tries the ``user_{uid}/kb_{kid}/`` convention first.
        Falls back to the provided *kb_id* argument, then to None.
        """
        # Try regex match on the full path
        rel = event_path
        # Try relative to the watch_dir first, then as-is
        m = _PATH_RE.search(event_path)
        if m:
            return int(m.group(1)), int(m.group(2))

        # Fallback: use provided kb_id
        if kb_id is not None:
            return 0, int(kb_id)

        return 0, None


# ---------------------------------------------------------------------------
# Watchdog event handler
# ---------------------------------------------------------------------------


class _WatcherHandler:
    """Handles file system events from watchdog and delegates to WatcherService."""

    def __init__(self, service: WatcherService, org_id: int, watch_dir: str) -> None:
        self._service = service
        self._org_id = org_id
        self._watch_dir = watch_dir.rstrip("/")

    def _should_process(self, path: str) -> bool:
        """Check if the file extension is supported."""
        _, ext = os.path.splitext(path)
        return ext.lower() in SUPPORTED_EXTENSIONS

    def on_created(self, event) -> None:
        if event.is_directory:
            return
        if not self._should_process(event.src_path):
            return
        self._dispatch(event.src_path, "created")

    def on_modified(self, event) -> None:
        if event.is_directory:
            return
        if not self._should_process(event.src_path):
            return
        self._dispatch(event.src_path, "modified")

    def on_moved(self, event) -> None:
        if event.is_directory:
            return
        if not self._should_process(event.dest_path):
            return
        self._dispatch(event.dest_path, "moved")

    def _dispatch(self, path: str, event_type: str) -> None:
        """Apply debouncing then delegate to the service."""
        coalesced = self._service._debouncer.touch(path, event_type)
        if coalesced is None:
            return  # debounced

        # Update scanned count
        with self._service._lock:
            self._service._files_scanned += 1

        logger.info(
            "[WATCHER] file_detected path=%s type=%s org_id=%s",
            path, coalesced, self._org_id,
        )

        try:
            self._service._handle_file(path, str(self._org_id), None)
        except Exception as e:
            logger.error(
                "[WATCHER] handle_file_error path=%s org_id=%s: %s",
                path, self._org_id, e, exc_info=True,
            )
