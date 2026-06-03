"""SMB share watcher — polls SMB shares for new/modified documents.

Implements polling-based scanning of SMB shares mirroring the local file
watcher flow: connect → list_path → download → hash → dedup → ingest → clean up.

Connection is per-scan (not persistent); one retry on connection error.
Socket timeout is 30 seconds.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.knowledge import Document
from app.services.document_processor import (
    SUPPORTED_EXTENSIONS,
    process_document_background,
)
from app.services.smb_auth import get_smb_auth

logger = logging.getLogger(__name__)

# Socket timeout for SMB connections (seconds)
_SMB_TIMEOUT = 30

# Max retries on connection error
_SMB_MAX_RETRIES = 1


class SMBShareWatcher:
    """Polls an SMB share for new/modified documents.

    Connects on-demand per scan (not persistent). One retry on connection error.
    Mirrors the local file watcher flow: download → hash → dedup → ingest → cleanup.
    """

    def __init__(
        self,
        host: str,
        share: str,
        username: str,
        password: str,
        domain: Optional[str] = None,
        kb_id: Optional[int] = None,
        poll_interval: int = 60,
    ) -> None:
        self.host = host
        self.share = share
        self.username = username
        self.password = password
        self.domain = domain
        self.kb_id = kb_id
        self.poll_interval = poll_interval
        self._last_scan_at: Optional[float] = None
        self._last_error: Optional[str] = None
        self._connected = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def test_connection(self) -> tuple[bool, str]:
        """Test connectivity to the SMB share.

        Connects to the SMB server, authenticates, and lists the root path.
        Returns (True, '') on success or (False, error_msg) on failure.
        """
        import smbprotocol.connection
        import smbprotocol.session
        import smbprotocol.share

        logger.info(
            "[SMB] test_connection host=%s share=%s username=%s",
            self.host, self.share, self.username,
        )

        try:
            conn = self._connect()
            try:
                share = conn.get_share(self.share)
                share.list_path("/")
                logger.info("[SMB] test_connection_ok host=%s share=%s", self.host, self.share)
                return True, ""
            finally:
                conn.close()
        except Exception as e:
            msg = str(e)
            logger.warning("[SMB] test_connection_failed host=%s share=%s error=%s", self.host, self.share, msg)
            return False, msg

    def scan(self) -> dict:
        """Scan the SMB share for new/modified documents.

        Connects, recursively lists all files, downloads each supported file
        to a temp directory, computes SHA-256, checks dedup via Document table,
        and triggers ingestion for new files. Cleans up temp files after.

        Returns {scanned, new, skipped, errors} counts.
        """
        import smbprotocol.connection
        import smbprotocol.session
        import smbprotocol.share

        result = {"scanned": 0, "new": 0, "skipped": 0, "errors": 0}

        logger.info(
            "[SMB] scan_start host=%s share=%s",
            self.host, self.share,
        )

        temp_base = None
        try:
            temp_base = tempfile.mkdtemp(prefix=f"rag-smb-{uuid.uuid4().hex[:8]}-")

            # Connect with retry
            conn = None
            last_exc: Optional[Exception] = None
            for attempt in range(_SMB_MAX_RETRIES + 1):
                try:
                    conn = self._connect()
                    break
                except Exception as e:
                    last_exc = e
                    if attempt < _SMB_MAX_RETRIES:
                        logger.warning(
                            "[SMB] scan_retry host=%s share=%s attempt=%d error=%s",
                            self.host, self.share, attempt + 1, e,
                        )
                        time.sleep(1)
                    else:
                        logger.error(
                            "[SMB] scan_connection_failed host=%s share=%s error=%s",
                            self.host, self.share, e,
                        )

            if conn is None:
                self._last_error = str(last_exc) if last_exc else "Connection failed"
                result["errors"] += 1
                self._connected = False
                return result

            self._connected = True
            self._last_error = None
            share = conn.get_share(self.share)

            # Recursively list all files
            files = self._list_all_files(share, "/")

            for file_info in files:
                result["scanned"] += 1
                try:
                    self._process_file(share, file_info, temp_base, result)
                except Exception as e:
                    logger.error(
                        "[SMB] scan_file_error host=%s share=%s path=%s error=%s",
                        self.host, self.share, file_info["long_name"], e,
                    )
                    result["errors"] += 1

        finally:
            # Clean up temp directory
            if temp_base and os.path.isdir(temp_base):
                try:
                    import shutil
                    shutil.rmtree(temp_base, ignore_errors=True)
                except Exception as e:
                    logger.warning("[SMB] cleanup_temp_failed dir=%s error=%s", temp_base, e)

        self._last_scan_at = time.time()
        logger.info(
            "[SMB] scan_complete host=%s share=%s scanned=%d new=%d skipped=%d errors=%d",
            self.host, self.share,
            result["scanned"], result["new"], result["skipped"], result["errors"],
        )
        return result

    def get_status(self) -> dict:
        """Return status dict for this share (for admin status endpoint)."""
        return {
            "host": self.host,
            "share": self.share,
            "connected": self._connected,
            "last_scan_at": self._last_scan_at,
            "last_error": self._last_error,
            "poll_interval": self.poll_interval,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> object:
        """Create and authenticate an SMB connection.

        Returns an smbprotocol.connection.Connection instance.
        Uses socket timeout of 30 seconds.
        """
        import smbprotocol.connection
        import smbprotocol.session
        import smbprotocol.share

        conn = smbprotocol.connection.Connection(
            uuid=None,  # Let library generate UUID
            is_encrypted=False,
            sock_timeout=_SMB_TIMEOUT,
        )
        conn.connect(self.host)

        auth_methods = []
        if self.domain:
            auth_methods.append(self.domain)
        auth_methods.append("")

        for auth_method in auth_methods:
            try:
                session = smbprotocol.session.Session(
                    conn,
                    self.username,
                    self.password,
                    requested_dialects=["SMB_3_1_1"],
                    auth_method=auth_method,
                )
                session.connect()
                return conn
            except Exception:
                continue

        raise ConnectionError(
            f"Failed to authenticate to SMB server {self.host} with user '{self.username}'"
        )

    def _list_all_files(self, share, base_path: str) -> list:
        """Recursively list all files under base_path on the SMB share."""
        files = []
        try:
            for entry in share.list_path(base_path, directory_only=False):
                full_path = f"{base_path}{entry['file_name']}"
                if entry["file_directory"]:
                    # It's a directory — recurse
                    files.extend(self._list_all_files(share, full_path + "/"))
                else:
                    # It's a file
                    files.append({
                        "path": full_path,
                        "file_name": entry["file_name"],
                        "long_name": entry["long_name"],
                        "size": entry["end_of_file"],
                    })
        except Exception as e:
            logger.warning("[SMB] list_path_failed path=%s error=%s", base_path, e)
        return files

    def _process_file(
        self, share, file_info: dict, temp_base: str, result: dict
    ) -> None:
        """Download, hash, dedup-check, and ingest a single SMB file."""
        import smbprotocol.share

        file_path = file_info["path"]
        file_name = file_info["long_name"] or file_info["file_name"]
        file_size = file_info["size"]

        _, ext = os.path.splitext(file_name)
        ext = ext.lower()

        # Skip non-supported extensions
        if ext not in SUPPORTED_EXTENSIONS:
            logger.debug(
                "[SMB] skip_unsupported_ext path=%s ext=%s",
                file_path, ext,
            )
            result["skipped"] += 1
            return

        # Skip hidden/system files
        if file_name.startswith("."):
            logger.debug("[SMB] skip_hidden_file path=%s", file_path)
            result["skipped"] += 1
            return

        # Download file to temp
        temp_path = os.path.join(temp_base, str(uuid.uuid4().hex), file_name)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)

        try:
            with open(temp_path, "wb") as f:
                with share.open_file(file_path, smbprotocol.share.ACCESS_MASK.GENERIC_READ) as smb_file:
                    data = smb_file.read()
                    f.write(data)
                    logger.info(
                        "[SMB] download_progress path=%s size=%d",
                        file_path, len(data),
                    )
        except Exception as e:
            logger.warning("[SMB] download_failed path=%s error=%s", file_path, e)
            raise

        try:
            # Compute SHA-256 hash
            file_hash = self._compute_hash(temp_path)
            hash_prefix = file_hash[:8] if file_hash else "none"

            # Determine kb_id: use per-share kb_id or fallback
            resolved_kb_id = self.kb_id

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
                        "[SMB] dedup_skip path=%s hash=%s kb_id=%s existing_doc_id=%s",
                        file_path, hash_prefix, resolved_kb_id, existing.id,
                    )
                    result["skipped"] += 1
                    return
            finally:
                db.close()

            # File is new — trigger ingestion
            self._trigger_ingestion_smb(temp_path, file_name, resolved_kb_id, file_path, file_hash)
            result["new"] += 1

        finally:
            # Clean up the downloaded temp file (ingestion copies it)
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except OSError:
                pass

    def _trigger_ingestion_smb(
        self,
        temp_path: str,
        file_name: str,
        kb_id: Optional[int],
        smb_path: str,
        file_hash: str,
    ) -> None:
        """Create DocumentUpload + ProcessingTask and enqueue background processing for an SMB file."""
        if kb_id is None:
            logger.warning(
                "[SMB] ingestion_skipped_no_kb smb_path=%s", smb_path,
            )
            return

        logger.info(
            "[SMB] ingestion_started smb_path=%s kb_id=%s file_name=%s",
            smb_path, kb_id, file_name,
        )

        # Get file size
        try:
            file_size = os.path.getsize(temp_path)
        except OSError:
            file_size = 0

        # Determine content type
        _, ext = os.path.splitext(file_name)
        from app.services.document_processor import CONTENT_TYPE_MAP
        content_type = CONTENT_TYPE_MAP.get(ext, "application/octet-stream")

        db: Session = SessionLocal()
        try:
            # Create DocumentUpload record
            from app.models.knowledge import DocumentUpload, ProcessingTask

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
                progress_message="Queued by SMB watcher",
            )
            db.add(task)
            db.commit()
            db.refresh(task)

            logger.info(
                "[SMB] records_created upload_id=%s task_id=%s smb_path=%s",
                upload.id, task.id, smb_path,
            )

            # Enqueue background processing
            import asyncio
            from concurrent.futures import ThreadPoolExecutor

            loop = asyncio.new_event_loop()
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="smb-ingest")

            def _run() -> None:
                try:
                    async def _do() -> None:
                        await process_document_background(
                            temp_path=temp_path,
                            file_name=file_name,
                            kb_id=kb_id,
                            task_id=task.id,
                            db=db,
                            user_id=0,
                        )
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(_do())
                except Exception as e:
                    logger.error(
                        "[SMB] ingestion_failed smb_path=%s task_id=%s error=%s",
                        smb_path, task.id, e, exc_info=True,
                    )
                finally:
                    loop.close()

            future = executor.submit(_run)
            future.add_done_callback(
                lambda f: executor.shutdown(wait=False, cancel_futures=True)
            )

        except Exception as e:
            logger.error(
                "[SMB] failed_to_create_ingestion_records smb_path=%s error=%s",
                smb_path, e, exc_info=True,
            )
            try:
                db.rollback()
            except Exception:
                pass
        finally:
            db.close()

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
            logger.warning("[SMB] hash_error path=%s: %s", path, e)
            return ""
