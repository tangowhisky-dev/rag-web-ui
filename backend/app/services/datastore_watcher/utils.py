"""Shared utilities for datastore watcher and handler.

Eliminates duplication of pattern matching and file counting between
DatastoreFileEventHandler and DataStoreWatcher.
"""
import fnmatch
import hashlib
import logging
import os
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Fallback locks for SQLite (which doesn't support GET_LOCK).
# Keyed by (datastore_id, file_path) — cleaned up after release to
# prevent unbounded growth over long-running sessions.
_file_locks: dict[tuple[int, str], threading.Lock] = {}
_file_locks_guard = threading.Lock()
_FILE_LOCKS_MAX = 5000


def _cleanup_file_locks() -> None:
    """Remove unlocked entries to prevent unbounded dict growth."""
    with _file_locks_guard:
        if len(_file_locks) <= _FILE_LOCKS_MAX:
            return
        stale = [
            key for key, lock in _file_locks.items()
            if not lock.locked()
        ]
        for key in stale:
            _file_locks.pop(key, None)


def acquire_file_lock(db, datastore_id: int, file_path: str) -> bool:
    """Acquire a per-file advisory lock to prevent concurrent processing.

    Uses MySQL's GET_LOCK() when available. Falls back to an in-process
    threading.Lock for SQLite (tests).

    Returns True if the lock was acquired, False if another process/thread
    holds it.
    """
    lock_name = f"ingest_{datastore_id}_{hashlib.md5(file_path.encode()).hexdigest()}"
    try:
        from sqlalchemy import text
        result = db.execute(text("SELECT GET_LOCK(:lock, 0)"), {"lock": lock_name}).scalar()
        return bool(result)
    except Exception:
        # SQLite or other DB without GET_LOCK — use in-process lock
        key = (datastore_id, file_path)
        with _file_locks_guard:
            lock = _file_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                _file_locks[key] = lock
            elif not lock.locked():
                # Reuse an existing unlocked entry
                pass
        acquired = lock.acquire(blocking=False)
        if not acquired:
            _cleanup_file_locks()
        return acquired


def release_file_lock(db, datastore_id: int, file_path: str) -> None:
    """Release a per-file advisory lock."""
    lock_name = f"ingest_{datastore_id}_{hashlib.md5(file_path.encode()).hexdigest()}"
    try:
        from sqlalchemy import text
        db.execute(text("SELECT RELEASE_LOCK(:lock)"), {"lock": lock_name})
    except Exception:
        # SQLite fallback
        key = (datastore_id, file_path)
        with _file_locks_guard:
            lock = _file_locks.get(key)
            if lock is not None and lock.locked():
                try:
                    lock.release()
                except RuntimeError:
                    pass  # not held by this thread
            # Remove the entry if it's no longer held
            if lock is not None and not lock.locked():
                _file_locks.pop(key, None)



def matches_pattern(filepath: str, pattern: str = "*") -> bool:
    """Check if a filepath matches the scan pattern.

    Excludes hidden files and temp/lock files regardless of pattern.
    """
    fname = os.path.basename(filepath)
    if fname.startswith("."):
        return False
    # Skip temp/lock files from common editors and office suites:
    #   ~$file.docx  — MS Office lock files
    #   .~file.txt   — Emacs/gedit temp files
    #   file.tmp     — generic temp
    #   file.swp     — vim swap
    #   file.bak     — backup
    if fname.startswith("~$") or fname.startswith(".~"):
        return False
    ext = os.path.splitext(fname)[1].lower()
    if ext in (".tmp", ".swp", ".swo", ".bak", ".lock"):
        return False
    if pattern == "*":
        return True

    patterns = [p.strip() for p in pattern.split(",")]
    for pat in patterns:
        if "*" in pat:
            if fnmatch.fnmatch(fname, pat):
                return True
        else:
            if fname == pat:
                return True
    return False


def count_files_in_folder(folder_path: str, scan_pattern: str = "*") -> int:
    """Count files matching pattern in folder.

    Excludes hidden files, temp files, and files with unsupported extensions.
    Returns 0 on any error.
    """
    try:
        from app.services.ingestion.document_converter import SUPPORTED_EXTENSIONS
        path = Path(folder_path)
        if not path.exists():
            return 0

        patterns = [p.strip() for p in scan_pattern.split(",")]
        all_files = set()

        for pattern in patterns:
            if "*" in pattern:
                matched = list(path.rglob(pattern))
            else:
                matched = list(path.glob(pattern))
            for f in matched:
                if not f.is_file():
                    continue
                if not matches_pattern(str(f), scan_pattern):
                    continue
                ext = f.suffix.lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    continue
                all_files.add(f)

        return len(all_files)
    except Exception:
        return 0
