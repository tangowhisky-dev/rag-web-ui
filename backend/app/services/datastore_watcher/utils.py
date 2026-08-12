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
# Keyed by (datastore_id, file_path) — cleaned up after use.
_file_locks: dict[tuple[int, str], threading.Lock] = {}
_file_locks_guard = threading.Lock()


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
        return lock.acquire(blocking=False)


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



def matches_pattern(filepath: str, pattern: str = "*") -> bool:
    """Check if a filepath matches the scan pattern.

    Excludes hidden files regardless of pattern.
    """
    fname = os.path.basename(filepath)
    if fname.startswith("."):
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

    Excludes hidden files. Returns 0 on any error.
    """
    try:
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
            all_files.update(f for f in matched if f.is_file() and not f.name.startswith("."))

        return len(all_files)
    except Exception:
        return 0
