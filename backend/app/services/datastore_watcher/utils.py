"""Shared utilities for datastore watcher and handler.

Eliminates duplication of pattern matching and file counting between
DatastoreFileEventHandler and DataStoreWatcher.
"""
import fnmatch
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


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
