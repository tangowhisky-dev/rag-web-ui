"""DataStore Watcher Service — global file watching with per-datastore batching.

Architecture:
- Single ``watchdog.Observer`` instance with ``recursive=True`` watching the root
  folder (/app/data/). On Linux this uses ``InotifyObserver`` (instant event
  delivery via inotify), on macOS ``FSEventsObserver``, on Windows ``ReadDirectoryChangesW``.
- Single ``DatastoreFileEventHandler`` that resolves datastore from event path
- Handler maintains a mapping of datastore_id -> (org_id, folder_path, min_interval)
- When an event fires, handler resolves datastore by checking folder_path prefix
- Per-datastore batch timers fire at configurable intervals
- Progress tracking for manual scans

File path convention inside a watched directory:
    kb_{kb_id}/{file_name}

The service parses the path to determine which knowledge base owns the file,
then routes it into the existing ``process_document_background()`` pipeline.

Progress tracking:
- Each scan is assigned an integer scan_id
- The scan_id is stored on the ProcessingTask so it can be matched to a running scan
- When a scan starts, the datastore is updated with scan_id and progress=0
- As files are processed, progress is updated
- When a scan ends, the scan_id is cleared

Cancellation:
- A running scan can be stopped by setting scan_id=None on the datastore
- The scan checks this between files and stops early if cancelled
"""

from __future__ import annotations

from app.db.session import SessionLocal
from app.services.datastore_watcher.watcher import DataStoreWatcher
from app.services.datastore_watcher.handler import (
    DatastoreFileEventHandler,
    _Debouncer,
    _SyntheticEvent,
)

__all__ = [
    "SessionLocal",
    "DataStoreWatcher",
    "DatastoreFileEventHandler",
    "_Debouncer",
    "_SyntheticEvent",
]
