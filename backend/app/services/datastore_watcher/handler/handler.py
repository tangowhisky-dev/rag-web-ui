"""DatastoreFileEventHandler — composed handler for filesystem events.

Combines all handler mixins into a single ``DatastoreFileEventHandler``
class that inherits from ``watchdog.events.FileSystemEventHandler``.

Also defines two helper classes:
- ``_Debouncer``: coalesces rapid repeated events for the same path.
- ``_SyntheticEvent``: synthetic file event for delayed dispatch after
  write-completion delay.

The handler maintains a mapping of datastore_id → (org_id, folder_path,
min_interval_seconds). When an event fires, it resolves the datastore
by checking which datastore's folder_path contains the event path.

Mixin composition (MRO order):
1. FolderMixin — folder registration, batch timers, datastore resolution
2. DispatchMixin — watchdog callbacks, event dispatch, debouncing
3. IngestMixin — file processing, manifest, ingestion pipeline
4. DeleteMixin — deletion flow, Qdrant/Neo4j cleanup
5. ChangesMixin — pending change queue, batch callback, progress tracking
6. FileSystemEventHandler — watchdog base class
"""

from __future__ import annotations

import logging
import os
import threading
import time as time_module
from typing import Any, Dict, List, Optional, Tuple

from watchdog.events import FileSystemEventHandler

from app.services.datastore_watcher.handler.folder import FolderMixin
from app.services.datastore_watcher.handler.dispatch import DispatchMixin
from app.services.datastore_watcher.handler.ingest import IngestMixin
from app.services.datastore_watcher.handler.delete import DeleteMixin
from app.services.datastore_watcher.handler.changes import ChangesMixin

logger = logging.getLogger(__name__)


# Root folder watched by the observer (all datastores live under this)
WATCH_ROOT = "/app/data"


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


class _SyntheticEvent:
    """A synthetic file event created for delayed dispatch after write-completion delay.

    Watchdog's observer fires events immediately, but we want to delay processing
    by ~1 second to let the file write complete (especially for editors like VS Code
    that write via temp files and renames). This class provides a minimal event-like
    object with the attributes needed by the handler (src_path, is_directory, event_type, etc).
    """

    def __init__(self, src_path: str, is_directory: bool = False, event_type: str = "modified") -> None:
        self.src_path = src_path
        self.is_directory = is_directory
        self.src_dir = os.path.dirname(src_path)
        self.dest_path = ""
        self.dest_dir = ""
        self.cookie = 0
        self.name = os.path.basename(src_path)
        self.dir = is_directory
        self.event_type = event_type


class DatastoreFileEventHandler(
    FolderMixin,
    DispatchMixin,
    IngestMixin,
    DeleteMixin,
    ChangesMixin,
    FileSystemEventHandler,
):
    """Global event handler that resolves datastore from event path.

    Maintains a mapping of datastore_id -> (org_id, folder_path, min_interval_seconds).
    When an event fires, it resolves the datastore by checking which datastore's
    folder_path contains the event path.

    Features:
    - Debouncing per-file (prevents duplicate events from editors)
    - Per-datastore batch processing with configurable intervals (default: 5 min)
    - org_id and datastore_id tracking for each event
    - File processing (ingestion/update) via _handle_file
    """

    def __init__(
        self,
        callback,
        executor,
        debounce_ms: int = 1000,
        default_min_interval_seconds: int = 300,  # 5 minutes
        debouncer: Optional[_Debouncer] = None,
        progress_lock: Optional[threading.Lock] = None,
    ) -> None:
        self.callback = callback
        self._executor = executor
        self.debounce_ms = debounce_ms / 1000.0
        self.default_min_interval_seconds = default_min_interval_seconds
        self.debouncer = debouncer

        # datastore_id -> (org_id, folder_path, min_interval_seconds)
        self.folder_paths: Dict[int, Tuple[int, str, int]] = {}

        # datastore_id -> list of pending changes
        self.pending_changes: Dict[int, List[Dict[str, Any]]] = {}

        # Per-file debouncing: file_path -> last call timestamp (monotonic)
        self._last_call: Dict[str, float] = {}

        # Track delayed-dispatch threads so we can wait for them on shutdown.
        self._delayed_threads: List[threading.Thread] = []
        self._delayed_threads_lock = threading.Lock()

        # Processing state flag — per-datastore set to allow independent
        # processing across different datastores (a slow ingestion for one
        # datastore doesn't block others).
        self._processing: set[int] = set()

        # Per-datastore batch timers for auto_process_enabled datastores.
        # When auto_process is enabled, file events are queued but NOT processed
        # immediately — they accumulate until the timer fires, then the batch
        # is processed. This matches the UI description: "File changes are
        # automatically processed every N minutes."
        # When auto_process is disabled (min_interval_seconds <= 0), events are
        # processed immediately (the original behavior).
        self._batch_timers: Dict[int, threading.Timer] = {}
        self._batch_timers_lock = threading.Lock()

        self._lock = threading.Lock()

        # Lock for scan progress updates — prevents race between event-driven
        # ingestion and manual scan when both update last_scan_processed
        # simultaneously. Shared with DataStoreWatcher to avoid double-counting.
        self._progress_lock = progress_lock if progress_lock is not None else threading.Lock()
