"""DatastoreFileEventHandler sub-package — filesystem event handling for datastore watches.

Re-exports the public API:
- DatastoreFileEventHandler: global handler that resolves datastore from event path
- _Debouncer: coalesces rapid repeated events for the same path
- _SyntheticEvent: synthetic file event for delayed dispatch
"""

from __future__ import annotations

from app.services.datastore_watcher.handler.handler import (
    DatastoreFileEventHandler,
    _Debouncer,
    _SyntheticEvent,
)

__all__ = [
    "DatastoreFileEventHandler",
    "_Debouncer",
    "_SyntheticEvent",
]
