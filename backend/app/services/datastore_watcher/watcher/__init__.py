"""DataStoreWatcher sub-package — lifecycle, scan orchestration, and recovery management.

Re-exports the public API:
- DataStoreWatcher: manages the watchdog observer, scan lifecycle, and progress tracking
- SessionLocal: re-exported for backward compatibility with tests that import it
  from ``app.services.datastore_watcher.watcher``
"""

from __future__ import annotations

from app.db.session import SessionLocal
from app.services.datastore_watcher.watcher.watcher import DataStoreWatcher

__all__ = [
    "DataStoreWatcher",
    "SessionLocal",
]
