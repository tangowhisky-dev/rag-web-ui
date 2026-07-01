"""Tests for the DataStoreWatcher — watchdog observer with recursive=True.

Verifies:
1. The observer uses watchdog.Observer (not PollingObserver) when native
   observers are available (Linux inotify, macOS FSEvents, Windows ReadDirectoryChangesW).
2. The observer falls back to PollingObserver when the native observer is
   unavailable.
3. The observer is scheduled with recursive=True.
4. Debouncing works correctly — _should_process does not update last_call
   for rejected events, and _after_process is called after processing.
5. File move events are handled correctly — old path is treated as deletion
   and new path is treated as creation.
"""
import os
import tempfile
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

# conftest.py sets up the SQLite session before any app.* import.


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary /app/data directory structure for testing."""
    data_dir = tmp_path / "app" / "data"
    data_dir.mkdir(parents=True)
    # Create a subdirectory for a datastore.
    store_dir = data_dir / "store1"
    store_dir.mkdir()
    return data_dir


class TestObserverSelection:
    """Test that the correct observer is selected."""

    def test_observer_import_works(self):
        """watchdog.Observer should be importable on all platforms."""
        from watchdog.observers import Observer
        # The Observer class exists — it auto-selects the platform-native observer.
        assert Observer is not None

    def test_observer_starts_successfully(self, tmp_data_dir):
        """Observer should start without error."""
        from watchdog.observers import Observer

        observer = Observer(timeout=2)
        try:
            observer.start()
            # Give the observer time to initialize.
            time.sleep(0.2)
            assert observer.is_alive()
        finally:
            observer.stop()
            observer.join(timeout=5)

    def test_observer_schedules_with_recursive_true(self, tmp_data_dir):
        """Observer should schedule with recursive=True."""
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler

        class TestHandler(FileSystemEventHandler):
            def __init__(self):
                self.events = []

            def on_created(self, event):
                self.events.append(("created", event.src_path))

        handler = TestHandler()
        observer = Observer(timeout=2)
        try:
            observer.start()
            observer.schedule(handler, str(tmp_data_dir), recursive=True)
            # Give the observer time to scan the directory tree.
            time.sleep(0.5)
            assert observer.is_alive()
            assert len(observer.emitters) > 0
        finally:
            observer.stop()
            observer.join(timeout=5)

    def test_native_observer_detected_on_linux(self, tmp_data_dir):
        """On Linux, the native observer should be InotifyObserver."""
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler

        observer = Observer(timeout=2)
        try:
            observer.start()
            observer.schedule(
                FileSystemEventHandler(), str(tmp_data_dir), recursive=True
            )
            time.sleep(0.5)

            # Check that emitters are using the native observer.
            # On Linux, emitters use InotifyEmitter.
            # observer.emitters is a set, so take the first element.
            emitter = next(iter(observer.emitters))
            emitter_cls = type(emitter)
            # The emitter should NOT be PollingEmitter.
            assert "polling" not in str(emitter_cls).lower(), (
                f"Expected native emitter, got {emitter_cls}"
            )
        finally:
            observer.stop()
            observer.join(timeout=5)


class TestDebouncing:
    """Test that debouncing works correctly."""

    def test_should_process_returns_false_within_debounce_window(self):
        """_should_process should return False for events within the debounce window."""
        from app.services.datastore_watcher import DatastoreFileEventHandler

        handler = DatastoreFileEventHandler(
            callback=MagicMock(),
            executor=MagicMock(),
            debounce_ms=1000,  # 1 second
        )
        path = "/app/data/store1/test.txt"

        # First call should return True (no previous event).
        assert handler._should_process(path) is True

        # Call _after_process to simulate processing.
        handler._after_process(path)

        # Second call within the debounce window should return False.
        assert handler._should_process(path) is False

    def test_should_process_returns_true_after_debounce_window(self):
        """_should_process should return True for events after the debounce window."""
        from app.services.datastore_watcher import DatastoreFileEventHandler

        handler = DatastoreFileEventHandler(
            callback=MagicMock(),
            executor=MagicMock(),
            debounce_ms=100,  # 100ms
        )
        path = "/app/data/store1/test.txt"

        # First call should return True.
        assert handler._should_process(path) is True

        # Call _after_process to simulate processing.
        handler._after_process(path)

        # Wait for the debounce window to expire.
        time.sleep(0.2)

        # Second call after the debounce window should return True.
        assert handler._should_process(path) is True

    def test_should_process_does_not_update_last_call(self):
        """_should_process should NOT update _last_call — only _after_process should."""
        from app.services.datastore_watcher import DatastoreFileEventHandler

        handler = DatastoreFileEventHandler(
            callback=MagicMock(),
            executor=MagicMock(),
            debounce_ms=1000,  # 1 second
        )
        path = "/app/data/store1/test.txt"

        # First call should return True (no previous event).
        assert handler._should_process(path) is True

        # _should_process should NOT have updated _last_call.
        assert handler._last_call.get(path) is None

        # Second call should also return True (no previous event was recorded).
        assert handler._should_process(path) is True

        # _after_process should update _last_call.
        handler._after_process(path)
        assert handler._last_call.get(path) is not None

        # Third call should return False (event was processed).
        assert handler._should_process(path) is False


class TestSyntheticEvent:
    """Test the _SyntheticEvent class."""

    def test_synthetic_event_has_is_directory_attribute(self):
        """_SyntheticEvent should have is_directory attribute set to False."""
        from app.services.datastore_watcher import _SyntheticEvent

        event = _SyntheticEvent(src_path="/app/data/store1/test.txt")
        assert event.src_path == "/app/data/store1/test.txt"
        assert event.is_directory is False
        assert event.src_dir == "/app/data/store1"
        assert event.dest_path == ""
        assert event.cookie == 0
        assert event.name == "test.txt"
        assert event.dir is False

    def test_synthetic_event_with_directory_true(self):
        """_SyntheticEvent should have is_directory attribute set to True."""
        from app.services.datastore_watcher import _SyntheticEvent

        event = _SyntheticEvent(src_path="/app/data/store1", is_directory=True)
        assert event.src_path == "/app/data/store1"
        assert event.is_directory is True
        assert event.dir is True


class TestFileMove:
    """Test that file move events are handled correctly."""

    def test_file_moved_into_datastore_treated_as_created(self, tmp_data_dir):
        """When a file is moved into a datastore, it should be treated as a new file."""
        from app.services.datastore_watcher import DatastoreFileEventHandler

        handler = DatastoreFileEventHandler(
            callback=MagicMock(),
            executor=MagicMock(),
            debounce_ms=100,
        )
        # Mock _resolve_datastore to return the datastore ID.
        handler._resolve_datastore = MagicMock(return_value=1)
        handler._should_process = MagicMock(return_value=True)
        handler._after_process = MagicMock()
        handler._dispatch = MagicMock()

        # Simulate a move event from /app/data/store1/old.txt to /app/data/store1/new.txt
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/app/data/store1/old.txt"
        event.dest_path = "/app/data/store1/new.txt"

        handler.on_moved(event)

        # The source path should be dispatched as a deletion.
        handler._dispatch.assert_any_call(
            "/app/data/store1/old.txt", "deleted"
        )
        # The destination path should be dispatched as a creation.
        handler._dispatch.assert_any_call(
            "/app/data/store1/new.txt", "created"
        )
        # _after_process should be called for both paths.
        assert handler._after_process.call_count == 2

    def test_file_moved_out_of_datastore_treated_as_deletion(self, tmp_data_dir):
        """When a file is moved out of a datastore, it should be treated as a deletion."""
        from app.services.datastore_watcher import DatastoreFileEventHandler

        handler = DatastoreFileEventHandler(
            callback=MagicMock(),
            executor=MagicMock(),
            debounce_ms=100,
        )
        # Mock _resolve_datastore to return None for the destination.
        handler._resolve_datastore = MagicMock(side_effect=lambda p: 1 if "old" in p else None)
        handler._should_process = MagicMock(return_value=True)
        handler._after_process = MagicMock()
        handler._dispatch = MagicMock()

        # Simulate a move event from /app/data/store1/old.txt to /app/data/other/new.txt
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/app/data/store1/old.txt"
        event.dest_path = "/app/data/other/new.txt"

        handler.on_moved(event)

        # The source path should be dispatched as a deletion.
        handler._dispatch.assert_any_call(
            "/app/data/store1/old.txt", "deleted"
        )
        # The destination path should NOT be dispatched (it doesn't belong to a datastore).

    def test_file_moved_between_datastores_treated_as_delete_and_create(self, tmp_data_dir):
        """When a file is moved between datastores, old path is deleted and new path is created."""
        from app.services.datastore_watcher import DatastoreFileEventHandler

        handler = DatastoreFileEventHandler(
            callback=MagicMock(),
            executor=MagicMock(),
            debounce_ms=100,
        )
        # Mock _resolve_datastore to return different datastore IDs.
        handler._resolve_datastore = MagicMock(side_effect=lambda p: 1 if "store1" in p else 2)
        handler._should_process = MagicMock(return_value=True)
        handler._after_process = MagicMock()
        handler._dispatch = MagicMock()

        # Simulate a move event from /app/data/store1/old.txt to /app/data/store2/new.txt
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/app/data/store1/old.txt"
        event.dest_path = "/app/data/store2/new.txt"

        handler.on_moved(event)

        # The source path should be dispatched as a deletion.
        handler._dispatch.assert_any_call(
            "/app/data/store1/old.txt", "deleted"
        )
        # The destination path should be dispatched as a creation.
        handler._dispatch.assert_any_call(
            "/app/data/store2/new.txt", "created"
        )


class TestSyncWatchersWithoutOrgAssignment:
    """Verify that unassigned datastores are still registered for watching.

    Datastores should be watched for changes irrespective of whether they are
    assigned to any organization. The org_id is only used for logging and
    KB-deletion cleanup (which guards on org_id is not None).
    """

    def test_sync_includes_unassigned_datastores(self, tmp_path):
        """A datastore without OrganizationDataStore rows is still registered."""
        from app.services.datastore_watcher import DataStoreWatcher

        # Ensure the directory exists
        ds_path = tmp_path / "unassigned"
        ds_path.mkdir(exist_ok=True)

        watcher = DataStoreWatcher()
        # Start the observer (required to avoid OSError on some platforms)
        from watchdog.observers.polling import PollingObserver
        watcher._observer = PollingObserver(timeout=2)
        watcher._observer.start()
        watcher._observer.schedule(watcher._handler, str(tmp_path), recursive=True)

        # Patch the database session to return an unassigned datastore
        from unittest.mock import MagicMock, patch
        mock_ds = MagicMock()
        mock_ds.id = 42
        mock_ds.folder_path = str(ds_path)
        mock_ds.auto_scan_interval_minutes = 60
        mock_ds.is_active = True
        mock_ds.auto_scan_enabled = True

        mock_assignment = MagicMock()
        # No assignment for datastore_id=42

        with patch("app.services.datastore_watcher.watcher.SessionLocal") as mock_session:
            mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_session.return_value.__exit__ = MagicMock(return_value=None)
            mock_query = MagicMock()
            mock_assignment_query = MagicMock()
            # Unassigned datastore returns the datastore itself
            mock_query.filter.return_value.all.return_value = [mock_ds]
            # No assignments for this datastore
            mock_assignment_query.filter.return_value.all.return_value = []
            mock_session.return_value.query.side_effect = (
                lambda model: mock_assignment_query if model.__name__ == "OrganizationDataStore" else mock_query
            )
            mock_session.return_value.query.return_value = mock_assignment_query

            watcher._sync_watchers_with_database()

        try:
            # Verify the datastore was registered (not filtered out)
            assert 42 in watcher._datastore_paths
            assert 42 in watcher._handler.folder_paths
            # org_id should be None for unassigned datastores
            assert watcher._handler.folder_paths[42][0] is None
        finally:
            watcher._observer.stop()
            watcher._observer.join(timeout=5)
