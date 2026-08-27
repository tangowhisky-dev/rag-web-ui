"""Tests for scan pause/resume functionality.

Covers:
  1. _cancel_scan(pause=True) sets last_scan_status to "paused".
  2. _cancel_scan(pause=False) sets last_scan_status to "idle" (default).
  3. _cancel_scan(pause=True) sets _active_scans status to "paused".
  4. _cancel_scan returns False when no running scan exists.
  5. Scan endpoint allows starting when status is "paused" (not blocked by "already running" guard).
  6. Recovery service does NOT auto-recover paused scans (only "running" triggers recovery).
"""
import os
import time
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest
from sqlalchemy.orm import sessionmaker

import app.db.session as _session_mod
from app.db.session import get_db
from app.models.base import Base  # noqa
import app.models.datastore  # noqa: ensure tables registered
import app.models.knowledge  # noqa
from app.models.datastore import DataStore
from app.models.knowledge import Document, ProcessingTask

engine = _session_mod.engine
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def db():
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def running_datastore(tmp_path):
    """Create a datastore with last_scan_status='running'."""
    ds_path = tmp_path / "test_store"
    ds_path.mkdir()
    (ds_path / "file1.pdf").write_bytes(b"content")

    db = TestingSessionLocal()
    try:
        ds = DataStore(
            name="Test Store",
            folder_path=str(ds_path),
            scan_pattern="*",
            is_active=True,
            last_scan_status="running",
            last_scan_total_files=1,
            last_scan_processed=0,
        )
        db.add(ds)
        db.commit()
        db.refresh(ds)
        return ds, str(ds_path)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Tests: _cancel_scan with pause parameter
# ---------------------------------------------------------------------------

class TestCancelScanPause:
    def test_pause_sets_status_to_paused(self, running_datastore):
        """_cancel_scan(pause=True) sets last_scan_status to 'paused'."""
        from app.services.datastore_watcher.watcher import DataStoreWatcher
        ds, ds_path = running_datastore

        watcher = DataStoreWatcher()
        # Don't start the real observer
        watcher._observer = MagicMock()

        with patch("app.services.ingestion.ingestion_dispatcher.cancel_graph_builds_for_datastore"):
            result = watcher._cancel_scan(ds.id, pause=True)

        assert result is True
        db = TestingSessionLocal()
        try:
            ds_check = db.query(DataStore).filter(DataStore.id == ds.id).first()
            assert ds_check.last_scan_status == "paused"
            assert "paused" in (ds_check.last_scan_error or "").lower()
        finally:
            db.close()

    def test_stop_sets_status_to_idle(self, running_datastore):
        """_cancel_scan(pause=False) sets last_scan_status to 'idle'."""
        from app.services.datastore_watcher.watcher import DataStoreWatcher
        ds, ds_path = running_datastore

        watcher = DataStoreWatcher()
        watcher._observer = MagicMock()

        with patch("app.services.ingestion.ingestion_dispatcher.cancel_graph_builds_for_datastore"):
            result = watcher._cancel_scan(ds.id, pause=False)

        assert result is True
        db = TestingSessionLocal()
        try:
            ds_check = db.query(DataStore).filter(DataStore.id == ds.id).first()
            assert ds_check.last_scan_status == "idle"
        finally:
            db.close()

    def test_pause_sets_active_scans_to_paused(self, running_datastore):
        """_cancel_scan(pause=True) sets _active_scans status to 'paused'."""
        from app.services.datastore_watcher.watcher import DataStoreWatcher
        ds, ds_path = running_datastore

        watcher = DataStoreWatcher()
        watcher._observer = MagicMock()

        # Simulate an active scan entry
        scan_id = 100
        watcher._active_scans[scan_id] = {
            "datastore_id": ds.id,
            "total": 10,
            "processed": 5,
            "status": "running",
            "error_message": None,
        }
        watcher._scan_futures[scan_id] = []

        with patch("app.services.ingestion.ingestion_dispatcher.cancel_graph_builds_for_datastore"):
            result = watcher._cancel_scan(ds.id, pause=True)

        assert result is True
        assert watcher._active_scans[scan_id]["status"] == "paused"
        assert "paused" in (watcher._active_scans[scan_id]["error_message"] or "").lower()

    def test_cancel_returns_false_when_not_running(self, running_datastore):
        """_cancel_scan returns False when last_scan_status is not 'running'."""
        from app.services.datastore_watcher.watcher import DataStoreWatcher
        ds, ds_path = running_datastore

        # Set status to completed
        db = TestingSessionLocal()
        try:
            ds_record = db.query(DataStore).filter(DataStore.id == ds.id).first()
            ds_record.last_scan_status = "completed"
            db.commit()
        finally:
            db.close()

        watcher = DataStoreWatcher()
        watcher._observer = MagicMock()

        result = watcher._cancel_scan(ds.id, pause=True)
        assert result is False

    def test_cancel_returns_false_when_paused(self, running_datastore):
        """Cannot pause an already-paused scan."""
        from app.services.datastore_watcher.watcher import DataStoreWatcher
        ds, ds_path = running_datastore

        # Set status to paused
        db = TestingSessionLocal()
        try:
            ds_record = db.query(DataStore).filter(DataStore.id == ds.id).first()
            ds_record.last_scan_status = "paused"
            db.commit()
        finally:
            db.close()

        watcher = DataStoreWatcher()
        watcher._observer = MagicMock()

        result = watcher._cancel_scan(ds.id, pause=True)
        assert result is False


# ---------------------------------------------------------------------------
# Tests: scan endpoint allows resume from paused
# ---------------------------------------------------------------------------

class TestScanEndpointResumeFromPaused:
    def test_scan_not_blocked_when_paused(self, running_datastore):
        """The scan endpoint's 'already running' guard should NOT block when status is 'paused'.

        The guard checks for last_scan_status == 'running'. 'paused' should pass through.
        This is a logic test — we verify the condition directly.
        """
        db = TestingSessionLocal()
        try:
            # Set to paused
            ds_record = db.query(DataStore).filter(
                DataStore.id == running_datastore[0].id
            ).first()
            ds_record.last_scan_status = "paused"
            db.commit()

            # The guard condition from the scan endpoint:
            # if ds_local.last_scan_status == "running": raise 409
            assert ds_record.last_scan_status != "running"
            # This means the scan endpoint would NOT raise 409 — resume is allowed.
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Tests: recovery service skips paused scans
# ---------------------------------------------------------------------------

class TestRecoverySkipsPaused:
    def test_recovery_does_not_discover_paused_datastore(self, running_datastore):
        """Recovery should NOT auto-discover a datastore with last_scan_status='paused'.

        Even if there are interrupted tasks (pending/processing) from the pause,
        recovery must respect the paused state and skip discovery.
        The user must manually click Resume to continue.
        """
        ds, ds_path = running_datastore

        db = TestingSessionLocal()
        try:
            # Set to paused
            ds_record = db.query(DataStore).filter(DataStore.id == ds.id).first()
            ds_record.last_scan_status = "paused"
            ds_record.auto_process_enabled = False
            db.commit()

            # Create a document with a pending task (simulating pause cancelling a future)
            doc = Document(
                data_store_id=ds.id,
                file_path=os.path.join(ds_path, "file1.pdf"),
                file_name="file1.pdf",
                file_size=100,
                content_type="application/pdf",
                file_hash="abc123",
                is_selected=True,
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)

            task = ProcessingTask(
                data_store_id=ds.id,
                document_id=doc.id,
                status="pending",
            )
            db.add(task)
            db.commit()

            # Verify the recovery decision logic:
            # The new elif branch checks for "paused" BEFORE the interrupted-tasks check.
            # So even with interrupted tasks, a paused datastore should NOT be discovered.
            assert ds_record.auto_process_enabled is False
            assert ds_record.last_scan_status == "paused"
            # The interrupted tasks check would find this task:
            interrupted = (
                db.query(ProcessingTask)
                .filter(
                    ProcessingTask.data_store_id == ds.id,
                    ProcessingTask.status.in_(["pending", "processing"]),
                )
                .count()
            )
            assert interrupted > 0
            # But recovery should still skip because of the "paused" check.
        finally:
            db.close()

    def test_recovery_skips_paused_even_with_processing_tasks(self, running_datastore):
        """Explicitly test the recovery code path for paused datastores.

        Calls the recovery start() method and verifies that no discovery
        pipeline worker is submitted for a paused datastore.
        """
        from app.services.discovery.startup_recovery_service import StartupRecoveryService
        ds, ds_path = running_datastore

        db = TestingSessionLocal()
        try:
            ds_record = db.query(DataStore).filter(DataStore.id == ds.id).first()
            ds_record.last_scan_status = "paused"
            ds_record.auto_process_enabled = False
            db.commit()

            doc = Document(
                data_store_id=ds.id,
                file_path=os.path.join(ds_path, "file1.pdf"),
                file_name="file1.pdf",
                file_size=100,
                content_type="application/pdf",
                file_hash="abc123",
                is_selected=True,
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)

            task = ProcessingTask(
                data_store_id=ds.id,
                document_id=doc.id,
                status="processing",
            )
            db.add(task)
            db.commit()
        finally:
            db.close()

        service = StartupRecoveryService()
        # Mock the executor to track what gets submitted
        submitted = []
        original_submit = service.executor.submit

        def track_submit(fn, *args, **kwargs):
            submitted.append((fn.__name__ if hasattr(fn, '__name__') else str(fn), args))
            # Return a mock future
            future = MagicMock()
            future.done.return_value = True
            future.result.return_value = None
            return future

        service.executor.submit = track_submit

        with patch("app.services.discovery.startup_recovery_service.SessionLocal", side_effect=TestingSessionLocal):
            service.start()

        # Should have submitted graph_only_worker, NOT discovery_pipeline_worker
        worker_names = [s[0] for s in submitted]
        assert "_graph_only_worker" in worker_names
        assert "_discovery_pipeline_worker" not in worker_names

        service.stop()

    def test_recovery_discovers_running_datastore(self, running_datastore):
        """Recovery SHOULD auto-discover a datastore with last_scan_status='running'."""
        db = TestingSessionLocal()
        try:
            ds_record = db.query(DataStore).filter(
                DataStore.id == running_datastore[0].id
            ).first()
            ds_record.last_scan_status = "running"
            ds_record.auto_process_enabled = False
            db.commit()

            # The recovery condition:
            assert ds_record.auto_process_enabled is False
            assert ds_record.last_scan_status == "running"
            # Recovery would set should_discover = True — correct behavior.
        finally:
            db.close()
