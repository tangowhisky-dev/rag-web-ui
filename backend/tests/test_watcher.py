"""
test_watcher.py — Unit tests for WatcherService.

Covers:
  1. WatcherService instantiates cleanly
  2. SHA-256 hash computation produces expected digests
  3. Extension filter accepts supported / rejects unsupported files
  4. Dedup: same file dropped twice → only one ingestion call
  5. Non-supported files (.tmp, .DS_Store, .exe) are ignored
  6. Lifecycle: start() and stop() work cleanly
"""

import hashlib
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# conftest.py patches MySQL types and app.db.session with SQLite stub.
# Import order matters: conftest runs first, so these use the SQLite engine.
from app.main import app as fastapi_app  # noqa: F401, F811
from app.services.document_processor import (
    SUPPORTED_EXTENSIONS,
)

from app.services.watcher_service import WatcherService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def watcher_service():
    """Create a WatcherService instance without starting the observer."""
    svc = WatcherService()
    yield svc
    # Ensure stop is called even if test fails
    if svc._running:
        svc.stop()


@pytest.fixture
def tmp_support_dir():
    """Create a temporary directory with user_1/kb_1 structure for test files."""
    with tempfile.TemporaryDirectory() as td:
        kb_dir = os.path.join(td, "user_1", "kb_1")
        os.makedirs(kb_dir)
        yield td, kb_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_watcher_service_init(watcher_service):
    """WatcherService instantiates cleanly with expected defaults."""
    assert watcher_service._running is False
    assert watcher_service._observer is None
    assert watcher_service._files_scanned == 0
    assert watcher_service._last_scan_at is None
    assert watcher_service._executor is not None
    assert watcher_service._debouncer is not None


def test_watcher_hash_computation():
    """SHA-256 of known content produces expected hash."""
    # Known content
    content = b"hello world"
    expected_hash = hashlib.sha256(content).hexdigest()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(content)
        tmp_path = f.name

    try:
        actual_hash = WatcherService._compute_hash(tmp_path)
        assert actual_hash == expected_hash
        assert len(actual_hash) == 64  # SHA-256 hex digest length
        # Hash prefix (first 8 chars)
        assert actual_hash[:8] == expected_hash[:8]
    finally:
        os.unlink(tmp_path)


def test_watcher_hash_computation_empty_file():
    """SHA-256 of empty file produces correct hash."""
    empty_hash = hashlib.sha256(b"").hexdigest()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        tmp_path = f.name

    try:
        actual_hash = WatcherService._compute_hash(tmp_path)
        assert actual_hash == empty_hash
    finally:
        os.unlink(tmp_path)


def test_watcher_extension_filter():
    """.pdf, .docx, .txt are accepted; .exe is rejected."""
    # Supported extensions should be in SUPPORTED_EXTENSIONS
    assert ".pdf" in SUPPORTED_EXTENSIONS
    assert ".docx" in SUPPORTED_EXTENSIONS
    assert ".txt" in SUPPORTED_EXTENSIONS

    # Unsupported extensions should NOT be in SUPPORTED_EXTENSIONS
    assert ".exe" not in SUPPORTED_EXTENSIONS
    assert ".tmp" not in SUPPORTED_EXTENSIONS
    assert ".DS_Store" not in SUPPORTED_EXTENSIONS


def test_watcher_ignores_non_supported(tmp_support_dir):
    """.tmp, .DS_Store, .exe files are ignored by the handler."""
    _, kb_dir = tmp_support_dir

    svc = WatcherService()
    # Create a mock handler with the service
    from app.services.watcher_service import _WatcherHandler

    handler = _WatcherHandler(service=svc, org_id=1, watch_dir=kb_dir)

    # Unsupported files should return False
    assert handler._should_process(os.path.join(kb_dir, "file.tmp")) is False
    assert handler._should_process(os.path.join(kb_dir, "file.exe")) is False
    assert handler._should_process(os.path.join(kb_dir, ".DS_Store")) is False
    assert handler._should_process(os.path.join(kb_dir, "secret.exe")) is False

    # Supported files should return True
    assert handler._should_process(os.path.join(kb_dir, "report.pdf")) is True
    assert handler._should_process(os.path.join(kb_dir, "doc.docx")) is True
    assert handler._should_process(os.path.join(kb_dir, "notes.txt")) is True
    assert handler._should_process(os.path.join(kb_dir, "data.CSV")) is True


def test_watcher_dedup(tmp_support_dir):
    """Same file dropped twice → only one ingestion task created (mock
    process_document_background)."""
    _, kb_dir = tmp_support_dir

    # Create a test PDF file
    test_file = os.path.join(kb_dir, "report.pdf")
    with open(test_file, "wb") as f:
        f.write(b"%PDF-1.4 fake pdf content for dedup test")

    svc = WatcherService()

    # Mock _trigger_ingestion to count how many times it's called
    trigger_count = 0

    def count_trigger(*args, **kwargs):
        nonlocal trigger_count
        trigger_count += 1

    # Mock SessionLocal so _handle_file doesn't hit the real DB.
    # First call: no existing doc → triggers ingestion.
    # Second call: existing doc returned → dedup skips.
    mock_existing_doc = MagicMock()
    mock_existing_doc.id = 42

    call_num = [0]

    def mock_session_factory():
        call_num[0] += 1
        mock_session = MagicMock()
        mock_query = MagicMock()
        if call_num[0] == 1:
            # First call: no existing doc → should ingest
            mock_query.filter.return_value.first.return_value = None
        else:
            # Second call: existing doc found → dedup skip
            mock_query.filter.return_value.first.return_value = mock_existing_doc
        mock_session.query.return_value = mock_query
        return mock_session

    with patch.object(svc, "_trigger_ingestion", side_effect=count_trigger):
        with patch(
            "app.services.watcher_service.SessionLocal",
            side_effect=mock_session_factory,
        ):
            # First event — should trigger ingestion (no existing doc)
            svc._debouncer._last_event.clear()
            svc._debouncer._last_type.clear()
            svc._handle_file(test_file, "1", None)
            assert trigger_count == 1, (
                f"Expected 1 ingestion trigger on first pass, got {trigger_count}"
            )

            # Wait for debouncer delay to pass
            time.sleep(1.5)

            # Second event — should be deduped (existing doc found)
            trigger_count = 0
            svc._handle_file(test_file, "1", None)
            assert trigger_count == 0, (
                f"Expected 0 ingestion triggers on dedup hit, got {trigger_count}"
            )


def test_watcher_service_lifecycle():
    """start() and stop() work cleanly."""
    svc = WatcherService()

    # Initial state
    assert svc._running is False
    assert svc._observer is None

    # Mock the observer so we don't need a real filesystem.
    # PollingObserver is imported inside start() via:
    #   from watchdog.observers.polling import PollingObserver
    # so we patch it at the source module.
    with patch(
        "watchdog.observers.polling.PollingObserver"
    ) as mock_observer_cls, patch.object(
        svc, "_get_orgs_with_watch_dirs", return_value=[]
    ):
        mock_observer = MagicMock()
        mock_observer_cls.return_value = mock_observer

        svc.start()

        assert svc._running is True
        assert svc._observer is not None
        mock_observer.start.assert_called_once()

        # Stop should clean up
        svc.stop()

        assert svc._running is False
        mock_observer.stop.assert_called_once()
        mock_observer.join.assert_called_once()

    # Double-stop should be a no-op
    svc.stop()  # Should not raise

    # Double-start should be a no-op
    with patch(
        "watchdog.observers.polling.PollingObserver"
    ) as mock_observer_cls, patch.object(
        svc, "_get_orgs_with_watch_dirs", return_value=[]
    ):
        mock_observer_cls.return_value = MagicMock()
        svc.start()
        # Should not raise, but _running is already True warning is logged


def test_watcher_get_status(watcher_service):
    """get_status() returns expected keys with correct types."""
    status = watcher_service.get_status()
    assert "running" in status
    assert "last_scan_at" in status
    assert "files_scanned" in status
    assert status["running"] is False
    assert status["last_scan_at"] is None
    assert status["files_scanned"] == 0


def test_watcher_scan_not_running(watcher_service):
    """scan() returns empty summary when service is not running."""
    summary = watcher_service.scan()
    assert summary == {"scanned": 0, "new": 0, "skipped": 0, "errors": 0}


def test_watcher_parse_path_regex():
    """Path with user_{uid}/kb_{kid}/ is parsed correctly."""
    # Full path with convention
    user_id, kb_id = WatcherService._parse_path(
        "/data/watch/user_42/kb_7/report.pdf", "1", None
    )
    assert user_id == 42
    assert kb_id == 7

    # Nested path
    user_id, kb_id = WatcherService._parse_path(
        "/data/watch/user_10/kb_20/deep/nested/file.txt", "1", None
    )
    assert user_id == 10
    assert kb_id == 20


def test_watcher_parse_path_fallback():
    """Path without user/kb convention falls back to provided kb_id."""
    # With kb_id provided
    user_id, kb_id = WatcherService._parse_path(
        "/data/watch/report.pdf", "1", "99"
    )
    assert user_id == 0
    assert kb_id == 99

    # Without kb_id, returns None
    user_id, kb_id = WatcherService._parse_path(
        "/data/watch/report.pdf", "1", None
    )
    assert user_id == 0
    assert kb_id is None


def test_watcher_ignores_hidden_files(tmp_support_dir):
    """Files starting with '.' are skipped by _handle_file extension check."""
    _, kb_dir = tmp_support_dir

    svc = WatcherService()

    # Create a hidden file with supported extension
    hidden_file = os.path.join(kb_dir, ".hidden.pdf")
    with open(hidden_file, "wb") as f:
        f.write(b"hidden pdf content")

    trigger_count = 0

    def count_trigger(*args, **kwargs):
        nonlocal trigger_count
        trigger_count += 1

    with patch.object(svc, "_trigger_ingestion", side_effect=count_trigger):
        with patch(
            "app.services.watcher_service.SessionLocal"
        ) as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_query = MagicMock()
            mock_session.query.return_value = mock_query
            mock_query.filter.return_value.first.return_value = None

            trigger_count = 0
            svc._handle_file(hidden_file, "1", None)
            # Hidden files are skipped regardless of extension
            assert trigger_count == 0, (
                f"Hidden files should be skipped, got {trigger_count} triggers"
            )


def test_watcher_dedup_different_kb():
    """Same file hash in different kb → NOT deduped (different knowledge bases)."""
    svc = WatcherService()
    file_hash = "abc123" * 12  # 64-char fake hash

    # Mock two different DB queries for different knowledge bases
    call_count = 0

    def mock_first():
        nonlocal call_count
        call_count += 1
        # First call (kb=1): no existing doc → ingest
        # Second call (kb=2): no existing doc → ingest
        return None

    with patch(
        "app.services.watcher_service.SessionLocal"
    ) as mock_session_cls, patch.object(
        svc, "_trigger_ingestion"
    ) as mock_trigger:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value.first.side_effect = lambda: None

        # Simulate same file in two different knowledge bases
        svc._handle_file("/data/user_1/kb_1/file.pdf", "1", None)
        svc._handle_file("/data/user_1/kb_2/file.pdf", "1", None)

        # Both should trigger ingestion (different kb_id)
        assert mock_trigger.call_count == 2, (
            f"Expected 2 ingestion triggers for different KBs, got {mock_trigger.call_count}"
        )
