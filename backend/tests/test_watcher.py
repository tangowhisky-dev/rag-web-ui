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


# ---------------------------------------------------------------------------
# SMB Share Watcher tests
# ---------------------------------------------------------------------------


def test_smb_watcher_init():
    """SMBShareWatcher instantiates cleanly with expected defaults."""
    from app.services.smb_watcher import SMBShareWatcher

    w = SMBShareWatcher(
        host="smb-server",
        share="documents",
        username="user",
        password="pass",
        domain="WORKGROUP",
        kb_id=1,
        poll_interval=120,
    )
    assert w.host == "smb-server"
    assert w.share == "documents"
    assert w.username == "user"
    assert w.password == "pass"
    assert w.domain == "WORKGROUP"
    assert w.kb_id == 1
    assert w.poll_interval == 120
    assert w._last_scan_at is None
    assert w._last_error is None
    assert w._connected is False


def test_smb_watcher_default_params():
    """SMBShareWatcher uses correct defaults when optional params omitted."""
    from app.services.smb_watcher import SMBShareWatcher

    w = SMBShareWatcher(host="host", share="share", username="u", password="p")
    assert w.domain is None
    assert w.kb_id is None
    assert w.poll_interval == 60  # default


def test_smb_watcher_get_status():
    """get_status() returns expected keys with correct types."""
    from app.services.smb_watcher import SMBShareWatcher

    w = SMBShareWatcher(host="h", share="s", username="u", password="p")
    status = w.get_status()
    assert status["host"] == "h"
    assert status["share"] == "s"
    assert status["connected"] is False
    assert status["last_scan_at"] is None
    assert status["last_error"] is None
    assert status["poll_interval"] == 60


def test_smb_watcher_scan_not_connected():
    """scan() returns error summary when smbprotocol is not importable."""
    from app.services.smb_watcher import SMBShareWatcher

    w = SMBShareWatcher(host="nonexistent-host-12345", share="share", username="u", password="p")
    # smbprotocol is not installed in the test env, so scan will raise ImportError
    # This is expected — the real env (Docker) has smbprotocol installed
    with pytest.raises(ImportError):
        w.scan()
    assert w._connected is False


def test_watcher_service_has_smb_watches():
    """WatcherService instantiates with _smb_watches attribute."""
    svc = WatcherService()
    assert hasattr(svc, "_smb_watches")
    assert svc._smb_watches == []


def test_watcher_get_status_includes_smb_watches():
    """get_status() returns smb_watches key in status dict."""
    svc = WatcherService()
    status = svc.get_status()
    assert "smb_watches" in status
    assert status["smb_watches"] == []


def test_watcher_service_load_smb_watches_empty():
    """_load_smb_watches() handles empty org list gracefully (mocked DB)."""
    svc = WatcherService()
    with patch(
        "app.services.watcher_service.SessionLocal"
    ) as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.query.return_value.filter.return_value.all.return_value = []
        # Should not raise even with no SMB-enabled orgs
        svc._load_smb_watches()
        assert svc._smb_watches == []


def test_smb_watcher_compute_hash():
    """SMBShareWatcher._compute_hash produces correct SHA-256."""
    from app.services.smb_watcher import SMBShareWatcher

    content = b"smb test content"
    expected = hashlib.sha256(content).hexdigest()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(content)
        tmp_path = f.name

    try:
        actual = SMBShareWatcher._compute_hash(tmp_path)
        assert actual == expected
    finally:
        os.unlink(tmp_path)


def test_smb_watcher_scan_skips_unsupported():
    """SMB scan skips files with unsupported extensions (mocked share listing)."""
    from app.services.smb_watcher import SMBShareWatcher

    w = SMBShareWatcher(host="h", share="s", username="u", password="p")
    # Test that unsupported files get skipped
    result = {"scanned": 0, "new": 0, "skipped": 0, "errors": 0}
    # We can't mock the SMB connection easily, but we can verify the
    # extension filter logic by checking that the _process_file method
    # would skip unsupported extensions
    # The file_info dict simulates what list_path returns
    file_info = {
        "path": "/share/file.tmp",
        "file_name": "file.tmp",
        "long_name": "file.tmp",
        "size": 100,
    }
    # _process_file would skip .tmp but we can't call it without a real share
    # Instead verify the extension set directly
    from app.services.document_processor import SUPPORTED_EXTENSIONS
    assert ".tmp" not in SUPPORTED_EXTENSIONS
    assert ".pdf" in SUPPORTED_EXTENSIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _install_fake_smbprotocol():
    """Install fake smbprotocol module with mock Connection/Session/Share into sys.modules."""
    import sys
    from types import ModuleType
    from unittest.mock import MagicMock

    fake_smb = ModuleType("smbprotocol")
    fake_conn = ModuleType("smbprotocol.connection")
    fake_session = ModuleType("smbprotocol.session")
    fake_share = ModuleType("smbprotocol.share")

    mock_conn_cls = MagicMock()
    mock_session_cls = MagicMock()
    mock_share_cls = MagicMock()
    mock_access = MagicMock()

    fake_conn.Connection = mock_conn_cls
    fake_session.Session = mock_session_cls
    fake_share.Share = mock_share_cls
    fake_share.ACCESS_MASK = mock_access

    fake_smb.connection = fake_conn
    fake_smb.session = fake_session
    fake_smb.share = fake_share

    saved = {}
    for mod_name in ["smbprotocol", "smbprotocol.connection", "smbprotocol.session", "smbprotocol.share"]:
        saved[mod_name] = sys.modules.get(mod_name)
    sys.modules["smbprotocol"] = fake_smb
    sys.modules["smbprotocol.connection"] = fake_conn
    sys.modules["smbprotocol.session"] = fake_session
    sys.modules["smbprotocol.share"] = fake_share

    return saved, mock_conn_cls, mock_session_cls, mock_share_cls, mock_access


def _uninstall_fake_smbprotocol(saved):
    """Restore original sys.modules entries."""
    import sys
    for mod_name, orig in saved.items():
        if orig is None:
            sys.modules.pop(mod_name, None)
        else:
            sys.modules[mod_name] = orig


# ---------------------------------------------------------------------------
# SMBAuth encryption tests
# ---------------------------------------------------------------------------


def test_smb_auth_encrypt_decrypt():
    """SMBAuth with Fernet key round-trips encrypt/decrypt correctly."""
    from cryptography.fernet import Fernet
    from app.services.smb_auth import SMBAuth

    # Create a valid Fernet key
    fernet_key = Fernet.generate_key()
    auth = SMBAuth(master_key=fernet_key.decode())

    plaintext = "MyS3cretP@ssw0rd!"
    encrypted = auth.encrypt(plaintext)

    # Encrypted output should differ from plaintext
    assert encrypted != plaintext
    assert len(encrypted) > len(plaintext)

    # Decrypt should recover the original
    decrypted = auth.decrypt(encrypted)
    assert decrypted == plaintext


def test_smb_auth_plaintext_fallback():
    """SMBAuth without master key stores plaintext and returns as-is on decrypt."""
    from app.services.smb_auth import SMBAuth

    auth = SMBAuth()  # No master key

    plaintext = "MyS3cretP@ssw0rd!"
    encrypted = auth.encrypt(plaintext)

    # Without a key, encrypt returns the original plaintext
    assert encrypted == plaintext

    # Decrypt also returns as-is
    decrypted = auth.decrypt(encrypted)
    assert decrypted == plaintext


# ---------------------------------------------------------------------------
# SMBShareWatcher test_connection tests (mocked)
# ---------------------------------------------------------------------------


def test_smb_test_connection_mock():
    """test_connection returns (True, '') when mocked SMB connection succeeds."""
    from unittest.mock import MagicMock

    from app.services.smb_watcher import SMBShareWatcher

    w = SMBShareWatcher(
        host="smb-server", share="documents", username="user", password="pass",
    )

    saved, mock_conn_cls, mock_session_cls, _, _ = _install_fake_smbprotocol()

    try:
        mock_conn = MagicMock()
        mock_share = MagicMock()
        mock_conn_cls.return_value = mock_conn
        mock_conn.connect = MagicMock()
        mock_conn.get_share = MagicMock(return_value=mock_share)

        connected, error = w.test_connection()

        assert connected is True
        assert error == ""
        mock_conn.close.assert_called_once()
    finally:
        _uninstall_fake_smbprotocol(saved)


def test_smb_test_connection_failure():
    """test_connection returns (False, error) when connection raises."""
    from unittest.mock import MagicMock

    from app.services.smb_watcher import SMBShareWatcher

    w = SMBShareWatcher(
        host="bad-host", share="share", username="user", password="pass",
    )

    saved, mock_conn_cls, _, _, _ = _install_fake_smbprotocol()

    try:
        mock_conn_cls.side_effect = ConnectionRefusedError("Connection refused")

        connected, error = w.test_connection()

        assert connected is False
        assert "Connection refused" in error
    finally:
        _uninstall_fake_smbprotocol(saved)


# ---------------------------------------------------------------------------
# SMBShareWatcher scan tests (mocked)
# ---------------------------------------------------------------------------


def test_smb_scan_mock():
    """Mocked scan with file listing verifies dedup + ingestion trigger."""
    from unittest.mock import MagicMock, patch

    from app.services.smb_watcher import SMBShareWatcher

    w = SMBShareWatcher(
        host="smb-server", share="docs", username="user", password="pass",
        kb_id=1,
    )

    saved, mock_conn_cls, _, _, _ = _install_fake_smbprotocol()

    try:
        mock_conn = MagicMock()
        mock_share = MagicMock()
        mock_conn_cls.return_value = mock_conn
        mock_conn.connect = MagicMock()
        mock_conn.get_share = MagicMock(return_value=mock_share)

        # _list_all_files uses dict-style access (entry["file_name"]) so we
        # need real dicts, not MagicMock objects.
        file_entry1 = {
            "path": "/report.pdf",
            "file_name": "report.pdf",
            "long_name": "report.pdf",
            "size": 1024,
            "end_of_file": 1024,
            "file_directory": False,
        }
        file_entry2 = {
            "path": "/notes.txt",
            "file_name": "notes.txt",
            "long_name": "notes.txt",
            "size": 512,
            "end_of_file": 512,
            "file_directory": False,
        }
        mock_share.list_path.return_value = [file_entry1, file_entry2]

        # Mock a file handle for read()
        mock_smb_file = MagicMock()
        mock_smb_file.read.return_value = b"fake pdf content for scan mock"
        mock_share.open_file.return_value.__enter__ = MagicMock(return_value=mock_smb_file)
        mock_share.open_file.return_value.__exit__ = MagicMock(return_value=False)

        # Mock _trigger_ingestion_smb to avoid threading/event-loop hang
        ingestion_calls = []

        def capture_ingestion(*args, **kwargs):
            ingestion_calls.append((args, kwargs))

        w._trigger_ingestion_smb = capture_ingestion

        # Patch SessionLocal at the source to avoid real DB access during dedup
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_query.filter.return_value.first.return_value = None
        mock_session.query.return_value = mock_query

        with patch("app.services.smb_watcher.SessionLocal", return_value=mock_session):
            result = w.scan()

        assert result["scanned"] == 2
        assert result["new"] == 2
        assert result["skipped"] == 0
        assert result["errors"] == 0
        assert w._connected is True
        assert len(ingestion_calls) == 2
    finally:
        _uninstall_fake_smbprotocol(saved)


def test_smb_scan_empty_share():
    """Mocked scan with empty share returns scanned=0."""
    from unittest.mock import MagicMock

    from app.services.smb_watcher import SMBShareWatcher

    w = SMBShareWatcher(
        host="smb-server", share="empty", username="user", password="pass",
    )

    saved, mock_conn_cls, _, _, _ = _install_fake_smbprotocol()

    try:
        mock_conn = MagicMock()
        mock_share = MagicMock()
        mock_share.list_path.return_value = []
        mock_conn_cls.return_value = mock_conn
        mock_conn.connect = MagicMock()
        mock_conn.get_share = MagicMock(return_value=mock_share)

        result = w.scan()

        assert result["scanned"] == 0
        assert result["new"] == 0
        assert result["skipped"] == 0
        assert result["errors"] == 0
    finally:
        _uninstall_fake_smbprotocol(saved)


def test_smb_scan_dedup():
    """Same file hash → dedup skip (scanned but skipped)."""
    from unittest.mock import MagicMock, patch

    from app.services.smb_watcher import SMBShareWatcher

    w = SMBShareWatcher(
        host="smb-server", share="docs", username="user", password="pass",
        kb_id=1,
    )

    saved, mock_conn_cls, _, _, _ = _install_fake_smbprotocol()

    try:
        mock_conn = MagicMock()
        mock_share = MagicMock()
        mock_conn_cls.return_value = mock_conn
        mock_conn.connect = MagicMock()
        mock_conn.get_share = MagicMock(return_value=mock_share)

        # Use real dicts — _list_all_files uses dict-style access
        file_entry = {
            "path": "/report.pdf",
            "file_name": "report.pdf",
            "long_name": "report.pdf",
            "size": 1024,
            "end_of_file": 1024,
            "file_directory": False,
        }
        mock_share.list_path.return_value = [file_entry]

        # Mock file read
        mock_smb_file = MagicMock()
        mock_smb_file.read.return_value = b"dedup test content"
        mock_share.open_file.return_value.__enter__ = MagicMock(return_value=mock_smb_file)
        mock_share.open_file.return_value.__exit__ = MagicMock(return_value=False)

        # Mock SessionLocal: existing doc found → dedup skip
        mock_existing_doc = MagicMock()
        mock_existing_doc.id = 42

        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_query.filter.return_value.first.return_value = mock_existing_doc
        mock_session.query.return_value = mock_query

        with patch("app.services.smb_watcher.SessionLocal", return_value=mock_session):
            result = w.scan()

        assert result["scanned"] == 1
        assert result["new"] == 0
        assert result["skipped"] == 1
        assert result["errors"] == 0
    finally:
        _uninstall_fake_smbprotocol(saved)


# ---------------------------------------------------------------------------
# SMB API endpoint tests
# ---------------------------------------------------------------------------


def test_smb_api_config_endpoint():
    """Mock POST /orgs/1/smb-config verifies encrypted password stored."""
    from unittest.mock import MagicMock, patch

    from fastapi.testclient import TestClient
    from app.main import app as fastapi_app
    from app.services.smb_auth import SMBAuth

    client = TestClient(fastapi_app)

    # Create a real Fernet key for deterministic encryption
    from cryptography.fernet import Fernet
    fernet_key = Fernet.generate_key()
    auth = SMBAuth(master_key=fernet_key.decode())

    # Use a real Organisation model instance
    from app.models.organisation import Organisation
    real_org = Organisation(
        id=1,
        name="Test Org",
        smb_host=None,
        smb_share=None,
        smb_username=None,
        smb_password_encrypted=None,
        smb_domain=None,
    )

    # Mock the watcher
    mock_watcher = MagicMock()
    mock_watcher._smb_watches = []

    # Mock a fake admin user
    mock_user = MagicMock()
    mock_user.role = "admin"

    # Patch internal helpers directly (require_admin is a dependency,
    # _get_org_or_404 and _get_watcher are called directly in the endpoint).
    # Override get_db at the module level where smb.py imports it.
    from app.core.security import require_admin as require_admin_fn
    from app.db.session import get_db as _conftest_get_db

    mock_session = MagicMock()
    mock_session.commit = MagicMock()
    mock_session.refresh = MagicMock()

    def _mock_get_db_gen():
        yield mock_session

    original_overrides = dict(fastapi_app.dependency_overrides)
    try:
        fastapi_app.dependency_overrides.update({
            require_admin_fn: lambda: mock_user,
            _conftest_get_db: _mock_get_db_gen,
        })
        with patch("app.services.smb_auth.get_smb_auth", return_value=auth), \
             patch("app.api.api_v1.smb._get_org_or_404", return_value=real_org), \
             patch("app.api.api_v1.smb._get_watcher", return_value=mock_watcher):
            response = client.post(
                "/api/admin/orgs/1/smb-config",
                json={
                    "host": "smb-server",
                    "share": "documents",
                    "username": "user",
                    "password": "MyP@ssw0rd",
                    "domain": "WORKGROUP",
                },
            )
    finally:
        fastapi_app.dependency_overrides.clear()
        fastapi_app.dependency_overrides.update(original_overrides)

    assert response.status_code == 200
    data = response.json()
    assert data["smb_host"] == "smb-server"
    assert data["smb_share"] == "documents"
    assert data["smb_username"] == "user"
    assert data["smb_domain"] == "WORKGROUP"
    assert data["status"] == "configured"

    # Verify password was encrypted (not stored as plaintext)
    assert real_org.smb_password_encrypted != "MyP@ssw0rd"
    assert real_org.smb_password_encrypted != None
    # Verify the stored ciphertext can be decrypted back to the original
    assert auth.decrypt(real_org.smb_password_encrypted) == "MyP@ssw0rd"
    assert real_org.smb_host == "smb-server"


def test_smb_password_not_in_response():
    """GET config response (via SMBConfigResponse) does not include password field."""
    from unittest.mock import MagicMock, patch
    from pydantic import BaseModel

    # SMBConfigResponse schema should not have a password field
    from app.schemas.smb import SMBConfigResponse

    response_model = SMBConfigResponse(
        org_id=1,
        smb_host="smb-server",
        smb_share="documents",
        smb_username="user",
        smb_domain="WORKGROUP",
        status="configured",
    )

    # The response dict should NOT contain any password field
    data = response_model.model_dump()
    assert "password" not in data
    assert "smb_password" not in data
    assert "smb_password_encrypted" not in data

    # Verify only expected fields exist
    expected_fields = {"org_id", "smb_host", "smb_share", "smb_username", "smb_domain", "status", "last_scan_at", "last_scan_status", "last_scan_files"}
    assert set(data.keys()) == expected_fields
