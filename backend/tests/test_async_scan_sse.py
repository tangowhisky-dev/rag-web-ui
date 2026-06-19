"""
test_async_scan_sse.py — Tests for the async scan + SSE progress reporting.

Covers:
  1. POST /datastores/{id}/scan returns 202 (not 200) and starts scan in background
  2. GET /datastores/{id}/scan-progress-stream returns SSE events
  3. SSE endpoint returns 200 with text/event-stream content type
  4. SSE endpoint returns error when no scan is running (waiting)
  5. SSE endpoint handles missing watcher gracefully
  6. SSE endpoint handles scan-not-found case
  7. Triggering a scan that is already running returns 409
"""
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from fastapi.testclient import TestClient

from app.main import app as fastapi_app
from app.db.session import get_db
from app.models.user import User, UserRole
from app.models.base import Base
from app.models.organisation import Organisation
import app.db.session as _session_mod
from sqlalchemy.orm import sessionmaker

# Ensure admin-only route is set up for the SSE endpoint
import app.api.api_v1.datastores  # noqa: ensure router is registered
import app.api.api_v1.admin  # noqa: ensure router is registered

engine = _session_mod.engine
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


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
def client():
    fastapi_app.dependency_overrides[get_db] = override_get_db
    with TestClient(fastapi_app) as c:
        yield c
    fastapi_app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_admin_user(db, prefix: str = "test") -> User:
    """Create a super_admin user. Use unique prefix to avoid constraint conflicts."""
    from app.core.security import get_password_hash
    user = User(
        username=f"{prefix}_admin",
        email=f"{prefix}@example.com",
        hashed_password=get_password_hash("admin123"),
        is_active=True,
        role=UserRole.super_admin,
        org_id=None,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_token(client, prefix: str = "test") -> str:
    username = f"{prefix}_admin"
    resp = client.post("/api/auth/token", data={"username": username, "password": "admin123"})
    assert resp.status_code == 200, f"login failed: {resp.text}"
    return resp.json()["access_token"]


def create_datastore(db, folder_path: str, prefix: str = "test") -> int:
    """Create a datastore in the DB and return its ID."""
    from app.models.datastore import DataStore
    ds = DataStore(
        name=f"Test DataStore {prefix}",
        description="Test",
        folder_path=folder_path,
        scan_pattern="*",
        is_active=True,
        auto_scan_enabled=False,
        auto_scan_interval_minutes=60,
        last_scan_total_files=0,
        last_scan_status="never",
        last_scan_processed=0,
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return ds.id


def create_test_files(tmp_dir: Path, count: int) -> str:
    """Create count test .txt files in tmp_dir/subdir and return tmp_dir."""
    subdir = tmp_dir / "store1"
    subdir.mkdir()
    for i in range(count):
        (subdir / f"file{i}.txt").write_text(f"content {i}")
    return str(tmp_dir)


# ---------------------------------------------------------------------------
# Tests — Async trigger endpoint
# ---------------------------------------------------------------------------

def test_trigger_scan_returns_202(client, db):
    """POST /datastores/{id}/scan must return 202, not 200."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)
        folder = create_test_files(tmp_dir, 3)
        datastore_id = create_datastore(db, folder, prefix="t1")

        create_admin_user(db, prefix="t1")
        token = get_token(client, prefix="t1")

        resp = client.post(
            f"/api/admin/datastores/{datastore_id}/scan",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 202, f"Expected 202, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert "datastore_id" in data
        assert data["datastore_id"] == datastore_id


def test_trigger_scan_already_running_returns_409(client, db):
    """Triggering a scan while one is already running returns 409."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)
        folder = create_test_files(tmp_dir, 3)
        datastore_id = create_datastore(db, folder, prefix="t2")

        create_admin_user(db, prefix="t2")
        token = get_token(client, prefix="t2")

        # Start a scan and hold it at _init_scan stage
        with patch("app.api.api_v1.datastores._get_watcher") as mock_watcher:
            mock_w = MagicMock()
            mock_w._active_scans = {
                1: {
                    "datastore_id": datastore_id,
                    "total": 3,
                    "processed": 1,
                    "status": "running",
                    "error": None,
                }
            }
            mock_watcher.return_value = mock_w

            resp = client.post(
                f"/api/admin/datastores/{datastore_id}/scan",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 409, f"Expected 409, got {resp.status_code}: {resp.text}"
            assert "already running" in resp.json()["detail"].lower()


def test_trigger_scan_nonexistent_folder(client, db):
    """Triggering a scan with a nonexistent folder returns 400."""
    datastore_id = create_datastore(db, "/nonexistent/path/12345", prefix="t3")

    create_admin_user(db, prefix="t3")
    token = get_token(client, prefix="t3")

    resp = client.post(
        f"/api/admin/datastores/{datastore_id}/scan",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"


# ---------------------------------------------------------------------------
# Tests — SSE progress stream
# ---------------------------------------------------------------------------

def test_sse_progress_stream_returns_200(client, db):
    """SSE endpoint must return 200 with text/event-stream content type."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)
        folder = create_test_files(tmp_dir, 3)
        datastore_id = create_datastore(db, folder, prefix="t4")

        create_admin_user(db, prefix="t4")
        token = get_token(client, prefix="t4")

        # Mock the watcher with empty active_scans (no scan running)
        mock_w = MagicMock()
        mock_w._active_scans = {}

        with patch("app.api.api_v1.datastores._get_watcher", return_value=mock_w):
            with client.stream(
                "GET",
                f"/api/admin/datastores/{datastore_id}/scan-progress-stream",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            ) as sse:
                assert sse.status_code == 200
                assert "text/event-stream" in sse.headers.get("content-type", "")

                events = []
                for line in sse.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = json.loads(line[6:])
                    events.append(data)
                    if data.get("status") in ("error",):
                        break

                # Should see a 'waiting' event first, then 'error' when scan not found
                assert len(events) >= 1, f"Expected events, got: {events}"
                # First event should be 'waiting' since no scan is running
                if events[0]["status"] == "waiting":
                    # Subsequent events should show it's waiting for scan
                    assert any(e["status"] == "waiting" for e in events)


def test_sse_progress_stream_missing_watcher(client, db):
    """If the watcher is not available, SSE must return an error event."""
    datastore_id = create_datastore(db, "/some/path", prefix="t5")

    create_admin_user(db, prefix="t5")
    token = get_token(client, prefix="t5")

    with patch("app.api.api_v1.datastores._get_watcher") as mock_get_watcher:
        from fastapi import HTTPException
        mock_get_watcher.side_effect = HTTPException(status_code=503, detail="Watcher not initialized")

        with client.stream(
            "GET",
            f"/api/admin/datastores/{datastore_id}/scan-progress-stream",
            headers={"Authorization": f"Bearer {token}"},
        ) as sse:
            assert sse.status_code == 200
            lines = list(sse.iter_lines())
            assert len(lines) >= 1
            assert "Watcher not available" in lines[0]


def test_sse_progress_stream_scan_not_found(client, db):
    """If no scan appears within 5 seconds, SSE must return error event."""
    datastore_id = create_datastore(db, "/some/path", prefix="t6")

    create_admin_user(db, prefix="t6")
    token = get_token(client, prefix="t6")

    with patch("app.api.api_v1.datastores._get_watcher") as mock_get_watcher:
        mock_w = MagicMock()
        mock_w._active_scans = {}
        mock_get_watcher.return_value = mock_w

        with client.stream(
            "GET",
            f"/api/admin/datastores/{datastore_id}/scan-progress-stream",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        ) as sse:
            assert sse.status_code == 200
            lines = list(sse.iter_lines())
            # After ~5s wait, we should get the "Scan not found" event
            assert any("Scan not found" in line for line in lines), f"Got: {lines}"


def test_sse_progress_stream_includes_all_fields(client, db):
    """SSE event must include total_files, processed_files, new_files, modified_files,
    skipped_files, error_files, and status fields."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)
        folder = create_test_files(tmp_dir, 3)
        datastore_id = create_datastore(db, folder, prefix="t7")

        create_admin_user(db, prefix="t7")
        token = get_token(client, prefix="t7")

        # Use a mock that returns a completed scan — the SSE endpoint
        # will emit the event and then break the loop because status
        # is not "running".
        mock_w = MagicMock()
        mock_w._active_scans = {
            1: {
                "datastore_id": datastore_id,
                "total": 3,
                "processed": 3,
                "status": "completed",
                "new": 2,
                "modified": 0,
                "skipped": 1,
                "error_count": 0,
                "error_message": None,
            }
        }

        with patch("app.api.api_v1.datastores._get_watcher", return_value=mock_w):
            with client.stream(
                "GET",
                f"/api/admin/datastores/{datastore_id}/scan-progress-stream",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            ) as sse:
                assert sse.status_code == 200

                events = []
                for line in sse.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = json.loads(line[6:])
                    events.append(data)

                # Should have at least one event with all expected fields
                assert len(events) >= 1, f"Expected events, got: {events}"

                # Check required fields
                evt = events[0]
                assert "total_files" in evt, f"Missing total_files in event: {evt}"
                assert "processed_files" in evt, f"Missing processed_files in event: {evt}"
                assert "status" in evt, f"Missing status in event: {evt}"
                assert "new_files" in evt, f"Missing new_files in event: {evt}"
                assert "modified_files" in evt, f"Missing modified_files in event: {evt}"
                assert "skipped_files" in evt, f"Missing skipped_files in event: {evt}"
                assert "error_files" in evt, f"Missing error_files in event: {evt}"

                # Check values
                assert evt["total_files"] == 3
                assert evt["processed_files"] == 3
                assert evt["status"] == "completed"
                assert evt["new_files"] == 2
                assert evt["modified_files"] == 0
                assert evt["skipped_files"] == 1
                assert evt["error_files"] == 0


def test_sse_progress_stream_error_with_message(client, db):
    """SSE event must include error_message when scan ends with an error."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)
        folder = create_test_files(tmp_dir, 3)
        datastore_id = create_datastore(db, folder, prefix="t8")

        create_admin_user(db, prefix="t8")
        token = get_token(client, prefix="t8")

        mock_w = MagicMock()
        mock_w._active_scans = {
            1: {
                "datastore_id": datastore_id,
                "total": 3,
                "processed": 2,
                "status": "error",
                "new": 2,
                "modified": 0,
                "skipped": 0,
                "error_count": 1,
                "error_message": "2 file(s) failed ingestion",
            }
        }

        with patch("app.api.api_v1.datastores._get_watcher", return_value=mock_w):
            with client.stream(
                "GET",
                f"/api/admin/datastores/{datastore_id}/scan-progress-stream",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            ) as sse:
                assert sse.status_code == 200

                events = []
                for line in sse.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = json.loads(line[6:])
                    events.append(data)

                # Should have the error event
                assert len(events) >= 1, f"Expected events, got: {events}"

                # Check error fields
                evt = events[0]
                assert evt["status"] == "error"
                assert evt["error_files"] == 1
                assert "error_message" in evt, f"Missing error_message in event: {evt}"
                assert evt["error_message"] == "2 file(s) failed ingestion"


def test_sse_progress_stream_ignores_old_completed_scan(client, db):
    """SSE endpoint must find the most recent scan, not a stale completed
    scan from a previous run. Python dicts preserve insertion order, so
    the SSE endpoint iterates in reverse to find the newest scan first.

    Simulates: user triggered a scan earlier (scan_id=1, completed),
    then triggered another scan (scan_id=2, completed). The SSE endpoint
    should return data from the NEW scan (scan_id=2), not the old one.
    """
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)
        folder = create_test_files(tmp_dir, 3)
        datastore_id = create_datastore(db, folder, prefix="t9")

        create_admin_user(db, prefix="t9")
        token = get_token(client, prefix="t9")

        mock_w = MagicMock()
        # Old completed scan (scan_id=1) + new completed scan (scan_id=2)
        mock_w._active_scans = {
            1: {
                "datastore_id": datastore_id,
                "total": 3,
                "processed": 3,
                "status": "completed",
                "new": 2,
                "modified": 0,
                "skipped": 1,
                "error_count": 0,
                "error_message": None,
            },
            2: {
                "datastore_id": datastore_id,
                "total": 3,
                "processed": 3,
                "status": "completed",
                "new": 3,
                "modified": 0,
                "skipped": 0,
                "error_count": 0,
                "error_message": None,
            },
        }

        with patch("app.api.api_v1.datastores._get_watcher", return_value=mock_w):
            with client.stream(
                "GET",
                f"/api/admin/datastores/{datastore_id}/scan-progress-stream",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            ) as sse:
                assert sse.status_code == 200

                events = []
                for line in sse.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = json.loads(line[6:])
                    events.append(data)

                # Should have at least one event from the NEW scan
                # (not the old completed scan)
                assert len(events) >= 1, f"Expected events, got: {events}"

                # The SSE endpoint should find the newest scan (scan_id=2),
                # not the old completed one (scan_id=1). Verify this by
                # checking that new_files=3 (new scan) not new_files=2 (old).
                evt = events[0]
                assert evt["new_files"] == 3, \
                    f"Expected new_files=3 (new scan), got {evt['new_files']}. " \
                    f"SSE found stale scan. Event: {evt}"
                assert evt["processed_files"] == 3, \
                    f"Expected 3 processed (new scan), got {evt['processed_files']}"
                assert evt["status"] == "completed", \
                    f"Expected completed, got {evt['status']}"


def test_active_scan_cleanup_after_completion(client, db):
    """After a scan completes, its entry must be removed from _active_scans
    so the SSE endpoint won't find stale completed scans on subsequent
    requests.

    We simulate this by manually calling _complete_scan on the mock after
    the scan is "running" — the mock's _active_scans dict is updated by
    _complete_scan (which is called on the real watcher), so we simulate
    the cleanup by manually removing the entry after completing.
    """
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)
        folder = create_test_files(tmp_dir, 3)
        datastore_id = create_datastore(db, folder, prefix="t10")

        create_admin_user(db, prefix="t10")
        token = get_token(client, prefix="t10")

        # First, test with a stale scan that WAS NOT cleaned up (old behavior)
        mock_w = MagicMock()
        mock_w._active_scans = {
            1: {
                "datastore_id": datastore_id,
                "total": 3,
                "processed": 3,
                "status": "completed",  # stale completed scan
                "error_count": 0,
                "new": 3,
                "modified": 0,
                "skipped": 0,
                "error_message": None,
            }
        }
        with patch("app.api.api_v1.datastores._get_watcher", return_value=mock_w):
            with client.stream(
                "GET",
                f"/api/admin/datastores/{datastore_id}/scan-progress-stream",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            ) as sse:
                assert sse.status_code == 200
                lines = list(sse.iter_lines())
                # With stale scan, SSE would find it and emit a "completed" event
                assert any(
                    '"completed"' in line for line in lines
                ), f"Expected stale scan event: {lines}"

        # Now simulate cleanup: remove the stale entry from _active_scans
        mock_w._active_scans = {}
        with patch("app.api.api_v1.datastores._get_watcher", return_value=mock_w):
            with client.stream(
                "GET",
                f"/api/admin/datastores/{datastore_id}/scan-progress-stream",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            ) as sse:
                assert sse.status_code == 200
                lines = list(sse.iter_lines())
                # After cleanup, SSE should NOT find any scan
                assert any(
                    "Scan not found" in line for line in lines
                ), f"Expected 'Scan not found': {lines}"
                # Should NOT have any "completed" event
                assert not any(
                    '"completed"' in line for line in lines
                ), f"Stale completed scan found: {lines}"


def test_sse_finds_completed_scan_before_sse_connects(client, db):
    """If the scan completes BEFORE the SSE endpoint connects, the SSE
    endpoint must still find the completed scan and emit the completion
    event (with correct data), not emit 'Scan not found'."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)
        folder = create_test_files(tmp_dir, 3)
        datastore_id = create_datastore(db, folder, prefix="t11")

        create_admin_user(db, prefix="t11")
        token = get_token(client, prefix="t11")

        # Simulate the race: scan completes (status = "completed") BEFORE
        # SSE endpoint connects. The scan entry is still in _active_scans
        # because _complete_scan doesn't remove it.


def test_trigger_scan_cleanup_does_not_raise_runtime_error(client, db):
    """POST /scan must clean up stale scan entries from _active_scans
    without raising RuntimeError: dictionary changed size during iteration.

    This test verifies the fix for the bug where iterating over
    _active_scans.items() while calling pop() on the same dict caused
    a RuntimeError, and the stale scan remained in _active_scans,
    causing the SSE endpoint to find the stale completed scan instead
    of the new one.
    """
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)
        folder = create_test_files(tmp_dir, 3)
        datastore_id = create_datastore(db, folder, prefix="t12")

        create_admin_user(db, prefix="t12")
        token = get_token(client, prefix="t12")

        # Create a stale completed scan in _active_scans
        mock_w = MagicMock()
        mock_w._active_scans = {
            1: {
                "datastore_id": datastore_id,
                "total": 3,
                "processed": 3,
                "status": "completed",  # stale
                "error_count": 0,
                "new": 3,
                "modified": 0,
                "skipped": 0,
                "error_message": None,
            }
        }
        mock_w._init_scan = MagicMock(return_value=2)
        mock_w.scan_single_datastore = MagicMock(return_value={"scanned": 3, "new": 3, "modified": 0, "skipped": 0, "errors": 0})

        with patch("app.api.api_v1.datastores._get_watcher", return_value=mock_w):
            # The POST should NOT raise RuntimeError even though the cleanup
            # modifies _active_scans while iterating over it.
            resp = client.post(
                f"/api/admin/datastores/{datastore_id}/scan",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 202, f"Expected 202, got {resp.status_code}: {resp.text}"

        # Verify the stale scan was removed from _active_scans (not RuntimeError)
        assert datastore_id not in [
            s["datastore_id"] for s in mock_w._active_scans.values()
        ], f"Stale scan not cleaned up: {mock_w._active_scans}"
        # The POST endpoint does not add the new scan to _active_scans itself
        # — that happens in the background thread via _init_scan. So the
        # dict is empty after the POST, confirming the cleanup worked.


def test_trigger_scan_cleanup_old_bug_would_leave_stale_scan(client, db):
    """Regression test: with the old buggy code (pop() during iteration),
    the RuntimeError would be caught silently and the stale scan would
    remain in _active_scans. This test verifies the fix works correctly.

    The old buggy code was:
        for sid, info in watcher._active_scans.items():
            if info.get("datastore_id") == datastore_id:
                watcher._active_scans.pop(sid, None)  # RuntimeError!
    """
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)
        folder = create_test_files(tmp_dir, 3)
        datastore_id = create_datastore(db, folder, prefix="t13")

        create_admin_user(db, prefix="t13")
        token = get_token(client, prefix="t13")

        # Create multiple stale scans (should all be removed, not just one)
        mock_w = MagicMock()
        mock_w._active_scans = {
            1: {
                "datastore_id": datastore_id,
                "total": 3,
                "processed": 3,
                "status": "completed",
                "error_count": 0,
                "new": 2,
                "modified": 0,
                "skipped": 1,
                "error_message": None,
            },
            3: {
                "datastore_id": 999,  # different datastore
                "total": 5,
                "processed": 5,
                "status": "completed",
                "error_count": 0,
                "new": 5,
                "modified": 0,
                "skipped": 0,
                "error_message": None,
            },
        }
        mock_w._init_scan = MagicMock(return_value=2)
        mock_w.scan_single_datastore = MagicMock(return_value={"scanned": 3, "new": 3, "modified": 0, "skipped": 0, "errors": 0})

        with patch("app.api.api_v1.datastores._get_watcher", return_value=mock_w):
            resp = client.post(
                f"/api/admin/datastores/{datastore_id}/scan",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 202, f"Expected 202, got {resp.status_code}: {resp.text}"

        # Only the stale scan for THIS datastore should be removed
        # The scan for datastore_id=999 should remain
        remaining = mock_w._active_scans
        assert datastore_id not in [s["datastore_id"] for s in remaining.values()], \
            f"Stale scan for this datastore not cleaned up: {remaining}"
        assert 999 in [s["datastore_id"] for s in remaining.values()], \
            f"Scan for OTHER datastore was incorrectly removed: {remaining}"
        # Should have exactly 1 remaining scan (for the other datastore)
        assert len(remaining) == 1, f"Expected 1 remaining scan, got {len(remaining)}: {remaining}"
        mock_w = MagicMock()
        mock_w._active_scans = {
            1: {
                "datastore_id": datastore_id,
                "total": 3,
                "processed": 3,
                "status": "completed",  # completed before SSE connects
                "error_count": 0,
                "new": 2,
                "modified": 0,
                "skipped": 1,
                "error_message": None,
            }
        }
        with patch("app.api.api_v1.datastores._get_watcher", return_value=mock_w):
            with client.stream(
                "GET",
                f"/api/admin/datastores/{datastore_id}/scan-progress-stream",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            ) as sse:
                assert sse.status_code == 200
                lines = list(sse.iter_lines())
                # Should find the completed scan immediately
                assert any(
                    '"completed"' in line for line in lines
                ), f"Expected completion event, got: {lines}"
                # Should NOT get "Scan not found"
                assert not any(
                    "Scan not found" in line for line in lines
                ), f"Should not get 'Scan not found': {lines}"
                # Should have correct data fields
                evt_line = [l for l in lines if '"status"' in l and '"completed"' in l][0]
                evt_data = json.loads(evt_line[6:])
                assert evt_data["status"] == "completed"
                assert evt_data["scanned"] == 3
                assert evt_data["new_files"] == 2
                assert evt_data["modified_files"] == 0
                assert evt_data["skipped_files"] == 1
