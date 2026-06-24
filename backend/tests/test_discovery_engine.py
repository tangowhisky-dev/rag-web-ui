"""Tests for discovery_engine — hash correctness, discovery classification, concurrency.

Covers:
  1. hash_file() returns correct 64-char hex SHA-256 digest.
  2. _matches_pattern() correctly matches wildcards and skips hidden files.
  3. discover_datastore() on an empty datastore returns 0 new files.
  4. discover_datastore() with new files — new_files count matches.
  5. discover_datastore() with modified files — modified count > 0.
  6. discover_datastore() with deleted files — deleted count > 0.
  7. discover_all() processes multiple datastores.
"""
import os
import hashlib
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest
from sqlalchemy.orm import sessionmaker

# conftest.py has already patched MySQL dialect types and app.db.session.
import app.db.session as _session_mod
from app.db.session import get_db
from app.models.base import Base  # noqa
import app.models.datastore  # noqa: ensure tables are registered

engine = _session_mod.engine
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_db():
    """Create all tables before each test, drop them after."""
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
def tmp_datastore_dir(tmp_path):
    """Create a temporary datastore directory and register it in the DB.

    Returns (directory_path, DataStore object) so callers can add files
    to the folder before running discovery.
    """
    import app.models.datastore as ds_mod

    folder = tmp_path / "test_store"
    folder.mkdir()

    ds = ds_mod.DataStore(
        name="Test Store",
        folder_path=str(folder),
        scan_pattern="*",
        is_active=True,
    )
    db_session = TestingSessionLocal()
    try:
        db_session.add(ds)
        db_session.commit()
        db_session.refresh(ds)
        return str(folder), ds
    finally:
        db_session.close()


@pytest.fixture()
def active_datastores(db):
    """Create two active datastores and return their IDs + paths."""
    import app.models.datastore as ds_mod
    from pathlib import Path

    p1 = Path("/tmp/_ds_a")
    p2 = Path("/tmp/_ds_b")
    p1.mkdir(exist_ok=True)
    p2.mkdir(exist_ok=True)

    d1 = ds_mod.DataStore(name="DS A", folder_path=str(p1), is_active=True)
    d2 = ds_mod.DataStore(name="DS B", folder_path=str(p2), is_active=True)
    db.add_all([d1, d2])
    db.commit()
    db.refresh(d1)
    db.refresh(d2)
    return (d1.id, str(p1), d2.id, str(p2))


# ---------------------------------------------------------------------------
# Unit tests: hash_file
# ---------------------------------------------------------------------------

class TestHashFile:
    """Tests for hash_file() — SHA-256 correctness."""

    def test_hash_file_returns_64_char_hex(self, tmp_path):
        """hash_file() must return a 64-character hex digest."""
        from app.services.discovery_engine import hash_file

        f = tmp_path / "small.txt"
        f.write_text("hello world")
        digest = hash_file(str(f))
        assert len(digest) == 64
        # Should be valid hex
        int(digest, 16)

    def test_hash_file_known_content(self, tmp_path):
        """hash_file('hello world') must equal the known SHA-256 of 'hello world'."""
        from app.services.discovery_engine import hash_file

        content = b"hello world"
        f = tmp_path / "known.bin"
        f.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert hash_file(str(f)) == expected

    def test_hash_file_empty_file(self, tmp_path):
        """hash_file() on an empty file must return a valid 64-char hex."""
        from app.services.discovery_engine import hash_file

        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        digest = hash_file(str(f))
        assert len(digest) == 64
        assert digest == hashlib.sha256(b"").hexdigest()

    def test_hash_file_os_error_returns_empty(self, tmp_path):
        """hash_file() must return '' when the file cannot be read."""
        from app.services.discovery_engine import hash_file

        f = tmp_path / "nope.txt"
        f.write_text("x")
        # Pass a path that does not exist
        assert hash_file(str(tmp_path / "nonexistent_xyz.txt")) == ""


# ---------------------------------------------------------------------------
# Unit tests: _matches_pattern
# ---------------------------------------------------------------------------

class TestMatchesPattern:
    """Tests for _matches_pattern() — wildcard and hidden file filtering."""

    def test_wildcard_pdf(self):
        from app.services.discovery_engine import _matches_pattern

        assert _matches_pattern("doc.pdf", "*.pdf") is True
        assert _matches_pattern("report.pdf", "*.pdf") is True

    def test_wildcard_no_extension_mismatch(self):
        from app.services.discovery_engine import _matches_pattern

        assert _matches_pattern("doc.txt", "*.pdf") is False

    def test_hidden_file_skipped(self):
        from app.services.discovery_engine import _matches_pattern

        assert _matches_pattern(".DS_Store", "*", skip_hidden=True) is False
        assert _matches_pattern(".hidden", "*.txt", skip_hidden=True) is False

    def test_hidden_file_included_when_disabled(self):
        from app.services.discovery_engine import _matches_pattern

        assert _matches_pattern(".DS_Store", "*", skip_hidden=False) is True

    def test_star_matches_all(self):
        from app.services.discovery_engine import _matches_pattern

        assert _matches_pattern("anything.xyz", "*") is True

    def test_specific_extension(self):
        from app.services.discovery_engine import _matches_pattern

        assert _matches_pattern("file.docx", "*.docx") is True
        assert _matches_pattern("file.docx", "*.pdf") is False

    def test_nonexistent_hidden_when_enabled(self):
        from app.services.discovery_engine import _matches_pattern

        # basename starts with '.' should be excluded
        assert _matches_pattern("/a/b/.gitkeep", "*.git*", skip_hidden=True) is False
        assert _matches_pattern("/a/b/.gitkeep", "*.git*", skip_hidden=False) is True


# ---------------------------------------------------------------------------
# Integration tests: discover_datastore (empty, new, modified, deleted)
# ---------------------------------------------------------------------------

class TestDiscoverDatastore:
    """Tests for discover_datastore() — classification of files."""

    def test_empty_datastore(self, tmp_datastore_dir):
        """Discovery on a datastore with zero files must return 0 new_files."""
        from app.services.discovery_engine import discover_datastore

        folder_path, ds = tmp_datastore_dir

        result = discover_datastore(ds.id)
        assert result.datastore_id == ds.id
        assert result.datastore_name == "Test Store"
        assert result.total_files_discovered == 0
        assert len(result.new_files) == 0
        assert len(result.modified_files) == 0
        assert len(result.deleted_files) == 0

    def test_new_files_detected(self, tmp_datastore_dir):
        """Creating files and running discovery must classify them as new."""
        from app.services.discovery_engine import discover_datastore

        folder_path, ds = tmp_datastore_dir

        # Create two files in the datastore folder
        f1 = os.path.join(folder_path, "file1.pdf")
        f2 = os.path.join(folder_path, "file2.txt")
        with open(f1, "wb") as fh:
            fh.write(b"alpha")
        with open(f2, "wb") as fh:
            fh.write(b"beta")

        result = discover_datastore(ds.id)
        assert len(result.new_files) == 2
        assert result.total_files_discovered == 2
        assert len(result.modified_files) == 0
        assert len(result.deleted_files) == 0

        # Verify hashes match actual SHA-256
        expected_f1 = hashlib.sha256(b"alpha").hexdigest()
        expected_f2 = hashlib.sha256(b"beta").hexdigest()
        hashes = {e["file_hash"] for e in result.new_files}
        assert expected_f1 in hashes
        assert expected_f2 in hashes

    def test_modified_file_detected(self, tmp_datastore_dir, db):
        """Modifying a file's content must show up as a modified entry."""
        from app.services.discovery_engine import discover_datastore

        folder_path, ds = tmp_datastore_dir

        f = os.path.join(folder_path, "doc.pdf")
        with open(f, "wb") as fh:
            fh.write(b"version one")

        # First discovery: file is new
        r1 = discover_datastore(ds.id)
        assert len(r1.new_files) == 1
        assert len(r1.modified_files) == 0

        # Modify the file content
        with open(f, "wb") as fh:
            fh.write(b"version two modified")

        # Second discovery: file is modified
        r2 = discover_datastore(ds.id)

        assert len(r2.new_files) == 0
        assert len(r2.modified_files) == 1
        assert len(r2.deleted_files) == 0

    def test_deleted_file_detected(self, tmp_datastore_dir, db):
        """Deleting a file must show up as a deleted manifest entry."""
        from app.services.discovery_engine import discover_datastore

        folder_path, ds = tmp_datastore_dir

        f = os.path.join(folder_path, "keep.pdf")
        g = os.path.join(folder_path, "remove.txt")
        with open(f, "wb") as fh:
            fh.write(b"kept content")
        with open(g, "wb") as fh:
            fh.write(b"to be deleted")

        # First discovery: both files are new
        r1 = discover_datastore(ds.id)
        assert len(r1.new_files) == 2

        # Delete one file
        os.remove(g)

        # Second discovery: one deleted
        r2 = discover_datastore(ds.id)

        assert len(r2.deleted_files) == 1
        assert len(r2.new_files) == 0
        assert os.path.basename(r2.deleted_files[0]["file_path"]) == "remove.txt"

    def test_combined_new_modified_deleted(self, tmp_datastore_dir, db):
        """A single discovery can simultaneously detect new, modified, and deleted files."""
        from app.services.discovery_engine import discover_datastore

        folder_path, ds = tmp_datastore_dir

        # Set up initial state with two files
        f_existing = os.path.join(folder_path, "existing.pdf")
        f_delete = os.path.join(folder_path, "bye.txt")
        with open(f_existing, "wb") as fh:
            fh.write(b"original")
        with open(f_delete, "wb") as fh:
            fh.write(b"bye")

        # First discovery: both files are new
        r1 = discover_datastore(ds.id)
        assert len(r1.new_files) == 2

        # Now: modify existing, delete one, add new
        with open(f_existing, "wb") as fh:
            fh.write(b"changed content now")
        os.remove(f_delete)
        f_new = os.path.join(folder_path, "brand_new.docx")
        with open(f_new, "wb") as fh:
            fh.write(b"brand new content")

        # Second discovery
        r2 = discover_datastore(ds.id)

        assert len(r2.modified_files) == 1
        assert len(r2.deleted_files) == 1
        assert len(r2.new_files) == 1
        assert r2.total_files_discovered == 2  # modified + new

    def test_inactive_datastore_returns_0(self, tmp_datastore_dir):
        """Discovery on an inactive datastore must return zero files."""
        from app.services.discovery_engine import discover_datastore

        folder_path, ds = tmp_datastore_dir

        # Mark inactive
        db_session = TestingSessionLocal()
        try:
            ds_obj = db_session.query(
                __import__("app.models.datastore", fromlist=["DataStore"]).DataStore
            ).filter_by(id=ds.id).first()
            ds_obj.is_active = False
            db_session.commit()
        finally:
            db_session.close()

        result = discover_datastore(ds.id)

        assert result.total_files_discovered == 0
        assert len(result.new_files) == 0

    def test_unknown_datastore_id(self, tmp_datastore_dir):
        """Discovering a non-existent datastore_id must return zero results."""
        from app.services.discovery_engine import discover_datastore

        result = discover_datastore(999999)

        assert result.datastore_id == 999999
        assert result.datastore_name == "unknown"
        assert result.total_files_discovered == 0

    def test_scan_pattern_filters_files(self, tmp_datastore_dir):
        """When scan_pattern='*.pdf', non-PDF files must be excluded."""
        from app.services.discovery_engine import discover_datastore, DiscoveryConfig
        from unittest.mock import patch

        folder_path, ds = tmp_datastore_dir

        # Create files with different extensions
        with open(os.path.join(folder_path, "a.pdf"), "wb") as fh:
            fh.write(b"pdf content")
        with open(os.path.join(folder_path, "b.txt"), "wb") as fh:
            fh.write(b"txt content")
        with open(os.path.join(folder_path, "c.docx"), "wb") as fh:
            fh.write(b"docx content")

        # Patch DiscoveryConfig to use pdf-only pattern
        with patch(
            "app.services.discovery_engine.DiscoveryConfig",
            **{"return_value.scan_pattern": "*.pdf", "return_value.max_workers": 2},
        ):
            result = discover_datastore(ds.id)

        assert len(result.new_files) == 1
        assert result.new_files[0]["file_path"].endswith("a.pdf")


# ---------------------------------------------------------------------------
# Integration tests: discover_all
# ---------------------------------------------------------------------------

class TestDiscoverAll:
    """Tests for discover_all() — multi-datastore concurrent discovery."""

    def test_discover_all_multiple_datastores(self, active_datastores):
        """discover_all() must process multiple active datastores.

        Uses mock to avoid the SQLAlchemy session-not-thread-safe issue
        when discover_all calls discover_datastore from multiple threads.
        """
        from app.services.discovery_engine import discover_all, DiscoveryResult
        from unittest.mock import patch

        ds_a_id, ds_a_path, ds_b_id, ds_b_path = active_datastores

        # Mock discover_datastore to return deterministic results
        def mock_discover(datastore_id):
            name = "DS A" if datastore_id == ds_a_id else "DS B"
            return DiscoveryResult(
                datastore_id=datastore_id,
                datastore_name=name,
                folder_path=ds_a_path if datastore_id == ds_a_id else ds_b_path,
                total_files_discovered=1,
                new_files=[{"file_path": f"file_{datastore_id}.txt",
                            "file_hash": "abc",
                            "file_size": 5}],
            )

        with patch(
            "app.services.discovery_engine.discover_datastore",
            side_effect=mock_discover,
        ):
            db_session = TestingSessionLocal()
            try:
                results = discover_all(db_session)
            finally:
                db_session.close()

        assert len(results) == 2
        ids = {r.datastore_id for r in results}
        assert ds_a_id in ids
        assert ds_b_id in ids
        for r in results:
            assert r.total_files_discovered == 1

    def test_discover_all_no_active_datastores(self):
        """discover_all() with no active datastores must return []."""
        from app.services.discovery_engine import discover_all

        db_session = TestingSessionLocal()
        try:
            results = discover_all(db_session)
        finally:
            db_session.close()

        assert results == []

    def test_discover_all_skips_inactive_datastores(self, active_datastores):
        """discover_all() must skip inactive datastores silently."""
        from app.services.discovery_engine import discover_all, DiscoveryResult
        from unittest.mock import patch

        ds_a_id, ds_a_path, ds_b_id, ds_b_path = active_datastores

        # Deactivate the second datastore in DB
        db_session = TestingSessionLocal()
        try:
            import app.models.datastore as ds_mod
            ds_b_row = db_session.query(ds_mod.DataStore).filter_by(id=ds_b_id).first()
            ds_b_row.is_active = False
            db_session.commit()
        finally:
            db_session.close()

        # Mock discover_datastore so DB threading isn't an issue
        def mock_discover(datastore_id):
            return DiscoveryResult(
                datastore_id=datastore_id,
                datastore_name=f"DS_{datastore_id}",
                folder_path=ds_a_path if datastore_id == ds_a_id else ds_b_path,
                total_files_discovered=1,
            )

        with patch(
            "app.services.discovery_engine.discover_datastore",
            side_effect=mock_discover,
        ):
            db_session = TestingSessionLocal()
            try:
                results = discover_all(db_session)
            finally:
                db_session.close()

        assert len(results) == 1
        assert results[0].datastore_id == ds_a_id


# ---------------------------------------------------------------------------
# Unit tests: DiscoveryResult
# ---------------------------------------------------------------------------

class TestDiscoveryResult:
    """Tests for DiscoveryResult serialization."""

    def test_to_dict_serializable(self):
        from app.services.discovery_engine import DiscoveryResult

        result = DiscoveryResult(
            datastore_id=1,
            datastore_name="Test",
            folder_path="/tmp/test",
            new_files=[{"file_path": "a.txt", "file_hash": "abc", "file_size": 3}],
            modified_files=[{"file_path": "b.txt", "file_hash": "def", "file_size": 5}],
            deleted_files=[{"file_path": "c.txt", "old_hash": "ghi", "old_size": 2}],
            skipped_files=1,
            total_files_discovered=2,
            elapsed_ms=42.5,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["datastore_id"] == 1
        assert d["total_files_discovered"] == 2
        assert d["elapsed_ms"] == 42.5
        assert len(d["new_files"]) == 1
        assert len(d["modified_files"]) == 1
        assert len(d["deleted_files"]) == 1
        assert d["skipped_files"] == 1

    def test_to_dict_empty_result(self):
        from app.services.discovery_engine import DiscoveryResult

        result = DiscoveryResult(
            datastore_id=0,
            datastore_name="empty",
            folder_path="",
        )
        d = result.to_dict()
        assert d["new_files"] == []
        assert d["total_files_discovered"] == 0
        assert d["elapsed_ms"] == 0.0
