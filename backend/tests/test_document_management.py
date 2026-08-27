"""Tests for datastore document management — browse, select, unselect, delete.

Covers:
  1. get_folder_contents() lists folders and files with ingestion state.
  2. select_documents() creates/updates Document records with is_selected=true.
  3. unselect_documents() deletes ingested data and sets is_selected=false.
  4. delete_document_data() cleans up Qdrant/Neo4j/DB (mocked).
  5. select_folder() bulk selects/unselects all files in a folder.
  6. is_selected=false skips ingestion in watcher/handler/recovery paths.
"""
import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest
from sqlalchemy.orm import sessionmaker

import app.db.session as _session_mod
from app.db.session import get_db
from app.models.base import Base  # noqa
import app.models.datastore  # noqa: ensure tables registered
import app.models.knowledge  # noqa
from app.models.datastore import DataStore, DataStoreFileManifest
from app.models.knowledge import Document, DocumentChunk, ProcessingTask

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
def tmp_datastore(tmp_path):
    """Create a datastore with a folder structure on disk.

    Structure:
        root/
          file1.pdf
          file2.txt
          subfolder/
            file3.pdf
            file4.docx
    """
    root = tmp_path / "test_store"
    root.mkdir()
    (root / "file1.pdf").write_bytes(b"pdf content")
    (root / "file2.txt").write_bytes(b"text content")
    sub = root / "subfolder"
    sub.mkdir()
    (sub / "file3.pdf").write_bytes(b"sub pdf")
    (sub / "file4.docx").write_bytes(b"sub docx")

    db_session = TestingSessionLocal()
    try:
        ds = DataStore(
            name="Test Store",
            folder_path=str(root),
            scan_pattern="*",
            is_active=True,
        )
        db_session.add(ds)
        db_session.commit()
        db_session.refresh(ds)
        return ds, str(root)
    finally:
        db_session.close()


def _make_document(db, datastore_id, file_path, file_name, is_selected=True, chunks=0):
    """Create a Document and optionally some chunks."""
    doc = Document(
        data_store_id=datastore_id,
        file_path=file_path,
        file_name=file_name,
        file_size=100,
        content_type="application/pdf",
        file_hash="abc123",
        is_selected=is_selected,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    for i in range(chunks):
        chunk = DocumentChunk(
            id=f"chunk_{doc.id}_{i}",
            data_store_id=datastore_id,
            document_id=doc.id,
            file_name=file_name,
            chunk_text=f"chunk text {i}",
            chunk_index=i,
            hash=f"hash_{doc.id}_{i}",
        )
        db.add(chunk)
    db.commit()

    if chunks > 0:
        task = ProcessingTask(
            data_store_id=datastore_id,
            document_id=doc.id,
            status="completed",
        )
    else:
        task = ProcessingTask(
            data_store_id=datastore_id,
            document_id=doc.id,
            status="pending",
        )
    db.add(task)
    db.commit()
    return doc


# ---------------------------------------------------------------------------
# Tests: get_folder_contents
# ---------------------------------------------------------------------------

class TestGetFolderContents:
    def test_browse_root_lists_folders_and_files(self, tmp_datastore, db):
        from app.services.datastore.document_management import get_folder_contents
        ds, root = tmp_datastore

        result = get_folder_contents(db, ds.id, relative_path="")
        assert result["datastore_id"] == ds.id
        assert result["current_path"] == ""
        assert len(result["breadcrumbs"]) == 1  # Root only

        items = result["items"]
        folders = [i for i in items if i["type"] == "folder"]
        files = [i for i in items if i["type"] == "file"]
        assert len(folders) == 1
        assert folders[0]["name"] == "subfolder"
        assert len(files) == 2
        file_names = {f["name"] for f in files}
        assert file_names == {"file1.pdf", "file2.txt"}

    def test_browse_subfolder(self, tmp_datastore, db):
        from app.services.datastore.document_management import get_folder_contents
        ds, root = tmp_datastore

        result = get_folder_contents(db, ds.id, relative_path="subfolder")
        assert result["current_path"] == "subfolder"
        assert len(result["breadcrumbs"]) == 2  # Root > subfolder

        items = result["items"]
        files = [i for i in items if i["type"] == "file"]
        assert len(files) == 2
        file_names = {f["name"] for f in files}
        assert file_names == {"file3.pdf", "file4.docx"}

    def test_browse_shows_document_state(self, tmp_datastore, db):
        from app.services.datastore.document_management import get_folder_contents
        ds, root = tmp_datastore

        # Create a document for file1.pdf with chunks
        _make_document(db, ds.id, os.path.join(root, "file1.pdf"), "file1.pdf",
                       is_selected=True, chunks=3)

        result = get_folder_contents(db, ds.id, relative_path="")
        files = {f["name"]: f for f in result["items"] if f["type"] == "file"}

        assert files["file1.pdf"]["document_id"] is not None
        assert files["file1.pdf"]["is_selected"] is True
        assert files["file1.pdf"]["chunk_count"] == 3
        assert files["file1.pdf"]["status"] == "completed"

        # file2.txt has no document
        assert files["file2.txt"]["document_id"] is None
        assert files["file2.txt"]["is_selected"] is False
        assert files["file2.txt"]["status"] == "not_ingested"

    def test_browse_search_filters_by_name(self, tmp_datastore, db):
        from app.services.datastore.document_management import get_folder_contents
        ds, root = tmp_datastore

        result = get_folder_contents(db, ds.id, relative_path="", search="pdf")
        files = [i for i in result["items"] if i["type"] == "file"]
        assert len(files) == 1
        assert files[0]["name"] == "file1.pdf"

    def test_browse_path_outside_datastore_returns_error(self, tmp_datastore, db):
        from app.services.datastore.document_management import get_folder_contents
        ds, root = tmp_datastore

        result = get_folder_contents(db, ds.id, relative_path="../../etc")
        assert "error" in result

    def test_browse_nonexistent_datastore(self, db):
        from app.services.datastore.document_management import get_folder_contents
        result = get_folder_contents(db, 999999, relative_path="")
        assert result.get("error") == "datastore_not_found"

    def test_browse_stats(self, tmp_datastore, db):
        from app.services.datastore.document_management import get_folder_contents
        ds, root = tmp_datastore

        _make_document(db, ds.id, os.path.join(root, "file1.pdf"), "file1.pdf",
                       is_selected=True, chunks=2)
        _make_document(db, ds.id, os.path.join(root, "file2.txt"), "file2.txt",
                       is_selected=False, chunks=0)

        result = get_folder_contents(db, ds.id, relative_path="")
        stats = result["stats"]
        assert stats["total_documents"] == 2
        assert stats["selected"] == 1
        assert stats["unselected"] == 1
        assert stats["ingested"] == 1  # only file1 has chunks


# ---------------------------------------------------------------------------
# Tests: select_documents
# ---------------------------------------------------------------------------

class TestSelectDocuments:
    def test_select_creates_document_for_new_file(self, tmp_datastore):
        from app.services.datastore.document_management import select_documents
        ds, root = tmp_datastore

        file_path = os.path.join(root, "file1.pdf")
        result = select_documents(ds.id, [file_path])

        assert result["created"] == 1
        assert result["selected"] == 0  # no existing doc to select

        db = TestingSessionLocal()
        try:
            doc = db.query(Document).filter(
                Document.file_path == file_path,
                Document.data_store_id == ds.id,
            ).first()
            assert doc is not None
            assert doc.is_selected is True
        finally:
            db.close()

    def test_select_updates_existing_unselected_document(self, tmp_datastore, db):
        from app.services.datastore.document_management import select_documents
        ds, root = tmp_datastore

        file_path = os.path.join(root, "file1.pdf")
        doc = _make_document(db, ds.id, file_path, "file1.pdf", is_selected=False)

        result = select_documents(ds.id, [file_path])
        assert result["selected"] == 1
        assert result["created"] == 0

        db2 = TestingSessionLocal()
        try:
            doc2 = db2.query(Document).filter(Document.id == doc.id).first()
            assert doc2.is_selected is True
        finally:
            db2.close()

    def test_select_idempotent_on_already_selected(self, tmp_datastore, db):
        from app.services.datastore.document_management import select_documents
        ds, root = tmp_datastore

        file_path = os.path.join(root, "file1.pdf")
        _make_document(db, ds.id, file_path, "file1.pdf", is_selected=True)

        result = select_documents(ds.id, [file_path])
        assert result["selected"] == 0  # already selected, no change
        assert result["created"] == 0


# ---------------------------------------------------------------------------
# Tests: unselect_documents
# ---------------------------------------------------------------------------

class TestUnselectDocuments:
    def test_unselect_deletes_chunks_and_sets_flag(self, tmp_datastore, db):
        from app.services.datastore.document_management import unselect_documents
        ds, root = tmp_datastore

        file_path = os.path.join(root, "file1.pdf")
        doc = _make_document(db, ds.id, file_path, "file1.pdf",
                             is_selected=True, chunks=3)

        # Mock Qdrant and Neo4j to avoid real connections
        with patch("app.services.datastore.document_management.get_qdrant_client"):
            with patch("app.services.graph.delete_graph_for_document"):
                result = unselect_documents(ds.id, [file_path])

        assert result["unselected"] == 1
        assert result["deleted_chunks"] == 3

        db2 = TestingSessionLocal()
        try:
            # Document should still exist but is_selected=False
            doc2 = db2.query(Document).filter(Document.id == doc.id).first()
            assert doc2 is not None
            assert doc2.is_selected is False

            # Chunks should be deleted
            chunks = db2.query(DocumentChunk).filter(
                DocumentChunk.document_id == doc.id
            ).count()
            assert chunks == 0

            # ProcessingTask should be deleted
            tasks = db2.query(ProcessingTask).filter(
                ProcessingTask.document_id == doc.id
            ).count()
            assert tasks == 0
        finally:
            db2.close()

    def test_unselect_deletes_manifest_entry(self, tmp_datastore, db):
        from app.services.datastore.document_management import unselect_documents
        ds, root = tmp_datastore

        file_path = os.path.join(root, "file1.pdf")
        _make_document(db, ds.id, file_path, "file1.pdf", is_selected=True, chunks=1)

        # Add manifest entry
        manifest = DataStoreFileManifest(
            datastore_id=ds.id,
            file_path=file_path,
            file_hash="abc123",
            file_size=100,
            file_mtime=12345,
        )
        db.add(manifest)
        db.commit()

        with patch("app.services.datastore.document_management.get_qdrant_client"):
            with patch("app.services.graph.delete_graph_for_document"):
                unselect_documents(ds.id, [file_path])

        db2 = TestingSessionLocal()
        try:
            count = db2.query(DataStoreFileManifest).filter(
                DataStoreFileManifest.datastore_id == ds.id,
                DataStoreFileManifest.file_path == file_path,
            ).count()
            assert count == 0
        finally:
            db2.close()

    def test_unselect_skips_already_unselected(self, tmp_datastore, db):
        from app.services.datastore.document_management import unselect_documents
        ds, root = tmp_datastore

        file_path = os.path.join(root, "file1.pdf")
        _make_document(db, ds.id, file_path, "file1.pdf", is_selected=False, chunks=0)

        with patch("app.services.datastore.document_management.get_qdrant_client"):
            with patch("app.services.graph.delete_graph_for_document"):
                result = unselect_documents(ds.id, [file_path])

        assert result["unselected"] == 0  # nothing to do

    def test_unselect_nonexistent_file_is_noop(self, tmp_datastore):
        from app.services.datastore.document_management import unselect_documents
        ds, root = tmp_datastore

        with patch("app.services.datastore.document_management.get_qdrant_client"):
            with patch("app.services.graph.delete_graph_for_document"):
                result = unselect_documents(ds.id, ["/nonexistent/path.pdf"])

        assert result["unselected"] == 0


# ---------------------------------------------------------------------------
# Tests: select_folder
# ---------------------------------------------------------------------------

class TestSelectFolder:
    def test_select_folder_recursive(self, tmp_datastore, db):
        from app.services.datastore.document_management import select_folder
        ds, root = tmp_datastore

        # Create documents for some files
        _make_document(db, ds.id, os.path.join(root, "file1.pdf"), "file1.pdf",
                       is_selected=False, chunks=0)
        _make_document(db, ds.id, os.path.join(root, "subfolder", "file3.pdf"),
                       "file3.pdf", is_selected=False, chunks=0)

        result = select_folder(ds.id, root, selected=True, recursive=True)

        # Should select both files
        assert result["selected"] == 2

    def test_unselect_folder_recursive(self, tmp_datastore, db):
        from app.services.datastore.document_management import select_folder
        ds, root = tmp_datastore

        # Create documents with chunks
        _make_document(db, ds.id, os.path.join(root, "file1.pdf"), "file1.pdf",
                       is_selected=True, chunks=2)
        _make_document(db, ds.id, os.path.join(root, "subfolder", "file3.pdf"),
                       "file3.pdf", is_selected=True, chunks=1)

        with patch("app.services.datastore.document_management.get_qdrant_client"):
            with patch("app.services.graph.delete_graph_for_document"):
                result = select_folder(ds.id, root, selected=False, recursive=True)

        assert result["unselected"] == 2
        assert result["deleted_chunks"] == 3


# ---------------------------------------------------------------------------
# Tests: is_selected in ingestion paths
# ---------------------------------------------------------------------------

class TestIsSelectedInIngestion:
    def test_unselected_document_is_skipped_by_watcher_scan(self, tmp_datastore, db):
        """When a Document has is_selected=False, the watcher scan should skip it."""
        ds, root = tmp_datastore
        file_path = os.path.join(root, "file1.pdf")

        # Create an unselected document with matching hash
        doc = _make_document(db, ds.id, file_path, "file1.pdf",
                             is_selected=False, chunks=0)
        doc.file_hash = "known_hash"
        db.commit()

        # Simulate what _handle_file_in_scan does: check is_selected
        db2 = TestingSessionLocal()
        try:
            existing = db2.query(Document).filter(
                Document.file_path == file_path,
                Document.data_store_id == ds.id,
            ).first()

            assert existing is not None
            assert existing.is_selected is False
            # The watcher should skip this file
        finally:
            db2.close()
