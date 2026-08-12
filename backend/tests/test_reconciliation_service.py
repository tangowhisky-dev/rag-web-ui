"""Tests for the startup reconciliation service.

Verifies that run_reconciliation handles missing external services
(Qdrant, Neo4j) gracefully and cleans up orphaned MySQL rows.
"""
import pytest
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker

from app.models.base import Base
import app.models.user  # noqa
import app.models.knowledge  # noqa
import app.models.datastore  # noqa
import app.models.organisation  # noqa


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


class TestRunReconciliation:
    """Test the top-level run_reconciliation function."""

    def test_returns_summary_dict_even_when_stores_unavailable(self):
        """run_reconciliation should return a summary dict even if
        Qdrant/Neo4j are unreachable."""
        from app.services.cleanup.reconciliation_service import run_reconciliation

        with patch("app.services.cleanup.reconciliation_service.SessionLocal") as mock_session_local:
            mock_db = MagicMock()
            mock_session_local.return_value = mock_db
            mock_db.query.return_value.all.return_value = []
            mock_db.query.return_value.outerjoin.return_value.filter.return_value.all.return_value = []

            summary = run_reconciliation()

        assert "mysql" in summary
        assert "qdrant" in summary
        assert "neo4j" in summary
        assert isinstance(summary["mysql"]["orphan_chunks"], int)
        assert isinstance(summary["qdrant"]["dropped_collections"], int)

    def test_mysql_orphan_chunk_cleanup(self, db_session):
        """Orphan DocumentChunk rows (document_id pointing to non-existent
        Document) should be deleted."""
        from app.services.cleanup.reconciliation_service import _reconcile_mysql
        from app.models.knowledge import DocumentChunk

        orphan = DocumentChunk(
            id="orphan_chunk_1",
            document_id=999999,
            file_name="test.txt",
            chunk_text="orphan chunk text",
            hash="abc123",
        )
        db_session.add(orphan)
        db_session.commit()

        with patch("app.services.cleanup.reconciliation_service.SessionLocal", return_value=db_session):
            summary: dict = {"mysql": {"orphan_chunks": 0, "orphan_tasks": 0}}
            _reconcile_mysql(summary)

        assert summary["mysql"]["orphan_chunks"] >= 1

        remaining = db_session.query(DocumentChunk).filter(
            DocumentChunk.id == "orphan_chunk_1"
        ).first()
        assert remaining is None

    def test_mysql_orphan_task_cleanup(self, db_session):
        """Orphan ProcessingTask rows (document_id pointing to non-existent
        Document) should be deleted."""
        from app.services.cleanup.reconciliation_service import _reconcile_mysql
        from app.models.knowledge import ProcessingTask

        orphan_task = ProcessingTask(
            document_id=999999,
            status="failed",
        )
        db_session.add(orphan_task)
        db_session.commit()
        orphan_id = orphan_task.id

        with patch("app.services.cleanup.reconciliation_service.SessionLocal", return_value=db_session):
            summary: dict = {"mysql": {"orphan_chunks": 0, "orphan_tasks": 0}}
            _reconcile_mysql(summary)

        assert summary["mysql"]["orphan_tasks"] >= 1

        remaining = db_session.query(ProcessingTask).filter(
            ProcessingTask.id == orphan_id
        ).first()
        assert remaining is None

    def test_mysql_does_not_delete_valid_chunks(self, db_session):
        """Chunks with a valid document_id should NOT be deleted."""
        from app.services.cleanup.reconciliation_service import _reconcile_mysql
        from app.models.knowledge import DocumentChunk, Document

        doc = Document(
            file_path="/test/file.txt",
            file_name="file.txt",
            file_size=100,
            content_type="text/plain",
        )
        db_session.add(doc)
        db_session.commit()

        valid_chunk = DocumentChunk(
            id="valid_chunk_1",
            document_id=doc.id,
            file_name="file.txt",
            chunk_text="valid chunk text",
            hash="def456",
        )
        db_session.add(valid_chunk)
        db_session.commit()

        with patch("app.services.cleanup.reconciliation_service.SessionLocal", return_value=db_session):
            summary: dict = {"mysql": {"orphan_chunks": 0, "orphan_tasks": 0}}
            _reconcile_mysql(summary)

        remaining = db_session.query(DocumentChunk).filter(
            DocumentChunk.id == "valid_chunk_1"
        ).first()
        assert remaining is not None


class TestQdrantReconciliation:
    """Test Qdrant collection reconciliation with mocked client."""

    def test_drops_stale_kb_collections(self):
        """Stale kb_ collections (KB deleted from MySQL) should be dropped."""
        from app.services.cleanup.reconciliation_service import _reconcile_qdrant

        mock_qdrant = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "kb_999"
        mock_qdrant.get_collections.return_value.collections = [mock_collection]

        with patch("app.services.infrastructure.utils.get_qdrant_client", return_value=mock_qdrant):
            with patch("app.services.cleanup.reconciliation_service.SessionLocal") as mock_sl:
                mock_db = MagicMock()
                mock_sl.return_value = mock_db
                mock_db.query.return_value.outerjoin.return_value.filter.return_value.all.return_value = []

                summary: dict = {"qdrant": {"dropped_collections": 0, "orphan_points": 0}}
                _reconcile_qdrant(summary, active_kb_ids=[1], active_ds_ids=[1])

        mock_qdrant.delete_collection.assert_any_call("kb_999")
        assert summary["qdrant"]["dropped_collections"] >= 1

    def test_drops_stale_ds_collections(self):
        """Stale ds_ collections (DataStore deleted from MySQL) should be dropped."""
        from app.services.cleanup.reconciliation_service import _reconcile_qdrant

        mock_qdrant = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "ds_888"
        mock_qdrant.get_collections.return_value.collections = [mock_collection]

        with patch("app.services.infrastructure.utils.get_qdrant_client", return_value=mock_qdrant):
            with patch("app.services.cleanup.reconciliation_service.SessionLocal") as mock_sl:
                mock_db = MagicMock()
                mock_sl.return_value = mock_db
                mock_db.query.return_value.outerjoin.return_value.filter.return_value.all.return_value = []

                summary: dict = {"qdrant": {"dropped_collections": 0, "orphan_points": 0}}
                _reconcile_qdrant(summary, active_kb_ids=[1], active_ds_ids=[1])

        mock_qdrant.delete_collection.assert_any_call("ds_888")
        assert summary["qdrant"]["dropped_collections"] >= 1

    def test_does_not_drop_active_collections(self):
        """Active collections should NOT be dropped."""
        from app.services.cleanup.reconciliation_service import _reconcile_qdrant

        mock_qdrant = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "kb_1"
        mock_qdrant.get_collections.return_value.collections = [mock_collection]
        mock_qdrant.scroll.return_value = ([], None)

        with patch("app.services.infrastructure.utils.get_qdrant_client", return_value=mock_qdrant):
            with patch("app.services.cleanup.reconciliation_service.SessionLocal") as mock_sl:
                mock_db = MagicMock()
                mock_sl.return_value = mock_db
                mock_db.query.return_value.outerjoin.return_value.filter.return_value.all.return_value = []
                mock_db.query.return_value.filter.return_value.all.return_value = []

                summary: dict = {"qdrant": {"dropped_collections": 0, "orphan_points": 0}}
                _reconcile_qdrant(summary, active_kb_ids=[1], active_ds_ids=[])

        mock_qdrant.delete_collection.assert_not_called()
        assert summary["qdrant"]["dropped_collections"] == 0


class TestNeo4jReconciliation:
    """Test Neo4j reconciliation with mocked driver."""

    def test_purge_stale_graph_data_called_with_active_kb_ids(self):
        """purge_stale_graph_data should be called with all active KB IDs."""
        from app.services.cleanup.reconciliation_service import _reconcile_neo4j

        with patch("app.services.graph.purge_stale_graph_data") as mock_purge:
            with patch("app.core.config.settings.NEO4J_URI", "bolt://localhost"):
                with patch("app.services.graph._get_driver") as mock_driver:
                    mock_session = MagicMock()
                    mock_driver.return_value.session.return_value.__enter__.return_value = mock_session
                    mock_session.run.return_value = []

                    summary: dict = {"neo4j": {"purged_kbs": 0, "purged_datastores": 0}}
                    _reconcile_neo4j(summary, active_kb_ids=[1, 2, 3], active_ds_ids=[10])

        mock_purge.assert_called_once_with(active_kb_ids=[1, 2, 3])

    def test_skips_when_neo4j_uri_not_set(self):
        """Should skip Neo4j reconciliation when NEO4J_URI is not configured."""
        from app.services.cleanup.reconciliation_service import _reconcile_neo4j

        with patch("app.core.config.settings.NEO4J_URI", ""):
            with patch("app.services.graph.purge_stale_graph_data") as mock_purge:
                summary: dict = {"neo4j": {"purged_kbs": 0, "purged_datastores": 0}}
                _reconcile_neo4j(summary, active_kb_ids=[1], active_ds_ids=[1])

        mock_purge.assert_not_called()
