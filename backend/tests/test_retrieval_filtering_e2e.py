"""End-to-end tests for agentic retrieval filtering.

These tests run against the live Docker stack (Qdrant, MySQL, LM Studio)
with ingested test documents in KB 1. They verify:
- kb_metadata tool returns real document metadata
- rag_retrieve with title_contains filter narrows results
- rag_retrieve with content_type filter narrows results
- rag_retrieve with sort by created_at desc returns newest first
- Filtered results contain no duplicate chunks in context
- Unfiltered retrieval returns all docs (backward compatible)
"""

import asyncio
import sys
import types
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest


# ── Fixtures ───────────────────────────────────────────────────────────────────

_LIVE_MYSQL_URL = "mysql+mysqlconnector://ragwebui:ragwebui@db:3306/ragwebui"


@pytest.fixture
def live_db():
    """Real MySQL database session for E2E tests.

    The conftest replaces app.db.session with a SQLite stub. We create a
    fresh MySQL engine and also patch app.db.session.SessionLocal so that
    any code inside the retrieval pipeline that creates its own session
    (e.g. _exact_search) also hits MySQL.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(_LIVE_MYSQL_URL, pool_pre_ping=True)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = Session()

    # Patch the conftest's SQLite stub so SessionLocal() returns MySQL sessions.
    with patch("app.db.session.SessionLocal", Session):
        yield db

    db.close()
    engine.dispose()


@pytest.fixture
def live_ctx(live_db):
    """ToolContext pointing at the live stack with KB 1."""
    from app.services.agentic_rag.tool_context import ToolContext
    ctx = MagicMock()
    ctx.db = live_db
    ctx.org_id = 1
    ctx.chat_id = None
    ctx.message_id = None
    ctx.user_id = 1
    ctx.state = {"kb_ids": [1]}
    return ctx


# ── kb_metadata E2E tests ──────────────────────────────────────────────────────

class TestKbMetadataE2E:
    def test_list_documents_returns_real_docs(self, live_ctx):
        from app.services.agentic_rag.tools.kb_metadata import KbMetadataTool, KbMetadataInput
        tool = KbMetadataTool()
        tool.ctx = live_ctx
        with patch_rbac():
            result = asyncio.run(tool._execute(KbMetadataInput(action="list_documents", limit=20)))
        assert result["ok"] is True
        docs = result["result"]["documents"]
        assert len(docs) >= 5, f"Expected >=5 docs, got {len(docs)}"
        # Each doc should have metadata fields
        for d in docs:
            assert "id" in d
            assert "title" in d
            assert "file_name" in d
            assert "file_created_at" in d or "file_modified_at" in d

    def test_unique_values_for_title(self, live_ctx):
        from app.services.agentic_rag.tools.kb_metadata import KbMetadataTool, KbMetadataInput
        tool = KbMetadataTool()
        tool.ctx = live_ctx
        with patch_rbac():
            result = asyncio.run(tool._execute(KbMetadataInput(action="unique_values", field="title", limit=50)))
        assert result["ok"] is True
        titles = result["result"]["values"]
        assert len(titles) >= 5, f"Expected >=5 titles, got {len(titles)}"

    def test_date_range_for_created_at(self, live_ctx):
        from app.services.agentic_rag.tools.kb_metadata import KbMetadataTool, KbMetadataInput
        tool = KbMetadataTool()
        tool.ctx = live_ctx
        with patch_rbac():
            result = asyncio.run(tool._execute(KbMetadataInput(action="date_range", field="created_at")))
        assert result["ok"] is True
        assert result["result"]["min"] is not None
        assert result["result"]["max"] is not None

    def test_unique_values_with_value_contains_filter(self, live_ctx):
        from app.services.agentic_rag.tools.kb_metadata import KbMetadataTool, KbMetadataInput
        tool = KbMetadataTool()
        tool.ctx = live_ctx
        with patch_rbac():
            result = asyncio.run(tool._execute(
                KbMetadataInput(action="unique_values", field="title", value_contains="Distributed", limit=50)
            ))
        assert result["ok"] is True
        titles = result["result"]["values"]
        assert all("Distributed" in t for t in titles), f"Expected all titles to contain 'Distributed', got {titles}"


# ── rag_retrieve filter E2E tests ──────────────────────────────────────────────

class TestRagRetrieveFilterE2E:
    def test_title_contains_filter_narrows_results(self, live_ctx):
        """Filtering by title_contains should return only docs from matching titles."""
        from app.services.agentic_rag.tools.rag_retrieve import _resolve_filter_to_doc_ids
        # Filter for "Distributed" — should match the DS 608 assignments
        doc_ids = _resolve_filter_to_doc_ids(live_ctx.db, [1], {"title_contains": "Distributed"})
        assert doc_ids is not None, "Filter should return a list, not None"
        assert len(doc_ids) >= 2, f"Expected >=2 docs with 'Distributed' in title, got {len(doc_ids)}"
        # Verify the doc_ids actually have "Distributed" in their titles
        from app.models.knowledge import Document
        for did in doc_ids:
            doc = live_ctx.db.query(Document).filter(Document.id == did).first()
            assert doc is not None
            assert "Distributed" in (doc.title or ""), f"Doc {did} title '{doc.title}' doesn't contain 'Distributed'"

    def test_content_type_filter_narrows_results(self, live_ctx):
        """Filtering by content_type should return only PDF docs."""
        from app.services.agentic_rag.tools.rag_retrieve import _resolve_filter_to_doc_ids
        doc_ids = _resolve_filter_to_doc_ids(live_ctx.db, [1], {"content_type": "application/pdf"})
        assert doc_ids is not None
        assert len(doc_ids) >= 3, f"Expected >=3 PDF docs, got {len(doc_ids)}"
        from app.models.knowledge import Document
        for did in doc_ids:
            doc = live_ctx.db.query(Document).filter(Document.id == did).first()
            assert doc.content_type == "application/pdf"

    def test_no_filters_returns_none(self, live_ctx):
        """No filters should return None (search all docs)."""
        from app.services.agentic_rag.tools.rag_retrieve import _resolve_filter_to_doc_ids
        assert _resolve_filter_to_doc_ids(live_ctx.db, [1], None) is None
        assert _resolve_filter_to_doc_ids(live_ctx.db, [1], {}) is None

    def test_nonexistent_filter_returns_empty(self, live_ctx):
        """A filter that matches nothing should return an empty list."""
        from app.services.agentic_rag.tools.rag_retrieve import _resolve_filter_to_doc_ids
        doc_ids = _resolve_filter_to_doc_ids(live_ctx.db, [1], {"title_contains": "XYZNonexistentTitle123"})
        assert doc_ids == []


# ── Qdrant filter E2E tests ────────────────────────────────────────────────────

class TestQdrantFilterE2E:
    def test_dense_search_with_doc_ids_filter(self, live_ctx):
        """Dense search with doc_ids should only return chunks from those docs."""
        from app.services.retrieval.retrieval import dense_search_docs
        # Get doc_ids for PDF docs only
        from app.models.knowledge import Document
        pdf_docs = live_ctx.db.query(Document).filter(
            Document.knowledge_base_id == 1,
            Document.content_type == "application/pdf"
        ).all()
        pdf_ids = [d.id for d in pdf_docs]
        assert len(pdf_ids) >= 3

        docs = dense_search_docs(
            query="network security",
            kb_ids=[1],
            datastore_ids=[],
            db=live_ctx.db,
            org_id=1,
            doc_ids=pdf_ids,
        )
        # All returned docs should have document_id in pdf_ids
        for d in docs:
            doc_id = d.metadata.get("document_id")
            assert doc_id in pdf_ids, f"Doc {doc_id} not in PDF ids {pdf_ids}"

    def test_exact_search_with_doc_ids_filter(self, live_ctx):
        """Exact search with doc_ids should only return chunks from those docs."""
        from app.services.retrieval.retrieval import exact_search_docs
        from app.models.knowledge import Document
        # Get doc_ids for the first 2 docs
        docs_db = live_ctx.db.query(Document).filter(
            Document.knowledge_base_id == 1
        ).limit(2).all()
        test_ids = [d.id for d in docs_db]

        docs = exact_search_docs(
            query="network",
            kb_ids=[1],
            datastore_ids=[],
            db=live_ctx.db,
            org_id=1,
            doc_ids=test_ids,
        )
        for d in docs:
            doc_id = d.metadata.get("document_id")
            assert doc_id in test_ids, f"Doc {doc_id} not in filtered ids {test_ids}"


# ── Sort E2E tests ─────────────────────────────────────────────────────────────

class TestSortE2E:
    def test_sort_by_file_modified_at_desc(self, live_ctx):
        """Sort by file_modified_at desc should put newest docs first."""
        from app.services.agentic_rag.tools.rag_retrieve import _sort_merged_docs
        from app.services.retrieval.retrieval import exact_search_docs
        # Use exact search (MySQL FTS) which doesn't need the embedding model
        docs = exact_search_docs(
            query="network",
            kb_ids=[1],
            datastore_ids=[],
            db=live_ctx.db,
            org_id=1,
        )
        if not docs:
            pytest.skip("No docs returned from exact search")
        # Convert to serialized format
        serialised = [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]
        sorted_docs = _sort_merged_docs(serialised, {"field": "file_modified_at", "direction": "desc"})
        # Verify sort order
        dates = [d["metadata"].get("_file_modified_at", "") for d in sorted_docs]
        assert dates == sorted(dates, reverse=True), f"Dates not in desc order: {dates}"


# ── Context verification E2E tests ─────────────────────────────────────────────

class TestContextVerificationE2E:
    def test_no_duplicate_content_in_filtered_results(self, live_ctx):
        """Filtered retrieval should not produce duplicate chunks in context."""
        from app.services.retrieval.retrieval import dense_search_docs
        from app.models.knowledge import Document
        pdf_ids = [d.id for d in live_ctx.db.query(Document).filter(
            Document.knowledge_base_id == 1,
            Document.content_type == "application/pdf"
        ).all()]

        docs = dense_search_docs(
            query="system",
            kb_ids=[1],
            datastore_ids=[],
            db=live_ctx.db,
            org_id=1,
            doc_ids=pdf_ids,
        )
        # Check for duplicate content
        contents = [d.page_content for d in docs]
        unique_contents = set(contents)
        assert len(contents) == len(unique_contents), \
            f"Found {len(contents) - len(unique_contents)} duplicate chunks in filtered results"

    def test_no_duplicate_content_hashes_in_filtered_results(self, live_ctx):
        """Filtered retrieval should not produce duplicate content hashes."""
        from app.services.retrieval.retrieval import dense_search_docs
        from app.services.agentic_rag.nodes import dedup_by_content_hash
        from app.models.knowledge import Document
        pdf_ids = [d.id for d in live_ctx.db.query(Document).filter(
            Document.knowledge_base_id == 1,
            Document.content_type == "application/pdf"
        ).all()]

        docs = dense_search_docs(
            query="monitoring",
            kb_ids=[1],
            datastore_ids=[],
            db=live_ctx.db,
            org_id=1,
            doc_ids=pdf_ids,
        )
        serialised = [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]
        deduped = dedup_by_content_hash(serialised)
        assert len(deduped) <= len(serialised), "Dedup should not increase count"
        # Verify no duplicate content hashes
        hashes = [d["metadata"].get("content_hash", d["page_content"][:100]) for d in deduped]
        assert len(hashes) == len(set(hashes)), "Duplicate content hashes after dedup"


# ── Helpers ────────────────────────────────────────────────────────────────────

def patch_rbac():
    """Patch enforce_rbac to allow KB 1."""
    from unittest.mock import patch
    def _allow(ctx, kb_ids=None, file_id=None):
        return {"kb_ids": kb_ids or [1], "file_id": file_id}
    return patch("app.services.agentic_rag.tools.kb_metadata.enforce_rbac", side_effect=_allow)
