"""Tests for inject_neighbor_context: prev/next chunk injection, dedup
against already-retrieved chunks, and hit-flag propagation.
"""
from unittest.mock import MagicMock

from sqlalchemy.orm import sessionmaker

import app.db.session as _session_mod
from app.models.base import Base  # noqa
import app.models.datastore  # noqa: ensure tables registered
import app.models.knowledge  # noqa
from app.models.knowledge import Document, DocumentChunk
from app.services.agentic_rag.tools._search_helpers import inject_neighbor_context

engine = _session_mod.engine
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


import pytest


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


def _seed_doc_with_chunks(db, n_chunks=5):
    doc = Document(
        file_path="/tmp/handbook.txt",
        file_name="handbook.txt",
        file_size=100,
        content_type="text/plain",
        file_hash="handbook.txt",
        knowledge_base_id=1,
        document_status="active",
    )
    db.add(doc)
    db.flush()
    for i in range(n_chunks):
        db.add(DocumentChunk(
            id=f"hash-{doc.id}-{i}",
            document_id=doc.id,
            file_name=doc.file_name,
            chunk_text=f"chunk {i} body text",
            chunk_index=i,
            chunk_metadata={"title": "Handbook", "document_id": doc.id},
            hash=f"hash-{doc.id}-{i}",
        ))
    db.commit()
    return doc


def _ldoc(content, document_id, chunk_index, **extra):
    m = MagicMock()
    m.metadata = {"document_id": document_id, "chunk_index": chunk_index, **extra}
    m.page_content = content
    return m


class TestNeighborInjection:

    def test_injects_prev_and_next(self, db):
        doc = _seed_doc_with_chunks(db, 5)
        hits = [_ldoc("chunk 2 body text", doc.id, 2)]
        out = inject_neighbor_context(hits, db, top_n=5, window=1)
        idx = [d.metadata.get("chunk_index") for d in out]
        assert idx == [2, 1, 3]
        # Injected chunks carry the neighbor flag + citation ref
        assert out[1].metadata["_is_neighbor"] is True
        assert out[2].metadata["_is_neighbor"] is True
        assert out[1].metadata["citation_ref"]["document_id"] == doc.id

    def test_does_not_reinject_existing_chunks(self, db):
        doc = _seed_doc_with_chunks(db, 5)
        # chunks 2 and 3 already retrieved — only chunk 1 should inject.
        hits = [_ldoc("chunk 2 body text", doc.id, 2),
                _ldoc("chunk 3 body text", doc.id, 3)]
        out = inject_neighbor_context(hits, db, top_n=5, window=1)
        idx = [d.metadata.get("chunk_index") for d in out]
        assert idx == [2, 3, 1, 4]
        # no duplicate (doc_id, chunk_index) pairs
        pairs = [(d.metadata["document_id"], d.metadata["chunk_index"]) for d in out]
        assert len(pairs) == len(set(pairs))

    def test_boundary_chunk_only_fetches_valid_indices(self, db):
        doc = _seed_doc_with_chunks(db, 3)
        hits = [_ldoc("chunk 0 body text", doc.id, 0)]
        out = inject_neighbor_context(hits, db, top_n=5, window=1)
        idx = [d.metadata.get("chunk_index") for d in out]
        assert idx == [0, 1]  # no chunk -1

    def test_top_n_limits_injection(self, db):
        doc = _seed_doc_with_chunks(db, 6)
        hits = [_ldoc(f"chunk {i} body text", doc.id, i) for i in (2, 4)]
        out = inject_neighbor_context(hits, db, top_n=1, window=1)
        idx = sorted(d.metadata.get("chunk_index") for d in out)
        # only the first hit (chunk 2) gets neighbors 1,3 — chunk 4 untouched
        assert idx == [1, 2, 3, 4]
