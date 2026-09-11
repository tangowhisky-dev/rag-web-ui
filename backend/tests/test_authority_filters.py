"""Tests for authority-aware retrieval: document_status / effective-date
filters in resolve_filter_to_doc_ids, hit enrichment via
enrich_hits_with_authority, and authority markers in evidence headers.
"""
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.orm import sessionmaker

import app.db.session as _session_mod
from app.models.base import Base  # noqa
import app.models.datastore  # noqa: ensure tables registered
import app.models.knowledge  # noqa
from app.models.knowledge import Document
from app.services.agentic_rag.agent_graph.tooling import _hit_to_doc_dict
from app.services.agentic_rag.tools._search_helpers import (
    enrich_hits_with_authority,
    resolve_filter_to_doc_ids,
)
from app.services.agentic_rag.tools.keyword_search import KeywordSearchTool
from app.services.agentic_rag.utils import _format_doc_parts, _format_effective_window

engine = _session_mod.engine
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


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


def _make_document(db, kb_id=1, file_name="doc.pdf", document_status="active",
                   effective_from=None, effective_to=None, version="1",
                   owner=None):
    doc = Document(
        file_path=f"/tmp/{file_name}",
        file_name=file_name,
        file_size=100,
        content_type="application/pdf",
        file_hash=file_name,
        knowledge_base_id=kb_id,
        document_status=document_status,
        effective_from=effective_from if effective_from is not None else datetime(1970, 1, 1),
        effective_to=effective_to,
        version=version,
        owner=owner,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


# ── resolve_filter_to_doc_ids ────────────────────────────────────────────────

class TestResolveFilterToDocIds:
    def test_no_filters_returns_none(self, db):
        doc_ids, meta = resolve_filter_to_doc_ids(db, [1], None)
        assert doc_ids is None
        assert meta == {"total_matching": None, "ignored_keys": []}

    def test_document_status_single(self, db):
        active = _make_document(db, file_name="a.pdf", document_status="active")
        _make_document(db, file_name="b.pdf", document_status="superseded")
        doc_ids, meta = resolve_filter_to_doc_ids(db, [1], {"document_status": "active"})
        assert doc_ids == [active.id]
        assert meta["total_matching"] == 1

    def test_document_status_list(self, db):
        a = _make_document(db, file_name="a.pdf", document_status="active")
        b = _make_document(db, file_name="b.pdf", document_status="superseded")
        _make_document(db, file_name="c.pdf", document_status="draft")
        doc_ids, _ = resolve_filter_to_doc_ids(
            db, [1], {"document_status": ["active", "superseded"]})
        assert sorted(doc_ids) == sorted([a.id, b.id])

    def test_obsolete_alias_maps_to_superseded(self, db):
        s = _make_document(db, file_name="s.pdf", document_status="superseded")
        doc_ids, meta = resolve_filter_to_doc_ids(db, [1], {"document_status": "obsolete"})
        assert doc_ids == [s.id]
        assert meta["total_matching"] == 1

    def test_unknown_status_matches_zero(self, db):
        _make_document(db, file_name="a.pdf", document_status="active")
        doc_ids, meta = resolve_filter_to_doc_ids(db, [1], {"document_status": "bogus"})
        assert doc_ids == []
        assert meta["total_matching"] == 0

    def test_exclude_status(self, db):
        a = _make_document(db, file_name="a.pdf", document_status="active")
        _make_document(db, file_name="d.pdf", document_status="draft")
        doc_ids, _ = resolve_filter_to_doc_ids(db, [1], {"exclude_status": "draft"})
        assert doc_ids == [a.id]

    def test_effective_as_of(self, db):
        current = _make_document(
            db, file_name="cur.pdf",
            effective_from=datetime(2024, 1, 1), effective_to=None)
        _make_document(db, file_name="exp.pdf",
                       effective_from=datetime(2023, 1, 1),
                       effective_to=datetime(2024, 6, 1))
        _make_document(db, file_name="fut.pdf",
                       effective_from=datetime(2027, 1, 1), effective_to=None)
        doc_ids, meta = resolve_filter_to_doc_ids(
            db, [1], {"effective_as_of": "2025-03-01"})
        # Only `current` covers 2025-03-01: `exp` expired 2024-06-01 and
        # `fut` does not take effect until 2027-01-01.
        assert doc_ids == [current.id]
        assert meta["total_matching"] == 1

    def test_effective_window_overlap(self, db):
        overlapping = _make_document(
            db, file_name="ov.pdf",
            effective_from=datetime(2023, 1, 1), effective_to=datetime(2024, 3, 15))
        ongoing = _make_document(
            db, file_name="on.pdf",
            effective_from=datetime(2024, 2, 1), effective_to=None)
        _make_document(db, file_name="early.pdf",
                       effective_from=datetime(2022, 1, 1),
                       effective_to=datetime(2023, 12, 31))
        doc_ids, _ = resolve_filter_to_doc_ids(
            db, [1], {"effective_window_start": "2024-01-01",
                      "effective_window_end": "2024-12-31"})
        assert sorted(doc_ids) == sorted([overlapping.id, ongoing.id])

    def test_effective_to_before_excludes_ongoing(self, db):
        expiring = _make_document(
            db, file_name="e.pdf", effective_to=datetime(2025, 12, 31))
        _make_document(db, file_name="o.pdf", effective_to=None)
        doc_ids, _ = resolve_filter_to_doc_ids(
            db, [1], {"effective_to_before": "2026-06-01"})
        assert doc_ids == [expiring.id]

    def test_version_and_owner_filters(self, db):
        d = _make_document(db, file_name="v.pdf", version="2.1", owner="finance-ops")
        _make_document(db, file_name="w.pdf", version="1", owner="legal")
        doc_ids, _ = resolve_filter_to_doc_ids(
            db, [1], {"version": "2.1", "owner": "finance"})
        assert doc_ids == [d.id]

    def test_unparseable_date_is_reported_ignored(self, db):
        _make_document(db, file_name="a.pdf")
        doc_ids, meta = resolve_filter_to_doc_ids(
            db, [1], {"effective_as_of": "not-a-date"})
        # The only filter was unusable → unfiltered result + honest reporting.
        assert doc_ids is None
        assert meta["ignored_keys"] == ["effective_as_of"]

    def test_unknown_keys_are_reported_ignored(self, db):
        _make_document(db, file_name="a.pdf")
        doc_ids, meta = resolve_filter_to_doc_ids(
            db, [1], {"bogus_key": "x"})
        assert doc_ids is None
        assert meta["ignored_keys"] == ["bogus_key"]

    def test_mixed_known_and_unknown_keys(self, db):
        a = _make_document(db, file_name="a.pdf", document_status="active")
        _make_document(db, file_name="b.pdf", document_status="draft")
        doc_ids, meta = resolve_filter_to_doc_ids(
            db, [1], {"document_status": "active", "bogus": 1})
        assert doc_ids == [a.id]
        assert meta["ignored_keys"] == ["bogus"]
        assert meta["total_matching"] == 1

    def test_document_ids_within_filters(self, db):
        a = _make_document(db, file_name="a.pdf")
        _make_document(db, file_name="b.pdf")
        doc_ids, meta = resolve_filter_to_doc_ids(
            db, [1], {"document_ids": [a.id]})
        assert doc_ids == [a.id]
        assert meta["total_matching"] == 1

    def test_no_cap_and_total_matching(self, db):
        for i in range(5):
            _make_document(db, file_name=f"d{i}.pdf")
        doc_ids, meta = resolve_filter_to_doc_ids(
            db, [1], {"document_status": "active"})
        assert len(doc_ids) == 5
        assert meta["total_matching"] == 5

    def test_scoped_to_kb(self, db):
        in_kb = _make_document(db, kb_id=1, file_name="in.pdf")
        _make_document(db, kb_id=2, file_name="out.pdf")
        doc_ids, _ = resolve_filter_to_doc_ids(db, [1], {"document_status": "active"})
        assert doc_ids == [in_kb.id]

    def test_scalar_document_ids_normalized(self, db):
        a = _make_document(db, file_name="a.pdf")
        doc_ids, meta = resolve_filter_to_doc_ids(
            db, [1], {"document_ids": a.id})
        assert doc_ids == [a.id]
        assert meta["total_matching"] == 1

    def test_nonnumeric_document_ids_ignored(self, db):
        _make_document(db, file_name="a.pdf")
        doc_ids, meta = resolve_filter_to_doc_ids(
            db, [1], {"document_ids": "abc"})
        assert doc_ids is None
        assert meta["ignored_keys"] == ["document_ids"]

    def test_falsy_numeric_value_is_ignored(self, db):
        _make_document(db, file_name="a.pdf")
        doc_ids, meta = resolve_filter_to_doc_ids(
            db, [1], {"document_ids": 0})
        assert doc_ids is None
        assert meta["ignored_keys"] == ["document_ids"]


# ── enrich_hits_with_authority ───────────────────────────────────────────────

class TestEnrichHitsWithAuthority:
    def test_stamps_fields_on_matching_hits(self, db):
        doc = _make_document(
            db, file_name="s.pdf", document_status="superseded",
            effective_from=datetime(2023, 1, 1), effective_to=datetime(2024, 6, 1),
            version="2.0")
        hits = [{"document_id": doc.id, "content": "x"}]
        out = enrich_hits_with_authority(hits, db)
        assert out[0]["document_status"] == "superseded"
        assert out[0]["effective_from"].startswith("2023-01-01")
        assert out[0]["effective_to"].startswith("2024-06-01")
        assert out[0]["version"] == "2.0"

    def test_null_effective_to_becomes_empty(self, db):
        doc = _make_document(db, file_name="a.pdf", effective_to=None)
        hits = enrich_hits_with_authority([{"document_id": doc.id}], db)
        assert hits[0]["effective_to"] == ""

    def test_hit_without_document_id_untouched(self, db):
        hits = [{"document_id": None, "content": "x"}, {"content": "y"}]
        out = enrich_hits_with_authority(hits, db)
        assert "document_status" not in out[0]
        assert "document_status" not in out[1]

    def test_hit_for_deleted_document_untouched(self, db):
        out = enrich_hits_with_authority([{"document_id": 9999}], db)
        assert "document_status" not in out[0]

    def test_empty_and_no_db_passthrough(self):
        assert enrich_hits_with_authority([], None) == []
        hits = [{"document_id": 1}]
        assert enrich_hits_with_authority(hits, None) is hits

    def test_db_error_degrades_gracefully(self):
        bad_db = MagicMock()
        bad_db.query.side_effect = RuntimeError("db down")
        hits = [{"document_id": 1}]
        assert enrich_hits_with_authority(hits, bad_db) is hits


# ── _hit_to_doc_dict propagation ─────────────────────────────────────────────

class TestHitToDocDict:
    def test_authority_fields_copied(self):
        hit = {
            "document_id": 7, "chunk_index": 2, "content": "text",
            "document_status": "superseded", "effective_from": "2023-01-01",
            "effective_to": "2024-06-01", "version": "2.1",
        }
        meta = _hit_to_doc_dict(hit)["metadata"]
        assert meta["document_status"] == "superseded"
        assert meta["effective_from"] == "2023-01-01"
        assert meta["effective_to"] == "2024-06-01"
        assert meta["version"] == "2.1"

    def test_absent_fields_not_stamped(self):
        meta = _hit_to_doc_dict({"document_id": 1, "content": "x"})["metadata"]
        assert "document_status" not in meta
        assert "effective_from" not in meta


# ── _format_effective_window / evidence headers ──────────────────────────────

class TestEffectiveWindowFormatting:
    def test_full_window(self):
        assert _format_effective_window(
            {"effective_from": "2024-01-01", "effective_to": "2025-06-01"}
        ) == "2024-01-01..2025-06-01"

    def test_open_end(self):
        assert _format_effective_window(
            {"effective_from": "2024-01-01", "effective_to": ""}
        ) == "2024-01-01.."

    def test_sentinel_start_is_open(self):
        assert _format_effective_window(
            {"effective_from": "1970-01-01", "effective_to": "2025-06-01"}
        ) == "..2025-06-01"

    def test_both_open_returns_empty(self):
        assert _format_effective_window(
            {"effective_from": "1970-01-01", "effective_to": None}) == ""
        assert _format_effective_window({}) == ""


class TestFormatDocParts:
    def _doc(self, **meta):
        return {"page_content": "chunk text", "metadata": meta}

    def test_superseded_chunk_header_shows_status_and_window(self):
        doc = self._doc(
            title="Policy", document_status="superseded",
            effective_from="2023-01-01", effective_to="2024-06-01",
            version="1",
            citation_ref={"citation_id": "E1", "citation_kind": "chunk",
                          "chunk_index": 3, "source_tool": "keyword_search"},
        )
        out = _format_doc_parts([doc], None)[0]
        assert 'status=superseded' in out
        assert 'effective=2023-01-01..2024-06-01' in out
        assert 'version=' not in out  # default version is noise

    def test_active_doc_shows_no_status(self):
        doc = self._doc(
            title="Policy", document_status="active",
            effective_from="1970-01-01", effective_to="", version="1",
            citation_ref={"citation_id": "E1", "citation_kind": "chunk"},
        )
        out = _format_doc_parts([doc], None)[0]
        assert 'status=' not in out
        assert 'effective=' not in out

    def test_nondefault_version_shown(self):
        doc = self._doc(
            title="Policy", document_status="active", version="2.1",
            citation_ref={"citation_id": "E1", "citation_kind": "chunk"},
        )
        assert 'version=2.1' in _format_doc_parts([doc], None)[0]

    def test_legacy_branch_shows_markers(self):
        # title_search metadata_only docs land here (no citation_ref).
        doc = self._doc(title="Old Policy", source="title_search",
                        document_status="draft",
                        effective_from="2026-01-01", effective_to="")
        out = _format_doc_parts([doc], None)[0]
        assert 'status=draft' in out
        assert 'effective=2026-01-01..' in out


# ── tool-level integration (keyword_search) ─────────────────────────────────

class TestKeywordSearchAuthority:
    def _ctx(self, db):
        ctx = MagicMock()
        ctx.org_id = 1
        ctx.db = db
        ctx.state = {"kb_ids": [1]}
        ctx.chat_id = 1
        ctx.message_id = 1
        return ctx

    @patch("app.services.agentic_rag.tools.keyword_search.sparse_search_docs")
    @patch("app.services.agentic_rag.tools.keyword_search.exact_search_docs")
    @patch("app.services.agentic_rag.tools.keyword_search.expand_synonyms", new_callable=AsyncMock)
    @patch("app.services.agentic_rag.tools.keyword_search.enforce_rbac")
    @patch("app.services.agentic_rag.tools.keyword_search.get_effective_datastore_ids")
    @patch("app.services.agentic_rag.tools.keyword_search.get_setting")
    def test_hits_carry_authority_fields(
            self, mock_setting, mock_ds, mock_rbac, mock_syn, mock_exact, mock_sparse, db):
        from langchain_core.documents import Document as LcDocument
        active = _make_document(db, file_name="a.pdf", document_status="active")
        sup = _make_document(db, file_name="s.pdf", document_status="superseded",
                             effective_to=datetime(2024, 1, 1))
        mock_setting.return_value = 0.0
        mock_ds.return_value = []
        mock_rbac.return_value = {"kb_ids": [1]}
        mock_syn.return_value = ("test", [])
        mock_exact.return_value = [
            LcDocument(page_content="hit a", metadata={
                "document_id": active.id, "chunk_index": 0, "content_hash": "ha",
                "score": 0.9}),
            LcDocument(page_content="hit s", metadata={
                "document_id": sup.id, "chunk_index": 0, "content_hash": "hs",
                "score": 0.8}),
        ]
        mock_sparse.return_value = []

        tool = KeywordSearchTool()
        tool.ctx = self._ctx(db)
        result = asyncio.run(tool.arun({"query": "test", "kb_ids": [1]}))

        assert result["ok"] is True
        hits = {h["document_id"]: h for h in result["result"]["hits"]}
        assert hits[active.id]["document_status"] == "active"
        assert hits[sup.id]["document_status"] == "superseded"
        assert hits[sup.id]["effective_to"].startswith("2024-01-01")

    @patch("app.services.agentic_rag.tools.keyword_search.sparse_search_docs")
    @patch("app.services.agentic_rag.tools.keyword_search.exact_search_docs")
    @patch("app.services.agentic_rag.tools.keyword_search.expand_synonyms", new_callable=AsyncMock)
    @patch("app.services.agentic_rag.tools.keyword_search.enforce_rbac")
    @patch("app.services.agentic_rag.tools.keyword_search.get_effective_datastore_ids")
    @patch("app.services.agentic_rag.tools.keyword_search.get_setting")
    def test_status_filter_narrows_doc_ids(
            self, mock_setting, mock_ds, mock_rbac, mock_syn, mock_exact, mock_sparse, db):
        active = _make_document(db, file_name="a.pdf", document_status="active")
        _make_document(db, file_name="s.pdf", document_status="superseded")
        mock_setting.return_value = 0.0
        mock_ds.return_value = []
        mock_rbac.return_value = {"kb_ids": [1]}
        mock_syn.return_value = ("test", [])
        mock_exact.return_value = []
        mock_sparse.return_value = []

        tool = KeywordSearchTool()
        tool.ctx = self._ctx(db)
        result = asyncio.run(tool.arun({
            "query": "test", "kb_ids": [1],
            "filters": {"document_status": "active"},
        }))

        assert result["ok"] is True
        # The search legs were scoped to just the active document.
        assert mock_exact.call_args.kwargs["doc_ids"] == [active.id]
        assert mock_sparse.call_args.kwargs["doc_ids"] == [active.id]
        assert result["result"]["matched_documents"] == 1

    @patch("app.services.agentic_rag.tools.keyword_search.sparse_search_docs")
    @patch("app.services.agentic_rag.tools.keyword_search.exact_search_docs")
    @patch("app.services.agentic_rag.tools.keyword_search.expand_synonyms", new_callable=AsyncMock)
    @patch("app.services.agentic_rag.tools.keyword_search.enforce_rbac")
    @patch("app.services.agentic_rag.tools.keyword_search.get_effective_datastore_ids")
    @patch("app.services.agentic_rag.tools.keyword_search.get_setting")
    def test_zero_match_filter_returns_no_hits(
            self, mock_setting, mock_ds, mock_rbac, mock_syn, mock_exact, mock_sparse, db):
        _make_document(db, file_name="a.pdf", document_status="active")
        mock_setting.return_value = 0.0
        mock_ds.return_value = []
        mock_rbac.return_value = {"kb_ids": [1]}

        tool = KeywordSearchTool()
        tool.ctx = self._ctx(db)
        result = asyncio.run(tool.arun({
            "query": "test", "kb_ids": [1],
            "filters": {"document_status": "superseded"},
        }))

        assert result["ok"] is True
        assert result["result"]["hits"] == []
        assert result["result"]["matched_documents"] == 0
        # Search legs must not run when the filter matched nothing.
        mock_exact.assert_not_called()
        mock_sparse.assert_not_called()
        mock_syn.assert_not_called()

    @patch("app.services.agentic_rag.tools.keyword_search.sparse_search_docs")
    @patch("app.services.agentic_rag.tools.keyword_search.exact_search_docs")
    @patch("app.services.agentic_rag.tools.keyword_search.expand_synonyms", new_callable=AsyncMock)
    @patch("app.services.agentic_rag.tools.keyword_search.enforce_rbac")
    @patch("app.services.agentic_rag.tools.keyword_search.get_effective_datastore_ids")
    @patch("app.services.agentic_rag.tools.keyword_search.get_setting")
    def test_document_ids_and_filters_intersect(
            self, mock_setting, mock_ds, mock_rbac, mock_syn, mock_exact, mock_sparse, db):
        a = _make_document(db, file_name="a.pdf", document_status="active")
        s = _make_document(db, file_name="s.pdf", document_status="superseded")
        mock_setting.return_value = 0.0
        mock_ds.return_value = []
        mock_rbac.return_value = {"kb_ids": [1]}
        mock_syn.return_value = ("test", [])
        mock_exact.return_value = []
        mock_sparse.return_value = []

        tool = KeywordSearchTool()
        tool.ctx = self._ctx(db)
        # document_ids allows both, but filter keeps only superseded → only s survives.
        result = asyncio.run(tool.arun({
            "query": "test", "kb_ids": [1],
            "document_ids": [a.id, s.id],
            "filters": {"document_status": "superseded"},
        }))

        assert result["ok"] is True
        assert mock_exact.call_args.kwargs["doc_ids"] == [s.id]

    @patch("app.services.agentic_rag.tools.keyword_search.sparse_search_docs")
    @patch("app.services.agentic_rag.tools.keyword_search.exact_search_docs")
    @patch("app.services.agentic_rag.tools.keyword_search.expand_synonyms", new_callable=AsyncMock)
    @patch("app.services.agentic_rag.tools.keyword_search.enforce_rbac")
    @patch("app.services.agentic_rag.tools.keyword_search.get_effective_datastore_ids")
    @patch("app.services.agentic_rag.tools.keyword_search.get_setting")
    def test_unknown_filter_key_runs_unfiltered_and_reports(
            self, mock_setting, mock_ds, mock_rbac, mock_syn, mock_exact, mock_sparse, db):
        _make_document(db, file_name="a.pdf")
        mock_setting.return_value = 0.0
        mock_ds.return_value = []
        mock_rbac.return_value = {"kb_ids": [1]}
        mock_syn.return_value = ("test", [])
        mock_exact.return_value = []
        mock_sparse.return_value = []

        tool = KeywordSearchTool()
        tool.ctx = self._ctx(db)
        result = asyncio.run(tool.arun({
            "query": "test", "kb_ids": [1],
            "filters": {"bogus_key": "x"},
        }))

        assert result["ok"] is True
        # Unrecognized filter must not silently "succeed" — run unfiltered
        # and tell the agent which keys were ignored.
        assert mock_exact.call_args.kwargs["doc_ids"] is None
        assert result["result"]["ignored_filter_keys"] == ["bogus_key"]

    def test_prepare_arguments_parses_stringified_filters(self):
        tool = KeywordSearchTool()
        args = tool.prepare_arguments({
            "query": "x", "kb_ids": [1],
            "filters": '{"document_status": "active"}',
            "document_ids": "5",
        })
        assert args["filters"] == {"document_status": "active"}
        assert args["document_ids"] == [5]
        # Malformed filters string is left for schema validation.
        args = tool.prepare_arguments({"query": "x", "kb_ids": [1], "filters": "{bad"})
        assert args["filters"] == "{bad"


# ── observation surfacing ────────────────────────────────────────────────────

class TestObservationSurfacing:
    def test_matched_documents_and_ignored_keys_reach_think_node(self):
        from app.services.agentic_rag.agent_graph.observations import (
            _observations_metadata_text)
        from app.services.agentic_rag.schemas import Observation
        obs = [Observation(
            tool="keyword_search", arguments={"query": "x"},
            result={"hits": [{"document_id": 1, "document_status": "superseded",
                              "score": 0.5}],
                    "search_type": "keyword",
                    "matched_documents": 3,
                    "ignored_filter_keys": ["bogus"]},
            error=None, tokens=0)]
        text = _observations_metadata_text(obs)
        assert "matched_documents=3" in text
        assert "ignored_filter_keys=['bogus']" in text
        assert "status_counts={'superseded': 1}" in text

    def test_all_active_hits_hide_status_counts(self):
        from app.services.agentic_rag.agent_graph.observations import (
            _observations_metadata_text)
        from app.services.agentic_rag.schemas import Observation
        obs = [Observation(
            tool="keyword_search", arguments={"query": "x"},
            result={"hits": [{"document_id": 1, "document_status": "active",
                              "score": 0.5}],
                    "search_type": "keyword"},
            error=None, tokens=0)]
        text = _observations_metadata_text(obs)
        assert "status_counts" not in text
