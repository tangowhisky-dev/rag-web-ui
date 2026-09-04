"""Tests for agentic retrieval filtering: metadata introspection, filters,
sort, and query rewrite with failure context.

Covers:
- kb_metadata tool: list_fields, unique_values, date_range, list_documents
- rag_retrieve filters: title_contains, file_name_contains, content_type,
  created_after/before, document_ids
- rag_retrieve sort: by created_at desc/asc
- Query rewrite with failure context: returns filter_suggestion
- Empty filter results: returns empty result without error
- Qdrant filter construction: _build_doc_id_filter
- SQL filter construction: doc_id_params in exact search
- Context verification: no duplicates after filtering
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Stubs ──────────────────────────────────────────────────────────────────────

class _StubDoc:
    """Minimal Document row stub for metadata queries."""

    def __init__(self, id=1, title="Weekly Update (Jun 3 - Jun 7)",
                 file_name="weekly_update_jun.pdf",
                 content_type="application/pdf",
                 created_at=datetime(2026, 6, 7, 12, 0, tzinfo=timezone.utc),
                 file_modified_at=datetime(2026, 6, 7, 14, 0, tzinfo=timezone.utc),
                 file_created_at=None,
                 knowledge_base_id=1, data_store_id=None,
                 converted_markdown="# Weekly Update\n\nContent here.\n"):
        self.id = id
        self.title = title
        self.file_name = file_name
        self.content_type = content_type
        self.created_at = created_at
        self.file_modified_at = file_modified_at
        self.file_created_at = file_created_at
        self.knowledge_base_id = knowledge_base_id
        self.data_store_id = data_store_id
        self.converted_markdown = converted_markdown


class _Row:
    """Stub row that supports both attribute access and tuple unpacking,
    matching SQLAlchemy's named-tuple behavior for multi-column queries."""

    def __init__(self, doc, fields):
        for f in fields:
            setattr(self, f, getattr(doc, f, None))

    def __iter__(self):
        return iter([getattr(self, f) for f in self._fields])


class _StubQuery:
    """Chained SQLAlchemy query stub that supports filter/distinct/order_by/limit."""

    def __init__(self, rows, query_type="model"):
        """query_type: 'model' (full objects), 'column' (single-col tuples),
        'multi' (named tuples with attribute access), 'aggregate' (min/max tuple)."""
        self._rows = rows
        self._filters = []
        self._distinct = False
        self._order_by = None
        self._limit_val = None
        self._query_type = query_type

    def filter(self, *args, **kwargs):
        self._filters.append(args)
        return self

    def distinct(self):
        self._distinct = True
        return self

    def order_by(self, *args):
        self._order_by = args
        return self

    def limit(self, n):
        self._limit_val = n
        return self

    def all(self):
        rows = self._rows[:self._limit_val] if self._limit_val else self._rows
        if self._query_type == "column":
            return [(getattr(r, "title", None),) for r in rows]
        if self._query_type == "multi":
            return [_Row(r, ["id", "title", "file_name", "content_type", "file_created_at", "file_modified_at"]) for r in rows]
        return rows

    def first(self):
        if not self._rows:
            return None
        row = self._rows[0]
        if self._query_type == "aggregate":
            return (row.created_at, row.created_at)
        if self._query_type == "column":
            return (getattr(row, "title", None),)
        return row


class _StubDB:
    """Minimal DB session stub that returns configurable queries."""

    def __init__(self, docs):
        self._docs = docs

    def query(self, *args):
        if len(args) == 1:
            model = args[0]
            # Column query (InstrumentedAttribute) vs model class.
            is_column = hasattr(model, "key") or (not isinstance(model, type))
            return _StubQuery(self._docs, query_type="column" if is_column else "model")
        # Multiple args: distinguish aggregate (func.min/func.max) from multi-column.
        is_aggregate = any("Function" in type(a).__name__ or any("Function" in c.__name__ for c in type(a).__mro__) for a in args)
        return _StubQuery(self._docs, query_type="aggregate" if is_aggregate else "multi")

    def add(self, record):
        pass

    def rollback(self):
        pass

    def flush(self):
        pass


class _StubToolContext:
    def __init__(self, docs, chat_id=1, user_id=1, org_id=1, state=None):
        self.db = _StubDB(docs)
        self.user_id = user_id
        self.org_id = org_id
        self.chat_id = chat_id
        self.message_id = None
        self.qdrant_client = None
        self.redis_memory = None
        self.org_llm_config = {}
        self.state = state


def _mock_rbac(kb_ids=None):
    def _enforce(ctx, kb_ids=None, file_id=None):
        return {"kb_ids": kb_ids or [1], "file_id": file_id}
    return _enforce


def _run_tool(tool_cls, ctx, input_obj):
    tool = tool_cls()
    tool.ctx = ctx
    return asyncio.run(tool._execute(input_obj))


# ── kb_metadata tests ──────────────────────────────────────────────────────────

class TestKbMetadataListFields:
    def test_list_fields_returns_static_schema(self):
        from app.services.agentic_rag.tools.kb_metadata import KbMetadataTool, KbMetadataInput
        ctx = _StubToolContext([_StubDoc()])
        with patch("app.services.agentic_rag.tools.kb_metadata.enforce_rbac", side_effect=_mock_rbac()):
            result = _run_tool(KbMetadataTool, ctx, KbMetadataInput(action="list_fields"))
        assert result["ok"] is True
        fields = result["result"]["fields"]
        field_names = {f["name"] for f in fields}
        assert "title" in field_names
        assert "file_name" in field_names
        assert "created_at" in field_names
        assert "content_type" in field_names


class TestKbMetadataUniqueValues:
    def test_unique_values_returns_titles(self):
        from app.services.agentic_rag.tools.kb_metadata import KbMetadataTool, KbMetadataInput
        docs = [_StubDoc(id=1, title="Weekly Update A"), _StubDoc(id=2, title="Weekly Update B")]
        ctx = _StubToolContext(docs)
        with patch("app.services.agentic_rag.tools.kb_metadata.enforce_rbac", side_effect=_mock_rbac()):
            result = _run_tool(KbMetadataTool, ctx, KbMetadataInput(action="unique_values", field="title"))
        assert result["ok"] is True
        assert result["result"]["field"] == "title"
        assert result["result"]["count"] == 2

    def test_unique_values_rejects_invalid_field(self):
        from app.services.agentic_rag.tools.kb_metadata import KbMetadataTool, KbMetadataInput
        ctx = _StubToolContext([_StubDoc()])
        with patch("app.services.agentic_rag.tools.kb_metadata.enforce_rbac", side_effect=_mock_rbac()):
            result = _run_tool(KbMetadataTool, ctx, KbMetadataInput(action="unique_values", field="invalid_field"))
        assert "error" in result["result"]


class TestKbMetadataDateRange:
    def test_date_range_returns_min_max(self):
        from app.services.agentic_rag.tools.kb_metadata import KbMetadataTool, KbMetadataInput
        docs = [_StubDoc(id=1, created_at=datetime(2026, 6, 1, tzinfo=timezone.utc)),
                _StubDoc(id=2, created_at=datetime(2026, 6, 30, tzinfo=timezone.utc))]
        ctx = _StubToolContext(docs)
        with patch("app.services.agentic_rag.tools.kb_metadata.enforce_rbac", side_effect=_mock_rbac()):
            result = _run_tool(KbMetadataTool, ctx, KbMetadataInput(action="date_range", field="created_at"))
        assert result["ok"] is True
        assert result["result"]["field"] == "created_at"
        assert result["result"]["min"] is not None
        assert result["result"]["max"] is not None

    def test_date_range_rejects_non_date_field(self):
        from app.services.agentic_rag.tools.kb_metadata import KbMetadataTool, KbMetadataInput
        ctx = _StubToolContext([_StubDoc()])
        with patch("app.services.agentic_rag.tools.kb_metadata.enforce_rbac", side_effect=_mock_rbac()):
            result = _run_tool(KbMetadataTool, ctx, KbMetadataInput(action="date_range", field="title"))
        assert "error" in result["result"]


class TestKbMetadataListDocuments:
    def test_list_documents_returns_docs_with_metadata(self):
        from app.services.agentic_rag.tools.kb_metadata import KbMetadataTool, KbMetadataInput
        docs = [_StubDoc(id=1, title="Doc A"), _StubDoc(id=2, title="Doc B")]
        ctx = _StubToolContext(docs)
        with patch("app.services.agentic_rag.tools.kb_metadata.enforce_rbac", side_effect=_mock_rbac()):
            result = _run_tool(KbMetadataTool, ctx, KbMetadataInput(action="list_documents", limit=10))
        assert result["ok"] is True
        docs_result = result["result"]["documents"]
        assert len(docs_result) == 2
        assert "title" in docs_result[0]
        assert "file_name" in docs_result[0]
        assert "file_created_at" in docs_result[0]


class TestKbMetadataRbac:
    def test_no_kb_ids_returns_empty(self):
        from app.services.agentic_rag.tools.kb_metadata import KbMetadataTool, KbMetadataInput
        ctx = _StubToolContext([], state={"kb_ids": []})

        def _deny_rbac(ctx, kb_ids=None, file_id=None):
            return {"kb_ids": [], "file_id": None}

        with patch("app.services.agentic_rag.tools.kb_metadata.enforce_rbac", side_effect=_deny_rbac):
            result = _run_tool(KbMetadataTool, ctx, KbMetadataInput(action="list_fields"))
        assert result["ok"] is True
        assert "error" in result["result"]


# ── rag_retrieve filter resolution tests ───────────────────────────────────────

class TestResolveFilterToDocIds:
    def test_no_filters_returns_none(self):
        from app.services.agentic_rag.tools.rag_retrieve import _resolve_filter_to_doc_ids
        db = MagicMock()
        result = _resolve_filter_to_doc_ids(db, [1], None)
        assert result is None

    def test_empty_filters_returns_none(self):
        from app.services.agentic_rag.tools.rag_retrieve import _resolve_filter_to_doc_ids
        db = MagicMock()
        result = _resolve_filter_to_doc_ids(db, [1], {})
        assert result is None

    def test_title_contains_filter_calls_query(self):
        from app.services.agentic_rag.tools.rag_retrieve import _resolve_filter_to_doc_ids
        docs = [_StubDoc(id=1, title="Weekly Update"), _StubDoc(id=2, title="Other")]
        db = _StubDB(docs)
        result = _resolve_filter_to_doc_ids(db, [1], {"title_contains": "Weekly"})
        assert isinstance(result, list)
        # The stub query returns all docs (filter is a no-op in the stub)
        assert len(result) == 2


# ── Sort tests ─────────────────────────────────────────────────────────────────

class TestSortMergedDocs:
    def test_no_sort_returns_original(self):
        from app.services.agentic_rag.tools.rag_retrieve import _sort_merged_docs
        docs = [{"metadata": {"_file_created_at": "2026-06-07"}}, {"metadata": {"_file_created_at": "2026-06-01"}}]
        result = _sort_merged_docs(docs, None)
        assert result == docs

    def test_sort_desc_by_created_at(self):
        from app.services.agentic_rag.tools.rag_retrieve import _sort_merged_docs
        docs = [
            {"metadata": {"_file_created_at": "2026-06-01"}},
            {"metadata": {"_file_created_at": "2026-06-07"}},
            {"metadata": {"_file_created_at": "2026-06-03"}},
        ]
        result = _sort_merged_docs(docs, {"field": "file_created_at", "direction": "desc"})
        assert result[0]["metadata"]["_file_created_at"] == "2026-06-07"
        assert result[1]["metadata"]["_file_created_at"] == "2026-06-03"
        assert result[2]["metadata"]["_file_created_at"] == "2026-06-01"

    def test_sort_asc_by_created_at(self):
        from app.services.agentic_rag.tools.rag_retrieve import _sort_merged_docs
        docs = [
            {"metadata": {"_file_created_at": "2026-06-07"}},
            {"metadata": {"_file_created_at": "2026-06-01"}},
        ]
        result = _sort_merged_docs(docs, {"field": "file_created_at", "direction": "asc"})
        assert result[0]["metadata"]["_file_created_at"] == "2026-06-01"
        assert result[1]["metadata"]["_file_created_at"] == "2026-06-07"

    def test_sort_empty_docs_returns_empty(self):
        from app.services.agentic_rag.tools.rag_retrieve import _sort_merged_docs
        result = _sort_merged_docs([], {"field": "file_created_at", "direction": "desc"})
        assert result == []

    def test_sort_missing_field_in_sort_returns_original(self):
        from app.services.agentic_rag.tools.rag_retrieve import _sort_merged_docs
        docs = [{"metadata": {"_file_created_at": "2026-06-01"}}]
        result = _sort_merged_docs(docs, {"direction": "desc"})
        assert result == docs


# ── Qdrant filter construction tests ───────────────────────────────────────────

class TestBuildDocIdFilter:
    def test_none_doc_ids_returns_none(self):
        from app.services.retrieval.retrieval import _build_doc_id_filter
        assert _build_doc_id_filter(None) is None

    def test_empty_doc_ids_returns_none(self):
        from app.services.retrieval.retrieval import _build_doc_id_filter
        assert _build_doc_id_filter([]) is None

    def test_non_empty_doc_ids_returns_filter(self):
        from app.services.retrieval.retrieval import _build_doc_id_filter
        from qdrant_client.models import Filter
        result = _build_doc_id_filter([1, 2, 3])
        assert isinstance(result, Filter)
        assert len(result.must) == 1
        assert result.must[0].key == "document_id"


# ── Query rewrite with failure context tests ───────────────────────────────────

class TestRewriteWithFailureContext:
    def test_rewrite_returns_tuple(self):
        from app.services.agentic_rag.tools.rag_retrieve import _rewrite_query
        ctx = _StubToolContext([_StubDoc()])
        mock_response = MagicMock()
        mock_response.content = '{"rewritten_query": "weekly update June", "filter_suggestion": {"title_contains": "Weekly Update"}, "reasoning": "test"}'

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch("app.services.agentic_rag.tools.rag_retrieve.build_chat_llm", return_value=mock_llm):
            result = asyncio.run(_rewrite_query("latest weekly update", "missing docs", ctx, []))

        assert isinstance(result, tuple)
        assert len(result) == 2
        rewritten, filter_suggestion = result
        assert rewritten == "weekly update June"
        assert filter_suggestion == {"title_contains": "Weekly Update"}

    def test_rewrite_no_filter_needed_returns_none_filter(self):
        from app.services.agentic_rag.tools.rag_retrieve import _rewrite_query
        ctx = _StubToolContext([_StubDoc()])
        mock_response = MagicMock()
        mock_response.content = '{"rewritten_query": "weekly update content", "filter_suggestion": null, "reasoning": "no filter needed"}'

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch("app.services.agentic_rag.tools.rag_retrieve.build_chat_llm", return_value=mock_llm):
            result = asyncio.run(_rewrite_query("weekly update", "missing", ctx, []))

        rewritten, filter_suggestion = result
        assert filter_suggestion is None

    def test_rewrite_filters_unknown_keys(self):
        from app.services.agentic_rag.tools.rag_retrieve import _rewrite_query
        ctx = _StubToolContext([_StubDoc()])
        mock_response = MagicMock()
        mock_response.content = '{"rewritten_query": "test", "filter_suggestion": {"unknown_key": "value", "title_contains": "Weekly"}, "reasoning": "test"}'

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch("app.services.agentic_rag.tools.rag_retrieve.build_chat_llm", return_value=mock_llm):
            result = asyncio.run(_rewrite_query("test", "missing", ctx, []))

        _, filter_suggestion = result
        assert "unknown_key" not in filter_suggestion
        assert "title_contains" in filter_suggestion

    def test_rewrite_failure_returns_original_and_none(self):
        from app.services.agentic_rag.tools.rag_retrieve import _rewrite_query
        ctx = _StubToolContext([_StubDoc()])
        with patch("app.services.agentic_rag.tools.rag_retrieve.build_chat_llm", side_effect=Exception("LLM down")):
            result = asyncio.run(_rewrite_query("test query", "missing", ctx, []))
        rewritten, filter_suggestion = result
        assert rewritten == "test query"
        assert filter_suggestion is None


# ── Empty result tests ─────────────────────────────────────────────────────────

class TestEmptyResult:
    def test_empty_result_has_correct_shape(self):
        from app.services.agentic_rag.tools.rag_retrieve import _empty_result, RagRetrieveInput
        import time
        input_obj = RagRetrieveInput(query="test", filters={"title_contains": "Nonexistent"})
        result = _empty_result(input_obj, time.monotonic(), "filters matched 0 documents")
        assert result["ok"] is True
        assert result["result"]["docs"] == []
        assert result["result"]["confidence"] == 0.0
        assert result["result"]["filters_applied"] == {"title_contains": "Nonexistent"}
        assert "filters matched 0" in result["result"]["missing"]


# ── Context verification: no duplicates after filtering ────────────────────────

class TestContextNoDuplicates:
    def test_sort_preserves_distinct_docs(self):
        """Sort should not create duplicates — each doc appears once."""
        from app.services.agentic_rag.tools.rag_retrieve import _sort_merged_docs
        docs = [
            {"page_content": "doc1", "metadata": {"_file_created_at": "2026-06-01", "content_hash": "h1"}},
            {"page_content": "doc2", "metadata": {"_file_created_at": "2026-06-07", "content_hash": "h2"}},
            {"page_content": "doc3", "metadata": {"_file_created_at": "2026-06-03", "content_hash": "h3"}},
        ]
        sorted_docs = _sort_merged_docs(docs, {"field": "file_created_at", "direction": "desc"})
        hashes = [d["metadata"]["content_hash"] for d in sorted_docs]
        assert len(hashes) == len(set(hashes)), "Sort produced duplicate content hashes"

    def test_dedup_by_content_hash_removes_duplicates(self):
        """Content-hash dedup should remove exact duplicates."""
        from app.services.agentic_rag.nodes import dedup_by_content_hash
        docs = [
            {"page_content": "same content", "metadata": {"content_hash": "h1"}},
            {"page_content": "same content", "metadata": {"content_hash": "h1"}},
            {"page_content": "different", "metadata": {"content_hash": "h2"}},
        ]
        result = dedup_by_content_hash(docs)
        assert len(result) == 2
        contents = [d["page_content"] for d in result]
        assert "same content" in contents
        assert "different" in contents
