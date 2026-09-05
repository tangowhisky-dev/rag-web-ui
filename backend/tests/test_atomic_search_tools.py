"""Tests for atomic search tools (search_exact, search_sparse, search_dense, rerank_results, graph_expand)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.agentic_rag.tools.search_dense import SearchDenseInput, SearchDenseTool
from app.services.agentic_rag.tools.search_exact import SearchExactInput, SearchExactTool
from app.services.agentic_rag.tools.search_sparse import SearchSparseInput, SearchSparseTool
from app.services.agentic_rag.tools.rerank_results import RerankResultsInput, RerankResultsTool
from app.services.agentic_rag.tools.graph_expand import GraphExpandInput, GraphExpandTool


class TestSearchExactTool:
    def test_schema_has_required_fields(self):
        schema = SearchExactInput.model_json_schema()
        assert "query" in schema["required"]
        assert "kb_ids" in schema["properties"]
        assert "document_ids" in schema["properties"]
        assert "filters" in schema["properties"]
        assert "top_k" in schema["properties"]

    def test_returns_empty_when_no_kb_ids(self):
        ctx = MagicMock()
        ctx.org_id = 1
        ctx.db = MagicMock()
        ctx.state = {"kb_ids": []}
        ctx.chat_id = 1
        # enforce_rbac returns empty kb_ids
        with patch("app.services.agentic_rag.tools.search_exact.enforce_rbac", return_value={"kb_ids": []}):
            tool = SearchExactTool()
            tool.ctx = ctx
            result = asyncio.run(tool.arun({"query": "test", "kb_ids": []}))
        assert result["ok"] is True
        assert result["result"]["hits"] == []
        assert result["result"]["count"] == 0

    @patch("app.services.agentic_rag.tools.search_exact.exact_search_docs")
    @patch("app.services.agentic_rag.tools.search_exact.expand_synonyms", new_callable=AsyncMock)
    @patch("app.services.agentic_rag.tools.search_exact.enforce_rbac")
    @patch("app.services.agentic_rag.tools.search_exact.get_effective_datastore_ids")
    @patch("app.services.agentic_rag.tools.search_exact.get_setting")
    def test_returns_hits_with_citation_ref(self, mock_setting, mock_ds, mock_rbac, mock_syn, mock_search):
        from langchain_core.documents import Document
        mock_setting.return_value = 0.0
        mock_ds.return_value = []
        mock_rbac.return_value = {"kb_ids": [1]}
        mock_syn.return_value = ("test", [])
        mock_search.return_value = [
            Document(page_content="test content", metadata={
                "document_id": 1, "chunk_index": 0, "page": 1,
                "title": "Test Doc", "file_name": "test.pdf",
                "content_hash": "abc123", "qdrant_point_id": "uuid-1",
            })
        ]
        ctx = MagicMock()
        ctx.org_id = 1
        ctx.db = MagicMock()
        ctx.state = {"kb_ids": [1]}
        ctx.chat_id = 1
        ctx.message_id = 1
        tool = SearchExactTool()
        tool.ctx = ctx
        result = asyncio.run(tool.arun({"query": "test", "kb_ids": [1]}))
        assert result["ok"] is True
        assert len(result["result"]["hits"]) == 1
        hit = result["result"]["hits"][0]
        assert hit["document_id"] == 1
        assert hit["citation_ref"]["citation_kind"] == "chunk"
        assert hit["citation_ref"]["source_tool"] == "search_exact"
        assert hit["citation_ref"]["document_id"] == 1


class TestSearchSparseTool:
    def test_schema_matches_dense(self):
        sparse_schema = SearchSparseInput.model_json_schema()
        dense_schema = SearchDenseInput.model_json_schema()
        assert set(sparse_schema["properties"].keys()) == set(dense_schema["properties"].keys())


class TestSearchDenseTool:
    def test_prepare_arguments_normalizes_kb_ids(self):
        tool = SearchDenseTool()
        result = tool.prepare_arguments({"kb_ids": "5", "query": "test"})
        assert result["kb_ids"] == [5]

    def test_prepare_arguments_handles_int(self):
        tool = SearchDenseTool()
        result = tool.prepare_arguments({"kb_ids": 5, "query": "test"})
        assert result["kb_ids"] == [5]

    def test_prepare_arguments_handles_list(self):
        tool = SearchDenseTool()
        result = tool.prepare_arguments({"kb_ids": [1, 2, 3], "query": "test"})
        assert result["kb_ids"] == [1, 2, 3]

    @patch("app.services.agentic_rag.tools.search_dense.dense_search_docs")
    @patch("app.services.agentic_rag.tools.search_dense.enforce_rbac")
    @patch("app.services.agentic_rag.tools.search_dense.get_effective_datastore_ids")
    @patch("app.services.agentic_rag.tools.search_dense.get_setting")
    def test_returns_hits_with_citation_ref(self, mock_setting, mock_ds, mock_rbac, mock_search):
        from langchain_core.documents import Document
        mock_setting.return_value = 0.0
        mock_ds.return_value = []
        mock_rbac.return_value = {"kb_ids": [1]}
        mock_search.return_value = [
            Document(page_content="semantic content", metadata={
                "document_id": 2, "chunk_index": 3, "page": 5,
                "title": "Semantic Doc", "file_name": "sem.pdf",
                "content_hash": "def456", "qdrant_point_id": "uuid-2",
            })
        ]
        ctx = MagicMock()
        ctx.org_id = 1
        ctx.db = MagicMock()
        ctx.state = {"kb_ids": [1]}
        ctx.chat_id = 1
        ctx.message_id = 1
        tool = SearchDenseTool()
        tool.ctx = ctx
        result = asyncio.run(tool.arun({"query": "semantic test", "kb_ids": [1]}))
        assert result["ok"] is True
        assert len(result["result"]["hits"]) == 1
        hit = result["result"]["hits"][0]
        assert hit["citation_ref"]["source_tool"] == "search_dense"
        assert hit["citation_ref"]["citation_kind"] == "chunk"


class TestRerankResultsTool:
    def test_empty_hits_returns_empty(self):
        ctx = MagicMock()
        tool = RerankResultsTool()
        tool.ctx = ctx
        result = asyncio.run(tool.arun({"query": "test", "hits": []}))
        assert result["ok"] is True
        assert result["result"]["hits"] == []
        assert result["result"]["output_count"] == 0

    @patch("app.services.agentic_rag.tools.rerank_results.rerank")
    @patch("app.services.agentic_rag.tools.rerank_results.dedup_by_content_hash")
    @patch("app.services.agentic_rag.tools.rerank_results.semantic_dedup")
    @patch("app.services.agentic_rag.tools.rerank_results.get_setting")
    def test_no_top_n_cap(self, mock_setting, mock_semidedup, mock_hashdedup, mock_rerank):
        """Verify that top_n=None returns all hits passing threshold."""
        from langchain_core.documents import Document
        mock_setting.return_value = -10.0  # low threshold so all pass
        mock_hashdedup.side_effect = lambda x: x
        mock_semidedup.side_effect = lambda x, **kw: x
        # Mock rerank to return all 5 docs with scores
        input_hits = [
            {"content": f"doc {i}", "document_id": i, "chunk_index": 0,
             "title": f"Doc{i}", "content_hash": f"h{i}", "citation_ref": {}}
            for i in range(5)
        ]
        reranked_docs = [
            Document(page_content=f"doc {i}", metadata={
                "document_id": i, "chunk_index": 0, "title": f"Doc{i}",
                "content_hash": f"h{i}", "_reranker_score": 1.0 - i * 0.1,
                "citation_ref": {},
            })
            for i in range(5)
        ]
        mock_rerank.return_value = reranked_docs
        ctx = MagicMock()
        ctx.org_id = 1
        ctx.db = MagicMock()
        ctx.chat_id = 1
        ctx.message_id = 1
        # Populate retrieved_docs so hit provenance validation passes
        ctx.state = {
            "retrieved_docs": [
                {"page_content": f"doc {i}", "metadata": {"document_id": i, "chunk_index": 0, "content_hash": f"h{i}"}}
                for i in range(5)
            ]
        }
        tool = RerankResultsTool()
        tool.ctx = ctx
        result = asyncio.run(tool.arun({"query": "test", "hits": input_hits, "top_n": None}))
        assert result["ok"] is True
        assert len(result["result"]["hits"]) == 5  # no cap

    @patch("app.services.agentic_rag.tools.rerank_results.rerank")
    @patch("app.services.agentic_rag.tools.rerank_results.dedup_by_content_hash")
    @patch("app.services.agentic_rag.tools.rerank_results.semantic_dedup")
    @patch("app.services.agentic_rag.tools.rerank_results.get_setting")
    def test_top_n_caps_results(self, mock_setting, mock_semidedup, mock_hashdedup, mock_rerank):
        """When top_n is specified, results are capped."""
        from langchain_core.documents import Document
        mock_setting.return_value = -10.0
        mock_hashdedup.side_effect = lambda x: x
        mock_semidedup.side_effect = lambda x, **kw: x
        input_hits = [
            {"content": f"doc {i}", "document_id": i, "chunk_index": 0,
             "title": f"Doc{i}", "content_hash": f"h{i}", "citation_ref": {}}
            for i in range(15)
        ]
        reranked_docs = [
            Document(page_content=f"doc {i}", metadata={
                "document_id": i, "chunk_index": 0, "title": f"Doc{i}",
                "content_hash": f"h{i}", "_reranker_score": 1.0,
                "citation_ref": {},
            })
            for i in range(15)
        ]
        mock_rerank.return_value = reranked_docs
        ctx = MagicMock()
        ctx.org_id = 1
        ctx.db = MagicMock()
        ctx.chat_id = 1
        ctx.message_id = 1
        ctx.state = {
            "retrieved_docs": [
                {"page_content": f"doc {i}", "metadata": {"document_id": i, "chunk_index": 0, "content_hash": f"h{i}"}}
                for i in range(15)
            ]
        }
        tool = RerankResultsTool()
        tool.ctx = ctx
        result = asyncio.run(tool.arun({"query": "test", "hits": input_hits, "top_n": 5}))
        assert result["ok"] is True
        assert len(result["result"]["hits"]) == 5

    @patch("app.services.agentic_rag.tools.rerank_results.rerank")
    @patch("app.services.agentic_rag.tools.rerank_results.dedup_by_content_hash")
    @patch("app.services.agentic_rag.tools.rerank_results.semantic_dedup")
    @patch("app.services.agentic_rag.tools.rerank_results.get_setting")
    def test_citation_ref_source_tool_updated(self, mock_setting, mock_semidedup, mock_hashdedup, mock_rerank):
        """Reranked hits should have citation_ref.source_tool = 'rerank_results'."""
        from langchain_core.documents import Document
        mock_setting.return_value = -10.0
        mock_hashdedup.side_effect = lambda x: x
        mock_semidedup.side_effect = lambda x, **kw: x
        input_hits = [
            {"content": "doc 0", "document_id": 1, "chunk_index": 0,
             "title": "Doc0", "content_hash": "h0",
             "citation_ref": {"document_id": 1, "source_tool": "search_dense"}}
        ]
        mock_rerank.return_value = [
            Document(page_content="doc 0", metadata={
                "document_id": 1, "chunk_index": 0, "title": "Doc0",
                "content_hash": "h0", "_reranker_score": 1.0,
                "citation_ref": {"document_id": 1, "source_tool": "search_dense"},
            })
        ]
        ctx = MagicMock()
        ctx.org_id = 1
        ctx.db = MagicMock()
        ctx.chat_id = 1
        ctx.message_id = 1
        ctx.state = {
            "retrieved_docs": [
                {"page_content": "doc 0", "metadata": {"document_id": 1, "chunk_index": 0, "content_hash": "h0"}}
            ]
        }
        tool = RerankResultsTool()
        tool.ctx = ctx
        result = asyncio.run(tool.arun({"query": "test", "hits": input_hits}))
        hit = result["result"]["hits"][0]
        assert hit["citation_ref"]["source_tool"] == "rerank_results"

    @patch("app.services.agentic_rag.tools.rerank_results.get_setting")
    def test_fabricated_hits_rejected(self, mock_setting):
        """Hits not in retrieved_docs are rejected to prevent LLM hallucination."""
        mock_setting.return_value = -10.0
        input_hits = [
            {"content": "fabricated content", "document_id": 999, "chunk_index": 0,
             "title": "Fake Doc", "content_hash": "fake_hash", "citation_ref": {}}
        ]
        ctx = MagicMock()
        ctx.org_id = 1
        ctx.db = MagicMock()
        ctx.chat_id = 1
        ctx.message_id = 1
        ctx.state = {"retrieved_docs": []}  # no real docs retrieved
        tool = RerankResultsTool()
        tool.ctx = ctx
        result = asyncio.run(tool.arun({"query": "test", "hits": input_hits}))
        assert result["ok"] is True
        assert result["result"]["output_count"] == 0
        assert result["result"]["rejected_fabricated"] == 1
        assert "rejected" in result["error"]

    @patch("app.services.agentic_rag.tools.rerank_results.rerank")
    @patch("app.services.agentic_rag.tools.rerank_results.dedup_by_content_hash")
    @patch("app.services.agentic_rag.tools.rerank_results.semantic_dedup")
    @patch("app.services.agentic_rag.tools.rerank_results.get_setting")
    def test_auto_fallback_to_retrieved_docs(self, mock_setting, mock_semidedup, mock_hashdedup, mock_rerank):
        """When LLM hits are rejected, auto-fallback to reranking retrieved_docs."""
        from langchain_core.documents import Document
        mock_setting.return_value = -10.0
        mock_hashdedup.side_effect = lambda x: x
        mock_semidedup.side_effect = lambda x, **kw: x
        mock_rerank.return_value = [
            Document(page_content="real doc", metadata={
                "document_id": 1, "chunk_index": 0, "title": "Real",
                "content_hash": "real_hash", "_reranker_score": 1.0,
                "citation_ref": {},
            })
        ]
        fabricated_hits = [
            {"content": "fake", "document_id": 999, "chunk_index": 0,
             "title": "Fake", "content_hash": "fake_hash", "citation_ref": {}}
        ]
        ctx = MagicMock()
        ctx.org_id = 1
        ctx.db = MagicMock()
        ctx.chat_id = 1
        ctx.message_id = 1
        ctx.state = {
            "retrieved_docs": [
                {"page_content": "real doc", "metadata": {
                    "document_id": 1, "chunk_index": 0, "content_hash": "real_hash",
                    "_reranker_score": 0.5,
                }}
            ]
        }
        tool = RerankResultsTool()
        tool.ctx = ctx
        result = asyncio.run(tool.arun({"query": "test", "hits": fabricated_hits}))
        assert result["ok"] is True
        assert result["result"]["output_count"] == 1
        assert result["result"]["rejected_fabricated"] == 1


class TestGraphExpandTool:
    def test_no_seeds_returns_empty(self):
        ctx = MagicMock()
        ctx.org_id = 1
        ctx.db = MagicMock()
        ctx.state = {"kb_ids": [1]}
        ctx.chat_id = 1
        with patch("app.services.agentic_rag.tools.graph_expand.enforce_rbac", return_value={"kb_ids": [1]}):
            tool = GraphExpandTool()
            tool.ctx = ctx
            result = asyncio.run(tool.arun({"kb_ids": [1], "seed_document_ids": None, "seed_chunk_ids": None}))
        assert result["ok"] is True
        assert result["result"]["hits"] == []

    @patch("app.services.agentic_rag.tools.graph_expand.expand_docs_via_graph")
    @patch("app.services.agentic_rag.tools.graph_expand.enforce_rbac")
    @patch("app.services.agentic_rag.tools.graph_expand.get_effective_datastore_ids")
    def test_failure_is_non_fatal(self, mock_ds, mock_rbac, mock_expand):
        """Graph expansion failures return empty hits, not errors."""
        mock_ds.return_value = []
        mock_rbac.return_value = {"kb_ids": [1]}
        mock_expand.side_effect = Exception("Neo4j connection failed")
        ctx = MagicMock()
        ctx.org_id = 1
        ctx.db = MagicMock()
        ctx.state = {"kb_ids": [1]}
        ctx.chat_id = 1
        ctx.message_id = 1
        tool = GraphExpandTool()
        tool.ctx = ctx
        result = asyncio.run(tool.arun({"kb_ids": [1], "seed_chunk_ids": ["uuid-1"]}))
        assert result["ok"] is True
        assert result["result"]["hits"] == []
        assert result["error"] is None


class TestToolRegistry:
    def test_build_tools_returns_atomic_search_tools(self):
        from app.services.agentic_rag.tools import build_tools
        ctx = MagicMock()
        ctx.state = {}
        tools = build_tools(ctx)
        names = {t.name for t in tools}
        assert "search_exact" in names
        assert "search_sparse" in names
        assert "search_dense" in names
        assert "rerank_results" in names
        assert "graph_expand" in names
        assert "rag_retrieve" not in names

    def test_applicable_tools_excludes_rerank_without_search(self):
        from app.services.agentic_rag.tools import applicable_tools
        ctx = MagicMock()
        ctx.state = {"tool_call_counts": {}}
        tools = applicable_tools(ctx)
        names = {t.name for t in tools}
        assert "rerank_results" not in names
        assert "graph_expand" not in names

    def test_applicable_tools_includes_rerank_after_search(self):
        from app.services.agentic_rag.tools import applicable_tools
        ctx = MagicMock()
        ctx.state = {"tool_call_counts": {"search_dense": 1}}
        tools = applicable_tools(ctx)
        names = {t.name for t in tools}
        assert "rerank_results" in names
        assert "graph_expand" in names
