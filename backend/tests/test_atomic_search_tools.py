"""Tests for search tools (keyword_search, semantic_search, rerank_results, graph_expand)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.agentic_rag.tools.keyword_search import KeywordSearchInput, KeywordSearchTool
from app.services.agentic_rag.tools.semantic_search import SemanticSearchInput, SemanticSearchTool
from app.services.agentic_rag.tools.rerank_results import RerankResultsInput, RerankResultsTool
from app.services.agentic_rag.tools.graph_expand import GraphExpandInput, GraphExpandTool


class TestKeywordSearchTool:
    def test_schema_has_required_fields(self):
        schema = KeywordSearchInput.model_json_schema()
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
        with patch("app.services.agentic_rag.tools.keyword_search.enforce_rbac", return_value={"kb_ids": []}):
            tool = KeywordSearchTool()
            tool.ctx = ctx
            result = asyncio.run(tool.arun({"query": "test", "kb_ids": []}))
        assert result["ok"] is True
        assert result["result"]["hits"] == []
        assert result["result"]["count"] == 0

    @patch("app.services.agentic_rag.tools.keyword_search.sparse_search_docs")
    @patch("app.services.agentic_rag.tools.keyword_search.exact_search_docs")
    @patch("app.services.agentic_rag.tools.keyword_search.expand_synonyms", new_callable=AsyncMock)
    @patch("app.services.agentic_rag.tools.keyword_search.enforce_rbac")
    @patch("app.services.agentic_rag.tools.keyword_search.get_effective_datastore_ids")
    @patch("app.services.agentic_rag.tools.keyword_search.get_setting")
    def test_returns_merged_hits_with_citation_ref(self, mock_setting, mock_ds, mock_rbac, mock_syn, mock_exact, mock_sparse):
        from langchain_core.documents import Document
        mock_setting.return_value = 0.0
        mock_ds.return_value = []
        mock_rbac.return_value = {"kb_ids": [1]}
        mock_syn.return_value = ("test", [])
        mock_exact.return_value = [
            Document(page_content="exact hit", metadata={
                "document_id": 1, "chunk_index": 0, "page": 1,
                "title": "Test Doc", "file_name": "test.pdf",
                "content_hash": "abc123", "qdrant_point_id": "uuid-1",
                "score": 0.9,
            })
        ]
        mock_sparse.return_value = [
            Document(page_content="sparse hit", metadata={
                "document_id": 2, "chunk_index": 5, "page": 3,
                "title": "Sparse Doc", "file_name": "sparse.pdf",
                "content_hash": "def456", "qdrant_point_id": "uuid-2",
                "score": 0.7,
            })
        ]
        ctx = MagicMock()
        ctx.org_id = 1
        ctx.db = MagicMock()
        ctx.state = {"kb_ids": [1]}
        ctx.chat_id = 1
        ctx.message_id = 1
        tool = KeywordSearchTool()
        tool.ctx = ctx
        result = asyncio.run(tool.arun({"query": "test", "kb_ids": [1]}))
        assert result["ok"] is True
        assert len(result["result"]["hits"]) == 2
        # Sorted by score descending
        assert result["result"]["hits"][0]["document_id"] == 1
        assert result["result"]["hits"][1]["document_id"] == 2
        for hit in result["result"]["hits"]:
            assert hit["citation_ref"]["source_tool"] == "keyword_search"
            assert hit["citation_ref"]["citation_kind"] == "chunk"

    @patch("app.services.agentic_rag.tools.keyword_search.sparse_search_docs")
    @patch("app.services.agentic_rag.tools.keyword_search.exact_search_docs")
    @patch("app.services.agentic_rag.tools.keyword_search.expand_synonyms", new_callable=AsyncMock)
    @patch("app.services.agentic_rag.tools.keyword_search.enforce_rbac")
    @patch("app.services.agentic_rag.tools.keyword_search.get_effective_datastore_ids")
    @patch("app.services.agentic_rag.tools.keyword_search.get_setting")
    def test_dedup_by_content_hash(self, mock_setting, mock_ds, mock_rbac, mock_syn, mock_exact, mock_sparse):
        """Same content_hash from both backends should appear once."""
        from langchain_core.documents import Document
        mock_setting.return_value = 0.0
        mock_ds.return_value = []
        mock_rbac.return_value = {"kb_ids": [1]}
        mock_syn.return_value = ("test", [])
        mock_exact.return_value = [
            Document(page_content="same content", metadata={
                "document_id": 1, "chunk_index": 0,
                "content_hash": "dup1", "score": 0.5,
            })
        ]
        mock_sparse.return_value = [
            Document(page_content="same content", metadata={
                "document_id": 1, "chunk_index": 0,
                "content_hash": "dup1", "score": 0.8,
            })
        ]
        ctx = MagicMock()
        ctx.org_id = 1
        ctx.db = MagicMock()
        ctx.state = {"kb_ids": [1]}
        ctx.chat_id = 1
        ctx.message_id = 1
        tool = KeywordSearchTool()
        tool.ctx = ctx
        result = asyncio.run(tool.arun({"query": "test", "kb_ids": [1]}))
        assert result["ok"] is True
        assert len(result["result"]["hits"]) == 1
        assert result["result"]["hits"][0]["score"] == 0.8

    @patch("app.services.agentic_rag.tools.keyword_search.sparse_search_docs")
    @patch("app.services.agentic_rag.tools.keyword_search.exact_search_docs")
    @patch("app.services.agentic_rag.tools.keyword_search.expand_synonyms", new_callable=AsyncMock)
    @patch("app.services.agentic_rag.tools.keyword_search.enforce_rbac")
    @patch("app.services.agentic_rag.tools.keyword_search.get_effective_datastore_ids")
    @patch("app.services.agentic_rag.tools.keyword_search.get_setting")
    def test_partial_failure_still_returns_hits(self, mock_setting, mock_ds, mock_rbac, mock_syn, mock_exact, mock_sparse):
        """If one backend fails, the other backend's results still return."""
        from langchain_core.documents import Document
        mock_setting.return_value = 0.0
        mock_ds.return_value = []
        mock_rbac.return_value = {"kb_ids": [1]}
        mock_syn.return_value = ("test", [])
        mock_exact.side_effect = Exception("MySQL FTS failed")
        mock_sparse.return_value = [
            Document(page_content="sparse result", metadata={
                "document_id": 3, "chunk_index": 0,
                "content_hash": "sparse_only", "score": 0.6,
            })
        ]
        ctx = MagicMock()
        ctx.org_id = 1
        ctx.db = MagicMock()
        ctx.state = {"kb_ids": [1]}
        ctx.chat_id = 1
        ctx.message_id = 1
        tool = KeywordSearchTool()
        tool.ctx = ctx
        result = asyncio.run(tool.arun({"query": "test", "kb_ids": [1]}))
        assert result["ok"] is True
        assert len(result["result"]["hits"]) == 1
        assert result["result"]["hits"][0]["document_id"] == 3


class TestSemanticSearchTool:
    def test_schema_has_required_fields(self):
        schema = SemanticSearchInput.model_json_schema()
        assert "query" in schema["required"]
        assert "kb_ids" in schema["properties"]
        assert "document_ids" in schema["properties"]
        assert "filters" in schema["properties"]
        assert "top_k" in schema["properties"]

    def test_prepare_arguments_normalizes_kb_ids(self):
        tool = SemanticSearchTool()
        result = tool.prepare_arguments({"kb_ids": "5", "query": "test"})
        assert result["kb_ids"] == [5]

    def test_prepare_arguments_handles_int(self):
        tool = SemanticSearchTool()
        result = tool.prepare_arguments({"kb_ids": 5, "query": "test"})
        assert result["kb_ids"] == [5]

    def test_prepare_arguments_handles_list(self):
        tool = SemanticSearchTool()
        result = tool.prepare_arguments({"kb_ids": [1, 2, 3], "query": "test"})
        assert result["kb_ids"] == [1, 2, 3]

    @patch("app.services.agentic_rag.tools.semantic_search.dense_search_docs")
    @patch("app.services.agentic_rag.tools.semantic_search.enforce_rbac")
    @patch("app.services.agentic_rag.tools.semantic_search.get_effective_datastore_ids")
    @patch("app.services.agentic_rag.tools.semantic_search.get_setting")
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
        tool = SemanticSearchTool()
        tool.ctx = ctx
        result = asyncio.run(tool.arun({"query": "semantic test", "kb_ids": [1]}))
        assert result["ok"] is True
        assert len(result["result"]["hits"]) == 1
        hit = result["result"]["hits"][0]
        assert hit["citation_ref"]["source_tool"] == "semantic_search"
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
        mock_setting.return_value = -10.0
        mock_hashdedup.side_effect = lambda x: x
        mock_semidedup.side_effect = lambda x, **kw: x
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
        assert len(result["result"]["hits"]) == 5

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
        mock_rerank.return_value = [
            Document(page_content="doc 0", metadata={
                "document_id": 1, "chunk_index": 0, "title": "Doc0",
                "content_hash": "h0", "_reranker_score": 1.0,
                "citation_ref": {"document_id": 1, "source_tool": "semantic_search"},
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
        result = asyncio.run(tool.arun({"query": "test"}))
        hit = result["result"]["hits"][0]
        assert hit["citation_ref"]["source_tool"] == "rerank_results"

    @patch("app.services.agentic_rag.tools.rerank_results.get_setting")
    def test_no_retrieved_docs_returns_error(self, mock_setting):
        """When no docs in state, return an error telling the LLM to search first."""
        mock_setting.return_value = -10.0
        ctx = MagicMock()
        ctx.org_id = 1
        ctx.db = MagicMock()
        ctx.chat_id = 1
        ctx.message_id = 1
        ctx.state = {"retrieved_docs": []}
        tool = RerankResultsTool()
        tool.ctx = ctx
        result = asyncio.run(tool.arun({"query": "test"}))
        assert result["ok"] is True
        assert result["result"]["output_count"] == 0
        assert "search tool first" in result["error"]

    @patch("app.services.agentic_rag.tools.rerank_results.rerank")
    @patch("app.services.agentic_rag.tools.rerank_results.dedup_by_content_hash")
    @patch("app.services.agentic_rag.tools.rerank_results.semantic_dedup")
    @patch("app.services.agentic_rag.tools.rerank_results.get_setting")
    def test_reranks_docs_from_state(self, mock_setting, mock_semidedup, mock_hashdedup, mock_rerank):
        """Reranker reads docs from state.retrieved_docs — no hits argument needed."""
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
        result = asyncio.run(tool.arun({"query": "test"}))
        assert result["ok"] is True
        assert result["result"]["output_count"] == 1


class TestGraphExpandTool:
    def test_no_retrieved_docs_returns_empty(self):
        ctx = MagicMock()
        ctx.org_id = 1
        ctx.db = MagicMock()
        ctx.state = {"kb_ids": [1], "retrieved_docs": []}
        ctx.chat_id = 1
        with patch("app.services.agentic_rag.tools.graph_expand.enforce_rbac", return_value={"kb_ids": [1]}):
            tool = GraphExpandTool()
            tool.ctx = ctx
            result = asyncio.run(tool.arun({"kb_ids": [1]}))
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
        ctx.state = {
            "kb_ids": [1],
            "retrieved_docs": [
                {"page_content": "doc", "metadata": {"document_id": 42, "qdrant_point_id": "uuid-1"}}
            ],
        }
        ctx.chat_id = 1
        ctx.message_id = 1
        tool = GraphExpandTool()
        tool.ctx = ctx
        with patch("app.services.infrastructure.get_qdrant_client") as mock_q:
            mock_q.return_value.scroll.return_value = ([MagicMock(id="uuid-1")], None)
            result = asyncio.run(tool.arun({"kb_ids": [1]}))
        assert result["ok"] is True
        assert result["result"]["hits"] == []
        assert result["error"] is None


class TestToolRegistry:
    def test_build_tools_returns_search_tools(self):
        from app.services.agentic_rag.tools import build_tools
        ctx = MagicMock()
        ctx.state = {}
        tools = build_tools(ctx)
        names = {t.name for t in tools}
        assert "keyword_search" in names
        assert "semantic_search" in names
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
        ctx.state = {"tool_call_counts": {"semantic_search": 1}}
        tools = applicable_tools(ctx)
        names = {t.name for t in tools}
        assert "rerank_results" in names
        assert "graph_expand" in names
