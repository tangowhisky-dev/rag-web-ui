"""Tests for sort/rerank ordering in _run_retrieval_pass.

Verifies that:
1. Sort is applied AFTER filtering (not before reranking), so user's
   explicit sort order is preserved instead of being overridden by
   reranker relevance score.
2. Without sort, reranker score order is preserved (current behavior).
3. Exact-only search with sort skips reranker entirely.
4. Sort on empty result set (all docs filtered out) is a no-op.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from app.services.agentic_rag.tools.rag_retrieve import _sort_merged_docs


class TestSortMergedDocs:
    """Unit tests for _sort_merged_docs in isolation."""

    def test_sort_by_created_at_desc(self):
        docs = [
            {"page_content": "a", "metadata": {"_created_at": "2026-01-01"}},
            {"page_content": "b", "metadata": {"_created_at": "2026-06-01"}},
            {"page_content": "c", "metadata": {"_created_at": "2026-03-01"}},
        ]
        result = _sort_merged_docs(docs, {"field": "created_at", "direction": "desc"})
        assert result[0]["page_content"] == "b"
        assert result[1]["page_content"] == "c"
        assert result[2]["page_content"] == "a"

    def test_sort_by_created_at_asc(self):
        docs = [
            {"page_content": "a", "metadata": {"_created_at": "2026-06-01"}},
            {"page_content": "b", "metadata": {"_created_at": "2026-01-01"}},
        ]
        result = _sort_merged_docs(docs, {"field": "created_at", "direction": "asc"})
        assert result[0]["page_content"] == "b"
        assert result[1]["page_content"] == "a"

    def test_no_sort_returns_original(self):
        docs = [{"page_content": "a"}, {"page_content": "b"}]
        result = _sort_merged_docs(docs, None)
        assert result == docs

    def test_empty_docs_returns_empty(self):
        result = _sort_merged_docs([], {"field": "created_at", "direction": "desc"})
        assert result == []

    def test_missing_metadata_field_falls_back(self):
        docs = [
            {"page_content": "a", "metadata": {}},
            {"page_content": "b", "metadata": {"_created_at": "2026-01-01"}},
        ]
        result = _sort_merged_docs(docs, {"field": "created_at", "direction": "desc"})
        # Should not crash, should return docs in some order
        assert len(result) == 2


class TestSortRerankOrder:
    """Integration tests for the sort-after-filter flow in _run_retrieval_pass.

    These test the ordering logic without requiring a live Qdrant/MySQL stack.
    We mock the retrieval nodes and verify the state transitions.
    """

    def _make_docs(self, n: int, scores: list[float] | None = None,
                   dates: list[str] | None = None) -> list[dict]:
        docs = []
        for i in range(n):
            d = {"page_content": f"doc_{i}", "metadata": {"content_hash": f"h_{i}"}}
            if scores:
                d["_reranker_score"] = scores[i]
            if dates:
                d["metadata"]["_created_at"] = dates[i]
            docs.append(d)
        return docs

    @pytest.mark.asyncio
    async def test_sort_preserved_after_filter(self):
        """Sort by created_at desc should be preserved after reranking+filtering."""
        from app.services.agentic_rag.tools.rag_retrieve import _run_retrieval_pass

        ctx = MagicMock()
        ctx.db = None
        ctx.org_id = 1

        scored_docs = self._make_docs(
            3,
            scores=[0.9, 0.5, 0.8],
            dates=["2026-01-01", "2026-06-01", "2026-03-01"],
        )

        with patch("app.services.agentic_rag.tools.rag_retrieve.expand_query_node", return_value={}), \
             patch("app.services.agentic_rag.tools.rag_retrieve.dense_retrieval_node", return_value={"dense_docs": []}), \
             patch("app.services.agentic_rag.tools.rag_retrieve.sparse_retrieval_node", return_value={"sparse_docs": []}), \
             patch("app.services.agentic_rag.tools.rag_retrieve.exact_retrieval_node", return_value={"exact_docs": scored_docs}), \
             patch("app.services.agentic_rag.tools.rag_retrieve.merge_node", return_value={"retrieved_docs": scored_docs}), \
             patch("app.services.agentic_rag.tools.rag_retrieve.reranking_node", return_value={
                 "retrieved_docs": scored_docs,
                 "all_scored_docs": scored_docs,
                 "retrieval_confidence": 0.8,
             }), \
             patch("app.services.agentic_rag.tools.rag_retrieve.filter_node", return_value={
                 "retrieved_docs": scored_docs,
             }):
            state = await _run_retrieval_pass(
                ctx, query="test", kb_ids=[1], org_id=1,
                file_markdown=None, legs=["dense", "sparse", "exact"],
                level={"dense_min_score": 0.0, "sparse_min_score": 0.0,
                       "exact_min_score": 0.0, "rerank_threshold": 0.0},
                sort={"field": "created_at", "direction": "desc"},
            )

        result = state["retrieved_docs"]
        # Sort should be by created_at desc, NOT by reranker score
        assert result[0]["metadata"]["_created_at"] == "2026-06-01"
        assert result[1]["metadata"]["_created_at"] == "2026-03-01"
        assert result[2]["metadata"]["_created_at"] == "2026-01-01"

    @pytest.mark.asyncio
    async def test_no_sort_preserves_reranker_order(self):
        """Without sort, reranker score order should be preserved."""
        from app.services.agentic_rag.tools.rag_retrieve import _run_retrieval_pass

        ctx = MagicMock()
        ctx.db = None
        ctx.org_id = 1

        # Reranker returns docs in score order: 0.9, 0.8, 0.5
        scored_docs = self._make_docs(
            3,
            scores=[0.9, 0.8, 0.5],
            dates=["2026-01-01", "2026-06-01", "2026-03-01"],
        )

        with patch("app.services.agentic_rag.tools.rag_retrieve.expand_query_node", return_value={}), \
             patch("app.services.agentic_rag.tools.rag_retrieve.dense_retrieval_node", return_value={"dense_docs": []}), \
             patch("app.services.agentic_rag.tools.rag_retrieve.sparse_retrieval_node", return_value={"sparse_docs": []}), \
             patch("app.services.agentic_rag.tools.rag_retrieve.exact_retrieval_node", return_value={"exact_docs": scored_docs}), \
             patch("app.services.agentic_rag.tools.rag_retrieve.merge_node", return_value={"retrieved_docs": scored_docs}), \
             patch("app.services.agentic_rag.tools.rag_retrieve.reranking_node", return_value={
                 "retrieved_docs": scored_docs,
                 "all_scored_docs": scored_docs,
                 "retrieval_confidence": 0.8,
             }), \
             patch("app.services.agentic_rag.tools.rag_retrieve.filter_node", return_value={
                 "retrieved_docs": scored_docs,
             }):
            state = await _run_retrieval_pass(
                ctx, query="test", kb_ids=[1], org_id=1,
                file_markdown=None, legs=["dense", "sparse", "exact"],
                level={"dense_min_score": 0.0, "sparse_min_score": 0.0,
                       "exact_min_score": 0.0, "rerank_threshold": 0.0},
                sort=None,
            )

        result = state["retrieved_docs"]
        # No sort → reranker score order preserved (0.9, 0.8, 0.5)
        assert result[0]["_reranker_score"] == 0.9
        assert result[1]["_reranker_score"] == 0.8
        assert result[2]["_reranker_score"] == 0.5

    @pytest.mark.asyncio
    async def test_exact_only_with_sort_skips_reranker(self):
        """legs=['exact'] with sort should skip reranking_node and filter_node."""
        from app.services.agentic_rag.tools.rag_retrieve import _run_retrieval_pass

        ctx = MagicMock()
        ctx.db = None
        ctx.org_id = 1

        merged_docs = self._make_docs(
            3,
            dates=["2026-01-01", "2026-06-01", "2026-03-01"],
        )

        reranker_called = False
        filter_called = False

        def _track_reranker(*args, **kwargs):
            nonlocal reranker_called
            reranker_called = True
            return {"retrieved_docs": [], "all_scored_docs": [], "retrieval_confidence": 0.0}

        def _track_filter(*args, **kwargs):
            nonlocal filter_called
            filter_called = True
            return {"retrieved_docs": []}

        with patch("app.services.agentic_rag.tools.rag_retrieve.expand_query_node", return_value={}), \
             patch("app.services.agentic_rag.tools.rag_retrieve.exact_retrieval_node", return_value={"exact_docs": merged_docs}), \
             patch("app.services.agentic_rag.tools.rag_retrieve.merge_node", return_value={"retrieved_docs": merged_docs}), \
             patch("app.services.agentic_rag.tools.rag_retrieve.reranking_node", side_effect=_track_reranker), \
             patch("app.services.agentic_rag.tools.rag_retrieve.filter_node", side_effect=_track_filter):
            state = await _run_retrieval_pass(
                ctx, query="test", kb_ids=[1], org_id=1,
                file_markdown=None, legs=["exact"],
                level={"dense_min_score": 0.0, "sparse_min_score": 0.0,
                       "exact_min_score": 0.0, "rerank_threshold": 0.0},
                sort={"field": "created_at", "direction": "desc"},
            )

        assert not reranker_called, "reranking_node should be skipped for exact-only + sort"
        assert not filter_called, "filter_node should be skipped for exact-only + sort"
        result = state["retrieved_docs"]
        assert result[0]["metadata"]["_created_at"] == "2026-06-01"
        assert result[1]["metadata"]["_created_at"] == "2026-03-01"
        assert result[2]["metadata"]["_created_at"] == "2026-01-01"

    @pytest.mark.asyncio
    async def test_sort_on_empty_after_filter(self):
        """Sort on empty result set (all filtered out) should be a no-op."""
        from app.services.agentic_rag.tools.rag_retrieve import _run_retrieval_pass

        ctx = MagicMock()
        ctx.db = None
        ctx.org_id = 1

        with patch("app.services.agentic_rag.tools.rag_retrieve.expand_query_node", return_value={}), \
             patch("app.services.agentic_rag.tools.rag_retrieve.dense_retrieval_node", return_value={"dense_docs": []}), \
             patch("app.services.agentic_rag.tools.rag_retrieve.sparse_retrieval_node", return_value={"sparse_docs": []}), \
             patch("app.services.agentic_rag.tools.rag_retrieve.exact_retrieval_node", return_value={"exact_docs": []}), \
             patch("app.services.agentic_rag.tools.rag_retrieve.merge_node", return_value={"retrieved_docs": []}), \
             patch("app.services.agentic_rag.tools.rag_retrieve.reranking_node", return_value={
                 "retrieved_docs": [],
                 "all_scored_docs": [],
                 "retrieval_confidence": 0.0,
             }), \
             patch("app.services.agentic_rag.tools.rag_retrieve.filter_node", return_value={
                 "retrieved_docs": [],
             }):
            state = await _run_retrieval_pass(
                ctx, query="test", kb_ids=[1], org_id=1,
                file_markdown=None, legs=["dense", "sparse", "exact"],
                level={"dense_min_score": 0.0, "sparse_min_score": 0.0,
                       "exact_min_score": 0.0, "rerank_threshold": 0.5},
                sort={"field": "created_at", "direction": "desc"},
            )

        assert state["retrieved_docs"] == []
