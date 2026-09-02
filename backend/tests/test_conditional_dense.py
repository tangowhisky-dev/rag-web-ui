"""Tests for conditional dense leg with quality gate (Phase 3).

Tests that the dense leg is skipped when exact+sparse reranker scores
are above the fast-accept threshold, and run when they're below.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.agentic_rag.tools.rag_retrieve import _run_retrieval_pass


def _make_docs(n: int, scores: list[float] | None = None) -> list[dict]:
    docs = []
    for i in range(n):
        d = {"page_content": f"doc_{i}", "metadata": {"content_hash": f"h_{i}"}}
        if scores:
            d["metadata"]["_reranker_score"] = scores[i]
        docs.append(d)
    return docs


class TestConditionalDenseLeg:
    @pytest.mark.asyncio
    async def test_dense_skipped_when_score_above_threshold(self):
        """When exact+sparse reranker score >= threshold, dense is skipped."""
        ctx = MagicMock()
        ctx.db = MagicMock()
        ctx.org_id = 1
        ctx.state = {"excluded_terms": []}

        scored_docs = _make_docs(3, scores=[0.9, 0.8, 0.7])

        dense_called = False

        async def _track_dense(*args, **kwargs):
            nonlocal dense_called
            dense_called = True
            return {"dense_docs": []}

        with patch("app.services.agentic_rag.tools.rag_retrieve.expand_query_node", return_value={}), \
             patch("app.services.agentic_rag.tools.rag_retrieve._expand_synonyms", new=AsyncMock(return_value=("query", []))), \
             patch("app.services.agentic_rag.tools.rag_retrieve.sparse_retrieval_node", new=AsyncMock(return_value={"sparse_docs": scored_docs})), \
             patch("app.services.agentic_rag.tools.rag_retrieve.exact_retrieval_node", new=AsyncMock(return_value={"exact_docs": scored_docs})), \
             patch("app.services.agentic_rag.tools.rag_retrieve.dense_retrieval_node", new=AsyncMock(side_effect=_track_dense)), \
             patch("app.services.agentic_rag.tools.rag_retrieve.merge_node", return_value={"retrieved_docs": scored_docs}), \
             patch("app.services.agentic_rag.tools.rag_retrieve.reranking_node", return_value={
                 "retrieved_docs": scored_docs,
                 "all_scored_docs": scored_docs,
                 "retrieval_confidence": 0.9,
             }), \
             patch("app.services.agentic_rag.tools.rag_retrieve.filter_node", return_value={
                 "retrieved_docs": scored_docs,
             }), \
             patch("app.services.settings_service.get_setting", return_value=0.7):
            state = await _run_retrieval_pass(
                ctx, query="test", kb_ids=[1], org_id=1,
                file_markdown=None, legs=["dense", "sparse", "exact"],
                level={"dense_min_score": 0.0, "sparse_min_score": 0.0,
                       "exact_min_score": 0.0, "rerank_threshold": 0.0},
            )

        assert not dense_called, "dense_retrieval_node should be skipped when score >= threshold"

    @pytest.mark.asyncio
    async def test_dense_run_when_score_below_threshold(self):
        """When exact+sparse reranker score < threshold, dense is run."""
        ctx = MagicMock()
        ctx.db = MagicMock()
        ctx.org_id = 1
        ctx.state = {"excluded_terms": []}

        scored_docs = _make_docs(3, scores=[0.3, 0.2, 0.1])
        dense_docs = _make_docs(2, scores=[0.5, 0.4])

        dense_called = False

        async def _track_dense(*args, **kwargs):
            nonlocal dense_called
            dense_called = True
            return {"dense_docs": dense_docs}

        with patch("app.services.agentic_rag.tools.rag_retrieve.expand_query_node", return_value={}), \
             patch("app.services.agentic_rag.tools.rag_retrieve._expand_synonyms", new=AsyncMock(return_value=("query", []))), \
             patch("app.services.agentic_rag.tools.rag_retrieve.sparse_retrieval_node", new=AsyncMock(return_value={"sparse_docs": scored_docs})), \
             patch("app.services.agentic_rag.tools.rag_retrieve.exact_retrieval_node", new=AsyncMock(return_value={"exact_docs": scored_docs})), \
             patch("app.services.agentic_rag.tools.rag_retrieve.dense_retrieval_node", new=AsyncMock(side_effect=_track_dense)), \
             patch("app.services.agentic_rag.tools.rag_retrieve.merge_node", return_value={"retrieved_docs": scored_docs + dense_docs}), \
             patch("app.services.agentic_rag.tools.rag_retrieve.reranking_node", return_value={
                 "retrieved_docs": scored_docs + dense_docs,
                 "all_scored_docs": scored_docs + dense_docs,
                 "retrieval_confidence": 0.5,
             }), \
             patch("app.services.agentic_rag.tools.rag_retrieve.filter_node", return_value={
                 "retrieved_docs": scored_docs + dense_docs,
             }), \
             patch("app.services.settings_service.get_setting", return_value=0.7):
            state = await _run_retrieval_pass(
                ctx, query="test", kb_ids=[1], org_id=1,
                file_markdown=None, legs=["dense", "sparse", "exact"],
                level={"dense_min_score": 0.0, "sparse_min_score": 0.0,
                       "exact_min_score": 0.0, "rerank_threshold": 0.0},
            )

        assert dense_called, "dense_retrieval_node should be run when score < threshold"

    @pytest.mark.asyncio
    async def test_no_dense_leg_runs_all_concurrently(self):
        """When dense is not in legs, all legs run concurrently (no quality gate)."""
        ctx = MagicMock()
        ctx.db = MagicMock()
        ctx.org_id = 1
        ctx.state = {"excluded_terms": []}

        scored_docs = _make_docs(3, scores=[0.9, 0.8, 0.7])

        with patch("app.services.agentic_rag.tools.rag_retrieve.expand_query_node", return_value={}), \
             patch("app.services.agentic_rag.tools.rag_retrieve._expand_synonyms", new=AsyncMock(return_value=("query", []))), \
             patch("app.services.agentic_rag.tools.rag_retrieve.sparse_retrieval_node", new=AsyncMock(return_value={"sparse_docs": scored_docs})), \
             patch("app.services.agentic_rag.tools.rag_retrieve.exact_retrieval_node", new=AsyncMock(return_value={"exact_docs": scored_docs})), \
             patch("app.services.agentic_rag.tools.rag_retrieve.merge_node", return_value={"retrieved_docs": scored_docs}), \
             patch("app.services.agentic_rag.tools.rag_retrieve.reranking_node", return_value={
                 "retrieved_docs": scored_docs,
                 "all_scored_docs": scored_docs,
                 "retrieval_confidence": 0.9,
             }), \
             patch("app.services.agentic_rag.tools.rag_retrieve.filter_node", return_value={
                 "retrieved_docs": scored_docs,
             }), \
             patch("app.services.settings_service.get_setting", return_value=0.7):
            state = await _run_retrieval_pass(
                ctx, query="test", kb_ids=[1], org_id=1,
                file_markdown=None, legs=["sparse", "exact"],
                level={"dense_min_score": 0.0, "sparse_min_score": 0.0,
                       "exact_min_score": 0.0, "rerank_threshold": 0.0},
            )

        assert len(state["retrieved_docs"]) == 3
