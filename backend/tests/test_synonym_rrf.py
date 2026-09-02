"""Tests for synonym expansion and RRF fusion (Phase 2).

Tests the RRF fusion algorithm, synonym expansion prompt parsing,
and Redis caching behavior.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.documents import Document as LCDoc

from app.services.retrieval.retrieval import _rrf_fuse


class TestRRFFusion:
    """Unit tests for Reciprocal Rank Fusion."""

    def _make_docs(self, hashes: list[str]) -> list[LCDoc]:
        return [LCDoc(page_content=f"content_{h}", metadata={"content_hash": h}) for h in hashes]

    def test_empty_input(self):
        assert _rrf_fuse([]) == []

    def test_single_list_passthrough(self):
        docs = self._make_docs(["a", "b", "c"])
        assert _rrf_fuse([docs]) == docs

    def test_two_lists_dedup(self):
        list1 = self._make_docs(["a", "b", "c"])
        list2 = self._make_docs(["b", "c", "d"])
        fused = _rrf_fuse([list1, list2])
        # Should have 4 unique docs (a, b, c, d)
        assert len(fused) == 4
        # b and c appear in both lists, so they should rank higher
        fused_hashes = [d.metadata["content_hash"] for d in fused]
        assert "a" in fused_hashes
        assert "b" in fused_hashes
        assert "c" in fused_hashes
        assert "d" in fused_hashes
        # b and c should be before a and d (they have higher fused scores)
        assert fused_hashes.index("b") < fused_hashes.index("a")
        assert fused_hashes.index("c") < fused_hashes.index("a")

    def test_three_lists(self):
        list1 = self._make_docs(["a", "b"])
        list2 = self._make_docs(["b", "c"])
        list3 = self._make_docs(["c", "a"])
        fused = _rrf_fuse([list1, list2, list3])
        assert len(fused) == 3
        # All three appear in 2 lists each, so scores should be close

    def test_no_metadata_hash_uses_content(self):
        docs1 = [LCDoc(page_content="hello", metadata={})]
        docs2 = [LCDoc(page_content="hello", metadata={})]
        fused = _rrf_fuse([docs1, docs2])
        # Should dedup by content hash (computed from page_content)
        assert len(fused) == 1


class TestSynonymExpansion:
    """Tests for _expand_synonyms in rag_retrieve."""

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        from app.services.agentic_rag.tools.rag_retrieve import _expand_synonyms

        ctx = MagicMock()
        ctx.db = MagicMock()
        ctx.org_id = 1

        cached = {"corrected": "Flaschenöffner", "synonyms": ["bottle opener"]}

        with patch("redis.asyncio.from_url") as mock_redis:
            mock_r = AsyncMock()
            mock_r.get = AsyncMock(return_value='{"corrected": "Flaschenöffner", "synonyms": ["bottle opener"]}')
            mock_r.aclose = AsyncMock()
            mock_redis.return_value = mock_r

            with patch("app.services.settings_service.get_setting", side_effect=[3, 300]):
                corrected, synonyms = await _expand_synonyms("bieröffner", ctx)

        assert corrected == "Flaschenöffner"
        assert synonyms == ["bottle opener"]

    @pytest.mark.asyncio
    async def test_llm_call_on_cache_miss(self):
        from app.services.agentic_rag.tools.rag_retrieve import _expand_synonyms

        ctx = MagicMock()
        ctx.db = MagicMock()
        ctx.org_id = 1

        # Mock LLM response
        mock_resp = MagicMock()
        mock_resp.content = '{"corrected_query": "Flaschenöffner", "queries": ["bottle opener"]}'

        with patch("redis.asyncio.from_url") as mock_redis:
            mock_r = AsyncMock()
            mock_r.get = AsyncMock(return_value=None)
            mock_r.setex = AsyncMock()
            mock_r.aclose = AsyncMock()
            mock_redis.return_value = mock_r

            with patch("app.services.agentic_rag.llm_factory.build_chat_llm", return_value=MagicMock(ainvoke=AsyncMock(return_value=mock_resp))), \
                 patch("app.services.settings_service.get_setting", side_effect=[3, 300]):
                corrected, synonyms = await _expand_synonyms("bieröffner", ctx)

        assert corrected == "Flaschenöffner"
        assert "bottle opener" in synonyms

    @pytest.mark.asyncio
    async def test_filters_out_original_query(self):
        from app.services.agentic_rag.tools.rag_retrieve import _expand_synonyms

        ctx = MagicMock()
        ctx.db = MagicMock()
        ctx.org_id = 1

        # LLM returns the original query as a "synonym" — should be filtered
        mock_resp = MagicMock()
        mock_resp.content = '{"corrected_query": null, "queries": ["mutex", "mutual exclusion"]}'

        with patch("redis.asyncio.from_url") as mock_redis:
            mock_r = AsyncMock()
            mock_r.get = AsyncMock(return_value=None)
            mock_r.setex = AsyncMock()
            mock_r.aclose = AsyncMock()
            mock_redis.return_value = mock_r

            with patch("app.services.agentic_rag.llm_factory.build_chat_llm", return_value=MagicMock(ainvoke=AsyncMock(return_value=mock_resp))), \
                 patch("app.services.settings_service.get_setting", side_effect=[3, 300]):
                corrected, synonyms = await _expand_synonyms("mutex", ctx)

        assert corrected == "mutex"
        assert "mutex" not in [s.lower() for s in synonyms]
        assert "mutual exclusion" in synonyms

    @pytest.mark.asyncio
    async def test_llm_failure_returns_empty(self):
        from app.services.agentic_rag.tools.rag_retrieve import _expand_synonyms

        ctx = MagicMock()
        ctx.db = MagicMock()
        ctx.org_id = 1

        with patch("redis.asyncio.from_url") as mock_redis:
            mock_r = AsyncMock()
            mock_r.get = AsyncMock(return_value=None)
            mock_r.aclose = AsyncMock()
            mock_redis.return_value = mock_r

            with patch("app.services.agentic_rag.llm_factory.build_chat_llm", side_effect=Exception("LLM unavailable")), \
                 patch("app.services.settings_service.get_setting", side_effect=[3, 300]):
                corrected, synonyms = await _expand_synonyms("test", ctx)

        assert corrected == "test"
        assert synonyms == []

    @pytest.mark.asyncio
    async def test_malformed_json_returns_empty(self):
        from app.services.agentic_rag.tools.rag_retrieve import _expand_synonyms

        ctx = MagicMock()
        ctx.db = MagicMock()
        ctx.org_id = 1

        mock_resp = MagicMock()
        mock_resp.content = "This is not JSON at all"

        with patch("redis.asyncio.from_url") as mock_redis:
            mock_r = AsyncMock()
            mock_r.get = AsyncMock(return_value=None)
            mock_r.aclose = AsyncMock()
            mock_redis.return_value = mock_r

            with patch("app.services.agentic_rag.llm_factory.build_chat_llm", return_value=MagicMock(ainvoke=AsyncMock(return_value=mock_resp))), \
                 patch("app.services.settings_service.get_setting", side_effect=[3, 300]):
                corrected, synonyms = await _expand_synonyms("test", ctx)

        assert corrected == "test"
        assert synonyms == []
