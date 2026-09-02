"""Tests for KB profiling (Phase 4).

Tests profile computation, Redis caching, multi-KB merging, and
prompt formatting.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from app.services.agentic_rag.kb_profile import (
    _compute_profile,
    merge_profiles,
    format_profile_summary,
    _cache_key,
)


class TestFormatProfileSummary:
    def test_empty_profile_returns_empty(self):
        assert format_profile_summary({}) == ""
        assert format_profile_summary(None) == ""

    def test_full_profile(self):
        profile = {
            "doc_count": 47,
            "titles": ["Weekly Update W01", "Weekly Update W02"],
            "content_types": ["application/pdf"],
            "date_range": {"min": "2026-01-01T00:00:00", "max": "2026-06-01T00:00:00"},
            "avg_chunk_len": 500,
            "fields": {"title_contains": True, "created_after": True},
        }
        text = format_profile_summary(profile)
        assert "[KB Profile]" in text
        assert "47" in text
        assert "application/pdf" in text
        assert "2026-01-01" in text
        assert "title_contains" in text

    def test_no_titles_omits_line(self):
        profile = {
            "doc_count": 5,
            "titles": [],
            "content_types": ["text/plain"],
            "date_range": {},
            "fields": {"file_name_contains": True},
        }
        text = format_profile_summary(profile)
        assert "Sample titles" not in text
        assert "file_name_contains" in text


class TestMergeProfiles:
    def test_empty_list_returns_empty(self):
        assert merge_profiles([]) == {}

    def test_single_profile_passthrough(self):
        p = {"doc_count": 10, "titles": ["A"], "content_types": ["pdf"], "fields": {"title_contains": True}}
        assert merge_profiles([p]) == p

    def test_merge_two_profiles(self):
        p1 = {
            "kb_id": 1,
            "doc_count": 40,
            "titles": ["Doc A"],
            "content_types": ["application/pdf"],
            "date_range": {"min": "2026-01-01", "max": "2026-03-01"},
            "avg_chunk_len": 400,
            "fields": {"title_contains": True, "created_after": True},
        }
        p2 = {
            "kb_id": 2,
            "doc_count": 20,
            "titles": ["Doc B"],
            "content_types": ["text/plain"],
            "date_range": {"min": "2026-02-01", "max": "2026-06-01"},
            "avg_chunk_len": 600,
            "fields": {"file_name_contains": True},
        }
        merged = merge_profiles([p1, p2])
        assert merged["doc_count"] == 60
        assert "application/pdf" in merged["content_types"]
        assert "text/plain" in merged["content_types"]
        assert merged["date_range"]["min"] == "2026-01-01"
        assert merged["date_range"]["max"] == "2026-06-01"
        # Weighted avg: (400*40 + 600*20) / 60 = (16000 + 12000) / 60 = 466
        assert merged["avg_chunk_len"] == 466
        # Fields are tracked per-KB
        assert "title_contains" in merged["fields"]
        assert "file_name_contains" in merged["fields"]


class TestComputeProfile:
    def test_empty_kb_returns_empty(self):
        db = MagicMock()
        db.query.return_value.filter.return_value.all.return_value = []
        result = _compute_profile(1, 999, db)
        assert result == {}

    def test_profile_with_docs(self):
        from datetime import datetime, timezone
        dt1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        dt2 = datetime(2026, 6, 1, tzinfo=timezone.utc)

        doc1 = MagicMock()
        doc1.title = "Weekly Update W01"
        doc1.content_type = "application/pdf"
        doc1.created_at = dt1
        doc1.file_name = "weekly_w01.pdf"

        doc2 = MagicMock()
        doc2.title = "Weekly Update W02"
        doc2.content_type = "application/pdf"
        doc2.created_at = dt2
        doc2.file_name = "weekly_w02.pdf"

        chunk = MagicMock()
        chunk.chunk_text = "A" * 500

        db = MagicMock()
        # Document query
        db.query.return_value.filter.return_value.all.return_value = [doc1, doc2]
        # Chunk query
        db.query.return_value.filter.return_value.limit.return_value.all.return_value = [chunk]

        result = _compute_profile(1, 1, db)
        assert result["doc_count"] == 2
        assert "application/pdf" in result["content_types"]
        assert result["avg_chunk_len"] == 500


class TestCacheKey:
    def test_key_format(self):
        key = _cache_key(1, 5)
        assert key == "kb_profile:1:5"


class TestProfileKbAsync:
    @pytest.mark.asyncio
    async def test_profile_kb_uses_cache(self):
        from app.services.agentic_rag.kb_profile import profile_kb

        cached = {"kb_id": 1, "doc_count": 10, "cached": True}
        with patch("app.services.agentic_rag.kb_profile._redis_get", new_callable=AsyncMock, return_value=cached):
            result = await profile_kb(1, 1, MagicMock())
            assert result == cached

    @pytest.mark.asyncio
    async def test_profile_kb_computes_on_miss(self):
        from app.services.agentic_rag.kb_profile import profile_kb

        computed = {"kb_id": 1, "doc_count": 5}
        with patch("app.services.agentic_rag.kb_profile._redis_get", new_callable=AsyncMock, return_value=None), \
             patch("app.services.agentic_rag.kb_profile._compute_profile", return_value=computed), \
             patch("app.services.agentic_rag.kb_profile._redis_set", new_callable=AsyncMock):
            result = await profile_kb(1, 1, MagicMock())
            assert result == computed
