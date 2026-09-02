"""Tests for query intent extraction folded into rewrite_query_node (Phase 6).

Tests the two-line output parsing, malformed output handling with retry,
and the QueryIntent schema validation.
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, patch

from app.services.agentic_rag.utils import (
    _parse_rewrite_with_intent,
    _validate_query_intent,
    resolve_retrieval_query,
)
from app.services.agentic_rag.schemas import QueryIntent


class TestParseRewriteWithIntent:
    def test_valid_two_line_output(self):
        raw = 'What is mutex?\n{"suggested_filters": null, "suggested_sort": null, "suggested_legs": null, "reasoning": ""}'
        q, i = _parse_rewrite_with_intent(raw)
        assert q == "What is mutex?"
        assert i is not None
        assert '"suggested_filters": null' in i

    def test_single_line_no_intent(self):
        raw = "What is mutex?"
        q, i = _parse_rewrite_with_intent(raw)
        assert q == "What is mutex?"
        assert i is None

    def test_markdown_fences_stripped(self):
        raw = '```\nWhat is mutex?\n{"suggested_filters": null}\n```'
        q, i = _parse_rewrite_with_intent(raw)
        assert "mutex" in q
        assert i is not None

    def test_empty_output(self):
        q, i = _parse_rewrite_with_intent("")
        assert q == ""
        assert i is None

    def test_json_not_on_second_line(self):
        raw = 'What is mutex?\nSome explanation\n{"suggested_filters": null}'
        q, i = _parse_rewrite_with_intent(raw)
        assert q == "What is mutex?"
        assert i is not None  # finds JSON on third line

    def test_no_json_found(self):
        raw = "What is mutex?\nThis is a test"
        q, i = _parse_rewrite_with_intent(raw)
        assert i is None


class TestValidateQueryIntent:
    def test_valid_intent(self):
        raw = '{"suggested_filters": {"title_contains": "Weekly"}, "suggested_sort": null, "suggested_legs": null, "reasoning": "test"}'
        result = _validate_query_intent(raw)
        assert result is not None
        assert result["suggested_filters"] == {"title_contains": "Weekly"}

    def test_null_values(self):
        raw = '{"suggested_filters": null, "suggested_sort": null, "suggested_legs": null, "reasoning": ""}'
        result = _validate_query_intent(raw)
        assert result is not None
        assert result["suggested_filters"] is None

    def test_invalid_json(self):
        result = _validate_query_intent("{not valid json}")
        assert result is None

    def test_non_dict_json(self):
        result = _validate_query_intent('["array", "not", "dict"]')
        assert result is None

    def test_unknown_keys_rejected(self):
        raw = '{"suggested_filters": null, "unknown_field": "bad"}'
        result = _validate_query_intent(raw)
        assert result is None


class TestQueryIntentSchema:
    def test_default_values(self):
        qi = QueryIntent()
        assert qi.suggested_filters is None
        assert qi.suggested_sort is None
        assert qi.suggested_legs is None
        assert qi.reasoning == ""

    def test_with_filters(self):
        qi = QueryIntent(
            suggested_filters={"title_contains": "Weekly Update"},
            suggested_sort={"field": "created_at", "direction": "desc"},
            reasoning="User asked for latest weekly update",
        )
        assert qi.suggested_filters == {"title_contains": "Weekly Update"}
        assert qi.suggested_sort["field"] == "created_at"


class TestResolveRetrievalQueryIntent:
    @pytest.mark.asyncio
    async def test_self_contained_skips_intent(self):
        """Self-contained query should skip LLM call and return None intent."""
        rewritten, provenance, intent = await resolve_retrieval_query(
            query="what is mutex?",
            original_query="what is mutex?",
            recent_history=[],
            kb_profile_text="[KB Profile]\nDocuments: 10",
        )
        assert rewritten == "what is mutex?"
        assert provenance["reason"] == "self_contained"
        assert intent is None

    @pytest.mark.asyncio
    async def test_malformed_output_triggers_retry(self):
        """Malformed intent output should trigger one retry."""
        from langchain_core.messages import HumanMessage

        history = [HumanMessage(content="tell me about Linux")]
        # First call: no JSON (malformed). Second call: valid JSON.
        call_count = 0

        async def mock_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "How does Linux differ?"
            return 'How does Linux differ?\n{"suggested_filters": null, "suggested_sort": null, "suggested_legs": null, "reasoning": ""}'

        with patch("app.services.agentic_rag.utils._call_rewriter", new=AsyncMock(side_effect=mock_call)):
            rewritten, provenance, intent = await resolve_retrieval_query(
                query="how does it differ?",
                original_query="how does it differ?",
                recent_history=history,
                provenance_sources=["tell me about Linux"],
                kb_profile_text="[KB Profile]\nDocuments: 10",
            )
        assert call_count == 2  # Initial + retry
        assert intent is not None
        assert intent["suggested_filters"] is None

    @pytest.mark.asyncio
    async def test_retry_also_malformed_returns_none(self):
        """If retry also produces malformed output, intent should be None."""
        from langchain_core.messages import HumanMessage

        history = [HumanMessage(content="tell me about Linux")]

        async def mock_call(*args, **kwargs):
            return "How does Linux differ?"  # No JSON on either call

        with patch("app.services.agentic_rag.utils._call_rewriter", new=AsyncMock(side_effect=mock_call)):
            rewritten, provenance, intent = await resolve_retrieval_query(
                query="how does it differ?",
                original_query="how does it differ?",
                recent_history=history,
                provenance_sources=["tell me about Linux"],
                kb_profile_text="[KB Profile]\nDocuments: 10",
            )
        assert intent is None
        # Provenance validation passes because "Linux" is in provenance sources
        assert rewritten == "How does Linux differ?"

    @pytest.mark.asyncio
    async def test_no_kb_profile_no_intent(self):
        """Without KB profile text, intent should always be None."""
        from langchain_core.messages import HumanMessage

        history = [HumanMessage(content="tell me about Linux")]

        with patch("app.services.agentic_rag.utils._call_rewriter", new=AsyncMock(return_value="How does Windows differ from Linux?")):
            rewritten, provenance, intent = await resolve_retrieval_query(
                query="how does it differ?",
                original_query="how does it differ?",
                recent_history=history,
                provenance_sources=["tell me about Linux"],
                kb_profile_text="",  # No KB profile
            )
        assert intent is None
