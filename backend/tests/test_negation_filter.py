"""Tests for negation extraction and post-filtering (Phase 1).

Tests the regex-based _extract_negation_terms function (ported from
retrievalagent) and the excluded_terms flow from rewrite_query_node
through rag_retrieve post-filtering to finalize_node prompt injection.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from app.services.agentic_rag.nodes import (
    _extract_negation_terms,
    _content_contains_exclusion,
)


class TestExtractNegationTerms:
    """Unit tests for the regex negation extractor."""

    def test_english_but_not(self):
        assert _extract_negation_terms("networking but not Linux") == ["Linux"]

    def test_english_without(self):
        assert _extract_negation_terms("documents without references") == ["references"]

    def test_english_except(self):
        assert _extract_negation_terms("all chapters except appendix") == ["appendix"]

    def test_english_not_from(self):
        assert _extract_negation_terms("results not from June") == ["June"]

    def test_german_aber_nicht(self):
        result = _extract_negation_terms("Netzwerk aber nicht Linux")
        assert "Linux" in result

    def test_german_ohne(self):
        result = _extract_negation_terms("Dokumente ohne Referenzen")
        assert "Referenzen" in result

    def test_french_sans(self):
        result = _extract_negation_terms("documents sans références")
        assert "références" in result

    def test_french_mais_pas(self):
        result = _extract_negation_terms("réseau mais pas Linux")
        assert "Linux" in result

    def test_italian_senza(self):
        result = _extract_negation_terms("documenti senza riferimenti")
        assert "riferimenti" in result

    def test_italian_ma_non(self):
        result = _extract_negation_terms("rete ma non Linux")
        assert "Linux" in result

    def test_no_negation_returns_empty(self):
        assert _extract_negation_terms("what is networking") == []
        assert _extract_negation_terms("") == []

    def test_multiple_negations(self):
        result = _extract_negation_terms("networking but not Linux and without Windows")
        assert "Linux" in result
        assert "Windows" in result

    def test_stops_at_conjunction(self):
        result = _extract_negation_terms("networking but not Linux and TCP")
        # "and" is a halt word, so "TCP" should not be included
        assert result == ["Linux"]

    def test_strips_leading_articles(self):
        result = _extract_negation_terms("documents but not the appendix")
        # "the" is a stopword, should be stripped
        assert result == ["appendix"]

    def test_multi_word_term(self):
        result = _extract_negation_terms("networking but not TCP protocol")
        assert "TCP protocol" in result

    def test_deduplication(self):
        result = _extract_negation_terms("networking but not Linux without Linux")
        # Should only have "Linux" once
        assert result == ["Linux"]

    def test_prefix_deduplication(self):
        # "Linux kernel" should be dropped if "Linux" is already extracted
        result = _extract_negation_terms("networking but not Linux kernel without Linux")
        assert "Linux" in result
        assert "Linux kernel" not in result


class TestContentContainsExclusion:
    """Unit tests for the exclusion content matcher."""

    def test_simple_match(self):
        assert _content_contains_exclusion("Linux is great", "Linux") is True

    def test_case_insensitive(self):
        assert _content_contains_exclusion("LINUX is great", "linux") is True

    def test_no_match(self):
        assert _content_contains_exclusion("Windows is great", "Linux") is False

    def test_camel_case_spacing(self):
        # Value "LinuxKernel" (camelCase) should match text "Linux Kernel"
        assert _content_contains_exclusion("uses Linux Kernel internally", "LinuxKernel") is True

    def test_empty_text(self):
        assert _content_contains_exclusion("", "Linux") is False

    def test_empty_value(self):
        assert _content_contains_exclusion("some text", "") is True  # empty string is in any string


class TestExcludedTermsPostFilter:
    """Tests for the _apply_excluded_terms_filter function in rag_retrieve."""

    def test_filter_drops_matching_docs(self):
        from app.services.agentic_rag.tools.rag_retrieve import _apply_excluded_terms_filter

        ctx = MagicMock()
        ctx.state = {"excluded_terms": ["Linux"]}

        state = {
            "retrieved_docs": [
                {"page_content": "Linux networking guide", "metadata": {}},
                {"page_content": "Windows networking guide", "metadata": {}},
                {"page_content": "macOS networking guide", "metadata": {}},
            ]
        }
        _apply_excluded_terms_filter(state, ctx)
        assert len(state["retrieved_docs"]) == 2
        assert "Linux" not in state["retrieved_docs"][0]["page_content"]
        assert "Linux" not in state["retrieved_docs"][1]["page_content"]

    def test_no_excluded_terms_is_noop(self):
        from app.services.agentic_rag.tools.rag_retrieve import _apply_excluded_terms_filter

        ctx = MagicMock()
        ctx.state = {"excluded_terms": []}

        state = {"retrieved_docs": [{"page_content": "Linux", "metadata": {}}]}
        _apply_excluded_terms_filter(state, ctx)
        assert len(state["retrieved_docs"]) == 1

    def test_empty_docs_is_noop(self):
        from app.services.agentic_rag.tools.rag_retrieve import _apply_excluded_terms_filter

        ctx = MagicMock()
        ctx.state = {"excluded_terms": ["Linux"]}

        state = {"retrieved_docs": []}
        _apply_excluded_terms_filter(state, ctx)
        assert state["retrieved_docs"] == []

    def test_filter_checks_title_too(self):
        from app.services.agentic_rag.tools.rag_retrieve import _apply_excluded_terms_filter

        ctx = MagicMock()
        ctx.state = {"excluded_terms": ["Linux"]}

        state = {
            "retrieved_docs": [
                {"page_content": "networking guide", "metadata": {"_title": "Linux Networking"}},
                {"page_content": "networking guide", "metadata": {"_title": "Windows Networking"}},
            ]
        }
        _apply_excluded_terms_filter(state, ctx)
        assert len(state["retrieved_docs"]) == 1
        assert "Linux" not in state["retrieved_docs"][0]["metadata"]["_title"]

    def test_no_ctx_state_is_noop(self):
        from app.services.agentic_rag.tools.rag_retrieve import _apply_excluded_terms_filter

        ctx = MagicMock()
        ctx.state = None

        state = {"retrieved_docs": [{"page_content": "Linux", "metadata": {}}]}
        _apply_excluded_terms_filter(state, ctx)
        assert len(state["retrieved_docs"]) == 1
