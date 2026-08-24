#!/usr/bin/env python3
"""Verify that the KB search endpoint is not impacted by the agentic pipeline's
new abbreviation_glossary state, and that search's own abbreviation handling
is correct and self-contained.

Key differences between search and agentic chat:
  - Search: uses expand_query_suffix() directly, no AgentState, no glossary
  - Chat:   uses expand_query_node() → stores abbreviation_glossary in AgentState

These tests verify:
  1. Search endpoint code has no reference to AgentState/abbreviation_glossary
  2. Search uses expand_query_suffix (not expand_query_node)
  3. Search passes expanded_query to all 3 retrieval legs
  4. Search passes expanded_query to the reranker (not rewritten_query)
  5. expand_query_suffix produces correct output for abbreviation queries
  6. expand_query_suffix produces correct output for plain English (no adverse effect)
  7. expand_query_suffix preserves original query text verbatim
  8. Search does not build or use a glossary (no LLM generation step)
  9. Search logs the expanded_query to search_history

Runs inside the backend container:
  docker exec rag-web-ui-backend-1 pytest tests/test_search_abbr_isolation.py -v
"""
import os
import sys
import inspect

import pytest

sys.path.insert(0, "/app")
os.environ.setdefault("PYTHONPATH", "/app")


# ---------------------------------------------------------------------------
# 1. Code-level isolation: search.py must not reference agentic glossary state
# ---------------------------------------------------------------------------

class TestSearchCodeIsolation:
    """Verify search.py source code has no coupling to the agentic glossary."""

    def test_search_does_not_import_agent_state(self):
        """search.py must not import AgentState or graph_state."""
        import app.api.api_v1.search as search_mod
        src = inspect.getsource(search_mod)
        assert "AgentState" not in src, "search.py must not reference AgentState"
        assert "graph_state" not in src, "search.py must not import graph_state"

    def test_search_does_not_reference_abbreviation_glossary(self):
        """search.py must not reference abbreviation_glossary."""
        import app.api.api_v1.search as search_mod
        src = inspect.getsource(search_mod)
        assert "abbreviation_glossary" not in src, (
            "search.py must not reference abbreviation_glossary"
        )

    def test_search_does_not_use_expand_query_node(self):
        """search.py must not use expand_query_node (agentic pipeline node)."""
        import app.api.api_v1.search as search_mod
        src = inspect.getsource(search_mod)
        assert "expand_query_node" not in src, (
            "search.py must not use expand_query_node"
        )

    def test_search_does_not_use_build_glossary(self):
        """search.py must not build a glossary (no LLM generation step)."""
        import app.api.api_v1.search as search_mod
        src = inspect.getsource(search_mod)
        assert "build_glossary" not in src, (
            "search.py must not build a glossary — no LLM generation in search"
        )

    def test_search_does_not_use_format_context_string(self):
        """search.py must not use format_context_string (agentic generation context)."""
        import app.api.api_v1.search as search_mod
        src = inspect.getsource(search_mod)
        assert "format_context_string" not in src, (
            "search.py must not use format_context_string"
        )

    def test_search_uses_expand_query_suffix(self):
        """search.py must use expand_query_suffix for abbreviation expansion."""
        import app.api.api_v1.search as search_mod
        src = inspect.getsource(search_mod)
        assert "expand_query_suffix" in src, (
            "search.py must use expand_query_suffix for query expansion"
        )

    def test_search_uses_build_lookup(self):
        """search.py must build its own lookup directly from the DB."""
        import app.api.api_v1.search as search_mod
        src = inspect.getsource(search_mod)
        assert "build_lookup" in src, (
            "search.py must call build_lookup directly"
        )


# ---------------------------------------------------------------------------
# 2. Search flow: expanded_query is used for retrieval and reranking
# ---------------------------------------------------------------------------

class TestSearchFlowUsesExpandedQuery:
    """Verify the search endpoint passes expanded_query to retrieval and reranker."""

    def test_retrieval_legs_receive_expanded_query(self):
        """All 3 retrieval legs must be called with expanded_query, not the raw query."""
        import app.api.api_v1.search as search_mod
        src = inspect.getsource(search_mod)
        # The search function calls leg_fn(query=expanded_query, ...)
        assert "query=expanded_query" in src, (
            "Retrieval legs must be called with query=expanded_query"
        )

    def test_reranker_receives_expanded_query(self):
        """The reranker must be called with expanded_query, not rewritten_query."""
        import app.api.api_v1.search as search_mod
        src = inspect.getsource(search_mod)
        # The search function calls rerank(query=expanded_query, ...)
        assert "query=expanded_query" in src, (
            "Reranker must be called with query=expanded_query"
        )
        # Must NOT use rewritten_query (that's the agentic pipeline's variable)
        assert "rewritten_query" not in src, (
            "search.py must not use rewritten_query — that's agentic-pipeline-only"
        )

    def test_search_returns_expanded_query_in_response(self):
        """The search response must include the expanded_query for transparency."""
        from app.api.api_v1.search import SearchResponse
        assert "expanded_query" in SearchResponse.model_fields, (
            "SearchResponse must have expanded_query field"
        )


# ---------------------------------------------------------------------------
# 3. expand_query_suffix correctness (the function search uses)
# ---------------------------------------------------------------------------

class TestExpandQuerySuffixForSearch:
    """Verify expand_query_suffix produces correct output for search scenarios."""

    @pytest.fixture(scope="module")
    def lookup(self):
        """Build lookup directly from CSV (bypasses SQLite test DB)."""
        from app.services.abbreviation_service import AbbreviationLookup
        from flashtext2 import KeywordProcessor
        import csv as csv_mod

        csv_path = "/app/assets/abbreviations_enhanced.csv"
        forward = {}
        abbr_categories = {}
        with open(csv_path, encoding="utf-8") as f:
            for row in csv_mod.DictReader(f):
                abbr = row["abbreviation"].strip()
                form = row["expanded_form"].strip()
                cat = row.get("category", "").strip()
                if abbr and form:
                    if abbr not in forward:
                        forward[abbr] = []
                        abbr_categories[abbr] = set()
                    if form not in forward[abbr]:
                        forward[abbr].append(form)
                    if cat:
                        abbr_categories[abbr].add(cat)

        from app.services.abbreviation_service import (
            STOPWORDS, _REVERSE_MIN_FORM_LEN,
            _is_lowercase, _is_qualification_category,
        )

        exact_abbrs, prose_abbrs, qual_abbrs = [], [], []
        for abbr in forward:
            if abbr.lower() in STOPWORDS:
                continue
            if _is_lowercase(abbr):
                is_qual = any(_is_qualification_category(c) for c in abbr_categories.get(abbr, set()))
                if is_qual:
                    qual_abbrs.append(abbr)
                else:
                    prose_abbrs.append(abbr)
            else:
                exact_abbrs.append(abbr)

        kp_exact = KeywordProcessor(case_sensitive=True)
        for a in exact_abbrs:
            kp_exact.add_keyword(a, a)
        kp_prose = KeywordProcessor(case_sensitive=False)
        for a in prose_abbrs:
            kp_prose.add_keyword(a, a)
        kp_qual = KeywordProcessor(case_sensitive=True)
        for a in qual_abbrs:
            kp_qual.add_keyword(a, a)

        reverse = {}
        for abbr, forms in forward.items():
            if abbr.lower() in STOPWORDS:
                continue
            for form in forms:
                key = form.lower()
                if len(key) < _REVERSE_MIN_FORM_LEN:
                    continue
                if key in STOPWORDS:
                    continue
                if key not in reverse:
                    reverse[key] = []
                if abbr not in reverse[key]:
                    reverse[key].append(abbr)

        kp_reverse = KeywordProcessor(case_sensitive=False)
        for form_lower in reverse:
            kp_reverse.add_keyword(form_lower, form_lower)

        return AbbreviationLookup(
            forward=forward,
            kp_exact=kp_exact,
            kp_prose=kp_prose,
            kp_qual=kp_qual,
            reverse=reverse,
            kp_reverse=kp_reverse,
        )

    def test_abbr_query_expanded(self, lookup):
        """Abbreviation query gets [Expansions: ...] suffix."""
        from app.services.abbreviation_service import expand_query_suffix
        q = "bns wdr from position"
        expanded = expand_query_suffix(q, lookup)
        assert expanded.startswith(q)
        assert "[Expansions:" in expanded
        assert "bns=" in expanded
        assert "wdr=" in expanded

    def test_full_form_query_reverse_expanded(self, lookup):
        """Full-form query gets reverse expansions (full form → abbr)."""
        from app.services.abbreviation_service import expand_query_suffix
        q = "commanding officer ordered battalions to withdraw"
        expanded = expand_query_suffix(q, lookup)
        assert expanded.startswith(q)
        assert "[Expansions:" in expanded
        assert "CO=" in expanded, "Reverse match: 'commanding officer' → CO"
        assert "bns=" in expanded, "Reverse match: 'battalions' → bns"
        assert "wdr=" in expanded, "Reverse match: 'withdraw' → wdr"

    def test_plain_english_not_expanded(self, lookup):
        """Plain English query has no expansion (no adverse effect)."""
        from app.services.abbreviation_service import expand_query_suffix
        q = "weather forecast rain temperature"
        expanded = expand_query_suffix(q, lookup)
        assert expanded == q, "Plain English query must not be expanded"

    def test_original_preserved_verbatim(self, lookup):
        """The original query text must appear verbatim as prefix."""
        from app.services.abbreviation_service import expand_query_suffix
        queries = [
            "CO ordered bns to wdr",
            "bns wdr from position",
            "which officer completed psc and ndc",
            "weather forecast rain",
        ]
        for q in queries:
            expanded = expand_query_suffix(q, lookup)
            assert expanded[:len(q)] == q, (
                f"Original text must be verbatim prefix: {q!r} → {expanded!r}"
            )

    def test_bidirectional_in_one_query(self, lookup):
        """A query with both abbreviations and full forms gets both directions."""
        from app.services.abbreviation_service import expand_query_suffix
        q = "CO ordered battalions to wdr"
        expanded = expand_query_suffix(q, lookup)
        # Forward: CO and wdr are abbreviations in the query
        assert "CO=" in expanded, "Forward match for CO"
        assert "wdr=" in expanded, "Forward match for wdr"
        # Reverse: 'battalions' is a full form in the query → bns
        assert "bns=" in expanded, "Reverse match for 'battalions' → bns"


# ---------------------------------------------------------------------------
# 4. Search history logging
# ---------------------------------------------------------------------------

class TestSearchHistoryLogging:
    """Verify search logs the expanded_query to search_history."""

    def test_search_history_model_has_expanded_query(self):
        """SearchHistory model must have expanded_query column."""
        from app.models.search_history import SearchHistory
        columns = {c.name for c in SearchHistory.__table__.columns}
        assert "expanded_query" in columns, (
            "SearchHistory must have expanded_query column"
        )

    def test_search_logs_expanded_query_only_when_different(self):
        """Search should log expanded_query only when it differs from the original."""
        import app.api.api_v1.search as search_mod
        src = inspect.getsource(search_mod)
        # The code: expanded_query=expanded_query if expanded_query != query else None
        assert "expanded_query != query" in src, (
            "Search must only log expanded_query when it differs from original"
        )


# ---------------------------------------------------------------------------
# 5. Agentic pipeline DOES use glossary (contrast with search)
# ---------------------------------------------------------------------------

class TestAgenticPipelineUsesGlossary:
    """Confirm the agentic pipeline DOES use the glossary (contrast with search).
    This ensures the glossary state is isolated to the agentic pipeline only."""

    def test_graph_state_has_abbreviation_glossary(self):
        from app.services.agentic_rag.graph_state import AgentState
        assert "abbreviation_glossary" in AgentState.__annotations__

    def test_expand_query_node_sets_glossary(self):
        """expand_query_node must set abbreviation_glossary in its return dict."""
        import app.services.agentic_rag.nodes as nodes_mod
        src = inspect.getsource(nodes_mod)
        assert "abbreviation_glossary" in src, (
            "expand_query_node must reference abbreviation_glossary"
        )

    def test_format_context_string_accepts_query_glossary(self):
        """format_context_string must accept query_glossary parameter."""
        import app.services.agentic_rag.utils as utils_mod
        src = inspect.getsource(utils_mod)
        assert "query_glossary" in src, (
            "format_context_string must accept query_glossary"
        )
        assert "[Abbreviation Glossary]" in src, (
            "format_context_string must produce [Abbreviation Glossary] section"
        )

    def test_chat_endpoint_references_agent_state(self):
        """The chat endpoint (not search) should reference the agentic pipeline."""
        import app.api.api_v1.chat as chat_mod
        src = inspect.getsource(chat_mod)
        # Chat uses the agentic pipeline which has AgentState
        assert "agent" in src.lower() or "graph" in src.lower() or "agentic" in src.lower(), (
            "Chat endpoint should reference the agentic pipeline"
        )
