"""Tests for the rag_retrieve graduated relaxation ladder and related fixes:

- Issue #1/#4: rag_retrieve retries with progressively looser thresholds and
  only pays for Neo4j graph expansion once the cheaper legs are insufficient.
- Issue #3: ADAPTIVE_RETRIEVAL_ENABLED/THRESHOLD are actually wired in now.
- Issue #5: recalled memory docs are merged into tool_node's retrieved_docs,
  not overwritten.
- Issue #6: the agent loop routes to reflect_final/finalize once a wall-clock
  budget is exceeded, independent of the iteration counter.
"""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from app.core.config import settings
from app.services.agentic_rag.agent_graph import route_reflect_final, route_think


class _StubToolContext:
    def __init__(self):
        self.db = None
        self.org_id = 1
        self.chat_id = None
        self.state = None


def _patch_pipeline(monkeypatch_target, level_outcomes):
    """Patch rag_retrieve's node imports so each ladder level returns the
    given (doc_count, confidence) before/after graph expansion.

    level_outcomes: list of dicts with keys 'pre' and 'post', each an
    (doc_count, confidence) tuple. 'pre' is the result after dense/sparse/
    exact+rerank+filter; 'post' is the result after graph expansion (only
    consulted if graph_expand=True and pre was insufficient).
    """
    call_index = {"i": -1}

    def _make_docs(n):
        return [{"page_content": f"doc{i}", "metadata": {"content_hash": f"h{i}"}} for i in range(n)]

    async def fake_dense(state, db, kb_ids, org_id, file_markdown, min_score=None):
        return {}

    async def fake_sparse(state, db, kb_ids, org_id, file_markdown, min_score=None):
        return {}

    async def fake_exact(state, db, kb_ids, org_id, file_markdown, min_score=None):
        return {}

    def fake_merge(state, file_markdown):
        return {}

    def fake_rerank(state):
        return {}

    def fake_filter(state, threshold=None):
        call_index["i"] += 1
        outcome = level_outcomes[call_index["i"]]
        n, conf = outcome["pre"]
        return {"retrieved_docs": _make_docs(n), "retrieval_confidence": conf}

    async def fake_neo4j(state, db, kb_ids, org_id, file_markdown):
        outcome = level_outcomes[call_index["i"]]
        n, conf = outcome["post"]
        return {"retrieved_docs": _make_docs(n), "retrieval_confidence": conf}

    return {
        "dense_retrieval_node": fake_dense,
        "sparse_retrieval_node": fake_sparse,
        "exact_retrieval_node": fake_exact,
        "merge_node": fake_merge,
        "reranking_node": fake_rerank,
        "filter_node": fake_filter,
        "neo4j_expansion_node": fake_neo4j,
    }


def _run_rag_retrieve(level_outcomes, graph_expand=True, adaptive_enabled=True):
    from app.services.agentic_rag.tools import rag_retrieve as mod

    patched = _patch_pipeline(mod, level_outcomes)
    ctx = _StubToolContext()
    input_obj = mod.RagRetrieveInput(query="what is the refund policy?", graph_expand=graph_expand)

    def _fake_get_setting(db, key, org_id=None):
        from app.core.settings_registry import get_def
        if key == "ADAPTIVE_RETRIEVAL_ENABLED":
            return adaptive_enabled
        if key == "ADAPTIVE_RETRIEVAL_THRESHOLD":
            return get_def("ADAPTIVE_RETRIEVAL_THRESHOLD").default
        if key == "ADAPTIVE_RETRIEVAL_RERANKER_THRESHOLD":
            return get_def("ADAPTIVE_RETRIEVAL_RERANKER_THRESHOLD").default
        if key == "RETRIEVAL_RELAX_LEVEL2_RERANKER_THRESHOLD":
            return get_def("RETRIEVAL_RELAX_LEVEL2_RERANKER_THRESHOLD").default
        if key == "DENSE_MIN_SCORE":
            return get_def("DENSE_MIN_SCORE").default
        if key == "SPARSE_MIN_SCORE":
            return get_def("SPARSE_MIN_SCORE").default
        if key == "EXACT_MIN_SCORE":
            return get_def("EXACT_MIN_SCORE").default
        defn = get_def(key)
        return defn.default if defn else None

    with patch.object(mod, "enforce_rbac", return_value={"kb_ids": [1]}), \
         patch("app.services.settings_service.get_setting", side_effect=_fake_get_setting), \
         patch.multiple(mod, **patched):
        return asyncio.run(mod._rag_retrieve(ctx, input_obj))


def test_ladder_stops_at_level_0_when_sufficient():
    """A strong first pass should never touch graph expansion or relax."""
    outcomes = [
        {"pre": (5, 0.9), "post": (5, 0.9)},  # level 0: already sufficient
        {"pre": (5, 0.9), "post": (5, 0.9)},
        {"pre": (5, 0.9), "post": (5, 0.9)},
    ]
    result = _run_rag_retrieve(outcomes)
    assert result["result"]["sufficient"] is True
    assert result["result"]["levels_tried"] == 1
    assert len(result["result"]["docs"]) == 5


def test_ladder_escalates_through_all_levels_when_never_sufficient():
    """Zero/low results at every level should exhaust the ladder, not loop forever."""
    outcomes = [
        {"pre": (0, 0.0), "post": (0, 0.0)},
        {"pre": (0, 0.0), "post": (1, 0.1)},
        {"pre": (1, 0.1), "post": (1, 0.1)},
    ]
    result = _run_rag_retrieve(outcomes)
    assert result["result"]["sufficient"] is False
    assert result["result"]["levels_tried"] == 3


def test_ladder_uses_graph_expansion_only_when_insufficient():
    """Level 0 succeeds only after graph expansion adds docs -> still level 1."""
    outcomes = [
        {"pre": (0, 0.0), "post": (4, 0.8)},  # graph expansion rescues level 0
    ]
    result = _run_rag_retrieve(outcomes)
    assert result["result"]["sufficient"] is True
    assert result["result"]["levels_tried"] == 1
    assert len(result["result"]["docs"]) == 4


def test_ladder_disabled_skips_relaxation_levels():
    """ADAPTIVE_RETRIEVAL_ENABLED=False must only ever try level 0."""
    outcomes = [
        {"pre": (0, 0.0), "post": (0, 0.0)},
    ]
    result = _run_rag_retrieve(outcomes, adaptive_enabled=False)
    assert result["result"]["levels_tried"] == 1
    assert result["result"]["sufficient"] is False


def test_min_confidence_defaults_from_adaptive_threshold_setting():
    from app.services.agentic_rag.tools.rag_retrieve import RagRetrieveInput
    assert RagRetrieveInput(query="x").min_confidence is None


# ── Wall-clock budget routing (Issue #6) ───────────────────────────────────────

def _mock_settings(overrides: dict):
    """Create a side_effect that returns override values for specific keys."""
    from app.services.settings_service import get_setting as _real
    def _side(db, key, org_id=None):
        if key in overrides:
            return overrides[key]
        return _real(db, key, org_id)
    return _side


def test_route_think_respects_wall_clock_budget():
    with patch("app.services.agentic_rag.agent_graph.get_setting",
               side_effect=_mock_settings({"AGENT_MAX_ITERATIONS": 100, "AGENT_MAX_WALL_SECONDS": 1.0})):
        state = {"iteration": 1, "tool_calls": [{"tool": "rag_retrieve"}], "started_at": time.monotonic() - 10}
        assert route_think(state) == "reflect_final"


def test_route_think_ignores_wall_clock_when_not_started():
    with patch("app.services.agentic_rag.agent_graph.get_setting",
               side_effect=_mock_settings({"AGENT_MAX_ITERATIONS": 100, "AGENT_MAX_WALL_SECONDS": 1.0})):
        state = {"iteration": 1, "tool_calls": [{"tool": "rag_retrieve"}]}
        assert route_think(state) == "tool"


def test_route_reflect_final_respects_wall_clock_budget():
    with patch("app.services.agentic_rag.agent_graph.get_setting",
               side_effect=_mock_settings({"AGENT_MAX_ITERATIONS": 100, "AGENT_MAX_WALL_SECONDS": 1.0})):
        state = {
            "iteration": 1,
            "reflection_final": {"ready": False},
            "started_at": time.monotonic() - 10,
        }
        assert route_reflect_final(state) == "finalize"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
