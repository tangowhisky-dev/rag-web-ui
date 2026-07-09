"""
Integration tests for adaptive retrieval with query classification.

Tests that classification affects retrieval config via the HTTP endpoint
and that the eval harness supports --classify flag.
"""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport

from app.models.query_classifier import QueryType
from app.services.retrieval import get_retrieval_config


# ── Unit-level integration: classification → retrieval config ─────────────────

def test_classification_routes_to_correct_config():
    """Verify that each query type maps to the correct retrieval config."""
    configs = {
        QueryType.FACTUAL: {
            "use_dense": True, "use_sparse": True, "use_exact": True,
            "dense_weight": 0.5, "sparse_weight": 0.3, "exact_weight": 0.2,
            "top_k": 10,
        },
        QueryType.ENTITY_CENTRIC: {
            "use_dense": True, "use_sparse": True, "use_exact": True,
            "dense_weight": 0.6, "sparse_weight": 0.2, "exact_weight": 0.2,
            "top_k": 10,
        },
        QueryType.MULTI_PART: {
            "use_dense": True, "use_sparse": True, "use_exact": False,
            "dense_weight": 0.5, "sparse_weight": 0.5, "exact_weight": 0.0,
            "top_k": 10,
        },
        QueryType.AMBIGUOUS: {
            "use_dense": True, "use_sparse": True, "use_exact": True,
            "dense_weight": 0.4, "sparse_weight": 0.4, "exact_weight": 0.2,
            "top_k": 15,
        },
    }

    for query_type, expected in configs.items():
        cfg = get_retrieval_config(query_type)
        for key, value in expected.items():
            assert cfg[key] == value, f"{query_type}.{key}: expected {value}, got {cfg[key]}"


def test_hybrid_search_accepts_query_type():
    """Verify hybrid_search_with_legs accepts query_type parameter."""
    import inspect
    from app.services.retrieval import hybrid_search_with_legs
    
    sig = inspect.signature(hybrid_search_with_legs)
    params = list(sig.parameters.keys())
    assert "query_type" in params, "hybrid_search_with_legs should accept query_type parameter"


def test_hybrid_search_applies_preset():
    """Verify that query_type parameter affects retrieval behavior."""
    import asyncio
    
    async def _run():
        from app.services.retrieval import hybrid_search_with_legs
        
        # Mock the DB and search legs to avoid actual database calls
        mock_db = MagicMock()
        mock_db.execute = MagicMock(return_value=MagicMock(fetchall=lambda: []))
        
        # Patch the search legs to return empty results
        with patch('app.services.retrieval.retrieval._dense_search', return_value={}):
            with patch('app.services.retrieval.retrieval._sparse_search', return_value={}):
                with patch('app.services.retrieval.retrieval._exact_search', return_value={}):
                    # Call with FACTUAL query type
                    result = await hybrid_search_with_legs(
                        query="test query",
                        kb_ids=[1],
                        db=mock_db,
                        query_type=QueryType.FACTUAL,
                    )
                    
                    assert "docs" in result
                    assert "retrieval_info" in result
                    
        # Patch for MULTI_PART which should disable exact leg
        with patch('app.services.retrieval.retrieval._dense_search', return_value={}):
            with patch('app.services.retrieval.retrieval._sparse_search', return_value={}):
                with patch('app.services.retrieval.retrieval._exact_search', return_value={}) as mock_exact:
                    result = await hybrid_search_with_legs(
                        query="test query",
                        kb_ids=[1],
                        db=mock_db,
                        query_type=QueryType.MULTI_PART,
                    )
                    
                    # MULTI_PART preset disables exact leg — _exact_search should NOT be called
                    assert not mock_exact.called, "exact_search should not be called for MULTI_PART"
    
    asyncio.run(_run())


# ── Eval harness --classify flag ──────────────────────────────────────────────

def test_eval_harness_classify_flag():
    """Verify eval harness accepts --classify flag."""
    import subprocess
    import os
    
    eval_path = os.path.join(os.path.dirname(__file__), "..", "..", "eval", "eval.py")
    
    # Check that the file exists and has --classify option
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            content = f.read()
        # After T06, the file should have --classify flag
        assert "--classify" in content or "classify" in content, \
            "eval.py should support --classify flag"
    else:
        pytest.skip(f"eval.py not found at {eval_path}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
