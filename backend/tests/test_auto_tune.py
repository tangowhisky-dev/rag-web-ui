"""
Tests for the auto-tuning service.

Covers config generation, settings patching, scoring, persistence, and
convergence behavior.
"""

import json
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from app.services.auto_tune import (
    TunableParams,
    _SettingsPatcher,
    _token_f1,
    _retrieval_hit,
    run_tuning_loop,
    save_best_config,
    load_best_config,
    EvalResult,
    TuningResult,
)
from app.core.config import settings


# ── Config generation tests ────────────────────────────────────────────────────

def test_tunable_params_defaults():
    """Default params match current settings."""
    params = TunableParams()
    assert params.dense_weight == 0.5
    assert params.sparse_weight == 0.3
    assert params.exact_weight == 0.2
    assert params.top_k == 10
    assert params.reranker_threshold == -2.0
    assert params.rrf_k == 60


def test_tunable_params_to_dict():
    """to_dict returns clean dict without class constants."""
    params = TunableParams()
    d = params.to_dict()
    assert "DENSE_WEIGHT_RANGE" not in d
    assert set(d.keys()) == {"dense_weight", "sparse_weight", "exact_weight", "top_k", "reranker_threshold", "rrf_k"}


def test_random_sample_within_bounds():
    """Random sample stays within defined bounds."""
    for _ in range(20):
        params = TunableParams.random_sample()
        assert 0.1 <= params.dense_weight <= 0.9
        assert 0.1 <= params.sparse_weight <= 0.9
        assert 0.0 <= params.exact_weight <= 0.5
        assert 5 <= params.top_k <= 20
        assert -5.0 <= params.reranker_threshold <= 5.0
        assert 30 <= params.rrf_k <= 90


def test_grid_candidates_count():
    """Grid candidates respects n parameter."""
    candidates = TunableParams.grid_candidates(5)
    assert len(candidates) <= 5


def test_grid_candidates_unique():
    """Grid candidates are unique."""
    candidates = TunableParams.grid_candidates(50)
    dicts = [c.to_dict() for c in candidates]
    assert len(dicts) == len(set(json.dumps(d, sort_keys=True) for d in dicts))


# ── Settings patcher tests ─────────────────────────────────────────────────────

def test_settings_patcher_applies_and_restores():
    """Patcher applies config and restores after."""
    original_dense = settings.HYBRID_DENSE_WEIGHT
    original_k = 60
    
    params = TunableParams(dense_weight=0.8, sparse_weight=0.1, exact_weight=0.1, top_k=15, reranker_threshold=1.0, rrf_k=40)
    
    with _SettingsPatcher(params):
        assert settings.HYBRID_DENSE_WEIGHT == 0.8
        assert settings.HYBRID_SPARSE_WEIGHT == 0.1
        assert settings.HYBRID_EXACT_WEIGHT == 0.1
        assert settings.RETRIEVAL_TOP_K == 15
        assert settings.RERANKER_SCORE_THRESHOLD == 1.0
        
        import app.services.retrieval as retrieval_mod
        assert retrieval_mod._RRF_K == 40
    
    # Restored
    assert settings.HYBRID_DENSE_WEIGHT == original_dense
    import app.services.retrieval as retrieval_mod
    assert retrieval_mod._RRF_K == original_k


def test_settings_patcher_restores_on_exception():
    """Patcher restores settings even if an exception occurs."""
    params = TunableParams(dense_weight=0.9)
    
    try:
        with _SettingsPatcher(params):
            assert settings.HYBRID_DENSE_WEIGHT == 0.9
            raise ValueError("test")
    except ValueError:
        pass
    
    # Should be restored
    assert settings.HYBRID_DENSE_WEIGHT != 0.9


# ── Scoring tests ──────────────────────────────────────────────────────────────

def test_token_f1_perfect_match():
    """Perfect match returns F1=1.0."""
    f1 = _token_f1("the answer is 42", ["the answer is 42"])
    assert f1 == 1.0


def test_token_f1_partial_match():
    """Partial match returns F1 between 0 and 1."""
    f1 = _token_f1("the answer is 42", ["the answer is 43"])
    assert 0.0 < f1 < 1.0


def test_token_f1_no_match():
    """No match returns F1=0.0."""
    f1 = _token_f1("hello world", ["goodbye moon"])
    assert f1 == 0.0


def test_token_f1_case_insensitive():
    """F1 is case-insensitive."""
    f1 = _token_f1("The Answer", ["the answer"])
    assert f1 == 1.0


def test_retrieval_hit_found():
    """Retrieval hit returns 1.0 when answer is in context."""
    hit = _retrieval_hit(["the capital is Paris"], ["Paris"])
    assert hit == 1.0


def test_retrieval_hit_not_found():
    """Retrieval hit returns 0.0 when answer is not in context."""
    hit = _retrieval_hit(["the capital is London"], ["Paris"])
    assert hit == 0.0


# ── Persistence tests ──────────────────────────────────────────────────────────

def test_save_and_load_config():
    """Config save/load round-trip."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        output_path = f.name
    
    try:
        params = TunableParams(dense_weight=0.7, sparse_weight=0.2, exact_weight=0.1, top_k=15)
        result = TuningResult(
            best_config=params,
            best_result=EvalResult(mean_f1=0.85, mean_em=0.7, hit_rate=0.9),
            history=[{"iteration": 1, "f1": 0.85}],
            n_iterations=1,
            converged_at=1,
        )
        
        save_best_config(result, kb_id=42, output_path=output_path)
        
        loaded = load_best_config(output_path)
        assert loaded is not None
        assert loaded["dense_weight"] == 0.7
        assert loaded["sparse_weight"] == 0.2
        assert loaded["top_k"] == 15
    finally:
        os.unlink(output_path)


def test_load_nonexistent_config():
    """Loading nonexistent config returns None."""
    result = load_best_config("/nonexistent/path/best_config.json")
    assert result is None


# ── Tuning loop tests ──────────────────────────────────────────────────────────

def test_tuning_loop_returns_result():
    """Tuning loop returns valid result structure."""
    questions = [
        {"question": "What is RRF?", "answers": ["Reciprocal Rank Fusion"]},
    ]
    
    # Mock the eval to avoid HTTP calls
    with patch("app.services.auto_tune._run_eval") as mock_eval:
        mock_eval.return_value = EvalResult(mean_f1=0.5, mean_em=0.3, hit_rate=0.6)
        
        result = run_tuning_loop(
            questions=questions,
            kb_id=1,
            base_url="http://localhost:8000/api",
            max_iterations=3,
            patience=2,
            seed=42,
        )
    
    assert result.best_config is not None
    assert result.best_result is not None
    assert len(result.history) >= 1
    assert result.n_iterations >= 1


def test_tuning_loop_converges():
    """Tuning loop converges when F1 stops improving."""
    questions = [{"question": "test", "answers": ["test"]}]
    
    with patch("app.services.auto_tune._run_eval") as mock_eval:
        # Return constant F1 — should converge quickly
        mock_eval.return_value = EvalResult(mean_f1=0.5)
        
        result = run_tuning_loop(
            questions=questions,
            kb_id=1,
            base_url="http://localhost:8000/api",
            max_iterations=20,
            patience=3,
            seed=42,
        )
    
    assert result.converged_at is not None
    assert result.converged_at <= 20


def test_tuning_loop_fallback_on_all_failures():
    """Tuning loop returns defaults when all evals fail."""
    questions = [{"question": "test", "answers": ["test"]}]
    
    with patch("app.services.auto_tune._run_eval") as mock_eval:
        mock_eval.return_value = EvalResult(errors=1)
        
        result = run_tuning_loop(
            questions=questions,
            kb_id=1,
            base_url="http://localhost:8000/api",
            max_iterations=2,
            patience=2,
            seed=42,
        )
    
    assert result.best_config is not None
    assert result.n_iterations >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
