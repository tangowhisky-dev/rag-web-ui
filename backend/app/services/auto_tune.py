"""
Eval-driven auto-tuning loop for retrieval configuration.

Searches the retrieval config space (RRF weights, top_k, reranker threshold,
RRF_K constant) to find the best configuration measured by F1 score on a
held-out eval dataset.

Usage:
    python -m app.cli.tune --iterations 20 --state ingest_state.json
"""

import json
import logging
import os
import random
import statistics
import time
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings

logger = logging.getLogger(__name__)


# ── Tunable parameters ──────────────────────────────────────────────────────────

@dataclass
class TunableParams:
    """Parameters that can be tuned for the retrieval pipeline."""
    dense_weight: float = 0.5
    sparse_weight: float = 0.3
    exact_weight: float = 0.2
    top_k: int = 10
    reranker_threshold: float = -2.0
    rrf_k: int = 60

    # Search space bounds
    DENSE_WEIGHT_RANGE: Tuple[float, float, float] = (0.1, 0.9, 0.1)
    SPARSE_WEIGHT_RANGE: Tuple[float, float, float] = (0.1, 0.9, 0.1)
    EXACT_WEIGHT_RANGE: Tuple[float, float, float] = (0.0, 0.5, 0.1)
    TOP_K_RANGE: Tuple[int, int, int] = (5, 20, 5)
    RERANKER_THRESHOLD_RANGE: Tuple[float, float, float] = (-5.0, 5.0, 1.0)
    RRF_K_RANGE: Tuple[int, int, int] = (30, 90, 10)

    @classmethod
    def random_sample(cls) -> "TunableParams":
        """Generate a random config within bounds."""
        return cls(
            dense_weight=round(random.uniform(*cls.DENSE_WEIGHT_RANGE[:2]), 1),
            sparse_weight=round(random.uniform(*cls.SPARSE_WEIGHT_RANGE[:2]), 1),
            exact_weight=round(random.uniform(*cls.EXACT_WEIGHT_RANGE[:2]), 1),
            top_k=random.choice(range(*cls.TOP_K_RANGE)),
            reranker_threshold=round(random.uniform(*cls.RERANKER_THRESHOLD_RANGE[:2]), 1),
            rrf_k=random.choice(range(*cls.RRF_K_RANGE)),
        )

    @classmethod
    def grid_candidates(cls, n: int) -> List["TunableParams"]:
        """Generate n candidate configs via grid sampling."""
        # Generate full grid
        dense_weights = list(_range_float(*cls.DENSE_WEIGHT_RANGE))
        sparse_weights = list(_range_float(*cls.SPARSE_WEIGHT_RANGE))
        exact_weights = list(_range_float(*cls.EXACT_WEIGHT_RANGE))
        top_ks = list(range(*cls.TOP_K_RANGE))
        thresholds = list(_range_float(*cls.RERANKER_THRESHOLD_RANGE))
        rrf_ks = list(range(*cls.RRF_K_RANGE))

        grid = list(product(
            dense_weights, sparse_weights, exact_weights,
            top_ks, thresholds, rrf_ks
        ))

        # Sample if grid is too large
        if len(grid) > n:
            random.shuffle(grid)
            grid = grid[:n]

        return [
            cls(
                dense_weight=g[0], sparse_weight=g[1], exact_weight=g[2],
                top_k=g[3], reranker_threshold=g[4], rrf_k=g[5],
            )
            for g in grid
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dict for JSON serialization (excludes class constants)."""
        return {
            "dense_weight": self.dense_weight,
            "sparse_weight": self.sparse_weight,
            "exact_weight": self.exact_weight,
            "top_k": self.top_k,
            "reranker_threshold": self.reranker_threshold,
            "rrf_k": self.rrf_k,
        }


def _range_float(start: float, stop: float, step: float) -> List[float]:
    """Generate a list of floats from start to stop (exclusive) with given step."""
    result = []
    current = start
    while current < stop:
        result.append(round(current, 2))
        current += step
    return result


# ── Settings patcher ────────────────────────────────────────────────────────────

class _SettingsPatcher:
    """Context manager that patches settings for tuning and restores after."""

    def __init__(self, params: TunableParams):
        self.params = params
        self._originals: Dict[str, Any] = {}

    def __enter__(self):
        self._originals = {
            "HYBRID_DENSE_WEIGHT": settings.HYBRID_DENSE_WEIGHT,
            "HYBRID_SPARSE_WEIGHT": settings.HYBRID_SPARSE_WEIGHT,
            "HYBRID_EXACT_WEIGHT": settings.HYBRID_EXACT_WEIGHT,
            "RETRIEVAL_TOP_K": settings.RETRIEVAL_TOP_K,
            "RERANKER_SCORE_THRESHOLD": settings.RERANKER_SCORE_THRESHOLD,
        }

        # Patch retrieval.py module-level _RRF_K
        import app.services.retrieval as retrieval_mod
        self._originals["_RRF_K"] = retrieval_mod._RRF_K
        retrieval_mod._RRF_K = self.params.rrf_k

        settings.HYBRID_DENSE_WEIGHT = self.params.dense_weight
        settings.HYBRID_SPARSE_WEIGHT = self.params.sparse_weight
        settings.HYBRID_EXACT_WEIGHT = self.params.exact_weight
        settings.RETRIEVAL_TOP_K = self.params.top_k
        settings.RERANKER_SCORE_THRESHOLD = self.params.reranker_threshold

        return self

    def __exit__(self, *exc):
        settings.HYBRID_DENSE_WEIGHT = self._originals["HYBRID_DENSE_WEIGHT"]
        settings.HYBRID_SPARSE_WEIGHT = self._originals["HYBRID_SPARSE_WEIGHT"]
        settings.HYBRID_EXACT_WEIGHT = self._originals["HYBRID_EXACT_WEIGHT"]
        settings.RETRIEVAL_TOP_K = self._originals["RETRIEVAL_TOP_K"]
        settings.RERANKER_SCORE_THRESHOLD = self._originals["RERANKER_SCORE_THRESHOLD"]

        import app.services.retrieval as retrieval_mod
        retrieval_mod._RRF_K = self._originals["_RRF_K"]


# ── Scoring ─────────────────────────────────────────────────────────────────────

def _token_f1(prediction: str, ground_truths: List[str]) -> float:
    """Max token-F1 over all ground truth answers (SQuAD metric)."""
    import string
    from collections import Counter

    def _normalise(text: str) -> List[str]:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text.split()

    best = 0.0
    pred_tokens = Counter(_normalise(prediction))
    for gt in ground_truths:
        gt_tokens = Counter(_normalise(gt))
        common = sum((pred_tokens & gt_tokens).values())
        if common == 0:
            continue
        precision = common / sum(pred_tokens.values())
        recall = common / sum(gt_tokens.values())
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


def _retrieval_hit(contexts: List[str], ground_truths: List[str]) -> float:
    """1.0 if any ground truth answer appears in any chunk."""
    all_text = " ".join(c.lower() for c in contexts)
    for gt in ground_truths:
        if gt.lower() in all_text:
            return 1.0
    return 0.0


@dataclass
class EvalResult:
    """Result of evaluating a single config."""
    mean_f1: float = 0.0
    mean_em: float = 0.0
    hit_rate: float = 0.0
    mean_latency_ms: float = 0.0
    n_questions: int = 0
    errors: int = 0


# ── Eval runner ─────────────────────────────────────────────────────────────────

def _run_eval(
    questions: List[Dict[str, Any]],
    kb_id: int,
    base_url: str,
    username: str,
    password: str,
    email: str,
) -> EvalResult:
    """Run eval queries against the current settings and return metrics."""
    import requests

    session = requests.Session()
    timeout = 60

    # Authenticate
    try:
        r = session.post(
            f"{base_url}/auth/register",
            json={"username": username, "password": password, "email": email},
            timeout=timeout,
        )
        if r.status_code not in (200, 201, 400):
            r.raise_for_status()

        r = session.post(
            f"{base_url}/auth/token",
            data={"username": username, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=timeout,
        )
        r.raise_for_status()
        token = r.json()["access_token"]
    except Exception as e:
        logger.error("[TUNE] auth failed: %s", e)
        return EvalResult(errors=1)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }

    f1_scores = []
    em_scores = []
    hit_rates = []
    latencies = []
    errors = 0

    for q in questions:
        try:
            r = session.post(
                f"{base_url}/query",
                json={
                    "question": q["question"],
                    "kb_ids": [kb_id],
                    "use_dense": True,
                    "use_sparse": True,
                    "use_exact": True,
                    "use_graph_rag": False,
                    "generate_answer": False,
                },
                headers=headers,
                timeout=timeout,
            )
            r.raise_for_status()
            resp = r.json()

            answer = resp.get("answer", "")
            contexts = [c.get("content", "") for c in resp.get("contexts", [])]
            latency_ms = resp.get("latency_ms", 0)

            score_text = answer if answer else " ".join(contexts)
            f1 = _token_f1(score_text, q["answers"]) if score_text else 0.0
            em = 1.0 if any(
                _normalise_eq(score_text, gt) for gt in q["answers"]
            ) else 0.0
            hit = _retrieval_hit(contexts, q["answers"])

            f1_scores.append(f1)
            em_scores.append(em)
            hit_rates.append(hit)
            latencies.append(latency_ms)

        except Exception as e:
            logger.warning("[TUNE] query failed: %s", e)
            errors += 1

    n = len(f1_scores)
    return EvalResult(
        mean_f1=round(sum(f1_scores) / n, 4) if n else 0.0,
        mean_em=round(sum(em_scores) / n, 4) if n else 0.0,
        hit_rate=round(sum(hit_rates) / n, 4) if n else 0.0,
        mean_latency_ms=round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        n_questions=n,
        errors=errors,
    )


def _normalise_eq(a: str, b: str) -> bool:
    import string
    a = a.lower().translate(str.maketrans("", "", string.punctuation))
    b = b.lower().translate(str.maketrans("", "", string.punctuation))
    return a == b


# ── Tuning loop ─────────────────────────────────────────────────────────────────

@dataclass
class TuningResult:
    """Result of the tuning loop."""
    best_config: TunableParams
    best_result: EvalResult
    history: List[Dict[str, Any]]
    n_iterations: int
    converged_at: Optional[int]


def run_tuning_loop(
    questions: List[Dict[str, Any]],
    kb_id: int,
    base_url: str,
    username: str = "tune_user",
    password: str = "tune_pass",
    email: str = "tune@example.com",
    max_iterations: int = 20,
    patience: int = 5,
    seed: Optional[int] = None,
) -> TuningResult:
    """
    Run the auto-tuning loop.

    Generates candidate configs, evaluates each, and tracks the best.
    Converges when F1 stops improving for `patience` iterations.

    Args:
        questions: List of {question, answers} dicts.
        kb_id: Knowledge base ID to query.
        base_url: API base URL (e.g., http://localhost:8000/api).
        username/password/email: Auth credentials for eval.
        max_iterations: Maximum configs to evaluate.
        patience: Stop if F1 doesn't improve for this many iterations.
        seed: Random seed for reproducibility.

    Returns:
        TuningResult with best config, metrics, and convergence history.
    """
    if seed is not None:
        random.seed(seed)

    logger.info(
        "[TUNE] starting loop | questions=%d | max_iterations=%d | patience=%d",
        len(questions), max_iterations, patience,
    )

    best_config: Optional[TunableParams] = None
    best_result: Optional[EvalResult] = None
    best_f1 = -1.0
    history: List[Dict[str, Any]] = []
    no_improve_count = 0
    converged_at: Optional[int] = None

    # Generate candidates — mix of grid and random
    grid = TunableParams.grid_candidates(max_iterations)
    random_samples = [TunableParams.random_sample() for _ in range(max_iterations // 2)]
    candidates = grid + random_samples
    random.shuffle(candidates)

    for i, params in enumerate(candidates[:max_iterations]):
        start = time.time()

        with _SettingsPatcher(params):
            result = _run_eval(questions, kb_id, base_url, username, password, email)

        elapsed_ms = round((time.time() - start) * 1000, 1)

        entry = {
            "iteration": i + 1,
            "config": params.to_dict(),
            "f1": result.mean_f1,
            "em": result.mean_em,
            "hit_rate": result.hit_rate,
            "latency_ms": result.mean_latency_ms,
            "elapsed_ms": elapsed_ms,
        }
        history.append(entry)

        logger.info(
            "[TUNE] iteration=%d | dense=%.1f sparse=%.1f exact=%.1f top_k=%d "
            "threshold=%.1f rrf_k=%d | f1=%.4f em=%.4f hit=%.4f | %.0fms",
            i + 1,
            params.dense_weight, params.sparse_weight, params.exact_weight,
            params.top_k, params.reranker_threshold, params.rrf_k,
            result.mean_f1, result.mean_em, result.hit_rate,
            elapsed_ms,
        )

        if result.mean_f1 > best_f1:
            best_f1 = result.mean_f1
            best_config = params
            best_result = result
            no_improve_count = 0
            logger.info("[TUNE] new best | f1=%.4f", best_f1)
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            converged_at = i + 1
            logger.info("[TUNE] converged at iteration %d (no improvement for %d)", converged_at, patience)
            break

    if best_config is None or best_result is None:
        # All evals failed — return defaults
        best_config = TunableParams()
        best_result = EvalResult()

    result = TuningResult(
        best_config=best_config,
        best_result=best_result,
        history=history,
        n_iterations=len(history),
        converged_at=converged_at,
    )

    logger.info(
        "[TUNE] done | iterations=%d | best_f1=%.4f | config=%s",
        len(history), best_f1, best_config.to_dict(),
    )

    return result


# ── Config persistence ─────────────────────────────────────────────────────────

TUNING_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".gsd", "tuning")


def save_best_config(result: TuningResult, kb_id: int, output_path: Optional[str] = None) -> str:
    """Save best config and convergence history to disk."""
    path = output_path or os.path.join(TUNING_DIR, "best_config.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kb_id": kb_id,
        "n_iterations": result.n_iterations,
        "converged_at": result.converged_at,
        "best_config": result.best_config.to_dict(),
        "best_metrics": {
            "f1": result.best_result.mean_f1,
            "em": result.best_result.mean_em,
            "hit_rate": result.best_result.hit_rate,
            "latency_ms": result.best_result.mean_latency_ms,
        },
        "f1_history": [h["f1"] for h in result.history],
        "history": result.history,
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info("[TUNE] saved best config to %s | f1=%.4f", path, result.best_result.mean_f1)
    return path


def load_best_config(path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load best config from disk and patch settings."""
    path = path or os.path.join(TUNING_DIR, "best_config.json")
    if not os.path.exists(path):
        logger.info("[TUNE] no best config at %s — using defaults", path)
        return None

    with open(path) as f:
        payload = json.load(f)

    config = payload.get("best_config", {})
    settings.HYBRID_DENSE_WEIGHT = config.get("dense_weight", settings.HYBRID_DENSE_WEIGHT)
    settings.HYBRID_SPARSE_WEIGHT = config.get("sparse_weight", settings.HYBRID_SPARSE_WEIGHT)
    settings.HYBRID_EXACT_WEIGHT = config.get("exact_weight", settings.HYBRID_EXACT_WEIGHT)
    settings.RETRIEVAL_TOP_K = config.get("top_k", settings.RETRIEVAL_TOP_K)
    settings.RERANKER_SCORE_THRESHOLD = config.get("reranker_threshold", settings.RERANKER_SCORE_THRESHOLD)

    import app.services.retrieval as retrieval_mod
    retrieval_mod._RRF_K = config.get("rrf_k", retrieval_mod._RRF_K)

    logger.info(
        "[TUNE] loaded best config from %s | f1=%.4f | config=%s",
        path, payload.get("best_metrics", {}).get("f1", 0), config,
    )
    return config
