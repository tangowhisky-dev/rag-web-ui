"""
Retrieval confidence scoring.

Single-mode reranker scoring — all docs passed to this function have already
cleared the reranker threshold, so every doc is considered genuinely relevant.
Signals:

  A. Top score      (60 pts) — best reranker logit, normalised over [-10, 10]
                               Dominant signal: one perfect chunk → ~55 pts.
  B. Evidence count (10 pts) — log(n+1) / log(11), saturates at 10 chunks.
                               TOP_K-independent: only chunks that cleared the
                               reranker threshold count. Graph-expanded chunks
                               compete in the same pool — good ones raise B and
                               C, marginal ones barely move B and drag C down.
  C. Mean score     (30 pts) — mean logit of all passing docs, same normalisation
                               as A. Penalises cases where one good chunk is
                               surrounded by many marginal ones. Keeps B honest.

Score → level
  ≥ 80  very_high
  ≥ 55  high
  ≥ 30  medium
  >  0  low
     0  none

The same function is used by chat_service.py (streaming chat) and query.py
(stateless eval endpoint).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional

from langchain_core.documents import Document as LangchainDocument

from app.core.config import settings


def _normalise_doc(doc: Any) -> LangchainDocument:
    """Accept Document objects or serialized dicts from the RAG graph."""
    if isinstance(doc, LangchainDocument):
        return doc
    if isinstance(doc, dict):
        return LangchainDocument(
            page_content=doc.get("page_content", ""),
            metadata=dict(doc.get("metadata", {})),
        )
    return LangchainDocument(page_content=str(doc), metadata={})


# ── Level ──────────────────────────────────────────────────────────────────────

LEVELS = ("none", "low", "medium", "high", "very_high")

# Reranker logit normalisation range for ms-marco-MiniLM-L-12-v2.
_RERANKER_SCORE_MIN = -10.0
_RERANKER_SCORE_MAX = 10.0


@dataclass
class ConfidenceResult:
    level: str                  # one of LEVELS
    score: int                  # 0-100
    suggestion: Optional[str]
    breakdown: dict             # per-signal scores for transparency


def _level_and_suggestion(score: int, failed_legs: list) -> tuple[str, Optional[str]]:
    if score == 0:
        level = "none"
    elif score < 30:
        level = "low"
    elif score < 55:
        level = "medium"
    elif score < 80:
        level = "high"
    else:
        level = "very_high"

    suggestion: Optional[str] = None
    if failed_legs:
        suggestion = (
            f"Some knowledge sources were unavailable "
            f"({', '.join(failed_legs)}). Results may be incomplete."
        )
    elif level == "none":
        suggestion = (
            "No relevant documents found. "
            "Answering from intrinsic knowledge only."
        )
    elif level == "low":
        suggestion = (
            "Few relevant documents found. "
            "Try more specific keywords or check that the relevant documents have been ingested."
        )
    elif level == "medium":
        suggestion = (
            "Some relevant documents found. "
            "Results may be partial — consider rephrasing for better coverage."
        )
    # high / very_high → no suggestion needed

    return level, suggestion


def score_retrieval(
    docs: List[Any],
    retrieval_info: dict,
) -> ConfidenceResult:
    """
    Compute retrieval confidence from docs + retrieval_info.

    All docs have already cleared the reranker threshold.

    retrieval_info shape (from hybrid_search_with_legs):
      {
        "legs": {
          "dense":         {"status": "ok"|"failed"|"disabled", "count": N},
          "sparse": {...},
          "exact":         {...},
          "graph":         {...},
        },
        "failed_legs": ["dense", ...]
      }
    """
    docs = [_normalise_doc(d) for d in docs]
    failed_legs = retrieval_info.get("failed_legs", [])

    # Zero docs → none, regardless of leg stats.
    if not docs:
        legs = retrieval_info.get("legs", {})
        enabled_legs = [k for k, v in legs.items() if v["status"] != "disabled"]
        return ConfidenceResult(
            level="none",
            score=0,
            suggestion="No relevant documents found. Answering from intrinsic knowledge only.",
            breakdown={
                "total": 0,
                "enabled_legs": enabled_legs,
                "failed_legs": failed_legs,
                "docs_returned": 0,
            },
        )

    # A: top score (60 pts)
    reranker_scores = [
        doc.metadata["_reranker_score"]
        for doc in docs
        if "_reranker_score" in doc.metadata
    ]

    if reranker_scores:
        top_score = max(reranker_scores)
        mean_score = sum(reranker_scores) / len(reranker_scores)
        span = _RERANKER_SCORE_MAX - _RERANKER_SCORE_MIN
        norm_top = max(0.0, min((top_score - _RERANKER_SCORE_MIN) / span, 1.0))
        norm_mean = max(0.0, min((mean_score - _RERANKER_SCORE_MIN) / span, 1.0))
    else:
        top_score = mean_score = 0.0
        norm_top = norm_mean = 0.5

    a = norm_top * 60

    # B: evidence count (10 pts)
    b = min(math.log(len(docs) + 1) / math.log(11), 1.0) * 10

    # C: mean score (30 pts)
    c = norm_mean * 30

    score = round(a + b + c)
    level, suggestion = _level_and_suggestion(score, failed_legs)

    breakdown = {
        "top_reranker_score": round(top_score, 3),
        "mean_reranker_score": round(mean_score, 3),
        "top_score_pts": round(a),
        "evidence_count_pts": round(b),
        "mean_score_pts": round(c),
        "total": score,
        "docs_returned": len(docs),
        "failed_legs": failed_legs,
    }

    return ConfidenceResult(level=level, score=score, suggestion=suggestion, breakdown=breakdown)
