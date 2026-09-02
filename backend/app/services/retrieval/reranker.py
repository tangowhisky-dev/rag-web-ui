"""
Cross-encoder reranker service.

Uses fastembed TextCrossEncoder (ONNX runtime) to re-score (query, chunk) pairs
after RRF merging. Replaced sentence-transformers CrossEncoder to:
  - Remove the PyTorch dependency (~2 GB) — fastembed uses ONNX Runtime (~200 MB)
  - Run ~1.8× faster on CPU (ONNX vs PyTorch inference)
  - Scores are bit-for-bit identical to the previous implementation

Model: Xenova/ms-marco-MiniLM-L-12-v2  (ONNX conversion of cross-encoder/ms-marco-MiniLM-L-12-v2)
  - Trained on MS MARCO passage ranking (127M query-passage pairs)
  - 12-layer MiniLM — fast enough for CPU, ~4ms per chunk on modern hardware
  - Outputs a raw logit (higher = more relevant); no fixed 0–1 scale

Score distribution (empirical, ms-marco-MiniLM-L-12-v2):
  Scores are bimodal — relevant chunks cluster 1–10, irrelevant cluster -5 to -11.
  There is almost no middle ground. 0.0 is a reliable cutoff for this model.

Integration point: called after neo4j_expansion node ,
and in chat_history_retrieval_node() for prior-answer scoring.
"""

import logging
import os
import threading
from typing import Any, List, Optional

from langchain_core.documents import Document as LangchainDocument

from app.core.config import settings
from app.services.agentic_rag.retry import with_retry_sync

logger = logging.getLogger(__name__)

# Module-level singleton — loaded once on first use, reused across all requests.
# TextCrossEncoder is stateless between calls so it is safe to share.
_cross_encoder = None
_cross_encoder_lock = threading.Lock()


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        with _cross_encoder_lock:
            # Double-check after acquiring lock — another thread may have
            # loaded the model while we were waiting.
            if _cross_encoder is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                model_name = settings.RERANKER_MODEL
                cache_dir = settings.RERANKER_CACHE_DIR

                os.makedirs(cache_dir, exist_ok=True)

                logger.debug("Reranker: loading model=%s cache_dir=%s", model_name, cache_dir)
                _cross_encoder = TextCrossEncoder(model_name=model_name, cache_dir=cache_dir)
                logger.debug("Reranker: model loaded")

    return _cross_encoder


@with_retry_sync(max_attempts=3)
def rerank(
    query: str,
    docs: List[LangchainDocument],
    score_threshold: Optional[float] = None,
    db: Any = None,
    org_id: Any = None,
) -> List[LangchainDocument]:
    """
    Re-score docs against query using the cross-encoder and filter by threshold.

    All chunks scoring above the threshold are returned, ordered by score.
    No top_n cap — if 8 out of 10 chunks are relevant, all 8 pass.

    Args:
        query:           The retrieval query.
        docs:            Candidates from RRF merge.
        score_threshold: Min logit to pass. Defaults to RERANKER_SCORE_THRESHOLD.

    Returns:
        Docs re-ordered by cross-encoder score, filtered by threshold only.
        Each doc gets metadata["_reranker_score"] set.
    """
    if not docs:
        return docs

    if score_threshold is None:
        if db is not None:
            from app.services.settings_service import get_setting
            score_threshold = get_setting(db, "RERANKER_SCORE_THRESHOLD", org_id)
        else:
            from app.core.settings_registry import get_def
            score_threshold = get_def("RERANKER_SCORE_THRESHOLD").default

    encoder = _get_cross_encoder()

    # Prepend document title to each passage so the cross-encoder gets
    # document-level context. A chunk about "withdrawal procedures" from
    # "Tactical Operations Doctrine" scores higher for a "tactical doctrine"
    # query when the title is visible to the reranker.
    passages = []
    for doc in docs:
        title = (doc.metadata or {}).get("title", "")
        if title:
            passages.append(f"{title}\n\n{doc.page_content}")
        else:
            passages.append(doc.page_content)
    scores: List[float] = list(encoder.rerank(query, passages))

    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    logger.debug(
        "Reranker: query=%r | input=%d | threshold=%.2f | score range=[%.3f, %.3f]",
        query[:80],
        len(docs),
        score_threshold,
        scored[-1][0] if scored else 0.0,
        scored[0][0] if scored else 0.0,
    )

    for rank, (score, doc) in enumerate(scored):
        snippet = doc.page_content[:80].replace("\n", " ")
        # logger.debug("  reranker[%d] score=%.4f text=%r", rank, score, snippet)

    result = []
    for score, doc in scored:
        if score < score_threshold:
            break  # sorted descending — nothing below this will pass
        doc.metadata["_reranker_score"] = round(score, 4)
        result.append(doc)

    logger.debug(
        "Reranker: %d/%d chunks passed threshold=%.2f",
        len(result), len(scored), score_threshold,
    )
    return result

def preload_cross_encoder() -> None:
    """Eagerly load the cross-encoder reranker at app startup.

    Safe to call even if the model was already loaded (lazy path).
    On failure, logs a warning but does not raise — the lazy path
    will still attempt to load on first use.
    """
    try:
        _get_cross_encoder()
        logger.debug("Cross-encoder reranker loaded: %s", settings.RERANKER_MODEL)
    except Exception as exc:
        logger.warning("Cross-encoder preload failed (will retry on first use): %s", exc)
