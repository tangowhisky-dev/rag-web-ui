"""
Historical memory retrieval service.

Queries past assistant messages from MySQL (the messages table), builds
LangchainDocument objects for each, reranks them against the retrieval
query, and returns the top-K most relevant as serialised context blocks
with _source_type="historical_memory".

This reaches into MySQL to surface assistant responses that have been
pushed beyond the sliding window and are no longer in recent_lc_history.

Integration point: called from the agentic RAG pipeline (agentic_rag/)
when the user's query indicates
"forgetting" — e.g. "what did you say about X earlier?", "summarise
your previous answer on Y".

Configuration (Settings):
  HISTORICAL_MEMORY_ENABLED   — enable/disable (default True)
  HISTORICAL_MEMORY_TOP_K     — number of docs returned (default 5)
  HISTORICAL_MEMORY_SCORE_THRESHOLD — cross-encoder threshold (default 2.0)
"""

import logging
from typing import List, Optional

from langchain_core.documents import Document as LangchainDocument
from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from app.core.config import settings

def _strip_base64_prefix(content: str) -> str:
    """Strip the base64-encoded context prefix from assistant messages.

    Assistant messages stored in the DB have the format:
        <base64_json>{"context": [...], "rewritten_query": "..."}
    __LLM_RESPONSE__<actual_response_text>

    This function extracts only the actual response text for meaningful
    comparison in the reranker.
    """
    if "__LLM_RESPONSE__" in content:
        return content.split("__LLM_RESPONSE__")[-1]
    return content


def _is_empty_message(content: str) -> bool:
    """Check if a message has no meaningful content."""
    return not content or not content.strip()


logger = logging.getLogger(__name__)


# Maximum number of past assistant messages to fetch in one query.
# The reranker can handle more, but 50 is a practical cap to keep
# query latency bounded even for very long conversations.
_HISTORICAL_FETCH_LIMIT = 50


from app.services.infrastructure import _serialise_doc
from app.services.settings_service import get_setting


def retrieve_historical_memory(
    chat_id: int,
    query: str,
    db: Session,
    top_k: int = 5,
    score_threshold: float = 2.0,
    org_id: Optional[int] = None,
) -> List[dict]:
    """
    Query past assistant messages from MySQL, rerank against query,
    and return the top-K most relevant as serialised context blocks.

    Args:
        chat_id:            The chat session to search.
        query:              The retrieval query (usually the original user query).
        db:                 SQLAlchemy Session bound to MySQL.
        top_k:              Maximum number of docs to return (default 5).
        score_threshold:    Minimum reranker score to pass (default 2.0).

    Returns:
        List of serialised dicts, each with:
          - page_content: The assistant message text.
          - metadata: Dict with _source_type="historical_memory", id, and content_length.
        Returns [] when:
          - No assistant messages found in the database.
          - All messages score below threshold.
          - The reranker is disabled (returns last `top_k` raw docs without scores).
          - Any query failure occurs.

    Edge cases handled:
        - No messages in database → returns []
        - Reranker disabled → returns last K raw docs (most recent) with score=0
        - Query fails (MySQL error) → returns []
        - All scores below threshold → returns []
    """
    t0 = logger.isEnabledFor(logging.DEBUG) and __import__("time").monotonic()

    # ── 1. Query past assistant messages from MySQL ──────────────────────
    sql = text(
        """
        SELECT id, content, LENGTH(content) AS content_length
        FROM   messages
        WHERE  chat_id = :chat_id
          AND  role = 'assistant'
        ORDER  BY id ASC
        LIMIT  :limit
        """
    ).bindparams(
        bindparam("chat_id", value=chat_id),
        bindparam("limit", value=_HISTORICAL_FETCH_LIMIT),
    )

    try:
        rows = db.execute(sql).fetchall()
    except Exception as exc:
        logger.warning(
            "historical_memory: MySQL query failed for chat_id=%d: %s",
            chat_id, exc,
        )
        return []

    if not rows:
        logger.debug(
            "historical_memory: no assistant messages for chat_id=%d", chat_id,
        )
        return []

    # ── 2. Build LangchainDocument for each message ──────────────────────
    docs: List[LangchainDocument] = []
    for row in rows:
        msg_id = row.id
        raw_content = row.content or ""
        content = _strip_base64_prefix(raw_content)
        # Skip messages with no actual text (e.g., cancelled generations
        # where only the base64 context prefix was stored with no response).
        if _is_empty_message(content):
            logger.debug(
                "historical_memory: skipping empty message id=%d", msg_id,
            )
            continue
        content_length = row.content_length or len(raw_content)
        doc = LangchainDocument(
            page_content=content,
            metadata={
                "_source_type": "historical_memory",
                "message_id": msg_id,
                "content_length": content_length,
            },
        )
        docs.append(doc)

    logger.info(
        "historical_memory: chat_id=%d | fetched=%d | query=%r",
        chat_id, len(docs), query[:80],
    )

    # ── 3. Rerank — disabled path: return last K raw ────────────────────
    hist_enabled = get_setting(db, "HISTORICAL_MEMORY_ENABLED", org_id)
    reranker_enabled = get_setting(db, "RERANKER_ENABLED", org_id)
    if not hist_enabled or not reranker_enabled:
        # No reranker available: return the last `top_k` (most recent) docs raw.
        result = docs[-top_k:] if top_k > 0 else []
        for d in result:
            d.metadata["_reranker_score"] = 0.0
        return [_serialise_doc(d) for d in result]

    # ── 4. Rerank — enabled path ────────────────────────────────────────
    try:
        from app.services.retrieval import rerank
        ranked = rerank(
            query=query,
            docs=docs,
            score_threshold=score_threshold,
        )
    except Exception as exc:
        logger.warning(
            "historical_memory: reranker failed for chat_id=%d: %s — returning []",
            chat_id, exc,
        )
        return []

    # ── 5. Return top-K serialised dicts ────────────────────────────────
    result = ranked[:top_k]

    latency_ms = 0
    if t0:
        latency_ms = round((__import__("time").monotonic() - t0) * 1000, 1)

    logger.info(
        "historical_memory: chat_id=%d | candidates=%d | passed_threshold=%d | returned=%d | latency_ms=%.1f",
        chat_id, len(docs), len(ranked), len(result), latency_ms,
    )

    return [_serialise_doc(d) for d in result]
