"""
POST /api/search — standalone KB search (Google-style).

Runs abbreviation expansion + 3-leg retrieval + merge/dedup + cross-encoder
reranking, then returns ranked chunk results. No LLM rewrite, no generation,
no chat session. Logs each search to search_history for auditing.
"""
import time
import logging
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.models.knowledge import KnowledgeBase
from app.models.search_history import SearchHistory
from app.services.retrieval import (
    get_effective_datastore_ids,
    dense_search_docs,
    sparse_search_docs,
    exact_search_docs,
    dedup_by_content_hash,
    semantic_dedup,
    rerank,
)
from app.services.settings_service import get_setting
from app.services.abbreviation_service import build_lookup, expand_query_suffix
from app.services.infrastructure.utils import _serialise_doc

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Request / Response schemas ────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    kb_ids: List[int]


class SearchResultItem(BaseModel):
    chunk_text: str
    original_text: Optional[str] = None
    file_name: str
    document_id: int
    kb_id: Optional[int] = None
    data_store_id: Optional[int] = None
    chunk_index: Optional[int] = None
    reranker_score: float


class SearchResponse(BaseModel):
    query: str
    expanded_query: str
    results: List[SearchResultItem]
    total: int
    latency_ms: int


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("", response_model=SearchResponse)
def search(
    *,
    db: Session = Depends(get_db),
    body: SearchRequest,
    current_user: User = Depends(get_current_user),
) -> Any:
    """Standalone KB search — retrieval + reranking, no LLM generation."""
    t0 = time.monotonic()

    if not body.query.strip():
        raise HTTPException(status_code=422, detail="Query must not be empty")
    if not body.kb_ids:
        raise HTTPException(status_code=422, detail="At least one knowledge base must be selected")

    # Verify all KBs exist and belong to this user
    kbs = (
        db.query(KnowledgeBase)
        .filter(
            KnowledgeBase.id.in_(body.kb_ids),
            KnowledgeBase.user_id == current_user.id,
        )
        .all()
    )
    if len(kbs) != len(body.kb_ids):
        raise HTTPException(status_code=404, detail="One or more knowledge bases not found")

    org_id = current_user.org_id
    query = body.query.strip()

    # 1. Abbreviation expansion
    lookup = build_lookup(db, org_id)
    expanded_query = expand_query_suffix(query, lookup)

    # 2. Resolve linked datastores
    datastore_ids = get_effective_datastore_ids(body.kb_ids, org_id, db)

    # 3. Run 3 retrieval legs (sync — FastAPI runs sync endpoints in a threadpool)
    all_docs: list[dict] = []
    for leg_fn in (dense_search_docs, sparse_search_docs, exact_search_docs):
        try:
            docs = leg_fn(
                query=expanded_query,
                kb_ids=body.kb_ids,
                datastore_ids=datastore_ids,
                db=db,
                org_id=org_id,
            )
            all_docs.extend(_serialise_doc(d) for d in docs)
        except Exception as exc:
            logger.warning("[SEARCH] %s failed: %s", leg_fn.__name__, exc)

    # 4. Merge + dedup
    merged = dedup_by_content_hash(all_docs)
    threshold = get_setting(db, "DEDUP_SEMANTIC_THRESHOLD", org_id)
    if threshold < 1.0 and len(merged) > 1:
        merged = semantic_dedup(merged, threshold)

    # 5. Rerank with cross-encoder (score all, no threshold filter)
    from langchain_core.documents import Document as LangchainDocument
    lc_docs = [
        LangchainDocument(page_content=d.get("page_content", ""), metadata=d.get("metadata", {}))
        for d in merged
    ]
    try:
        reranked = rerank(query=expanded_query, docs=lc_docs, score_threshold=float("-inf"))
    except Exception as exc:
        logger.warning("[SEARCH] rerank failed: %s", exc)
        reranked = lc_docs

    # 6. Build response items sorted by reranker score
    results: List[SearchResultItem] = []
    for doc in reranked:
        meta = doc.metadata or {}
        results.append(SearchResultItem(
            chunk_text=doc.page_content or "",
            original_text=meta.get("original_text"),
            file_name=meta.get("file_name", "Unknown"),
            document_id=meta.get("document_id", 0),
            kb_id=meta.get("kb_id"),
            data_store_id=meta.get("data_store_id"),
            chunk_index=meta.get("chunk_index"),
            reranker_score=meta.get("_reranker_score", 0.0),
        ))

    latency_ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "[SEARCH] query=%r | kb_ids=%s | results=%d | latency=%dms",
        query[:80], body.kb_ids, len(results), latency_ms,
    )

    # 7. Log to search_history
    db.add(SearchHistory(
        user_id=current_user.id,
        query=query,
        expanded_query=expanded_query if expanded_query != query else None,
        kb_ids=body.kb_ids,
        result_count=len(results),
        latency_ms=latency_ms,
    ))
    db.commit()

    return SearchResponse(
        query=query,
        expanded_query=expanded_query,
        results=results,
        total=len(results),
        latency_ms=latency_ms,
    )
