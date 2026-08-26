"""
POST /api/search — standalone KB search (Google-style).
GET  /api/search/history — recent searches for the current user.
POST /api/search/suggestions — LLM-generated query suggestions from history.

Runs abbreviation expansion + 3-leg retrieval + merge/dedup + cross-encoder
reranking, then returns ranked chunk results. No LLM rewrite, no generation,
no chat session. Logs each search to search_history for auditing.
"""
import json
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
    title: Optional[str] = None
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
            title=meta.get("title"),
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


# ── Recent searches ──────────────────────────────────────────────────────────

class SearchHistoryItem(BaseModel):
    id: int
    query: str
    result_count: int
    created_at: str


@router.get("/history", response_model=List[SearchHistoryItem])
def get_search_history(
    *,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    limit: int = 10,
) -> Any:
    """Return recent searches for the current user, newest first."""
    rows = (
        db.query(SearchHistory)
        .filter(SearchHistory.user_id == current_user.id)
        .order_by(SearchHistory.created_at.desc())
        .limit(min(limit, 50))
        .all()
    )
    return [
        SearchHistoryItem(
            id=r.id,
            query=r.query,
            result_count=r.result_count,
            created_at=r.created_at.isoformat() if r.created_at else "",
        )
        for r in rows
    ]


# ── LLM query suggestions ────────────────────────────────────────────────────

class SuggestionResponse(BaseModel):
    suggestions: List[str]


@router.post("/suggestions", response_model=SuggestionResponse)
async def get_suggestions(
    *,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Generate 3 query suggestions from the user's recent search history.

    Uses the configured chat LLM with a concise system prompt. Falls back
    to an empty list if no LLM is configured or the call fails.
    """
    # Fetch recent searches (up to 20 for context)
    rows = (
        db.query(SearchHistory)
        .filter(SearchHistory.user_id == current_user.id)
        .order_by(SearchHistory.created_at.desc())
        .limit(20)
        .all()
    )
    if not rows:
        return SuggestionResponse(suggestions=[])

    recent_queries = [r.query for r in rows]

    # Resolve LLM config
    model_name = get_setting(db, "QUERY_MODEL", None) or get_setting(db, "OPENAI_MODEL", None)
    api_base = get_setting(db, "OPENAI_API_BASE", None)
    api_key = get_setting(db, "OPENAI_API_KEY", None)
    if not model_name or not api_base:
        return SuggestionResponse(suggestions=[])

    if not api_key:
        api_key = "not-required"

    system_prompt = (
        "You are a search assistant. Given the user's recent search queries, "
        "suggest 3 new queries they might want to search next. "
        "Return ONLY a JSON array of 3 strings, no explanation."
    )
    user_prompt = "Recent searches:\n" + "\n".join(f"- {q}" for q in recent_queries[:15])

    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key, base_url=api_base)
        resp = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=200,
        )
        content = resp.choices[0].message.content or ""
        # Parse JSON array from response — handle markdown code fences
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        suggestions = json.loads(content)
        if isinstance(suggestions, list):
            suggestions = [s.strip() for s in suggestions if isinstance(s, str)][:3]
        else:
            suggestions = []
        return SuggestionResponse(suggestions=suggestions)
    except Exception as exc:
        logger.warning("[SEARCH] suggestion LLM call failed: %s", exc)
        return SuggestionResponse(suggestions=[])
