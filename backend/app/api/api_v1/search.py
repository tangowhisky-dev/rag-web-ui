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

def _run_retrieval_legs(
    expanded_query: str,
    kb_ids: list[int],
    datastore_ids: list[int],
    db: Session,
    org_id: int,
) -> list[dict]:
    """Run dense, sparse, and exact retrieval legs, returning serialised docs."""
    all_docs: list[dict] = []
    for leg_fn in (dense_search_docs, sparse_search_docs, exact_search_docs):
        try:
            docs = leg_fn(
                query=expanded_query,
                kb_ids=kb_ids,
                datastore_ids=datastore_ids,
                db=db,
                org_id=org_id,
            )
            all_docs.extend(_serialise_doc(d) for d in docs)
        except Exception as exc:
            logger.warning("[SEARCH] %s failed: %s", leg_fn.__name__, exc)
    return all_docs


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
    all_docs = _run_retrieval_legs(expanded_query, body.kb_ids, datastore_ids, db, org_id)

    # 4. Merge + dedup
    merged = dedup_by_content_hash(all_docs)
    threshold = get_setting(db, "DEDUP_SEMANTIC_THRESHOLD", org_id)
    if threshold < 1.0 and len(merged) > 1:
        merged = semantic_dedup(merged, threshold)

    # 5. Rerank with cross-encoder (apply RERANKER_SCORE_THRESHOLD from settings)
    from langchain_core.documents import Document as LangchainDocument
    lc_docs = [
        LangchainDocument(page_content=d.get("page_content", ""), metadata=d.get("metadata", {}))
        for d in merged
    ]
    try:
        reranked = rerank(query=expanded_query, docs=lc_docs, db=db, org_id=org_id)
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
    logger.debug(
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
    """Return recent distinct searches for the current user, newest first.

    Duplicate queries are collapsed — only the most recent occurrence of
    each distinct query is returned.
    """
    rows = (
        db.query(SearchHistory)
        .filter(SearchHistory.user_id == current_user.id)
        .order_by(SearchHistory.created_at.desc())
        .limit(min(limit * 5, 100))
        .all()
    )
    seen: set[str] = set()
    distinct: list[SearchHistory] = []
    for r in rows:
        key = r.query.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        distinct.append(r)
        if len(distinct) >= limit:
            break
    return [
        SearchHistoryItem(
            id=r.id,
            query=r.query,
            result_count=r.result_count,
            created_at=r.created_at.isoformat() if r.created_at else "",
        )
        for r in distinct
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
    # Fetch the user's last 3 search queries
    rows = (
        db.query(SearchHistory)
        .filter(SearchHistory.user_id == current_user.id)
        .order_by(SearchHistory.created_at.desc())
        .limit(3)
        .all()
    )

    # Fetch the user's last 3 chat queries (role=user messages across their chats)
    from sqlalchemy import text as _text
    chat_rows = db.execute(_text(
        "SELECT m.content FROM messages m "
        "JOIN chats c ON m.chat_id = c.id "
        "WHERE c.user_id = :uid AND m.role = 'user' "
        "ORDER BY m.created_at DESC LIMIT 3"
    ), {"uid": current_user.id})
    recent_chat_queries = [r[0].strip() for r in chat_rows if r[0] and r[0].strip()]

    if not rows and not recent_chat_queries:
        return SuggestionResponse(suggestions=[])

    recent_searches = [r.query for r in rows]

    # Resolve utility LLM config (utility model + its own API base/key).
    from app.services.agentic_rag.llm_factory import get_org_llm
    org_id = current_user.org_id if hasattr(current_user, "org_id") else None
    llm_cfg = get_org_llm(org_id, db, role="utility")
    model_name = llm_cfg["model_name"]
    api_base = llm_cfg["api_base"]
    api_key = llm_cfg["api_key"]
    if not model_name or not api_base:
        return SuggestionResponse(suggestions=[])

    system_prompt = (
        "You are a search assistant. Given the user's recent search history and chat queries, "
        "suggest 3 new queries they might want to search next.\n\n"
        "Each suggestion must be a self-contained question or search phrase.\n"
        "Aim for variety:\n"
        "- one that broadens the scope (a wider search around the topic),\n"
        "- one that narrows the scope (a more specific or pinpoint query),\n"
        "- one that is a natural continuation of the user's recent interests.\n"
        "Do not repeat queries the user has already searched or asked.\n"
        "Return ONLY a JSON array of 3 strings, no explanation."
    )
    parts = []
    if recent_searches:
        parts.append("Recent searches:\n" + "\n".join(f"- {q}" for q in recent_searches))
    if recent_chat_queries:
        parts.append("Recent chat questions:\n" + "\n".join(f"- {q}" for q in recent_chat_queries))
    user_prompt = "\n\n".join(parts)

    try:
        from openai import AsyncOpenAI
        from app.services.infrastructure.reasoning_tags import extract_reasoning
        client = AsyncOpenAI(api_key=api_key, base_url=api_base)
        resp = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=1.0,
        )
        msg = resp.choices[0].message
        raw = msg.content or ""
        # Thinking models (e.g. Gemma, Qwen) may put all output in
        # reasoning_content and leave content empty. Fall back to
        # reasoning_content to find the JSON.
        if not raw.strip():
            raw = getattr(msg, "reasoning_content", None) or ""
        # Extract reasoning and clean the answer — same approach as the
        # retrieval pipeline (tool_call_parser.parse_think_response).
        # extract_reasoning returns (reasoning, answer_text, is_complete).
        _reasoning, content, _is_complete = extract_reasoning(raw)
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
