"""
POST /api/query          — stateless RAG query, returns JSON (no SSE, no chat session)
GET  /api/query/kb/{id}/ingest-status — KB processing readiness check

The query endpoint now uses the same agentic RAG pipeline as the chat endpoint.
It is stateless: no chat session is persisted, but a transient thread id is used
for the agent graph checkpoint.
"""
import time
import logging
import uuid
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.models.knowledge import KnowledgeBase, ProcessingTask
from app.services.retrieval import score_retrieval
from app.services.agentic_rag import run_agentic_rag

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Request / Response schemas ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    kb_ids: List[int]
    # All globally enabled retrieval sources are always used. Chat-level toggles
    # were removed; this endpoint no longer accepts per-request leg flags.
    # generate_answer=False skips LLM generation for faster retrieval benchmarks.
    generate_answer: bool = True


class ContextChunk(BaseModel):
    content: str
    metadata: dict


class QueryResponse(BaseModel):
    question: str
    answer: Optional[str]
    contexts: List[ContextChunk]
    confidence: str                  # "high" | "low" | "none"
    suggestion: Optional[str]
    retrieval_info: dict
    latency_ms: int


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("", response_model=QueryResponse)
async def query(
    *,
    db: Session = Depends(get_db),
    body: QueryRequest,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Stateless RAG query — no chat session created, nothing persisted.

    The response contains:
    - answer           LLM answer (or null if generate_answer=False)
    - contexts         retrieved chunks with metadata
    - confidence       high / low / none
    - suggestion       human-readable hint when confidence != high
    - retrieval_info   per-leg status (ok / failed / disabled + count)
    - latency_ms       wall-clock time for the full call
    """
    t0 = time.monotonic()

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

    # Use a transient chat id for the stateless agent graph checkpoint so
    # concurrent stateless queries do not collide in the checkpoint store.
    transient_chat_id = abs(hash(uuid.uuid4().hex)) % (2**31)

    # Run the same agentic pipeline as chat, but collect events into a JSON response.
    answer: Optional[str] = None
    docs: List[dict] = []
    retrieval_confidence = 0.0

    async for event in run_agentic_rag(
        query=body.question,
        knowledge_base_ids=body.kb_ids,
        db=db,
        chat_id=transient_chat_id,
    ):
        if event.get("event") == "context":
            docs = event.get("docs", docs)
            retrieval_confidence = event.get("score", retrieval_confidence) / 100.0
        elif event.get("event") == "done":
            answer = event.get("full_response") or answer

    # Resolve retrieval leg flags before closing the DB session.
    from app.services.settings_service import get_setting
    org_id = current_user.org_id
    dense_enabled = get_setting(db, "RETRIEVAL_DENSE_ENABLED", org_id)
    sparse_enabled = get_setting(db, "RETRIEVAL_SPARSE_ENABLED", org_id)
    exact_enabled = get_setting(db, "RETRIEVAL_EXACT_ENABLED", org_id)
    graph_enabled = get_setting(db, "RETRIEVAL_GRAPH_ENABLED", org_id)

    # Release the DB connection now that the agentic pipeline is done.
    db.close()

    # Build a basic retrieval_info map: all globally enabled sources are active.
    retrieval_info = {
        "legs": {
            "dense": {"status": "ok" if dense_enabled else "disabled", "count": 0},
            "sparse": {"status": "ok" if sparse_enabled else "disabled", "count": 0},
            "exact": {"status": "ok" if exact_enabled else "disabled", "count": 0},
            "graph": {"status": "ok" if graph_enabled else "disabled", "count": 0},
        }
    }

    # Score retrieval confidence using the same helper as the legacy path.
    confidence_result = score_retrieval(docs, retrieval_info)

    latency_ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "[QUERY] question=%r | kb_ids=%s | docs=%d | confidence=%s | latency=%dms",
        body.question[:80], body.kb_ids, len(docs), confidence_result.level, latency_ms,
    )

    return QueryResponse(
        question=body.question,
        answer=answer,
        contexts=[ContextChunk(content=d.get("page_content", ""), metadata=d.get("metadata", {})) for d in docs],
        confidence=confidence_result.level,
        suggestion=confidence_result.suggestion,
        retrieval_info={**retrieval_info, "confidence_breakdown": confidence_result.breakdown},
        latency_ms=latency_ms,
    )


# ── KB ingest status ───────────────────────────────────────────────────────────

class IngestStatus(BaseModel):
    kb_id: int
    total: int
    completed: int
    failed: int
    pending: int
    ready: bool       # True when total > 0 and completed == total and failed == 0


@router.get("/kb/{kb_id}/ingest-status", response_model=IngestStatus)
def ingest_status(
    *,
    db: Session = Depends(get_db),
    kb_id: int,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Returns processing status for every document task in a knowledge base.
    Poll this until ready=True before running eval queries.
    """
    kb = (
        db.query(KnowledgeBase)
        .filter(
            KnowledgeBase.id == kb_id,
            KnowledgeBase.user_id == current_user.id,
        )
        .first()
    )
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    tasks = (
        db.query(ProcessingTask)
        .filter(ProcessingTask.knowledge_base_id == kb_id)
        .all()
    )

    total     = len(tasks)
    completed = sum(1 for t in tasks if t.status == "completed")
    failed    = sum(1 for t in tasks if t.status == "failed")
    pending   = total - completed - failed

    return IngestStatus(
        kb_id=kb_id,
        total=total,
        completed=completed,
        failed=failed,
        pending=pending,
        ready=(total > 0 and completed == total and failed == 0),
    )
