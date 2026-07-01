"""RAGGraph state, Pydantic schemas, and SSE event constants.

Split from rag_graph.py for maintainability.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from typing_extensions import TypedDict

from app.core.config import settings

# ---------------------------------------------------------------------------
# Type alias for SQLAlchemy Session (imported lazily to avoid circular deps)
# ---------------------------------------------------------------------------
_Session = Any  # sqlalchemy.orm.Session


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class RAGGraphState(TypedDict):
    # ── Query lifecycle ────────────────────────────────────────────────────
    query: str
    rewritten_query: str
    sub_queries: List[str]             # from decompose_query

    # ── Routing (preserved from v1) ───────────────────────────────────────
    sources: List[str]                 # ["kb", "file_current", "file_prior", "chat_history"]
    chat_history_docs: list            # from chat_history_retrieval_node
    file_ids_needed: List[int]
    router_rationale: str

    # ── File ──────────────────────────────────────────────────────────────
    file_markdown: Optional[str]

    # ── Retrieval ─────────────────────────────────────────────────────────
    retrieved_docs: list               # accumulates across all retry attempts
    retrieval_attempt: int             # 0=first, 1=widened, 2=keyword
    keyword_iterations: list           # [{sub_query, iteration, keywords, results_found}]

    # ── Grading / coverage ────────────────────────────────────────────────
    draft_answer: str
    coverage_result: dict              # sub_query → "covered"|"partially_covered"|"not_covered"
    uncovered_sub_queries: List[str]

    # ── Final answer ──────────────────────────────────────────────────────
    merged_context: str
    answer: str
    _usage: dict

    # ── Observability ─────────────────────────────────────────────────────
    agent_steps: list

    # ── Run-time context injected by run_stream ───────────────────────────
    knowledge_base_ids: List[int]
    recent_lc_history: list
    existing_summary: Optional[str]
    use_dense: bool
    use_sparse: bool
    use_exact: bool
    use_graph_rag: bool
    temperature: float
    model_name: Optional[str]
    display_query: Optional[str]
    api_base: Optional[str]
    query_model: Optional[str]
    org_id: Optional[int]
    _db: _Session | None


# ---------------------------------------------------------------------------
# SSE event type constants
# ---------------------------------------------------------------------------

EVENT_AGENT_STEP = "agent_step"
EVENT_REWRITTEN  = "rewritten_query"
EVENT_CONTEXT    = "context"
EVENT_TOKEN      = "token"
EVENT_DONE       = "done"


# ---------------------------------------------------------------------------
# Pydantic schemas for structured LLM calls
# ---------------------------------------------------------------------------

class _RouterOutput(BaseModel):
    sources: List[str]
    rationale: str
    file_ids_needed: List[int] = []

class _SectionOutput(BaseModel):
    indices: List[int]

class _SubQueriesOutput(BaseModel):
    sub_queries: List[str]

class _CoverageItem(BaseModel):
    sub_query: str
    status: str  # "covered" | "partially_covered" | "not_covered"

class _CoverageOutput(BaseModel):
    coverage: List[_CoverageItem]

class _KeywordsOutput(BaseModel):
    broad_keywords: List[str]
    narrow_keywords: List[str]
