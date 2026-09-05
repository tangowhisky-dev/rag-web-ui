"""LangGraph-compatible state definitions for the agentic RAG agent.

Defines AgentState with all data flowing through the graph, including
accumulator reducers for automatic accumulation across nodes.
"""

from __future__ import annotations

from typing import Annotated, Any, List, Optional

from langgraph.graph import MessagesState

from app.services.agentic_rag.schemas import Plan, LastAnswerObject, Observation


def accumulate(existing: list, new: list) -> list:
    """Default reducer: append new items to existing list.

    If new items contain __reset__ markers, replace the list entirely
    (removing the markers).
    """
    has_reset = any(isinstance(item, dict) and item.get("__reset__") for item in new)
    if has_reset:
        return [
            item for item in new
            if not (isinstance(item, dict) and item.get("__reset__"))
        ]
    return existing + new


def _last_value(_old: Any, new: Any) -> Any:
    """Reducer that keeps the most recent value.

    Required for keys that are overwritten by parallel Send() branches so
    LangGraph accepts multiple updates to the same key in one step.
    """
    return new


class AgentState(MessagesState):
    """State for the agent graph. Extends MessagesState with agentic fields."""

    # ── Query state ─────────────────────────────────────────────────────
    original_query: Annotated[str, _last_value] = ""
    # KB profile loaded by load_context_node (doc count, fields, content types, date range).
    # Injected into plan/think prompts so the agent has KB context without calling kb_metadata.
    kb_profile: Annotated[dict, _last_value] = {}

    # ── Merged retrieval state ──────────────────────────────────────────
    # Citable evidence ONLY: never seed this with recalled conversational
    # memory (see recalled_memories).
    retrieved_docs: Annotated[List[dict], _last_value] = []

    # ── Conversational memory (NOT citable evidence) ────────────────────
    # Hits from the long-term semantic store. May inform reference
    # resolution and planning; must never enter retrieved_docs, citations,
    # retrieval confidence, or faithfulness scoring.
    recalled_memories: Annotated[List[dict], _last_value] = []

    # ── Compaction state ──────────────────────────────────────────────────
    # Structured summary of older conversation turns, produced when the
    # rendered prompt exceeds the context budget.
    compaction_summary: Annotated[Optional[str], _last_value] = None
    compaction_triggered: Annotated[bool, _last_value] = False

    # ── Generation state ────────────────────────────────────────────────
    answer: str = ""
    answer_usage: Annotated[Optional[dict], _last_value] = None  # Provider token usage captured during streaming
    cited_doc_indices: List[int] = []  # 1-based doc indices cited by the final answer, in display order (legacy format)
    cited_docs: Annotated[list, _last_value] = []  # Cited evidence docs (both evidence and legacy formats)

    # ── Retry budget state ──────────────────────────────────────────────
    answer_evaluation_attempts: Annotated[int, _last_value] = 0

    # ── Synthesis state ─────────────────────────────────────────────────
    final_answer: str = ""

    # ── Final evaluation state ──────────────────────────────────────────
    # Computed by answer_evaluation_node — no automatic retry, UI decides
    final_confidence: Annotated[float, _last_value] = 0.0
    confidence_level: str = "none"
    faithfulness: int = 0
    completeness: int = 0
    retrieval_score: int = 0
    # Best retrieval confidence from merged docs (max reranker score or
    # kb_read/kb_search_documents confidence). Written by tool_node, read
    # by answer_evaluation_node for the final confidence formula.
    best_retrieval_confidence: Annotated[float, _last_value] = 0.0
    evaluation_flags: Annotated[List[str], _last_value] = []

    # ── Configuration ───────────────────────────────────────────────────
    # All configuration keys use _last_value because parallel Send() branches
    # pass the same values into each agent_subgraph invocation; without an
    # Annotated reducer LangGraph rejects concurrent writes to the same key.
    kb_ids: Annotated[List[int], _last_value] = []
    org_id: Annotated[Optional[int], _last_value] = None
    chat_id: Annotated[Optional[int], _last_value] = None
    user_id: Annotated[Optional[int], _last_value] = None
    message_id: Annotated[Optional[int], _last_value] = None
    file_markdown: Annotated[Optional[str], _last_value] = None
    generate_answer: Annotated[bool, _last_value] = True  # If False, skip LLM generation (retrieval-only mode)

    # ── Agent loop state ────────────────────────────────────────────────
    plan: Annotated[Optional[Plan], _last_value] = None
    observations: Annotated[List[Observation], accumulate] = []
    iteration: Annotated[int, _last_value] = 0
    tool_calls: Annotated[List[dict], _last_value] = []
    precomputed_answer: Annotated[str, _last_value] = ""
    tool_call_counts: Annotated[dict, _last_value] = {}
    last_answer_object: Annotated[Optional[LastAnswerObject], _last_value] = None
    needs_clarification: Annotated[bool, _last_value] = False
    clarification_question: Annotated[Optional[str], _last_value] = None
    # Sufficiency check result
    sufficient: Annotated[bool, _last_value] = False

    # ── Accumulated structured data (map-reduce for aggregate queries) ──
    # extract_data appends {label, value, unit, context} rows here across
    # multiple batches. chart_generate reads from this instead of
    # retrieved_docs. Not subject to compaction — it's small structured
    # data, not raw document content.
    accumulated_data: Annotated[List[dict], _last_value] = []

    # Wall-clock start of the current turn (time.monotonic()). Must be
    # declared: LangGraph silently drops updates for undeclared keys, which
    # previously made AGENT_MAX_WALL_SECONDS a no-op.
    started_at: Annotated[Optional[float], _last_value] = None
    # Set by tool_node when _verify_execution is already satisfied so
    # route_think can short-circuit straight to finalize.
    force_finalize: Annotated[bool, _last_value] = False
    # Pre-populated tool calls from plan_node for the first think round.
    precomputed_tool_calls: Annotated[List[dict], _last_value] = []

    # ── Clarification state ─────────────────────────────────────────────
    # Number of clarification rounds already spent this turn; capped by
    # AGENT_MAX_CLARIFICATIONS so plan → clarify → plan cannot loop.
    clarification_count: Annotated[int, _last_value] = 0
    # The user's answer to the clarification question, merged into query
    # resolution on resume.
    clarification_response: Annotated[str, _last_value] = ""

    # ── Metadata ────────────────────────────────────────────────────────
    latency_ms: int = 0
    model_used: str = ""
