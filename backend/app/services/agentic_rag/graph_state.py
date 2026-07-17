"""LangGraph-compatible state definitions for the agentic RAG agent.

Defines AgentState with all data flowing through the graph, including
accumulator reducers for automatic accumulation across nodes.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, List, Literal, Optional, Union

from langgraph.graph import MessagesState


def accumulate(existing: list, new: list) -> list:
    """Default reducer: append new items to existing list.

    If new items contain __reset__ markers, replace the list entirely
    (removing the markers).
    """
    has_reset = any(isinstance(item, dict) and item.get("__reset__") for item in new)
    if has_reset:
        return [item for item in new if not item.get("__reset__")]
    return existing + new


def set_union(a: set, b: set) -> set:
    """Union reducer: merge two sets."""
    return a | b


def _last_value(_old: Any, new: Any) -> Any:
    """Reducer that keeps the most recent value.

    Required for keys that are overwritten by parallel Send() branches so
    LangGraph accepts multiple updates to the same key in one step.
    """
    return new


class AgentState(MessagesState):
    """State for the agent graph. Extends MessagesState with agentic fields."""

    # ── Query state ─────────────────────────────────────────────────────
    # Annotated with last-value reducer because parallel Send() branches to
    # agent_subgraph each write these keys in the same step.
    original_query: Annotated[str, _last_value] = ""
    rewritten_query: Annotated[str, _last_value] = ""
    is_complex: Annotated[bool, _last_value] = False
    subtasks: Annotated[List[str], _last_value] = []
    subtask_independence: Annotated[List[bool], _last_value] = []
    current_subtask_index: Annotated[int, _last_value] = 0

    # ── Clarification state ─────────────────────────────────────────────
    question_is_clear: Annotated[bool, _last_value] = True
    pending_query: Annotated[str, _last_value] = ""
    clarification_questions: Annotated[List[str], _last_value] = []

    # ── Per-leg retrieval state (separated for observability) ───────────
    dense_docs: Annotated[List[dict], accumulate] = []
    sparse_docs: Annotated[List[dict], accumulate] = []
    exact_docs: Annotated[List[dict], accumulate] = []
    graph_docs: Annotated[List[dict], accumulate] = []  # Graph expansion docs
    
    # Per-leg status tracking
    # Annotated with a merge reducer so parallel Send() branches (multiple subtasks)
    # can each write their own leg results without LangGraph throwing
    # "Can receive only one value per step".
    leg_results: Annotated[dict, lambda a, b: {**a, **b}] = {}
    failed_legs: Annotated[List[str], accumulate] = []  # Legs that failed (for confidence messages)
    leg_doc_counts: Annotated[dict, lambda a, b: {**a, **b}] = {}  # {leg_name: count} for sufficiency check
    
    # ── Memory state ────────────────────────────────────────────────────
    historical_memory_docs: Annotated[List[dict], accumulate] = []

    # ── Merged retrieval state ──────────────────────────────────────────
    # All scored docs (with _reranker_score) — used by adaptive reranking.
    all_scored_docs: Annotated[List[dict], _last_value] = []
    # Filtered docs after applying threshold — used for generation.
    retrieved_docs: Annotated[List[dict], _last_value] = []
    retrieved_contexts: Annotated[List[str], _last_value] = []
    retrieval_keys: Annotated[set, set_union] = set()  # Track what we've already retrieved
    # Annotated with last-value reducer so parallel subgraphs can each write
    # without LangGraph throwing "Can receive only one value per step".
    retrieval_iterations: Annotated[int, _last_value] = 0
    retrieval_confidence: Annotated[float, _last_value] = 0.0

    # ── Sufficiency check state ─────────────────────────────────────────
    sufficiency_met: Annotated[bool, _last_value] = False
    sufficiency_message: Annotated[str, _last_value] = ""
    needs_graph_expansion: Annotated[bool, _last_value] = False

    # ── Generation state ────────────────────────────────────────────────
    answer: str = ""
    answer_usage: Optional[dict] = None  # Token usage captured during streaming
    thinking_chunks: List[str] = []
    is_chart_query: bool = False
    chart_data: Optional[Any] = None
    chart_retries: int = 0

    # ── Retry budget state ──────────────────────────────────────────────
    # Annotated with last-value reducer so parallel subgraphs can each write
    # without LangGraph throwing "Can receive only one value per step".
    adaptive_reran: Annotated[bool, _last_value] = False
    adaptive_rerunning: Annotated[bool, _last_value] = False  # True only when adaptive actually expanded
    answer_evaluation_attempts: Annotated[int, _last_value] = 0
    graph_expansion_done: Annotated[bool, _last_value] = False

    # ── Subtask context state ───────────────────────────────────────────
    # Each subagent accumulates its retrieved context here; the main
    # orchestrator uses these bundles to generate the final answer.
    subtask_contexts: Annotated[List[dict], accumulate] = []  # {question, retrieved_docs, retrieved_contexts, retrieval_confidence, leg_results, failed_legs}

    # ── Synthesis state ─────────────────────────────────────────────────
    subtask_answers: Annotated[List[dict], accumulate] = []  # legacy field, kept for compatibility
    final_answer: str = ""

    # ── Final evaluation state ─────────────────────────────────────────
    # Computed by answer_evaluation_node — no automatic retry, UI decides
    final_confidence: Annotated[float, _last_value] = 0.0
    confidence_level: str = "none"
    faithfulness: int = 0
    completeness: int = 0
    needs_retry: bool = False  # Always False — user-initiated retry only

    # ── Configuration ───────────────────────────────────────────────────
    # All configuration keys use _last_value because parallel Send() branches
    # pass the same values into each agent_subgraph invocation; without an
    # Annotated reducer LangGraph rejects concurrent writes to the same key.
    kb_ids: Annotated[List[int], _last_value] = []
    org_id: Annotated[Optional[int], _last_value] = None
    chat_id: Annotated[Optional[int], _last_value] = None
    user_id: Annotated[Optional[int], _last_value] = None
    file_markdown: Annotated[Optional[str], _last_value] = None
    generate_answer: Annotated[bool, _last_value] = True  # If False, skip LLM generation (retrieval-only mode)

    # ── Metadata ────────────────────────────────────────────────────────
    latency_ms: int = 0
    model_used: str = ""

    # ── Streaming / event data ──────────────────────────────────────────
    _task_list: Optional[List[dict]] = None  # Subtask list for task_list events
    _confidence: str = ""  # Confidence level for context events
