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
    failed_legs: List[str] = []  # Legs that failed (for confidence messages)
    leg_doc_counts: Annotated[dict, lambda a, b: {**a, **b}] = {}  # {leg_name: count} for sufficiency check
    
    # ── Memory state ────────────────────────────────────────────────────
    historical_memory_docs: Annotated[List[dict], accumulate] = []

    # ── Merged retrieval state ──────────────────────────────────────────
    retrieved_docs: Annotated[List[dict], accumulate] = []
    retrieved_contexts: Annotated[List[str], accumulate] = []
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
    thinking_chunks: List[str] = []
    is_chart_query: bool = False
    chart_data: Optional[Any] = None
    chart_retries: int = 0

    # ── Retry budget state ──────────────────────────────────────────────
    # Annotated with last-value reducer so parallel subgraphs can each write
    # without LangGraph throwing "Can receive only one value per step".
    adaptive_reran: Annotated[bool, _last_value] = False
    answer_evaluation_attempts: Annotated[int, _last_value] = 0
    needs_retry: Annotated[bool, _last_value] = False
    graph_expansion_done: Annotated[bool, _last_value] = False

    # ── Subtask context state ───────────────────────────────────────────
    # Each subagent accumulates its retrieved context here; the main
    # orchestrator uses these bundles to generate the final answer.
    subtask_contexts: Annotated[List[dict], accumulate] = []  # {question, retrieved_docs, retrieved_contexts, retrieval_confidence, leg_results, failed_legs}

    # ── Synthesis state ─────────────────────────────────────────────────
    subtask_answers: Annotated[List[dict], accumulate] = []  # legacy field, kept for compatibility
    final_answer: str = ""
    final_confidence: str = ""

    # ── Configuration ───────────────────────────────────────────────────
    kb_ids: List[int] = []
    org_id: Optional[int] = None
    file_markdown: Optional[str] = None
    existing_summary: str = ""
    generate_answer: bool = True  # If False, skip LLM generation (retrieval-only mode)

    # ── Metadata ────────────────────────────────────────────────────────
    latency_ms: int = 0
    model_used: str = ""

    # ── Streaming / event data ──────────────────────────────────────────
    _task_list: Optional[List[dict]] = None  # Subtask list for task_list events
    _confidence: str = ""  # Confidence level for context events
