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


class AgentState(MessagesState):
    """State for the agent graph. Extends MessagesState with agentic fields."""

    # ── Query state ─────────────────────────────────────────────────────
    original_query: str = ""
    rewritten_query: str = ""
    is_complex: bool = False
    subtasks: List[str] = []
    current_subtask_index: int = 0

    # ── Clarification state ─────────────────────────────────────────────
    question_is_clear: bool = True
    pending_query: str = ""
    clarification_questions: List[str] = []

    # ── Retrieval state ─────────────────────────────────────────────────
    retrieved_docs: Annotated[List[dict], accumulate] = []
    retrieved_contexts: Annotated[List[str], accumulate] = []
    retrieval_keys: Annotated[set, set_union] = set()  # Track what we've already retrieved
    retrieval_iterations: int = 0
    retrieval_confidence: float = 0.0

    # ── Per-leg retrieval tracking ──────────────────────────────────────
    leg_results: dict = {}  # {leg_name: {"status": ok/failed/disabled, "count": N}}
    failed_legs: List[str] = []  # Legs that failed (for confidence messages)
    leg_doc_counts: dict = {}  # {leg_name: count} for sufficiency check

    # ── Sufficiency check state ─────────────────────────────────────────
    sufficiency_met: bool = False
    sufficiency_message: str = ""
    needs_graph_expansion: bool = False

    # ── Generation state ────────────────────────────────────────────────
    answer: str = ""
    thinking_chunks: List[str] = []
    is_chart_query: bool = False
    chart_data: Optional[Any] = None

    # ── Synthesis state ─────────────────────────────────────────────────
    subtask_answers: Annotated[List[dict], accumulate] = []  # {subtask, answer, docs}
    final_answer: str = ""

    # ── Configuration ───────────────────────────────────────────────────
    kb_ids: List[int] = []
    org_id: Optional[int] = None
    file_markdown: Optional[str] = None
    existing_summary: str = ""

    # ── Metadata ────────────────────────────────────────────────────────
    latency_ms: int = 0
    model_used: str = ""

    # ── Streaming / event data ──────────────────────────────────────────
    _task_list: Optional[List[dict]] = None  # Subtask list for task_list events
    _confidence: str = ""  # Confidence level for context events
