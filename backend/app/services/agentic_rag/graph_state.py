"""LangGraph-compatible state definitions for the agentic RAG agent.

Defines AgentState with all data flowing through the graph, including
accumulator reducers for automatic accumulation across nodes.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, List, Literal, Optional, Union

from langgraph.graph import MessagesState

from app.services.agentic_rag.schemas import Plan, LastAnswerObject, Observation, Artifact


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
    
    # ── Merged retrieval state ──────────────────────────────────────────
    # All scored docs (with _reranker_score) — used by adaptive reranking.
    all_scored_docs: Annotated[List[dict], _last_value] = []
    # Filtered docs after applying threshold — used for generation.
    retrieved_docs: Annotated[List[dict], _last_value] = []
    retrieval_keys: Annotated[set, set_union] = set()  # Track what we've already retrieved
    # Annotated with last-value reducer so parallel subgraphs can each write
    # without LangGraph throwing "Can receive only one value per step".
    retrieval_iterations: Annotated[int, _last_value] = 0
    retrieval_confidence: Annotated[float, _last_value] = 0.0

    # ── Sufficiency check state ─────────────────────────────────────────
    sufficiency_met: Annotated[bool, _last_value] = False
    sufficiency_message: Annotated[str, _last_value] = ""
    needs_graph_expansion: Annotated[bool, _last_value] = False

    # ── Compaction state ──────────────────────────────────────────────────
    # Structured summary of older conversation turns, generated when the
    # message history grows beyond COMPACTION_HISTORY_THRESHOLD.
    compaction_summary: Annotated[Optional[str], _last_value] = None
    compaction_triggered: Annotated[bool, _last_value] = False

    # ── Generation state ────────────────────────────────────────────────
    answer: str = ""
    answer_usage: Optional[dict] = None  # Token usage captured during streaming
    cited_doc_indices: List[int] = []  # 1-based doc indices cited by the final answer, in display order
    thinking_chunks: List[str] = []
    is_chart_query: bool = False
    chart_data: Optional[Any] = None
    chart_retries: int = 0

    # ── Retry budget state ──────────────────────────────────────────────
    # Annotated with last-value reducer so parallel subgraphs can each write
    # without LangGraph throwing "Can receive only one value per step".
    needs_retry: Annotated[bool, _last_value] = False
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

    # ── Final evaluation state ──────────────────────────────────────────
    # Computed by answer_evaluation_node — no automatic retry, UI decides
    final_confidence: Annotated[float, _last_value] = 0.0
    confidence_level: str = "none"
    faithfulness: int = 0
    completeness: int = 0
    retrieval_score: int = 0
    confidence_match: bool = True
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
    tool_call_count: Annotated[dict, _last_value] = {}
    last_answer_object: Annotated[Optional[LastAnswerObject], _last_value] = None
    artifacts: Annotated[List[Artifact], accumulate] = []
    needs_clarification: Annotated[bool, _last_value] = False
    clarification_question: Annotated[Optional[str], _last_value] = None

    # ── Metadata ────────────────────────────────────────────────────────
    latency_ms: int = 0
    model_used: str = ""

    # ── Streaming / event data ──────────────────────────────────────────
    _task_list: Optional[List[dict]] = None  # Subtask list for task_list events
    _confidence: str = ""  # Confidence level for context events
