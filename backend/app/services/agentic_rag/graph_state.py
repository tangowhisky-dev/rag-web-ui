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
    # ``original_query`` is the user's exact wording and is authoritative for
    # planning, finalization and evaluation. ``rewritten_query`` is the
    # standalone retrieval string and is used only by retrieval/reranking.
    original_query: Annotated[str, _last_value] = ""
    rewritten_query: Annotated[str, _last_value] = ""
    expanded_query: Annotated[str, _last_value] = ""
    # Abbreviation glossary built once by expand_query_node, reused by all
    # downstream LLM calls (rewrite, plan, think, finalize, evaluation).
    abbreviation_glossary: Annotated[str, _last_value] = ""
    # Where each reference in rewritten_query was resolved from, or the
    # reason resolution was skipped/rejected.
    resolution_provenance: Annotated[Optional[dict], _last_value] = None
    # Negated terms extracted by rewrite_query_node via regex (e.g. "but not Linux" → ["Linux"]).
    # Used by rag_retrieve for post-filtering and by finalize_node as a generation guardrail.
    excluded_terms: Annotated[List[str], _last_value] = []
    # KB profile loaded by load_context_node (doc count, fields, content types, date range).
    # Injected into plan/think/rewrite prompts so the agent has KB context without calling kb_metadata.
    kb_profile: Annotated[dict, _last_value] = {}
    # Query intent (suggested filters/sort/legs) extracted by rewrite_query_node.
    # Folded into the existing rewrite LLM call — no separate node. null when no KB profile
    # or when the LLM output is malformed (after one retry).
    query_intent: Annotated[Optional[dict], _last_value] = None

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
    # Citable evidence ONLY: never seed this with recalled conversational
    # memory (see recalled_memories).
    retrieved_docs: Annotated[List[dict], _last_value] = []
    retrieval_confidence: Annotated[float, _last_value] = 0.0

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
    cited_doc_indices: List[int] = []  # 1-based doc indices cited by the final answer, in display order

    # ── Retry budget state ──────────────────────────────────────────────
    # Annotated with last-value reducer so parallel subgraphs can each write
    # without LangGraph throwing "Can receive only one value per step".
    adaptive_reran: Annotated[bool, _last_value] = False
    answer_evaluation_attempts: Annotated[int, _last_value] = 0
    graph_expansion_done: Annotated[bool, _last_value] = False

    # ── Synthesis state ─────────────────────────────────────────────────
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
    needs_clarification: Annotated[bool, _last_value] = False
    clarification_question: Annotated[Optional[str], _last_value] = None
    reflection_final: Annotated[Optional[dict], _last_value] = None  # {ready: bool, reasoning: str} from reflect_final_node

    # Wall-clock start of the current turn (time.monotonic()). Must be
    # declared: LangGraph silently drops updates for undeclared keys, which
    # previously made AGENT_MAX_WALL_SECONDS a no-op.
    started_at: Annotated[Optional[float], _last_value] = None
    # Set by tool_node when _verify_execution is already satisfied so
    # route_tool can short-circuit straight to reflect_final.
    force_finalize: Annotated[bool, _last_value] = False
    # Recovery tool calls injected by reflect_node for the next think round.
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
