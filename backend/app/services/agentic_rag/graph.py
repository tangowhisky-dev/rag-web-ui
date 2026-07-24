"""Compiled LangGraph StateGraph for the agentic RAG pipeline.

Two-level architecture:
  Main graph:  START -> rewrite -> compaction -> classify -> route_by_dependencies
               -> [Send(agent, ...) | sequential_subtask_loop]
               -> prepare_final_context -> generate -> [chart_validation ->] answer_evaluation
               -> finalize_answer -> END
  Agent subgraph: START -> exact -> sparse -> dense
                        -> merge -> neo4j_expansion -> reranking(-inf) -> filter(-2.0)
                        -> sufficiency_check -> [adaptive_reranking(-5.0)]
                        -> collect_context -> END
  Sequential loop: START -> enrich_subtask_query -> [agent subgraph nodes] -> collect_context
                   -> increment_index -> check_done -> [loop back | return]

Subagents are retrieval-only: they return ranked contexts. The main orchestrator
aggregates all subtask contexts and generates/validates the final answer once.

All node dependencies (db, kb_ids, etc.) are injected via functools.partial
so the compiled graph can call each node with only (state, config).
"""

from __future__ import annotations

from functools import partial
from typing import Any, List, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from langchain_openai import ChatOpenAI

from app.core.config import settings
import logging

from .graph_state import AgentState
from .nodes import (
    rewrite_query_node,
    rewrite_subtask_query_node,
    load_subtask_memory_node,
    enrich_subtask_query_node,
    classify_query_node,
    request_clarification_node,
    dense_retrieval_node,
    sparse_retrieval_node,
    exact_retrieval_node,
    merge_node,
    neo4j_expansion_node,
    reranking_node,
    filter_node,
    sufficiency_check_node,
    adaptive_reranking_node,
    collect_context_node,
    prepare_final_context_node,
    finalize_answer_node,
    generating_node,
    chart_validation_node,
    answer_evaluation_node,
    save_memory_node,
    compaction_node,
)


# ---------------------------------------------------------------------------
# Routing / edge functions
# ---------------------------------------------------------------------------

def _trim_history_for_subgraph(state: AgentState) -> list:
    """Return the last few conversation turns to carry into subagents."""
    from .nodes import select_recent_history

    return select_recent_history(state.get("messages", []), max_pairs=2)


def _subgraph_send_kwargs(state: AgentState) -> dict:
    """Return the shared context every subgraph invocation should receive."""
    return {
        "chat_id": state.get("chat_id"),
        "user_id": state.get("user_id"),
        "kb_ids": state.get("kb_ids", []),
        "org_id": state.get("org_id"),
        "file_markdown": state.get("file_markdown"),
    }


def route_by_dependencies(state: AgentState) -> list[Send] | str:
    """Decide which path to take after classification.

    Routes subtasks based on per-subtask routing flags:
    - needs_retrieval=True  → agent_subgraph (full retrieval pipeline)
    - needs_retrieval=False → chat_subgraph (pass state to generating)
    - needs_file_content=True → file_context_subgraph (pass file content to generating)

    Independent subtasks are fanned out in parallel via Send().
    Dependent subtasks run sequentially.
    Mixed subtasks (some retrieval, some chat) are fanned out as separate Send()
    calls — the main orchestrator collects all contexts at prepare_final_context.
    """
    if not state.get("question_is_clear", True):
        return "request_clarification"

    subtasks = state.get("subtasks", [])
    dependencies = state.get("subtask_dependencies", [[] for _ in subtasks])
    subtask_routing = state.get("subtask_routing", [])
    needs_file_content = state.get("needs_file_content", False)
    needs_file_metadata = state.get("needs_file_metadata", False)
    file_markdown = state.get("file_markdown")

    # If there are no subtasks, route as a single default subtask.
    if not subtasks:
        single_routing = {"needs_retrieval": True, "needs_file_content": False, "needs_file_metadata": False}
        return [Send("agent_subgraph", {
            **_subgraph_send_kwargs(state),
            "original_query": state.get("original_query", ""),
            "rewritten_query": state.get("rewritten_query", state.get("original_query", "")),
            "subgraph_history": _trim_history_for_subgraph(state),
            "subtasks": [state.get("original_query", "")],
            "is_complex": False,
            "current_subtask_index": 0,
            "needs_retrieval": True,
            "needs_file_content": needs_file_content,
            "needs_file_metadata": needs_file_metadata,
            "subtask_routing": [single_routing],
        })]

    # Check if any subtask has dependencies
    has_dependencies = any(len(deps) > 0 for deps in dependencies)

    # Detect circular dependencies — fall back to parallel if cycles found
    if has_dependencies and _has_circular_deps(dependencies):
        logger.warning(
            "[ROUTE] circular dependency detected in %d subtasks — "
            "falling back to parallel execution",
            len(subtasks),
        )
        has_dependencies = False

    if not has_dependencies:
        # Independent subtasks — route each to its appropriate subgraph.
        sends = []
        for i, subtask in enumerate(subtasks):
            routing = subtask_routing[i] if i < len(subtask_routing) else {
                "needs_retrieval": True,
                "needs_file_content": False,
                "needs_file_metadata": False,
            }
            node = "agent_subgraph"
            if not routing.get("needs_retrieval", True):
                node = "chat_subgraph"
            elif routing.get("needs_file_content", False) or routing.get("needs_file_metadata", False):
                node = "file_context_subgraph"

            sends.append(Send(node, {
                **_subgraph_send_kwargs(state),
                "original_query": subtask,
                "rewritten_query": subtask,
                "subgraph_history": _trim_history_for_subgraph(state),
                "subtasks": [subtask],
                "is_complex": False,
                "current_subtask_index": 0,
                "needs_retrieval": routing.get("needs_retrieval", True),
                "needs_file_content": routing.get("needs_file_content", False),
                "needs_file_metadata": routing.get("needs_file_metadata", False),
                "subtask_routing": [routing],
            }))
        return sends if sends else "agent_subgraph"

    # Has dependencies — run sequentially via loop
    return "sequential_subtask_loop"


def _has_circular_deps(dependencies: list[list[int]]) -> bool:
    """Detect cycles in the subtask dependency graph using DFS.

    Returns True if a cycle exists.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * len(dependencies)

    def dfs(node: int) -> bool:
        color[node] = GRAY
        for dep in dependencies[node]:
            if dep < 0 or dep >= len(dependencies):
                continue  # skip invalid indices
            if color[dep] == GRAY:
                return True  # back edge = cycle
            if color[dep] == WHITE and dfs(dep):
                return True
        color[node] = BLACK
        return False

    return any(color[i] == WHITE and dfs(i) for i in range(len(dependencies)))


def route_after_sufficiency(state: AgentState) -> str:
    """Route based on sufficiency check result."""
    if state.get("needs_adaptive_reranking", False):
        return "adaptive_reranking"
    return "collect_context"


def route_after_generating(state: AgentState) -> str:
    """Route after generating to answer_evaluation (conditional) or chart_validation."""
    if state.get("is_chart_query", False):
        return "chart_validation"
    return "answer_evaluation"


def route_after_chart_validation(state: AgentState) -> str:
    """Route after chart validation in the main orchestrator."""
    if not state.get("is_chart_query", False):
        return "answer_evaluation"

    valid = state.get("chart_data", {}).get("valid", False)
    retries = state.get("chart_retries", 0)

    if valid or retries >= 3:
        return "answer_evaluation"
    return "generating"


def route_after_answer_evaluation(state: AgentState) -> str:
    """Route after answer evaluation.

    If needs_retry is set and we haven't exhausted retries (attempts < 2),
    route back to generating. Otherwise proceed to finalize_answer.
    """
    if state.get("needs_retry", False) and state.get("answer_evaluation_attempts", 0) < 2:
        return "generating"
    return "finalize_answer"


# ---------------------------------------------------------------------------
# Sequential subtask loop subgraph
# ---------------------------------------------------------------------------

def increment_subtask_index(state: AgentState) -> dict:
    """Increment the current subtask index after a subtask completes."""
    idx = state.get("current_subtask_index", 0)
    return {"current_subtask_index": idx + 1}


def route_sequential_check(state: AgentState) -> str:
    """Route after incrementing index in the sequential subtask loop.

    If there are more subtasks to process, loop back to enrichment + exact_retrieval.
    Otherwise, exit the subgraph by going to END.
    """
    subtasks = state.get("subtasks", [])
    idx = state.get("current_subtask_index", 0)

    if idx < len(subtasks):
        return "enrich_subtask_query"
    return END


def build_sequential_subtask_loop(
    db: Any,
    kb_ids: List[int],
    org_id: Optional[int] = None,
    file_markdown: Optional[str] = None,
    api_base: Optional[str] = None,
    generate_answer: bool = True,
) -> StateGraph:
    """Build the sequential subtask loop subgraph.

    Executes dependent subtasks one-by-one. Each subtask reuses the same
    ``rewritten_query`` produced by the main graph's rewrite_query node.
    After each subtask completes, the index is incremented and the loop
    either continues or exits to collect_context.
    """
    builder = StateGraph(AgentState)

    # ── Retrieval legs (sequential) ─────────────────────────────────────
    builder.add_node(
        "dense_retrieval",
        partial(dense_retrieval_node, db=db, kb_ids=kb_ids,
                org_id=org_id, file_markdown=file_markdown),
    )
    builder.add_node(
        "sparse_retrieval",
        partial(sparse_retrieval_node, db=db, kb_ids=kb_ids,
                org_id=org_id, file_markdown=file_markdown),
    )
    builder.add_node(
        "exact_retrieval",
        partial(exact_retrieval_node, db=db, kb_ids=kb_ids,
                org_id=org_id, file_markdown=file_markdown),
    )

    # ── Post-retrieval pipeline ──────────────────────────────────────────
    builder.add_node("merge", partial(merge_node, file_markdown=file_markdown))
    builder.add_node(
        "neo4j_expansion",
        partial(neo4j_expansion_node, db=db, kb_ids=kb_ids,
                org_id=org_id, file_markdown=file_markdown),
    )
    builder.add_node("reranking", reranking_node)
    builder.add_node("filter", filter_node)
    builder.add_node("sufficiency_check", sufficiency_check_node)
    builder.add_node("adaptive_reranking", adaptive_reranking_node)

    # ── Subtask loop control ────────────────────────────────────────────
    builder.add_node("enrich_subtask_query", enrich_subtask_query_node)
    builder.add_node("collect_context", collect_context_node)
    builder.add_node("increment_index", increment_subtask_index)

    # ── Edges ────────────────────────────────────────────────────────────
    builder.add_edge(START, "enrich_subtask_query")
    builder.add_edge("enrich_subtask_query", "exact_retrieval")
    builder.add_edge("exact_retrieval", "sparse_retrieval")
    builder.add_edge("sparse_retrieval", "dense_retrieval")
    builder.add_edge("dense_retrieval", "merge")
    builder.add_edge("merge", "neo4j_expansion")
    builder.add_edge("neo4j_expansion", "reranking")
    builder.add_edge("reranking", "filter")
    builder.add_edge("filter", "sufficiency_check")
    builder.add_conditional_edges(
        "sufficiency_check",
        route_after_sufficiency,
        {
            "adaptive_reranking": "adaptive_reranking",
            "collect_context": "collect_context",
        },
    )
    builder.add_edge("adaptive_reranking", "collect_context")
    builder.add_edge("collect_context", "increment_index")
    builder.add_conditional_edges(
        "increment_index",
        route_sequential_check,
        {
            "enrich_subtask_query": "enrich_subtask_query",
            END: END,
        },
    )

    return builder


# ---------------------------------------------------------------------------
# Agent subgraph builder
# ---------------------------------------------------------------------------

def build_agent_subgraph(
    db: Any,
    kb_ids: List[int],
    org_id: Optional[int] = None,
    file_markdown: Optional[str] = None,
    api_base: Optional[str] = None,
    generate_answer: bool = True,
) -> StateGraph:
    """Build and return the agent subgraph (not yet compiled).

    Simplified pipeline per subtask (retrieval-only):
      rewrite_subtask_query -> load_subtask_memory
        -> exact_retrieval -> sparse_retrieval -> dense_retrieval
        -> merge -> neo4j_expansion -> reranking(-inf) -> filter(-2.0)
        -> sufficiency_check -> [adaptive_reranking(-5.0)]
        -> collect_context -> END

    All 4 legs run first, merge, then one reranking pass scores everything.
    Filter applies the standard threshold. If sufficiency fails, adaptive
    re-filters with the lower threshold (no re-running retrieval or reranker).

    The main orchestrator then consumes all subtask_contexts to generate,
    validate, and finalize the final answer.
    """
    builder = StateGraph(AgentState)

    # ── Subtask rewrite + memory ────────────────────────────────────────
    builder.add_node("rewrite_subtask_query", rewrite_subtask_query_node)
    builder.add_node("load_subtask_memory", partial(load_subtask_memory_node, db=db))

    # ── Retrieval legs (sequential) ─────────────────────────────────────
    builder.add_node(
        "dense_retrieval",
        partial(dense_retrieval_node, db=db, kb_ids=kb_ids,
                org_id=org_id, file_markdown=file_markdown),
    )
    builder.add_node(
        "sparse_retrieval",
        partial(sparse_retrieval_node, db=db, kb_ids=kb_ids,
                org_id=org_id, file_markdown=file_markdown),
    )
    builder.add_node(
        "exact_retrieval",
        partial(exact_retrieval_node, db=db, kb_ids=kb_ids,
                org_id=org_id, file_markdown=file_markdown),
    )

    # ── Post-retrieval pipeline ──────────────────────────────────────────
    builder.add_node("merge", partial(merge_node, file_markdown=file_markdown))
    builder.add_node(
        "neo4j_expansion",
        partial(neo4j_expansion_node, db=db, kb_ids=kb_ids,
                org_id=org_id, file_markdown=file_markdown),
    )
    builder.add_node("reranking", reranking_node)
    builder.add_node("filter", filter_node)
    builder.add_node("sufficiency_check", sufficiency_check_node)
    builder.add_node("adaptive_reranking", adaptive_reranking_node)

    # ── Subgraph output ─────────────────────────────────────────────────
    builder.add_node("collect_context", collect_context_node)

    # ── Edges ────────────────────────────────────────────────────────────
    builder.add_edge(START, "rewrite_subtask_query")
    builder.add_edge("rewrite_subtask_query", "load_subtask_memory")
    builder.add_edge("load_subtask_memory", "exact_retrieval")
    builder.add_edge("exact_retrieval", "sparse_retrieval")
    builder.add_edge("sparse_retrieval", "dense_retrieval")
    builder.add_edge("dense_retrieval", "merge")
    builder.add_edge("merge", "neo4j_expansion")
    builder.add_edge("neo4j_expansion", "reranking")
    builder.add_edge("reranking", "filter")
    builder.add_edge("filter", "sufficiency_check")
    builder.add_conditional_edges(
        "sufficiency_check",
        route_after_sufficiency,
        {
            "adaptive_reranking": "adaptive_reranking",
            "collect_context": "collect_context",
        },
    )
    builder.add_edge("adaptive_reranking", "collect_context")
    builder.add_edge("collect_context", END)

    return builder


# ---------------------------------------------------------------------------
# Chat subgraph builder (needs_retrieval=False)
# ---------------------------------------------------------------------------

def build_chat_subgraph() -> StateGraph:
    """Build a minimal subgraph for subtasks that only need conversation context.

    This subgraph collects the current messages state into subtask_contexts so
    prepare_final_context can merge it with retrieval results from other subtasks.
    It passes through full conversation history to the generation node via the
    checkpointer, with no document retrieval.
    """
    builder = StateGraph(AgentState)
    # No nodes needed — just collect_context to record this subtask's context type.
    builder.add_node("collect_context", collect_context_node)
    builder.add_edge(START, "collect_context")
    return builder


# ---------------------------------------------------------------------------
# File context subgraph builder (needs_file_content/metadata)
# ---------------------------------------------------------------------------

def build_file_context_subgraph(
    file_markdown: Optional[str] = None,
) -> StateGraph:
    """Build a minimal subgraph for subtasks that need uploaded file content.

    Collects the file_markdown as context in subtask_contexts so prepare_final_context
    can merge it with retrieval results from other subtasks.
    """
    builder = StateGraph(AgentState)
    builder.add_node("collect_context", collect_context_node)
    builder.add_edge(START, "collect_context")
    return builder


# ---------------------------------------------------------------------------
# Main graph builder
# ---------------------------------------------------------------------------

def build_main_graph(
    db: Any = None,
    kb_ids: Optional[List[int]] = None,
    org_id: Optional[int] = None,
    file_markdown: Optional[str] = None,
    api_base: Optional[str] = None,
    generate_answer: bool = True,
    query_model: Optional[str] = None,
    checkpointer: Any = None,
    store: Any = None,
) -> Any:
    """Build and compile the main LangGraph StateGraph."""
    builder = StateGraph(AgentState)

    # --- Nodes ---
    builder.add_node("rewrite_query", rewrite_query_node)
    builder.add_node("compaction", compaction_node)
    builder.add_node("classify_query", classify_query_node)
    builder.add_node("request_clarification", request_clarification_node)

    # Build and compile the agent subgraph (retrieval-only subagents)
    agent_builder = build_agent_subgraph(
        db=db,
        kb_ids=kb_ids or [],
        org_id=org_id,
        file_markdown=file_markdown,
        api_base=api_base,
        generate_answer=generate_answer,
    )
    agent_subgraph = agent_builder.compile()
    builder.add_node("agent_subgraph", agent_subgraph)

    # Build and compile the sequential subtask loop subgraph
    seq_builder = build_sequential_subtask_loop(
        db=db,
        kb_ids=kb_ids or [],
        org_id=org_id,
        file_markdown=file_markdown,
        api_base=api_base,
        generate_answer=generate_answer,
    )
    seq_subgraph = seq_builder.compile()
    builder.add_node("sequential_subtask_loop", seq_subgraph)

    # Build chat subgraph (minimal — no retrieval)
    chat_sg = build_chat_subgraph().compile()
    builder.add_node("chat_subgraph", chat_sg)

    # Build file context subgraph (minimal — passes file_markdown into context)
    file_sg = build_file_context_subgraph(file_markdown=file_markdown).compile()
    builder.add_node("file_context_subgraph", file_sg)

    # Main orchestrator: aggregates subagent contexts and generates the final answer
    builder.add_node("prepare_final_context", prepare_final_context_node)
    builder.add_node("generating", generating_node)
    builder.add_node(
        "answer_evaluation",
        partial(answer_evaluation_node, llm=ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0.0,
            openai_api_base=api_base or settings.OPENAI_API_BASE,
            openai_api_key=settings.OPENAI_API_KEY,
            streaming=False,
        )),
    )
    builder.add_node("chart_validation", chart_validation_node)
    builder.add_node("finalize_answer", finalize_answer_node)
    builder.add_node("save_memory", save_memory_node)

    # --- Edges ---
    builder.add_edge(START, "rewrite_query")
    builder.add_edge("rewrite_query", "compaction")
    builder.add_edge("compaction", "classify_query")

    # Conditional routing after classification (Send for parallel, loop for sequential, chat-only)
    builder.add_conditional_edges(
        "classify_query",
        route_by_dependencies,
        {
            "request_clarification": "request_clarification",
            "agent_subgraph": "agent_subgraph",
            "sequential_subtask_loop": "sequential_subtask_loop",
            "chat_subgraph": "chat_subgraph",
            "file_context_subgraph": "file_context_subgraph",
        },
    )

    # Clarification loops back to rewrite
    builder.add_edge("request_clarification", "rewrite_query")

    # All subgraph branches converge at prepare_final_context
    # (collect_context runs inside each subgraph)
    builder.add_edge("agent_subgraph", "prepare_final_context")
    builder.add_edge("sequential_subtask_loop", "prepare_final_context")
    builder.add_edge("chat_subgraph", "prepare_final_context")
    builder.add_edge("file_context_subgraph", "prepare_final_context")
    builder.add_edge("prepare_final_context", "generating")

    # Generation -> chart_validation (if chart query) or answer_evaluation
    builder.add_conditional_edges(
        "generating",
        route_after_generating,
        {
            "chart_validation": "chart_validation",
            "answer_evaluation": "answer_evaluation",
        },
    )

    # Chart validation -> answer_evaluation (valid/exhausted) or re-generate
    builder.add_conditional_edges(
        "chart_validation",
        route_after_chart_validation,
        {
            "answer_evaluation": "answer_evaluation",
            "generating": "generating",
        },
    )

    # Answer evaluation -> finalize -> save to long-term memory -> END
    builder.add_conditional_edges(
        "answer_evaluation",
        route_after_answer_evaluation,
        {"finalize_answer": "finalize_answer"},
    )
    builder.add_edge("finalize_answer", "save_memory")
    builder.add_edge("save_memory", END)

    return builder.compile(
        checkpointer=checkpointer,
        store=store,
        interrupt_before=["request_clarification"],
    )
