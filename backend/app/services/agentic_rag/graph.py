"""Compiled LangGraph StateGraph for the agentic RAG pipeline.

Two-level architecture:
  Main graph:  START → summarize_history → rewrite → classify → [clarification | Send(agent, ...)]
               → prepare_final_context → generate → [chart_validation →] answer_evaluation
               → finalize_answer → END
  Agent subgraph: START → rewrite_subtask_query → dense → sparse → exact → merge
                        → sufficiency_check → [graph_expansion →] reranking
                        → adaptive_reranking → collect_context → END

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

from .graph_state import AgentState
from .nodes import (
    rewrite_query_node,
    rewrite_subtask_query_node,
    classify_query_node,
    request_clarification_node,
    load_historical_memory_node,
    dense_retrieval_node,
    sparse_retrieval_node,
    exact_retrieval_node,
    merge_node,
    reranking_node,
    graph_expansion_node,
    collect_context_node,
    prepare_final_context_node,
    finalize_answer_node,
    sufficiency_check_node,
    generating_node,
    adaptive_reranking_node,
    chart_validation_node,
    summarize_history_node,
    answer_evaluation_node,
)


# ---------------------------------------------------------------------------
# Routing / edge functions
# ---------------------------------------------------------------------------

def route_after_classify(state: AgentState) -> list[Send] | str:
    """Decide which path to take after classification.
    
    Returns a list of Send objects for parallel independent subtask fan-out,
    a list of sequential Send objects for dependent subtasks, or a string
    for simple queries / clarification.
    """
    if not state.get("question_is_clear", True):
        return "request_clarification"
    
    subtasks = state.get("subtasks", [])
    subtask_independence = state.get("subtask_independence", [True] * len(subtasks))
    
    if len(subtasks) > 1:
        # Separate independent vs dependent subtasks
        independent_subtasks = []
        dependent_subtasks = []
        
        for i, subtask in enumerate(subtasks):
            is_independent = subtask_independence[i] if i < len(subtask_independence) else True
            if is_independent:
                independent_subtasks.append(subtask)
            else:
                dependent_subtasks.append(subtask)
        
        sends = []
        
        # Independent subtasks: parallel fan-out via Send()
        for subtask in independent_subtasks:
            sends.append(Send("agent_subgraph", {
                "original_query": subtask,
                "rewritten_query": subtask,
                "messages": [],
                "subtasks": [subtask],
                "is_complex": False,
                "current_subtask_index": 0,
            }))
        
        # Dependent subtasks: sequential execution via explicit Send chaining
        if len(dependent_subtasks) > 1:
            # Chain dependent subtasks sequentially through the agent subgraph
            sends.append(Send("agent_subgraph", {
                "original_query": " ".join(dependent_subtasks),
                "rewritten_query": " ".join(dependent_subtasks),
                "messages": [],
                "subtasks": dependent_subtasks,
                "is_complex": True,
                "current_subtask_index": 0,
            }))
        elif dependent_subtasks:
            # Single dependent subtask
            sends.append(Send("agent_subgraph", {
                "original_query": dependent_subtasks[0],
                "rewritten_query": dependent_subtasks[0],
                "messages": [],
                "subtasks": dependent_subtasks,
                "is_complex": False,
                "current_subtask_index": 0,
            }))
        
        return sends if sends else "agent_subgraph"
    
    # Simple query: route through full agent_subgraph pipeline
    return "agent_subgraph"


def route_after_sufficiency(state: AgentState) -> str:
    """Route based on sufficiency check result."""
    if state.get("needs_graph_expansion", False):
        return "graph_expansion"
    return "adaptive_reranking"


def route_after_generating(state: AgentState) -> str:
    """Route after generating to answer_evaluation (conditional) or chart_validation."""
    if state.get("is_chart_query", False):
        return "chart_validation"
    return "answer_evaluation"


def route_after_chart_validation(state: AgentState) -> str:
    """Route after chart validation in the main orchestrator.

    Valid chart JSON proceeds to answer evaluation. Invalid JSON is sent
    back to generation for up to 3 retries; after that it proceeds as
    low-confidence.
    """
    if not state.get("is_chart_query", False):
        return "answer_evaluation"

    valid = state.get("chart_data", {}).get("valid", False)
    retries = state.get("chart_retries", 0)

    if valid or retries >= 3:
        return "answer_evaluation"
    return "generating"


def route_after_adaptive_reranking(state: AgentState) -> str:
    """Route after adaptive reranking: collect context and end subagent."""
    return "collect_context"


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

    Sequential pipeline per subtask (retrieval-only):
      rewrite_subtask_query
        → exact_retrieval → sparse_retrieval → dense_retrieval
        → merge → reranking → sufficiency_check
        → [graph_expansion → reranking → sufficiency_check]
        → adaptive_reranking → collect_context → END

    The main orchestrator then consumes all subtask_contexts to generate,
    validate, and finalize the final answer.

    All db/kb parameters are injected via functools.partial so LangGraph
    only sees (state, config) at each node.
    """
    builder = StateGraph(AgentState)

    # ── Subtask rewrite ─────────────────────────────────────────────────
    builder.add_node("rewrite_subtask_query", rewrite_subtask_query_node)

    # ── Retrieval legs (sequential, each handles its own retry) ─────────
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
    builder.add_node("sufficiency_check", sufficiency_check_node)
    builder.add_node(
        "graph_expansion",
        partial(graph_expansion_node, db=db, kb_ids=kb_ids,
                org_id=org_id, file_markdown=file_markdown),
    )
    # reranking_node is sync and needs no db injection
    builder.add_node("reranking", reranking_node)
    builder.add_node(
        "adaptive_reranking",
        partial(adaptive_reranking_node, db=db, kb_ids=kb_ids,
                org_id=org_id, file_markdown=file_markdown),
    )

    # ── Subgraph output ─────────────────────────────────────────────────
    builder.add_node("collect_context", collect_context_node)

    # ── Edges ────────────────────────────────────────────────────────────
    # Retrieval: sequential so all three legs always run and feed merge
    builder.add_edge(START, "rewrite_subtask_query")
    builder.add_edge("rewrite_subtask_query", "exact_retrieval")
    builder.add_edge("exact_retrieval", "sparse_retrieval")
    builder.add_edge("sparse_retrieval", "dense_retrieval")
    builder.add_edge("dense_retrieval", "merge")
    builder.add_edge("merge", "reranking")

    # Reranker is the single authoritative ranking step; confidence is computed
    # from reranker scores. If retrieval is insufficient, expand via Neo4j and
    # rerank again; otherwise collect the context and end the subagent.
    builder.add_edge("reranking", "sufficiency_check")
    builder.add_conditional_edges(
        "sufficiency_check",
        route_after_sufficiency,
        {
            "graph_expansion": "graph_expansion",
            "adaptive_reranking": "adaptive_reranking",
        },
    )
    builder.add_edge("graph_expansion", "reranking")

    builder.add_edge("adaptive_reranking", "collect_context")
    builder.add_edge("collect_context", END)

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
) -> Any:
    """Build and compile the main LangGraph StateGraph.

    Returns the compiled graph ready for execution.
    """
    builder = StateGraph(AgentState)

    # --- Nodes ---
    builder.add_node("load_historical_memory", partial(load_historical_memory_node, db=db))
    builder.add_node("summarize_history", summarize_history_node)
    builder.add_node("rewrite_query", rewrite_query_node)
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

    # --- Edges ---
    builder.add_edge(START, "load_historical_memory")
    builder.add_edge("load_historical_memory", "summarize_history")
    builder.add_edge("summarize_history", "rewrite_query")
    builder.add_edge("rewrite_query", "classify_query")

    # Conditional routing after classification (Send for parallel subtasks)
    builder.add_conditional_edges(
        "classify_query",
        route_after_classify,
        {
            "request_clarification": "request_clarification",
            "agent_subgraph": "agent_subgraph",
        },
    )

    # Clarification loops back to rewrite
    builder.add_edge("request_clarification", "rewrite_query")

    # Subagents return contexts; main orchestrator generates the final answer
    builder.add_edge("agent_subgraph", "prepare_final_context")
    builder.add_edge("prepare_final_context", "generating")

    # Generation → chart_validation (if chart query) or answer_evaluation
    builder.add_conditional_edges(
        "generating",
        route_after_generating,
        {
            "chart_validation": "chart_validation",
            "answer_evaluation": "answer_evaluation",
        },
    )

    # Chart validation → answer_evaluation (valid/exhausted) or re-generate
    builder.add_conditional_edges(
        "chart_validation",
        route_after_chart_validation,
        {
            "answer_evaluation": "answer_evaluation",
            "generating": "generating",
        },
    )

    # Answer evaluation → finalize (no automatic retry)
    builder.add_edge("answer_evaluation", "finalize_answer")

    builder.add_edge("finalize_answer", END)

    # Compile with an InMemorySaver checkpointer.
    # AsyncSqliteSaver.from_conn_string() returns an async context manager,
    # but build_main_graph is synchronous. Using InMemorySaver avoids the
    # complexity of managing an async connection here.
    from langgraph.checkpoint.memory import InMemorySaver
    checkpointer = InMemorySaver()

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["request_clarification"],
    )
