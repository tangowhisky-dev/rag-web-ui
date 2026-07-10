"""Compiled LangGraph StateGraph for the agentic RAG pipeline.

Two-level architecture:
  Main graph:  START → rewrite → classify → [direct_retrieval | agent_subgraph] → synthesize → END
  Agent subgraph: START → orchestrator → direct_retrieval → sufficiency_check → [graph_expansion | reranking | adaptive_reranking | generating | chart_validation] → collect/fallback → END

The main graph handles the simple/complex routing and final synthesis.
The agent subgraph handles iterative retrieval with all documented steps:
  1. rewriting → 2. keyword_search → 3. dense_search → 4. sparse_search →
  5. sufficiency_check → 6. graph_expansion → 7. reranking →
  8. adaptive_reranking (conditional) → 9. generating → 10. chart_validation (conditional) → collect

All node dependencies (db, kb_ids, etc.) are injected via functools.partial
so the compiled graph can call each node with only (state, config).
"""

from __future__ import annotations

from functools import partial
from typing import Any, List, Optional

from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI

from app.core.config import settings

from .graph_state import AgentState
from .nodes import (
    rewrite_query_node,
    classify_query_node,
    request_clarification_node,
    direct_retrieval_node,
    orchestrator_node,
    collect_answer_node,
    synthesize_node,
    fallback_response_node,
    sufficiency_check_node,
    generating_node,
    adaptive_reranking_node,
    chart_validation_node,
    compress_context_node,
    should_compress_context,
)


# ---------------------------------------------------------------------------
# Routing / edge functions
# ---------------------------------------------------------------------------

def route_after_classify(state: AgentState) -> str:
    """Decide which path to take after classification."""
    if not state.get("question_is_clear", True):
        return "request_clarification"
    if state.get("is_complex", False) and len(state.get("subtasks", [])) > 1:
        return "agent_subgraph"
    return "direct_retrieval"


def route_after_orchestrator(state: AgentState) -> str:
    """Route from orchestrator in the agent subgraph."""
    tool_call_count = sum(
        1 for m in state.get("messages", [])
        if hasattr(m, "tool_calls") and m.tool_calls
    )
    iteration_count = state.get("retrieval_iterations", 0)

    if iteration_count >= 8 or tool_call_count >= 20:
        return "fallback_response"

    return "direct_retrieval"


def route_after_sufficiency(state: AgentState) -> str:
    """Route based on sufficiency check result."""
    if state.get("needs_graph_expansion", False):
        return "graph_expansion"
    return "reranking"


def route_after_generating(state: AgentState) -> str:
    """Route after generating to chart validation (conditional) or collect."""
    if state.get("is_chart_query", False):
        return "chart_validation"
    return "collect_answer"


# ---------------------------------------------------------------------------
# Agent subgraph builder
# ---------------------------------------------------------------------------

def build_agent_subgraph(
    db: Any,
    kb_ids: List[int],
    org_id: Optional[int] = None,
    file_markdown: Optional[str] = None,
    use_dense: bool = True,
    use_sparse: bool = True,
    use_exact: bool = True,
    use_graph_rag: bool = False,
    api_base: Optional[str] = None,
) -> StateGraph:
    """Build and return the agent subgraph (not yet compiled).

    Injects ``db``, ``kb_ids``, etc. via ``functools.partial`` so LangGraph
    only needs to pass (state, config) to each node.
    """
    builder = StateGraph(AgentState)

    # Nodes — wrap in partial to inject db, kb_ids, etc.
    builder.add_node(
        "orchestrator",
        partial(orchestrator_node, llm=ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0.0,
            openai_api_base=api_base or settings.OPENAI_API_BASE,
            openai_api_key=settings.OPENAI_API_KEY,
            streaming=False,
        )),
    )

    # Retrieval (all legs bundled into single node for efficiency)
    builder.add_node(
        "direct_retrieval",
        partial(direct_retrieval_node, db=db, kb_ids=kb_ids,
                org_id=org_id, file_markdown=file_markdown,
                use_dense=use_dense, use_sparse=use_sparse,
                use_exact=use_exact, use_graph_rag=use_graph_rag),
    )

    # Post-retrieval nodes (documented pipeline steps)
    builder.add_node("sufficiency_check", sufficiency_check_node)
    builder.add_node("graph_expansion", partial(direct_retrieval_node, db=db, kb_ids=kb_ids,
                    org_id=org_id, file_markdown=file_markdown,
                    use_dense=use_dense, use_sparse=use_sparse,
                    use_exact=use_exact, use_graph_rag=True))
    builder.add_node("reranking", partial(direct_retrieval_node, db=db, kb_ids=kb_ids,
                         org_id=org_id, file_markdown=file_markdown,
                         use_dense=use_dense, use_sparse=use_sparse,
                         use_exact=use_exact, use_graph_rag=use_graph_rag))
    builder.add_node("adaptive_reranking", adaptive_reranking_node)
    builder.add_node("generating", generating_node)
    builder.add_node("chart_validation", chart_validation_node)
    builder.add_node("compress_context", compress_context_node)
    builder.add_node("fallback_response", fallback_response_node)
    builder.add_node("collect_answer", collect_answer_node)

    # Edges — structured per documented pipeline
    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "direct_retrieval": "direct_retrieval",
            "fallback_response": "fallback_response",
        },
    )

    # After retrieval → sufficiency check
    builder.add_conditional_edges(
        "direct_retrieval",
        route_after_sufficiency,
        {
            "graph_expansion": "graph_expansion",
            "reranking": "reranking",
        },
    )

    # Graph expansion → reranking → adaptive_reranking → generating
    builder.add_edge("graph_expansion", "reranking")
    builder.add_edge("reranking", "adaptive_reranking")
    builder.add_edge("adaptive_reranking", "generating")

    # Generating routes to chart validation (conditional) or collect
    builder.add_conditional_edges(
        "generating",
        route_after_generating,
        {
            "chart_validation": "chart_validation",
            "collect_answer": "collect_answer",
        },
    )

    # Chart validation → collect
    builder.add_edge("chart_validation", "collect_answer")

    # Fallback → collect
    builder.add_edge("fallback_response", "collect_answer")

    # Collect → END
    builder.add_edge("collect_answer", END)

    return builder


# ---------------------------------------------------------------------------
# Main graph builder
# ---------------------------------------------------------------------------

def build_main_graph(
    db: Any = None,
    kb_ids: Optional[List[int]] = None,
    org_id: Optional[int] = None,
    file_markdown: Optional[str] = None,
    use_dense: bool = True,
    use_sparse: bool = True,
    use_exact: bool = True,
    use_graph_rag: bool = False,
    api_base: Optional[str] = None,
) -> Any:
    """Build and compile the main LangGraph StateGraph.

    Returns the compiled graph ready for execution.
    """
    builder = StateGraph(AgentState)

    # --- Nodes ---
    builder.add_node("rewrite_query", rewrite_query_node)
    builder.add_node("classify_query", classify_query_node)
    builder.add_node("request_clarification", request_clarification_node)
    builder.add_node("synthesize", synthesize_node)

    # Build and compile the agent subgraph
    agent_builder = build_agent_subgraph(
        db=db,
        kb_ids=kb_ids or [],
        org_id=org_id,
        file_markdown=file_markdown,
        use_dense=use_dense,
        use_sparse=use_sparse,
        use_exact=use_exact,
        use_graph_rag=use_graph_rag,
        api_base=api_base,
    )
    agent_subgraph = agent_builder.compile()
    builder.add_node("agent_subgraph", agent_subgraph)

    # --- Edges ---
    builder.add_edge(START, "rewrite_query")
    builder.add_edge("rewrite_query", "classify_query")

    # Conditional routing after classification
    builder.add_conditional_edges(
        "classify_query",
        route_after_classify,
        {
            "request_clarification": "request_clarification",
            "direct_retrieval": "direct_retrieval",
            "agent_subgraph": "agent_subgraph",
        },
    )

    # Clarification loops back to classification
    builder.add_edge("request_clarification", "classify_query")

    # Simple path: direct_retrieval → synthesize
    builder.add_node(
        "direct_retrieval",
        partial(direct_retrieval_node, db=db, kb_ids=kb_ids,
                org_id=org_id, file_markdown=file_markdown,
                use_dense=use_dense, use_sparse=use_sparse,
                use_exact=use_exact, use_graph_rag=use_graph_rag),
    )
    builder.add_edge("direct_retrieval", "synthesize")
    builder.add_edge("agent_subgraph", "synthesize")
    builder.add_edge("synthesize", END)

    return builder.compile()
