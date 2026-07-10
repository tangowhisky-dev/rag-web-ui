"""Main LangGraph agent graph compilation.

Wires nodes into a StateGraph with proper edges and conditional routing.
Produces a compiled graph that can be executed via astream().
"""

from __future__ import annotations

from typing import Any, List, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from langchain_openai import ChatOpenAI

from .graph_state import AgentState
from .nodes import (
    classify_query_node,
    collect_answer_node,
    direct_retrieval_node,
    generate_node,
    fallback_response_node,
    orchestrator_node,
    request_clarification_node,
    rewrite_query_node,
    should_compress_context,
    summarize_history_node,
    synthesize_node,
    compress_context_node,
)
from .edges import (
    route_after_classification,
    route_after_clarification,
    route_after_orchestrator,
    route_after_should_compress,
)


def create_agent_graph(
    llm: ChatOpenAI | None = None,
    checkpointer: MemorySaver | None = None,
) -> Any:
    """Create the compiled LangGraph agent graph.
    
    Architecture:
    
    Main Graph:
        START -> rewrite -> classify -> [clarification|direct_retrieval|agent] -> synthesize -> END
    
    Agent Subgraph:
        START -> orchestrator -> [tools->should_compress->orchestrator|fallback|collect] -> END
    """
    
    # ── Agent subgraph (self-correcting retrieval loop) ─────────────────
    agent_builder = StateGraph(AgentState)
    
    agent_builder.add_node("orchestrator", orchestrator_node)
    agent_builder.add_node("collect", collect_answer_node)
    agent_builder.add_node("compress", compress_context_node)
    agent_builder.add_node("fallback", fallback_response_node)
    
    agent_builder.add_edge(START, "orchestrator")
    agent_builder.add_conditional_edges(
        "orchestrator", route_after_orchestrator,
        {"tools": "should_compress", "fallback": "fallback", "collect": "collect"},
    )
    
    # Tool execution placeholder — in full implementation, ToolNode(tools_list) goes here
    # For now, should_compress loops back to orchestrator since we don't have tools wired yet
    agent_builder.add_edge("compress", "should_compress")
    agent_builder.add_conditional_edges(
        "should_compress", route_after_should_compress,
        {"orchestrator": "orchestrator", "compress": "compress"},
    )
    agent_builder.add_edge("fallback", "collect")
    agent_builder.add_edge("collect", END)
    
    agent_subgraph = StateGraph(AgentState)
    agent_subgraph.add_node("agent", agent_builder.compile())
    agent_subgraph.add_edge(START, "agent")
    agent_subgraph.add_edge("agent", END)
    
    agent_graph = agent_subgraph.compile()
    
    # ── Main graph ──────────────────────────────────────────────────────
    graph_builder = StateGraph(AgentState)
    
    graph_builder.add_node("rewrite", rewrite_query_node)
    graph_builder.add_node("classify", classify_query_node)
    graph_builder.add_node("clarification", request_clarification_node)
    graph_builder.add_node("direct_retrieval", direct_retrieval_node)
    graph_builder.add_node("synthesize", synthesize_node)
    
    # The "agent" node is a special node — it invokes the agent subgraph
    graph_builder.add_node("agent", agent_graph)
    
    graph_builder.add_edge(START, "rewrite")
    graph_builder.add_conditional_edges(
        "classify", route_after_classification,
        {"clarification": "clarification", "direct": "direct_retrieval", "agent": "agent"},
    )
    graph_builder.add_conditional_edges(
        "clarification", route_after_clarification,
        {"classify": "classify"},
    )
    graph_builder.add_edge("direct_retrieval", "synthesize")
    graph_builder.add_edge("agent", "synthesize")
    graph_builder.add_edge("synthesize", END)
    
    main_graph = graph_builder.compile(checkpointer=checkpointer)
    return main_graph
