"""Graph builder — assembles and compiles the LangGraph agent loop.

Wires all nodes (load_context, expand_query, rewrite_query, plan,
clarify_interrupt, think, tool, reflect, reflect_final, finalize,
answer_scoring, save_memory) and their conditional edges into a
StateGraph, then compiles it with the Redis checkpointer (if available).
"""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, StateGraph

from app.services.agentic_rag.graph_state import AgentState
from app.services.agentic_rag.nodes import expand_query_node, rewrite_query_node

from .finalization import finalize_node, save_memory_node
from .load_context import load_context_node
from .planning import plan_node, route_plan
from .reflection import answer_scoring_node, clarify_interrupt_node, reflect_final_node, reflect_node
from .thinking import route_think, think_node
from .tooling import route_reflect_final, route_tool, tool_node


def build_agent_graph(ctx):
    """Compile and return the agent loop graph."""
    graph = StateGraph(AgentState)

    graph.add_node("load_context", partial(load_context_node, ctx=ctx))
    # Resolve query-role LLM config for rewrite_query_node
    from app.services.agentic_rag.llm_factory import get_org_llm
    query_cfg = get_org_llm(ctx.org_id, ctx.db, role="query")
    graph.add_node("rewrite_query", partial(rewrite_query_node,
                                            api_base=query_cfg["api_base"],
                                            api_key=query_cfg["api_key"],
                                            query_model=query_cfg["model_name"],
                                            db=ctx.db,
                                            org_id=ctx.org_id))
    graph.add_node("expand_query", partial(expand_query_node, db=ctx.db, org_id=ctx.org_id))
    graph.add_node("plan", partial(plan_node, ctx=ctx))
    graph.add_node("clarify_interrupt", clarify_interrupt_node)
    graph.add_node("think", partial(think_node, ctx=ctx))
    graph.add_node("tool", partial(tool_node, ctx=ctx))
    graph.add_node("reflect", partial(reflect_node, ctx=ctx))
    graph.add_node("reflect_final", partial(reflect_final_node, ctx=ctx))
    graph.add_node("finalize", partial(finalize_node, ctx=ctx))
    graph.add_node("answer_scoring", partial(answer_scoring_node, ctx=ctx))
    graph.add_node("save_memory", partial(save_memory_node, ctx=ctx))

    graph.set_entry_point("load_context")
    graph.add_edge("load_context", "expand_query")
    graph.add_edge("expand_query", "rewrite_query")
    graph.add_edge("rewrite_query", "plan")
    graph.add_conditional_edges("plan", route_plan)
    # Back through expansion + rewrite, not straight to plan: the clarification
    # answer has to reach the retrieval query, which was computed from the
    # original ambiguous message.
    graph.add_edge("clarify_interrupt", "expand_query")
    graph.add_conditional_edges("think", route_think)
    graph.add_conditional_edges("tool", route_tool)
    graph.add_edge("reflect", "think")
    graph.add_conditional_edges("reflect_final", route_reflect_final)
    graph.add_edge("finalize", "answer_scoring")
    graph.add_edge("answer_scoring", "save_memory")
    graph.add_edge("save_memory", END)

    checkpointer = getattr(ctx.redis_memory, "checkpointer", None) if ctx.redis_memory else None
    return graph.compile(checkpointer=checkpointer)
