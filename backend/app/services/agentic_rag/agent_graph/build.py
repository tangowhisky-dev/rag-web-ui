"""Graph builder — assembles and compiles the LangGraph agent loop.

New topology (atomic tools redesign):
  load_context → plan → think → tool → sufficiency_check → finalize
                                                       ↓
                                                      think (if not sufficient)

  finalize → answer_scoring → save_memory → END

Legacy nodes (rewrite_query, expand_query, reflect, reflect_final) are
removed from the active graph. Their functions remain importable for
backward compatibility during migration.
"""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, StateGraph

from app.services.agentic_rag.graph_state import AgentState

from .finalization import finalize_node, save_memory_node
from .load_context import load_context_node
from .planning import plan_node, route_plan
from .reflection import answer_scoring_node, clarify_interrupt_node
from .sufficiency import route_sufficiency, sufficiency_check_node
from .thinking import route_think, think_node
from .tooling import tool_node


def build_agent_graph(ctx):
    """Compile and return the agent loop graph."""
    graph = StateGraph(AgentState)

    graph.add_node("load_context", partial(load_context_node, ctx=ctx))
    graph.add_node("plan", partial(plan_node, ctx=ctx))
    graph.add_node("clarify_interrupt", clarify_interrupt_node)
    graph.add_node("think", partial(think_node, ctx=ctx))
    graph.add_node("tool", partial(tool_node, ctx=ctx))
    graph.add_node("sufficiency_check", partial(sufficiency_check_node, ctx=ctx))
    graph.add_node("finalize", partial(finalize_node, ctx=ctx))
    graph.add_node("answer_scoring", partial(answer_scoring_node, ctx=ctx))
    graph.add_node("save_memory", partial(save_memory_node, ctx=ctx))

    graph.set_entry_point("load_context")
    graph.add_edge("load_context", "plan")
    graph.add_conditional_edges("plan", route_plan)
    # Clarification goes back to plan (not expand_query) in the new topology.
    graph.add_edge("clarify_interrupt", "plan")
    graph.add_conditional_edges("think", route_think)
    graph.add_edge("tool", "sufficiency_check")
    graph.add_conditional_edges("sufficiency_check", route_sufficiency)
    graph.add_edge("finalize", "answer_scoring")
    graph.add_edge("answer_scoring", "save_memory")
    graph.add_edge("save_memory", END)

    checkpointer = getattr(ctx.redis_memory, "checkpointer", None) if ctx.redis_memory else None
    return graph.compile(checkpointer=checkpointer)
