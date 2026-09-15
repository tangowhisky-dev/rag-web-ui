"""Graph builder for agentic-v2.

Topology:
  load_context → think ⇄ tool → post_process → END

The think node calls the LLM with all tools. If tool calls are emitted,
the tool node dispatches them and loops back to think. If no tool calls,
the text is the answer → post_process normalizes and persists it.
"""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, StateGraph

from app.services.agentic_rag.graph_state import AgentState

from ..agent_graph.load_context import load_context_node
from .post_process import post_process_node_v2
from .thinking import route_think_v2, think_node_v2
from .tooling import tool_node_v2


def build_agent_graph_v2(ctx):
    """Compile and return the agentic-v2 graph."""
    graph = StateGraph(AgentState)

    graph.add_node("load_context", partial(load_context_node, ctx=ctx))
    graph.add_node("think", partial(think_node_v2, ctx=ctx))
    graph.add_node("tool", partial(tool_node_v2, ctx=ctx))
    graph.add_node("post_process", partial(post_process_node_v2, ctx=ctx))

    graph.set_entry_point("load_context")
    graph.add_edge("load_context", "think")
    graph.add_conditional_edges("think", route_think_v2)
    graph.add_edge("tool", "think")
    graph.add_edge("post_process", END)

    checkpointer = getattr(ctx.redis_memory, "checkpointer", None) if ctx.redis_memory else None
    return graph.compile(checkpointer=checkpointer)


def build_agent_graph_fast(ctx):
    """Compile the fast pipeline: committed plan, no ReAct loop.

    Topology:
      load_context → fast_plan → fast_step ⇄ tool → post_process → END

    - fast_plan resolves intent/query and emits the full ordered plan in one
      utility-model call (or no steps for direct/history-only answers).
    - fast_step materializes each planned step's tool_calls between rounds —
      literal args pass through, needs_prior args are compiled by one
      utility-model call that sees prior observations.
    - tool is the existing tool_node_v2 — same dispatch, dedup, timeline and
      debug events, and retrieved_docs merge as the agentic loop.
    - post_process is the existing post_process_node_v2 — identical answer
      generation, citation normalization, persistence, and memory write-back.
    """
    from .fast_plan import fast_plan_node, fast_step_node, route_fast_plan, route_fast_step

    graph = StateGraph(AgentState)

    graph.add_node("load_context", partial(load_context_node, ctx=ctx))
    graph.add_node("fast_plan", partial(fast_plan_node, ctx=ctx))
    graph.add_node("fast_step", partial(fast_step_node, ctx=ctx))
    graph.add_node("tool", partial(tool_node_v2, ctx=ctx))
    graph.add_node("post_process", partial(post_process_node_v2, ctx=ctx))

    graph.set_entry_point("load_context")
    graph.add_edge("load_context", "fast_plan")
    graph.add_conditional_edges("fast_plan", route_fast_plan)
    graph.add_edge("tool", "fast_step")
    graph.add_conditional_edges("fast_step", route_fast_step)
    graph.add_edge("post_process", END)

    checkpointer = getattr(ctx.redis_memory, "checkpointer", None) if ctx.redis_memory else None
    return graph.compile(checkpointer=checkpointer)
