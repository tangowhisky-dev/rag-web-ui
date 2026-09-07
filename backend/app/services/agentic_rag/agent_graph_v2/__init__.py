"""Agent graph v2 — unified loop topology.

load_context → think ⇄ tool → post_process → END

The think node calls the LLM with all tools bound. If the LLM emits tool
calls, the graph dispatches them in the tool node and loops back to think.
If the LLM emits no tool calls, the text IS the final answer → post_process.

No separate planner, sufficiency checker, or finalizer. The LLM decides
when it has enough evidence by not calling more tools.
"""

from __future__ import annotations

from .build import build_agent_graph_v2

__all__ = ["build_agent_graph_v2"]
