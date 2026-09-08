"""State for the agentic-v2 graph.

Reuses the existing AgentState (from graph_state.py) to avoid duplicating
the schema and all its reducers. The v2 pipeline only uses a subset of the
fields, but keeping the same state class means the checkpointer, streaming,
and tool infrastructure all work without modification.
"""

from __future__ import annotations

from app.services.agentic_rag.graph_state import AgentState

# Re-export for convenience — v2 uses the same state.
AgentStateV2 = AgentState

__all__ = ["AgentStateV2"]
