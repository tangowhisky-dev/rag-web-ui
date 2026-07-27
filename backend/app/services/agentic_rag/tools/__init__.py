"""Agent tool registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.agentic_rag.tool_context import ToolContext

from .rag_retrieve import make_rag_retrieve_tool


def build_tools(ctx: "ToolContext") -> list:
    """Return tool instances bound to the given ToolContext."""
    return [
        make_rag_retrieve_tool(ctx),
    ]
