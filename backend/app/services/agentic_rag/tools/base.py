"""Base class for agent tools that bind to ToolContext."""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.services.agentic_rag.tool_context import ToolContext


class BaseAgentTool(BaseTool):
    """BaseTool subclass that receives a ToolContext and dispatches to _execute.

    Tools return an envelope dict:
        {"ok": bool, "result": dict, "error": str|None, "tokens": int, "terminate": bool}
    The ``terminate`` field defaults to False. When True, the tool node sets
    ``force_finalize = True`` to short-circuit the agent loop.
    """

    ctx: Optional[ToolContext] = Field(default=None, exclude=True)
    # Human-readable label shown in the frontend during tool execution.
    # Short, action-oriented, third-person: "Retrieving from knowledge base".
    ui_label: str = "Running tool"

    def prepare_arguments(self, args: dict) -> dict:
        """Normalize/validate arguments before execution. Override in subclasses."""
        return args

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """Parse validated input and call the concrete implementation."""
        kwargs.pop("run_manager", None)
        if args and isinstance(args[0], dict):
            kwargs = args[0]
        kwargs = self.prepare_arguments(kwargs)
        input_obj = self.args_schema(**kwargs)
        return await self._execute(input_obj)

    async def _execute(self, input_obj: BaseModel) -> dict:
        """Concrete tool logic; overridden by subclasses."""
        raise NotImplementedError

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Synchronous entry-point is not supported; use arun instead."""
        raise NotImplementedError("Use arun() for agent tools.")
