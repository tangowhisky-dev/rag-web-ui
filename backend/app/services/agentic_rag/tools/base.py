"""Base class for agent tools that bind to ToolContext."""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.services.agentic_rag.tool_context import ToolContext


class BaseAgentTool(BaseTool):
    """BaseTool subclass that receives a ToolContext and dispatches to _execute."""

    ctx: Optional[ToolContext] = Field(default=None, exclude=True)

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """Parse validated input and call the concrete implementation."""
        kwargs.pop("run_manager", None)
        if args and isinstance(args[0], dict):
            kwargs = args[0]
        input_obj = self.args_schema(**kwargs)
        return await self._execute(input_obj)

    async def _execute(self, input_obj: BaseModel) -> dict:
        """Concrete tool logic; overridden by subclasses."""
        raise NotImplementedError

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Synchronous entry-point is not supported; use arun instead."""
        raise NotImplementedError("Use arun() for agent tools.")
