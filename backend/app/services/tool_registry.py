"""
Tool registry for agentic tool calling.

Provides a global registry of callable tools that the LLM can invoke.
Tools are registered with their OpenAI-compatible JSON Schema definitions
and executed via execute_tool().

Built-in tools are registered in builtin_tools.py which imports this module.
chat_service.py passes list_tools() to the LLM and calls execute_tool() when
the LLM returns tool_calls.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class ToolDefinition:
    """A registered tool callable by the LLM."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema object for the parameters
    handler: Callable[..., Any]


@dataclass
class ToolResult:
    """Result of a tool execution."""
    tool_name: str
    output: Any          # JSON-serializable output (dict, list, str)
    error: Optional[str] = None
    latency_ms: float = 0.0

    @property
    def success(self) -> bool:
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "output": self.output,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 1),
        }


# ── Registry ──────────────────────────────────────────────────────────────────

class ToolRegistry:
    """Registry of callable tools with OpenAI-compatible schema export."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        """Register a tool. Overwrites existing registration with same name."""
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
        )
        logger.debug("[TOOL] registered tool=%s", name)

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Return tool definition by name, or None if not registered."""
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return all registered tools in OpenAI tools= format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]


# ── Global singleton ──────────────────────────────────────────────────────────

_registry = ToolRegistry()


def register_tool(
    name: str,
    description: str,
    parameters: Dict[str, Any],
) -> Callable:
    """
    Decorator to register a function as a tool in the global registry.

    Usage:
        @register_tool("my_tool", "Does something", {"type": "object", ...})
        def my_tool(arg1: str) -> dict:
            ...
    """
    def decorator(fn: Callable) -> Callable:
        _registry.register(name, description, parameters, fn)
        return fn
    return decorator


def execute_tool(name: str, args: Dict[str, Any]) -> ToolResult:
    """
    Execute a registered tool by name with the given arguments.

    Always returns a ToolResult — never raises. Errors are captured in
    ToolResult.error so the calling loop can feed the error back to the LLM.

    Args:
        name: Tool name as returned by list_tools().
        args: Arguments dict parsed from the LLM's tool_call.function.arguments.

    Returns:
        ToolResult with output or error and latency_ms.
    """
    tool = _registry.get(name)
    if tool is None:
        logger.warning("[TOOL] unknown tool=%s", name)
        return ToolResult(
            tool_name=name,
            output=None,
            error=f"Unknown tool: {name!r}. Available: {list(_registry._tools.keys())}",
        )

    t0 = time.perf_counter()
    try:
        output = tool.handler(**args)
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "[TOOL] tool=%s latency_ms=%.1f success=true",
            name, latency_ms,
        )
        return ToolResult(tool_name=name, output=output, latency_ms=latency_ms)
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.warning(
            "[TOOL] tool=%s latency_ms=%.1f success=false error=%s",
            name, latency_ms, exc,
        )
        return ToolResult(tool_name=name, output=None, error=str(exc), latency_ms=latency_ms)
