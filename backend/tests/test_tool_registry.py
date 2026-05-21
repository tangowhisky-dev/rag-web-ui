"""Tests for the tool registry."""

import pytest
from app.services.tool_registry import (
    ToolRegistry,
    ToolResult,
    execute_tool,
    register_tool,
    _registry,
)


# ── Registry unit tests ───────────────────────────────────────────────────────

class TestToolRegistry:
    def setup_method(self):
        """Use a fresh registry for each test."""
        self.reg = ToolRegistry()

    def test_register_and_get(self):
        self.reg.register("my_tool", "Does something", {"type": "object"}, lambda: None)
        tool = self.reg.get("my_tool")
        assert tool is not None
        assert tool.name == "my_tool"
        assert tool.description == "Does something"

    def test_get_unknown_returns_none(self):
        assert self.reg.get("nonexistent") is None

    def test_list_tools_openai_format(self):
        self.reg.register(
            "test_tool",
            "A test tool",
            {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
            lambda x: x,
        )
        tools = self.reg.list_tools()
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "test_tool"
        assert tools[0]["function"]["description"] == "A test tool"
        assert "parameters" in tools[0]["function"]

    def test_register_overwrites_existing(self):
        self.reg.register("dup", "v1", {}, lambda: "v1")
        self.reg.register("dup", "v2", {}, lambda: "v2")
        assert self.reg.get("dup").description == "v2"

    def test_list_tools_empty_registry(self):
        assert self.reg.list_tools() == []


class TestExecuteTool:
    def test_executes_registered_tool(self):
        _registry.register("add", "Add two numbers", {}, lambda a, b: a + b)
        result = execute_tool("add", {"a": 2, "b": 3})
        assert result.success
        assert result.output == 5
        assert result.error is None
        assert result.latency_ms >= 0

    def test_unknown_tool_returns_error(self):
        result = execute_tool("totally_unknown_tool_xyz", {})
        assert not result.success
        assert result.error is not None
        assert "Unknown tool" in result.error

    def test_handler_exception_captured(self):
        _registry.register("bad_tool", "Fails always", {}, lambda: 1 / 0)
        result = execute_tool("bad_tool", {})
        assert not result.success
        assert "division by zero" in result.error

    def test_tool_result_to_dict(self):
        result = ToolResult(tool_name="my_tool", output={"key": "val"}, latency_ms=12.3)
        d = result.to_dict()
        assert d["tool_name"] == "my_tool"
        assert d["output"] == {"key": "val"}
        assert d["error"] is None
        assert d["latency_ms"] == 12.3

    def test_register_tool_decorator(self):
        @register_tool("decorated", "Decorated tool", {"type": "object"})
        def my_fn(x: int) -> int:
            return x * 2

        result = execute_tool("decorated", {"x": 5})
        assert result.success
        assert result.output == 10
