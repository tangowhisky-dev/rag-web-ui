"""Tests for the enterprise agent loop tools and utilities."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.services.agentic_rag.token_budget import count_tokens
from app.services.agentic_rag.tool_context import ToolContext
from app.services.agentic_rag.tools import applicable_tools, build_tools
from app.services.agentic_rag.tools.chart_generate import ChartGenerateTool
from app.services.agentic_rag.tools.code_execute import CodeExecuteTool
from app.services.agentic_rag.tools.extract_data import ExtractDataTool


def _make_ctx(has_file: bool = True, has_data: bool = False) -> ToolContext:
    state = SimpleNamespace(
        retrieved_docs=[{"page_content": "revenue was 100"}] if has_data else [],
        last_answer_object=SimpleNamespace(data=[{"label": "x", "value": 1}]) if has_data else None,
    )
    return ToolContext(
        db=MagicMock(),
        user_id=1,
        org_id=1,
        chat_id=1 if has_file else None,
        redis_memory=None,
        org_llm_config={},
        state=state,
    )


class TestToolRegistry:
    def test_build_tools_returns_all_tools(self):
        ctx = _make_ctx()
        tools = build_tools(ctx)
        names = {t.name for t in tools}
        expected = {
            "rag_retrieve",
            "file_read",
            "file_summarize",
            "file_extract_table",
            "code_execute",
            "chart_generate",
            "summarize_answer",
            "extract_data",
            "clarify",
        }
        assert names == expected

    def test_applicable_tools_excludes_file_tools_without_file(self):
        ctx = _make_ctx(has_file=False)
        tools = applicable_tools(ctx)
        names = {t.name for t in tools}
        assert "file_read" not in names
        assert "file_summarize" not in names
        assert "file_extract_table" not in names
        assert "rag_retrieve" in names

    def test_applicable_tools_includes_chart_when_data_present(self):
        ctx = _make_ctx(has_file=True, has_data=True)
        tools = applicable_tools(ctx)
        names = {t.name for t in tools}
        assert "chart_generate" in names
        assert "extract_data" in names


class TestChartGenerateTool:
    def _tool(self):
        tool = ChartGenerateTool()
        tool.ctx = _make_ctx()
        return tool

    def test_bar_chart(self):
        tool = self._tool()
        result = asyncio.run(
            tool.arun(
                {
                    "chart_type": "bar",
                    "data": [
                        {"label": "A", "value": 10},
                        {"label": "B", "value": 20},
                    ],
                    "title": "Demo",
                }
            )
        )
        assert result["ok"] is True
        assert "bar" in result["result"]["chart_option"]["series"][0]["type"]
        assert result["result"]["valid"] is True

    def test_pie_chart(self):
        tool = self._tool()
        result = asyncio.run(
            tool.arun(
                {
                    "chart_type": "pie",
                    "data": [{"label": "A", "value": 5}],
                    "title": "Demo",
                }
            )
        )
        assert result["ok"] is True
        assert result["result"]["chart_type"] == "pie"

    def test_rejects_empty_data(self):
        tool = self._tool()
        result = asyncio.run(tool.arun({"chart_type": "bar", "data": []}))
        assert result["ok"] is False


class TestCodeExecuteTool:
    def test_simple_math(self):
        tool = CodeExecuteTool()
        tool.ctx = _make_ctx()
        result = asyncio.run(tool.arun({"code": "result = 2 + 3"}))
        assert result["ok"] is True
        assert result["result"]["result"] == 5

    def test_disallowed_import(self):
        tool = CodeExecuteTool()
        tool.ctx = _make_ctx()
        result = asyncio.run(tool.arun({"code": "import os"}))
        assert result["ok"] is False
        assert "Disallowed" in result["error"]

    def test_timeout(self):
        tool = CodeExecuteTool()
        tool.ctx = _make_ctx()
        result = asyncio.run(tool.arun({"code": "while True: pass", "timeout_s": 1}))
        assert result["ok"] is False


class TestExtractDataTool:
    def test_rule_based_extraction(self):
        tool = ExtractDataTool()
        tool.ctx = _make_ctx(has_data=True)
        result = asyncio.run(
            tool.arun(
                {
                    "source": "retrieved_docs",
                    "focus": "revenue",
                }
            )
        )
        assert result["ok"] is True
        assert result["result"]["count"] >= 1
        assert any("100" in str(d["value"]) for d in result["result"]["data"])


class TestTokenBudget:
    def test_count_tokens_positive(self):
        # Tiktoken (or heuristic fallback) should return positive counts for non-empty text.
        assert count_tokens("") == 0
        assert count_tokens("hello world") > 0
