"""Tests for the enterprise agent loop tools and utilities."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.services.agentic_rag.agent_graph import _observations_text, _tried_rag_retrieve_queries
from app.services.agentic_rag.schemas import Observation
from app.services.agentic_rag.token_budget import count_tokens
from app.services.agentic_rag.tool_context import ToolContext
from app.services.agentic_rag.tools import applicable_tools, build_tools
from app.services.agentic_rag.tools.chart_generate import ChartGenerateTool
from app.services.agentic_rag.tools.code_execute import CodeExecuteTool
from app.services.agentic_rag.tools.extract_data import ExtractDataTool


def _make_ctx(has_file: bool = True, has_data: bool = False) -> ToolContext:
    # AgentState is a plain dict (TypedDict/MessagesState) at runtime.
    state = {
        "retrieved_docs": [{"page_content": "revenue was 100"}] if has_data else [],
        "last_answer_object": SimpleNamespace(data=[{"label": "x", "value": 1}]) if has_data else None,
        "file_markdown": "# file content" if has_file else None,
    }
    return ToolContext(
        db=MagicMock(),
        user_id=1,
        org_id=1,
        chat_id=1 if has_file else None,
        message_id=1,
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

    def test_radar_chart(self):
        tool = self._tool()
        result = asyncio.run(
            tool.arun(
                {
                    "chart_type": "radar",
                    "data": [{"label": "Speed", "value": 80}, {"label": "Power", "value": 60}],
                    "title": "Demo",
                }
            )
        )
        assert result["ok"] is True
        assert result["result"]["chart_option"]["series"][0]["type"] == "radar"
        assert len(result["result"]["chart_option"]["radar"]["indicator"]) == 2

    def test_gauge_chart(self):
        tool = self._tool()
        result = asyncio.run(
            tool.arun({"chart_type": "gauge", "data": [{"label": "Completion", "value": 72}], "title": "Demo"})
        )
        assert result["ok"] is True
        assert result["result"]["chart_option"]["series"][0]["type"] == "gauge"

    def test_funnel_chart(self):
        tool = self._tool()
        result = asyncio.run(
            tool.arun(
                {
                    "chart_type": "funnel",
                    "data": [{"label": "Visit", "value": 100}, {"label": "Signup", "value": 40}],
                    "title": "Demo",
                }
            )
        )
        assert result["ok"] is True
        assert result["result"]["chart_option"]["series"][0]["type"] == "funnel"

    def test_effect_scatter_chart(self):
        tool = self._tool()
        result = asyncio.run(
            tool.arun(
                {
                    "chart_type": "effectScatter",
                    "data": [{"label": "A", "value": 10}, {"label": "B", "value": 20}],
                    "title": "Demo",
                }
            )
        )
        assert result["ok"] is True
        assert result["result"]["chart_option"]["series"][0]["type"] == "effectScatter"

    def test_unsupported_type_still_errors(self):
        tool = self._tool()
        result = asyncio.run(tool.arun({"chart_type": "sankey", "data": [{"label": "A", "value": 1}]}))
        assert result["ok"] is False
        assert "Unsupported chart type" in result["error"]


class TestSubstituteChartMarkers:
    def test_replaces_marker_inline(self):
        from app.services.agentic_rag.agent_graph import _substitute_chart_markers

        text = "Here is the chart: [[CHART_1]]\n\nDone."
        result = _substitute_chart_markers(text, [{"series": [{"type": "bar"}]}])
        assert "[[CHART_1]]" not in result
        assert "```echarts" in result
        assert result.index("```echarts") < result.index("Done.")

    def test_appends_when_marker_missing(self):
        from app.services.agentic_rag.agent_graph import _substitute_chart_markers

        text = "No marker here."
        result = _substitute_chart_markers(text, [{"series": [{"type": "bar"}]}])
        assert result.startswith("No marker here.")
        assert "```echarts" in result

    def test_multiple_charts_each_substituted(self):
        from app.services.agentic_rag.agent_graph import _substitute_chart_markers

        text = "First [[CHART_1]] then [[CHART_2]]."
        result = _substitute_chart_markers(
            text,
            [{"series": [{"type": "bar"}]}, {"series": [{"type": "pie"}]}],
        )
        assert "[[CHART_1]]" not in result
        assert "[[CHART_2]]" not in result
        assert result.count("```echarts") == 2
        assert result.index('"bar"') < result.index('"pie"')

    def test_no_charts_returns_text_unchanged(self):
        from app.services.agentic_rag.agent_graph import _substitute_chart_markers

        text = "No charts this turn."
        assert _substitute_chart_markers(text, []) == text


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
        assert "not allowed" in result["error"]

    def test_timeout(self):
        tool = CodeExecuteTool()
        tool.ctx = _make_ctx()
        result = asyncio.run(tool.arun({"code": "while True: pass", "timeout_s": 1}))
        assert result["ok"] is False

    def test_print_is_captured_in_stdout(self):
        # RestrictedPython routes print() through a PrintCollector, not real
        # stdout — must not crash with 'NoneType has no attribute _call_print'.
        tool = CodeExecuteTool()
        tool.ctx = _make_ctx()
        result = asyncio.run(tool.arun({"code": "print('hello')\nresult = 1"}))
        assert result["ok"] is True
        assert "hello" in result["result"]["stdout"]

    def test_compile_failure_disarms_alarm(self):
        # A syntax error must not leave signal.alarm() armed — otherwise it
        # fires later inside the event loop and crashes an unrelated call.
        import signal

        tool = CodeExecuteTool()
        tool.ctx = _make_ctx()
        result = asyncio.run(tool.arun({"code": "1. \"bad\",", "timeout_s": 5}))
        assert result["ok"] is False
        assert "compilation failed" in result["error"].lower()
        assert signal.alarm(0) == 0  # returns previous remaining time; must be disarmed

        # A subsequent normal call must succeed without an errant TimeoutError.
        result2 = asyncio.run(tool.arun({"code": "result = 1 + 1"}))
        assert result2["ok"] is True
        assert result2["result"]["result"] == 2


class TestObservationsText:
    def test_non_retrieval_tool_result_is_rendered(self):
        # Regression: code_execute (and other non-rag_retrieve tools) have no
        # "docs" key, so the old formatter showed doc_count=0 and hid the
        # actual result — causing the LLM to re-issue the same call repeatedly.
        obs = Observation(tool="code_execute", arguments={"code": "print(391)"}, result={"stdout": "391\n", "result": ""})
        text = _observations_text([obs], full=True)
        assert "391" in text
        assert "doc_count=0" not in text


class TestTriedRagRetrieveQueries:
    def test_dedups_and_preserves_order(self):
        # Regression: the LLM would sometimes resubmit an identical
        # rag_retrieve query the ladder already exhausted, wasting an
        # iteration. This list is surfaced in the think prompt to discourage it.
        observations = [
            Observation(tool="rag_retrieve", arguments={"query": "race condition"}, result={"docs": [], "sufficient": False}),
            Observation(tool="code_execute", arguments={"code": "1+1"}, result={"result": 2}),
            Observation(tool="rag_retrieve", arguments={"query": "mutual exclusion condition"}, result={"docs": [], "sufficient": False}),
            Observation(tool="rag_retrieve", arguments={"query": "race condition"}, result={"docs": [], "sufficient": False}),
        ]
        assert _tried_rag_retrieve_queries(observations) == ["race condition", "mutual exclusion condition"]

    def test_empty_when_no_rag_retrieve_calls(self):
        observations = [Observation(tool="code_execute", arguments={"code": "1+1"}, result={"result": 2})]
        assert _tried_rag_retrieve_queries(observations) == []


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


class TestConvergence:
    """Regression coverage for the wasted-iteration bug: once the plan is
    deterministically satisfied, the loop must stop without extra LLM calls."""

    def _satisfied_state(self) -> dict:
        from app.services.agentic_rag.schemas import Plan, Subtask

        plan = Plan(
            intent="rag",
            subtasks=[Subtask(id="a", description="find x", tool_hint="rag_retrieve", depends_on=[], expected_output="answer")],
        )
        obs = Observation(
            tool="rag_retrieve",
            arguments={"query": "what is mutex"},
            result={"docs": [{"page_content": "a mutex is..."}]},
            error=None,
            tokens=10,
        )
        return {
            "plan": plan,
            "observations": [obs],
            "tool_call_count": {"rag_retrieve": 1},
            "iteration": 0,
            "original_query": "what is mutex",
            "messages": [],
        }

    def test_verify_execution_ready_when_plan_satisfied(self):
        from app.services.agentic_rag.agent_graph import _build_execution_summary, _verify_execution

        summary = _build_execution_summary(self._satisfied_state())
        ready, _reasoning = _verify_execution(summary)
        assert ready is True

    def test_route_tool_skips_reflect_when_force_finalize(self):
        from app.services.agentic_rag.agent_graph import route_tool

        assert route_tool({"force_finalize": True}) == "reflect_final"
        assert route_tool({"force_finalize": False}) == "reflect"

    def test_think_node_short_circuits_without_llm_call(self):
        # If this ever calls the LLM again despite an already-satisfied plan,
        # build_chat_llm would be invoked and fail against the mocked ctx.db —
        # the absence of that failure is the regression signal.
        from app.services.agentic_rag.agent_graph import think_node

        ctx = _make_ctx(has_file=False)
        result = asyncio.run(think_node(self._satisfied_state(), ctx))
        assert result["tool_calls"] == []
        assert result["precomputed_answer"] == ""


class TestLoadContextNodeResetsPerTurnState:
    """The checkpointer restores turn 1's full state at the start of turn 2.
    load_context_node must reset per-turn loop state (observations, iteration,
    tool_call_count, force_finalize, ...) so turn 2 starts clean, while leaving
    conversation-level state (messages, last_answer_object) untouched."""

    def _turn1_leftover_state(self) -> dict:
        return {
            "original_query": "what's next?",
            "observations": [
                Observation(
                    tool="rag_retrieve",
                    arguments={"query": "turn 1 query"},
                    result={"docs": [{"page_content": "turn 1 doc chunk"}]},
                    error=None,
                    tokens=10,
                )
            ],
            "iteration": 3,
            "tool_call_count": {"rag_retrieve": 2},
            "force_finalize": True,
            "reflection_final": {"ready": False, "reasoning": "turn 1 reasoning"},
            "precomputed_answer": "turn 1 answer",
            "tool_calls": [{"tool": "chart_generate", "arguments": {}}],
            "all_scored_docs": [{"page_content": "turn 1 scored doc"}],
            "retrieval_confidence": 0.9,
            "compaction_triggered": True,
            "answer_evaluation_attempts": 2,
            "evaluation_flags": ["low_confidence"],
            "adaptive_reran": True,
        }

    def test_resets_loop_state_at_start_of_next_turn(self):
        from app.services.agentic_rag.agent_graph import load_context_node
        from app.services.agentic_rag.graph_state import accumulate

        ctx = ToolContext(
            db=MagicMock(), user_id=1, org_id=1, chat_id=None, message_id=None,
            redis_memory=None, org_llm_config={},
        )

        update = asyncio.run(load_context_node(self._turn1_leftover_state(), ctx))

        assert update["iteration"] == 0
        assert update["tool_call_count"] == {}
        assert update["force_finalize"] is False
        assert update["reflection_final"] is None
        assert update["precomputed_answer"] == ""
        assert update["tool_calls"] == []
        assert update["all_scored_docs"] == []
        assert update["retrieval_confidence"] == 0.0
        assert update["compaction_triggered"] is False
        assert update["answer_evaluation_attempts"] == 0
        assert update["evaluation_flags"] == []
        assert update["adaptive_reran"] is False

        # observations uses the accumulate reducer; applying the __reset__
        # marker on top of turn 1's list must clear it, not append to it.
        turn1_observations = self._turn1_leftover_state()["observations"]
        merged = accumulate(turn1_observations, update["observations"])
        assert merged == []
