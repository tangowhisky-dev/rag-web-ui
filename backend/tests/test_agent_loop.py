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
            "kb_search_documents",
            "current_datetime",
            "file_read",
            "file_summarize",
            "file_extract_table",
            "code_execute",
            "chart_generate",
            "summarize_answer",
            "extract_data",
            "kb_grep",
            "kb_read",
            "kb_outline",
            "kb_metadata",
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

    def test_applicable_tools_includes_chart_after_code_execute(self):
        """A successful code_execute observation should make chart_generate
        available even without retrieved_docs or last_answer_object.data."""
        ctx = _make_ctx(has_file=True, has_data=False)
        ctx.state["observations"] = [
            Observation(tool="code_execute", arguments={"code": "print([1,2,3])"}, result={"stdout": "[1, 2, 3]"}),
        ]
        tools = applicable_tools(ctx)
        names = {t.name for t in tools}
        assert "chart_generate" in names
        assert "extract_data" in names

    def test_applicable_tools_excludes_chart_after_failed_code_execute(self):
        """A failed code_execute observation should NOT make chart_generate available."""
        ctx = _make_ctx(has_file=True, has_data=False)
        ctx.state["observations"] = [
            Observation(tool="code_execute", arguments={"code": "error"}, result={}, error="SyntaxError"),
        ]
        tools = applicable_tools(ctx)
        names = {t.name for t in tools}
        assert "chart_generate" not in names
        assert "extract_data" not in names


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


class TestCodeExecuteGuards:
    """Tests for the RestrictedPython guard functions added to fix list
    comprehension, tuple unpacking, and augmented assignment failures."""

    def test_list_comprehension_works(self):
        tool = CodeExecuteTool()
        tool.ctx = _make_ctx()
        result = asyncio.run(tool.arun({
            "code": "fibs = [0, 1]\n[fibs.append(fibs[-1] + fibs[-2]) for _ in range(10)]\nresult = fibs",
        }))
        assert result["ok"] is True
        assert result["result"]["result"] == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

    def test_for_loop_with_unpacking_works(self):
        tool = CodeExecuteTool()
        tool.ctx = _make_ctx()
        result = asyncio.run(tool.arun({
            "code": "pairs = [(1, 2), (3, 4)]\ntotal = 0\nfor a, b in pairs:\n    total += a + b\nresult = total",
        }))
        assert result["ok"] is True
        assert result["result"]["result"] == 10

    def test_augmented_assignment_works(self):
        tool = CodeExecuteTool()
        tool.ctx = _make_ctx()
        result = asyncio.run(tool.arun({"code": "x = 5\nx += 3\nx *= 2\nresult = x"}))
        assert result["ok"] is True
        assert result["result"]["result"] == 16


class TestExtractDataChartFallback:
    """Tests for extract_data reading chart_options when lao.data is empty."""

    def test_extract_from_chart_options(self):
        from app.services.agentic_rag.tools.extract_data import _extract_from_chart_options

        chart_opts = [{
            "xAxis": {"data": ["A", "B", "C"]},
            "series": [{"type": "bar", "data": [10, 20, 30]}],
        }]
        points = _extract_from_chart_options(chart_opts)
        assert len(points) == 3
        assert points[0] == {"label": "A", "value": 10.0, "unit": None, "context": "Chart data point 1"}
        assert points[2] == {"label": "C", "value": 30.0, "unit": None, "context": "Chart data point 3"}

    def test_extract_from_empty_chart_options(self):
        from app.services.agentic_rag.tools.extract_data import _extract_from_chart_options

        assert _extract_from_chart_options([]) == []
        assert _extract_from_chart_options([{"series": []}]) == []

    def test_extract_data_uses_chart_options_when_no_data(self):
        from app.services.agentic_rag.schemas import LastAnswerObject

        tool = ExtractDataTool()
        ctx = _make_ctx(has_file=False, has_data=False)
        ctx.state["last_answer_object"] = LastAnswerObject(
            summary="Here is a chart.",
            key_points=[],
            data=None,
            chart_options=[{
                "xAxis": {"data": ["Q1", "Q2"]},
                "series": [{"type": "bar", "data": [100, 200]}],
            }],
        )
        tool.ctx = ctx
        result = asyncio.run(tool.arun({"source": "last_answer"}))
        assert result["ok"] is True
        assert result["result"]["count"] == 2
        assert result["result"]["data"][0]["label"] == "Q1"
        assert result["result"]["data"][0]["value"] == 100.0


class TestRetryHelpers:
    """Tests for the tool_node retry infrastructure."""

    def test_is_transient_error_detects_network_errors(self):
        from app.services.agentic_rag.agent_graph import _is_transient_error

        assert _is_transient_error("Connection timed out")
        assert _is_transient_error("network unreachable")
        assert _is_transient_error("Connection reset by peer")
        assert _is_transient_error("I/O error on network share")

    def test_is_transient_error_rejects_argument_errors(self):
        from app.services.agentic_rag.agent_graph import _is_transient_error

        assert not _is_transient_error("name '_iter_unpack_sequence_' is not defined")
        assert not _is_transient_error("No numeric values found")
        assert not _is_transient_error("Tool not available")

    def test_correction_hints_for_code_execute(self):
        from app.services.agentic_rag.agent_graph import _correction_hints

        hints = _correction_hints("code_execute", "_iter_unpack_sequence_ is not defined")
        assert "for-loops" in hints

    def test_correction_hints_for_chart_generate(self):
        from app.services.agentic_rag.agent_graph import _correction_hints

        hints = _correction_hints("chart_generate", "No numeric values found")
        assert "value" in hints

    def test_correction_hints_default(self):
        from app.services.agentic_rag.agent_graph import _correction_hints

        hints = _correction_hints("unknown_tool", "some error")
        assert "Fix the arguments" in hints


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
