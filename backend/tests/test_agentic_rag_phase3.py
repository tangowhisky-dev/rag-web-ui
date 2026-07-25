"""Focused tests for Phase 3 hardening of the agentic RAG pipeline."""

import pytest

from app.services.agentic_rag.nodes import (
    validate_echarts_json,
    chart_validation_node,
    adaptive_reranking_node,
)
from app.services.agentic_rag.graph import (
    route_after_chart_validation,
    route_after_answer_evaluation,
)
from app.services.agentic_rag.graph_state import AgentState


# ── ECharts JSON validation ────────────────────────────────────────────────

def test_validate_echarts_json_valid_line_chart():
    answer = """```json
{
  "xAxis": {"type": "category", "data": ["A", "B"]},
  "yAxis": {"type": "value"},
  "series": [{"type": "line", "data": [1, 2]}]
}
```"""
    valid, message = validate_echarts_json(answer)
    assert valid is True
    assert "valid" in message.lower()


def test_validate_echarts_json_missing_series():
    answer = """```json
{"xAxis": {}, "yAxis": {}}
```"""
    valid, message = validate_echarts_json(answer)
    assert valid is False
    assert "series" in message.lower()


def test_validate_echarts_json_cartesian_missing_axes():
    answer = """```json
{"series": [{"type": "bar", "data": [1, 2]}]}
```"""
    valid, message = validate_echarts_json(answer)
    assert valid is False
    assert "xaxis" in message.lower() and "yaxis" in message.lower()


def test_validate_echarts_json_invalid_json():
    answer = "```json\n{not valid json}\n```"
    valid, message = validate_echarts_json(answer)
    assert valid is False
    assert "json" in message.lower()


# ── Chart validation routing ───────────────────────────────────────────────

def test_chart_validation_node_no_op_for_non_chart():
    state = AgentState(is_chart_query=False)
    result = chart_validation_node(state)
    assert result["chart_validated"] is False
    assert result["chart_data"] is None


def test_chart_validation_node_retries_invalid_chart():
    state = AgentState(
        is_chart_query=True,
        answer="This is not JSON",
        chart_retries=0,
    )
    result = chart_validation_node(state)
    assert result["chart_validated"] is False
    assert result["chart_retries"] == 1


def test_chart_validation_node_gives_up_after_three_retries():
    state = AgentState(
        is_chart_query=True,
        answer="Still not JSON",
        chart_retries=3,
    )
    result = chart_validation_node(state)
    # After 3 failed attempts we mark as validated to avoid infinite loops.
    assert result["chart_validated"] is True
    assert result["chart_data"]["valid"] is False


def test_route_after_chart_validation_retries_then_proceeds():
    invalid_state = AgentState(
        is_chart_query=True,
        chart_data={"valid": False},
        chart_retries=1,
    )
    assert route_after_chart_validation(invalid_state) == "generating"

    exhausted_state = AgentState(
        is_chart_query=True,
        chart_data={"valid": False},
        chart_retries=3,
    )
    assert route_after_chart_validation(exhausted_state) == "answer_evaluation"

    valid_state = AgentState(
        is_chart_query=True,
        chart_data={"valid": True},
        chart_retries=0,
    )
    assert route_after_chart_validation(valid_state) == "answer_evaluation"


# ── Answer evaluation retry cap ────────────────────────────────────────────

def test_route_after_answer_evaluation_caps_retries():
    retry_state = AgentState(needs_retry=True, answer_evaluation_attempts=1)
    assert route_after_answer_evaluation(retry_state) == "generating"

    exhausted_state = AgentState(needs_retry=True, answer_evaluation_attempts=2)
    assert route_after_answer_evaluation(exhausted_state) == "finalize_answer"

    done_state = AgentState(needs_retry=False, answer_evaluation_attempts=1)
    assert route_after_answer_evaluation(done_state) == "finalize_answer"


# ── Adaptive reranking once per subtask ────────────────────────────────────

def test_adaptive_reranking_skips_if_already_reran():
    state = AgentState(
        retrieval_confidence=0.1,
        retrieved_docs=[{"page_content": "doc"}],
        adaptive_reran=True,
    )
    result = adaptive_reranking_node(state, db=None)
    assert result["adaptive_rerunning"] is False


def test_adaptive_reranking_no_op_when_confidence_high():
    state = AgentState(
        retrieval_confidence=0.5,
        retrieved_docs=[{"page_content": "doc"}],
        adaptive_reran=False,
    )
    result = adaptive_reranking_node(state, db=None)
    assert result["adaptive_rerunning"] is False
    assert result["adaptive_reran"] is True
