"""Tests for agent loop guardrails: iteration caps and token budgets."""

from unittest.mock import patch

from app.services.agentic_rag.agent_graph import route_think
from app.services.agentic_rag.token_budget import count_tokens


def test_route_think_routes_to_tool_when_calls_present():
    state = {"iteration": 1, "tool_calls": [{"tool": "search_dense"}]}
    with patch("app.services.agentic_rag.agent_graph.thinking.get_setting", return_value=8):
        assert route_think(state) == "tool"


def test_route_think_routes_to_finalize_at_max_iterations():
    state = {"iteration": 3, "tool_calls": []}
    with patch("app.services.agentic_rag.agent_graph.thinking.get_setting", return_value=3):
        # In the new topology, route_think returns "finalize" at max iterations
        assert route_think(state) == "finalize"


def test_route_think_routes_to_sufficiency_check_when_no_calls():
    state = {"iteration": 2, "tool_calls": []}
    with patch("app.services.agentic_rag.agent_graph.thinking.get_setting", return_value=5):
        # In the new topology, no tool calls → check sufficiency (not reflect_final)
        assert route_think(state) == "sufficiency_check"


def test_count_tokens_handles_strings_and_lists():
    text = "This is a test sentence."
    assert count_tokens(text) > 0
    assert count_tokens([text, text]) > count_tokens(text)


def test_count_tokens_returns_positive_for_dict():
    payload = {"summary": "hello world", "key_points": ["a", "b"]}
    assert count_tokens(payload) > 0
