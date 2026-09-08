"""Tests for agent loop guardrails: tool-call budget and token budgets."""

from unittest.mock import patch

from app.services.agentic_rag.agent_graph_v2.thinking import route_think_v2
from app.services.agentic_rag.token_budget import count_tokens


def test_route_think_routes_to_tool_when_calls_present():
    state = {"iteration": 1, "tool_calls": [{"tool": "semantic_search"}]}
    with patch("app.services.agentic_rag.agent_graph_v2.thinking.get_setting", return_value=25):
        assert route_think_v2(state) == "tool"


def test_route_think_routes_to_post_process_at_max_iterations():
    state = {"iteration": 3, "tool_calls": []}
    with patch("app.services.agentic_rag.agent_graph_v2.thinking.get_setting", return_value=25):
        # In v2 topology, no tool calls at max iterations → post_process
        assert route_think_v2(state) == "post_process"


def test_route_think_routes_to_post_process_when_no_calls():
    state = {"iteration": 2, "tool_calls": []}
    with patch("app.services.agentic_rag.agent_graph_v2.thinking.get_setting", return_value=25):
        # In v2 topology, no tool calls → the LLM wrote the answer → post_process
        assert route_think_v2(state) == "post_process"


def test_count_tokens_handles_strings_and_lists():
    text = "This is a test sentence."
    assert count_tokens(text) > 0
    assert count_tokens([text, text]) > count_tokens(text)


def test_count_tokens_returns_positive_for_dict():
    payload = {"summary": "hello world", "key_points": ["a", "b"]}
    assert count_tokens(payload) > 0
