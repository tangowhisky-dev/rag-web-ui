"""Tests for agent loop guardrails: iteration caps and token budgets."""

from unittest.mock import patch

from app.core.config import settings
from app.services.agentic_rag.agent_graph import route_think
from app.services.agentic_rag.token_budget import count_tokens


def test_route_think_routes_to_tool_when_calls_present():
    state = {"iteration": 1, "tool_calls": [{"tool": "rag_retrieve"}]}
    assert route_think(state) == "tool"


def test_route_think_routes_to_reflect_final_at_max_iterations():
    with patch.object(settings, "AGENT_MAX_ITERATIONS", 3):
        state = {"iteration": 3, "tool_calls": []}
        assert route_think(state) == "reflect_final"


def test_route_think_routes_to_reflect_final_when_no_calls():
    with patch.object(settings, "AGENT_MAX_ITERATIONS", 5):
        state = {"iteration": 2, "tool_calls": []}
        assert route_think(state) == "reflect_final"


def test_count_tokens_handles_strings_and_lists():
    text = "This is a test sentence."
    assert count_tokens(text) > 0
    assert count_tokens([text, text]) > count_tokens(text)


def test_count_tokens_returns_positive_for_dict():
    payload = {"summary": "hello world", "key_points": ["a", "b"]}
    assert count_tokens(payload) > 0
