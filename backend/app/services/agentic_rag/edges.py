"""Conditional routing logic for the LangGraph agent graph."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import START, END
from langgraph.types import Command

from .graph_state import AgentState
from .nodes import (
    collect_answer_node,
    fallback_response_node,
    generate_node,
    orchestrator_node,
    request_clarification_node,
    should_compress_context,
)

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI


def route_after_classification(state: AgentState) -> Command:
    """Route from classify node to clarification, direct retrieval, or agent subgraph."""
    if not state.get("question_is_clear", True):
        return Command(goto="clarification")
    elif state.get("is_complex", False):
        return Command(goto="agent")  # enter nested subgraph
    else:
        return Command(goto="direct_retrieval")


def route_after_clarification(state: AgentState) -> Command:
    """After clarification, go back to classification."""
    return Command(goto="classify")


def route_after_orchestrator(state: AgentState) -> Command:
    """Route from orchestrator based on budget or tool call needs."""
    if state.get("_orchestrator_result") == "fallback":
        return Command(goto="fallback")
    elif state.get("_orchestrator_result") == "collect":
        return Command(goto="collect")
    else:
        # Default: generate directly without tools
        return Command(goto="collect")


def route_after_should_compress(state: AgentState) -> Command:
    """Route based on whether token budget is exceeded."""
    result = should_compress_context(state)
    if result == "compress":
        return Command(goto="compress")
    return Command(goto="orchestrator")


def route_after_fallback(state: AgentState) -> Command:
    """After fallback, collect the answer."""
    return Command(goto="collect")


def route_after_collect(state: AgentState) -> Command:
    """After collecting in agent subgraph, go to END of subgraph."""
    return Command(goto=END)
