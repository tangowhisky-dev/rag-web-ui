"""Regression tests for AgenticRAGTransformer token handling."""

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from app.services.agentic_rag.streaming import AgenticRAGTransformer


def _token_events(transformer: AgenticRAGTransformer) -> list[dict]:
    """Return token events pushed to the events channel."""
    return [item for _, item in transformer.events._items if item.get("event") == "token"]


@pytest.fixture
def transformer():
    t = AgenticRAGTransformer()
    t.events._subscribed = True
    return t


def _msg_event(payload, node: str):
    """Build a v3 messages protocol event."""
    return {
        "method": "messages",
        "params": {
            "namespace": [],
            "data": (payload, {"langgraph_node": node}),
        },
    }


def test_classifier_message_not_emitted_as_token(transformer):
    """Structured classifier output must not leak into the answer token stream."""
    classifier_json = '{"is_clear": true, "questions": ["q1"], "clarification_needed": ""}'
    transformer.process(_msg_event(AIMessage(content=classifier_json), "classify_query"))
    assert _token_events(transformer) == []


def _custom_event(payload: dict):
    """Build a v3 custom protocol event."""
    return {
        "method": "custom",
        "params": {
            "namespace": [],
            "data": payload,
        },
    }


def test_generating_message_does_not_emit_token(transformer):
    """Generating-node message events now only carry usage, not answer tokens."""
    for chunk in ["Hello ", "world"]:
        transformer.process(_msg_event(AIMessageChunk(content=chunk), "generating"))
    assert _token_events(transformer) == []


def test_custom_token_event_emits_token(transformer):
    """Explicit token events from generating_node are forwarded unchanged."""
    for chunk in ["Hello ", "world"]:
        transformer.process(_custom_event({"event": "token", "content": chunk}))
    tokens = _token_events(transformer)
    assert len(tokens) == 2
    assert [t["content"] for t in tokens] == ["Hello ", "world"]


def test_finalize_answer_message_not_emitted_as_token(transformer):
    """The final AIMessage from finalize_answer is not re-streamed as tokens."""
    transformer.process(_msg_event(AIMessage(content="Final answer."), "finalize_answer"))
    assert _token_events(transformer) == []
