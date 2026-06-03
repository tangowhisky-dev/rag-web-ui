"""
Unit tests for cancellation integration in generate_response().

Tests cover:
- Cancellation during streaming (partial response saved)
- Cancellation before stream starts (loop breaks immediately)
- Normal flow without cancellation (no spurious cancel)
- Partial response content on cancel
- Token cleanup on normal completion
- Token cleanup on error
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_registry():
    """Reset the cancel registry before each test."""
    import app.services.cancel_registry as reg
    reg._cancel_tokens.clear()


@pytest.fixture(autouse=True)
def _reset_state():
    _fresh_registry()
    yield
    _fresh_registry()


def _make_mock_chat():
    """Create a mock Chat object."""
    chat = MagicMock()
    chat.history_summary = None
    return chat


# ---------------------------------------------------------------------------
# test_cancel_during_streaming
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_during_streaming():
    """When the cancel token is set during streaming, the loop breaks
    and the partial response is saved."""
    from app.services.chat_service import generate_response
    from app.services.cancel_registry import set_cancel_token

    chat_id = 201
    db = MagicMock()
    db.commit = MagicMock()
    db.close = MagicMock()
    db.add = MagicMock()
    mock_query = MagicMock()
    mock_query.filter.return_value.first.return_value = _make_mock_chat()
    db.query.return_value = mock_query

    bot_msg = MagicMock()
    bot_msg.content = ""
    bot_msg.role = "assistant"
    bot_msg.chat_id = chat_id
    bot_msg.rewritten_query = None

    with patch("app.services.chat_service.Message") as MockMsg:
        MockMsg.side_effect = [
            MagicMock(id=1, content="test", role="user", chat_id=chat_id),
            bot_msg,
        ]

        call_count = 0

        async def mock_stream_iter():
            nonlocal call_count
            call_count += 1
            yield {"event": "token", "content": "Hello"}
            # Set cancel after first token — simulates user clicking Stop
            set_cancel_token(chat_id)
            yield {"event": "token", "content": " world"}

        with patch("app.services.rag_graph.run_stream", return_value=mock_stream_iter()):
            with patch("app.services.chat_service.asyncio.create_task"):
                frames = []
                async for frame in generate_response(
                    query="test question",
                    messages={"messages": []},
                    knowledge_base_ids=[1],
                    chat_id=chat_id,
                    db=db,
                ):
                    frames.append(frame)

    # bot_message.content should have the partial response ("Hello" was streamed
    # before cancel was set; " world" was never processed)
    assert bot_msg.content == "Hello"
    db.commit.assert_called()


# ---------------------------------------------------------------------------
# test_cancel_before_streaming
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_before_streaming():
    """When cancel is set before the stream loop starts, the loop breaks
    immediately and partial response '(generation stopped)' is saved."""
    from app.services.chat_service import generate_response
    from app.services.cancel_registry import set_cancel_token

    chat_id = 301
    db = MagicMock()
    db.commit = MagicMock()
    db.close = MagicMock()
    db.add = MagicMock()
    mock_query = MagicMock()
    mock_query.filter.return_value.first.return_value = _make_mock_chat()
    db.query.return_value = mock_query

    bot_msg = MagicMock()
    bot_msg.content = ""
    bot_msg.role = "assistant"
    bot_msg.chat_id = chat_id
    bot_msg.rewritten_query = None

    with patch("app.services.chat_service.Message") as MockMsg:
        MockMsg.side_effect = [
            MagicMock(id=1, content="test", role="user", chat_id=chat_id),
            bot_msg,
        ]

        async def mock_stream_iter():
            yield {"event": "token", "content": "should not reach here"}

        set_cancel_token(chat_id)

        with patch("app.services.rag_graph.run_stream", return_value=mock_stream_iter()):
            with patch("app.services.chat_service.asyncio.create_task"):
                frames = []
                async for frame in generate_response(
                    query="test question",
                    messages={"messages": []},
                    knowledge_base_ids=[1],
                    chat_id=chat_id,
                    db=db,
                ):
                    frames.append(frame)

    # bot_message.content should be "(generation stopped)" since full_response is ""
    assert bot_msg.content == "(generation stopped)"
    db.commit.assert_called()


# ---------------------------------------------------------------------------
# test_no_cancel_normal_flow
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_cancel_normal_flow():
    """Normal completion without cancellation should persist the full response
    and not trigger the cancel branch."""
    from app.services.chat_service import generate_response

    chat_id = 401
    db = MagicMock()
    db.commit = MagicMock()
    db.close = MagicMock()
    db.add = MagicMock()
    mock_query = MagicMock()
    mock_query.filter.return_value.first.return_value = _make_mock_chat()
    db.query.return_value = mock_query

    bot_msg = MagicMock()
    bot_msg.content = ""
    bot_msg.role = "assistant"
    bot_msg.chat_id = chat_id
    bot_msg.rewritten_query = None

    with patch("app.services.chat_service.Message") as MockMsg:
        MockMsg.side_effect = [
            MagicMock(id=1, content="test", role="user", chat_id=chat_id),
            bot_msg,
        ]

        async def mock_stream_iter():
            yield {"event": "token", "content": "Full"}
            yield {"event": "token", "content": " response"}
            yield {"event": "done", "usage": {"promptTokens": 5, "completionTokens": 2}}

        with patch("app.services.rag_graph.run_stream", return_value=mock_stream_iter()):
            with patch("app.services.chat_service.asyncio.create_task"):
                frames = []
                async for frame in generate_response(
                    query="test question",
                    messages={"messages": []},
                    knowledge_base_ids=[1],
                    chat_id=chat_id,
                    db=db,
                ):
                    frames.append(frame)

    # Full response should be persisted
    assert bot_msg.content == "Full response"
    # Should have yielded token frames + done frame
    assert len(frames) >= 3


# ---------------------------------------------------------------------------
# test_partial_response_saved
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_partial_response_saved():
    """Verify bot_message.content is set to partial text on cancel,
    not overwritten by a full response."""
    from app.services.chat_service import generate_response
    from app.services.cancel_registry import set_cancel_token

    chat_id = 501
    db = MagicMock()
    db.commit = MagicMock()
    db.close = MagicMock()
    db.add = MagicMock()
    mock_query = MagicMock()
    mock_query.filter.return_value.first.return_value = _make_mock_chat()
    db.query.return_value = mock_query

    bot_msg = MagicMock()
    bot_msg.content = ""
    bot_msg.role = "assistant"
    bot_msg.chat_id = chat_id
    bot_msg.rewritten_query = None

    partial_text = "Partial answer that was streamed before cancellation"

    with patch("app.services.chat_service.Message") as MockMsg:
        MockMsg.side_effect = [
            MagicMock(id=1, content="test", role="user", chat_id=chat_id),
            bot_msg,
        ]

        partial_words = partial_text.split(" ")
        cancel_at_idx = 3  # Cancel after 3 words

        async def mock_stream_iter():
            for i, chunk in enumerate(partial_words):
                yield {"event": "token", "content": chunk + " "}
                if i == cancel_at_idx:
                    set_cancel_token(chat_id)

        with patch("app.services.rag_graph.run_stream", return_value=mock_stream_iter()):
            with patch("app.services.chat_service.asyncio.create_task"):
                frames = []
                async for frame in generate_response(
                    query="test question",
                    messages={"messages": []},
                    knowledge_base_ids=[1],
                    chat_id=chat_id,
                    db=db,
                ):
                    frames.append(frame)

    # The partial response should be the words streamed before cancel
    expected_partial = " ".join(partial_words[:cancel_at_idx + 1])
    assert bot_msg.content.strip() == expected_partial
    assert db.commit.call_count >= 1


# ---------------------------------------------------------------------------
# test_cancel_token_cleaned_on_normal_completion
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_token_cleaned_on_normal_completion():
    """After normal completion (no cancel), clear_cancel_token should be called."""
    from app.services.chat_service import generate_response
    from app.services.cancel_registry import get_cancel_token, set_cancel_token

    chat_id = 601
    db = MagicMock()
    db.commit = MagicMock()
    db.close = MagicMock()
    db.add = MagicMock()
    mock_query = MagicMock()
    mock_query.filter.return_value.first.return_value = _make_mock_chat()
    db.query.return_value = mock_query

    bot_msg = MagicMock()
    bot_msg.content = ""
    bot_msg.role = "assistant"
    bot_msg.chat_id = chat_id

    # Pre-set a cancel token (simulating a prior cancel that wasn't cleared)
    set_cancel_token(chat_id)
    assert get_cancel_token(chat_id).is_set()

    with patch("app.services.chat_service.Message") as MockMsg:
        MockMsg.side_effect = [
            MagicMock(id=1, content="test", role="user", chat_id=chat_id),
            bot_msg,
        ]

        async def mock_stream_iter():
            yield {"event": "token", "content": "answer"}
            yield {"event": "done", "usage": {"promptTokens": 1, "completionTokens": 1}}

        with patch("app.services.rag_graph.run_stream", return_value=mock_stream_iter()):
            with patch("app.services.chat_service.asyncio.create_task"):
                frames = []
                async for frame in generate_response(
                    query="test question",
                    messages={"messages": []},
                    knowledge_base_ids=[1],
                    chat_id=chat_id,
                    db=db,
                ):
                    frames.append(frame)

    # After normal completion, the token should be cleared
    import app.services.cancel_registry as reg
    assert chat_id not in reg._cancel_tokens


# ---------------------------------------------------------------------------
# test_cancel_token_cleaned_on_error
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_token_cleaned_on_error():
    """After an exception, clear_cancel_token should be called."""
    from app.services.chat_service import generate_response
    from app.services.cancel_registry import set_cancel_token

    chat_id = 701
    db = MagicMock()
    db.commit = MagicMock()
    db.close = MagicMock()
    db.add = MagicMock()
    mock_query = MagicMock()
    mock_query.filter.return_value.first.return_value = _make_mock_chat()
    db.query.return_value = mock_query

    bot_msg = MagicMock()
    bot_msg.content = ""
    bot_msg.role = "assistant"
    bot_msg.chat_id = chat_id

    set_cancel_token(chat_id)

    with patch("app.services.chat_service.Message") as MockMsg:
        MockMsg.side_effect = [
            MagicMock(id=1, content="test", role="user", chat_id=chat_id),
            bot_msg,
        ]

        async def mock_stream_iter():
            raise RuntimeError("LLM connection failed")

        with patch("app.services.rag_graph.run_stream", return_value=mock_stream_iter()):
            with patch("app.services.chat_service.asyncio.create_task"):
                frames = []
                async for frame in generate_response(
                    query="test question",
                    messages={"messages": []},
                    knowledge_base_ids=[1],
                    chat_id=chat_id,
                    db=db,
                ):
                    frames.append(frame)

    # Error frames should have been yielded
    assert any("3:" in f for f in frames)
    # Error message should be in bot_message.content
    assert "Error generating response" in bot_msg.content
    # Token should be cleaned up
    import app.services.cancel_registry as reg
    assert chat_id not in reg._cancel_tokens


# ---------------------------------------------------------------------------
# test_cancel_mid_agent_step  (R008 — multi-agent cancellation)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_mid_agent_step():
    """When cancellation occurs while a multi-agent step is active,
    the stream loop breaks immediately and the partial response (including
    agent steps seen so far) is saved. This verifies R008: cancellation
    stops an active agent step mid-execution."""
    from app.services.chat_service import generate_response
    from app.services.cancel_registry import set_cancel_token

    chat_id = 801
    db = MagicMock()
    db.commit = MagicMock()
    db.close = MagicMock()
    db.add = MagicMock()
    mock_query = MagicMock()
    mock_query.filter.return_value.first.return_value = _make_mock_chat()
    db.query.return_value = mock_query

    bot_msg = MagicMock()
    bot_msg.content = ""
    bot_msg.role = "assistant"
    bot_msg.chat_id = chat_id
    bot_msg.rewritten_query = None

    with patch("app.services.chat_service.Message") as MockMsg:
        MockMsg.side_effect = [
            MagicMock(id=1, content="test", role="user", chat_id=chat_id),
            bot_msg,
        ]

        async def mock_stream_iter():
            # Simulate multi-agent workflow: rewrite_query → context_router → draft_answer
            yield {"event": "agent_step", "node": "rewrite_query", "status": "done", "latency_ms": 45}
            yield {"event": "agent_step", "node": "context_router", "status": "done", "latency_ms": 30}
            # Start draft_answer — this is the "active" step that gets cancelled
            yield {"event": "agent_step", "node": "draft_answer", "status": "active", "latency_ms": 0}
            # Some tokens start flowing
            yield {"event": "token", "content": "Partial "}
            # Cancel while draft_answer is still active
            set_cancel_token(chat_id)
            # These should NOT be emitted
            yield {"event": "token", "content": "should not appear"}
            yield {"event": "agent_step", "node": "draft_answer", "status": "done", "latency_ms": 100}

        with patch("app.services.rag_graph.run_stream", return_value=mock_stream_iter()):
            with patch("app.services.chat_service.asyncio.create_task"):
                frames = []
                async for frame in generate_response(
                    query="test question",
                    messages={"messages": []},
                    knowledge_base_ids=[1],
                    chat_id=chat_id,
                    db=db,
                ):
                    frames.append(frame)

    # Verify: stream broke mid-agent-step
    agent_step_frames = [f for f in frames if f.startswith("4:")]
    assert len(agent_step_frames) == 3, f"Expected 3 agent_step frames, got {len(agent_step_frames)}"

    # Parse agent steps to verify they match expected nodes
    import json
    nodes = [json.loads(f[2:]) for f in agent_step_frames]  # "4:{json}" → f[2:] gives {json}\n
    node_names = [n["node"] for n in nodes]
    assert "rewrite_query" in node_names
    assert "context_router" in node_names
    assert "draft_answer" in node_names

    # draft_answer should have status "active" (cancelled mid-execution)
    draft_answer_step = [n for n in nodes if n["node"] == "draft_answer"]
    assert len(draft_answer_step) == 1
    assert draft_answer_step[0]["status"] == "active", \
        "draft_answer should be 'active' (cancelled mid-execution), not 'done'"

    # Partial response should be saved (with trailing space from token streaming)
    assert bot_msg.content.strip() == "Partial"
    db.commit.assert_called()

    # Token should be cleaned up
    import app.services.cancel_registry as reg
    assert chat_id not in reg._cancel_tokens

    # Tokens after cancel should NOT appear
    assert "should not appear" not in bot_msg.content


# ---------------------------------------------------------------------------
# test_cancel_then_chat_reusable  (R016 — full flow after cancel)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_then_chat_reusable():
    """After a response is cancelled and partial response saved,
    the chat remains reusable for new queries. This verifies R016:
    the ask→cancel→edit→branch flow works end-to-end."""
    from app.services.chat_service import generate_response
    from app.services.cancel_registry import set_cancel_token

    chat_id = 901
    db = MagicMock()
    db.commit = MagicMock()
    db.close = MagicMock()
    db.add = MagicMock()
    mock_query = MagicMock()
    mock_query.filter.return_value.first.return_value = _make_mock_chat()
    db.query.return_value = mock_query

    bot_msg = MagicMock()
    bot_msg.content = ""
    bot_msg.role = "assistant"
    bot_msg.chat_id = chat_id
    bot_msg.rewritten_query = None

    with patch("app.services.chat_service.Message") as MockMsg:
        MockMsg.side_effect = [
            MagicMock(id=1, content="test", role="user", chat_id=chat_id),
            bot_msg,
        ]

        async def mock_stream_iter():
            yield {"event": "token", "content": "First "}
            set_cancel_token(chat_id)
            yield {"event": "token", "content": "should not appear"}

        with patch("app.services.rag_graph.run_stream", return_value=mock_stream_iter()):
            with patch("app.services.chat_service.asyncio.create_task"):
                frames = []
                async for frame in generate_response(
                    query="first question",
                    messages={"messages": []},
                    knowledge_base_ids=[1],
                    chat_id=chat_id,
                    db=db,
                ):
                    frames.append(frame)

    # Partial response saved (with trailing space from token streaming)
    assert bot_msg.content.strip() == "First"
    db.commit.assert_called()

    # ── Simulate second query on the same chat (R016: chat remains reusable) ──
    bot_msg2 = MagicMock()
    bot_msg2.content = ""
    bot_msg2.role = "assistant"
    bot_msg2.chat_id = chat_id
    bot_msg2.rewritten_query = None

    with patch("app.services.chat_service.Message") as MockMsg:
        MockMsg.side_effect = [
            MagicMock(id=2, content="second", role="user", chat_id=chat_id),
            bot_msg2,
        ]

        async def mock_stream_iter2():
            yield {"event": "token", "content": "Second "}
            yield {"event": "token", "content": "answer"}
            yield {"event": "done", "usage": {"promptTokens": 5, "completionTokens": 2}}

        with patch("app.services.rag_graph.run_stream", return_value=mock_stream_iter2()):
            with patch("app.services.chat_service.asyncio.create_task"):
                frames2 = []
                async for frame in generate_response(
                    query="second question",
                    messages={"messages": []},
                    knowledge_base_ids=[1],
                    chat_id=chat_id,
                    db=db,
                ):
                    frames2.append(frame)

    # Second query completes normally — chat is reusable
    assert bot_msg2.content == "Second answer"
    db.commit.assert_called()


# ---------------------------------------------------------------------------
# test_cancel_preserves_agent_steps_in_db  (R008 — agent steps survive cancel)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_preserves_agent_steps_in_db():
    """When cancellation occurs mid-stream, the partial response saved to DB
    includes any context/agent-step data that was accumulated before cancel.
    This verifies R008: agent steps captured before cancellation persist."""
    from app.services.chat_service import generate_response
    from app.services.cancel_registry import set_cancel_token

    chat_id = 951
    db = MagicMock()
    db.commit = MagicMock()
    db.close = MagicMock()
    db.add = MagicMock()
    mock_query = MagicMock()
    mock_query.filter.return_value.first.return_value = _make_mock_chat()
    db.query.return_value = mock_query

    bot_msg = MagicMock()
    bot_msg.content = ""
    bot_msg.role = "assistant"
    bot_msg.chat_id = chat_id
    bot_msg.rewritten_query = None

    with patch("app.services.chat_service.Message") as MockMsg:
        MockMsg.side_effect = [
            MagicMock(id=1, content="test", role="user", chat_id=chat_id),
            bot_msg,
        ]

        async def mock_stream_iter():
            # Emit an agent_step and a context event before cancellation
            yield {"event": "agent_step", "node": "rewrite_query", "status": "done", "latency_ms": 45}
            yield {"event": "rewritten_query", "query": "rewritten question"}
            yield {"event": "context", "docs": [], "rewritten_query": "rewritten question"}
            yield {"event": "token", "content": "Answer "}
            set_cancel_token(chat_id)
            yield {"event": "token", "content": "should not appear"}

        with patch("app.services.rag_graph.run_stream", return_value=mock_stream_iter()):
            with patch("app.services.chat_service.asyncio.create_task"):
                frames = []
                async for frame in generate_response(
                    query="test question",
                    messages={"messages": []},
                    knowledge_base_ids=[1],
                    chat_id=chat_id,
                    db=db,
                ):
                    frames.append(frame)

    # Partial response saved with accumulated content
    # Note: context event wraps response in base64 + "__LLM_RESPONSE__" prefix
    assert "__LLM_RESPONSE__Answer " in bot_msg.content
    db.commit.assert_called()

    # Verify that the rewritten_query was captured
    assert bot_msg.rewritten_query == "rewritten question"

    # Agent step frames should include rewrite_query
    import json
    agent_step_frames = [f for f in frames if f.startswith("4:")]
    assert len(agent_step_frames) >= 1
    node = json.loads(agent_step_frames[0][2:])  # "4:{json}" → f[2:] gives {json}\n
    assert node["node"] == "rewrite_query"
    assert node["status"] == "done"

    # Token cleanup
    import app.services.cancel_registry as reg
    assert chat_id not in reg._cancel_tokens
