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
