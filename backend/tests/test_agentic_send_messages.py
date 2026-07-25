"""Tests for the Send() messages fix.

Verifies that route_by_dependencies sends subtask history via a separate
"subgraph_history" field rather than overwriting "messages", so that
LangGraph's MessagesState reducer never appends duplicate message objects
when subgraphs complete.
"""

import pytest
import sys
from unittest.mock import MagicMock

# Mock the langgraph.checkpoint.redis and langgraph.store.redis modules before
# importing anything that uses them. This is a pre-existing import error that
# blocks the test suite on local machines without the Docker Redis cluster.
_mock_modules = [
    "langgraph.checkpoint.redis", "langgraph.checkpoint.redis.aio",
    "langgraph.store.redis", "langgraph.store.redis.aio",
    "langgraph.store.redis.base", "langgraph.store.redis.hash",
    "redisvl", "redisvl.index", "redisvl.index.index",
]
for _mod in _mock_modules:
    sys.modules[_mod] = MagicMock()

from app.services.agentic_rag.graph import route_by_dependencies
from app.services.agentic_rag.nodes import _build_generation_messages, select_recent_history
from app.services.agentic_rag.schemas import SubtaskRouting
from langchain_core.messages import HumanMessage, AIMessage


# ---------------------------------------------------------------------------
# 1. Send() state — subgraph_history not messages
# ---------------------------------------------------------------------------

class TestRouteByDependenciesSendState:
    """route_by_dependencies must NOT set 'messages' in Send() payloads."""

    def _make_state(self, subtasks=None, dependencies=None, subtask_routing=None,
                    question_is_clear=True, needs_retrieval=True,
                    needs_file_content=False, needs_file_metadata=False):
        """Build a minimal AgentState-like dict for route_by_dependencies."""
        return {
            "question_is_clear": question_is_clear,
            "original_query": "test query",
            "subtasks": subtasks or ["test query"],
            "subtask_dependencies": dependencies or [[]],
            "subtask_routing": subtask_routing or [SubtaskRouting(
                needs_retrieval=needs_retrieval,
                needs_file_content=needs_file_content,
                needs_file_metadata=needs_file_metadata,
            )],
            "needs_file_content": needs_file_content,
            "needs_file_metadata": needs_file_metadata,
            "kb_ids": [1],
            "org_id": None,
            "chat_id": None,
            "user_id": None,
            "file_markdown": None,
            "rewritten_query": "test query",
        }

    def test_single_subtask_no_messages_in_send(self):
        """Single subtask Send() must use subgraph_history, not messages."""
        sends = route_by_dependencies(self._make_state())
        assert isinstance(sends, list)
        assert len(sends) == 1
        send_kwarg = sends[0].arg
        # CRITICAL: "messages" must NOT be in the Send() payload.
        assert "messages" not in send_kwarg, (
            f"Send() must not set 'messages' — got keys: {list(send_kwarg.keys())}"
        )
        # subgraph_history must be present.
        assert "subgraph_history" in send_kwarg, (
            "Send() must set 'subgraph_history' for the subgraph to access prior turns."
        )

    def test_multiple_independent_subtasks_no_messages_in_send(self):
        """Independent subtasks sent via Send() must all use subgraph_history."""
        state = self._make_state(
            subtasks=["query A", "query B"],
            dependencies=[[], []],  # no dependencies
            subtask_routing=[
                SubtaskRouting(needs_retrieval=True, needs_file_content=False, needs_file_metadata=False),
                SubtaskRouting(needs_retrieval=True, needs_file_content=False, needs_file_metadata=False),
            ],
        )
        sends = route_by_dependencies(state)
        assert isinstance(sends, list)
        for send_obj in sends:
            send_kwarg = send_obj.arg
            assert "messages" not in send_kwarg, (
                f"Send() must not set 'messages' in independent subtask — got keys: {list(send_kwarg.keys())}"
            )
            assert "subgraph_history" in send_kwarg

    def test_multiple_subtasks_different_routing_no_messages_in_send(self):
        """Mixed routing (retrieval vs chat) Send() payloads must not set 'messages'."""
        state = self._make_state(
            subtasks=["query A", "query B"],
            dependencies=[[], []],
            subtask_routing=[
                SubtaskRouting(needs_retrieval=True, needs_file_content=False, needs_file_metadata=False),
                SubtaskRouting(needs_retrieval=False, needs_file_content=False, needs_file_metadata=False),
            ],
        )
        sends = route_by_dependencies(state)
        assert isinstance(sends, list)
        for send_obj in sends:
            send_kwarg = send_obj.arg
            assert "messages" not in send_kwarg, (
                f"Send() must not set 'messages' — got keys: {list(send_kwarg.keys())}"
            )

    def test_chat_subtask_no_messages_in_send(self):
        """Chat subtask (needs_retrieval=False) must use subgraph_history, not messages."""
        state = self._make_state(
            subtasks=["what did I say"],
            subtask_routing=[
                SubtaskRouting(needs_retrieval=False, needs_file_content=False, needs_file_metadata=False),
            ],
        )
        sends = route_by_dependencies(state)
        assert isinstance(sends, list)
        send_kwarg = sends[0].arg
        assert "messages" not in send_kwarg
        assert "subgraph_history" in send_kwarg

    def test_file_context_subtask_no_messages_in_send(self):
        """File context subtask (needs_file_content=True) must use subgraph_history."""
        state = self._make_state(
            subtasks=["summarize this"],
            subtask_routing=[
                SubtaskRouting(needs_retrieval=True, needs_file_content=True, needs_file_metadata=False),
            ],
            needs_file_content=True,
        )
        sends = route_by_dependencies(state)
        assert isinstance(sends, list)
        send_kwarg = sends[0].arg
        assert "messages" not in send_kwarg
        assert "subgraph_history" in send_kwarg


# ---------------------------------------------------------------------------
# 2. _build_generation_messages — no duplicates
# ---------------------------------------------------------------------------

class TestBuildGenerationMessagesNoDuplication:
    """_build_generation_messages must produce a clean, deduplicated message list."""

    def _make_mock_state(self, messages, **overrides):
        """Build a dict that looks like AgentState for _build_generation_messages."""
        state = {
            "messages": messages,
            "original_query": messages[-1].content if messages else "query",
            "retrieved_docs": [],
            "file_markdown": None,
            "compaction_summary": None,
            "compaction_triggered": False,
            "needs_retrieval": True,
            "needs_file_content": False,
            "needs_file_metadata": False,
        }
        state.update(overrides)
        return state

    def test_no_duplicate_user_messages(self):
        """User messages with identical content must be deduplicated."""
        msgs = [
            HumanMessage(content="same content"),  # from parent
            AIMessage(content="some answer"),
            HumanMessage(content="same content"),  # duplicate from subgraph Send()
            HumanMessage(content="current query"),
        ]
        state = self._make_mock_state(msgs)
        result = _build_generation_messages(state)

        # Count user messages that appear in the history portion (not the final one)
        user_in_history = [m for m in result[:-1] if m.get("role") == "user"]
        # Each unique user content should appear only once in history.
        user_contents = [m["content"] for m in user_in_history]
        assert len(user_contents) == len(set(user_contents)), (
            f"Duplicated user messages found: {user_contents}"
        )

    def test_messages_end_with_user_query(self):
        """The last entry in generated messages must always be the user query."""
        msgs = [
            HumanMessage(content="first query"),
            AIMessage(content="assistant answer to first"),
            HumanMessage(content="explain mutex"),  # current query
        ]
        state = self._make_mock_state(msgs)
        result = _build_generation_messages(state)

        assert len(result) >= 2
        assert result[-1]["role"] == "user"
        assert "explain mutex" in result[-1]["content"]

    def test_system_message_at_start(self):
        """System message must always be at position 0."""
        msgs = [HumanMessage(content="query")]
        state = self._make_mock_state(msgs)
        result = _build_generation_messages(state)

        assert result[0]["role"] == "system"

    def test_no_system_messages_in_conversation_history(self):
        """System/context messages in state.messages must be filtered out."""
        msgs = [
            # Simulate a leaked system message from a prior LLM call
            {"role": "system", "content": "You are a helpful AI assistant..."},
            HumanMessage(content="first query"),
            AIMessage(content="first answer"),
            HumanMessage(content="second query"),
        ]
        state = self._make_mock_state(msgs)
        result = _build_generation_messages(state)

        # The system message from state.messages should be filtered out.
        # Only the freshly generated system prompt should be at position 0.
        # After that, only user/assistant messages.
        history_roles = [m.get("role") for m in result[1:]]
        assert "system" not in history_roles, (
            "System messages from state.messages should not leak into the generation prompt."
        )


# ---------------------------------------------------------------------------
# 3. select_recent_history creates new objects (documented behavior)
# ---------------------------------------------------------------------------

class TestSelectRecentHistoryBehavior:
    """document: select_recent_history creates new message objects."""

    def test_new_message_objects(self):
        """select_recent_history creates new HumanMessage/AIMessage instances."""
        orig_h = HumanMessage(content="hello")
        orig_a = AIMessage(content="world")
        msgs = [orig_h, orig_a, HumanMessage(content="next")]
        result = select_recent_history(msgs, max_pairs=2)

        # Each item in result is a NEW object (not the same as input).
        assert result[0] is not orig_h, (
            "select_recent_history must create new HumanMessage objects."
        )
        assert result[1] is not orig_a, (
            "select_recent_history must create new AIMessage objects."
        )
        # AI messages must be truncated to 400 chars.
        long_msg = AIMessage(content="x" * 500)
        msgs2 = [HumanMessage(content="q"), long_msg, HumanMessage(content="q2")]
        result2 = select_recent_history(msgs2, max_pairs=2)
        assert len(result2[1].content) <= 400, (
            "select_recent_history must truncate AI messages to 400 chars."
        )


# ---------------------------------------------------------------------------
# 4. SubgraphHistory vs Messages — isolation test
# ---------------------------------------------------------------------------

class TestSubgraphHistoryIsolation:
    """subgraph_history field is isolated from the messages channel."""

    def test_send_payload_keys_for_single_subtask(self):
        """Verify the exact keys in the single-subtask Send() payload."""
        state = {
            "question_is_clear": True,
            "original_query": "test",
            "subtasks": ["test"],
            "subtask_dependencies": [[]],
            "subtask_routing": [SubtaskRouting(
                needs_retrieval=True,
                needs_file_content=False,
                needs_file_metadata=False,
            )],
            "needs_file_content": False,
            "needs_file_metadata": False,
            "kb_ids": [1],
            "org_id": None,
            "chat_id": None,
            "user_id": None,
            "file_markdown": None,
            "rewritten_query": "test",
        }
        sends = route_by_dependencies(state)
        payload_keys = list(sends[0].arg.keys())

        assert "messages" not in payload_keys
        assert "subgraph_history" in payload_keys
        # Verify expected keys are present
        expected_keys = [
            "kb_ids", "org_id", "chat_id", "user_id", "file_markdown",
            "original_query", "rewritten_query", "subgraph_history",
            "subtasks", "is_complex", "current_subtask_index",
            "needs_retrieval", "needs_file_content", "needs_file_metadata",
            "subtask_routing",
        ]
        for key in expected_keys:
            assert key in payload_keys, f"Send() payload missing key: {key}"
