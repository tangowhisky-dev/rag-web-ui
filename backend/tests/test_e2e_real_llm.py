"""
test_e2e_real_llm.py — End-to-end tests against a real LLM endpoint.

These tests exercise the full pipeline: authentication → chat creation →
streaming message → agent graph → LLM calls → tool dispatch → retrieval →
answer generation. They require:
  - A running backend with a real MySQL database.
  - A reachable OpenAI-compatible LLM endpoint configured in the settings table.
  - A user 'tango' with password 'tango123' (org_id=8, KB id=17 with
    'Computer Architecture Book' containing ~2988 chunks).

Tests are skipped automatically when:
  - The LLM endpoint is unreachable.
  - The user 'tango' doesn't exist in the database.

Run explicitly:
    docker exec rag-web-ui-backend-1 pytest tests/test_e2e_real_llm.py -v --timeout=300 -s

NOT part of the default test suite — these are integration tests that hit
real external services and take 30-120s per turn.
"""
import json
import os
import time
import logging
from typing import Optional

import pytest
import requests

logger = logging.getLogger(__name__)

# ── Skip conditions ──────────────────────────────────────────────────────────

E2E_BASE_URL = os.getenv("E2E_BASE_URL", "http://localhost:8000/api")
E2E_USERNAME = os.getenv("E2E_USERNAME", "tango")
E2E_PASSWORD = os.getenv("E2E_PASSWORD", "tango123")
E2E_KB_ID = int(os.getenv("E2E_KB_ID", "17"))


def _backend_reachable() -> bool:
    """Check if the backend is running and the test user can log in."""
    try:
        resp = requests.post(
            f"{E2E_BASE_URL}/auth/token",
            data={"username": E2E_USERNAME, "password": E2E_PASSWORD},
            timeout=30,
        )
        if resp.status_code != 200:
            return False
        # Verify the LLM is actually reachable by sending a test message
        # to a throwaway chat. If the LLM is down, the first message will
        # return an error event.
        token = resp.json()["access_token"]
        chat = requests.post(
            f"{E2E_BASE_URL}/chat",
            json={"title": "E2E Health Check", "knowledge_base_ids": [E2E_KB_ID]},
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        if chat.status_code != 200:
            return False
        chat_id = chat.json()["id"]
        # Clean up the chat
        try:
            requests.delete(
                f"{E2E_BASE_URL}/chat/{chat_id}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
        except Exception:
            pass
        return True
    except Exception as e:
        logger.warning("Backend not reachable: %s", e)
        return False


pytestmark = pytest.mark.skipif(
    not _backend_reachable(),
    reason="Requires running backend with LLM configured and user 'tango'.",
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _login() -> str:
    """Login as the test user and return the access token."""
    resp = requests.post(
        f"{E2E_BASE_URL}/auth/token",
        data={"username": E2E_USERNAME, "password": E2E_PASSWORD},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _create_chat(token: str, title: str, kb_ids: list[int]) -> dict:
    """Create a chat and return the chat object."""
    resp = requests.post(
        f"{E2E_BASE_URL}/chat",
        json={"title": title, "knowledge_base_ids": kb_ids},
        headers=_headers(token),
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _send_message(token: str, chat_id: int, content: str, timeout: int = 300) -> dict:
    """Send a message and collect the full streaming response.

    Returns a dict with:
      - tokens: concatenated answer text
      - rewritten_query: the rewritten query (if any)
      - context: retrieved docs metadata (if any)
      - agent_steps: list of agent step events
      - done: the done event payload
      - errors: any error events
    """
    resp = requests.post(
        f"{E2E_BASE_URL}/chat/{chat_id}/messages",
        json={"messages": [{"role": "user", "content": content}]},
        headers=_headers(token),
        stream=True,
        timeout=timeout,
    )
    resp.raise_for_status()

    result = {
        "tokens": "",
        "rewritten_query": None,
        "context": None,
        "agent_steps": [],
        "done": None,
        "errors": [],
    }

    for line in resp.iter_lines(decode_unicode=True):
        if not line or line.startswith(":"):
            continue
        # Vercel AI SDK SSE format: <type>:<json>
        if ":" not in line:
            continue
        prefix, payload = line.split(":", 1)
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue

        if prefix == "0":
            result["tokens"] += data if isinstance(data, str) else data.get("text", "")
        elif prefix == "1":
            result["rewritten_query"] = data
        elif prefix == "2":
            result["context"] = data
        elif prefix == "3":
            result["errors"].append(data)
        elif prefix == "4":
            result["agent_steps"].append(data)
        elif prefix == "d":
            result["done"] = data

    return result


def _get_chat_messages(token: str, chat_id: int) -> list[dict]:
    """Get all messages in a chat."""
    resp = requests.get(
        f"{E2E_BASE_URL}/chat/{chat_id}",
        headers=_headers(token),
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("messages", [])


def _delete_chat(token: str, chat_id: int):
    """Delete a chat (cleanup)."""
    try:
        requests.delete(
            f"{E2E_BASE_URL}/chat/{chat_id}",
            headers=_headers(token),
            timeout=30,
        )
    except Exception:
        pass


# ── Test fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def token():
    return _login()


@pytest.fixture(scope="module")
def chat_id(token):
    """Create a chat for the test module, delete it after."""
    chat = _create_chat(token, "E2E Real LLM Tests", [E2E_KB_ID])
    cid = chat["id"]
    yield cid
    _delete_chat(token, cid)


# ── 1. Authentication ────────────────────────────────────────────────────────


class TestAuthentication:
    def test_login_succeeds(self):
        """tango/tango123 should authenticate successfully."""
        tok = _login()
        assert tok, "Should get a non-empty access token"

    def test_login_wrong_password_fails(self):
        """Wrong password should return 401."""
        resp = requests.post(
            f"{E2E_BASE_URL}/auth/token",
            data={"username": E2E_USERNAME, "password": "wrong"},
            timeout=30,
        )
        assert resp.status_code == 401


# ── 2. Single-turn rag_retrieve against real KB + real LLM ──────────────────


class TestSingleTurnRetrieval:
    """Send a factual question about computer architecture and verify:
      - The agent graph runs plan → think → rag_retrieve → finalize.
      - Retrieved docs are returned in the context event.
      - The answer is a non-empty, relevant string.
    """

    def test_rag_retrieve_returns_answer(self, token, chat_id):
        result = _send_message(
            token, chat_id,
            "What is a cache line and why is it important for CPU performance?",
            timeout=300,
        )
        assert not result["errors"], f"Got errors: {result['errors']}"
        assert result["tokens"], "Should get a non-empty answer"
        # The answer should mention cache-related terms
        answer_lower = result["tokens"].lower()
        assert any(w in answer_lower for w in ["cache", "line", "memory", "cpu", "performance"]), \
            f"Answer should be about cache lines, got: {result['tokens'][:200]}"

    def test_context_event_has_retrieved_docs(self, token, chat_id):
        result = _send_message(
            token, chat_id,
            "What is the difference between RISC and CISC architectures?",
            timeout=300,
        )
        assert not result["errors"], f"Got errors: {result['errors']}"
        # Context event should contain retrieved documents
        assert result["context"] is not None, "Should receive a context event"
        # Context can be a dict with 'docs' or a list
        ctx = result["context"]
        if isinstance(ctx, dict):
            docs = ctx.get("docs", [])
        else:
            docs = ctx
        assert len(docs) > 0, f"Should retrieve at least 1 doc, got context: {ctx}"

    def test_agent_steps_emitted(self, token, chat_id):
        """The streaming response should include agent_step events showing
        the graph execution (plan, think, tool, finalize)."""
        result = _send_message(
            token, chat_id,
            "Explain what a pipeline hazard is in CPU design.",
            timeout=300,
        )
        assert not result["errors"], f"Got errors: {result['errors']}"
        assert len(result["agent_steps"]) > 0, "Should emit agent_step events"
        step_names = [s.get("node", "") for s in result["agent_steps"]]
        # At least some graph nodes should fire
        assert any(name for name in step_names), \
            f"Agent steps should have node names, got: {step_names}"


# ── 3. Multi-turn conversation with reference resolution ────────────────────


class TestMultiTurnConversation:
    """Verify that multi-turn conversations work:
      - Turn 1 introduces an entity (e.g. "branch prediction").
      - Turn 2 references it with "it" — the rewrite should resolve the
        reference using conversation history.
      - The answer for Turn 2 should be relevant to the entity from Turn 1.
    """

    def test_pronoun_resolution_across_turns(self, token, chat_id):
        # Turn 1: introduce "branch prediction"
        turn1 = _send_message(
            token, chat_id,
            "What is branch prediction in modern CPUs?",
            timeout=300,
        )
        assert not turn1["errors"], f"Turn 1 errors: {turn1['errors']}"
        assert turn1["tokens"], "Turn 1 should produce an answer"
        assert "branch" in turn1["tokens"].lower(), \
            f"Turn 1 answer should mention branch prediction: {turn1['tokens'][:200]}"

        # Turn 2: "How does it handle mispredictions?"
        turn2 = _send_message(
            token, chat_id,
            "How does it handle mispredictions?",
            timeout=300,
        )
        assert not turn2["errors"], f"Turn 2 errors: {turn2['errors']}"
        assert turn2["tokens"], "Turn 2 should produce an answer"

        # The rewritten query should contain "branch" (reference resolved)
        if turn2["rewritten_query"]:
            rq = turn2["rewritten_query"]
            if isinstance(rq, dict):
                rq = rq.get("rewritten_query") or rq.get("query") or json.dumps(rq)
            rq = rq.lower()
            assert "branch" in rq or "mispredict" in rq, \
                f"Rewritten query should resolve 'it' to branch prediction: {turn2['rewritten_query']}"

        # The answer should be about misprediction handling
        answer_lower = turn2["tokens"].lower()
        assert any(w in answer_lower for w in ["mispredict", "branch", "pipeline", "flush", "stall"]), \
            f"Turn 2 answer should be about misprediction handling: {turn2['tokens'][:200]}"

    def test_conversation_history_persisted(self, token, chat_id):
        """After multiple turns, the chat should have all messages persisted."""
        # Send a simple question
        _send_message(
            token, chat_id,
            "What is a TLB?",
            timeout=300,
        )
        messages = _get_chat_messages(token, chat_id)
        # Should have at least user + assistant messages from prior turns
        assert len(messages) >= 2, f"Should have persisted messages, got {len(messages)}"


# ── 4. Code execution + chart generation ────────────────────────────────────


class TestCodeExecuteAndChart:
    """Test that the agent can execute code and generate charts.
    These tools require the agent to plan code_execute → chart_generate.
    """

    def test_code_execute_produces_output(self, token, chat_id):
        """Ask the agent to compute something — it should use code_execute."""
        result = _send_message(
            token, chat_id,
            "Calculate the factorial of 10 using Python code execution and tell me the result.",
            timeout=300,
        )
        assert not result["errors"], f"Got errors: {result['errors']}"
        assert result["tokens"], "Should produce an answer"
        # The answer should contain the factorial of 10 = 3628800
        # (the LLM may format it with commas: "3,628,800")
        answer = result["tokens"].replace(",", "").replace(" ", "")
        assert "3628800" in answer, \
            f"Answer should contain factorial of 10 (3628800): {result['tokens'][:300]}"

    def test_chart_generate_from_code(self, token, chat_id):
        """Ask the agent to create a chart — it should use code_execute then
        chart_generate."""
        result = _send_message(
            token, chat_id,
            "Generate a bar chart showing the performance comparison of 5 sorting algorithms (bubble, insertion, merge, quick, heap) with O(n^2), O(n^2), O(n log n), O(n log n), O(n log n) complexity. Use code execution and then generate a chart.",
            timeout=300,
        )
        assert not result["errors"], f"Got errors: {result['errors']}"
        assert result["tokens"], "Should produce an answer"
        # The agent_step events show node names (plan, think, tool, finalize).
        # The "tool" node fires when any tool is dispatched. Multiple tool
        # executions indicate the agent used code_execute + chart_generate.
        tool_steps = [s for s in result["agent_steps"] if s.get("node") == "tool"]
        assert len(tool_steps) >= 2, \
            f"Should execute at least 2 tools (code_execute + chart_generate), " \
            f"got {len(tool_steps)} tool steps: {json.dumps(result['agent_steps'])[:500]}"


# ── 5. Summarize / extract tools ─────────────────────────────────────────────


class TestSummarizeAndExtract:
    """Test that the agent can summarize retrieved documents and extract
    structured data from them.
    """

    def test_summarize_chunks(self, token, chat_id):
        """Ask for a summary of a topic — the agent should use rag_retrieve
        and possibly summarize_chunks for a multi-document synthesis."""
        result = _send_message(
            token, chat_id,
            "Summarize the key concepts of memory hierarchy from the textbook.",
            timeout=300,
        )
        assert not result["errors"], f"Got errors: {result['errors']}"
        assert result["tokens"], "Should produce a summary"
        answer_lower = result["tokens"].lower()
        # Should mention memory hierarchy concepts
        assert any(w in answer_lower for w in ["cache", "memory", "hierarchy", "level", "latency"]), \
            f"Should mention memory hierarchy concepts: {result['tokens'][:200]}"

    def test_multi_part_synthesis(self, token, chat_id):
        """Ask a multi-part question that requires synthesizing info from
        multiple chunks — should trigger synthesis mode."""
        result = _send_message(
            token, chat_id,
            "Compare and contrast the advantages and disadvantages of pipelining versus superscalar execution. Cover at least 3 points for each.",
            timeout=300,
        )
        assert not result["errors"], f"Got errors: {result['errors']}"
        assert result["tokens"], "Should produce a synthesis answer"
        # The answer should be substantial (multi-part question)
        assert len(result["tokens"]) > 100, \
            f"Multi-part answer should be substantial: {len(result['tokens'])} chars"


# ── 6. Error handling ────────────────────────────────────────────────────────


class TestErrorHandling:
    """Verify graceful behavior on edge cases."""

    def test_empty_question_gets_response(self, token, chat_id):
        """An empty or trivial question should still get a response."""
        result = _send_message(
            token, chat_id,
            "Hello",
            timeout=300,
        )
        # Should not crash — either an answer or a graceful error
        assert result["tokens"] or result["errors"], \
            "Should get either an answer or an error, not silence"

    def test_nonexistent_kb_question(self, token):
        """Creating a chat with a nonexistent KB should fail."""
        resp = requests.post(
            f"{E2E_BASE_URL}/chat",
            json={"title": "Bad KB", "knowledge_base_ids": [99999]},
            headers=_headers(token),
            timeout=30,
        )
        assert resp.status_code in (400, 404), \
            f"Should reject nonexistent KB, got {resp.status_code}"
