"""
Unit tests for rag_graph.py

Tests cover:
- RAGGraphState TypedDict structure
- rewrite_query_node (mocked LLM)
- context_router_node (mocked LLM)
- run_stream interface contract (importable, yields expected event shapes)
"""
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.rag_graph import (
    RAGGraphState,
    rewrite_query_node,
    context_router_node,
    run_stream,
    EVENT_AGENT_STEP,
    EVENT_REWRITTEN,
    EVENT_CONTEXT,
    EVENT_TOKEN,
    EVENT_DONE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state(**overrides) -> dict:
    """Return a minimal RAGGraphState-compatible dict."""
    base = {
        "query": "What is RAG?",
        "rewritten_query": "",
        "route": "kb",
        "sources": [],
        "file_ids_needed": [],
        "router_rationale": "",
        "file_markdown": None,
        "retrieved_docs": [],
        "graded_docs": [],
        "merged_context": "",
        "answer": "",
        "agent_steps": [],
        "knowledge_base_ids": [1],
        "recent_lc_history": [],
        "existing_summary": None,
        "use_dense": True,
        "use_sparse": True,
        "use_exact": True,
        "use_graph_rag": False,
        "temperature": 0.0,
        "model_name": None,
        "display_query": None,
    }
    base.update(overrides)
    return base


def _make_llm_response(content: str):
    """Create a mock LLM response object."""
    resp = MagicMock()
    resp.content = content
    return resp


# ---------------------------------------------------------------------------
# RAGGraphState structure
# ---------------------------------------------------------------------------

class TestRAGGraphState:
    def test_required_keys_present(self):
        """RAGGraphState TypedDict should accept all required keys."""
        state: RAGGraphState = _base_state()  # type: ignore[assignment]
        assert "query" in state
        assert "rewritten_query" in state
        assert "sources" in state
        assert "file_ids_needed" in state
        assert "router_rationale" in state
        assert "agent_steps" in state

    def test_agent_steps_list(self):
        state = _base_state()
        assert isinstance(state["agent_steps"], list)

    def test_sources_list(self):
        state = _base_state()
        assert isinstance(state["sources"], list)


# ---------------------------------------------------------------------------
# rewrite_query_node
# ---------------------------------------------------------------------------

class TestRewriteQueryNode:
    @pytest.mark.asyncio
    async def test_returns_rewritten_query(self):
        """Node should update rewritten_query with LLM output."""
        state = _base_state(query="how does rag work?")
        mock_response = _make_llm_response("RAG retrieval augmented generation mechanism")

        with patch("app.services.rag_graph._get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm

            result = await rewrite_query_node(state)

        assert result["rewritten_query"] == "RAG retrieval augmented generation mechanism"

    @pytest.mark.asyncio
    async def test_appends_agent_step(self):
        """Node should append an agent_step dict."""
        state = _base_state()
        mock_response = _make_llm_response("rewritten query")

        with patch("app.services.rag_graph._get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm

            result = await rewrite_query_node(state)

        steps = result["agent_steps"]
        assert len(steps) == 1
        assert steps[0]["node"] == "rewrite_query"
        assert "latency_ms" in steps[0]
        assert steps[0]["status"] == "done"

    @pytest.mark.asyncio
    async def test_preserves_existing_agent_steps(self):
        """Node should append to existing agent_steps, not replace."""
        existing = [{"node": "prior", "latency_ms": 5.0, "status": "done"}]
        state = _base_state(agent_steps=existing)
        mock_response = _make_llm_response("rewritten")

        with patch("app.services.rag_graph._get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm

            result = await rewrite_query_node(state)

        assert len(result["agent_steps"]) == 2
        assert result["agent_steps"][0]["node"] == "prior"

    @pytest.mark.asyncio
    async def test_falls_back_to_original_on_empty_llm_response(self):
        """Empty LLM response should fall back to original query."""
        state = _base_state(query="original query")
        mock_response = _make_llm_response("   ")  # whitespace only

        with patch("app.services.rag_graph._get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm

            result = await rewrite_query_node(state)

        assert result["rewritten_query"] == "original query"


# ---------------------------------------------------------------------------
# context_router_node
# ---------------------------------------------------------------------------

class TestContextRouterNode:
    @pytest.mark.asyncio
    async def test_parses_sources(self):
        """Node should parse LLM JSON and set sources."""
        state = _base_state(rewritten_query="what is in the document?", file_markdown="some content")
        json_resp = json.dumps({
            "sources": ["kb", "file_current"],
            "rationale": "query references a document",
            "file_ids_needed": [],
        })
        mock_response = _make_llm_response(json_resp)

        with patch("app.services.rag_graph._get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm

            result = await context_router_node(state)

        assert "kb" in result["sources"]
        assert "file_current" in result["sources"]
        assert result["router_rationale"] == "query references a document"

    @pytest.mark.asyncio
    async def test_appends_agent_step(self):
        """Node should append an agent_step dict with sources."""
        state = _base_state()
        json_resp = json.dumps({
            "sources": ["kb"],
            "rationale": "general question",
            "file_ids_needed": [],
        })
        mock_response = _make_llm_response(json_resp)

        with patch("app.services.rag_graph._get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm

            result = await context_router_node(state)

        steps = result["agent_steps"]
        assert len(steps) == 1
        assert steps[0]["node"] == "context_router"
        assert "sources" in steps[0]
        assert steps[0]["status"] == "done"

    @pytest.mark.asyncio
    async def test_fallback_on_bad_json(self):
        """Malformed LLM JSON should fall back to kb-only."""
        state = _base_state()
        mock_response = _make_llm_response("not valid json {{{")

        with patch("app.services.rag_graph._get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm

            result = await context_router_node(state)

        assert result["sources"] == ["kb"]
        assert "fallback" in result["router_rationale"]

    @pytest.mark.asyncio
    async def test_file_ids_parsed_as_ints(self):
        """file_ids_needed should be coerced to ints."""
        state = _base_state()
        json_resp = json.dumps({
            "sources": ["file_prior"],
            "rationale": "referencing prior files",
            "file_ids_needed": ["42", "7"],
        })
        mock_response = _make_llm_response(json_resp)

        with patch("app.services.rag_graph._get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm

            result = await context_router_node(state)

        assert result["file_ids_needed"] == [42, 7]


# ---------------------------------------------------------------------------
# run_stream interface contract
# ---------------------------------------------------------------------------

def _make_mock_llm(content: str):
    """Return a mock LLM whose ainvoke always returns content."""
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=_make_llm_response(content))
    return mock_llm


def _router_json(sources=None, rationale="general", file_ids=None):
    return json.dumps({
        "sources": sources or ["kb"],
        "rationale": rationale,
        "file_ids_needed": file_ids or [],
    })


def _grade_json(relevant=True):
    return json.dumps({"relevant": relevant})


class TestRunStreamInterface:
    @pytest.mark.asyncio
    async def test_importable_and_async_generator(self):
        """run_stream should be importable and return an async generator."""
        import inspect
        assert inspect.isasyncgenfunction(run_stream)

    @pytest.mark.asyncio
    async def test_yields_expected_event_keys(self):
        """run_stream should yield dicts with 'event' key (full graph mocked)."""
        # Each node calls _get_llm once; return different payloads in call order:
        # rewrite, router, (extract skips LLM for small/no file), grade (no docs → skips), answer
        call_payloads = [
            "rewritten query",           # rewrite_query_node
            _router_json(),              # context_router_node
            "The answer is here.",       # generate_answer_node
        ]
        call_iter = iter(call_payloads)

        def side_effect(*args, **kwargs):
            payload = next(call_iter, "fallback")
            return _make_mock_llm(payload)

        events = []
        with patch("app.services.rag_graph._get_llm", side_effect=side_effect):
            async for evt in run_stream(
                query="test",
                file_markdown=None,
                db=None,
                chat_id=1,
                knowledge_base_ids=[],
                recent_lc_history=[],
                existing_summary=None,
            ):
                events.append(evt)

        assert len(events) > 0
        for evt in events:
            assert "event" in evt

    @pytest.mark.asyncio
    async def test_yields_done_event(self):
        """run_stream should always yield a 'done' event last."""
        call_payloads = iter([
            "rewritten query",
            _router_json(),
            "The answer.",
        ])

        def side_effect(*args, **kwargs):
            return _make_mock_llm(next(call_payloads, "fallback"))

        events = []
        with patch("app.services.rag_graph._get_llm", side_effect=side_effect):
            async for evt in run_stream(
                query="test",
                file_markdown=None,
                db=None,
                chat_id=1,
                knowledge_base_ids=[],
                recent_lc_history=[],
                existing_summary=None,
            ):
                events.append(evt)

        assert events[-1]["event"] == EVENT_DONE

    @pytest.mark.asyncio
    async def test_done_event_has_full_response(self):
        """done event should have full_response and usage fields."""
        call_payloads = iter([
            "rewritten query",
            _router_json(),
            "The answer.",
        ])

        def side_effect(*args, **kwargs):
            return _make_mock_llm(next(call_payloads, "fallback"))

        with patch("app.services.rag_graph._get_llm", side_effect=side_effect):
            async for evt in run_stream(
                query="test",
                file_markdown=None,
                db=None,
                chat_id=1,
                knowledge_base_ids=[],
                recent_lc_history=[],
                existing_summary=None,
            ):
                if evt["event"] == EVENT_DONE:
                    assert "full_response" in evt
                    assert "usage" in evt
                    assert "promptTokens" in evt["usage"]
                    assert "completionTokens" in evt["usage"]
                    break

    def test_event_constants_defined(self):
        """All SSE event type constants should be non-empty strings."""
        for const in [EVENT_AGENT_STEP, EVENT_REWRITTEN, EVENT_CONTEXT, EVENT_TOKEN, EVENT_DONE]:
            assert isinstance(const, str)
            assert len(const) > 0
