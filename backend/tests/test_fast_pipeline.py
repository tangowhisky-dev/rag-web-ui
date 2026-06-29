"""
Tests for fast_pipeline.py — fast_stream() pipeline.

Covers:
  1. Standard retrieval (high confidence → single context event)
  2. Adaptive retrieval (low confidence → two context events, expanded docs)
  3. Historical memory prepending with [HIST-N] formatting
  4. Historical memory skipped when no chat_id
  5. Historical memory skipped when disabled
  6. Confidence boundary: score=55 exactly → single event (not adaptive)
  7. Adaptive no expansion when expanded docs == standard docs
  8. Query rewrite failure — non-fatal fallback
  9. Retrieval failure — pipeline continues with empty docs
  10. LLM generation failure — error message yielded
  11. Context string construction with file_markdown
  12. Historical memory retrieval exception — non-fatal fallback

Uses conftest's in-memory SQLite stub for DB mocking.
Mocks patch at their import locations:
  - settings → app.core.config.settings
  - _rewrite_query → app.services.chat_service._rewrite_query
  - hybrid_search_with_legs → app.services.retrieval.hybrid_search_with_legs
  - score_retrieval → app.services.confidence.score_retrieval
  - retrieve_historical_memory → app.services.historical_memory.retrieve_historical_memory
  - _get_llm → app.services.fast_pipeline._get_llm
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document as LangchainDocument


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_kb_doc(score: float, source: str = "test.txt", content: str = "") -> LangchainDocument:
    """Create a knowledge-base LangchainDocument with _reranker_score."""
    if not content:
        content = f"Relevant document with reranker score {score}."
    return LangchainDocument(
        page_content=content,
        metadata={"_reranker_score": score, "source": source},
    )


def _make_hist_doc(content: str, message_id: int = 1) -> dict:
    """Create a historical-memory doc (dict from retrieve_historical_memory)."""
    return {
        "page_content": content,
        "metadata": {"_source_type": "historical_memory", "message_id": message_id},
    }


def _make_mock_chunk(text: str = "Hello world"):
    """Create a mock LLM chunk with .content and optional .usage_metadata."""
    chunk = MagicMock()
    chunk.content = text
    chunk.usage_metadata = {"input_tokens": 10, "output_tokens": 5}
    return chunk


def _make_settings(**overrides):
    """Create a mock settings object matching the attrs fast_pipeline reads."""
    defaults = {
        "HISTORICAL_MEMORY_ENABLED": True,
        "RERANKER_SCORE_THRESHOLD": -2.0,
        "HISTORICAL_MEMORY_TOP_K": 5,
        "HISTORICAL_MEMORY_SCORE_THRESHOLD": 0.3,
        "RERANKER_ENABLED": True,
        "OPENAI_API_BASE": "http://localhost:11434/v1",
        "OPENAI_API_KEY": "test-key",
        "OPENAI_MODEL": "llama3",
    }
    defaults.update(overrides)
    return MagicMock(**defaults)


def _make_retrieval_result(docs, failed_legs=None, graph_count=0, graph_expanded=0):
    """Create a retrieval_result dict matching hybrid_search_with_legs output."""
    return {
        "docs": docs,
        "retrieval_info": {
            "legs": {
                "dense": {"status": "ok", "count": max(len(docs) // 2, 0)},
                "qdrant_sparse": {"status": "ok", "count": max(len(docs) // 2, 0)},
                "exact": {"status": "disabled", "count": 0},
                "graph": {"count": graph_count, "expanded": graph_expanded},
            },
            "failed_legs": failed_legs or [],
        },
    }


def _make_confidence_result(score, level=None, suggestion=None):
    """Create a ConfidenceResult-like mock for score_retrieval."""
    result = MagicMock()
    result.score = score
    if level is None:
        if score >= 80:
            level = "very_high"
        elif score >= 55:
            level = "high"
        elif score >= 30:
            level = "medium"
        elif score > 0:
            level = "low"
        else:
            level = "none"
    result.level = level
    result.suggestion = suggestion or ""
    result.breakdown = {
        "top_reranker_score": 0.0,
        "mean_reranker_score": 0.0,
        "top_score_pts": 0.0,
        "evidence_count_pts": 0.0,
        "mean_score_pts": 0.0,
        "total": score,
        "docs_returned": 0,
        "failed_legs": [],
    }
    return result


async def _collect_events(gen):
    """Run an async generator and return all emitted events as a list."""
    events = []
    async for event in gen:
        events.append(event)
    return events


# ── Test 1: Standard retrieval — single context event (high confidence) ───────

@pytest.mark.asyncio
async def test_standard_retrieval_single_context_event():
    """
    Standard flow: confidence >= 55 → one context event emitted with standard docs.
    No adaptive expansion. No historical memory (no chat_id).
    """
    docs = [
        _make_kb_doc(3.0, source="doc1.txt"),
        _make_kb_doc(2.5, source="doc2.txt"),
        _make_kb_doc(1.0, source="doc3.txt"),
    ]
    retrieval_result = _make_retrieval_result(docs)
    conf_result = _make_confidence_result(65, level="high")
    mock_db = MagicMock()

    with patch("app.core.config.settings", _make_settings()):
        with patch("app.services.chat_service._rewrite_query", return_value="rewritten query"):
            with patch("app.services.retrieval.hybrid_search_with_legs", return_value=retrieval_result):
                with patch("app.services.confidence.score_retrieval", return_value=conf_result):
                    with patch("app.services.fast_pipeline._get_llm") as mock_llm_factory:
                        mock_llm = AsyncMock()
                        mock_llm.astream = AsyncMock(
                            return_value=iter([_make_mock_chunk("Test answer")])
                        )
                        mock_llm_factory.return_value = mock_llm

                        with patch("app.models.knowledge.KnowledgeBaseDataStore"):
                            from app.services.fast_pipeline import fast_stream

                            events = await _collect_events(
                                fast_stream(
                                    query="test query",
                                    knowledge_base_ids=[1],
                                    db=mock_db,
                                    recent_lc_history=[],
                                    existing_summary=None,
                                )
                            )

    context_events = [e for e in events if e.get("event") == "context"]
    assert len(context_events) == 1, f"Expected 1 context event, got {len(context_events)}"

    ctx = context_events[0]
    assert ctx["score"] == 0.65
    assert ctx["confidence"] == "high"
    assert ctx["docs"] is not None
    assert len(ctx["docs"]) == 3
    assert "adaptive" not in ctx["breakdown"]

    event_types = [e.get("event") for e in events]
    assert "rewritten_query" in event_types
    assert "done" in event_types


# ── Test 2: Adaptive retrieval — double context event (low confidence) ─────────

@pytest.mark.asyncio
async def test_adaptive_retrieval_double_context_event():
    """
    Low confidence (<55) → two context events: standard first, then adaptive
    with expanded docs. The adaptive event carries adaptive=True,
    expanded_from, expanded_to, threshold_used.
    """
    standard_docs = [
        _make_kb_doc(3.0, source="doc1.txt"),
        _make_kb_doc(2.5, source="doc2.txt"),
    ]
    full_pool = standard_docs + [
        _make_kb_doc(-3.0, source="doc3.txt"),
        _make_kb_doc(-3.5, source="doc4.txt"),
        _make_kb_doc(-4.0, source="doc5.txt"),
    ]
    retrieval_result = _make_retrieval_result(full_pool)
    conf_result = _make_confidence_result(40, level="medium")
    mock_db = MagicMock()

    with patch("app.core.config.settings", _make_settings()):
        with patch("app.services.chat_service._rewrite_query", return_value="rewritten query"):
            with patch("app.services.retrieval.hybrid_search_with_legs", return_value=retrieval_result):
                with patch("app.services.confidence.score_retrieval", return_value=conf_result):
                    with patch("app.services.fast_pipeline._get_llm") as mock_llm_factory:
                        mock_llm = AsyncMock()
                        mock_llm.astream = AsyncMock(
                            return_value=iter([_make_mock_chunk("Adaptive answer")])
                        )
                        mock_llm_factory.return_value = mock_llm

                        with patch("app.models.knowledge.KnowledgeBaseDataStore"):
                            from app.services.fast_pipeline import fast_stream

                            events = await _collect_events(
                                fast_stream(
                                    query="vague question",
                                    knowledge_base_ids=[1],
                                    db=mock_db,
                                    recent_lc_history=[],
                                    existing_summary=None,
                                )
                            )

    context_events = [e for e in events if e.get("event") == "context"]
    assert len(context_events) == 2, f"Expected 2 context events, got {len(context_events)}"

    # First event: standard docs (2 docs)
    first_ctx = context_events[0]
    assert len(first_ctx["docs"]) == 2
    assert "adaptive" not in first_ctx["breakdown"]

    # Second event: adaptive expanded docs (5 docs)
    second_ctx = context_events[1]
    assert second_ctx["breakdown"].get("adaptive") is True
    assert second_ctx["breakdown"].get("threshold_used") == -5.0
    assert second_ctx["breakdown"].get("expanded_from") == 2
    assert second_ctx["breakdown"].get("expanded_to") == 5
    assert len(second_ctx["docs"]) == 5


# ── Test 3: Historical memory prepending ──────────────────────────────────────

@pytest.mark.asyncio
async def test_historical_memory_prepend():
    """
    chat_id provided + HISTORICAL_MEMORY_ENABLED → historical docs retrieved
    and prepended to context string with [HIST-N] formatting.
    """
    kb_docs = [_make_kb_doc(3.0, source="kb.txt", content="Knowledge base content.")]
    retrieval_result = _make_retrieval_result(kb_docs)
    conf_result = _make_confidence_result(70, level="high")
    hist_docs = [
        _make_hist_doc("Last session: we discussed RAG architecture.", message_id=10),
        _make_hist_doc("You mentioned vector databases are important.", message_id=9),
    ]
    mock_db = MagicMock()

    with patch("app.core.config.settings", _make_settings()):
        with patch("app.services.chat_service._rewrite_query", return_value="rewritten query"):
            with patch("app.services.retrieval.hybrid_search_with_legs", return_value=retrieval_result):
                with patch("app.services.confidence.score_retrieval", return_value=conf_result):
                    with patch("app.services.fast_pipeline._get_llm") as mock_llm_factory:
                        mock_llm = AsyncMock()
                        mock_llm.astream = AsyncMock(
                            return_value=iter([_make_mock_chunk("Answer with history")])
                        )
                        mock_llm_factory.return_value = mock_llm

                        with patch("app.models.knowledge.KnowledgeBaseDataStore"):
                            with patch(
                                "app.services.historical_memory.retrieve_historical_memory",
                                return_value=hist_docs,
                            ) as mock_hist:
                                from app.services.fast_pipeline import fast_stream

                                events = await _collect_events(
                                    fast_stream(
                                        query="what did we talk about?",
                                        knowledge_base_ids=[1],
                                        db=mock_db,
                                        recent_lc_history=[],
                                        existing_summary=None,
                                        chat_id=42,
                                    )
                                )

    assert mock_hist.called
    assert mock_hist.call_args.kwargs["chat_id"] == 42

    context_events = [e for e in events if e.get("event") == "context"]
    assert len(context_events) >= 1

    first_ctx = context_events[0]
    assert len(first_ctx["docs"]) == 1


# ── Test 4: Historical memory skipped when no chat_id ─────────────────────────

@pytest.mark.asyncio
async def test_historical_memory_skipped_when_no_chat_id():
    """chat_id=None → retrieve_historical_memory is NOT called."""
    docs = [_make_kb_doc(3.0, source="doc.txt")]
    retrieval_result = _make_retrieval_result(docs)
    conf_result = _make_confidence_result(70, level="high")
    mock_db = MagicMock()

    with patch("app.core.config.settings", _make_settings()):
        with patch("app.services.chat_service._rewrite_query", return_value="rewritten"):
            with patch("app.services.retrieval.hybrid_search_with_legs", return_value=retrieval_result):
                with patch("app.services.confidence.score_retrieval", return_value=conf_result):
                    with patch("app.services.fast_pipeline._get_llm") as mock_llm_factory:
                        mock_llm = AsyncMock()
                        mock_llm.astream = AsyncMock(
                            return_value=iter([_make_mock_chunk("Answer")])
                        )
                        mock_llm_factory.return_value = mock_llm

                        with patch("app.models.knowledge.KnowledgeBaseDataStore"):
                            with patch(
                                "app.services.historical_memory.retrieve_historical_memory"
                            ) as mock_hist:
                                from app.services.fast_pipeline import fast_stream

                                events = await _collect_events(
                                    fast_stream(
                                        query="test",
                                        knowledge_base_ids=[1],
                                        db=mock_db,
                                        recent_lc_history=[],
                                        existing_summary=None,
                                        chat_id=None,
                                    )
                                )

    assert not mock_hist.called


# ── Test 5: Historical memory skipped when disabled ───────────────────────────

@pytest.mark.asyncio
async def test_historical_memory_skipped_when_disabled():
    """
    HISTORICAL_MEMORY_ENABLED=False → retrieve_historical_memory is NOT called
    even when chat_id is provided.
    """
    docs = [_make_kb_doc(3.0, source="doc.txt")]
    retrieval_result = _make_retrieval_result(docs)
    conf_result = _make_confidence_result(70, level="high")
    mock_db = MagicMock()

    with patch("app.core.config.settings", _make_settings(HISTORICAL_MEMORY_ENABLED=False)):
        with patch("app.services.chat_service._rewrite_query", return_value="rewritten"):
            with patch("app.services.retrieval.hybrid_search_with_legs", return_value=retrieval_result):
                with patch("app.services.confidence.score_retrieval", return_value=conf_result):
                    with patch("app.services.fast_pipeline._get_llm") as mock_llm_factory:
                        mock_llm = AsyncMock()
                        mock_llm.astream = AsyncMock(
                            return_value=iter([_make_mock_chunk("Answer")])
                        )
                        mock_llm_factory.return_value = mock_llm

                        with patch("app.models.knowledge.KnowledgeBaseDataStore"):
                            with patch(
                                "app.services.historical_memory.retrieve_historical_memory"
                            ) as mock_hist:
                                from app.services.fast_pipeline import fast_stream

                                events = await _collect_events(
                                    fast_stream(
                                        query="test",
                                        knowledge_base_ids=[1],
                                        db=mock_db,
                                        recent_lc_history=[],
                                        existing_summary=None,
                                        chat_id=42,
                                    )
                                )

    assert not mock_hist.called


# ── Test 6: Confidence boundary — score=55 exactly → single event ──────────────

@pytest.mark.asyncio
async def test_adaptive_confidence_boundary():
    """
    confidence score = 55 → single context event (not adaptive).
    The code gate is `conf_score < 55` — so 55 is NOT adaptive.
    """
    docs = [_make_kb_doc(3.0, source="doc1.txt"), _make_kb_doc(2.5, source="doc2.txt")]
    retrieval_result = _make_retrieval_result(docs)
    conf_result = _make_confidence_result(55, level="high")
    mock_db = MagicMock()

    with patch("app.core.config.settings", _make_settings()):
        with patch("app.services.chat_service._rewrite_query", return_value="rewritten"):
            with patch("app.services.retrieval.hybrid_search_with_legs", return_value=retrieval_result):
                with patch("app.services.confidence.score_retrieval", return_value=conf_result):
                    with patch("app.services.fast_pipeline._get_llm") as mock_llm_factory:
                        mock_llm = AsyncMock()
                        mock_llm.astream = AsyncMock(
                            return_value=iter([_make_mock_chunk("Boundary answer")])
                        )
                        mock_llm_factory.return_value = mock_llm

                        with patch("app.models.knowledge.KnowledgeBaseDataStore"):
                            from app.services.fast_pipeline import fast_stream

                            events = await _collect_events(
                                fast_stream(
                                    query="test boundary",
                                    knowledge_base_ids=[1],
                                    db=mock_db,
                                    recent_lc_history=[],
                                    existing_summary=None,
                                )
                            )

    context_events = [e for e in events if e.get("event") == "context"]
    assert len(context_events) == 1, f"Expected 1 context event at boundary, got {len(context_events)}"
    assert "adaptive" not in context_events[0]["breakdown"]


# ── Test 7: Adaptive no expansion when few docs ───────────────────────────────

@pytest.mark.asyncio
async def test_adaptive_no_expansion_when_few_docs():
    """
    Low confidence but all docs pass both thresholds → expanded_to == expanded_from.
    The adaptive event is still emitted but with the same doc count.
    """
    docs = [
        _make_kb_doc(3.0, source="doc1.txt"),
        _make_kb_doc(2.5, source="doc2.txt"),
        _make_kb_doc(2.0, source="doc3.txt"),
    ]
    # All pass standard (-2.0) and adaptive (-5.0) thresholds
    retrieval_result = _make_retrieval_result(docs)
    conf_result = _make_confidence_result(35, level="medium")
    mock_db = MagicMock()

    with patch("app.core.config.settings", _make_settings()):
        with patch("app.services.chat_service._rewrite_query", return_value="rewritten"):
            with patch("app.services.retrieval.hybrid_search_with_legs", return_value=retrieval_result):
                with patch("app.services.confidence.score_retrieval", return_value=conf_result):
                    with patch("app.services.fast_pipeline._get_llm") as mock_llm_factory:
                        mock_llm = AsyncMock()
                        mock_llm.astream = AsyncMock(
                            return_value=iter([_make_mock_chunk("Same-count answer")])
                        )
                        mock_llm_factory.return_value = mock_llm

                        with patch("app.models.knowledge.KnowledgeBaseDataStore"):
                            from app.services.fast_pipeline import fast_stream

                            events = await _collect_events(
                                fast_stream(
                                    query="test few docs",
                                    knowledge_base_ids=[1],
                                    db=mock_db,
                                    recent_lc_history=[],
                                    existing_summary=None,
                                )
                            )

    context_events = [e for e in events if e.get("event") == "context"]
    assert len(context_events) == 2, f"Expected 2 context events, got {len(context_events)}"

    first_ctx = context_events[0]
    assert len(first_ctx["docs"]) == 3

    second_ctx = context_events[1]
    assert second_ctx["breakdown"].get("adaptive") is True
    assert second_ctx["breakdown"].get("expanded_from") == 3
    assert second_ctx["breakdown"].get("expanded_to") == 3
    assert len(second_ctx["docs"]) == 3


# ── Test 8: Query rewrite failure — non-fatal fallback ─────────────────────────

@pytest.mark.asyncio
async def test_rewrite_fallback_to_original_query():
    """
    If _rewrite_query raises, the original query is used instead.
    The pipeline continues normally.
    """
    docs = [_make_kb_doc(3.0, source="doc.txt")]
    retrieval_result = _make_retrieval_result(docs)
    conf_result = _make_confidence_result(70, level="high")
    mock_db = MagicMock()

    with patch("app.core.config.settings", _make_settings()):
        with patch(
            "app.services.chat_service._rewrite_query",
            side_effect=Exception("LLM unavailable"),
        ):
            with patch("app.services.retrieval.hybrid_search_with_legs", return_value=retrieval_result):
                with patch("app.services.confidence.score_retrieval", return_value=conf_result):
                    with patch("app.services.fast_pipeline._get_llm") as mock_llm_factory:
                        mock_llm = AsyncMock()
                        mock_llm.astream = AsyncMock(
                            return_value=iter([_make_mock_chunk("Fallback answer")])
                        )
                        mock_llm_factory.return_value = mock_llm

                        with patch("app.models.knowledge.KnowledgeBaseDataStore"):
                            from app.services.fast_pipeline import fast_stream

                            events = await _collect_events(
                                fast_stream(
                                    query="original query",
                                    knowledge_base_ids=[1],
                                    db=mock_db,
                                    recent_lc_history=[],
                                    existing_summary=None,
                                )
                            )

    rewrite_events = [e for e in events if e.get("event") == "rewritten_query"]
    assert len(rewrite_events) == 1
    assert rewrite_events[0]["query"] == "original query"
    assert any(e.get("event") == "context" for e in events)
    assert any(e.get("event") == "done" for e in events)


# ── Test 9: Retrieval failure — pipeline continues with empty docs ─────────────

@pytest.mark.asyncio
async def test_retrieval_failure_continues_with_empty_docs():
    """
    If hybrid_search_with_legs raises, the pipeline continues with empty docs.
    """
    conf_result = _make_confidence_result(0, level="none")
    mock_db = MagicMock()

    with patch("app.core.config.settings", _make_settings()):
        with patch("app.services.chat_service._rewrite_query", return_value="rewritten"):
            with patch(
                "app.services.retrieval.hybrid_search_with_legs",
                side_effect=Exception("Connection timeout"),
            ):
                with patch("app.services.confidence.score_retrieval", return_value=conf_result):
                    with patch("app.services.fast_pipeline._get_llm") as mock_llm_factory:
                        mock_llm = AsyncMock()
                        mock_llm.astream = AsyncMock(
                            return_value=iter([_make_mock_chunk("Empty context answer")])
                        )
                        mock_llm_factory.return_value = mock_llm

                        with patch("app.models.knowledge.KnowledgeBaseDataStore"):
                            from app.services.fast_pipeline import fast_stream

                            events = await _collect_events(
                                fast_stream(
                                    query="test",
                                    knowledge_base_ids=[1],
                                    db=mock_db,
                                    recent_lc_history=[],
                                    existing_summary=None,
                                )
                            )

    context_events = [e for e in events if e.get("event") == "context"]
    assert len(context_events) >= 1
    assert any(e.get("event") == "done" for e in events)


# ── Test 10: LLM generation failure — error message yielded ───────────────────

@pytest.mark.asyncio
async def test_llm_generation_failure():
    """
    If llm.astream raises, an error message is yielded and the pipeline completes.
    """
    docs = [_make_kb_doc(3.0, source="doc.txt")]
    retrieval_result = _make_retrieval_result(docs)
    conf_result = _make_confidence_result(70, level="high")
    mock_db = MagicMock()

    with patch("app.core.config.settings", _make_settings()):
        with patch("app.services.chat_service._rewrite_query", return_value="rewritten"):
            with patch("app.services.retrieval.hybrid_search_with_legs", return_value=retrieval_result):
                with patch("app.services.confidence.score_retrieval", return_value=conf_result):
                    with patch("app.services.fast_pipeline._get_llm") as mock_llm_factory:
                        mock_llm = AsyncMock()
                        mock_llm.astream = AsyncMock(
                            side_effect=Exception("Model unavailable")
                        )
                        mock_llm_factory.return_value = mock_llm

                        with patch("app.models.knowledge.KnowledgeBaseDataStore"):
                            from app.services.fast_pipeline import fast_stream

                            events = await _collect_events(
                                fast_stream(
                                    query="test",
                                    knowledge_base_ids=[1],
                                    db=mock_db,
                                    recent_lc_history=[],
                                    existing_summary=None,
                                )
                            )

    error_events = [
        e for e in events
        if e.get("event") == "token"
        and "error in synthesizing" in e.get("content", "").lower()
    ]
    assert len(error_events) == 1
    assert any(e.get("event") == "done" for e in events)


# ── Test 11: Context string construction with file_markdown ────────────────────

@pytest.mark.asyncio
async def test_context_string_with_file_markdown():
    """When file_markdown is provided, it is appended to the context string."""
    kb_docs = [_make_kb_doc(3.0, source="kb.txt", content="KB content.")]
    retrieval_result = _make_retrieval_result(kb_docs)
    conf_result = _make_confidence_result(70, level="high")
    mock_db = MagicMock()

    with patch("app.core.config.settings", _make_settings()):
        with patch("app.services.chat_service._rewrite_query", return_value="rewritten"):
            with patch("app.services.retrieval.hybrid_search_with_legs", return_value=retrieval_result):
                with patch("app.services.confidence.score_retrieval", return_value=conf_result):
                    with patch("app.services.fast_pipeline._get_llm") as mock_llm_factory:
                        mock_llm = AsyncMock()
                        mock_llm.astream = AsyncMock(
                            return_value=iter([_make_mock_chunk("Answer")])
                        )
                        mock_llm_factory.return_value = mock_llm

                        with patch("app.models.knowledge.KnowledgeBaseDataStore"):
                            from app.services.fast_pipeline import fast_stream

                            file_content = "# Readme\nThis is the file content."
                            events = await _collect_events(
                                fast_stream(
                                    query="test",
                                    knowledge_base_ids=[1],
                                    db=mock_db,
                                    recent_lc_history=[],
                                    existing_summary=None,
                                    file_markdown=file_content,
                                )
                            )

    assert any(e.get("event") == "done" for e in events)


# ── Test 12: Historical memory retrieval exception — non-fatal fallback ────────

@pytest.mark.asyncio
async def test_historical_memory_retrieval_exception():
    """
    If retrieve_historical_memory raises, historical_docs defaults to [] and
    the pipeline continues normally.
    """
    kb_docs = [_make_kb_doc(3.0, source="doc.txt")]
    retrieval_result = _make_retrieval_result(kb_docs)
    conf_result = _make_confidence_result(70, level="high")
    mock_db = MagicMock()

    with patch("app.core.config.settings", _make_settings()):
        with patch("app.services.chat_service._rewrite_query", return_value="rewritten"):
            with patch("app.services.retrieval.hybrid_search_with_legs", return_value=retrieval_result):
                with patch("app.services.confidence.score_retrieval", return_value=conf_result):
                    with patch("app.services.fast_pipeline._get_llm") as mock_llm_factory:
                        mock_llm = AsyncMock()
                        mock_llm.astream = AsyncMock(
                            return_value=iter([_make_mock_chunk("Answer")])
                        )
                        mock_llm_factory.return_value = mock_llm

                        with patch("app.models.knowledge.KnowledgeBaseDataStore"):
                            with patch(
                                "app.services.historical_memory.retrieve_historical_memory",
                                side_effect=Exception("DB connection lost"),
                            ):
                                from app.services.fast_pipeline import fast_stream

                                events = await _collect_events(
                                    fast_stream(
                                        query="test",
                                        knowledge_base_ids=[1],
                                        db=mock_db,
                                        recent_lc_history=[],
                                        existing_summary=None,
                                        chat_id=99,
                                    )
                                )

    assert any(e.get("event") == "context" for e in events)
    assert any(e.get("event") == "done" for e in events)
