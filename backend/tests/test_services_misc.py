"""Unit tests for small, side-effect-free service modules."""
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document as LangchainDocument


# ── Reasoning tags ───────────────────────────────────────────────────────────


def test_strip_reasoning_tags_html_style():
    from app.services.infrastructure.reasoning_tags import strip_reasoning_tags

    text = "prefix<reasoning>hidden</reasoning>suffix"
    assert strip_reasoning_tags(text) == "prefixsuffix"


def test_strip_reasoning_tags_think_tag():
    from app.services.infrastructure.reasoning_tags import strip_reasoning_tags

    text = "pre<think>hidden</think>post"
    assert strip_reasoning_tags(text) == "prepost"


def test_strip_reasoning_tags_channel_style():
    from app.services.infrastructure.reasoning_tags import strip_reasoning_tags

    text = "pre<|channel>thought hidden thought <channel|>post"
    assert strip_reasoning_tags(text) == "prepost"


def test_strip_reasoning_tags_partial_block():
    from app.services.infrastructure.reasoning_tags import strip_reasoning_tags

    assert strip_reasoning_tags("prefix<reasoning>incomplete") == "prefix"


# ── ProgressTimeout ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_progress_timeout_fires_on_inactivity():
    from app.services.infrastructure.progress_timeout import ProgressTimeout

    callback = MagicMock()
    async with ProgressTimeout(silence_seconds=1, on_timeout=callback) as pt:
        await asyncio.sleep(1.5)
        assert pt._watcher is not None
    assert callback.called


@pytest.mark.asyncio
async def test_progress_timeout_resets_with_ping():
    from app.services.infrastructure.progress_timeout import ProgressTimeout

    callback = MagicMock()
    async with ProgressTimeout(silence_seconds=1, on_timeout=callback) as pt:
        await asyncio.sleep(0.5)
        pt.ping()
        await asyncio.sleep(0.5)
        pt.ping()
    assert not callback.called


# ── Retrieval confidence ─────────────────────────────────────────────────────


def test_score_retrieval_empty_docs():
    from app.services.retrieval.confidence import score_retrieval

    result = score_retrieval([], {"legs": {"dense": {"status": "ok"}}, "failed_legs": []})
    assert result.level == "none"
    assert result.score == 0
    assert result.breakdown["total"] == 0


def test_score_retrieval_very_high():
    from app.services.retrieval.confidence import score_retrieval

    docs = [
        LangchainDocument(page_content="x", metadata={"_reranker_score": 10.0}),
        LangchainDocument(page_content="x", metadata={"_reranker_score": 10.0}),
    ]
    result = score_retrieval(docs, {"legs": {"dense": {"status": "ok"}}, "failed_legs": []})
    assert result.score >= 80
    assert result.level == "very_high"
    assert result.breakdown["docs_returned"] == 2


def test_score_retrieval_with_failed_legs_suggestion():
    from app.services.retrieval.confidence import score_retrieval

    docs = [
        LangchainDocument(page_content="x", metadata={"_reranker_score": 10.0}),
    ]
    info = {"legs": {"dense": {"status": "ok"}, "sparse": {"status": "failed"}}, "failed_legs": ["sparse"]}
    result = score_retrieval(docs, info)
    assert "sparse" in result.suggestion


def test_score_retrieval_normalises_dict_documents():
    from app.services.retrieval.confidence import score_retrieval

    docs = [{"page_content": "x", "metadata": {"_reranker_score": 10.0}}]
    result = score_retrieval(docs, {"legs": {}, "failed_legs": []})
    assert result.score >= 80


# ── Prompt loader ───────────────────────────────────────────────────────────


def test_load_chart_instructions_returns_string_or_empty():
    from app.services.prompts.loader import load_chart_instructions

    result = load_chart_instructions()
    assert isinstance(result, str)


def test_append_chart_instructions_idempotent():
    from app.services.prompts.loader import append_chart_instructions

    prompt = "System prompt"
    first = append_chart_instructions(prompt)
    second = append_chart_instructions(first)
    assert first == second
    assert "System prompt" in second
