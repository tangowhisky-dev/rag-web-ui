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
    from app.services.infrastructure.progress_timeout import ProgressTimeout, ProgressTimeoutError

    callback = MagicMock()
    with pytest.raises(ProgressTimeoutError):
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


def test_append_chart_placeholder_instructions_single_chart():
    from app.services.prompts.loader import append_chart_placeholder_instructions

    result = append_chart_placeholder_instructions("System prompt", 1)
    assert "[[CHART_1]]" in result
    assert "System prompt" in result


def test_append_chart_placeholder_instructions_multiple_charts():
    from app.services.prompts.loader import append_chart_placeholder_instructions

    result = append_chart_placeholder_instructions("System prompt", 3)
    assert "[[CHART_1]]" in result
    assert "[[CHART_2]]" in result
    assert "[[CHART_3]]" in result


def test_append_chart_placeholder_instructions_noop_when_zero_charts():
    from app.services.prompts.loader import append_chart_placeholder_instructions

    result = append_chart_placeholder_instructions("System prompt", 0)
    assert result == "System prompt"


# ── Citation normalization ───────────────────────────────────────────────────


def test_normalize_citations_plain_numeric():
    from app.services.agentic_rag.utils import normalize_citations

    docs = [{"page_content": "a"}, {"page_content": "b"}]
    answer = "First point [1](1). Second point [2](2)."
    normalized, cited = normalize_citations(answer, docs)
    assert normalized == "First point [1](1). Second point [2](2)."
    assert cited == [1, 2]


def test_normalize_citations_kb_label_both_sides():
    # The model sometimes cites using the full "KB-N" label instead of the
    # bare numeral it was instructed to use, e.g. in chart-generating turns.
    from app.services.agentic_rag.utils import normalize_citations

    docs = [{"page_content": "a"}, {"page_content": "b"}]
    answer = "Revenue grew [KB-2](KB-2) last quarter."
    normalized, cited = normalize_citations(answer, docs)
    assert normalized == "Revenue grew [1](1) last quarter."
    assert cited == [2]


def test_normalize_citations_kb_label_mixed_sides():
    from app.services.agentic_rag.utils import normalize_citations

    docs = [{"page_content": "a"}, {"page_content": "b"}]
    answer = "See [KB-2](2) and also [1](KB-1)."
    normalized, cited = normalize_citations(answer, docs)
    assert normalized == "See [1](1) and also [2](2)."
    assert cited == [2, 1]


def test_normalize_citations_double_digit_indices():
    from app.services.agentic_rag.utils import normalize_citations

    docs = [{"page_content": f"doc{i}"} for i in range(1, 13)]
    answer = "Cited [10](10), [11](11), and [12](12)."
    normalized, cited = normalize_citations(answer, docs)
    assert normalized == "Cited [1](1), [2](2), and [3](3)."
    assert cited == [10, 11, 12]
