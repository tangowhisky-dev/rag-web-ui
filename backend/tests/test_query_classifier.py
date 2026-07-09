"""
Test suite for LLM-based query classification.

Covers 20 queries across 4 types (FACTUAL, ENTITY_CENTRIC, MULTI_PART, AMBIGUOUS)
plus latency benchmarks and fallback behavior.
"""
import asyncio
import statistics
import time
import pytest
from unittest.mock import patch, AsyncMock

from app.services.chat import classify_query
from app.models.query_classifier import QueryType, QueryClassification
from app.core.config import settings


# ── Test queries grouped by expected type ─────────────────────────────────────

FACTUAL_QUERIES = [
    ("What is RRF?", QueryType.FACTUAL),
    ("Define BM25 scoring", QueryType.FACTUAL),
    ("What does SPLADE stand for?", QueryType.FACTUAL),
    ("Explain cosine similarity", QueryType.FACTUAL),
    ("What is TF-IDF?", QueryType.FACTUAL),
]

ENTITY_CENTRIC_QUERIES = [
    ("What did Apple acquire?", QueryType.ENTITY_CENTRIC),
    ("Who founded Microsoft?", QueryType.ENTITY_CENTRIC),
    ("What companies does Amazon own?", QueryType.ENTITY_CENTRIC),
    ("What products does Google make?", QueryType.ENTITY_CENTRIC),
    ("What is Tesla known for?", QueryType.ENTITY_CENTRIC),
]

MULTI_PART_QUERIES = [
    ("Compare RRF and BM25", QueryType.MULTI_PART),
    ("Differences between dense and sparse retrieval", QueryType.MULTI_PART),
    ("Pros and cons of vector databases", QueryType.MULTI_PART),
    ("Contrast Qdrant and Pinecone", QueryType.MULTI_PART),
    ("Advantages of hybrid search", QueryType.MULTI_PART),
]

AMBIGUOUS_QUERIES = [
    ("Tell me about that", QueryType.AMBIGUOUS),
    ("What do you think?", QueryType.AMBIGUOUS),
    ("More info please", QueryType.AMBIGUOUS),
    ("Can you elaborate?", QueryType.AMBIGUOUS),
    ("What else?", QueryType.AMBIGUOUS),
]

ALL_QUERIES = (
    FACTUAL_QUERIES
    + ENTITY_CENTRIC_QUERIES
    + MULTI_PART_QUERIES
    + AMBIGUOUS_QUERIES
)


# ── Mock-based tests (no LLM required) ────────────────────────────────────────

def _mock_classify_response(expected_type: str) -> AsyncMock:
    """Create a mock LLM response that returns the expected type."""
    mock_choice = AsyncMock()
    mock_choice.message.content = expected_type
    mock_response = AsyncMock()
    mock_response.choices = [mock_choice]
    return mock_response


@pytest.mark.parametrize("query,expected_type", ALL_QUERIES)
def test_classify_query_mocked(query, expected_type):
    """Test classification with mocked LLM responses."""
    async def _run():
        mock_response = _mock_classify_response(expected_type.value)
        
        with patch('app.services.chat.chat_service.AsyncOpenAI') as MockClient:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client
            
            result = await classify_query(query)
        
        assert result.type == expected_type, f"Expected {expected_type}, got {result.type} for query: {query}"
        assert result.confidence == 1.0, f"Expected confidence 1.0 for exact match"
        assert result.latency_ms > 0, "Latency should be measured"
        assert result.fallback is False, "Should not be fallback"
    
    asyncio.run(_run())


@pytest.mark.parametrize("query,expected_type", ALL_QUERIES)
def test_classify_query_fuzzy_match(query, expected_type):
    """Test fuzzy matching when LLM returns extra text."""
    async def _run():
        # Simulate LLM returning extra text around the type
        mock_response = _mock_classify_response(f"The category is {expected_type.value} based on the query.")
        
        with patch('app.services.chat.chat_service.AsyncOpenAI') as MockClient:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client
            
            result = await classify_query(query)
        
        assert result.type == expected_type
        assert result.confidence == 0.5, "Fuzzy match should have 0.5 confidence"
    
    asyncio.run(_run())


def test_classify_query_disabled():
    """Test that disabled classifier returns fallback."""
    async def _run():
        original = settings.QUERY_CLASSIFIER_ENABLED
        settings.QUERY_CLASSIFIER_ENABLED = False
        
        result = await classify_query("What is the capital of France?")
        
        settings.QUERY_CLASSIFIER_ENABLED = original
        
        assert result.type == QueryType.FACTUAL
        assert result.fallback is True
        assert result.confidence == 0.0
    
    asyncio.run(_run())


def test_classify_query_llm_error_fallback():
    """Test fallback behavior when LLM raises an exception."""
    async def _run():
        with patch('app.services.chat.chat_service.AsyncOpenAI') as MockClient:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(side_effect=Exception("Connection error"))
            MockClient.return_value = mock_client
            
            result = await classify_query("test query")
        
        assert result.type == QueryType.FACTUAL
        assert result.fallback is True
        assert result.confidence == 0.0
        assert result.latency_ms > 0
    
    asyncio.run(_run())


def test_classify_query_empty_response_fallback():
    """Test fallback when LLM returns empty response."""
    async def _run():
        mock_choice = AsyncMock()
        mock_choice.message.content = ""
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        
        with patch('app.services.chat.chat_service.AsyncOpenAI') as MockClient:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client
            
            result = await classify_query("test query")
        
        # Empty response doesn't match any enum — falls back to FACTUAL with 0 confidence
        assert result.type == QueryType.FACTUAL
        assert result.confidence == 0.0
    
    asyncio.run(_run())


# ── Latency benchmark (mocked) ────────────────────────────────────────────────

def test_classification_latency_benchmark():
    """Benchmark classification latency with mocked LLM — should be near-instant."""
    async def _run():
        latencies = []
        
        for i in range(100):
            mock_response = _mock_classify_response(QueryType.FACTUAL.value)
            
            with patch('app.services.chat.chat_service.AsyncOpenAI') as MockClient:
                mock_client = AsyncMock()
                mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
                MockClient.return_value = mock_client
                
                result = await classify_query(f"test query {i}")
                latencies.append(result.latency_ms)
        
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        median = statistics.median(latencies)
        
        # With mocked LLM, latency should be minimal (mock overhead only)
        assert p95 < 100, f"p95 latency {p95}ms exceeds 100ms target"
        assert median < 50, f"median latency {median}ms exceeds 50ms target"
        print(f"Latency benchmark: p95={p95:.1f}ms median={median:.1f}ms (mocked LLM)")
    
    asyncio.run(_run())


# ── Retrieval config tests ────────────────────────────────────────────────────

def test_retrieval_config_presets():
    """Test that retrieval config presets are correctly structured."""
    from app.services.retrieval import get_retrieval_config
    
    # FACTUAL: all legs enabled, balanced weights
    cfg = get_retrieval_config(QueryType.FACTUAL)
    assert cfg["use_dense"] is True
    assert cfg["use_sparse"] is True
    assert cfg["use_exact"] is True
    assert cfg["dense_weight"] == 0.5
    assert cfg["sparse_weight"] == 0.3
    assert cfg["exact_weight"] == 0.2
    
    # ENTITY_CENTRIC: dense emphasis
    cfg = get_retrieval_config(QueryType.ENTITY_CENTRIC)
    assert cfg["dense_weight"] == 0.6
    assert cfg["sparse_weight"] == 0.2
    
    # MULTI_PART: dense + sparse, no exact
    cfg = get_retrieval_config(QueryType.MULTI_PART)
    assert cfg["use_exact"] is False
    assert cfg["dense_weight"] == 0.5
    assert cfg["sparse_weight"] == 0.5
    
    # AMBIGUOUS: conservative top_k
    cfg = get_retrieval_config(QueryType.AMBIGUOUS)
    assert cfg["top_k"] == 15


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
