"""
Tests for entity-aware retrieval: extraction, expansion, boost, and integration.
"""

import json
from dataclasses import dataclass
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.services.graph.entity_extractor import (
    Entity,
    EntityNeighbor,
    apply_entity_boost,
    expand_query_entities,
    extract_entities_from_query,
    extract_expand_boost,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _doc(text: str, score: float = 0.8) -> Document:
    return Document(page_content=text, metadata={"score": score})


def _llm_resp(entities: list) -> MagicMock:
    """Build a mock OpenAI response with a JSON entity list."""
    msg = MagicMock()
    msg.content = json.dumps({"entities": entities})
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _mock_get_def(**overrides) -> MagicMock:
    """Mock for settings_registry.get_def returning SettingDef-like objects with .default."""
    def _side_effect(key):
        m = MagicMock()
        m.default = overrides.get(key)
        return m
    return MagicMock(side_effect=_side_effect)


# ── T01: LLM entity extraction ────────────────────────────────────────────────

class TestExtractEntitiesFromQuery:
    def test_returns_entities_from_llm(self):
        mock_resp = _llm_resp([
            {"name": "Apple", "type": "ORG"},
            {"name": "Tim Cook", "type": "PERSON"},
        ])
        with patch("app.services.graph.entity_extractor._get_llm_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = mock_resp
            result = extract_entities_from_query("What did Apple CEO Tim Cook announce?")

        assert len(result) == 2
        assert result[0].name == "Apple"
        assert result[0].type == "ORG"
        assert result[1].name == "Tim Cook"
        assert result[1].type == "PERSON"

    def test_deduplicates_by_name(self):
        mock_resp = _llm_resp([
            {"name": "Apple", "type": "ORG"},
            {"name": "Apple", "type": "ORG"},  # duplicate
        ])
        with patch("app.services.graph.entity_extractor._get_llm_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = mock_resp
            result = extract_entities_from_query("Apple Apple")

        assert len(result) == 1

    def test_empty_query_returns_empty(self):
        result = extract_entities_from_query("")
        assert result == []

    def test_whitespace_query_returns_empty(self):
        result = extract_entities_from_query("   ")
        assert result == []

    def test_llm_failure_returns_empty(self):
        with patch("app.services.graph.entity_extractor._get_llm_client") as mock_client:
            mock_client.return_value.chat.completions.create.side_effect = Exception("LLM error")
            result = extract_entities_from_query("What did Apple acquire?")

        assert result == []

    def test_empty_entity_list_from_llm(self):
        mock_resp = _llm_resp([])
        with patch("app.services.graph.entity_extractor._get_llm_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = mock_resp
            result = extract_entities_from_query("What is the weather today?")

        assert result == []

    def test_filters_short_entity_names(self):
        mock_resp = _llm_resp([
            {"name": "A", "type": "ORG"},   # too short
            {"name": "Google", "type": "ORG"},
        ])
        with patch("app.services.graph.entity_extractor._get_llm_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = mock_resp
            result = extract_entities_from_query("A Google query")

        assert len(result) == 1
        assert result[0].name == "Google"


# ── T02: Neo4j entity expansion ───────────────────────────────────────────────

class TestExpandQueryEntities:
    def _mock_neo4j(self, records: list) -> MagicMock:
        """Build a mock Neo4j session that returns the given records."""
        mock_record_list = []
        for r in records:
            rec = MagicMock()
            rec.get = lambda key, default=None, _r=r: _r.get(key, default)
            mock_record_list.append(rec)

        mock_result = MagicMock()
        mock_result.__iter__ = lambda self: iter(mock_record_list)

        mock_session = MagicMock()
        mock_session.__enter__ = lambda s: s
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.run.return_value = mock_result

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session
        return mock_driver

    def test_returns_neighbors(self):
        entities = [Entity(name="Apple", type="ORG")]
        driver = self._mock_neo4j([
            {"name": "Beats Electronics", "type": "ORG", "rel": "ACQUIRED"},
            {"name": "Tim Cook", "type": "PERSON", "rel": "LED_BY"},
        ])

        with patch("app.services.graph.entity_extractor._get_neo4j_driver", return_value=driver):
            with patch("app.core.settings_registry.get_def", new=_mock_get_def(GRAPHRAG_ENABLED=True)):
                result = expand_query_entities(entities, [1])

        assert len(result) == 2
        assert result[0].name == "Beats Electronics"
        assert result[0].relation == "ACQUIRED"

    def test_empty_entities_returns_empty(self):
        result = expand_query_entities([], [1])
        assert result == []

    def test_graphrag_disabled_returns_empty(self):
        entities = [Entity(name="Apple", type="ORG")]
        with patch("app.core.settings_registry.get_def", new=_mock_get_def(GRAPHRAG_ENABLED=False)):
            result = expand_query_entities(entities, [1])
        assert result == []

    def test_neo4j_failure_returns_empty(self):
        entities = [Entity(name="Apple", type="ORG")]
        with patch("app.services.graph.entity_extractor._get_neo4j_driver") as mock_driver_fn:
            mock_driver_fn.return_value.session.side_effect = Exception("connection refused")
            with patch("app.core.settings_registry.get_def", new=_mock_get_def(GRAPHRAG_ENABLED=True)):
                result = expand_query_entities(entities, [1])
        assert result == []

    def test_deduplicates_neighbors(self):
        entities = [Entity(name="Apple", type="ORG")]
        driver = self._mock_neo4j([
            {"name": "Beats", "type": "ORG", "rel": "ACQUIRED"},
            {"name": "Beats", "type": "ORG", "rel": "ACQUIRED"},  # duplicate
        ])

        with patch("app.services.graph.entity_extractor._get_neo4j_driver", return_value=driver):
            with patch("app.core.settings_registry.get_def", new=_mock_get_def(GRAPHRAG_ENABLED=True)):
                result = expand_query_entities(entities, [1])

        assert len(result) == 1


# ── T03: Entity boost scoring ─────────────────────────────────────────────────

class TestApplyEntityBoost:
    def test_boosts_matching_doc(self):
        docs = [_doc("Apple acquired Beats in 2014.", score=0.8)]
        entities = [Entity(name="Apple", type="ORG")]

        with patch("app.core.settings_registry.get_def", new=_mock_get_def(ENTITY_BOOST_FACTOR=0.1)):
            result = apply_entity_boost(docs, entities)

        assert result[0].metadata["score"] > 0.8
        assert "entity_matches" in result[0].metadata

    def test_no_boost_without_mention(self):
        docs = [_doc("Google released a new product.", score=0.8)]
        entities = [Entity(name="Apple", type="ORG")]

        with patch("app.core.settings_registry.get_def", new=_mock_get_def(ENTITY_BOOST_FACTOR=0.1)):
            result = apply_entity_boost(docs, entities)

        assert result[0].metadata.get("score") == 0.8
        assert "entity_matches" not in result[0].metadata

    def test_empty_entities_returns_docs_unchanged(self):
        docs = [_doc("Some text.", score=0.5)]
        result = apply_entity_boost(docs, [])
        assert result[0].metadata["score"] == 0.5

    def test_neighbor_boost_half_weight(self):
        docs = [_doc("Beats Electronics is a great company.", score=1.0)]
        entities = [Entity(name="Apple", type="ORG")]  # not in text
        neighbors = [EntityNeighbor(name="Beats Electronics", type="ORG", relation="ACQUIRED")]

        with patch("app.core.settings_registry.get_def", new=_mock_get_def(ENTITY_BOOST_FACTOR=0.1)):
            result = apply_entity_boost(docs, entities, neighbors)

        # "Beats Electronics" appears once → count = int(1 * 0.5) = 0 → no boost
        # (half-weight rounds down for single hits)
        assert result[0].metadata["score"] == pytest.approx(1.0, rel=1e-3)

    def test_multiple_entity_mentions_cumulative(self):
        docs = [_doc("Apple Apple Apple.", score=1.0)]
        entities = [Entity(name="Apple", type="ORG")]

        with patch("app.core.settings_registry.get_def", new=_mock_get_def(ENTITY_BOOST_FACTOR=0.1)):
            result = apply_entity_boost(docs, entities)

        # 3 mentions × 0.1 factor → multiplier 1.3 → score = 1.3
        assert result[0].metadata["score"] == pytest.approx(1.3, rel=1e-3)

    def test_entity_matches_metadata_stored(self):
        docs = [_doc("Apple and Google compete.", score=0.5)]
        entities = [Entity(name="Apple", type="ORG"), Entity(name="Google", type="ORG")]

        with patch("app.core.settings_registry.get_def", new=_mock_get_def(ENTITY_BOOST_FACTOR=0.1)):
            result = apply_entity_boost(docs, entities)

        matches = {m["entity"]: m["count"] for m in result[0].metadata["entity_matches"]}
        assert "Apple" in matches
        assert "Google" in matches


# ── Integration: full pipeline ────────────────────────────────────────────────

class TestExtractExpandBoost:
    def test_full_pipeline(self):
        docs = [
            _doc("Apple acquired Beats in 2014.", score=0.8),
            _doc("Samsung released a new phone.", score=0.6),
        ]

        mock_resp = _llm_resp([{"name": "Apple", "type": "ORG"}])

        with patch("app.services.graph.entity_extractor._get_llm_client") as mock_llm, \
             patch("app.services.graph.entity_extractor._get_neo4j_driver") as mock_neo4j, \
             patch("app.core.settings_registry.get_def", new=_mock_get_def(GRAPHRAG_ENABLED=False, ENTITY_BOOST_FACTOR=0.1)):

            mock_llm.return_value.chat.completions.create.return_value = mock_resp

            result = extract_expand_boost("What did Apple acquire?", docs, [1])

        # Apple-mentioning doc should have higher score
        apple_doc = next(d for d in result if "Apple" in d.page_content)
        samsung_doc = next(d for d in result if "Samsung" in d.page_content)
        assert apple_doc.metadata["score"] > samsung_doc.metadata["score"]

    def test_returns_unchanged_on_no_entities(self):
        docs = [_doc("Some text.", score=0.5)]

        mock_resp = _llm_resp([])
        with patch("app.services.graph.entity_extractor._get_llm_client") as mock_llm, \
             patch("app.services.graph.entity_extractor.settings") as mock_settings:

            mock_llm.return_value.chat.completions.create.return_value = mock_resp

            result = extract_expand_boost("general query", docs, [1])

        assert result[0].metadata["score"] == 0.5
