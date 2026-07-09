"""Tests for built-in tools: search_documents, extract_entities, summarize_chunks."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.services.tool_registry import _registry, execute_tool


# Ensure built-in tools are registered
import app.services.builtin_tools  # noqa: F401


# ── Registration tests ────────────────────────────────────────────────────────

class TestBuiltinToolRegistration:
    def test_search_documents_registered(self):
        assert _registry.get("search_documents") is not None

    def test_extract_entities_registered(self):
        assert _registry.get("extract_entities") is not None

    def test_summarize_chunks_registered(self):
        assert _registry.get("summarize_chunks") is not None

    def test_all_tools_in_list_tools(self):
        names = [t["function"]["name"] for t in _registry.list_tools()]
        assert "search_documents" in names
        assert "extract_entities" in names
        assert "summarize_chunks" in names

    def test_search_documents_has_required_params(self):
        tool = _registry.get("search_documents")
        required = tool.parameters.get("required", [])
        assert "query" in required
        assert "kb_ids" in required

    def test_extract_entities_has_required_params(self):
        tool = _registry.get("extract_entities")
        required = tool.parameters.get("required", [])
        assert "text" in required

    def test_summarize_chunks_has_required_params(self):
        tool = _registry.get("summarize_chunks")
        required = tool.parameters.get("required", [])
        assert "chunks" in required
        assert "instruction" in required


# ── Functional tests ──────────────────────────────────────────────────────────

class TestExtractEntitiesTool:
    def test_returns_entity_list(self):
        from app.services.graph.entity_extractor import Entity

        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps({
            "entities": [
                {"name": "Microsoft", "type": "ORG"},
                {"name": "Bill Gates", "type": "PERSON"},
            ]
        })

        with patch("app.services.graph.entity_extractor._get_llm_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = mock_resp
            result = execute_tool("extract_entities", {"text": "Bill Gates founded Microsoft."})

        assert result.success
        assert isinstance(result.output, list)
        assert len(result.output) == 2
        assert result.output[0]["name"] == "Microsoft"

    def test_returns_empty_list_for_no_entities(self):
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps({"entities": []})

        with patch("app.services.graph.entity_extractor._get_llm_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = mock_resp
            result = execute_tool("extract_entities", {"text": "what is the weather?"})

        assert result.success
        assert result.output == []

    def test_llm_failure_returns_empty_list(self):
        with patch("app.services.graph.entity_extractor._get_llm_client") as mock_client:
            mock_client.return_value.chat.completions.create.side_effect = Exception("timeout")
            result = execute_tool("extract_entities", {"text": "Apple acquired Beats."})

        # entity_extractor swallows LLM errors and returns [] — tool succeeds with empty list
        assert result.success
        assert result.output == []


class TestSummarizeChunksTool:
    def test_returns_summary(self):
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "Key theme: AI acquisition strategy."

        with patch("openai.OpenAI") as mock_oai:
            mock_oai.return_value.chat.completions.create.return_value = mock_resp
            result = execute_tool("summarize_chunks", {
                "chunks": ["Apple acquired Beats for $3B.", "Google acquired DeepMind."],
                "instruction": "Summarize key themes",
            })

        assert result.success
        assert result.output["summary"] == "Key theme: AI acquisition strategy."
        assert result.output["chunk_count"] == 2

    def test_empty_chunks_returns_empty_summary(self):
        result = execute_tool("summarize_chunks", {"chunks": [], "instruction": "Summarize"})
        assert result.success
        assert result.output == {"summary": "", "chunk_count": 0}
