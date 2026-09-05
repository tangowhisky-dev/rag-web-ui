"""Tests for the new CitationRef schema and evidence-based citation pipeline."""

import re
from unittest.mock import MagicMock

import pytest

from app.services.agentic_rag.schemas import CitationRef, LastAnswerObject
from app.services.agentic_rag.utils import format_context_string, normalize_evidence_citations


class TestCitationRefSchema:
    def test_chunk_citation(self):
        ref = CitationRef(document_id=1, citation_kind="chunk", chunk_index=5, page=3)
        assert ref.document_id == 1
        assert ref.citation_kind == "chunk"
        assert ref.chunk_index == 5
        assert ref.page == 3
        assert ref.citation_id == ""

    def test_file_citation(self):
        ref = CitationRef(document_id=2, citation_kind="file")
        assert ref.citation_kind == "file"
        assert ref.chunk_index is None

    def test_section_citation(self):
        ref = CitationRef(document_id=3, citation_kind="section", section="Introduction",
                          start_char=100, end_char=500)
        assert ref.citation_kind == "section"
        assert ref.section == "Introduction"
        assert ref.start_char == 100
        assert ref.end_char == 500

    def test_range_citation(self):
        ref = CitationRef(document_id=4, citation_kind="range",
                          start_char=200, end_char=300,
                          start_line=10, end_line=15)
        assert ref.citation_kind == "range"
        assert ref.start_line == 10
        assert ref.end_line == 15

    def test_grep_citation(self):
        ref = CitationRef(document_id=5, citation_kind="grep",
                          match_line=42, quoted_text="some text")
        assert ref.citation_kind == "grep"
        assert ref.match_line == 42

    def test_outline_citation(self):
        ref = CitationRef(document_id=6, citation_kind="outline")
        assert ref.citation_kind == "outline"

    def test_table_citation(self):
        ref = CitationRef(document_id=7, citation_kind="table", section="Results")
        assert ref.citation_kind == "table"

    def test_coerce_kb_label_document_id(self):
        ref = CitationRef.model_validate({"document_id": "KB-2", "citation_kind": "chunk"})
        assert ref.document_id == 2

    def test_default_kind_is_chunk(self):
        ref = CitationRef(document_id=1)
        assert ref.citation_kind == "chunk"

    def test_invalid_kind_rejected(self):
        with pytest.raises(Exception):
            CitationRef(document_id=1, citation_kind="invalid_kind")


class TestFormatContextString:
    def test_evidence_based_labeling(self):
        """Docs with citation_ref should use [E1], [E2] labels."""
        docs = [
            {
                "page_content": "Content about topic A",
                "metadata": {
                    "document_id": 1,
                    "title": "Doc A",
                    "citation_ref": {
                        "document_id": 1,
                        "citation_kind": "chunk",
                        "chunk_index": 0,
                        "source_tool": "search_dense",
                    },
                },
            },
            {
                "page_content": "Content about topic B",
                "metadata": {
                    "document_id": 2,
                    "title": "Doc B",
                    "citation_ref": {
                        "document_id": 2,
                        "citation_kind": "section",
                        "section": "Overview",
                        "source_tool": "kb_read",
                    },
                },
            },
        ]
        result = format_context_string(docs)
        assert "[E1]" in result
        assert "[E2]" in result
        assert "kind=chunk" in result
        assert "kind=section" in result
        assert "source=search_dense" in result
        assert "section=Overview" in result

    def test_legacy_labeling_without_citation_ref(self):
        """Docs without citation_ref should use [KB-N] labels."""
        docs = [
            {
                "page_content": "Content about topic A",
                "metadata": {
                    "document_id": 1,
                    "title": "Doc A",
                    "source": "doc_a.pdf",
                },
            },
        ]
        result = format_context_string(docs)
        assert "[KB-1]" in result
        assert "[E1]" not in result

    def test_file_markdown_appended(self):
        docs = []
        result = format_context_string(docs, file_markdown="# File Content")
        assert "[File Content]" in result
        assert "# File Content" in result


class TestNormalizeEvidenceCitations:
    def test_basic_renumbering(self):
        evidence = [
            {"page_content": "A", "metadata": {"citation_ref": {"citation_id": "E1"}}},
            {"page_content": "B", "metadata": {"citation_ref": {"citation_id": "E2"}}},
            {"page_content": "C", "metadata": {"citation_ref": {"citation_id": "E3"}}},
        ]
        answer = "According to [E2] and [E1], the result is clear."
        normalized, cited = normalize_evidence_citations(answer, evidence)
        assert "[1]" in normalized  # E2 → [1] (first cited)
        assert "[2]" in normalized  # E1 → [2] (second cited)
        assert len(cited) == 2
        assert cited[0] == evidence[1]  # E2
        assert cited[1] == evidence[0]  # E1

    def test_out_of_range_stripped(self):
        evidence = [
            {"page_content": "A", "metadata": {"citation_ref": {"citation_id": "E1"}}},
        ]
        answer = "See [E1] and [E99]."
        normalized, cited = normalize_evidence_citations(answer, evidence)
        assert "[1]" in normalized
        assert "[E99]" not in normalized
        assert len(cited) == 1

    def test_no_evidence_strips_all(self):
        answer = "See [E1] and [E2]."
        normalized, cited = normalize_evidence_citations(answer, [])
        assert "[E1]" not in normalized
        assert "[E2]" not in normalized
        assert cited == []

    def test_empty_answer(self):
        normalized, cited = normalize_evidence_citations("", [{"x": 1}])
        assert normalized == ""
        assert cited == []

    def test_code_blocks_protected(self):
        evidence = [
            {"page_content": "A", "metadata": {"citation_ref": {"citation_id": "E1"}}},
        ]
        answer = "See [E1]. Code: `array[1] = 0`"
        normalized, cited = normalize_evidence_citations(answer, evidence)
        # [1] inside code should NOT be touched (it's not [E1])
        assert "array[1] = 0" in normalized
        # [E1] outside code should be renumbered
        assert "[1]" in normalized

    def test_deduplicates_repeated_citations(self):
        evidence = [
            {"page_content": "A", "metadata": {"citation_ref": {"citation_id": "E1"}}},
            {"page_content": "B", "metadata": {"citation_ref": {"citation_id": "E2"}}},
        ]
        answer = "[E1] says X. [E1] also says Y. [E2] says Z."
        normalized, cited = normalize_evidence_citations(answer, evidence)
        # E1 → [1], E2 → [2], each appears twice but maps to same number
        assert normalized.count("[1]") == 2
        assert normalized.count("[2]") == 1
        assert len(cited) == 2


class TestLastAnswerObjectCitations:
    def test_citations_use_new_schema(self):
        ref = CitationRef(document_id=1, citation_kind="chunk", chunk_index=3)
        lao = LastAnswerObject(summary="test", citations=[ref])
        assert len(lao.citations) == 1
        assert lao.citations[0].citation_kind == "chunk"

    def test_citations_default_empty(self):
        lao = LastAnswerObject(summary="test")
        assert lao.citations == []
