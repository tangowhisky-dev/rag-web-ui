"""
Tests for multi-document synthesis: synthesize_documents tool, synthesis prompt
detection, and report generation.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

# Ensure tools registered
import app.services.builtin_tools  # noqa: F401
from app.services.tool_registry import _registry, execute_tool
from app.services.export import generate_synthesis_report


# ── synthesize_documents registration ────────────────────────────────────────

class TestSynthesizeDocumentsTool:
    def test_registered(self):
        assert _registry.get("synthesize_documents") is not None

    def test_in_list_tools(self):
        names = [t["function"]["name"] for t in _registry.list_tools()]
        assert "synthesize_documents" in names

    def test_required_params(self):
        tool = _registry.get("synthesize_documents")
        required = tool.parameters.get("required", [])
        assert "topic" in required
        assert "sub_queries" in required
        assert "kb_ids" in required

    def test_empty_sub_queries_returns_empty(self):
        result = execute_tool("synthesize_documents", {
            "topic": "Q4 earnings",
            "sub_queries": [],
            "kb_ids": [1],
        })
        assert result.success
        assert result.output["total_unique"] == 0
        assert result.output["chunks"] == []
        assert result.output["sub_queries_run"] == 0

    def test_deduplicates_chunks_across_sub_queries(self):
        """Same chunk returned by two sub-queries should appear only once."""
        from langchain_core.documents import Document

        dup_doc = Document(
            page_content="Apple revenue grew 12% in Q4.",
            metadata={"score": 0.9, "source": "Q4.pdf"},
        )
        unique_doc = Document(
            page_content="Google launched new AI products this quarter.",
            metadata={"score": 0.7, "source": "Q4_google.pdf"},
        )

        mock_result_1 = {"docs": [dup_doc]}
        mock_result_2 = {"docs": [dup_doc, unique_doc]}  # dup_doc appears again

        async def _mock_search(query, kb_ids, **kwargs):
            if "Apple" in query:
                return mock_result_1
            return mock_result_2

        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.distinct.return_value.all.return_value = []

        with patch("app.db.session.SessionLocal", return_value=mock_session), \
             patch("app.services.retrieval.hybrid_search_with_legs", side_effect=_mock_search):
            result = execute_tool("synthesize_documents", {
                "topic": "Q4 earnings",
                "sub_queries": ["Apple Q4 revenue", "Google Q4 products"],
                "kb_ids": [1],
            })

        assert result.success
        assert result.output["total_unique"] == 2  # dup_doc counted once
        assert result.output["sub_queries_run"] == 2

    def test_handles_sub_query_failure_gracefully(self):
        """If one sub-query fails, others still return results."""
        from langchain_core.documents import Document

        good_doc = Document(
            page_content="Revenue grew 12%.",
            metadata={"score": 0.9, "source": "Q4.pdf"},
        )

        call_count = [0]

        async def _mock_search(query, kb_ids, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("connection error")
            return {"docs": [good_doc]}

        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.distinct.return_value.all.return_value = []

        with patch("app.db.session.SessionLocal", return_value=mock_session), \
             patch("app.services.retrieval.hybrid_search_with_legs", side_effect=_mock_search):
            result = execute_tool("synthesize_documents", {
                "topic": "Q4 earnings",
                "sub_queries": ["failed query", "good query"],
                "kb_ids": [1],
            })

        assert result.success
        assert result.output["total_unique"] == 1  # failed sub-query skipped

    def test_returns_total_unique(self):
        from langchain_core.documents import Document

        docs = [
            Document(page_content=f"Content {i}", metadata={"score": 0.5, "source": "doc.pdf"})
            for i in range(5)
        ]

        async def _mock_search(query, kb_ids, **kwargs):
            return {"docs": docs}

        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.distinct.return_value.all.return_value = []

        with patch("app.db.session.SessionLocal", return_value=mock_session), \
             patch("app.services.retrieval.hybrid_search_with_legs", side_effect=_mock_search):
            result = execute_tool("synthesize_documents", {
                "topic": "test",
                "sub_queries": ["q1", "q2"],
                "kb_ids": [1],
            })

        assert result.success
        assert result.output["total_unique"] == 5  # same docs deduped

    def test_sorted_by_score_descending(self):
        from langchain_core.documents import Document

        docs = [
            Document(page_content="Low score content", metadata={"score": 0.3, "source": "a.pdf"}),
            Document(page_content="High score content", metadata={"score": 0.9, "source": "b.pdf"}),
        ]

        async def _mock_search(query, kb_ids, **kwargs):
            return {"docs": docs}

        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.distinct.return_value.all.return_value = []

        with patch("app.db.session.SessionLocal", return_value=mock_session), \
             patch("app.services.retrieval.hybrid_search_with_legs", side_effect=_mock_search):
            result = execute_tool("synthesize_documents", {
                "topic": "test",
                "sub_queries": ["q1"],
                "kb_ids": [1],
            })

        chunks = result.output["chunks"]
        assert chunks[0]["score"] >= chunks[-1]["score"]


# ── generate_synthesis_report ─────────────────────────────────────────────────

class TestGenerateSynthesisReport:
    def _trace(self, sources):
        return [{
            "tool_name": "synthesize_documents",
            "output": {"chunks": [{"source": s, "content": "text", "score": 0.8} for s in sources]},
            "error": None,
            "latency_ms": 100.0,
        }]

    def test_includes_query(self):
        report = generate_synthesis_report("The answer.", [], "My synthesis query", [1])
        assert "My synthesis query" in report

    def test_includes_header(self):
        report = generate_synthesis_report("Answer text.", [], "q", [1])
        assert "# Synthesis Report" in report

    def test_includes_answer(self):
        report = generate_synthesis_report("## Key themes\n- Theme 1", [], "q", [1])
        assert "## Key themes" in report
        assert "Theme 1" in report

    def test_builds_sources_section(self):
        trace = self._trace(["Q4.pdf", "Q4.pdf", "Annual.pdf"])
        report = generate_synthesis_report("Answer.", trace, "q", [1])
        assert "## Sources" in report
        assert "Q4.pdf" in report
        assert "Annual.pdf" in report

    def test_deduplicates_sources_by_count(self):
        trace = self._trace(["Q4.pdf", "Q4.pdf", "Other.pdf"])
        report = generate_synthesis_report("Answer.", trace, "q", [1])
        # Q4.pdf (2 chunks) should appear before Other.pdf (1 chunk)
        q4_pos = report.index("Q4.pdf")
        other_pos = report.index("Other.pdf")
        assert q4_pos < other_pos

    def test_empty_tool_trace_no_sources_section(self):
        report = generate_synthesis_report("Answer.", [], "q", [1])
        assert "## Sources" not in report

    def test_includes_kb_ids(self):
        report = generate_synthesis_report("Answer.", [], "q", [1, 2, 3])
        assert "1, 2, 3" in report

    def test_handles_search_documents_trace(self):
        """search_documents returns a list, not a dict with chunks."""
        trace = [{
            "tool_name": "search_documents",
            "output": [
                {"source": "doc1.pdf", "content": "text", "score": 0.8},
                {"source": "doc2.pdf", "content": "text", "score": 0.7},
            ],
            "error": None,
            "latency_ms": 50.0,
        }]
        report = generate_synthesis_report("Answer.", trace, "q", [1])
        assert "doc1.pdf" in report


# ── Config ────────────────────────────────────────────────────────────────────

class TestSynthesisConfig:
    def test_synthesis_mode_enabled_in_registry(self):
        from app.core.settings_registry import get_def
        defn = get_def("SYNTHESIS_MODE_ENABLED")
        assert defn is not None
        assert isinstance(defn.default, bool)

    def test_default_is_true(self):
        from app.core.settings_registry import get_def
        assert get_def("SYNTHESIS_MODE_ENABLED").default is True
