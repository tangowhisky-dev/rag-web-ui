"""Tests for kb_grep, kb_outline, and kb_read agent tools.

Covers:
- Basic functionality (grep finds lines, outline extracts headings, read returns content)
- RBAC enforcement (unauthorized KBs/documents are denied)
- Section extraction by heading name
- Character range slicing
- Token truncation
- Max results cap for grep
- Datastore document access
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest


# ── Stubs ──────────────────────────────────────────────────────────────────────

class _StubDoc:
    """Minimal Document stub."""
    def __init__(self, id=1, title="Test Doc", file_name="test.pdf",
                 converted_markdown="# Heading 1\n\nSome text.\n\n## Subheading\n\nMore text.\n",
                 knowledge_base_id=1, data_store_id=None):
        self.id = id
        self.title = title
        self.file_name = file_name
        self.converted_markdown = converted_markdown
        self.knowledge_base_id = knowledge_base_id
        self.data_store_id = data_store_id


class _StubQuery:
    """Chained SQLAlchemy query stub."""
    def __init__(self, docs):
        self._docs = docs
        self._filters = []

    def filter(self, *args, **kwargs):
        self._filters.append(args)
        return self

    def all(self):
        return self._docs

    def first(self):
        return self._docs[0] if self._docs else None


class _StubDB:
    """Minimal DB session stub."""
    def __init__(self, docs):
        self._docs = docs
        self._query_docs = _StubQuery(docs)

    def query(self, model):
        return _StubQuery(self._docs)

    def add(self, record):
        pass

    def rollback(self):
        pass

    def flush(self):
        pass


class _StubToolContext:
    def __init__(self, docs, chat_id=1, user_id=1, org_id=1, state=None):
        self.db = _StubDB(docs)
        self.user_id = user_id
        self.org_id = org_id
        self.chat_id = chat_id
        self.message_id = None
        self.qdrant_client = None
        self.redis_memory = None
        self.org_llm_config = {}
        self.state = state


# ── RBAC mock helper ───────────────────────────────────────────────────────────

def _mock_rbac(kb_ids=None, file_id=None):
    """Return a mock enforce_rbac that allows the given kb_ids."""
    def _enforce(ctx, kb_ids=None, file_id=None):
        return {"kb_ids": kb_ids or [1], "file_id": file_id}
    return _enforce


def _mock_rbac_empty(kb_ids=None, file_id=None):
    """Return a mock enforce_rbac that denies all access."""
    def _enforce(ctx, kb_ids=None, file_id=None):
        return {"kb_ids": [], "file_id": None}
    return _enforce


def _mock_datastore_ids(ds_ids=None):
    """Return a mock get_effective_datastore_ids."""
    def _get(kb_ids, org_id, db):
        return ds_ids or []
    return _get


# ── kb_grep tests ──────────────────────────────────────────────────────────────

def _run_tool(tool_cls, ctx, input_obj):
    """Run a tool's _execute method in an asyncio event loop."""
    tool = tool_cls()
    tool.ctx = ctx
    return asyncio.run(tool._execute(input_obj))


def test_kb_grep_finds_matching_lines():
    from app.services.agentic_rag.tools.kb_grep import KbGrepTool, KbGrepInput
    markdown = "# Leadership\n\nIntegrity is key.\n\nVision matters.\n\nAccountability counts.\n"
    doc = _StubDoc(converted_markdown=markdown)
    ctx = _StubToolContext([doc])

    with patch("app.services.agentic_rag.tools.kb_grep.enforce_rbac", side_effect=_mock_rbac()), \
         patch("app.services.retrieval.retrieval.get_effective_datastore_ids", side_effect=_mock_datastore_ids()):
        result = _run_tool(KbGrepTool, ctx, KbGrepInput(pattern="integrity"))

    assert result["ok"] is True
    matches = result["result"]["matches"]
    assert len(matches) == 1
    assert "Integrity" in matches[0]["line_text"]
    assert matches[0]["document_id"] == 1


def test_kb_grep_respects_rbac():
    from app.services.agentic_rag.tools.kb_grep import KbGrepTool, KbGrepInput
    doc = _StubDoc(converted_markdown="secret content")
    ctx = _StubToolContext([doc])

    with patch("app.services.agentic_rag.tools.kb_grep.enforce_rbac", side_effect=_mock_rbac_empty()):
        result = _run_tool(KbGrepTool, ctx, KbGrepInput(pattern="secret"))

    assert result["ok"] is False
    assert "No authorized" in result["error"]


def test_kb_grep_max_results_cap():
    from app.services.agentic_rag.tools.kb_grep import KbGrepTool, KbGrepInput
    markdown = "\n".join(f"match line {i}" for i in range(100))
    doc = _StubDoc(converted_markdown=markdown)
    ctx = _StubToolContext([doc])

    with patch("app.services.agentic_rag.tools.kb_grep.enforce_rbac", side_effect=_mock_rbac()), \
         patch("app.services.retrieval.retrieval.get_effective_datastore_ids", side_effect=_mock_datastore_ids()):
        result = _run_tool(KbGrepTool, ctx, KbGrepInput(pattern="match", max_results=10))

    assert result["ok"] is True
    assert len(result["result"]["matches"]) == 10


def test_kb_grep_case_insensitive_default():
    from app.services.agentic_rag.tools.kb_grep import KbGrepTool, KbGrepInput
    markdown = "INTEGRITY is important\n"
    doc = _StubDoc(converted_markdown=markdown)
    ctx = _StubToolContext([doc])

    with patch("app.services.agentic_rag.tools.kb_grep.enforce_rbac", side_effect=_mock_rbac()), \
         patch("app.services.retrieval.retrieval.get_effective_datastore_ids", side_effect=_mock_datastore_ids()):
        result = _run_tool(KbGrepTool, ctx, KbGrepInput(pattern="integrity"))

    assert len(result["result"]["matches"]) == 1


def test_kb_grep_invalid_regex():
    from app.services.agentic_rag.tools.kb_grep import KbGrepTool, KbGrepInput
    doc = _StubDoc(converted_markdown="text")
    ctx = _StubToolContext([doc])

    with patch("app.services.agentic_rag.tools.kb_grep.enforce_rbac", side_effect=_mock_rbac()), \
         patch("app.services.retrieval.retrieval.get_effective_datastore_ids", side_effect=_mock_datastore_ids()):
        result = _run_tool(KbGrepTool, ctx, KbGrepInput(pattern="[unclosed"))

    assert result["ok"] is False
    assert "Invalid regex" in result["error"]


# ── kb_outline tests ───────────────────────────────────────────────────────────

def test_kb_outline_extracts_headings():
    from app.services.agentic_rag.tools.kb_outline import KbOutlineTool, KbOutlineInput
    markdown = "# Title\n\nIntro text.\n\n## Section A\n\nContent A.\n\n### Subsection\n\nDeep.\n\n## Section B\n\nContent B.\n"
    doc = _StubDoc(converted_markdown=markdown)
    ctx = _StubToolContext([doc])

    with patch("app.services.agentic_rag.tools.kb_outline.enforce_rbac", side_effect=_mock_rbac()), \
         patch("app.services.retrieval.retrieval.get_effective_datastore_ids", side_effect=_mock_datastore_ids()):
        result = _run_tool(KbOutlineTool, ctx, KbOutlineInput(document_id=1))

    assert result["ok"] is True
    headings = result["result"]["headings"]
    assert len(headings) == 4
    assert headings[0]["level"] == 1
    assert headings[0]["text"] == "Title"
    assert headings[1]["level"] == 2
    assert headings[1]["text"] == "Section A"
    assert headings[2]["level"] == 3
    assert headings[2]["text"] == "Subsection"


def test_kb_outline_respects_rbac():
    from app.services.agentic_rag.tools.kb_outline import KbOutlineTool, KbOutlineInput
    doc = _StubDoc(converted_markdown="# Secret\n\nConfidential.\n")
    ctx = _StubToolContext([doc])

    with patch("app.services.agentic_rag.tools.kb_outline.enforce_rbac", side_effect=_mock_rbac_empty()):
        result = _run_tool(KbOutlineTool, ctx, KbOutlineInput(document_id=1))

    assert result["ok"] is False
    assert "No authorized" in result["error"]


def test_kb_outline_no_headings():
    from app.services.agentic_rag.tools.kb_outline import KbOutlineTool, KbOutlineInput
    markdown = "Just plain text with no headings at all.\n\nMore text.\n"
    doc = _StubDoc(converted_markdown=markdown)
    ctx = _StubToolContext([doc])

    with patch("app.services.agentic_rag.tools.kb_outline.enforce_rbac", side_effect=_mock_rbac()), \
         patch("app.services.retrieval.retrieval.get_effective_datastore_ids", side_effect=_mock_datastore_ids()):
        result = _run_tool(KbOutlineTool, ctx, KbOutlineInput(document_id=1))

    assert result["ok"] is True
    assert result["result"]["headings"] == []


def test_kb_outline_document_not_found():
    from app.services.agentic_rag.tools.kb_outline import KbOutlineTool, KbOutlineInput
    ctx = _StubToolContext([])  # no docs

    with patch("app.services.agentic_rag.tools.kb_outline.enforce_rbac", side_effect=_mock_rbac()), \
         patch("app.services.retrieval.retrieval.get_effective_datastore_ids", side_effect=_mock_datastore_ids()):
        result = _run_tool(KbOutlineTool, ctx, KbOutlineInput(document_id=999))

    assert result["ok"] is False
    assert "not found" in result["error"]


# ── kb_read tests ──────────────────────────────────────────────────────────────

def test_kb_read_full_document():
    from app.services.agentic_rag.tools.kb_read import KbReadTool, KbReadInput
    markdown = "# Title\n\nSome content here.\n"
    doc = _StubDoc(converted_markdown=markdown)
    ctx = _StubToolContext([doc])

    with patch("app.services.agentic_rag.tools.kb_outline.enforce_rbac", side_effect=_mock_rbac()), \
         patch("app.services.retrieval.retrieval.get_effective_datastore_ids", side_effect=_mock_datastore_ids()):
        result = _run_tool(KbReadTool, ctx, KbReadInput(document_id=1))

    assert result["ok"] is True
    assert "Some content" in result["result"]["content"]
    assert result["result"]["section"] is None


def test_kb_read_by_section():
    from app.services.agentic_rag.tools.kb_read import KbReadTool, KbReadInput
    markdown = "# Title\n\nIntro.\n\n## Integrity\n\nIntegrity is key.\n\n## Vision\n\nVision matters.\n"
    doc = _StubDoc(converted_markdown=markdown)
    ctx = _StubToolContext([doc])

    with patch("app.services.agentic_rag.tools.kb_outline.enforce_rbac", side_effect=_mock_rbac()), \
         patch("app.services.retrieval.retrieval.get_effective_datastore_ids", side_effect=_mock_datastore_ids()):
        result = _run_tool(KbReadTool, ctx, KbReadInput(document_id=1, section="Integrity"))

    assert result["ok"] is True
    assert result["result"]["section"] == "Integrity"
    assert "Integrity is key" in result["result"]["content"]
    # Should NOT include the next section
    assert "Vision matters" not in result["result"]["content"]


def test_kb_read_by_char_range():
    from app.services.agentic_rag.tools.kb_read import KbReadTool, KbReadInput
    markdown = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    doc = _StubDoc(converted_markdown=markdown)
    ctx = _StubToolContext([doc])

    with patch("app.services.agentic_rag.tools.kb_outline.enforce_rbac", side_effect=_mock_rbac()), \
         patch("app.services.retrieval.retrieval.get_effective_datastore_ids", side_effect=_mock_datastore_ids()):
        result = _run_tool(KbReadTool, ctx, KbReadInput(document_id=1, start_char=10, end_char=20))

    assert result["ok"] is True
    assert result["result"]["content"] == "ABCDEFGHIJ"
    assert result["result"]["char_range"] == [10, 20]


def test_kb_read_respects_rbac():
    from app.services.agentic_rag.tools.kb_read import KbReadTool, KbReadInput
    doc = _StubDoc(converted_markdown="secret content")
    ctx = _StubToolContext([doc])

    with patch("app.services.agentic_rag.tools.kb_outline.enforce_rbac", side_effect=_mock_rbac_empty()):
        result = _run_tool(KbReadTool, ctx, KbReadInput(document_id=1))

    assert result["ok"] is False
    assert "No authorized" in result["error"]


def test_kb_read_section_not_found_returns_full():
    from app.services.agentic_rag.tools.kb_read import KbReadTool, KbReadInput
    markdown = "# Title\n\nContent.\n"
    doc = _StubDoc(converted_markdown=markdown)
    ctx = _StubToolContext([doc])

    with patch("app.services.agentic_rag.tools.kb_outline.enforce_rbac", side_effect=_mock_rbac()), \
         patch("app.services.retrieval.retrieval.get_effective_datastore_ids", side_effect=_mock_datastore_ids()):
        result = _run_tool(KbReadTool, ctx, KbReadInput(document_id=1, section="Nonexistent"))

    assert result["ok"] is True
    # Falls back to full document
    assert "Content" in result["result"]["content"]
    assert result["result"]["section"] is None


def test_kb_read_truncates_to_max_tokens():
    from app.services.agentic_rag.tools.kb_read import KbReadTool, KbReadInput
    # Large content — 10000 chars
    markdown = "A" * 10000
    doc = _StubDoc(converted_markdown=markdown)
    ctx = _StubToolContext([doc])

    with patch("app.services.agentic_rag.tools.kb_outline.enforce_rbac", side_effect=_mock_rbac()), \
         patch("app.services.retrieval.retrieval.get_effective_datastore_ids", side_effect=_mock_datastore_ids()):
        result = _run_tool(KbReadTool, ctx, KbReadInput(document_id=1, max_tokens=500))

    assert result["ok"] is True
    assert result["result"]["truncated"] is True
    # 500 tokens * 4 chars/token = 2000 chars max
    assert len(result["result"]["content"]) <= 2000


def test_kb_read_document_not_found():
    from app.services.agentic_rag.tools.kb_read import KbReadTool, KbReadInput
    ctx = _StubToolContext([])

    with patch("app.services.agentic_rag.tools.kb_outline.enforce_rbac", side_effect=_mock_rbac()), \
         patch("app.services.retrieval.retrieval.get_effective_datastore_ids", side_effect=_mock_datastore_ids()):
        result = _run_tool(KbReadTool, ctx, KbReadInput(document_id=999))

    assert result["ok"] is False
    assert "not found" in result["error"]


# ── RBAC: unauthorized document ────────────────────────────────────────────────

def test_kb_read_unauthorized_document():
    """Document exists but belongs to a KB not linked to this chat."""
    from app.services.agentic_rag.tools.kb_read import KbReadTool, KbReadInput
    doc = _StubDoc(knowledge_base_id=999, converted_markdown="secret")
    ctx = _StubToolContext([doc])

    # enforce_rbac returns kb_ids=[1] (authorized), but doc is in KB 999
    with patch("app.services.agentic_rag.tools.kb_outline.enforce_rbac", side_effect=_mock_rbac(kb_ids=[1])), \
         patch("app.services.retrieval.retrieval.get_effective_datastore_ids", side_effect=_mock_datastore_ids()):
        result = _run_tool(KbReadTool, ctx, KbReadInput(document_id=1))

    assert result["ok"] is False
    assert "Access denied" in result["error"]


def test_kb_outline_unauthorized_document():
    from app.services.agentic_rag.tools.kb_outline import KbOutlineTool, KbOutlineInput
    doc = _StubDoc(knowledge_base_id=999, converted_markdown="# Secret\n")
    ctx = _StubToolContext([doc])

    with patch("app.services.agentic_rag.tools.kb_outline.enforce_rbac", side_effect=_mock_rbac(kb_ids=[1])), \
         patch("app.services.retrieval.retrieval.get_effective_datastore_ids", side_effect=_mock_datastore_ids()):
        result = _run_tool(KbOutlineTool, ctx, KbOutlineInput(document_id=1))

    assert result["ok"] is False
    assert "Access denied" in result["error"]


# ── Datastore document access ──────────────────────────────────────────────────

def test_kb_read_datastore_document():
    """Document from a datastore linked to an authorized KB should be accessible."""
    from app.services.agentic_rag.tools.kb_read import KbReadTool, KbReadInput
    doc = _StubDoc(knowledge_base_id=None, data_store_id=5, converted_markdown="# DS Doc\n\nContent.\n")
    ctx = _StubToolContext([doc])

    with patch("app.services.agentic_rag.tools.kb_outline.enforce_rbac", side_effect=_mock_rbac(kb_ids=[1])), \
         patch("app.services.retrieval.retrieval.get_effective_datastore_ids", side_effect=_mock_datastore_ids(ds_ids=[5])):
        result = _run_tool(KbReadTool, ctx, KbReadInput(document_id=1))

    assert result["ok"] is True
    assert "Content" in result["result"]["content"]


def test_kb_grep_searches_datastore_documents():
    from app.services.agentic_rag.tools.kb_grep import KbGrepTool, KbGrepInput
    ds_doc = _StubDoc(id=2, knowledge_base_id=None, data_store_id=5, converted_markdown="datastore match line\n")
    ctx = _StubToolContext([ds_doc])

    with patch("app.services.agentic_rag.tools.kb_grep.enforce_rbac", side_effect=_mock_rbac(kb_ids=[1])), \
         patch("app.services.retrieval.retrieval.get_effective_datastore_ids", side_effect=_mock_datastore_ids(ds_ids=[5])):
        result = _run_tool(KbGrepTool, ctx, KbGrepInput(pattern="datastore"))

    assert result["ok"] is True
    assert len(result["result"]["matches"]) == 1
    assert result["result"]["matches"][0]["document_id"] == 2


# ── Tool registration ──────────────────────────────────────────────────────────

def test_tools_registered():
    """Verify all three tools are in the registry."""
    from app.services.agentic_rag.tools import _TOOL_CLASSES
    names = set()
    for cls in _TOOL_CLASSES:
        try:
            instance = cls()
            names.add(instance.name)
        except Exception:
            pass
    assert "kb_grep" in names
    assert "kb_read" in names
    assert "kb_outline" in names


def test_applicable_tools_filters_without_kb():
    """KB tools should be filtered out when state has no kb_ids."""
    from app.services.agentic_rag.tools import applicable_tools
    from app.services.agentic_rag.tool_context import ToolContext

    ctx = ToolContext(db=None, user_id=1, org_id=1, chat_id=1, state={"kb_ids": [], "file_markdown": None})
    tools = applicable_tools(ctx)
    names = {t.name for t in tools}
    assert "kb_grep" not in names
    assert "kb_read" not in names
    assert "kb_outline" not in names


def test_applicable_tools_includes_with_kb():
    """KB tools should be available when state has kb_ids."""
    from app.services.agentic_rag.tools import applicable_tools
    from app.services.agentic_rag.tool_context import ToolContext

    ctx = ToolContext(db=None, user_id=1, org_id=1, chat_id=1, state={"kb_ids": [1], "file_markdown": None})
    tools = applicable_tools(ctx)
    names = {t.name for t in tools}
    assert "kb_grep" in names
    assert "kb_read" in names
    assert "kb_outline" in names


# ── Tool call budget ───────────────────────────────────────────────────────────

def test_tool_call_budget_includes_kb_tools():
    from app.services.agentic_rag.agent_graph import _tool_call_budget
    with patch("app.services.agentic_rag.agent_graph.get_setting", side_effect=lambda db, key, org_id=None: {
        "AGENT_MAX_RETRIEVALS": 3, "AGENT_MAX_CODE_EXEC": 3,
        "AGENT_MAX_KB_GREP": 5, "AGENT_MAX_KB_READ": 10,
    }.get(key, 0)):
        budget = _tool_call_budget(None, None)
    assert "kb_grep" in budget
    assert "kb_read" in budget
    assert "kb_outline" in budget
    assert budget["kb_grep"] == 5
    assert budget["kb_read"] == 10
    assert budget["kb_outline"] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
