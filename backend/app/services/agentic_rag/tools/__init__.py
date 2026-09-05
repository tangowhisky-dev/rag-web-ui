"""Agent tool registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.agentic_rag.tool_context import ToolContext

from .chart_generate import ChartGenerateTool
from .code_execute import CodeExecuteTool
from .current_datetime import CurrentDatetimeTool
from .extract_data import ExtractDataTool
from .file_extract_table import FileExtractTableTool
from .file_read import FileReadTool
from .file_summarize import FileSummarizeTool
from .graph_expand import GraphExpandTool
from .kb_grep import KbGrepTool
from .kb_metadata import KbMetadataTool
from .kb_outline import KbOutlineTool
from .kb_read import KbReadTool
from .kb_search_documents import KbSearchDocumentsTool
from .rerank_results import RerankResultsTool
from .search_dense import SearchDenseTool
from .search_exact import SearchExactTool
from .search_sparse import SearchSparseTool
from .summarize_answer import SummarizeAnswerTool


_TOOL_CLASSES = [
    # Atomic search tools
    SearchExactTool,
    SearchSparseTool,
    SearchDenseTool,
    RerankResultsTool,
    GraphExpandTool,
    # Discovery
    KbSearchDocumentsTool,
    KbMetadataTool,
    KbOutlineTool,
    CurrentDatetimeTool,
    # Read
    KbReadTool,
    FileReadTool,
    FileSummarizeTool,
    FileExtractTableTool,
    # Processing
    CodeExecuteTool,
    ChartGenerateTool,
    SummarizeAnswerTool,
    ExtractDataTool,
    KbGrepTool,
]

ALL_TOOLS = _TOOL_CLASSES


def build_tools(ctx: "ToolContext") -> list:
    """Return tool instances bound to the given ToolContext."""
    tools = []
    for cls in _TOOL_CLASSES:
        tool = cls()
        tool.ctx = ctx
        tools.append(tool)
    return tools


def _has_chart_data(state) -> bool:
    if state is None:
        return False
    lao = state.get("last_answer_object")
    has_data = (lao is not None and getattr(lao, "data", None)) or bool(state.get("retrieved_docs"))
    # A successful code_execute or extract_data observation earlier in
    # this turn produces data that chart_generate can consume. Without
    # this check, a plan that runs code_execute → chart_generate fails
    # because chart_generate is filtered out after code_execute succeeds.
    if not has_data:
        for raw_obs in state.get("observations") or []:
            if isinstance(raw_obs, dict):
                tool = raw_obs.get("tool", "")
                err = raw_obs.get("error")
            else:
                tool = getattr(raw_obs, "tool", "")
                err = getattr(raw_obs, "error", None)
            if tool in ("code_execute", "extract_data") and not err:
                has_data = True
                break
    return has_data


def _filter_tools_by_name(tools: list, excluded: tuple[str, ...]) -> list:
    return [t for t in tools if t.name not in excluded]


def applicable_tools(ctx: "ToolContext") -> list:
    """Filter tools based on the current turn context.

    - File tools only if a file is attached.
    - Chart only if there is data to chart (last_answer_object.data,
      retrieved docs, or a successful code_execute / extract_data
      observation earlier in the same turn).
    - rerank_results and graph_expand only after at least one search tool
      has been called (deferred tool gating).
    - extract_data only after a read or search tool has been called.
    """
    tools = build_tools(ctx)
    state = ctx.state
    # state is a dict (AgentState/MessagesState) at runtime.
    has_file = bool(state.get("file_markdown")) if state is not None else False
    has_data = _has_chart_data(state)

    # Deferred tool gating: check tool_call_counts for prior tool use
    counts = state.get("tool_call_counts", {}) if state is not None else {}
    has_search = any(counts.get(t, 0) > 0 for t in ("search_exact", "search_sparse", "search_dense"))
    # extract_data is available when there's data to extract from (retrieved_docs,
    # last_answer_object.data, or a successful code_execute/extract_data observation)
    # OR after a search/read tool has been called.
    has_read = has_search or any(counts.get(t, 0) > 0 for t in ("kb_read", "kb_search_documents"))

    if not has_file:
        tools = _filter_tools_by_name(tools, ("file_read", "file_summarize", "file_extract_table"))
    if not has_data and not has_read:
        tools = _filter_tools_by_name(tools, ("chart_generate", "extract_data"))
    elif not has_data:
        tools = _filter_tools_by_name(tools, ("chart_generate",))
    if not has_search:
        tools = _filter_tools_by_name(tools, ("rerank_results", "graph_expand"))

    # KB tools (kb_grep, kb_read, kb_outline, kb_metadata) are always
    # available — every chat has KBs linked (ChatCreate requires
    # knowledge_base_ids). Each KB tool handles empty kb_ids gracefully
    # via enforce_rbac.

    return tools
