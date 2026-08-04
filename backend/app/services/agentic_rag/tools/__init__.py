"""Agent tool registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.agentic_rag.tool_context import ToolContext

from .chart_generate import ChartGenerateTool
from .code_execute import CodeExecuteTool
from .extract_data import ExtractDataTool
from .file_extract_table import FileExtractTableTool
from .file_read import FileReadTool
from .file_summarize import FileSummarizeTool
from .rag_retrieve import RagRetrieveTool
from .summarize_answer import SummarizeAnswerTool


_TOOL_CLASSES = [
    RagRetrieveTool,
    FileReadTool,
    FileSummarizeTool,
    FileExtractTableTool,
    CodeExecuteTool,
    ChartGenerateTool,
    SummarizeAnswerTool,
    ExtractDataTool,
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


def applicable_tools(ctx: "ToolContext") -> list:
    """Filter tools based on the current turn context.

    - File tools only if a file is attached.
    - Chart only if there is data to chart (last_answer_object.data or retrieved docs).
    """
    tools = build_tools(ctx)
    state = ctx.state
    # state is a dict (AgentState/MessagesState) at runtime.
    has_file = bool(state.get("file_markdown")) if state is not None else False
    if state is not None:
        lao = state.get("last_answer_object")
        has_data = (lao is not None and getattr(lao, "data", None)) or bool(state.get("retrieved_docs"))
    else:
        has_data = False

    if not has_file:
        tools = [t for t in tools if t.name not in ("file_read", "file_summarize", "file_extract_table")]
    if not has_data:
        tools = [t for t in tools if t.name not in ("chart_generate", "extract_data")]

    return tools
