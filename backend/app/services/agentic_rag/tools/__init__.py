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
from .kb_grep import KbGrepTool
from .kb_outline import KbOutlineTool
from .kb_read import KbReadTool
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
    KbGrepTool,
    KbReadTool,
    KbOutlineTool,
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
    - Chart only if there is data to chart (last_answer_object.data,
      retrieved docs, or a successful code_execute / extract_data
      observation earlier in the same turn).
    """
    tools = build_tools(ctx)
    state = ctx.state
    # state is a dict (AgentState/MessagesState) at runtime.
    has_file = bool(state.get("file_markdown")) if state is not None else False
    if state is not None:
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
    else:
        has_data = False

    if not has_file:
        tools = [t for t in tools if t.name not in ("file_read", "file_summarize", "file_extract_table")]
    if not has_data:
        tools = [t for t in tools if t.name not in ("chart_generate", "extract_data")]

    has_kb = bool(state.get("kb_ids")) if state is not None else False
    if not has_kb:
        tools = [t for t in tools if t.name not in ("kb_grep", "kb_read", "kb_outline")]

    return tools
