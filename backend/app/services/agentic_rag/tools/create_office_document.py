"""Wrapper tool that delegates Office document generation to the sub-agent.

The main agent calls this single tool instead of orchestrating 4 office tools.
The sub-agent runs its own think→tool loop with a focused prompt and only
office tools, then returns file metadata.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CreateOfficeDocumentInput(BaseModel):
    """Input for the office document creation tool."""

    request: str = Field(
        ...,
        description="Natural language description of what to create. "
        "Include: format (pptx/docx/xlsx), title, content structure, "
        "and any specific requirements. Example: "
        "'Create a 3-slide PPTX about risk management. "
        "Slide 1: Title and overview. Slide 2: Key principles. "
        "Slide 3: Best practices.'",
    )


class CreateOfficeDocumentTool(BaseTool):
    """Tool that delegates to the office sub-agent."""

    name: str = "create_office_document"
    description: str = (
        "Create an Office document (PowerPoint/pptx, Word/docx, or Excel/xlsx). "
        "Pass a natural language description of what to create — the tool handles "
        "loading design guidelines, generating the document, inspecting quality, "
        "and fixing issues automatically. Returns file_id for download. "
        "Data from accumulated_data is used automatically for data-driven documents. "
        "For text-only documents, describe the content structure in the request."
    )
    prompt_snippet: str = "Create Office artifacts (DOCX, PPTX, XLSX)"
    prompt_guidelines: list[str] = [
        "create_office_document: Use for explicit requests to create downloadable DOCX, PPTX, or XLSX files. For data-driven artifacts, extract/prepare structured data first.",
        "create_office_document: Handles embedded charts internally — do NOT call chart_generate separately for charts that belong inside a document.",
        "create_office_document: Supported formats: pptx, docx, xlsx ONLY. If the user asks for PDF/TXT/CSV/JSON/HTML, tell them only pptx/docx/xlsx are supported.",
        "create_office_document: The tool call is the ONLY way to produce a file. Writing a description without calling the tool is a failure.",
    ]
    args_schema: type = CreateOfficeDocumentInput
    ctx: Any = None

    async def arun(self, tool_input: str | dict[str, Any], **kwargs: Any) -> dict:
        """Run the office sub-agent."""
        if isinstance(tool_input, str):
            import json
            try:
                args = json.loads(tool_input)
            except Exception:
                args = {"request": tool_input}
        else:
            args = tool_input

        request = args.get("request", "")
        if not request:
            return {
                "ok": False,
                "result": {},
                "error": "Missing 'request' field. Describe what document to create.",
                "tokens": 0,
                "terminate": False,
            }

        ctx = self.ctx
        if ctx is None:
            return {
                "ok": False,
                "result": {},
                "error": "No context available",
                "tokens": 0,
                "terminate": False,
            }

        max_iter = 6
        try:
            from app.services.settings_service import get_setting
            max_iter = get_setting(ctx.db, "OFFICE_SUBAGENT_MAX_ITERATIONS", ctx.org_id) or 6
        except Exception:
            pass

        # Lazy import to avoid circular dependency
        from app.services.agentic_rag.office_subagent import run_office_subagent

        try:
            result = await run_office_subagent(
                ctx=ctx,
                request=request,
                max_iterations=max_iter,
            )
        except Exception as exc:
            logger.exception("[create_office_document] sub-agent failed: %s", exc)
            return {
                "ok": False,
                "result": {},
                "error": f"Office sub-agent failed: {exc}",
                "tokens": 0,
                "terminate": False,
            }

        # Return as tool result — the main agent sees this as an observation
        return {
            "ok": result["ok"],
            "result": {
                "file_id": result["file_id"],
                "file_name": result["file_name"],
                "format": result["format"],
                "summary": result["summary"],
            },
            "error": result["error"],
            "tokens": 0,
            "terminate": False,
        }

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use arun() for agent tools.")
