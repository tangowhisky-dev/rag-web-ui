"""kb_read tool — read a section or character range of a KB document.

Reads converted_markdown (the mirrored markdown stored in the database),
never the original file on disk. This is the "read" in the search/browse/
read triad — the agent uses it after kb_grep and kb_outline to read
specific sections that search tools missed.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.services.agentic_rag.tool_context import ToolContext, write_audit
from app.services.agentic_rag.token_budget import count_tokens
from app.services.agentic_rag.tools.base import BaseAgentTool
from app.services.agentic_rag.tools.kb_outline import _load_authorized_document

logger = logging.getLogger(__name__)


class KbReadInput(BaseModel):
    document_id: int = Field(description="Document ID from search results, kb_grep matches, or kb_outline.")
    section: Optional[str] = Field(default=None, description="Heading text to read (e.g. 'Integrity'). Reads from this heading until the next heading of same or higher level.")
    start_char: Optional[int] = Field(default=None, description="Start character offset (from kb_outline or kb_grep). If omitted with end_char, reads from beginning.")
    end_char: Optional[int] = Field(default=None, description="End character offset. If omitted, reads to end of section or document.")
    max_tokens: int = Field(default=4000, ge=500, le=16000, description="Token budget for returned content.")


class KbReadTool(BaseAgentTool):
    name: str = "kb_read"
    ui_label: str = "Reading KB document"
    description: str = (
        "Read a specific section or character range of a KB document's markdown. "
        "Use after kb_outline to read the relevant section, or after kb_grep to "
        "read context around a matching line. Use when search tools return "
        "insufficient evidence for a specific document."
    )
    args_schema: type[BaseModel] = KbReadInput

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use arun() for agent tools.")

    async def _execute(self, input_obj: KbReadInput) -> dict:
        t0 = time.monotonic()
        ctx: ToolContext = self.ctx

        doc, error = await _load_authorized_document(ctx, input_obj.document_id)
        if error:
            return error

        markdown = doc.converted_markdown or ""
        if not markdown:
            return {"ok": False, "result": {}, "error": "Document has no converted markdown.", "tokens": 0}

        section_name = None
        char_start = 0
        char_end = len(markdown)

        if input_obj.section:
            section_name, char_start, char_end = _extract_section(markdown, input_obj.section)
            if char_start is None:
                # Section not found — return full document so the agent can
                # see the structure and try a different section name.
                section_name = None
                char_start = 0
                char_end = len(markdown)
        elif input_obj.start_char is not None or input_obj.end_char is not None:
            char_start = max(0, input_obj.start_char or 0)
            char_end = min(len(markdown), input_obj.end_char or len(markdown))

        content = markdown[char_start:char_end]

        # Token-truncate
        tokens = count_tokens(content)
        truncated = False
        if tokens > input_obj.max_tokens:
            max_chars = input_obj.max_tokens * 4
            content = content[:max_chars]
            char_end = char_start + len(content)
            truncated = True
            tokens = count_tokens(content)

        latency_ms = round((time.monotonic() - t0) * 1000)
        write_audit(ctx, "kb_read", input_obj.model_dump(),
                     {"document_id": doc.id, "section": section_name, "truncated": truncated,
                      "char_range": [char_start, char_end]},
                     latency_ms=latency_ms, status="ok")

        # Determine citation_kind based on read mode
        if section_name:
            citation_kind = "section"
        elif input_obj.start_char is not None or input_obj.end_char is not None:
            citation_kind = "range"
        else:
            citation_kind = "file"

        # Compute line numbers for range reads
        start_line = end_line = None
        if citation_kind == "range":
            start_line = markdown[:char_start].count("\n") + 1
            end_line = markdown[:char_end].count("\n") + 1

        return {
            "ok": True,
            "result": {
                "document_id": doc.id,
                "title": doc.title or doc.file_name,
                "file_name": doc.file_name,
                "section": section_name,
                "content": content,
                "total_tokens": tokens,
                "truncated": truncated,
                "char_range": [char_start, char_end],
                "start_line": start_line,
                "end_line": end_line,
                "citation_ref": {
                    "document_id": doc.id,
                    "citation_kind": citation_kind,
                    "chunk_index": None,
                    "section": section_name,
                    "start_char": char_start if citation_kind in ("section", "range") else None,
                    "end_char": char_end if citation_kind in ("section", "range") else None,
                    "start_line": start_line,
                    "end_line": end_line,
                    "quoted_text": content[:200],
                    "source_tool": "kb_read",
                    "citation_id": "",
                },
            },
            "error": None,
            "tokens": tokens,
        }


def _extract_section(markdown: str, section: str) -> tuple[Optional[str], Optional[int], Optional[int]]:
    """Extract content under a heading, up to the next heading of same/higher level.

    Returns (section_name, start_char, end_char). If the heading is not
    found, returns (None, None, None).
    """
    pattern = re.compile(rf"(?m)^(#+)\s+{re.escape(section)}.*$", re.IGNORECASE)
    m = pattern.search(markdown)
    if not m:
        return None, None, None

    heading_level = len(m.group(1))
    start = m.start()
    # Find the next heading of same or higher level.
    rest = markdown[m.end():]
    next_heading = re.search(rf"(?m)^({'#' * heading_level})\s+", rest)
    if next_heading:
        end = m.end() + next_heading.start()
    else:
        end = len(markdown)

    return section, start, end
