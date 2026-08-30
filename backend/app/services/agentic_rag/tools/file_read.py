"""file_read tool — read an attached file or a section of it."""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

from pydantic import BaseModel, Field

from app.models.chat import ChatFile
from app.services.agentic_rag.schemas import CitationRef, LastAnswerObject
from app.services.agentic_rag.tool_context import ToolContext, enforce_rbac, write_audit
from app.services.agentic_rag.token_budget import count_tokens
from app.services.agentic_rag.tools.base import BaseAgentTool

logger = logging.getLogger(__name__)


class FileReadInput(BaseModel):
    file_id: Optional[int] = Field(default=None, description="Specific file; default most recent in chat.")
    section: Optional[str] = Field(default=None, description="Heading text, 'page:N', or 'chunk:I'.")
    max_tokens: int = Field(default=4000, ge=500, le=16000)


def _resolve_file(ctx: ToolContext, file_id: Optional[int]) -> tuple:
    if file_id is None and ctx.chat_id:
        cf = (
            ctx.db.query(ChatFile)
            .filter(ChatFile.chat_id == ctx.chat_id)
            .order_by(ChatFile.id.desc())
            .first()
        )
        file_id = cf.id if cf else None

    if not file_id:
        return None, {"ok": False, "result": {}, "error": "No file specified and no attached file found.", "tokens": 0}

    rbac = enforce_rbac(ctx, file_id=file_id)
    if rbac.get("file_id") is None:
        return None, {"ok": False, "result": {}, "error": "Access denied to file.", "tokens": 0}
    file_id = rbac["file_id"]

    cf = ctx.db.query(ChatFile).filter(ChatFile.id == file_id).first()
    if not cf or not cf.markdown_content:
        return None, {"ok": False, "result": {}, "error": "File not found or not processed.", "tokens": 0}

    return cf, None


def _extract_section(content: str, section: Optional[str]) -> tuple:
    section_name = None
    if section:
        if section.startswith("page:"):
            page = section.split(":", 1)[1]
            marker = f"<!-- page {page} -->"
            if marker in content:
                idx = content.find(marker)
                content = content[idx:idx + 8000]
                section_name = f"page {page}"
        elif section.startswith("chunk:"):
            chunk_idx = int(section.split(":", 1)[1])
            chunks = content.split("\n\n")
            content = chunks[chunk_idx] if 0 <= chunk_idx < len(chunks) else ""
            section_name = f"chunk {chunk_idx}"
        else:
            pattern = re.compile(rf"(?m)^#+\s+{re.escape(section)}.*$", re.IGNORECASE)
            m = pattern.search(content)
            if m:
                start = m.start()
                next_heading = re.search(r"(?m)^#+\s+", content[m.end():])
                end = m.end() + (next_heading.start() if next_heading else len(content) - m.end())
                content = content[start:end]
                section_name = section
    return content, section_name


class FileReadTool(BaseAgentTool):
    name: str = "file_read"
    ui_label: str = "Reading file"
    description: str = (
        "Read content from an attached file. Use for questions about a specific "
        "section or when the user says 'this file' without asking for a summary."
    )
    args_schema: type[BaseModel] = FileReadInput

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use arun() for agent tools.")

    async def _execute(self, input_obj: FileReadInput) -> dict:
        t0 = time.monotonic()
        ctx: ToolContext = self.ctx

        cf, err = _resolve_file(ctx, input_obj.file_id)
        if err:
            return err

        content = cf.markdown_content
        content, section_name = _extract_section(content, input_obj.section)

        tokens = count_tokens(content)
        truncated = False
        if tokens > input_obj.max_tokens:
            max_chars = input_obj.max_tokens * 4
            content = content[:max_chars]
            truncated = True
            tokens = count_tokens(content)

        latency_ms = round((time.monotonic() - t0) * 1000)
        write_audit(ctx, "file_read", input_obj.model_dump(), {"file_name": cf.file_name, "section": section_name}, latency_ms=latency_ms, status="ok")

        return {
            "ok": True,
            "result": {
                "file_name": cf.file_name,
                "section": section_name,
                "content": content,
                "total_tokens": tokens,
                "truncated": truncated,
            },
            "error": None,
            "tokens": tokens,
        }
