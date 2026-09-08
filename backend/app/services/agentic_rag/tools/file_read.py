"""file_read tool — read a portion of a KB document or attached chat file.

Replaces the former kb_read and file_read tools. Uses pi-style offset/limit
for portion selection: the model uses kb_outline (which returns line numbers)
or kb_grep (which returns line numbers) to locate the relevant section, then
calls file_read with offset/limit to read only that portion.

Sources:
  - document_id: KB or datastore document (RBAC via authorized KBs/datastores).
    Content is read from Document.converted_markdown.
  - file_id: attached chat file (RBAC via chat ownership). Defaults to most
    recent attached file if omitted and no document_id given.
    Content is read from ChatFile.markdown_content.

If neither document_id nor file_id is provided, defaults to the most recent
attached chat file. If no file is attached, returns an error.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.models.chat import ChatFile
from app.services.agentic_rag.tool_context import ToolContext, enforce_rbac, write_audit
from app.services.agentic_rag.token_budget import count_tokens
from app.services.agentic_rag.tools.base import BaseAgentTool
from app.services.agentic_rag.tools.kb_outline import _load_authorized_document

logger = logging.getLogger(__name__)


class FileReadInput(BaseModel):
    document_id: Optional[int] = Field(
        default=None,
        description="KB or datastore document ID. If provided, reads from the KB document.",
    )
    file_id: Optional[int] = Field(
        default=None,
        description="Attached chat file ID. If omitted (and no document_id), defaults to most recent attached file.",
    )
    offset: Optional[int] = Field(
        default=None, ge=1,
        description="Line number to start reading from (1-indexed). If omitted, starts from line 1.",
    )
    limit: Optional[int] = Field(
        default=None, ge=1,
        description="Maximum number of lines to read. If omitted, reads to end of file (subject to max_tokens).",
    )
    max_tokens: int = Field(
        default=50000, ge=500,
        description="Token budget for returned content. If exceeded, content is truncated and a continuation hint is returned.",
    )


def _resolve_chat_file(ctx: ToolContext, file_id: Optional[int]) -> tuple[Optional[ChatFile], Optional[dict]]:
    """Resolve an attached chat file with RBAC. Returns (file, None) or (None, error)."""
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


class FileReadTool(BaseAgentTool):
    name: str = "file_read"
    ui_label: str = "Reading file"
    description: str = (
        "Read a portion of a KB document or attached chat file by line range. "
        "Use kb_outline or kb_grep first to find the right line numbers, then "
        "call file_read with offset/limit to read only what you need."
    )
    prompt_snippet: str = "Read KB document or attached file content (line range)"
    prompt_guidelines: list[str] = [
        "file_read: Use for targeted reads after locating content via kb_outline, kb_grep, or search results. Read only the required lines with offset/limit; use larger limits only when full-document context is genuinely needed.",
        "file_read: If the response includes a continuation_hint, call again with the suggested offset to read the next portion.",
        "file_read: Use document_id for KB documents, file_id for attached chat files. If neither is provided, defaults to the most recent attached file.",
    ]
    args_schema: type[BaseModel] = FileReadInput

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use arun() for agent tools.")

    async def _execute(self, input_obj: FileReadInput) -> dict:
        t0 = time.monotonic()
        ctx: ToolContext = self.ctx

        # ── Source resolution ──────────────────────────────────────────────
        if input_obj.document_id is not None:
            doc, error = await _load_authorized_document(ctx, input_obj.document_id)
            if error:
                return error
            markdown = doc.converted_markdown or ""
            if not markdown:
                return {"ok": False, "result": {}, "error": "Document has no converted markdown.", "tokens": 0}
            source_type = "kb"
            doc_id: Optional[int] = doc.id
            file_id: Optional[int] = None
            title = doc.title or doc.file_name
            file_name = doc.file_name
        else:
            cf, error = _resolve_chat_file(ctx, input_obj.file_id)
            if error:
                return error
            markdown = cf.markdown_content or ""
            if not markdown:
                return {"ok": False, "result": {}, "error": "File has no markdown content.", "tokens": 0}
            source_type = "chat"
            doc_id = None
            file_id = cf.id
            title = cf.file_name
            file_name = cf.file_name

        # ── Line-range slicing (pi-style offset/limit) ─────────────────────
        lines = markdown.split("\n")
        total_lines = len(lines)
        start_idx = (input_obj.offset or 1) - 1  # 0-indexed
        if start_idx >= total_lines:
            return {
                "ok": False,
                "result": {},
                "error": f"offset {input_obj.offset} exceeds total lines {total_lines}.",
                "tokens": 0,
            }
        end_idx = total_lines
        if input_obj.limit is not None:
            end_idx = min(start_idx + input_obj.limit, total_lines)

        content = "\n".join(lines[start_idx:end_idx])

        # ── Token truncation ───────────────────────────────────────────────
        tokens = count_tokens(content)
        truncated = False
        if tokens > input_obj.max_tokens:
            max_chars = input_obj.max_tokens * 4
            content = content[:max_chars]
            # Recompute end line based on actual content length
            actual_lines = content.count("\n") + 1
            end_idx = start_idx + actual_lines
            truncated = True
            tokens = count_tokens(content)

        start_line = start_idx + 1  # 1-indexed
        end_line = end_idx

        # ── Continuation hint (pi-style) ───────────────────────────────────
        continuation_hint = ""
        if end_line < total_lines or truncated:
            next_offset = end_line + 1
            continuation_hint = (
                f"[Showing lines {start_line}-{end_line} of {total_lines}. "
                f"Use offset={next_offset} to continue.]"
            )

        # ── Citation ref ───────────────────────────────────────────────────
        citation_kind = "range" if (input_obj.offset is not None or input_obj.limit is not None) else "file"
        citation_ref = {
            "document_id": doc_id,
            "citation_kind": citation_kind,
            "chunk_index": None,
            "section": None,
            "start_char": None,
            "end_char": None,
            "start_line": start_line,
            "end_line": end_line,
            "page": None,
            "match_line": None,
            "quoted_text": content[:200],
            "source_tool": "file_read",
            "citation_id": "",
        }

        latency_ms = round((time.monotonic() - t0) * 1000)
        write_audit(
            ctx, "file_read", input_obj.model_dump(),
            {
                "source_type": source_type,
                "document_id": doc_id,
                "file_id": file_id,
                "start_line": start_line,
                "end_line": end_line,
                "total_lines": total_lines,
                "truncated": truncated,
            },
            latency_ms=latency_ms, status="ok",
        )

        return {
            "ok": True,
            "result": {
                "source_type": source_type,
                "document_id": doc_id,
                "file_id": file_id,
                "title": title,
                "file_name": file_name,
                "content": content,
                "total_tokens": tokens,
                "truncated": truncated,
                "start_line": start_line,
                "end_line": end_line,
                "total_lines": total_lines,
                "continuation_hint": continuation_hint,
                "citation_ref": citation_ref,
            },
            "error": None,
            "tokens": tokens,
        }
