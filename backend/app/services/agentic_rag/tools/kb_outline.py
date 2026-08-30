"""kb_outline tool — return heading structure of a KB document.

Gives the agent a "table of contents" so it can decide which section to
read with kb_read. Pure regex parse of converted_markdown — no LLM call.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.models.knowledge import Document
from app.services.agentic_rag.tool_context import ToolContext, enforce_rbac, write_audit
from app.services.agentic_rag.token_budget import count_tokens
from app.services.agentic_rag.tools.base import BaseAgentTool

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


class KbOutlineInput(BaseModel):
    document_id: int = Field(description="Document ID from rag_retrieve results, kb_grep matches, or kb_outline.")


class KbOutlineTool(BaseAgentTool):
    name: str = "kb_outline"
    ui_label: str = "Reading document outline"
    description: str = (
        "Get the heading structure (table of contents) of a KB document. "
        "Returns heading levels, text, and character offsets. Use after "
        "kb_grep to see which sections exist before reading with kb_read."
    )
    args_schema: type[BaseModel] = KbOutlineInput

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use arun() for agent tools.")

    async def _execute(self, input_obj: KbOutlineInput) -> dict:
        t0 = time.monotonic()
        ctx: ToolContext = self.ctx

        doc, error = await _load_authorized_document(ctx, input_obj.document_id)
        if error:
            return error

        markdown = doc.converted_markdown or ""
        headings = [
            {"level": len(m.group(1)), "text": m.group(2).strip(), "char_offset": m.start()}
            for m in _HEADING_RE.finditer(markdown)
        ]

        latency_ms = round((time.monotonic() - t0) * 1000)
        write_audit(ctx, "kb_outline", input_obj.model_dump(),
                     {"document_id": doc.id, "heading_count": len(headings)},
                     latency_ms=latency_ms, status="ok")

        return {
            "ok": True,
            "result": {
                "document_id": doc.id,
                "title": doc.title or doc.file_name,
                "file_name": doc.file_name,
                "headings": headings,
                "total_chars": len(markdown),
            },
            "error": None,
            "tokens": count_tokens(str(headings)),
        }


# ── Shared RBAC + document loader ──────────────────────────────────────────────

async def _load_authorized_document(ctx: ToolContext, document_id: int) -> tuple[Optional[Document], Optional[dict]]:
    """Load a document and verify it belongs to an authorized KB or datastore.

    Returns (document, None) on success, (None, error_dict) on failure.
    """
    rbac = enforce_rbac(ctx)
    authorized_kb_ids = set(rbac["kb_ids"])
    if not authorized_kb_ids:
        return None, {"ok": False, "result": {}, "error": "No authorized knowledge bases for this chat.", "tokens": 0}

    from app.services.retrieval.retrieval import get_effective_datastore_ids
    authorized_ds_ids = set(get_effective_datastore_ids(list(authorized_kb_ids), ctx.org_id, ctx.db))

    doc = ctx.db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        return None, {"ok": False, "result": {}, "error": f"Document {document_id} not found.", "tokens": 0}

    doc_kb = doc.knowledge_base_id
    doc_ds = doc.data_store_id
    if (doc_kb is None or doc_kb not in authorized_kb_ids) and (doc_ds is None or doc_ds not in authorized_ds_ids):
        logger.warning("RBAC: document %s not in authorized KBs/datastores for chat %s", document_id, ctx.chat_id)
        return None, {"ok": False, "result": {}, "error": "Access denied to document.", "tokens": 0}

    return doc, None
