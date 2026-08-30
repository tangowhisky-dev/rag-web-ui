"""kb_grep tool — keyword/regex search across KB document markdown.

Last-resort exploration tool for the agent when rag_retrieve's sufficiency
check fails. Searches raw converted markdown (not chunks, not embeddings)
for exact terms or patterns that vector similarity may have missed.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Optional

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.models.knowledge import Document
from app.services.agentic_rag.tool_context import ToolContext, enforce_rbac, write_audit
from app.services.agentic_rag.token_budget import count_tokens
from app.services.agentic_rag.tools.base import BaseAgentTool

logger = logging.getLogger(__name__)


def _safe_writer():
    """Return the LangGraph stream writer if available, else None."""
    try:
        from langgraph.config import get_stream_writer
        return get_stream_writer()
    except (RuntimeError, KeyError, ImportError):
        return None


def _emit_progress(phase: str, message: str, **extra: Any) -> None:
    writer = _safe_writer()
    if writer:
        payload: dict[str, Any] = {"event": "progress", "phase": phase, "message": message}
        payload.update(extra)
        writer(payload)


class KbGrepInput(BaseModel):
    pattern: str = Field(description="Search term or regex pattern to find in document text.")
    kb_ids: Optional[list[int]] = Field(default=None, description="Specific KBs to search; default all authorized KBs for this chat.")
    document_ids: Optional[list[int]] = Field(default=None, description="Restrict to specific documents; default all documents in authorized KBs.")
    max_results: int = Field(default=50, ge=1, le=200, description="Maximum matching lines to return.")
    case_insensitive: bool = Field(default=True, description="Case-insensitive matching.")


class KbGrepTool(BaseAgentTool):
    name: str = "kb_grep"
    ui_label: str = "Searching KB documents"
    description: str = (
        "Search for exact terms or regex patterns across all documents in authorized "
        "knowledge bases. Returns matching lines with document IDs and line numbers. "
        "Use as a last resort when rag_retrieve returns insufficient=false and you "
        "need to find specific keywords that vector search may have missed."
    )
    args_schema: type[BaseModel] = KbGrepInput

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use arun() for agent tools.")

    async def _execute(self, input_obj: KbGrepInput) -> dict:
        t0 = time.monotonic()
        ctx: ToolContext = self.ctx

        rbac = enforce_rbac(ctx, kb_ids=input_obj.kb_ids)
        kb_ids = rbac["kb_ids"]
        if not kb_ids:
            return {"ok": False, "result": {}, "error": "No authorized knowledge bases for this chat.", "tokens": 0}

        # Resolve datastore IDs linked to authorized KBs.
        from app.services.retrieval.retrieval import get_effective_datastore_ids
        ds_ids = get_effective_datastore_ids(kb_ids, ctx.org_id, ctx.db)

        # Build document query scoped to authorized KBs + datastores.
        from sqlalchemy import or_
        query = ctx.db.query(Document).filter(
            or_(
                Document.knowledge_base_id.in_(kb_ids),
                Document.data_store_id.in_(ds_ids) if ds_ids else False,
            )
        )
        if input_obj.document_ids:
            query = query.filter(Document.id.in_(input_obj.document_ids))

        documents = query.all()
        if not documents:
            return {
                "ok": True,
                "result": {"matches": [], "total_matches": 0, "documents_searched": 0, "pattern": input_obj.pattern},
                "error": None,
                "tokens": 0,
            }

        _emit_progress("kb_grep", f"Searching {len(documents)} documents …")

        flags = re.IGNORECASE if input_obj.case_insensitive else 0
        try:
            regex = re.compile(input_obj.pattern, flags)
        except re.error as exc:
            return {"ok": False, "result": {}, "error": f"Invalid regex pattern: {exc}", "tokens": 0}

        matches: list[dict] = []
        for doc in documents:
            markdown = doc.converted_markdown or ""
            if not markdown:
                continue
            for line_num, line in enumerate(markdown.splitlines(), 1):
                if regex.search(line):
                    matches.append({
                        "document_id": doc.id,
                        "title": doc.title or doc.file_name,
                        "file_name": doc.file_name,
                        "line_number": line_num,
                        "line_text": line.strip()[:200],
                    })
                    if len(matches) >= input_obj.max_results:
                        break
            if len(matches) >= input_obj.max_results:
                break

        latency_ms = round((time.monotonic() - t0) * 1000)
        result_summary = {
            "total_matches": len(matches),
            "documents_searched": len(documents),
            "pattern": input_obj.pattern,
        }
        write_audit(ctx, "kb_grep", input_obj.model_dump(), result_summary, latency_ms=latency_ms, status="ok")

        tokens = count_tokens(str(matches))
        return {
            "ok": True,
            "result": {
                "matches": matches,
                **result_summary,
            },
            "error": None,
            "tokens": tokens,
        }
