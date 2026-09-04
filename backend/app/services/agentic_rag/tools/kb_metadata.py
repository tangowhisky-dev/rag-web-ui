"""kb_metadata tool — introspect KB document metadata for filtering.

Lets the agent discover what documents exist and what fields it can filter
on before calling rag_retrieve. This makes the agent autonomous — it doesn't
need the user to pre-filter.

Actions:
  list_fields    — static schema of filterable fields
  unique_values  — distinct values for a field (e.g. all titles)
  date_range     — min/max dates for a field
  list_documents — recent documents with title, filename, date, type
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.models.knowledge import Document
from app.services.agentic_rag.tool_context import ToolContext, enforce_rbac, write_audit
from app.services.agentic_rag.tools.base import BaseAgentTool

logger = logging.getLogger(__name__)

# Static schema returned by list_fields — no DB query needed.
_FILTERABLE_FIELDS = {
    "fields": [
        {"name": "title", "type": "text", "filter": "title_contains"},
        {"name": "file_name", "type": "text", "filter": "file_name_contains"},
        {"name": "content_type", "type": "text", "filter": "content_type"},
        {"name": "created_at", "type": "date", "filter": "created_after/created_before"},
        {"name": "file_modified_at", "type": "date", "filter": "modified_after/modified_before"},
        {"name": "file_size", "type": "int", "filter": "document_ids"},
        {"name": "document_id", "type": "int", "filter": "document_ids"},
    ]
}

# Fields that support unique_values and date_range actions.
_TEXT_FIELDS = {"title", "file_name", "content_type"}
_DATE_FIELDS = {"created_at", "file_modified_at"}


class KbMetadataInput(BaseModel):
    action: str = Field(
        description=(
            "One of: list_fields, unique_values, date_range, list_documents. "
            "list_fields: returns available filter fields (no field needed). "
            "unique_values: returns distinct values for a field. "
            "date_range: returns min/max dates for a field. "
            "list_documents: returns recent documents."
        )
    )
    field: Optional[str] = Field(
        default=None,
        description="Field name for unique_values or date_range. Required for those actions.",
    )
    value_contains: Optional[str] = Field(
        default=None,
        description="Filter unique_values results to those containing this substring.",
    )
    limit: int = Field(default=50, ge=1, le=200, description="Max results for unique_values or list_documents.")
    kb_ids: Optional[list[int]] = Field(default=None, description="Specific KBs; default all authorized KBs for this chat.")


class KbMetadataTool(BaseAgentTool):
    name: str = "kb_metadata"
    ui_label: str = "Inspecting KB metadata"
    description: str = (
        "Inspect knowledge base metadata to discover what documents exist and "
        "what fields you can filter on. Call BEFORE rag_retrieve when the query "
        "implies filtering by title, date, file type, or filename. "
        "Actions: list_fields (available filter fields), "
        "unique_values (distinct values for a field), "
        "date_range (min/max dates for a field), "
        "list_documents (recent documents with title, filename, date, type)."
    )
    args_schema: type[BaseModel] = KbMetadataInput

    async def _execute(self, input_obj: KbMetadataInput) -> dict:
        t0 = time.monotonic()
        ctx: ToolContext = self.ctx

        rbac = enforce_rbac(ctx, kb_ids=input_obj.kb_ids)
        kb_ids = rbac["kb_ids"]
        if not kb_ids and ctx.state is not None:
            kb_ids = ctx.state.get("kb_ids", [])

        if not kb_ids:
            return _empty_metadata_result(input_obj, t0, "No knowledge bases available")

        action = input_obj.action.strip().lower()

        if action == "list_fields":
            result = _FILTERABLE_FIELDS
        elif action == "unique_values":
            result = _unique_values(ctx.db, kb_ids, input_obj)
        elif action == "date_range":
            result = _date_range(ctx.db, kb_ids, input_obj)
        elif action == "list_documents":
            result = _list_documents(ctx.db, kb_ids, input_obj)
        else:
            return _empty_metadata_result(input_obj, t0, f"Unknown action: {action}")

        latency_ms = round((time.monotonic() - t0) * 1000)
        write_audit(ctx, "kb_metadata", input_obj.model_dump(), result, tokens_in=0, tokens_out=0, status="ok", latency_ms=latency_ms)

        return {"ok": True, "result": result, "error": None, "tokens": 0}


def _empty_metadata_result(input_obj: KbMetadataInput, t0: float, reason: str) -> dict:
    latency_ms = round((time.monotonic() - t0) * 1000)
    return {
        "ok": True,
        "result": {"error": reason},
        "error": None,
        "tokens": 0,
    }


def _unique_values(db, kb_ids: list[int], input_obj: KbMetadataInput) -> dict:
    """Return distinct values for a text field."""
    field = input_obj.field
    if not field or field not in _TEXT_FIELDS:
        return {"error": f"unique_values requires field in {sorted(_TEXT_FIELDS)}"}

    col = getattr(Document, field, None)
    if col is None:
        return {"error": f"Unknown field: {field}"}

    q = db.query(col).filter(Document.knowledge_base_id.in_(kb_ids)).distinct()
    if input_obj.value_contains:
        q = q.filter(col.ilike(f"%{input_obj.value_contains}%"))
    values = [v for v, in q.limit(input_obj.limit).all() if v]
    return {"field": field, "values": values, "count": len(values)}


def _date_range(db, kb_ids: list[int], input_obj: KbMetadataInput) -> dict:
    """Return min/max dates for a date field."""
    field = input_obj.field
    if not field or field not in _DATE_FIELDS:
        return {"error": f"date_range requires field in {sorted(_DATE_FIELDS)}"}

    col = getattr(Document, field, None)
    if col is None:
        return {"error": f"Unknown field: {field}"}

    from sqlalchemy import func
    row = db.query(func.min(col), func.max(col)).filter(Document.knowledge_base_id.in_(kb_ids)).first()
    min_val, max_val = row if row else (None, None)

    def _iso(v):
        return v.isoformat() if hasattr(v, "isoformat") else str(v) if v else None

    return {"field": field, "min": _iso(min_val), "max": _iso(max_val)}


def _list_documents(db, kb_ids: list[int], input_obj: KbMetadataInput) -> dict:
    """Return recent documents with metadata."""
    rows = (
        db.query(Document.id, Document.title, Document.file_name, Document.content_type, Document.file_created_at, Document.file_modified_at)
        .filter(Document.knowledge_base_id.in_(kb_ids))
        .order_by(Document.file_modified_at.desc())
        .limit(input_obj.limit)
        .all()
    )

    def _iso(v):
        return v.isoformat() if hasattr(v, "isoformat") else str(v) if v else None

    docs = [
        {
            "id": r.id,
            "title": r.title or r.file_name,
            "file_name": r.file_name,
            "content_type": r.content_type,
            "file_created_at": _iso(r.file_created_at),
            "file_modified_at": _iso(r.file_modified_at),
        }
        for r in rows
    ]
    return {"documents": docs, "count": len(docs)}
