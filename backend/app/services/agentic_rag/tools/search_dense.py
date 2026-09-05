"""Dense vector search tool — semantic/conceptual chunk retrieval."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from app.services.agentic_rag.tool_context import ToolContext, enforce_rbac, write_audit
from app.services.agentic_rag.tools.base import BaseAgentTool
from app.services.retrieval import get_effective_datastore_ids
from app.services.retrieval.retrieval import dense_search_docs
from app.services.settings_service import get_setting

from ._search_helpers import _emit_progress, resolve_filter_to_doc_ids

logger = logging.getLogger(__name__)


class SearchDenseInput(BaseModel):
    query: str = Field(description="Search query for semantic/conceptual matching.")
    kb_ids: List[int] = Field(default_factory=list, description="Knowledge base IDs to search.")
    document_ids: Optional[List[int]] = Field(default=None, description="Restrict to these document IDs.")
    filters: Optional[dict] = Field(default=None, description="Metadata filters: title_contains, file_name_contains, content_type, file_modified_after, file_modified_before, file_created_after, file_created_before.")
    top_k: int = Field(default=20, description="Maximum hits to return.")


class SearchDenseTool(BaseAgentTool):
    name: str = "search_dense"
    description: str = (
        "Dense vector search. Best for semantic/conceptual matching. "
        "Returns ranked chunks with scores and citation metadata."
    )
    args_schema: type = SearchDenseInput
    ui_label: str = "Searching (dense)"

    def prepare_arguments(self, args: dict) -> dict:
        """Normalize kb_ids to list of ints."""
        kb_ids = args.get("kb_ids", [])
        if isinstance(kb_ids, (str, int)):
            kb_ids = [kb_ids]
        args["kb_ids"] = [int(k) for k in kb_ids]
        return args

    async def _execute(self, input_obj: SearchDenseInput) -> dict:
        ctx = self.ctx
        if ctx is None:
            return {"ok": False, "result": {}, "error": "No context", "tokens": 0, "terminate": False}

        rbac = enforce_rbac(ctx, kb_ids=input_obj.kb_ids)
        kb_ids = rbac["kb_ids"]
        if not kb_ids and ctx.state is not None:
            kb_ids = ctx.state.get("kb_ids", [])
        if not kb_ids:
            return {"ok": True, "result": {"hits": [], "query_used": input_obj.query, "search_type": "dense", "count": 0}, "error": None, "tokens": 0, "terminate": False}

        datastore_ids = get_effective_datastore_ids(kb_ids, ctx.org_id, ctx.db) if ctx.db else []

        doc_ids = input_obj.document_ids
        if input_obj.filters:
            doc_ids = resolve_filter_to_doc_ids(ctx.db, kb_ids, input_obj.filters)
            if doc_ids is not None:
                _emit_progress("filtering", f"Filtering to {len(doc_ids)} matching documents …")
                if not doc_ids:
                    return {"ok": True, "result": {"hits": [], "query_used": input_obj.query, "search_type": "dense", "count": 0}, "error": None, "tokens": 0, "terminate": False}

        min_score = get_setting(ctx.db, "DENSE_MIN_SCORE", ctx.org_id)

        try:
            docs = dense_search_docs(
                query=input_obj.query,
                kb_ids=kb_ids,
                datastore_ids=datastore_ids,
                db=ctx.db,
                org_id=ctx.org_id,
                top_k=input_obj.top_k,
                min_score=min_score,
                doc_ids=doc_ids,
            )
        except Exception as exc:
            logger.warning("[search_dense] failed: %s", exc)
            return {"ok": False, "result": {}, "error": str(exc), "tokens": 0, "terminate": False}

        hits = []
        for doc in docs:
            meta = doc.metadata or {}
            hit = {
                "document_id": meta.get("document_id"),
                "chunk_index": meta.get("chunk_index"),
                "page": meta.get("page"),
                "title": meta.get("title", ""),
                "file_name": meta.get("file_name", ""),
                "content": doc.page_content,
                "score": meta.get("score", 0.0),
                "content_hash": meta.get("content_hash", ""),
                "qdrant_point_id": meta.get("qdrant_point_id", ""),
                "citation_ref": {
                    "document_id": meta.get("document_id"),
                    "citation_kind": "chunk",
                    "chunk_index": meta.get("chunk_index"),
                    "page": meta.get("page"),
                    "quoted_text": doc.page_content[:200],
                    "source_tool": "search_dense",
                    "citation_id": "",
                },
            }
            hits.append(hit)

        write_audit(ctx, "search_dense", input_obj.model_dump(),
                     {"hit_count": len(hits)}, status="ok")

        return {
            "ok": True,
            "result": {
                "hits": hits,
                "query_used": input_obj.query,
                "search_type": "dense",
                "count": len(hits),
            },
            "error": None,
            "tokens": sum(len(h["content"]) for h in hits) // 4,
            "terminate": False,
        }
