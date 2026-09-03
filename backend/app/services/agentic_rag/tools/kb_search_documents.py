"""kb_search_documents tool — document-level retrieval by title.

Queries the documents table directly (no chunks, no Qdrant, no reranker).
Finds documents by title, deduplicates same-title versions (keeps latest
by created_at), and returns the full converted_markdown of each matching
document. This is the primary retrieval strategy for document-specific
queries like "what is in the latest weekly update" — the reranker scores
chunks against the query, but for named documents the user wants the
full file, not the top-scoring fragments.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from pydantic import BaseModel, Field
from sqlalchemy import or_, and_

from app.models.knowledge import Document
from app.services.agentic_rag.tool_context import ToolContext, enforce_rbac, write_audit
from app.services.agentic_rag.token_budget import count_tokens
from app.services.agentic_rag.tools.base import BaseAgentTool

logger = logging.getLogger(__name__)


class KbSearchDocumentsInput(BaseModel):
    title_contains: str = Field(
        description="Case-insensitive substring to match against document titles. "
        "Example: 'Weekly Update' matches 'Weekly Update Aug 21-28'."
    )
    kb_ids: Optional[list[int]] = Field(default=None, description="Optional KB id override.")
    sort_field: str = Field(default="created_at", description="Metadata field to sort by.")
    sort_direction: str = Field(default="desc", description="Sort direction: 'desc' (newest first) or 'asc'.")
    top_n: int = Field(
        default=1, ge=1, le=10,
        description="Number of documents to return after same-title deduplication. "
        "1 = only the latest version. 3 = latest 3 versions (for synthesizing across versions).",
    )
    max_tokens_per_doc: int = Field(
        default=4000, ge=500, le=32000,
        description="Token budget per document. The full markdown is truncated if it exceeds this. "
        "Note: some tokenizers (e.g. Gemma) count tokens at ~3x the rate of the estimator, "
        "so 4000 estimated tokens may be ~12000 actual tokens for the provider.",
    )


class KbSearchDocumentsTool(BaseAgentTool):
    name: str = "kb_search_documents"
    ui_label: str = "Searching documents by title"
    description: str = (
        "Find and read full documents by title from the knowledge base. "
        "Queries the document table directly — no chunk retrieval, no reranking. "
        "Returns the complete converted markdown of matching documents, deduplicated "
        "to the latest version by created_at. Use when the query names a specific "
        "document (e.g. 'weekly update', 'Q3 report', 'onboarding guide') or asks "
        "for the latest/most recent version of a document. For conceptual queries "
        "that don't name a specific document, use rag_retrieve instead."
    )
    args_schema: type[BaseModel] = KbSearchDocumentsInput

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use arun() for agent tools.")

    async def _execute(self, input_obj: KbSearchDocumentsInput) -> dict:
        t0 = time.monotonic()
        ctx: ToolContext = self.ctx

        rbac = enforce_rbac(ctx, kb_ids=input_obj.kb_ids)
        kb_ids = rbac["kb_ids"]
        if not kb_ids and ctx.state is not None:
            kb_ids = ctx.state.get("kb_ids", [])
        if not kb_ids:
            return {"ok": False, "result": {}, "error": "No authorized knowledge bases for this chat.", "tokens": 0}

        from app.services.retrieval.retrieval import get_effective_datastore_ids
        datastore_ids = get_effective_datastore_ids(kb_ids, ctx.org_id, ctx.db) if ctx.db else []

        # Query documents by title (case-insensitive ilike).
        q = ctx.db.query(Document).filter(
            or_(
                Document.knowledge_base_id.in_(kb_ids),
                and_(Document.knowledge_base_id.is_(None), Document.data_store_id.isnot(None)),
            )
        ).filter(
            Document.title.ilike(f"%{input_obj.title_contains}%")
        )

        # Also match on file_name as a fallback — sometimes the title is
        # auto-generated and doesn't contain the user's search term, but
        # the file_name does.
        q = q.filter(
            or_(
                Document.title.ilike(f"%{input_obj.title_contains}%"),
                Document.file_name.ilike(f"%{input_obj.title_contains}%"),
            )
        )

        # Sort
        sort_field = input_obj.sort_field
        sort_col = getattr(Document, sort_field, Document.created_at)
        if input_obj.sort_direction == "asc":
            q = q.order_by(sort_col.asc())
        else:
            q = q.order_by(sort_col.desc())

        rows = q.limit(50).all()
        if not rows:
            logger.debug("[kb_search_documents] no documents matching title=%r", input_obj.title_contains)
            return {
                "ok": True,
                "result": {
                    "docs": [],
                    "document_count": 0,
                    "title_contains": input_obj.title_contains,
                },
                "error": None,
                "tokens": 0,
            }

        # Deduplicate same-title versions: keep only the latest per title.
        # If top_n > 1, keep the top N latest versions per title.
        latest_per_title: dict[str, list[Document]] = {}
        for doc in rows:
            title_key = (doc.title or doc.file_name or "").strip().lower()
            if not title_key:
                title_key = f"__untitled_{doc.id}"
            latest_per_title.setdefault(title_key, []).append(doc)

        selected: list[Document] = []
        for title_key, docs in latest_per_title.items():
            # docs are already sorted by created_at desc from the query
            selected.extend(docs[:input_obj.top_n])

        # Re-sort selected by created_at desc (interleaved across titles)
        selected.sort(key=lambda d: d.created_at or d.updated_at, reverse=True)

        # Build doc dicts with full markdown content.
        docs_result: list[dict] = []
        total_tokens = 0
        for doc in selected:
            markdown = doc.converted_markdown or ""
            if not markdown:
                logger.debug("[kb_search_documents] doc %d has no converted_markdown, skipping", doc.id)
                continue

            tokens = count_tokens(markdown)
            truncated = False
            if tokens > input_obj.max_tokens_per_doc:
                max_chars = input_obj.max_tokens_per_doc * 4
                markdown = markdown[:max_chars]
                tokens = count_tokens(markdown)
                truncated = True

            created_iso = doc.created_at.isoformat() if doc.created_at else ""
            modified_iso = (doc.modified_at or doc.created_at).isoformat() if (doc.modified_at or doc.created_at) else ""

            doc_dict = {
                "page_content": markdown,
                "metadata": {
                    "document_id": doc.id,
                    "title": doc.title or doc.file_name,
                    "file_name": doc.file_name,
                    "content_type": doc.content_type,
                    "created_at": created_iso,
                    "_created_at": created_iso,
                    "_modified_at": modified_iso,
                    "source": "kb_search_documents",
                    "_reranker_score": 1.0,  # Document-level match — full relevance
                    "truncated": truncated,
                    "total_tokens": tokens,
                },
            }
            docs_result.append(doc_dict)
            total_tokens += tokens

        latency_ms = round((time.monotonic() - t0) * 1000)
        logger.debug(
            "[kb_search_documents] title=%r | matched=%d | selected=%d | tokens=%d | latency=%dms",
            input_obj.title_contains, len(rows), len(docs_result), total_tokens, latency_ms,
        )

        write_audit(
            ctx, "kb_search_documents", input_obj.model_dump(),
            {
                "title_contains": input_obj.title_contains,
                "matched_documents": len(rows),
                "returned_documents": len(docs_result),
                "top_n": input_obj.top_n,
            },
            tokens_in=0, tokens_out=total_tokens, status="ok", latency_ms=latency_ms,
        )

        return {
            "ok": True,
            "result": {
                "docs": docs_result,
                "document_count": len(docs_result),
                "title_contains": input_obj.title_contains,
                "top_n": input_obj.top_n,
            },
            "error": None,
            "tokens": total_tokens,
        }
