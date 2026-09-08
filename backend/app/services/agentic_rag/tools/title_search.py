"""title_search tool — document-level retrieval by title/filename/metadata.

Queries the documents table directly (no chunks, no Qdrant, no reranker).
Finds documents by title/filename/date filters, deduplicates same-title
versions (keeps latest by file_modified_at → file_created_at), and returns
the full converted_markdown of each matching document. This is the primary
retrieval strategy for document-specific queries like "what is in the latest
weekly update" — the reranker scores chunks against the query, but for named
documents the user wants the full file, not the top-scoring fragments.

Supports metadata_only mode for discovery/aggregate queries: returns just
title, dates, and type without markdown content, so the LLM can discover
many documents without blowing up the context window.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field
from sqlalchemy import or_

from app.models.knowledge import Document
from app.services.agentic_rag.tool_context import ToolContext, enforce_rbac, write_audit
from app.services.agentic_rag.token_budget import count_tokens
from app.services.agentic_rag.tools.base import BaseAgentTool
from app.services.retrieval.retrieval import get_effective_datastore_ids

logger = logging.getLogger(__name__)


def _parse_date(s: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


class TitleSearchInput(BaseModel):
    title_contains: Optional[str] = Field(
        default=None,
        description="Case-insensitive substring to match against document titles "
        "or file names. Example: 'Weekly Update' matches 'Weekly Update Aug 21-28'. "
        "If null, matches all documents (use with date filters for broad queries).",
    )
    kb_ids: Optional[list[int]] = Field(default=None, description="Optional KB id override.")
    content_type: Optional[str] = Field(
        default=None,
        description="Filter by MIME type, e.g. 'application/pdf'.",
    )
    modified_after: Optional[str] = Field(
        default=None,
        description="ISO date string (e.g. '2026-01-01'). Only return documents with "
        "file_modified_at >= this date. Use for 'this year', 'since June', etc.",
    )
    modified_before: Optional[str] = Field(
        default=None,
        description="ISO date string (e.g. '2026-12-31'). Only return documents with "
        "file_modified_at <= this date.",
    )
    sort_field: str = Field(
        default="file_modified_at",
        description="Metadata field to sort by: 'file_modified_at', 'file_created_at', 'title', 'file_name'.",
    )
    sort_direction: str = Field(default="desc", description="Sort direction: 'desc' (newest first) or 'asc'.")
    top_n: int = Field(
        default=3, ge=1,
        description="Max documents to return after deduplication. Reason about this "
        "based on the query: 3 for 'latest' queries, 10-20 for comparing a few "
        "versions, 50+ for aggregate queries that need all matching documents. "
        "Use metadata_only=true when requesting many documents to avoid token overflow.",
    )
    max_tokens_per_doc: int = Field(
        default=50000, ge=500,
        description="Token budget per document. The full markdown is truncated if it exceeds this. "
        "Set high to read full documents, or low to skim. If truncated, use file_read to read the rest.",
    )
    metadata_only: bool = Field(
        default=False,
        description="If true, return only title, file_name, file_modified_at, file_created_at, "
        "content_type, document_id — no markdown content. Use for discovery queries "
        "('how many documents match X', 'list all weekly updates') to save tokens. "
        "Follow up with a second call (metadata_only=false) to read specific documents.",
    )


class TitleSearchTool(BaseAgentTool):
    name: str = "title_search"
    ui_label: str = "Searching documents by title"
    description: str = "Find and read full documents by title, filename, content type, or date range. Queries the document table directly — no chunk retrieval, no reranking. Returns complete converted markdown."
    prompt_snippet: str = "Retrieve documents by title/filename/metadata"
    prompt_guidelines: list[str] = [
        "title_search: Best for finding documents by title, filename, type, author, or date. Use metadata_only=true for discovery or aggregation; use full content for content questions.",
        "title_search: For conceptual queries that don't name a specific document, use semantic_search or keyword_search instead.",
    ]
    args_schema: type[BaseModel] = TitleSearchInput

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use arun() for agent tools.")

    async def _execute(self, input_obj: TitleSearchInput) -> dict:
        t0 = time.monotonic()
        ctx: ToolContext = self.ctx

        rbac = enforce_rbac(ctx, kb_ids=input_obj.kb_ids)
        kb_ids = rbac["kb_ids"]
        if not kb_ids and ctx.state is not None:
            kb_ids = ctx.state.get("kb_ids", [])
        if not kb_ids:
            return {"ok": False, "result": {}, "error": "No authorized knowledge bases for this chat.", "tokens": 0}

        # Build query — search KB documents plus documents in datastores
        # linked to those KBs. The datastore association is resolved
        # internally via get_effective_datastore_ids so the agent never
        # sees which datastore collections are being searched.
        ds_ids = get_effective_datastore_ids(kb_ids, ctx.org_id, ctx.db)
        q = ctx.db.query(Document).filter(
            or_(
                Document.knowledge_base_id.in_(kb_ids),
                Document.data_store_id.in_(ds_ids) if ds_ids else False,
            )
        )

        if input_obj.title_contains:
            q = q.filter(or_(
                Document.title.ilike(f"%{input_obj.title_contains}%"),
                Document.file_name.ilike(f"%{input_obj.title_contains}%"),
            ))
        if input_obj.content_type:
            q = q.filter(Document.content_type == input_obj.content_type)
        if input_obj.modified_after:
            after = _parse_date(input_obj.modified_after)
            if after:
                q = q.filter(Document.file_modified_at >= after)
        if input_obj.modified_before:
            before = _parse_date(input_obj.modified_before)
            if before:
                q = q.filter(Document.file_modified_at <= before)

        # Sort
        sort_col = getattr(Document, input_obj.sort_field, Document.file_modified_at)
        if input_obj.sort_direction == "asc":
            q = q.order_by(sort_col.asc())
        else:
            q = q.order_by(sort_col.desc())

        rows = q.limit(200).all()
        if not rows:
            logger.debug(
                "[title_search] no documents matching title=%r modified_after=%r modified_before=%r",
                input_obj.title_contains, input_obj.modified_after, input_obj.modified_before,
            )
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
        latest_per_title: dict[str, Document] = {}
        for doc in rows:
            title_key = (doc.title or doc.file_name or "").strip().lower()
            if not title_key:
                title_key = f"__untitled_{doc.id}"
            # rows are already sorted by the sort_field desc, so the first
            # occurrence per title is the latest version.
            if title_key not in latest_per_title:
                latest_per_title[title_key] = doc

        # Sort deduplicated docs by file_modified_at → file_created_at desc and cap at top_n.
        selected = sorted(
            latest_per_title.values(),
            key=lambda d: d.file_modified_at or d.file_created_at or d.created_at,
            reverse=True,
        )[:input_obj.top_n]

        # Build doc dicts.
        docs_result: list[dict] = []
        total_tokens = 0
        for doc in selected:
            file_created_iso = (doc.file_created_at or doc.created_at).isoformat() if (doc.file_created_at or doc.created_at) else ""
            file_modified_iso = (doc.file_modified_at or doc.file_created_at or doc.created_at).isoformat() if (doc.file_modified_at or doc.file_created_at or doc.created_at) else ""

            if input_obj.metadata_only:
                doc_dict = {
                    "page_content": "",
                    "metadata": {
                        "document_id": doc.id,
                        "title": doc.title or doc.file_name,
                        "file_name": doc.file_name,
                        "content_type": doc.content_type,
                        "file_created_at": file_created_iso,
                        "_file_created_at": file_created_iso,
                        "_file_modified_at": file_modified_iso,
                        "source": "title_search",
                    },
                }
                docs_result.append(doc_dict)
                continue

            markdown = doc.converted_markdown or ""
            if not markdown:
                logger.debug("[title_search] doc %d has no converted_markdown, skipping", doc.id)
                continue

            tokens = count_tokens(markdown)
            truncated = False
            if tokens > input_obj.max_tokens_per_doc:
                max_chars = input_obj.max_tokens_per_doc * 4
                markdown = markdown[:max_chars]
                tokens = count_tokens(markdown)
                truncated = True

            doc_dict = {
                "page_content": markdown,
                "metadata": {
                    "document_id": doc.id,
                    "title": doc.title or doc.file_name,
                    "file_name": doc.file_name,
                    "content_type": doc.content_type,
                    "file_created_at": file_created_iso,
                    "_file_created_at": file_created_iso,
                    "_file_modified_at": file_modified_iso,
                    "source": "title_search",
                    "_reranker_score": 1.0,
                    "truncated": truncated,
                    "total_tokens": tokens,
                    "citation_ref": {
                        "document_id": doc.id,
                        "citation_kind": "file",
                        "quoted_text": (markdown or "")[:200],
                        "source_tool": "title_search",
                        "citation_id": "",
                    },
                },
            }
            docs_result.append(doc_dict)
            total_tokens += tokens

        latency_ms = round((time.monotonic() - t0) * 1000)
        logger.debug(
            "[title_search] title=%r | matched=%d | selected=%d | metadata_only=%s | tokens=%d | latency=%dms",
            input_obj.title_contains, len(rows), len(docs_result), input_obj.metadata_only, total_tokens, latency_ms,
        )

        write_audit(
            ctx, "title_search", input_obj.model_dump(),
            {
                "title_contains": input_obj.title_contains,
                "matched_documents": len(rows),
                "returned_documents": len(docs_result),
                "top_n": input_obj.top_n,
                "metadata_only": input_obj.metadata_only,
            },
            tokens_in=0, tokens_out=total_tokens, status="ok", latency_ms=latency_ms,
        )

        return {
            "ok": True,
            "result": {
                "docs": docs_result,
                "document_count": len(docs_result),
                "total_matching": len(latest_per_title),
                "title_contains": input_obj.title_contains,
                "top_n": input_obj.top_n,
                "metadata_only": input_obj.metadata_only,
            },
            "error": None,
            "tokens": total_tokens,
        }
