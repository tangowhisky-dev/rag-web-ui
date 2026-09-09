"""Keyword search tool — merges MySQL FTS + SPLADE sparse vector search.

Runs both backends, deduplicates by content hash, and returns the merged
result set ranked by score. The model doesn't choose between strict and
expanded keyword matching — it just asks for keyword matches.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from app.services.agentic_rag.tool_context import ToolContext, enforce_rbac, write_audit
from app.services.agentic_rag.tools.base import BaseAgentTool
from app.services.retrieval import get_effective_datastore_ids
from app.services.retrieval.retrieval import exact_search_docs, sparse_search_docs
from app.services.settings_service import get_setting

from ._search_helpers import _emit_progress, expand_synonyms, resolve_filter_to_doc_ids

logger = logging.getLogger(__name__)


class KeywordSearchInput(BaseModel):
    query: str = Field(description="Search query — keywords, terms, code, identifiers, or distinctive phrases.")
    kb_ids: List[int] = Field(default_factory=list, description="Knowledge base IDs to search.")
    document_ids: Optional[List[int]] = Field(default=None, description="Restrict to these document IDs.")
    filters: Optional[dict] = Field(default=None, description="Metadata filters: title_contains, file_name_contains, content_type, file_modified_after, file_modified_before, file_created_after, file_created_before.")
    top_k: int = Field(default=20, description="Maximum hits to return after merge and dedup.")


class KeywordSearchTool(BaseAgentTool):
    name: str = "keyword_search"
    description: str = "Hybrid keyword search across chunk text. Runs strict MySQL full-text search and expanded SPLADE sparse matching, merges and deduplicates results. Best as the first search when the query contains specific technical terms, identifiers, acronyms, code, error messages, jargon, or distinctive phrases; the SPLADE expansion also captures related keyword overlaps. Prefer semantic_search only when the question is fully paraphrased or contains no specific technical terms."
    prompt_snippet: str = "Keyword retrieval (strict + expanded, merged)"
    prompt_guidelines: list[str] = [
        "keyword_search: Best as the first search when the query contains identifiers, acronyms, code, error messages, jargon, or distinctive terminology. It runs strict MySQL FTS plus SPLADE sparse expansion, so it also captures related keyword overlaps.",
        "keyword_search: Prefer keyword_search over semantic_search when the query includes any specific technical term, even if the overall question is conceptual.",
        "keyword_search: Fall back to semantic_search if keyword_search returns weak or irrelevant results.",
    ]
    args_schema: type = KeywordSearchInput
    ui_label: str = "Searching (keyword)"

    def prepare_arguments(self, args: dict) -> dict:
        """Normalize kb_ids to list of ints."""
        kb_ids = args.get("kb_ids", [])
        if isinstance(kb_ids, (str, int)):
            kb_ids = [kb_ids]
        args["kb_ids"] = [int(k) for k in kb_ids]
        return args

    async def _execute(self, input_obj: KeywordSearchInput) -> dict:
        ctx = self.ctx
        if ctx is None:
            return {"ok": False, "result": {}, "error": "No context", "tokens": 0, "terminate": False}

        rbac = enforce_rbac(ctx, kb_ids=input_obj.kb_ids)
        kb_ids = rbac["kb_ids"]
        if not kb_ids and ctx.state is not None:
            kb_ids = ctx.state.get("kb_ids", [])
        if not kb_ids:
            return {"ok": True, "result": {"hits": [], "query_used": input_obj.query, "search_type": "keyword", "count": 0}, "error": None, "tokens": 0, "terminate": False}

        datastore_ids = get_effective_datastore_ids(kb_ids, ctx.org_id, ctx.db) if ctx.db else []

        doc_ids = input_obj.document_ids
        if input_obj.filters:
            doc_ids = resolve_filter_to_doc_ids(ctx.db, kb_ids, input_obj.filters)
            if doc_ids is not None:
                _emit_progress("filtering", f"Filtering to {len(doc_ids)} matching documents …")
                if not doc_ids:
                    return {"ok": True, "result": {"hits": [], "query_used": input_obj.query, "search_type": "keyword", "count": 0}, "error": None, "tokens": 0, "terminate": False}

        # Synonym expansion (Redis-cached) — keyword search benefits from variants
        query, extra_queries = await expand_synonyms(input_obj.query, ctx)

        # Run both backends
        exact_min = get_setting(ctx.db, "EXACT_MIN_SCORE", ctx.org_id)
        sparse_min = get_setting(ctx.db, "SPARSE_MIN_SCORE", ctx.org_id)

        all_docs: list = []
        errors: list[str] = []

        try:
            exact_docs = exact_search_docs(
                query=query,
                kb_ids=kb_ids,
                datastore_ids=datastore_ids,
                db=ctx.db,
                org_id=ctx.org_id,
                top_k=input_obj.top_k,
                min_score=exact_min,
                doc_ids=doc_ids,
                extra_queries=extra_queries,
            )
            all_docs.extend(exact_docs)
        except Exception as exc:
            logger.warning("[keyword_search] exact leg failed: %s", exc)
            errors.append(f"exact: {exc}")

        try:
            sparse_docs = sparse_search_docs(
                query=query,
                kb_ids=kb_ids,
                datastore_ids=datastore_ids,
                db=ctx.db,
                org_id=ctx.org_id,
                top_k=input_obj.top_k,
                min_score=sparse_min,
                doc_ids=doc_ids,
                extra_queries=extra_queries,
            )
            all_docs.extend(sparse_docs)
        except Exception as exc:
            logger.warning("[keyword_search] sparse leg failed: %s", exc)
            errors.append(f"sparse: {exc}")

        if not all_docs and errors:
            return {"ok": False, "result": {}, "error": "; ".join(errors), "tokens": 0, "terminate": False}

        # Deduplicate by content_hash, keep highest score per hash
        seen: dict[str, Any] = {}
        for doc in all_docs:
            meta = doc.metadata or {}
            h = meta.get("content_hash", "")
            if not h:
                h = doc.page_content[:200]
            score = meta.get("score", 0.0)
            if h not in seen or score > seen[h].metadata.get("score", 0.0):
                seen[h] = doc

        # Sort by score descending, take top_k
        merged = sorted(seen.values(), key=lambda d: d.metadata.get("score", 0.0), reverse=True)[:input_obj.top_k]

        hits = []
        for doc in merged:
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
                    "source_tool": "keyword_search",
                    "citation_id": "",
                },
            }
            hits.append(hit)

        write_audit(ctx, "keyword_search", input_obj.model_dump(),
                     {"hit_count": len(hits), "exact_count": len(exact_docs) if 'exact_docs' in dir() else 0,
                      "sparse_count": len(sparse_docs) if 'sparse_docs' in dir() else 0},
                     status="ok")

        return {
            "ok": True,
            "result": {
                "hits": hits,
                "query_used": query,
                "search_type": "keyword",
                "count": len(hits),
            },
            "error": None,
            "tokens": sum(len(h["content"]) for h in hits) // 4,
            "terminate": False,
        }
