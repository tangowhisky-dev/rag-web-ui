"""Semantic search tool — dense vector search for conceptual/conceptual matching."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from app.services.agentic_rag.tool_context import ToolContext, enforce_rbac, write_audit
from app.services.agentic_rag.tools.base import BaseAgentTool
from app.services.retrieval import get_effective_datastore_ids
from app.services.retrieval.retrieval import dense_search_docs
from app.services.retrieval.reranker import rerank, soft_elbow_truncate
from app.services.settings_service import get_setting

from ._search_helpers import _emit_progress, enrich_hits_with_authority, inject_neighbor_context, resolve_filter_to_doc_ids

logger = logging.getLogger(__name__)


class SemanticSearchInput(BaseModel):
    query: str = Field(description="Search query for semantic/conceptual matching.")
    kb_ids: List[int] = Field(default_factory=list, description="Knowledge base IDs to search.")
    document_ids: Optional[List[int]] = Field(default=None, description="Restrict to these document IDs.")
    filters: Optional[dict] = Field(default=None, description="Metadata filters: title_contains, file_name_contains, content_type, file_modified_after, file_modified_before, file_created_after, file_created_before, document_status (draft|active|superseded; 'obsolete' is accepted), exclude_status, effective_as_of (ISO date — document must be in force on that date), effective_window_start/effective_window_end (validity overlap), effective_from_after/before, effective_to_after/before, version, owner.")
    top_k: int = Field(default=20, description="Maximum hits to return.")


class SemanticSearchTool(BaseAgentTool):
    name: str = "semantic_search"
    description: str = "Dense vector search for semantic/conceptual matching. Finds chunks by meaning, not exact wording. Best for paraphrased, natural-language, or purely conceptual questions that do not contain specific technical identifiers or acronyms. Use as a fallback when keyword_search returns weak or irrelevant results."
    prompt_snippet: str = "Semantic retrieval (dense vectors)"
    prompt_guidelines: list[str] = [
        "semantic_search: Best for conceptual, natural-language, paraphrased, and meaning-based questions that do not contain specific identifiers, acronyms, or distinctive technical terms.",
        "semantic_search: Use when keyword_search returns weak or irrelevant results, or when the user's wording differs substantially from the document wording.",
        "semantic_search: Results are cross-encoder reranked and soft-elbow filtered before returning. No separate rerank call needed.",
        "semantic_search: For 'current/latest/in-force' questions pass filters={\"document_status\":\"active\",\"effective_as_of\":\"<today>\"} (call current_datetime first if needed). Leave unfiltered for history or version comparisons — hits carry document_status/effective-window tags so you can reason about conflicting versions.",
    ]
    args_schema: type = SemanticSearchInput
    ui_label: str = "Searching (semantic)"

    def prepare_arguments(self, args: dict) -> dict:
        """Normalize kb_ids/document_ids to int lists; parse a stringified
        filters dict (some LLMs serialize nested objects as JSON strings)."""
        for key in ("kb_ids", "document_ids"):
            val = args.get(key)
            if val is None:
                continue
            if isinstance(val, (str, int)):
                val = [val]
            try:
                args[key] = [int(k) for k in val]
            except (TypeError, ValueError):
                pass  # leave raw — schema validation reports the bad value
        filt = args.get("filters")
        if isinstance(filt, str):
            try:
                import json as _json
                parsed = _json.loads(filt)
                if isinstance(parsed, dict):
                    args["filters"] = parsed
            except (ValueError, TypeError):
                pass  # leave — schema validation will report it
        return args

    async def _execute(self, input_obj: SemanticSearchInput) -> dict:
        ctx = self.ctx
        if ctx is None:
            return {"ok": False, "result": {}, "error": "No context", "tokens": 0, "terminate": False}

        rbac = enforce_rbac(ctx, kb_ids=input_obj.kb_ids)
        kb_ids = rbac["kb_ids"]
        if not kb_ids and ctx.state is not None:
            kb_ids = ctx.state.get("kb_ids", [])
        if not kb_ids:
            return {"ok": True, "result": {"hits": [], "query_used": input_obj.query, "search_type": "semantic", "count": 0}, "error": None, "tokens": 0, "terminate": False}

        datastore_ids = get_effective_datastore_ids(kb_ids, ctx.org_id, ctx.db) if ctx.db else []

        doc_ids = input_obj.document_ids
        filter_meta: dict = {}
        if input_obj.filters:
            filter_doc_ids, filter_meta = resolve_filter_to_doc_ids(ctx.db, kb_ids, input_obj.filters)
            if filter_doc_ids is not None:
                # Intersect with an explicit document_ids restriction —
                # filters narrowing to zero searchable docs means zero hits,
                # not an unfiltered search.
                doc_ids = (
                    sorted(set(filter_doc_ids) & set(input_obj.document_ids))
                    if input_obj.document_ids else filter_doc_ids
                )
                _emit_progress("filtering", f"Filtering to {len(doc_ids)} matching documents …")
                if not doc_ids:
                    return {"ok": True, "result": {"hits": [], "query_used": input_obj.query, "search_type": "semantic", "count": 0, "matched_documents": filter_meta.get("total_matching") or 0}, "error": None, "tokens": 0, "terminate": False}

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
            logger.warning("[semantic_search] failed: %s", exc)
            return {"ok": False, "result": {}, "error": str(exc), "tokens": 0, "terminate": False}

        # dense_search_docs fetches a pool of top_k * 4 candidates.
        # Rerank with cross-encoder and apply soft-elbow truncation.
        # Skip rerank only when the pool is very small (<= top_k // 2).
        if len(docs) > input_obj.top_k // 2:
            score_threshold = get_setting(ctx.db, "RERANKER_SCORE_THRESHOLD", ctx.org_id)
            elbow_enabled = get_setting(ctx.db, "ELBOW_CUT_ENABLED", ctx.org_id)
            try:
                docs = rerank(
                    query=input_obj.query,
                    docs=docs,
                    score_threshold=score_threshold,
                    db=ctx.db,
                    org_id=ctx.org_id,
                )
                if elbow_enabled:
                    docs = soft_elbow_truncate(docs, max_keep=input_obj.top_k)
            except Exception as exc:
                logger.warning("[semantic_search] rerank failed, using raw scores: %s", exc)
                docs = sorted(docs, key=lambda d: d.metadata.get("score", 0.0), reverse=True)[:input_obj.top_k]

        # Inject prev/next chunks for top evidence and reorder by file position.
        try:
            docs = inject_neighbor_context(docs, ctx.db)
        except Exception as exc:
            logger.warning("[semantic_search] neighbor injection failed: %s", exc)

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
                "_is_neighbor": bool(meta.get("_is_neighbor")),
                "citation_ref": {
                    "document_id": meta.get("document_id"),
                    "citation_kind": "chunk",
                    "chunk_index": meta.get("chunk_index"),
                    "page": meta.get("page"),
                    "quoted_text": doc.page_content[:200],
                    "source_tool": "semantic_search",
                    "citation_id": "",
                },
            }
            hits.append(hit)

        # Tag each hit with the document's lifecycle status / validity window
        # (resolved live from MySQL — safe under post-ingestion edits).
        hits = enrich_hits_with_authority(hits, ctx.db)

        write_audit(ctx, "semantic_search", input_obj.model_dump(),
                     {"hit_count": len(hits)}, status="ok")

        result_payload = {
            "hits": hits,
            "query_used": input_obj.query,
            "search_type": "semantic",
            "count": len(hits),
        }
        if filter_meta.get("total_matching") is not None:
            result_payload["matched_documents"] = filter_meta["total_matching"]
        if filter_meta.get("ignored_keys"):
            result_payload["ignored_filter_keys"] = filter_meta["ignored_keys"]

        return {
            "ok": True,
            "result": result_payload,
            "error": None,
            "tokens": sum(len(h["content"]) for h in hits) // 4,
            "terminate": False,
        }
