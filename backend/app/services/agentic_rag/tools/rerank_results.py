"""Cross-encoder reranker tool — dedups and reranks hits from search tools."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from langchain_core.documents import Document as LangchainDocument
from pydantic import BaseModel, Field

from app.services.agentic_rag.tool_context import ToolContext, write_audit
from app.services.agentic_rag.tools.base import BaseAgentTool
from app.services.retrieval.reranker import rerank
from app.services.retrieval.retrieval import dedup_by_content_hash, semantic_dedup
from app.services.settings_service import get_setting

logger = logging.getLogger(__name__)


class RerankResultsInput(BaseModel):
    query: str = Field(description="The search query to rerank against.")
    top_n: Optional[int] = Field(default=None, description="Maximum hits after reranking. If None, all hits passing the threshold are returned (no hard cap).")


class RerankResultsTool(BaseAgentTool):
    name: str = "rerank_results"
    description: str = (
        "Cross-encoder reranker. Deduplicates and reranks all retrieved docs from state. "
        "Call AFTER one or more search tools when you have multiple hits and need to prioritize. "
        "Only pass the query — the reranker reads hits from state automatically. "
        "No hard top_n cap — all hits passing the threshold are returned."
    )
    args_schema: type = RerankResultsInput
    ui_label: str = "Reranking results"

    async def _execute(self, input_obj: RerankResultsInput) -> dict:
        ctx = self.ctx
        if ctx is None:
            return {"ok": False, "result": {}, "error": "No context", "tokens": 0, "terminate": False}

        # Read hits directly from state — the LLM never passes hits as
        # arguments. This avoids fabricated hits and saves output tokens.
        retrieved_docs = []
        if ctx.state is not None:
            retrieved_docs = ctx.state.get("retrieved_docs", [])

        if not retrieved_docs:
            return {"ok": True, "result": {"hits": [], "query_used": input_obj.query, "input_count": 0, "output_count": 0, "best_score": 0.0}, "error": "No retrieved docs to rerank. Call a search tool first.", "tokens": 0, "terminate": False}

        # Convert retrieved_docs to dict format for dedup functions
        docs_as_dicts = []
        for doc in retrieved_docs:
            if not isinstance(doc, dict):
                continue
            meta = doc.get("metadata", {}) or {}
            docs_as_dicts.append({
                "page_content": doc.get("page_content", ""),
                "metadata": {
                    "document_id": meta.get("document_id"),
                    "chunk_index": meta.get("chunk_index"),
                    "page": meta.get("page"),
                    "title": meta.get("title", ""),
                    "file_name": meta.get("file_name", ""),
                    "content_hash": meta.get("content_hash", ""),
                    "qdrant_point_id": meta.get("qdrant_point_id", ""),
                    "citation_ref": meta.get("citation_ref", {}),
                },
            })

        # Dedup by content hash, then semantic dedup
        deduped = dedup_by_content_hash(docs_as_dicts)
        dedup_threshold = get_setting(ctx.db, "DEDUP_SEMANTIC_THRESHOLD", ctx.org_id)
        deduped = semantic_dedup(deduped, threshold=dedup_threshold)

        # Convert to LangchainDocument for rerank()
        lc_docs = []
        for d in deduped:
            lc_docs.append(LangchainDocument(
                page_content=d["page_content"],
                metadata=d["metadata"],
            ))

        # Rerank with cross-encoder (threshold from settings)
        score_threshold = get_setting(ctx.db, "RERANKER_SCORE_THRESHOLD", ctx.org_id)
        reranked = rerank(
            query=input_obj.query,
            docs=lc_docs,
            score_threshold=score_threshold,
            db=ctx.db,
            org_id=ctx.org_id,
        )

        # Apply top_n if specified (no hard cap otherwise)
        if input_obj.top_n is not None:
            reranked = reranked[:input_obj.top_n]

        # Convert back to hit dicts with updated CitationRef
        hits = []
        for doc in reranked:
            meta = doc.metadata or {}
            citation_ref = meta.get("citation_ref", {})
            citation_ref["source_tool"] = "rerank_results"
            hit = {
                "document_id": meta.get("document_id"),
                "chunk_index": meta.get("chunk_index"),
                "page": meta.get("page"),
                "title": meta.get("title", ""),
                "file_name": meta.get("file_name", ""),
                "content": doc.page_content,
                "_reranker_score": meta.get("_reranker_score", 0.0),
                "content_hash": meta.get("content_hash", ""),
                "qdrant_point_id": meta.get("qdrant_point_id", ""),
                "citation_ref": citation_ref,
            }
            hits.append(hit)

        scores = [h.get("_reranker_score", 0) for h in hits]
        write_audit(ctx, "rerank_results", input_obj.model_dump(),
                     {"input_count": len(retrieved_docs), "output_count": len(hits),
                      "best_score": max(scores) if scores else 0.0}, status="ok")

        return {
            "ok": True,
            "result": {
                "hits": hits,
                "query_used": input_obj.query,
                "input_count": len(retrieved_docs),
                "output_count": len(hits),
                "best_score": max(scores) if scores else 0.0,
                "threshold": score_threshold,
            },
            "error": None,
            "tokens": sum(len(h["content"]) for h in hits) // 4,
            "terminate": False,
        }
