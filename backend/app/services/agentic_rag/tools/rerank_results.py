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
    hits: List[dict] = Field(description="Hits from one or more search tools. Each hit is a dict with 'content', 'document_id', 'chunk_index', 'title', 'content_hash', etc.")
    top_n: Optional[int] = Field(default=None, description="Maximum hits after reranking. If None, all hits passing the threshold are returned (no hard cap).")


class RerankResultsTool(BaseAgentTool):
    name: str = "rerank_results"
    description: str = (
        "Cross-encoder reranker. Deduplicates and reranks hits from search tools. "
        "Call AFTER one or more search tools when you have multiple hits and need to prioritize. "
        "Pass the hits array from the search tool observation. If the hits are invalid or empty, "
        "the reranker will automatically use all retrieved docs from state. "
        "No hard top_n cap — all hits passing the threshold are returned."
    )
    args_schema: type = RerankResultsInput
    ui_label: str = "Reranking results"

    async def _execute(self, input_obj: RerankResultsInput) -> dict:
        ctx = self.ctx
        if ctx is None:
            return {"ok": False, "result": {}, "error": "No context", "tokens": 0, "terminate": False}
        if not input_obj.hits:
            return {"ok": True, "result": {"hits": [], "query_used": input_obj.query, "input_count": 0, "output_count": 0, "best_score": 0.0}, "error": None, "tokens": 0, "terminate": False}

        # ── Hit provenance validation ───────────────────────────────────
        # The LLM can fabricate hits with plausible content and fake
        # document_ids. Only accept hits that match documents already in
        # state.retrieved_docs (from actual search/read tools).
        # Validate by content_hash (primary) or document_id+chunk_index.
        retrieved_docs = []
        if ctx.state is not None:
            retrieved_docs = ctx.state.get("retrieved_docs", [])

        # Build lookup indices from retrieved_docs
        valid_hashes: set[str] = set()
        valid_doc_chunks: set[tuple] = set()
        for doc in retrieved_docs:
            if not isinstance(doc, dict):
                continue
            meta = doc.get("metadata", {}) or {}
            ch = meta.get("content_hash", "")
            if ch:
                valid_hashes.add(ch)
            did = meta.get("document_id")
            ci = meta.get("chunk_index")
            if did is not None and ci is not None:
                valid_doc_chunks.add((did, ci))

        validated_hits = []
        rejected_count = 0
        for hit in input_obj.hits:
            if not isinstance(hit, dict):
                rejected_count += 1
                continue
            hit_hash = hit.get("content_hash", "")
            hit_did = hit.get("document_id")
            hit_ci = hit.get("chunk_index")
            is_valid = (
                (hit_hash and hit_hash in valid_hashes) or
                (hit_did is not None and hit_ci is not None and (hit_did, hit_ci) in valid_doc_chunks)
            )
            if is_valid:
                validated_hits.append(hit)
            else:
                rejected_count += 1
                logger.warning(
                    "[rerank_results] rejected fabricated hit: document_id=%s chunk_index=%s content_hash=%s",
                    hit_did, hit_ci, hit_hash[:16] if hit_hash else "none",
                )

        if rejected_count:
            logger.warning(
                "[rerank_results] rejected %d/%d hits not found in retrieved_docs (likely LLM-fabricated)",
                rejected_count, len(input_obj.hits),
            )

        if not validated_hits:
            # Auto-fallback: use retrieved_docs from state instead of failing.
            # The LLM frequently fabricates hits because it can't copy large
            # hit arrays through its context window. Rather than wasting a
            # think round on an error, rerank whatever is in retrieved_docs.
            if retrieved_docs:
                logger.info(
                    "[rerank_results] auto-fallback: reranking %d docs from retrieved_docs",
                    len(retrieved_docs),
                )
                validated_hits = []
                for doc in retrieved_docs:
                    if not isinstance(doc, dict):
                        continue
                    meta = doc.get("metadata", {}) or {}
                    validated_hits.append({
                        "content": doc.get("page_content", ""),
                        "document_id": meta.get("document_id"),
                        "chunk_index": meta.get("chunk_index"),
                        "page": meta.get("page"),
                        "title": meta.get("title", ""),
                        "file_name": meta.get("file_name", ""),
                        "content_hash": meta.get("content_hash", ""),
                        "qdrant_point_id": meta.get("qdrant_point_id", ""),
                        "citation_ref": meta.get("citation_ref", {}),
                        "_reranker_score": meta.get("_reranker_score", meta.get("score", 0.0)),
                    })
            else:
                return {
                    "ok": True,
                    "result": {
                        "hits": [],
                        "query_used": input_obj.query,
                        "input_count": len(input_obj.hits),
                        "output_count": 0,
                        "best_score": 0.0,
                        "rejected_fabricated": rejected_count,
                        "auto_fallback": False,
                    },
                    "error": f"All {len(input_obj.hits)} hits were rejected and no retrieved_docs available. Call a search tool first.",
                    "tokens": 0,
                    "terminate": False,
                }

        # Convert validated hits to dict format for dedup functions
        docs_as_dicts = []
        for hit in validated_hits:
            docs_as_dicts.append({
                "page_content": hit.get("content", ""),
                "metadata": {
                    "document_id": hit.get("document_id"),
                    "chunk_index": hit.get("chunk_index"),
                    "page": hit.get("page"),
                    "title": hit.get("title", ""),
                    "file_name": hit.get("file_name", ""),
                    "content_hash": hit.get("content_hash", ""),
                    "qdrant_point_id": hit.get("qdrant_point_id", ""),
                    "citation_ref": hit.get("citation_ref", {}),
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
                     {"input_count": len(input_obj.hits), "validated_count": len(validated_hits),
                      "rejected_count": rejected_count, "output_count": len(hits),
                      "best_score": max(scores) if scores else 0.0}, status="ok")

        return {
            "ok": True,
            "result": {
                "hits": hits,
                "query_used": input_obj.query,
                "input_count": len(input_obj.hits),
                "validated_count": len(validated_hits),
                "rejected_fabricated": rejected_count,
                "output_count": len(hits),
                "best_score": max(scores) if scores else 0.0,
                "threshold": score_threshold,
            },
            "error": None,
            "tokens": sum(len(h["content"]) for h in hits) // 4,
            "terminate": False,
        }
