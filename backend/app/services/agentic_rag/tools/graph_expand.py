"""Neo4j graph expansion tool — finds related chunks via entity relationships."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from langchain_core.documents import Document as LangchainDocument
from pydantic import BaseModel, Field

from app.services.agentic_rag.tool_context import ToolContext, enforce_rbac, write_audit
from app.services.agentic_rag.tools.base import BaseAgentTool
from app.services.graph.expand import expand_docs_via_graph
from app.services.retrieval import get_effective_datastore_ids

from ._search_helpers import enrich_hits_with_authority

logger = logging.getLogger(__name__)


class GraphExpandInput(BaseModel):
    kb_ids: List[int] = Field(default_factory=list, description="Knowledge base IDs to search within.")
    seed_entity_names: List[str] = Field(default_factory=list, description="Named seed entities from the retrieved evidence.")
    rel_type: Optional[str] = Field(default=None, description="Optional relationship type to follow (e.g. REPORTS_TO, DEPENDS_ON, GOVERNS).")
    target_entity_names: List[str] = Field(default_factory=list, description="Optional target-entity hints to narrow the far end of the path.")
    hops: int = Field(default=1, ge=1, le=3, description="Number of entity-relationship hops to traverse (default 1, max 3).")
    top_k: int = Field(default=10, description="Maximum expanded chunks to return.")


class GraphExpandTool(BaseAgentTool):
    name: str = "graph_expand"
    description: str = "Targeted Neo4j graph expansion. Follows typed entity relationships from named seed entities."
    prompt_snippet: str = "Retrieve graph-connected knowledge (Neo4j entity relationships)"
    prompt_guidelines: list[str] = [
        "graph_expand: Use only when the answer depends on a relationship or multi-hop connection that direct retrieval cannot establish. Pass the seed entity names in seed_entity_names, a relationship type in rel_type when it is clear, and target_entity_names when the far entity is known. hops defaults to 1; use 2 or 3 only for explicit multi-hop connection questions. Do not expand weak/noisy seeds or just because the query contains multiple entities.",
    ]
    args_schema: type = GraphExpandInput
    ui_label: str = "Expanding via graph"

    def prepare_arguments(self, args: dict) -> dict:
        """Normalize list inputs and scalar fields."""
        for key in ("kb_ids", "seed_entity_names", "target_entity_names"):
            val = args.get(key, [])
            if isinstance(val, str):
                val = [v.strip() for v in val.split(",") if v.strip()]
            if not isinstance(val, list):
                val = [val] if val is not None else []
            args[key] = val
        args["kb_ids"] = [int(k) for k in args["kb_ids"]]
        return args

    async def _execute(self, input_obj: GraphExpandInput) -> dict:
        ctx = self.ctx
        if ctx is None:
            return {"ok": False, "result": {}, "error": "No context", "tokens": 0, "terminate": False}

        rbac = enforce_rbac(ctx, kb_ids=input_obj.kb_ids)
        kb_ids = rbac["kb_ids"]
        if not kb_ids and ctx.state is not None:
            kb_ids = ctx.state.get("kb_ids", [])
        if not kb_ids:
            return {"ok": True, "result": {"hits": [], "count": 0}, "error": None, "tokens": 0, "terminate": False}

        if not ctx.state:
            return {"ok": True, "result": {"hits": [], "count": 0}, "error": None, "tokens": 0, "terminate": False}

        # Build the set of already-retrieved chunks to exclude from expansion.
        # Also collect seed document IDs for the legacy fallback when the LLM
        # does not supply explicit seed entity names.
        retrieved_docs = ctx.state.get("retrieved_docs", [])
        seed_document_ids = list({
            (d.get("metadata", {}) or {}).get("document_id")
            for d in retrieved_docs
            if isinstance(d, dict) and (d.get("metadata", {}) or {}).get("document_id") is not None
        })

        if not input_obj.seed_entity_names and not seed_document_ids:
            return {"ok": True, "result": {"hits": [], "count": 0}, "error": None, "tokens": 0, "terminate": False}

        from app.services.infrastructure import get_qdrant_client
        from qdrant_client.models import Filter, FieldCondition, MatchAny

        datastore_ids = get_effective_datastore_ids(kb_ids, ctx.org_id, ctx.db) if ctx.db else []

        # Prefer qdrant_point_id already present in retrieved evidence.
        seed_docs: list[LangchainDocument] = []
        for d in retrieved_docs:
            meta = d.get("metadata", {}) or {}
            pid = meta.get("qdrant_point_id")
            if pid:
                seed_docs.append(LangchainDocument(
                    page_content="",
                    metadata={"qdrant_point_id": str(pid)},
                ))

        # Legacy fallback: scroll Qdrant to get all chunks for the seed documents
        # when the LLM has not provided named seed entities and we have no points.
        if not input_obj.seed_entity_names and not seed_docs:
            client = get_qdrant_client()
            collections = [f"kb_{kb_id}" for kb_id in kb_ids]
            if datastore_ids:
                collections += [f"ds_{ds_id}" for ds_id in datastore_ids]

            for collection in collections:
                try:
                    points, _ = client.scroll(
                        collection_name=collection,
                        scroll_filter=Filter(
                            must=[
                                FieldCondition(
                                    key="document_id",
                                    match=MatchAny(any=seed_document_ids),
                                )
                            ]
                        ),
                        limit=100,
                        with_payload=False,
                        with_vectors=False,
                    )
                    for pt in points:
                        seed_docs.append(LangchainDocument(
                            page_content="",
                            metadata={"qdrant_point_id": str(pt.id)},
                        ))
                except Exception as exc:
                    logger.debug("[graph_expand] scroll collection %s failed: %s", collection, exc)

        try:
            expanded = expand_docs_via_graph(
                docs=seed_docs,
                kb_ids=kb_ids,
                db=ctx.db,
                org_id=ctx.org_id,
                datastore_ids=datastore_ids,
                seed_entity_names=input_obj.seed_entity_names,
                rel_type=input_obj.rel_type,
                target_entity_names=input_obj.target_entity_names,
                hops=input_obj.hops,
            )
        except Exception as exc:
            logger.warning("[graph_expand] failed: %s", exc)
            return {"ok": True, "result": {"hits": [], "count": 0}, "error": None, "tokens": 0, "terminate": False}

        hits = []
        for doc in expanded[:input_obj.top_k]:
            meta = doc.metadata or {}
            hit = {
                "document_id": meta.get("document_id"),
                "chunk_index": meta.get("chunk_index"),
                "page": meta.get("page"),
                "title": meta.get("title", ""),
                "file_name": meta.get("file_name", ""),
                "content": doc.page_content,
                "content_hash": meta.get("content_hash", ""),
                "qdrant_point_id": meta.get("qdrant_point_id", ""),
                "graph_path": meta.get("_graph_path"),
                "citation_ref": {
                    "document_id": meta.get("document_id"),
                    "citation_kind": "chunk",
                    "chunk_index": meta.get("chunk_index"),
                    "page": meta.get("page"),
                    "quoted_text": doc.page_content[:200],
                    "source_tool": "graph_expand",
                    "citation_id": "",
                },
            }
            hits.append(hit)

        hits = enrich_hits_with_authority(hits, ctx.db)

        write_audit(ctx, "graph_expand", input_obj.model_dump(),
                     {"hit_count": len(hits)}, status="ok")

        return {
            "ok": True,
            "result": {"hits": hits, "count": len(hits)},
            "error": None,
            "tokens": sum(len(h["content"]) for h in hits) // 4,
            "terminate": False,
        }
