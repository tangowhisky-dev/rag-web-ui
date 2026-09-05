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

logger = logging.getLogger(__name__)


class GraphExpandInput(BaseModel):
    kb_ids: List[int] = Field(default_factory=list, description="Knowledge base IDs to search within.")
    seed_document_ids: Optional[List[int]] = Field(default=None, description="Document IDs to use as seeds. The tool will find their chunks and traverse the graph.")
    seed_chunk_ids: Optional[List[str]] = Field(default=None, description="Qdrant point UUIDs to use as seeds directly.")
    top_k: int = Field(default=10, description="Maximum expanded chunks to return.")


class GraphExpandTool(BaseAgentTool):
    name: str = "graph_expand"
    description: str = (
        "Graph expansion via Neo4j. Finds related chunks through entity relationships. "
        "Call when initial search results are insufficient and the KB has graph data. "
        "Pass seed_document_ids or seed_chunk_ids from prior search results."
    )
    args_schema: type = GraphExpandInput
    ui_label: str = "Expanding via graph"

    def prepare_arguments(self, args: dict) -> dict:
        """Normalize kb_ids to list of ints."""
        kb_ids = args.get("kb_ids", [])
        if isinstance(kb_ids, (str, int)):
            kb_ids = [kb_ids]
        args["kb_ids"] = [int(k) for k in kb_ids]
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

        if not input_obj.seed_document_ids and not input_obj.seed_chunk_ids:
            return {"ok": True, "result": {"hits": [], "count": 0}, "error": None, "tokens": 0, "terminate": False}

        from app.services.infrastructure import get_qdrant_client
        from qdrant_client.models import Filter, FieldCondition, MatchAny

        datastore_ids = get_effective_datastore_ids(kb_ids, ctx.org_id, ctx.db) if ctx.db else []

        # Build seed docs with qdrant_point_id in metadata
        seed_docs: list[LangchainDocument] = []

        if input_obj.seed_chunk_ids:
            for pid in input_obj.seed_chunk_ids:
                seed_docs.append(LangchainDocument(
                    page_content="",
                    metadata={"qdrant_point_id": str(pid)},
                ))

        if input_obj.seed_document_ids:
            # Scroll Qdrant to find chunk point IDs for these documents
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
                                    match=MatchAny(any=input_obj.seed_document_ids),
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

        if not seed_docs:
            return {"ok": True, "result": {"hits": [], "count": 0}, "error": None, "tokens": 0, "terminate": False}

        try:
            expanded = expand_docs_via_graph(
                docs=seed_docs,
                kb_ids=kb_ids,
                db=ctx.db,
                org_id=ctx.org_id,
                datastore_ids=datastore_ids,
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

        write_audit(ctx, "graph_expand", input_obj.model_dump(),
                     {"hit_count": len(hits)}, status="ok")

        return {
            "ok": True,
            "result": {"hits": hits, "count": len(hits)},
            "error": None,
            "tokens": sum(len(h["content"]) for h in hits) // 4,
            "terminate": False,
        }
