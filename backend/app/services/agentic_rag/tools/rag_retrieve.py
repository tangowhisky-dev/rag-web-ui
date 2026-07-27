"""rag_retrieve tool — wraps the existing 3-leg retrieval pipeline."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.services.agentic_rag.nodes import (
    adaptive_reranking_node,
    dense_retrieval_node,
    exact_retrieval_node,
    filter_node,
    merge_node,
    neo4j_expansion_node,
    reranking_node,
    sparse_retrieval_node,
)
from app.services.agentic_rag.tool_context import ToolContext, enforce_rbac, write_audit

logger = logging.getLogger(__name__)


class RagRetrieveInput(BaseModel):
    """Input schema for rag_retrieve."""

    query: str = Field(description="Search query.")
    kb_ids: Optional[list[int]] = Field(default=None, description="Optional KB id override.")
    datastore_ids: Optional[list[int]] = Field(default=None)
    top_k: Optional[int] = Field(default=None)
    legs: Optional[list[str]] = Field(default=None)
    graph_expand: bool = Field(default=True)
    min_confidence: float = Field(default=0.3)


class _RagRetrieveTool(BaseTool):
    """Search the knowledge base using dense, sparse, exact and graph legs."""

    name: str = "rag_retrieve"
    description: str = (
        "Search the attached knowledge bases. Returns ranked document chunks, "
        "confidence, and sufficiency. Use when the user needs facts from documents."
    )
    args_schema: type[BaseModel] = RagRetrieveInput
    ctx: Optional[ToolContext] = Field(default=None, exclude=True)

    async def _arun(self, *args, **kwargs) -> dict:
        """Dispatch to the internal retrieval pipeline."""
        kwargs.pop("run_manager", None)
        if args and isinstance(args[0], dict):
            kwargs = args[0]
        input_obj = RagRetrieveInput(**kwargs)
        return await _rag_retrieve(self.ctx, input_obj)


    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Synchronous entry-point is not supported; use arun instead."""
        raise NotImplementedError("Use arun() for agent tools.")


async def _rag_retrieve(ctx: ToolContext, input_obj: RagRetrieveInput) -> dict:
    t0 = time.monotonic()
    rbac = enforce_rbac(ctx, kb_ids=input_obj.kb_ids)
    kb_ids = rbac["kb_ids"]
    if not kb_ids and ctx.state is not None:
        kb_ids = getattr(ctx.state, "kb_ids", [])
    org_id = ctx.org_id
    file_markdown = None
    if ctx.state is not None:
        file_markdown = getattr(ctx.state, "file_markdown", None)

    state: dict[str, Any] = {
        "rewritten_query": input_obj.query,
        "original_query": input_obj.query,
        "kb_ids": kb_ids,
        "org_id": org_id,
        "file_markdown": file_markdown,
    }

    legs = input_obj.legs or ["dense", "sparse", "exact"]
    coros = []
    if "dense" in legs:
        coros.append(dense_retrieval_node(state, ctx.db, kb_ids, org_id, file_markdown))
    if "sparse" in legs:
        coros.append(sparse_retrieval_node(state, ctx.db, kb_ids, org_id, file_markdown))
    if "exact" in legs:
        coros.append(exact_retrieval_node(state, ctx.db, kb_ids, org_id, file_markdown))

    leg_results = await asyncio.gather(*coros, return_exceptions=True)
    for r in leg_results:
        if isinstance(r, Exception):
            logger.warning("[rag_retrieve] leg failed: %s", r)
        else:
            state.update(r)

    merge_result = merge_node(state, file_markdown)
    state.update(merge_result)

    rerank_result = reranking_node(state)
    state.update(rerank_result)

    filter_result = filter_node(state)
    state.update(filter_result)

    if input_obj.graph_expand:
        try:
            neo4j = await neo4j_expansion_node(state, ctx.db, kb_ids, org_id, file_markdown)
            state.update(neo4j)
        except Exception as exc:
            logger.warning("[rag_retrieve] graph expansion failed: %s", exc)

    docs = state.get("retrieved_docs", [])
    confidence = float(state.get("retrieval_confidence", 0.0))

    if confidence < input_obj.min_confidence and not state.get("adaptive_reran"):
        adaptive = adaptive_reranking_node(state, ctx.db)
        state.update(adaptive)
        docs = state.get("retrieved_docs", docs)
        confidence = float(state.get("retrieval_confidence", confidence))

    confidence_level = "low"
    if confidence > 0.7:
        confidence_level = "high"
    elif confidence > 0.3:
        confidence_level = "medium"

    latency_ms = round((time.monotonic() - t0) * 1000)
    result_summary = {
        "doc_count": len(docs),
        "confidence": confidence,
        "confidence_level": confidence_level,
    }
    write_audit(
        ctx,
        "rag_retrieve",
        input_obj.model_dump(),
        result_summary,
        tokens_in=0,
        tokens_out=0,
        status="ok",
        latency_ms=latency_ms,
    )

    return {
        "ok": True,
        "result": {
            "docs": docs,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "query_used": input_obj.query,
            "legs_run": legs,
            "sufficient": len(docs) >= 3 and confidence > input_obj.min_confidence,
        },
        "error": None,
        "tokens": len(str(docs)) // 4,
    }


def make_rag_retrieve_tool(ctx: ToolContext) -> _RagRetrieveTool:
    tool = _RagRetrieveTool()
    tool.ctx = ctx
    return tool


RagRetrieveTool = _RagRetrieveTool
