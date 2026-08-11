"""rag_retrieve tool — wraps the existing 3-leg retrieval pipeline.

Implements a graduated relaxation ladder: if the first pass isn't sufficient,
retry with progressively looser leg/reranker thresholds instead of leaving
the decision to "try again" entirely to the calling LLM.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.agentic_rag.nodes import (
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
    min_confidence: Optional[float] = Field(
        default=None,
        description="Confidence bar (0-1) below which the graduated relaxation ladder kicks in. Defaults to settings.ADAPTIVE_RETRIEVAL_THRESHOLD/100.",
    )


class _RagRetrieveTool(BaseTool):
    """Search the knowledge base using dense, sparse, exact and graph legs."""

    name: str = "rag_retrieve"
    ui_label: str = "Retrieving from knowledge base"
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


def _is_sufficient(docs: list, confidence: float, min_confidence: float) -> bool:
    return len(docs) >= 3 and confidence > min_confidence


# Graduated relaxation ladder. Level 0 is the normal, tightest pass. Each
# subsequent level loosens leg minimums and the reranker filter threshold.
# `min_score=None` means "use the leg's default from settings".
# Resolved per-request via the settings service (org-overridable).
def _relaxation_levels(db, org_id) -> list[dict[str, Any]]:
    from app.services.settings_service import get_setting
    dense_min = get_setting(db, "DENSE_MIN_SCORE", org_id)
    sparse_min = get_setting(db, "SPARSE_MIN_SCORE", org_id)
    exact_min = get_setting(db, "EXACT_MIN_SCORE", org_id)
    return [
        {
            "dense_min_score": None,
            "sparse_min_score": None,
            "exact_min_score": None,
            "rerank_threshold": None,
        },
        {
            "dense_min_score": max(0.0, dense_min - 0.15),
            "sparse_min_score": sparse_min * 0.5,
            "exact_min_score": exact_min * 0.5,
            "rerank_threshold": get_setting(db, "ADAPTIVE_RETRIEVAL_RERANKER_THRESHOLD", org_id),
        },
        {
            "dense_min_score": 0.0,
            "sparse_min_score": 0.0,
            "exact_min_score": 0.0,
            "rerank_threshold": get_setting(db, "RETRIEVAL_RELAX_LEVEL2_RERANKER_THRESHOLD", org_id),
        },
    ]


async def _run_retrieval_pass(
    ctx: ToolContext,
    query: str,
    kb_ids: list[int],
    org_id: Optional[int],
    file_markdown: Optional[str],
    legs: list[str],
    level: dict[str, Any],
) -> dict:
    """Run one dense+sparse+exact+rerank+filter pass at a given relaxation level.

    Graph expansion is intentionally NOT part of this pass — it's a separate,
    more expensive step only invoked by the caller when this pass alone isn't
    sufficient (see Issue #6: skip graph expansion when already sufficient).
    """
    state: dict[str, Any] = {
        "rewritten_query": query,
        "original_query": query,
        "kb_ids": kb_ids,
        "org_id": org_id,
        "file_markdown": file_markdown,
    }

    coros = []
    if "dense" in legs:
        coros.append(dense_retrieval_node(state, ctx.db, kb_ids, org_id, file_markdown, min_score=level["dense_min_score"]))
    if "sparse" in legs:
        coros.append(sparse_retrieval_node(state, ctx.db, kb_ids, org_id, file_markdown, min_score=level["sparse_min_score"]))
    if "exact" in legs:
        coros.append(exact_retrieval_node(state, ctx.db, kb_ids, org_id, file_markdown, min_score=level["exact_min_score"]))

    leg_results = await asyncio.gather(*coros, return_exceptions=True)
    for r in leg_results:
        if isinstance(r, Exception):
            logger.warning("[rag_retrieve] leg failed: %s", r)
        else:
            state.update(r)

    state.update(merge_node(state, file_markdown))
    state.update(reranking_node(state))
    state.update(filter_node(state, threshold=level["rerank_threshold"]))
    return state


async def _rag_retrieve(ctx: ToolContext, input_obj: RagRetrieveInput) -> dict:
    t0 = time.monotonic()
    rbac = enforce_rbac(ctx, kb_ids=input_obj.kb_ids)
    kb_ids = rbac["kb_ids"]
    if not kb_ids and ctx.state is not None:
        kb_ids = ctx.state.get("kb_ids", [])
    org_id = ctx.org_id
    file_markdown = None
    if ctx.state is not None:
        file_markdown = ctx.state.get("file_markdown", None)

    legs = input_obj.legs or ["dense", "sparse", "exact"]
    from app.services.settings_service import get_setting
    min_confidence = (
        input_obj.min_confidence
        if input_obj.min_confidence is not None
        else get_setting(ctx.db, "ADAPTIVE_RETRIEVAL_THRESHOLD", ctx.org_id) / 100.0
    )

    all_levels = _relaxation_levels(ctx.db, ctx.org_id)
    adaptive_enabled = get_setting(ctx.db, "ADAPTIVE_RETRIEVAL_ENABLED", ctx.org_id)
    levels = all_levels if adaptive_enabled else all_levels[:1]

    state: dict[str, Any] = {}
    docs: list = []
    confidence = 0.0
    levels_tried = 0
    for i, level in enumerate(levels):
        levels_tried = i + 1
        state = await _run_retrieval_pass(ctx, input_obj.query, kb_ids, org_id, file_markdown, legs, level)
        docs = state.get("retrieved_docs", [])
        confidence = float(state.get("retrieval_confidence", 0.0))
        if _is_sufficient(docs, confidence, min_confidence):
            break

        # Not sufficient from vector/sparse/exact legs alone at this level —
        # graph expansion is worth its latency cost now. Recheck afterward.
        if input_obj.graph_expand:
            try:
                neo4j = await neo4j_expansion_node(state, ctx.db, kb_ids, org_id, file_markdown)
                state.update(neo4j)
                docs = state.get("retrieved_docs", docs)
                confidence = float(state.get("retrieval_confidence", confidence))
            except Exception as exc:
                logger.warning("[rag_retrieve] graph expansion failed: %s", exc)

        if _is_sufficient(docs, confidence, min_confidence):
            break

        logger.info(
            "[rag_retrieve] level %d insufficient (docs=%d confidence=%.2f) — %s",
            i, len(docs), confidence,
            "trying next relaxation level" if i < len(levels) - 1 else "no more levels, giving up",
        )

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
        "levels_tried": levels_tried,
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
            "levels_tried": levels_tried,
            "sufficient": _is_sufficient(docs, confidence, min_confidence),
        },
        "error": None,
        "tokens": len(str(docs)) // 4,
    }


def make_rag_retrieve_tool(ctx: ToolContext) -> _RagRetrieveTool:
    tool = _RagRetrieveTool()
    tool.ctx = ctx
    return tool


RagRetrieveTool = _RagRetrieveTool
