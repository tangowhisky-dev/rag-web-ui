"""rag_retrieve tool — wraps the existing 3-leg retrieval pipeline.

Implements a graduated relaxation ladder: if the first pass isn't sufficient,
retry with progressively looser leg/reranker thresholds instead of leaving
the decision to "try again" entirely to the calling LLM.

Sufficiency is checked via an LLM-based collection-level evaluation after
each relaxation level. If all levels are exhausted and the result is still
insufficient, the query is rewritten once and the ladder re-runs with the
new query.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.agentic_rag.llm_factory import build_chat_llm
from app.services.agentic_rag.nodes import (
    dense_retrieval_node,
    exact_retrieval_node,
    expand_query_node,
    filter_node,
    merge_node,
    neo4j_expansion_node,
    reranking_node,
    sparse_retrieval_node,
)
from app.services.agentic_rag.tool_context import ToolContext, enforce_rbac, write_audit

logger = logging.getLogger(__name__)


def _safe_writer():
    """Return the LangGraph stream writer if available, else None.

    The rag_retrieve tool runs inside a tool_node (graph context), so
    progress events can be emitted. This helper mirrors the one in
    nodes.py to avoid a cross-module import for a 4-line function.
    """
    try:
        from langgraph.config import get_stream_writer
        return get_stream_writer()
    except (RuntimeError, KeyError, ImportError):
        return None


def _emit_progress(phase: str, message: str, **extra: Any) -> None:
    """Emit a progress event for the UI, following the existing pattern."""
    writer = _safe_writer()
    if writer:
        payload: dict[str, Any] = {"event": "progress", "phase": phase, "message": message}
        payload.update(extra)
        writer(payload)


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
    """Heuristic fallback used when the LLM sufficiency check is unavailable."""
    return len(docs) >= 3 and confidence > min_confidence


def _extract_json_block(text: str) -> str | None:
    """Return the first well-formed JSON object from *text*."""
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        return _extract_balanced(m.group(1), ("{", "}"))
    return _extract_balanced(text, ("{", "}"))


def _extract_balanced(text: str, chars: tuple[str, str]) -> str | None:
    start_char, end_char = chars
    start = text.find(start_char)
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


# ── LLM-based sufficiency check ──────────────────────────────────────────────

_SUFFICIENCY_PROMPT = """\
User question: {query}

Retrieved document excerpts:
{previews}

Do these documents contain sufficient information to fully answer the user's question?
Judge by actual content, not topic similarity. A document about the right topic that \
does not contain the specific answer is NOT sufficient.

Return ONLY a JSON object:
{{"sufficient": true/false, "missing": "what's missing if not sufficient, or empty string"}}

If the documents are sufficient, set "missing" to an empty string.
"""


async def _llm_sufficiency_check(
    query: str,
    docs: list,
    confidence: float,
    ctx: ToolContext,
    min_confidence: float,
) -> tuple[bool, str]:
    """LLM-based check: do these docs contain enough to answer the query?

    Returns ``(sufficient, missing_description)``.
    Falls back to the old heuristic on LLM failure.
    """
    if not docs:
        return False, "No documents were found for this query."

    # Truncate docs to keep the prompt small — top 5 docs × ~500 chars each.
    previews = []
    for i, doc in enumerate(docs[:5]):
        content = str(doc.get("page_content", ""))[:500]
        previews.append(f"[Doc {i + 1}] {content}")
    prompt = _SUFFICIENCY_PROMPT.format(query=query, previews="\n".join(previews))

    _emit_progress("sufficiency_check", "Evaluating retrieval sufficiency …")

    try:
        llm = build_chat_llm(ctx.org_id, ctx.db, role="query", temperature=0.0)
        response = await llm.ainvoke([{"role": "user", "content": prompt}])
        block = _extract_json_block(str(response.content))
        if block:
            result = json.loads(block)
            sufficient = bool(result.get("sufficient", False))
            missing = str(result.get("missing", "")).strip()
            logger.info(
                "[rag_retrieve] LLM sufficiency: sufficient=%s missing=%s",
                sufficient, missing[:100] if missing else "(none)",
            )
            return sufficient, missing
    except Exception as exc:
        logger.warning("[rag_retrieve] sufficiency LLM check failed: %s — falling back to heuristic", exc)

    # Fallback: old heuristic
    heuristic_ok = _is_sufficient(docs, confidence, min_confidence)
    return heuristic_ok, "" if heuristic_ok else "Heuristic check could not confirm sufficiency."


# ── Query rewriting ──────────────────────────────────────────────────────────

_REWRITE_PROMPT = """\
The query "{query}" did not retrieve sufficient documents from the knowledge base.
Missing information: {missing}

Rewrite the query to improve retrieval. Consider:
- Using different terminology or synonyms
- Simplifying overly complex phrasing
- Removing unnecessary qualifiers
- Breaking a multi-part question into a simpler form

Return ONLY the rewritten query string, no explanation, no quotes.
"""


async def _rewrite_query(
    original_query: str,
    missing: str,
    ctx: ToolContext,
) -> str:
    """Rewrite a query that failed to retrieve sufficient docs.

    Returns the rewritten query, or the original if rewriting fails or
    produces an identical string.
    """
    _emit_progress("query_rewrite", "Rewriting query for better retrieval …")

    prompt = _REWRITE_PROMPT.format(query=original_query, missing=missing or "unknown")
    try:
        llm = build_chat_llm(ctx.org_id, ctx.db, role="query", temperature=0.0)
        response = await llm.ainvoke([{"role": "user", "content": prompt}])
        rewritten = str(response.content).strip().strip('"').strip("'").strip()
        if rewritten and rewritten.lower() != original_query.lower():
            logger.info("[rag_retrieve] query rewritten: '%s' -> '%s'", original_query, rewritten)
            return rewritten
    except Exception as exc:
        logger.warning("[rag_retrieve] query rewrite failed: %s", exc)
    return original_query


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

    # Expand abbreviations in the query for retrieval legs.
    # The reranker uses rewritten_query (unexpanded) to preserve user intent.
    state.update(expand_query_node(state, db=ctx.db, org_id=ctx.org_id))

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

    state.update(merge_node(state, file_markdown, ctx.db, ctx.org_id))
    state.update(reranking_node(state))
    state.update(filter_node(state, threshold=level["rerank_threshold"]))
    return state


async def _run_relaxation_ladder(
    ctx: ToolContext,
    query: str,
    kb_ids: list[int],
    org_id: Optional[int],
    file_markdown: Optional[str],
    legs: list[str],
    levels: list[dict[str, Any]],
    min_confidence: float,
    graph_expand: bool,
) -> tuple[dict[str, Any], list, float, int, bool, str]:
    """Run the relaxation ladder for a single query string.

    Returns ``(state, docs, confidence, levels_tried, sufficient, missing)``.
    """
    state: dict[str, Any] = {}
    docs: list = []
    confidence = 0.0
    levels_tried = 0
    sufficient = False
    missing = ""

    for i, level in enumerate(levels):
        levels_tried = i + 1
        state = await _run_retrieval_pass(ctx, query, kb_ids, org_id, file_markdown, legs, level)
        docs = state.get("retrieved_docs", [])
        confidence = float(state.get("retrieval_confidence", 0.0))

        # LLM-based sufficiency check (falls back to heuristic on error).
        if docs:
            sufficient, missing = await _llm_sufficiency_check(query, docs, confidence, ctx, min_confidence)
        if sufficient:
            break

        # Not sufficient from vector/sparse/exact legs alone at this level —
        # graph expansion is worth its latency cost now. Recheck afterward.
        if graph_expand:
            try:
                neo4j = await neo4j_expansion_node(state, ctx.db, kb_ids, org_id, file_markdown)
                state.update(neo4j)
                docs = state.get("retrieved_docs", docs)
                confidence = float(state.get("retrieval_confidence", confidence))
            except Exception as exc:
                logger.warning("[rag_retrieve] graph expansion failed: %s", exc)

            if docs:
                sufficient, missing = await _llm_sufficiency_check(query, docs, confidence, ctx, min_confidence)
        if sufficient:
            break

        logger.info(
            "[rag_retrieve] level %d insufficient (docs=%d confidence=%.2f missing=%s) — %s",
            i, len(docs), confidence, missing[:80] if missing else "(none)",
            "trying next relaxation level" if i < len(levels) - 1 else "no more levels",
        )

    return state, docs, confidence, levels_tried, sufficient, missing


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

    # ── Pass 1: relaxation ladder with the original query ──────────────
    state, docs, confidence, levels_tried, sufficient, missing = await _run_relaxation_ladder(
        ctx, input_obj.query, kb_ids, org_id, file_markdown, legs, levels,
        min_confidence, input_obj.graph_expand,
    )

    # ── Pass 2: if still insufficient, rewrite the query and re-run ────
    query_used = input_obj.query
    query_rewritten = False
    if not sufficient and missing:
        rewritten = await _rewrite_query(input_obj.query, missing, ctx)
        if rewritten != input_obj.query:
            query_used = rewritten
            query_rewritten = True
            _emit_progress(
                "query_rewrite",
                "Retrying with rewritten query …",
                rewritten_query=rewritten,
                original_query=input_obj.query,
            )
            state, docs, confidence, levels_tried, sufficient, missing = await _run_relaxation_ladder(
                ctx, rewritten, kb_ids, org_id, file_markdown, legs, levels,
                min_confidence, input_obj.graph_expand,
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
        "sufficient": sufficient,
        "query_rewritten": query_rewritten,
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
            "query_used": query_used,
            "original_query": input_obj.query,
            "query_rewritten": query_rewritten,
            "legs_run": legs,
            "levels_tried": levels_tried,
            "sufficient": sufficient,
            "missing": missing if not sufficient else "",
        },
        "error": None,
        "tokens": len(str(docs)) // 4,
    }


def make_rag_retrieve_tool(ctx: ToolContext) -> _RagRetrieveTool:
    tool = _RagRetrieveTool()
    tool.ctx = ctx
    return tool


RagRetrieveTool = _RagRetrieveTool
