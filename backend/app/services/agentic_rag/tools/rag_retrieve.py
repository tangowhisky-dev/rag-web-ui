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
from sqlalchemy import or_, and_

from app.core.config import settings
from app.services.agentic_rag.llm_factory import build_chat_llm
from app.services.settings_service import get_setting
from app.services.agentic_rag.nodes import (
    _content_contains_exclusion,
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
from app.services.agentic_rag.prompts import SUFFICIENCY_CHECK_PROMPT, RETRIEVAL_REWRITE_PROMPT, SYNONYM_EXPANSION_PROMPT

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


def _resolve_filter_to_doc_ids(
    db: Any,
    kb_ids: list[int],
    filters: dict | None,
) -> list[int] | None:
    """Translate metadata filters to a list of document_ids via MySQL.

    Returns None when no filters are provided (search all docs).
    Returns an empty list if filters match zero documents.
    """
    if not filters:
        return None

    from app.models.knowledge import Document
    from datetime import datetime as _dt

    q = db.query(Document.id).filter(
        or_(
            Document.knowledge_base_id.in_(kb_ids),
            and_(Document.knowledge_base_id.is_(None), Document.data_store_id.isnot(None)),
        )
    )

    if filters.get("title_contains"):
        q = q.filter(Document.title.ilike(f"%{filters['title_contains']}%"))
    if filters.get("file_name_contains"):
        q = q.filter(Document.file_name.ilike(f"%{filters['file_name_contains']}%"))
    if filters.get("content_type"):
        q = q.filter(Document.content_type == filters["content_type"])
    if filters.get("created_after"):
        try:
            after = _dt.fromisoformat(filters["created_after"])
            q = q.filter(Document.created_at >= after)
        except (ValueError, TypeError):
            pass
    if filters.get("created_before"):
        try:
            before = _dt.fromisoformat(filters["created_before"])
            q = q.filter(Document.created_at <= before)
        except (ValueError, TypeError):
            pass
    if filters.get("document_ids"):
        q = q.filter(Document.id.in_(filters["document_ids"]))

    return [r[0] for r in q.limit(200).all()]


def _sort_merged_docs(docs: list[dict], sort: dict | None) -> list[dict]:
    """Sort merged docs by a metadata field. Falls back to original order on errors."""
    if not sort or not sort.get("field") or not docs:
        return docs
    field = sort["field"]
    reverse = sort.get("direction", "desc") == "desc"
    meta_key = f"_{field}"

    def _sort_key(doc: dict) -> str:
        return doc.get("metadata", {}).get(meta_key, "") or ""

    try:
        return sorted(docs, key=_sort_key, reverse=reverse)
    except Exception:
        return docs


def _empty_result(input_obj: RagRetrieveInput, t0: float, reason: str) -> dict:
    """Build an empty-result response when filters match nothing."""
    latency_ms = round((time.monotonic() - t0) * 1000)
    return {
        "ok": True,
        "result": {
            "docs": [],
            "confidence": 0.0,
            "confidence_level": "none",
            "query_used": input_obj.query,
            "original_query": input_obj.query,
            "query_rewritten": False,
            "legs_run": input_obj.legs or ["dense", "sparse", "exact"],
            "levels_tried": 0,
            "sufficient": False,
            "missing": reason,
            "filters_applied": input_obj.filters,
        },
        "error": None,
        "tokens": 0,
    }


class RagRetrieveInput(BaseModel):
    """Input schema for rag_retrieve."""

    query: str = Field(description="Search query.")
    kb_ids: Optional[list[int]] = Field(default=None, description="Optional KB id override.")
    datastore_ids: Optional[list[int]] = Field(default=None)
    top_k: Optional[int] = Field(default=None)
    legs: Optional[list[str]] = Field(
        default=None,
        description=(
            "Retrieval legs to run. Options: 'dense' (semantic vector search), "
            "'sparse' (SPLADE keyword), 'exact' (MySQL fulltext). "
            "Default: all three. Use ['exact','sparse'] for literal lookups "
            "(filenames, IDs, exact titles). Use ['dense'] for conceptual queries. "
            "The dense leg is automatically skipped if exact+sparse scores are high enough."
        ),
    )
    graph_expand: bool = Field(default=True)
    min_confidence: Optional[float] = Field(
        default=None,
        description="Confidence bar (0-1) below which the graduated relaxation ladder kicks in. Defaults to settings.ADAPTIVE_RETRIEVAL_THRESHOLD/100.",
    )
    filters: Optional[dict] = Field(
        default=None,
        description=(
            "Metadata filters to narrow retrieval before scoring. "
            "All conditions are AND-combined. Supported keys: "
            "title_contains (str), file_name_contains (str), "
            "content_type (str, e.g. 'application/pdf'), "
            "created_after (ISO date string), created_before (ISO date string), "
            "document_ids (list[int]). "
            'Example: {"title_contains": "Weekly Update", "created_after": "2026-06-01"}'
        ),
    )
    sort: Optional[dict] = Field(
        default=None,
        description=(
            "Sort merged results by metadata field before reranking. "
            'Example: {"field": "created_at", "direction": "desc"}. '
            "Use when the query implies recency or ordering ('latest', 'most recent')."
        ),
    )


class _RagRetrieveTool(BaseTool):
    """Search the knowledge base using dense, sparse, exact and graph legs."""

    name: str = "rag_retrieve"
    ui_label: str = "Retrieving from knowledge base"
    description: str = (
        "Search the attached knowledge bases. Returns ranked document chunks, "
        "confidence, and sufficiency. Use when the user needs facts from documents. "
        "Supports metadata filters (title_contains, content_type, created_after/before, "
        "document_ids), sort (by created_at or other metadata fields), and leg selection "
        "(dense/sparse/exact). For literal lookups (filenames, IDs), use legs=['exact','sparse']. "
        "For conceptual queries, use legs=['dense'] or omit (all legs run)."
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

    # Include all retrieved chunks, truncated per-chunk to keep the prompt
    # bounded. The sufficiency check needs to see the full picture — limiting
    # to top-5 can miss chunks from a second document that are needed to
    # answer a multi-document query.
    from app.services.agentic_rag.prompts import SUFFICIENCY_CHECK_USER_PROMPT

    previews = []
    for i, doc in enumerate(docs):
        content = str(doc.get("page_content", ""))[:500]
        previews.append(f"[Doc {i + 1}] {content}")
    user_prompt = SUFFICIENCY_CHECK_USER_PROMPT.format(query=query, previews="\n".join(previews))

    _emit_progress("sufficiency_check", "Evaluating retrieval sufficiency …")

    try:
        llm = build_chat_llm(ctx.org_id, ctx.db, role="query", temperature=0.0)
        response = await llm.ainvoke([
            {"role": "system", "content": SUFFICIENCY_CHECK_PROMPT},
            {"role": "user", "content": user_prompt},
        ])
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

_ALLOWED_FILTER_KEYS = {"title_contains", "file_name_contains", "content_type", "created_after", "created_before", "document_ids"}


def _build_failure_snippets(failed_docs: list | None) -> str:
    """Build top-3 doc snippets for the rewrite prompt."""
    if not failed_docs:
        return "(no documents were returned)"
    snippets = []
    for doc in failed_docs[:3]:
        content = str(doc.get("page_content", ""))[:200]
        title = doc.get("metadata", {}).get("title", "")
        snippets.append(f"[{title}] {content}" if title else content)
    return "\n".join(snippets)


def _parse_rewrite_response(raw: str, original_query: str) -> tuple[str, dict | None]:
    """Parse the LLM rewrite response into (rewritten_query, filter_suggestion).

    Handles both JSON and plain-text responses. Returns (original, None) if
    the response is empty or identical to the original query.
    """
    block = _extract_json_block(raw)
    if block:
        result = json.loads(block)
        rewritten = str(result.get("rewritten_query", "")).strip().strip('"').strip("'").strip()
        filter_suggestion = result.get("filter_suggestion")
        if isinstance(filter_suggestion, dict):
            filter_suggestion = {k: v for k, v in filter_suggestion.items() if k in _ALLOWED_FILTER_KEYS} or None
        else:
            filter_suggestion = None
        if rewritten and rewritten.lower() != original_query.lower():
            return rewritten, filter_suggestion
        if filter_suggestion:
            return original_query, filter_suggestion
        return original_query, None
    # Fallback: treat raw response as plain rewritten query.
    rewritten = raw.strip('"').strip("'").strip()
    if rewritten and rewritten.lower() != original_query.lower():
        return rewritten, None
    return original_query, None


async def _rewrite_query(
    original_query: str,
    missing: str,
    ctx: ToolContext,
    failed_docs: list | None = None,
) -> tuple[str, dict | None]:
    """Rewrite a query that failed to retrieve sufficient docs.

    Returns ``(rewritten_query, filter_suggestion)``. The filter_suggestion
    is a dict suitable for `_resolve_filter_to_doc_ids`, or None when no
    filter is needed. Falls back to ``(original_query, None)`` on failure.
    """
    _emit_progress("query_rewrite", "Rewriting query for better retrieval …")

    top_snippets = _build_failure_snippets(failed_docs)
    prompt = RETRIEVAL_REWRITE_PROMPT.format(
        query=original_query,
        missing=missing or "unknown",
        top_snippets=top_snippets,
    )
    try:
        llm = build_chat_llm(ctx.org_id, ctx.db, role="query", temperature=0.0)
        response = await llm.ainvoke([{"role": "user", "content": prompt}])
        raw = str(response.content).strip()
        rewritten, filter_suggestion = _parse_rewrite_response(raw, original_query)
        if rewritten != original_query or filter_suggestion:
            logger.info("[rag_retrieve] query rewritten: '%s' -> '%s' filter=%s", original_query, rewritten, filter_suggestion)
            return rewritten, filter_suggestion
    except Exception as exc:
        logger.warning("[rag_retrieve] query rewrite failed: %s", exc)
    return original_query, None


# Graduated relaxation ladder. Level 0 is the normal, tightest pass. Level 1
# loosens leg minimums and the reranker filter threshold. After both levels
# fail, the query is rewritten once and the 2-level ladder re-runs.
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
    ]


async def _expand_synonyms(query: str, ctx: ToolContext) -> tuple[str, list[str]]:
    """Expand query with spell-corrected + synonym variants via LLM.

    Uses the same `query` LLM role as rewrite_query_node.
    Cached in Redis (key: synonyms:{org_id}:{sha256(query)}).

    Returns (corrected_query, synonyms). corrected_query is the spell-corrected
    query (or original if no correction needed). synonyms is a list of
    alternative terms (may be empty).
    """
    import hashlib
    import json as _json

    from app.services.agentic_rag.llm_factory import build_chat_llm
    from app.services.settings_service import get_setting

    n = get_setting(ctx.db, "SYNONYM_VARIANTS", ctx.org_id)
    cache_ttl = get_setting(ctx.db, "SYNONYM_CACHE_TTL", ctx.org_id)

    # Check Redis cache
    cache_key = f"synonyms:{ctx.org_id}:{hashlib.sha256(query.encode()).hexdigest()}"
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        try:
            cached = await r.get(cache_key)
            if cached:
                obj = _json.loads(cached)
                return obj.get("corrected", query), obj.get("synonyms", [])
        finally:
            await r.aclose()
    except Exception:
        pass  # Redis unavailable — proceed without cache

    # Call LLM with query role
    try:
        llm = build_chat_llm(ctx.org_id, ctx.db, role="query", temperature=0.0)
        prompt = SYNONYM_EXPANSION_PROMPT.format(n=n)
        resp = await llm.ainvoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ])
        raw = resp.content if isinstance(resp.content, str) else str(resp.content)
        # Parse JSON from response
        import re as _re
        json_match = _re.search(r'\{[^{}]*\}', raw, _re.DOTALL)
        if not json_match:
            return query, []
        obj = _json.loads(json_match.group())
        corrected = obj.get("corrected_query") or query
        synonyms = obj.get("queries") or []
        # Filter out empty strings and the original query
        synonyms = [s for s in synonyms if s and s.lower() != query.lower() and s.lower() != corrected.lower()]
    except Exception as exc:
        logger.warning("[rag_retrieve] synonym expansion failed: %s", exc)
        return query, []

    # Cache result
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        try:
            await r.setex(cache_key, cache_ttl, _json.dumps({"corrected": corrected, "synonyms": synonyms}))
        finally:
            await r.aclose()
    except Exception:
        pass

    logger.debug("[rag_retrieve] synonyms for %r: corrected=%r, synonyms=%s", query, corrected, synonyms)
    return corrected, synonyms


async def _run_retrieval_pass(
    ctx: ToolContext,
    query: str,
    kb_ids: list[int],
    org_id: Optional[int],
    file_markdown: Optional[str],
    legs: list[str],
    level: dict[str, Any],
    doc_ids: Optional[list[int]] = None,
    sort: Optional[dict] = None,
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

    # Expand synonyms for sparse/exact legs (not dense — embeddings handle semantics).
    # Uses the same query LLM role as rewrite_query_node. Cached in Redis.
    extra_queries: list[str] = []
    corrected_query = None
    if "sparse" in legs or "exact" in legs:
        try:
            corrected_query, extra_queries = await _expand_synonyms(query, ctx)
        except Exception as exc:
            logger.warning("[rag_retrieve] synonym expansion skipped: %s", exc)

    # Phase 3: Conditional dense leg — run exact+sparse first, check if
    # their reranker scores are high enough to skip the dense embedding API call.
    non_dense_legs = [l for l in legs if l != "dense"]
    run_dense = "dense" in legs

    if run_dense and non_dense_legs:
        # Run exact+sparse first (without dense)
        coros_nd = []
        if "sparse" in legs:
            coros_nd.append(sparse_retrieval_node(state, ctx.db, kb_ids, org_id, file_markdown, min_score=level["sparse_min_score"], doc_ids=doc_ids, extra_queries=extra_queries))
        if "exact" in legs:
            coros_nd.append(exact_retrieval_node(state, ctx.db, kb_ids, org_id, file_markdown, min_score=level["exact_min_score"], doc_ids=doc_ids, extra_queries=extra_queries))

        nd_results = await asyncio.gather(*coros_nd, return_exceptions=True)
        for r in nd_results:
            if isinstance(r, Exception):
                logger.warning("[rag_retrieve] non-dense leg failed: %s", r)
            else:
                state.update(r)

        # Merge + rerank the exact+sparse results to get a quality score
        state.update(merge_node(state, file_markdown, ctx.db, ctx.org_id))
        pre_docs = state.get("retrieved_docs", [])
        if pre_docs:
            state.update(reranking_node(state))
            scored_docs = state.get("all_scored_docs", [])
            best_score = max(
                (d.get("metadata", {}).get("_reranker_score", 0.0) for d in scored_docs),
                default=0.0,
            )
            fast_accept = get_setting(ctx.db, "ADAPTIVE_RETRIEVAL_FAST_ACCEPT_SCORE", ctx.org_id)
            if best_score >= fast_accept:
                logger.info("[rag_retrieve] fast-accept: best reranker score %.3f >= %.2f, skipping dense leg",
                            best_score, fast_accept)
                run_dense = False
            else:
                logger.debug("[rag_retrieve] best reranker score %.3f < %.2f, running dense leg",
                             best_score, fast_accept)
                # Reset state for the full run (dense + exact + sparse)
                # Keep the exact/sparse docs but re-merge with dense results
                # by resetting the merged state and running all legs together.
                # Actually, simpler: just run dense separately and merge.
                state["retrieved_docs"] = []
                state["all_scored_docs"] = []

        if run_dense:
            # Run dense leg separately and merge with existing exact/sparse docs
            dense_result = await dense_retrieval_node(state, ctx.db, kb_ids, org_id, file_markdown, min_score=level["dense_min_score"], doc_ids=doc_ids)
            state.update(dense_result)
            # Re-merge all docs (dense + exact + sparse)
            # Reset per-leg docs to force merge to include all
            state["retrieved_docs"] = []
            state.update(merge_node(state, file_markdown, ctx.db, ctx.org_id))
    else:
        # No dense in legs, or no non-dense legs — run all concurrently
        coros = []
        if "dense" in legs:
            coros.append(dense_retrieval_node(state, ctx.db, kb_ids, org_id, file_markdown, min_score=level["dense_min_score"], doc_ids=doc_ids))
        if "sparse" in legs:
            coros.append(sparse_retrieval_node(state, ctx.db, kb_ids, org_id, file_markdown, min_score=level["sparse_min_score"], doc_ids=doc_ids, extra_queries=extra_queries))
        if "exact" in legs:
            coros.append(exact_retrieval_node(state, ctx.db, kb_ids, org_id, file_markdown, min_score=level["exact_min_score"], doc_ids=doc_ids, extra_queries=extra_queries))

        leg_results = await asyncio.gather(*coros, return_exceptions=True)
        for r in leg_results:
            if isinstance(r, Exception):
                logger.warning("[rag_retrieve] leg failed: %s", r)
            else:
                state.update(r)

        state.update(merge_node(state, file_markdown, ctx.db, ctx.org_id))

    # Fast path: exact-only search with explicit sort skips the reranker
    # quality gate. Exact FTS matches are already high-precision and the
    # user explicitly wants sorted order, not relevance order.
    if legs == ["exact"] and sort:
        merged = state.get("retrieved_docs", [])
        state["retrieved_docs"] = _sort_merged_docs(merged, sort)
        _apply_excluded_terms_filter(state, ctx)
        return state

    # Score all docs (quality gate — always runs), then filter by threshold.
    # Note: if the conditional dense path already reranked, this reranks
    # the combined set (dense + exact + sparse) which is correct.
    state.update(reranking_node(state))
    state.update(filter_node(state, threshold=level["rerank_threshold"]))

    # Apply metadata sort AFTER filtering so user's explicit sort order
    # is preserved instead of being overridden by reranker relevance score.
    if sort:
        scored = state.get("retrieved_docs", [])
        state["retrieved_docs"] = _sort_merged_docs(scored, sort)

    # Drop docs containing negated terms extracted by rewrite_query_node.
    _apply_excluded_terms_filter(state, ctx)
    return state


def _apply_excluded_terms_filter(state: dict, ctx: ToolContext) -> None:
    """Post-filter retrieved docs by excluded_terms from AgentState.

    Drops any doc whose page_content or title contains an excluded term.
    No-op if excluded_terms is empty or ctx.state is unavailable.
    """
    excluded = []
    if ctx.state:
        excluded = ctx.state.get("excluded_terms", [])
    if not excluded:
        return
    docs = state.get("retrieved_docs", [])
    if not docs:
        return
    filtered = []
    for doc in docs:
        content = doc.get("page_content", "")
        title = doc.get("metadata", {}).get("_title", "") or doc.get("metadata", {}).get("title", "")
        text = f"{content} {title}"
        if not any(_content_contains_exclusion(text, term) for term in excluded):
            filtered.append(doc)
    if len(filtered) != len(docs):
        logger.debug("[rag_retrieve] excluded_terms filter: %d → %d docs (dropped %d containing: %s)",
                     len(docs), len(filtered), len(docs) - len(filtered), excluded)
    state["retrieved_docs"] = filtered


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
    doc_ids: Optional[list[int]] = None,
    sort: Optional[dict] = None,
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
        state = await _run_retrieval_pass(ctx, query, kb_ids, org_id, file_markdown, legs, level, doc_ids=doc_ids, sort=sort)
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


def _confidence_level(confidence: float) -> str:
    """Map a 0-1 confidence score to a label."""
    if confidence > 0.7:
        return "high"
    if confidence > 0.3:
        return "medium"
    return "low"


async def _try_rewrite_retry(
    ctx: ToolContext,
    input_obj: RagRetrieveInput,
    docs: list,
    missing: str,
    kb_ids: list[int],
    org_id: Optional[int],
    file_markdown: Optional[str],
    legs: list[str],
    levels: list[dict[str, Any]],
    min_confidence: float,
    doc_ids: list[int] | None,
) -> tuple[str, dict, list, float, int, bool, str, list[int] | None]:
    """Attempt a rewrite+retry when the first pass was insufficient.

    Returns (query_used, state, docs, confidence, levels_tried, sufficient, missing, doc_ids).
    If the rewrite doesn't change the query, returns the original values unchanged.
    """
    rewritten, filter_suggestion = await _rewrite_query(input_obj.query, missing, ctx, docs)
    if rewritten == input_obj.query and not filter_suggestion:
        return input_obj.query, {}, docs, 0.0, 0, False, missing, doc_ids

    merged_filters = dict(input_obj.filters or {})
    if filter_suggestion:
        merged_filters.update(filter_suggestion)
        doc_ids = _resolve_filter_to_doc_ids(ctx.db, kb_ids, merged_filters)

    _emit_progress(
        "query_rewrite",
        "Retrying with rewritten query …",
        rewritten_query=rewritten,
        original_query=input_obj.query,
    )
    state, docs, confidence, levels_tried, sufficient, missing = await _run_relaxation_ladder(
        ctx, rewritten, kb_ids, org_id, file_markdown, legs, levels,
        min_confidence, input_obj.graph_expand, doc_ids=doc_ids, sort=input_obj.sort,
    )
    return rewritten, state, docs, confidence, levels_tried, sufficient, missing, doc_ids


async def _rag_retrieve(ctx: ToolContext, input_obj: RagRetrieveInput) -> dict:
    t0 = time.monotonic()
    rbac = enforce_rbac(ctx, kb_ids=input_obj.kb_ids)
    kb_ids = rbac["kb_ids"]
    if not kb_ids and ctx.state is not None:
        kb_ids = ctx.state.get("kb_ids", [])
    org_id = ctx.org_id
    file_markdown = ctx.state.get("file_markdown") if ctx.state else None

    legs = input_obj.legs or ["dense", "sparse", "exact"]
    from app.services.settings_service import get_setting
    min_confidence = (
        input_obj.min_confidence
        if input_obj.min_confidence is not None
        else get_setting(ctx.db, "ADAPTIVE_RETRIEVAL_THRESHOLD", ctx.org_id) / 100.0
    )

    # Resolve metadata filters to document_ids via MySQL.
    doc_ids = _resolve_filter_to_doc_ids(ctx.db, kb_ids, input_obj.filters)
    if doc_ids is not None:
        _emit_progress("filtering", f"Filtering to {len(doc_ids)} matching documents …")
        if not doc_ids:
            logger.info("[rag_retrieve] filters matched 0 documents — returning empty")
            return _empty_result(input_obj, t0, "filters matched 0 documents")

    all_levels = _relaxation_levels(ctx.db, ctx.org_id)
    adaptive_enabled = get_setting(ctx.db, "ADAPTIVE_RETRIEVAL_ENABLED", ctx.org_id)
    levels = all_levels if adaptive_enabled else all_levels[:1]

    # ── Pass 1: relaxation ladder with the original query ──────────────
    state, docs, confidence, levels_tried, sufficient, missing = await _run_relaxation_ladder(
        ctx, input_obj.query, kb_ids, org_id, file_markdown, legs, levels,
        min_confidence, input_obj.graph_expand, doc_ids=doc_ids, sort=input_obj.sort,
    )

    # ── Pass 2: if still insufficient, rewrite the query and re-run ────
    query_used = input_obj.query
    query_rewritten = False
    if not sufficient and missing:
        query_used, state, docs, confidence, levels_tried, sufficient, missing, doc_ids = await _try_rewrite_retry(
            ctx, input_obj, docs, missing, kb_ids, org_id, file_markdown, legs, levels, min_confidence, doc_ids,
        )
        query_rewritten = query_used != input_obj.query

    conf_level = _confidence_level(confidence)
    latency_ms = round((time.monotonic() - t0) * 1000)
    result_summary = {
        "doc_count": len(docs),
        "confidence": confidence,
        "confidence_level": conf_level,
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
            "confidence_level": conf_level,
            "query_used": query_used,
            "original_query": input_obj.query,
            "query_rewritten": query_rewritten,
            "legs_run": legs,
            "levels_tried": levels_tried,
            "sufficient": sufficient,
            "missing": missing if not sufficient else "",
            "filters_applied": input_obj.filters,
            "sort_applied": input_obj.sort,
        },
        "error": None,
        "tokens": len(str(docs)) // 4,
    }


def make_rag_retrieve_tool(ctx: ToolContext) -> _RagRetrieveTool:
    tool = _RagRetrieveTool()
    tool.ctx = ctx
    return tool


RagRetrieveTool = _RagRetrieveTool
