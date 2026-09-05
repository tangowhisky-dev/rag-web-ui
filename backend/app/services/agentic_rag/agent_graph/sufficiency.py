"""Sufficiency check node — replaces reflect_final_node.

Three-tier check:
1. Budget: if total tool calls >= AGENT_TOTAL_TOOL_BUDGET or iterations >=
   AGENT_MAX_ITERATIONS or wall clock exceeded → finalize.
2. force_finalize: if any tool returned terminate=True or the deterministic
   execution check passed → finalize.
3. LLM sufficiency: ask the LLM whether the current evidence is sufficient
   to answer the query. If yes → finalize. If no and budget remains → think.
"""

from __future__ import annotations

import logging
from typing import Any

from app.services.agentic_rag.llm_factory import build_chat_llm
from app.services.agentic_rag.nodes import _agent_step
from app.services.agentic_rag.prompts import SUFFICIENCY_CHECK_PROMPT
from app.services.settings_service import get_setting

from .execution_check import _build_execution_summary, _verify_execution
from .helpers import _coerce_observation, _wall_clock_exceeded, _writer
from .observations import _observations_metadata_text

logger = logging.getLogger(__name__)


async def sufficiency_check_node(state, ctx) -> dict:
    """Check whether the agent has sufficient evidence to finalize.

    Replaces reflect_final_node. Three tiers:
    1. Budget exhausted → finalize.
    2. force_finalize or deterministic check passed → finalize.
    3. LLM sufficiency judgment → finalize or think.
    """
    with _agent_step("sufficiency_check"):
        writer = _writer()
        iteration = state.get("iteration", 0)
        max_iter = get_setting(ctx.db, "AGENT_MAX_ITERATIONS", ctx.org_id)
        total_budget = get_setting(ctx.db, "AGENT_TOTAL_TOOL_BUDGET", ctx.org_id)

        counts = dict(state.get("tool_call_counts", {}))
        total_calls = sum(counts.values())

        # Tier 1: Budget exhausted
        if total_calls >= total_budget:
            logger.debug("[sufficiency] total budget exhausted (%d/%d), finalizing", total_calls, total_budget)
            return {"sufficient": True, "force_finalize": True}

        if iteration >= max_iter or _wall_clock_exceeded(state):
            logger.debug("[sufficiency] iteration/wall-clock limit reached, finalizing")
            return {"sufficient": True, "force_finalize": True}

        # Tier 2: force_finalize or deterministic check
        if state.get("force_finalize"):
            logger.debug("[sufficiency] force_finalize set, finalizing")
            return {"sufficient": True}

        summary = _build_execution_summary(state)
        ready, reasoning = _verify_execution(summary)
        if ready:
            logger.debug("[sufficiency] deterministic check passed: %s", reasoning[:200])
            return {"sufficient": True}

        # Tier 3: LLM sufficiency judgment
        sufficient = await _llm_sufficiency_check(state, ctx, summary)
        if sufficient:
            logger.debug("[sufficiency] LLM judged evidence sufficient")
            return {"sufficient": True}

        logger.debug("[sufficiency] not sufficient, routing back to think")
        return {"sufficient": False}


async def _llm_sufficiency_check(state: dict, ctx: Any, summary: dict) -> bool:
    """Ask the LLM whether the current evidence is sufficient to answer."""
    observations = state.get("observations", [])
    obs_text = _observations_metadata_text(observations)
    query = state.get("original_query", "")

    # Build evidence previews from retrieved_docs so the LLM can judge
    # content quality, not just hit counts.
    docs = state.get("retrieved_docs", [])
    evidence_parts = []
    for i, doc in enumerate(docs[:10], 1):
        if not isinstance(doc, dict):
            continue
        content = (doc.get("page_content") or "")[:200]
        meta = doc.get("metadata", {}) or {}
        title = meta.get("title", meta.get("file_name", ""))
        evidence_parts.append(f"[E{i}] {title}: {content}")
    evidence_text = "\n".join(evidence_parts) if evidence_parts else "No retrieved documents yet."

    # Deterministic shortcut: if we've already done 2+ search rounds and
    # the evidence hasn't grown, more searches won't help. Finalize.
    search_calls = summary["search"]["calls"]
    if search_calls >= 3 and len(docs) <= 5:
        logger.debug("[sufficiency] deterministic shortcut: %d searches, %d docs — finalizing", search_calls, len(docs))
        return True

    # Deterministic shortcut: if the first search returned 10+ hits and
    # we've already reranked, the evidence is likely sufficient.
    if search_calls >= 1 and len(docs) >= 10:
        has_rerank = any(
            getattr(o, "tool", None) == "rerank_results" and not getattr(o, "error", None)
            for o in observations
        )
        if has_rerank:
            logger.debug("[sufficiency] deterministic shortcut: %d docs after rerank — finalizing", len(docs))
            return True

    try:
        llm = build_chat_llm(ctx.org_id, ctx.db, role="query", temperature=0.0)
        prompt = SUFFICIENCY_CHECK_PROMPT.format(
            query=query[:500],
            observations=obs_text[:2000],
            evidence=evidence_text[:2000],
            hit_count=summary["search"]["hits"],
            search_calls=summary["search"]["calls"],
            remaining_budget=summary["remaining_budget"]["total"],
        )
        resp = await llm.ainvoke([{"role": "user", "content": prompt}])
        raw = str(resp.content).strip().lower()
        # Accept JSON {"sufficient": true/false} or plain text "yes"/"no"
        if "{" in raw:
            import json
            try:
                obj = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
                return bool(obj.get("sufficient", False))
            except (json.JSONDecodeError, ValueError):
                pass
        return raw.startswith("yes") or raw.startswith("true")
    except Exception as exc:
        logger.warning("[sufficiency] LLM check failed: %s, defaulting to not sufficient", exc)
        return False


def route_sufficiency(state) -> str:
    """Route after sufficiency_check: sufficient → finalize, not → think."""
    if state.get("sufficient") or state.get("force_finalize"):
        return "finalize"
    return "think"
