"""Wrapper tool for parallel retrieval via sub-agents.

The main agent calls this tool when it reasons that a query has multiple
independent sub-questions that can be searched in parallel. For simple
queries, the main agent calls search tools directly (no sub-agent overhead).

The tool accepts a list of sub-queries, spawns one retrieval sub-agent per
sub-query, runs them concurrently, and returns merged evidence with citations.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RetrieveParallelInput(BaseModel):
    """Input for parallel retrieval."""

    queries: list[str] = Field(
        ...,
        description="List of 2-4 independent sub-queries to search in parallel. "
        "Each sub-query should be a self-contained question that can be "
        "searched independently. Example: "
        '["What are the principles of risk management?", '
        '"What are the applications of risk management in cybersecurity?"]',
    )


class RetrieveParallelTool(BaseTool):
    """Tool that spawns parallel retrieval sub-agents."""

    name: str = "retrieve_parallel"
    description: str = (
        "Retrieve evidence for MULTIPLE independent sub-queries in parallel. "
        "Use ONLY when the user's question has 2+ distinct parts that can be "
        "searched independently (e.g. comparisons, multi-topic questions). "
        "For simple single-topic queries, use search_dense/search_exact directly. "
        "Returns merged evidence chunks with citation metadata. "
        "Pass 2-4 sub-queries as a list."
    )
    args_schema: type = RetrieveParallelInput
    ctx: Any = None

    async def arun(self, tool_input: str | dict[str, Any], **kwargs: Any) -> dict:
        """Run parallel retrieval sub-agents."""
        if isinstance(tool_input, str):
            import json
            try:
                args = json.loads(tool_input)
            except Exception:
                return {
                    "ok": False,
                    "result": {},
                    "error": "Invalid JSON input. Pass {\"queries\": [...]}",
                    "tokens": 0,
                    "terminate": False,
                }
        else:
            args = tool_input

        queries = args.get("queries", [])
        if not queries or len(queries) < 2:
            return {
                "ok": False,
                "result": {},
                "error": "retrieve_parallel requires 2+ sub-queries. "
                "For single queries, use search_dense or search_exact directly.",
                "tokens": 0,
                "terminate": False,
            }
        if len(queries) > 4:
            queries = queries[:4]  # Cap at 4

        ctx = self.ctx
        if ctx is None:
            return {
                "ok": False,
                "result": {},
                "error": "No context available",
                "tokens": 0,
                "terminate": False,
            }

        # Lazy import
        from app.services.agentic_rag.retrieval_subagent import run_retrieval_subagents_parallel
        from app.services.agentic_rag.agent_graph.helpers import _writer as _get_writer

        writer = _get_writer()
        writer({"event": "retrieve_parallel", "status": "started",
                "queries": queries})

        tool_budget = 25
        try:
            from app.services.settings_service import get_setting
            tool_budget = get_setting(ctx.db, "AGENT_TOTAL_TOOL_BUDGET", ctx.org_id) or 25
        except Exception:
            pass

        try:
            results = await run_retrieval_subagents_parallel(
                ctx=ctx,
                sub_queries=queries,
                tool_budget=tool_budget,
            )
        except Exception as exc:
            logger.exception("[retrieve_parallel] sub-agents failed: %s", exc)
            return {
                "ok": False,
                "result": {},
                "error": f"Retrieval sub-agents failed: {exc}",
                "tokens": 0,
                "terminate": False,
            }

        # Merge evidence from all sub-agents
        all_evidence: list[dict] = []
        seen_hashes: set[str] = set()
        summaries: list[str] = []

        for result in results:
            query = result.get("query", "")
            summary = result.get("summary", "")
            evidence = result.get("evidence", [])
            ok = result.get("ok", False)

            summaries.append(f"[{query}] {'Found' if ok else 'No evidence'}: {summary[:100]}")

            for chunk in evidence:
                # Deduplicate by content hash or content prefix
                content = chunk.get("content", "")
                dedup_key = content[:200]
                if dedup_key in seen_hashes:
                    continue
                seen_hashes.add(dedup_key)
                all_evidence.append(chunk)

        # Cap total evidence to keep main agent context manageable
        all_evidence = all_evidence[:25]

        writer({"event": "retrieve_parallel", "status": "done",
                "evidence_count": len(all_evidence),
                "sub_queries": len(queries)})

        return {
            "ok": len(all_evidence) > 0,
            "result": {
                "hits": all_evidence,
                "count": len(all_evidence),
                "sub_query_summaries": summaries,
            },
            "error": None,
            "tokens": sum(len(e.get("content", "")) for e in all_evidence) // 4,
            "terminate": False,
        }

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use arun() for agent tools.")
