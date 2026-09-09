"""Trace retrieval sub-agent pipeline: log every LLM call, tool selection,
arguments, results, and dedup hits to diagnose looping and budget exhaustion.

Usage (inside backend container):
    python -m trace_subagent --query "vulnerabilities of GMR-1 and GMR-2"
    python -m trace_subagent --query "vulnerabilities of GMR-1 and GMR-2" --budget 10
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from typing import Any

# Set up logging BEFORE any app imports so we capture everything
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stderr,
)
# Silence httpx noise
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx2").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)

logger = logging.getLogger("trace")


async def run_trace(query: str, budget: int, chat_id: int | None):
    from app.db.session import SessionLocal
    from app.services.agentic_rag.tool_context import ToolContext
    from app.services.agentic_rag.retrieval_subagent import (
        run_retrieval_subagent,
        _build_retrieval_user_prompt,
        RETRIEVAL_SUBAGENT_PROMPT,
    )
    from app.services.agentic_rag.tool_call_parser import parse_think_response
    from app.services.agentic_rag.agent_graph.observations import _tool_descriptions_text
    from app.services.agentic_rag.tools import build_tools
    from app.services.agentic_rag.llm_factory import build_chat_llm
    from app.services.settings_service import get_setting
    from app.services.agentic_rag.agent_graph.tooling import _run_tool
    from app.services.agentic_rag.schemas import Observation
    from app.models.chat import Chat

    db = SessionLocal()

    # Find a chat with KB access if not specified
    if chat_id is None:
        chat = db.query(Chat).filter(Chat.knowledge_bases.any()).first()
        if not chat:
            logger.error("No chat with KB access found in DB")
            return
        chat_id = chat.id
        logger.info("Using chat_id=%d (user_id=%d)", chat_id, chat.user_id)
    else:
        chat = db.query(Chat).filter(Chat.id == chat_id).first()
        if not chat:
            logger.error("Chat %d not found", chat_id)
            return

    ctx = ToolContext(
        db=db,
        user_id=chat.user_id,
        org_id=chat.org_id,
        chat_id=chat_id,
        qdrant_client=None,  # tools will get their own
    )

    # Decompose query into sub-queries (same as main agent does)
    sub_queries = query.replace(" and ", "|").split("|")
    sub_queries = [s.strip() for s in sub_queries if s.strip()]
    if len(sub_queries) < 2:
        sub_queries = [query]

    logger.info("=" * 80)
    logger.info("QUERY: %s", query)
    logger.info("SUB-QUERIES: %s", sub_queries)
    logger.info("BUDGET: %d per sub-agent", budget)
    logger.info("=" * 80)

    # Run each sub-agent sequentially (for clear logging)
    for sq_idx, sq in enumerate(sub_queries):
        logger.info("\n{'='*80}")
        logger.info("SUB-AGENT %d: %s", sq_idx + 1, sq)
        logger.info("{'='*80}")

        subagent_id = f"trace-{sq_idx}"
        t0 = time.monotonic()

        # Manually run the sub-agent loop with detailed logging
        all_tools = build_tools(ctx)
        retrieval_tool_names = {
            "keyword_search", "semantic_search",
            "title_search", "file_read", "kb_outline", "kb_grep",
            "graph_expand",
        }
        tools = {t.name: t for t in all_tools if t.name in retrieval_tool_names}
        tools_list = list(tools.values())
        tools_text = _tool_descriptions_text(tools_list)

        system = RETRIEVAL_SUBAGENT_PROMPT
        observations: list[Observation] = []
        counts: dict[str, int] = {}
        seen_signatures: set[str] = set()
        iteration = 0
        total_llm_calls = 0

        while True:
            iteration += 1
            calls_used = sum(counts.values())
            total_attempts = len(observations)
            if calls_used >= budget or total_attempts >= budget or iteration > budget + 2:
                logger.info("[iter %d] BUDGET EXHAUSTED (calls=%d, attempts=%d, iter=%d, budget=%d)",
                            iteration, calls_used, total_attempts, iteration, budget)
                break

            user = _build_retrieval_user_prompt(
                sq, tools_text, iteration, budget, calls_used, observations,
            )

            logger.info("\n--- iter %d | calls_used=%d/%d ---", iteration, calls_used, budget)
            logger.info("USER PROMPT (last 500 chars):\n%s", user[-500:])

            try:
                tool_temp = get_setting(ctx.db, "TOOL_CALL_TEMPERATURE", ctx.org_id)
                llm = build_chat_llm(ctx.org_id, ctx.db, role="chat", temperature=tool_temp)
                if iteration == 1:
                    from app.services.agentic_rag.llm_factory import get_org_llm
                    cfg = get_org_llm(ctx.org_id, ctx.db, role="chat")
                    logger.info("MODEL: %s (base=%s)", cfg["model_name"], cfg["api_base"])
                resp = await llm.bind_tools(tools_list).ainvoke([
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ])
                total_llm_calls += 1
            except Exception as exc:
                logger.warning("LLM call failed: %s", exc)
                break

            parsed = parse_think_response(resp, mode="auto")
            tool_calls = parsed.tool_calls

            logger.info("REASONING (full):\n%s", parsed.reasoning or "(none)")
            logger.info("FINAL_ANSWER (if any):\n%s", parsed.final_answer or "(none)")
            logger.info("RAW CONTENT (first 1000):\n%s", (resp.content if isinstance(resp.content, str) else str(resp.content))[:1000])
            logger.info("ADDITIONAL_KWARGS: %s", json.dumps(resp.additional_kwargs, default=str)[:500] if resp.additional_kwargs else "(none)")

            if not tool_calls:
                logger.info("No tool calls — sub-agent wrote final JSON")
                break

            logger.info("TOOL_CALLS: %d", len(tool_calls))
            for tc_idx, tc in enumerate(tool_calls):
                name = tc.get("tool")
                args = tc.get("arguments", {})
                logger.info("  [%d] %s(%s)", tc_idx, name, json.dumps(args, default=str)[:200])

            # Execute tool calls
            for tc in tool_calls:
                name = tc.get("tool")
                args = tc.get("arguments", {})
                tool = tools.get(name)

                if name == "file_read":
                    sig_key = f"{args.get('document_id')}:{args.get('offset', 1)}"
                else:
                    sig_key = (
                        args.get("query")
                        or args.get("title_contains")
                        or args.get("pattern")
                        or str(args.get("document_id") or "")
                    )
                sig = f"{name}:{sig_key}"
                if sig in seen_signatures:
                    logger.info("  DEDUP HIT: %s already tried — skipped", sig)
                    observations.append(Observation(
                        tool=name, arguments=args, result={},
                        error=f"Duplicate call: {sig} already tried.",
                        tokens=0,
                    ))
                    continue
                seen_signatures.add(sig)
                logger.info("  DEDUP: added signature %s", sig)

                if tool is None:
                    logger.info("  Tool '%s' not available", name)
                    observations.append(Observation(
                        tool=name, arguments=args, result={},
                        error=f"Tool '{name}' not available", tokens=0,
                    ))
                    continue

                if calls_used >= budget:
                    logger.info("  BUDGET EXCEEDED mid-batch")
                    observations.append(Observation(
                        tool=name, arguments=args, result={},
                        error=f"Budget ({budget}) exhausted.", tokens=0,
                    ))
                    continue

                t_tool = time.monotonic()
                result = await _run_tool(tool, name, args)
                tool_ms = (time.monotonic() - t_tool) * 1000

                obs = Observation(
                    tool=result["tool"], arguments=result["arguments"],
                    result=result.get("result", {}), error=result.get("error"),
                    tokens=result.get("tokens", 0),
                )
                observations.append(obs)
                counts[name] = counts.get(name, 0) + 1
                calls_used += 1

                # Log result summary
                if obs.error:
                    logger.info("  RESULT: ERROR: %s (%.0fms)", obs.error[:150], tool_ms)
                else:
                    r = obs.result or {}
                    if "hits" in r:
                        hits = r["hits"]
                        top = hits[0] if hits else {}
                        logger.info("  RESULT: %d hits, top: title=%s doc_id=%s score=%.3f (%.0fms)",
                                    len(hits),
                                    (top.get("title") or "")[:50],
                                    top.get("document_id"),
                                    top.get("score", 0),
                                    tool_ms)
                    elif "docs" in r:
                        docs = r["docs"]
                        top = docs[0] if docs else {}
                        logger.info("  RESULT: %d docs, top: title=%s doc_id=%s (%.0fms)",
                                    len(docs),
                                    (top.get("title") or (top.get("metadata", {}) or {}).get("title", ""))[:50],
                                    top.get("id") or top.get("document_id") or (top.get("metadata", {}) or {}).get("document_id"),
                                    tool_ms)
                    else:
                        logger.info("  RESULT: %s (%.0fms)", json.dumps(r, default=str)[:150], tool_ms)

        elapsed = time.monotonic() - t0
        logger.info("\nSUB-AGENT %d SUMMARY: %s", sq_idx + 1, sq)
        logger.info("  Total iterations: %d", iteration)
        logger.info("  Total LLM calls: %d", total_llm_calls)
        logger.info("  Total tool calls: %d", sum(counts.values()))
        logger.info("  Tool call breakdown: %s", dict(counts))
        logger.info("  Dedup signatures: %d unique, %d blocked",
                    len(seen_signatures),
                    sum(1 for o in observations if o.error and "Duplicate" in (o.error or "")))
        logger.info("  Elapsed: %.1fs", elapsed)
        logger.info("  Observations with errors: %d",
                    sum(1 for o in observations if o.error))

    db.close()
    logger.info("\nDONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="vulnerabilities of GMR-1 and GMR-2")
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--chat-id", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(run_trace(args.query, args.budget, args.chat_id))
