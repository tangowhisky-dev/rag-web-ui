"""Trace office sub-agent pipeline: log every LLM call, tool selection,
arguments, results, and progress events to diagnose failures and verify
the intended flow.

Usage (inside backend container):
    python -m trace_office --query "give me 3 slides deck for vulnerabilities of GMR-2"
    python -m trace_office --query "give me 3 slides deck for vulnerabilities of GMR-2" --budget 20
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stderr,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)

logger = logging.getLogger("trace_office")


async def run_trace(query: str, budget: int, chat_id: int | None):
    from app.db.session import SessionLocal
    from app.services.agentic_rag.tool_context import ToolContext
    from app.services.agentic_rag.office_subagent import (
        run_office_subagent,
        OFFICE_SUBAGENT_PROMPT,
        _build_subagent_user_prompt,
        _format_evidence_for_subagent,
        _format_accumulated_data,
    )
    from app.services.agentic_rag.tool_call_parser import parse_think_response
    from app.services.agentic_rag.agent_graph.observations import _tool_descriptions_text
    from app.services.agentic_rag.tools import build_tools
    from app.services.agentic_rag.llm_factory import build_chat_llm, get_org_llm
    from app.services.settings_service import get_setting
    from app.services.agentic_rag.agent_graph.tooling import _run_tool
    from app.services.agentic_rag.schemas import Observation
    from app.models.chat import Chat

    db = SessionLocal()

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

    # Build a minimal state with some mock evidence for GMR-2
    mock_state = {
        "retrieved_docs": [
            {
                "page_content": "The GMR-2 cipher is a type of stream cipher with 64-bit key. The internal states include a shift register S, an encryption-key register K, a counter c, and a toggle-bit t. It is used in Inmarsat satellite phones.",
                "metadata": {"title": "A Real-time Inversion Attack on the GMR-2 Cipher", "document_id": 1416},
            },
            {
                "page_content": "We present a ciphertext-only attack on the GEO-Mobile Radio Interface-2 (GMR-2) system. The GMR-2 is a satellite communication standard adopted by Inmarsat.",
                "metadata": {"title": "A Practical Ciphertext-Only Attack", "document_id": 1418},
            },
        ],
        "accumulated_data": [],
        "generated_files": [],
    }

    ctx = ToolContext(
        db=db,
        user_id=chat.user_id,
        org_id=chat.org_id,
        chat_id=chat_id,
        qdrant_client=None,
        state=mock_state,
    )

    logger.info("=" * 80)
    logger.info("QUERY: %s", query)
    logger.info("BUDGET: %d", budget)
    logger.info("=" * 80)

    # Manually run the sub-agent loop with detailed logging
    # (mirrors run_office_subagent but with full tracing)
    all_tools = build_tools(ctx)
    office_tool_names = {"office_load_skill", "office_generate", "office_inspect", "office_edit"}
    tools = {t.name: t for t in all_tools if t.name in office_tool_names}
    tools_list = list(tools.values())
    tools_text = _tool_descriptions_text(tools_list)

    docs = ctx.state.get("retrieved_docs", [])
    evidence_text = _format_evidence_for_subagent(docs)
    accumulated_data = ctx.state.get("accumulated_data", [])
    accumulated_data_text = _format_accumulated_data(accumulated_data)

    logger.info("EVIDENCE TEXT (first 500 chars):\n%s", evidence_text[:500])
    logger.info("ACCUMULATED_DATA: %s", accumulated_data_text[:200] if accumulated_data_text else "(empty)")

    system = OFFICE_SUBAGENT_PROMPT
    observations: list[Observation] = []
    counts: dict[str, int] = {}
    summary = ""
    iteration = 0
    total_llm_calls = 0
    progress_events: list[dict] = []

    # Snapshot pre-existing files
    pre_existing_file_ids = set()
    if ctx.state:
        for f in ctx.state.get("generated_files", []) or []:
            if f.get("file_id"):
                pre_existing_file_ids.add(f["file_id"])

    logger.info("PRE_EXISTING_FILE_IDS: %s", pre_existing_file_ids)

    while True:
        iteration += 1
        calls_used = sum(counts.values())
        total_attempts = len(observations)
        if calls_used >= budget or total_attempts >= budget or iteration > budget + 2:
            logger.info("[iter %d] BUDGET EXHAUSTED (calls=%d, attempts=%d, iter=%d, budget=%d)",
                        iteration, calls_used, total_attempts, iteration, budget)
            break

        user = _build_subagent_user_prompt(
            query, evidence_text, accumulated_data_text,
            tools_text, iteration, budget, calls_used, observations,
        )

        logger.info("\n--- iter %d | calls_used=%d/%d ---", iteration, calls_used, budget)
        logger.info("USER PROMPT (last 800 chars):\n%s", user[-800:])

        progress_events.append({"event": "thinking", "iteration": iteration})
        logger.info("PROGRESS: thinking (iter %d)", iteration)

        try:
            tool_temp = get_setting(ctx.db, "TOOL_CALL_TEMPERATURE", ctx.org_id)
            llm = build_chat_llm(ctx.org_id, ctx.db, role="chat", temperature=tool_temp)
            if iteration == 1:
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

        if not tool_calls:
            logger.info("No tool calls — sub-agent wrote summary")
            if isinstance(parsed.final_answer, str) and parsed.final_answer.strip():
                summary = parsed.final_answer.strip()
            logger.info("SUMMARY: %s", summary[:300])
            progress_events.append({"event": "done", "summary": summary[:200]})
            break

        logger.info("TOOL_CALLS: %d", len(tool_calls))
        for tc_idx, tc in enumerate(tool_calls):
            name = tc.get("tool")
            args = tc.get("arguments", {})
            logger.info("  [%d] %s(%s)", tc_idx, name, json.dumps(args, default=str)[:300])

        for tc in tool_calls:
            name = tc.get("tool")
            args = tc.get("arguments", {})
            tool = tools.get(name)

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

            progress_events.append({"event": "tool_call", "tool": name, "iteration": iteration})
            logger.info("PROGRESS: tool_call %s (iter %d)", name, iteration)

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

            if obs.error:
                logger.info("  RESULT: ERROR: %s (%.0fms)", obs.error[:300], tool_ms)
            else:
                r = obs.result or {}
                if obs.tool == "office_generate" and r.get("file_id"):
                    logger.info("  RESULT: file_id=%s file_name=%s format=%s slides=%s (%.0fms)",
                                r.get("file_id"), r.get("file_name"),
                                r.get("format"),
                                r.get("slide_count"),
                                tool_ms)
                    progress_events.append({"event": "file_created", "file_id": r["file_id"],
                                           "file_name": r.get("file_name", "")})
                elif obs.tool == "office_load_skill":
                    content_len = len(r.get("skill_content", ""))
                    logger.info("  RESULT: skill loaded, format=%s, content_len=%d (%.0fms)",
                                r.get("format"), content_len, tool_ms)
                else:
                    logger.info("  RESULT: %s (%.0fms)", json.dumps(r, default=str)[:300], tool_ms)

            progress_events.append({"event": "tool_done", "tool": obs.tool,
                                   "error": bool(obs.error), "iteration": iteration})

    # Final summary
    generated_files = ctx.state.get("generated_files", []) if ctx.state else []
    new_files = [
        f for f in generated_files
        if f.get("file_id") and f["file_id"] not in pre_existing_file_ids
    ]

    logger.info("\n" + "=" * 80)
    logger.info("OFFICE SUB-AGENT SUMMARY")
    logger.info("=" * 80)
    logger.info("  Total iterations: %d", iteration)
    logger.info("  Total LLM calls: %d", total_llm_calls)
    logger.info("  Total tool calls: %d", sum(counts.values()))
    logger.info("  Tool call breakdown: %s", dict(counts))
    logger.info("  Observations with errors: %d", sum(1 for o in observations if o.error))
    logger.info("  New files generated: %d", len(new_files))
    for f in new_files:
        logger.info("    → file_id=%s file_name=%s format=%s", f.get("file_id"), f.get("file_name"), f.get("format"))
    logger.info("  Summary: %s", summary[:300] if summary else "(none)")
    logger.info("  Progress events emitted: %d", len(progress_events))
    for pe in progress_events:
        logger.info("    → %s", json.dumps(pe, default=str))
    logger.info("  Elapsed: %.1fs", 0)

    db.close()
    logger.info("\nDONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="give me 3 slides deck for vulnerabilities of GMR-2")
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--chat-id", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(run_trace(args.query, args.budget, args.chat_id))
