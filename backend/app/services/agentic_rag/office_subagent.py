"""Office document generation sub-agent.

A mini think→tool loop dedicated to Office document generation. Has only
4 tools (office_load_skill, office_generate, office_inspect, office_edit)
and a focused prompt with field names and iteration guidance.

The main agent calls `create_office_document` (a wrapper tool) which runs
this sub-agent internally. The sub-agent:
1. Reads evidence content and accumulated_data from shared state
2. Loads the OfficeCLI skill (design guidelines)
3. Generates the document (with retry on validation errors)
4. Optionally inspects and edits
5. Returns file metadata + summary

This isolation solves the problem where the main agent's 21-tool context
distracted gemma-4-12b from calling office_generate after searching.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage

from app.services.agentic_rag.llm_factory import build_chat_llm
from app.services.agentic_rag.tool_call_parser import parse_think_response
from app.services.agentic_rag.token_budget import count_tokens
from app.services.agentic_rag.schemas import Observation

# Lazy imports to avoid circular dependency
# (agent_graph → tools → create_office_document → office_subagent → agent_graph)

logger = logging.getLogger(__name__)


OFFICE_SUBAGENT_PROMPT = """\
You are an Office document generation specialist. You create polished,\
 downloadable Office documents using OfficeCLI tools.

# Available Tools

- office_load_skill: Load design guidelines. Call ONCE first.\
 Args: {{"format": "pptx|docx|xlsx", "skill": "base"}}
- office_generate: Create or append to a document.\
 Returns file_id. Args: {{"format": "...", "title": "...", "append": false,\
 "slides": [...], "sections": [...], "sheets": [...]}}
- office_inspect: Check document quality.\
 Args: {{"mode": "outline|issues|screenshot|validate", "file_id": N}}\
 (file_id comes from office_generate result)
- office_edit: Fix issues.\
 Args: {{"file_id": N, "commands": [...]}}\
 (file_id comes from office_generate result)

# Field Names (CRITICAL — wrong names cause validation errors)

PPTX slides: {{"layout": "title|title_and_content|blank",\
 "title": "Slide Title", "subtitle": "...",\
 "bullets": ["bullet 1", "bullet 2"],\
 "chart": {{"type": "bar|line|pie", "title": "..."}},\
 "speaker_notes": "..."}}

DOCX sections: {{"heading": "Section Heading", "level": 1,\
 "paragraphs": ["paragraph 1", "paragraph 2"],\
 "table": {{"headers": [...], "rows": [[...]]}},\
 "chart": {{"type": "bar|line|pie", "title": "..."}}}}

XLSX sheets: {{"name": "Sheet Name",\
 "headers": ["Col1", "Col2"],\
 "rows": [["val1", "val2"]],\
 "chart": {{"type": "bar|line|pie", "title": "..."}}}}

Do NOT use "content" for sections — use "heading" and "paragraphs".\
 Do NOT use "title" for sections — use "heading".

# Process

1. Call office_load_skill with the target format.
2. Call office_generate with append=false and the document structure.\
 For multi-slide decks: 1-2 slides per call, then append=true for the rest.
3. If office_generate returns an error: read the error, fix the field names\
 or structure, and call office_generate again.
4. Optionally call office_inspect to check quality.
5. If issues found: call office_edit to fix them.
6. Write a brief summary of what was created.

# Rules

- Data is read automatically from state.accumulated_data — do NOT pass data values.\
 Pass only structure (titles, headings, bullet text, chart types).
- For text-only documents: provide paragraphs/bullets directly.
- Supported formats: pptx, docx, xlsx ONLY.
- You have a limited tool-call budget. The prompt shows how many calls remain.
- When done, write a brief plain-text summary (no tool calls) describing\
 what was created: file name, format, number of slides/sections/sheets,\
 and key content.
"""


def _build_subagent_user_prompt(
    request: str,
    evidence_text: str,
    accumulated_data_text: str,
    tools_text: str,
    iteration: int,
    tool_budget: int,
    calls_used: int,
    observations: list[Observation],
) -> str:
    """Build the user prompt for the office sub-agent."""
    parts: list[str] = []
    parts.append(f"Document request: {request}\n\n")
    if evidence_text:
        parts.append(f"Source evidence (use this content in the document):\n{evidence_text}\n\n")
    if accumulated_data_text:
        parts.append(f"Structured data available (read automatically by office_generate):\n{accumulated_data_text}\n\n")
    parts.append(f"Available tools:\n{tools_text}\n\n")

    # Show prior observations (tool calls + results)
    if observations:
        parts.append("Prior tool calls:\n")
        for i, obs in enumerate(observations, 1):
            parts.append(f"  {i}. {obs.tool}({json.dumps(obs.arguments, default=str)[:200]})")
            if obs.error:
                parts.append(f"     → ERROR: {obs.error[:300]}\n")
            else:
                result_summary = json.dumps(obs.result, default=str)[:200] if obs.result else "(empty)"
                parts.append(f"     → {result_summary}\n")
        parts.append("\n")

    remaining = tool_budget - calls_used
    parts.append(f"Tool calls remaining: {remaining}/{tool_budget}\n")
    if remaining <= 0:
        parts.append("\nYou have exhausted your tool-call budget. Write a summary of what was created (or failed to create).")
    else:
        parts.append("\nCall the next tool, or write a plain-text summary if the document is created.")
    return "".join(parts)


def _format_evidence_for_subagent(docs: list[dict], max_chars: int = 2000) -> str:
    """Format retrieved docs as evidence text for the sub-agent."""
    if not docs:
        return ""
    parts: list[str] = []
    total = 0
    for doc in docs[:8]:
        if not isinstance(doc, dict):
            continue
        content = (doc.get("page_content") or "")[:500]
        meta = doc.get("metadata") or {}
        title = meta.get("title") or meta.get("file_name") or "Unknown"
        chunk = f"[{title}] {content}"
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n\n".join(parts)


def _format_accumulated_data(data: list) -> str:
    """Format accumulated_data for the sub-agent prompt."""
    if not data:
        return ""
    try:
        return json.dumps(data, default=str)[:2000]
    except Exception:
        return str(data)[:2000]


async def run_office_subagent(
    ctx,
    request: str,
    tool_budget: int = 20,
    subagent_id: str = "",
) -> dict:
    """Run the office sub-agent loop.

    Args:
        ctx: ToolContext (shared with main agent — has state with accumulated_data, etc.)
        request: Natural language document request from the main agent.
        tool_budget: Per-subagent tool-call budget (from OFFICE_SUBAGENT_TOOL_BUDGET).
        subagent_id: Unique identifier for progress event streaming.

    Returns:
        dict with keys: ok, file_id, file_name, format, summary, error
    """
    # Lazy imports (break circular dependency)
    from app.services.agentic_rag.agent_graph.helpers import _writer as _get_writer
    from app.services.agentic_rag.agent_graph.tooling import _run_tool
    from app.services.agentic_rag.agent_graph.observations import _tool_descriptions_text
    from app.services.agentic_rag.tools import build_tools
    from app.services.settings_service import get_setting

    writer = _get_writer()
    writer({"event": "subagent_progress", "subagent_id": subagent_id,
            "sub_query": request[:200], "status": "started",
            "subagent_type": "office"})

    # Build the 4 office tools — these share ctx so they can read/write state
    all_tools = build_tools(ctx)
    office_tool_names = {"office_load_skill", "office_generate", "office_inspect", "office_edit"}
    tools = {t.name: t for t in all_tools if t.name in office_tool_names}
    tools_list = list(tools.values())
    tools_text = _tool_descriptions_text(tools_list)

    # Get evidence and data from shared state
    docs = ctx.state.get("retrieved_docs", []) if ctx.state else []
    evidence_text = _format_evidence_for_subagent(docs)
    accumulated_data = ctx.state.get("accumulated_data", []) if ctx.state else []
    accumulated_data_text = _format_accumulated_data(accumulated_data)

    system = OFFICE_SUBAGENT_PROMPT
    observations: list[Observation] = []
    counts: dict[str, int] = {}
    summary = ""
    seen_signatures: set[str] = set()

    # Snapshot generated_files at start — files from previous turns persist
    # in the checkpointer. We only want to report success if NEW files were
    # created during THIS subagent run.
    pre_existing_file_ids = set()
    if ctx.state:
        for f in ctx.state.get("generated_files", []) or []:
            if f.get("file_id"):
                pre_existing_file_ids.add(f["file_id"])

    writer({"event": "subagent_progress", "subagent_id": subagent_id,
            "sub_query": request[:200], "status": "started",
            "subagent_type": "office"})

    iteration = 0
    while True:
        iteration += 1
        calls_used = sum(counts.values())
        # Budget counts ALL attempts (successful + dedup-blocked + errors).
        total_attempts = len(observations)
        if calls_used >= tool_budget or total_attempts >= tool_budget or iteration > tool_budget + 2:
            break

        user = _build_subagent_user_prompt(
            request, evidence_text, accumulated_data_text,
            tools_text, iteration, tool_budget, calls_used, observations,
        )

        writer({"event": "subagent_progress", "subagent_id": subagent_id,
                "sub_query": request[:200], "status": "tool_call",
                "tool": "thinking", "label": "Office: thinking",
                "subagent_type": "office", "iteration": iteration})

        try:
            tool_temp = get_setting(ctx.db, "TOOL_CALL_TEMPERATURE", ctx.org_id)
            if iteration == 1:
                from app.services.agentic_rag.llm_factory import get_org_llm
                cfg = get_org_llm(ctx.org_id, ctx.db, role="chat")
                logger.info("[office_subagent %s] model=%s base=%s",
                            subagent_id, cfg["model_name"], cfg["api_base"])
            llm = build_chat_llm(ctx.org_id, ctx.db, role="chat", temperature=tool_temp)
            resp = await llm.bind_tools(tools_list).ainvoke([
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ])
        except Exception as exc:
            logger.warning("[office_subagent %s] LLM call failed: %s", subagent_id, exc)
            break

        parsed = parse_think_response(resp, mode="auto")
        tool_calls = parsed.tool_calls

        if not tool_calls:
            # Sub-agent wrote a summary — we're done
            if isinstance(parsed.final_answer, str) and parsed.final_answer.strip():
                summary = parsed.final_answer.strip()
            if not summary:
                errors = [o.error for o in observations if o.error]
                if errors and not ctx.state.get("generated_files"):
                    summary = f"Failed to generate document: {errors[-1]}"
            generated_files = ctx.state.get("generated_files", []) if ctx.state else []
            new_files = [
                f for f in generated_files
                if f.get("file_id") and f["file_id"] not in pre_existing_file_ids
            ]
            writer({"event": "subagent_progress", "subagent_id": subagent_id,
                    "sub_query": request[:200], "status": "done",
                    "subagent_type": "office",
                    "evidence_count": len(new_files),
                    "summary": summary[:300] if summary else ""})
            break

        # Execute tool calls
        for tc in tool_calls:
            name = tc.get("tool")
            args = tc.get("arguments", {})
            tool = tools.get(name)

            # Dedup guard: skip identical tool+key-arg combinations.
            # office_generate with append=true is allowed to repeat (appending
            # slides/sections to the same file), so we include append in the
            # signature for office_generate.
            if name == "office_generate":
                sig_key = f"{args.get('format')}:{args.get('append', False)}"
            elif name == "office_edit":
                sig_key = f"{args.get('file_id')}:{json.dumps(args.get('commands', []), default=str)[:100]}"
            elif name == "office_inspect":
                sig_key = f"{args.get('file_id')}:{args.get('mode', 'outline')}"
            else:
                sig_key = str(args.get("format") or "")
            sig = f"{name}:{sig_key}"
            if sig in seen_signatures:
                observations.append(Observation(
                    tool=name, arguments=args, result={},
                    error=f"Duplicate call: {sig} already tried. Try a different approach.",
                    tokens=0,
                ))
                continue
            seen_signatures.add(sig)

            if tool is None:
                observations.append(Observation(
                    tool=name, arguments=args, result={},
                    error=f"Tool '{name}' not available", tokens=0,
                ))
                continue

            if calls_used >= tool_budget:
                observations.append(Observation(
                    tool=name, arguments=args, result={},
                    error=f"Tool-call budget ({tool_budget}) exhausted. Write your summary.",
                    tokens=0,
                ))
                continue

            label = getattr(tool, "ui_label", f"office: {name}")
            writer({"event": "subagent_progress", "subagent_id": subagent_id,
                    "sub_query": request[:200], "status": "tool_call",
                    "tool": name, "label": label,
                    "subagent_type": "office", "iteration": iteration})

            result = await _run_tool(tool, name, args)
            obs = Observation(
                tool=result["tool"], arguments=result["arguments"],
                result=result.get("result", {}), error=result.get("error"),
                tokens=result.get("tokens", 0),
            )
            observations.append(obs)
            counts[name] = counts.get(name, 0) + 1
            calls_used += 1

            # Sync observations to ctx.state so office_inspect/office_edit can
            # find file_id from office_generate.
            if ctx.state is not None:
                ctx.state["_office_subagent_observations"] = observations

            # Emit observation
            summary_text = ""
            file_created = False
            if obs.result:
                if obs.tool == "office_generate" and obs.result.get("file_id"):
                    summary_text = f"Created: {obs.result.get('file_name', '')}"
                    file_created = True
                    writer({"event": "file", "file_id": obs.result["file_id"],
                            "file_name": obs.result.get("file_name", ""),
                            "format": obs.result.get("format", "")})
                else:
                    summary_text = json.dumps(obs.result, default=str)[:150]
            if obs.error:
                logger.warning("[office_subagent %s] tool %s failed: %s",
                               subagent_id, obs.tool, obs.error)
            writer({"event": "subagent_progress", "subagent_id": subagent_id,
                    "sub_query": request[:200], "status": "tool_done",
                    "tool": obs.tool, "label": label,
                    "hit_count": 1 if file_created else 0,
                    "error": bool(obs.error), "summary": summary_text,
                    "subagent_type": "office", "iteration": iteration})

    # Collect results from state — only files generated during THIS run count.
    generated_files = ctx.state.get("generated_files", []) if ctx.state else []
    new_files = [
        f for f in generated_files
        if f.get("file_id") and f["file_id"] not in pre_existing_file_ids
    ]
    if new_files:
        latest = new_files[-1]
        return {
            "ok": True,
            "file_id": latest.get("file_id"),
            "file_name": latest.get("file_name"),
            "format": latest.get("format"),
            "summary": summary or f"Created {latest.get('file_name', 'document')}",
            "error": None,
        }

    # No file generated — check observations for errors
    errors = [o.error for o in observations if o.error]
    error_msg = errors[-1] if errors else "No document was generated"
    return {
        "ok": False,
        "file_id": None,
        "file_name": None,
        "format": None,
        "summary": summary or f"Failed to generate document: {error_msg}",
        "error": error_msg,
    }
