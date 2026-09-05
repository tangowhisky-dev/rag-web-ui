"""Tool-call parser for the enterprise agent loop.

Supports three tiers:
1. Native function-calling (response.tool_calls).
2. JSON-text fallback (extract {"tool_calls": [...]} / {"tool": ...} / {"final_answer": ...}).
3. Final-answer default (treat raw content as the final answer).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

import json_repair
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)


class ParsedThinkResponse:
    """Normalized result of parsing a `think_node` LLM response."""

    def __init__(self, final_answer: Optional[str] = None, tool_calls: Optional[list[dict]] = None):
        self.final_answer = final_answer
        self.tool_calls = tool_calls or []


def _repair_json_brackets(text: str) -> str:
    """Best-effort repair for the extra/missing closing bracket local models
    sometimes emit (e.g. one stray '}' before the final ']'). Drops closing
    brackets that don't match any open scope and appends closers for any
    scope still open at the end.
    """
    pairs = {"}": "{", "]": "["}
    stack: list[str] = []
    out: list[str] = []
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
        elif ch in "{[":
            stack.append(ch)
            out.append(ch)
        elif ch in "}]":
            if stack and stack[-1] == pairs[ch]:
                stack.pop()
                out.append(ch)
            # else: unmatched closing bracket — drop it silently.
        else:
            out.append(ch)
    while stack:
        opener = stack.pop()
        out.append("}" if opener == "{" else "]")
    return "".join(out)


def _extract_json_block(text: str) -> Optional[str]:
    """Extract the first JSON object or array from a markdown/code-fenced string."""
    # Try code fences first.
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    # Fall back to the first { ... } or [ ... ] block.
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    return match.group(0).strip() if match else None


def _normalize_tool_calls(raw: Any) -> list[dict]:
    """Coerce native tool calls or parsed JSON into a list of {tool, arguments} dicts."""
    calls: list[dict] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                name = item.get("function", {}).get("name") if "function" in item else item.get("tool") or item.get("name")
                args = item.get("function", {}).get("arguments") if "function" in item else item.get("arguments", item.get("args", {}))
                if isinstance(args, str):
                    try:
                        args = json_repair.loads(args)
                    except Exception:
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                if name:
                    calls.append({"tool": name, "arguments": args})
    return calls


def _dispatch_parsed_json(parsed: Any) -> Optional[ParsedThinkResponse]:
    """Map a parsed JSON object to tool calls or a final answer."""
    if "tool_calls" in parsed:
        tool_calls = _normalize_tool_calls(parsed["tool_calls"])
        if tool_calls:
            return ParsedThinkResponse(tool_calls=tool_calls)
    elif "tool" in parsed or "name" in parsed:
        name = parsed.get("tool") or parsed.get("name")
        args = parsed.get("arguments", parsed.get("args", {}))
        if name:
            return ParsedThinkResponse(tool_calls=[{"tool": name, "arguments": args}])
    elif "final_answer" in parsed:
        return ParsedThinkResponse(final_answer=parsed["final_answer"])
    elif isinstance(parsed, dict) and len(parsed) == 1:
        # Malformed shorthand some local models emit, e.g.
        # {"search_dense": {"query": "..."}} instead of the
        # documented {"tool": ..., "arguments": ...} shape.
        # Treat the single key as the tool name.
        (name, args), = parsed.items()
        if isinstance(args, dict):
            return ParsedThinkResponse(tool_calls=[{"tool": name, "arguments": args}])
    elif isinstance(parsed, dict) and "chart_type" in parsed and "data" in parsed:
        # Some local models, asked for a chart, write the
        # chart_generate *arguments* directly as the answer body
        # instead of an actual tool call — e.g.
        # {"chart_type": "bar", "data": [...], "title": "..."}.
        # Recognize the tool's own argument shape and dispatch it
        # as a real chart_generate call instead of letting this
        # raw JSON leak into the final answer text verbatim.
        return ParsedThinkResponse(tool_calls=[{"tool": "chart_generate", "arguments": parsed}])
    return None


def _parse_json_text_fallback(raw: str) -> Optional[ParsedThinkResponse]:
    """Tier 2: try to extract tool calls or final answer from JSON text."""
    try:
        block = _extract_json_block(raw)
        if block:
            try:
                parsed = json_repair.loads(block)
            except Exception:
                try:
                    parsed = json.loads(block)
                except json.JSONDecodeError:
                    parsed = json.loads(_repair_json_brackets(block))
            result = _dispatch_parsed_json(parsed)
            if result is not None:
                return result
    except Exception as exc:
        logger.warning("[tool_call_parser] JSON fallback parse failed: %s", exc)
    return None


def parse_think_response(
    response: AIMessage,
    mode: str = "auto",
) -> ParsedThinkResponse:
    """Parse an LLM response into either a final answer or a list of tool calls."""
    tool_calls: list[dict] = []
    final_answer: Optional[str] = None

    # Tier 1: native function-calling.
    if mode in ("auto", "native") and getattr(response, "tool_calls", None):
        tool_calls = _normalize_tool_calls(response.tool_calls)
        if tool_calls:
            logger.debug("[tool_call_parser] native tool_calls: %s", tool_calls)
            return ParsedThinkResponse(tool_calls=tool_calls)

    raw = str(response.content) if response.content else ""

    if mode == "auto":
        logger.warning("[tool_call_parser] gateway returned no native tool_calls — falling back to JSON-text parsing")

    # Tier 2: JSON-text fallback.
    if mode in ("auto", "json_text"):
        result = _parse_json_text_fallback(raw)
        if result is not None:
            return result

    # Tier 3: final-answer default.
    return ParsedThinkResponse(final_answer=raw)
