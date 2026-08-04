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

from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)


class ParsedThinkResponse:
    """Normalized result of parsing a `think_node` LLM response."""

    def __init__(self, final_answer: Optional[str] = None, tool_calls: Optional[list[dict]] = None):
        self.final_answer = final_answer
        self.tool_calls = tool_calls or []


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
                        args = json.loads(args)
                    except Exception:
                        args = {}
                if name:
                    calls.append({"tool": name, "arguments": args})
    return calls


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
        try:
            block = _extract_json_block(raw)
            if block:
                parsed = json.loads(block)
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
                    final_answer = parsed["final_answer"]
                    return ParsedThinkResponse(final_answer=final_answer)
                elif isinstance(parsed, dict) and len(parsed) == 1:
                    # Malformed shorthand some local models emit, e.g.
                    # {"rag_retrieve": {"query": "..."}} instead of the
                    # documented {"tool": ..., "arguments": ...} shape.
                    # Treat the single key as the tool name.
                    (name, args), = parsed.items()
                    if isinstance(args, dict):
                        return ParsedThinkResponse(tool_calls=[{"tool": name, "arguments": args}])
        except Exception as exc:
            logger.warning("[tool_call_parser] JSON fallback parse failed: %s", exc)

    # Tier 3: final-answer default.
    return ParsedThinkResponse(final_answer=raw)
