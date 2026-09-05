"""Low-level helpers for the agent loop.

Contains utilities with no dependency on other agent_graph sub-modules:
observation coercion, stream-writer access, per-tool call budgets,
transient-error detection, correction hints, balanced-text extraction,
chart-marker substitution, JSON-block extraction, and wall-clock budget
checking.
"""

from __future__ import annotations

import json
import re
import time

from app.services.agentic_rag.schemas import Observation
from app.services.settings_service import get_setting
from langgraph.config import get_stream_writer


def _coerce_observation(obs: Observation | dict) -> Observation:
    """Coerce a dict (e.g. restored from Redis checkpoint) to an Observation.

    Redis checkpoint serializes Pydantic models as LangChain constructor
    dicts: {"lc": 2, "type": "constructor", "id": [...], "kwargs": {...}}.
    The actual fields live under "kwargs".
    """
    if isinstance(obs, Observation):
        return obs
    if isinstance(obs, dict):
        if "kwargs" in obs and "lc" in obs:
            return Observation(**obs["kwargs"])
        return Observation(**obs)
    return Observation(tool=str(obs))


def _writer():
    """Return a stream writer if one is available, else a no-op."""
    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda x: None


# Per-turn call caps for tools that can be invoked in a tight loop.
# Resolved per-request via the settings service (org-overridable).
def _tool_call_budget(db, org_id) -> dict:
    return {
        # Atomic search tools
        "search_exact": get_setting(db, "AGENT_MAX_SEARCH_EXACT", org_id),
        "search_sparse": get_setting(db, "AGENT_MAX_SEARCH_SPARSE", org_id),
        "search_dense": get_setting(db, "AGENT_MAX_SEARCH_DENSE", org_id),
        "rerank_results": get_setting(db, "AGENT_MAX_RERANK", org_id),
        "graph_expand": get_setting(db, "AGENT_MAX_GRAPH_EXPAND", org_id),
        # Discovery
        "kb_search_documents": get_setting(db, "AGENT_MAX_KB_SEARCH", org_id),
        # Read
        "kb_grep": get_setting(db, "AGENT_MAX_KB_GREP", org_id),
        "kb_read": get_setting(db, "AGENT_MAX_KB_READ", org_id),
        "kb_outline": get_setting(db, "AGENT_MAX_KB_READ", org_id),
        # Processing
        "code_execute": get_setting(db, "AGENT_MAX_CODE_EXEC", org_id),
        "extract_data": get_setting(db, "AGENT_MAX_EXTRACT_DATA", org_id),
        "chart_generate": get_setting(db, "AGENT_MAX_CHART_GENERATE", org_id),
        # Office document generation
        "office_load_skill": get_setting(db, "AGENT_MAX_OFFICE_LOAD_SKILL", org_id),
        "office_generate": get_setting(db, "AGENT_MAX_OFFICE_GENERATE", org_id),
        "office_inspect": get_setting(db, "AGENT_MAX_OFFICE_INSPECT", org_id),
        "office_edit": get_setting(db, "AGENT_MAX_OFFICE_EDIT", org_id),
    }


def _total_tool_budget(db, org_id) -> int:
    """Total tool-call budget across all tools per turn."""
    return get_setting(db, "AGENT_TOTAL_TOOL_BUDGET", org_id)

# Error patterns that indicate a transient infrastructure failure rather
# than a bad-argument error.  Transient failures retry with the same
# arguments (plus backoff); argument failures call the correction LLM.
_TRANSIENT_ERROR_PATTERNS = (
    "timeout", "timed out", "connection", "network", "unreachable",
    "temporarily", "broken pipe", "reset by peer", "i/o error",
    "errno 5", "errno 11", "errno 104", "errno 110",
    "rate limit", "429", "too many requests", "quota", "throttle",
    "503", "502", "service unavailable", "bad gateway",
)



def _is_transient_error(error: str) -> bool:
    return any(p in (error or "").lower() for p in _TRANSIENT_ERROR_PATTERNS)


def _extract_balanced(text: str, chars: tuple[str, str]) -> str | None:
    """Return the first balanced *chars* region in *text* while respecting strings."""
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


def _substitute_chart_markers(text: str, chart_options: list[dict]) -> str:
    """Replace [[CHART_N]] placeholders with the real ECharts fence.

    Any chart whose marker the model omitted is appended at the end, so a
    chart is never silently dropped even if placement wasn't followed.
    """
    result = text
    for i, option in enumerate(chart_options, start=1):
        marker = f"[[CHART_{i}]]"
        fence = f"```echarts\n{json.dumps(option)}\n```"
        if marker in result:
            result = result.replace(marker, fence, 1)
        else:
            result = f"{result}\n\n{fence}"
    return result


def _substitute_office_markers(text: str, office_files: list[dict]) -> str:
    """Replace [[DOC_N]] placeholders with markdown download links.

    Any file whose marker the model omitted is appended at the end.
    The frontend intercepts office:// links and renders file chips.
    """
    result = text
    for i, f in enumerate(office_files, start=1):
        marker = f"[[DOC_{i}]]"
        link = f"[📄 {f.get('file_name', 'document')}](office://{f['file_id']})"
        if marker in result:
            result = result.replace(marker, link, 1)
        else:
            result = f"{result}\n\n{link}"
    return result


def _extract_json_block(text: str) -> str | None:
    """Return the first well-formed JSON object or array string from *text*.

    Tries markdown fenced blocks first, then scans for balanced braces or brackets.
    """
    if not text:
        return None
    # Prefer a fenced ```json ... ``` block.
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        block = _extract_balanced(m.group(1), ("{", "}")) or _extract_balanced(m.group(1), ("[", "]"))
        if block:
            return block
    # Fall back to the first inline balanced object or array.
    return _extract_balanced(text, ("{", "}")) or _extract_balanced(text, ("[", "]"))


def _wall_clock_exceeded(state) -> bool:
    started_at = state.get("started_at")
    if started_at is None:
        return False
    from app.db.session import SessionLocal
    org_id = state.get("org_id")
    _db = SessionLocal()
    try:
        max_seconds = get_setting(_db, "AGENT_MAX_WALL_SECONDS", org_id)
    finally:
        _db.close()
    return (time.monotonic() - started_at) >= max_seconds
