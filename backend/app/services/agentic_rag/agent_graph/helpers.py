"""Low-level helpers for the agent loop.

Contains utilities with no dependency on other agent_graph sub-modules:
observation coercion, stream-writer access, per-tool call budgets,
transient-error detection, correction hints, balanced-text extraction,
chart-marker substitution, JSON-block extraction, wall-clock budget
checking, and the unified timeline event emitter.
"""

from __future__ import annotations

import json
import re
import time
import uuid

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
    except (RuntimeError, KeyError):
        return lambda x: None


# ── Unified timeline event emitter ───────────────────────────────────────────
#
# All chain-of-thought display events (phases, thinking, tool calls, tool
# results, subagent lifecycle) flow through this single helper as
# `{"event": "timeline", ...}` dicts.  The frontend builds one ordered
# list from these events — no more reconstructing a timeline from
# disconnected agent_step + tool_call + subagent_progress + thinking arrays.
#
# Step types:
#   phase        — agent loop phase (Analyzing query, Thinking, etc.)
#   thinking     — reasoning content from a thinking model (inline in CoT)
#   tool_call    — main-agent tool about to run
#   tool_result  — main-agent tool finished
#   subagent_start — subagent spawned (retrieval or office)
#   subagent_step  — subagent internal step (tool call / tool result / thinking)
#   subagent_done  — subagent finished
#
# Every step has a unique `id`.  Updates to the same step (e.g. active →
# complete, thinking content accumulation) reuse the same id.

_timeline_counter = 0

# Per-request debug stream: when the chat message body carries debug=true,
# stage internals (tool args, observation payloads, think/finalize prompts)
# are emitted as `type="debug"` timeline events. Default off — production
# streams carry nothing extra. Set per request via set_debug_stream() in
# generate_response; propagates through asyncio.gather/create_task children.
from contextvars import ContextVar

_agent_debug_stream: ContextVar[bool] = ContextVar("agent_debug_stream", default=False)


def set_debug_stream(enabled: bool) -> None:
    _agent_debug_stream.set(enabled)


def debug_emit(stage: str, data: dict) -> None:
    """Emit a `type="debug"` timeline event when debug streaming is on."""
    if not _agent_debug_stream.get():
        return
    _emit_timeline(type="debug", stage=stage, data=data)


def _timeline_id(prefix: str = "s") -> str:
    """Generate a unique timeline step id."""
    global _timeline_counter
    _timeline_counter += 1
    return f"{prefix}-{_timeline_counter}"


def _emit_timeline(**kwargs) -> str:
    """Emit a timeline event and return the step id.

    The caller passes the step type and any fields.  `id` is auto-generated
    if not provided.  `ts` is auto-stamped.
    """
    writer = _writer()
    step_id = kwargs.pop("id", None) or _timeline_id(kwargs.get("type", "s"))
    payload = {"event": "timeline", "id": step_id, "ts": time.time(), **kwargs}
    writer(payload)
    return step_id


def _compact_value(v, depth: int = 0):
    """Truncate a tool-arg value for observability — keeps shape, bounds size."""
    if isinstance(v, str):
        return v if len(v) <= 200 else f"{v[:200]}…[{len(v)} chars]"
    if isinstance(v, list):
        items = [_compact_value(x, depth + 1) for x in v[:10]]
        if len(v) > 10:
            items.append(f"…[+{len(v) - 10} items]")
        return items
    if isinstance(v, dict):
        if depth >= 2:
            return "{…}"
        return {k: _compact_value(x, depth + 1) for k, x in v.items()}
    return v


def _compact_args(args: dict | None) -> dict:
    """Compact a tool-call arguments dict for timeline events.

    Raw args can be huge (office tools carry full document content), so
    strings/lists/dicts are bounded — consumers see *what* was passed
    without hauling the payload over the SSE stream.
    """
    return {k: _compact_value(v) for k, v in (args or {}).items()}


_HIT_BRIEF_KEYS = (
    "document_id", "file_id", "chunk_index", "page", "title", "file_name",
    "score", "_reranker_score", "_is_neighbor", "document_status",
    "effective_from", "effective_to", "version", "owner", "content_hash",
    "source",
)


def _hit_brief(item):
    """Per-hit/doc brief for debug streams — identity + authority metadata,
    content only as a short preview (the full text would flood the stream)."""
    if not isinstance(item, dict):
        return _compact_value(item)
    meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    brief: dict = {}
    for src in (item, meta):
        for k in _HIT_BRIEF_KEYS:
            if brief.get(k) is None and src.get(k) is not None:
                brief[k] = src[k]
    content = item.get("page_content") or item.get("content") or ""
    if content:
        brief["content_preview"] = content[:160]
        brief["content_len"] = len(content)
    return brief


def _result_brief(result):
    """Compact a tool result for debug streams — keeps the metadata that
    matters (per-hit identity/authority fields, counts, errors) while
    bounding content strings. Dict/list nesting otherwise collapses at
    depth 2 in _compact_value and loses exactly the fields evaluators need."""
    if not isinstance(result, dict):
        return _compact_value(result)
    out: dict = {}
    for k, v in result.items():
        if k in ("hits", "docs") and isinstance(v, list):
            items = [_hit_brief(x) for x in v[:15]]
            if len(v) > 15:
                items.append(f"…[+{len(v) - 15} items]")
            out[k] = items
        elif k in ("content", "page_content", "markdown") and isinstance(v, str):
            out[k] = f"{v[:300]}…[{len(v)} chars]" if len(v) > 300 else v
            out[f"{k}_len"] = len(v)
        else:
            out[k] = _compact_value(v)
    return out


# Per-turn call caps are intentionally NOT enforced here except for
# clarify (human-in-the-loop), which has its own safety cap.
# The only tool usage guards are:
#   1. AGENT_TOTAL_TOOL_BUDGET — total calls across all tools per user query.
#   2. AGENT_MAX_CLARIFY — per-query cap on clarification calls.
#   3. AGENT_MAX_SAME_TOOL_REPEAT — blocks consecutive calls to the same
#      tool with similar/different arguments, preventing local-model loops.
def _tool_call_budget(db, org_id) -> dict:
    return {
        "clarify": get_setting(db, "AGENT_MAX_CLARIFY", org_id),
    }


def _total_tool_budget(db, org_id) -> int:
    """Total tool-call budget across all tools per user query."""
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
    """Remove [[DOC_N]] placeholders from the answer text.

    The frontend renders download chips via the officeFiles prop, so the
    answer text should not contain any marker placeholders. Strip all
    [[DOC_N]] markers (known and unknown indices) and clean up leftover
    whitespace/punctuation around them.
    """
    # Remove all [[DOC_N]] markers regardless of index.
    result = re.sub(r"\[\[DOC_\d+\]\]", "", text)
    # Clean up double spaces left behind by marker removal.
    result = re.sub(r"  +", " ", result)
    # Clean up empty lines left behind.
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


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
