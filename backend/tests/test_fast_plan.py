"""Unit tests for the fast-pipeline planner + step resolver.

fast_plan runs one utility-model call producing intent + ordered steps;
fast_step materializes each step's tool_calls between tool rounds (literal
args pass through, needs_prior args are compiled by a utility call).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.agentic_rag.agent_graph_v2.fast_plan import (
    _MAX_TOOL_CALLS,
    _normalize_steps,
    _parse_plan_json,
    _search_step_calls,
    _step_to_tool_calls,
    fast_plan_node,
    fast_step_node,
    route_fast_plan,
    route_fast_step,
)


def _ctx():
    c = MagicMock()
    c.db = MagicMock()
    c.org_id = None
    return c


# ── _parse_plan_json ──────────────────────────────────────────────────────────


def test_parse_plain_json():
    assert _parse_plan_json('{"intent": "retrieve"}') == {"intent": "retrieve"}


def test_parse_fenced_json():
    assert _parse_plan_json('```json\n{"intent": "retrieve"}\n```') == {"intent": "retrieve"}


def test_parse_garbage_returns_none():
    assert _parse_plan_json("not json at all") is None
    assert _parse_plan_json("") is None


def test_parse_non_object_returns_none():
    assert _parse_plan_json('[1, 2, 3]') is None


def test_parse_strips_think_block():
    """Thinking-model output: <think> contains braces; JSON still extracts."""
    out = '<think>let me plan {"x": 1}</think>{"intent": "retrieve", "steps": []}'
    assert _parse_plan_json(out) == {"intent": "retrieve", "steps": []}


# ── _normalize_steps ─────────────────────────────────────────────────────────


def test_normalize_legacy_sub_queries_becomes_search_step():
    steps = _normalize_steps({"sub_queries": [{"query": "x"}]})
    assert steps == [{"tool": "search", "sub_queries": [{"query": "x"}]}]


def test_normalize_explicit_steps():
    steps = _normalize_steps({"steps": [
        {"tool": "search", "sub_queries": [{"query": "a"}]},
        {"tool": "code_execute", "spec": {"task": "t"}, "needs_prior": True},
    ]})
    assert [s["tool"] for s in steps] == ["search", "code_execute"]


def test_normalize_empty():
    assert _normalize_steps({}) == []


# ── _search_step_calls ───────────────────────────────────────────────────────


def test_subquery_expands_to_hybrid_pair():
    calls = _search_step_calls(
        {"tool": "search", "sub_queries": [
            {"query": "meal cap", "tool_hint": "any",
             "filters": {"document_status": "active"}}]},
        kb_ids=[3],
    )
    assert [c["tool"] for c in calls] == ["keyword_search", "semantic_search"]
    for c in calls:
        assert c["arguments"]["query"] == "meal cap"
        assert c["arguments"]["filters"] == {"document_status": "active"}
        assert c["arguments"]["kb_ids"] == [3]


def test_title_hint_pairs_with_hybrid():
    calls = _search_step_calls(
        {"tool": "search", "sub_queries": [{"query": "travel policy", "tool_hint": "title_search"}]},
        kb_ids=[3],
    )
    tools = [c["tool"] for c in calls]
    assert "title_search" in tools
    assert "keyword_search" in tools and "semantic_search" in tools
    ts = next(c for c in calls if c["tool"] == "title_search")
    assert ts["arguments"]["title_contains"] == "travel policy"
    assert ts["arguments"]["metadata_only"] is True


def test_file_read_hint_emits_per_document():
    calls = _search_step_calls(
        {"tool": "search", "sub_queries": [
            {"query": "", "tool_hint": "file_read", "document_ids": [10, 20]}]},
        kb_ids=[3],
    )
    assert [c["tool"] for c in calls] == ["file_read", "file_read"]
    assert [c["arguments"]["document_id"] for c in calls] == [10, 20]


def test_multi_part_compare_gets_pair_per_part():
    calls = _search_step_calls(
        {"tool": "search", "sub_queries": [
            {"query": "old price"}, {"query": "new price"}]},
        kb_ids=[3],
    )
    assert len(calls) == 4
    assert {c["arguments"]["query"] for c in calls} == {"old price", "new price"}


def test_tool_calls_capped():
    step = {"tool": "search",
            "sub_queries": [{"query": f"q{i}"} for i in range(10)]}
    calls = _search_step_calls(step, kb_ids=[3])
    assert len(calls) <= _MAX_TOOL_CALLS


# ── _step_to_tool_calls (non-search steps) ───────────────────────────────────


@pytest.mark.asyncio
async def test_literal_args_pass_through():
    state = {"kb_ids": [3], "observations": [], "fast_plan": {}}
    calls = await _step_to_tool_calls(
        {"tool": "kb_metadata", "args": {"kb_ids": [3]}}, state, _ctx())
    assert calls == [{"tool": "kb_metadata", "arguments": {"kb_ids": [3]}}]


@pytest.mark.asyncio
async def test_needs_prior_uses_arg_compiler():
    state = {"kb_ids": [3], "observations": [], "fast_plan": {},
             "original_query": "compute 17% of the total"}
    resp = MagicMock(); resp.content = '{"code": "print(17/100*200)"}'
    with patch("app.services.agentic_rag.agent_graph_v2.fast_plan.build_chat_llm") as m:
        m.return_value.ainvoke = AsyncMock(return_value=resp)
        calls = await _step_to_tool_calls(
            {"tool": "code_execute", "spec": {"task": "17% of 200"}, "needs_prior": True},
            state, _ctx())
    assert calls == [{"tool": "code_execute",
                      "arguments": {"code": "print(17/100*200)"}}]


@pytest.mark.asyncio
async def test_non_fast_tool_skipped():
    state = {"kb_ids": [3], "observations": [], "fast_plan": {}}
    calls = await _step_to_tool_calls(
        {"tool": "create_office_document", "args": {}}, state, _ctx())
    assert calls == []


@pytest.mark.asyncio
async def test_arg_compiler_failure_returns_empty():
    state = {"kb_ids": [3], "observations": [], "fast_plan": {}}
    with patch("app.services.agentic_rag.agent_graph_v2.fast_plan.build_chat_llm") as m:
        m.return_value.ainvoke = AsyncMock(side_effect=RuntimeError("llm down"))
        calls = await _step_to_tool_calls(
            {"tool": "summarize", "spec": {"task": "summarize file"}, "needs_prior": True},
            state, _ctx())
    assert calls == []


# ── routes ───────────────────────────────────────────────────────────────────


def test_route_tool_when_calls():
    assert route_fast_plan({"tool_calls": [{"tool": "x", "arguments": {}}]}) == "tool"
    assert route_fast_step({"tool_calls": [{"tool": "x", "arguments": {}}]}) == "tool"


def test_route_post_process_when_empty():
    assert route_fast_plan({"tool_calls": []}) == "post_process"
    assert route_fast_step({"tool_calls": []}) == "post_process"


# ── fast_plan_node ────────────────────────────────────────────────────────────


def _state(query="q", **kw):
    return {"original_query": query, "kb_ids": [3], "messages": [],
            "last_answer_object": None, "kb_profile": {}, "observations": [],
            "file_markdown": None, **kw}


@pytest.mark.asyncio
async def test_answer_from_history_emits_no_tool_calls():
    resp = MagicMock()
    resp.content = '{"intent": "answer_from_history", "resolved_query": "summarize the travel answer"}'
    with patch("app.services.agentic_rag.agent_graph_v2.fast_plan.build_chat_llm") as m:
        m.return_value.ainvoke = AsyncMock(return_value=resp)
        out = await fast_plan_node(_state("summarize this"), _ctx())
    assert out["tool_calls"] == []
    assert out["fast_plan"]["intent"] == "answer_from_history"


@pytest.mark.asyncio
async def test_direct_intent_emits_no_tool_calls():
    resp = MagicMock()
    resp.content = '{"intent": "direct", "resolved_query": "what is 2+2"}'
    with patch("app.services.agentic_rag.agent_graph_v2.fast_plan.build_chat_llm") as m:
        m.return_value.ainvoke = AsyncMock(return_value=resp)
        out = await fast_plan_node(_state("what is 2+2"), _ctx())
    assert out["tool_calls"] == []
    assert out["fast_plan"]["intent"] == "direct"


@pytest.mark.asyncio
async def test_planner_failure_falls_back_to_hybrid():
    with patch("app.services.agentic_rag.agent_graph_v2.fast_plan.build_chat_llm") as m:
        m.return_value.ainvoke = AsyncMock(side_effect=RuntimeError("llm down"))
        out = await fast_plan_node(_state("what is the meal cap"), _ctx())
    tools = [c["tool"] for c in out["tool_calls"]]
    assert tools == ["keyword_search", "semantic_search"]
    assert all(c["arguments"]["query"] == "what is the meal cap" for c in out["tool_calls"])


@pytest.mark.asyncio
async def test_plan_with_processing_step_cursor_advances():
    resp = MagicMock()
    resp.content = json.dumps({
        "intent": "retrieve",
        "resolved_query": "chart the extracted values",
        "steps": [
            {"tool": "search", "sub_queries": [{"query": "sales data"}]},
            {"tool": "extract_data", "spec": {"task": "pull numbers"}, "needs_prior": True},
            {"tool": "chart_generate", "spec": {"task": "bar chart"}, "needs_prior": True},
        ],
    })
    with patch("app.services.agentic_rag.agent_graph_v2.fast_plan.build_chat_llm") as m:
        m.return_value.ainvoke = AsyncMock(return_value=resp)
        out = await fast_plan_node(_state("chart the extracted values"), _ctx())
    # Step 0 compiled (search round); cursor advanced to 1.
    assert out["fast_plan"]["cursor"] == 1
    assert len(out["fast_plan"]["steps"]) == 3
    assert {c["tool"] for c in out["tool_calls"]} == {"keyword_search", "semantic_search"}


# ── fast_step_node ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fast_step_emits_next_literal_step():
    state = _state("q")
    state["fast_plan"] = {
        "intent": "retrieve", "steps": [
            {"tool": "search", "sub_queries": [{"query": "a"}]},
            {"tool": "kb_metadata", "args": {"kb_ids": [3]}},
        ],
        "cursor": 1, "repairs": 0, "last_emitted": 2,
    }
    state["observations"] = [
        {"tool": "keyword_search", "result": {"hits": []}, "error": None},
        {"tool": "semantic_search", "result": {"hits": []}, "error": None},
    ]
    out = await fast_step_node(state, _ctx())
    assert out["tool_calls"] == [{"tool": "kb_metadata", "arguments": {"kb_ids": [3]}}]
    assert out["fast_plan"]["cursor"] == 2


@pytest.mark.asyncio
async def test_fast_step_done_when_steps_exhausted():
    state = _state("q")
    state["fast_plan"] = {"intent": "retrieve", "steps": [{"tool": "search"}],
                          "cursor": 1, "repairs": 0, "last_emitted": 2}
    state["observations"] = [
        {"tool": "keyword_search", "result": {}, "error": None},
        {"tool": "semantic_search", "result": {}, "error": None},
    ]
    out = await fast_step_node(state, _ctx())
    assert out["tool_calls"] == []


@pytest.mark.asyncio
async def test_fast_step_repair_recompiles_spec_step():
    """Errored needs_prior step → arg compiler re-runs with error context."""
    state = _state("q")
    state["fast_plan"] = {
        "intent": "retrieve", "steps": [
            {"tool": "code_execute", "spec": {"task": "compute"}, "needs_prior": True},
            {"tool": "summarize", "args": {"text": "x"}},
        ],
        "cursor": 1, "repairs": 0, "last_emitted": 1,
    }
    state["observations"] = [
        {"tool": "code_execute", "result": {}, "error": "NameError: foo"},
    ]
    resp = MagicMock(); resp.content = '{"code": "print(4)"}'
    with patch("app.services.agentic_rag.agent_graph_v2.fast_plan.build_chat_llm") as m:
        m.return_value.ainvoke = AsyncMock(return_value=resp)
        out = await fast_step_node(state, _ctx())
    assert out["tool_calls"] == [{"tool": "code_execute", "arguments": {"code": "print(4)"}}]
    assert out["fast_plan"]["repairs"] == 1
    assert out["fast_plan"]["cursor"] == 1   # cursor not advanced during repair
    # The error was fed into the compiler prompt.
    assert "NameError" in m.return_value.ainvoke.call_args[0][0][1].content


@pytest.mark.asyncio
async def test_fast_step_literal_args_failure_abandons():
    """Literal-args steps that fail are not retried — args can't change."""
    state = _state("q")
    state["fast_plan"] = {
        "intent": "retrieve",
        "steps": [{"tool": "kb_metadata", "args": {}}],
        "cursor": 1, "repairs": 0, "last_emitted": 1,
    }
    state["observations"] = [{"tool": "kb_metadata", "result": {}, "error": "boom"}]
    out = await fast_step_node(state, _ctx())
    assert out["tool_calls"] == []


@pytest.mark.asyncio
async def test_fast_step_abandons_after_repair_used():
    state = _state("q")
    state["fast_plan"] = {
        "intent": "retrieve",
        "steps": [{"tool": "code_execute", "args": {"code": "bad"}},
                  {"tool": "summarize", "args": {"text": "x"}}],
        "cursor": 1, "repairs": 1, "last_emitted": 1,
    }
    state["observations"] = [{"tool": "code_execute", "result": {}, "error": "still bad"}]
    out = await fast_step_node(state, _ctx())
    assert out["tool_calls"] == []   # give up → post_process


import json  # noqa: E402  (used by test_plan_with_processing_step_cursor_advances)
