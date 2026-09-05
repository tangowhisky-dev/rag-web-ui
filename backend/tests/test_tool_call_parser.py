"""Tests for the tool-call parser used by think_node."""

from langchain_core.messages import AIMessage

from app.services.agentic_rag.tool_call_parser import parse_think_response


def _msg(content, tool_calls=None):
    return AIMessage(content=content, tool_calls=tool_calls or [])


class TestParseThinkResponse:
    def test_native_tool_call(self):
        resp = _msg(
            "",
            tool_calls=[{"id": "tc_1", "name": "search_dense", "args": {"query": "revenue"}}],
        )
        parsed = parse_think_response(resp, mode="native")
        assert parsed.final_answer is None
        assert len(parsed.tool_calls) == 1
        assert parsed.tool_calls[0]["tool"] == "search_dense"
        assert parsed.tool_calls[0]["arguments"]["query"] == "revenue"

    def test_json_text_tool_calls(self):
        resp = _msg('```json\n{"tool_calls": [{"tool": "chart_generate", "arguments": {"chart_type": "bar"}}]}\n```')
        parsed = parse_think_response(resp, mode="json_text")
        assert len(parsed.tool_calls) == 1
        assert parsed.tool_calls[0]["tool"] == "chart_generate"

    def test_json_text_single_tool(self):
        resp = _msg('{"tool": "search_dense", "arguments": {"query": "q"}}')
        parsed = parse_think_response(resp, mode="json_text")
        assert len(parsed.tool_calls) == 1

    def test_json_text_final_answer(self):
        resp = _msg('{"final_answer": "The revenue is 100."}')
        parsed = parse_think_response(resp, mode="json_text")
        assert parsed.final_answer == "The revenue is 100."
        assert parsed.tool_calls == []

    def test_auto_tries_native_then_json(self):
        # No native tool calls, but valid JSON-text tool call.
        resp = _msg('{"tool": "extract_data", "arguments": {"focus": "revenue"}}')
        parsed = parse_think_response(resp, mode="auto")
        assert parsed.tool_calls[0]["tool"] == "extract_data"

    def test_plain_text_becomes_final_answer(self):
        resp = _msg("I don't know.")
        parsed = parse_think_response(resp, mode="auto")
        assert parsed.final_answer == "I don't know."

    def test_json_text_extra_closing_brace_is_repaired(self):
        # Some local models emit one stray '}' before the array close.
        resp = _msg(
            '{ "tool_calls": [ { "tool": "search_dense", "arguments": '
            '{ "query": "CPU scheduling" } } } ] }'
        )
        parsed = parse_think_response(resp, mode="json_text")
        assert len(parsed.tool_calls) == 1
        assert parsed.tool_calls[0]["tool"] == "search_dense"
        assert parsed.tool_calls[0]["arguments"]["query"] == "CPU scheduling"

    def test_json_text_missing_closing_brace_is_repaired(self):
        resp = _msg('{ "tool": "chart_generate", "arguments": { "chart_type": "pie" }')
        parsed = parse_think_response(resp, mode="json_text")
        assert len(parsed.tool_calls) == 1
        assert parsed.tool_calls[0]["tool"] == "chart_generate"

    def test_malformed_json_fallback(self):
        resp = _msg("not { valid json")
        parsed = parse_think_response(resp, mode="json_text")
        assert parsed.final_answer == "not { valid json"

    def test_shorthand_single_key_tool_call(self):
        # Malformed shape some local models emit instead of the documented
        # {"tool": ..., "arguments": ...} — must not be swallowed as a literal answer.
        resp = _msg('{"search_dense": {"query": "race condition definition"}}')
        parsed = parse_think_response(resp, mode="json_text")
        assert parsed.final_answer is None
        assert len(parsed.tool_calls) == 1
        assert parsed.tool_calls[0]["tool"] == "search_dense"
        assert parsed.tool_calls[0]["arguments"]["query"] == "race condition definition"

    def test_chart_generate_args_written_as_prose_answer(self):
        # Some local models, asked for a chart, write chart_generate's own
        # argument shape directly into the answer body (surrounded by prose)
        # instead of emitting an actual tool call. Must be recognized and
        # dispatched as a real chart_generate call, not leaked as raw JSON
        # text in the final answer.
        resp = _msg(
            "Based on the provided documents, here is the comparison:\n\n"
            "```\n"
            '{\n'
            '  "chart_type": "bar",\n'
            '  "title": "Comparison of Average Waiting Times",\n'
            '  "x_label": "Scheduling Algorithm",\n'
            '  "y_label": "Average Waiting Time (ms)",\n'
            '  "data": [\n'
            '    {"label": "FCFS", "value": 28},\n'
            '    {"label": "SJF", "value": 13},\n'
            '    {"label": "Round Robin", "value": 23}\n'
            '  ]\n'
            '}\n'
            "```\n\n"
            "### Data Summary\n- FCFS: 28ms\n"
        )
        parsed = parse_think_response(resp, mode="json_text")
        assert parsed.final_answer is None
        assert len(parsed.tool_calls) == 1
        assert parsed.tool_calls[0]["tool"] == "chart_generate"
        assert parsed.tool_calls[0]["arguments"]["chart_type"] == "bar"
        assert len(parsed.tool_calls[0]["arguments"]["data"]) == 3
