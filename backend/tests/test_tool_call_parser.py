"""Tests for the tool-call parser used by think_node."""

from langchain_core.messages import AIMessage

from app.services.agentic_rag.tool_call_parser import parse_think_response


def _msg(content, tool_calls=None):
    return AIMessage(content=content, tool_calls=tool_calls or [])


class TestParseThinkResponse:
    def test_native_tool_call(self):
        resp = _msg(
            "",
            tool_calls=[{"id": "tc_1", "name": "rag_retrieve", "args": {"query": "revenue"}}],
        )
        parsed = parse_think_response(resp, mode="native")
        assert parsed.final_answer is None
        assert len(parsed.tool_calls) == 1
        assert parsed.tool_calls[0]["tool"] == "rag_retrieve"
        assert parsed.tool_calls[0]["arguments"]["query"] == "revenue"

    def test_json_text_tool_calls(self):
        resp = _msg('```json\n{"tool_calls": [{"tool": "chart_generate", "arguments": {"chart_type": "bar"}}]}\n```')
        parsed = parse_think_response(resp, mode="json_text")
        assert len(parsed.tool_calls) == 1
        assert parsed.tool_calls[0]["tool"] == "chart_generate"

    def test_json_text_single_tool(self):
        resp = _msg('{"tool": "rag_retrieve", "arguments": {"query": "q"}}')
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

    def test_malformed_json_fallback(self):
        resp = _msg("not { valid json")
        parsed = parse_think_response(resp, mode="json_text")
        assert parsed.final_answer == "not { valid json"
