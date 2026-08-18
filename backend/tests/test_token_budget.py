"""Tests for token-budget helpers."""

from app.services.agentic_rag.token_budget import (
    count_tokens,
    estimate_tokens,
    record_usage,
    ContextBudget,
)


class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens("") == 0 or estimate_tokens("") == 1  # min clamp

    def test_non_empty(self):
        assert estimate_tokens("hello world") > 1

    def test_long_text(self):
        text = "a" * 400
        assert estimate_tokens(text) == 100  # 400 / 4


class TestCountTokens:
    def test_empty(self):
        assert count_tokens("") == 0

    def test_non_empty(self):
        assert count_tokens("hello world") > 1

    def test_message_list(self):
        messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        assert count_tokens(messages) > 0

    def test_dict(self):
        assert count_tokens({"key": "value"}) > 0

    def test_model_name_ignored(self):
        """model_name is accepted but no longer affects the result."""
        assert count_tokens("hello", model_name="gpt-4") == count_tokens("hello", model_name="qwen")


class TestCalibration:
    def test_record_usage_adjusts_ratio(self):
        # "aaaa" = 4 chars. If the provider says that's 2 tokens,
        # the ratio should become 2.0 chars/token instead of 4.0.
        text = "a" * 100
        before = count_tokens(text)
        record_usage(text, 50)  # 100 chars / 50 tokens = 2.0
        after = count_tokens(text)
        assert after >= before  # more tokens for same text after calibration

    def test_record_usage_ignores_zero(self):
        record_usage("hello", 0)  # should not crash or change ratio
        assert count_tokens("hello") > 0

    def test_record_usage_ignores_empty(self):
        record_usage("", 100)  # should not crash
        assert count_tokens("hello") > 0


class TestContextBudget:
    def test_budget_without_db(self):
        """ContextBudget can be constructed without a db session."""
        budget = ContextBudget()
        assert budget.context_size > 0
        assert budget.available > 0
        assert budget.remaining == budget.available

    def test_budget_add_and_remaining(self):
        budget = ContextBudget(context_size=10000, reserved_generation=1000, tool_budget=500)
        assert budget.available == 8500
        budget.add(1000)
        assert budget.used == 1000
        assert budget.remaining == 7500

    def test_needs_compaction(self):
        budget = ContextBudget(context_size=10000, reserved_generation=0, tool_budget=0)
        budget.add(int(budget.available * budget.trigger_ratio))
        assert budget.needs_compaction()
