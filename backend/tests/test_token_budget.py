"""Tests for token-budget helpers."""

import pytest

from app.services.agentic_rag.token_budget import count_tokens, get_tokenizer


class TestGetTokenizer:
    def test_openai_model(self):
        enc = get_tokenizer("gpt-4")
        assert enc is not None

    def test_unknown_model_falls_back(self):
        enc = get_tokenizer("some/local-model")
        assert enc is not None


class TestCountTokens:
    def test_empty(self):
        assert count_tokens("") == 0

    def test_non_empty(self):
        assert count_tokens("hello world") > 1

    def test_message_list(self):
        messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        assert count_tokens(messages) > 0
