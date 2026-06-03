"""Unit tests for cancel_registry.py — in-memory cancel token registry."""

import asyncio

import pytest

from app.services.cancel_registry import (
    clear_cancel_token,
    get_cancel_token,
    is_cancelled,
    set_cancel_token,
)

# ---------------------------------------------------------------------------
# Helpers — each test gets a fresh event loop via the fixture below.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _fresh_registry():
    """Reset the module-level registry before every test."""
    import app.services.cancel_registry as reg
    reg._cancel_tokens.clear()
    yield
    reg._cancel_tokens.clear()


# ---------------------------------------------------------------------------
# test_create_and_set
# ---------------------------------------------------------------------------

def test_create_and_set():
    """get_cancel_token creates a token; set_cancel_token sets it."""
    chat_id = 1
    token = get_cancel_token(chat_id)
    assert isinstance(token, asyncio.Event)
    assert not token.is_set()

    set_cancel_token(chat_id)
    assert token.is_set()


# ---------------------------------------------------------------------------
# test_clear
# ---------------------------------------------------------------------------

def test_clear():
    """set then clear — token is_set() returns False after clear."""
    chat_id = 2
    get_cancel_token(chat_id)
    set_cancel_token(chat_id)

    clear_cancel_token(chat_id)
    assert not is_cancelled(chat_id)


# ---------------------------------------------------------------------------
# test_is_cancelled
# ---------------------------------------------------------------------------

def test_is_cancelled():
    """After set, is_cancelled returns True; after clear, returns False."""
    chat_id = 3

    assert not is_cancelled(chat_id)  # no token yet

    set_cancel_token(chat_id)
    assert is_cancelled(chat_id)

    clear_cancel_token(chat_id)
    assert not is_cancelled(chat_id)


# ---------------------------------------------------------------------------
# test_is_cancelled_nonexistent
# ---------------------------------------------------------------------------

def test_is_cancelled_nonexistent():
    """Non-existent chat_id returns False, not an exception."""
    assert is_cancelled(99999) is False


# ---------------------------------------------------------------------------
# test_set_before_create
# ---------------------------------------------------------------------------

def test_set_before_create():
    """set on non-existent chat_id creates and sets (race safety)."""
    chat_id = 5
    # Never called get_cancel_token — just set directly.
    set_cancel_token(chat_id)

    # Now get should find it already set.
    token = get_cancel_token(chat_id)
    assert token.is_set()
    assert is_cancelled(chat_id)


# ---------------------------------------------------------------------------
# test_multiple_chats
# ---------------------------------------------------------------------------

def test_multiple_chats():
    """Independent tokens for different chat_ids — cancelling one doesn't affect others."""
    chat_id_a = 10
    chat_id_b = 20

    token_a = get_cancel_token(chat_id_a)
    token_b = get_cancel_token(chat_id_b)

    # Different Event objects.
    assert token_a is not token_b

    # Cancelling A should not affect B.
    set_cancel_token(chat_id_a)
    assert is_cancelled(chat_id_a)
    assert not is_cancelled(chat_id_b)

    # Clearing A should not affect B.
    clear_cancel_token(chat_id_a)
    assert not is_cancelled(chat_id_a)
    assert not is_cancelled(chat_id_b)  # B was never set either

    # Set B and verify independence.
    set_cancel_token(chat_id_b)
    assert not is_cancelled(chat_id_a)
    assert is_cancelled(chat_id_b)
