"""Unit tests for cancel_registry.py — scoped cancellation with Redis fallback."""

import asyncio

import pytest

from app.services.infrastructure import (
    clear_cancel_token,
    get_cancel_token,
    is_cancelled,
    set_cancel_token,
    set_cancel,
    clear_cancel,
    get_cancel_event,
)
from app.services.infrastructure import cancel_registry as reg


# ---------------------------------------------------------------------------
# Helpers — each test gets a fresh registry.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _fresh_registry():
    """Reset the module-level registries before every test."""
    reg._async_tokens.clear()
    reg._thread_tokens.clear()
    # Clear Redis keys too (best-effort)
    r = reg._get_redis()
    if r is not None:
        for key in r.scan_iter("cancel:*"):
            r.delete(key)
    yield
    reg._async_tokens.clear()
    reg._thread_tokens.clear()
    if r is not None:
        for key in r.scan_iter("cancel:*"):
            r.delete(key)


# ---------------------------------------------------------------------------
# Legacy API tests (backward compatibility)
# ---------------------------------------------------------------------------

def test_create_and_set():
    """get_cancel_token creates a token; set_cancel_token sets it."""
    chat_id = 1
    token = get_cancel_token(chat_id)
    assert isinstance(token, asyncio.Event)
    assert not token.is_set()

    set_cancel_token(chat_id)
    assert token.is_set()


def test_clear():
    """set then clear — is_cancelled returns False after clear."""
    chat_id = 2
    get_cancel_token(chat_id)
    set_cancel_token(chat_id)

    clear_cancel_token(chat_id)
    assert not is_cancelled(chat_id)


def test_is_cancelled():
    """After set, is_cancelled returns True; after clear, returns False."""
    chat_id = 3

    assert not is_cancelled(chat_id)  # no token yet

    set_cancel_token(chat_id)
    assert is_cancelled(chat_id)

    clear_cancel_token(chat_id)
    assert not is_cancelled(chat_id)


def test_is_cancelled_nonexistent():
    """Non-existent chat_id returns False, not an exception."""
    assert is_cancelled(99999) is False


def test_set_before_create():
    """set on non-existent chat_id creates and sets (race safety)."""
    chat_id = 5
    set_cancel_token(chat_id)

    token = get_cancel_token(chat_id)
    assert token.is_set()
    assert is_cancelled(chat_id)


def test_multiple_chats():
    """Independent tokens for different chat_ids."""
    chat_id_a = 10
    chat_id_b = 20

    token_a = get_cancel_token(chat_id_a)
    token_b = get_cancel_token(chat_id_b)

    assert token_a is not token_b

    set_cancel_token(chat_id_a)
    assert is_cancelled(chat_id_a)
    assert not is_cancelled(chat_id_b)

    clear_cancel_token(chat_id_a)
    assert not is_cancelled(chat_id_a)
    assert not is_cancelled(chat_id_b)

    set_cancel_token(chat_id_b)
    assert not is_cancelled(chat_id_a)
    assert is_cancelled(chat_id_b)


# ---------------------------------------------------------------------------
# Scoped API tests
# ---------------------------------------------------------------------------

def test_scoped_set_and_check():
    """set_cancel / is_cancelled with scope + id."""
    set_cancel("doc", 100)
    assert is_cancelled("doc", 100)
    assert not is_cancelled("doc", 101)
    assert not is_cancelled("ds", 100)

    clear_cancel("doc", 100)
    assert not is_cancelled("doc", 100)


def test_scoped_thread_event():
    """get_cancel_event returns a threading.Event that gets set."""
    import threading

    evt = get_cancel_event("ds", 200)
    assert isinstance(evt, threading.Event)
    assert not evt.is_set()

    set_cancel("ds", 200)
    assert evt.is_set()

    clear_cancel("ds", 200)
    # get_cancel_event creates a new event after clear
    evt2 = get_cancel_event("ds", 200)
    assert not evt2.is_set()


def test_scoped_idempotent():
    """Calling set_cancel multiple times is safe."""
    set_cancel("kb", 300)
    set_cancel("kb", 300)
    set_cancel("kb", 300)
    assert is_cancelled("kb", 300)


def test_scoped_clear_nonexistent():
    """clear_cancel on non-existent scope is safe."""
    clear_cancel("doc", 99999)  # should not raise


def test_legacy_and_scoped_coexist():
    """Legacy is_cancelled(chat_id) delegates to scoped is_cancelled('chat', chat_id)."""
    set_cancel("chat", 42)
    assert is_cancelled(42)  # legacy
    assert is_cancelled("chat", 42)  # scoped

    clear_cancel("chat", 42)
    assert not is_cancelled(42)
    assert not is_cancelled("chat", 42)


def test_redis_cross_scope_independence():
    """Cancelling one scope doesn't affect another with the same id."""
    set_cancel("doc", 500)
    assert is_cancelled("doc", 500)
    assert not is_cancelled("ds", 500)
    assert not is_cancelled("kb", 500)
    assert not is_cancelled("chat", 500)


@pytest.mark.asyncio
async def test_wait_for_cancel_returns_immediately_if_set():
    """wait_for_cancel returns immediately if already cancelled."""
    set_cancel("doc", 600)
    await asyncio.wait_for(
        reg.wait_for_cancel("doc", 600),
        timeout=1.0,
    )


@pytest.mark.asyncio
async def test_heartbeat_detects_redis_cancellation():
    """heartbeat_cancel_check detects cancellation set via Redis."""
    # Set cancel after a short delay
    async def _set_later():
        await asyncio.sleep(0.2)
        set_cancel("ds", 700)

    asyncio.create_task(_set_later())

    # Heartbeat should detect it and return
    await asyncio.wait_for(
        reg.heartbeat_cancel_check("ds", 700, interval=0.1),
        timeout=2.0,
    )
    assert is_cancelled("ds", 700)
