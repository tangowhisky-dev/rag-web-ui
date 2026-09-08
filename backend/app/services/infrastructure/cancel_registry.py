"""Cancel token registry for streaming and ingestion cancellation.

Provides two layers of cancellation signalling:

1. **In-memory** (fast path): ``asyncio.Event`` for async contexts and
   ``threading.Event`` for thread contexts, keyed by scope + id.
2. **Redis** (durable fallback + cross-process): a Redis key with TTL
   so cancellation survives backend restarts and is visible to all
   threads/processes.

Scopes:
    - ``"chat"``  — chat streaming cancellation (stop button)
    - ``"doc"``   — per-document ingestion/graph cancellation
    - ``"ds"``    — per-datastore scan/ingestion/graph cancellation
    - ``"kb"``    — per-knowledge-base cancellation (KB deletion)

Usage:
    # Signal cancellation
    set_cancel("chat", chat_id)
    set_cancel("doc", document_id)

    # Check cancellation (checks in-memory first, then Redis)
    if is_cancelled("chat", chat_id):
        break

    # Await cancellation in async context
    await wait_for_cancel("chat", chat_id)

    # Clear after handling
    clear_cancel("chat", chat_id)

The legacy chat-only functions (``set_cancel_token``, ``get_cancel_token``,
``clear_cancel_token``, ``is_cancelled``) remain for backward compatibility
and delegate to the new scoped API.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Dict, Optional

import redis

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── In-memory tokens ─────────────────────────────────────────────────────────
# Async tokens for async contexts (chat streaming, agent loop)
_async_tokens: Dict[str, asyncio.Event] = {}
# Thread tokens for thread contexts (ingestion, graph build)
_thread_tokens: Dict[str, threading.Event] = {}

# ── Redis client (lazy init, with fallback) ──────────────────────────────────
_REDIS: Optional[redis.Redis] = None

_CANCEL_TTL = 3600  # 1 hour — tokens auto-expire if not cleared


def _get_redis() -> Optional[redis.Redis]:
    """Return a shared synchronous Redis client, or None if unavailable."""
    global _REDIS
    if _REDIS is not None:
        try:
            _REDIS.ping()
            return _REDIS
        except Exception:
            _REDIS = None
    try:
        client = redis.from_url(
            settings.REDIS_URL,
            socket_timeout=1,
            socket_connect_timeout=1,
        )
        client.ping()
        _REDIS = client
        return _REDIS
    except Exception:
        return None


def _redis_key(scope: str, id: int) -> str:
    return f"cancel:{scope}:{id}"


# ── Scoped API ───────────────────────────────────────────────────────────────


def set_cancel(scope: str, id: int, reason: str = "user_requested") -> None:
    """Signal cancellation for *scope* + *id*.

    Sets both the in-memory events (async + thread) and the Redis key.
    Idempotent — safe to call multiple times.
    """
    key = f"{scope}:{id}"

    # In-memory async event
    if key not in _async_tokens:
        _async_tokens[key] = asyncio.Event()
    _async_tokens[key].set()

    # In-memory thread event
    if key not in _thread_tokens:
        _thread_tokens[key] = threading.Event()
    _thread_tokens[key].set()

    # Redis (durable + cross-process)
    r = _get_redis()
    if r is not None:
        try:
            r.set(_redis_key(scope, id), reason, ex=_CANCEL_TTL)
        except Exception as e:
            logger.debug("[CANCEL] Redis set failed for %s:%s — %s", scope, id, e)


def get_cancel_event(scope: str, id: int) -> threading.Event:
    """Get or create a threading.Event for *scope* + *id*.

    For use in thread contexts (ingestion, graph build).  The event is
    also set when ``set_cancel`` is called from any thread or process.
    """
    key = f"{scope}:{id}"
    if key not in _thread_tokens:
        _thread_tokens[key] = threading.Event()
    return _thread_tokens[key]


async def wait_for_cancel(scope: str, id: int) -> None:
    """Await cancellation in an async context.

    Polls Redis every 1 second (the heartbeat safety net) so cancellation
    is detected even if the in-memory event was not set (e.g. backend
    restart, cross-process cancellation).
    """
    key = f"{scope}:{id}"
    if key not in _async_tokens:
        _async_tokens[key] = asyncio.Event()

    async_evt = _async_tokens[key]
    if async_evt.is_set():
        return

    # Race: in-memory event vs Redis heartbeat
    while not async_evt.is_set():
        # Check Redis
        if is_cancelled(scope, id):
            async_evt.set()
            return
        # Wait up to 1s on the in-memory event, then re-check Redis
        try:
            await asyncio.wait_for(async_evt.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            continue


def clear_cancel(scope: str, id: int) -> None:
    """Clear cancellation for *scope* + *id* (cleanup after work completes)."""
    key = f"{scope}:{id}"
    _async_tokens.pop(key, None)
    _thread_tokens.pop(key, None)

    r = _get_redis()
    if r is not None:
        try:
            r.delete(_redis_key(scope, id))
        except Exception:
            pass


# ── Heartbeat helper ─────────────────────────────────────────────────────────


async def heartbeat_cancel_check(
    scope: str,
    id: int,
    interval: float = 1.0,
) -> None:
    """Background task that polls Redis for cancellation.

    Run alongside long-running operations as a safety net.  If Redis
    shows cancellation (e.g. set by another process or before a restart),
    this propagates it to the in-memory events so fast-path checks trigger.

    Exits when cancellation is detected.
    """
    while True:
        if is_cancelled(scope, id):
            # is_cancelled already propagates to in-memory events
            return
        await asyncio.sleep(interval)


# ── Legacy chat-only API (backward compatibility) ────────────────────────────


def set_cancel_token(chat_id: int) -> None:
    """Set (signal) the cancel token for *chat_id*. Legacy — use set_cancel."""
    set_cancel("chat", chat_id)


def get_cancel_token(chat_id: int) -> asyncio.Event:
    """Get or create an asyncio.Event for *chat_id*. Legacy — use wait_for_cancel."""
    key = f"chat:{chat_id}"
    if key not in _async_tokens:
        _async_tokens[key] = asyncio.Event()
    return _async_tokens[key]


def clear_cancel_token(chat_id: int) -> None:
    """Remove the token for *chat_id*. Legacy — use clear_cancel."""
    clear_cancel("chat", chat_id)


def is_cancelled(*args) -> bool:
    """Return True if cancellation is signalled.

    Two calling conventions:
    - ``is_cancelled(chat_id: int)`` — legacy, checks the ``"chat"`` scope.
    - ``is_cancelled(scope: str, id: int)`` — scoped, checks any scope.
    """
    if len(args) == 1:
        # Legacy: is_cancelled(chat_id: int)
        return _is_cancelled_scoped("chat", args[0])
    elif len(args) == 2:
        # Scoped: is_cancelled(scope: str, id: int)
        return _is_cancelled_scoped(args[0], args[1])
    return False


def _is_cancelled_scoped(scope: str, id: int) -> bool:
    """Check if cancellation has been signalled for *scope* + *id*.

    Checks in-memory first (fast path), then Redis (durable fallback).
    """
    key = f"{scope}:{id}"

    # Fast path: in-memory
    async_evt = _async_tokens.get(key)
    if async_evt is not None and async_evt.is_set():
        return True
    thread_evt = _thread_tokens.get(key)
    if thread_evt is not None and thread_evt.is_set():
        return True

    # Durable fallback: Redis
    r = _get_redis()
    if r is not None:
        try:
            if r.exists(_redis_key(scope, id)):
                # Propagate to in-memory so subsequent checks are fast
                set_cancel(scope, id, reason="detected_via_redis")
                return True
        except Exception:
            pass

    return False
