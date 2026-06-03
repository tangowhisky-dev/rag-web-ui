"""In-memory cancel token registry for streaming cancellation.

Provides asyncio.Event objects keyed by chat_id so that the streaming
pipeline can signal cancellation to async generators during iteration.
"""

from __future__ import annotations

import asyncio
from typing import Dict

_cancel_tokens: Dict[int, asyncio.Event] = {}


def set_cancel_token(chat_id: int) -> None:
    """Set (signal) the asyncio.Event for *chat_id*.

    If no token exists yet, creates one and sets it — this handles the race
    where a cancel request arrives after the stream has already completed but
    before the backend has cleared the token.
    """
    if chat_id not in _cancel_tokens:
        _cancel_tokens[chat_id] = asyncio.Event()
    _cancel_tokens[chat_id].set()


def get_cancel_token(chat_id: int) -> asyncio.Event:
    """Get or create an asyncio.Event for *chat_id*.

    Callers that need to *await* cancellation (e.g. inside an async generator)
    should use this function to obtain the token and then ``await token.wait()``
    or check ``token.is_set()`` at iteration boundaries.
    """
    if chat_id not in _cancel_tokens:
        _cancel_tokens[chat_id] = asyncio.Event()
    return _cancel_tokens[chat_id]


def clear_cancel_token(chat_id: int) -> None:
    """Remove the token for *chat_id* (cleanup after stream ends).

    Safe to call even if no token exists — silently does nothing.
    """
    _cancel_tokens.pop(chat_id, None)


def is_cancelled(chat_id: int) -> bool:
    """Return ``True`` if a token exists for *chat_id* and is set.

    Returns ``False`` for non-existent chat_ids instead of raising — callers
    should treat an absent token as "not cancelled".
    """
    token = _cancel_tokens.get(chat_id)
    return token is not None and token.is_set()
