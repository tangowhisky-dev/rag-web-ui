"""ProgressTimeout — silence-based async context manager for document processing.

Fires an on_timeout callback only when no progress ping has been received
within the configured silence window, preventing false-positive stuck detection
on large OCR or multi-chunk jobs.
"""

import asyncio
import time
from typing import Callable


class ProgressTimeout:
    """Async context manager that calls *on_timeout* if no ping arrives within
    *silence_seconds* seconds.

    Usage::

        async def _mark_failed():
            ...

        async with ProgressTimeout(silence_seconds=300, on_timeout=_mark_failed) as pt:
            for chunk in document.chunks():
                process(chunk)
                pt.ping()
    """

    def __init__(self, silence_seconds: int, on_timeout: Callable[[], None]) -> None:
        self._silence_seconds = silence_seconds
        self._on_timeout = on_timeout
        self._last_ping: float = time.monotonic()
        self._watcher: asyncio.Task | None = None
        # Poll frequently enough to catch the boundary without busy-waiting.
        self._poll_interval: int = max(1, min(silence_seconds // 6, 10))

    def ping(self) -> None:
        """Reset the silence clock.  Call this after each unit of progress."""
        self._last_ping = time.monotonic()

    async def _watch(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._poll_interval)
                elapsed = time.monotonic() - self._last_ping
                if elapsed > self._silence_seconds:
                    self._on_timeout()
                    return
        except asyncio.CancelledError:
            return

    async def __aenter__(self) -> "ProgressTimeout":
        self._last_ping = time.monotonic()
        self._watcher = asyncio.ensure_future(self._watch())
        return self

    async def __aexit__(self, *_) -> None:
        if self._watcher is not None:
            self._watcher.cancel()
            try:
                await self._watcher
            except asyncio.CancelledError:
                pass
            self._watcher = None
