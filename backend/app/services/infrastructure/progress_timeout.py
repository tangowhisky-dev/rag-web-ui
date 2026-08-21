"""ProgressTimeout — silence-based async context manager for document processing.

Fires an on_timeout callback only when no progress ping has been received
within the configured silence window, preventing false-positive stuck detection
on large OCR or multi-chunk jobs.

When the timeout fires, the callback is invoked and the host coroutine is
cancelled.  ``__aexit__`` converts the resulting ``CancelledError`` into a
``ProgressTimeoutError`` (a regular ``Exception`` subclass) so the caller's
``except Exception`` block can handle cleanup (rollback, delete document,
mark task failed).
"""

import asyncio
import time
from typing import Callable, Optional


class ProgressTimeoutError(Exception):
    """Raised when no progress ping arrives within the silence window."""


class ProgressTimeout:
    """Async context manager that calls *on_timeout* and cancels the host
    coroutine if no ping arrives within *silence_seconds* seconds.

    Usage::

        async def _mark_failed():
            ...

        try:
            async with ProgressTimeout(600, _mark_failed) as pt:
                for chunk in document.chunks():
                    process(chunk)
                    pt.ping()
        except ProgressTimeoutError:
            # host coroutine was cancelled by the timeout
            ...

    If the body completes normally before the timeout fires, the watcher
    task is cancelled in ``__aexit__`` and no exception is raised.
    """

    def __init__(self, silence_seconds: int, on_timeout: Callable[[], None]) -> None:
        self._silence_seconds = silence_seconds
        self._on_timeout = on_timeout
        self._last_ping: float = time.monotonic()
        self._watcher: Optional[asyncio.Task] = None
        self._host_task: Optional[asyncio.Task] = None
        self._timed_out: bool = False
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
                    self._timed_out = True
                    self._on_timeout()
                    # Cancel the host coroutine so it stops doing work.
                    if self._host_task is not None:
                        self._host_task.cancel()
                    return
        except asyncio.CancelledError:
            return

    async def __aenter__(self) -> "ProgressTimeout":
        self._last_ping = time.monotonic()
        self._host_task = asyncio.current_task()
        self._watcher = asyncio.ensure_future(self._watch())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        # Always cancel and await the watcher.
        if self._watcher is not None:
            self._watcher.cancel()
            try:
                await self._watcher
            except asyncio.CancelledError:
                pass
            self._watcher = None

        # If the timeout fired and caused a CancelledError, convert it to
        # ProgressTimeoutError so the caller's `except Exception` catches it.
        # Returning True suppresses the original CancelledError.
        if self._timed_out and exc_type is asyncio.CancelledError:
            raise ProgressTimeoutError(
                f"Processing timed out — no progress for {self._silence_seconds}s"
            )

        # Normal exit — don't suppress.
        return False
