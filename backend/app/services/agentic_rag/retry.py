"""Shared retry helpers for the agentic RAG pipeline.

All external calls (retrieval, tools, LLM generation) should use `with_retry`
to get a uniform 3-attempt policy with exponential backoff. Retry attempts are
logged and surfaced through progress events when an emitter is provided.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Optional, TypeVar

from tenacity import (
    AsyncRetrying,
    Retrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

RETRY_EXCEPTIONS = (Exception,)


def _log_retry_attempt(retry_state: RetryCallState) -> None:
    """Log each retry attempt."""
    fn_name = retry_state.fn.__name__ if retry_state.fn else "unknown"
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "[RETRY] %s attempt %d failed: %s",
        fn_name,
        retry_state.attempt_number,
        exception,
        exc_info=exception is not None,
    )


def _make_retry_kwargs(max_attempts: int, min_wait: float, max_wait: float) -> dict:
    return {
        "stop": stop_after_attempt(max_attempts),
        "wait": wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        "retry": retry_if_exception_type(RETRY_EXCEPTIONS),
        "before_sleep": _log_retry_attempt,
        "reraise": True,
    }


def with_retry(
    fn: Optional[F] = None,
    *,
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
) -> F:
    """Decorator that retries an async function up to `max_attempts` times.

    Usage:
        @with_retry
        async def my_func(...): ...

        @with_retry(max_attempts=5)
        async def my_func(...): ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            async for attempt in AsyncRetrying(**_make_retry_kwargs(max_attempts, min_wait, max_wait)):
                with attempt:
                    return await func(*args, **kwargs)
            raise RuntimeError("Retry loop exited unexpectedly")

        return async_wrapper  # type: ignore[return-value]

    if fn is not None:
        return decorator(fn)
    return decorator  # type: ignore[return-value]


def with_retry_sync(
    fn: Optional[F] = None,
    *,
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
) -> F:
    """Decorator that retries a synchronous function up to `max_attempts` times.

    Usage:
        @with_retry_sync
        def my_func(...): ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in Retrying(**_make_retry_kwargs(max_attempts, min_wait, max_wait)):
                with attempt:
                    return func(*args, **kwargs)
            raise RuntimeError("Retry loop exited unexpectedly")

        return sync_wrapper  # type: ignore[return-value]

    if fn is not None:
        return decorator(fn)
    return decorator  # type: ignore[return-value]
