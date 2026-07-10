"""Retry wrapper for agentic pipeline operations.

Wraps any subtask function with attempt tracking, progressive backoff,
and configurable threshold widening on retries.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Coroutine

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    backoff_base_seconds: float = 0.5
    threshold_widening: bool = True  # Lower thresholds on each retry


@dataclass
class RetryResult:
    """Result of a retried operation."""
    success: bool
    result: Any
    attempt: int  # which attempt succeeded (1-based)
    error: str | None = None


class RetryExhaustedError(Exception):
    """Raised when all retry attempts fail."""

    def __init__(self, last_error: str, attempt: int):
        super().__init__(f"Operation failed after {attempt} attempts: {last_error}")
        self.last_error = last_error
        self.attempt = attempt


async def with_retry(
    fn: Callable[..., Coroutine[Any, Any, Any]],
    *args: Any,
    yield_callback: Callable,
    config: RetryConfig | None = None,
    **kwargs: Any,
) -> RetryResult:
    """Execute a function with retry logic.

    Args:
        fn: Async function to execute.
        args/kwargs: Arguments to pass to fn.
        yield_callback: Async callback to emit progress events on each attempt.
            Signature: async def callback(phase: str, message: str, details: dict)
        config: Retry configuration. Defaults to 3 attempts with backoff.

    Returns:
        RetryResult with success status, result, and attempt count.
    """
    config = config or RetryConfig(max_attempts=3)

    last_error = None
    for attempt in range(1, config.max_attempts + 1):
        try:
            await yield_callback(
                "retry_attempt",
                f"Attempt {attempt}/{config.max_attempts}" if attempt > 1 else "Executing...",
                {"attempt": attempt, "max_attempts": config.max_attempts},
            )

            result = await fn(*args, **kwargs)
            return RetryResult(success=True, result=result, attempt=attempt)

        except Exception as exc:
            last_error = str(exc)
            logger.warning("[RETRY] attempt %d/%d failed: %s", attempt, config.max_attempts, last_error)

            if attempt < config.max_attempts:
                # Emit retrying progress event
                await yield_callback(
                    "retrying",
                    f"Retrying ({attempt}/{config.max_attempts - 1} remaining)...",
                    {"attempt": attempt, "error": last_error, "backoff": config.backoff_base_seconds * attempt},
                )
                # Brief backoff before retry
                await asyncio.sleep(config.backoff_base_seconds * attempt)

    # All attempts failed
    raise RetryExhaustedError(last_error, config.max_attempts)


class Retriever:
    """Encapsulates a full retrieval-and-evaluation cycle with retry support.

    This wraps a complete retrieval loop: search → rank → assess → expand → re-rank.
    The caller provides a function that performs one full cycle.
    """

    def __init__(self, config: RetryConfig | None = None):
        self.config = config or RetryConfig(max_attempts=3, backoff_base_seconds=0.3)

    async def run(
        self,
        cycle_fn: Callable[..., Coroutine[Any, Any, Any]],
        *args: Any,
        yield_callback: Callable,
        **kwargs: Any,
    ) -> RetryResult:
        """Run retrieval cycles with retry on failure.

        Args:
            cycle_fn: A function that performs one full retrieval cycle
                and raises an exception if it fails (e.g., no docs returned,
                or confidence score too low).
            yield_callback: Progress event emitter.
            *args, **kwargs: Arguments passed to cycle_fn.

        Returns:
            RetryResult with docs, confidence, and metadata.
        """
        last_error = None
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                await yield_callback(
                    "retrieve_attempt",
                    f"Attempt {attempt}/{self.config.max_attempts}" if attempt > 1 else "Retrieving...",
                    {"attempt": attempt, "max_attempts": self.config.max_attempts},
                )

                result = await cycle_fn(*args, **kwargs)

                if attempt < self.config.max_attempts:
                    await yield_callback(
                        "attempt_complete",
                        f"Retrieval succeeded on attempt {attempt}",
                        {"attempt": attempt, "doc_count": len(result.get("docs", []))},
                    )

                result["attempt"] = attempt
                return RetryResult(success=True, result=result, attempt=attempt)

            except Exception as exc:
                last_error = str(exc)
                logger.warning("[RETRIEVE_RETRY] attempt %d/%d failed: %s", attempt, self.config.max_attempts, last_error)

                if attempt < self.config.max_attempts:
                    await yield_callback(
                        "retrying",
                        f"Retrying retrieval ({self.config.max_attempts - attempt} remaining)...",
                        {"attempt": attempt, "error": last_error},
                    )
                    await asyncio.sleep(self.config.backoff_base_seconds * attempt)

        raise RetryExhaustedError(last_error, self.config.max_attempts)


class Generator:
    """Encapsulates answer generation with retry support."""

    def __init__(self, config: RetryConfig | None = None):
        self.config = config or RetryConfig(max_attempts=2, backoff_base_seconds=0.3)

    async def run(
        self,
        generate_fn: Callable[..., Coroutine[Any, Any, Any]],
        *args: Any,
        yield_callback: Callable,
        **kwargs: Any,
    ) -> RetryResult:
        """Run generation with retry on failure.

        Args:
            generate_fn: Function that generates an answer.
            yield_callback: Progress event emitter.
            *args, **kwargs: Arguments passed to generate_fn.

        Returns:
            RetryResult with answer text and metadata.
        """
        last_error = None
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                await yield_callback(
                    "generate_attempt",
                    f"Attempt {attempt}/{self.config.max_attempts}" if attempt > 1 else "Generating...",
                    {"attempt": attempt, "max_attempts": self.config.max_attempts},
                )

                # Generation is streamed, so we collect the result at the end
                result_parts = []
                async for chunk in generate_fn(*args, **kwargs):
                    if chunk.get("event") == "done":
                        result_parts.append(chunk)
                        break
                    result_parts.append(chunk)

                # Extract answer from done event
                answer = ""
                for chunk in result_parts:
                    if chunk.get("event") == "done":
                        answer = chunk.get("full_response", "")
                        break

                return RetryResult(success=True, result={"answer": answer}, attempt=attempt)

            except Exception as exc:
                last_error = str(exc)
                logger.warning("[GENERATE_RETRY] attempt %d/%d failed: %s", attempt, self.config.max_attempts, last_error)

                if attempt < self.config.max_attempts:
                    await yield_callback(
                        "retrying",
                        f"Retrying generation ({self.config.max_attempts - attempt} remaining)...",
                        {"attempt": attempt, "error": last_error},
                    )
                    await asyncio.sleep(self.config.backoff_base_seconds * attempt)

        raise RetryExhaustedError(last_error, self.config.max_attempts)
