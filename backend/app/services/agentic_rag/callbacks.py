"""SSE event callback handlers for LangGraph agent graph.

Bridge between LangGraph node transitions and the existing SSE event
protocol (p: progress, t: task_list, th: thinking, 0: token, etc.).
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator

logger = logging.getLogger(__name__)


class SSEEventEmitter:
    """Manages SSE event emission for graph execution."""

    def __init__(self) -> None:
        self._event_queue: asyncio.Queue[dict] = asyncio.Queue()

    async def emit(self, event: dict) -> None:
        """Emit an event to the SSE queue."""
        await self._event_queue.put(event)

    async def emit_progress(self, phase: str, message: str, details: dict | None = None) -> None:
        """Emit a progress event (prefix: p:)."""
        event = {"event": "progress", "phase": phase, "message": message}
        if details:
            event["details"] = details
        await self.emit(event)

    async def emit_task_list(self, tasks: list[dict]) -> None:
        """Emit a task_list event (prefix: t:)."""
        await self.emit({"event": "task_list", "tasks": tasks})

    async def emit_rewritten_query(self, query: str) -> None:
        """Emit a rewritten_query event (prefix: 1:)."""
        await self.emit({"event": "rewritten_query", "query": query})

    async def emit_context(self, docs: list, confidence: str, **kwargs) -> None:
        """Emit a context event (prefix: 2:)."""
        await self.emit({"event": "context", "docs": docs, "confidence": confidence, **kwargs})

    async def emit_token(self, content: str) -> None:
        """Emit a token event (prefix: 0:)."""
        await self.emit({"event": "token", "content": content})

    async def emit_thinking(self, content: str, done: bool = False) -> None:
        """Emit a thinking event (prefix: th:)."""
        await self.emit({"event": "thinking", "content": content, "done": done})

    async def emit_error(self, message: str) -> None:
        """Emit an error event (prefix: 3:)."""
        await self.emit({"event": "error", "message": message})

    async def emit_done(self, full_response: str, usage: dict | None = None) -> None:
        """Emit a done event (prefix: d:)."""
        event = {"event": "done", "full_response": full_response}
        if usage:
            event["usage"] = usage
        await self.emit(event)

    async def emit_answer_rewrite(self, content: str) -> None:
        """Emit an answer_rewrite event (internal normalization)."""
        await self.emit({"event": "answer_rewrite", "content": content})

    async def emit_evaluation(
        self,
        faithfulness: int,
        completeness: int,
        citation_quality: int,
        confidence_match: bool,
        flags: list[str],
    ) -> None:
        """Emit an evaluation event with answer quality metrics."""
        await self.emit({
            "event": "evaluation",
            "faithfulness": faithfulness,
            "completeness": completeness,
            "citation_quality": citation_quality,
            "confidence_match": confidence_match,
            "flags": flags,
        })

    async def drain(self) -> AsyncGenerator[dict, None]:
        """Yield all queued events until the queue is empty."""
        while not self._event_queue.empty():
            try:
                event = self._event_queue.get_nowait()
                yield event
            except asyncio.QueueEmpty:
                break
