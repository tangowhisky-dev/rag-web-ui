"""SSE event callback handlers for LangGraph agent graph.

Bridge between LangGraph node transitions and the existing SSE event
protocol (p: progress, t: task_list, th: thinking, 0: token, etc.).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler

from .utils import build_task_list_events

logger = logging.getLogger(__name__)


class SSEEventEmitter:
    """Manages SSE event emission for graph execution.
    
    This is the bridge between LangGraph's node execution and the
    existing SSE event protocol used by the frontend.
    """

    def __init__(self) -> None:
        self._event_queue: asyncio.Queue[dict] = asyncio.Queue()
        self._running = False

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

    async def drain(self) -> AsyncGenerator[dict, None]:
        """Yield all queued events until the queue is empty and no more are coming.
        
        Used after graph execution to flush remaining events.
        """
        while not self._event_queue.empty():
            try:
                event = self._event_queue.get_nowait()
                yield event
            except asyncio.QueueEmpty:
                break


# ---------------------------------------------------------------------------
# LangChain callback handler
# ---------------------------------------------------------------------------

class LangGraphCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that emits SSE events from graph transitions."""

    def __init__(self, emitter: SSEEventEmitter) -> None:
        self.emitter = emitter
        self.supported_methods = ["on_chain_start", "on_chain_end", "on_llm_start", "on_llm_end"]
        self.supported_types = ["chain", "llm"]
        self._node_entered = False

    async def on_chain_start(self, serialized: dict, inputs: dict | None, **kwargs: Any) -> None:
        node_name = serialized.get("name", "")
        await self.emitter.emit_progress(
            node_name,
            f"Starting {node_name}...",
            {"node": node_name},
        )

    async def on_chain_end(self, outputs: dict | None = None, **kwargs: Any) -> None:
        node_name = kwargs.get("name") or (kwargs.get("serialized", {}).get("name", "")) if isinstance(kwargs.get("serialized"), dict) else ""
        if node_name:
            await self.emitter.emit_progress(
                node_name,
                f"Finished {node_name}",
            )

    async def on_llm_start(self, serialized: dict, inputs: dict | None, **kwargs: Any) -> None:
        await self.emitter.emit_progress(
            "llm_call",
            f"Calling {serialized.get('name', 'LLM')}...",
        )

    async def on_llm_end(self, response: dict | None = None, **kwargs: Any) -> None:
        await self.emitter.emit_progress(
            "llm_call",
            "LLM call complete.",
        )

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        await self.emitter.emit_token(token)
