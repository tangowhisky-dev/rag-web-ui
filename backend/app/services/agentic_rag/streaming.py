"""LangGraph v3 streaming transformer for the agentic RAG runner.

Transforms raw protocol events from graph.astream_events(..., version="v3")
into the internal SSE event dictionaries consumed by chat_service.py.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langgraph.stream import ProtocolEvent, StreamChannel, StreamTransformer

logger = logging.getLogger(__name__)

class AgenticRAGTransformer(StreamTransformer):
    """Custom v3 stream transformer for agentic RAG.

    Requires updates (state deltas), messages (tokens + usage), and custom
    (explicit node-emitted events including agent_step, task_list, progress,
    context).

    Produces three side-channel projections:
      - events: StreamChannel[dict]  -> internal SSE event payloads
      - usage:  StreamChannel[dict]  -> final token usage metadata
      - state:  StreamChannel[dict]  -> final graph state snapshot
    """

    required_stream_modes = ("updates", "messages", "custom")

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self.events = StreamChannel[dict]()
        self.usage = StreamChannel[dict]()
        self.state = StreamChannel[dict]()

        self._input_tokens = 0
        self._output_tokens = 0
        self._all_docs: list[dict] = []
        self._final_state: Optional[dict] = None

        self._custom_handlers = {
            "agent_step": self._passthrough,
            "task_list": self._handle_task_list,
            "progress": self._passthrough,
            "thinking": self._passthrough,
            "context": self._handle_context,
            "token": self._handle_token,
            "plan": self._handle_plan,
            "tool_call": self._passthrough,
            "tool_observation": self._passthrough,
            "last_answer": self._handle_last_answer,
            "interrupt": self._handle_interrupt,
        }

    def init(self) -> dict[str, Any]:
        return {
            "agentic": self,
            "events": self.events,
            "usage": self.usage,
            "state": self.state,
        }

    def process(self, event: ProtocolEvent) -> bool:
        method = event.get("method", "")
        params = event.get("params", {})
        namespace = params.get("namespace", [])
        data = params.get("data", {})

        if method == "updates":
            self._process_updates(namespace, data)
        elif method == "messages":
            self._process_messages(namespace, data)
        elif method == "custom":
            self._process_custom(data)

        return True

    def finalize(self) -> None:
        self.usage.push(
            {
                "promptTokens": self._input_tokens,
                "completionTokens": self._output_tokens,
            }
        )
        self.usage.close()
        if self._final_state is not None:
            self.state.push(self._final_state)
        self.state.close()
        self.events.close()

    def fail(self, err: BaseException) -> None:  # type: ignore[override]
        logger.error("[STREAM] transformer failed: %s", err)
        self.usage.push(
            {
                "promptTokens": self._input_tokens,
                "completionTokens": self._output_tokens,
            }
        )
        self.usage.fail(err)
        self.state.fail(err)
        self.events.fail(err)

    # ------------------------------------------------------------------
    # Channel processors
    # ------------------------------------------------------------------

    def _process_updates(self, namespace: list[str], data: Any) -> None:
        if not isinstance(data, dict):
            return

        # Root-level updates carry main-graph state changes.
        if not namespace:
            self._process_root_updates(data)

    def _process_root_updates(self, data: dict[str, Any]) -> None:
        for node_name, node_update in data.items():
            if not isinstance(node_update, dict):
                continue

            rd = node_update.get("retrieved_docs")
            if isinstance(rd, list) and rd:
                self._emit_context(rd)

    def _process_messages(self, namespace: list[str], data: Any) -> None:
        """Process chat-model message events.

        Answer tokens are now emitted explicitly by the `finalize_node` via
        the custom stream writer, so this method only collects usage metadata
        from the finalizing node. Other nodes' LLM tokens are ignored.

        In the v3 protocol, `data` is a `(payload, metadata)` tuple where
        `metadata` carries the originating node name in `langgraph_node`.
        The namespace list is empty for these events, so we must read the
        node from metadata.
        """
        if not isinstance(data, (list, tuple)) or len(data) < 2:
            return

        payload, metadata = data[0], data[1]
        if not isinstance(metadata, dict):
            return

        node_name = metadata.get("langgraph_node")

        # Only collect usage from the finalizing node (tokens come via custom writer).
        if node_name not in ("finalize", "generating", "generate_answer"):
            return

        # Case 1: LangChain message chunk emitted by the model stream.
        if isinstance(payload, (BaseMessage, AIMessageChunk, AIMessage)):
            self._collect_usage_from_message(payload)
            return

        # Case 2: Protocol event dict (content-block-delta / message-finish).
        if isinstance(payload, dict):
            self._collect_usage_from_event(payload)

    def _collect_usage_from_message(self, payload: Any) -> None:
        usage = getattr(payload, "usage_metadata", None)
        if isinstance(usage, dict):
            self._input_tokens += usage.get("input_tokens", 0) or 0
            self._output_tokens += usage.get("output_tokens", 0) or 0

    def _collect_usage_from_event(self, payload: dict) -> None:
        if payload.get("event", "") != "message-finish":
            return
        usage = payload.get("usage") or {}
        self._input_tokens += usage.get("input_tokens", 0) or 0
        self._output_tokens += usage.get("output_tokens", 0) or 0
        # Some providers include usage under a nested 'usage' key inside metadata.
        if not usage:
            nested_metadata = payload.get("metadata") or {}
            usage = nested_metadata.get("usage") or {}
            self._input_tokens += usage.get("input_tokens", 0) or 0
            self._output_tokens += usage.get("output_tokens", 0) or 0

    def _process_custom(self, data: Any) -> None:
        if not isinstance(data, dict):
            return

        kind = data.get("event")
        handler = self._custom_handlers.get(kind)
        if handler is not None:
            handler(data, kind)

    def _passthrough(self, data: dict, kind: str) -> None:
        self.events.push({"event": kind, **{k: v for k, v in data.items() if k != "event"}})

    def _handle_task_list(self, data: dict, kind: str) -> None:
        self.events.push({"event": "task_list", "tasks": data.get("tasks", [])})

    def _handle_context(self, data: dict, kind: str) -> None:
        docs = data.get("docs", [])
        if isinstance(docs, list):
            self._emit_context(docs)

    def _handle_token(self, data: dict, kind: str) -> None:
        # Explicit per-token events emitted by generating_node via the
        # LangGraph stream writer. Forward them unchanged.
        content = data.get("content", "")
        if content:
            self.events.push({"event": "token", "content": content})

    def _handle_plan(self, data: dict, kind: str) -> None:
        self.events.push({"event": "plan", "plan": data.get("plan", {})})

    def _handle_last_answer(self, data: dict, kind: str) -> None:
        self.events.push({"event": "last_answer", "last_answer_object": data.get("last_answer_object", {})})

    def _handle_interrupt(self, data: dict, kind: str) -> None:
        self.events.push({"event": "interrupt", "question": data.get("question", "")})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _emit_context(self, docs: list[dict]) -> None:
        # Deduplicate by content_hash so repeated updates don't bloat the UI.
        seen = {d.get("metadata", {}).get("content_hash") for d in self._all_docs}
        seen.discard(None)
        for doc in docs:
            h = doc.get("metadata", {}).get("content_hash")
            if h is None or h not in seen:
                self._all_docs.append(doc)
                if h is not None:
                    seen.add(h)

        # Compute confidence from best reranker score in retrieved docs.
        best_score = 0.0
        for d in self._all_docs:
            score = d.get("metadata", {}).get("_reranker_score", 0.0) or 0.0
            if score > best_score:
                best_score = score
        if best_score > 0.8:
            conf_level, conf_score = "very_high", int(best_score * 100)
        elif best_score > 0.6:
            conf_level, conf_score = "high", int(best_score * 100)
        elif best_score > 0.3:
            conf_level, conf_score = "medium", int(best_score * 100)
        elif best_score > 0:
            conf_level, conf_score = "low", int(best_score * 100)
        else:
            conf_level, conf_score = "none", 0

        self.events.push(
            {
                "event": "context",
                "docs": list(self._all_docs),
                "confidence": conf_level,
                "score": conf_score,
            }
        )

    def set_final_state(self, state: dict) -> None:
        """Called by graph_runner once stream.output() resolves."""
        self._final_state = state
