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

# Nodes that should surface in the agent timeline UI.
# Subgraph-scoped nodes are prefixed by their parent namespace, so we match
# the last segment of the namespace path.
TRACKED_NODES = frozenset(
    (
        "load_historical_memory",
        "summarize_history",
        "rewrite_query",
        "classify_query",
        "request_clarification",
        "agent_subgraph",
        "rewrite_subtask_query",
        "dense_retrieval",
        "sparse_retrieval",
        "exact_retrieval",
        "merge",
        "neo4j_expansion",
        "reranking",
        "filter",
        "sufficiency_check",
        "adaptive_reranking",
        "collect_context",
        "prepare_final_context",
        "generating",
        "chart_validation",
        "answer_evaluation",
        "finalize_answer",
    )
)


def _namespace_node_name(namespace: list[str]) -> Optional[str]:
    """Return the tracked node name from a namespace path, if any.

    Namespaces look like [] for root graph events, or
    ["agent_subgraph:<task_id>", "merge:<task_id>"] for subgraph events.
    """
    for segment in reversed(namespace):
        name = segment.split(":", 1)[0]
        if name in TRACKED_NODES:
            return name
    return None


class AgenticRAGTransformer(StreamTransformer):
    """Custom v3 stream transformer for agentic RAG.

    Requires updates (state deltas), messages (tokens + usage), and custom
    (explicit node-emitted events including agent_step, task_list, progress,
    context, rewritten_query).

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
        self._task_texts: list[str] = []
        self._completed_subtasks = 0
        self._active_nodes: set[str] = set()

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
            return

        # Subgraph updates (retrieval happens inside agent_subgraph).
        # We stream context as soon as retrieved_docs are produced there.
        node_name = _namespace_node_name(namespace)
        if node_name in (
            "collect_context",
            "reranking",
            "adaptive_reranking",
            "prepare_final_context",
        ):
            for node_update in data.values():
                if not isinstance(node_update, dict):
                    continue
                rd = node_update.get("retrieved_docs")
                if isinstance(rd, list) and rd:
                    self._emit_context(rd)

    def _process_root_updates(self, data: dict[str, Any]) -> None:
        for node_name, node_update in data.items():
            if not isinstance(node_update, dict):
                continue

            if node_name == "classify_query":
                subtasks = node_update.get("subtasks")
                if isinstance(subtasks, list) and len(subtasks) > 1:
                    self._task_texts = subtasks
                    self._completed_subtasks = 0
                    self.events.push(
                        {
                            "event": "task_list",
                            "tasks": self._make_task_list(self._task_texts, 0),
                        }
                    )
                continue

            rd = node_update.get("retrieved_docs")
            if isinstance(rd, list) and rd:
                self._emit_context(rd)

    def _process_messages(self, namespace: list[str], data: Any) -> None:
        """Process chat-model message events.

        Answer tokens are now emitted explicitly by the `generating_node` via
        the custom stream writer, so this method only collects usage metadata
        from the generating node. Classifier LLM tokens, tool tokens, etc. are
        ignored.

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

        # Only collect usage from generating nodes (tokens come via custom writer).
        if node_name not in ("generating", "generate_answer"):
            return

        # Case 1: LangChain message chunk emitted by the model stream.
        if isinstance(payload, (BaseMessage, AIMessageChunk, AIMessage)):
            usage = getattr(payload, "usage_metadata", None)
            if isinstance(usage, dict):
                self._input_tokens += usage.get("input_tokens", 0) or 0
                self._output_tokens += usage.get("output_tokens", 0) or 0
            return

        # Case 2: Protocol event dict (content-block-delta / message-finish).
        if not isinstance(payload, dict):
            return

        event_type = payload.get("event", "")
        if event_type == "message-finish":
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
        if kind == "agent_step":
            self.events.push({"event": "agent_step", **{k: v for k, v in data.items() if k != "event"}})
            if data.get("node") == "agent_subgraph" and data.get("status") == "done" and self._task_texts:
                self._completed_subtasks = min(
                    self._completed_subtasks + 1, len(self._task_texts)
                )
                self.events.push(
                    {
                        "event": "task_list",
                        "tasks": self._make_task_list(
                            self._task_texts, self._completed_subtasks
                        ),
                    }
                )
        elif kind == "task_list":
            tasks = data.get("tasks", [])
            if isinstance(tasks, list) and tasks:
                self._task_texts = [t.get("text", t.get("id", "")) for t in tasks]
                self._completed_subtasks = sum(
                    1 for t in tasks if t.get("status") == "done"
                )
            self.events.push({"event": "task_list", "tasks": tasks})
        elif kind == "rewritten_query":
            self.events.push({"event": "rewritten_query", **{k: v for k, v in data.items() if k != "event"}})
        elif kind == "progress":
            self.events.push({"event": "progress", **{k: v for k, v in data.items() if k != "event"}})
        elif kind == "thinking":
            self.events.push({"event": "thinking", **{k: v for k, v in data.items() if k != "event"}})
        elif kind == "context":
            docs = data.get("docs", [])
            if isinstance(docs, list):
                self._emit_context(docs)
        elif kind == "token":
            # Explicit per-token events emitted by generating_node via the
            # LangGraph stream writer. Forward them unchanged.
            content = data.get("content", "")
            if content:
                self.events.push({"event": "token", "content": content})

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

        self.events.push(
            {
                "event": "context",
                "docs": list(self._all_docs),
                "confidence": "medium",
                "score": 50,
                "synthesis_mode": len(self._task_texts) > 1,
            }
        )

    def _make_task_list(self, texts: list[str], done_count: int) -> list[dict]:
        tasks = []
        for i, text in enumerate(texts):
            if i < done_count:
                status = "done"
            elif i == done_count:
                status = "active"
            else:
                status = "pending"
            tasks.append({"id": i, "text": text, "status": status})
        return tasks

    def set_final_state(self, state: dict) -> None:
        """Called by graph_runner once stream.output() resolves."""
        self._final_state = state
