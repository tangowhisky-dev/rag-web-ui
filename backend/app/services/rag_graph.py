"""
LangGraph-based multi-agent RAG orchestration.

T01: Schema definition and interface contract for run_stream.
T02: rewrite_query + context_router nodes.
T03: extract_file_sections, kb_retrieval, grade_documents nodes.
T04: merge_context, generate_answer nodes + StateGraph assembly + full run_stream().
"""

from __future__ import annotations

import time
from typing import Any, AsyncGenerator, List, Optional

from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class RAGGraphState(TypedDict):
    """Shared state passed between graph nodes."""

    query: str
    rewritten_query: str
    route: str                    # "file" | "kb" | "both"
    file_markdown: Optional[str]
    retrieved_docs: list          # raw docs from KB retrieval
    graded_docs: list             # docs that passed relevance grading
    merged_context: str           # final formatted context string
    answer: str
    agent_steps: list             # each: {"node": str, "latency_ms": float, "status": str}


# ---------------------------------------------------------------------------
# SSE event type constants (used by chat_service to map run_stream events)
# ---------------------------------------------------------------------------

EVENT_AGENT_STEP    = "agent_step"      # graph node started/finished
EVENT_REWRITTEN     = "rewritten_query" # query after rewrite node
EVENT_CONTEXT       = "context"         # retrieved + graded docs + confidence
EVENT_TOKEN         = "token"           # streaming answer token
EVENT_DONE          = "done"            # final event, carries full_response + usage


# ---------------------------------------------------------------------------
# run_stream() — async generator interface contract
#
# Interface is stable from T01 onward.  Full graph wiring arrives in T04.
# ---------------------------------------------------------------------------

async def run_stream(
    query: str,
    file_markdown: Optional[str],
    db: Any,
    chat_id: int,
    knowledge_base_ids: List[int],
    recent_lc_history: list,
    existing_summary: Optional[str],
    use_dense: bool = True,
    use_sparse: bool = True,
    use_exact: bool = True,
    use_graph_rag: bool = False,
    temperature: float = 0.0,
    model_name: Optional[str] = None,
    display_query: Optional[str] = None,
) -> AsyncGenerator[dict, None]:
    """
    Async generator that runs the full RAG graph and streams events.

    Yield shapes (keyed by "event"):
      {"event": "agent_step",    "node": str, "latency_ms": float, "status": str}
      {"event": "rewritten_query","query": str}
      {"event": "context",       "docs": list, "confidence": str, "score": float,
                                 "suggestion": str, "failed_legs": list,
                                 "breakdown": dict, "query_classification": dict,
                                 "tool_trace": list, "synthesis_mode": bool}
      {"event": "token",         "content": str}
      {"event": "done",          "full_response": str,
                                 "usage": {"promptTokens": int, "completionTokens": int}}

    Full multi-node graph (rewrite → route → retrieve → grade →
    merge → generate) is wired in T02-T04.  This stub allows chat_service
    to compile and be tested against the interface contract.
    """
    start = time.monotonic()

    # Stub: emit one placeholder agent_step so the frontend has something to
    # render while T02-T04 implement the real nodes.
    yield {
        "event": EVENT_AGENT_STEP,
        "node": "placeholder",
        "latency_ms": round((time.monotonic() - start) * 1000, 2),
        "status": "pending",
        "message": "LangGraph graph not yet wired (T02-T04).",
    }

    # Stub: emit a pass-through rewritten_query
    yield {
        "event": EVENT_REWRITTEN,
        "query": display_query or query,
    }

    # Stub: emit empty context
    yield {
        "event": EVENT_CONTEXT,
        "docs": [],
        "confidence": "low",
        "score": 0.0,
        "suggestion": "Graph not yet wired.",
        "failed_legs": [],
        "breakdown": {},
        "query_classification": {"type": "SIMPLE", "confidence": 0.0, "latency_ms": 0, "fallback": True},
        "tool_trace": [],
        "synthesis_mode": False,
    }

    # Stub: emit a placeholder answer token
    stub_msg = "[Graph not yet wired — T02-T04 pending]"
    yield {"event": EVENT_TOKEN, "content": stub_msg}

    yield {
        "event": EVENT_DONE,
        "full_response": stub_msg,
        "usage": {"promptTokens": 0, "completionTokens": 0},
    }
