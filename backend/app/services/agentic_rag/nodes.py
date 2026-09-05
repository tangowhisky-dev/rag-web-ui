"""LangGraph node implementations for the agentic RAG pipeline."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.config import get_stream_writer

from app.core.settings_registry import get_def

from app.services.retrieval import dedup_by_content_hash

from .graph_state import AgentState
from .utils import format_context_string

logger = logging.getLogger(__name__)


def select_recent_history(messages: list, max_pairs: int | None = None, db: Any = None, org_id: Any = None) -> list:
    """Return up to ``max_pairs`` of recent user/assistant turns.

    The last message is assumed to be the current user query and is excluded.
    Assistant turns are not truncated: the resolver, think and finalize nodes
    need the full prior answer to resolve references and compare approaches.
    Truncating here loses facts that cannot be recovered downstream.
    """
    from langchain_core.messages import HumanMessage, AIMessage

    if max_pairs is None:
        from app.services.settings_service import get_setting
        max_pairs = get_setting(db, "AGENT_HISTORY_PAIRS", org_id) if db is not None else get_def("AGENT_HISTORY_PAIRS").default

    history: list = []
    # Skip the final message (current query).
    for m in messages[:-1]:
        if isinstance(m, HumanMessage):
            history.append(HumanMessage(content=m.content))
        elif isinstance(m, AIMessage):
            history.append(AIMessage(content=m.content))
    # Keep the most recent max_pairs * 2 messages.
    return history[-(max_pairs * 2):]


def history_to_text(messages: list) -> str:
    """Render selected history messages as a ``User:``/``Assistant:`` block."""
    lines = []
    for msg in messages:
        role = "User" if getattr(msg, "type", "") == "human" else "Assistant"
        lines.append(f"  {role}: {msg.content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Conversation text helper (shared by the single compaction implementation)
# ---------------------------------------------------------------------------

def _messages_to_conversation_text(messages: list) -> str:
    """Convert LangChain messages to readable conversation text for summarization.

    No per-turn truncation: the summarizer cannot preserve facts that were
    removed before it ever saw them.
    """
    parts = []
    for m in messages:
        if isinstance(m, HumanMessage):
            parts.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            parts.append(f"Assistant: {m.content}")
    return "\n\n".join(parts)


@contextmanager
def _agent_step(name: str) -> Generator[None, None, None]:
    """Emit agent_step active/done lifecycle events around a node.

    No-op when called outside a LangGraph runnable context (e.g. unit tests).
    """
    writer = _safe_writer()
    if writer is not None:
        writer({"event": "agent_step", "node": name, "status": "active", "latency_ms": 0})
    try:
        yield
    finally:
        if writer is not None:
            writer({"event": "agent_step", "node": name, "status": "done", "latency_ms": 0})


def _safe_writer():
    """Return the stream writer if inside a graph context, else None.

    Retrieval nodes are called both from the graph (where stream events work)
    and from search tools (where there is no graph context). This
    helper lets nodes emit progress events safely in both cases.
    """
    try:
        return get_stream_writer()
    except (RuntimeError, KeyError):
        return None


# ── Answer Generation ──────────────────────────────────────────────────────


def _get_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    streaming: bool = False,
) -> ChatOpenAI:
    """Build a ChatOpenAI from explicit overrides or app-level settings.

    Prefer build_chat_llm() / get_org_llm() for per-org resolution.
    This fallback path is used when org context is unavailable (e.g. compaction
    fallback in agent_graph.py).
    """
    # Resolve from app-level settings when not explicitly provided
    if model_name is None or api_base is None or api_key is None:
        from app.services.settings_service import get_setting
        from app.db.session import SessionLocal
        _db = SessionLocal()
        try:
            if model_name is None:
                model_name = get_setting(_db, "QUERY_MODEL", None) or get_setting(_db, "OPENAI_MODEL", None)
            if api_base is None:
                api_base = get_setting(_db, "OPENAI_API_BASE", None)
            if api_key is None:
                api_key = get_setting(_db, "OPENAI_API_KEY", None)
        finally:
            _db.close()
    # Local servers (LM Studio, Ollama) don't require a key, but the OpenAI
    # client rejects None/empty — supply a placeholder when unset.
    if not api_key:
        api_key = "not-required"
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_base=api_base,
        openai_api_key=api_key,
        streaming=streaming,
    )


# ---------------------------------------------------------------------------
# Node: answer_evaluation
# ---------------------------------------------------------------------------

async def answer_evaluation_node(
    state: AgentState,
    llm: ChatOpenAI | None = None,
    ctx: Any = None,
) -> dict:
    """Evaluate final answer quality, extract structured data, and compute confidence.

    Single LLM call combines:
    - faithfulness/completeness scoring (needs query + cited context + answer)
    - summary/key_points/data extraction (needs answer)
    - followups/retry_strategy generation (needs answer + query)
    """
    with _agent_step("answer_evaluation"):
        from .evaluator import evaluate_answer

        answer = state.get("answer", "")
        query = state.get("original_query", "")
        # Use cited_docs from state (set by finalize_node for both evidence
        # and legacy citation paths). Fall back to cited_doc_indices for
        # backward compatibility, then to all docs.
        cited_docs = state.get("cited_docs", [])
        if cited_docs:
            docs = cited_docs
        else:
            all_docs = state.get("retrieved_docs", [])
            cited_indices = state.get("cited_doc_indices", [])
            if cited_indices:
                docs = [all_docs[i - 1] for i in cited_indices if 0 < i <= len(all_docs)]
            else:
                docs = all_docs

        retrieval_conf = state.get("best_retrieval_confidence", 0.0)

        if not answer:
            return {
                "answer_evaluation_attempts": state.get("answer_evaluation_attempts", 0) + 1,
                "final_confidence": 0.0,
                "confidence_level": "none",
                "faithfulness": 0,
                "completeness": 0,
                "retrieval_score": 0,
            }

        _db = ctx.db if ctx is not None else None
        _org_id = ctx.org_id if ctx is not None else None
        context_text = format_context_string(docs, state.get("file_markdown"), db=_db, org_id=_org_id)

        # Retrieval confidence level.
        if retrieval_conf > 0.8:
            conf_level = "very_high"
        elif retrieval_conf > 0.6:
            conf_level = "high"
        elif retrieval_conf > 0.3:
            conf_level = "medium"
        else:
            conf_level = "low"

        # Resolve eval LLM kwargs from org context.
        eval_kwargs: dict = {}
        if ctx is not None:
            try:
                from app.services.agentic_rag.llm_factory import get_org_llm
                query_cfg = get_org_llm(ctx.org_id, ctx.db, role="query")
                eval_kwargs = {
                    "api_base": query_cfg["api_base"],
                    "api_key": query_cfg["api_key"],
                    "query_model": query_cfg["model_name"],
                }
            except Exception:
                pass

        try:
            evaluation = await evaluate_answer(
                query=query,
                answer=answer,
                context_preview=context_text,
                confidence_level=conf_level,
                **eval_kwargs,
            )
            faithfulness = evaluation.faithfulness
            completeness = evaluation.completeness
            eval_flags = evaluation.flags
        except Exception as exc:
            logger.warning("[ANSWER_EVALUATION] failed: %s", exc)
            faithfulness = 50
            completeness = 50
            eval_flags = ["Evaluation unavailable"]
            evaluation = None

        # If the answer says "no information found", the retrieval score
        # should be low — the retrieved evidence was not used to answer the
        # query, even if citations exist (they cite irrelevant context).
        answer_lower = answer.lower()
        says_no_info = (
            "no information" in answer_lower
            or "do not contain" in answer_lower
            or "no mention" in answer_lower
            or "not possible to determine" in answer_lower
            or "no relevant" in answer_lower
        )
        if says_no_info:
            retrieval_score = 0
        else:
            retrieval_score = min(retrieval_conf, 1.0) * 100
        final_confidence = (
            0.4 * retrieval_score +
            0.3 * faithfulness +
            0.3 * completeness
        )
        final_confidence = round(min(final_confidence / 100.0, 1.0), 3)

        # Final confidence level.
        if final_confidence > 0.8:
            confidence_level = "very_high"
        elif final_confidence > 0.6:
            confidence_level = "high"
        elif final_confidence > 0.3:
            confidence_level = "medium"
        elif final_confidence > 0:
            confidence_level = "low"
        else:
            confidence_level = "none"

        # Update LastAnswerObject with LLM-extracted fields.
        updates = {
            "answer_evaluation_attempts": state.get("answer_evaluation_attempts", 0) + 1,
            "final_confidence": final_confidence,
            "confidence_level": confidence_level,
            "faithfulness": faithfulness,
            "completeness": completeness,
            "retrieval_score": int(retrieval_score),
            "evaluation_flags": eval_flags,
        }

        if evaluation is not None:
            # Update the LastAnswerObject with LLM-extracted fields.
            lao = state.get("last_answer_object")
            if lao is not None:
                from app.services.agentic_rag.schemas import LastAnswerObject, DataPoint
                # Convert data dicts to DataPoint objects if needed.
                data_points = None
                if evaluation.data:
                    try:
                        data_points = [DataPoint(**d) if isinstance(d, dict) else d for d in evaluation.data]
                    except Exception:
                        data_points = None
                lao.summary = evaluation.summary
                lao.key_points = evaluation.key_points
                lao.data = data_points
                lao.followups = evaluation.followups
                lao.retry_strategy = evaluation.retry_strategy
                updates["last_answer_object"] = lao

        return updates
