import asyncio
import json
import logging
import re
import time
from typing import List, AsyncGenerator, Optional
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)
from langchain_core.messages import HumanMessage, AIMessage
from openai import AsyncOpenAI
from app.core.config import settings
from app.models.chat import Chat, Message, MessageCitation, ChatFile
from app.services.infrastructure import get_cancel_token, clear_cancel_token
# ── SSE flush helpers ─────────────────────────────────────────────────────────
# Uvicorn buffers SSE responses by default. These helpers force the HTTP
# server to flush buffered data to the client so events arrive progressively.

async def _sse_flush():
    """Force-flush the SSE response buffer.
    
    Appends an empty SSE comment (':\\n') which signals the client to flush
    its buffer, and yields control back to the event loop.
    """
    yield ':\n'  # SSE comment — triggers client-side flush
    await asyncio.sleep(0)


async def stream_flush():
    """Wait for the async generator to produce at least one SSE flush chunk."""
    async for _ in _sse_flush():
        pass  # consume the generator


def get_effective_llm_config(org_id: Optional[int], db: Session) -> dict:
    """Return LLM config dict for the given org, falling back to .env settings.

    Keys: api_base, api_key, model_name, query_model.
    Reads from the unified settings service (3-tier precedence:
    org override → app value → .env/config.py default).
    """
    from app.services.agentic_rag.llm_factory import get_org_llm

    chat_cfg = get_org_llm(org_id, db, role="chat")
    query_cfg = get_org_llm(org_id, db, role="query")

    return {
        "api_base": chat_cfg["api_base"],
        "api_key": chat_cfg["api_key"],
        "model_name": chat_cfg["model_name"],
        "query_model": query_cfg["model_name"],
    }

# ── Constants ─────────────────────────────────────────────────────────────────

_IDENTITY_PATTERNS = re.compile(
    r"(who\s+are\s+you|what\s+are\s+you|introduce\s+yourself|tell\s+me\s+about\s+yourself|"
    r"what\s+is\s+your\s+name|what('s| is)\s+your\s+purpose|"
    r"what\s+can\s+you\s+(do|help)|how\s+can\s+you\s+help|"
    r"what\s+(tools|capabilities|features)(/\w+)?\s+.{0,40}?(have|access|available))",
    re.IGNORECASE,
)


def _is_identity_question(query: str) -> bool:
    """True if every clause in the (possibly compound) query is an identity question.

    Splits on '?' so phrasings like "Who are you? What can you help me with?"
    are recognized even though each clause is checked independently. Uses
    `search` (not an anchored full-string match) per clause so minor extra
    wording ("with?", "to?") doesn't prevent a match.
    """
    clauses = [c.strip() for c in query.strip().rstrip("?").split("?") if c.strip()]
    if not clauses:
        return False
    return all(_IDENTITY_PATTERNS.search(c) for c in clauses)

_IDENTITY_RESPONSE = (
    "I'm professional AI based Knowledge Assistant that answers questions using "
    "the documents and knowledge bases you've uploaded. "
    "Ask me anything about your content and I'll retrieve the most relevant information "
    "and give you a clear, cited answer."
)


# ── LLM helpers ───────────────────────────────────────────────────────────────

from app.services.infrastructure import strip_reasoning_tags


def _strip_think(text: str) -> str:
    """Remove reasoning tag blocks emitted by reasoning models."""
    return strip_reasoning_tags(text)


# ── SSE event formatter ───────────────────────────────────────────────────────

# ── Stream context ────────────────────────────────────────────────────────────

class _StreamContext:
    def __init__(self, query, bot_message, user_message, db, chat_id,
                 bot_message_id, user_message_id):
        self.query = query
        self.bot_message = bot_message
        self.user_message = user_message
        self.db = db
        self.chat_id = chat_id
        self.bot_message_id = bot_message_id
        self.user_message_id = user_message_id
        self.full_response = ""
        self.rewritten_q = None
        self.buffered_citations: list[tuple[int, int, int, int, dict]] = []
        self.confidence_level: str | None = None
        self.confidence_score: int | None = None
        self.confidence_breakdown: dict | None = None
        self.confidence_suggestion: str | None = None
        self.confidence_failed_legs = None
        self.final_confidence: float | None = None
        self.final_confidence_level: str | None = None
        self.faithfulness: int | None = None
        self.completeness: int | None = None
        self.retrieval_score: int | None = None
        self.interrupt = False


# ── Event handlers ─────────────────────────────────────────────────────────────

async def _handle_agent_step(event, ctx):
    yield f'4:{json.dumps({k: v for k, v in event.items() if k != "event"})}\n'
    await stream_flush()


async def _handle_rewritten_query(event, ctx):
    ctx.rewritten_q = event.get("query", ctx.query)
    ctx.bot_message.rewritten_query = ctx.rewritten_q
    yield f'1:{json.dumps({"rewritten_query": ctx.rewritten_q})}\n'
    await asyncio.sleep(0)


async def _handle_expanded_query(event, ctx):
    expanded_q = event.get("query", "")
    ctx.user_message.expanded_query = expanded_q
    ctx.db.commit()
    yield f'eq:{json.dumps({"expanded_query": expanded_q})}\n'
    await asyncio.sleep(0)


async def _handle_context(event, ctx):
    ctx.confidence_level = event.get("confidence")
    ctx.confidence_score = event.get("score")
    ctx.confidence_breakdown = event.get("breakdown")
    ctx.confidence_suggestion = event.get("suggestion")
    ctx.confidence_failed_legs = event.get("failed_legs")
    context_payload = {k: v for k, v in event.items() if k != "event"}
    yield f'2:{json.dumps(context_payload)}\n'
    yield ':\n'


async def _handle_token(event, ctx):
    content = event.get("content", "")
    if isinstance(content, str):
        ctx.full_response += content
    elif isinstance(content, list):
        for chunk in content:
            if isinstance(chunk, str):
                ctx.full_response += chunk
            elif isinstance(chunk, dict) and "text" in chunk:
                ctx.full_response += chunk["text"]
    # logger.debug("[CHAT SSE] yield token %r", content)
    yield f'0:{json.dumps(content)}\n'
    yield ':\n'  # SSE flush comment — force chunk to leave backend buffer


async def _handle_answer_rewrite(event, ctx):
    ctx.full_response = event.get("content", ctx.full_response)
    ctx.buffered_citations = []
    citation_idx = 0
    for doc in event.get("citations", []):
        document_id = doc.get("metadata", {}).get("document_id")
        chunk_index = doc.get("metadata", {}).get("chunk_index")
        if document_id is not None and chunk_index is not None:
            citation_idx += 1
            meta = {**(doc.get("metadata", {}) or {})}
            for rk in ("score", "dense_rank", "sparse_rank", "exact_rank", "retrieval_leg"):
                v = doc.get(rk)
                if v is not None:
                    meta[rk] = v
            ctx.buffered_citations.append((
                ctx.bot_message.id, document_id, chunk_index, citation_idx, meta,
            ))
    rewrite_payload = {
        "content": ctx.full_response,
        "citations": event.get("citations", []),
    }
    yield f'r:{json.dumps(rewrite_payload)}\n'
    await asyncio.sleep(0)


async def _handle_done(event, ctx):
    usage = event.get("usage", {"promptTokens": 0, "completionTokens": 0})
    ctx.final_confidence = usage.get("final_confidence")
    ctx.final_confidence_level = usage.get("confidence_level")
    ctx.faithfulness = usage.get("faithfulness")
    ctx.completeness = usage.get("completeness")
    ctx.retrieval_score = usage.get("retrieval_score")
    yield f'd:{json.dumps({"finishReason": "stop", "usage": usage, "messageId": ctx.bot_message_id, "userMessageId": ctx.user_message_id})}\n'
    await asyncio.sleep(0)


async def _handle_plan(event, ctx):
    yield f'pl:{json.dumps({k: v for k, v in event.items() if k != "event"})}\n'
    await asyncio.sleep(0)


async def _handle_tool_call(event, ctx):
    yield f'tc:{json.dumps({k: v for k, v in event.items() if k != "event"})}\n'
    await asyncio.sleep(0)


async def _handle_tool_observation(event, ctx):
    yield f'to:{json.dumps({k: v for k, v in event.items() if k != "event"})}\n'
    await asyncio.sleep(0)


async def _handle_tool_retry(event, ctx):
    yield f'tr:{json.dumps({k: v for k, v in event.items() if k != "event"})}\n'
    await asyncio.sleep(0)


async def _handle_last_answer(event, ctx):
    yield f'la:{json.dumps({k: v for k, v in event.items() if k != "event"})}\n'
    await asyncio.sleep(0)


async def _handle_progress(event, ctx):
    yield f'p:{json.dumps({k: v for k, v in event.items() if k != "event"})}\n'
    await asyncio.sleep(0)


async def _handle_task_list(event, ctx):
    yield f't:{json.dumps({k: v for k, v in event.items() if k != "event"})}\n'
    await asyncio.sleep(0)


async def _handle_thinking(event, ctx):
    yield f'th:{json.dumps({k: v for k, v in event.items() if k != "event"})}\n'
    await asyncio.sleep(0)


async def _handle_interrupt(event, ctx):
    question = event.get("question", "")
    thread_id = event.get("thread_id", "")
    logger.debug(
        "[CHAT] clarification interrupt | chat_id=%d thread_id=%s",
        ctx.chat_id, thread_id,
    )
    from app.models.clarification import ClarificationRequest as ClarificationRequestModel
    clar_req = ClarificationRequestModel(
        chat_id=ctx.chat_id,
        assistant_message_id=ctx.bot_message.id,
        question=question,
        rationale="Query needs clarification from user",
        status="pending",
        attempt=1,
    )
    ctx.db.add(clar_req)
    ctx.db.commit()
    ctx.db.refresh(clar_req)
    interrupt_payload = {
        "question": question,
        "clarification_id": clar_req.id,
        "attempt": 1,
        "max_attempts": 2,
    }
    yield f'c:{json.dumps(interrupt_payload)}\n'
    await asyncio.sleep(0)
    ctx.interrupt = True


EVENT_HANDLERS = {
    "agent_step": _handle_agent_step,
    "rewritten_query": _handle_rewritten_query,
    "expanded_query": _handle_expanded_query,
    "context": _handle_context,
    "token": _handle_token,
    "answer_rewrite": _handle_answer_rewrite,
    "done": _handle_done,
    "plan": _handle_plan,
    "tool_call": _handle_tool_call,
    "tool_observation": _handle_tool_observation,
    "tool_retry": _handle_tool_retry,
    "last_answer": _handle_last_answer,
    "progress": _handle_progress,
    "task_list": _handle_task_list,
    "thinking": _handle_thinking,
    "interrupt": _handle_interrupt,
}


def _link_file_to_chat(db: Session, file_id: int, user_message_id: int) -> None:
    from app.models.chat import ChatFile
    chat_file = db.query(ChatFile).filter(ChatFile.id == file_id).first()
    if chat_file:
        chat_file.message_id = user_message_id
        db.commit()


def _create_user_message(
    db: Session,
    chat_id: int,
    query: str,
    display_query: Optional[str],
    parent_message_id: Optional[int],
    file_id: Optional[int],
) -> tuple:
    if parent_message_id is not None:
        user_message = db.query(Message).filter(Message.id == parent_message_id).first()
        if not user_message:
            return None, f'3:{json.dumps({"error": "parent_message_id not found"})}\n'
        return user_message, None
    user_message = Message(content=display_query or query, role="user", chat_id=chat_id)
    db.add(user_message)
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if chat:
        from datetime import datetime, timezone
        chat.updated_at = datetime.now(timezone.utc)
    db.commit()
    if file_id:
        _link_file_to_chat(db, file_id, user_message.id)
    return user_message, None


async def _handle_identity_question(
    bot_message: Message, user_message_id: int, db: Session,
) -> AsyncGenerator[str, None]:
    logger.debug("[CHAT] identity shortcut — skipping RAG")
    yield f'0:{json.dumps(_IDENTITY_RESPONSE)}\n'
    yield f'd:{{"finishReason":"stop","usage":{{"promptTokens":0,"completionTokens":0}},"messageId":{bot_message.id},"userMessageId":{user_message_id}}}\n'
    bot_message.content = _IDENTITY_RESPONSE
    db.commit()


def _persist_response_metadata(
    bot_message: Message, db: Session, chat_id: int, ctx: "_StreamContext",
) -> None:
    bot_message.content = ctx.full_response
    _chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if _chat:
        from datetime import datetime, timezone
        _chat.updated_at = datetime.now(timezone.utc)
    if ctx.confidence_level is not None:
        bot_message.confidence_level = ctx.confidence_level
    if ctx.confidence_score is not None:
        bot_message.confidence_score = ctx.confidence_score
    if ctx.confidence_breakdown is not None:
        bot_message.confidence_breakdown = json.dumps(ctx.confidence_breakdown)
    if ctx.final_confidence is not None:
        bot_message.final_confidence = ctx.final_confidence
    if ctx.final_confidence_level is not None:
        bot_message.final_confidence_level = ctx.final_confidence_level
    if ctx.faithfulness is not None:
        bot_message.faithfulness = ctx.faithfulness
    if ctx.completeness is not None:
        bot_message.completeness = ctx.completeness
    if ctx.retrieval_score is not None:
        bot_message.retrieval_score = ctx.retrieval_score
    try:
        db.commit()
        logger.debug(
            "[CHAT] confidence persisted | chat_id=%d | level=%s score=%s final=%.3f",
            chat_id, ctx.confidence_level, ctx.confidence_score, ctx.final_confidence or 0,
        )
    except Exception as commit_err:
        logger.warning("[CHAT] failed to persist confidence: %s", commit_err)
        try:
            db.rollback()
        except Exception:
            pass


def _persist_citations(db: Session, chat_id: int, buffered_citations: list) -> None:
    if not buffered_citations:
        return
    try:
        for msg_id, document_id, chunk_index, citation_index, metadata in buffered_citations:
            db.add(
                MessageCitation(
                    message_id=msg_id,
                    document_id=document_id,
                    chunk_index=chunk_index,
                    citation_index=citation_index,
                    citation_metadata=metadata,
                )
            )
        db.commit()
        logger.debug(
            "[CHAT] citations persisted | chat_id=%d | count=%d",
            chat_id, len(buffered_citations),
        )
    except Exception as cit_err:
        logger.warning("[CHAT] failed to persist citations: %s", cit_err)
        try:
            db.rollback()
        except Exception:
            pass


async def _process_stream_events(
    stream_iter: AsyncGenerator, ctx: "_StreamContext", chat_id: int,
) -> AsyncGenerator[str, None]:
    async for event in stream_iter:
        if get_cancel_token(chat_id).is_set():
            logger.debug("[CHAT] cancelled | chat_id=%d | response_length=%d chars", chat_id, len(ctx.full_response))
            break
        event_type = event.get("event")
        handler = EVENT_HANDLERS.get(event_type)
        if handler:
            async for chunk in handler(event, ctx):
                yield chunk
            if ctx.interrupt:
                break


def _finalize_stream(
    bot_message: Message, db: Session, chat_id: int, ctx: "_StreamContext",
) -> None:
    if get_cancel_token(chat_id).is_set():
        bot_message.content = ctx.full_response or "(generation stopped)"
        db.commit()
        logger.debug("[CHAT] partial response saved | chat_id=%d | chars=%d", chat_id, len(bot_message.content))
        clear_cancel_token(chat_id)
        return
    _persist_response_metadata(bot_message, db, chat_id, ctx)
    clear_cancel_token(chat_id)
    _persist_citations(db, chat_id, ctx.buffered_citations)


async def _emit_response_error(
    exc: Exception, db: Session, chat_id: int,
    bot_message_id: int, user_message_id: int, bot_message: Optional[Message],
) -> AsyncGenerator[str, None]:
    error_message = f"Error generating response: {str(exc)}"
    logger.error(error_message, exc_info=True)
    yield f'3:{json.dumps(error_message)}\n'
    yield f'd:{{"finishReason":"error","messageId":{bot_message_id},"userMessageId":{user_message_id}}}\n'
    if bot_message is not None and db.is_active:
        try:
            bot_message.content = error_message
            db.commit()
        except Exception as commit_err:
            logger.warning("[CHAT] failed to persist error message: %s", commit_err)
            try:
                db.rollback()
            except Exception:
                pass
    clear_cancel_token(chat_id)


# ── Main entry point ──────────────────────────────────────────────────────────

async def generate_response(
    query: str,
    messages: dict,
    knowledge_base_ids: List[int],
    chat_id: int,
    db: Session,
    display_query: Optional[str] = None,
    file_id: Optional[int] = None,
    file_markdown: Optional[str] = None,
    org_id: Optional[int] = None,
    user_id: Optional[int] = None,
    parent_message_id: Optional[int] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream a chat response for the given query.

    Delegates the full RAG pipeline (query rewrite -> routing -> retrieval ->
    grading -> generation) to run_agentic_rag(), which emits typed events
    that are forwarded as Vercel AI SDK SSE frames:

      0:  token             (streaming answer text)
      1:  rewritten_query   (standalone question after rewrite node)
      2:  context           (retrieved docs + confidence metadata)
      3:  error             (exception message)
      4:  agent_step        (LangGraph node start / finish event)
      d:  done              (finish reason + token usage)

    file_markdown is forwarded to run_stream; the graph routes the query
    internally — no special-casing in this function.
    """
    logger.debug("=" * 70)
    logger.debug("[CHAT] chat_id=%s | kb_ids=%s | query=%r", chat_id, knowledge_base_ids, query)

    _bot_message_id: int = 0
    _user_message_id: int = 0
    bot_message = None

    try:
        user_message, error_frame = _create_user_message(
            db, chat_id, query, display_query, parent_message_id, file_id,
        )
        if error_frame is not None:
            yield error_frame
            return

        bot_message = Message(
            content="", role="assistant", chat_id=chat_id,
            parent_message_id=user_message.id,
        )
        db.add(bot_message)
        db.commit()

        if _is_identity_question(query):
            async for chunk in _handle_identity_question(bot_message, user_message.id, db):
                yield chunk
            return

        # NOTE: we intentionally do NOT hard-fail here when no KB is attached.
        # The agentic loop supports plenty of intents that need no documents at
        # all (code_execute, chart_generate on inline data, plain conversation,
        # identity questions with unusual phrasing, etc.) — short-circuiting to
        # an error before planning even starts would block all of those. If the
        # query genuinely needs retrieval, rag_retrieve returns zero docs when
        # knowledge_base_ids is empty and the agent explains it found nothing,
        # which is the correct behavior for a non-RAG-only agentic pipeline.

        # The Redis/LangGraph checkpointer is the single source of truth for
        # conversation history. We no longer pass a sliding window or MySQL
        # rolling summary into the graph.
        prior_messages = messages["messages"][:-1]

        logger.debug("[CHAT] prior_messages=%d | delegating history to Redis checkpoint",
                    len(prior_messages))

        _bot_message_id = getattr(bot_message, "id", 0) or 0
        _user_message_id = getattr(user_message, "id", 0) or 0

        ctx = _StreamContext(
            query=query,
            bot_message=bot_message,
            user_message=user_message,
            db=db,
            chat_id=chat_id,
            bot_message_id=_bot_message_id,
            user_message_id=_user_message_id,
        )
        ctx.rewritten_q = display_query or query

        # ── Agentic pipeline: single autonomous agent ───────────────────────
        # New agentic agent: rewrite -> search -> stream in real-time
        from app.services.agentic_rag import run_agentic_rag
        stream_iter = run_agentic_rag(
            query=query,
            file_markdown=file_markdown,
            db=db,
            chat_id=chat_id,
            knowledge_base_ids=knowledge_base_ids,
            display_query=display_query,
            org_id=org_id,
            user_id=user_id,
            message_id=_bot_message_id,
        )

        async for chunk in _process_stream_events(stream_iter, ctx, chat_id):
            yield chunk

        logger.debug("[CHAT] stream complete | response_length=%d chars", len(ctx.full_response))

        _finalize_stream(bot_message, db, chat_id, ctx)

        # ── Post-turn: schedule summary update (fire-and-forget) ──────────
    except Exception as e:
        async for chunk in _emit_response_error(e, db, chat_id, _bot_message_id, _user_message_id, bot_message):
            yield chunk
# ── SSE flush helpers ─────────────────────────────────────────────────────────
# Uvicorn buffers SSE responses by default. These helpers force the HTTP
# server to flush buffered data to the client so events arrive progressively.

async def _sse_flush():
    """Force-flush the SSE response buffer.

    Appends an empty SSE comment (':\\n') which signals the client to flush
    its buffer, and yields control back to the event loop.
    """
    yield ':\n'  # SSE comment — triggers client-side flush
    await asyncio.sleep(0)


async def stream_flush():
    """Wait for the async generator to produce at least one SSE flush chunk."""
    async for _ in _sse_flush():
        pass  # consume the generator
