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


async def classify_query(query: str, api_base: Optional[str] = None, query_model: Optional[str] = None, api_key: Optional[str] = None) -> "QueryClassification":
    """
    Classify a query into one of 4 types using LLM-based zero-shot classification.

    Returns QueryClassification with type, confidence, latency_ms, and fallback flag.
    On any failure, returns FACTUAL with fallback=True (safe default).
    """
    from app.models.query_classifier import QueryType, QueryClassification

    start = time.perf_counter()

    try:
        if not settings.QUERY_CLASSIFIER_ENABLED:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info("[CLASSIFY] disabled | latency=%.1fms query=%s", elapsed_ms, query[:80])
            return QueryClassification(
                type=QueryType.FACTUAL, confidence=0.0,
                latency_ms=elapsed_ms, fallback=True
            )

        # Fall back to app-level settings when no explicit overrides provided
        if api_key is None or api_base is None or query_model is None:
            from app.services.settings_service import get_setting
            from app.db.session import SessionLocal
            _db = SessionLocal()
            try:
                if api_key is None:
                    api_key = get_setting(_db, "QUERY_API_KEY", None) or get_setting(_db, "OPENAI_API_KEY", None)
                if api_base is None:
                    api_base = get_setting(_db, "QUERY_API_BASE", None) or get_setting(_db, "OPENAI_API_BASE", None)
                if query_model is None:
                    query_model = get_setting(_db, "QUERY_MODEL", None) or get_setting(_db, "OPENAI_MODEL", None)
            finally:
                _db.close()

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
        )

        prompt = settings.QUERY_CLASSIFIER_PROMPT.format(query=query)

        response = await client.chat.completions.create(
            model=query_model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=10,
            temperature=0,
            stream=False,
            extra_body={"thinking": {"type": "disabled"}},
        )
        
        raw = (response.choices[0].message.content or "").strip().upper()
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # Parse response — expect single enum value
        confidence = 0.0
        query_type = QueryType.FACTUAL  # safe default
        
        # Exact match
        if raw in QueryType.__members__:
            query_type = QueryType[raw]
            confidence = 1.0
        # Fuzzy match — check if any enum value is contained in response
        else:
            for member in QueryType:
                if member.value in raw:
                    query_type = member
                    confidence = 0.5
                    break
        
        logger.info("[CLASSIFY] type=%s confidence=%.2f latency=%.1fms query=%s",
                    query_type.value, confidence, elapsed_ms, query[:80])
        
        return QueryClassification(
            type=query_type, confidence=confidence,
            latency_ms=elapsed_ms, fallback=False
        )
        
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.error("[CLASSIFY] fallback due to error: %s | latency=%.1fms query=%s",
                     str(e), elapsed_ms, query[:80])
        return QueryClassification(
            type=QueryType.FACTUAL, confidence=0.0,
            latency_ms=elapsed_ms, fallback=True
        )



# ── SSE event formatter ───────────────────────────────────────────────────────

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
    logger.info("=" * 70)
    logger.info("[CHAT] chat_id=%s | kb_ids=%s | query=%r", chat_id, knowledge_base_ids, query)

    try:
        _bot_message_id: int = 0  # cached before db.close() so error handler can use it
        # ── Persist user message ───────────────────────────────────────────
        # When branching (parent_message_id provided), the user message already
        # exists — it was created by edit_message. Skip creation and link the
        # assistant reply to it directly.
        if parent_message_id is not None:
            user_message = db.query(Message).filter(Message.id == parent_message_id).first()
            if not user_message:
                yield f'3:{json.dumps({"error": "parent_message_id not found"})}\n'
                return
        else:
            user_message = Message(content=display_query or query, role="user", chat_id=chat_id)
            db.add(user_message)
            # Bump chat.updated_at so the sidebar can sort by last activity
            chat = db.query(Chat).filter(Chat.id == chat_id).first()
            if chat:
                from datetime import datetime, timezone
                chat.updated_at = datetime.now(timezone.utc)
            db.commit()

            # Link chat_file to the user message so UI can show the filename
            if file_id:
                from app.models.chat import ChatFile
                chat_file = db.query(ChatFile).filter(ChatFile.id == file_id).first()
                if chat_file:
                    chat_file.message_id = user_message.id
                    db.commit()

        # ── Persist bot placeholder ────────────────────────────────────────
        # Link assistant reply to the user message via parent_message_id so
        # branch navigation can find the correct reply for each user branch.
        bot_message = Message(
            content="", role="assistant", chat_id=chat_id,
            parent_message_id=user_message.id,
        )
        db.add(bot_message)
        db.commit()

        # ── Identity shortcut ──────────────────────────────────────────────
        if _is_identity_question(query):
            logger.info("[CHAT] identity shortcut — skipping RAG")
            yield f'0:{json.dumps(_IDENTITY_RESPONSE)}\n'
            yield f'd:{{"finishReason":"stop","usage":{{"promptTokens":0,"completionTokens":0}},"messageId":{bot_message.id},"userMessageId":{_user_message_id}}}\n'
            bot_message.content = _IDENTITY_RESPONSE
            db.commit()
            return

        # ── Knowledge base check ────────────────────────────────────────────
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

        logger.info("[CHAT] prior_messages=%d | delegating history to Redis checkpoint",
                    len(prior_messages))

        full_response = ""
        rewritten_q = display_query or query

        # Confidence capture from the context event
        _confidence_level: str | None = None
        _confidence_score: int | None = None
        _confidence_breakdown: dict | None = None
        _confidence_suggestion: str | None = None
        # Final answer evaluation from the done event
        _final_confidence: float | None = None
        _final_confidence_level: str | None = None
        _faithfulness: int | None = None
        _completeness: int | None = None
        _retrieval_score: int | None = None

        # Cache the message ID so the error handler can reference it even if
        # the bot_message instance becomes detached.
        _bot_message_id = getattr(bot_message, "id", 0) or 0
        _user_message_id = getattr(user_message, "id", 0) or 0
        buffered_citations: list[tuple[int, int, int, int, dict]] = []
        # (message_id, document_id, chunk_index, citation_index, metadata)

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

        async for event in stream_iter:
            # Check for cancellation before processing each event
            if get_cancel_token(chat_id).is_set():
                logger.info("[CHAT] cancelled | chat_id=%d | response_length=%d chars", chat_id, len(full_response))
                break

            event_type = event.get("event")

            if event_type == "agent_step":
                # Forward graph-node events for AgentTimeline rendering
                yield f'4:{json.dumps({k: v for k, v in event.items() if k != "event"})}\n'
                await stream_flush()

            elif event_type == "rewritten_query":
                rewritten_q = event.get("query", query)
                bot_message.rewritten_query = rewritten_q
                yield f'1:{json.dumps({"rewritten_query": rewritten_q})}\n'
                await asyncio.sleep(0)

            elif event_type == "context":
                # Capture confidence data for persistence (written after stream)
                _confidence_level = event.get("confidence")
                _confidence_score = event.get("score")
                _confidence_breakdown = event.get("breakdown")
                _confidence_suggestion = event.get("suggestion")
                _confidence_failed_legs = event.get("failed_legs")

                # Stream confidence metadata only; docs/citations are now supplied
                # by the answer_rewrite event after the final answer is normalized.
                context_payload = {k: v for k, v in event.items() if k != "event"}
                yield f'2:{json.dumps(context_payload)}\n'
                yield ':\n'

            elif event_type == "token":
                content = event.get("content", "")
                if isinstance(content, str):
                    full_response += content
                elif isinstance(content, list):
                    for chunk in content:
                        if isinstance(chunk, str):
                            full_response += chunk
                        elif isinstance(chunk, dict) and "text" in chunk:
                            full_response += chunk["text"]
                # logger.debug("[CHAT SSE] yield token %r", content)
                yield f'0:{json.dumps(content)}\n'
                yield ':\n'  # SSE flush comment — force chunk to leave backend buffer
            elif event_type == "answer_rewrite":
                # Citation normalisation: replace accumulated streamed text with
                # the citation-linked version. Frontend handles this via event type 'r'.
                full_response = event.get("content", full_response)

                # Buffer only the citations actually used by the normalized answer,
                # in display order (1..M).  Each item already carries the doc's
                # metadata from graph_runner.
                # Use a separate counter (not enumerate) so that docs without
                # document_id/chunk_index are skipped without creating gaps
                # in the citation_index sequence.
                buffered_citations = []
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
                        buffered_citations.append((
                            bot_message.id, document_id, chunk_index, citation_idx, meta,
                        ))

                # Forward normalized content + cited docs to the frontend so the
                # citation list matches the [1], [2], ... markers exactly.
                rewrite_payload = {
                    "content": full_response,
                    "citations": event.get("citations", []),
                }
                yield f'r:{json.dumps(rewrite_payload)}\n'
                await asyncio.sleep(0)

            elif event_type == "done":
                usage = event.get("usage", {"promptTokens": 0, "completionTokens": 0})
                # Capture final answer evaluation metrics for persistence
                _final_confidence = usage.get("final_confidence")
                _final_confidence_level = usage.get("confidence_level")
                _faithfulness = usage.get("faithfulness")
                _completeness = usage.get("completeness")
                _retrieval_score = usage.get("retrieval_score")
                yield f'd:{json.dumps({"finishReason": "stop", "usage": usage, "messageId": _bot_message_id, "userMessageId": _user_message_id})}\n'
                await asyncio.sleep(0)

            # ── Enterprise agent loop SSE events ──────────────────────────────
            elif event_type == "plan":
                yield f'pl:{json.dumps({k: v for k, v in event.items() if k != "event"})}\n'
                await asyncio.sleep(0)

            elif event_type == "tool_call":
                yield f'tc:{json.dumps({k: v for k, v in event.items() if k != "event"})}\n'
                await asyncio.sleep(0)

            elif event_type == "tool_observation":
                yield f'to:{json.dumps({k: v for k, v in event.items() if k != "event"})}\n'
                await asyncio.sleep(0)

            elif event_type == "tool_retry":
                yield f'tr:{json.dumps({k: v for k, v in event.items() if k != "event"})}\n'
                await asyncio.sleep(0)

            elif event_type == "last_answer":
                yield f'la:{json.dumps({k: v for k, v in event.items() if k != "event"})}\n'
                await asyncio.sleep(0)

            # ── New agentic agent SSE events ──────────────────────────────────
            elif event_type == "progress":
                # Transient progress/status messages — forwarded as 'p:' event
                yield f'p:{json.dumps({k: v for k, v in event.items() if k != "event"})}\n'
                await asyncio.sleep(0)

            elif event_type == "task_list":
                # Subtask list with status — forwarded as 't:' event
                yield f't:{json.dumps({k: v for k, v in event.items() if k != "event"})}\n'
                await asyncio.sleep(0)

            elif event_type == "thinking":
                # Thinking model chain-of-thought — forwarded as 'th:' event
                yield f'th:{json.dumps({k: v for k, v in event.items() if k != "event"})}\n'
                await asyncio.sleep(0)

            elif event_type == "interrupt":
                # Human-in-the-loop clarification — pause streaming, create
                # ClarificationRequest in DB, and signal the frontend to poll.
                question = event.get("question", "")
                thread_id = event.get("thread_id", "")

                logger.info(
                    "[CHAT] clarification interrupt | chat_id=%d thread_id=%s",
                    chat_id, thread_id,
                )

                # Create ClarificationRequest in DB so frontend can poll
                from app.models.clarification import ClarificationRequest as ClarificationRequestModel
                clar_req = ClarificationRequestModel(
                    chat_id=chat_id,
                    assistant_message_id=bot_message.id,
                    question=question,
                    rationale="Query needs clarification from user",
                    status="pending",
                    attempt=1,
                )
                db.add(clar_req)
                db.commit()
                db.refresh(clar_req)

                # Forward interrupt event to frontend
                interrupt_payload = {
                    "question": question,
                    "clarification_id": clar_req.id,
                    "attempt": 1,
                    "max_attempts": 2,
                }
                yield f'c:{json.dumps(interrupt_payload)}\n'
                await asyncio.sleep(0)

                # Break the stream — the agent will resume after user responds
                break

        logger.info("[CHAT] stream complete | response_length=%d chars", len(full_response))

        # ── Handle cancellation after stream ends ──────────────────────────
        if get_cancel_token(chat_id).is_set():
            # Save partial response (stream was cancelled)
            bot_message.content = full_response or "(generation stopped)"
            db.commit()
            logger.info("[CHAT] partial response saved | chat_id=%d | chars=%d", chat_id, len(bot_message.content))
            clear_cancel_token(chat_id)
            return

        # ── Persist final answer ───────────────────────────────────────────
        bot_message.content = full_response

        # Bump chat.updated_at so sidebar sorts by last activity
        _chat = db.query(Chat).filter(Chat.id == chat_id).first()
        if _chat:
            from datetime import datetime, timezone
            _chat.updated_at = datetime.now(timezone.utc)

        # Persist retrieval confidence for this message
        if _confidence_level is not None:
            bot_message.confidence_level = _confidence_level
        if _confidence_score is not None:
            bot_message.confidence_score = _confidence_score
        if _confidence_breakdown is not None:
            bot_message.confidence_breakdown = json.dumps(_confidence_breakdown)

        # Persist final answer evaluation metrics
        if _final_confidence is not None:
            bot_message.final_confidence = _final_confidence
        if _final_confidence_level is not None:
            bot_message.final_confidence_level = _final_confidence_level
        if _faithfulness is not None:
            bot_message.faithfulness = _faithfulness
        if _completeness is not None:
            bot_message.completeness = _completeness
        if _retrieval_score is not None:
            bot_message.retrieval_score = _retrieval_score

        try:
            db.commit()
            logger.info(
                "[CHAT] confidence persisted | chat_id=%d | level=%s score=%s final=%.3f",
                chat_id, _confidence_level, _confidence_score, _final_confidence or 0,
            )
        except Exception as commit_err:
            logger.warning("[CHAT] failed to persist confidence: %s", commit_err)
            try:
                db.rollback()
            except Exception:
                pass
        clear_cancel_token(chat_id)

        # ── Persist buffered citations to message_citations table ──────────
        if buffered_citations:
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
                logger.info(
                    "[CHAT] citations persisted | chat_id=%d | count=%d",
                    chat_id, len(buffered_citations),
                )
            except Exception as cit_err:
                logger.warning("[CHAT] failed to persist citations: %s", cit_err)
                try:
                    db.rollback()
                except Exception:
                    pass

        # ── Post-turn: schedule summary update (fire-and-forget) ──────────
    except Exception as e:
        error_message = f"Error generating response: {str(e)}"
        logger.error(error_message, exc_info=True)
        yield f'3:{json.dumps(error_message)}\n'
        yield f'd:{{"finishReason":"error","messageId":{_bot_message_id},"userMessageId":{_user_message_id}}}\n'
        if 'bot_message' in locals() and db.is_active:
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
