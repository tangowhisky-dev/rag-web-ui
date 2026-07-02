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
from app.models.knowledge import KnowledgeBase
from app.services.retrieval import hybrid_search_with_legs
from app.services.confidence import score_retrieval
from app.services.cancel_registry import get_cancel_token, clear_cancel_token


def get_effective_llm_config(org_id: Optional[int], db: Session) -> dict:
    """Return LLM config dict for the given org, falling back to .env settings.

    Keys: api_base, model_name, query_model.
    When org_id is None or no OrgLLMConfig row exists, all values are None
    (callers should fall back to settings).
    """
    api_base = None
    model_name = None
    query_model = None

    if org_id is not None:
        from app.models.org_llm_config import OrgLLMConfig
        row = db.query(OrgLLMConfig).filter(OrgLLMConfig.org_id == org_id).first()
        if row:
            api_base = row.api_base or None
            model_name = row.model_name or None
            query_model = row.query_model or None

    return {
        "api_base": api_base or settings.OPENAI_API_BASE,
        "model_name": model_name or settings.OPENAI_MODEL,
        "query_model": query_model or settings.effective_query_model,
    }

# ── Constants ─────────────────────────────────────────────────────────────────

# Number of most-recent full user/assistant turn-pairs to include verbatim.
# 3 pairs = 6 messages (3 human + 3 assistant).
_SLIDING_WINDOW_PAIRS = 3
_SLIDING_WINDOW_MESSAGES = _SLIDING_WINDOW_PAIRS * 2  # 6

_IDENTITY_PATTERNS = re.compile(
    r"^\s*(who\s+are\s+you|what\s+are\s+you|introduce\s+yourself|tell\s+me\s+about\s+yourself|"
    r"what\s+is\s+your\s+name|what('s| is)\s+your\s+purpose|what\s+can\s+you\s+do)\s*\??\s*$",
    re.IGNORECASE,
)

_IDENTITY_RESPONSE = (
    "I'm professional AI based Knowledge Assistant that answers questions using "
    "the documents and knowledge bases you've uploaded. "
    "Ask me anything about your content and I'll retrieve the most relevant information "
    "and give you a clear, cited answer."
)


def _is_identity_question(query: str) -> bool:
    return bool(_IDENTITY_PATTERNS.match(query.strip()))


_SYNTHESIS_KEYWORDS = re.compile(
    r"\b(summarize|summary|themes?|across|compare|comparison|overview|report|"
    r"aggregate|synthesis|synthesize|all\s+\w+|key\s+(findings?|points?|themes?))\b",
    re.IGNORECASE,
)

def _is_synthesis_query(query: str, query_type: str = "") -> bool:
    """
    Heuristic: True for MULTI_PART queries that contain synthesis keywords.
    These queries benefit from synthesize_documents rather than single-shot retrieval.
    """
    if query_type.upper() not in ("MULTI_PART", "AMBIGUOUS"):
        return False
    return bool(_SYNTHESIS_KEYWORDS.search(query))


_SYNTHESIS_SYSTEM_PROMPT = (
    "You are a professional research assistant that synthesizes information from multiple documents.\n\n"
    "## Your task\n"
    "The user wants a comprehensive synthesis across multiple documents. Follow this workflow:\n\n"
    "1. **Gather coverage**: Call `synthesize_documents` with the topic and 3–6 targeted sub-queries "
    "to retrieve a broad, deduplicated set of relevant chunks.\n"
    "2. **Extract entities** (optional): Call `extract_entities` on the combined text if entity context "
    "would enrich the synthesis.\n"
    "3. **Summarize**: Call `summarize_chunks` with a precise instruction to synthesize the gathered chunks.\n"
    "4. **Write the report**: After tool calls are done, write a structured Markdown report with:\n"
    "   - `## Executive Summary` — 2–3 sentence overview\n"
    "   - `## Key Themes` — bullet points for each major theme\n"
    "   - `## Details` — elaboration per theme with [citation:N] references\n"
    "   - `## Sources` — list of source documents cited\n\n"
    "## Citation rules\n"
    "Cite using EXACTLY [citation:N] format where N is the chunk number from context (1-indexed). "
    "Do NOT use [1], (1), or any other format.\n\n"
    "## Style\n"
    "Be precise, professional, and concise. Do not pad. Do not repeat yourself."
)


# ── LLM helpers ───────────────────────────────────────────────────────────────

from app.services.reasoning_tags import strip_reasoning_tags


def _strip_think(text: str) -> str:
    """Remove reasoning tag blocks emitted by reasoning models."""
    return strip_reasoning_tags(text)


# ── Summary ───────────────────────────────────────────────────────────────────

async def _summarise_older_messages(
    messages_to_summarise: List[dict],
    existing_summary: str | None,
    chat_id: int,
) -> str:
    """
    Produce (or update) a rolling summary that covers all dialogue *outside*
    the sliding window.

    If an existing summary exists, it is passed in and the new batch of
    messages is folded into it — so the summary is always cumulative.
    """
    # Build a plain-text transcript of the messages being summarised
    transcript_lines = []
    for m in messages_to_summarise:
        role = "User" if m["role"] == "user" else "Assistant"
        # Strip the base64 context prefix that assistant messages contain
        content = m["content"]
        if "__LLM_RESPONSE__" in content:
            content = content.split("__LLM_RESPONSE__")[-1]
        transcript_lines.append(f"{role}: {content.strip()}")
    transcript = "\n".join(transcript_lines)

    if existing_summary:
        system_prompt = (
            "You are a precise dialogue summariser. "
            "You will be given a running summary of an earlier part of a conversation "
            "and a new batch of exchanges to fold into it.\n\n"
            "Rules:\n"
            "- Produce a single, compact summary that covers everything: the existing summary PLUS the new exchanges.\n"
            "- Preserve every fact, decision, preference, and piece of information the user provided or the assistant stated.\n"
            "- Capture questions asked and answers given — especially facts extracted from documents.\n"
            "- Keep it dense but readable — use short bullet points or tightly written prose.\n"
            "- Do NOT omit details; losing information defeats the purpose.\n"
            "- Output ONLY the updated summary — no preamble, no labels, no extra text."
        )
        user_prompt = (
            f"EXISTING SUMMARY:\n{existing_summary}\n\n"
            f"NEW EXCHANGES TO FOLD IN:\n{transcript}"
        )
    else:
        system_prompt = (
            "You are a precise dialogue summariser. "
            "You will be given a conversation transcript to summarise.\n\n"
            "Rules:\n"
            "- Produce a compact summary that captures every significant fact, question, answer, and decision.\n"
            "- Include what documents or topics were discussed and what was found.\n"
            "- Keep it dense but readable — use short bullet points or tightly written prose.\n"
            "- Do NOT omit details; losing information defeats the purpose.\n"
            "- Output ONLY the summary — no preamble, no labels, no extra text."
        )
        user_prompt = f"CONVERSATION TRANSCRIPT:\n{transcript}"

    logger.info("[SUMMARY] chat_id=%d | summarising %d messages | has_existing=%s",
                chat_id, len(messages_to_summarise), bool(existing_summary))

    client = AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_API_BASE,
    )
    response = await client.chat.completions.create(
        model=settings.effective_query_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        stream=False,
    )
    summary = _strip_think(response.choices[0].message.content or "")
    logger.info("[SUMMARY] chat_id=%d | summary_length=%d chars", chat_id, len(summary))
    return summary


async def _maybe_update_summary(
    chat_id: int,
    all_prior_messages: List[dict],
    existing_summary: str | None,
) -> None:
    """
    Called as a fire-and-forget background task after the response stream
    completes. Checks whether there are messages beyond the sliding window
    and, if so, summarises them and persists the result.

    Uses a fresh DB session so it doesn't interfere with the main request.
    """
    from app.db.session import SessionLocal

    # Messages outside the window are everything except the last N
    # (window) + the pair just completed (2 more = current user + bot turn).
    # At call time all_prior_messages already includes only historical turns
    # (not the current one) — the current pair gets appended here.
    # Actually: all_prior_messages is the full history BEFORE the current turn.
    # The current turn was just committed. We need to reload from DB.
    db = SessionLocal()
    try:
        chat = db.query(Chat).filter(Chat.id == chat_id).first()
        if not chat:
            return

        # Full ordered message list from DB (just committed, so current turn is included)
        db_messages = (
            db.query(Message)
            .filter(Message.chat_id == chat_id)
            .order_by(Message.id)
            .all()
        )

        # Convert to plain dicts, excluding empty bot placeholders
        all_msgs = [
            {"role": m.role, "content": m.content}
            for m in db_messages
            if m.content.strip()
        ]

        total = len(all_msgs)
        if total == 0:
            return

        # Build incrementally: if a summary already exists it covers everything
        # up to the previous turn, so only fold in the latest turn (last 2 msgs).
        # On first call (no existing summary) summarise the full history.
        messages_to_summarise = all_msgs[-2:] if existing_summary else all_msgs

        summary = await _summarise_older_messages(
            messages_to_summarise=messages_to_summarise,
            existing_summary=existing_summary,
            chat_id=chat_id,
        )

        chat.history_summary = summary
        db.commit()
        logger.info("[SUMMARY] chat_id=%d | summary persisted (%d chars)", chat_id, len(summary))

    except Exception as e:
        logger.error("[SUMMARY] chat_id=%d | error: %s", chat_id, e)
    finally:
        try:
            db.close()
        except Exception:
            pass  # session may already be closed by caller or during cancellation


# ── Query rewrite ──────────────────────────────────────────────────────────────

async def _rewrite_query(
    query: str,
    recent_history: List,  # LangChain message objects
    api_base: Optional[str] = None,
    query_model: Optional[str] = None,
) -> str:
    """
    Condense the current query + recent chat history into a self-contained
    standalone question for retrieval.
    """
    if not recent_history:
        return query

    # Build messages manually — avoids LangChain template curly-brace hazards
    # and lets us set max_tokens to prevent the small model from answering instead of rewriting
    system_msg = (
        "You are a search query rewriter for a document retrieval system. "
        "Your ONLY job is to rewrite the user's latest message into a self-contained search query "
        "that can be sent to a vector database. "
        "Use the chat history solely to resolve pronouns and references — "
        "never to answer, evaluate, or judge the question.\n\n"
        "Rules:\n"
        "1. Output a standalone question or keyword phrase — nothing else.\n"
        "2. Resolve pronouns and references from history "
        "(e.g. 'it' → the specific topic discussed).\n"
        "3. Do NOT answer the question. Do NOT say whether information exists or not.\n"
        "4. Do NOT add information not needed to resolve an ambiguous reference.\n"
        "5. Keep the output short — one sentence or a keyword phrase, maximum 30 words.\n\n"
        "Examples:\n"
        "History: [user: tell me about Linux, assistant: Linux is an open-source OS...]\n"
        "Query: 'any other worthwhile OS you like to mention?'\n"
        "Output: 'other notable operating systems worth mentioning'\n\n"
        "History: [user: summarise assignment 1, assistant: ...summary...]\n"
        "Query: 'what is question 1'\n"
        "Output: 'What is Question 1 in Assignment 1?'\n\n"
        "History: [user: tell me about the StreamVC paper]\n"
        "Query: 'what model does it use'\n"
        "Output: 'What model architecture does StreamVC use?'"
    )
    messages: list[dict] = [{"role": "system", "content": system_msg}]
    for m in recent_history:
        if isinstance(m, HumanMessage):
            messages.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            # Truncate long AI responses to avoid flooding the rewrite context
            messages.append({"role": "assistant", "content": m.content[:400]})
    messages.append({"role": "user", "content": query})

    from openai import AsyncOpenAI as _OAI
    client = _OAI(api_key=settings.OPENAI_API_KEY, base_url=api_base or settings.OPENAI_API_BASE)
    resp = await client.chat.completions.create(
        model=query_model or settings.effective_query_model,
        messages=messages,
        max_tokens=60,
        temperature=0,
        stream=False,
        extra_body={"thinking": {"type": "disabled"}},  # Qwen3: suppress reasoning preamble
    )
    raw_rewrite = (resp.choices[0].message.content or "").strip()

    had_think = bool(strip_reasoning_tags(raw_rewrite) != raw_rewrite)
    standalone = _strip_think(raw_rewrite) or query

    # Strip meta-commentary preamble that some models emit before the actual rewrite
    # e.g. "The user is asking me to rewrite... Here is the rewritten query: ..."
    # The actual rewritten query is always the last sentence / after the last colon.
    if re.search(r"\buser\b.*\brewrite\b|\brewritten\b|\bstandalone\b", standalone, re.IGNORECASE):
        # Take everything after the last colon if present, else last sentence
        if ":" in standalone:
            candidate = standalone.rsplit(":", 1)[-1].strip()
        else:
            sentences = re.split(r"(?<=[.?!])\s+", standalone)
            candidate = sentences[-1].strip()
        if len(candidate) > 5:
            standalone = candidate

    logger.info("[STEP 1] raw_rewrite=%r | had_think=%s | standalone=%r",
                raw_rewrite[:300], had_think, standalone)

    # Guard: if the rewriter echoed the previous assistant response instead of
    # rewriting the query, fall back to the original.  Answer-like patterns are
    # a clear sign the model ignored the "do not answer" instruction.
    answer_patterns = [
        r"\bthere\s+is\s+no\s+information\b",
        r"\bthe\s+text\s+only\s+discusses\b",
        r"\bthe\s+context\s+does?\s+not\s+contain\b",
        r"\bthe\s+provided\s+context\s+does?\s+not\b",
        r"\bi\s+cannot\s+answer\b",
        r"\bi\s+don't\s+have\s+enough\b",
        r"\bno\s+information\s+found\b",
    ]
    if any(re.search(p, standalone, re.IGNORECASE) for p in answer_patterns):
        logger.warning("[STEP 1] rewriter echoed assistant response — falling back to original query")
        standalone = query

    return standalone


# ── Query classification ──────────────────────────────────────────────────────

async def classify_query(query: str, api_base: Optional[str] = None, query_model: Optional[str] = None) -> "QueryClassification":
    """
    Classify a query into one of 4 types using LLM-based zero-shot classification.
    
    Returns QueryClassification with type, confidence, latency_ms, and fallback flag.
    On any failure, returns FACTUAL with fallback=True (safe default).
    """
    from app.schemas.chat import QueryType, QueryClassification
    
    start = time.perf_counter()
    
    try:
        if not settings.QUERY_CLASSIFIER_ENABLED:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info("[CLASSIFY] disabled | latency=%.1fms query=%s", elapsed_ms, query[:80])
            return QueryClassification(
                type=QueryType.FACTUAL, confidence=0.0,
                latency_ms=elapsed_ms, fallback=True
            )
        
        client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=api_base or settings.OPENAI_API_BASE,
        )
        
        prompt = settings.QUERY_CLASSIFIER_PROMPT.format(query=query)
        
        response = await client.chat.completions.create(
            model=query_model or settings.effective_query_model,
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



# ── Main entry point ──────────────────────────────────────────────────────────

async def generate_response(
    query: str,
    messages: dict,
    knowledge_base_ids: List[int],
    chat_id: int,
    db: Session,
    use_dense:     bool = True,
    use_sparse:    bool = True,
    use_exact:     bool = True,
    use_graph_rag: bool = False,
    temperature:   float = 0.0,
    model_name:    Optional[str] = None,
    display_query: Optional[str] = None,
    file_id: Optional[int] = None,
    file_markdown: Optional[str] = None,
    answering_mode: str = "agentic",
    api_base: Optional[str] = None,
    query_model: Optional[str] = None,
    org_id: Optional[int] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream a chat response for the given query.

    Delegates the full RAG pipeline (query rewrite → routing → retrieval →
    grading → generation) to rag_graph.run_stream(), which emits typed
    events that are forwarded as Vercel AI SDK SSE frames:

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
    if api_base is not None:
        logger.info("[CHAT] api_base=%s", api_base)

    try:
        # ── Persist user message ───────────────────────────────────────────
        user_message = Message(content=display_query or query, role="user", chat_id=chat_id)
        db.add(user_message)
        db.commit()

        # Link chat_file to the user message so UI can show the filename
        if file_id:
            from app.models.chat import ChatFile
            chat_file = db.query(ChatFile).filter(ChatFile.id == file_id).first()
            if chat_file:
                chat_file.message_id = user_message.id
                db.commit()

        # ── Persist bot placeholder ────────────────────────────────────────
        bot_message = Message(content="", role="assistant", chat_id=chat_id)
        db.add(bot_message)
        db.commit()

        # ── Identity shortcut ──────────────────────────────────────────────
        if _is_identity_question(query):
            logger.info("[CHAT] identity shortcut — skipping RAG")
            yield f'0:{json.dumps(_IDENTITY_RESPONSE)}\n'
            yield f'd:{{"finishReason":"stop","usage":{{"promptTokens":0,"completionTokens":0}},"messageId":{bot_message.id}}}\n'
            bot_message.content = _IDENTITY_RESPONSE
            db.commit()
            return

        # ── Check knowledge bases ──────────────────────────────────────────
        knowledge_bases = (
            db.query(KnowledgeBase)
            .filter(KnowledgeBase.id.in_(knowledge_base_ids))
            .all()
        )
        if not knowledge_bases:
            error_msg = "I don't have any knowledge base to help answer your question."
            yield f'0:"{error_msg}"\n'
            yield f'd:{{"finishReason":"stop","usage":{{"promptTokens":0,"completionTokens":0}},"messageId":{bot_message.id}}}\n'
            bot_message.content = error_msg
            db.commit()
            return

        # ── Load chat-level state (existing summary) ───────────────────────
        chat = db.query(Chat).filter(Chat.id == chat_id).first()
        existing_summary: str | None = chat.history_summary if chat else None

        # ── Build sliding window from prior messages ───────────────────────
        prior_messages = messages["messages"][:-1]
        window_messages = prior_messages[-_SLIDING_WINDOW_MESSAGES:]

        recent_lc_history = []
        for m in window_messages:
            if m["role"] == "user":
                recent_lc_history.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                content = m["content"]
                if "__LLM_RESPONSE__" in content:
                    content = content.split("__LLM_RESPONSE__")[-1]
                recent_lc_history.append(AIMessage(content=content))

        logger.info("[CHAT] sliding_window=%d msgs | has_summary=%s | mode=%s",
                    len(window_messages), bool(existing_summary), answering_mode)

        full_response = ""
        rewritten_q = display_query or query

        # Confidence capture from the context event
        _confidence_level: str | None = None
        _confidence_score: int | None = None
        _confidence_breakdown: dict | None = None
        _confidence_suggestion: str | None = None

        # ── Route to answering pipeline based on mode ──────────────────────
        if answering_mode in ("fast", "thinking"):
            from app.services.fast_pipeline import fast_stream
            stream_iter = fast_stream(
                query=query,
                knowledge_base_ids=knowledge_base_ids,
                db=db,
                recent_lc_history=recent_lc_history,
                existing_summary=existing_summary,
                use_dense=use_dense,
                use_sparse=use_sparse,
                use_exact=use_exact,
                use_graph_rag=use_graph_rag,
                temperature=temperature,
                model_name=model_name or (
                    settings.effective_reasoning_model if answering_mode == "thinking"
                    else settings.OPENAI_MODEL
                ),
                display_query=display_query,
                file_markdown=file_markdown,
                api_base=api_base,
                query_model=query_model,
                org_id=org_id,
                chat_id=chat_id,
            )
        else:
            # Agentic: full LangGraph pipeline
            from app.services.rag_graph import run_stream
            stream_iter = run_stream(
                query=query,
                file_markdown=file_markdown,
                db=db,
                chat_id=chat_id,
                knowledge_base_ids=knowledge_base_ids,
                recent_lc_history=recent_lc_history,
                existing_summary=existing_summary,
                use_dense=use_dense,
                use_sparse=use_sparse,
                use_exact=use_exact,
                use_graph_rag=use_graph_rag,
                temperature=temperature,
                model_name=model_name,
                display_query=display_query,
                api_base=api_base,
                query_model=query_model,
                org_id=org_id,
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

            elif event_type == "rewritten_query":
                rewritten_q = event.get("query", query)
                bot_message.rewritten_query = rewritten_q
                yield f'1:{json.dumps({"rewritten_query": rewritten_q})}\n'

            elif event_type == "context":
                # Capture confidence data for persistence (written after stream)
                _confidence_level = event.get("confidence")
                _confidence_score = event.get("score")
                _confidence_breakdown = event.get("breakdown")
                _confidence_suggestion = event.get("suggestion")
                _confidence_failed_legs = event.get("failed_legs")

                # Store retrieved docs as citations in the new message_citations table
                raw_docs = event.get("docs", [])
                # Serialize LangChain Document objects to plain dicts
                docs = [
                    {"page_content": d.page_content, "metadata": d.metadata}
                    if hasattr(d, "page_content") else d
                    for d in raw_docs
                ]
                # Persist each doc as a MessageCitation row
                for idx, doc in enumerate(docs, start=1):
                    # Retrieval metadata has document_id and chunk_index
                    document_id = doc.get("metadata", {}).get("document_id")
                    chunk_index = doc.get("metadata", {}).get("chunk_index")
                    if document_id is not None and chunk_index is not None:
                        meta = {**(doc.get("metadata", {}) or {})}
                        # Only store ranking fields if they actually exist
                        for rk in ("score", "dense_rank", "sparse_rank", "exact_rank", "retrieval_leg"):
                            v = doc.get(rk)
                            if v is not None:
                                meta[rk] = v
                        citation = MessageCitation(
                            message_id=bot_message.id,
                            document_id=document_id,
                            chunk_index=chunk_index,
                            citation_index=idx,
                            citation_metadata=meta,
                        )
                        db.add(citation)
                db.flush()

                # Build context payload for streaming (unchanged)
                context_payload = {
                    k: (
                        [{"page_content": d.page_content, "metadata": d.metadata} if hasattr(d, "page_content") else d for d in v]
                        if k == "docs" else v
                    )
                    for k, v in event.items() if k != "event"
                }
                yield f'2:{json.dumps(context_payload)}\n'

            elif event_type == "token":
                content = event.get("content", "")
                full_response += content
                yield f'0:{json.dumps(content)}\n'

            elif event_type == "answer_rewrite":
                # Citation normalisation: replace accumulated streamed text with
                # the citation-linked version. Frontend handles this via event type 'r'.
                full_response = event.get("content", full_response)
                yield f'r:{json.dumps({"content": full_response})}\n'

            elif event_type == "done":
                usage = event.get("usage", {"promptTokens": 0, "completionTokens": 0})
                yield f'd:{json.dumps({"finishReason": "stop", "usage": usage, "messageId": bot_message.id})}\n'

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

        # Persist retrieval confidence for this message
        if _confidence_level is not None:
            bot_message.confidence_level = _confidence_level
        if _confidence_score is not None:
            bot_message.confidence_score = _confidence_score
        if _confidence_breakdown is not None:
            bot_message.confidence_breakdown = json.dumps(_confidence_breakdown)

        db.commit()
        logger.info(
            "[CHAT] confidence persisted | chat_id=%d | level=%s score=%d",
            chat_id, _confidence_level, _confidence_score,
        )
        clear_cancel_token(chat_id)

        # ── Post-turn: schedule summary update (fire-and-forget) ──────────
        def _log_task_error(task: asyncio.Task) -> None:
            exc = task.exception()
            if exc:
                logger.error("[SUMMARY] background task raised: %s", exc)

        task = asyncio.create_task(
            _maybe_update_summary(
                chat_id=chat_id,
                all_prior_messages=prior_messages,
                existing_summary=existing_summary,
            )
        )
        task.add_done_callback(_log_task_error)

    except Exception as e:
        error_message = f"Error generating response: {str(e)}"
        logger.error(error_message)
        yield f'3:{json.dumps(error_message)}\n'
        yield f'd:{{"finishReason":"error","messageId":{bot_message.id}}}\n'
        if 'bot_message' in locals():
            bot_message.content = error_message
            db.commit()
        clear_cancel_token(chat_id)
    finally:
        try:
            db.close()
        except Exception:
            pass  # session may already be closed by caller or during cancellation
