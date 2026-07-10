"""Single autonomous agentic agent - rewrite, search, stream, review.

Replaces the old supervisor -> worker -> critic loop.  This agent:

1. Rewrites the query using chat history
2. Decides simple vs complex (heuristic, no LLM call)
3. For simple queries: direct answer with streaming
4. For complex queries: decompose -> iterate subtasks -> stream per subtask -> synthesize
5. Streams everything in real-time (tokens, progress, thinking traces)
6. Lightweight post-review only - never blocks streaming

Event protocol (SSE prefixes):
  p:  progress       - transient status messages
  t:  task_list      - subtask list with status
  th: thinking       - reasoning model chain-of-thought
  0:  token          - streaming answer text
  1:  rewritten_query - standalone query (internal, not in response)
  2:  context        - retrieved documents
  3:  error          - exception message
  d:  done           - finish reason + usage
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, AsyncGenerator, List, Optional

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.services.retrieval import hybrid_search_with_legs, get_effective_datastore_ids
from app.services.retrieval import score_retrieval
from app.services.infrastructure import strip_reasoning_tags
from app.services.prompts.loader import append_chart_instructions

from .prompts import _ANSWER_SYSTEM_PROMPT, _THINKING_KEYWORDS

logger = logging.getLogger(__name__)

# Apply chart instructions to system prompt
_ANSWER_SYSTEM_PROMPT = append_chart_instructions(_ANSWER_SYSTEM_PROMPT)

_PREVIEW_CHARS = 120


def _preview(text: str, n: int = _PREVIEW_CHARS) -> str:
    text = text.strip().replace("\n", " ")
    return text[:n] + "\u2026" if len(text) > n else text


# Helpers

def _is_complex_query(query: str, rewritten: str) -> bool:
    """Heuristic: does this query likely need subtask decomposition?"""
    combined = (query + " " + rewritten).lower()
    # Multi-part indicators
    multi = re.search(
        r"\b(and|or|but|yet|also|plus|along with|as well as|in addition)\b.*"
        r"\b(and|or|but|yet|also|plus|along with|as well as|in addition)\b",
        combined,
    )
    if multi:
        return True
    # Count distinct question-like phrases
    questions = re.findall(r"\b(what|how|why|when|where|which|compare|list|explain)\b", combined)
    if len(questions) >= 3:
        return True
    # Length heuristic - very long queries likely cover multiple topics
    if len(rewritten.split()) > 30:
        return True
    return False


async def _classify_complexity(
    query: str,
    rewritten: str,
    history: list,
) -> dict:
    """Decide if query is simple or complex. Returns (complex, subtasks)."""
    complex = _is_complex_query(query, rewritten)

    if not complex:
        return {"complex": False, "subtasks": None}

    # For complex queries, use LLM to decompose
    try:
        from openai import AsyncOpenAI as _OAI
        client = _OAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE,
        )

        system_prompt = (
            "You are a query decomposer. Break the user's complex question into "
            "2-5 focused subtasks. Each subtask should be a clear, standalone question "
            "that can be answered by searching a knowledge base.\n\n"
            "Rules:\n"
            "- Return EXACTLY 2-5 subtasks (fewer if the query is simple enough).\n"
            "- Each subtask should cover a distinct aspect of the original query.\n"
            "- Subtasks should be concise (max 20 words each).\n"
            "- Do NOT overlap - each subtask addresses a unique part.\n"
            "- If the query is actually simple, return just 1 subtask.\n\n"
            "Output format: a JSON array of strings, nothing else.\n\n"
            "Example:\n"
            "Input: 'Compare the PCB structure in Linux vs Windows and explain how context switching works'\n"
            'Output: ["What is the PCB structure in Linux?", "What is the PCB structure in Windows?", "How does context switching work in operating systems?"]\n\n'
            "Input: 'What is photosynthesis?'\n"
            'Output: ["What is photosynthesis?"]'
        )

        messages = [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": rewritten}]

        resp = await client.chat.completions.create(
            model=settings.effective_query_model,
            messages=messages,
            max_tokens=200,
            temperature=0,
            stream=False,
            extra_body={"thinking": {"type": "disabled"}},
        )
        raw = (resp.choices[0].message.content or "[]").strip()
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            subtasks = json.loads(m.group(0))
            if isinstance(subtasks, list) and len(subtasks) > 0:
                return {"complex": True, "subtasks": subtasks}
    except Exception as e:
        logger.warning("[AGENT] complexity decomposition failed: %s - using default", e)

    # Fallback: single subtask
    return {"complex": True, "subtasks": [rewritten]}


def _select_model(subtask_text: str, is_complex: bool) -> str:
    """Auto-select model based on query nature."""
    lower = subtask_text.lower()
    is_thinking = any(kw in lower for kw in _THINKING_KEYWORDS)
    if is_thinking or is_complex:
        return settings.REASONING_MODEL or settings.OPENAI_MODEL
    return settings.OPENAI_MODEL


from app.services.infrastructure import _serialise_doc


def _get_llm(model_name: Optional[str] = None, temperature: float = 0.0, api_base: Optional[str] = None):
    return ChatOpenAI(
        model=model_name or settings.OPENAI_MODEL,
        temperature=temperature,
        openai_api_base=api_base or settings.OPENAI_API_BASE,
        openai_api_key=settings.OPENAI_API_KEY,
        streaming=True,
    )


# Core pipeline: search + rerank

async def _search_and_rerank(
    query: str,
    kb_ids: List[int],
    db: Any,
    use_dense: bool = True,
    use_sparse: bool = True,
    use_exact: bool = True,
    use_graph_rag: bool = False,
    org_id: Optional[int] = None,
    file_markdown: Optional[str] = None,
) -> tuple:
    """Run hybrid search + reranking. Returns (docs, retrieval_info, failed_legs)."""
    datastore_ids = get_effective_datastore_ids(kb_ids, org_id, db)

    retrieval_result = await hybrid_search_with_legs(
        query=query,
        kb_ids=kb_ids,
        db=db,
        use_dense=use_dense,
        use_sparse=use_sparse,
        use_exact=use_exact,
        use_graph_rag=use_graph_rag,
        datastore_ids=datastore_ids,
        return_full_pool=True,
    )

    docs = retrieval_result.get("docs", [])
    retrieval_info = retrieval_result.get("retrieval_info", {})
    failed_legs = retrieval_info.get("failed_legs", [])
    return docs, retrieval_info, failed_legs


# Core pipeline: generate answer

async def _generate_answer(
    query: str,
    context_text: str,
    model_name: str,
    api_base: Optional[str] = None,
    existing_summary: Optional[str] = None,
    is_thinking: bool = False,
) -> AsyncGenerator[dict, None]:
    """Stream an answer from context. Yields token/thinking/done events."""
    messages: list = [{"role": "system", "content": _ANSWER_SYSTEM_PROMPT}]

    if existing_summary:
        messages.append({
            "role": "system",
            "content": f"[Conversation summary so far]\n{existing_summary}",
        })

    context_section = f"\nContext:\n{context_text}\n\nQuestion: {query}" if context_text else query
    messages.append({"role": "user", "content": context_section})

    llm = _get_llm(model_name, 0.0, api_base=api_base)
    streamed_parts: list[str] = []
    thinking_parts: list[str] = []
    thinking_active = False
    usage: dict = {"promptTokens": 0, "completionTokens": 0}

    try:
        async for chunk in llm.astream(messages):
            token: str = chunk.content or ""

            if is_thinking:
                # Detect thinking tag blocks
                stripped = strip_reasoning_tags(token)
                if stripped != token:
                    # Token contains reasoning tags - extract thinking content
                    full_match = re.search(r'</think>(.*?)</think>', token, re.DOTALL)
                    if full_match:
                        thinking_parts.append(full_match.group(1))
                        token = token[full_match.end():]
                    else:
                        # Partial/closed thinking block
                        open_match = re.search(r'</think>', token)
                        if open_match:
                            thinking_active = True
                            token = token[open_match.end():]

                if thinking_active and stripped:
                    thinking_parts.append(stripped)
                    # Emit thinking content periodically
                    if len(thinking_parts) % 50 == 0 or stripped.endswith(('\n', ' ')):
                        yield {
                            "event": "thinking",
                            "content": "".join(thinking_parts),
                            "done": False,
                        }

                if not token and not thinking_active:
                    continue

            if token:
                streamed_parts.append(token)
                yield {"event": "token", "content": token}

            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                usage = {
                    "promptTokens": chunk.usage_metadata.get("input_tokens", 0),
                    "completionTokens": chunk.usage_metadata.get("output_tokens", 0),
                }
    except Exception as exc:
        logger.error("[AGENT] generation failed: %s", exc)
        err_msg = "I encountered an error generating the response. Please try again."
        yield {"event": "token", "content": err_msg}
        streamed_parts.append(err_msg)

    # Emit final thinking block if any
    if thinking_parts:
        yield {
            "event": "thinking",
            "content": "".join(thinking_parts),
            "done": True,
        }

    answer = "".join(streamed_parts)

    # Normalise citation syntax: [2] -> [2](2)
    normalised = re.sub(
        r'\[(\d+)\](?!\()',
        lambda m: f'[{m.group(1)}]({m.group(1)})',
        answer,
    )
    if normalised != answer:
        yield {"event": "answer_rewrite", "content": normalised}

    final_answer = normalised or answer
    yield {"event": "done", "full_response": final_answer, "usage": usage}


# Simple path: direct answer

async def _direct_answer(
    query: str,
    kb_ids: List[int],
    db: Any,
    api_base: Optional[str],
    model_name: Optional[str],
    temperature: float,
    file_markdown: Optional[str],
    existing_summary: Optional[str],
    use_dense: bool,
    use_sparse: bool,
    use_exact: bool,
    use_graph_rag: bool,
    org_id: Optional[int],
    chat_id: Optional[int],
) -> AsyncGenerator[dict, None]:
    """Simple query path: rewrite -> search -> rerank -> stream answer."""
    # 1. Rewrite
    yield {"event": "progress", "phase": "rewriting", "message": "Rewriting query..."}
    rewritten = await _rewrite_query(query, [], api_base=api_base)
    yield {"event": "rewritten_query", "query": rewritten}

    # 2. Search
    yield {"event": "progress", "phase": "searching", "message": "Searching knowledge base..."}
    docs, retrieval_info, failed_legs = await _search_and_rerank(
        rewritten, kb_ids, db, use_dense, use_sparse, use_exact,
        use_graph_rag, org_id, file_markdown,
    )
    yield {"event": "progress", "phase": "searching", "message": f"Found {len(docs)} relevant chunks",
           "details": {"chunks_found": len(docs)}}

    # 3. Rerank - filter by threshold
    yield {"event": "progress", "phase": "reranking", "message": "Reranking for relevance..."}
    threshold = settings.RERANKER_SCORE_THRESHOLD
    filtered = [d for d in docs if d.metadata.get("_reranker_score", -float("inf")) >= threshold]
    filtered.sort(key=lambda d: d.metadata.get("_reranker_score", -float("inf")), reverse=True)
    yield {"event": "progress", "phase": "reranking", "message": f"Shortlisted {len(filtered)} chunks",
           "details": {"reranked": len(filtered)}}

    # 4. Compute confidence and emit context
    conf_result = score_retrieval(filtered, retrieval_info)
    conf_score = conf_result.score

    serialised = [_serialise_doc(d) for d in filtered]
    yield {
        "event": "context",
        "docs": serialised,
        "confidence": conf_result.level,
        "score": conf_score,
        "suggestion": conf_result.suggestion or "",
        "failed_legs": failed_legs,
        "breakdown": conf_result.breakdown,
        "query_classification": {"type": "FACTUAL", "confidence": 1.0, "latency_ms": 0, "fallback": False},
        "tool_trace": ["rewrite_query", "kb_retrieval", "generate_answer"],
        "synthesis_mode": False,
    }

    # 5. Build context string
    context_parts = []
    for i, doc in enumerate(serialised, 1):
        content = doc.get("page_content", "").strip()
        source = doc.get("metadata", {}).get("source", "")
        header = f"[KB-{i}]" + (f" ({source})" if source else "")
        context_parts.append(f"{header}\n{content}")
    if file_markdown:
        context_parts.append(f"[File Content]\n{file_markdown}")
    merged = "\n\n---\n\n".join(context_parts)

    # 6. Select model
    effective_model = model_name or settings.OPENAI_MODEL
    is_thinking = False  # simple path - no thinking model

    # 7. Stream answer
    yield {"event": "progress", "phase": "generating", "message": "Generating answer..."}
    async for event in _generate_answer(
        rewritten, merged, effective_model, api_base,
        existing_summary, is_thinking,
    ):
        yield event


# Complex path: subtask decomposition

async def _complex_answer(
    query: str,
    kb_ids: List[int],
    db: Any,
    api_base: Optional[str],
    model_name: Optional[str],
    temperature: float,
    file_markdown: Optional[str],
    existing_summary: Optional[str],
    use_dense: bool,
    use_sparse: bool,
    use_exact: bool,
    use_graph_rag: bool,
    org_id: Optional[int],
    chat_id: Optional[int],
    subtasks: List[str],
) -> AsyncGenerator[dict, None]:
    """Complex query path: decompose -> iterate subtasks -> synthesize -> review."""
    total = len(subtasks)

    # Emit initial task list
    initial_tasks = [
        {"id": i, "text": s, "status": "pending", "progress": None}
        for i, s in enumerate(subtasks)
    ]
    yield {"event": "task_list", "tasks": initial_tasks}

    all_answers = []
    all_docs = []

    for idx, subtask in enumerate(subtasks):
        # Update task status to running
        yield {"event": "task_list", "tasks": [
            {"id": i, "text": s, "status": "running" if i == idx else ("done" if i < idx else "pending"),
             "progress": None}
            for i, s in enumerate(subtasks)
        ]}

        # Rewrite subtask query
        yield {
            "event": "progress",
            "phase": "rewriting",
            "message": f"Rewriting query {idx + 1}/{total}...",
            "details": {"subtask_index": idx, "subtask_total": total},
        }
        rewritten = await _rewrite_query(subtask, [], api_base=api_base)

        # Search
        yield {
            "event": "progress",
            "phase": "searching",
            "message": f"Searching knowledge base {idx + 1}/{total}...",
            "details": {"subtask_index": idx, "subtask_total": total},
        }
        docs, retrieval_info, failed_legs = await _search_and_rerank(
            rewritten, kb_ids, db, use_dense, use_sparse, use_exact,
            use_graph_rag, org_id, file_markdown,
        )
        yield {
            "event": "progress",
            "phase": "searching",
            "message": f"Found {len(docs)} relevant chunks",
            "details": {"subtask_index": idx, "subtask_total": total, "chunks_found": len(docs)},
        }

        # Rerank - filter by threshold
        yield {
            "event": "progress",
            "phase": "reranking",
            "message": "Reranking for relevance...",
            "details": {"subtask_index": idx, "subtask_total": total},
        }
        threshold = settings.RERANKER_SCORE_THRESHOLD
        filtered = [d for d in docs if d.metadata.get("_reranker_score", -float("inf")) >= threshold]
        filtered.sort(key=lambda d: d.metadata.get("_reranker_score", -float("inf")), reverse=True)
        yield {
            "event": "progress",
            "phase": "reranking",
            "message": f"Shortlisted {len(filtered)} chunks",
            "details": {"subtask_index": idx, "subtask_total": total, "reranked": len(filtered)},
        }

        # Compute confidence and emit context
        conf_result = score_retrieval(filtered, retrieval_info)
        conf_score = conf_result.score

        yield {
            "event": "context",
            "docs": [_serialise_doc(d) for d in filtered],
            "confidence": conf_result.level,
            "score": conf_score,
            "suggestion": conf_result.suggestion or "",
            "failed_legs": failed_legs,
            "subtask_index": idx,
            "breakdown": conf_result.breakdown,
            "query_classification": {"type": "FACTUAL", "confidence": 1.0, "latency_ms": 0, "fallback": False},
            "tool_trace": ["rewrite_query", "kb_retrieval", "generate_answer"],
            "synthesis_mode": False,
        }

        # Build context string
        context_parts = []
        serialised = [_serialise_doc(d) for d in filtered]
        for i, doc in enumerate(serialised, 1):
            content = doc.get("page_content", "").strip()
            source = doc.get("metadata", {}).get("source", "")
            header = f"[KB-{i}]" + (f" ({source})" if source else "")
            context_parts.append(f"{header}\n{content}")
        if file_markdown:
            context_parts.append(f"[File Content]\n{file_markdown}")
        merged = "\n\n---\n\n".join(context_parts)

        # Select model
        effective_model = _select_model(rewritten, len(subtasks) > 1)
        is_thinking = effective_model != settings.OPENAI_MODEL

        # Stream answer for this subtask
        if is_thinking:
            yield {
                "event": "progress",
                "phase": "generating",
                "message": f"Thinking through subtask {idx + 1}/{total}...",
                "details": {"subtask_index": idx, "subtask_total": total, "model": effective_model, "model_type": "thinking"},
            }
        else:
            yield {
                "event": "progress",
                "phase": "generating",
                "message": f"Generating answer {idx + 1}/{total}...",
                "details": {"subtask_index": idx, "subtask_total": total, "model": effective_model, "model_type": "fast"},
            }

        subtask_answer = ""
        async for event in _generate_answer(
            rewritten, merged, effective_model, api_base,
            existing_summary, is_thinking,
        ):
            if event.get("event") == "token":
                subtask_answer += event.get("content", "")
            elif event.get("event") == "done":
                subtask_answer = event.get("full_response", subtask_answer)
            yield event

        # Update task to done
        yield {"event": "task_list", "tasks": [
            {"id": i, "text": s, "status": "done" if i <= idx else "pending",
             "progress": None}
            for i, s in enumerate(subtasks)
        ]}

        all_answers.append((subtask, subtask_answer))
        all_docs.extend(filtered)

    # Final synthesis
    yield {"event": "progress", "phase": "synthesizing", "message": "Synthesizing final answer..."}

    # Build combined answer
    synthesis_parts = []
    for subtask_text, answer_text in all_answers:
        synthesis_parts.append(f"### {subtask_text}\n\n{answer_text}")
    combined = "\n\n---\n\n".join(synthesis_parts)

    # Emit final context
    all_serialised = [_serialise_doc(d) for d in all_docs]
    yield {
        "event": "context",
        "docs": all_serialised[:10],
        "confidence": "high" if len(all_docs) > 5 else ("medium" if all_docs else "low"),
        "total_docs": len(all_docs),
        "breakdown": {
            "iterations": 1,
            "subtasks": len(subtasks),
        },
        "query_classification": {"type": "MULTI_PART", "confidence": 1.0, "latency_ms": 0, "fallback": False},
        "synthesis_mode": True,
    }

    # Stream final combined answer
    yield {"event": "progress", "phase": "synthesizing", "message": "Generating final summary..."}
    answer_chars = list(combined)
    i = 0
    while i < len(answer_chars):
        chunk_size = min(50, len(answer_chars) - i)
        chunk = "".join(answer_chars[i:i + chunk_size])
        yield {"event": "token", "content": chunk}
        i += chunk_size

    yield {"event": "done", "full_response": combined, "usage": {}}


# Query rewrite (reused from chat_service)

async def _rewrite_query(
    query: str,
    recent_history: list,
    api_base: Optional[str] = None,
) -> str:
    """Condense query into a standalone question for retrieval."""
    if not recent_history:
        return query

    system_msg = (
        "You are a search query rewriter for a document retrieval system. "
        "Your ONLY job is to rewrite the user's latest message into a self-contained search query "
        "that can be sent to a vector database. "
        "Use the chat history solely to resolve pronouns and references - "
        "never to answer, evaluate, or judge the question.\n\n"
        "Rules:\n"
        "1. Output a standalone question or keyword phrase - nothing else.\n"
        "2. Resolve pronouns and references from history.\n"
        "3. Do NOT answer the question.\n"
        "4. Keep the output short - one sentence or a keyword phrase, maximum 30 words.\n\n"
        "Examples:\n"
        "History: [user: tell me about Linux, assistant: Linux is an open-source OS...]\n"
        "Query: 'any other worthwhile OS you like to mention?'\n"
        "Output: 'other notable operating systems worth mentioning'\n\n"
        "History: [user: tell me about the StreamVC paper]\n"
        "Query: 'what model does it use'\n"
        "Output: 'What model architecture does StreamVC use?'"
    )

    messages: list[dict] = [{"role": "system", "content": system_msg}]
    for m in recent_history:
        if isinstance(m, HumanMessage):
            messages.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            messages.append({"role": "assistant", "content": m.content[:400]})
    messages.append({"role": "user", "content": query})

    from openai import AsyncOpenAI as _OAI
    client = _OAI(api_key=settings.OPENAI_API_KEY, base_url=api_base or settings.OPENAI_API_BASE)
    resp = await client.chat.completions.create(
        model=settings.effective_query_model,
        messages=messages,
        max_tokens=60,
        temperature=0,
        stream=False,
        extra_body={"thinking": {"type": "disabled"}},
    )
    raw = (resp.choices[0].message.content or "").strip()
    standalone = strip_reasoning_tags(raw) or query

    # Guard: if rewriter echoed assistant response, fall back to original
    answer_patterns = [
        r"\bthere\s+is\s+no\s+information\b",
        r"\bthe\s+context\s+does?\s+not\s+contain\b",
        r"\bi\s+cannot\s+answer\b",
        r"\bi\s+don't\s+have\s+enough\b",
        r"\bno\s+information\s+found\b",
    ]
    if any(re.search(p, standalone, re.IGNORECASE) for p in answer_patterns):
        standalone = query

    return standalone


# Public API

async def run_agentic_rag(
    query: str,
    chat_id: int,
    knowledge_base_ids: List[int],
    db: Any,
    recent_lc_history: list,
    existing_summary: Optional[str] = None,
    file_markdown: Optional[str] = None,
    use_dense: bool = True,
    use_sparse: bool = True,
    use_exact: bool = True,
    use_graph_rag: bool = False,
    temperature: float = 0.0,
    model_name: Optional[str] = None,
    display_query: Optional[str] = None,
    api_base: Optional[str] = None,
    query_model: Optional[str] = None,
    org_id: Optional[int] = None,
) -> AsyncGenerator[dict, None]:
    """
    Single autonomous agentic agent. No supervisor, no workers, no critic gate.

    Streams everything in real-time: tokens, progress, thinking traces.

    Feature flag: USE_LANGGRAPH routes to the LangGraph-based pipeline.
    When disabled (default), uses the existing generator-based pipeline.
    """
    from app.core.config import settings

    if settings.USE_LANGGRAPH:
        logger.info("[AGENT] using LangGraph pipeline")
        from .graph_runner import run_agentic_rag_via_graph
        async for event in run_agentic_rag_via_graph(
            query=query,
            kb_ids=knowledge_base_ids,
            db=db,
            recent_lc_history=recent_lc_history,
            existing_summary=existing_summary,
            file_markdown=file_markdown,
            use_dense=use_dense,
            use_sparse=use_sparse,
            use_exact=use_exact,
            use_graph_rag=use_graph_rag,
            temperature=temperature,
            model_name=model_name,
            api_base=api_base,
            org_id=org_id,
            chat_id=chat_id,
        ):
            yield event
    else:
        t0 = time.monotonic()
        effective_model = model_name or settings.OPENAI_MODEL

        # 1. Rewrite query
        yield {"event": "progress", "phase": "rewriting", "message": "Rewriting query..."}
        rewritten = await _rewrite_query(query, recent_lc_history, api_base=api_base)
        yield {"event": "rewritten_query", "query": rewritten}

        # 2. Decide simple vs complex
        complexity = await _classify_complexity(query, rewritten, recent_lc_history)

        if not complexity["complex"]:
            # Simple path
            yield {"event": "progress", "phase": "simple", "message": "Answering directly..."}
            async for event in _direct_answer(
                query=query,
                kb_ids=knowledge_base_ids,
                db=db,
                api_base=api_base,
                model_name=effective_model,
                temperature=temperature,
                file_markdown=file_markdown,
                existing_summary=existing_summary,
                use_dense=use_dense,
                use_sparse=use_sparse,
                use_exact=use_exact,
                use_graph_rag=use_graph_rag,
                org_id=org_id,
                chat_id=chat_id,
            ):
                yield event
        else:
            # Complex path
            subtasks = complexity["subtasks"]
            yield {
                "event": "progress",
                "phase": "decomposition",
                "message": f"Breaking down into {len(subtasks)} subtasks...",
                "details": {"subtask_total": len(subtasks)},
            }
            async for event in _complex_answer(
                query=query,
                kb_ids=knowledge_base_ids,
                db=db,
                api_base=api_base,
                model_name=effective_model,
                temperature=temperature,
                file_markdown=file_markdown,
                existing_summary=existing_summary,
                use_dense=use_dense,
                use_sparse=use_sparse,
                use_exact=use_exact,
                use_graph_rag=use_graph_rag,
                org_id=org_id,
                chat_id=chat_id,
                subtasks=subtasks,
            ):
                yield event

        logger.info("[AGENT] total latency=%.1fms query=%r", (time.monotonic() - t0) * 1000, query[:80])
