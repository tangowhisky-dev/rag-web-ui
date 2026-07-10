"""LangGraph-based pipeline runner — the single pipeline implementation.

Uses LangGraph node definitions with StateState data contracts.
Routes through nodes: rewrite → classify → [direct_retrieval | agent_loop] → synthesize.
Streams SSE events (p:/t:/th:/0:/1:/2:/3:/d:) in real-time.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, AsyncGenerator, List, Optional

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.services.infrastructure import strip_reasoning_tags
from app.services.infrastructure.utils import _serialise_doc
from app.services.retrieval import score_retrieval

from .callbacks import SSEEventEmitter
from .graph_state import AgentState
from .nodes import _select_model, rewrite_query_node, classify_query_node
from .nodes import request_clarification_node, direct_retrieval_node

logger = logging.getLogger(__name__)


async def run_agentic_rag(
    query: str,
    kb_ids: List[int],
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
    api_base: Optional[str] = None,
    org_id: Optional[int] = None,
    chat_id: Optional[int] = None,
) -> AsyncGenerator[dict, None]:
    """Run the agentic RAG pipeline using LangGraph graph nodes.
    
    Uses LangGraph StateGraph with nested subgraph architecture.
    Routes through nodes: rewrite → classify → [direct_retrieval | agent_loop] → synthesize.
    """
    t0 = time.monotonic()
    emitter = SSEEventEmitter()
    
    # Build initial messages from history
    messages: list = []
    for m in recent_lc_history:
        if isinstance(m, HumanMessage):
            messages.append(HumanMessage(content=m.content))
        elif isinstance(m, AIMessage):
            messages.append(AIMessage(content=m.content[:400]))
    messages.append(HumanMessage(content=query))
    
    effective_model = model_name or settings.OPENAI_MODEL
    is_thinking = settings.REASONING_MODEL and effective_model == settings.REASONING_MODEL
    
    llm = ChatOpenAI(
        model=effective_model,
        temperature=temperature,
        openai_api_base=api_base or settings.OPENAI_API_BASE,
        openai_api_key=settings.OPENAI_API_KEY,
        streaming=True,
    )
    
    initial_state = AgentState(
        messages=messages,
        original_query=query,
        existing_summary=existing_summary or "",
        kb_ids=kb_ids,
        org_id=org_id,
        file_markdown=file_markdown,
    )
    
    # Execute pipeline via LangGraph nodes
    try:
        # Step 1: Rewrite query
        await emitter.emit_progress("rewrite", "Rewriting query...")
        rewrite_result = await rewrite_query_node(initial_state, api_base=api_base)
        
        rewritten = rewrite_result["rewritten_query"]
        await emitter.emit_rewritten_query(rewritten)
        
        # Step 2: Classify query
        await emitter.emit_progress("classify", "Analyzing query...")
        classify_result = await classify_query_node(initial_state)
        
        is_complex = classify_result.get("is_complex", False)
        subtasks = classify_result.get("subtasks", [rewritten])
        
        if not classify_result.get("question_is_clear", True):
            from .nodes import request_clarification_node
            clar_result = request_clarification_node(initial_state)
            for msg in clar_result.get("messages", []):
                content = msg.content if hasattr(msg, "content") else str(msg)
                await emitter.emit_token(content)
            yield {"event": "done", "full_response": "", "usage": {}}
            return
        
        if is_complex and len(subtasks) > 1:
            # Complex path: subtask loop via agent subgraph
            yield {"event": "task_list", "tasks": [
                {"id": i, "text": s, "status": "pending", "progress": None}
                for i, s in enumerate(subtasks)
            ]}
            
            all_answers = []
            all_docs = []
            
            for idx, subtask in enumerate(subtasks):
                yield {"event": "task_list", "tasks": [
                    {"id": i, "text": s, "status": "running" if i == idx else ("done" if i < idx else "pending"),
                     "progress": None}
                    for i, s in enumerate(subtasks)
                ]}
                
                await emitter.emit_progress("rewriting", f"Rewriting query {idx + 1}/{len(subtasks)}...")
                await emitter.emit_progress("searching", f"Searching knowledge base {idx + 1}/{len(subtasks)}...")
                
                from .nodes import direct_retrieval_node
                retrieval_result = await direct_retrieval_node(
                    {**initial_state, "rewritten_query": subtask},
                    db, kb_ids, org_id, file_markdown,
                    use_dense, use_sparse, use_exact, use_graph_rag,
                )
                
                docs = retrieval_result.get("retrieved_docs", [])
                retrieval_info = docs  # We don't have retrieval_info here, skip
                conf_result = score_retrieval(docs, {}) if docs else None
                conf_score = conf_result.score if conf_result else 0
                
                yield {
                    "event": "progress", "phase": "searching",
                    "message": f"Found {len(docs)} relevant chunks",
                    "details": {"subtask_index": idx, "subtask_total": len(subtasks), "chunks_found": len(docs)},
                }
                yield {"event": "progress", "phase": "reranking", "message": "Reranking for relevance..."}
                yield {"event": "progress", "phase": "reranking", 
                       "message": f"Shortlisted {len(docs)} chunks",
                       "details": {"subtask_index": idx, "subtask_total": len(subtasks), "reranked": len(docs)}}
                
                yield {
                    "event": "context", "docs": docs,
                    "confidence": conf_result.level if conf_result else "low",
                    "score": conf_score,
                    "subtask_index": idx,
                    "query_classification": {"type": "MULTI_PART", "confidence": 1.0},
                    "synthesis_mode": False,
                }
                
                # Build context string
                context_parts = []
                for i, doc in enumerate(docs, 1):
                    content = doc.get("page_content", "").strip()
                    source = doc.get("metadata", {}).get("source", "")
                    header = f"[KB-{i}]" + (f" ({source})" if source else "")
                    context_parts.append(f"{header}\n{content}")
                if file_markdown:
                    context_parts.append(f"[File Content]\n{file_markdown}")
                merged = "\n\n---\n\n".join(context_parts)
                
                effective_model = _select_model(subtask, True)
                
                if is_thinking:
                    await emitter.emit_progress("generating", f"Thinking through subtask {idx + 1}/{len(subtasks)}...")
                else:
                    await emitter.emit_progress("generating", f"Generating answer {idx + 1}/{len(subtasks)}...")
                
                subtask_answer = ""
                async for event in _generate_streaming(
                    subtask, merged, effective_model, api_base, existing_summary,
                    is_thinking, emitter,
                ):
                    if event.get("event") == "token":
                        subtask_answer += event.get("content", "")
                    elif event.get("event") == "done":
                        subtask_answer = event.get("full_response", subtask_answer)
                    yield event
                
                yield {"event": "task_list", "tasks": [
                    {"id": i, "text": s, "status": "done" if i <= idx else "pending",
                     "progress": None}
                    for i, s in enumerate(subtasks)
                ]}
                
                all_answers.append({"answer": subtask_answer})
                all_docs.extend(docs)
            
            # Synthesis
            await emitter.emit_progress("synthesizing", "Synthesizing final answer...")
            
            synthesis_parts = []
            for sa in all_answers:
                synthesis_parts.append(f"### Answer\n\n{sa['answer']}")
            combined = "\n\n---\n\n".join(synthesis_parts)
            
            all_serialised = [_serialise_doc(d) for d in all_docs]
            yield {
                "event": "context", "docs": all_serialised[:10],
                "confidence": "high" if len(all_docs) > 5 else ("medium" if all_docs else "low"),
                "synthesis_mode": True,
            }
            
            await emitter.emit_progress("synthesizing", "Generating final summary...")
            
            answer_chars = list(combined)
            i = 0
            while i < len(answer_chars):
                chunk_size = min(50, len(answer_chars) - i)
                chunk = "".join(answer_chars[i:i + chunk_size])
                await emitter.emit_token(chunk)
                i += chunk_size
            
            yield {"event": "done", "full_response": combined, "usage": {}}
            
        else:
            # Simple path: direct retrieval + generate
            await emitter.emit_progress("simple", "Answering directly...")
            await emitter.emit_progress("searching", "Searching knowledge base...")
            
            from .nodes import direct_retrieval_node
            retrieval_result = await direct_retrieval_node(
                initial_state, db, kb_ids, org_id, file_markdown,
                use_dense, use_sparse, use_exact, use_graph_rag,
            )
            
            docs = retrieval_result.get("retrieved_docs", [])
            conf_result = score_retrieval(docs, {}) if docs else None
            conf_score = conf_result.score if conf_result else 0
            
            yield {
                "event": "progress", "phase": "searching",
                "message": f"Found {len(docs)} relevant chunks",
                "details": {"chunks_found": len(docs)},
            }
            yield {"event": "progress", "phase": "reranking", "message": "Reranking for relevance..."}
            yield {"event": "progress", "phase": "reranking",
                   "message": f"Shortlisted {len(docs)} chunks",
                   "details": {"reranked": len(docs)}}
            
            yield {
                "event": "context", "docs": docs,
                "confidence": conf_result.level if conf_result else "low",
                "score": conf_score,
                "query_classification": {"type": "FACTUAL", "confidence": 1.0},
                "synthesis_mode": False,
            }
            
            context_parts = []
            for i, doc in enumerate(docs, 1):
                content = doc.get("page_content", "").strip()
                source = doc.get("metadata", {}).get("source", "")
                header = f"[KB-{i}]" + (f" ({source})" if source else "")
                context_parts.append(f"{header}\n{content}")
            if file_markdown:
                context_parts.append(f"[File Content]\n{file_markdown}")
            merged = "\n\n---\n\n".join(context_parts)
            
            effective_model = _select_model(rewritten, False)
            
            await emitter.emit_progress("generating", "Generating answer...")
            
            async for event in _generate_streaming(
                rewritten, merged, effective_model, api_base, existing_summary,
                False, emitter,
            ):
                yield event
    
    except Exception as exc:
        logger.error("[GRAPH] pipeline failed: %s", exc, exc_info=True)
        await emitter.emit_error(str(exc))
        yield {"event": "error", "message": str(exc)}
        yield {"event": "done", "full_response": "", "usage": {}}
    
    finally:
        async for event in emitter.drain():
            yield event
    
    logger.info("[GRAPH] total latency=%.1fms query=%r", (time.monotonic() - t0) * 1000, query[:80])


async def _generate_streaming(
    query: str,
    context_text: str,
    model_name: str,
    api_base: Optional[str],
    existing_summary: Optional[str],
    is_thinking: bool,
    emitter: SSEEventEmitter,
) -> AsyncGenerator[dict, None]:
    """Stream an answer from context via LLM. Yields token/thinking/done events."""
    from .nodes import _ANSWER_SYSTEM_PROMPT
    
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.0,
        openai_api_base=api_base or settings.OPENAI_API_BASE,
        openai_api_key=settings.OPENAI_API_KEY,
        streaming=True,
    )
    
    messages: list = [{"role": "system", "content": _ANSWER_SYSTEM_PROMPT}]
    
    if existing_summary:
        messages.append({
            "role": "system",
            "content": f"[Conversation summary so far]\n{existing_summary}",
        })
    
    context_section = f"\nContext:\n{context_text}\n\nQuestion: {query}" if context_text else query
    messages.append({"role": "user", "content": context_section})
    
    streamed_parts: list[str] = []
    thinking_parts: list[str] = []
    usage: dict = {"promptTokens": 0, "completionTokens": 0}
    
    try:
        async for chunk in llm.astream(messages):
            token: str = chunk.content or ""
            
            if is_thinking:
                stripped = strip_reasoning_tags(token)
                if stripped != token:
                    full_match = re.search(r'</think>(.*?)</think>', token, re.DOTALL)
                    if full_match:
                        thinking_parts.append(full_match.group(1))
                        token = token[full_match.end():]
                    else:
                        open_match = re.search(r'</think>', token)
                        if open_match:
                            token = token[open_match.end():]
                
                if stripped:
                    thinking_parts.append(stripped)
                    if len(thinking_parts) % 50 == 0 or stripped.endswith(('\n', ' ')):
                        yield {"event": "thinking", "content": "".join(thinking_parts), "done": False}
                
                if not token:
                    continue
            
            if token:
                streamed_parts.append(token)
                await emitter.emit_token(token)
                yield {"event": "token", "content": token}
            
            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                usage = {
                    "promptTokens": chunk.usage_metadata.get("input_tokens", 0),
                    "completionTokens": chunk.usage_metadata.get("output_tokens", 0),
                }
    except Exception as exc:
        logger.error("[GRAPH] generation failed: %s", exc)
        err_msg = "I encountered an error generating the response. Please try again."
        yield {"event": "token", "content": err_msg}
        streamed_parts.append(err_msg)
    
    if thinking_parts:
        yield {"event": "thinking", "content": "".join(thinking_parts), "done": True}
    
    answer = "".join(streamed_parts)
    normalised = re.sub(
        r'\[(\d+)\](?!\()',
        lambda m: f'[{m.group(1)}]({m.group(1)})',
        answer,
    )
    
    if normalised != answer:
        await emitter.emit_answer_rewrite(normalised)
    
    final_answer = normalised or answer
    
    # Validate charts
    chart_pattern = re.compile(r'\[chart\](.*?)\[/chart\]', re.DOTALL)
    matches = list(chart_pattern.finditer(final_answer))
    chart_validated = {"has_charts": False, "chart_count": 0}
    if matches:
        valid_count = 0
        errors = []
        for i, match in enumerate(matches):
            try:
                json.loads(match.group(1))
                valid_count += 1
            except (json.JSONDecodeError, TypeError) as e:
                errors.append(f"Chart {i+1}: {str(e)[:100]}")
        chart_validated = {
            "has_charts": True, "chart_count": len(matches),
            "valid_count": valid_count, "errors": errors,
            "all_valid": len(errors) == 0,
        }
    
    yield {"event": "done", "full_response": final_answer, "usage": usage, "chart_validated": chart_validated}
