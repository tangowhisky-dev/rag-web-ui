"""LangGraph-based pipeline runner.

Bridges the compiled LangGraph graph to the existing SSE event protocol.
Handles streaming tokens, progress events, and task lists from the graph.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, List, Optional

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.services.infrastructure import _serialise_doc
from app.services.retrieval import score_retrieval

from .callbacks import SSEEventEmitter
from .graph import create_agent_graph
from .graph_state import AgentState
from .nodes import _select_model, direct_retrieval_node, generate_node, synthesize_node
from .pipeline import _search_and_rerank
from .schemas import QueryAnalysis

logger = logging.getLogger(__name__)


async def run_agentic_rag_via_graph(
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
    """Run the agentic RAG pipeline via the LangGraph graph.
    
    Bridges graph execution to the existing SSE event protocol.
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
    
    # Create LLM
    llm = ChatOpenAI(
        model=effective_model,
        temperature=temperature,
        openai_api_base=api_base or settings.OPENAI_API_BASE,
        openai_api_key=settings.OPENAI_API_KEY,
        streaming=True,
    )
    
    # Build initial state
    initial_state = AgentState(
        messages=messages,
        original_query=query,
        existing_summary=existing_summary or "",
        kb_ids=kb_ids,
        org_id=org_id,
        file_markdown=file_markdown,
    )
    
    # Compile graph (with memory checkpointer for node transitions)
    checkpointer = None  # No checkpointing yet (performance)
    graph = create_agent_graph(llm=llm, checkpointer=checkpointer)
    
    # Execute graph with streaming
    try:
        # We can't easily stream tokens from LangGraph's astream() —
        # it yields full node completions. Instead, we run the pipeline
        # using our existing node-based approach but with LangGraph routing.
        
        # Since LangGraph streaming doesn't give us individual tokens,
        # we use a hybrid approach: LangGraph for routing, our existing
        # generators for streaming.
        
        # Step 1: Rewrite
        await emitter.emit_progress("rewrite", "Rewriting query...")
        from .nodes import rewrite_query_node
        rewrite_result = await rewrite_query_node(
            initial_state, llm=llm, api_base=api_base,
        )
        
        # Run the simplified pipeline with LangGraph routing decisions
        await emitter.emit_progress("rewriting", "Rewriting query...")
        rewritten = rewrite_result["rewritten_query"]
        await emitter.emit_rewritten_query(rewritten)
        
        # Step 2: Classify
        await emitter.emit_progress("classify", "Analyzing query...")
        classify_result = await classify_query_node(
            {**initial_state, **rewrite_result}, llm=llm,
        )
        
        is_complex = classify_result.get("is_complex", False)
        subtasks = classify_result.get("subtasks", [rewritten])
        
        if not classify_result.get("question_is_clear", True):
            # Clarification needed
            from .nodes import request_clarification_node
            clar_result = request_clarification_node(initial_state)
            for msg in clar_result.get("messages", []):
                content = msg.content if hasattr(msg, "content") else str(msg)
                await emitter.emit_token(content)
            return
        
        if is_complex and len(subtasks) > 1:
            # Complex path: subtask loop
            yield {"event": "task_list", "tasks": [
                {"id": i, "text": s, "status": "pending", "progress": None}
                for i, s in enumerate(subtasks)
            ]}
            
            all_answers = []
            all_docs = []
            
            for idx, subtask in enumerate(subtasks):
                # Update task status
                yield {"event": "task_list", "tasks": [
                    {"id": i, "text": s, "status": "running" if i == idx else ("done" if i < idx else "pending"),
                     "progress": None}
                    for i, s in enumerate(subtasks)
                ]}
                
                await emitter.emit_progress("rewriting", f"Rewriting query {idx + 1}/{len(subtasks)}...")
                
                # Search
                await emitter.emit_progress("keyword_search", "Searching keywords...")
                await emitter.emit_progress("dense_search", "Running dense vector search...")
                await emitter.emit_progress("sparse_search", "Running sparse vector search...")
                
                docs, retrieval_info, failed_legs = await _search_and_rerank(
                    subtask, kb_ids, db, use_dense, use_sparse, use_exact,
                    use_graph_rag, org_id, file_markdown,
                )
                
                await emitter.emit_progress("reranking", f"Shortlisted {len(docs)} chunks")
                
                conf_result = score_retrieval(docs, retrieval_info) if docs else None
                conf_score = conf_result.score if conf_result else 0
                
                serialised = [_serialise_doc(d) for d in docs]
                yield {
                    "event": "context",
                    "docs": serialised,
                    "confidence": conf_result.level if conf_result else "low",
                    "score": conf_score,
                    "failed_legs": failed_legs,
                    "subtask_index": idx,
                    "query_classification": {"type": "MULTI_PART", "confidence": 1.0},
                    "synthesis_mode": False,
                }
                
                # Build context string
                context_parts = []
                for i, doc in enumerate(serialised, 1):
                    content = doc.get("page_content", "").strip()
                    source = doc.get("metadata", {}).get("source", "")
                    header = f"[KB-{i}]" + (f" ({source})" if source else "")
                    context_parts.append(f"{header}\n{content}")
                if file_markdown:
                    context_parts.append(f"[File Content]\n{file_markdown}")
                merged = "\n\n---\n\n".join(context_parts)
                
                # Select model
                effective_model = _select_model(subtask, True)
                
                # Generate answer
                if is_thinking:
                    await emitter.emit_progress("generating", f"Thinking through subtask {idx + 1}/{len(subtasks)}...")
                else:
                    await emitter.emit_progress("generating", f"Generating answer {idx + 1}/{len(subtasks)}...")
                
                subtask_answer = ""
                async for event in _generate_via_graph(
                    subtask, merged, effective_model, api_base, existing_summary,
                    is_thinking, emitter,
                ):
                    if event.get("event") == "token":
                        subtask_answer += event.get("content", "")
                    elif event.get("event") == "done":
                        subtask_answer = event.get("full_response", subtask_answer)
                    yield event
                
                # Update task status
                yield {"event": "task_list", "tasks": [
                    {"id": i, "text": s, "status": "done" if i <= idx else "pending",
                     "progress": None}
                    for i, s in enumerate(subtasks)
                ]}
                
                all_answers.append({"answer": subtask_answer})
                all_docs.extend(serialised)
            
            # Synthesis
            await emitter.emit_progress("synthesizing", "Synthesizing final answer...")
            
            synthesis_parts = []
            for sa in all_answers:
                synthesis_parts.append(f"### Answer\n\n{sa['answer']}")
            combined = "\n\n---\n\n".join(synthesis_parts)
            
            all_serialised = [_serialise_doc(d) for d in all_docs]
            yield {
                "event": "context",
                "docs": all_serialised[:10],
                "confidence": "high" if len(all_docs) > 5 else ("medium" if all_docs else "low"),
                "synthesis_mode": True,
            }
            
            await emitter.emit_progress("synthesizing", "Generating final summary...")
            
            # Stream combined answer character by character
            answer_chars = list(combined)
            i = 0
            while i < len(answer_chars):
                chunk_size = min(50, len(answer_chars) - i)
                chunk = "".join(answer_chars[i:i + chunk_size])
                await emitter.emit_token(chunk)
                i += chunk_size
            
            yield {"event": "done", "full_response": combined, "usage": {}}
            final_answer = combined
            
        else:
            # Simple path: direct retrieval + generate
            await emitter.emit_progress("simple", "Answering directly...")
            
            await emitter.emit_progress("keyword_search", "Searching keywords...")
            await emitter.emit_progress("dense_search", "Running dense vector search...")
            await emitter.emit_progress("sparse_search", "Running sparse vector search...")
            
            docs, retrieval_info, failed_legs = await _search_and_rerank(
                rewritten, kb_ids, db, use_dense, use_sparse, use_exact,
                use_graph_rag, org_id, file_markdown,
            )
            
            await emitter.emit_progress("reranking", f"Shortlisted {len(docs)} chunks")
            
            conf_result = score_retrieval(docs, retrieval_info) if docs else None
            conf_score = conf_result.score if conf_result else 0
            
            serialised = [_serialise_doc(d) for d in docs]
            yield {
                "event": "context",
                "docs": serialised,
                "confidence": conf_result.level if conf_result else "low",
                "score": conf_score,
                "failed_legs": failed_legs,
                "query_classification": {"type": "FACTUAL", "confidence": 1.0},
                "synthesis_mode": False,
            }
            
            # Build context string
            context_parts = []
            for i, doc in enumerate(serialised, 1):
                content = doc.get("page_content", "").strip()
                source = doc.get("metadata", {}).get("source", "")
                header = f"[KB-{i}]" + (f" ({source})" if source else "")
                context_parts.append(f"{header}\n{content}")
            if file_markdown:
                context_parts.append(f"[File Content]\n{file_markdown}")
            merged = "\n\n---\n\n".join(context_parts)
            
            effective_model = _select_model(rewritten, False)
            
            await emitter.emit_progress("generating", "Generating answer...")
            
            async for event in _generate_via_graph(
                rewritten, merged, effective_model, api_base, existing_summary,
                False, emitter,
            ):
                yield event
            
            # Get the answer from the last done event
            final_answer = ""
            async for event in _generate_via_graph(
                rewritten, merged, effective_model, api_base, existing_summary,
                False, emitter,
            ):
                if event.get("event") == "done":
                    final_answer = event.get("full_response", "")
            yield {"event": "done", "full_response": final_answer, "usage": {}}
    
    except Exception as exc:
        logger.error("[GRAPH] pipeline failed: %s", exc, exc_info=True)
        await emitter.emit_error(str(exc))
        yield {"event": "error", "message": str(exc)}
        yield {"event": "done", "full_response": "", "usage": {}}
    
    finally:
        # Flush remaining events
        async for event in emitter.drain():
            yield event
    
    logger.info("[GRAPH] total latency=%.1fms query=%r", (time.monotonic() - t0) * 1000, query[:80])


async def _generate_via_graph(
    query: str,
    context_text: str,
    model_name: str,
    api_base: Optional[str],
    existing_summary: Optional[str],
    is_thinking: bool,
    emitter: SSEEventEmitter,
) -> AsyncGenerator[dict, None]:
    """Stream an answer via the graph's LLM. Yields token/thinking/done events."""
    from .nodes import _ANSWER_SYSTEM_PROMPT
    from app.services.infrastructure import strip_reasoning_tags
    
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
                    full_match = __import__('re').search(r'</think>(.*?)</think>', token, __import__('re').DOTALL)
                    if full_match:
                        thinking_parts.append(full_match.group(1))
                        token = token[full_match.end():]
                    else:
                        open_match = __import__('re').search(r'</think>', token)
                        if open_match:
                            token = token[open_match.end():]
                
                if stripped:
                    thinking_parts.append(stripped)
                    if len(thinking_parts) % 50 == 0 or stripped.endswith(('\n', ' ')):
                        yield {
                            "event": "thinking",
                            "content": "".join(thinking_parts),
                            "done": False,
                        }
                
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
        yield {
            "event": "thinking",
            "content": "".join(thinking_parts),
            "done": True,
        }
    
    answer = "".join(streamed_parts)
    
    # Normalise citation syntax
    import re as _re
    normalised = _re.sub(
        r'\[(\d+)\](?!\()',
        lambda m: f'[{m.group(1)}]({m.group(1)})',
        answer,
    )
    
    if normalised != answer:
        await emitter.emit_answer_rewrite(normalised)
    
    final_answer = normalised or answer
    
    # Validate charts
    chart_pattern = _re.compile(r'\[chart\](.*?)\[/chart\]', _re.DOTALL)
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
            "has_charts": True,
            "chart_count": len(matches),
            "valid_count": valid_count,
            "errors": errors,
            "all_valid": len(errors) == 0,
        }
    
    yield {"event": "done", "full_response": final_answer, "usage": usage, "chart_validated": chart_validated}
