"""
Fast and Thinking answering pipelines.

query_rewrite → hybrid_search (dense+sparse+exact+neo4j) → stream answer

Both yield the same event dict shapes as rag_graph.run_stream so that
chat_service.generate_response can forward them unchanged.

Step data schema emitted per node:
  rewrite_query    : { rewritten_query: str }
  kb_retrieval     : { docs_found: int, chunks: [{preview, source, score}] }
  graph_enrichment : { graph_docs: int, enriched_docs: int,
                       context_lines: [str] }   (only if graph data found)
  generate_answer  : { usage: {promptTokens, completionTokens} }
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, AsyncGenerator, List, Optional

logger = logging.getLogger(__name__)

_ANSWER_SYSTEM_PROMPT = """\
You are a helpful assistant. Answer the user's question using ONLY the provided context.
If the context is insufficient, say so clearly.

FORMATTING RULES:
- Use ### headers to divide multi-part answers (e.g., "### 1. Definition", "### 2. How It Works").
- Use numbered lists for sequential steps or algorithms.
- Use bullet points with **bold terms** for features, attributes, or comparisons.
- Use inline code for variable names, identifiers, and technical terms (e.g., `wait()`, `Available[j]`).
- For simple single-concept questions, plain prose is fine — do not force structure.

CITATION RULES:
When you use information from a chunk, cite it as a markdown link with ONLY the number as both text and href:
  Example: process scheduling [1](1) involves saving the CPU state [2](2).
The number must match the [KB-N] label of the chunk you are citing.
Do NOT invent citations. Only cite chunks you actually used.
"""

_PREVIEW_CHARS = 120  # characters shown per chunk in the timeline


def _preview(text: str, n: int = _PREVIEW_CHARS) -> str:
    text = text.strip().replace("\n", " ")
    return text[:n] + "…" if len(text) > n else text


def _filter_by_score(docs: list, threshold: float) -> list:
    """Filter docs by _reranker_score >= threshold, sorted descending by score."""
    filtered = [
        d for d in docs
        if d.metadata.get("_reranker_score", -float("inf")) >= threshold
    ]
    filtered.sort(
        key=lambda d: d.metadata.get("_reranker_score", -float("inf")),
        reverse=True,
    )
    return filtered


def _get_llm(model_name: str, temperature: float = 0.0, api_base: Optional[str] = None):
    from langchain_openai import ChatOpenAI
    from app.core.config import settings
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_base=api_base or settings.OPENAI_API_BASE,
        openai_api_key=settings.OPENAI_API_KEY,
        streaming=True,
    )


def _serialise_doc(doc: Any) -> dict:
    if hasattr(doc, "page_content"):
        return {"page_content": doc.page_content, "metadata": dict(doc.metadata)}
    if isinstance(doc, dict):
        return doc
    return {"page_content": str(doc), "metadata": {}}


async def _grade_answer_quality(
    query: str,
    answer: str,
    context_chunks: list[str],
    model_name: str,
    api_base: Optional[str] = None,
) -> dict:
    """
    Grade answer quality using the query model.

    Returns dict with keys:
        faithfulness  : float 0.00-1.00 (all claims backed by context?)
        completeness  : float 0.00-1.00 (does it answer the question?)
        coherence     : float 0.00-1.00 (well-structured?)
        verdict       : str "pass", "needs_improvement", or "unsatisfactory"
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    from app.core.config import settings

    grading_prompt = f"""You are grading an AI assistant's answer to a knowledge-base query.

Question: {query}

Answer:
{answer}

Context chunks used:
{chr(10).join(context_chunks[:10]) if context_chunks else '(none)'}

Rate on a scale of 0.00 to 1.00:
- faithfulness: are ALL factual claims in the answer directly supported by the context chunks? Penalize any claim not grounded in the provided context.
- completeness: does the answer address all parts of the question? Penalize for missing key information.
- coherence: is the answer well-structured and easy to understand? Penalize for rambling, repetition, or poor formatting.

Return ONLY valid JSON with no markdown code fences:
{{"faithfulness": <float>, "completeness": <float>, "coherence": <float>}}"""

    system_msg = SystemMessage(content="You are a precise grader. Return ONLY valid JSON with three float keys.")
    user_msg = HumanMessage(content=grading_prompt)

    try:
        # Non-streaming call for structured JSON output
        strict_llm = ChatOpenAI(
            model=model_name,
            temperature=0.0,
            openai_api_base=api_base or settings.OPENAI_API_BASE,
            openai_api_key=settings.OPENAI_API_KEY,
            streaming=False,
        )
        response = await strict_llm.ainvoke([system_msg, user_msg])
        content = response.content.strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            content = re.sub(r'^```(?:json)?\s*|\s*```$', '', content, flags=re.MULTILINE).strip()

        scores = _parse_grading_json(content)
    except Exception as exc:
        logger.warning("[QUALITY] grading failed (non-fatal): %s", exc)
        scores = {"faithfulness": 0.0, "completeness": 0.0, "coherence": 0.0}

    scores["verdict"] = _compute_verdict(scores)
    return scores


def _parse_grading_json(raw: str) -> dict:
    """Parse grading JSON from LLM response, extracting first JSON object found."""
    # Try direct parse first
    try:
        result = json.loads(raw)
        return {
            "faithfulness": max(0.0, min(1.0, float(result.get("faithfulness", 0.5)))),
            "completeness": max(0.0, min(1.0, float(result.get("completeness", 0.5)))),
            "coherence": max(0.0, min(1.0, float(result.get("coherence", 0.5)))),
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Try to extract JSON from markdown-fenced block
    match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            return {
                "faithfulness": max(0.0, min(1.0, float(result.get("faithfulness", 0.5)))),
                "completeness": max(0.0, min(1.0, float(result.get("completeness", 0.5)))),
                "coherence": max(0.0, min(1.0, float(result.get("coherence", 0.5)))),
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    logger.warning("[QUALITY] could not parse grading JSON, returning defaults")
    return {"faithfulness": 0.5, "completeness": 0.5, "coherence": 0.5}


def _compute_verdict(scores: dict) -> str:
    """Determine pass/needs_improvement/unsatisfactory from scores."""
    if any(scores.get(k, 0) < 0.50 for k in ("faithfulness", "completeness", "coherence")):
        return "unsatisfactory"
    if any(scores.get(k, 0) < 0.70 for k in ("faithfulness", "completeness", "coherence")):
        return "needs_improvement"
    return "pass"


async def fast_stream(
    query: str,
    knowledge_base_ids: List[int],
    db: Any,
    recent_lc_history: list,
    existing_summary: Optional[str],
    use_dense: bool = True,
    use_sparse: bool = True,
    use_exact: bool = True,
    use_graph_rag: bool = False,
    temperature: float = 0.0,
    model_name: Optional[str] = None,
    display_query: Optional[str] = None,
    file_markdown: Optional[str] = None,
    api_base: Optional[str] = None,
    query_model: Optional[str] = None,
    org_id: Optional[int] = None,
    chat_id: Optional[int] = None,
) -> AsyncGenerator[dict, None]:
    """
    Lean RAG pipeline: rewrite → hybrid_search → stream answer.
    Yields event dicts in the same shapes as rag_graph.run_stream.
    """
    from app.core.config import settings
    from app.services.retrieval import hybrid_search_with_legs
    from app.services.chat_service import _rewrite_query

    effective_model = model_name or query_model or settings.OPENAI_MODEL

    # ── 1. Query rewrite ──────────────────────────────────────────────────────
    # Emit "active" first so the UI shows "Rewriting query…"
    yield {"event": "agent_step", "node": "rewrite_query", "status": "active", "latency_ms": None}

    t0 = time.monotonic()
    try:
        # Only pass last 2 pairs (4 msgs) to rewriter — it only needs pronoun/reference
        # resolution; more history causes topic drift and query contamination.
        rewritten = await _rewrite_query(query, recent_lc_history, api_base=api_base, query_model=query_model)
    except Exception as exc:
        logger.warning("[FAST] rewrite failed (non-fatal): %s", exc)
        rewritten = query

    rewrite_ms = round((time.monotonic() - t0) * 1000, 1)
    logger.info("[FAST] rewrite latency_ms=%.1f | rewritten=%r", rewrite_ms, rewritten[:80])

    yield {
        "event": "agent_step",
        "node": "rewrite_query",
        "status": "done",
        "latency_ms": rewrite_ms,
        "rewritten_query": rewritten,
    }
    yield {
        "event": "rewritten_query",
        "query": rewritten,
    }

    # ── 1.5. Historical memory retrieval (before hybrid search) ──────────
    historical_docs: list = []
    if chat_id is not None and settings.HISTORICAL_MEMORY_ENABLED:
        try:
            from app.services.historical_memory import retrieve_historical_memory
            historical_docs = retrieve_historical_memory(
                chat_id=chat_id,
                query=rewritten,
                db=db,
                top_k=settings.HISTORICAL_MEMORY_TOP_K,
                score_threshold=settings.HISTORICAL_MEMORY_SCORE_THRESHOLD,
            )
        except Exception as exc:
            logger.warning("[HIST] retrieval failed (non-fatal): %s", exc)
            historical_docs = []
    if historical_docs:
        logger.info("[HIST] chat_id=%d | retrieved=%d", chat_id, len(historical_docs))

    # ── 2. Hybrid retrieval ───────────────────────────────────────────────────
    yield {"event": "agent_step", "node": "kb_retrieval", "status": "active", "latency_ms": None}

    t1 = time.monotonic()
    retrieval_result: dict = {}
    raw_docs: list = []
    try:
        # Get linked datastores for these KBs
        datastore_ids = []
        if knowledge_base_ids and db:
            from app.models.knowledge import KnowledgeBaseDataStore
            datastore_links = (
                db.query(KnowledgeBaseDataStore.data_store_id)
                .filter(KnowledgeBaseDataStore.knowledge_base_id.in_(knowledge_base_ids))
                .distinct()
                .all()
            )
            datastore_ids = [row.data_store_id for row in datastore_links]
        # Also resolve datastores linked to the user's organization (standalone DataStores)
        if org_id and db and datastore_ids is not None:
            from app.models.datastore import OrganizationDataStore
            ds_org_links = (
                db.query(OrganizationDataStore.data_store_id)
                .filter(OrganizationDataStore.org_id == org_id)
                .distinct()
                .all()
            )
            org_ds_ids = [row.data_store_id for row in ds_org_links]
            for ds_id in org_ds_ids:
                if ds_id not in datastore_ids:
                    datastore_ids.append(ds_id)
            if org_ds_ids:
                logger.info("[FAST] Found %d org-linked datastores for org_id=%s", len(org_ds_ids), org_id)

        retrieval_result = await hybrid_search_with_legs(
            query=rewritten,
            kb_ids=knowledge_base_ids,
            db=db,
            use_dense=use_dense,
            use_sparse=use_sparse,
            use_exact=use_exact,
            use_graph_rag=use_graph_rag,
            datastore_ids=datastore_ids,
            return_full_pool=True,
        )
        raw_docs = retrieval_result.get("docs", [])
    except Exception as exc:
        logger.error("[FAST] retrieval failed: %s", exc)

    retrieval_ms = round((time.monotonic() - t1) * 1000, 1)
    retrieval_info = retrieval_result.get("retrieval_info", {})
    failed_legs = retrieval_info.get("failed_legs", [])
    logger.info("[FAST] retrieval latency_ms=%.1f | docs=%d | failed_legs=%s",
                retrieval_ms, len(raw_docs), failed_legs)

    # Separate graph-expanded docs (added by Neo4j expansion) from regular docs
    graph_expanded: list = []
    kb_docs: list = []
    for doc in raw_docs:
        meta = doc.metadata if hasattr(doc, "metadata") else doc.get("metadata", {})
        legs = meta.get("_legs", [])
        if isinstance(legs, list) and "graph" in legs:
            graph_expanded.append(doc)
        else:
            kb_docs.append(doc)

    # Build chunk previews for kb_retrieval step (ordered as reranker returned them)
    chunk_previews = []
    for doc in kb_docs:
        d = _serialise_doc(doc)
        meta = d["metadata"]
        chunk_previews.append({
            "preview": _preview(d["page_content"]),
            "source": meta.get("source") or meta.get("file_name") or "",
            "score": round(float(meta.get("_rrf_score", meta.get("score", 0))), 4),
        })

    yield {
        "event": "agent_step",
        "node": "kb_retrieval",
        "status": "done",
        "latency_ms": retrieval_ms,
        "docs_found": len(raw_docs),
        "chunks": chunk_previews,
    }

    # ── 2b. Graph enrichment step (only if graph data was retrieved) ──────────
    graph_info = retrieval_info.get("legs", {}).get("graph", {})
    enriched_count = graph_info.get("count", 0)
    expanded_count = graph_info.get("expanded", 0)
    has_graph = enriched_count > 0 or expanded_count > 0 or bool(graph_expanded)

    if has_graph or graph_expanded:
        # Build context lines — first 3 lines of each graph-expanded chunk
        context_lines: list[str] = []
        for doc in graph_expanded[:6]:
            d = _serialise_doc(doc)
            lines = d["page_content"].strip().splitlines()
            first_lines = " ".join(l.strip() for l in lines[:3] if l.strip())
            source = d["metadata"].get("source") or d["metadata"].get("file_name") or ""
            context_lines.append(f"{_preview(first_lines, 150)}" + (f" [{source}]" if source else ""))

        yield {
            "event": "agent_step",
            "node": "graph_enrichment",
            "status": "done",
            "latency_ms": 0,
            "graph_docs": expanded_count,
            "enriched_docs": enriched_count,
            "context_lines": context_lines,
        }


    # ── 2c. Adaptive retrieval with confidence-based context events ─────────────
    # Use the same reranker-aware confidence scorer as rag_graph.
    # raw_docs already have _reranker_score in metadata (set by rerank()).
    from app.services.confidence import score_retrieval

    # Filter raw_docs by standard threshold → standard_docs
    standard_docs = _filter_by_score(raw_docs, settings.RERANKER_SCORE_THRESHOLD)

    # Score confidence on standard docs only (not on all raw_docs)
    conf_result = score_retrieval(standard_docs, retrieval_info)
    conf_score = conf_result.score  # 0-100 scale
    conf_score_01 = round(conf_score / 100, 2)  # normalise 0-100 → 0-1 for frontend

    # Count graph vs KB docs for breakdown
    def _count_graph(docs_list):
        g = sum(
            1 for d in docs_list
            if isinstance(d.metadata.get("_legs", []), list) and "graph" in d.metadata.get("_legs", [])
        )
        return len(docs_list) - g, g  # (kb, graph)

    std_kb, std_graph = _count_graph(standard_docs)

    # Standard context event (always emitted)
    standard_serialised = [_serialise_doc(d) for d in standard_docs]
    yield {
        "event": "context",
        "docs": standard_serialised,
        "confidence": conf_result.level,
        "score": conf_score_01,
        "suggestion": conf_result.suggestion or "",
        "failed_legs": failed_legs,
        "breakdown": {
            **conf_result.breakdown,
            "kb_docs": std_kb,
            "graph_docs": std_graph,
            "file_chars": len(file_markdown or ""),
        },
        "query_classification": {
            "type": "FACTUAL",
            "confidence": conf_score_01,
            "latency_ms": 0,
            "fallback": False,
        },
        "tool_trace": ["rewrite_query", "kb_retrieval", "generate_answer"],
        "synthesis_mode": False,
    }

    # ── 2d. Adaptive expansion for low-confidence queries ─────────────────────
    expanded_docs = standard_docs  # default for high confidence
    expanded_from = len(standard_docs)
    expanded_to = len(standard_docs)

    if settings.ADAPTIVE_RETRIEVAL_ENABLED and conf_score < settings.ADAPTIVE_RETRIEVAL_THRESHOLD:
        adaptive_threshold = settings.ADAPTIVE_RETRIEVAL_RERANKER_THRESHOLD
        expanded_docs = _filter_by_score(raw_docs, adaptive_threshold)
        expanded_from = len(standard_docs)
        expanded_to = len(expanded_docs)
        exp_kb, exp_graph = _count_graph(expanded_docs)

        adaptive_serialised = [_serialise_doc(d) for d in expanded_docs]
        yield {
            "event": "context",
            "docs": adaptive_serialised,
            "confidence": conf_result.level,
            "score": conf_score_01,
            "suggestion": conf_result.suggestion or "",
            "failed_legs": failed_legs,
            "breakdown": {
                **conf_result.breakdown,
                "kb_docs": exp_kb,
                "graph_docs": exp_graph,
                "adaptive": True,
                "threshold_used": adaptive_threshold,
                "expanded_from": expanded_from,
                "expanded_to": expanded_to,
                "file_chars": len(file_markdown or ""),
            },
            "query_classification": {
                "type": "FACTUAL",
                "confidence": conf_score_01,
                "latency_ms": 0,
                "fallback": False,
            },
            "tool_trace": ["rewrite_query", "kb_retrieval", "generate_answer"],
            "synthesis_mode": False,
        }
        logger.info(
            "[ADAPTIVE] standard_count=%d expanded_count=%d conf_score=%d",
            expanded_from, expanded_to, conf_score,
        )

    # Serialise expanded_docs for LLM context (standard docs when high conf)
    serialised_docs = [_serialise_doc(d) for d in expanded_docs]

    # ── 3. Build context string ────────────────────────────────────────────────
    context_parts: list[str] = []
    # Prepend historical memory blocks BEFORE KB docs
    if historical_docs:
        for i, doc in enumerate(historical_docs, 1):
            content = doc.get("page_content", "").strip()
            if content:
                context_parts.append(f"[HIST-{i}]\n{content}")
    for i, doc in enumerate(serialised_docs, 1):
        content = doc.get("page_content", "").strip()
        source = doc.get("metadata", {}).get("source", "")
        header = f"[KB-{i}]" + (f" ({source})" if source else "")
        context_parts.append(f"{header}\n{content}")
    if file_markdown:
        context_parts.append(f"[File Content]\n{file_markdown}")
    merged = "\n\n---\n\n".join(context_parts)

    # ── 4. Build messages ──────────────────────────────────────────────────────
    # Generator context: summary only.
    # recent_lc_history is intentionally excluded — raw prior answers pollute
    # the context and cause the LLM to treat its own previous responses as
    # user statements. The summary (built after every turn) provides all
    # necessary prior context in a clean, condensed form.
    messages: list = [{"role": "system", "content": _ANSWER_SYSTEM_PROMPT}]
    if existing_summary:
        messages.append({
            "role": "system",
            "content": f"[Conversation summary so far]\n{existing_summary}",
        })

    messages.append({
        "role": "user",
        "content": f"Context:\n{merged}\n\nQuestion: {rewritten}" if merged else rewritten,
    })

    # ── 5. Stream answer ──────────────────────────────────────────────────────
    yield {"event": "agent_step", "node": "generate_answer", "status": "active", "latency_ms": None}

    t2 = time.monotonic()
    llm = _get_llm(effective_model, temperature, api_base=api_base)
    streamed_parts: list[str] = []
    usage: dict = {"promptTokens": 0, "completionTokens": 0}

    try:
        async for chunk in llm.astream(messages):
            token: str = chunk.content or ""
            if token:
                streamed_parts.append(token)
                yield {"event": "token", "content": token}
            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                usage = {
                    "promptTokens": chunk.usage_metadata.get("input_tokens", 0),
                    "completionTokens": chunk.usage_metadata.get("output_tokens", 0),
                }
    except Exception as exc:
        logger.error("[FAST] generation failed: %s", exc)
        err_msg = "I encountered an error in synthesizing the response. Please try again."
        yield {"event": "token", "content": err_msg}
        streamed_parts.append(err_msg)

    answer_ms = round((time.monotonic() - t2) * 1000, 1)
    logger.info("[FAST] generation latency_ms=%.1f | tokens=%d | model=%s",
                answer_ms, len(streamed_parts), effective_model)

    yield {
        "event": "agent_step",
        "node": "generate_answer",
        "status": "done",
        "latency_ms": answer_ms,
        "usage": usage,
    }

    # Normalise citation syntax: [2] → [2](2)
    raw_answer = "".join(streamed_parts)
    normalised = re.sub(r'\[(\d+)\](?!\()', lambda m: f'[{m.group(1)}]({m.group(1)})', raw_answer)
    if normalised != raw_answer:
        yield {"event": "answer_rewrite", "content": normalised}

    # ── 6. Quality grading (post-stream, only for low-confidence retrieval) ──
    final_answer = normalised or raw_answer
    if conf_score < 55 and settings.ANSWER_QUALITY_GRADING_ENABLED:
        # Build chunk previews for the grading prompt
        grading_chunks = [
            _preview(d.get("page_content", ""), _PREVIEW_CHARS) + (" [" + d.get("metadata", {}).get("source", "") + "]" if d.get("metadata", {}).get("source") else "")
            for d in expanded_docs
        ]
        try:
            grading_scores = await _grade_answer_quality(
                query=rewritten,
                answer=final_answer,
                context_chunks=grading_chunks,
                model_name=effective_model,
                api_base=api_base,
            )
            logger.info(
                "[QUALITY] conf=%d | verdict=%s | faith=%.2f complete=%.2f coherent=%.2f",
                conf_score, grading_scores["verdict"],
                grading_scores["faithfulness"], grading_scores["completeness"],
                grading_scores["coherence"],
            )

            if grading_scores["verdict"] == "needs_improvement":
                # Regenerate with feedback appended to user message
                t3 = time.monotonic()
                feedback_msg = (
                    f"\n\n[Quality feedback — please improve: "
                    f"faithfulness={grading_scores['faithfulness']:.2f}, "
                    f"completeness={grading_scores['completeness']:.2f}, "
                    f"coherence={grading_scores['coherence']:.2f}.]"
                )
                feedback_messages = list(messages)  # shallow copy
                feedback_messages[-1] = {
                    "role": feedback_messages[-1]["role"],
                    "content": feedback_messages[-1]["content"] + feedback_msg,
                }
                yield {"event": "agent_step", "node": "regenerate_answer", "status": "active", "latency_ms": None}
                regenerate_llm = _get_llm(effective_model, temperature, api_base=api_base)
                regen_parts: list[str] = []
                async for chunk in regenerate_llm.astream(feedback_messages):
                    token: str = chunk.content or ""
                    if token:
                        regen_parts.append(token)
                        yield {"event": "token", "content": token}
                regen_ms = round((time.monotonic() - t3) * 1000, 1)
                final_answer = "".join(regen_parts)
                yield {
                    "event": "agent_step",
                    "node": "regenerate_answer",
                    "status": "done",
                    "latency_ms": regen_ms,
                }
                logger.info("[QUALITY] regenerated after feedback in %.1fms", regen_ms)

            elif grading_scores["verdict"] == "unsatisfactory":
                # Append disclaimer to the answer
                disclaimer = (
                    "\n\n[Disclaimer: This answer could not meet quality standards. "
                    "The information may be incomplete or inaccurate. "
                    "Please verify with the source documents.]"
                )
                final_answer = final_answer + disclaimer
                logger.info("[QUALITY] unsatisfactory — disclaimer appended")

        except Exception as exc:
            logger.warning("[QUALITY] grading failed (non-fatal): %s", exc)

    yield {"event": "done", "full_response": final_answer, "usage": usage}
