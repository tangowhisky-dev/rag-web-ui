"""Finalize and save-memory nodes — answer generation and persistence.

finalize_node generates the final answer (streamed from the LLM or
precomputed), substitutes chart markers, normalizes citations, and
extracts a LastAnswerObject for the next turn's context.

save_memory_node persists the final answer, plan, last_answer_object,
and tool calls to the DB message row.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.messages import AIMessage

from app.models.chat import Message
from app.services.agentic_rag.llm_factory import build_chat_llm
from app.services.agentic_rag.nodes import _agent_step, history_to_text, select_recent_history
from app.services.agentic_rag.prompts import (
    FINALIZE_ANSWER_PROMPT,
    FINALIZE_GUARDRAIL_PROMPT,
    LAST_ANSWER_EXTRACT_PROMPT,
)
from app.services.agentic_rag.schemas import LastAnswerObject, Plan
from app.services.agentic_rag.token_budget import count_tokens
from app.services.agentic_rag.utils import format_context_string, group_docs_by_document, normalize_citations
from app.services.infrastructure import is_cancelled
from app.services.settings_service import get_setting

from .compaction import _compact_if_needed
from .helpers import _coerce_observation, _extract_json_block, _substitute_chart_markers, _writer
from .observations import _non_retrieval_observations_text

logger = logging.getLogger(__name__)


def _log_duplicate_check(docs: list[dict]) -> None:
    """Log duplicate chunks before final generation. Debug-only safeguard."""
    if not docs:
        return
    from app.services.infrastructure import content_hash
    seen: dict[str, int] = {}
    dup_hashes: list[str] = []
    for d in docs:
        meta = d.get("metadata", {}) if isinstance(d, dict) else {}
        h = meta.get("content_hash") or content_hash(d.get("page_content", ""))
        seen[h] = seen.get(h, 0) + 1
    duplicates = sum(count - 1 for count in seen.values() if count > 1)
    dup_hashes = [h[:12] for h, c in seen.items() if c > 1]
    if duplicates:
        logger.warning(
            "[finalize] DUPLICATE CHUNKS: %d duplicates in %d docs (hashes: %s)",
            duplicates, len(docs), dup_hashes[:10],
        )
    else:
        logger.debug("[finalize] no duplicate chunks in %d docs", len(docs))


def _build_finalize_prompt(
    docs: list[dict],
    file_markdown,
    plan,
    chart_options: list[dict],
    query: str,
    retrieval_query: str,
    summary_text: str,
    history_text: str,
    observations: list,
    glossary: str,
    ctx: "ToolContext",
    excluded_terms: list[str] | None = None,
) -> tuple[str, str]:
    """Build the finalize system+user prompt. Returns (system, user)."""
    context_text = format_context_string(docs, file_markdown, db=ctx.db, org_id=ctx.org_id, query_glossary=glossary)
    # Non-retrieval tool results (code_execute, chart_generate, etc.)
    # are not in retrieved_docs; surface them separately. Retrieval
    # results are already in context_text — don't duplicate.
    non_rag_text = _non_retrieval_observations_text(observations)

    # Chart docs are only appended when the plan intent is "chart" or
    # a chart_generate observation exists.
    plan_intent = plan.intent if isinstance(plan, Plan) else ""
    include_charts = plan_intent == "chart" or bool(chart_options)

    answer_prompt = FINALIZE_ANSWER_PROMPT
    if chart_options:
        # Valid chart JSON already exists — have the model place a
        # marker instead of freehand-writing (and risking malformed) JSON.
        from app.services.prompts.loader import append_chart_placeholder_instructions
        answer_prompt = append_chart_placeholder_instructions(answer_prompt, len(chart_options))
    elif include_charts:
        from app.services.prompts.loader import append_chart_instructions
        answer_prompt = append_chart_instructions(answer_prompt)

    system = FINALIZE_GUARDRAIL_PROMPT + "\n\n" + answer_prompt

    # Order: stable → volatile for prefix cache reuse.
    # Compaction summary changes rarely (only when compaction fires).
    # History grows by 1 pair per turn — the old prefix cache-hits.
    # Retrieved context, tool results, query all change every turn.
    parts: list[str] = []
    if summary_text:
        parts.append(f"Earlier conversation summary:\n{summary_text}\n\n")
    if history_text:
        parts.append(
            "Conversation so far (intent only — cite nothing from here):\n"
            f"{history_text}\n\n"
        )
    if excluded_terms:
        parts.append(f"User excluded topics: {', '.join(excluded_terms)}. Do not discuss these.\n\n")
    if non_rag_text:
        parts.append(f"Tool results:\n{non_rag_text}\n\n")
    parts.append(f"Retrieved context (the only citable evidence):\n{context_text}\n\n")
    if retrieval_query and retrieval_query != query:
        parts.append(f"Resolved retrieval query: {retrieval_query}\n\n")
    parts.append(f"User query: {query}\n\n")
    parts.append("Provide a concise, accurate answer.")
    user = "".join(parts)

    return system, user


async def _stream_final_answer(
    ctx: "ToolContext",
    system: str,
    user: str,
    writer,
) -> tuple[str, Optional[dict]]:
    """Stream the final answer from the LLM. Returns (final, answer_usage)."""
    answer_usage: Optional[dict] = None
    try:
        gen_temp = get_setting(ctx.db, "GENERATION_TEMPERATURE", ctx.org_id)
        llm = build_chat_llm(ctx.org_id, ctx.db, role="chat", temperature=gen_temp, streaming=True)
        final = ""
        async for chunk in llm.astream([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]):
            if ctx.chat_id is not None and is_cancelled(ctx.chat_id):
                logger.info("[finalize_node] cancel detected, aborting LLM stream | chat_id=%d", ctx.chat_id)
                break
            # Capture provider-reported usage when the backend sends
            # it, so reported tokens are measured rather than guessed.
            chunk_usage = getattr(chunk, "usage_metadata", None)
            if chunk_usage:
                answer_usage = {
                    "input_tokens": chunk_usage.get("input_tokens", 0),
                    "output_tokens": chunk_usage.get("output_tokens", 0),
                    "total_tokens": chunk_usage.get("total_tokens", 0),
                }
                # Calibrate the token estimator using the exact
                # prompt_tokens reported by the provider.
                from app.services.agentic_rag.token_budget import record_usage
                record_usage(system + user, chunk_usage.get("input_tokens", 0))
            content = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
            if content:
                writer({"event": "token", "content": content})
                final += content
        if not final:
            final = "I'm sorry, I couldn't generate a response at this time."
    except Exception as exc:
        logger.warning("[finalize_node] generation failed: %s", exc)
        final = "I'm sorry, I couldn't generate a response at this time."
    return final, answer_usage


async def _build_last_answer_object(
    raw_final: str,
    final: str,
    chart_options: list[dict],
    ctx: "ToolContext",
) -> LastAnswerObject:
    """Extract chart data for raw_for_extraction and construct LastAnswerObject."""
    # Extraction (Call 4, below) wants the raw marker text — not the
    # substituted chart JSON — so it isn't fed a large embedded blob.
    raw_for_extraction = raw_final
    # Append a readable summary of chart data so the extraction LLM can
    # see the actual values (the marker text alone says "[[CHART_1]]").
    if chart_options:
        chart_parts: list[str] = []
        for i, opt in enumerate(chart_options, 1):
            series = opt.get("series", [])
            xaxis = opt.get("xAxis", {})
            labels = xaxis.get("data", []) if isinstance(xaxis, dict) else []
            for s in series:
                values = s.get("data", [])
                pairs = ", ".join(
                    f"{labels[j]}={values[j]}"
                    for j in range(min(len(labels), len(values)))
                )
                chart_parts.append(f"Chart {i} ({s.get('type', 'chart')}): {pairs}")
        raw_for_extraction += "\n\nChart data:\n" + "\n".join(chart_parts)

    # Build a lightweight LastAnswerObject. Try LLM extraction for data/chart.
    lao = LastAnswerObject(
        summary=final[:500],
        key_points=[s.strip("- ") for s in final.splitlines() if s.strip()][:8],
        data=None,
        citations=[],
        chart_option=None,
        chart_options=[],
        followups=[],
    )

    # Use a structured extraction for data if any numeric content; otherwise cheap.
    llm_query = build_chat_llm(ctx.org_id, ctx.db, role="query", temperature=0.0)
    extracted: Optional[LastAnswerObject] = None
    for attempt in range(2):
        try:
            raw = await llm_query.ainvoke([
                {"role": "user", "content": LAST_ANSWER_EXTRACT_PROMPT.format(answer=raw_for_extraction[:3000])},
            ])
            block = _extract_json_block(str(raw.content))
            if block:
                extracted = LastAnswerObject.model_validate_json(block)
                break
        except Exception as exc:
            logger.debug("[finalize_node] last_answer_object extraction attempt %d failed: %s", attempt + 1, exc)
    if extracted:
        lao = extracted

    lao.chart_options = chart_options
    lao.chart_option = chart_options[0] if chart_options else None

    return lao


async def finalize_node(state, ctx) -> dict:
    """Generate final answer if not precomputed; extract LastAnswerObject."""
    with _agent_step("finalize"):
        writer = _writer()
        precomputed = state.get("precomputed_answer", "")
        # The answer model always sees the user's exact wording; only
        # retrieval sees the resolved standalone query.
        query = state.get("original_query", "") or state.get("rewritten_query", "")
        retrieval_query = state.get("rewritten_query", "") or query
        observations = state.get("observations", [])
        docs = state.get("retrieved_docs", [])
        # Log duplicate chunk detection before final generation.
        _log_duplicate_check(docs)
        # Group chunks by document so the LLM sees contiguous chunks from
        # the same document together, and citation indices map correctly.
        docs = group_docs_by_document(docs)
        plan = state.get("plan")
        compaction_updates: dict = {}
        answer_usage: Optional[dict] = None

        # Collect every valid chart_generate result up front so both the
        # prompt instructions and the post-generation substitution use the
        # same list, regardless of the precomputed/streamed branch below.
        chart_options: list[dict] = []
        for raw_obs in observations:
            obs = _coerce_observation(raw_obs)
            if obs.tool == "chart_generate" and obs.result.get("chart_option"):
                chart_options.append(obs.result["chart_option"])

        if precomputed:
            final = precomputed
        else:
            recent = select_recent_history(state.get("messages", []), max_pairs=get_setting(ctx.db, "AGENT_HISTORY_PAIRS", ctx.org_id))
            history_text = history_to_text(recent)
            summary_text = state.get("compaction_summary") or ""
            system, user = _build_finalize_prompt(
                docs, state.get("file_markdown"), plan, chart_options,
                query, retrieval_query, summary_text, history_text,
                observations, state.get("abbreviation_glossary", ""), ctx,
                excluded_terms=state.get("excluded_terms", []),
            )

            # Runtime compaction before the generation LLM call. trim_docs=True
            # because this prompt is dominated by retrieved_docs — summarising
            # conversation history cannot fix an evidence-payload overflow.
            compaction_updates, compaction_local = await _compact_if_needed(
                state, user, system_overhead=count_tokens(system), ctx=ctx, trim_docs=True,
            )
            if compaction_local:
                state = {**state, **compaction_local}
                observations = state.get("observations", [])
                docs = state.get("retrieved_docs", docs)
                recent = select_recent_history(state.get("messages", []), max_pairs=get_setting(ctx.db, "AGENT_HISTORY_PAIRS", ctx.org_id))
                history_text = history_to_text(recent)
                summary_text = state.get("compaction_summary") or ""
                system, user = _build_finalize_prompt(
                    docs, state.get("file_markdown"), plan, chart_options,
                    query, retrieval_query, summary_text, history_text,
                    observations, state.get("abbreviation_glossary", ""), ctx,
                    excluded_terms=state.get("excluded_terms", []),
                )

            final, answer_usage = await _stream_final_answer(ctx, system, user, writer)

        # Keep that copy before substituting, then rewrite citations and
        # stream the display-ready answer immediately, without waiting on
        # Call 4 (last_answer_object extraction) or Call 5 (confidence score).
        raw_final = final
        final = _substitute_chart_markers(final, chart_options)
        final, cited_doc_indices = normalize_citations(final, docs)
        cited_docs = [docs[i - 1] for i in cited_doc_indices]
        writer({"event": "answer_rewrite", "content": final, "citations": cited_docs})

        lao = await _build_last_answer_object(raw_final, final, chart_options, ctx)
        writer({"event": "last_answer", "last_answer_object": lao.model_dump()})

        # Persist the assistant turn into the checkpointed conversation.
        # Without this the thread holds user questions only, so every
        # downstream consumer of history (reference resolution, think,
        # compaction) reads half a conversation and invents the rest.
        # The `add_messages` reducer appends, and the id is stable per
        # assistant message row so a resume-after-interrupt replaces rather
        # than duplicates the turn.
        message_id = state.get("message_id")
        answer_message = AIMessage(
            content=final,
            id=f"assistant-{message_id}" if message_id else None,
        )

        updates = {
            **compaction_updates,
            "final_answer": final,
            "answer": final,
            "last_answer_object": lao,
            "retrieved_docs": docs,
            "cited_doc_indices": cited_doc_indices,
            "messages": [*compaction_updates.get("messages", []), answer_message],
        }
        if answer_usage:
            updates["answer_usage"] = answer_usage
        return updates

async def save_memory_node(state, ctx) -> dict:
    """Persist final answer, last_answer_object, and tool calls to the DB message row."""
    with _agent_step("save_memory"):
        message_id = state.get("message_id")
        if not message_id:
            return {}

        msg = ctx.db.query(Message).filter(Message.id == message_id).first()
        if not msg:
            return {}

        msg.content = state.get("final_answer", "")
        plan = state.get("plan")
        if plan:
            msg.plan = plan.model_dump() if isinstance(plan, Plan) else plan
        lao = state.get("last_answer_object")
        if lao:
            msg.last_answer_object = lao.model_dump() if isinstance(lao, LastAnswerObject) else lao
        observations = state.get("observations", [])
        msg.tool_calls = [_coerce_observation(obs).model_dump() for obs in observations]
        msg.final_confidence = state.get("final_confidence")
        msg.final_confidence_level = state.get("confidence_level")
        msg.faithfulness = state.get("faithfulness")
        msg.completeness = state.get("completeness")

        try:
            ctx.db.commit()
        except Exception as exc:
            logger.warning("[save_memory_node] failed to commit message updates: %s", exc)
            ctx.db.rollback()
        return {}
