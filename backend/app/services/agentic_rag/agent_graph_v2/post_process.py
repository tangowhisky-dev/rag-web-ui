"""Post-process node for agentic-v2.

Runs after the think node emits no tool calls (the LLM wrote the answer).
This node does NOT call the LLM — it:
1. Takes the answer text from precomputed_answer (or generates a fallback from evidence).
2. Substitutes chart markers and office file markers.
3. Normalizes citations.
4. Builds the LastAnswerObject.
5. Saves to DB.
6. Runs answer scoring (optional, single LLM call for metadata only).

If the LLM signaled final_answer=true but didn't write text (empty precomputed),
we generate the answer from evidence using the same streaming approach as the
old finalize node — but this is the fallback path, not the primary one.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.messages import AIMessage

from app.models.chat import Message
from app.services.agentic_rag.llm_factory import build_chat_llm
from app.services.agentic_rag.nodes import answer_evaluation_node
from app.services.agentic_rag.schemas import LastAnswerObject, Plan
from app.services.agentic_rag.token_budget import count_tokens
from app.services.agentic_rag.utils import (
    format_context_string,
    group_docs_by_document,
    normalize_citations,
    normalize_evidence_citations,
)
from app.services.infrastructure import is_cancelled
from app.services.settings_service import get_setting

from ..agent_graph.compaction import _compact_if_needed
from ..agent_graph.finalization import _build_finalize_prompt, _stream_final_answer
from ..agent_graph.helpers import _coerce_observation, _substitute_chart_markers, _substitute_office_markers, _writer
from ..agent_graph.observations import _non_retrieval_observations_text

logger = logging.getLogger(__name__)


def _collect_chart_options(observations: list) -> list[dict]:
    chart_options: list[dict] = []
    for raw_obs in observations:
        obs = _coerce_observation(raw_obs)
        if obs.tool == "chart_generate" and obs.result.get("chart_option"):
            chart_options.append(obs.result["chart_option"])
    return chart_options


def _collect_office_files(observations: list, generated_files: list | None = None) -> list[dict]:
    """Collect office file metadata from THIS TURN's observations only.

    state.generated_files persists across turns via the checkpointer, so
    collecting from it would show files from previous turns. Only
    create_office_document observations (this turn) are reliable.
    """
    office_files: list[dict] = []
    seen_ids: set = set()

    # From create_office_document wrapper observations (this turn only)
    for raw_obs in observations:
        obs = _coerce_observation(raw_obs)
        if obs.tool == "create_office_document" and obs.result.get("file_id"):
            fid = obs.result["file_id"]
            if fid not in seen_ids:
                seen_ids.add(fid)
                office_files.append({
                    "file_id": fid,
                    "file_name": obs.result.get("file_name", ""),
                    "format": obs.result.get("format", ""),
                    "summary": obs.result.get("summary", ""),
                    "title": obs.result.get("title"),
                    "slide_count": obs.result.get("slide_count"),
                    "sheet_count": obs.result.get("sheet_count"),
                    "chart_count": obs.result.get("chart_count"),
                })

    return office_files


def _extract_office_summary(observations: list) -> str:
    """Extract the office sub-agent's summary from create_office_document observations.

    The sub-agent writes a brief plain-text summary describing what was
    created. This should be the answer text — not the main LLM's
    reproduction of the document's slide/section content.
    """
    for raw_obs in reversed(observations):
        obs = _coerce_observation(raw_obs)
        if obs.tool == "create_office_document" and obs.result.get("summary"):
            return obs.result["summary"]
    return ""


def _build_last_answer_object(
    final: str,
    chart_options: list[dict],
    cited_docs: list[dict],
    office_files: list[dict] | None = None,
) -> LastAnswerObject:
    citations = []
    for doc in cited_docs:
        meta = doc.get("metadata", {}) if isinstance(doc, dict) else {}
        cref = meta.get("citation_ref")
        if cref and isinstance(cref, dict):
            # Normalize: ensure document_id is int or None, citation_kind is valid
            doc_id = cref.get("document_id")
            if doc_id is not None:
                try:
                    doc_id = int(doc_id)
                except (TypeError, ValueError):
                    doc_id = None
            kind = cref.get("citation_kind", "chunk")
            if kind not in ("chunk", "file", "section", "range", "grep", "table", "outline"):
                kind = "chunk"
            # Skip citations with no document_id — can't link to a source
            if doc_id is None:
                continue
            citations.append({
                **cref,
                "document_id": doc_id,
                "citation_kind": kind,
            })
    return LastAnswerObject(
        summary="",
        key_points=[],
        data=None,
        citations=citations,
        chart_options=chart_options,
        office_files=office_files or [],
        followups=[],
    )


async def post_process_node_v2(state, ctx) -> dict:
    """Post-process the answer: substitute markers, normalize citations, save, score."""
    # "Finalizing answer" phase is emitted by the think node when the LLM
    # starts streaming content (the final answer). For the fallback path
    # (no precomputed answer), we emit it here before _stream_final_answer.
    # We do NOT use _agent_step("finalize") here — that would emit a
    # duplicate "Finalizing answer" phase after the answer already streamed.
    writer = _writer()

    # Cancellation check: if the chat was cancelled, skip all generation,
    # citation normalization, scoring, and DB persistence. The chat_service
    # layer already saved the partial response to the DB message row.
    chat_id = ctx.chat_id if ctx is not None else None
    if chat_id is not None and is_cancelled(chat_id):
        logger.debug("[post_process_v2] cancelled — skipping generation and persistence | chat_id=%s", chat_id)
        return {
            "final_answer": "",
            "answer": "",
            "cited_docs": [],
            "messages": [],
        }

    precomputed = state.get("precomputed_answer", "")
    reasoning_content = state.get("reasoning_content", "")
    query = state.get("original_query", "")
    observations = state.get("observations", [])
    docs = state.get("retrieved_docs", [])
    answer_usage: Optional[dict] = None

    chart_options = _collect_chart_options(observations)
    office_files = _collect_office_files(observations)
    office_summary = _extract_office_summary(observations)

    # When create_office_document was called, use the sub-agent's summary
    # as the answer — not the main LLM's reproduction of slide content.
    # The sub-agent's summary is a brief description of what was created.
    if office_files and office_summary:
        precomputed = office_summary

    if precomputed and precomputed.strip():
        # The LLM wrote the answer in the think node. Use it directly.
        # Do NOT group docs here: the model's [E-N] citations refer to
        # the ungrouped retrieved_docs order shown in the think prompt.
        final = precomputed
    else:
        # Fallback: the LLM signaled final_answer=true but didn't write text.
        # Generate from evidence using the old finalize approach. Group
        # first so the model sees contiguous chunks and cites the grouped
        # order, which matches the normalization below.
        # Emit "Finalizing answer" phase here since the think node didn't
        # stream any content tokens (no precomputed answer).
        from ..agent_graph.helpers import _emit_timeline
        _finalize_phase = _emit_timeline(type="phase", label="Finalizing answer", status="active")
        docs = group_docs_by_document(docs)
        from app.services.agentic_rag.nodes import history_to_text, select_recent_history
        recent = select_recent_history(
            state.get("messages", []),
            max_pairs=get_setting(ctx.db, "AGENT_HISTORY_PAIRS", ctx.org_id),
        )
        history_text = history_to_text(recent)
        summary_text = state.get("compaction_summary") or ""
        system, user = _build_finalize_prompt(
            docs, state.get("file_markdown"), Plan(), chart_options,
            query, query, summary_text, history_text, observations, ctx, None,
        )
        compaction_updates, compaction_local = await _compact_if_needed(
            state, user, system_overhead=count_tokens(system), ctx=ctx, trim_docs=True,
        )
        if compaction_local:
            state = {**state, **compaction_local}
            docs = state.get("retrieved_docs", docs)
            recent = select_recent_history(
                state.get("messages", []),
                max_pairs=get_setting(ctx.db, "AGENT_HISTORY_PAIRS", ctx.org_id),
            )
            history_text = history_to_text(recent)
            summary_text = state.get("compaction_summary") or ""
            system, user = _build_finalize_prompt(
                docs, state.get("file_markdown"), Plan(), chart_options,
                query, query, summary_text, history_text, observations, ctx, None,
            )
        final, answer_usage, fallback_reasoning = await _stream_final_answer(ctx, system, user, writer, docs)
        # Use fallback reasoning if the think node didn't produce any.
        if fallback_reasoning and not reasoning_content:
            reasoning_content = fallback_reasoning
        # Close the "Finalizing answer" phase for the fallback path.
        _emit_timeline(id=_finalize_phase, type="phase", label="Finalizing answer", status="complete")

    # Always emit a final th: done event so the frontend closes the
    # final-answer reasoning panel. Without this, the UI stays stuck
    # on "Thinking..." because _stream_final_answer only emits
    # done=False chunks, and the precomputed path emits no th: events.
    writer({"event": "thinking", "content": reasoning_content, "done": True, "phase": "answer"})

    # Substitute chart and office markers.
    final = _substitute_chart_markers(final, chart_options)
    if office_files:
        final = _substitute_office_markers(final, office_files)

    # Normalize citations.
    has_evidence = any(
        (d.get("metadata", {}) or {}).get("citation_ref")
        for d in docs if isinstance(d, dict)
    )
    if has_evidence:
        for i, doc in enumerate(docs, 1):
            meta = doc.get("metadata", {}) if isinstance(doc, dict) else {}
            cref = meta.get("citation_ref") or {}
            if not cref.get("citation_id"):
                cref["citation_id"] = f"E{i}"
        final, cited_evidence = normalize_evidence_citations(final, docs)
        cited_docs = cited_evidence
    else:
        final, cited_doc_indices = normalize_citations(final, docs)
        cited_docs = [docs[i - 1] for i in cited_doc_indices]

    writer({"event": "answer_rewrite", "content": final, "citations": cited_docs})

    # Build LastAnswerObject.
    lao = _build_last_answer_object(final, chart_options, cited_docs, office_files)
    writer({"event": "last_answer", "last_answer_object": lao.model_dump()})

    # Persist assistant message.
    message_id = state.get("message_id")
    answer_message = AIMessage(
        content=final,
        id=f"assistant-{message_id}" if message_id else None,
    )

    # Make the final answer, cited evidence, and LAO available to
    # answer_evaluation so it can score rather than bailing out.
    # (state["last_answer_object"] still holds the previous turn's LAO
    # from load_context — must be overwritten before evaluation runs.)
    state["answer"] = final
    state["cited_docs"] = cited_docs
    state["last_answer_object"] = lao

    updates: dict = {
        "final_answer": final,
        "answer": final,
        "last_answer_object": lao,
        "retrieved_docs": docs,
        "cited_docs": cited_docs,
        "messages": [answer_message],
    }
    if answer_usage:
        updates["answer_usage"] = answer_usage

    # ── Answer scoring (optional, metadata only) ────────────────────
    try:
        scoring_updates = await answer_evaluation_node(state, ctx=ctx)
        updates.update(scoring_updates)
        # Update LAO with followups if scoring produced them.
        if scoring_updates.get("last_answer_object"):
            writer({"event": "last_answer", "last_answer_object": scoring_updates["last_answer_object"].model_dump()})
    except Exception as exc:
        logger.warning("[post_process_v2] answer scoring failed: %s", exc)

    # Ensure the persisted LAO includes any followups/fields added by
    # answer scoring so they survive a page refresh.
    lao = updates.get("last_answer_object", lao)

    # ── Save to DB ──────────────────────────────────────────────────
    if message_id:
        try:
            msg = ctx.db.query(Message).filter(Message.id == message_id).first()
            if msg:
                # Prepend reasoning as <think> tags so the frontend's
                # parseThinkContent can extract it on page reload.
                # The answer_rewrite event sends just the answer (no tags)
                # so the live streaming display is unaffected.
                saved_content = final
                if reasoning_content:
                    close_tag = "/think>"
                    saved_content = "<think" + ">" + reasoning_content + "<" + close_tag + "\n\n" + final
                msg.content = saved_content
                msg.last_answer_object = lao.model_dump()
                msg.tool_calls = [_coerce_observation(obs).model_dump() for obs in observations]
                # Persist answer scoring fields if scoring produced them.
                if "final_confidence" in updates:
                    msg.final_confidence = updates.get("final_confidence")
                if "final_confidence_level" in updates:
                    msg.final_confidence_level = updates.get("final_confidence_level")
                if "confidence_level" in updates:
                    msg.confidence_level = updates.get("confidence_level")
                if "faithfulness" in updates:
                    msg.faithfulness = updates.get("faithfulness")
                if "completeness" in updates:
                    msg.completeness = updates.get("completeness")
                if "retrieval_score" in updates:
                    msg.retrieval_score = updates.get("retrieval_score")
                ctx.db.commit()
        except Exception as exc:
            logger.warning("[post_process_v2] DB save failed: %s", exc)
            ctx.db.rollback()

    return updates
