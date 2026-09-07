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
from app.services.agentic_rag.nodes import _agent_step, answer_evaluation_node
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


def _collect_office_files(observations: list) -> list[dict]:
    office_files: list[dict] = []
    for raw_obs in observations:
        obs = _coerce_observation(raw_obs)
        if obs.tool == "office_generate" and obs.result.get("file_id"):
            office_files.append({
                "file_id": obs.result["file_id"],
                "file_name": obs.result["file_name"],
                "format": obs.result["format"],
                "title": obs.result.get("title"),
                "slide_count": obs.result.get("slide_count"),
                "sheet_count": obs.result.get("sheet_count"),
                "chart_count": obs.result.get("chart_count"),
            })
    return office_files


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
        if cref:
            citations.append(cref)
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
    with _agent_step("finalize"):
        writer = _writer()
        precomputed = state.get("precomputed_answer", "")
        query = state.get("original_query", "")
        observations = state.get("observations", [])
        docs = state.get("retrieved_docs", [])
        docs = group_docs_by_document(docs)
        answer_usage: Optional[dict] = None

        chart_options = _collect_chart_options(observations)
        office_files = _collect_office_files(observations)

        if precomputed and precomputed.strip():
            # The LLM wrote the answer in the think node. Use it directly.
            final = precomputed
            writer({"event": "answer_rewrite", "content": final, "citations": []})
        else:
            # Fallback: the LLM signaled final_answer=true but didn't write text.
            # Generate from evidence using the old finalize approach.
            from app.services.agentic_rag.nodes import history_to_text, select_recent_history
            recent = select_recent_history(
                state.get("messages", []),
                max_pairs=get_setting(ctx.db, "AGENT_HISTORY_PAIRS", ctx.org_id),
            )
            history_text = history_to_text(recent)
            summary_text = state.get("compaction_summary") or ""
            system, user = _build_finalize_prompt(
                docs, state.get("file_markdown"), Plan(), chart_options,
                query, query, summary_text, history_text, observations, ctx, office_files,
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
                    query, query, summary_text, history_text, observations, ctx, office_files,
                )
            final, answer_usage = await _stream_final_answer(ctx, system, user, writer)

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

        # ── Save to DB ──────────────────────────────────────────────────
        if message_id:
            try:
                msg = ctx.db.query(Message).filter(Message.id == message_id).first()
                if msg:
                    msg.content = final
                    msg.last_answer_object = lao.model_dump()
                    msg.tool_calls = [_coerce_observation(obs).model_dump() for obs in observations]
                    ctx.db.commit()
            except Exception as exc:
                logger.warning("[post_process_v2] DB save failed: %s", exc)
                ctx.db.rollback()

        # ── Answer scoring (optional, metadata only) ────────────────────
        try:
            scoring_updates = await answer_evaluation_node(state, ctx=ctx)
            updates.update(scoring_updates)
            # Update LAO with followups if scoring produced them.
            if scoring_updates.get("last_answer_object"):
                writer({"event": "last_answer", "last_answer_object": scoring_updates["last_answer_object"].model_dump()})
        except Exception as exc:
            logger.warning("[post_process_v2] answer scoring failed: %s", exc)

        return updates
