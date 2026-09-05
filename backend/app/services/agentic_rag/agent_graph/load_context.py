"""Load-context node — initialises per-turn state from DB and memory.

Loads the previous-answer object, recalled conversational memory, and file
metadata into graph state. Resets all per-turn loop state (observations,
iteration, tool_call_counts, force_finalize, etc.) so the checkpointer
doesn't carry over stale values from the previous turn.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from app.models.chat import Message
from app.services.agentic_rag.kb_profile import profile_kb, merge_profiles
from app.services.agentic_rag.nodes import _agent_step
from app.services.agentic_rag.schemas import LastAnswerObject

logger = logging.getLogger(__name__)


async def load_context_node(state, ctx) -> dict:
    """Load previous-answer object, recalled memory, and file metadata into state."""
    with _agent_step("load_context"):
        last_obj: Optional[LastAnswerObject] = None
        if ctx.chat_id and ctx.message_id:
            # The current assistant message may already exist; find the previous assistant message.
            prev = (
                ctx.db.query(Message)
                .filter(Message.chat_id == ctx.chat_id, Message.role == "assistant")
                .filter(Message.id != ctx.message_id)
                .order_by(Message.id.desc())
                .first()
            )
            if prev and prev.last_answer_object:
                try:
                    last_obj = LastAnswerObject(**prev.last_answer_object)
                except Exception:
                    last_obj = None

        recalled: list[dict] = []
        if ctx.redis_memory and getattr(ctx.redis_memory, "search_memory", None):
            try:
                recalled = await ctx.redis_memory.search_memory(
                    query=state.get("original_query", ""),
                    user_id=ctx.user_id,
                    chat_id=ctx.chat_id,
                    limit=3,
                )
            except Exception as exc:
                logger.warning("[load_context] memory search failed: %s", exc)

        # Load KB profiles (per-KB, cached in Redis, merged into a single dict).
        kb_profile: dict = {}
        kb_ids = state.get("kb_ids", [])
        if kb_ids:
            try:
                per_kb = [await profile_kb(ctx.org_id, kb_id, ctx.db) for kb_id in kb_ids]
                kb_profile = merge_profiles([p for p in per_kb if p])
            except Exception as exc:
                logger.warning("[load_context] KB profile load failed: %s", exc)

        return {
            "last_answer_object": last_obj,
            # Recalled conversational memory is NOT citable evidence: it stays
            # out of retrieved_docs so it can never be rendered as a [KB-n]
            # chunk, cited, or scored for faithfulness.
            "recalled_memories": recalled,
            "org_id": ctx.org_id,
            "user_id": ctx.user_id,
            "chat_id": ctx.chat_id,
            "message_id": ctx.message_id,
            "started_at": time.monotonic(),
            # Reset per-turn loop state; the checkpointer otherwise carries it
            # over from the previous turn (e.g. force_finalize would silently
            # kill tool calls this turn; observations would leak last turn's
            # doc chunks into this turn's think_node prompt).
            "observations": [{"__reset__": True}],
            "iteration": 0,
            "tool_call_counts": {},
            "force_finalize": False,
            "precomputed_answer": "",
            "tool_calls": [],
            "retrieved_docs": [],
            "cited_doc_indices": [],
            "compaction_triggered": False,
            "answer_evaluation_attempts": 0,
            "answer_usage": None,
            "final_answer": "",
            "answer": "",
            "clarification_count": 0,
            "clarification_response": "",
            "needs_clarification": False,
            "kb_profile": kb_profile,
            # New atomic-tools state
            "sufficient": False,
        }
