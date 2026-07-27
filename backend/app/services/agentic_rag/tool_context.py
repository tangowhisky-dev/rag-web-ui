"""Shared context, RBAC, and audit helpers for every agent tool."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.models.chat import Chat, ChatFile, ToolCallAudit
from app.services.agentic_rag.token_budget import count_tokens

logger = logging.getLogger(__name__)


@dataclass
class ToolContext:
    """Runtime context injected into every agent tool."""

    db: Session
    user_id: int
    org_id: Optional[int] = None
    chat_id: Optional[int] = None
    message_id: Optional[int] = None
    qdrant_client: Any = None
    redis_memory: Any = None
    org_llm_config: dict = field(default_factory=dict)
    state: Any = None


def enforce_rbac(
    ctx: ToolContext,
    kb_ids: Optional[list] = None,
    file_id: Optional[int] = None,
) -> dict:
    """Filter requested resources to those the authenticated user may access.

    Returns a dict with the filtered ``kb_ids`` and ``file_id`` (None if denied).
    """
    result = {"kb_ids": [], "file_id": file_id}
    kb_ids = list(kb_ids) if kb_ids else []

    if ctx.chat_id is None:
        # Outside a chat, do not allow resource access.
        return {"kb_ids": [], "file_id": None}

    chat = (
        ctx.db.query(Chat)
        .filter(Chat.id == ctx.chat_id, Chat.user_id == ctx.user_id)
        .first()
    )
    if chat is None:
        logger.warning("RBAC: chat %s not owned by user %s", ctx.chat_id, ctx.user_id)
        return result

    allowed_kb_ids = {kb.id for kb in chat.knowledge_bases}
    if kb_ids:
        result["kb_ids"] = [k for k in kb_ids if k in allowed_kb_ids]
    else:
        result["kb_ids"] = sorted(allowed_kb_ids)

    if file_id is not None:
        cf = (
            ctx.db.query(ChatFile)
            .filter(ChatFile.id == file_id, ChatFile.chat_id == ctx.chat_id)
            .first()
        )
        if cf is None:
            logger.warning("RBAC: file %s not in chat %s", file_id, ctx.chat_id)
            result["file_id"] = None

    return result


def write_audit(
    ctx: ToolContext,
    tool: str,
    arguments: dict,
    result_summary: dict,
    tokens_in: int = 0,
    tokens_out: int = 0,
    latency_ms: int = 0,
    status: str = "ok",
    error: Optional[str] = None,
) -> None:
    """Persist a row to the ``tool_call_audit`` table."""
    if ctx.chat_id is None or ctx.db is None:
        return

    message_id = ctx.message_id
    iteration = 0
    if ctx.state is not None:
        message_id = message_id or getattr(ctx.state, "message_id", None)
        iteration = getattr(ctx.state, "iteration", 0)

    if tokens_in == 0:
        tokens_in = count_tokens(json.dumps(arguments, default=str))
    if tokens_out == 0:
        tokens_out = count_tokens(json.dumps(result_summary, default=str))

    try:
        record = ToolCallAudit(
            chat_id=ctx.chat_id,
            message_id=message_id,
            iteration=iteration,
            tool_name=tool,
            arguments=arguments,
            result_summary=result_summary,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            status=status,
            error=error,
        )
        ctx.db.add(record)
        ctx.db.flush()
    except Exception as exc:
        logger.warning("Failed to write tool_call_audit row: %s", exc)
