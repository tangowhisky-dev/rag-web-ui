"""Export endpoints — single-message and whole-chat export.

Renders an individual assistant message as PDF, Word (.docx), or PNG image,
or exports the entire conversation as a Markdown file with role headers
and rewritten-query annotations.
"""
from typing import Any, Literal

from fastapi import Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.api.api_v1.chat import router
from app.db.session import get_db
from app.models.user import User
from app.models.chat import Chat, Message
from app.api.api_v1.rbac import chat_owner_filter as _chat_owner_filter
from app.core.security import get_current_user


@router.get("/{chat_id}/messages/{message_id}/export")
def export_message(
    *,
    db: Session = Depends(get_db),
    chat_id: int,
    message_id: int,
    format: Literal["pdf", "word", "image"] = Query(...),
    current_user: User = Depends(get_current_user),
) -> Response:
    """Export a single assistant message as PDF, Word (.docx), or PNG image."""
    chat = (
        db.query(Chat)
        .filter(Chat.id == chat_id, _chat_owner_filter(current_user))
        .first()
    )
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    msg = (
        db.query(Message)
        .filter(Message.id == message_id, Message.chat_id == chat_id, Message.role == "assistant")
        .first()
    )
    if not msg:
        raise HTTPException(status_code=404, detail="Message not found")

    # Strip the base64 context prefix stored in legacy messages
    content = msg.content
    if "__LLM_RESPONSE__" in content:
        content = content.split("__LLM_RESPONSE__", 1)[1]

    from app.services.export import export_message as _export
    try:
        data, media_type, filename = _export(content, format)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Export failed: {exc}")

    return Response(
        content=data,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{chat_id}/export")
def export_chat(
    *,
    db: Session = Depends(get_db),
    chat_id: int,
    current_user: User = Depends(get_current_user),
) -> Response:
    """Export all messages of a chat as a Markdown file."""
    chat = (
        db.query(Chat)
        .filter(Chat.id == chat_id, _chat_owner_filter(current_user))
        .first()
    )
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    msgs = (
        db.query(Message)
        .filter(Message.chat_id == chat_id)
        .order_by(Message.id)
        .all()
    )

    lines = [f"# {chat.title}\n"]
    for msg in msgs:
        content = msg.content
        if "__LLM_RESPONSE__" in content:
            content = content.split("__LLM_RESPONSE__", 1)[1]
        if msg.role == "user":
            lines.append(f"**User**\n\n{content.strip()}\n")
        else:
            # Include rewritten query when it differs from the preceding user message
            header = "**Assistant**"
            if msg.rewritten_query:
                header += f"\n\n> \u26fa *Rewritten query: {msg.rewritten_query.strip()}*"
            lines.append(f"{header}\n\n{content.strip()}\n")

    md = "\n---\n\n".join(lines)
    return Response(
        content=md,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="chat-{chat_id}.md"'},
    )
