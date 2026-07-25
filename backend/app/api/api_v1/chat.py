import asyncio
import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import List, Any, Literal, Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import Response, StreamingResponse
from sqlalchemy.orm import Session, joinedload
from app.db.session import get_db
from app.models.user import User
from app.models.chat import Chat, Message, ChatFile
from app.core.storage import delete_ephemeral_chat_files
from app.models.knowledge import KnowledgeBase
from app.api.api_v1.rbac import chat_owner_filter as _chat_owner_filter
from app.schemas.chat import (
    ChatCreate,
    ChatResponse,
    ChatUpdate,
    MessageEditRequest,
    MessageResponse,
    SearchResult,
)
from app.core.security import get_current_user
from app.services.chat import generate_response, get_effective_llm_config
from app.services.infrastructure import set_cancel_token
from app.services.ingestion import SUPPORTED_EXTENSIONS
from app.services.ingestion import MAX_FILE_SIZE, _convert_to_markdown
from app.services.agentic_rag.redis_memory import delete_chat_redis_sync

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("", response_model=ChatResponse)
def create_chat(
    *,
    db: Session = Depends(get_db),
    chat_in: ChatCreate,
    current_user: User = Depends(get_current_user)
) -> Any:
    # Verify knowledge bases exist and belong to user
    knowledge_bases = (
        db.query(KnowledgeBase)
        .filter(
            KnowledgeBase.id.in_(chat_in.knowledge_base_ids),
            KnowledgeBase.user_id == current_user.id,
        )
        .all()
    )
    if len(knowledge_bases) != len(chat_in.knowledge_base_ids):
        raise HTTPException(
            status_code=400,
            detail="One or more knowledge bases not found"
        )
    
    chat = Chat(
        title=chat_in.title,
        user_id=current_user.id,
        org_id=current_user.org_id,
    )
    chat.knowledge_bases = knowledge_bases
    
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return chat

@router.get("", response_model=List[ChatResponse])
def get_chats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    skip: int = 0,
    limit: int = 100
) -> Any:
    chats = (
        db.query(Chat)
        .filter(_chat_owner_filter(current_user))
        .offset(skip)
        .limit(limit)
        .all()
    )
    return chats



@router.get("/search", response_model=List[SearchResult])
def search_chats(
    q: str = Query(..., min_length=1),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Full-text search across messages in the current user's chats."""
    import time as _time
    from sqlalchemy import text as sa_text
    from sqlalchemy.exc import OperationalError

    t0 = _time.monotonic()
    mode = "fulltext"
    rows = []

    # Get all chat ids belonging to this user
    chat_ids = [
        r.id for r in db.query(Chat.id).filter(_chat_owner_filter(current_user)).all()
    ]
    if not chat_ids:
        logger.info("[SEARCH] query=%r result_count=0 latency_ms=0 mode=%s", q, mode)
        return []

    from sqlalchemy import bindparam as _bp
    try:
        if len(q) < 4:
            raise OperationalError("query too short for FULLTEXT", None, None)
        sql = sa_text(
            "SELECT m.id AS message_id, m.chat_id, m.content, c.title AS chat_title "
            "FROM messages m JOIN chats c ON c.id = m.chat_id "
            "WHERE m.chat_id IN :chat_ids "
            "AND MATCH(m.content) AGAINST(:q IN BOOLEAN MODE) "
            "LIMIT 20"
        ).bindparams(_bp("chat_ids", expanding=True))
        result = db.execute(sql, {"chat_ids": chat_ids, "q": q})
        rows = result.fetchall()
    except OperationalError:
        mode = "like"
        sql = sa_text(
            "SELECT m.id AS message_id, m.chat_id, m.content, c.title AS chat_title "
            "FROM messages m JOIN chats c ON c.id = m.chat_id "
            "WHERE m.chat_id IN :chat_ids "
            "AND m.content LIKE :pat "
            "LIMIT 20"
        ).bindparams(_bp("chat_ids", expanding=True))
        result = db.execute(sql, {"chat_ids": chat_ids, "pat": f"%{q}%"})
        rows = result.fetchall()

    latency_ms = round((_time.monotonic() - t0) * 1000)
    logger.info(
        "[SEARCH] query=%r result_count=%d latency_ms=%d mode=%s",
        q, len(rows), latency_ms, mode,
    )
    return [
        SearchResult(
            chat_id=row.chat_id,
            chat_title=row.chat_title,
            snippet=row.content[:120],
            message_id=row.message_id,
        )
        for row in rows
    ]


@router.post("/{chat_id}/cancel")
def cancel_chat_stream(
    *,
    db: Session = Depends(get_db),
    chat_id: int,
    current_user: User = Depends(get_current_user)
) -> Any:
    """Signal cancellation for an in-progress streaming response.

    Sets the cancel token in the registry so that the streaming pipeline
    can detect cancellation on its next iteration step.
    """
    chat = (
        db.query(Chat)
        .filter(
            Chat.id == chat_id,
            _chat_owner_filter(current_user)
        )
        .first()
    )
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    set_cancel_token(chat_id)
    logger.info("[CHAT] cancelled chat_id=%d user_id=%d", chat_id, current_user.id)
    return {"status": "cancelled"}


@router.get("/{chat_id}", response_model=ChatResponse)
def get_chat(
    *,
    db: Session = Depends(get_db),
    chat_id: int,
    include_messages: bool = Query(True),
    current_user: User = Depends(get_current_user)
) -> Any:
    chat = (
        db.query(Chat)
        .filter(
            Chat.id == chat_id,
            _chat_owner_filter(current_user)
        )
        .first()
    )
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    from app.schemas.chat import ChatResponse
    chat_data = ChatResponse.model_validate(chat).model_dump()

    if not include_messages:
        chat_data["messages"] = []
        return chat_data

    # Build a file info lookup: message_id → (file_id, file_name)
    file_map: dict[int, tuple[int, str]] = {}
    chat_files = (
        db.query(ChatFile)
        .filter(ChatFile.chat_id == chat_id, ChatFile.message_id.isnot(None))
        .all()
    )
    for cf in chat_files:
        if cf.message_id:
            file_map[cf.message_id] = (cf.id, cf.file_name)

    for msg_dict in chat_data.get("messages", []):
        info = file_map.get(msg_dict["id"])
        msg_dict["file_name"] = info[1] if info else None
        msg_dict["file_id"]   = info[0] if info else None

    return chat_data


@router.get("/{chat_id}/messages/paginated")
def get_messages_paginated(
    *,
    db: Session = Depends(get_db),
    chat_id: int,
    limit: int = Query(20, ge=1, le=100),
    before_id: Optional[int] = Query(None),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Returns up to `limit` messages ending just before `before_id` (cursor).
    Messages are returned in ascending ID order (oldest first).
    `has_more=True` means there are older messages available.
    """
    chat = (
        db.query(Chat)
        .filter(Chat.id == chat_id, _chat_owner_filter(current_user))
        .first()
    )
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    q = db.query(Message).filter(Message.chat_id == chat_id)
    if before_id is not None:
        q = q.filter(Message.id < before_id)

    # Fetch newest-first so we get the correct page, then reverse for display
    messages = q.order_by(Message.id.desc()).limit(limit).all()
    messages = list(reversed(messages))  # chronological order

    # has_more = there exist messages older than what we just returned
    has_more = False
    if messages:
        oldest_id = messages[0].id
        has_more = (
            db.query(Message.id)
            .filter(Message.chat_id == chat_id, Message.id < oldest_id)
            .first()
        ) is not None

    # Build file_map for this page only
    msg_ids = [m.id for m in messages]
    file_map: dict[int, tuple[int, str]] = {}
    if msg_ids:
        chat_files = (
            db.query(ChatFile)
            .filter(ChatFile.chat_id == chat_id, ChatFile.message_id.in_(msg_ids))
            .all()
        )
        for cf in chat_files:
            if cf.message_id:
                file_map[cf.message_id] = (cf.id, cf.file_name)

    from app.schemas.chat import MessageResponse
    from app.models.chat import MessageCitation
    from app.models.knowledge import DocumentChunk
    result = []
    for msg in messages:
        msg_dict = MessageResponse.model_validate(msg).model_dump()
        info = file_map.get(msg.id)
        msg_dict["file_name"] = info[1] if info else None
        msg_dict["file_id"]   = info[0] if info else None

        # Reconstruct citations from message_citations table
        citations = (
            db.query(MessageCitation)
            .filter(MessageCitation.message_id == msg.id)
            .order_by(MessageCitation.citation_index)
            .all()
        )
        if citations:
            msg_dict["citations"] = []
            for cit in citations:
                # Look up chunk by (document_id, chunk_index) for the citation text
                chunk = (
                    db.query(DocumentChunk)
                    .filter(
                        DocumentChunk.document_id == cit.document_id,
                        DocumentChunk.chunk_index == cit.chunk_index,
                    )
                    .first()
                )
                if chunk:
                    # Keep citation_metadata as-is from DB (raw fields from 2: event)
                    # Only add "text" for the citation content
                    entry = {**(cit.citation_metadata or {})}  # type: ignore[misc]
                    entry["text"] = chunk.chunk_text
                    msg_dict["citations"].append(entry)
        else:
            msg_dict["citations"] = []

        result.append(msg_dict)

    return {"messages": result, "has_more": has_more}

@router.post("/{chat_id}/messages")
async def create_message(
    *,
    db: Session = Depends(get_db),
    chat_id: int,
    messages: dict,
    current_user: User = Depends(get_current_user)
) -> StreamingResponse:
    chat = (
        db.query(Chat)
        .options(joinedload(Chat.knowledge_bases))
        .filter(
            Chat.id == chat_id,
            _chat_owner_filter(current_user)
        )
        .first()
    )
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Get the last user message
    last_message = messages["messages"][-1]
    if last_message["role"] != "user":
        raise HTTPException(status_code=400, detail="Last message must be from user")

    # Optional per-request overrides
    temperature: float = float(messages.get("temperature", 0.0))
    model_name: Optional[str] = messages.get("model_name") or None
    file_id: Optional[int] = messages.get("file_id") or None

    # ── File context injection ─────────────────────────────────────────────────
    # Build augmented query from: attached file (current turn) + any files from
    # prior turns in the same chat (multi-turn context).
    query_text = last_message["content"]
    display_query = query_text          # shown in UI / stored in DB
    file_context_parts: list[str] = []

    # Current-turn file
    current_file_markdown: Optional[str] = None
    if file_id:
        chat_file = (
            db.query(ChatFile)
            .filter(ChatFile.id == file_id, ChatFile.chat_id == chat_id)
            .first()
        )
        if not chat_file:
            raise HTTPException(status_code=404, detail="File not found")
        if chat_file.status == "processing":
            raise HTTPException(status_code=409, detail="File is still processing. Please wait and retry.")
        if chat_file.status == "error":
            raise HTTPException(status_code=422, detail=chat_file.error_message or "File processing failed.")
        if chat_file.markdown_content:
            current_file_markdown = chat_file.markdown_content
            file_context_parts.append(f"## Attached File: {chat_file.file_name}\n\n{chat_file.markdown_content}")

    # Prior-turn files in this chat (multi-turn context)
    prior_files = (
        db.query(ChatFile)
        .filter(
            ChatFile.chat_id == chat_id,
            ChatFile.status == "ready",
            ChatFile.message_id.isnot(None),   # already linked to a sent message
            ChatFile.id != file_id if file_id else True,
        )
        .order_by(ChatFile.id.asc())
        .all()
    )
    for pf in prior_files:
        if pf.markdown_content:
            file_context_parts.append(f"## Previously Uploaded File: {pf.file_name}\n\n{pf.markdown_content}")

    if file_context_parts:
        query_text = "\n\n".join(file_context_parts) + "\n\n" + query_text

    knowledge_base_ids = [kb.id for kb in chat.knowledge_bases]
    llm_cfg = get_effective_llm_config(current_user.org_id, db)

    async def response_stream():
        async for chunk in generate_response(
            query=query_text,
            messages=messages,
            knowledge_base_ids=knowledge_base_ids,
            chat_id=chat_id,
            db=db,
            temperature=temperature,
            model_name=model_name,
            display_query=display_query,
            file_id=file_id,
            file_markdown=current_file_markdown,
            api_base=llm_cfg["api_base"],
            query_model=llm_cfg["query_model"],
            org_id=current_user.org_id,
            user_id=current_user.id,
        ):
            yield chunk

    return StreamingResponse(
        response_stream(),
        media_type="text/event-stream",
        headers={
            "x-vercel-ai-data-stream": "v1",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/{chat_id}/messages/with-file")
async def create_message_with_file(
    *,
    db: Session = Depends(get_db),
    chat_id: int,
    file: UploadFile = File(...),
    message: str = Form(...),
    messages: str = Form(...),
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Accept multipart/form-data with a file + message; prepend file content as context."""
    import json as _json

    chat = (
        db.query(Chat)
        .options(joinedload(Chat.knowledge_bases))
        .filter(Chat.id == chat_id, _chat_owner_filter(current_user))
        .first()
    )
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    # ── Validate file type ────────────────────────────────────────────────────
    filename = file.filename or "upload"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    # ── Read & validate file size ─────────────────────────────────────────────
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"File exceeds 10 MB limit ({len(file_bytes) / 1024 / 1024:.1f} MB).",
        )

    # ── Convert to markdown via a temp file ──────────────────────────────────
    tmp_path: Optional[str] = None
    try:
        suffix = ext if ext else ".tmp"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        file_content = _convert_to_markdown(tmp_path, filename)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    logger.info("[with-file] converted '%s' → %d chars of markdown", filename, len(file_content))

    # ── Build augmented query ─────────────────────────────────────────────────
    augmented_query = f"## File Context: {filename}\n\n{file_content}\n\n{message}"

    # ── Parse messages JSON ───────────────────────────────────────────────────
    try:
        messages_data = _json.loads(messages)
    except Exception:
        raise HTTPException(status_code=400, detail="'messages' must be valid JSON.")

    knowledge_base_ids = [kb.id for kb in chat.knowledge_bases]
    llm_cfg = get_effective_llm_config(current_user.org_id, db)

    async def response_stream():
        async for chunk in generate_response(
            query=augmented_query,
            messages=messages_data,
            knowledge_base_ids=knowledge_base_ids,
            chat_id=chat_id,
            db=db,
            display_query=message,
            api_base=llm_cfg["api_base"],
            query_model=llm_cfg["query_model"],
            org_id=current_user.org_id,
            user_id=current_user.id,
        ):
            yield chunk

    return StreamingResponse(
        response_stream(),
        media_type="text/event-stream",
        headers={
            "x-vercel-ai-data-stream": "v1",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete("/{chat_id}/messages/{message_id}")
def delete_message(
    *,
    db: Session = Depends(get_db),
    chat_id: int,
    message_id: int,
    current_user: User = Depends(get_current_user),
) -> Any:
    """Delete a single assistant message (and the user message that prompted it)."""
    chat = (
        db.query(Chat)
        .filter(Chat.id == chat_id, _chat_owner_filter(current_user))
        .first()
    )
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    msg = (
        db.query(Message)
        .filter(Message.id == message_id, Message.chat_id == chat_id)
        .first()
    )
    if not msg:
        raise HTTPException(status_code=404, detail="Message not found")

    db.delete(msg)
    db.commit()
    return {"status": "success"}


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
                header += f"\n\n> ⛲ *Rewritten query: {msg.rewritten_query.strip()}*"
            lines.append(f"{header}\n\n{content.strip()}\n")

    md = "\n---\n\n".join(lines)
    return Response(
        content=md,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="chat-{chat_id}.md"'},
    )


@router.patch("/{chat_id}", response_model=ChatResponse)
def update_chat(
    *,
    db: Session = Depends(get_db),
    chat_id: int,
    chat_in: ChatUpdate,
    current_user: User = Depends(get_current_user)
) -> Any:
    chat = (
        db.query(Chat)
        .filter(
            Chat.id == chat_id,
            _chat_owner_filter(current_user)
        )
        .first()
    )
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    if chat_in.title is not None:
        chat.title = chat_in.title
    if chat_in.pinned is not None:
        chat.pinned = chat_in.pinned
    db.commit()
    db.refresh(chat)
    return chat


@router.delete("/{chat_id}")
def delete_chat(
    *,
    db: Session = Depends(get_db),
    chat_id: int,
    current_user: User = Depends(get_current_user)
) -> Any:
    chat = (
        db.query(Chat)
        .filter(
            Chat.id == chat_id,
            _chat_owner_filter(current_user)
        )
        .first()
    )
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    db.delete(chat)
    db.commit()
    # Clean up ephemeral uploaded files for this chat
    delete_ephemeral_chat_files(chat_id)
    # Clean up Redis checkpoints and long-term memory for this chat
    delete_chat_redis_sync(chat_id)
    return {"status": "success"}

@router.patch("/messages/{message_id}", response_model=MessageResponse)
def edit_message(
    *,
    db: Session = Depends(get_db),
    message_id: int,
    body: MessageEditRequest,
    current_user: User = Depends(get_current_user),
) -> Any:
    """Edit a sent user message by creating a new branch.

    The original message is kept intact. A new Message row is created with
    the same ``parent_message_id`` as the original (or the original message's
    own id when it has no parent) and an incremented ``branch_index``.  This
    preserves the full conversation history while giving the UI a clean new
    branch to stream the assistant response into.
    """
    original = (
        db.query(Message)
        .join(Chat, Chat.id == Message.chat_id)
        .filter(Message.id == message_id, _chat_owner_filter(current_user))
        .first()
    )
    if not original:
        raise HTTPException(status_code=404, detail="Message not found")

    # Determine the shared parent for this branch group
    shared_parent_id = original.parent_message_id if original.parent_message_id is not None else original.id

    # Find the highest branch_index among existing siblings
    max_branch = (
        db.query(Message.branch_index)
        .filter(
            Message.parent_message_id == shared_parent_id,
            Message.chat_id == original.chat_id,
        )
        .order_by(Message.branch_index.desc())
        .first()
    )
    next_index = (max_branch[0] + 1) if max_branch else 1

    new_message = Message(
        content=body.content,
        role=original.role,
        chat_id=original.chat_id,
        parent_message_id=shared_parent_id,
        branch_index=next_index,
    )
    db.add(new_message)
    db.commit()
    db.refresh(new_message)
    logger.info(
        "message.branch_created chat_id=%s original_id=%s new_id=%s branch_index=%s",
        original.chat_id, original.id, new_message.id, next_index,
    )
    return new_message


@router.get("/messages/{message_id}/siblings", response_model=List[MessageResponse])
def get_message_siblings(
    *,
    db: Session = Depends(get_db),
    message_id: int,
    current_user: User = Depends(get_current_user),
) -> Any:
    """Return all branch siblings of a message (messages sharing the same parent).

    The response includes the original message (branch_index=0) and all
    edited variants, ordered by branch_index ascending.
    """
    target = (
        db.query(Message)
        .join(Chat, Chat.id == Message.chat_id)
        .filter(Message.id == message_id, _chat_owner_filter(current_user))
        .first()
    )
    if not target:
        raise HTTPException(status_code=404, detail="Message not found")

    # The root of the branch group
    shared_parent_id = target.parent_message_id if target.parent_message_id is not None else target.id

    # If target IS the root (no parent), include itself + all children
    if target.parent_message_id is None:
        siblings = (
            db.query(Message)
            .filter(
                (Message.id == shared_parent_id) |
                (Message.parent_message_id == shared_parent_id),
                Message.chat_id == target.chat_id,
            )
            .order_by(Message.branch_index)
            .all()
        )
    else:
        siblings = (
            db.query(Message)
            .filter(
                (Message.id == shared_parent_id) |
                (Message.parent_message_id == shared_parent_id),
                Message.chat_id == target.chat_id,
            )
            .order_by(Message.branch_index)
            .all()
        )

    return siblings


# ── Clarification State Machine ─────────────────────────────────────────────────

from app.models.clarification import ClarificationRequest as ClarificationRequestModel


class ClarificationSubmitRequest(BaseModel):
    """Request body for user clarification response."""
    chat_id: int
    clarification_id: int  # The ClarificationRequest.id from the pending endpoint
    response: str          # User's clarification answer


class ClarificationPendingResponse(BaseModel):
    """Response from the pending clarification check."""
    pending: bool
    question: str = ""
    options: list[str] = []
    rationale: str = ""
    clarification_id: int = 0
    attempt: int = 1
    max_attempts: int = 2


@router.get("/clarification/pending", response_model=ClarificationPendingResponse)
async def get_pending_clarification(
    *,
    db: Session = Depends(get_db),
    chat_id: int = Query(...),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Check if there is a pending clarification request for this chat.

    The frontend polls this endpoint every 2 seconds while waiting for
    the agent to respond. If the agent needs clarification, it creates
    a ClarificationRequest row and yields a 'c:' SSE event. The frontend
    polls this endpoint to get the question details.
    """
    # Verify chat ownership
    chat = (
        db.query(Chat)
        .filter(
            Chat.id == chat_id,
            _chat_owner_filter(current_user),
        )
        .first()
    )
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Find the most recent pending clarification for this chat
    pending = (
        db.query(ClarificationRequestModel)
        .filter(
            ClarificationRequestModel.chat_id == chat_id,
            ClarificationRequestModel.status == "pending",
        )
        .order_by(ClarificationRequestModel.created_at.desc())
        .first()
    )

    if not pending:
        return ClarificationPendingResponse(pending=False)

    return ClarificationPendingResponse(
        pending=True,
        question=pending.question or "",
        options=pending.options or [],
        rationale=pending.rationale or "",
        clarification_id=pending.id,
        attempt=pending.attempt,
        max_attempts=2,
    )


@router.post("/clarification", response_model=dict)
async def submit_clarification(
    *,
    db: Session = Depends(get_db),
    body: ClarificationSubmitRequest,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Submit user's clarification response to an in-progress agent query.

    This endpoint is called when the user responds to a clarification request.
    It updates the ClarificationRequest row with the user's response, marks it
    as answered, and resumes the paused LangGraph execution with
    Command(resume=...).
    """
    # Verify chat ownership
    chat = (
        db.query(Chat)
        .filter(
            Chat.id == body.chat_id,
            _chat_owner_filter(current_user),
        )
        .first()
    )
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Find and verify the clarification request
    clarification = (
        db.query(ClarificationRequestModel)
        .filter(
            ClarificationRequestModel.id == body.clarification_id,
            ClarificationRequestModel.chat_id == body.chat_id,
            ClarificationRequestModel.status == "pending",
        )
        .first()
    )
    if not clarification:
        raise HTTPException(status_code=404, detail="Clarification request not found or already answered")

    # Update the clarification request with user response
    clarification.user_response = body.response
    clarification.status = "answered"
    clarification.answered_at = datetime.now(timezone.utc)

    # Store clarification response as a user message so the LLM sees it
    # in the normal conversation history (same as any other user query).
    clarification_msg = Message(
        content=body.response,
        role="user",
        chat_id=body.chat_id,
    )

    db.add(clarification_msg)
    db.commit()
    db.refresh(clarification)

    # Resume the paused LangGraph execution.
    # The thread_id matches what run_agentic_rag() used.
    # Command(resume=...) becomes the return value of interrupt() inside
    # request_clarification_node, which then routes back to classify_query.
    from langgraph.types import Command
    from app.services.agentic_rag.graph import build_main_graph
    from app.services.agentic_rag.redis_memory import get_redis_memory
    from app.services.agentic_rag.streaming import AgenticRAGTransformer

    memory = await get_redis_memory()
    thread_id = f"chat-{body.chat_id}"
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Build the graph with the same config as the original run.
        # We need the kb_ids and file_markdown from the chat to resume properly.
        kb_ids = [kb.id for kb in chat.knowledge_bases]
        graph = build_main_graph(
            db=None,
            kb_ids=kb_ids,
            org_id=current_user.org_id,
            checkpointer=memory.checkpointer,
            store=memory.store,
        )

        # Resume the interrupted graph — the resume value becomes the
        # return value of interrupt() in request_clarification_node.
        # The graph will re-run from classify_query with the clarification
        # response now in the checkpoint history.
        resumed_stream = await graph.astream_events(
            Command(resume=body.response),
            config=config,
            version="v3",
            transformers=[AgenticRAGTransformer],
        )

        # Drain the resumed stream — this re-runs the full pipeline
        # (rewrite -> classify -> retrieve -> generate) and waits for completion.
        async def _drain_resumed() -> None:
            async for _ in resumed_stream:
                pass

        asyncio.create_task(_drain_resumed())
        # Wait for the resumed stream to complete
        await resumed_stream.output()

        logger.info(
            "[CLARIFICATION] chat_id=%d clarification_id=%d resumed successfully | user=%d",
            body.chat_id, clarification.id, current_user.id,
        )

    except Exception as exc:
        logger.error(
            "[CLARIFICATION] chat_id=%d clarification_id=%d resume failed: %s",
            body.chat_id, clarification.id, exc, exc_info=True,
        )

    logger.info(
        "[CLARIFICATION] chat_id=%d clarification_id=%d response_id=%d user=%d",
        body.chat_id, clarification.id, clarification_msg.id, current_user.id,
    )

    return {
        "status": "received",
        "clarification_id": clarification.id,
        "message": "Clarification response received. Agent is processing your answer.",
    }