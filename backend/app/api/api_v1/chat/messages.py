"""Message endpoints — creation, pagination, editing, and branch siblings.

Handles both the text-only and multipart file-attachment message creation
paths, cursor-based paginated retrieval with active-branch resolution,
message deletion, branch-creating edits, and sibling listing for the
branch-switcher UI.
"""
import logging
import os
import tempfile
from typing import Any, Optional

from fastapi import Depends, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session, joinedload

from app.api.api_v1.chat import router
from app.db.session import get_db
from app.models.user import User
from app.models.chat import Chat, Message, ChatFile
from app.api.api_v1.rbac import chat_owner_filter as _chat_owner_filter
from app.schemas.chat import (
    MessageEditRequest,
    MessageResponse,
)
from app.core.security import get_current_user
from app.services.chat import generate_response
from app.services.ingestion import SUPPORTED_EXTENSIONS
from app.services.ingestion import MAX_FILE_SIZE, _convert_to_markdown

logger = logging.getLogger(__name__)


def _select_active_branch_messages(
    all_user_msgs: list[Message],
    active_branches: dict,
) -> list[Message]:
    # Group by branch parent and pick active branch per group
    # Branch parent = parent_message_id if set, else the message itself
    branch_groups: dict[int, list[Message]] = {}
    for um in all_user_msgs:
        group_key = um.parent_message_id if um.parent_message_id is not None else um.id
        branch_groups.setdefault(group_key, []).append(um)

    selected_user_msgs: list[Message] = []
    for group_key, siblings in branch_groups.items():
        if len(siblings) == 1:
            # No branching — just use the single message
            selected_user_msgs.append(siblings[0])
        else:
            # Pick the active branch, or default to latest (highest branch_index)
            active_id = active_branches.get(str(group_key))
            selected = None
            if active_id:
                for s in siblings:
                    if s.id == active_id:
                        selected = s
                        break
            if selected is None:
                selected = max(siblings, key=lambda m: m.branch_index)
            selected_user_msgs.append(selected)
    return selected_user_msgs


def _collect_turn_messages(
    paged_user_msgs: list[Message],
    db: Session,
    chat_id: int,
) -> list[Message]:
    result_msgs: list[Message] = []
    for um in paged_user_msgs:
        result_msgs.append(um)
        assistant = (
            db.query(Message)
            .filter(
                Message.parent_message_id == um.id,
                Message.chat_id == chat_id,
                Message.role == "assistant",
            )
            .order_by(Message.id.desc())
            .first()
        )
        if assistant:
            result_msgs.append(assistant)
    return result_msgs


def _build_message_file_map(
    db: Session,
    chat_id: int,
    msg_ids: list[int],
) -> dict[int, tuple[int, str]]:
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
    return file_map


def _serialize_messages_with_citations(
    result_msgs: list[Message],
    db: Session,
    file_map: dict[int, tuple[int, str]],
) -> list[dict]:
    from app.schemas.chat import MessageResponse
    from app.models.chat import MessageCitation
    from app.models.knowledge import DocumentChunk
    result = []
    for msg in result_msgs:
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
                chunk = (
                    db.query(DocumentChunk)
                    .filter(
                        DocumentChunk.document_id == cit.document_id,
                        DocumentChunk.chunk_index == cit.chunk_index,
                    )
                    .first()
                )
                if chunk:
                    entry = {**(cit.citation_metadata or {})}  # type: ignore[misc]
                    entry["text"] = chunk.chunk_text
                    msg_dict["citations"].append(entry)
        else:
            msg_dict["citations"] = []

        result.append(msg_dict)
    return result


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

    Branch handling: when a user message has branch siblings (edited
    variants), only the active branch is returned — the one selected by
    the user (tracked in chat.active_branches), or the latest branch
    (highest branch_index) by default. The paired assistant reply (linked
    via parent_message_id) is included right after each user message.
    """
    chat = (
        db.query(Chat)
        .filter(Chat.id == chat_id, _chat_owner_filter(current_user))
        .first()
    )
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    active_branches = chat.active_branches or {}

    # -- Build the list of "turns" (user message + paired assistant) --------
    # A "turn root" is a user message that is either:
    #   - a root message (parent_message_id is None), or
    #   - a branch child (parent_message_id points to another user message)
    # For branch groups, only the active/latest branch is included.

    # Step 1: find all user messages in this chat
    user_msgs_q = db.query(Message).filter(
        Message.chat_id == chat_id,
        Message.role == "user",
    )
    if before_id is not None:
        user_msgs_q = user_msgs_q.filter(Message.id < before_id)
    all_user_msgs = user_msgs_q.order_by(Message.id.asc()).all()

    # Step 2: group by branch parent and pick active branch per group
    selected_user_msgs = _select_active_branch_messages(all_user_msgs, active_branches)

    # Step 3: sort by id (chronological) and apply limit
    selected_user_msgs.sort(key=lambda m: m.id)
    # Paginate: take the last `limit` turns (newest), then reverse for chronological
    paged_user_msgs = selected_user_msgs[-limit:]

    # has_more: are there older turns beyond what we returned?
    has_more = len(selected_user_msgs) > len(paged_user_msgs)

    # Step 4: for each selected user message, find its paired assistant reply
    result_msgs = _collect_turn_messages(paged_user_msgs, db, chat_id)

    # Step 5: build file_map and serialize (same as before)
    msg_ids = [m.id for m in result_msgs]
    file_map = _build_message_file_map(db, chat_id, msg_ids)
    result = _serialize_messages_with_citations(result_msgs, db, file_map)

    return {"messages": result, "has_more": has_more}

def _build_message_file_context(
    db: Session,
    chat_id: int,
    file_id: Optional[int],
    query_text: str,
) -> tuple[str, Optional[str]]:
    """Build augmented query with file context from current and prior turns.

    Returns (augmented_query, current_file_markdown).
    """
    file_context_parts: list[str] = []
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

    prior_files = (
        db.query(ChatFile)
        .filter(
            ChatFile.chat_id == chat_id,
            ChatFile.status == "ready",
            ChatFile.message_id.isnot(None),
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
    return query_text, current_file_markdown


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
    file_id: Optional[int] = messages.get("file_id") or None
    # When branching (edit), the frontend sends the branched user message's
    # DB id so the assistant reply can be linked to it via parent_message_id.
    parent_message_id: Optional[int] = messages.get("parent_message_id") or None

    # -- File context injection -----------------------------------------------
    # Build augmented query from: attached file (current turn) + any files from
    # prior turns in the same chat (multi-turn context).
    query_text = last_message["content"]
    display_query = query_text          # shown in UI / stored in DB
    query_text, current_file_markdown = _build_message_file_context(
        db, chat_id, file_id, query_text
    )

    knowledge_base_ids = [kb.id for kb in chat.knowledge_bases]

    async def response_stream():
        async for chunk in generate_response(
            query=query_text,
            messages=messages,
            knowledge_base_ids=knowledge_base_ids,
            chat_id=chat_id,
            db=db,
            display_query=display_query,
            file_id=file_id,
            file_markdown=current_file_markdown,
            org_id=current_user.org_id,
            user_id=current_user.id,
            parent_message_id=parent_message_id,
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

    # -- Validate file type ----------------------------------------------------
    filename = file.filename or "upload"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    # -- Read & validate file size ---------------------------------------------
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"File exceeds 10 MB limit ({len(file_bytes) / 1024 / 1024:.1f} MB).",
        )

    # -- Convert to markdown via a temp file -----------------------------------
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

    logger.debug("[with-file] converted '%s' -> %d chars of markdown", filename, len(file_content))

    # -- Build augmented query -------------------------------------------------
    augmented_query = f"## File Context: {filename}\n\n{file_content}\n\n{message}"

    # -- Parse messages JSON ---------------------------------------------------
    try:
        messages_data = _json.loads(messages)
    except Exception:
        raise HTTPException(status_code=400, detail="'messages' must be valid JSON.")

    # Parse optional per-message KB scope
    knowledge_base_ids = [kb.id for kb in chat.knowledge_bases]

    async def response_stream():
        async for chunk in generate_response(
            query=augmented_query,
            messages=messages_data,
            knowledge_base_ids=knowledge_base_ids,
            chat_id=chat_id,
            db=db,
            display_query=message,
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


@router.patch("/{chat_id}/messages/{message_id}", response_model=MessageResponse)
def edit_message(
    *,
    db: Session = Depends(get_db),
    chat_id: int,
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
        .filter(Message.id == message_id, Message.chat_id == chat_id, _chat_owner_filter(current_user))
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


@router.get("/{chat_id}/messages/{message_id}/siblings")
def get_message_siblings(
    *,
    db: Session = Depends(get_db),
    chat_id: int,
    message_id: int,
    current_user: User = Depends(get_current_user),
) -> Any:
    """Return all branch siblings of a user message, each paired with its
    assistant reply.

    Response: list of {user: MessageResponse, assistant: MessageResponse|null}
    ordered by branch_index ascending.
    """
    target = (
        db.query(Message)
        .join(Chat, Chat.id == Message.chat_id)
        .filter(Message.id == message_id, Message.chat_id == chat_id, _chat_owner_filter(current_user))
        .first()
    )
    if not target:
        raise HTTPException(status_code=404, detail="Message not found")

    # The root of the branch group
    shared_parent_id = target.parent_message_id if target.parent_message_id is not None else target.id

    # All user-message siblings (the original + edited branches)
    user_siblings = (
        db.query(Message)
        .filter(
            (Message.id == shared_parent_id) |
            (Message.parent_message_id == shared_parent_id),
            Message.chat_id == target.chat_id,
            Message.role == "user",
        )
        .order_by(Message.branch_index)
        .all()
    )

    # For each user sibling, find its paired assistant reply
    result = []
    for user_msg in user_siblings:
        assistant_msg = (
            db.query(Message)
            .filter(
                Message.parent_message_id == user_msg.id,
                Message.chat_id == chat_id,
                Message.role == "assistant",
            )
            .order_by(Message.id.desc())
            .first()
        )
        from app.schemas.chat import MessageResponse
        result.append({
            "user": MessageResponse.model_validate(user_msg).model_dump(),
            "assistant": MessageResponse.model_validate(assistant_msg).model_dump() if assistant_msg else None,
        })

    return result
