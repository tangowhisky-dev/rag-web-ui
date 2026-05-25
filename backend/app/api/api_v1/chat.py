import logging
import os
import tempfile
import time
from typing import List, Any, Literal, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import StreamingResponse, Response
from sqlalchemy.orm import Session, joinedload
from app.db.session import get_db
from app.models.user import User
from app.models.chat import Chat, Message, ChatFile
from app.core.storage import delete_ephemeral_chat_files
from app.models.knowledge import KnowledgeBase
from app.schemas.chat import (
    ChatCreate,
    ChatResponse,
    ChatUpdate,
    MessageCreate,
    MessageEditRequest,
    MessageResponse,
    SearchResult,
)
from app.core.security import get_current_user
from app.services.chat_service import generate_response
from app.services.document_processor import _convert_to_markdown, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

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
            KnowledgeBase.user_id == current_user.id
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
        use_graph_rag=chat_in.use_graph_rag,
        use_dense=chat_in.use_dense,
        use_sparse=chat_in.use_sparse,
        use_exact=chat_in.use_exact,
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
        .filter(Chat.user_id == current_user.id)
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
        r.id for r in db.query(Chat.id).filter(Chat.user_id == current_user.id).all()
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


@router.get("/{chat_id}", response_model=ChatResponse)
def get_chat(
    *,
    db: Session = Depends(get_db),
    chat_id: int,
    current_user: User = Depends(get_current_user)
) -> Any:
    chat = (
        db.query(Chat)
        .filter(
            Chat.id == chat_id,
            Chat.user_id == current_user.id
        )
        .first()
    )
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

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

    # Build the response manually so file_name/file_id are reliably serialized.
    # Relying on transient attributes on SQLAlchemy objects is not safe with FastAPI's
    # jsonable_encoder when no response_model is declared.
    from app.schemas.chat import ChatResponse, MessageResponse

    # Serialize via Pydantic, then patch file info into message dicts
    chat_data = ChatResponse.model_validate(chat).model_dump()
    for msg_dict in chat_data.get("messages", []):
        info = file_map.get(msg_dict["id"])
        msg_dict["file_name"] = info[1] if info else None
        msg_dict["file_id"]   = info[0] if info else None

    return chat_data

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
            Chat.user_id == current_user.id
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

    async def response_stream():
        # After generate_response creates the user Message, link the chat_file to it
        user_msg_id: Optional[int] = None
        async for chunk in generate_response(
            query=query_text,
            messages=messages,
            knowledge_base_ids=knowledge_base_ids,
            chat_id=chat_id,
            db=db,
            use_dense=chat.use_dense,
            use_sparse=chat.use_sparse,
            use_exact=chat.use_exact,
            use_graph_rag=chat.use_graph_rag,
            temperature=temperature,
            model_name=model_name,
            display_query=display_query,
            file_id=file_id,
            file_markdown=current_file_markdown,
        ):
            yield chunk

    return StreamingResponse(
        response_stream(),
        media_type="text/event-stream",
        headers={"x-vercel-ai-data-stream": "v1"},
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
        .filter(Chat.id == chat_id, Chat.user_id == current_user.id)
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

    async def response_stream():
        async for chunk in generate_response(
            query=augmented_query,
            messages=messages_data,
            knowledge_base_ids=knowledge_base_ids,
            chat_id=chat_id,
            db=db,
            use_dense=chat.use_dense,
            use_sparse=chat.use_sparse,
            use_exact=chat.use_exact,
            use_graph_rag=chat.use_graph_rag,
            display_query=message,
        ):
            yield chunk

    return StreamingResponse(
        response_stream(),
        media_type="text/event-stream",
        headers={"x-vercel-ai-data-stream": "v1"},
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
        .filter(Chat.id == chat_id, Chat.user_id == current_user.id)
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
        .filter(Chat.id == chat_id, Chat.user_id == current_user.id)
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

    from app.services.export_service import export_message as _export
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
        .filter(Chat.id == chat_id, Chat.user_id == current_user.id)
        .first()
    )
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    msgs = (
        db.query(Message)
        .filter(Message.chat_id == chat_id)
        .order_by(Message.created_at)
        .all()
    )

    lines = [f"# {chat.title}\n"]
    for msg in msgs:
        content = msg.content
        if "__LLM_RESPONSE__" in content:
            content = content.split("__LLM_RESPONSE__", 1)[1]
        label = "**User**" if msg.role == "user" else "**Assistant**"
        lines.append(f"{label}\n\n{content.strip()}\n")

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
            Chat.user_id == current_user.id
        )
        .first()
    )
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    if chat_in.title is not None:
        chat.title = chat_in.title
    if chat_in.pinned is not None:
        chat.pinned = chat_in.pinned
    if chat_in.use_dense is not None:
        chat.use_dense = chat_in.use_dense
    if chat_in.use_sparse is not None:
        chat.use_sparse = chat_in.use_sparse
    if chat_in.use_exact is not None:
        chat.use_exact = chat_in.use_exact
    if chat_in.use_graph_rag is not None:
        chat.use_graph_rag = chat_in.use_graph_rag
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
            Chat.user_id == current_user.id
        )
        .first()
    )
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    db.delete(chat)
    db.commit()
    # Clean up ephemeral uploaded files for this chat
    delete_ephemeral_chat_files(chat_id)
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
        .filter(Message.id == message_id, Chat.user_id == current_user.id)
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
        .filter(Message.id == message_id, Chat.user_id == current_user.id)
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