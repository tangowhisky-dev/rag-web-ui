import logging
import os
import tempfile
from typing import List, Any, Literal, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import StreamingResponse, Response
from sqlalchemy.orm import Session, joinedload
from app.db.session import get_db
from app.models.user import User
from app.models.chat import Chat, Message
from app.models.knowledge import KnowledgeBase
from app.schemas.chat import (
    ChatCreate,
    ChatResponse,
    ChatUpdate,
    MessageCreate,
    MessageResponse
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
    return chat

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

    # Get knowledge base IDs
    knowledge_base_ids = [kb.id for kb in chat.knowledge_bases]

    async def response_stream():
        async for chunk in generate_response(
            query=last_message["content"],
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
        ):
            yield chunk

    return StreamingResponse(
        response_stream(),
        media_type="text/event-stream",
        headers={
            "x-vercel-ai-data-stream": "v1"
        }
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
    return {"status": "success"}