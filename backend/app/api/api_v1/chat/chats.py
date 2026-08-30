"""Chat CRUD, search, and stream-cancellation endpoints.

Routes for creating, listing, searching, fetching, updating, and deleting
chats, plus the cancel-stream signal that lets the frontend stop an
in-progress SSE response.
"""
import logging
from typing import List, Any

from fastapi import Depends, HTTPException, Query
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func

from app.api.api_v1.chat import router
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
    SearchResult,
)
from app.core.security import get_current_user
from app.services.infrastructure import set_cancel_token
from app.services.agentic_rag.redis_memory import delete_chat_redis_sync

logger = logging.getLogger(__name__)


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
    # Sort by latest message timestamp, falling back to chat creation date.
    # Chats with no messages sort by created_at.
    latest_msg_subq = (
        db.query(
            Message.chat_id.label("chat_id"),
            func.max(Message.created_at).label("last_msg_at"),
        )
        .group_by(Message.chat_id)
        .subquery()
    )
    chats = (
        db.query(Chat)
        .outerjoin(latest_msg_subq, latest_msg_subq.c.chat_id == Chat.id)
        .filter(_chat_owner_filter(current_user))
        .order_by(func.coalesce(latest_msg_subq.c.last_msg_at, Chat.created_at).desc())
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

    # Build a file info lookup: message_id -> (file_id, file_name)
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
        .options(joinedload(Chat.knowledge_bases))
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
    if chat_in.knowledge_base_ids is not None:
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
        chat.knowledge_bases = knowledge_bases
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
    delete_chat_redis_sync(chat_id, user_id=current_user.id)
    return {"status": "success"}
