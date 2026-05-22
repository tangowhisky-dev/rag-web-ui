"""
Chat-scoped file upload API.

Flow:
  POST /api/chat/{chat_id}/files          → upload file, start async markdown conversion
  GET  /api/chat/{chat_id}/files/{file_id} → poll status
  DELETE /api/chat/{chat_id}/files/{file_id} → remove file record
"""

import asyncio
import logging
import os
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.api.api_v1.auth import get_current_user
from app.core.config import settings
from app.core.storage import save_ephemeral_file, delete_ephemeral_chat_files
from app.db.session import get_db
from app.models.chat import Chat, ChatFile
from app.models.user import User

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
    ".txt", ".md", ".html", ".htm", ".csv", ".json", ".xml", ".eml", ".epub",
    ".jpg", ".jpeg", ".png", ".gif", ".zip",
}


def _convert_to_markdown(tmp_path: str, filename: str) -> str:
    """Convert a file to markdown using markitdown."""
    try:
        from markitdown import MarkItDown
        md = MarkItDown()
        result = md.convert(tmp_path)
        return result.text_content or ""
    except Exception as exc:
        logger.warning("[chat_files] markitdown failed for %s: %s", filename, exc)
        # Fallback: read as plain text
        try:
            with open(tmp_path, "r", errors="replace") as f:
                return f.read()
        except Exception:
            return ""


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token (English average)."""
    return max(1, len(text) // 4)


def _file_token_budget() -> int:
    """25% of total context window reserved for file content."""
    return settings.OPENAI_MODEL_CONTEXT_SIZE // 4


async def _process_file(db_session_factory, file_id: int, tmp_path: str, filename: str) -> None:
    """Background task: convert file → markdown, estimate tokens, update DB."""
    from app.db.session import SessionLocal
    db: Session = SessionLocal()
    try:
        chat_file = db.query(ChatFile).filter(ChatFile.id == file_id).first()
        if not chat_file:
            return

        loop = asyncio.get_event_loop()
        markdown = await loop.run_in_executor(None, _convert_to_markdown, tmp_path, filename)
        token_count = _estimate_tokens(markdown)
        budget = _file_token_budget()

        if token_count > budget:
            chat_file.status = "error"
            chat_file.error_message = (
                f"File content is too large for the context window "
                f"({token_count:,} tokens > {budget:,} token budget). "
                f"Please upload a smaller file or split it into parts."
            )
            # Remove markdown to save space; metadata kept for error display
            chat_file.markdown_content = None
            chat_file.token_count = token_count
            logger.warning(
                "[chat_files] file_id=%d too large: %d tokens > budget %d",
                file_id, token_count, budget,
            )
        else:
            chat_file.markdown_content = markdown
            chat_file.token_count = token_count
            chat_file.status = "ready"
            logger.info(
                "[chat_files] file_id=%d ready: %d tokens (%s)",
                file_id, token_count, filename,
            )

        db.commit()
    except Exception as exc:
        logger.error("[chat_files] processing error for file_id=%d: %s", file_id, exc)
        try:
            cf = db.query(ChatFile).filter(ChatFile.id == file_id).first()
            if cf:
                cf.status = "error"
                cf.error_message = str(exc)
                db.commit()
        except Exception:
            pass
    finally:
        db.close()
        # Original file kept; entire ephemeral/{chat_id}/ dir is removed on chat delete


@router.post("/{chat_id}/files")
async def upload_chat_file(
    chat_id: int,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Upload a file to a chat. Starts async markdown conversion immediately."""
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == current_user.id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    filename = file.filename or "upload"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"File exceeds 10 MB limit ({len(file_bytes) / 1024 / 1024:.1f} MB).",
        )

    # Insert the DB row first so we have a unique id to use as the on-disk filename prefix.
    # This prevents collisions when the same filename is uploaded multiple times to the same chat.
    chat_file = ChatFile(
        chat_id=chat_id,
        file_name=filename,
        file_size=len(file_bytes),
        content_type=file.content_type or "application/octet-stream",
        status="processing",
    )
    db.add(chat_file)
    db.commit()
    db.refresh(chat_file)

    # Stored as {file_id}_{original_filename} — unique even if the same name is re-uploaded
    stored_name = f"{chat_file.id}_{filename}"
    stored_path = save_ephemeral_file(chat_id, stored_name, file_bytes)

    background_tasks.add_task(_process_file, None, chat_file.id, stored_path, filename)

    return {
        "id": chat_file.id,
        "file_name": chat_file.file_name,
        "file_size": chat_file.file_size,
        "status": chat_file.status,
    }


@router.get("/{chat_id}/files/{file_id}")
def get_chat_file_status(
    chat_id: int,
    file_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Poll processing status of a chat file."""
    chat_file = (
        db.query(ChatFile)
        .filter(ChatFile.id == file_id, ChatFile.chat_id == chat_id)
        .first()
    )
    if not chat_file:
        raise HTTPException(status_code=404, detail="File not found")

    # Verify ownership via chat
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == current_user.id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    return {
        "id": chat_file.id,
        "file_name": chat_file.file_name,
        "file_size": chat_file.file_size,
        "token_count": chat_file.token_count,
        "status": chat_file.status,
        "error_message": chat_file.error_message,
    }


@router.delete("/{chat_id}/files/{file_id}", status_code=204)
def delete_chat_file(
    chat_id: int,
    file_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a chat file record."""
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == current_user.id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    chat_file = db.query(ChatFile).filter(ChatFile.id == file_id, ChatFile.chat_id == chat_id).first()
    if chat_file:
        db.delete(chat_file)
        db.commit()
