"""Centralized ingestion dispatch.

All ingestion triggers (KB upload, watcher events, manual scan, startup
recovery, retry endpoint) call through here instead of directly invoking
``process_document_background`` with their own event-loop boilerplate.

This eliminates the duplicated loop-creation / task-status-update logic
that was copy-pasted across four modules.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.knowledge import ProcessingTask
from app.services.ingestion.document_processor import process_document_background

logger = logging.getLogger(__name__)


def run_ingestion_in_thread(
    file_path: str,
    file_name: str,
    task_id: int,
    document_id: int,
    kb_id: Optional[int] = None,
    data_store_id: Optional[int] = None,
    enable_ocr: Optional[bool] = None,
    user_id: Optional[int] = None,
    file_hash: Optional[str] = None,
    file_size: Optional[int] = None,
    content_type: Optional[str] = None,
) -> None:
    """Run ingestion in a brand-new event loop (for thread contexts).

    Creates a new asyncio loop, runs ``process_document_background`` to
    completion, then updates the ProcessingTask status in a fresh session.

    This is the single replacement for the three duplicated ``_run_ingestion``
    implementations in watcher.py, handler.py, and startup_recovery_service.py.
    """
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)

        async def _do() -> None:
            await process_document_background(
                temp_path=file_path,
                file_name=file_name,
                kb_id=kb_id,
                task_id=task_id,
                document_id=document_id,
                data_store_id=data_store_id,
                enable_ocr=enable_ocr,
                user_id=user_id,
                file_hash=file_hash,
                file_size=file_size,
                content_type=content_type,
                file_path=file_path if data_store_id is not None else None,
                db=None,
            )

        loop.run_until_complete(_do())

        _mark_task_status(task_id, "completed", progress=100,
                          message="Ingestion completed")
        logger.info(
            "ingestion_completed task_id=%s path=%s",
            task_id, file_path,
        )
    except Exception as e:
        logger.error(
            "ingestion_failed task_id=%s error=%s",
            task_id, e, exc_info=True,
        )
        _mark_task_status(task_id, "failed", progress=0,
                          message=f"Ingestion failed: {e}")
        raise
    finally:
        loop.close()


def _mark_task_status(
    task_id: int,
    status: str,
    progress: int = 0,
    message: str = "",
) -> None:
    """Update ProcessingTask status in a fresh session (best-effort)."""
    try:
        db: Session = SessionLocal()
        try:
            task = db.query(ProcessingTask).filter(
                ProcessingTask.id == task_id
            ).first()
            if task:
                task.status = status
                task.progress = progress
                task.progress_message = message
                db.commit()
        finally:
            db.close()
    except Exception:
        pass
