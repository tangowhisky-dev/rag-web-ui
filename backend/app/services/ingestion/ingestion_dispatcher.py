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
import threading
from typing import Optional

from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.knowledge import ProcessingTask
from app.services.ingestion.document_processor import (
    process_document_background,
    GraphBuildRequest,
)

logger = logging.getLogger(__name__)

# Track in-flight graph builds to prevent duplicate workers for the same task.
# Guarded by _active_graph_lock.  A task_id is added when a graph build starts
# and removed when it finishes (success or failure).  If a second caller tries
# to start a graph build for a task that's already in-flight, it's skipped.
_active_graph_builds: set[int] = set()
_active_graph_lock = threading.Lock()


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
    completion (parse → embed → Qdrant upsert), then updates the
    ProcessingTask status.  If the function returns a GraphBuildRequest,
    fires the Neo4j graph build in a separate daemon thread so the
    ingestion future resolves immediately at Qdrant upsert.

    This is the single replacement for the three duplicated ``_run_ingestion``
    implementations in watcher.py, handler.py, and startup_recovery_service.py.
    """
    loop = asyncio.new_event_loop()
    graph_request: Optional[GraphBuildRequest] = None
    try:
        asyncio.set_event_loop(loop)

        async def _do() -> Optional[GraphBuildRequest]:
            return await process_document_background(
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

        graph_request = loop.run_until_complete(_do())

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

    # Fire graph build in a separate daemon thread with its own event loop.
    # This decouples graph extraction (slow, LLM-bound) from the ingestion
    # pipeline so scans complete at Qdrant upsert.  Graph status is tracked
    # via ProcessingTask.graph_status and retried by the recovery service.
    if graph_request is not None:
        _start_graph_build_thread(graph_request)


def run_graph_build_in_thread(req: GraphBuildRequest) -> None:
    """Run Neo4j graph build in a dedicated thread with its own event loop.

    Updates ProcessingTask.graph_status to pending → completed/failed.
    Calls delete_graph_for_document first to ensure idempotency on retry.
    Skips if a graph build for the same task is already in-flight.
    """
    # Deduplication guard: skip if another thread is already building this graph.
    with _active_graph_lock:
        if req.task_id in _active_graph_builds:
            logger.info(
                "graph_build_skipped task_id=%s — already in-flight",
                req.task_id,
            )
            return
        _active_graph_builds.add(req.task_id)

    try:
        _set_graph_status(req.task_id, "pending")

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)

            async def _do() -> None:
                from app.services.graph import build_graph_for_document
                from app.services.graph.graph_service import delete_graph_for_document

                # Delete existing graph nodes for this document first (idempotent
                # retry — a previous attempt may have written partial data).
                delete_graph_for_document(
                    kb_id=req.kb_id,
                    document_id=req.document_id,
                    data_store_id=req.data_store_id,
                )

                await build_graph_for_document(
                    kb_id=req.kb_id,
                    document_id=req.document_id,
                    file_name=req.file_name,
                    chunks=req.chunks,
                    chunk_ids=req.chunk_ids,
                    data_store_id=req.data_store_id,
                    task_id=req.task_id,
                )

            loop.run_until_complete(_do())
            _set_graph_status(req.task_id, "completed", error=None)
            logger.info(
                "graph_build_completed task_id=%s document_id=%s",
                req.task_id, req.document_id,
            )
        except Exception as e:
            logger.warning(
                "graph_build_failed task_id=%s document_id=%s error=%s",
                req.task_id, req.document_id, e,
                exc_info=True,
            )
            _set_graph_status(req.task_id, "failed", error=str(e)[:1000])
        finally:
            loop.close()
    finally:
        with _active_graph_lock:
            _active_graph_builds.discard(req.task_id)


def _start_graph_build_thread(req: GraphBuildRequest) -> None:
    """Launch graph build as a daemon thread so it doesn't block the caller."""
    t = threading.Thread(
        target=run_graph_build_in_thread,
        args=(req,),
        name=f"graph-build-{req.task_id}",
        daemon=True,
    )
    t.start()


def _set_graph_status(
    task_id: Optional[int], status: str, error: Optional[str] = None,
) -> None:
    """Update ProcessingTask.graph_status in a fresh session (best-effort).

    On failure, encodes the retry count as a ``[retry:N]`` prefix in
    graph_error so the recovery service can limit retry attempts.
    """
    if task_id is None:
        return
    try:
        db: Session = SessionLocal()
        try:
            task = db.query(ProcessingTask).filter(
                ProcessingTask.id == task_id
            ).first()
            if task:
                task.graph_status = status
                if status == "failed" and error:
                    # Extract current retry count from existing error prefix.
                    retries = 0
                    if task.graph_error and task.graph_error.startswith("[retry:"):
                        try:
                            retries = int(task.graph_error.split("]")[0].split(":")[1])
                        except (ValueError, IndexError):
                            pass
                    task.graph_error = f"[retry:{retries + 1}] {error[:900]}"
                else:
                    task.graph_error = error
                db.commit()
        finally:
            db.close()
    except Exception:
        pass


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
