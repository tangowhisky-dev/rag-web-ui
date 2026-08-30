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
import time
from typing import Dict, Optional, Set

from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.knowledge import ProcessingTask, Document
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

# Cancellation registry: task_id → threading.Event.
# When a scan is cancelled, events are set for all in-flight graph builds
# belonging to that datastore.  The graph build loop checks the event
# between extraction batches and aborts if set.
_graph_cancel_events: Dict[int, threading.Event] = {}
# Reverse index: datastore_id → set of task_ids with in-flight graph builds.
_graph_builds_by_datastore: Dict[int, Set[int]] = {}
_graph_cancel_lock = threading.Lock()

# Global semaphore limiting concurrent graph build threads.
# Prevents thread explosion when many documents complete ingestion
# simultaneously (e.g. 1000-file scan).  Threads waiting here consume
# no GPU — they're blocked on the semaphore, not making LLM calls.
# The LLM call concurrency is separately capped by the global LLM
# semaphore in graph_service.py.
_global_graph_thread_sem = threading.Semaphore(8)

# Track in-flight ingestions by datastore so delete can wait for them.
# datastore_id → set of task_ids currently being ingested.
_active_ingestions_by_datastore: Dict[int, Set[int]] = {}
_active_ingestions_lock = threading.Lock()


def register_ingestion(datastore_id: int, task_id: int) -> None:
    """Register an in-flight ingestion for a datastore."""
    with _active_ingestions_lock:
        _active_ingestions_by_datastore.setdefault(datastore_id, set()).add(task_id)


def unregister_ingestion(datastore_id: int, task_id: int) -> None:
    """Remove an in-flight ingestion after completion."""
    with _active_ingestions_lock:
        tasks = _active_ingestions_by_datastore.get(datastore_id)
        if tasks:
            tasks.discard(task_id)
            if not tasks:
                del _active_ingestions_by_datastore[datastore_id]


def wait_for_ingestions(datastore_id: int, timeout: float = 30.0) -> int:
    """Wait for all in-flight ingestions for a datastore to finish.

    Returns the number of ingestions that were still running when timeout
    was reached (0 = all finished).
    """
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with _active_ingestions_lock:
            tasks = _active_ingestions_by_datastore.get(datastore_id)
            if not tasks:
                return 0
            remaining = len(tasks)
        if remaining == 0:
            return 0
        time.sleep(0.5)
    with _active_ingestions_lock:
        tasks = _active_ingestions_by_datastore.get(datastore_id, set())
        return len(tasks)


def is_datastore_deleted(datastore_id: int) -> bool:
    """Check if a datastore has been deleted from the database."""
    from app.models.datastore import DataStore
    db: Session = SessionLocal()
    try:
        return db.query(DataStore).filter(DataStore.id == datastore_id).first() is None
    finally:
        db.close()


def _is_graph_ingestion_paused(datastore_id: int) -> bool:
    """Check if graph ingestion is paused for a datastore."""
    from app.models.datastore import DataStore
    db: Session = SessionLocal()
    try:
        ds = db.query(DataStore).filter(DataStore.id == datastore_id).first()
        return ds is not None and bool(ds.graph_ingestion_paused)
    finally:
        db.close()


def cancel_graph_builds_for_datastore(datastore_id: int) -> int:
    """Signal cancellation for all in-flight graph builds belonging to a datastore.

    Also marks pending (not-yet-started) graph builds as failed in the DB so
    the recovery service won't retry them.

    Returns the number of builds cancelled.
    """
    cancelled = 0

    # 1. Signal in-flight graph builds to stop via threading.Event.
    with _graph_cancel_lock:
        task_ids = list(_graph_builds_by_datastore.get(datastore_id, set()))
    for task_id in task_ids:
        event = _graph_cancel_events.get(task_id)
        if event:
            event.set()
            cancelled += 1

    # 2. Mark pending graph builds as failed in DB so recovery won't retry.
    try:
        db: Session = SessionLocal()
        try:
            tasks = db.query(ProcessingTask).filter(
                ProcessingTask.data_store_id == datastore_id,
                ProcessingTask.graph_status == "pending",
            ).all()
            for t in tasks:
                t.graph_status = "failed"
                t.graph_error = "Cancelled by admin"
                cancelled += 1
            db.commit()
        finally:
            db.close()
    except Exception:
        pass

    return cancelled


def _clear_needs_reprocess(document_id: Optional[int]) -> None:
    if document_id is None:
        return
    try:
        clear_db = SessionLocal()
        try:
            doc = clear_db.query(Document).filter(Document.id == document_id).first()
            if doc and doc.needs_reprocess:
                doc.needs_reprocess = False
                clear_db.commit()
        finally:
            clear_db.close()
    except Exception:
        pass


def _maybe_start_graph_build(
    graph_request: Optional[GraphBuildRequest],
    task_id: int,
) -> None:
    if graph_request is None:
        return
    task_status = _get_task_status(task_id)
    if task_status == "completed":
        if graph_request.data_store_id is not None:
            if is_datastore_deleted(graph_request.data_store_id):
                logger.info(
                    "graph_build_skipped task_id=%s — datastore %s deleted",
                    task_id, graph_request.data_store_id,
                )
                return
            graph_status = _get_graph_status(task_id)
            if graph_status == "failed":
                logger.info(
                    "graph_build_skipped task_id=%s — scan cancelled",
                    task_id,
                )
                return
        _start_graph_build_thread(graph_request)
    else:
        logger.warning(
            "graph_build_skipped task_id=%s status=%s (not completed)",
            task_id, task_status,
        )


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
    skip_conversion: bool = False,
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
    ds_id_for_tracking = data_store_id

    # Register this ingestion so delete can wait for it
    if ds_id_for_tracking is not None:
        register_ingestion(ds_id_for_tracking, task_id)

    try:
        asyncio.set_event_loop(loop)

        # Bail out early if the datastore was already deleted
        if ds_id_for_tracking is not None and is_datastore_deleted(ds_id_for_tracking):
            logger.info(
                "ingestion_skipped task_id=%s — datastore %s deleted",
                task_id, ds_id_for_tracking,
            )
            _mark_task_status(task_id, "failed", progress=0,
                              message="Datastore deleted")
            return

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
                skip_conversion=skip_conversion,
            )

        graph_request = loop.run_until_complete(_do())

        _mark_task_status(task_id, "completed", progress=100,
                          message="Ingestion completed")

        _clear_needs_reprocess(document_id)

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
        try:
            loop.close()
        except Exception:
            pass
        # Unregister this ingestion so delete can proceed
        if ds_id_for_tracking is not None:
            unregister_ingestion(ds_id_for_tracking, task_id)

    _maybe_start_graph_build(graph_request, task_id)


def _acquire_graph_thread_slot(
    cancel_event: threading.Event, req: GraphBuildRequest,
) -> bool:
    while True:
        if cancel_event.is_set():
            logger.info(
                "graph_build_skipped task_id=%s — cancelled while waiting for thread slot",
                req.task_id,
            )
            _set_graph_status(req.task_id, "pending")
            return False
        if _global_graph_thread_sem.acquire(blocking=False):
            return True
        time.sleep(0.1)


def _check_graph_build_eligible(req: GraphBuildRequest) -> bool:
    existing_status = _get_graph_status(req.task_id)
    if existing_status == _TASK_NOT_FOUND:
        logger.info(
            "graph_build_skipped task_id=%s — task no longer exists",
            req.task_id,
        )
        return False
    if existing_status == "failed":
        logger.info(
            "graph_build_skipped task_id=%s — already cancelled",
            req.task_id,
        )
        return False
    if req.data_store_id is not None and is_datastore_deleted(req.data_store_id):
        logger.info(
            "graph_build_skipped task_id=%s — datastore %s deleted",
            req.task_id, req.data_store_id,
        )
        return False
    if req.data_store_id is not None and _is_graph_ingestion_paused(req.data_store_id):
        logger.info(
            "graph_build_skipped task_id=%s — graph ingestion paused for datastore %s",
            req.task_id, req.data_store_id,
        )
        _set_graph_status(req.task_id, "pending")
        return False
    return True


def _execute_graph_build(req: GraphBuildRequest, cancel_event: threading.Event) -> None:
    _set_graph_status(req.task_id, "pending")

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)

        async def _do() -> int:
            from app.services.graph import build_graph_for_document
            from app.services.graph.graph_service import delete_graph_for_document

            delete_graph_for_document(
                kb_id=req.kb_id,
                document_id=req.document_id,
                data_store_id=req.data_store_id,
            )

            skipped = await build_graph_for_document(
                kb_id=req.kb_id,
                document_id=req.document_id,
                file_name=req.file_name,
                chunks=req.chunks,
                chunk_ids=req.chunk_ids,
                data_store_id=req.data_store_id,
                task_id=req.task_id,
                cancel_event=cancel_event,
            )
            return skipped

        skipped_batches = loop.run_until_complete(_do())

        if skipped_batches is not None and skipped_batches > 0:
            _set_graph_status(req.task_id, "pending", error=None)
            logger.info(
                "graph_build_partial task_id=%s document_id=%s skipped_batches=%d — marked pending for retry",
                req.task_id, req.document_id, skipped_batches,
            )
        else:
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
        try:
            loop.close()
        except Exception:
            pass


def _cleanup_graph_build(req: GraphBuildRequest, sem_acquired: bool) -> None:
    if sem_acquired:
        _global_graph_thread_sem.release()
    with _active_graph_lock:
        _active_graph_builds.discard(req.task_id)
    with _graph_cancel_lock:
        _graph_cancel_events.pop(req.task_id, None)
        if req.data_store_id is not None:
            _graph_builds_by_datastore.get(req.data_store_id, set()).discard(req.task_id)


def run_graph_build_in_thread(req: GraphBuildRequest) -> None:
    """Run Neo4j graph build in a dedicated thread with its own event loop.

    Updates ProcessingTask.graph_status to pending → completed/failed.
    Calls delete_graph_for_document first to ensure idempotency on retry.
    Skips if a graph build for the same task is already in-flight.
    Aborts early if the task's graph_status was already set to "failed"
    (e.g. cancelled by admin before the thread started).
    """
    with _active_graph_lock:
        if req.task_id in _active_graph_builds:
            logger.info(
                "graph_build_skipped task_id=%s — already in-flight",
                req.task_id,
            )
            return
        _active_graph_builds.add(req.task_id)

    cancel_event = threading.Event()
    with _graph_cancel_lock:
        _graph_cancel_events[req.task_id] = cancel_event
        if req.data_store_id is not None:
            _graph_builds_by_datastore.setdefault(req.data_store_id, set()).add(req.task_id)

    _thread_sem_acquired = False
    try:
        _thread_sem_acquired = _acquire_graph_thread_slot(cancel_event, req)
        if not _thread_sem_acquired:
            return

        if not _check_graph_build_eligible(req):
            return

        _execute_graph_build(req, cancel_event)
    finally:
        _cleanup_graph_build(req, _thread_sem_acquired)


_TASK_NOT_FOUND = "__task_not_found__"


def _get_graph_status(task_id: int) -> str:
    """Read the current graph_status for a task (best-effort).

    Returns _TASK_NOT_FOUND if the task row doesn't exist, or the
    actual graph_status value (which may be None, "pending",
    "completed", or "failed").
    """
    try:
        db: Session = SessionLocal()
        try:
            task = db.query(ProcessingTask).filter(
                ProcessingTask.id == task_id
            ).first()
            return task.graph_status if task else _TASK_NOT_FOUND
        finally:
            db.close()
    except Exception:
        return _TASK_NOT_FOUND


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


def _get_task_status(task_id: int) -> str | None:
    """Read the current ProcessingTask status (best-effort)."""
    try:
        db: Session = SessionLocal()
        try:
            task = db.query(ProcessingTask).filter(
                ProcessingTask.id == task_id
            ).first()
            return task.status if task else None
        finally:
            db.close()
    except Exception:
        return None
