"""Per-document management endpoints (3-phase pipeline editor).

GET  /datastores/{id}/documents/{doc}/markdown       — get converted markdown
PUT  /datastores/{id}/documents/{doc}/markdown       — save edited markdown (optimistic lock)
POST /datastores/{id}/documents/{doc}/reconvert      — re-run conversion from source file
GET  /datastores/{id}/documents/{doc}/ingest-status  — current ingestion/conversion/graph status

All endpoints are admin-only and enforce organisation scope.
"""

import logging
import os

from fastapi import Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.core.security import get_admin_org_ids, require_admin
from app.db.session import get_db
from app.models.user import User

from app.api.api_v1.datastores import router, _datastore_in_scope
from app.api.api_v1.datastores.schemas import UpdateMarkdownRequest
from app.api.api_v1.datastores.crud import _get_datastore_or_404

logger = logging.getLogger(__name__)


def _get_document_or_404(db: Session, document_id: int) -> "Document":
    from app.models.knowledge import Document
    doc = db.query(Document).filter(Document.id == document_id).first()
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


def _verify_document_in_datastore(db: Session, datastore_id: int, document_id: int):
    """Ensure the document belongs to the given datastore."""
    doc = _get_document_or_404(db, document_id)
    if doc.data_store_id != datastore_id:
        raise HTTPException(status_code=404, detail="Document not found in this datastore")
    return doc


@router.get("/datastores/{datastore_id}/documents/{document_id}/markdown")
def get_document_markdown(
    datastore_id: int,
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Get the converted markdown for a document (editor source)."""
    admin_org_ids = get_admin_org_ids(db, current_user)
    _get_datastore_or_404(db, datastore_id)
    if not _datastore_in_scope(db, datastore_id, admin_org_ids):
        raise HTTPException(status_code=404, detail="DataStore not found")

    doc = _verify_document_in_datastore(db, datastore_id, document_id)

    if doc.conversion_status == "processing":
        raise HTTPException(status_code=409, detail="Conversion in progress")
    if doc.conversion_status == "pending":
        raise HTTPException(status_code=409, detail="Conversion pending")
    if not doc.converted_markdown:
        if doc.conversion_status == "error":
            raise HTTPException(
                status_code=422,
                detail=f"Conversion failed: {doc.conversion_error or 'unknown error'}",
            )
        raise HTTPException(
            status_code=409,
            detail="Markdown not available — run re-convert first",
        )

    return {
        "document_id": doc.id,
        "markdown": doc.converted_markdown,
        "conversion_status": doc.conversion_status,
        "lock_version": doc.lock_version,
        "title": doc.title,
    }


@router.put("/datastores/{datastore_id}/documents/{document_id}/markdown")
def update_document_markdown(
    datastore_id: int,
    document_id: int,
    body: UpdateMarkdownRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Save edited markdown and earmark for reprocessing.

    1. Validate non-empty markdown.
    2. Optimistic lock check.
    3. Persist new markdown.
    4. Set needs_reprocess=True so the next scan re-ingests using
       the saved markdown (without re-converting the source file).
    Returns 202 Accepted.
    """
    admin_org_ids = get_admin_org_ids(db, current_user)
    _get_datastore_or_404(db, datastore_id)
    if not _datastore_in_scope(db, datastore_id, admin_org_ids):
        raise HTTPException(status_code=404, detail="DataStore not found")

    doc = _verify_document_in_datastore(db, datastore_id, document_id)

    # Optimistic lock check
    if doc.lock_version != body.lock_version:
        raise HTTPException(
            status_code=409,
            detail=f"Document was modified by another editor. Expected lock_version={doc.lock_version}, got {body.lock_version}.",
        )

    # Check conversion is done
    if not doc.converted_markdown and doc.conversion_status != "completed":
        raise HTTPException(
            status_code=409,
            detail="Document has not been converted yet — run re-convert first",
        )

    # Persist new markdown + bump lock version + earmark for reprocessing
    doc.converted_markdown = body.markdown
    doc.conversion_status = "completed"
    doc.lock_version = doc.lock_version + 1
    doc.needs_reprocess = True
    db.commit()

    logger.info(
        "[EDITOR] markdown_saved doc_id=%s datastore_id=%s — earmarked for reprocessing",
        document_id, datastore_id,
    )

    return JSONResponse(
        status_code=202,
        content={
            "document_id": document_id,
            "lock_version": doc.lock_version,
            "needs_reprocess": True,
            "message": "Markdown saved. File will be re-ingested on next process cycle.",
        },
    )


@router.post("/datastores/{datastore_id}/documents/{document_id}/reconvert")
def reconvert_document(
    datastore_id: int,
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Re-run conversion from the source file. Overwrites current markdown."""
    admin_org_ids = get_admin_org_ids(db, current_user)
    _get_datastore_or_404(db, datastore_id)
    if not _datastore_in_scope(db, datastore_id, admin_org_ids):
        raise HTTPException(status_code=404, detail="DataStore not found")

    doc = _verify_document_in_datastore(db, datastore_id, document_id)

    if not os.path.isfile(doc.file_path):
        raise HTTPException(status_code=404, detail="Source file not found on disk")

    # Capture plain strings before starting the thread — the SQLAlchemy
    # object will be detached once the request session closes.
    _file_path = doc.file_path
    _file_name = doc.file_name

    import threading

    def _do_reconvert():
        import asyncio
        from app.db.session import SessionLocal
        from app.services.ingestion.document_processor import convert_document
        from app.services.infrastructure.progress_timeout import ProgressTimeout

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            rdb = SessionLocal()
            try:
                def _on_timeout():
                    try:
                        from app.models.knowledge import Document as _Doc
                        d = rdb.query(_Doc).filter(_Doc.id == document_id).first()
                        if d:
                            d.conversion_status = "error"
                            d.conversion_error = "Conversion timeout — no activity for 600s"
                            rdb.commit()
                    except Exception:
                        rdb.rollback()

                async def _run_with_timeout():
                    async with ProgressTimeout(
                        silence_seconds=600,
                        on_timeout=_on_timeout,
                    ):
                        return await convert_document(
                            document_id=document_id,
                            file_path=_file_path,
                            file_name=_file_name,
                            db=rdb,
                        )

                try:
                    loop.run_until_complete(_run_with_timeout())
                    # Mark for reprocessing — the markdown was regenerated
                    # and needs to be re-ingested on next process cycle.
                    from app.models.knowledge import Document as _Doc
                    d = rdb.query(_Doc).filter(_Doc.id == document_id).first()
                    if d:
                        d.needs_reprocess = True
                        rdb.commit()
                except Exception as e:
                    logger.error("reconvert_failed document_id=%s: %s", document_id, e)
                    try:
                        from app.models.knowledge import Document as _Doc
                        d = rdb.query(_Doc).filter(_Doc.id == document_id).first()
                        if d:
                            d.conversion_status = "error"
                            d.conversion_error = str(e)[:500]
                            rdb.commit()
                    except Exception:
                        rdb.rollback()
            finally:
                rdb.close()
        finally:
            loop.close()

    doc.conversion_status = "pending"
    db.commit()

    t = threading.Thread(target=_do_reconvert, name=f"reconvert-{document_id}", daemon=True)
    t.start()

    return JSONResponse(
        status_code=202,
        content={
            "document_id": document_id,
            "conversion_status": "pending",
            "message": "Re-convert queued",
        },
    )


@router.get("/datastores/{datastore_id}/documents/{document_id}/ingest-status")
def get_ingest_status(
    datastore_id: int,
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Get the current ingestion/conversion/graph status for a document."""
    admin_org_ids = get_admin_org_ids(db, current_user)
    _get_datastore_or_404(db, datastore_id)
    if not _datastore_in_scope(db, datastore_id, admin_org_ids):
        raise HTTPException(status_code=404, detail="DataStore not found")

    doc = _verify_document_in_datastore(db, datastore_id, document_id)

    # Get latest task
    from app.models.knowledge import ProcessingTask
    latest_task = (
        db.query(ProcessingTask)
        .filter(ProcessingTask.document_id == document_id)
        .order_by(ProcessingTask.id.desc())
        .first()
    )

    from app.models.knowledge import DocumentChunk
    chunk_count = db.query(DocumentChunk).filter(
        DocumentChunk.document_id == document_id,
        DocumentChunk.data_store_id == datastore_id,
    ).count()

    return {
        "document_id": doc.id,
        "conversion_status": doc.conversion_status,
        "conversion_error": doc.conversion_error,
        "ingest_status": latest_task.status if latest_task else None,
        "ingest_progress": latest_task.progress if latest_task else 0,
        "ingest_message": latest_task.progress_message if latest_task else None,
        "ingest_error": latest_task.error_message if latest_task else None,
        "graph_status": latest_task.graph_status if latest_task else None,
        "graph_error": latest_task.graph_error if latest_task else None,
        "chunk_count": chunk_count,
        "lock_version": doc.lock_version,
    }
