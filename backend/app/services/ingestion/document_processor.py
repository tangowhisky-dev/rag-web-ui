"""Document ingestion pipeline — upload, preview, background processing.

This is the main entry point for document processing. It imports helpers
from document_converter.py (markitdown/OCR) and document_qdrant.py
(Qdrant collections, embedding, upsert) to keep the monoliths split.
"""

import asyncio
import hashlib
import logging
import os
import traceback
from typing import Optional, List, Dict, Set, Tuple

from fastapi import UploadFile
from langchain_core.documents import Document as LangchainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import PointIdsList
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import SessionLocal
from app.core.storage import get_abs_path, save_file, move_file, delete_file
from app.services.infrastructure.progress_timeout import ProgressTimeout
from app.services.ingestion.markdown_cleaner import clean_markdown
from app.services.ingestion.document_converter import (
    SUPPORTED_EXTENSIONS,
    CONTENT_TYPE_MAP,
    _convert_to_markdown,
)
from app.services.ingestion.document_qdrant import (
    _get_qdrant_collection_name,
    _ensure_qdrant_collection,
    _chunk_id_to_point_id,
    _embed_texts_batch,
    _build_qdrant_points,
    _upsert_to_qdrant,
    UploadResult,
    TextChunk,
    PreviewResult,
)
from app.services.infrastructure import get_qdrant_client

from app.models.knowledge import ProcessingTask, Document, DocumentChunk, DocumentUpload

logger = logging.getLogger(__name__)


async def upload_document(file: UploadFile, kb_id: int, user_id: int) -> UploadResult:
    """Step 1: Upload document to local storage"""
    content = await file.read()
    file_size = len(content)

    file_hash = hashlib.sha256(content).hexdigest()

    # Clean and normalize filename
    file_name = "".join(c for c in file.filename if c.isalnum() or c in ('-', '_', '.')).strip()
    object_path = f"user_{user_id}/kb_{kb_id}/{file_name}"

    _, ext = os.path.splitext(file_name)
    ext = ext.lower()
    content_type = CONTENT_TYPE_MAP.get(ext, "application/octet-stream")

    try:
        save_file(object_path, content)
    except Exception as e:
        logging.error(f"Failed to save file to storage: {str(e)}")
        raise

    return UploadResult(
        file_path=object_path,
        file_name=file_name,
        file_size=file_size,
        content_type=content_type,
        file_hash=file_hash
    )


async def preview_document(file_path: str, chunk_size: int = None, chunk_overlap: int = None) -> PreviewResult:
    """Step 2: Generate preview chunks"""
    if chunk_size is None:
        from app.services.settings_service import get_setting
        from app.db.session import SessionLocal
        _db = SessionLocal()
        try:
            chunk_size = get_setting(_db, "CHUNK_SIZE", None)
        finally:
            _db.close()
    if chunk_overlap is None:
        from app.services.settings_service import get_setting
        from app.db.session import SessionLocal
        _db = SessionLocal()
        try:
            chunk_overlap = int(get_setting(_db, "CHUNK_SIZE", None) * get_setting(_db, "OVERLAP_PERCENTAGE", None))
        finally:
            _db.close()
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    abs_path = get_abs_path(file_path)

    try:
        # Convert to markdown using markitdown (handles all supported formats)
        markdown_text = _convert_to_markdown(abs_path, os.path.basename(file_path))

        # ── Cleanup pass ─────────────────────────────────────────────────────
        _fname = os.path.basename(file_path)
        _chars_before = len(markdown_text)
        try:
            markdown_text = clean_markdown(markdown_text)
            logger.info(
                "[CLEANUP] chars_before=%d chars_after=%d file=%s",
                _chars_before, len(markdown_text), _fname,
            )
        except Exception as _ce:
            logger.warning(
                "[CLEANUP] fallback to raw markdown. reason=%s file=%s",
                str(_ce)[:200], _fname,
            )

        # Wrap in a LangchainDocument so we can reuse RecursiveCharacterTextSplitter
        doc = LangchainDocument(
            page_content=markdown_text,
            metadata={"source": os.path.basename(file_path)},
        )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents([doc])
        preview_chunks = [
            TextChunk(
                content=chunk.page_content,
                metadata=chunk.metadata
            )
            for chunk in chunks
        ]

        return PreviewResult(
            chunks=preview_chunks,
            total_chunks=len(chunks)
        )
    except Exception as e:
        logging.error(f"Failed to preview document {file_path}: {str(e)}")
        raise


async def process_document_background(
    temp_path: str,
    file_name: str,
    kb_id: Optional[int] = None,  # None for DataStore files
    task_id: Optional[int] = None,
    db: Session = None,
    user_id: int = None,
    chunk_size: int = None,
    chunk_overlap: int = None,
    enable_ocr: Optional[bool] = None,
    document_id: Optional[int] = None,  # For updating existing documents
    data_store_id: Optional[int] = None,  # For linking to datastore
    file_path: Optional[str] = None,  # Original file path (for in-place processing)
    file_hash: Optional[str] = None,  # Pre-computed hash
    file_size: Optional[int] = None,  # Pre-computed size
    content_type: Optional[str] = None,  # Pre-computed content type
) -> None:
    """Process document in background.

    enable_ocr: None = global setting, True = force on, False = force off.
    document_id: If provided, update existing document instead of creating new one.
    data_store_id: Link document to a datastore.
    file_path: Original file path (for in-place processing, not copying).
    """
    logger = logging.getLogger(__name__)
    if db is None:
        db = SessionLocal()
        should_close_db = True
    else:
        should_close_db = False

    if chunk_size is None:
        from app.services.settings_service import get_setting
        chunk_size = get_setting(db, "CHUNK_SIZE", None)
    if chunk_overlap is None:
        from app.services.settings_service import get_setting
        chunk_overlap = int(get_setting(db, "CHUNK_SIZE", None) * get_setting(db, "OVERLAP_PERCENTAGE", None))

    task = db.get(ProcessingTask, task_id)
    if not task:
        logger.error(f"Task {task_id} not found")
        return

    # Track cleanup state so the except block always knows what to delete and
    # what DB objects to roll back — regardless of how far we got.
    permanent_path: Optional[str] = None   # set after move_file succeeds
    document: Optional[Document] = None    # set after Document record committed
    progress_db = None

    try:
        # Separate session for progress writes so progress commits don't
        # survive a main-transaction rollback (M5).
        progress_db = SessionLocal()

        def _set_progress(pct: int, msg: str) -> None:
            """Write progress to DB using a separate session so progress
            commits are independent of the main transaction."""
            try:
                ptask = progress_db.query(ProcessingTask).filter(
                    ProcessingTask.id == task_id
                ).first()
                if ptask:
                    ptask.progress = pct
                    ptask.progress_message = msg
                    progress_db.commit()
                pt.ping()
            except Exception:
                try:
                    progress_db.rollback()
                except Exception:
                    pass

        task.status = "processing"
        task.progress = 0
        task.progress_message = "Starting…"
        db.commit()

        from app.services.settings_service import get_setting
        silence_s = get_setting(db, "PROCESSING_TIMEOUT_SILENCE_S", None)

        def _on_timeout() -> None:
            logger.warning(
                "[PROGRESS_TIMEOUT] task_id=%s silence_s=%s — "
                "task may still be processing (graph build is non-fatal)",
                task_id, silence_s,
            )

        async with ProgressTimeout(silence_s, _on_timeout) as pt:

            local_temp_path = get_abs_path(temp_path)
            logger.info(f"Task {task_id}: Using file at {local_temp_path}")
    
            # ── Step 1: Parse ────────────────────────────────────────────────────
            _set_progress(5, "Parsing document…")
            logger.info(f"Task {task_id}: Converting document (enable_ocr={enable_ocr})")
            # Run in a thread pool — conversion is synchronous and CPU/IO-bound.
            # Blocking the event loop here starves the poll endpoint for 60-120s on
            # large PDFs, causing ECONNRESET storms on the frontend.
            loop = asyncio.get_event_loop()
            markdown_text = await loop.run_in_executor(
                None, lambda: _convert_to_markdown(local_temp_path, file_name, enable_ocr=enable_ocr)
            )
    
            # ── Cleanup pass ─────────────────────────────────────────────────────
            _chars_before = len(markdown_text)
            try:
                markdown_text = clean_markdown(markdown_text)
                logger.info(
                    "[CLEANUP] chars_before=%d chars_after=%d file=%s",
                    _chars_before, len(markdown_text), file_name,
                )
            except Exception as _ce:
                logger.warning(
                    "[CLEANUP] fallback to raw markdown. reason=%s file=%s",
                    str(_ce)[:200], file_name,
                )
    
            if not markdown_text or not markdown_text.strip():
                raise ValueError(
                    f"Document produced no extractable text. "
                    f"The file may be empty, password-protected, or in an unreadable format."
                )
    
            # ── Step 2: Chunk ────────────────────────────────────────────────────
            _set_progress(20, "Splitting into chunks…")
            logger.info(f"Task {task_id}: Splitting document into chunks")
            doc = LangchainDocument(
                page_content=markdown_text,
                metadata={"source": file_name},
            )
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            chunks = await loop.run_in_executor(
                None, lambda: text_splitter.split_documents([doc])
            )
            logger.info(f"Task {task_id}: Document split into {len(chunks)} chunks")
    
            if not chunks:
                raise ValueError(
                    "Document produced no chunks after splitting. "
                    "It may contain only whitespace or unsupported content."
                )
    
            # ── Step 3: Ensure Qdrant collection ─────────────────────────────────
            _set_progress(25, f"Preparing vector store ({len(chunks)} chunks)…")
            if data_store_id is not None:
                collection_name = f"ds_{data_store_id}"
                logger.info(f"Task {task_id}: Ensuring Qdrant collection {collection_name}")
            elif kb_id is not None:
                collection_name = f"kb_{kb_id}"
                logger.info(f"Task {task_id}: Ensuring Qdrant collection {collection_name}")
            else:
                logger.error(f"Task {task_id}: No collection — neither data_store_id nor kb_id provided")
                return
            _ensure_qdrant_collection(get_qdrant_client(), collection_name)
    
            # ── Step 4: Move to permanent storage (DataStore files stay in place) ───
            if data_store_id is not None:
                # DataStore: file stays in its original location
                permanent_path = file_path if file_path else temp_path
                logger.info(f"Task {task_id}: DataStore file stays in place: {permanent_path}")
            else:
                # KnowledgeBase: copy to uploads
                _permanent_path = f"user_{user_id}/kb_{kb_id}/{file_name}"
                logger.info(f"Task {task_id}: Moving file to permanent storage")
                move_file(temp_path, _permanent_path)
                permanent_path = _permanent_path          # mark: file now at permanent location
                local_perm_path = get_abs_path(permanent_path)
                logger.info(f"Task {task_id}: File moved to {permanent_path}")
    
            # ── Step 5: Create Document record ───────────────────────────────────
            logger.info(f"Task {task_id}: Creating document record")
            
            # Use provided values or fall back to task.upload values
            doc_file_path = file_path if file_path else permanent_path
            if data_store_id is not None:
                # DataStore: use pre-computed values from watcher
                doc_file_hash = file_hash if file_hash else None
                doc_file_size = file_size if file_size else None
                doc_content_type = content_type if content_type else None
            else:
                # KnowledgeBase: use task.upload values
                doc_file_hash = file_hash if file_hash else task.document_upload.file_hash
                doc_file_size = file_size if file_size else task.document_upload.file_size
                doc_content_type = content_type if content_type else task.document_upload.content_type
            
            if document_id:
                # Update existing document
                document = db.get(Document, document_id)
                if document:
                    document.file_path = doc_file_path
                    document.file_hash = doc_file_hash
                    document.file_size = doc_file_size
                    document.content_type = doc_content_type
                    document.data_store_id = data_store_id
                    document.knowledge_base_id = kb_id if kb_id else None
                    logger.info(f"Task {task_id}: Updated document ID {document.id}")
                else:
                    logger.error(f"Task {task_id}: Document {document_id} not found for update")
                    return
            else:
                # Create new document
                document = Document(
                    file_name=file_name,
                    file_path=doc_file_path,
                    file_hash=doc_file_hash,
                    file_size=doc_file_size,
                    content_type=doc_content_type,
                    knowledge_base_id=kb_id if kb_id else None,
                    data_store_id=data_store_id,
                )
                db.add(document)
                db.commit()
                db.refresh(document)
                logger.info(f"Task {task_id}: Document record created with ID {document.id}")
    
            # ── Step 6: Delete old chunks, then build new ones ───────────────────
            _set_progress(30, "Cleaning up old chunks…")
            logger.info(f"Task {task_id}: Deleting old chunks for document_id={document.id}")
            # Delete old chunks from DB (must happen before new chunks are added,
            # so that if the Qdrant upsert fails, old chunks are still present)
            # Build scope filter explicitly (M7: was a Python ternary that
            # evaluated incorrectly for edge cases).
            from sqlalchemy import and_
            if data_store_id is not None:
                scope_filter = DocumentChunk.data_store_id == data_store_id
            else:
                scope_filter = and_(
                    DocumentChunk.kb_id == kb_id,
                    DocumentChunk.data_store_id.is_(None),
                )
            old_chunk_ids = db.query(DocumentChunk).filter(
                DocumentChunk.document_id == document.id,
                scope_filter,
            ).with_entities(DocumentChunk.id).all()
            old_chunk_ids = [cid[0] for cid in old_chunk_ids]
            if old_chunk_ids:
                logger.info(f"Task {task_id}: Deleting {len(old_chunk_ids)} old chunks")
                # Delete from Qdrant first (so DB rollback doesn't orphan Qdrant points)
                point_ids = [_chunk_id_to_point_id(cid) for cid in old_chunk_ids]
                collection_name = f"ds_{data_store_id}" if data_store_id else f"kb_{kb_id}"
                get_qdrant_client().delete(
                    collection_name=collection_name,
                    points_selector=PointIdsList(points=point_ids),
                )
                # Delete from DB — rebuild scope_filter (SQLAlchemy clauses are
                # single-use after being consumed by a query)
                if data_store_id is not None:
                    scope_filter = DocumentChunk.data_store_id == data_store_id
                else:
                    scope_filter = and_(
                        DocumentChunk.kb_id == kb_id,
                        DocumentChunk.data_store_id.is_(None),
                    )
                db.query(DocumentChunk).filter(
                    DocumentChunk.document_id == document.id,
                    scope_filter,
                ).delete(synchronize_session="fetch")
                db.commit()
    
            _set_progress(35, "Building chunk records…")
            logger.info(f"Task {task_id}: Building {len(chunks)} chunk records")
    
            def _build_chunk_records():
                """CPU-bound: hashing + object construction. Returns payloads only — no db.add here."""
                payloads = []
                db_chunks = []
                for i, chunk in enumerate(chunks):
                    scope_prefix = f"ds:{data_store_id}" if data_store_id else f"kb:{kb_id}"
                    chunk_id = hashlib.sha256(
                        f"{scope_prefix}:{file_name}:{chunk.page_content}".encode()
                    ).hexdigest()
                    chunk.metadata["source"] = file_name
                    source_metadata = {
                        k: v for k, v in chunk.metadata.items()
                        if k not in ("kb_id", "document_id", "chunk_id", "file_name")
                    }
                    db_chunks.append(DocumentChunk(
                        id=chunk_id,
                        document_id=document.id,
                        kb_id=kb_id if kb_id else None,
                        data_store_id=data_store_id if data_store_id else None,
                        file_name=file_name,
                        chunk_text=chunk.page_content,
                        chunk_index=i,
                        chunk_metadata=source_metadata,
                        hash=hashlib.sha256(
                            (chunk.page_content + str(chunk.metadata)).encode()
                        ).hexdigest(),
                    ))
                    payloads.append((chunk_id, chunk.page_content, source_metadata, i))
                return payloads, db_chunks
    
            qdrant_payloads, db_chunks = await loop.run_in_executor(None, _build_chunk_records)
            # db.add must happen on the event-loop thread — SQLAlchemy sessions are not thread-safe
            for doc_chunk in db_chunks:
                db.add(doc_chunk)
    
            # ── Step 7: Upsert to Qdrant ─────────────────────────────────────────
            _set_progress(40, f"Embedding {len(qdrant_payloads)} chunks…")
            logger.info(f"Task {task_id}: Upserting {len(qdrant_payloads)} chunks to Qdrant")
            await _upsert_to_qdrant(
                qdrant_payloads, kb_id, document.id, file_name,
                data_store_id=data_store_id,
                progress_cb=_set_progress,
                progress_start=40,
                progress_end=80,
                pt=pt,
            )
            logger.info(f"Task {task_id}: Chunks added to Qdrant")
    
            # ── Step 8: Commit chunks + mark task complete ───────────────────────
            _set_progress(82, "Saving to database…")
            task.status = "completed"
            task.progress = 90
            task.progress_message = "Finalising…"
            # Commit directly — SQLAlchemy sessions are not thread-safe, running
            # db.commit() in run_in_executor risks concurrent access with the event loop.
            db.commit()
            task.document_id = document.id
            upload = task.document_upload
            if upload:
                upload.status = "completed"
            db.commit()
            logger.info("[PROGRESS_TIMEOUT] task_id=%s completed_ok=true", task_id)
            logger.info(f"Task {task_id}: Processing completed successfully")
    
            # ── Step 9: Build Neo4j knowledge graph (non-fatal, awaited) ──
            # Await the graph build directly so the event loop stays alive until
            # it completes.  This prevents "Task was destroyed but it is pending"
            # when the loop would otherwise close while the graph task is running.
            if get_setting(db, "GRAPHRAG_ENABLED", None):
                _doc_id = document.id   # capture plain int before session closes
                _chunks = [p[1] for p in qdrant_payloads]
                _chunk_ids = [p[0] for p in qdrant_payloads]
                _task_id = task_id
    
                async def _build_graph() -> None:
                    # Mark graph extraction as in-progress in the DB
                    from app.db.session import SessionLocal as _SessionLocal
                    from app.models.knowledge import ProcessingTask as _PT
                    _db = _SessionLocal()
                    try:
                        _t = _db.query(_PT).filter(_PT.id == _task_id).first()
                        if _t:
                            _t.graph_status = "pending"
                            _db.commit()
                    finally:
                        _db.close()
    
                    try:
                        from app.services.graph import build_graph_for_document
                        await build_graph_for_document(
                            kb_id=kb_id,
                            document_id=_doc_id,
                            file_name=file_name,
                            chunks=_chunks,
                            chunk_ids=_chunk_ids,
                            data_store_id=data_store_id,
                            pt=pt,
                        )
                        _db2 = _SessionLocal()
                        try:
                            _t = _db2.query(_PT).filter(_PT.id == _task_id).first()
                            if _t:
                                _t.graph_status = "completed"
                                _t.graph_error = None
                                _db2.commit()
                        finally:
                            _db2.close()
                    except Exception as _e:
                        logger.warning(
                            f"Task {task_id}: Neo4j graph build failed (non-fatal): {_e}",
                            exc_info=True,
                        )
                        _db3 = _SessionLocal()
                        try:
                            _t = _db3.query(_PT).filter(_PT.id == _task_id).first()
                            if _t:
                                _t.graph_status = "failed"
                                _t.graph_error = str(_e)[:1000]
                                _db3.commit()
                        finally:
                            _db3.close()
    
                try:
                    await _build_graph()
                    logger.info(f"Task {task_id}: Knowledge graph built in Neo4j")
                except asyncio.CancelledError:
                    logger.warning(f"Task {task_id}: Graph build cancelled")
                except Exception as _e:
                    logger.warning(
                        f"Task {task_id}: Graph build failed (non-fatal): {_e}",
                        exc_info=True,
                    )
    
    except Exception as e:
        logger.error(f"Task {task_id}: Error processing document: {str(e)}")
        logger.error(f"Task {task_id}: Stack trace: {traceback.format_exc()}")

        # ── Rollback uncommitted DB state ────────────────────────────────────
        # Any chunk records added to the session but not yet committed are
        # discarded here.  If the Document record was already committed we
        # delete it explicitly so we don't leave a document with no chunks.
        try:
            db.rollback()
        except Exception:
            pass

        if document is not None:
            try:
                db.delete(document)
                db.commit()
                logger.info(f"Task {task_id}: Document record rolled back")
            except Exception as del_err:
                logger.warning(f"Task {task_id}: Could not delete document record: {del_err}")

        # ── Mark task failed ─────────────────────────────────────────────────
        try:
            task.status = "failed"
            task.error_message = str(e)
            db.commit()
        except Exception:
            pass

        # ── Delete the file ──────────────────────────────────────────────────
        # For DataStore files, the file stays in its original location and should NOT be deleted.
        # For KB files, delete the file from uploads after processing.
        if data_store_id is None and permanent_path is not None:
            file_to_delete = permanent_path
            try:
                logger.info(f"Task {task_id}: Cleaning up file at {file_to_delete}")
                delete_file(file_to_delete)
                logger.info(f"Task {task_id}: File cleaned up")
            except Exception:
                logger.warning(f"Task {task_id}: Failed to clean up file at {file_to_delete}")

        # Clean up temp file if move_file failed (permanent_path was never set)
        if data_store_id is None and permanent_path is None and temp_path:
            try:
                delete_file(temp_path)
                logger.info(f"Task {task_id}: Temp file cleaned up at {temp_path}")
            except Exception:
                logger.warning(f"Task {task_id}: Failed to clean up temp file at {temp_path}")

        # Clean up DocumentUpload record if this was a KB upload
        if task and task.document_upload_id:
            try:
                upload = db.query(DocumentUpload).filter(
                    DocumentUpload.id == task.document_upload_id
                ).first()
                if upload:
                    db.delete(upload)
                    db.commit()
                    logger.info(f"Task {task_id}: DocumentUpload record cleaned up")
            except Exception:
                db.rollback()

    finally:
        if progress_db is not None:
            try:
                progress_db.close()
            except Exception:
                pass
        if should_close_db and db:
            db.close()
