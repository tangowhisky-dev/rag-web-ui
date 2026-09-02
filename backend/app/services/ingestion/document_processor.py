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
from dataclasses import dataclass
from datetime import datetime, timezone
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
    extract_title,
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

from app.models.knowledge import ProcessingTask, Document, DocumentChunk, DocumentUpload, KnowledgeBase
from app.models.datastore import DataStore, OrganizationDataStore

logger = logging.getLogger(__name__)


@dataclass
class GraphBuildRequest:
    """Data needed to build a Neo4j knowledge graph for a document.

    Returned by ``process_document_background`` so the caller can fire
    the graph build as a separate background task, decoupled from the
    ingestion pipeline (which completes at Qdrant upsert).
    """
    document_id: int
    file_name: str
    chunks: list[str]
    chunk_ids: list[str]
    kb_id: Optional[int]
    data_store_id: Optional[int]
    task_id: Optional[int]


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
            chunk_size = int(get_setting(_db, "CHUNK_SIZE", None) or 1500)
        finally:
            _db.close()
    if chunk_overlap is None:
        from app.services.settings_service import get_setting
        from app.db.session import SessionLocal
        _db = SessionLocal()
        try:
            _cs = int(get_setting(_db, "CHUNK_SIZE", None) or 1500)
            _op = float(get_setting(_db, "OVERLAP_PERCENTAGE", None) or 0.10)
            chunk_overlap = int(_cs * _op)
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
            logger.debug(
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
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
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


# ── Phase 1: Convert ──────────────────────────────────────────────────────────

async def convert_document(
    document_id: int,
    file_path: str,
    file_name: str,
    enable_ocr: Optional[bool] = None,
    db: Session = None,
    progress_cb: Optional[callable] = None,
) -> str:
    """Convert a file to markdown and store it in Document.converted_markdown.

    Phase 1 of the 3-phase pipeline.  Sets conversion_status to 'completed'
    on success, 'error' on failure.  Returns the markdown text.
    Does not touch chunks, vectors, or graph.

    Async — uses run_in_executor for the synchronous _convert_to_markdown call.
    """
    should_close_db = False
    if db is None:
        db = SessionLocal()
        should_close_db = True

    try:
        document = db.get(Document, document_id)
        if not document:
            raise ValueError(f"Document {document_id} not found")

        document.conversion_status = "processing"
        db.commit()

        loop = asyncio.get_event_loop()
        local_path = get_abs_path(file_path)

        # Convert
        markdown_text = await loop.run_in_executor(
            None, lambda: _convert_to_markdown(
                local_path, file_name, enable_ocr=enable_ocr, progress_cb=progress_cb,
            )
        )

        # Cleanup pass
        try:
            markdown_text = clean_markdown(markdown_text)
        except Exception as _ce:
            logger.warning("[CLEANUP] fallback to raw markdown. reason=%s file=%s",
                           str(_ce)[:200], file_name)

        if not markdown_text or not markdown_text.strip():
            raise ValueError(
                "Document produced no extractable text. "
                "The file may be empty, password-protected, or in an unreadable format."
            )

        # Extract title
        doc_title = extract_title(markdown_text, file_name, abs_path=local_path)

        # Store markdown + title on the document
        document.converted_markdown = markdown_text
        document.title = doc_title
        document.conversion_status = "completed"
        document.conversion_error = None
        db.commit()

        logger.debug("[CONVERT] document_id=%s file=%s chars=%d title=%r",
                    document_id, file_name, len(markdown_text), doc_title)
        return markdown_text

    except Exception as e:
        logger.error("[CONVERT] document_id=%s error=%s", document_id, e, exc_info=True)
        try:
            db.rollback()
            doc = db.get(Document, document_id)
            if doc:
                doc.conversion_status = "error"
                doc.conversion_error = str(e)[:2000]
                db.commit()
        except Exception:
            db.rollback()
        raise
    finally:
        if should_close_db and db:
            db.close()


# ── Phase 2: Ingest (chunk + embed + Qdrant) ──────────────────────────────────

async def _prepare_ingestion(
    document_id: int,
    file_name: str,
    data_store_id: Optional[int],
    kb_id: Optional[int],
    task_id: Optional[int],
    markdown_text: Optional[str],
    db: Session,
    _prog: callable,
):
    """Setup and validation for ingestion."""
    document = db.get(Document, document_id)
    if not document:
        raise ValueError(f"Document {document_id} not found")

    task = db.get(ProcessingTask, task_id) if task_id else None

    # Reset graph_status — a stale "failed" from a previous run
    # should not persist into the new ingestion cycle.
    if task and task.graph_status in ("failed", "pending"):
        task.graph_status = None
        task.graph_error = None
        db.commit()

    # Read markdown from argument or DB
    if markdown_text is None:
        markdown_text = document.converted_markdown
        if not markdown_text:
            raise ValueError(f"Document {document_id} has no converted_markdown")

    # Check datastore deleted
    if data_store_id is not None:
        from app.services.ingestion.ingestion_dispatcher import is_datastore_deleted
        if is_datastore_deleted(data_store_id):
            logger.debug("[INGEST] document_id=%s — datastore %s deleted, aborting",
                        document_id, data_store_id)
            return None

    # Chunk size from settings (with defaults to prevent None crash)
    from app.services.settings_service import get_setting
    chunk_size = int(get_setting(db, "CHUNK_SIZE", None) or 1500)
    overlap_pct = float(get_setting(db, "OVERLAP_PERCENTAGE", None) or 0.10)
    chunk_overlap = int(chunk_size * overlap_pct)

    loop = asyncio.get_event_loop()

    # Chunk
    _prog(20, "Splitting into chunks…")
    doc = LangchainDocument(page_content=markdown_text, metadata={"source": file_name})
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = await loop.run_in_executor(None, lambda: text_splitter.split_documents([doc]))
    logger.debug("[INGEST] document_id=%s chunks=%d", document_id, len(chunks))

    if not chunks:
        raise ValueError("Document produced no chunks after splitting.")

    # Ensure Qdrant collection
    _prog(25, f"Preparing vector store ({len(chunks)} chunks)…")
    if data_store_id is not None:
        collection_name = f"ds_{data_store_id}"
    elif kb_id is not None:
        collection_name = f"kb_{kb_id}"
    else:
        raise ValueError("Neither data_store_id nor kb_id provided")
    _ensure_qdrant_collection(get_qdrant_client(), collection_name)

    return document, task, chunks, collection_name, loop


def _resolve_org_id_for_abbr(
    kb_id: Optional[int],
    data_store_id: Optional[int],
    db: Session,
) -> Optional[int]:
    if kb_id:
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
        return kb.org_id if kb else None
    elif data_store_id:
        link = db.query(OrganizationDataStore).filter(
            OrganizationDataStore.data_store_id == data_store_id,
            OrganizationDataStore.is_active == True,  # noqa: E712
        ).first()
        return link.org_id if link else None
    return None


def _build_single_chunk(
    i: int,
    chunk,
    data_store_id: Optional[int],
    kb_id: Optional[int],
    file_name: str,
    document: Document,
    doc_title: str,
    abbr_lookup,
):
    from app.services.abbreviation_service import expand_suffix
    scope_prefix = f"ds:{data_store_id}" if data_store_id else f"kb:{kb_id}"
    original_text = chunk.page_content
    expanded_text = expand_suffix(original_text, abbr_lookup) if not abbr_lookup.is_empty else original_text
    chunk_id = hashlib.sha256(
        f"{scope_prefix}:{file_name}:{i}:{original_text}".encode()
    ).hexdigest()
    chunk.metadata["source"] = file_name
    source_metadata = {
        k: v for k, v in chunk.metadata.items()
        if k not in ("kb_id", "document_id", "chunk_id", "file_name")
    }
    if doc_title:
        source_metadata["title"] = doc_title
    if not abbr_lookup.is_empty and expanded_text != original_text:
        source_metadata["original_text"] = original_text
    # Store document-level metadata in the Qdrant payload so retrieval
    # can filter/sort on these fields without a MySQL round-trip.
    source_metadata["_created_at"] = document.created_at.isoformat() if document.created_at else None
    source_metadata["_modified_at"] = (document.modified_at or document.created_at).isoformat() if (document.modified_at or document.created_at) else None
    source_metadata["_content_type"] = document.content_type
    source_metadata["_file_size"] = document.file_size
    db_chunk = DocumentChunk(
        id=chunk_id,
        document_id=document.id,
        kb_id=kb_id if kb_id else None,
        data_store_id=data_store_id if data_store_id else None,
        file_name=file_name,
        chunk_text=expanded_text,
        chunk_index=i,
        chunk_metadata=source_metadata,
        hash=hashlib.sha256((original_text + str(chunk.metadata)).encode()).hexdigest(),
    )
    payload = (chunk_id, expanded_text, source_metadata, i)
    return payload, db_chunk


async def _process_chunks(
    document: Document,
    file_name: str,
    data_store_id: Optional[int],
    kb_id: Optional[int],
    db: Session,
    chunks: list,
    collection_name: str,
    loop,
    _prog: callable,
):
    """Chunk processing and embedding."""
    # Delete old chunks
    _prog(30, "Cleaning up old chunks…")
    from sqlalchemy import and_
    if data_store_id is not None:
        scope_filter = DocumentChunk.data_store_id == data_store_id
    else:
        scope_filter = and_(DocumentChunk.kb_id == kb_id, DocumentChunk.data_store_id.is_(None))
    old_chunk_ids = db.query(DocumentChunk).filter(
        DocumentChunk.document_id == document.id, scope_filter,
    ).with_entities(DocumentChunk.id).all()
    old_chunk_ids = [cid[0] for cid in old_chunk_ids]
    if old_chunk_ids:
        point_ids = [_chunk_id_to_point_id(cid) for cid in old_chunk_ids]
        try:
            get_qdrant_client().delete(
                collection_name=collection_name,
                points_selector=PointIdsList(points=point_ids),
            )
        except Exception as e:
            logger.warning("[INGEST] Qdrant delete old chunks failed: %s", e)
        # Rebuild scope_filter (single-use)
        if data_store_id is not None:
            scope_filter = DocumentChunk.data_store_id == data_store_id
        else:
            scope_filter = and_(DocumentChunk.kb_id == kb_id, DocumentChunk.data_store_id.is_(None))
        db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document.id, scope_filter,
        ).delete(synchronize_session="fetch")
        db.commit()

    # Build chunk records
    _prog(35, "Building chunk records…")
    doc_title = document.title or file_name

    def _build_chunk_records():
        from app.services.abbreviation_service import build_lookup
        org_id_for_abbr = _resolve_org_id_for_abbr(kb_id, data_store_id, db)
        abbr_lookup = build_lookup(db, org_id_for_abbr)
        payloads = []
        db_chunks = []
        for i, chunk in enumerate(chunks):
            payload, db_chunk = _build_single_chunk(
                i, chunk, data_store_id, kb_id, file_name, document, doc_title, abbr_lookup,
            )
            db_chunks.append(db_chunk)
            payloads.append(payload)
        return payloads, db_chunks

    qdrant_payloads, db_chunks = await loop.run_in_executor(None, _build_chunk_records)
    for dc in db_chunks:
        db.add(dc)

    return qdrant_payloads, db_chunks


async def _update_search_indices(
    qdrant_payloads: list,
    kb_id: Optional[int],
    document: Document,
    file_name: str,
    data_store_id: Optional[int],
    task: Optional[ProcessingTask],
    task_id: Optional[int],
    db: Session,
    progress_cb: Optional[callable],
    pt: Optional[ProgressTimeout],
    _prog: callable,
) -> Optional[GraphBuildRequest]:
    """Update Qdrant/graph indices."""
    # Upsert to Qdrant
    _prog(40, f"Embedding {len(qdrant_payloads)} chunks…")
    await _upsert_to_qdrant(
        qdrant_payloads, kb_id, document.id, file_name,
        data_store_id=data_store_id,
        progress_cb=progress_cb or (lambda *_: None),
        progress_start=40,
        progress_end=80,
        pt=pt,
    )

    # Commit + mark task complete
    _prog(82, "Saving to database…")
    if task:
        task.status = "completed"
        task.progress = 90
        task.progress_message = "Finalising…"
        task.document_id = document.id
        upload = task.document_upload
        if upload:
            upload.status = "completed"
    db.commit()
    logger.debug("[INGEST] document_id=%s completed chunks=%d", document.id, len(qdrant_payloads))

    # Return graph build request
    from app.services.settings_service import get_setting as _gs
    if _gs(db, "GRAPHRAG_ENABLED", None):
        return GraphBuildRequest(
            document_id=document.id,
            file_name=file_name,
            chunks=[p[1] for p in qdrant_payloads],
            chunk_ids=[p[0] for p in qdrant_payloads],
            kb_id=kb_id,
            data_store_id=data_store_id,
            task_id=task_id,
        )
    return None


async def ingest_document(
    document_id: int,
    file_name: str,
    data_store_id: Optional[int] = None,
    kb_id: Optional[int] = None,
    task_id: Optional[int] = None,
    markdown_text: Optional[str] = None,
    db: Session = None,
    progress_cb: Optional[callable] = None,
    pt: Optional[ProgressTimeout] = None,
) -> Optional[GraphBuildRequest]:
    """Chunk a document's markdown, embed, and store in Qdrant.

    Phase 2 of the 3-phase pipeline.  If markdown_text is None, reads from
    Document.converted_markdown.  Deletes existing chunks (MySQL + Qdrant)
    for this document before re-chunking.  Returns a GraphBuildRequest
    for the caller to queue phase 3.
    """
    should_close_db = False
    if db is None:
        db = SessionLocal()
        should_close_db = True

    def _prog(pct: int, msg: str) -> None:
        if progress_cb:
            progress_cb(pct, msg)
        if pt:
            pt.ping()

    try:
        result = await _prepare_ingestion(
            document_id, file_name, data_store_id, kb_id, task_id,
            markdown_text, db, _prog,
        )
        if result is None:
            return None
        document, task, chunks, collection_name, loop = result

        qdrant_payloads, db_chunks = await _process_chunks(
            document, file_name, data_store_id, kb_id, db,
            chunks, collection_name, loop, _prog,
        )

        return await _update_search_indices(
            qdrant_payloads, kb_id, document, file_name,
            data_store_id, task, task_id, db, progress_cb, pt, _prog,
        )

    except Exception as e:
        logger.error("[INGEST] document_id=%s error=%s", document_id, e, exc_info=True)
        try:
            db.rollback()
            if task_id:
                fail_task = db.get(ProcessingTask, task_id)
                if fail_task:
                    fail_task.status = "failed"
                    fail_task.error_message = str(e)[:2000]
                    db.commit()
        except Exception:
            db.rollback()
        raise
    finally:
        if should_close_db and db:
            db.close()


# ── Orchestrator: full pipeline (convert → ingest) ────────────────────────────

def _prepare_file(
    temp_path: str,
    file_path: Optional[str],
    data_store_id: Optional[int],
    user_id: int,
    kb_id: Optional[int],
    file_name: str,
    task_id: Optional[int],
) -> str:
    """Handle file movement/copying for datastore vs KB."""
    if data_store_id is not None:
        permanent_path = file_path if file_path else temp_path
        logger.debug(f"Task {task_id}: DataStore file stays in place: {permanent_path}")
    else:
        _permanent_path = f"user_{user_id}/kb_{kb_id}/{file_name}"
        logger.debug(f"Task {task_id}: Moving file to permanent storage")
        move_file(temp_path, _permanent_path)
        permanent_path = _permanent_path
        logger.debug(f"Task {task_id}: File moved to {permanent_path}")
    return permanent_path


def _resolve_doc_metadata(
    data_store_id: Optional[int],
    file_hash: Optional[str],
    file_size: Optional[int],
    content_type: Optional[str],
    file_name: str,
    task: ProcessingTask,
) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    if data_store_id is not None:
        doc_file_hash = file_hash if file_hash else None
        doc_file_size = file_size if file_size else None
        doc_content_type = content_type if content_type else CONTENT_TYPE_MAP.get(
            os.path.splitext(file_name)[1].lower(), "application/octet-stream",
        )
    else:
        doc_file_hash = file_hash if file_hash else task.document_upload.file_hash
        doc_file_size = file_size if file_size else task.document_upload.file_size
        doc_content_type = content_type if content_type else task.document_upload.content_type
    return doc_file_hash, doc_file_size, doc_content_type


def _get_file_mtime(doc_file_path: str) -> datetime:
    try:
        return datetime.fromtimestamp(os.stat(doc_file_path).st_mtime, tz=timezone.utc)
    except (OSError, TypeError):
        return datetime.now(timezone.utc)


def _get_or_create_document(
    file_path: Optional[str],
    permanent_path: str,
    data_store_id: Optional[int],
    file_hash: Optional[str],
    file_size: Optional[int],
    content_type: Optional[str],
    file_name: str,
    task: ProcessingTask,
    document_id: Optional[int],
    kb_id: Optional[int],
    db: Session,
    task_id: Optional[int],
) -> Optional[Tuple[Document, bool]]:
    """Create or update DB document record."""
    doc_file_path = file_path if file_path else permanent_path
    doc_file_hash, doc_file_size, doc_content_type = _resolve_doc_metadata(
        data_store_id, file_hash, file_size, content_type, file_name, task,
    )

    if document_id:
        document = db.get(Document, document_id)
        if document:
            document.file_path = doc_file_path
            document.file_hash = doc_file_hash
            document.file_size = doc_file_size
            document.content_type = doc_content_type
            document.data_store_id = data_store_id
            document.knowledge_base_id = kb_id if kb_id else None
            document.modified_at = _get_file_mtime(doc_file_path)
            db.commit()
            logger.debug(f"Task {task_id}: Updated document ID {document.id}")
        else:
            logger.error(f"Task {task_id}: Document {document_id} not found")
            return None
    else:
        document = Document(
            file_name=file_name,
            file_path=doc_file_path,
            file_hash=doc_file_hash,
            file_size=doc_file_size,
            content_type=doc_content_type,
            knowledge_base_id=kb_id if kb_id else None,
            data_store_id=data_store_id,
            modified_at=_get_file_mtime(doc_file_path),
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        logger.debug(f"Task {task_id}: Document record created with ID {document.id}")
        return document, True

    return document, False


async def _convert_or_reuse_markdown(
    skip_conversion: bool,
    document: Document,
    permanent_path: str,
    file_path: Optional[str],
    temp_path: str,
    data_store_id: Optional[int],
    file_name: str,
    enable_ocr: Optional[bool],
    db: Session,
    task_id: Optional[int],
    _set_progress: callable,
) -> Optional[str]:
    """Convert file or reuse existing markdown."""
    if skip_conversion:
        # Use existing converted_markdown (e.g. markdown was edited)
        _set_progress(5, "Re-ingesting edited markdown…")
        markdown_text = None  # ingest_document will read from Document.converted_markdown
        logger.debug(f"Task {task_id}: Skipping conversion, using existing markdown")
    else:
        _set_progress(5, "Converting document…")
        markdown_text = await convert_document(
            document_id=document.id,
            file_path=permanent_path if data_store_id is None else (file_path or temp_path),
            file_name=file_name,
            enable_ocr=enable_ocr,
            db=db,
            progress_cb=_set_progress,  # pings ProgressTimeout between OCR pages
        )
    return markdown_text


def _rollback_document_on_failure(
    document: Optional[Document],
    document_was_created: bool,
    task_id: Optional[int],
    db: Session,
) -> None:
    if document is not None and document_was_created:
        try:
            db.refresh(document)
            if not document.converted_markdown:
                db.delete(document)
                db.commit()
                logger.debug(f"Task {task_id}: Document record rolled back (no markdown)")
            else:
                logger.debug(f"Task {task_id}: Keeping document — markdown exists")
        except Exception as del_err:
            logger.warning(f"Task {task_id}: Could not delete document record: {del_err}")


def _mark_task_failed(
    task_id: Optional[int],
    e: Exception,
    db: Session,
) -> None:
    try:
        task = db.get(ProcessingTask, task_id)
        if task:
            task.status = "failed"
            task.error_message = str(e)[:2000]
            db.commit()
    except Exception:
        pass


def _cleanup_files_on_failure(
    data_store_id: Optional[int],
    permanent_path: Optional[str],
    temp_path: str,
    task_id: Optional[int],
) -> None:
    if data_store_id is None and permanent_path is not None:
        try:
            delete_file(permanent_path)
            logger.debug(f"Task {task_id}: File cleaned up at {permanent_path}")
        except Exception:
            logger.warning(f"Task {task_id}: Failed to clean up file at {permanent_path}")
    if data_store_id is None and permanent_path is None and temp_path:
        try:
            delete_file(temp_path)
            logger.debug(f"Task {task_id}: Temp file cleaned up at {temp_path}")
        except Exception:
            pass


def _cleanup_upload_on_failure(
    task: Optional[ProcessingTask],
    db: Session,
) -> None:
    if task and task.document_upload_id:
        try:
            upload = db.query(DocumentUpload).filter(
                DocumentUpload.id == task.document_upload_id
            ).first()
            if upload:
                db.delete(upload)
                db.commit()
        except Exception:
            db.rollback()


def _cleanup_on_failure(
    e: Exception,
    task_id: Optional[int],
    db: Session,
    document: Optional[Document],
    document_was_created: bool,
    data_store_id: Optional[int],
    permanent_path: Optional[str],
    temp_path: str,
    task: ProcessingTask,
) -> None:
    """Failure cleanup logic."""
    logger.error(f"Task {task_id}: Error processing document: {str(e)}")
    logger.error(f"Task {task_id}: Stack trace: {traceback.format_exc()}")

    try:
        db.rollback()
    except Exception:
        pass

    _rollback_document_on_failure(document, document_was_created, task_id, db)
    _mark_task_failed(task_id, e, db)
    _cleanup_files_on_failure(data_store_id, permanent_path, temp_path, task_id)
    _cleanup_upload_on_failure(task, db)


async def process_document_full(
    temp_path: str,
    file_name: str,
    kb_id: Optional[int] = None,
    task_id: Optional[int] = None,
    db: Session = None,
    user_id: int = None,
    chunk_size: int = None,
    chunk_overlap: int = None,
    enable_ocr: Optional[bool] = None,
    document_id: Optional[int] = None,
    data_store_id: Optional[int] = None,
    file_path: Optional[str] = None,
    file_hash: Optional[str] = None,
    file_size: Optional[int] = None,
    content_type: Optional[str] = None,
    skip_conversion: bool = False,
) -> Optional[GraphBuildRequest]:
    """Full pipeline: convert → ingest.  Replaces process_document_background.

    Returns a GraphBuildRequest if graph extraction is enabled and the
    document was successfully ingested, so the caller can fire the graph
    build as a separate background task. Returns None if graph is disabled
    or ingestion failed.
    """
    if db is None:
        db = SessionLocal()
        should_close_db = True
    else:
        should_close_db = False

    task = db.get(ProcessingTask, task_id)
    if not task:
        logger.error(f"Task {task_id} not found")
        return

    permanent_path: Optional[str] = None
    document: Optional[Document] = None
    document_was_created = False  # track whether we created it (for rollback)
    progress_db = None

    try:
        progress_db = SessionLocal()

        def _set_progress(pct: int, msg: str) -> None:
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
        silence_s = get_setting(db, "PROCESSING_TIMEOUT_SILENCE_S", None) or 600

        def _on_timeout() -> None:
            logger.warning(
                "[PROGRESS_TIMEOUT] task_id=%s silence_s=%s — cancelling",
                task_id, silence_s, silence_s,
            )
            try:
                fail_db = SessionLocal()
                try:
                    ptask = fail_db.query(ProcessingTask).filter(
                        ProcessingTask.id == task_id
                    ).first()
                    if ptask:
                        ptask.status = "failed"
                        ptask.error_message = f"Processing timed out — no progress for {silence_s}s"
                        fail_db.commit()
                finally:
                    fail_db.close()
            except Exception:
                pass

        async with ProgressTimeout(silence_s, _on_timeout) as pt:

            local_temp_path = get_abs_path(temp_path)
            logger.debug(f"Task {task_id}: Using file at {local_temp_path}")

            # ── Step 1: Move to permanent storage (DataStore files stay in place) ───
            permanent_path = _prepare_file(
                temp_path, file_path, data_store_id, user_id, kb_id, file_name, task_id,
            )

            # ── Step 2: Create or update Document record ──────────────────────
            doc_result = _get_or_create_document(
                file_path, permanent_path, data_store_id, file_hash, file_size,
                content_type, file_name, task, document_id, kb_id, db, task_id,
            )
            if doc_result is None:
                return
            document, document_was_created = doc_result

            # ── Phase 1: Convert ──────────────────────────────────────────────
            markdown_text = await _convert_or_reuse_markdown(
                skip_conversion, document, permanent_path, file_path, temp_path,
                data_store_id, file_name, enable_ocr, db, task_id, _set_progress,
            )

            # ── Phase 2: Ingest ────────────────────────────────────────────────
            graph_request = await ingest_document(
                document_id=document.id,
                file_name=file_name,
                data_store_id=data_store_id,
                kb_id=kb_id,
                task_id=task_id,
                markdown_text=markdown_text,
                db=db,
                progress_cb=_set_progress,
                pt=pt,
            )

            logger.debug("[PROGRESS_TIMEOUT] task_id=%s completed_ok=true", task_id)
            logger.debug(f"Task {task_id}: Processing completed successfully")
            return graph_request

    except Exception as e:
        _cleanup_on_failure(
            e, task_id, db, document, document_was_created,
            data_store_id, permanent_path, temp_path, task,
        )
        return None

    finally:
        if progress_db is not None:
            try:
                progress_db.close()
            except Exception:
                pass
        if should_close_db and db:
            db.close()


# Backward-compatible alias
process_document_background = process_document_full
