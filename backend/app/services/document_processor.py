import asyncio
import logging
import os
import re
import uuid
import hashlib
import traceback
from app.db.session import SessionLocal
from typing import Optional, List, Dict, Set, Tuple
from fastapi import UploadFile
from markitdown import MarkItDown
from openai import OpenAI as SyncOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument
from pydantic import BaseModel
from sqlalchemy.orm import Session
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointIdsList,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from fastembed import SparseTextEmbedding
from app.core.config import settings
from app.core.storage import get_abs_path, save_file, move_file, delete_file
from app.services.markdown_cleaner import clean_markdown
from app.models.knowledge import ProcessingTask, Document, DocumentChunk
from app.services.chunk_record import ChunkRecord

# ── Module-level singletons (lazy) ────────────────────────────────────────────
_qdrant_client: Optional[QdrantClient] = None
_sparse_embedder: Optional[SparseTextEmbedding] = None

_markitdown: Optional[MarkItDown] = None
_EMBED_BATCH_SIZE = 32
_QDRANT_UPSERT_BATCH = 100

# Supported file extensions (markitdown handles all of these)
SUPPORTED_EXTENSIONS = {
    # Documents
    ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
    # Text / Markup
    ".txt", ".md", ".html", ".htm", ".mhtml",
    # Data formats
    ".csv", ".json", ".xml",
    # Email
    ".msg", ".eml",
    # Books
    ".epub",
    # Images (OCR via markitdown-ocr)
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff",
    # Archives (recursively processes contents)
    ".zip",
}

CONTENT_TYPE_MAP = {
    ".pdf":   "application/pdf",
    ".docx":  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc":   "application/msword",
    ".pptx":  "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".ppt":   "application/vnd.ms-powerpoint",
    ".xlsx":  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls":   "application/vnd.ms-excel",
    ".txt":   "text/plain",
    ".md":    "text/markdown",
    ".html":  "text/html",
    ".htm":   "text/html",
    ".mhtml": "message/rfc822",
    ".csv":   "text/csv",
    ".json":  "application/json",
    ".xml":   "application/xml",
    ".msg":   "application/vnd.ms-outlook",
    ".eml":   "message/rfc822",
    ".epub":  "application/epub+zip",
    ".jpg":   "image/jpeg",
    ".jpeg":  "image/jpeg",
    ".png":   "image/png",
    ".gif":   "image/gif",
    ".bmp":   "image/bmp",
    ".tiff":  "image/tiff",
    ".zip":   "application/zip",
}


def _get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    return _qdrant_client


def _get_sparse_embedder() -> SparseTextEmbedding:
    global _sparse_embedder
    if _sparse_embedder is None:
        _sparse_embedder = SparseTextEmbedding(
            model_name=settings.SPLADE_MODEL,
            cache_dir=settings.FASTEMBED_CACHE_DIR,
        )
    return _sparse_embedder


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

def _get_markitdown() -> MarkItDown:
    """
    Lazy singleton for MarkItDown converter.

    When VISION_MODEL is configured, the markitdown-ocr plugin is
    activated with an llm_client so it can OCR embedded images in PDFs,
    DOCX, PPTX, and XLSX files automatically.

    When VISION_MODEL is unset, MarkItDown is initialised without a
    client — markitdown-ocr still loads (if installed) but silently skips
    OCR, which is identical to the previous behaviour.
    """
    global _markitdown
    if _markitdown is None:
        vision_model = settings.VISION_MODEL
        if vision_model:
            vision_client = SyncOpenAI(
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.effective_vision_api_base,
            )
            _markitdown = MarkItDown(
                enable_plugins=True,
                llm_client=vision_client,
                llm_model=vision_model,
                llm_prompt=(
                    "Extract all text from this image into clean, naturally flowing paragraphs, "
                    "while preserving document structure and any table or sub-element layout.\n\n"
                    "Rules:\n"
                    "- Remove unnatural line breaks within sentences\n"
                    "- Join split words and sentences caused by column layout or line wrapping\n"
                    "- Keep proper paragraph breaks where the topic clearly changes\n"
                    "- Preserve tables using Markdown table syntax\n"
                    "- Preserve all original meaning and technical terms exactly\n"
                    "- Output only the extracted text, no explanations or commentary"
                ),
            )
            logging.getLogger(__name__).info(
                "[markitdown] OCR enabled — vision_model=%s base=%s",
                vision_model, settings.effective_vision_api_base,
            )
        else:
            _markitdown = MarkItDown()
            logging.getLogger(__name__).info(
                "[markitdown] OCR disabled — VISION_MODEL not set"
            )
    return _markitdown


def _convert_to_markdown(abs_path: str, file_name: str, enable_ocr: Optional[bool] = None) -> str:
    """
    Convert any supported file to clean Markdown text using markitdown.

    enable_ocr: tri-state override.
      None  → use global VISION_MODEL setting (default behaviour)
      True  → force OCR on (requires VISION_MODEL to be configured)
      False → force OCR off for this document regardless of global setting

    Think traces (<think>...</think>) are stripped before returning.
    Falls back gracefully to raw UTF-8 if conversion fails.
    """
    logger = logging.getLogger(__name__)
    try:
        if enable_ocr is False:
            # Per-document OCR override: use a plain MarkItDown with no vision client.
            md_instance = MarkItDown()
            logger.info("[markitdown] OCR disabled for this document (per-document override)")
        elif enable_ocr is True:
            # Explicit on: use the configured singleton (which has OCR if set up).
            md_instance = _get_markitdown()
            if settings.VISION_MODEL is None:
                logger.warning("[markitdown] OCR requested but VISION_MODEL not set — falling back to text-only")
        else:
            # Default: respect global setting.
            md_instance = _get_markitdown()
        result = md_instance.convert(abs_path)
        markdown_text = result.text_content or ""

        # Strip thinking traces from reasoning models used for OCR.
        # Pass 1: remove complete <think>...</think> blocks.
        cleaned = _THINK_RE.sub("", markdown_text)
        # Pass 2: remove any unclosed <think> prefix (truncated by max_tokens).
        cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL).strip()

        if len(cleaned) < len(markdown_text):
            stripped_chars = len(markdown_text) - len(cleaned)
            logger.info(
                "[markitdown] stripped %d chars of thinking traces from %s",
                stripped_chars, file_name,
            )

        logger.info(
            "[markitdown] converted %s → %d chars of markdown (ocr=%s)",
            file_name, len(cleaned), bool(settings.VISION_MODEL),
        )
        return cleaned
    except Exception as e:
        logger.warning(
            "[markitdown] conversion failed for %s (%s) — falling back to raw text",
            file_name, e
        )
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception:
            return ""


def _chunk_id_to_point_id(chunk_id: str) -> str:
    """Convert a SHA-256 hex chunk ID to a deterministic UUID for Qdrant."""
    return str(uuid.uuid5(uuid.NAMESPACE_OID, chunk_id))


def _ensure_qdrant_collection(client: QdrantClient, kb_id: int) -> None:
    """Create the Qdrant collection for a knowledge base if it does not exist."""
    collection_name = f"kb_{kb_id}"
    existing = {c.name for c in client.get_collections().collections}
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=settings.DENSE_EMBEDDING_DIM,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            },
        )


async def _embed_texts_batch(
    texts: List[str],
    progress_cb=None,
    progress_start: int = 0,
    progress_end: int = 100,
) -> List[List[float]]:
    """Compute dense embeddings via the OpenAI-compatible API, in batches.

    progress_cb(pct, msg) is called after each batch, with pct mapped between
    progress_start and progress_end so callers can slot this into a larger bar.
    """
    client = AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_API_BASE,
    )
    all_embeddings: List[List[float]] = []
    total_batches = max(1, (len(texts) + _EMBED_BATCH_SIZE - 1) // _EMBED_BATCH_SIZE)
    for batch_idx, i in enumerate(range(0, len(texts), _EMBED_BATCH_SIZE)):
        batch = texts[i : i + _EMBED_BATCH_SIZE]
        response = await client.embeddings.create(
            input=batch,
            model=settings.DENSE_EMBEDDINGS_MODEL,
        )
        all_embeddings.extend(r.embedding for r in response.data)
        if progress_cb is not None:
            frac = (batch_idx + 1) / total_batches
            pct = int(progress_start + frac * (progress_end - progress_start))
            done = min(i + _EMBED_BATCH_SIZE, len(texts))
            progress_cb(pct, f"Embedding chunks {done}/{len(texts)}…")
    return all_embeddings


def _build_qdrant_points(
    chunk_payloads: List[Tuple[str, str, dict, int]],  # (chunk_id, text, metadata, index)
    dense_embeddings: List[List[float]],
    sparse_embeddings,
    kb_id: int,
    document_id: int,
    file_name: str,
) -> List[PointStruct]:
    """Build Qdrant PointStruct list from pre-computed embeddings."""
    points = []
    for (chunk_id, chunk_text, source_meta, chunk_index), dense_emb, sparse_emb in zip(
        chunk_payloads, dense_embeddings, sparse_embeddings
    ):
        points.append(
            PointStruct(
                id=_chunk_id_to_point_id(chunk_id),
                vector={
                    "dense": dense_emb,
                    "sparse": SparseVector(
                        indices=sparse_emb.indices.tolist(),
                        values=sparse_emb.values.tolist(),
                    ),
                },
                payload={
                    "chunk_text": chunk_text,
                    "kb_id": kb_id,
                    "document_id": document_id,
                    "file_name": file_name,
                    "chunk_index": chunk_index,
                    "qdrant_point_id": _chunk_id_to_point_id(chunk_id),  # explicit UUID for Neo4j cross-reference
                    **source_meta,
                },
            )
        )
    return points


async def _upsert_to_qdrant(
    chunk_payloads: List[Tuple[str, str, dict, int]],
    kb_id: int,
    document_id: int,
    file_name: str,
    progress_cb=None,   # optional callable(pct: int, msg: str)
    progress_start: int = 40,
    progress_end: int = 80,
) -> None:
    """Compute both vector types and upsert all points to Qdrant.

    progress_cb is called after each embedding batch with the current progress
    percentage (mapped from progress_start to progress_end) and a message.
    """
    if not chunk_payloads:
        return
    texts = [p[1] for p in chunk_payloads]
    dense_embs = await _embed_texts_batch(
        texts,
        progress_cb=progress_cb,
        progress_start=progress_start,
        progress_end=progress_end,
    )
    # fastembed BM25 tokenization is Python/numpy — does NOT release the GIL.
    # Running the full 2795-text corpus in one executor call still blocks the
    # event loop for 10-30s. Batch it with asyncio.sleep(0) yields between
    # batches so poll requests get served.
    loop = asyncio.get_event_loop()
    sparse_embs: list = []
    embedder = _get_sparse_embedder()
    for batch_start in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[batch_start : batch_start + _EMBED_BATCH_SIZE]
        batch_sparse = await loop.run_in_executor(
            None, lambda b=batch: list(embedder.embed(b))
        )
        sparse_embs.extend(batch_sparse)
        await asyncio.sleep(0)  # yield — let poll requests through

    # Build point structs and upsert in batches, yielding the event loop between
    # each batch via asyncio.sleep(0). Pure-Python object construction holds the
    # GIL even inside run_in_executor, so we must yield explicitly to let poll
    # requests through — otherwise the event loop is blocked for 10-30 seconds
    # while 2795 PointStruct objects are built, causing ECONNRESET on the frontend.
    client = _get_qdrant_client()
    _ensure_qdrant_collection(client, kb_id)
    n = len(chunk_payloads)
    for batch_start in range(0, n, _QDRANT_UPSERT_BATCH):
        batch_end = min(batch_start + _QDRANT_UPSERT_BATCH, n)
        batch_chunks = chunk_payloads[batch_start:batch_end]
        batch_dense = dense_embs[batch_start:batch_end]
        batch_sparse = sparse_embs[batch_start:batch_end]

        def _build_upsert_batch(bc=batch_chunks, bd=batch_dense, bs=batch_sparse):
            pts = _build_qdrant_points(bc, bd, bs, kb_id, document_id, file_name)
            client.upsert(collection_name=f"kb_{kb_id}", points=pts)

        await loop.run_in_executor(None, _build_upsert_batch)
        # Yield the event loop so poll requests can be served between batches
        await asyncio.sleep(0)

class UploadResult(BaseModel):
    file_path: str
    file_name: str
    file_size: int
    content_type: str
    file_hash: str

class TextChunk(BaseModel):
    content: str
    metadata: Optional[Dict] = None

class PreviewResult(BaseModel):
    chunks: List[TextChunk]
    total_chunks: int

async def process_document(file_path: str, file_name: str, kb_id: int, document_id: int, chunk_size: int = None, chunk_overlap: int = None) -> None:
    """Process document and store in vector database with incremental updates"""
    logger = logging.getLogger(__name__)
    # Use env-configured defaults when callers do not supply explicit values.
    # WARNING: chunk_size and chunk_overlap must stay consistent across all
    # documents in a knowledge base. Do not change CHUNK_SIZE / OVERLAP_PERCENTAGE
    # in .env after documents have been ingested — re-upload existing documents
    # to re-index them with the new settings.
    if chunk_size is None:
        chunk_size = settings.CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = settings.chunk_overlap
    
    try:
        preview_result = await preview_document(file_path, chunk_size, chunk_overlap)
        
        # Initialize chunk record manager
        chunk_manager = ChunkRecord(kb_id)
        
        # Get existing chunk hashes for this file
        existing_hashes = chunk_manager.list_chunks(file_name)
        
        # Prepare new chunks
        new_chunks = []
        current_hashes = set()
        
        for chunk_index, chunk in enumerate(preview_result.chunks):
            # Calculate chunk hash
            chunk_hash = hashlib.sha256(
                (chunk.content + str(chunk.metadata)).encode()
            ).hexdigest()
            current_hashes.add(chunk_hash)
            
            # Skip if chunk hasn't changed
            if chunk_hash in existing_hashes:
                continue
            
            # Create unique ID for the chunk
            chunk_id = hashlib.sha256(
                f"{kb_id}:{file_name}:{chunk_hash}".encode()
            ).hexdigest()
            
            # chunk_metadata holds only variable source metadata (page number, source path)
            # chunk_text and chunk_index are stored as proper columns
            metadata = {k: v for k, v in chunk.metadata.items()
                        if k not in ("kb_id", "document_id", "chunk_id", "file_name")}
            
            new_chunks.append({
                "id": chunk_id,
                "kb_id": kb_id,
                "document_id": document_id,
                "file_name": file_name,
                "chunk_text": chunk.content,
                "chunk_index": chunk_index,
                "metadata": metadata,
                "hash": chunk_hash
            })
        
        # Add new chunks to MySQL + Qdrant
        if new_chunks:
            logger.info(f"Adding {len(new_chunks)} new/updated chunks")
            chunk_manager.add_chunks(new_chunks)
            chunk_payloads = [
                (c["id"], c["chunk_text"], c.get("metadata") or {}, c["chunk_index"])
                for c in new_chunks
            ]
            await _upsert_to_qdrant(chunk_payloads, kb_id, document_id, file_name)
        
        # Delete removed chunks from MySQL + Qdrant
        chunks_to_delete = chunk_manager.get_deleted_chunks(current_hashes, file_name)
        if chunks_to_delete:
            logger.info(f"Removing {len(chunks_to_delete)} deleted chunks")
            chunk_manager.delete_chunks(chunks_to_delete)
            point_ids = [_chunk_id_to_point_id(cid) for cid in chunks_to_delete]
            _get_qdrant_client().delete(
                collection_name=f"kb_{kb_id}",
                points_selector=PointIdsList(points=point_ids),
            )
        
        logger.info("Document processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise

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
        chunk_size = settings.CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = settings.chunk_overlap
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
            logging.getLogger(__name__).info(
                "[CLEANUP] chars_before=%d chars_after=%d file=%s",
                _chars_before, len(markdown_text), _fname,
            )
        except Exception as _ce:
            logging.getLogger(__name__).warning(
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
    kb_id: int,
    task_id: int,
    db: Session = None,
    user_id: int = None,
    chunk_size: int = None,
    chunk_overlap: int = None,
    enable_ocr: Optional[bool] = None,
) -> None:
    """Process document in background.

    enable_ocr: None = global setting, True = force on, False = force off.
    """
    logger = logging.getLogger(__name__)
    if chunk_size is None:
        chunk_size = settings.CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = settings.chunk_overlap

    if db is None:
        db = SessionLocal()
        should_close_db = True
    else:
        should_close_db = False

    task = db.query(ProcessingTask).get(task_id)
    if not task:
        logger.error(f"Task {task_id} not found")
        return

    # Track cleanup state so the except block always knows what to delete and
    # what DB objects to roll back — regardless of how far we got.
    permanent_path: Optional[str] = None   # set after move_file succeeds
    document: Optional[Document] = None    # set after Document record committed

    try:
        def _set_progress(pct: int, msg: str) -> None:
            """Write progress to DB immediately — uses a fresh merge so it always
            succeeds even if the main session has a pending rollback."""
            try:
                task.progress = pct
                task.progress_message = msg
                db.commit()
            except Exception:
                try:
                    db.rollback()
                except Exception:
                    pass

        task.status = "processing"
        task.progress = 0
        task.progress_message = "Starting…"
        db.commit()

        local_temp_path = get_abs_path(temp_path)
        logger.info(f"Task {task_id}: Using file at {local_temp_path}")

        # ── Step 1: Parse ────────────────────────────────────────────────────
        _set_progress(5, "Parsing document…")
        logger.info(f"Task {task_id}: Converting document with markitdown (enable_ocr={enable_ocr})")
        # Run in a thread pool — markitdown is synchronous and CPU/IO-bound.
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
        logger.info(f"Task {task_id}: Ensuring Qdrant collection kb_{kb_id}")
        _ensure_qdrant_collection(_get_qdrant_client(), kb_id)

        # ── Step 4: Move to permanent storage ────────────────────────────────
        _permanent_path = f"user_{user_id}/kb_{kb_id}/{file_name}"
        logger.info(f"Task {task_id}: Moving file to permanent storage")
        move_file(temp_path, _permanent_path)
        permanent_path = _permanent_path          # mark: file now at permanent location
        local_perm_path = get_abs_path(permanent_path)
        logger.info(f"Task {task_id}: File moved to {permanent_path}")

        # ── Step 5: Create Document record ───────────────────────────────────
        logger.info(f"Task {task_id}: Creating document record")
        document = Document(
            file_name=file_name,
            file_path=permanent_path,
            file_hash=task.document_upload.file_hash,
            file_size=task.document_upload.file_size,
            content_type=task.document_upload.content_type,
            knowledge_base_id=kb_id,
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        logger.info(f"Task {task_id}: Document record created with ID {document.id}")

        # ── Step 6: Build chunk records (no commit yet) ───────────────────────
        _set_progress(35, "Building chunk records…")
        logger.info(f"Task {task_id}: Building {len(chunks)} chunk records")

        def _build_chunk_records():
            """CPU-bound: hashing + object construction. Returns payloads only — no db.add here."""
            payloads = []
            db_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_id = hashlib.sha256(
                    f"{kb_id}:{file_name}:{chunk.page_content}".encode()
                ).hexdigest()
                chunk.metadata["source"] = file_name
                source_metadata = {
                    k: v for k, v in chunk.metadata.items()
                    if k not in ("kb_id", "document_id", "chunk_id", "file_name")
                }
                db_chunks.append(DocumentChunk(
                    id=chunk_id,
                    document_id=document.id,
                    kb_id=kb_id,
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
            progress_cb=_set_progress,
            progress_start=40,
            progress_end=80,
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
        logger.info(f"Task {task_id}: Processing completed successfully")

        # ── Step 9: Build Neo4j knowledge graph (non-fatal) ──────────────────
        if settings.GRAPHRAG_ENABLED:
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
                    from app.services.graph_service import build_graph_for_document
                    await build_graph_for_document(
                        kb_id=kb_id,
                        document_id=_doc_id,
                        file_name=file_name,
                        chunks=_chunks,
                        chunk_ids=_chunk_ids,
                    )
                    logger.info(f"Task {task_id}: Knowledge graph built in Neo4j")
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
            asyncio.create_task(_build_graph())

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
        # Delete from whatever location the file is currently at.
        # * Before move_file ran  → delete temp_path
        # * After  move_file ran  → delete permanent_path
        file_to_delete = permanent_path if permanent_path is not None else temp_path
        try:
            logger.info(f"Task {task_id}: Cleaning up file at {file_to_delete}")
            delete_file(file_to_delete)
            logger.info(f"Task {task_id}: File cleaned up")
        except Exception:
            logger.warning(f"Task {task_id}: Failed to clean up file at {file_to_delete}")

    finally:
        if should_close_db and db:
            db.close()
