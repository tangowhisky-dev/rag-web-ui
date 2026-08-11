"""Qdrant helpers — collection management, embedding, upsert, chunking.

Split from document_processor.py for maintainability.
"""

import asyncio
import hashlib
import logging
import uuid
from typing import Dict, List, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument
from openai import AsyncOpenAI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    PointIdsList,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from app.services.infrastructure import content_hash, get_qdrant_client, get_sparse_embedder


def _get_embedding_dim() -> int:
    """Resolve DENSE_EMBEDDING_DIM from app-level settings."""
    from app.services.settings_service import get_setting
    from app.db.session import SessionLocal
    _db = SessionLocal()
    try:
        return get_setting(_db, "DENSE_EMBEDDING_DIM", None) or 1024
    finally:
        _db.close()

logger = logging.getLogger(__name__)

_EMBED_BATCH_SIZE = 32
_QDRANT_UPSERT_BATCH = 100


def _get_qdrant_collection_name(data_store_id: Optional[int], kb_id: Optional[int]) -> str:
    """Determine Qdrant collection name based on document source.
    
    - DataStore documents: ds_{data_store_id}
    - Direct KB uploads: kb_{kb_id}
    """
    if data_store_id:
        return f"ds_{data_store_id}"
    elif kb_id:
        return f"kb_{kb_id}"
    else:
        raise ValueError("Either data_store_id or kb_id must be provided")


def _ensure_qdrant_collection(client: QdrantClient, collection_name: str) -> None:
    """Create a Qdrant collection if it does not exist.

    Handles the race condition where two concurrent ingestion tasks both
    try to create the same collection.  A 409 Conflict simply means
    another thread already created it — that is fine.
    """
    existing = {c.name for c in client.get_collections().collections}
    if collection_name not in existing:
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=_get_embedding_dim(),
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                },
            )
        except UnexpectedResponse as e:
            # 409 means another thread created the collection first — harmless
            if "409" in str(e) or "already exists" in str(e).lower():
                logger.debug(
                    "Qdrant collection %s already created by concurrent task", collection_name
                )
            else:
                raise


def _chunk_id_to_point_id(chunk_id: str) -> str:
    """Convert a SHA-256 hex chunk ID to a deterministic UUID for Qdrant."""
    return str(uuid.uuid5(uuid.NAMESPACE_OID, chunk_id))


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
    # Embeddings API key/base are super_admin-only (app scope).
    from app.services.settings_service import get_setting
    from app.db.session import SessionLocal
    _db = SessionLocal()
    try:
        api_key = get_setting(_db, "EMBEDDING_API_KEY", None) or get_setting(_db, "OPENAI_API_KEY", None)
        api_base = get_setting(_db, "EMBEDDING_API_BASE", None) or get_setting(_db, "OPENAI_API_BASE", None)
        embed_model = get_setting(_db, "DENSE_EMBEDDINGS_MODEL", None)
    finally:
        _db.close()
    # Local servers don't require a key; supply a placeholder when unset.
    if not api_key:
        api_key = "not-required"
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=api_base,
    )
    all_embeddings: List[List[float]] = []
    total_batches = max(1, (len(texts) + _EMBED_BATCH_SIZE - 1) // _EMBED_BATCH_SIZE)
    for batch_idx, i in enumerate(range(0, len(texts), _EMBED_BATCH_SIZE)):
        batch = texts[i : i + _EMBED_BATCH_SIZE]
        response = await client.embeddings.create(
            input=batch,
            model=embed_model,
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
    kb_id: Optional[int] = None,
    document_id: int = None,
    file_name: str = "",
    data_store_id: Optional[int] = None,
) -> List[PointStruct]:
    """Build Qdrant PointStruct list from pre-computed embeddings.
    
    Payload includes both kb_id and data_store_id for proper source tracking.
    """
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
                    "kb_id": kb_id if kb_id else None,
                    "data_store_id": data_store_id,
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
    kb_id: Optional[int] = None,
    document_id: int = None,
    file_name: str = "",
    data_store_id: Optional[int] = None,
    progress_cb=None,   # optional callable(pct: int, msg: str)
    progress_start: int = 40,
    progress_end: int = 80,
    pt=None,            # optional ProgressTimeout for periodic pings
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
    if pt:
        pt.ping()  # signal progress after dense embeddings complete
    # fastembed BM25 tokenization is Python/numpy — does NOT release the GIL.
    # Running the full 2795-text corpus in one executor call still blocks the
    # event loop for 10-30s. Batch it with asyncio.sleep(0) yields between
    # batches so poll requests get served.
    loop = asyncio.get_event_loop()
    sparse_embs: list = []
    embedder = get_sparse_embedder()
    for batch_start in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[batch_start : batch_start + _EMBED_BATCH_SIZE]
        batch_sparse = await loop.run_in_executor(
            None, lambda b=batch: list(embedder.embed(b))
        )
        sparse_embs.extend(batch_sparse)
        if pt:
            pt.ping()  # signal progress after sparse embeddings complete
        await asyncio.sleep(0)  # yield — let poll requests through

    # Build point structs and upsert in batches, yielding the event loop between
    # each batch via asyncio.sleep(0). Pure-Python object construction holds the
    # GIL even inside run_in_executor, so we must yield explicitly to let poll
    # requests through — otherwise the event loop is blocked for 10-30 seconds
    # while 2795 PointStruct objects are built, causing ECONNRESET on the frontend.
    client = get_qdrant_client()
    collection_name = _get_qdrant_collection_name(data_store_id, kb_id)
    _ensure_qdrant_collection(client, collection_name)
    n = len(chunk_payloads)
    for batch_start in range(0, n, _QDRANT_UPSERT_BATCH):
        batch_end = min(batch_start + _QDRANT_UPSERT_BATCH, n)
        batch_chunks = chunk_payloads[batch_start:batch_end]
        batch_dense = dense_embs[batch_start:batch_end]
        batch_sparse = sparse_embs[batch_start:batch_end]

        def _build_upsert_batch(bc=batch_chunks, bd=batch_dense, bs=batch_sparse):
            pts = _build_qdrant_points(bc, bd, bs, kb_id, document_id, file_name, data_store_id)
            client.upsert(collection_name=collection_name, points=pts)

        await loop.run_in_executor(None, _build_upsert_batch)
        if pt:
            pt.ping()  # signal progress after each upsert batch
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
