"""Shared utility functions used across service modules."""

from __future__ import annotations

import hashlib
import logging
import threading
from typing import Any, Optional

from fastembed import SparseTextEmbedding
from openai import OpenAI as SyncOpenAI
from qdrant_client import QdrantClient

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Module-level singletons (lazy-initialised) ────────────────────────────────

_qdrant_client: Optional[QdrantClient] = None
_openai_client: Optional[SyncOpenAI] = None
_sparse_embedder: Optional[SparseTextEmbedding] = None
_singleton_lock = threading.Lock()


def _serialise_doc(doc: Any) -> dict:
    """Serialise a document (LangchainDocument, dict, or generic) to a dict.

    Used across the retrieval pipeline to normalise different document
    representations into a common {"page_content": ..., "metadata": ...} shape.
    """
    if isinstance(doc, dict):
        return doc
    if hasattr(doc, "page_content"):
        return {"page_content": doc.page_content, "metadata": dict(doc.metadata)}
    return {"page_content": str(doc), "metadata": {}}


def content_hash(text: str) -> str:
    """Return a SHA-256 hex digest of *text*.

    Used for deduplication of chunk content.  SHA-256 was chosen over MD5
    to avoid hash collisions and to keep a single canonical implementation
    in one place.
    """
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        with _singleton_lock:
            if _qdrant_client is None:
                _qdrant_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    return _qdrant_client


def get_openai_client() -> SyncOpenAI:
    """Singleton OpenAI client for dense embeddings (super_admin-only settings).

    Uses EMBEDDING_API_KEY / EMBEDDING_API_BASE (app scope), falling back to
    OPENAI_API_KEY / OPENAI_API_BASE from .env.
    """
    global _openai_client
    if _openai_client is None:
        from app.services.settings_service import get_setting
        from app.db.session import SessionLocal
        _db = SessionLocal()
        try:
            api_key = get_setting(_db, "EMBEDDING_API_KEY", None) or get_setting(_db, "OPENAI_API_KEY", None)
            api_base = get_setting(_db, "EMBEDDING_API_BASE", None) or get_setting(_db, "OPENAI_API_BASE", None)
        finally:
            _db.close()
        if not api_key:
            api_key = "not-required"
        _openai_client = SyncOpenAI(
            api_key=api_key,
            base_url=api_base,
        )
    return _openai_client


def get_sparse_embedder() -> SparseTextEmbedding:
    global _sparse_embedder
    if _sparse_embedder is None:
        with _singleton_lock:
            if _sparse_embedder is None:
                _sparse_embedder = SparseTextEmbedding(
                    model_name=settings.SPLADE_MODEL,
                    cache_dir=settings.FASTEMBED_CACHE_DIR,
                )
                # ── SPLADE truncation patch ──────────────────────────────
                # ANOMALY: prithivida/Splade_PP_en_v1 ships with two limits
                # in tokenizer_config.json:
                #   max_length:        128   (HF default, conservative)
                #   model_max_length:  512   (BERT's actual capacity)
                # fastembed's load_tokenizer() takes min(512, 128) = 128 and
                # calls tokenizer.enable_truncation(max_length=128).
                #
                # 128 tokens (~600 chars) is shorter than our default
                # CHUNK_SIZE of 1500 chars.  With the title prepended
                # (f"{title}\n\n{chunk_text}" in _upsert_to_qdrant), a
                # typical chunk is 260-315 tokens.  SPLADE silently
                # truncated at 128, dropping ~50% of every chunk from the
                # sparse index.  Keywords in the second half of a chunk
                # were invisible to sparse retrieval.
                #
                # The underlying BERT model supports 512 position embeddings.
                # We override the truncation to 512 after loading.  This
                # covers our full chunk size with room to spare.
                #
                # ⚠ If SPLADE_MODEL is changed to a non-BERT sparse model
                # (e.g. BM25, MiniCOIL, Bm42), this patch may be wrong —
                # those models have different token limits or no token
                # limit at all.  Audit the new model's tokenizer_config.json
                # and adjust or remove this override accordingly.
                _sparse_embedder.model.tokenizer.enable_truncation(max_length=512)
                logger.info("SPLADE truncation raised to 512 tokens (was 128)")
    return _sparse_embedder


def preload_sparse_embedder() -> None:
    """Eagerly load the SPLADE sparse embedder at app startup.

    Safe to call even if the embedder was already loaded (lazy path).
    On failure, logs a warning but does not raise — the lazy path
    will still attempt to load on first use.
    """
    try:
        get_sparse_embedder()
        logger.info("Sparse embedder loaded: %s", settings.SPLADE_MODEL)
    except Exception as exc:
        logger.warning("Sparse embedder preload failed (will retry on first use): %s", exc)


