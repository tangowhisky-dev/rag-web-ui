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
        _qdrant_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    return _qdrant_client


def get_openai_client() -> SyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = SyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE,
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
    return _sparse_embedder
