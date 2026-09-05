"""
3-leg hybrid retrieval with per-leg candidate APIs:

  Leg 1 — Dense   : Qdrant cosine-similarity search on Qwen3 embeddings
  Leg 2 — Sparse  : Qdrant learned sparse-vector search (SPLADE via FastEmbed)
  Leg 3 — Exact   : MySQL InnoDB FULLTEXT search (BM25/TF-IDF, server-side)

Each leg is called independently by the agentic RAG pipeline via the
single-leg public APIs (dense_search_docs, sparse_search_docs,
exact_search_docs).  The caller merges and reranks the results.

Configuration (.env / settings):
  RETRIEVAL_TOP_K              — number of documents returned           (default 10)
  RETRIEVAL_DENSE_ENABLED      — enable/disable dense leg               (default true)
  RETRIEVAL_SPARSE_ENABLED — enable/disable sparse leg           (default true)
  RETRIEVAL_EXACT_ENABLED      — enable/disable exact leg               (default true)
  RETRIEVAL_GRAPH_ENABLED      — enable/disable graph enrichment        (default true)
"""

import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional

from langchain_core.documents import Document as LangchainDocument
from openai import OpenAI as SyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector, NearestQuery, Mmr, Filter, FieldCondition, MatchAny
from fastembed import SparseTextEmbedding
from sqlalchemy import text, bindparam
from sqlalchemy.orm import Session

from app.core.config import settings
from app.services.settings_service import get_setting
from app.services.infrastructure import content_hash, get_qdrant_client, get_openai_client, get_sparse_embedder
from app.services.agentic_rag.retry import with_retry_sync

logger = logging.getLogger(__name__)


def get_effective_datastore_ids(
    kb_ids: List[int],
    org_id: Optional[int],
    db: Session,
) -> List[int]:
    """Resolve all datastore IDs explicitly linked to the given knowledge bases.

    Only datastores linked via KnowledgeBaseDataStore are returned — org-level
    assignment (OrganizationDataStore) makes a datastore *visible* for linking
    in the UI but does NOT make it queryable. A user must explicitly link a
    datastore to their KB for it to be searched at query time.

    IMPORTANT: This function creates its own fresh SessionLocal() session instead
    of using the passed ``db`` session. The passed session may be shared across
    LangGraph nodes and can become corrupted when a MySQL connection drops, which
    would cascade failures to every subsequent node. A fresh session per call
    isolates failures and allows the pool to provision a new connection.
    """
    from app.db.session import SessionLocal

    datastore_ids: list[int] = []

    for attempt in range(3):
        fresh_db: Session | None = None
        try:
            fresh_db = SessionLocal()
            if kb_ids and fresh_db:
                from app.models.knowledge import KnowledgeBaseDataStore

                datastore_links = (
                    fresh_db.query(KnowledgeBaseDataStore.data_store_id)
                    .filter(KnowledgeBaseDataStore.knowledge_base_id.in_(kb_ids))
                    .distinct()
                    .all()
                )
                datastore_ids = [row.data_store_id for row in datastore_links]
            break
        except Exception as exc:
            logger.warning("get_effective_datastore_ids failed (attempt %d): %s", attempt + 1, exc)
            try:
                if fresh_db is not None:
                    fresh_db.rollback()
            except Exception:
                pass
            if fresh_db is not None:
                try:
                    fresh_db.close()
                except Exception:
                    pass
            if attempt == 2:
                raise
            import time
            time.sleep(0.1 * (2 ** attempt))

    return datastore_ids


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class _Candidate:
    doc: LangchainDocument
    content_hash: str
    dense_rank: int = -1           # -1 = absent from this leg
    sparse_rank: int = -1
    exact_rank: int = -1





def _build_doc_id_filter(doc_ids: Optional[List[int]]) -> Optional[Filter]:
    """Build a Qdrant payload filter restricting results to the given document_ids.

    Returns None when doc_ids is None or empty (no filtering).
    """
    if not doc_ids:
        return None
    return Filter(must=[
        FieldCondition(key="document_id", match=MatchAny(any=doc_ids))
    ])


def _qdrant_payload_to_doc(payload: dict) -> LangchainDocument:
    chunk_text = payload.get("chunk_text", "")
    metadata = {k: v for k, v in payload.items() if k != "chunk_text"}
    return LangchainDocument(page_content=chunk_text, metadata=metadata)


# ── Search legs ───────────────────────────────────────────────────────────────

@with_retry_sync(max_attempts=3)
def _dense_search(query: str, kb_ids: List[int], datastore_ids: List[int], db: Session, candidates: int, org_id: Optional[int] = None, min_score: Optional[float] = None, doc_ids: Optional[List[int]] = None) -> Dict[str, _Candidate]:
    """Qdrant cosine-similarity search using the dense (OpenAI) embedding.

    Searches both KB collections (kb_{kb_id}) and DataStore collections (ds_{datastore_id}).
    Uses native Qdrant MMR when QDRANT_MMR_DIVERSITY > 0 to diversify results.
    Returns dense vectors in metadata for downstream semantic dedup.
    ``min_score`` overrides settings.DENSE_MIN_SCORE for this call (used by the
    graduated relaxation ladder in atomic search tools).
    """
    from app.services.settings_service import get_setting
    embed_model = get_setting(db, "DENSE_EMBEDDINGS_MODEL", None)
    logger.debug("[DENSE] embedding request | model=%s | query=%r", embed_model, query[:120])
    response = get_openai_client().embeddings.create(
        input=query,
        model=embed_model,
    )
    query_vector = response.data[0].embedding
    logger.debug("[DENSE] embedding response | dim=%d | first5=%s",
                len(query_vector), [round(v, 4) for v in query_vector[:5]])

    # Build MMR-wrapped query if diversity > 0.
    # QDRANT_MMR_DIVERSITY=0.0 means pure relevance (no MMR).
    diversity = get_setting(db, "QDRANT_MMR_DIVERSITY", org_id)
    if diversity > 0.0:
        query_obj = NearestQuery(nearest=query_vector, mmr=Mmr(diversity=diversity))
        logger.debug("[DENSE] using native MMR | diversity=%.2f", diversity)
    else:
        query_obj = query_vector

    result: Dict[str, _Candidate] = {}
    rank = 0
    min_score = get_setting(db, "DENSE_MIN_SCORE", org_id) if min_score is None else min_score
    if min_score > 0.0:
        logger.debug("[DENSE] applying min_cosine=%.2f", min_score)

    # Build Qdrant payload filter from doc_ids (metadata pre-filter).
    qdrant_filter = _build_doc_id_filter(doc_ids) if doc_ids else None
    if qdrant_filter:
        logger.debug("[DENSE] filtering to %d document_ids", len(doc_ids))

    def _process_hits(hits, collection_name: str):
        nonlocal rank
        filtered = 0
        for hit in hits:
            score = getattr(hit, 'score', -1)
            if min_score > 0.0 and score < min_score:
                filtered += 1
                continue
            pid = str(hit.id)
            if pid in result:
                continue
            doc = _qdrant_payload_to_doc(hit.payload or {})
            # Store the Qdrant similarity score in metadata so downstream
            # tools (search_dense, rerank_results) can access it for
            # confidence scoring.
            doc.metadata["score"] = float(score)
            # Store dense vector for downstream semantic dedup.
            vec = hit.vector
            if isinstance(vec, dict):
                vec = vec.get("dense")
            if vec:
                doc.metadata["_dense_vector"] = vec
            h = content_hash(doc.page_content)
            result[pid] = _Candidate(
                doc=doc,
                content_hash=h,
                dense_rank=rank,
            )
            logger.debug("[DENSE]   rank=%d score=%.4f text=%r", rank, score, doc.page_content[:80])
            rank += 1
        if filtered:
            logger.debug("[DENSE] %s | filtered_by_score=%d", collection_name, filtered)

    # Search KB collections
    for kb_id in kb_ids:
        logger.debug("[DENSE] qdrant query | collection=kb_%d | using=dense | limit=%d", kb_id, candidates)
        try:
            hits = get_qdrant_client().query_points(
                collection_name=f"kb_{kb_id}",
                query=query_obj,
                using="dense",
                limit=candidates,
                with_payload=True,
                with_vectors=True,
                query_filter=qdrant_filter,
            ).points
        except Exception as e:
            logger.warning("dense_search: Qdrant query failed for kb_%d: %s", kb_id, e)
            continue
        logger.debug("[DENSE] qdrant response | kb_%d | hits=%d", kb_id, len(hits))
        _process_hits(hits, f"kb_{kb_id}")

    # Search DataStore collections
    for ds_id in datastore_ids:
        logger.debug("[DENSE] qdrant query | collection=ds_%d | using=dense | limit=%d", ds_id, candidates)
        try:
            hits = get_qdrant_client().query_points(
                collection_name=f"ds_{ds_id}",
                query=query_obj,
                using="dense",
                limit=candidates,
                with_payload=True,
                with_vectors=True,
                query_filter=qdrant_filter,
            ).points
        except Exception as e:
            logger.warning("dense_search: Qdrant query failed for ds_%d: %s", ds_id, e)
            continue
        logger.debug("[DENSE] qdrant response | ds_%d | hits=%d", ds_id, len(hits))
        _process_hits(hits, f"ds_{ds_id}")

    logger.debug("[DENSE] unique candidates=%d", len(result))
    return result


@with_retry_sync(max_attempts=3)
def _sparse_search(query: str, kb_ids: List[int], datastore_ids: List[int], db: Session, candidates: int, org_id: Optional[int] = None, min_score: Optional[float] = None, doc_ids: Optional[List[int]] = None) -> Dict[str, _Candidate]:
    """Qdrant learned-sparse search (SPLADE via FastEmbed).

    Searches both KB collections (kb_{kb_id}) and DataStore collections (ds_{datastore_id}).
    Uses native Qdrant MMR when QDRANT_MMR_DIVERSITY > 0 to diversify results.
    Returns dense vectors in metadata for downstream semantic dedup.
    ``min_score`` overrides settings.SPARSE_MIN_SCORE for this call (used by the
    graduated relaxation ladder in atomic search tools).
    """
    logger.debug("[SPARSE] SPLADE embed | model=%s | query=%r", settings.SPLADE_MODEL, query[:120])
    sparse_emb = next(iter(get_sparse_embedder().embed([query])))
    query_sparse = SparseVector(
        indices=sparse_emb.indices.tolist(),
        values=sparse_emb.values.tolist(),
    )
    logger.debug("[SPARSE] SPLADE response | nnz=%d | top_terms_indices=%s | top_values=%s",
                len(sparse_emb.indices),
                sparse_emb.indices[:5].tolist(),
                [round(v, 4) for v in sparse_emb.values[:5].tolist()])

    # Build MMR-wrapped query if diversity > 0.
    # QDRANT_MMR_DIVERSITY=0.0 means pure relevance (no MMR).
    diversity = get_setting(db, "QDRANT_MMR_DIVERSITY", org_id)
    if diversity > 0.0:
        query_obj = NearestQuery(nearest=query_sparse, mmr=Mmr(diversity=diversity))
        logger.debug("[SPARSE] using native MMR | diversity=%.2f", diversity)
    else:
        query_obj = query_sparse

    result: Dict[str, _Candidate] = {}
    rank = 0
    min_score = get_setting(db, "SPARSE_MIN_SCORE", org_id) if min_score is None else min_score
    if min_score > -float("inf"):
        logger.debug("[SPARSE] applying min_score=%.2f", min_score)

    # Build Qdrant payload filter from doc_ids (metadata pre-filter).
    qdrant_filter = _build_doc_id_filter(doc_ids) if doc_ids else None
    if qdrant_filter:
        logger.debug("[SPARSE] filtering to %d document_ids", len(doc_ids))

    def _process_hits(hits, collection_name: str):
        nonlocal rank
        filtered = 0
        for hit in hits:
            score = getattr(hit, 'score', -1)
            if min_score > -float("inf") and score < min_score:
                filtered += 1
                continue
            pid = str(hit.id)
            if pid in result:
                continue
            doc = _qdrant_payload_to_doc(hit.payload or {})
            doc.metadata["score"] = float(score)
            # Store dense vector for downstream semantic dedup.
            # Qdrant returns all named vectors when with_vectors=True.
            vec = hit.vector
            if isinstance(vec, dict):
                vec = vec.get("dense")
            if vec:
                doc.metadata["_dense_vector"] = vec
            h = content_hash(doc.page_content)
            result[pid] = _Candidate(
                doc=doc,
                content_hash=h,
                sparse_rank=rank,
            )
            logger.debug("[SPARSE]   rank=%d score=%.4f text=%r", rank, score, doc.page_content[:80])
            rank += 1
        if filtered:
            logger.debug("[SPARSE] %s | filtered_by_score=%d", collection_name, filtered)

    # Search KB collections
    for kb_id in kb_ids:
        logger.debug("[SPARSE] qdrant query | collection=kb_%d | using=sparse | limit=%d", kb_id, candidates)
        try:
            hits = get_qdrant_client().query_points(
                collection_name=f"kb_{kb_id}",
                query=query_obj,
                using="sparse",
                limit=candidates,
                with_payload=True,
                with_vectors=True,
                query_filter=qdrant_filter,
            ).points
        except Exception as e:
            logger.warning("sparse_search: Qdrant query failed for kb_%d: %s", kb_id, e)
            continue
        _process_hits(hits, f"kb_{kb_id}")

    # Search DataStore collections
    for ds_id in datastore_ids:
        logger.debug("[SPARSE] qdrant query | collection=ds_%d | using=sparse | limit=%d", ds_id, candidates)
        try:
            hits = get_qdrant_client().query_points(
                collection_name=f"ds_{ds_id}",
                query=query_obj,
                using="sparse",
                limit=candidates,
                with_payload=True,
                with_vectors=True,
                query_filter=qdrant_filter,
            ).points
        except Exception as e:
            logger.warning("sparse_search: Qdrant query failed for ds_%d: %s", ds_id, e)
            continue
        logger.debug("[SPARSE] qdrant response | ds_%d | hits=%d", ds_id, len(hits))
        _process_hits(hits, f"ds_{ds_id}")

    logger.debug("[SPARSE] unique candidates=%d", len(result))
    return result


def _parse_raw_meta(raw_meta) -> dict:
    if isinstance(raw_meta, str):
        try:
            return json.loads(raw_meta)
        except (ValueError, TypeError):
            return {}
    if isinstance(raw_meta, dict):
        return raw_meta
    return {}


def _enrich_meta_from_row(meta: dict, row) -> None:
    # Ensure document_id and chunk_index are in metadata —
    # they're columns on document_chunks but not always in
    # the chunk_metadata JSON. Without these, citations from
    # exact-retrieval docs can't be stored in message_citations.
    if "document_id" not in meta and hasattr(row, "document_id"):
        meta["document_id"] = row.document_id
    if "chunk_index" not in meta and hasattr(row, "chunk_index"):
        meta["chunk_index"] = row.chunk_index
    # file_name is a column on document_chunks but stripped from
    # chunk_metadata during ingestion. Add it so downstream consumers
    # (search endpoint, citations) can display the source filename.
    if "file_name" not in meta and hasattr(row, "file_name") and row.file_name:
        meta["file_name"] = row.file_name
    # title comes from the documents JOIN, not chunk_metadata.
    if "title" not in meta and hasattr(row, "title") and row.title:
        meta["title"] = row.title
    # Store file_modified_at from the JOIN for recency-aware dedup.
    if hasattr(row, "file_modified_at") and row.file_modified_at:
        meta["_file_modified_at"] = row.file_modified_at.isoformat() if hasattr(row.file_modified_at, "isoformat") else str(row.file_modified_at)
    # Store file_created_at from the JOIN for sort-by-recency in atomic search tools.
    if hasattr(row, "file_created_at") and row.file_created_at:
        meta["_file_created_at"] = row.file_created_at.isoformat() if hasattr(row.file_created_at, "isoformat") else str(row.file_created_at)


def _normalize_metadata(raw_meta, row) -> dict:
    meta = _parse_raw_meta(raw_meta)
    _enrich_meta_from_row(meta, row)
    # Store the FTS score so downstream tools can access it for confidence.
    if hasattr(row, 'fts_score') and row.fts_score is not None:
        meta["score"] = float(row.fts_score)
    return meta


def _run_fts_query_with_retry(query: str, kb_ids: List[int], datastore_ids: List[int], kb_sql, ds_sql, candidates: int, doc_id_params: Optional[dict] = None):
    from app.db.session import SessionLocal

    kb_rows = []
    ds_rows = []
    doc_id_params = doc_id_params or {}

    for attempt in range(3):
        fresh_db: Session | None = None
        try:
            fresh_db = SessionLocal()
            if kb_ids:
                kb_rows = fresh_db.execute(kb_sql, {"query": query, "kb_ids": kb_ids, "candidates": candidates, **doc_id_params}).fetchall()
            if datastore_ids:
                ds_rows = fresh_db.execute(ds_sql, {"query": query, "ds_ids": datastore_ids, "candidates": candidates, **doc_id_params}).fetchall()
            return kb_rows, ds_rows
        except Exception as e:
            logger.warning("exact_search: MySQL FTS query failed (attempt %d): %s", attempt + 1, e)
            try:
                if fresh_db is not None:
                    fresh_db.rollback()
            except Exception:
                pass
            if fresh_db is not None:
                try:
                    fresh_db.close()
                except Exception:
                    pass
            if attempt == 2:
                return None
            # Give the pool a moment to replace a bad connection before retrying.
            import time
            time.sleep(0.1 * (2 ** attempt))
    return None


def _filter_and_dedup_rows(all_rows, min_score: float) -> Dict[str, _Candidate]:
    result: Dict[str, _Candidate] = {}
    filtered = 0
    for rank, row in enumerate(all_rows):
        if min_score > 0.0 and (row.fts_score or 0) < min_score:
            filtered += 1
            continue
        chunk_text = row.chunk_text or ""
        h = content_hash(chunk_text)
        if h not in result:
            meta = _normalize_metadata(row.chunk_metadata, row)
            result[h] = _Candidate(
                doc=LangchainDocument(
                    page_content=chunk_text,
                    metadata=meta,
                ),
                content_hash=h,
                exact_rank=rank,
            )
    if filtered:
        logger.debug("[EXACT] returned=%d | filtered_by_score=%d (min=%.2f)", len(result), filtered, min_score)
    return result


@with_retry_sync(max_attempts=3)
def _exact_search(query: str, kb_ids: List[int], datastore_ids: List[int], db: Session, candidates: int, org_id: Optional[int] = None, min_score: Optional[float] = None, doc_ids: Optional[List[int]] = None) -> Dict[str, _Candidate]:
    """MySQL InnoDB FULLTEXT search — exact keyword / BM25 scoring, server-side.
    
    Searches both KB documents and DataStore documents.
    ``min_score`` overrides settings.EXACT_MIN_SCORE for this call (used by the
    graduated relaxation ladder in atomic search tools).

    IMPORTANT: This function creates its own fresh SessionLocal() session for
    each retry attempt instead of using the passed ``db`` session. The passed
    session may be shared across LangGraph nodes and can become corrupted when
    a MySQL connection drops, which would cause all retries to fail on the
    same dead connection. A fresh session per retry lets the pool provision
    a new connection.
    """
    if not query.strip():
        return {}

    # Build optional doc_id filter clause for MySQL queries.
    doc_id_clause = " AND d.id IN :doc_ids" if doc_ids else ""
    doc_id_params = {"doc_ids": list(doc_ids)} if doc_ids else {}
    if doc_ids:
        logger.debug("[EXACT] filtering to %d document_ids", len(doc_ids))

    # bindparam for doc_ids is only declared when the placeholder is in the SQL text.
    kb_binds = [bindparam("kb_ids", expanding=True)]
    ds_binds = [bindparam("ds_ids", expanding=True)]
    if doc_ids:
        kb_binds.append(bindparam("doc_ids", expanding=True))
        ds_binds.append(bindparam("doc_ids", expanding=True))

    # Query KB documents — JOIN documents for file_modified_at and title.
    # Title matches get 2x weight: a chunk from a document whose title
    # matches the query is more relevant than one matching only in body.
    kb_sql = text(
        f"""
        SELECT dc.chunk_text, dc.chunk_metadata, dc.kb_id, dc.document_id, dc.chunk_index,
               dc.file_name, d.title,
               COALESCE(d.file_modified_at, d.file_created_at, d.created_at) AS file_modified_at,
               COALESCE(d.file_created_at, d.created_at) AS file_created_at,
               (MATCH(dc.chunk_text) AGAINST(:query IN NATURAL LANGUAGE MODE)
                + COALESCE(MATCH(d.title) AGAINST(:query IN NATURAL LANGUAGE MODE), 0) * 2.0
               ) AS fts_score
        FROM   document_chunks dc
        JOIN   documents d ON dc.document_id = d.id
        WHERE  dc.kb_id IN :kb_ids
          AND  dc.data_store_id IS NULL{doc_id_clause}
          AND  (MATCH(dc.chunk_text) AGAINST(:query IN NATURAL LANGUAGE MODE) > 0
                OR MATCH(d.title) AGAINST(:query IN NATURAL LANGUAGE MODE) > 0)
        ORDER  BY fts_score DESC
        LIMIT  :candidates
        """
    ).bindparams(*kb_binds)

    # Query DataStore documents — same title-weighted scoring
    ds_sql = text(
        f"""
        SELECT dc.chunk_text, dc.chunk_metadata, dc.kb_id, dc.document_id, dc.chunk_index,
               dc.file_name, d.title,
               COALESCE(d.file_modified_at, d.file_created_at, d.created_at) AS file_modified_at,
               COALESCE(d.file_created_at, d.created_at) AS file_created_at,
               (MATCH(dc.chunk_text) AGAINST(:query IN NATURAL LANGUAGE MODE)
                + COALESCE(MATCH(d.title) AGAINST(:query IN NATURAL LANGUAGE MODE), 0) * 2.0
               ) AS fts_score
        FROM   document_chunks dc
        JOIN   documents d ON dc.document_id = d.id
        WHERE  dc.data_store_id IN :ds_ids{doc_id_clause}
          AND  (MATCH(dc.chunk_text) AGAINST(:query IN NATURAL LANGUAGE MODE) > 0
                OR MATCH(d.title) AGAINST(:query IN NATURAL LANGUAGE MODE) > 0)
        ORDER  BY fts_score DESC
        LIMIT  :candidates
        """
    ).bindparams(*ds_binds)

    logger.debug("[EXACT] MySQL FTS query | query=%r | kb_ids=%s | ds_ids=%s | candidates=%d", 
                query[:120], kb_ids, datastore_ids, candidates)

    rows = _run_fts_query_with_retry(query, kb_ids, datastore_ids, kb_sql, ds_sql, candidates, doc_id_params)
    if rows is None:
        return {}
    kb_rows, ds_rows = rows

    # Merge results and re-rank by FTS score
    all_rows = list(kb_rows) + list(ds_rows)
    all_rows.sort(key=lambda r: r.fts_score or 0, reverse=True)
    all_rows = all_rows[:candidates]

    logger.debug("[EXACT] MySQL FTS response | rows=%d (kb=%d, ds=%d)", len(all_rows), len(kb_rows), len(ds_rows))
    if all_rows:
        for i, row in enumerate(all_rows[:5]):
            logger.debug("  exact[%d] fts_score=%.4f text=%r", i, row.fts_score, (row.chunk_text or "")[:80])

    min_score = get_setting(db, "EXACT_MIN_SCORE", org_id) if min_score is None else min_score
    return _filter_and_dedup_rows(all_rows, min_score)


# ── Recency-aware dedup helpers (shared by leg nodes and merge_node) ──────────

def _get_modified_at(doc: dict) -> str:
    """Extract _file_modified_at from a serialized doc's metadata as a sortable string.

    Falls back to empty string (sorts oldest) if missing.
    """
    return doc.get("metadata", {}).get("_file_modified_at", "")


def dedup_by_content_hash(docs: list[dict]) -> list[dict]:
    """Recency-aware exact dedup by content_hash.

    When two docs share the same content_hash, keeps the one from the document
    with the latest _file_modified_at. Used by both retrieval leg nodes (per-leg
    dedup) and merge_node (cross-leg dedup).
    """
    by_hash: dict[str, dict] = {}
    for doc in docs:
        meta = doc.get("metadata", {})
        h = meta.get("content_hash") or content_hash(doc.get("page_content", ""))
        if h not in by_hash:
            by_hash[h] = doc
        else:
            # Keep the one with the latest _file_modified_at
            if _get_modified_at(doc) > _get_modified_at(by_hash[h]):
                by_hash[h] = doc
    return list(by_hash.values())


def semantic_dedup(docs: list[dict], threshold: float) -> list[dict]:
    """Semantic near-duplicate removal using dense cosine similarity.

    Greedy newest-first: for each chunk, if its dense vector is >threshold
    similar to an already-kept chunk from a *different* document, drop it.
    Chunks from the same document are never deduped against each other
    (they may be legitimately similar adjacent sections).

    Chunks without _dense_vector pass through untouched.
    """
    if threshold >= 1.0 or len(docs) <= 1:
        return docs

    import numpy as np

    # Sort newest-first by _file_modified_at
    sorted_docs = sorted(docs, key=_get_modified_at, reverse=True)

    kept: list[dict] = []
    for doc in sorted_docs:
        meta = doc.get("metadata", {})
        vec = meta.get("_dense_vector")
        doc_id = meta.get("document_id")

        if vec is None:
            kept.append(doc)
            continue

        vec_np = np.array(vec, dtype=np.float32)
        vec_norm = np.linalg.norm(vec_np)
        if vec_norm == 0:
            kept.append(doc)
            continue

        is_dup = False
        for kept_doc in kept:
            kept_meta = kept_doc.get("metadata", {})
            if kept_meta.get("document_id") == doc_id:
                continue  # same document, different section
            kept_vec = kept_meta.get("_dense_vector")
            if kept_vec is None:
                continue
            kept_np = np.array(kept_vec, dtype=np.float32)
            kept_norm = np.linalg.norm(kept_np)
            if kept_norm == 0:
                continue
            sim = float(np.dot(vec_np, kept_np) / (vec_norm * kept_norm))
            if sim > threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(doc)
    return kept


# ── Single-leg public API (used by agentic RAG nodes) ─────────────────────────

# Shared candidate pool multiplier — large enough for downstream reranking.
_LEG_POOL_MULTIPLIER = 4


def _candidates_to_docs(candidates: Dict[str, _Candidate], leg: str) -> List[LangchainDocument]:
    """Convert a candidate dict to an ordered list of LangchainDocuments.

    Preserves per-leg rank and marks which leg produced each doc.
    """
    docs: List[LangchainDocument] = []
    for c in sorted(candidates.values(), key=lambda x: getattr(x, f"{leg}_rank"), reverse=False):
        if getattr(c, f"{leg}_rank") < 0:
            continue
        c.doc.metadata["_legs"] = [leg]
        c.doc.metadata["_leg_rank"] = getattr(c, f"{leg}_rank")
        docs.append(c.doc)
    return docs


def dense_search_docs(
    query: str,
    kb_ids: List[int],
    datastore_ids: List[int],
    db: Session,
    org_id: Optional[int] = None,
    top_k: Optional[int] = None,
    min_score: Optional[float] = None,
    doc_ids: Optional[List[int]] = None,
) -> List[LangchainDocument]:
    """Run only the dense leg and return its ranked candidate docs."""
    candidates = top_k or get_setting(db, "RETRIEVAL_TOP_K", org_id)
    pool = candidates * _LEG_POOL_MULTIPLIER
    return _candidates_to_docs(
        _dense_search(query, kb_ids, datastore_ids, db, pool, org_id, min_score=min_score, doc_ids=doc_ids), "dense"
    )


def _rrf_fuse(ranked_lists: List[List[LangchainDocument]], k: int = 60) -> List[LangchainDocument]:
    """Reciprocal Rank Fusion — fuse multiple ranked lists into one.

    score(doc) = sum(1 / (k + rank(doc))) across all input lists.
    Docs are deduplicated by content_hash; the highest-scoring copy wins.
    """
    if not ranked_lists:
        return []
    if len(ranked_lists) == 1:
        return ranked_lists[0]

    scores: Dict[str, float] = {}
    best_doc: Dict[str, LangchainDocument] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            h = doc.metadata.get("content_hash") or content_hash(doc.page_content)
            score = 1.0 / (k + rank)
            scores[h] = scores.get(h, 0.0) + score
            if h not in best_doc:
                best_doc[h] = doc

    fused = sorted(best_doc.values(), key=lambda d: scores.get(
        d.metadata.get("content_hash") or content_hash(d.page_content), 0.0
    ), reverse=True)
    return fused


def sparse_search_docs(
    query: str,
    kb_ids: List[int],
    datastore_ids: List[int],
    db: Session,
    org_id: Optional[int] = None,
    top_k: Optional[int] = None,
    min_score: Optional[float] = None,
    doc_ids: Optional[List[int]] = None,
    extra_queries: Optional[List[str]] = None,
) -> List[LangchainDocument]:
    """Run only the sparse leg and return its ranked candidate docs.

    When extra_queries (synonyms) are provided, runs one search per query
    and RRF-fuses the results.
    """
    candidates = top_k or get_setting(db, "RETRIEVAL_TOP_K", org_id)
    pool = candidates * _LEG_POOL_MULTIPLIER
    main_results = _candidates_to_docs(
        _sparse_search(query, kb_ids, datastore_ids, db, pool, org_id, min_score=min_score, doc_ids=doc_ids), "sparse"
    )
    if not extra_queries:
        return main_results
    ranked_lists = [main_results]
    for sq in extra_queries:
        try:
            ranked_lists.append(_candidates_to_docs(
                _sparse_search(sq, kb_ids, datastore_ids, db, pool, org_id, min_score=min_score, doc_ids=doc_ids), "sparse"
            ))
        except Exception as exc:
            logger.warning("[sparse_search] synonym query %r failed: %s", sq, exc)
    return _rrf_fuse(ranked_lists)


def exact_search_docs(
    query: str,
    kb_ids: List[int],
    datastore_ids: List[int],
    db: Session,
    org_id: Optional[int] = None,
    top_k: Optional[int] = None,
    min_score: Optional[float] = None,
    doc_ids: Optional[List[int]] = None,
    extra_queries: Optional[List[str]] = None,
) -> List[LangchainDocument]:
    """Run only the exact (MySQL FTS) leg and return its ranked candidate docs.

    When extra_queries (synonyms) are provided, runs one search per query
    and RRF-fuses the results.
    """
    candidates = top_k or get_setting(db, "RETRIEVAL_TOP_K", org_id)
    pool = candidates * _LEG_POOL_MULTIPLIER
    main_results = _candidates_to_docs(
        _exact_search(query, kb_ids, datastore_ids, db, pool, org_id, min_score=min_score, doc_ids=doc_ids), "exact"
    )
    if not extra_queries:
        return main_results
    ranked_lists = [main_results]
    for sq in extra_queries:
        try:
            ranked_lists.append(_candidates_to_docs(
                _exact_search(sq, kb_ids, datastore_ids, db, pool, org_id, min_score=min_score, doc_ids=doc_ids), "exact"
            ))
        except Exception as exc:
            logger.warning("[exact_search] synonym query %r failed: %s", sq, exc)
    return _rrf_fuse(ranked_lists)
