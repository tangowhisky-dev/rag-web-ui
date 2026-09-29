"""
3-leg hybrid retrieval with per-leg candidate APIs:

  Leg 1 — Dense   : Qdrant cosine-similarity search on dense embeddings
  Leg 2 — Sparse  : Qdrant learned sparse-vector search (SPLADE via FastEmbed)
  Leg 3 — BM25    : Qdrant server-side BM25 on chunk text + title pseudo-points
                    (replaces the old MySQL InnoDB FULLTEXT exact leg)

Each leg is called independently by the agentic RAG pipeline via the
single-leg public APIs (dense_search_docs, sparse_search_docs,
bm25_search_docs).  lexical_search_docs fuses the two keyword legs
(SPLADE + BM25 + BM25-title) into ONE query_batch_points call per
collection.  The caller merges and reranks the results.

Configuration (.env / settings):
  RETRIEVAL_TOP_K              — number of documents returned           (default 10)
  RETRIEVAL_DENSE_ENABLED      — enable/disable dense leg               (default true)
  RETRIEVAL_SPARSE_ENABLED     — enable/disable sparse leg              (default true)
  RETRIEVAL_BM25_ENABLED       — enable/disable bm25 keyword leg        (default true)
  RETRIEVAL_GRAPH_ENABLED      — enable/disable graph enrichment        (default true)
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from langchain_core.documents import Document as LangchainDocument
from openai import OpenAI as SyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector, NearestQuery, Mmr, Filter, FieldCondition, MatchAny, QueryRequest
from fastembed import SparseTextEmbedding
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
            # tools (semantic_search, rerank_results) can access it for
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

    # Search KB + DataStore collections — independent per-collection queries
    # are fanned out across a small thread pool (the Qdrant client is
    # thread-safe); executor.map preserves submission order so hit ranks
    # stay identical to the old sequential loop.
    collections = [f"kb_{kb_id}" for kb_id in kb_ids] + [f"ds_{ds_id}" for ds_id in datastore_ids]

    def _query_collection(collection_name: str):
        try:
            return collection_name, get_qdrant_client().query_points(
                collection_name=collection_name,
                query=query_obj,
                using="dense",
                limit=candidates,
                with_payload=True,
                with_vectors=True,
                query_filter=qdrant_filter,
            ).points
        except Exception as e:
            logger.warning("dense_search: Qdrant query failed for %s: %s", collection_name, e)
            return collection_name, []

    if collections:
        with ThreadPoolExecutor(max_workers=min(8, len(collections))) as ex:
            for collection_name, hits in ex.map(_query_collection, collections):
                logger.debug("[DENSE] qdrant response | %s | hits=%d", collection_name, len(hits))
                _process_hits(hits, collection_name)

    logger.debug("[DENSE] unique candidates=%d", len(result))
    return result


def _sparse_embed_queries(queries: List[str], db: Session, org_id: Optional[int]):
    """SPLADE-embed all query variants in one batch, applying the configured
    native MMR wrap. Returns per-variant query objects for QueryRequest."""
    logger.debug("[SPARSE] SPLADE embed | model=%s | queries=%d", settings.SPLADE_MODEL, len(queries))
    sparse_embs = list(get_sparse_embedder().embed(queries))
    query_vectors = [
        SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
        for e in sparse_embs
    ]
    logger.debug("[SPARSE] SPLADE response | variants=%d | nnz=%s",
                len(query_vectors), [len(e.indices) for e in sparse_embs])

    # Build MMR-wrapped query if diversity > 0.
    # QDRANT_MMR_DIVERSITY=0.0 means pure relevance (no MMR).
    diversity = get_setting(db, "QDRANT_MMR_DIVERSITY", org_id)
    if diversity > 0.0:
        logger.debug("[SPARSE] using native MMR | diversity=%.2f", diversity)
        return [NearestQuery(nearest=sv, mmr=Mmr(diversity=diversity)) for sv in query_vectors]
    return query_vectors


def _sparse_process_hits(result: Dict[str, _Candidate], hits, qi: int, collection_name: str, min_score: float) -> None:
    """Fold one collection's SPLADE hits into a per-query candidate dict."""
    rank = len(result)
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
        logger.debug("[SPARSE]   q=%d rank=%d score=%.4f text=%r", qi, rank, score, doc.page_content[:80])
        rank += 1
    if filtered:
        logger.debug("[SPARSE] %s | q=%d filtered_by_score=%d", collection_name, qi, filtered)


def _sparse_requests(query_objs, candidates: int, qdrant_filter) -> list:
    return [
        QueryRequest(
            query=qo,
            using="sparse",
            limit=candidates,
            with_payload=True,
            with_vector=True,
            filter=qdrant_filter,
        )
        for qo in query_objs
    ]


@with_retry_sync(max_attempts=3)
def _sparse_search(queries: List[str], kb_ids: List[int], datastore_ids: List[int], db: Session, candidates: int, org_id: Optional[int] = None, min_score: Optional[float] = None, doc_ids: Optional[List[int]] = None) -> List[Dict[str, _Candidate]]:
    """Qdrant learned-sparse search (SPLADE via FastEmbed).

    Searches both KB collections (kb_{kb_id}) and DataStore collections
    (ds_{datastore_id}) for every query variant. All variants are embedded in
    a single batched SPLADE call, then each collection is queried once via
    ``query_batch_points`` (one HTTP round trip per collection instead of one
    per variant). Collections are fanned out over a small thread pool — the
    Qdrant client is thread-safe.

    Returns one candidate dict per query (aligned with ``queries``) — the
    caller RRF-fuses the per-query ranked lists.
    ``min_score`` overrides settings.SPARSE_MIN_SCORE for this call (used by the
    graduated relaxation ladder in atomic search tools).
    """
    query_objs = _sparse_embed_queries(queries, db, org_id)

    min_score = get_setting(db, "SPARSE_MIN_SCORE", org_id) if min_score is None else min_score
    if min_score > -float("inf"):
        logger.debug("[SPARSE] applying min_score=%.2f", min_score)

    # Build Qdrant payload filter from doc_ids (metadata pre-filter).
    qdrant_filter = _build_doc_id_filter(doc_ids) if doc_ids else None
    if qdrant_filter:
        logger.debug("[SPARSE] filtering to %d document_ids", len(doc_ids))

    results: List[Dict[str, _Candidate]] = [dict() for _ in queries]

    # One batched request per collection (all query variants in a single
    # HTTP call), collections fanned out over a small thread pool.
    collections = [f"kb_{kb_id}" for kb_id in kb_ids] + [f"ds_{ds_id}" for ds_id in datastore_ids]

    def _query_collection(collection_name: str):
        try:
            return collection_name, get_qdrant_client().query_batch_points(
                collection_name=collection_name,
                requests=_sparse_requests(query_objs, candidates, qdrant_filter),
            )
        except Exception as e:
            logger.warning("sparse_search: Qdrant query failed for %s: %s", collection_name, e)
            return collection_name, []

    if collections:
        with ThreadPoolExecutor(max_workers=min(8, len(collections))) as ex:
            for collection_name, responses in ex.map(_query_collection, collections):
                for qi, resp in enumerate(responses):
                    _sparse_process_hits(results[qi], resp.points, qi, collection_name, min_score)
                logger.debug("[SPARSE] qdrant response | %s | hits=%s",
                            collection_name, [len(r.points) for r in responses])

    logger.debug("[SPARSE] unique candidates=%s", [len(r) for r in results])
    return results


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

    When extra_queries (synonyms) are provided, all variants are embedded in
    a single batched SPLADE call and each collection is queried once via
    query_batch_points; the per-variant ranked lists are RRF-fused.
    """
    candidates = top_k or get_setting(db, "RETRIEVAL_TOP_K", org_id)
    pool = candidates * _LEG_POOL_MULTIPLIER
    queries = [query] + list(extra_queries or [])
    per_query = _sparse_search(queries, kb_ids, datastore_ids, db, pool, org_id, min_score=min_score, doc_ids=doc_ids)
    ranked_lists = [_candidates_to_docs(cands, "sparse") for cands in per_query]
    if len(ranked_lists) == 1:
        return ranked_lists[0]
    return _rrf_fuse(ranked_lists)


# ── BM25 leg (Qdrant-native keyword search) ───────────────────────────────────

_BM25_MODEL = "qdrant/bm25"
_BM25_CHUNK_VECTOR = "bm25"
_BM25_TITLE_VECTOR = "bm25_title"
_TITLE_EXPANSION_CAP = 5000  # safety bound on chunks fetched for title-matched docs


def _bm25_requests(queries: List[str], candidates: int, qdrant_filter) -> list:
    """Chunk-match + title-pseudo-point QueryRequests for every variant."""
    from qdrant_client.models import Document
    return [
        QueryRequest(
            query=Document(text=q, model=_BM25_MODEL),
            using=_BM25_CHUNK_VECTOR,
            limit=candidates,
            with_payload=True,
            filter=qdrant_filter,
        )
        for q in queries
    ] + [
        QueryRequest(
            query=Document(text=q, model=_BM25_MODEL),
            using=_BM25_TITLE_VECTOR,
            limit=candidates,
            with_payload=True,
            filter=qdrant_filter,
        )
        for q in queries
    ]


def _bm25_process_chunk_hits(result: Dict[str, _Candidate], hits) -> None:
    """Fold one collection's bm25 chunk hits into a per-query candidate dict."""
    rank = len(result)
    for hit in hits:
        score = float(getattr(hit, "score", 0.0) or 0.0)
        pid = str(hit.id)
        if pid in result:
            continue
        doc = _qdrant_payload_to_doc(hit.payload or {})
        doc.metadata["score"] = score
        doc.metadata["_bm25_chunk_score"] = score
        doc.metadata["_qpid"] = pid
        h = content_hash(doc.page_content)
        result[pid] = _Candidate(doc=doc, content_hash=h, exact_rank=rank)
        rank += 1


def _bm25_collect_title_hits(hits, qi: int, collection_name: str,
                             title_scores: List[Dict[int, float]],
                             matched_title_docs: set) -> None:
    for hit in hits:
        doc_id = (hit.payload or {}).get("document_id")
        if doc_id is None:
            continue
        matched_title_docs.add((collection_name, int(doc_id)))
        s = float(getattr(hit, "score", 0.0) or 0.0)
        title_scores[qi][int(doc_id)] = max(title_scores[qi].get(int(doc_id), 0.0), s)


def _bm25_expand_titles(results: List[Dict[str, _Candidate]],
                        title_scores: List[Dict[int, float]],
                        matched_title_docs: set,
                        title_weight: float) -> None:
    """Fetch all chunks of title-matched documents and add
    ``title_weight × title_score`` to each (title pseudo-points carry no
    chunk_text, so exclude them via the _title_point payload flag)."""
    from qdrant_client.models import MatchValue

    if not matched_title_docs:
        return
    by_collection: Dict[str, List[int]] = {}
    for coll, did in matched_title_docs:
        by_collection.setdefault(coll, []).append(did)
    for collection_name, dids in by_collection.items():
        expansion_filter = Filter(
            must=[FieldCondition(key="document_id", match=MatchAny(any=dids))],
            must_not=[FieldCondition(key="_title_point", match=MatchValue(value=True))],
        )
        try:
            offset = None
            fetched = 0
            while fetched < _TITLE_EXPANSION_CAP:
                points, offset = get_qdrant_client().scroll(
                    collection_name=collection_name,
                    scroll_filter=expansion_filter,
                    limit=500,
                    with_payload=True,
                    with_vectors=False,
                    offset=offset,
                )
                if not points:
                    break
                fetched += len(points)
                for p in points:
                    payload = p.payload or {}
                    did = int(payload.get("document_id"))
                    for qi in range(len(results)):
                        t_score = title_scores[qi].get(did)
                        if not t_score:
                            continue
                        pid = str(p.id)
                        result = results[qi]
                        rec = result.get(pid)
                        if rec is None:
                            doc = _qdrant_payload_to_doc(payload)
                            doc.metadata["_bm25_chunk_score"] = 0.0
                            doc.metadata["score"] = title_weight * t_score
                            doc.metadata["_qpid"] = pid
                            h = content_hash(doc.page_content)
                            rec = _Candidate(doc=doc, content_hash=h, exact_rank=0)
                            result[pid] = rec
                        else:
                            rec.doc.metadata["score"] = rec.doc.metadata["score"] + title_weight * t_score
                if offset is None:
                    break
        except Exception as e:
            logger.warning("bm25_search: title expansion failed for %s: %s", collection_name, e)


def _bm25_finalize(results: List[Dict[str, _Candidate]], min_score: float, candidates: int) -> List[Dict[str, _Candidate]]:
    """Apply combined-score min_score filter, sort desc, cap, re-rank."""
    final: List[Dict[str, _Candidate]] = []
    for qi in range(len(results)):
        cands = list(results[qi].values())
        kept = [c for c in cands if min_score <= 0.0 or c.doc.metadata.get("score", 0.0) >= min_score]
        kept.sort(key=lambda c: c.doc.metadata.get("score", 0.0), reverse=True)
        kept = kept[:candidates]
        out: Dict[str, _Candidate] = {}
        for rank, c in enumerate(kept):
            c.exact_rank = rank
            out[str(c.doc.metadata.get("_qpid", rank))] = c
        final.append(out)
    return final


@with_retry_sync(max_attempts=3)
def _bm25_search(queries: List[str], kb_ids: List[int], datastore_ids: List[int], db: Session, candidates: int, org_id: Optional[int] = None, min_score: Optional[float] = None, doc_ids: Optional[List[int]] = None) -> List[Dict[str, _Candidate]]:
    """Qdrant BM25 keyword search — the Qdrant-native replacement for the
    MySQL InnoDB FTS leg.

    Two sub-queries per variant, batched per collection via
    ``query_batch_points`` (server-side inference — ``Document`` inputs, no
    embedder call):

    1. ``bm25`` field on chunk points — lexical chunk match.
    2. ``bm25_title`` field on per-document title pseudo-points — lexical
       title match. Hits resolve back to *all chunks* of the matched
       document via a payload-filtered scroll, each scoring
       ``title_weight × title_score`` — replicating the old MySQL title
       branch (MATCH(title) weight 2.0 returning all chunks of the doc).

    Combined per-chunk score = bm25_chunk + title_weight × bm25_title(doc),
    identical in shape to the MySQL leg's chunk + 2×title merge.
    ``min_score`` overrides settings.BM25_MIN_SCORE for this call (used by the
    graduated relaxation ladder in atomic search tools).
    """
    min_score = get_setting(db, "BM25_MIN_SCORE", org_id) if min_score is None else min_score
    title_weight = float(get_setting(db, "BM25_TITLE_WEIGHT", org_id) or 2.0)

    qdrant_filter = _build_doc_id_filter(doc_ids) if doc_ids else None
    if qdrant_filter:
        logger.debug("[BM25] filtering to %d document_ids", len(doc_ids))

    results: List[Dict[str, _Candidate]] = [dict() for _ in queries]
    # doc_id -> title score, per query variant
    title_scores: List[Dict[int, float]] = [dict() for _ in queries]
    collections = [f"kb_{kb_id}" for kb_id in kb_ids] + [f"ds_{ds_id}" for ds_id in datastore_ids]

    def _query_collection(collection_name: str):
        n = len(queries)
        try:
            responses = get_qdrant_client().query_batch_points(
                collection_name=collection_name,
                requests=_bm25_requests(queries, candidates, qdrant_filter),
            )
            return collection_name, responses[:n], responses[n:]
        except Exception as e:
            logger.warning("bm25_search: Qdrant query failed for %s: %s", collection_name, e)
            return collection_name, [], []

    matched_title_docs: set = set()
    if collections:
        with ThreadPoolExecutor(max_workers=min(8, len(collections))) as ex:
            for collection_name, chunk_resps, title_resps in ex.map(_query_collection, collections):
                for qi, resp in enumerate(chunk_resps):
                    _bm25_process_chunk_hits(results[qi], resp.points)
                for qi, resp in enumerate(title_resps):
                    _bm25_collect_title_hits(resp.points, qi, collection_name, title_scores, matched_title_docs)

    _bm25_expand_titles(results, title_scores, matched_title_docs, title_weight)

    final = _bm25_finalize(results, min_score, candidates)
    logger.debug("[BM25] unique candidates=%s", [len(r) for r in final])
    return final


@with_retry_sync(max_attempts=3)
def _lexical_search(queries: List[str], kb_ids: List[int], datastore_ids: List[int], db: Session, candidates: int, org_id: Optional[int] = None, sparse_min_score: Optional[float] = None, bm25_min_score: Optional[float] = None, doc_ids: Optional[List[int]] = None) -> "Tuple[List[Dict[str, _Candidate]], List[Dict[str, _Candidate]]]":
    """Fused keyword retrieval — SPLADE + BM25 + BM25-title in ONE
    ``query_batch_points`` call per collection (the batch is the union of all
    three request sets, so it is a single HTTP round trip).

    Returns ``(bm25_results, sparse_results)`` — two per-query candidate
    dicts, identical in shape to ``_bm25_search`` and ``_sparse_search``
    outputs, so callers can rank/fuse each leg as before.

    Failure decomposition: ``query_batch_points`` is atomic per call — if the
    batch fails (e.g. a collection missing the bm25 vectors because it has not
    been migrated), the request is retried as two smaller batches so the
    SPLADE leg still returns hits instead of losing both legs.
    """
    sparse_min = get_setting(db, "SPARSE_MIN_SCORE", org_id) if sparse_min_score is None else sparse_min_score
    bm25_min = get_setting(db, "BM25_MIN_SCORE", org_id) if bm25_min_score is None else bm25_min_score
    title_weight = float(get_setting(db, "BM25_TITLE_WEIGHT", org_id) or 2.0)

    query_objs = _sparse_embed_queries(queries, db, org_id)

    qdrant_filter = _build_doc_id_filter(doc_ids) if doc_ids else None
    if qdrant_filter:
        logger.debug("[LEXICAL] filtering to %d document_ids", len(doc_ids))

    sparse_results: List[Dict[str, _Candidate]] = [dict() for _ in queries]
    bm25_results: List[Dict[str, _Candidate]] = [dict() for _ in queries]
    title_scores: List[Dict[int, float]] = [dict() for _ in queries]
    collections = [f"kb_{kb_id}" for kb_id in kb_ids] + [f"ds_{ds_id}" for ds_id in datastore_ids]

    client = get_qdrant_client()

    def _query_collection(collection_name: str):
        n = len(queries)
        sparse_reqs = _sparse_requests(query_objs, candidates, qdrant_filter)
        bm25_reqs = _bm25_requests(queries, candidates, qdrant_filter)
        try:
            responses = client.query_batch_points(
                collection_name=collection_name,
                requests=sparse_reqs + bm25_reqs,
            )
            return collection_name, responses[:n], responses[n:2 * n], responses[2 * n:]
        except Exception as e:
            logger.warning("lexical_search: fused batch failed for %s, splitting legs: %s", collection_name, e)
        # Decompose so a broken/missing bm25 schema doesn't kill SPLADE.
        sparse_resps: list = []
        bm25_resps: list = []
        title_resps: list = []
        try:
            sparse_resps = client.query_batch_points(
                collection_name=collection_name, requests=sparse_reqs,
            )
        except Exception as e:
            logger.warning("lexical_search: sparse sub-batch failed for %s: %s", collection_name, e)
        try:
            rest = client.query_batch_points(
                collection_name=collection_name, requests=bm25_reqs,
            )
            bm25_resps, title_resps = rest[:n], rest[n:]
        except Exception as e:
            logger.warning("lexical_search: bm25 sub-batch failed for %s: %s", collection_name, e)
        return collection_name, sparse_resps, bm25_resps, title_resps

    matched_title_docs: set = set()
    if collections:
        with ThreadPoolExecutor(max_workers=min(8, len(collections))) as ex:
            for collection_name, sparse_resps, bm25_resps, title_resps in ex.map(_query_collection, collections):
                for qi, resp in enumerate(sparse_resps):
                    _sparse_process_hits(sparse_results[qi], resp.points, qi, collection_name, sparse_min)
                for qi, resp in enumerate(bm25_resps):
                    _bm25_process_chunk_hits(bm25_results[qi], resp.points)
                for qi, resp in enumerate(title_resps):
                    _bm25_collect_title_hits(resp.points, qi, collection_name, title_scores, matched_title_docs)

    _bm25_expand_titles(bm25_results, title_scores, matched_title_docs, title_weight)
    bm25_final = _bm25_finalize(bm25_results, bm25_min, candidates)

    logger.debug("[LEXICAL] sparse=%s bm25=%s",
                 [len(r) for r in sparse_results], [len(r) for r in bm25_final])
    return bm25_final, sparse_results


def lexical_search_docs(
    query: str,
    kb_ids: List[int],
    datastore_ids: List[int],
    db: Session,
    org_id: Optional[int] = None,
    top_k: Optional[int] = None,
    doc_ids: Optional[List[int]] = None,
    extra_queries: Optional[List[str]] = None,
) -> "Tuple[List[LangchainDocument], List[LangchainDocument]]":
    """Run the fused keyword legs (Qdrant BM25 + SPLADE) in one batched
    request per collection. Returns ``(bm25_docs, sparse_docs)`` — each list
    RRF-fused across query variants, ready for cross-leg dedup + rerank."""
    candidates = top_k or get_setting(db, "RETRIEVAL_TOP_K", org_id)
    pool = candidates * _LEG_POOL_MULTIPLIER
    queries = [query] + list(extra_queries or [])
    bm25_per_query, sparse_per_query = _lexical_search(
        queries, kb_ids, datastore_ids, db, pool, org_id, doc_ids=doc_ids,
    )
    bm25_lists = [_candidates_to_docs(cands, "exact") for cands in bm25_per_query]
    sparse_lists = [_candidates_to_docs(cands, "sparse") for cands in sparse_per_query]
    bm25_docs = bm25_lists[0] if len(bm25_lists) == 1 else _rrf_fuse(bm25_lists)
    sparse_docs = sparse_lists[0] if len(sparse_lists) == 1 else _rrf_fuse(sparse_lists)
    return bm25_docs, sparse_docs


def bm25_search_docs(
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
    """Run only the Qdrant BM25 keyword leg and return its ranked candidate docs.

    All variants are queried in one ``query_batch_points`` call per collection
    (server-side inference), and the per-variant ranked lists are RRF-fused.
    """
    candidates = top_k or get_setting(db, "RETRIEVAL_TOP_K", org_id)
    pool = candidates * _LEG_POOL_MULTIPLIER
    queries = [query] + list(extra_queries or [])
    per_query = _bm25_search(queries, kb_ids, datastore_ids, db, pool, org_id, min_score=min_score, doc_ids=doc_ids)
    ranked_lists = [_candidates_to_docs(cands, "exact") for cands in per_query]
    if len(ranked_lists) == 1:
        return ranked_lists[0]
    return _rrf_fuse(ranked_lists)
