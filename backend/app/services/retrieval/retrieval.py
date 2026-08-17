"""
3-leg hybrid retrieval fused with Reciprocal Rank Fusion (RRF), with optional
Neo4j graph enrichment after merge:

  Leg 1 — Dense   : Qdrant cosine-similarity search on Qwen3 embeddings
  Leg 2 — Sparse  : Qdrant learned sparse-vector search (SPLADE via FastEmbed)
  Leg 3 — Exact   : MySQL InnoDB FULLTEXT search (BM25/TF-IDF, server-side)

  Graph enrichment (post-merge, not a scored leg):
    When RETRIEVAL_GRAPH_ENABLED=true, after RRF merge the top-K docs are
    enriched with entity/relationship triples from Neo4j. Neo4j is
    queried by (document_id, chunk_index) from each doc's Qdrant payload — the
    cross-reference link established at ingest. Enriched docs then go to the
    reranker so it sees the expanded context.

Configuration (.env / settings):
  HYBRID_DENSE_WEIGHT          — RRF weight for the dense leg          (default 0.5)
  HYBRID_SPARSE_WEIGHT  — RRF weight for the Qdrant sparse leg  (default 0.3)
  HYBRID_EXACT_WEIGHT          — RRF weight for the MySQL exact leg     (default 0.2)
  RETRIEVAL_TOP_K              — number of documents returned           (default 10)
  RETRIEVAL_DENSE_ENABLED      — enable/disable dense leg               (default true)
  RETRIEVAL_SPARSE_ENABLED — enable/disable sparse leg           (default true)
  RETRIEVAL_EXACT_ENABLED      — enable/disable exact leg               (default true)
  RETRIEVAL_GRAPH_ENABLED      — enable/disable graph enrichment        (default true)

Absent-leg design
-----------------
A document absent from a leg (no hit, score=0, or leg disabled) contributes
0 to its RRF score from that leg.  It can still surface via the other legs —
this is the correct behaviour: a paraphrase match with no exact keyword
overlap should be returned by the dense/sparse legs, not suppressed.
"""

import json
import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from langchain_core.documents import Document as LangchainDocument
from openai import OpenAI as SyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector
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


# RRF smoothing constant — standard value from the original paper (k=60).
_RRF_K = 60


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class _Candidate:
    doc: LangchainDocument
    content_hash: str
    dense_rank: int = -1           # -1 = absent from this leg
    sparse_rank: int = -1
    exact_rank: int = -1

    def rrf_score(
        self,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.3,
        exact_weight: float = 0.2,
    ) -> float:
        score = 0.0
        if self.dense_rank >= 0:
            score += dense_weight / (_RRF_K + self.dense_rank)
        if self.sparse_rank >= 0:
            score += sparse_weight / (_RRF_K + self.sparse_rank)
        if self.exact_rank >= 0:
            score += exact_weight / (_RRF_K + self.exact_rank)
        return score





def _qdrant_payload_to_doc(payload: dict) -> LangchainDocument:
    chunk_text = payload.get("chunk_text", "")
    metadata = {k: v for k, v in payload.items() if k != "chunk_text"}
    return LangchainDocument(page_content=chunk_text, metadata=metadata)


# ── Search legs ───────────────────────────────────────────────────────────────

@with_retry_sync(max_attempts=3)
def _dense_search(query: str, kb_ids: List[int], datastore_ids: List[int], db: Session, candidates: int, org_id: Optional[int] = None, min_score: Optional[float] = None) -> Dict[str, _Candidate]:
    """Qdrant cosine-similarity search using the dense (OpenAI) embedding.
    
    Searches both KB collections (kb_{kb_id}) and DataStore collections (ds_{datastore_id}).
    ``min_score`` overrides settings.DENSE_MIN_SCORE for this call (used by the
    graduated relaxation ladder in rag_retrieve).
    """
    # DENSE_EMBEDDINGS_MODEL is super_admin-only (app scope).
    from app.services.settings_service import get_setting
    embed_model = get_setting(db, "DENSE_EMBEDDINGS_MODEL", None)
    logger.info("[DENSE] embedding request | model=%s | query=%r", embed_model, query[:120])
    response = get_openai_client().embeddings.create(
        input=query,
        model=embed_model,
    )
    query_vector = response.data[0].embedding
    logger.info("[DENSE] embedding response | dim=%d | first5=%s",
                len(query_vector), [round(v, 4) for v in query_vector[:5]])

    result: Dict[str, _Candidate] = {}
    rank = 0
    min_score = get_setting(db, "DENSE_MIN_SCORE", org_id) if min_score is None else min_score
    if min_score > 0.0:
        logger.info("[DENSE] applying min_cosine=%.2f", min_score)

    # Search KB collections
    for kb_id in kb_ids:
        logger.info("[DENSE] qdrant query | collection=kb_%d | using=dense | limit=%d", kb_id, candidates)
        try:
            hits = get_qdrant_client().query_points(
                collection_name=f"kb_{kb_id}",
                query=query_vector,
                using="dense",
                limit=candidates,
                with_payload=True,
            ).points
        except Exception as e:
            logger.warning("dense_search: Qdrant query failed for kb_%d: %s", kb_id, e)
            continue
        logger.info("[DENSE] qdrant response | kb_%d | hits=%d", kb_id, len(hits))
        filtered = 0
        for hit in hits:
            score = getattr(hit, 'score', -1)
            if min_score > 0.0 and score < min_score:
                filtered += 1
                continue
            text = (hit.payload or {}).get("chunk_text", "")
            h = content_hash(text)
            if h not in result:
                result[h] = _Candidate(
                    doc=_qdrant_payload_to_doc(hit.payload or {}),
                    content_hash=h,
                    dense_rank=rank,
                )
                logger.debug("[DENSE]   rank=%d score=%.4f text=%r", rank, score, text[:80])
                rank += 1
        if filtered:
            logger.info("[DENSE] kb_%d | returned=%d | filtered_by_score=%d", kb_id, len(result), filtered)
    
    # Search DataStore collections
    for ds_id in datastore_ids:
        logger.info("[DENSE] qdrant query | collection=ds_%d | using=dense | limit=%d", ds_id, candidates)
        try:
            hits = get_qdrant_client().query_points(
                collection_name=f"ds_{ds_id}",
                query=query_vector,
                using="dense",
                limit=candidates,
                with_payload=True,
            ).points
        except Exception as e:
            logger.warning("dense_search: Qdrant query failed for ds_%d: %s", ds_id, e)
            continue
        filtered = 0
        for hit in hits:
            score = getattr(hit, 'score', -1)
            if min_score > 0.0 and score < min_score:
                filtered += 1
                continue
            text = (hit.payload or {}).get("chunk_text", "")
            h = content_hash(text)
            if h not in result:
                result[h] = _Candidate(
                    doc=_qdrant_payload_to_doc(hit.payload or {}),
                    content_hash=h,
                    dense_rank=rank,
                )
                logger.debug("[DENSE]   rank=%d score=%.4f text=%r", rank, score, text[:80])
                rank += 1
        if filtered:
            logger.info("[DENSE] ds_%d | returned=%d | filtered_by_score=%d", ds_id, len(result), filtered)
    logger.info("[DENSE] unique candidates=%d", len(result))
    return result


@with_retry_sync(max_attempts=3)
def _sparse_search(query: str, kb_ids: List[int], datastore_ids: List[int], db: Session, candidates: int, org_id: Optional[int] = None, min_score: Optional[float] = None) -> Dict[str, _Candidate]:
    """Qdrant learned-sparse search (SPLADE via FastEmbed).
    
    Searches both KB collections (kb_{kb_id}) and DataStore collections (ds_{datastore_id}).
    ``min_score`` overrides settings.SPARSE_MIN_SCORE for this call (used by the
    graduated relaxation ladder in rag_retrieve).
    """
    logger.info("[SPARSE] SPLADE embed | model=%s | query=%r", settings.SPLADE_MODEL, query[:120])
    sparse_emb = next(iter(get_sparse_embedder().embed([query])))
    query_sparse = SparseVector(
        indices=sparse_emb.indices.tolist(),
        values=sparse_emb.values.tolist(),
    )
    logger.info("[SPARSE] SPLADE response | nnz=%d | top_terms_indices=%s | top_values=%s",
                len(sparse_emb.indices),
                sparse_emb.indices[:5].tolist(),
                [round(v, 4) for v in sparse_emb.values[:5].tolist()])

    result: Dict[str, _Candidate] = {}
    rank = 0
    min_score = get_setting(db, "SPARSE_MIN_SCORE", org_id) if min_score is None else min_score
    if min_score > -float("inf"):
        logger.info("[SPARSE] applying min_score=%.2f", min_score)
    # Search KB collections
    for kb_id in kb_ids:
        logger.info("[SPARSE] qdrant query | collection=kb_%d | using=sparse | limit=%d", kb_id, candidates)
        try:
            hits = get_qdrant_client().query_points(
                collection_name=f"kb_{kb_id}",
                query=query_sparse,
                using="sparse",
                limit=candidates,
                with_payload=True,
            ).points
        except Exception as e:
            logger.warning("sparse_search: Qdrant query failed for kb_%d: %s", kb_id, e)
            continue
        filtered = 0
        for hit in hits:
            score = getattr(hit, 'score', -1)
            if min_score > -float("inf") and score < min_score:
                filtered += 1
                continue
            text = (hit.payload or {}).get("chunk_text", "")
            h = content_hash(text)
            if h not in result:
                result[h] = _Candidate(
                    doc=_qdrant_payload_to_doc(hit.payload or {}),
                    content_hash=h,
                    sparse_rank=rank,
                )
                logger.debug("[SPARSE]   rank=%d score=%.4f text=%r", rank, score, text[:80])
                rank += 1
        if filtered:
            logger.info("[SPARSE] kb_%d | returned=%d | filtered_by_score=%d", kb_id, len(result), filtered)
    
    # Search DataStore collections
    for ds_id in datastore_ids:
        logger.info("[SPARSE] qdrant query | collection=ds_%d | using=sparse | limit=%d", ds_id, candidates)
        try:
            hits = get_qdrant_client().query_points(
                collection_name=f"ds_{ds_id}",
                query=query_sparse,
                using="sparse",
                limit=candidates,
                with_payload=True,
            ).points
        except Exception as e:
            logger.warning("sparse_search: Qdrant query failed for ds_%d: %s", ds_id, e)
            continue
        logger.info("[SPARSE] qdrant response | ds_%d | hits=%d", ds_id, len(hits))
        filtered = 0
        for hit in hits:
            score = getattr(hit, 'score', -1)
            if min_score > -float("inf") and score < min_score:
                filtered += 1
                continue
            text = (hit.payload or {}).get("chunk_text", "")
            h = content_hash(text)
            if h not in result:
                result[h] = _Candidate(
                    doc=_qdrant_payload_to_doc(hit.payload or {}),
                    content_hash=h,
                    sparse_rank=rank,
                )
                logger.debug("[SPARSE]   rank=%d score=%.4f text=%r", rank, score, text[:80])
                rank += 1
        if filtered:
            logger.info("[SPARSE] ds_%d | returned=%d | filtered_by_score=%d", ds_id, len(result), filtered)
    logger.info("[SPARSE] unique candidates=%d", len(result))
    return result


@with_retry_sync(max_attempts=3)
def _exact_search(query: str, kb_ids: List[int], datastore_ids: List[int], db: Session, candidates: int, org_id: Optional[int] = None, min_score: Optional[float] = None) -> Dict[str, _Candidate]:
    """MySQL InnoDB FULLTEXT search — exact keyword / BM25 scoring, server-side.
    
    Searches both KB documents and DataStore documents.
    ``min_score`` overrides settings.EXACT_MIN_SCORE for this call (used by the
    graduated relaxation ladder in rag_retrieve).

    IMPORTANT: This function creates its own fresh SessionLocal() session for
    each retry attempt instead of using the passed ``db`` session. The passed
    session may be shared across LangGraph nodes and can become corrupted when
    a MySQL connection drops, which would cause all retries to fail on the
    same dead connection. A fresh session per retry lets the pool provision
    a new connection.
    """
    from app.db.session import SessionLocal

    if not query.strip():
        return {}

    # Query KB documents (direct uploads)
    kb_sql = text(
        """
        SELECT chunk_text, chunk_metadata, kb_id, document_id, chunk_index,
               MATCH(chunk_text) AGAINST(:query IN NATURAL LANGUAGE MODE) AS fts_score
        FROM   document_chunks
        WHERE  kb_id IN :kb_ids
          AND  data_store_id IS NULL
          AND  MATCH(chunk_text) AGAINST(:query IN NATURAL LANGUAGE MODE) > 0
        ORDER  BY fts_score DESC
        LIMIT  :candidates
        """
    ).bindparams(bindparam("kb_ids", expanding=True))

    # Query DataStore documents
    ds_sql = text(
        """
        SELECT chunk_text, chunk_metadata, kb_id, document_id, chunk_index,
               MATCH(chunk_text) AGAINST(:query IN NATURAL LANGUAGE MODE) AS fts_score
        FROM   document_chunks
        WHERE  data_store_id IN :ds_ids
          AND  MATCH(chunk_text) AGAINST(:query IN NATURAL LANGUAGE MODE) > 0
        ORDER  BY fts_score DESC
        LIMIT  :candidates
        """
    ).bindparams(bindparam("ds_ids", expanding=True))

    logger.info("[EXACT] MySQL FTS query | query=%r | kb_ids=%s | ds_ids=%s | candidates=%d", 
                query[:120], kb_ids, datastore_ids, candidates)
    
    kb_rows = []
    ds_rows = []
    
    for attempt in range(3):
        fresh_db: Session | None = None
        try:
            fresh_db = SessionLocal()
            if kb_ids:
                kb_rows = fresh_db.execute(kb_sql, {"query": query, "kb_ids": kb_ids, "candidates": candidates}).fetchall()
            if datastore_ids:
                ds_rows = fresh_db.execute(ds_sql, {"query": query, "ds_ids": datastore_ids, "candidates": candidates}).fetchall()
            break
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
                return {}
            # Give the pool a moment to replace a bad connection before retrying.
            import time
            time.sleep(0.1 * (2 ** attempt))

    # Merge results and re-rank by FTS score
    all_rows = list(kb_rows) + list(ds_rows)
    all_rows.sort(key=lambda r: r.fts_score or 0, reverse=True)
    all_rows = all_rows[:candidates]

    logger.info("[EXACT] MySQL FTS response | rows=%d (kb=%d, ds=%d)", len(all_rows), len(kb_rows), len(ds_rows))
    if all_rows:
        for i, row in enumerate(all_rows[:5]):
            logger.debug("  exact[%d] fts_score=%.4f text=%r", i, row.fts_score, (row.chunk_text or "")[:80])

    min_score = get_setting(db, "EXACT_MIN_SCORE", org_id) if min_score is None else min_score
    result: Dict[str, _Candidate] = {}
    filtered = 0
    for rank, row in enumerate(all_rows):
        if min_score > 0.0 and (row.fts_score or 0) < min_score:
            filtered += 1
            continue
        chunk_text = row.chunk_text or ""
        h = content_hash(chunk_text)
        if h not in result:
            raw_meta = row.chunk_metadata
            if isinstance(raw_meta, str):
                try:
                    meta = json.loads(raw_meta)
                except (ValueError, TypeError):
                    meta = {}
            elif isinstance(raw_meta, dict):
                meta = raw_meta
            else:
                meta = {}
            # Ensure document_id and chunk_index are in metadata —
            # they're columns on document_chunks but not always in
            # the chunk_metadata JSON. Without these, citations from
            # exact-retrieval docs can't be stored in message_citations.
            if "document_id" not in meta and hasattr(row, "document_id"):
                meta["document_id"] = row.document_id
            if "chunk_index" not in meta and hasattr(row, "chunk_index"):
                meta["chunk_index"] = row.chunk_index
            result[h] = _Candidate(
                doc=LangchainDocument(
                    page_content=chunk_text,
                    metadata=meta,
                ),
                content_hash=h,
                exact_rank=rank,
            )
    if filtered:
        logger.info("[EXACT] returned=%d | filtered_by_score=%d (min=%.2f)", len(result), filtered, min_score)
    return result


# ── RRF merge ─────────────────────────────────────────────────────────────────

def _rrf_merge_candidates(
    dense: Dict[str, "_Candidate"],
    sparse: Dict[str, "_Candidate"],
    exact: Dict[str, "_Candidate"],
    top_k: int,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.3,
    exact_weight: float = 0.2,
) -> list["_Candidate"]:
    merged: Dict[str, _Candidate] = {**dense}

    for h, c in sparse.items():
        if h in merged:
            merged[h].sparse_rank = c.sparse_rank
        else:
            merged[h] = c

    for h, c in exact.items():
        if h in merged:
            merged[h].exact_rank = c.exact_rank
        else:
            merged[h] = c

    ranked = sorted(
        merged.values(),
        key=lambda c: c.rrf_score(dense_weight, sparse_weight, exact_weight),
        reverse=True,
    )

    logger.info("[RRF] total unique candidates=%d | returning top_k=%d | weights=%.2f/%.2f/%.2f",
                len(ranked), top_k, dense_weight, sparse_weight, exact_weight)
    for i, c in enumerate(ranked[:top_k]):
        logger.info(
            "  rrf[%d] score=%.5f dense_rank=%s sparse_rank=%s exact_rank=%s text=%r",
            i, c.rrf_score(dense_weight, sparse_weight, exact_weight),
            c.dense_rank if c.dense_rank >= 0 else "-",
            c.sparse_rank if c.sparse_rank >= 0 else "-",
            c.exact_rank if c.exact_rank >= 0 else "-",
            c.doc.page_content[:80],
        )
    return ranked[:top_k]


# ── Public API ────────────────────────────────────────────────────────────────

def _run_leg(name: str, fn, *args) -> tuple[dict, str | None]:
    """Run a single retrieval leg, catching any exception.    # The decorated leg function already retries internally; this wrapper
    # catches the final failure so other legs can still run.    Returns (results_dict, error_message_or_None)."""
    try:
        return fn(*args), None
    except Exception as exc:
        logger.error("[LEG:%s] failed after retries: %s", name, exc)
        return {}, str(exc)


async def hybrid_search(
    query: str,
    kb_ids: List[int],
    db: Session,
    datastore_ids: Optional[List[int]] = None,
    org_id: Optional[int] = None,
) -> List[LangchainDocument]:
    """Run enabled retrieval legs in parallel (sync calls) and merge via RRF.

    Searches both KB collections and DataStore collections.
    All globally enabled retrieval sources are used; chat-level toggles were removed.

    When org_id is provided, org-overridable settings are resolved via the
    settings service (3-tier precedence: org → app → .env).
    """
    from app.services.settings_service import get_setting

    top_k = get_setting(db, "RETRIEVAL_TOP_K", org_id)
    pool = top_k * 4
    datastore_ids = datastore_ids or []

    enabled = {
        "dense": get_setting(db, "RETRIEVAL_DENSE_ENABLED", org_id),
        "sparse": get_setting(db, "RETRIEVAL_SPARSE_ENABLED", org_id),
        "exact": get_setting(db, "RETRIEVAL_EXACT_ENABLED", org_id),
        "graph": get_setting(db, "RETRIEVAL_GRAPH_ENABLED", org_id),
    }
    logger.info(
        "hybrid_search | kb_ids=%s | ds_ids=%s | top_k=%d | legs=%s",
        kb_ids, datastore_ids, top_k,
        [k for k, v in enabled.items() if v],
    )

    dense        = _dense_search(query, kb_ids, datastore_ids, db, pool, org_id)           if enabled["dense"]          else {}
    sparse = _sparse_search(query, kb_ids, datastore_ids, db, pool, org_id)  if enabled["sparse"]  else {}
    exact        = _exact_search(query, kb_ids, datastore_ids, db, pool, org_id)       if enabled["exact"]          else {}

    docs = [c.doc for c in _rrf_merge_candidates(dense, sparse, exact, top_k)]

    if enabled["graph"] and docs:
        try:
            from app.services.graph import enrich_docs_with_graph
            loop = asyncio.get_running_loop()
            docs = await loop.run_in_executor(None, lambda: enrich_docs_with_graph(docs))
        except Exception as e:
            logger.warning("hybrid_search: graph enrichment failed (non-fatal): %s", e)

    logger.info("hybrid_search returned %d documents", len(docs))
    return docs


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
) -> List[LangchainDocument]:
    """Run only the dense leg and return its ranked candidate docs."""
    candidates = top_k or get_setting(db, "RETRIEVAL_TOP_K", org_id)
    pool = candidates * _LEG_POOL_MULTIPLIER
    return _candidates_to_docs(
        _dense_search(query, kb_ids, datastore_ids, db, pool, org_id, min_score=min_score), "dense"
    )


def sparse_search_docs(
    query: str,
    kb_ids: List[int],
    datastore_ids: List[int],
    db: Session,
    org_id: Optional[int] = None,
    top_k: Optional[int] = None,
    min_score: Optional[float] = None,
) -> List[LangchainDocument]:
    """Run only the sparse leg and return its ranked candidate docs."""
    candidates = top_k or get_setting(db, "RETRIEVAL_TOP_K", org_id)
    pool = candidates * _LEG_POOL_MULTIPLIER
    return _candidates_to_docs(
        _sparse_search(query, kb_ids, datastore_ids, db, pool, org_id, min_score=min_score), "sparse"
    )


def exact_search_docs(
    query: str,
    kb_ids: List[int],
    datastore_ids: List[int],
    db: Session,
    org_id: Optional[int] = None,
    top_k: Optional[int] = None,
    min_score: Optional[float] = None,
) -> List[LangchainDocument]:
    """Run only the exact (MySQL FTS) leg and return its ranked candidate docs."""
    candidates = top_k or get_setting(db, "RETRIEVAL_TOP_K", org_id)
    pool = candidates * _LEG_POOL_MULTIPLIER
    return _candidates_to_docs(
        _exact_search(query, kb_ids, datastore_ids, db, pool, org_id, min_score=min_score), "exact"
    )
