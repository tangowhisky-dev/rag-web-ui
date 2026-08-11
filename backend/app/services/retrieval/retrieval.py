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
from app.models.query_classifier import QueryType
from app.services.infrastructure import content_hash, get_qdrant_client, get_openai_client, get_sparse_embedder
from app.services.agentic_rag.retry import with_retry_sync

logger = logging.getLogger(__name__)


def get_effective_datastore_ids(
    kb_ids: List[int],
    org_id: Optional[int],
    db: Session,
) -> List[int]:
    """Resolve all effective datastore IDs for a set of knowledge bases.

    Queries two tables:
      1. KnowledgeBaseDataStore  — KB-to-datastore links
      2. OrganizationDataStore   — org-to-datastore links (for standalone DataStores)

    Returns the deduplicated union of both.

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

            if org_id and fresh_db and datastore_ids is not None:
                from app.models.datastore import OrganizationDataStore

                ds_org_links = (
                    fresh_db.query(OrganizationDataStore.data_store_id)
                    .filter(OrganizationDataStore.org_id == org_id)
                    .distinct()
                    .all()
                )
                org_ds_ids = [row.data_store_id for row in ds_org_links]
                for ds_id in org_ds_ids:
                    if ds_id not in datastore_ids:
                        datastore_ids.append(ds_id)
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


def get_retrieval_config(query_type: QueryType, db: Optional[Session] = None, org_id: Optional[int] = None) -> dict:
    """
    Return retrieval config preset for a query type.

    Presets override default leg flags and weights based on query classification.
    Falls back to default settings when preset is not found.
    When db and org_id are provided, resolves org-overridable settings via the settings service.
    """
    if db is not None:
        from app.services.settings_service import get_setting
        presets_raw = get_setting(db, "RETRIEVAL_CONFIG_PRESETS", org_id)
        presets = json.loads(presets_raw) if isinstance(presets_raw, str) else (presets_raw or {})
        d_weight = get_setting(db, "HYBRID_DENSE_WEIGHT", org_id)
        s_weight = get_setting(db, "HYBRID_SPARSE_WEIGHT", org_id)
        e_weight = get_setting(db, "HYBRID_EXACT_WEIGHT", org_id)
        top_k = get_setting(db, "RETRIEVAL_TOP_K", org_id)
    else:
        presets = settings.retrieval_config_presets
        d_weight = settings.HYBRID_DENSE_WEIGHT
        s_weight = settings.HYBRID_SPARSE_WEIGHT
        e_weight = settings.HYBRID_EXACT_WEIGHT
        top_k = settings.RETRIEVAL_TOP_K

    preset = presets.get(query_type.value, {})

    config = {
        "dense_weight": preset.get("dense_weight", d_weight),
        "sparse_weight": preset.get("sparse_weight", s_weight),
        "exact_weight": preset.get("exact_weight", e_weight),
        "top_k": preset.get("top_k", top_k),
    }

    return config

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
def _dense_search(query: str, kb_ids: List[int], datastore_ids: List[int], candidates: int, min_score: Optional[float] = None) -> Dict[str, _Candidate]:
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
    min_score = settings.DENSE_MIN_SCORE if min_score is None else min_score
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
def _sparse_search(query: str, kb_ids: List[int], datastore_ids: List[int], candidates: int, min_score: Optional[float] = None) -> Dict[str, _Candidate]:
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
    min_score = settings.SPARSE_MIN_SCORE if min_score is None else min_score
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
def _exact_search(query: str, kb_ids: List[int], datastore_ids: List[int], db: Session, candidates: int, min_score: Optional[float] = None) -> Dict[str, _Candidate]:
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

    min_score = settings.EXACT_MIN_SCORE if min_score is None else min_score
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


async def hybrid_search_with_legs(
    query: str,
    kb_ids: List[int],
    db: Session,
    query_type: Optional[QueryType] = None,
    datastore_ids: Optional[List[int]] = None,
    return_full_pool: bool = False,
    org_id: Optional[int] = None,
) -> dict:
    """
    Like hybrid_search but returns a richer dict:
      {
        "docs": [...],
        "retrieval_info": {
          "legs": {
            "dense":         {"status": "ok"|"failed"|"disabled", "count": N, "error": str|None},
            "sparse": {...},
            "exact":         {...},
            "graph":         {"status": "ok"|"failed"|"disabled", "count": N, "error": str|None},
          },
          "failed_legs": ["dense", ...],
        }
      }
    Each retrieval leg runs independently — a failure in one never blocks the others.
    Graph enrichment runs after RRF merge: not a scored leg, just context expansion.

    Per-call flags AND with global .env settings: a leg is only enabled when
    both the chat-level flag and the global env flag are True.

    When query_type is provided, apply retrieval config preset to override
    default leg flags and weights.

    When org_id is provided, org-overridable settings are resolved via the
    settings service (3-tier precedence: org → app → .env).
    """
    from app.services.settings_service import get_setting

    # Resolve org-overridable settings
    dense_weight = get_setting(db, "HYBRID_DENSE_WEIGHT", org_id)
    sparse_weight = get_setting(db, "HYBRID_SPARSE_WEIGHT", org_id)
    exact_weight = get_setting(db, "HYBRID_EXACT_WEIGHT", org_id)

    if query_type is not None:
        preset = get_retrieval_config(query_type, db, org_id)
        top_k = preset["top_k"]
        # Apply per-preset weights (overrides global defaults when present)
        dense_weight = preset.get("dense_weight", dense_weight)
        sparse_weight = preset.get("sparse_weight", sparse_weight)
        exact_weight = preset.get("exact_weight", exact_weight)
        logger.info("[RETRIEVAL] query_type=%s | config=%s", query_type.value, preset)
    else:
        top_k = get_setting(db, "RETRIEVAL_TOP_K", org_id)

    pool  = top_k * 4

    enabled = {
        "dense": get_setting(db, "RETRIEVAL_DENSE_ENABLED", org_id),
        "sparse": get_setting(db, "RETRIEVAL_SPARSE_ENABLED", org_id),
        "exact": get_setting(db, "RETRIEVAL_EXACT_ENABLED", org_id),
        "graph": get_setting(db, "RETRIEVAL_GRAPH_ENABLED", org_id),
    }

    legs: dict[str, dict] = {}

    if enabled["dense"]:
        results, err = _run_leg("dense", _dense_search, query, kb_ids, datastore_ids or [], pool)
        legs["dense"] = {"status": "failed" if err else "ok", "count": len(results), "error": err}
        dense = results
    else:
        dense = {}
        legs["dense"] = {"status": "disabled", "count": 0, "error": None}

    if enabled["sparse"]:
        results, err = _run_leg("sparse", _sparse_search, query, kb_ids, datastore_ids or [], pool)
        legs["sparse"] = {"status": "failed" if err else "ok", "count": len(results), "error": err}
        sparse = results
    else:
        sparse = {}
        legs["sparse"] = {"status": "disabled", "count": 0, "error": None}

    if enabled["exact"]:
        results, err = _run_leg("exact", _exact_search, query, kb_ids, datastore_ids or [], db, pool)
        legs["exact"] = {"status": "failed" if err else "ok", "count": len(results), "error": err}
        exact = results
    else:
        exact = {}
        legs["exact"] = {"status": "disabled", "count": 0, "error": None}

    failed_legs = [k for k, v in legs.items() if v["status"] == "failed"]
    if failed_legs:
        logger.warning("hybrid_search_with_legs: failed legs=%s", failed_legs)

    candidates = _rrf_merge_candidates(
        dense, sparse, exact, top_k,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        exact_weight=exact_weight,
    )


    # Annotate each doc with which legs found it — used by confidence scoring.
    docs: List[LangchainDocument] = []
    for c in candidates:
        contributing = []
        if c.dense_rank >= 0:         contributing.append("dense")
        if c.sparse_rank >= 0: contributing.append("sparse")
        if c.exact_rank >= 0:         contributing.append("exact")
        c.doc.metadata["_legs"] = contributing
        docs.append(c.doc)

    logger.info("hybrid_search_with_legs: RRF returned %d docs | failed_legs=%s", len(docs), failed_legs)

    # ── Entity-aware boost (post-RRF, pre-graph) ───────────────────────────
    # For ENTITY_CENTRIC queries: extract entities from query, expand via Neo4j,
    # and boost chunks mentioning those entities.
    if get_setting(db, "ENTITY_AWARE_ENABLED", org_id) and query_type == QueryType.ENTITY_CENTRIC and docs:
        try:
            from app.services.graph.entity_extractor import extract_expand_boost
            docs = extract_expand_boost(query, docs, kb_ids, db=db, org_id=org_id)
        except Exception as exc:
            logger.warning("hybrid_search_with_legs: entity boost failed (non-fatal): %s", exc)

    # ── Graph expansion (post-RRF, pre-reranker) ───────────────────────────
    # Traverse Neo4j to find entity-connected chunks NOT returned by vector
    # search. Fetch their text from Qdrant by UUID. Merge into candidate pool.
    # Runs before enrichment so the reranker scores expanded chunks too.
    graph_expansion_count = 0
    if enabled["graph"] and docs:
        try:
            from app.services.graph import expand_docs_via_graph
            loop = asyncio.get_running_loop()
            expanded = await loop.run_in_executor(None, lambda: expand_docs_via_graph(docs, kb_ids, db, org_id))
            if expanded:
                existing_hashes = {content_hash(d.page_content) for d in docs}
                new_docs = [d for d in expanded if content_hash(d.page_content) not in existing_hashes]
                for d in new_docs:
                    d.metadata["_legs"] = ["graph"]
                docs = docs + new_docs
                graph_expansion_count = len(new_docs)
                logger.info(
                    "hybrid_search_with_legs: graph expansion added %d new chunks (total=%d)",
                    graph_expansion_count, len(docs),
                )
        except Exception as exc:
            logger.warning("hybrid_search_with_legs: graph expansion failed (non-fatal): %s", exc)

    # ── Graph enrichment (post-expansion, pre-reranker) ────────────────────
    # Appends entity relationship triples to each candidate's text so the
    # reranker and LLM both see the entity context alongside chunk text.
    graph_enriched_count = 0
    if enabled["graph"] and docs:
        try:
            from app.services.graph import enrich_docs_with_graph
            loop = asyncio.get_running_loop()
            docs = await loop.run_in_executor(None, lambda: enrich_docs_with_graph(docs))
            graph_enriched_count = sum(1 for d in docs if d.metadata.get("_graph_triples", 0) > 0)
            legs["graph"] = {
                "status": "ok",
                "count": graph_enriched_count,
                "expanded": graph_expansion_count,
                "error": None,
            }
        except Exception as exc:
            logger.warning("hybrid_search_with_legs: graph enrichment failed (non-fatal): %s", exc)
            legs["graph"] = {"status": "failed", "count": 0, "expanded": graph_expansion_count, "error": str(exc)}
    else:
        legs["graph"] = {"status": "disabled", "count": 0, "expanded": 0, "error": None}

    # ── Cross-encoder reranking (optional) ────────────────────────────────────
    if get_setting(db, "RERANKER_ENABLED", org_id) and docs:
        try:
            from app.services.retrieval import rerank
            if return_full_pool:
                # score_threshold=float('-inf') ensures ALL docs pass, each with
                # metadata["_reranker_score"] set — no threshold filtering applied.
                docs = rerank(query=query, docs=docs, score_threshold=float('-inf'))
                logger.info(
                    "hybrid_search_with_legs: reranker returned %d docs (full pool)",
                    len(docs),
                )
            else:
                docs = rerank(query=query, docs=docs)
                logger.info("hybrid_search_with_legs: reranker reduced to %d docs", len(docs))
        except Exception as exc:
            logger.warning("hybrid_search_with_legs: reranker failed (using RRF order): %s", exc)

    return {
        "docs": docs,
        "retrieval_info": {
            "legs": legs,
            "failed_legs": failed_legs,
        },
    }


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

    dense        = _dense_search(query, kb_ids, datastore_ids, pool)           if enabled["dense"]          else {}
    sparse = _sparse_search(query, kb_ids, datastore_ids, pool)  if enabled["sparse"]  else {}
    exact        = _exact_search(query, kb_ids, datastore_ids, db, pool)       if enabled["exact"]          else {}

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
    top_k: Optional[int] = None,
    min_score: Optional[float] = None,
) -> List[LangchainDocument]:
    """Run only the dense leg and return its ranked candidate docs."""
    candidates = top_k or settings.RETRIEVAL_TOP_K
    pool = candidates * _LEG_POOL_MULTIPLIER
    return _candidates_to_docs(
        _dense_search(query, kb_ids, datastore_ids, pool, min_score=min_score), "dense"
    )


def sparse_search_docs(
    query: str,
    kb_ids: List[int],
    datastore_ids: List[int],
    top_k: Optional[int] = None,
    min_score: Optional[float] = None,
) -> List[LangchainDocument]:
    """Run only the sparse leg and return its ranked candidate docs."""
    candidates = top_k or settings.RETRIEVAL_TOP_K
    pool = candidates * _LEG_POOL_MULTIPLIER
    return _candidates_to_docs(
        _sparse_search(query, kb_ids, datastore_ids, pool, min_score=min_score), "sparse"
    )


def exact_search_docs(
    query: str,
    kb_ids: List[int],
    datastore_ids: List[int],
    db: Session,
    top_k: Optional[int] = None,
    min_score: Optional[float] = None,
) -> List[LangchainDocument]:
    """Run only the exact (MySQL FTS) leg and return its ranked candidate docs."""
    candidates = top_k or settings.RETRIEVAL_TOP_K
    pool = candidates * _LEG_POOL_MULTIPLIER
    return _candidates_to_docs(
        _exact_search(query, kb_ids, datastore_ids, db, pool, min_score=min_score), "exact"
    )
