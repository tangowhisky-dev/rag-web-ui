"""
3-leg hybrid retrieval fused with Reciprocal Rank Fusion (RRF), with optional
Neo4j graph enrichment after merge:

  Leg 1 — Dense   : Qdrant cosine-similarity search on Qwen3 embeddings
  Leg 2 — Sparse  : Qdrant learned sparse-vector search (SPLADE via FastEmbed)
  Leg 3 — Exact   : MySQL InnoDB FULLTEXT search (BM25/TF-IDF, server-side)

  Graph enrichment (post-merge, not a scored leg):
    When GRAPHRAG_ENABLED=true and use_graph_rag=True, after RRF merge the top-K
    docs are enriched with entity/relationship triples from Neo4j. Neo4j is
    queried by (document_id, chunk_index) from each doc's Qdrant payload — the
    cross-reference link established at ingest. Enriched docs then go to the
    reranker so it sees the expanded context.

Configuration (.env / settings):
  HYBRID_DENSE_WEIGHT          — RRF weight for the dense leg          (default 0.5)
  HYBRID_QDRANT_SPARSE_WEIGHT  — RRF weight for the Qdrant sparse leg  (default 0.3)
  HYBRID_EXACT_WEIGHT          — RRF weight for the MySQL exact leg     (default 0.2)
  RETRIEVAL_TOP_K              — number of documents returned           (default 10)
  RETRIEVAL_DENSE_ENABLED      — enable/disable dense leg               (default true)
  RETRIEVAL_QDRANT_SPARSE_ENABLED — enable/disable sparse leg           (default true)
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
from app.schemas.chat import QueryType
from app.services.utils import content_hash, get_qdrant_client, get_openai_client, get_sparse_embedder

logger = logging.getLogger(__name__)


def get_retrieval_config(query_type: QueryType) -> dict:
    """
    Return retrieval config preset for a query type.
    
    Presets override default leg flags and weights based on query classification.
    Falls back to default settings when preset is not found.
    """
    presets = settings.retrieval_config_presets
    preset = presets.get(query_type.value, {})
    
    config = {
        "use_dense": preset.get("use_dense", True),
        "use_sparse": preset.get("use_sparse", True),
        "use_exact": preset.get("use_exact", True),
        "dense_weight": preset.get("dense_weight", settings.HYBRID_DENSE_WEIGHT),
        "sparse_weight": preset.get("sparse_weight", settings.HYBRID_QDRANT_SPARSE_WEIGHT),
        "exact_weight": preset.get("exact_weight", settings.HYBRID_EXACT_WEIGHT),
        "top_k": preset.get("top_k", settings.RETRIEVAL_TOP_K),
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
    qdrant_sparse_rank: int = -1
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
        if self.qdrant_sparse_rank >= 0:
            score += sparse_weight / (_RRF_K + self.qdrant_sparse_rank)
        if self.exact_rank >= 0:
            score += exact_weight / (_RRF_K + self.exact_rank)
        return score





def _qdrant_payload_to_doc(payload: dict) -> LangchainDocument:
    chunk_text = payload.get("chunk_text", "")
    metadata = {k: v for k, v in payload.items() if k != "chunk_text"}
    return LangchainDocument(page_content=chunk_text, metadata=metadata)


# ── Search legs ───────────────────────────────────────────────────────────────

def _dense_search(query: str, kb_ids: List[int], datastore_ids: List[int], candidates: int) -> Dict[str, _Candidate]:
    """Qdrant cosine-similarity search using the dense (OpenAI) embedding.
    
    Searches both KB collections (kb_{kb_id}) and DataStore collections (ds_{datastore_id}).
    """
    logger.info("[DENSE] embedding request | model=%s | query=%r", settings.DENSE_EMBEDDINGS_MODEL, query[:120])
    response = get_openai_client().embeddings.create(
        input=query,
        model=settings.DENSE_EMBEDDINGS_MODEL,
    )
    query_vector = response.data[0].embedding
    logger.info("[DENSE] embedding response | dim=%d | first5=%s",
                len(query_vector), [round(v, 4) for v in query_vector[:5]])

    result: Dict[str, _Candidate] = {}
    rank = 0
    
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
        for hit in hits:
            text = (hit.payload or {}).get("chunk_text", "")
            h = content_hash(text)
            if h not in result:
                result[h] = _Candidate(
                    doc=_qdrant_payload_to_doc(hit.payload or {}),
                    content_hash=h,
                    dense_rank=rank,
                )
                logger.debug("[DENSE]   rank=%d score=%.4f text=%r", rank, getattr(hit, 'score', -1), text[:80])
                rank += 1
    
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
        logger.info("[DENSE] qdrant response | ds_%d | hits=%d", ds_id, len(hits))
        for hit in hits:
            text = (hit.payload or {}).get("chunk_text", "")
            h = content_hash(text)
            if h not in result:
                result[h] = _Candidate(
                    doc=_qdrant_payload_to_doc(hit.payload or {}),
                    content_hash=h,
                    dense_rank=rank,
                )
                logger.debug("[DENSE]   rank=%d score=%.4f text=%r", rank, getattr(hit, 'score', -1), text[:80])
                rank += 1
    logger.info("[DENSE] unique candidates=%d", len(result))
    return result


def _qdrant_sparse_search(query: str, kb_ids: List[int], datastore_ids: List[int], candidates: int) -> Dict[str, _Candidate]:
    """Qdrant learned-sparse search (SPLADE via FastEmbed).
    
    Searches both KB collections (kb_{kb_id}) and DataStore collections (ds_{datastore_id}).
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
            logger.warning("qdrant_sparse_search: Qdrant query failed for kb_%d: %s", kb_id, e)
            continue
        logger.info("[SPARSE] qdrant response | kb_%d | hits=%d", kb_id, len(hits))
        for hit in hits:
            text = (hit.payload or {}).get("chunk_text", "")
            h = content_hash(text)
            if h not in result:
                result[h] = _Candidate(
                    doc=_qdrant_payload_to_doc(hit.payload or {}),
                    content_hash=h,
                    qdrant_sparse_rank=rank,
                )
                logger.debug("[SPARSE]   rank=%d score=%.4f text=%r", rank, getattr(hit, 'score', -1), text[:80])
                rank += 1
    
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
            logger.warning("qdrant_sparse_search: Qdrant query failed for ds_%d: %s", ds_id, e)
            continue
        logger.info("[SPARSE] qdrant response | ds_%d | hits=%d", ds_id, len(hits))
        for hit in hits:
            text = (hit.payload or {}).get("chunk_text", "")
            h = content_hash(text)
            if h not in result:
                result[h] = _Candidate(
                    doc=_qdrant_payload_to_doc(hit.payload or {}),
                    content_hash=h,
                    qdrant_sparse_rank=rank,
                )
                logger.debug("[SPARSE]   rank=%d score=%.4f text=%r", rank, getattr(hit, 'score', -1), text[:80])
                rank += 1
    logger.info("[SPARSE] unique candidates=%d", len(result))
    return result


def _exact_search(query: str, kb_ids: List[int], datastore_ids: List[int], db: Session, candidates: int) -> Dict[str, _Candidate]:
    """MySQL InnoDB FULLTEXT search — exact keyword / BM25 scoring, server-side.
    
    Searches both KB documents and DataStore documents.
    """
    if not query.strip():
        return {}

    # Query KB documents (direct uploads)
    kb_sql = text(
        """
        SELECT chunk_text, chunk_metadata, kb_id,
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
        SELECT chunk_text, chunk_metadata, kb_id,
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
    
    try:
        if kb_ids:
            kb_rows = db.execute(kb_sql, {"query": query, "kb_ids": kb_ids, "candidates": candidates}).fetchall()
        if datastore_ids:
            ds_rows = db.execute(ds_sql, {"query": query, "ds_ids": datastore_ids, "candidates": candidates}).fetchall()
    except Exception as e:
        logger.warning("exact_search: MySQL FTS query failed: %s", e)
        return {}

    # Merge results and re-rank by FTS score
    all_rows = list(kb_rows) + list(ds_rows)
    all_rows.sort(key=lambda r: r.fts_score or 0, reverse=True)
    all_rows = all_rows[:candidates]

    logger.info("[EXACT] MySQL FTS response | rows=%d (kb=%d, ds=%d)", len(all_rows), len(kb_rows), len(ds_rows))
    if all_rows:
        for i, row in enumerate(all_rows[:5]):
            logger.info("  exact[%d] fts_score=%.4f text=%r", i, row.fts_score, (row.chunk_text or "")[:80])

    result: Dict[str, _Candidate] = {}
    for rank, row in enumerate(all_rows):
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
            result[h] = _Candidate(
                doc=LangchainDocument(
                    page_content=chunk_text,
                    metadata=meta,
                ),
                content_hash=h,
                exact_rank=rank,
            )
    return result


# ── RRF merge ─────────────────────────────────────────────────────────────────

def _rrf_merge_candidates(
    dense: Dict[str, "_Candidate"],
    qdrant_sparse: Dict[str, "_Candidate"],
    exact: Dict[str, "_Candidate"],
    top_k: int,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.3,
    exact_weight: float = 0.2,
) -> list["_Candidate"]:
    merged: Dict[str, _Candidate] = {**dense}

    for h, c in qdrant_sparse.items():
        if h in merged:
            merged[h].qdrant_sparse_rank = c.qdrant_sparse_rank
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
            c.qdrant_sparse_rank if c.qdrant_sparse_rank >= 0 else "-",
            c.exact_rank if c.exact_rank >= 0 else "-",
            c.doc.page_content[:80],
        )
    return ranked[:top_k]


# ── Public API ────────────────────────────────────────────────────────────────

def _run_leg(name: str, fn, *args) -> tuple[dict, str | None]:
    """Run a single retrieval leg, catching any exception.
    Returns (results_dict, error_message_or_None)."""
    try:
        return fn(*args), None
    except Exception as exc:
        logger.error("[LEG:%s] failed: %s", name, exc)
        return {}, str(exc)


async def hybrid_search_with_legs(
    query: str,
    kb_ids: List[int],
    db: Session,
    use_dense:     bool = True,
    use_sparse:    bool = True,
    use_exact:     bool = True,
    use_graph_rag: bool = False,
    query_type: Optional[QueryType] = None,
    datastore_ids: Optional[List[int]] = None,
    return_full_pool: bool = False,
) -> dict:
    """
    Like hybrid_search but returns a richer dict:
      {
        "docs": [...],
        "retrieval_info": {
          "legs": {
            "dense":         {"status": "ok"|"failed"|"disabled", "count": N, "error": str|None},
            "qdrant_sparse": {...},
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
    """
    # Apply query-type preset if provided
    dense_weight = settings.HYBRID_DENSE_WEIGHT
    sparse_weight = settings.HYBRID_QDRANT_SPARSE_WEIGHT
    exact_weight = settings.HYBRID_EXACT_WEIGHT

    if query_type is not None:
        preset = get_retrieval_config(query_type)
        use_dense = preset["use_dense"]
        use_sparse = preset["use_sparse"]
        use_exact = preset["use_exact"]
        top_k = preset["top_k"]
        # Apply per-preset weights (overrides global defaults when present)
        dense_weight = preset.get("dense_weight", dense_weight)
        sparse_weight = preset.get("sparse_weight", sparse_weight)
        exact_weight = preset.get("exact_weight", exact_weight)
        logger.info("[RETRIEVAL] query_type=%s | config=%s", query_type.value, preset)
    else:
        top_k = settings.RETRIEVAL_TOP_K
    
    pool  = top_k * 4

    enabled = {
        "dense":         use_dense    and settings.RETRIEVAL_DENSE_ENABLED,
        "qdrant_sparse": use_sparse   and settings.RETRIEVAL_QDRANT_SPARSE_ENABLED,
        "exact":         use_exact    and settings.RETRIEVAL_EXACT_ENABLED,
        "graph":         use_graph_rag and settings.RETRIEVAL_GRAPH_ENABLED,
    }

    legs: dict[str, dict] = {}

    if enabled["dense"]:
        results, err = _run_leg("dense", _dense_search, query, kb_ids, datastore_ids or [], pool)
        legs["dense"] = {"status": "failed" if err else "ok", "count": len(results), "error": err}
        dense = results
    else:
        dense = {}
        legs["dense"] = {"status": "disabled", "count": 0, "error": None}

    if enabled["qdrant_sparse"]:
        results, err = _run_leg("qdrant_sparse", _qdrant_sparse_search, query, kb_ids, datastore_ids or [], pool)
        legs["qdrant_sparse"] = {"status": "failed" if err else "ok", "count": len(results), "error": err}
        qdrant_sparse = results
    else:
        qdrant_sparse = {}
        legs["qdrant_sparse"] = {"status": "disabled", "count": 0, "error": None}

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
        dense, qdrant_sparse, exact, top_k,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        exact_weight=exact_weight,
    )


    # Annotate each doc with which legs found it — used by confidence scoring.
    docs: List[LangchainDocument] = []
    for c in candidates:
        contributing = []
        if c.dense_rank >= 0:         contributing.append("dense")
        if c.qdrant_sparse_rank >= 0: contributing.append("qdrant_sparse")
        if c.exact_rank >= 0:         contributing.append("exact")
        c.doc.metadata["_legs"] = contributing
        docs.append(c.doc)

    logger.info("hybrid_search_with_legs: RRF returned %d docs | failed_legs=%s", len(docs), failed_legs)

    # ── Entity-aware boost (post-RRF, pre-graph) ───────────────────────────
    # For ENTITY_CENTRIC queries: extract entities from query, expand via Neo4j,
    # and boost chunks mentioning those entities.
    if settings.ENTITY_AWARE_ENABLED and query_type == QueryType.ENTITY_CENTRIC and docs:
        try:
            from app.services.entity_extractor import extract_expand_boost
            docs = extract_expand_boost(query, docs, kb_ids)
        except Exception as exc:
            logger.warning("hybrid_search_with_legs: entity boost failed (non-fatal): %s", exc)

    # ── Graph expansion (post-RRF, pre-reranker) ───────────────────────────
    # Traverse Neo4j to find entity-connected chunks NOT returned by vector
    # search. Fetch their text from Qdrant by UUID. Merge into candidate pool.
    # Runs before enrichment so the reranker scores expanded chunks too.
    graph_expansion_count = 0
    if enabled["graph"] and docs:
        try:
            from app.services.graph_service import expand_docs_via_graph
            loop = asyncio.get_running_loop()
            expanded = await loop.run_in_executor(None, lambda: expand_docs_via_graph(docs, kb_ids))
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
            from app.services.graph_service import enrich_docs_with_graph
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
    if settings.RERANKER_ENABLED and docs:
        try:
            from app.services.reranker import rerank
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
    use_graph_rag: bool = False,
    datastore_ids: Optional[List[int]] = None,
) -> List[LangchainDocument]:
    """Run enabled retrieval legs in parallel (sync calls) and merge via RRF.
    
    Searches both KB collections and DataStore collections.
    """
    top_k = settings.RETRIEVAL_TOP_K
    pool = top_k * 4
    datastore_ids = datastore_ids or []

    enabled = {
        "dense": settings.RETRIEVAL_DENSE_ENABLED,
        "qdrant_sparse": settings.RETRIEVAL_QDRANT_SPARSE_ENABLED,
        "exact": settings.RETRIEVAL_EXACT_ENABLED,
        "graph": use_graph_rag and settings.RETRIEVAL_GRAPH_ENABLED,
    }
    logger.info(
        "hybrid_search | kb_ids=%s | ds_ids=%s | top_k=%d | legs=%s",
        kb_ids, datastore_ids, top_k,
        [k for k, v in enabled.items() if v],
    )

    dense        = _dense_search(query, kb_ids, datastore_ids, pool)           if enabled["dense"]          else {}
    qdrant_sparse = _qdrant_sparse_search(query, kb_ids, datastore_ids, pool)  if enabled["qdrant_sparse"]  else {}
    exact        = _exact_search(query, kb_ids, datastore_ids, db, pool)       if enabled["exact"]          else {}

    docs = [c.doc for c in _rrf_merge_candidates(dense, qdrant_sparse, exact, top_k)]

    if enabled["graph"] and docs:
        try:
            from app.services.graph_service import enrich_docs_with_graph
            loop = asyncio.get_running_loop()
            docs = await loop.run_in_executor(None, lambda: enrich_docs_with_graph(docs))
        except Exception as e:
            logger.warning("hybrid_search: graph enrichment failed (non-fatal): %s", e)

    logger.info("hybrid_search returned %d documents", len(docs))
    return docs
