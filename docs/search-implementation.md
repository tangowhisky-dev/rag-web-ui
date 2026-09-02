# Search Implementation

## Overview

Retrieval uses a **3-leg hybrid pipeline** with per-leg candidate APIs:

- **Leg 1 — Dense**: Qdrant cosine-similarity search with native MMR diversity
- **Leg 2 — Sparse**: Qdrant learned-sparse search (SPLADE via FastEmbed) with native MMR diversity
- **Leg 3 — Exact**: MySQL InnoDB FULLTEXT search (BM25/TF-IDF, server-side)

Each leg is called independently by the agentic RAG pipeline via single-leg public APIs (`dense_search_docs`, `sparse_search_docs`, `exact_search_docs`). The agentic pipeline merges, reranks, and filters the results. Individual legs can be disabled via settings without re-indexing.

---

## Pipeline

### Agentic Pipeline (`agentic_rag/`)

The sole production path. The agent calls the `rag_retrieve` tool, which runs a graduated relaxation ladder across the three retrieval legs:

```
rewrite_query (standalone question via QUERY_MODEL)
    │
    ▼
rag_retrieve tool — _run_retrieval_pass()
    ├── dense_retrieval_node()    → dense_search_docs()  → Qdrant cosine + MMR
    ├── sparse_retrieval_node()   → sparse_search_docs() → Qdrant SPLADE + MMR
    └── exact_retrieval_node()    → exact_search_docs()  → MySQL FULLTEXT (JOIN documents)
    │          ↓ per-leg: fetch modified_at, recency-aware content_hash dedup
    │
    └── merge_node()              → cross-leg content_hash dedup + semantic dedup
    │
    ▼ (optional: RETRIEVAL_GRAPH_ENABLED)
neo4j_expansion_node — expand_docs_via_graph()
    │
    ▼
reranking_node — cross-encoder reranking (score_threshold configurable)
    │
    ▼
filter_node — threshold filter
    │
    ▼
LLM streaming answer
```

The relaxation ladder tries progressively looser `min_score` thresholds until enough results are found or all levels are exhausted.

---

## Diversity: Native Qdrant MMR

Both dense and sparse legs use Qdrant's native Maximal Marginal Relevance (MMR) to reduce candidate clustering — near-duplicate chunks from the same or similar documents are diversified at the Qdrant level before results reach the application.

MMR is controlled by a single setting:

| Setting | Default | Range | Effect |
|---|---|---|---|
| `QDRANT_MMR_DIVERSITY` | `0.3` | `0.0`–`1.0` | `0.0` = pure relevance (no MMR), `1.0` = pure diversity |

The query is wrapped in `NearestQuery(nearest=vector, mmr=Mmr(diversity=...))`. When `diversity=0.0`, the query is sent as a plain vector (no MMR overhead).

Both legs also request `with_vectors=True` so dense vectors are available in doc metadata (`_dense_vector`) for downstream semantic dedup.

---

## Deduplication: Recency-Aware Exact + Semantic

Deduplication happens at two levels, using a shared `dedup_by_content_hash` function:

### Per-leg dedup (in each retrieval node)

After fetching results, each leg node:
1. Fetches `modified_at` for unique document IDs (1 MySQL query for dense/sparse; exact leg gets it from the SQL JOIN).
2. Stores `_modified_at` (ISO string) in each doc's metadata.
3. Calls `dedup_by_content_hash()` — when two chunks share the same `content_hash`, the one from the document with the latest `modified_at` is kept.

### Cross-leg dedup (in merge_node)

`merge_node` runs two stages:

**Stage 1 — Exact content_hash dedup** (same shared function):
- Combines all legs' docs.
- Groups by `content_hash`.
- For duplicates across legs, keeps the one with the latest `_modified_at`.
- Citations naturally point to the latest document since its chunk is the one retained.

**Stage 2 — Semantic dedup** (`semantic_dedup`):
- Sorts docs newest-first by `_modified_at`.
- For each doc with a `_dense_vector`, computes cosine similarity against already-kept docs from *different* documents.
- If similarity > `DEDUP_SEMANTIC_THRESHOLD`, drops the doc (keeps the newer one).
- Chunks from the same document are never deduped against each other.
- Chunks without `_dense_vector` (e.g. exact-leg-only matches) pass through untouched.

| Setting | Default | Range | Effect |
|---|---|---|---|
| `DEDUP_SEMANTIC_THRESHOLD` | `0.95` | `0.0`–`1.0` | Cosine similarity above which chunks from different documents are collapsed. `1.0` = disabled. |

**Note:** The semantic threshold needs careful tuning. Too low → legitimate cross-document differences are suppressed. Too high → near-duplicate versions coexist in context. Test with real enterprise data before changing.

---

## Single-Leg Public APIs

`backend/app/services/retrieval/retrieval.py`

Each leg has a public API that the agentic pipeline calls independently:

```python
def dense_search_docs(query, kb_ids, datastore_ids, db, org_id=None, top_k=None, min_score=None) -> List[LangchainDocument]
def sparse_search_docs(query, kb_ids, datastore_ids, db, org_id=None, top_k=None, min_score=None) -> List[LangchainDocument]
def exact_search_docs(query, kb_ids, datastore_ids, db, org_id=None, top_k=None, min_score=None) -> List[LangchainDocument]
```

Each leg over-fetches by `top_k × 4` (`_LEG_POOL_MULTIPLIER`) to give the reranker a larger candidate pool. Results include per-leg rank metadata (`_legs`, `_leg_rank`) and `_dense_vector` (dense/sparse legs) for downstream use.

### Dense leg — `_dense_search()`

Embeds the query via the configured OpenAI-compatible embeddings endpoint, then queries Qdrant by cosine distance on the `dense` named vector with native MMR:

```python
query_obj = NearestQuery(nearest=query_vector, mmr=Mmr(diversity=diversity))
hits = get_qdrant_client().query_points(
    collection_name=f"kb_{kb_id}",
    query=query_obj, using="dense",
    limit=candidates, with_payload=True, with_vectors=True,
).points
```

### Sparse leg — `_sparse_search()`

Embeds query via SPLADE (FastEmbed, CPU-local) to produce a sparse `(indices, values)` vector, then queries Qdrant's `sparse` named vector index with native MMR. `with_vectors=True` returns the dense vectors too (Qdrant returns all named vectors).

### Exact leg — `_exact_search()`

MySQL InnoDB FULLTEXT in `NATURAL LANGUAGE MODE` (BM25/TF-IDF, server-side). JOINs the `documents` table to get `modified_at` in the same query — no extra DB round-trip:

```sql
SELECT dc.chunk_text, dc.chunk_metadata, dc.kb_id, dc.document_id, dc.chunk_index,
       COALESCE(d.modified_at, d.created_at) AS modified_at,
       MATCH(dc.chunk_text) AGAINST(:query IN NATURAL LANGUAGE MODE) AS fts_score
FROM   document_chunks dc
JOIN   documents d ON dc.document_id = d.id
WHERE  dc.kb_id IN :kb_ids
  AND  dc.data_store_id IS NULL
  AND  MATCH(dc.chunk_text) AGAINST(:query IN NATURAL LANGUAGE MODE) > 0
ORDER  BY fts_score DESC
LIMIT  :candidates
```

Only rows with `fts_score > 0` are returned (no keyword overlap = excluded).

---

## Document Versioning and Recency

The `Document` model has a `modified_at` column (nullable, indexed), set at ingestion time from the source file's mtime. For existing documents, the migration backfills `modified_at = created_at`. All queries use `COALESCE(modified_at, created_at)` so a value is always available.

This timestamp drives recency-aware dedup: when duplicate or near-duplicate chunks are found across document versions, the chunk from the latest `modified_at` document is kept, and its citation metadata points to that document.

---

## GraphRAG Enrichment

GraphRAG runs as an expansion step in the agentic pipeline, not as a scored retrieval leg.

- **Expansion** (`expand_docs_via_graph`): traverses entity edges from seed chunks to find chunks NOT in the top-K by similarity. These are tagged with `_legs: ["graph"]` in their metadata.

```
Qdrant — source of truth for TEXT and VECTORS
Neo4j  — source of truth for GRAPH TOPOLOGY
```

Vectors are never stored in Neo4j. Cross-reference uses `qdrant_point_id` (the exact UUID Qdrant assigns).

---

## Cross-Encoder Reranking

`backend/app/services/retrieval/reranker.py` — `rerank(query, docs, score_threshold)`

```python
def rerank(query, docs, score_threshold=None):
    # score_threshold defaults to RERANKER_SCORE_THRESHOLD (default -2.0)
    ...
```

Reranker is a HuggingFace cross-encoder (`Xenova/ms-marco-MiniLM-L-12-v2`) running on CPU via ONNX Runtime. It re-scores all candidates against the query and filters by threshold; no cap on number of results.

---

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `RETRIEVAL_TOP_K` | `10` | Chunks returned to LLM |
| `RETRIEVAL_DENSE_ENABLED` | `true` | Enable/disable dense leg |
| `RETRIEVAL_SPARSE_ENABLED` | `true` | Enable/disable sparse leg |
| `RETRIEVAL_EXACT_ENABLED` | `true` | Enable/disable exact leg |
| `DENSE_MIN_SCORE` | `0.5` | Minimum cosine similarity for dense leg |
| `SPARSE_MIN_SCORE` | `5.0` | Minimum score for sparse leg |
| `EXACT_MIN_SCORE` | `0.5` | Minimum FTS score for exact leg |
| `QDRANT_MMR_DIVERSITY` | `0.3` | Qdrant native MMR diversity (0=pure relevance, 1=pure diversity) |
| `DEDUP_SEMANTIC_THRESHOLD` | `0.95` | Cosine similarity for semantic dedup (1.0=disabled) |
| `RERANKER_SCORE_THRESHOLD` | `-2.0` | Minimum logit to pass reranker |

Disabling a leg affects retrieval only. Ingestion always writes to all three stores.

---

## Where retrieval is called

**Agentic pipeline (`agentic_rag/`):**
- The `rag_retrieve` tool calls the single-leg APIs (`dense_search_docs`, `sparse_search_docs`, `exact_search_docs`) in parallel, then merges via `merge_node`, reranks, and filters.
- The graduated relaxation ladder tries progressively looser thresholds until sufficient results are found.

The agentic pipeline is the sole production path. Both the chat and query API endpoints delegate to `run_agentic_rag`.
