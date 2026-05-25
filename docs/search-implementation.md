# Search Implementation

## Overview

Retrieval uses a **3-leg hybrid pipeline** fused with weighted Reciprocal Rank Fusion (RRF):

- **Leg 1 — Dense**: Qdrant cosine-similarity search on OpenAI-compatible embeddings
- **Leg 2 — Sparse**: Qdrant learned-sparse search (SPLADE via FastEmbed, CPU-local)
- **Leg 3 — Exact**: MySQL InnoDB FULLTEXT search (BM25/TF-IDF, server-side)

All three legs run independently; their ranked lists are merged by weighted RRF. Individual legs can be disabled via `.env` without re-indexing.

---

## Pipeline

### Fast / Thinking mode (`fast_pipeline.py`)

```
rewrite_query (standalone question via QUERY_MODEL)
    │
    ▼
hybrid_search_with_legs()
    ├── _dense_search()          → Qdrant cosine similarity (dense vectors)
    ├── _qdrant_sparse_search()  → Qdrant SPLADE sparse vectors
    └── _exact_search()          → MySQL InnoDB FULLTEXT (NATURAL LANGUAGE MODE)
    │          ↓
    └── _rrf_merge_candidates()  → weighted RRF → top-K documents
    │
    ▼ (optional: GRAPHRAG_ENABLED + use_graph_rag)
graph_service.py — expand_docs_via_graph() + enrich_docs_with_graph()
    │
    ▼ (optional: RERANKER_ENABLED)
reranker.py — cross-encoder reranking (score_threshold configurable)
    │
    ▼
LLM streaming answer
```

### Agentic mode (`rag_graph.py`)

Parallel sub-query retrieval with reinforced deduplication:

```
decompose_query → [sub_query_1, ..., sub_query_N]
    │
    ▼
asyncio.gather(hybrid_search_with_legs(sq) for sq in sub_queries)
    │
    ▼
_dedup_and_reinforce()
    └── Chunks found by N sub-queries get score × N (reinforced scoring)
    └── Deduplicated by content hash, sorted by reinforced score
    │
    ▼
[Retry 1 if uncovered: widened_retrieval]
    └── Same legs, reranker threshold relaxed to -5.0, merges with prior docs
    │
    ▼
[Retry 2 if still uncovered: keyword_search_loop]
    └── LLM extracts broad + narrow keywords → MySQL FULLTEXT only → merges
```

---

## Hybrid Search Entry Point

`backend/app/services/retrieval.py` — `hybrid_search_with_legs()`

```python
async def hybrid_search_with_legs(
    query: str,
    kb_ids: List[int],
    db: Session,
    use_dense: bool = True,
    use_sparse: bool = True,
    use_exact: bool = True,
    use_graph_rag: bool = False,
    query_type: Optional[QueryType] = None,
) -> dict:
    # Returns {"docs": [...], "retrieval_info": {"legs": {...}, "failed_legs": [...]}}
```

Each leg runs independently — a failure in one never blocks the others.

### Dense leg — `_dense_search()`

Embeds the query via the configured OpenAI-compatible embeddings endpoint, then queries Qdrant by cosine distance on the `dense` named vector.

```python
response = _get_openai_client().embeddings.create(input=query, model=settings.DENSE_EMBEDDINGS_MODEL)
hits = _get_qdrant_client().query_points(
    collection_name=f"kb_{kb_id}",
    query=response.data[0].embedding,
    using="dense", limit=candidates, with_payload=True,
).points
```

### Sparse leg — `_qdrant_sparse_search()`

Embeds query via SPLADE (FastEmbed, CPU-local) to produce a sparse `(indices, values)` vector, then queries Qdrant's `sparse` named vector index.

### Exact leg — `_exact_search()`

MySQL InnoDB FULLTEXT in `NATURAL LANGUAGE MODE` (BM25/TF-IDF, server-side, no client index):

```sql
SELECT chunk_text, chunk_metadata,
       MATCH(chunk_text) AGAINST(:query IN NATURAL LANGUAGE MODE) AS fts_score
FROM   document_chunks
WHERE  kb_id IN :kb_ids
  AND  MATCH(chunk_text) AGAINST(:query IN NATURAL LANGUAGE MODE) > 0
ORDER  BY fts_score DESC
LIMIT  :candidates
```

Only rows with `fts_score > 0` are returned (no keyword overlap = excluded).

### RRF merge — `_rrf_merge_candidates()`

```
score(doc) = HYBRID_DENSE_WEIGHT         / (60 + dense_rank)
           + HYBRID_QDRANT_SPARSE_WEIGHT / (60 + qdrant_sparse_rank)
           + HYBRID_EXACT_WEIGHT         / (60 + exact_rank)
```

A leg absent for a document (rank == -1) contributes 0. Constant 60 from Cormack et al. (2009). Each leg over-fetches by `top_k × 4` so documents ranked outside the per-leg top-K are still visible to the merge.

### Adaptive presets

When `query_type` is provided, `get_retrieval_config(query_type)` overrides leg weights and `top_k`:

| Query type | Dense | Sparse | Exact | top_k | Notes |
|---|---|---|---|---|---|
| FACTUAL | 0.5 | 0.3 | 0.2 | default | Balanced across all legs |
| ENTITY_CENTRIC | 0.6 | 0.2 | 0.2 | default | Dense-heavy; entity proximity matters |
| MULTI_PART | 0.5 | 0.5 | 0.0 | default | Dense + sparse; no exact |
| AMBIGUOUS | 0.5 | 0.3 | 0.2 | 15 | Conservative wider net |

### Reinforced scoring (Agentic only)

When multiple sub-queries retrieve the same chunk, scores are accumulated:

```python
# In _dedup_and_reinforce()
if content_hash in seen:
    prev["_reinforced_score"] += current_score
    prev["_retrieval_count"] += 1
```

A chunk central to 3 sub-queries ranks higher than one relevant to just 1, without any additional LLM calls.

---

## GraphRAG Enrichment

GraphRAG runs **after** RRF merge, not as a scored leg.

- **Expansion** (`expand_docs_via_graph`): traverses entity edges from seed chunks to find chunks NOT in the top-K by similarity. These are tagged with `_legs: ["graph"]` in their metadata.
- **Enrichment** (`enrich_docs_with_graph`): appends entity/relationship triples as `[Graph context]` text to every candidate (seed + expanded).

```
Qdrant — source of truth for TEXT and VECTORS
Neo4j  — source of truth for GRAPH TOPOLOGY
```

Vectors are never stored in Neo4j. Cross-reference uses `qdrant_point_id` (the exact UUID Qdrant assigns).

---

## Cross-Encoder Reranking

`backend/app/services/reranker.py` — `rerank(query, docs, score_threshold)`

```python
def rerank(query, docs, score_threshold=None):
    # score_threshold defaults to RERANKER_SCORE_THRESHOLD (default -2.0)
    # Agentic widened_retrieval passes score_threshold=-5.0 for looser filtering
    ...
```

Reranker is a HuggingFace cross-encoder running on CPU. It re-scores all candidates against the query and filters by threshold; no cap on number of results. Disabled by setting `RERANKER_ENABLED=false`.

---

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `RETRIEVAL_TOP_K` | `10` | Chunks returned to LLM |
| `HYBRID_DENSE_WEIGHT` | `0.5` | RRF weight for dense leg |
| `HYBRID_QDRANT_SPARSE_WEIGHT` | `0.3` | RRF weight for SPLADE leg |
| `HYBRID_EXACT_WEIGHT` | `0.2` | RRF weight for MySQL FTS leg |
| `RETRIEVAL_DENSE_ENABLED` | `true` | Enable/disable dense leg |
| `RETRIEVAL_QDRANT_SPARSE_ENABLED` | `true` | Enable/disable sparse leg |
| `RETRIEVAL_EXACT_ENABLED` | `true` | Enable/disable exact leg |
| `RERANKER_ENABLED` | `true` | Enable cross-encoder reranker |
| `RERANKER_SCORE_THRESHOLD` | `-2.0` | Minimum logit to pass reranker |

Disabling a leg affects retrieval only. Ingestion always writes to all three stores.

---

## Where retrieval is called

**Fast/Thinking:** `fast_pipeline.fast_stream()` calls `hybrid_search_with_legs()` once with the rewritten query.

**Agentic:** `rag_graph.parallel_retrieval_node()` calls `hybrid_search_with_legs()` once per sub-query via `asyncio.gather`, then deduplicates with reinforced scoring. `widened_retrieval_node()` re-runs for uncovered sub-queries with a relaxed reranker threshold. `keyword_search_loop_node()` calls `_exact_search()` directly (no vector legs) for the final fallback.
