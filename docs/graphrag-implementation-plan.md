# GraphRAG Implementation

## Architecture

GraphRAG uses Neo4j exclusively for entity/relationship storage and post-retrieval
context enrichment. It does NOT perform vector search — Qdrant owns all vector
search. This matches the Qdrant+Neo4j reference architecture described at
https://qdrant.tech/documentation/examples/graphrag-qdrant-neo4j/

```
Ingestion:
  document text
      │
      ├── Qdrant  ← dense + sparse vectors (existing 3-leg pipeline)
      ├── MySQL   ← chunk text for FTS (existing)
      └── Neo4j   ← Chunk nodes (keyed by qdrant_point_id UUID)
                     + Entity nodes + relationships (via LLMEntityRelationExtractor
                       or ReLiK, operating on context-batched, overlap-stripped text)

Query time:
  user query
      │
      ▼
  3-leg RRF (Qdrant dense + sparse + MySQL FTS)
      │
      ▼  (qdrant_point_id from Qdrant payload)
  Neo4j graph expansion — expand_docs_via_graph()
      │ MATCH Chunk by qdrant_point_id
      │ → FROM_CHUNK → Entity → relationships → neighbor Entity → FROM_CHUNK → Chunk
      │ → fetch expanded chunks from Qdrant, merge into candidate pool
      ▼
  Neo4j context enrichment — enrich_docs_with_graph()
      │ MATCH Chunk by qdrant_point_id
      │ → FROM_CHUNK → Entity → relationships → append [Graph context] triples
      ▼
  cross-encoder reranker
      │
      ▼
  LLM response generation
```

The cross-reference between Qdrant and Neo4j is the `qdrant_point_id` (a deterministic
UUID derived from the SHA-256 chunk ID via uuid5). Every Qdrant point payload contains
this field; every Neo4j Chunk node is written with it at ingest time.

---

## Configuration

```env
# Enable Neo4j graph building at ingest and enrichment at query time
GRAPHRAG_ENABLED=true

# Neo4j connection
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# LLM for entity/relationship extraction (uses OPENAI_API_BASE endpoint)
# When unset, falls back to the ReLiK dockerized service
GRAPHRAG_LLM=qwen/qwen3.5-4b

# Context window budget for the graph extraction LLM (characters).
# Consecutive chunks are merged and fed as a single batch up to this limit
# (80% used for text; 20% headroom for system prompt + JSON schema output).
# Rule of thumb: 1 token ≈ 3–4 chars.
#   2K token model  → NEO4J_LLM_CONTEXT=6000   (default)
#   4K token model  → NEO4J_LLM_CONTEXT=12000
#   8K token model  → NEO4J_LLM_CONTEXT=24000
NEO4J_LLM_CONTEXT=6000

# Maximum chunks per document to run graph extraction on.
# Chunks beyond this are still indexed in Qdrant/MySQL. Set to 0 for no limit.
GRAPHRAG_MAX_CHUNKS=300

# Hop limit for entity traversal at query time
GRAPHRAG_RETRIEVAL_HOPS=2

# Per-retrieval enable/disable toggle (separate from GRAPHRAG_ENABLED)
RETRIEVAL_GRAPH_ENABLED=true
```

---

## Key files

| File | Role |
|------|------|
| backend/app/services/graph_service.py | All Neo4j logic: ingest, enrich, expand, delete |
| backend/app/services/retrieval.py | Calls expand_docs_via_graph() + enrich_docs_with_graph() after RRF merge |
| backend/app/services/document_processor.py | Calls build_graph_for_document() at step 9 |
| backend/app/core/config.py | All GRAPHRAG_* and NEO4J_* settings |

---

## Ingestion — build_graph_for_document()

Called from `document_processor.py` step 9, after the chunk has been indexed in
Qdrant and MySQL. Non-fatal — a Neo4j failure never blocks the core 3-leg pipeline.

### Step 1: Write Chunk nodes
```cypher
MERGE (c:Chunk {qdrant_point_id: $point_id})
SET c.qdrant_collection = $collection,
    c.document_id = $document_id,
    c.chunk_index = $chunk_index,
    c.kb_id = $kb_id,
    c.file_name = $file_name
```
One node per Qdrant point. `qdrant_point_id` is the primary cross-reference key.

### Step 2: Build extraction batches
`_build_extraction_batches()` groups consecutive chunks into batches:

- Budget = `NEO4J_LLM_CONTEXT × 0.8` chars (80% — headroom for prompt + schema)
- Within each batch, chunk overlap is stripped: the longest suffix of chunk[i] that
  matches a prefix of chunk[i+1] is removed before concatenation
- Result: clean, non-redundant prose that fits the LLM context window

Example with 1500-char chunks and 6000-char budget (~4800 usable):
```
Chunk 0 (1500 chars) + deduped Chunk 1 (~1350 chars) + deduped Chunk 2 (~1350 chars)
= ~4200 chars → fits in batch 0

Chunk 3 starts batch 1, etc.
```

### Step 3: LLM extraction (up to 4 batches concurrent)
Each batch is sent to `LLMEntityRelationExtractor` as a single `TextChunk`.
The extractor writes `__Entity__` nodes and typed relationship edges to Neo4j.

### Step 4: FROM_CHUNK fan-out
After each batch's extraction, `FROM_CHUNK` edges are written from every extracted
entity to **all** Qdrant chunk nodes that contributed to that batch. Entity→chunk
links remain granular even though extraction ran on merged text.

```
Batch 0 text = chunks [C0, C1, C2] merged
Entities extracted: "Apple", "Beats"

FROM_CHUNK edges written:
  Apple → C0, Apple → C1, Apple → C2
  Beats → C0, Beats → C1, Beats → C2
```

### Concurrency
- Document-level: `_graph_semaphore = Semaphore(1)` — one document extracts at a time
- Batch-level: `_batch_sem = Semaphore(4)` — up to 4 batches of one document run concurrently

---

## Retrieval — expand_docs_via_graph()

Called from `retrieval.py` after RRF merge, before reranking.

1. Extract `qdrant_point_id` from each retrieved doc's metadata.
2. Traverse Neo4j: `chunk → entity → entity → chunk` to find entity-connected chunks
   NOT already in the result set.
3. Fetch those chunks from Qdrant by UUID (text + payload, no re-embedding).
4. Return as additional `LangchainDocument` objects with `_graph_expanded=True`.

---

## Retrieval — enrich_docs_with_graph()

Called after expansion + reranking, on the final top-k docs only.

For each doc:
1. Look up Neo4j Chunk by `qdrant_point_id`.
2. Traverse `FROM_CHUNK` edges to Entity nodes, collect entity-to-entity relationship
   triples (up to 40).
3. Append `[Graph context]\nA -[R]-> B\n...` to the chunk text.

---

## Deletion

### delete_graph_for_document(kb_id, document_id)
- Deletes all `Chunk` nodes for the document (`DETACH DELETE`)
- Cleans up orphaned `__Entity__` nodes (no remaining `FROM_CHUNK` connections)
- NOT gated on `GRAPHRAG_ENABLED` — data may exist from a prior run

### delete_graph_for_kb(kb_id)
- Deletes all nodes with `kb_id` property (`DETACH DELETE`)
- Cleans up fully isolated `__Entity__` nodes
- NOT gated on `GRAPHRAG_ENABLED`

Both functions check `NEO4J_URI` and return early if unset.

---

## Design decisions

**Why batch chunks instead of per-chunk extraction?**
Per-chunk extraction (the original approach) meant the LLM saw only ~1500 chars
at a time. Entities at chunk boundaries were often extracted as duplicates with
slightly different names; relationships spanning two chunks were invisible.
Batching to `NEO4J_LLM_CONTEXT` gives the LLM broader context, reduces duplicates,
and surfaces intra-document relationships at no extra LLM call overhead (fewer,
larger calls vs many small ones).

**Why strip overlap before batching?**
LangChain's `RecursiveCharacterTextSplitter` produces chunks with configurable
overlap (default ~150 chars). Without stripping, the same sentence appears twice
in the batch text, wasting context budget and potentially confusing the extractor.
`_strip_overlap()` finds the longest suffix of chunk[i] matching a prefix of
chunk[i+1] and removes it — O(overlap²) but bounded by `chunk_overlap * 2`.

**Why qdrant_point_id as the cross-reference key?**
The original implementation used `(document_id, chunk_index)`. Migrated to
`qdrant_point_id` (a deterministic uuid5 derived from the SHA-256 chunk hash)
because: it's a single indexed field instead of a compound key; it's the same
UUID Qdrant uses as the point ID so no separate mapping table is needed; and it's
stable across re-ingests as long as the chunk content doesn't change.

**Why no vectors in Neo4j?**
Neo4j's `setNodeVectorProperty` was called by the pipeline internals but served
no purpose — Qdrant owns all vector search. Keeping vectors out of Neo4j eliminates
the `List{}` error from empty embeddings and avoids duplicating the entire embedding
store in Neo4j.

**Why not store vectors on Neo4j for cross-document entity similarity?**
Cross-document entity merging via embedding similarity (e.g. "Apple" vs "Apple Inc.")
is a valid improvement but out of scope for the current pipeline. The `MERGE` on
entity name handles exact-match deduplication. Fuzzy merging would require a
separate post-ingest pass with a vector index on entity names.
