# Qdrant BM25 Migration Plan

Replaces the MySQL InnoDB FULLTEXT "exact" leg with Qdrant-native BM25 sparse
vectors. Motivation: at multi-million chunk scale, MySQL FTS requires either
index-bypassing OR'd cross-table MATCHes or large unbounded result transfers;
BM25 in Qdrant runs on the inverted index already colocated with the other
vector legs — one retrieval system instead of two.

## Facts verified (Phase 0)

- Qdrant **1.18.1** self-hosted; `qdrant-client` **1.19.0**.
- Server-side BM25 inference is built into the Qdrant binary (tokenizer +
  hashing, no ONNX, no external inference service). Verified:
  `{"text": "...", "model": "qdrant/bm25"}` works in both upsert and query;
  `nonexistent/model` fails with "InferenceService URL not configured".
- `models.Document(text=..., model="qdrant/bm25")` is accepted by the Python
  client in `PointStruct.vector` and in `query_points(query=...)`.
- **Vector schema update API (v1.18.0+)**: `PUT /collections/{name}/vectors/{vector_name}`
  adds named vectors — including `{"sparse": {"modifier": "idf"}}` — to an
  existing collection. Verified live on `ds_88` (add + delete). No recreation,
  no aliases, no downtime. Qdrant has no collection rename API; aliases are
  the rename mechanism (not needed here).
- Corpus stats (local): 292 chunks, avg ~1167 chars / ~176 words (~230 tokens),
  range 28–2105. Titles avg 36 chars. fastembed's default `avg_len=256` is
  close to real chunk length; re-measure at production scale.
- `idf` modifier: Qdrant maintains collection-level IDF in the sparse index
  and applies it at query time. Document-side vectors carry the saturated TF
  component; IDF stays live as the corpus changes.
- `avg_len` is a fixed hyperparameter (server-side default or fastembed's 256),
  not a live corpus stat. `Document.options` accepts `k`/`b`/`avg_len`
  overrides — verify honored values if defaults diverge at scale.

## Design

### Named vectors per chunk point

| vector        | content        | producer                              |
|---------------|----------------|----------------------------------------|
| `dense`       | title+chunk    | external embedding API (unchanged)     |
| `sparse`      | title+chunk    | SPLADE fastembed (unchanged)           |
| `bm25`        | chunk only     | `qdrant/bm25` server-side inference    |
| `bm25_title`  | —              | not on chunk points (see below)        |

`bm25` uses clean `chunk_text` (not the title-prefixed text used for
dense/sparse) to keep parity with the MySQL chunk branch.

### Titles: pseudo-points (option B)

One extra point per **document**:
`id = uuid5("title:" + str(document_id))`, carrying only a `bm25_title`
vector + payload `{"document_id": N, "_title_point": true}`.

- True per-document IDF (embedding the title into every chunk would inflate
  document frequency N_chunks× and depress title scores for long documents).
- Self-isolating: absent `dense`/`sparse`/`bm25` vectors mean it can never
  match those legs.
- Query path: `bm25_title` hits → document_ids → payload-filtered fetch of
  all their chunks, each scored `TITLE_WEIGHT × title_score` — replicates the
  old MySQL title branch (all chunks of a title-matching doc, weight 2.0).

Rejected alternatives: per-chunk `bm25_title` (IDF distortion), keeping a
MySQL title FTS query (blocks retiring the MySQL leg entirely).

### New leg: `bm25_search_docs` (`retrieval.py`)

Mirrors `_sparse_search`: per collection one `query_batch_points` carrying
(query + synonym variants) × (`bm25` + `bm25_title`) — all `Document` inputs.
Title expansion via payload filter on `document_id`. Combined score per point:
`bm25_score + BM25_TITLE_WEIGHT × title_score` → `_Candidate(exact_rank=…)`.
`doc_ids` payload filter and collection fan-out threading reused.

### Settings

- `KEYWORD_BACKEND` = `mysql` | `bm25` | `shadow` (shadow runs both, logs
  top-k overlap — comparison harness before cutover).
- `BM25_MIN_SCORE` (default 0.0 — calibrate from distributions at scale).
- `BM25_TITLE_WEIGHT` (default 2.0 — preserves MySQL-leg weighting).

### Call points (complete inventory)

Only two consumers of `exact_search_docs`:

1. `keyword_search` tool — all agentic paths flow here (fast_plan rounds,
   v1/v2 agent graphs, retrieval/office subagents via `_run_tool`).
2. `/search` endpoint (`search.py::_run_retrieval_legs`).

Not affected: `title_search` tool (metadata filters on `documents`, no FTS),
`chats.py` MATCH on `m.content` (chat-message search, different table).

## Migration (in-place, no downtime)

Per collection `ds_{id}` / `kb_{id}`:

1. `create_vector_name` for `bm25` + `bm25_title` (`{"sparse":{"modifier":"idf"}}`).
2. Scroll all points (payload only — `chunk_text` and `title` are already in
   payload); `update_vectors` batches add `bm25` via `Document` inputs.
3. Emit title pseudo-points per distinct `document_id`.
4. Verify: point-count parity + `has_vector` count for `bm25` == total;
   pseudo-point count == distinct doc count. In-memory progress registry,
   superadmin UI on `/dashboard/admin/data-sources`.

New ingests get all vectors automatically; updates/deletes flow through the
same code paths. Verified: no standalone title-edit path exists —
`_apply_document_metadata` (datastores/documents.py) only mutates status,
effective dates, version, owner; title changes always come through
re-conversion → re-ingestion, which rebuilds the pseudo-point.

## Implementation status (COMPLETE — MySQL FTS removed)

- `services/ingestion/document_qdrant.py` — `_ensure_bm25_vectors`
  (in-place schema add via `create_vector_name`), `_title_point_id`,
  `bm25` Document on chunk points, title pseudo-point upsert.
- All 7 Qdrant delete sites extended to remove title pseudo-points;
  `reconciliation_service._delete_orphan_points` treats pseudo-points as
  expected; `graph_expand` legacy fallback excludes them.
- `services/retrieval/retrieval.py` — the MySQL FTS leg is REMOVED
  (`exact_search_docs`, `_run_fts_query_with_retry`, `_filter_and_dedup_rows`,
  `_normalize_metadata` deleted). `_lexical_search` fuses SPLADE + BM25 +
  BM25-title into ONE `query_batch_points` per collection with
  failure-decomposition fallback. `lexical_search_docs` is the public API;
  `sparse_search_docs`/`bm25_search_docs` remain as single-leg exports.
- `services/datastore/bm25_migration.py` — in-memory migration job (schema
  add → scroll+update_vectors → pseudo-points → has_vector verify) with
  `/api/admin/datastores/bm25-migration` API and superadmin UI on
  `/dashboard/admin/data-sources`.
- Settings: `BM25_MIN_SCORE`, `BM25_TITLE_WEIGHT`, `RETRIEVAL_BM25_ENABLED`.
  `KEYWORD_BACKEND`, `EXACT_MIN_SCORE`, `RETRIEVAL_EXACT_ENABLED` removed.
- `document_chunks.chunk_text` FULLTEXT index and `documents.title`
  FULLTEXT index dropped — keyword retrieval no longer touches MySQL FTS.
  `document_chunks` rows remain: they are the metadata source of truth for
  neighbor injection, `doc_ids` filtering, `doc_chunks_have_vectors` parity,
  reconciliation, and citations.
- Call sites: `keyword_search` tool (all agentic paths) and `/search`
  endpoint both call `lexical_search_docs`.
- Live-verified on `ds_88`: migration 150 ms, `verified=true`;
  fused lexical call returns correct bm25 + sparse hits in one batch.

Remaining: re-measure `avg_len` at production scale (default 256 is close
to the ~230-token average observed). `kb_*` collections use the same
migration service if needed (UI is datastore-scoped per spec).
