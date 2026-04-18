# Search Implementation

## Overview

Retrieval uses a **hybrid** pipeline: BM25 sparse keyword search combined with ChromaDB dense vector similarity, fused via Reciprocal Rank Fusion (RRF). This covers the failure modes of each method individually:

- Dense search excels at semantic similarity (synonyms, paraphrases) but misses exact keyword matches.
- BM25 excels at exact keyword matches (product codes, names, numbers) but misses paraphrase relationships.
- RRF merges both ranked lists without requiring scores to be on the same scale.

---

## Pipeline

```
User query
    │
    ▼
[chat_service.py] Condense with chat history → standalone question
    │
    ▼
[retrieval.py] hybrid_search()
    ├── _dense_search()   → ChromaDB cosine similarity → Dict[hash, _Candidate]
    ├── _bm25_search()    → BM25Okapi over MySQL chunks → Dict[hash, _Candidate]
    └── _rrf_merge()      → RRF score → top-K LangchainDocuments
    │
    ▼
[chat_service.py] Build prompt → stream LLM response
```

---

## Files

| File | Role |
|------|------|
| `backend/app/services/retrieval.py` | Complete hybrid search implementation |
| `backend/app/services/chat_service.py` | Calls `hybrid_search()`, handles context streaming |
| `backend/app/core/config.py` | All tunable parameters |
| `.env` / `.env.example` | Runtime configuration |

---

## Implementation Detail

### Entry point — `hybrid_search()`

`backend/app/services/retrieval.py`

```python
async def hybrid_search(query, kb_ids, db, vector_stores) -> List[LangchainDocument]:
    top_k = settings.RETRIEVAL_TOP_K
    pool = top_k * 4   # each leg fetches 4× top_k so RRF has room to rerank

    dense  = _dense_search(query, vector_stores, pool)
    sparse = _bm25_search(query, kb_ids, db, pool)
    return _rrf_merge(dense, sparse, top_k)
```

`pool = top_k * 4` ensures that a document ranked #20 by one leg but #1 by the other is not prematurely discarded before the merge.

---

### Dense leg — `_dense_search()`

Queries every knowledge base's ChromaDB collection using cosine distance. ChromaDB returns `(document, distance)` pairs where distance is on the range `[0, 2]` (0 = identical vectors, 2 = opposite).

```python
def _dense_search(query, vector_stores, candidates):
    all_pairs = []
    for vs in vector_stores:
        all_pairs.extend(vs.similarity_search_with_score(query, k=candidates))

    all_pairs.sort(key=lambda pair: pair[1])   # ascending: best match first

    max_distance = settings.HYBRID_MAX_DENSE_DISTANCE
    result = {}
    rank = 0
    for doc, distance in all_pairs:
        if distance > max_distance:
            break   # sorted list — all remaining are farther away
        h = _content_hash(doc.page_content)
        if h not in result:
            result[h] = _Candidate(doc=doc, content_hash=h, dense_rank=rank)
            rank += 1
    return result
```

`HYBRID_MAX_DENSE_DISTANCE` acts as a hard cutoff. Documents beyond it carry no meaningful semantic signal and are excluded before the merge. The default `2.0` keeps everything; set it lower (e.g. `1.2`) to be selective.

---

### Sparse leg — `_bm25_search()`

Loads all `DocumentChunk` rows for the requested knowledge bases from MySQL, builds a `BM25Okapi` index in memory, and scores every chunk against the query.

```python
def _bm25_search(query, kb_ids, db, candidates):
    chunks = db.query(DocumentChunk).filter(DocumentChunk.kb_id.in_(kb_ids)).all()
    texts  = [(chunk.chunk_metadata or {}).get("page_content", "") for chunk in chunks]

    bm25   = BM25Okapi([t.lower().split() for t in texts])
    scores = bm25.get_scores(query.lower().split())

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    result = {}
    rank = 0
    for idx in ranked_indices:
        if scores[idx] == 0:
            break   # no query-term overlap — stop here
        ...
        result[h] = _Candidate(..., sparse_rank=rank)
        rank += 1
```

**Why stop at score == 0?**
BM25 = 0 is an absolute zero — the document shares no tokens with the query at all. Assigning it a rank position (e.g. rank 500) would inject a tiny but non-zero sparse score into the RRF merge that the document does not deserve. The `break` keeps the sparse leg semantically meaningful: every document that receives a sparse rank matched at least one query term.

This is **not** the same as excluding the document from results — it only means it contributes 0 from the sparse leg. It can still be returned if the dense leg ranks it highly (see the four cases below).

---

### RRF merge — `_rrf_merge()`

```python
def _rrf_merge(dense, sparse, top_k):
    merged = {**dense}
    for h, c in sparse.items():
        if h in merged:
            merged[h].sparse_rank = c.sparse_rank   # already in dense — add sparse rank
        else:
            merged[h] = c                            # sparse-only — dense_rank stays -1

    ranked = sorted(merged.values(), key=lambda c: c.rrf_score, reverse=True)
    return [c.doc for c in ranked[:top_k]]
```

#### RRF score formula

```
score(doc) = HYBRID_DENSE_WEIGHT  / (60 + dense_rank)
           + HYBRID_SPARSE_WEIGHT / (60 + sparse_rank)
```

An absent leg (`rank == -1`) contributes 0. The constant 60 is the standard smoothing value from the original RRF paper (Cormack et al., 2009) — it prevents the top-ranked document from dominating disproportionately.

#### The four cases

| BM25 score | Dense distance | Outcome | Rationale |
|------------|---------------|---------|-----------|
| > 0 | ≤ threshold | Both legs contribute | Strong match — full RRF score |
| = 0 | ≤ threshold | Dense leg only | Paraphrase / synonym match; dense signal is real and sufficient |
| > 0 | > threshold | Sparse leg only | Exact keyword match with no semantic overlap (e.g. codes, model numbers) |
| = 0 | > threshold | **Excluded** | No signal from either leg — not a relevant document |

---

### Deduplication — `_content_hash()`

Multiple knowledge bases may contain the same chunk (e.g. shared onboarding document). SHA-256 of the chunk text is used as the merge key so duplicates are collapsed to a single candidate before scoring.

```python
def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
```

---

### Scoring dataclass — `_Candidate`

```python
@dataclass
class _Candidate:
    doc: LangchainDocument
    content_hash: str
    dense_rank: int = -1   # -1 = absent from this leg
    sparse_rank: int = -1

    @property
    def rrf_score(self) -> float:
        score = 0.0
        if self.dense_rank >= 0:
            score += settings.HYBRID_DENSE_WEIGHT / (_RRF_K + self.dense_rank)
        if self.sparse_rank >= 0:
            score += settings.HYBRID_SPARSE_WEIGHT / (_RRF_K + self.sparse_rank)
        return score
```

`-1` sentinel is intentional — it cleanly separates "not ranked" from "ranked last". A rank of 0 is the best possible position.

---

## Configuration

All parameters live in `backend/app/core/config.py` and are set via `.env`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RETRIEVAL_TOP_K` | `6` | Number of documents returned to the LLM |
| `HYBRID_DENSE_WEIGHT` | `0.7` | Weight of the dense (embedding) leg in RRF |
| `HYBRID_SPARSE_WEIGHT` | `0.3` | Weight of the BM25 leg in RRF |
| `HYBRID_MAX_DENSE_DISTANCE` | `2.0` | Cosine distance cutoff for the dense leg (0–2); lower = more selective |

Weights do not need to sum to 1 — RRF is scale-agnostic. Raising `HYBRID_SPARSE_WEIGHT` benefits corpora with precise terminology (legal, medical, technical). Raising `HYBRID_DENSE_WEIGHT` benefits conversational or paraphrase-heavy queries.

---

## Where retrieval is called

`backend/app/services/chat_service.py`, Step 2 of `generate_response()`:

```python
# Step 2: Retrieve relevant documents via hybrid search (dense + BM25 + RRF)
docs = await hybrid_search(
    query=standalone_question,
    kb_ids=knowledge_base_ids,
    db=db,
    vector_stores=vector_stores,
)
```

The query passed in is the **condensed standalone question** — chat history context has already been folded in by Step 1 (the condense chain). This ensures the retrieval query is self-contained and does not rely on pronouns or references that BM25 would fail to resolve.

---

## Performance notes

- BM25 index is built in memory on every request from `DocumentChunk` rows. This is acceptable up to ~50 k short chunks (typically < 200 ms). For larger corpora a pre-built persistent index (e.g. Elasticsearch, Typesense) would be more appropriate.
- ChromaDB queries run concurrently per collection via `similarity_search_with_score`; results are merged in Python.
- The candidate pool (`top_k * 4`) adds overhead but is necessary for RRF correctness — without headroom, a document that is the best BM25 match but not in the top-K dense results would be invisible to the merge.
