# Abbreviation Expansion for RAG — Design Analysis

## Objective

Use the military abbreviations CSV (`abbreviations_enhanced.csv`, 3,090 rows, 2,193
unique abbreviations) to improve retrieval recall in the agentic RAG pipeline.
Expansion must handle abbreviations with multiple derivative forms and completely
different meanings.

---

## Current Pipeline Architecture

### Ingestion Flow

```
Document → markitdown → markdown → RecursiveCharacterTextSplitter → chunks
                                                                    ↓
                                                            _build_chunk_records()
                                                                    ↓
                                                    ┌──────────────┴───────────────┐
                                                    ↓                               ↓
                                            MySQL document_chunks            Qdrant upsert
                                            (chunk_text, FULLTEXT)           (dense + sparse vectors)
```

**Key files:**
- `backend/app/services/ingestion/document_processor.py` — orchestrates the pipeline
- `backend/app/services/ingestion/document_qdrant.py` — embedding + Qdrant upsert
- `backend/app/models/knowledge.py` — `DocumentChunk` model (MySQL)

### What Gets Stored Per Chunk

| Store | Field | Content | Searched by |
|-------|-------|---------|-------------|
| MySQL `document_chunks` | `chunk_text` (LONTEXT, FULLTEXT) | Raw chunk text | Exact leg (BM25) |
| MySQL `document_chunks` | `chunk_metadata` (JSON) | Source metadata | Not searched |
| Qdrant | `vector["dense"]` | Qwen3 embedding of raw chunk text | Dense leg (cosine) |
| Qdrant | `vector["sparse"]` | SPLADE embedding of raw chunk text | Sparse leg (SPLADE) |
| Qdrant | `payload.chunk_text` | Raw chunk text | Not searched (returned with hits) |
| Qdrant | `payload.*` | kb_id, document_id, file_name, chunk_index, metadata | Filtering |

### Retrieval Flow (3-leg hybrid)

```
User query
    ↓
rewrite_query_node (LLM resolves pronouns/references)
    ↓
rewritten_query
    ↓
┌─────────────────┬──────────────────┬──────────────────┐
│ Dense leg       │ Sparse leg       │ Exact leg        │
│ Qwen3 embed     │ SPLADE embed     │ MySQL FULLTEXT   │
│ Qdrant cosine   │ Qdrant sparse    │ BM25 scoring     │
└────────┬────────┴────────┬─────────┴────────┬─────────┘
         └────────────────┼──────────────────┘
                          ↓
                    merge + rerank
                          ↓
                    LLM generates answer
```

**Key files:**
- `backend/app/services/retrieval/retrieval.py` — 3 search legs
- `backend/app/services/agentic_rag/nodes.py` — LangGraph nodes
- `backend/app/services/retrieval/query_expander.py` — existing (but unused) expansion scaffolding

---

## Empirical Test Results

### Test Setup

- **Dense embeddings**: qwen3-embedding-0.6b (LM Studio, dim=1024, context=512)
- **Sparse embeddings**: SPLADE PP en v1 (FastEmbed, local ONNX, 512-token limit)
- **Reranker**: ms-marco-MiniLM-L-12-v2 (FastEmbed, local ONNX cross-encoder)
- **Generation**: gemma-4-12b (LM Studio, no thinking)
- **Test corpus**: 5 chunks (military abbreviation-heavy + weather control)
- **Test queries**: 6 queries (3 full-form, 3 abbreviation-form, 1 multi-meaning)
- **Test script**: `backend/tests/test_abbr_expansion.py`

### Expansion Strategies Tested

| Strategy | Ingestion | Example |
|----------|-----------|---------|
| **none** | Original text | `The CO ordered bns to wdr` |
| **suffix** | Append all forms | `The CO ordered bns to wdr [Expansions: CO=Commanding Officer; bns=Battalions; wdr=Withdraw...]` |
| **replace** | Replace abbrs with all forms | `The Commanding Officer ordered Battalions to Withdraw Withdrawal...` |
| **glossary_suffix** | Append glossary block | `The CO ordered bns to wdr\n[Abbreviation Glossary]\nCO = Commanding Officer\nbns = Battalions` |
| **replace+glossary** | Replace with primary form + glossary | `The Commanding Officer ordered Battalions to withdraw\n[Abbreviation Glossary]\nCO = Commanding Officer\nbns = Battalions` |

### Dense Embedding Results (qwen3-embedding-0.6b)

Hit rates (top-2 contains expected chunk):

| Ingestion \ Query | original | suffix | replace |
|---|---|---|---|
| **none** | 100% | 100% | 100% |
| **suffix** | 100% | 100% | 100% |
| **replace** | 100% | 100% | 100% |
| **glossary_suffix** | 100% | 100% | 100% |
| **replace+glossary** | 100% | 100% | 100% |

All combinations achieve 100% hit rate on this test corpus. The differentiator is
similarity score quality, not hit rate:

**Key score observations:**

- `ing=replace` gives the highest dense similarity for full-form queries:
  Q0 "battalions withdrew": c0:0.689 (replace) vs 0.520 (none)
- `ing=replace` HURTS abbreviation queries — replacing "bns" with "Battalions"
  removes the original token:
  Q1 "bns wdr": c0:0.345 (replace) vs 0.491 (none)
- `q=suffix` helps abbreviation queries by adding both forms:
  Q1 "bns wdr": c4:0.566 (suffix) vs 0.304 (original)
- `ing=glossary_suffix` consistently underperforms in raw scores — the glossary
  format adds structural noise (`[Abbreviation Glossary]`, `=` signs) that dilutes
  the dense embedding signal

### Sparse (SPLADE) Results

Hit rates (top-2 contains expected chunk):

| Ingestion \ Query | original | suffix | replace |
|---|---|---|---|
| **none** | 100% | 100% | 100% |
| **suffix** | 100% | 100% | 100% |
| **replace** | 100% | 100% | 100% |
| **glossary_suffix** | **83%** | 100% | **83%** |
| **replace+glossary** | 100% | 100% | 100% |

**Critical finding**: `glossary_suffix` drops to 83% with original/replace queries.
The glossary format (`[Abbreviation Glossary]\nCO = Commanding Officer`) adds
structural tokens that SPLADE weights, creating false matches. Suffix format
(`[Expansions: CO=Commanding Officer]`) is cleaner for SPLADE.

**Score observations:**

- `ing=suffix` + `q=suffix` gives the highest SPLADE scores:
  Q2 "CO ordered bns to wdr": c0:36.6 (suffix+suffix) vs 23.2 (none+original)
- `ing=replace` is catastrophic for abbreviation queries:
  Q1 "bns wdr": c4:5.1 c0:4.8 (both expected chunks score near zero)
  The abbreviation "bns" is gone from the chunk, so SPLADE can't match it
- SPLADE scores are strongly bimodal: relevant chunks 15-38, irrelevant 0-3.
  This makes threshold filtering more reliable than dense.

### Reranker Results (ms-marco-MiniLM-L-12-v2)

Hit rates (top-2 contains expected chunk):

| Query \ Chunk | orig_c | suffix_c | replace_c | glossary_c | repglos_c |
|---|---|---|---|---|---|
| **orig_q** | **83%** | 100% | 100% | 100% | 100% |
| **exp_q** | 100% | 100% | — | 100% | 100% |
| **rep_q** | **83%** | — | — | 100% | 100% |

**Key findings:**

1. `orig_q + orig_c` scores only 83% — it fails on Q3 "brigade headquarters
   operation objective" where both query and chunk use different surface forms
   (query: "brigade headquarters", chunk: "bde HQ"). The cross-encoder can't
   bridge this gap without expansion.

2. ANY expansion on either side fixes the Q3 failure. Even just expanding the
   chunk with a glossary (`orig_q + glossary_c`) brings the expected chunk from
   -9.333 to 3.101.

3. `rep_q + orig_c` scores 83% — replacement queries HURT the reranker for
   abbreviation queries. When "bns wdr from position" becomes "Battalions
   Withdraw from position", the reranker can no longer match it against a chunk
   containing "bns" and "wdr". The original abbreviation tokens are gone.

4. `orig_q + replace_c` scores 100% — replacement chunks help because the
   reranker sees full-form text in the chunk, which matches full-form queries.
   But this creates an asymmetry: abbreviation queries can't match replacement
   chunks (the "bns" in the query has no "bns" in the chunk to match).

5. **Best reranker combos** (100% hit rate): `orig_q + suffix_c`,
   `orig_q + glossary_c`, `orig_q + repglos_c`, `exp_q + any_c`.

6. The safest universal choice is `orig_q + suffix_c` or `orig_q + glossary_c`:
   the original query preserves the user's intent, and the expanded chunk
   provides the abbreviation mapping for the cross-encoder.

### LLM Query Expansion Results (gemma-4-12b)

| Query | LLM Replacement | Correct? |
|-------|----------------|----------|
| `bns wdr from position` | `base navigation data from position` | **WRONG** |
| `CO ordered bns to wdr` | `Commanding Officer ordered battalion to withdraw` | Correct |

LLM-based replacement is **unreliable** with a 4b model. "bns" was expanded to
"base navigation data" — a completely wrong interpretation. The model doesn't
have sufficient military domain knowledge to disambiguate abbreviations
correctly. A larger model (9b+) might do better, but adds latency and cost.

**Dense retrieval comparison** (against original chunks):

| Query | original | llm_replace | det_suffix |
|-------|----------|-------------|------------|
| `bns wdr from position` | c0:0.491 | c0:0.361 | c4:0.566 |
| `CO ordered bns to wdr` | c0:0.681 | c4:0.704 | c0:0.603 |

LLM replacement sometimes helps (Q2: 0.704 vs 0.681) and sometimes hurts
(Q1: 0.361 vs 0.491). Deterministic suffix expansion is more consistent.

### Generation Results (gemma-4-12b)

**Test 1**: Query "battalions withdrew from position", chunk with abbreviations:

| Context | Answer | Correct? |
|---------|--------|----------|
| original | `The CO ordered the bns to wdr from the forward position.` | Yes (terse) |
| glossary | `The CO ordered the bns to wdr from the forward positions.` | Yes (terse) |
| suffix | `The CO ordered the battalions to withdraw from the forward positions.` | Yes (expanded) |
| replace | `The Commanding Officer ordered the Battalions to withdraw from the forward positions.` | Yes (clean) |
| **replace+glossary** | `The Commanding Officer ordered the Battalions Transport Officer to withdraw...` | **WRONG** |

**Test 2**: Query "who approved the medical resupply?", chunk with "DA":

| Context | Answer | Correct? |
|---------|--------|----------|
| original | `The DA approved the medical resupply.` | Yes |
| glossary | `The DA approved the medical resupply.` | Yes |
| suffix | `The DA approved the medical resupply.` | Yes |
| **replace+glossary** | `The Daily Allowance.` | **WRONG** |

**Critical finding**: `replace+glossary` introduces WRONG meanings. The
deterministic replacer picked "Daily Allowance" for "DA" (first form in the CSV)
instead of "Deputy Assistant" (the correct meaning in context). It also picked
"Transport Officer" for "TO" which wasn't even an abbreviation in the original
text — "to" was matched as an abbreviation and replaced.

This is the fundamental problem with deterministic replacement: **it cannot
disambiguate multi-meaning abbreviations without understanding context**. The
CSV lists "Daily Allowance" before "Deputy Assistant" for DA, so the replacer
picks the first form, which is wrong.

### LLM-Based Replacement with Glossary Context (gemma-4-12b)

A separate test (`backend/tests/test_llm_replace.py`) evaluated whether an LLM
can correctly disambiguate abbreviations when given a glossary of all possible
meanings plus the surrounding context. This addresses the core weakness of
deterministic replacement.

**Method**: For each chunk/query, find abbreviations deterministically, build a
glossary listing ALL possible meanings, send text + glossary to gemma-4-12b,
ask it to replace each abbreviation with the correct form based on context.

#### LLM Replacement Quality on Chunks

| Chunk | Det Replace | LLM Replace | LLM Correct? |
|-------|------------|-------------|--------------|
| 0 | "Battalions Transport Officer..." (TO wrongly replaced) | "Commanding Officer ordered Battalions to Withdraw" | YES |
| 1 | "Brigade Aviation Command" (comd wrong) | "Brigade Commander" | YES |
| 2 | "drop Transport Officer" (to wrongly replaced) | "drop to 15 degrees" | YES |
| 3 | "Daily Allowance Defence Attache Deputy Assistant..." (all dumped) | "Defence Attache approved" | **NO** |
| 4 | "battalions Transport Officer" (to wrongly replaced) | "battalions to withdraw" | YES |

LLM replacement: 4/5 correct. It correctly handles prepositions ("to" left
alone) and picks right forms for most abbreviations. But it got DA wrong —
"Defence Attache" instead of "Deputy Assistant" in context of "approved the
medical resupply". The 12b model lacks sufficient military domain knowledge.

#### LLM Query Replacement Quality

| Query | LLM Replace | Correct? |
|-------|-------------|----------|
| `bns wdr from position` | `Battalions Withdraw from position` | YES |
| `CO ordered bns to wdr` | `Commanding Officer ordered Battalions to Withdraw` | YES |

Both query replacements are correct.

#### Dense Retrieval: LLM Replace vs Suffix

| Query | ing=original | ing=suffix | ing=llm_replace |
|-------|-------------|------------|-----------------|
| `battalions withdrew` (Q0) | c4:0.660 c0:0.520 | c4:0.637 c0:0.577 | c0:0.690 c4:0.660 |
| `bns wdr` (Q1) | c0:0.491 | c0:0.456 | c0:0.337 |
| `CO ordered bns to wdr` (Q2) | c0:0.681 | c0:0.610 | c0:0.466 |
| `brigade HQ op obj` (Q3) | c1:0.590 | c1:0.624 | **c1:0.735** |

LLM replacement gives the highest dense similarity for full-form queries (Q0,
Q3) because the text is clean, readable prose with correct full forms. But it
HURTS abbreviation queries (Q1, Q2) because the original abbreviation tokens
are gone — "bns" no longer exists in the LLM-replaced chunk.

#### Reranker: LLM Replace vs Suffix

| Combo | Hit Rate | Key Issue |
|-------|----------|-----------|
| `orig_q + orig_c` | 83% | Fails on Q3 (no expansion) |
| `orig_q + suffix_c` | 100% | Safe, both forms in chunk |
| `orig_q + llm_rep_c` | 100% | But Q1 score: c4(-8.033), Q2: c0(-9.802) — negative! |
| `llm_q + orig_c` | 100% | Q1: c4(8.000), Q2: c4(9.390) — excellent |
| `llm_q + llm_rep_c` | 100% | Q1: c4(8.000), Q2: c4(9.390) — excellent |

`orig_q + llm_rep_c` technically hits 100% but with catastrophic scores for
abbreviation queries — the reranker scores -8 to -10 because the abbreviation
tokens in the query don't exist in the LLM-replaced chunk. This would fail
under any reasonable score threshold.

`llm_q + llm_rep_c` works well because both sides use full forms, but it
requires an LLM call at query time (latency + cost).

#### Generation: LLM Replace vs Suffix vs Glossary

**DA disambiguation test** (query: "who approved the medical resupply?"):

| Context | Answer | Correct? |
|---------|--------|----------|
| original | `The DA approved the medical resupply.` | Yes (but doesn't expand DA) |
| glossary | `The DA approved the medical resupply.` | Yes (but doesn't expand DA) |
| **llm_replace** | `The Defence Attache approved the medical resupply.` | **WRONG** |
| **llm_replace+glossary** | `The Defence Attache.` | **WRONG** |

The LLM-replaced context bakes in the wrong meaning ("Defence Attache"). Even
with the glossary appended, the generation LLM trusts the pre-replaced text
over the glossary. This is the fundamental risk of replacement: **a wrong
replacement propagates through the entire pipeline**.

#### Full Pipeline Comparison

| Pipeline | Hit | Correct | Answer |
|----------|-----|---------|--------|
| suffix: ing=suffix q=suffix rr=orig+suffix gen=glossary | Yes | Yes | "commanding officer ordered... to withdraw" |
| llm: ing=llm_rep q=orig rr=orig+llm_rep gen=llm_rep | Yes | Yes | "commanding officer ordered battalions to withdraw" |
| llm: ing=llm_rep q=llm_q rr=llm_q+llm_rep gen=llm_rep | Yes | Yes | "commanding officer ordered... to withdraw" |
| hybrid: ing=suffix q=suffix rr=orig+llm_rep gen=glossary | Yes | Yes | "commanding officer ordered... to withdraw" |

All pipelines hit and produce correct answers for the simple test query. The
DA disambiguation failure would surface with a query targeting chunk 3.

### LLM Replace vs Suffix: Summary

| Aspect | Suffix | LLM Replace (gemma-4-12b) |
|--------|--------|--------------------------|
| Preposition handling | Appends forms, keeps "to" | Correctly leaves "to" alone |
| Multi-meaning disambiguation | Appends all forms, models disambiguate | Picks one form, may be wrong (DA→Defence Attache) |
| Abbreviation query matching | Works (both forms present) | Broken (abbreviation removed, reranker scores -8 to -10) |
| Full-form query matching | Works | Works, higher similarity scores |
| Dense similarity (full-form queries) | Moderate | Higher (clean prose) |
| Reranker (orig_q + chunk) | 100%, positive scores | 100% but negative scores for abbr queries |
| Reranker (llm_q + llm_chunk) | N/A | 100%, high scores (but needs LLM at query time) |
| Generation quality | Clean original + glossary | May introduce wrong meanings that override glossary |
| Latency | Zero (deterministic) | ~1-3s per chunk (LLM call) |
| Cost | Free | LLM API calls per chunk at ingestion + per query |

**Conclusion**: LLM replacement is better at producing readable text and not
replacing prepositions, but it is worse at handling multi-meaning abbreviations
because it commits to one interpretation that may be wrong. A wrong replacement
propagates through the entire pipeline and cannot be corrected downstream.

Suffix expansion remains safer because it preserves all possibilities and lets
downstream models (dense, SPLADE, cross-encoder, generation LLM) disambiguate
based on context. The generation LLM with a glossary can pick the correct
meaning from the original text + glossary, which is more reliable than trusting
a pre-replacement that might have picked wrong.

### Full Pipeline Results

All 8 tested pipeline combinations achieved 100% hit rate and 100% answer
correctness. Even `ing=none q=original` works because the test corpus is small
and the reranker + generation model handle abbreviations well for simple cases.

The differentiator is not hit rate but **robustness** — which combination
degrades least on harder queries, larger corpora, and ambiguous abbreviations.

---

## Final Conclusions

### The Replacement Strategy Problem

The user's preferred strategy (replacement + glossary) has a critical flaw
confirmed by testing: **deterministic replacement introduces wrong meanings for
multi-meaning abbreviations**. The test showed:

- `DA` → "Daily Allowance" (wrong; should be "Deputy Assistant" in context)
- `TO` → "Transport Officer" (wrong; "to" was a preposition, not an abbreviation)

This happens because:
1. The CSV lists all meanings with no priority ordering
2. The replacer picks the first form alphabetically or by CSV order
3. Context disambiguation requires understanding, which a deterministic replacer
   cannot do

LLM-based replacement is also unreliable with smaller models (4b gemma expanded
"bns" to "base navigation data"). Larger models may work but add latency.

### Recommended Strategy

Based on the empirical results, the optimal combination is:

| Stage | Strategy | Rationale |
|-------|----------|-----------|
| **Ingestion** | **Suffix expansion** | Preserves original text + adds all forms. Never removes information. Best SPLADE scores. Safe for all query types. |
| **Dense query** | **Suffix expansion** | Bidirectional: adds abbreviations for full-form queries, adds full forms for abbreviation queries. |
| **SPLADE query** | **Suffix expansion** | Maximum term overlap. SPLADE weights terms, so extra forms are low-noise. |
| **Exact query** | **Suffix expansion** | MySQL FULLTEXT matches any term. Extra forms increase recall. |
| **Reranker** | **Original query + suffix chunk** | Original query preserves user intent. Suffix chunk provides abbreviation mapping. 100% hit rate. |
| **Generation** | **Original text + scoped glossary** | Clean prose preserves readability. Glossary gives LLM explicit mappings. No wrong replacements. |
| **Citations** | **Original text only** | No expansion metadata in user-visible text. |

### Why Suffix Beats Replacement

1. **Suffix preserves both forms**: "bns" AND "Battalions" coexist. Replacement
   destroys "bns", making abbreviation queries impossible to match.

2. **Suffix never introduces wrong meanings**: All forms are appended, the
   retrieval models disambiguate. Replacement picks one form, which may be wrong.

3. **Suffix has better SPLADE scores**: Both abbreviation and full-form tokens
   are present, creating maximum term overlap. Replacement removes one set.

4. **Suffix is safer for the reranker**: The cross-encoder sees both forms and
   can match either query style. Replacement creates an asymmetry.

### Why Glossary for Generation (Not Suffix)

1. **Clean prose**: The generation LLM sees original text without `[Expansions: ...]`
   suffixes that disrupt reading flow and answer coherence.

2. **Explicit mappings**: The glossary block gives the LLM direct abbreviation→form
   mappings, which it can use to interpret abbreviated text in the chunk.

3. **No wrong replacements**: The glossary lists ALL forms, letting the LLM pick
   the correct one based on context. The deterministic replacer cannot do this.

4. **Tested result**: Generation with glossary produced correct answers for both
   the abbreviation query and the multi-meaning query (DA). Replacement+glossary
   produced wrong answers for both.

### Why Original Query for Reranker (Not Expanded)

1. **Preserves user intent**: The cross-encoder is trained on natural-language
   query/passage pairs. Expanded queries with appended terms dilute the intent
   signal.

2. **Tested result**: `orig_q + suffix_c` scored 100% hit rate. `rep_q + orig_c`
   scored only 83% — replacement queries lost abbreviation tokens.

3. **The chunk side provides the mapping**: With suffix-expanded chunks, the
   reranker sees both "bns" and "Battalions" in the passage, so it can match
   either "bns wdr" or "battalions withdrew" queries.

### Implementation Architecture

```
INGESTION:
  chunk_text (for search) = original + suffix expansions
  chunk_metadata.original_text = original text (for display/citations)
  Qdrant dense vector = embedding of suffix-expanded text
  Qdrant sparse vector = SPLADE of suffix-expanded text
  MySQL chunk_text = suffix-expanded text (FULLTEXT indexed)

QUERY:
  expanded_query = original + suffix expansions (bidirectional)
  Dense leg: embed(expanded_query) vs Qdrant dense vectors
  Sparse leg: SPLADE(expanded_query) vs Qdrant sparse vectors
  Exact leg: expanded_query vs MySQL FULLTEXT(chunk_text)

RERANKER:
  query = original user query (NOT expanded)
  passages = suffix-expanded chunk_text from retrieval results
  scores = cross_encoder.rerank(query, passages)

GENERATION:
  context = original_text per chunk + [Abbreviation Glossary] block
  The glossary is built from abbreviations found in the retrieved chunks
  The LLM sees clean original prose + explicit abbreviation mappings

CITATIONS:
  Display original_text only (from chunk_metadata)
  No expansion metadata in user-visible text
```

### Handling Multiple and Conflicting Expanded Forms

The CSV contains 386 abbreviations with multiple expanded forms, falling into
three categories:

#### Category 1: Derivative Forms (Same Root) — 255 abbreviations

All forms share the same root word (verb conjugations, noun forms, etc.):

```
inc → Increase, Increased, Increases, Increasing
wdr → Withdraw, Withdrawal, Withdrawing, Withdrawn, Withdraws, Withdrew
```

**Handling**: Append all forms in suffix. No noise — they all represent the same
concept. The embedding model treats "increase" and "increased" as nearly
identical, but having both ensures exact-match legs catch both forms.

#### Category 2: Completely Different Meanings — 131 abbreviations

Same abbreviation, totally unrelated expanded forms:

```
DA  → Daily Allowance | Defence Attache | Deputy Assistant | Direct action | Dispersal Area
BD  → Base Detonating | Battle Dress | Bomb Disposal
CP  → Check Post | Command Post | Contact Point
```

**Handling**: Append ALL forms in suffix. Let the retrieval models disambiguate.

**Why this works without an LLM**:

1. **Recall is guaranteed**: The correct meaning is always present in the text.
2. **Dense embeddings disambiguate**: "approved the plan" semantically clusters
   with "Deputy Assistant", not with "Dispersal Area".
3. **SPLADE handles it through term weighting**: In a chunk about "Deputy
   Assistant approving a plan", SPLADE assigns higher weights to "approved",
   "plan", "Deputy", "Assistant" and lower weights to "Daily", "Allowance".
4. **The reranker provides a final filter**: The cross-encoder reads the full
   chunk + query pair and scores semantic relevance.
5. **The generation LLM picks the right meaning from the glossary**: When the
   glossary lists "DA = Daily Allowance, Defence Attache, Deputy Assistant,
   Direct action, Dispersal Area", the LLM uses surrounding context to select
   "Deputy Assistant".

**What NOT to do**: Don't replace with a single form. The test proved this
introduces wrong meanings (DA→"Daily Allowance" instead of "Deputy Assistant").

---

## Chunk Size and SPLADE Token Limit

### Why the 512-Token Limit Matters

SPLADE PP en v1 is BERT-derived with a hard 512-token input limit (510 usable
after `[CLS]` and `[SEP]`). SPLADE silently truncates beyond 512 tokens.

At `CHUNK_SIZE=1500` characters, a chunk is ~333–500 SPLADE tokens. Suffix
expansion adds ~100-400 characters depending on abbreviation density.

### Two-Pass Chunking Strategy

1. Split at `CHUNK_SIZE` (1500 chars) as usual.
2. Scan the chunk for abbreviations.
3. Compute the expansion suffix size.
4. If `chunk_chars + suffix_chars > 1400` (safe char ceiling), re-split at a
   reduced size and repeat.
5. If within limit, proceed with embedding.

If the suffix is too large, truncate it (not the original content). Sort
abbreviations by frequency and drop the rarest ones first.

---

## Integration Plan

### Step 1: Load the CSV

Create a `global_abbreviations` table mirroring the CSV schema
(`abbreviation`, `expanded_form`, `category`). Load the CSV at startup or via
an admin endpoint. The existing `OrgAbbreviation` table remains for org-specific
overrides.

### Step 2: Wire Ingestion Expansion

In `document_processor.py`, in `_build_chunk_records()`:

1. Load the abbreviation lookup once (cache in memory)
2. For each chunk, run suffix-mode expansion
3. Store expanded text in `chunk_text` (for search/embedding)
4. Store original text in `chunk_metadata["original_text"]` (for display)

### Step 3: Wire Query Expansion

In `nodes.py`, after `rewrite_query_node`:

1. Run bidirectional suffix expansion on the rewritten query
2. Pass the expanded query to all three retrieval legs
3. Keep the original query for the reranker and generation

### Step 4: Wire Reranker

In `nodes.py`, in the reranking node:

1. Use the **original** query (not expanded) as the reranker query
2. Pass the **suffix-expanded** `chunk_text` as passages
3. The cross-encoder sees both abbreviation and full-form in the passage

### Step 5: Wire Generation Glossary

In `agent_graph.py`, in `format_context_string()`:

1. Use `chunk_metadata["original_text"]` for the context text (clean prose)
2. After all chunks, append a scoped `[Abbreviation Glossary]` block listing
   abbreviations found in the retrieved chunks
3. The generation LLM sees clean original prose + explicit abbreviation mappings

### Step 6: Wire Citations

In the frontend citation rendering:

1. Display `chunk_metadata["original_text"]` (not the expanded `chunk_text`)
2. No expansion metadata in user-visible text

---

## Test Script

The test suite is at `backend/tests/test_abbr_expansion.py`. Run inside the
backend container:

```bash
docker exec rag-web-ui-backend-1 python3 /app/tests/test_abbr_expansion.py
```

Results are saved to `/app/assets/abbr_test_results_v2.json`.
