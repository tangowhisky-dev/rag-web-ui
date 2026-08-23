# Abbreviation Expansion for RAG — Design Analysis

## Objective

Use the military abbreviations CSV (`abbreviations_enhanced.csv`, 3,090 rows, 2,193
unique abbreviations) to improve retrieval recall in the agentic RAG pipeline.
Expansion must be deterministic (no LLM calls) and handle abbreviations with
multiple derivative forms and completely different meanings.

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

### Existing Abbreviation Scaffolding (Dead Code)

- `backend/app/services/retrieval/query_expander.py` — `expand()` function and `load_org_abbreviations()`
- `backend/app/models/organisation.py` — `OrgAbbreviation` table (per-org, 1:1 mapping)
- `backend/app/api/api_v1/admin.py` — CRUD endpoints for org abbreviations
- **None of these are called in the actual pipeline.** The `expand` function is exported via
  `retrieval/__init__.py` but never imported by any node or tool.

### The Problem

If a chunk says "the CO ordered bns to wdr" and the user queries "commanding officer
ordered battalions to withdraw", none of the three legs match well:

- **Dense**: "CO" and "commanding officer" produce different embeddings (Qwen3 does not
  reliably know military abbreviations)
- **Sparse**: SPLADE does not expand "CO" to "Commanding Officer"
- **Exact**: MySQL FULLTEXT does not match "CO" to "commanding officer"

---

## Expansion Strategy

### Two Insertion Points (Both Required)

#### 1. Ingestion-Time Expansion (Chunk Enrichment)

**Where**: `document_processor.py`, in `_build_chunk_records()` (~line 464), after
chunking and before embedding.

**What happens**: For each chunk, scan for abbreviations, append all expanded forms
as a suffix block. The enriched text replaces `chunk_text` for embedding and FULLTEXT
indexing. The original text is preserved in `chunk_metadata["original_text"]` for
display in citations.

**Before expansion**:
```
chunk_text: "The CO ordered bns to wdr from the position"
```

**After expansion (suffix mode)**:
```
chunk_text: "The CO ordered bns to wdr from the position [Expansions: CO=Commanding Officer; bns=Battalions; wdr=Withdraw Withdrawal Withdrawing Withdrawn Withdraws Withdrew]"
chunk_metadata: {"original_text": "The CO ordered bns to wdr from the position", "source": "ops_report.pdf"}
```

**What gets stored**:
- MySQL `chunk_text`: enriched text — FULLTEXT indexed, searched by exact leg
- MySQL `chunk_metadata.original_text`: original text for display
- Qdrant `dense` vector: embedding of enriched text
- Qdrant `sparse` vector: SPLADE embedding of enriched text
- Qdrant `payload.chunk_text`: enriched text (returned with hits)
- Qdrant `payload.original_text`: original text (for citation display)

#### 2. Query-Time Expansion (Query Enrichment)

**Where**: `nodes.py`, after `rewrite_query_node` (~line 240) and before retrieval
nodes (~line 429).

**What happens**: Scan the rewritten query for abbreviations → append all expanded
forms. Also scan for full forms → append the abbreviation. This is bidirectional
expansion.

**Before expansion**:
```
rewritten_query: "battalions withdrew from position"
```

**After expansion**:
```
expanded_query: "battalions withdrew from position bns wdr Withdraw Withdrawal Withdrawing Withdrawn Withdraws Withdrew Battalion"
```

**What gets searched**:
- Dense leg: embedding of expanded query vs embedding of enriched chunks
- Sparse leg: SPLADE of expanded query vs SPLADE of enriched chunks
- Exact leg: expanded query string vs enriched `chunk_text` via MySQL FULLTEXT

The original (un-expanded) query is still used for LLM context and display.

---

## Handling Multiple and Conflicting Expanded Forms

### Three Categories of Multi-Form Abbreviations

The CSV contains 386 abbreviations with multiple expanded forms, falling into three
categories:

#### Category 1: Derivative Forms (Same Root) — 255 abbreviations

All forms share the same root word. These are verb conjugations, noun forms,
adjective forms.

```
inc → Increase, Increased, Increases, Increasing
wdr → Withdraw, Withdrawal, Withdrawing, Withdrawn, Withdraws, Withdrew
sp  → Support, Supported, Supporter, Supporting, Supportive, Supports
ack → Acknowledge, Acknowledged, Acknowledgement, Acknowledges, Acknowledging
```

**Handling**: Append all forms. No noise — they all represent the same concept.
The embedding model treats "increase" and "increased" as nearly identical, but
having both ensures exact-match legs (MySQL FULLTEXT, SPLADE) catch both forms.

#### Category 2: Completely Different Meanings — 131 abbreviations

Same abbreviation, totally unrelated expanded forms.

```
DA  → Daily Allowance | Defence Attache | Deputy Assistant | Direct action | Dispersal Area
BD  → Base Detonating | Battle Dress | Bomb Disposal
CP  → Check Post | Command Post | Contact Point
ARP → Air Raid Precautions | Ammunition Replenishment Point | Aviation Reconnaissance Patrol
```

**Handling**: Append ALL forms. Let the embedding model disambiguate.

**Why this works without an LLM**:

1. **Recall is guaranteed**: The correct meaning is always present in the text.
2. **The embedding model handles disambiguation**: "approved the plan" semantically
   clusters with "Deputy Assistant" and "Defence Attache", not with "Dispersal Area"
   or "Daily Allowance". Dense cosine similarity ranks the correct chunk higher
   because surrounding context creates a semantic signal that overrides noise.
3. **SPLADE handles it through term weighting**: In a chunk about "Deputy Assistant
   approving a plan", SPLADE assigns higher weights to "approved", "plan", "Deputy",
   "Assistant" and lower weights to "Daily", "Allowance", "Dispersal". Unrelated
   expansions become low-weight noise.
4. **MySQL FULLTEXT handles it through BM25**: "approved" and "plan" have higher
   TF-IDF scores than appended expansion terms because they appear in fewer
   documents. Expansion terms are common (appear in many chunks) so their IDF is low.
5. **The reranker (cross-encoder) provides a final filter**: The cross-encoder reads
   the full chunk + query pair and scores semantic relevance. Chunks where the
   expansion noise doesn't match the query intent get down-ranked.

**What NOT to do**: Don't try to disambiguate deterministically. You cannot know
which meaning of "DA" is correct without understanding the surrounding text, and
that's the embedding model's job. Building rules for this ("if 'approved' is
nearby, DA = Deputy Assistant") is rebuilding a language model with regex.

#### Category 3: Mixed (Partially Related) — varies

Some forms share a root, others are completely different.

```
dep → Depart, Departed, Departing, Departs, Departure, Depot, Depoted, Depoture
rel → Release, Released, Releases, Releasing, Relieve, Relieved, Relieves, Relieving
br  → Branch, Bridge, Bridged, Bridging
```

**Handling**: Same as Category 2 — append all forms. The derivative forms help
exact matching, the different meanings are disambiguated by the embedding model.

---

## Expansion Mode: Suffix

The expander supports three modes. **Suffix mode** is the correct choice for this
pipeline.

### Suffix mode (recommended)
```
"The CO ordered bns to wdr [Expansions: CO=Commanding Officer; bns=Battalions; wdr=Withdraw Withdrawal Withdrawing Withdrawn Withdraws Withdrew]"
```

- Keeps original text clean for dense embeddings (no parenthetical interruptions)
- SPLADE and MySQL FULLTEXT don't care about position — both index all tokens
- Original text is readable and displayable

### Append mode (not recommended)
```
"The CO (Commanding Officer) ordered bns (Battalions) to wdr (Withdraw Withdrawal Withdrawing Withdrawn Withdraws Withdrew)"
```

- Disrupts natural text flow, which hurts dense embeddings
- The embedding model sees a confusing sentence with parenthetical interruptions

### Replace mode (not recommended)
```
"The Commanding Officer ordered Battalions to Withdraw Withdrawal Withdrawing Withdrawn Withdraws Withdrew"
```

- Destroys the original abbreviation — user can never search for "CO" and get a hit
- Loses exact-match capability for abbreviations

---

## Integration Plan

### Step 1: Load the CSV

The `OrgAbbreviation` model has `short` (String 64) and `expansion` (String 512),
mapping to `abbreviation` and `expanded_form`. But it's per-org.

Options:
- **Option A**: Insert all 2,193 abbreviations for every org (simple, but duplicative)
- **Option B**: Create a new `GlobalAbbreviation` table (cleaner, not org-scoped)

Option B is cleaner. The military abbreviations are not org-specific. A new table
`global_abbreviations` with columns `abbreviation`, `expanded_form`, `category`
mirrors the CSV schema. A startup script or admin endpoint loads the CSV.

The existing `OrgAbbreviation` table remains for org-specific overrides.

### Step 2: Wire Ingestion Expansion

In `document_processor.py`, in `_build_chunk_records()`:

1. Load the abbreviation lookup once (cache in memory or load from DB)
2. For each chunk, run suffix-mode expansion on `chunk.page_content`
3. Store enriched text in `chunk_text` (for search)
4. Store original text in `chunk_metadata["original_text"]` (for display)

### Step 3: Wire Query Expansion

In `nodes.py`, after `rewrite_query_node`:

1. Load the abbreviation lookup
2. Run bidirectional expansion on the rewritten query
3. Pass the expanded query to all three retrieval legs
4. Keep the original query for LLM context and display

### Step 4: Update `query_expander.py`

The current `expand()` function assumes a 1:1 mapping (`Dict[str, str]`). It needs
to handle `Dict[str, List[str]]` and append all forms in suffix mode.

---

## What Will Be Stored After Integration

### MySQL `document_chunks` table

| Field | Content | Purpose |
|-------|---------|---------|
| `chunk_text` | Enriched text (original + suffix expansions) | FULLTEXT indexed, searched by exact leg |
| `chunk_metadata` | JSON with `original_text` field | Display in citations |
| Other fields | Unchanged | Ownership, dedup, etc. |

### Qdrant

| Field | Content | Purpose |
|-------|---------|---------|
| `vector["dense"]` | Qwen3 embedding of enriched text | Searched by dense leg |
| `vector["sparse"]` | SPLADE embedding of enriched text | Searched by sparse leg |
| `payload.chunk_text` | Enriched text | Returned with hits |
| `payload.original_text` | Original chunk text | Display in citations |
| Other payload fields | Unchanged | Filtering, metadata |

### What Gets Searched at Query Time

| Leg | Query | Search target | Mechanism |
|-----|-------|---------------|-----------|
| Dense | Embedding of expanded query | Embedding of enriched chunks | Qdrant cosine similarity |
| Sparse | SPLADE of expanded query | SPLADE of enriched chunks | Qdrant sparse vector search |
| Exact | Expanded query string | Enriched `chunk_text` | MySQL FULLTEXT (BM25) |

---

## Risks and Mitigations

### Risk 1: Chunk size inflation

Suffix expansions add text to every chunk. See the separate chunk-size analysis
below for the detailed treatment.

### Risk 2: False positives in exact search

A chunk mentioning "DA" (meaning Defence Attache) with expansion appending "Daily
Allowance" might match a user searching for "daily allowance".

**Mitigation**: The reranker (cross-encoder) reads the full chunk + query pair and
scores semantic relevance. Chunks where the expansion noise doesn't match the query
intent get down-ranked. The 3-leg merge + rerank pipeline already handles this.

### Risk 3: Existing chunks need re-ingestion

All current chunks in Qdrant and MySQL have un-expanded text.

**Mitigation**: Expansion only applies to new documents. For existing documents,
trigger re-ingestion via the existing document update flow.

### Risk 4: The `expand` function is never called

The scaffolding exists but is dead code.

**Mitigation**: This is actually good — there's no existing behavior to break. The
integration is purely additive.

---

## Chunk Size and SPLADE Token Limit

See [chunking.md](chunking.md) for the full chunking rationale. The critical
constraint is reproduced here.

### Why the 512-Token Limit Exists

The sparse leg uses **SPLADE PP en v1**, a BERT-derived model. BERT has a hard
512-token input limit (510 usable after `[CLS]` and `[SEP]`). SPLADE silently
truncates anything beyond 512 tokens — no error is raised, but the sparse vector
only represents the first ~500 tokens of the chunk.

The current `CHUNK_SIZE=1500` characters was chosen because English text tokenizes
at roughly 3–4.5 characters per BERT WordPiece token:

| Text type | Chars/token | Safe char ceiling |
|-----------|-------------|-------------------|
| Plain prose | ~4.5 | ~2295 |
| Mixed technical/prose | ~4.0 | ~2040 |
| Dense technical (jargon, codes) | ~3.0–3.5 | ~1530–1785 |

At 1500 characters, a chunk is ~333–500 SPLADE tokens — safely within the 512 limit.

### How Expansion Affects Chunk Size

Expansion adds a suffix block to each chunk. The suffix contains all expanded forms
for every abbreviation found in the chunk. The size of this suffix depends on:

1. How many abbreviations appear in the chunk
2. How many expanded forms each abbreviation has

### Ensuring Expanded Chunks Stay Within SPLADE's Limit

The approach is to **split first, then expand, then check the token budget, and
re-split if needed**. This is a two-pass chunking strategy:

#### Pass 1: Split at a reduced character target

Split at a character target that accounts for the expected expansion overhead.
The original `CHUNK_SIZE=1500` produces chunks of ~333–500 SPLADE tokens. If we
split at a reduced target (e.g., 1000–1200 characters), the pre-expansion chunk
is ~222–400 tokens, leaving ~112–290 tokens of headroom for the expansion suffix.

The reduction does not need to be a fixed number — it can be computed per-chunk
based on how many abbreviations are found:

1. Split at `CHUNK_SIZE` (1500 chars) as usual.
2. Scan the chunk for abbreviations.
3. Compute the expansion suffix size in characters.
4. If `chunk_chars + suffix_chars > SPLADE_SAFE_LIMIT`, re-split this chunk at a
   reduced size and repeat from step 2.
5. If within limit, proceed with embedding.

The `SPLADE_SAFE_LIMIT` is a character budget, not a token budget. Since we can't
tokenize with SPLADE's tokenizer at split time (it's expensive and not thread-safe),
we use a conservative character ceiling. At ~3 chars/token (worst case for technical
text), 512 tokens = 1536 characters. Use 1400 as the safe character ceiling to
leave margin for tokenizer overhead.

#### Pass 2: Truncate the suffix, not the content

If a chunk has many abbreviations and the full suffix would push it over the
character budget, truncate the suffix — never the original content. The suffix is
metadata for search; the original text is the actual content.

Suffix truncation strategy:
- Sort abbreviations found in the chunk by frequency (most common first)
- Append expansions until the character budget is exhausted
- Drop the remaining expansions

This ensures the most frequently occurring abbreviations in the chunk get their
expansions indexed, while rare ones may be dropped. Since rare abbreviations are
less likely to be queried, this has minimal impact on recall.

#### Why this works without limiting to 3 forms or capping at 100 characters

The user's constraint is: don't limit expansions to the first 3 forms per
abbreviation, and don't cap the suffix at 100 characters. The two-pass approach
respects this:

- **No 3-form limit**: All forms are appended for each abbreviation that fits
  within the budget. An abbreviation with 9 forms (like `op`) gets all 9 forms
  if the budget allows. The budget is per-chunk, not per-abbreviation.

- **No 100-character cap**: The suffix can be as long as the remaining token
  budget allows. A chunk with 2 abbreviations might have 200+ characters of
  suffix. A chunk with 10 abbreviations might have 400+ characters. The cap is
  the SPLADE token limit, not an arbitrary number.

- **The original content is never truncated**: If the suffix doesn't fit, the
  suffix is shortened, not the content. This preserves the semantic signal from
  the original text.

#### Concrete example

Chunk (1200 chars, ~300 SPLADE tokens):
```
The CO ordered the bns to wdr from the forward position. The MO reported
casualties. The adjt coordinated with HQ. The DA approved the medical resupply.
The op was conducted at first light. The recce team provided intelligence...
```

Abbreviations found: CO, bns, wdr, MO, adjt, HQ, DA, op, recce (9 abbreviations)

Expansion suffix (all forms):
```
[Expansions: CO=Commanding Officer; bns=Battalions; wdr=Withdraw Withdrawal
Withdrawing Withdrawn Withdraws Withdrew; MO=Medical Officer; adjt=Adjutant;
HQ=Headquarters; DA=Daily Allowance Defence Attache Deputy Assistant Direct
action Dispersal Area; op=Operate Operated Operates Operating Operation
Operational Operations Operator Operators; recce=Reconnaissance Reconnoitre
Reconnoitred Reconnoitres Reconnoitring]
```

Suffix size: ~400 chars (~100 SPLADE tokens)

Total: 1200 + 400 = 1600 chars (~400 SPLADE tokens) — within the 512-token limit.

If the chunk were 1400 chars with the same 9 abbreviations:
Total: 1400 + 400 = 1800 chars (~450 SPLADE tokens) — still within limit.

If the chunk were 1500 chars with 15 abbreviations and a 600-char suffix:
Total: 1500 + 600 = 2100 chars (~525 SPLADE tokens) — over the limit.

In this case, the two-pass approach kicks in:
1. Re-split the chunk at 1200 chars (producing two chunks: 1200 + 300)
2. Expand each sub-chunk
3. The 1200-char sub-chunk with ~10 abbreviations: 1200 + 450 = 1650 chars (~412 tokens) — within limit
4. The 300-char sub-chunk with ~5 abbreviations: 300 + 250 = 550 chars (~137 tokens) — within limit

This is the correct behavior: the chunk is split to accommodate the expansion,
not the other way around.

#### Implementation detail: when to re-split

Re-splitting is expensive (re-running the text splitter). To avoid it in the
common case:

1. Use a reduced initial split size (e.g., 1200 chars instead of 1500) that
   leaves ~300 chars of headroom for expansion. Most chunks will fit without
   re-splitting.

2. Only re-split if the expanded chunk exceeds the character budget. In practice,
   this will be rare because:
   - Most chunks contain 0–5 abbreviations
   - Most abbreviations have 1–6 forms
   - 5 abbreviations × 6 forms × ~15 chars/form = ~450 chars of suffix
   - 1200 + 450 = 1650 chars (~412 tokens) — within limit

3. For the rare chunk that needs re-splitting, split at `budget / 2` and expand
   each half. This is a single additional split operation, not a loop.

#### The character budget formula

```
SPLADE_SAFE_CHARS = 1400  # conservative ceiling at ~3 chars/token → ~467 tokens

expanded_chunk_chars = original_chunk_chars + expansion_suffix_chars

if expanded_chunk_chars <= SPLADE_SAFE_CHARS:
    proceed with embedding
else:
    re-split chunk at (SPLADE_SAFE_CHARS * 0.6) chars
    expand each sub-chunk
    proceed with embedding
```

The 0.6 ratio ensures each sub-chunk has 40% of the budget for expansion overhead.
At `SPLADE_SAFE_CHARS=1400`, each sub-chunk is 840 chars, leaving 560 chars for
expansion — more than enough for any realistic abbreviation density.

---

## Context for LLM Generation: Why Suffix Chunks Must Not Reach the LLM

### The Problem

If each chunk stored in the DB has an expansion suffix like:

```
The CO ordered bns to wdr from the position [Expansions: CO=Commanding Officer; bns=Battalions; wdr=Withdraw Withdrawal Withdrawing Withdrawn Withdraws Withdrew]
```

And this text is fed directly to the generation LLM, two things break:

1. **Continuity and flow**: The LLM sees `[Expansions: ...]` blocks interrupting
   every chunk. Multiple chunks in the context become a noisy mix of prose and
   metadata. The LLM may try to quote or reference the expansion block in its
   answer, producing output like "according to [Expansions: CO=Commanding Officer]..."

2. **Token waste**: The expansion suffix consumes context window tokens without
   adding information the LLM needs. The LLM already knows what "Commanding
   Officer" means — the suffix exists for the embedding model and BM25, not for
   the generation model.

### How Context Reaches the Generation LLM (Current Code Path)

The pipeline has three places where chunk text flows to an LLM:

#### Path 1: `think_node` (agent reasoning)

```
retrieved_docs → _observations_text(observations, full=True)
              → doc.get("page_content") per chunk
              → appended to think_user_prompt
              → LLM decides next tool call
```

File: `agent_graph.py:298-354`. Each doc's `page_content` is rendered as
`doc_{j}: {content}` in the observations text. The LLM reads this to judge
whether retrieval answered the query.

#### Path 2: `finalize_node` (answer generation)

```
retrieved_docs → format_context_string(docs, file_markdown)
              → _prune_contiguous_overlaps(docs)
              → doc.get("page_content") per chunk
              → "[KB-N] (source)\ncontent" joined by "\n\n---\n\n"
              → inserted as "Retrieved context (the only citable evidence):"
              → LLM generates the final answer
```

File: `agent_graph.py:1318`, `utils.py:34-56`. This is the primary context
string the generation LLM sees. The `page_content` of each retrieved chunk
becomes the body of a `[KB-N]` block.

#### Path 3: `answer_evaluation_node` (confidence scoring)

```
retrieved_docs → format_context_string(docs)
              → context_preview passed to evaluator LLM
```

File: `nodes.py:634`. The evaluator LLM reads the context to score
faithfulness and completeness.

#### Path 4: Citation display in the UI

```
cited_docs → writer({"event": "answer_rewrite", "citations": cited_docs})
           → frontend maps doc.page_content → citation.text
           → cleanChunkText(citation.text) rendered in citation popover
```

File: `agent_graph.py:1437-1439`, `frontend/src/app/dashboard/chat/[id]/page.tsx:557-571`,
`frontend/src/components/chat/answer.tsx:540`. The `page_content` of each cited
doc is sent to the frontend as `citation.text` and displayed in the citation
popover. `cleanChunkText` does light formatting (strip OCR markers, convert
bullets) but does not strip expansion suffixes.

### The Design: Separate Search Text from Display Text

The solution is to store **two versions of each chunk** and use the right one
at each stage:

| Field | Content | Used by |
|-------|---------|---------|
| `chunk_text` (MySQL) / `payload.chunk_text` (Qdrant) | Enriched text (original + suffix) | Embedding, SPLADE, FULLTEXT indexing — never sent to LLM |
| `chunk_metadata.original_text` | Original text only | LLM context, citation display |

At retrieval time, `_qdrant_payload_to_doc` (retrieval.py:112) currently does:

```python
def _qdrant_payload_to_doc(payload: dict) -> LangchainDocument:
    chunk_text = payload.get("chunk_text", "")
    metadata = {k: v for k, v in payload.items() if k != "chunk_text"}
    return LangchainDocument(page_content=chunk_text, metadata=metadata)
```

This sets `page_content` to `chunk_text` (the enriched version). After the
abbreviation integration, this function would be changed to prefer
`original_text` from the payload:

```python
def _qdrant_payload_to_doc(payload: dict) -> LangchainDocument:
    # Use original_text for LLM context and citations;
    # chunk_text (enriched) is only for embedding/indexing.
    original = payload.get("original_text")
    chunk_text = payload.get("chunk_text", "")
    metadata = {k: v for k, v in payload.items()
                if k not in ("chunk_text", "original_text")}
    return LangchainDocument(
        page_content=original if original else chunk_text,
        metadata=metadata,
    )
```

Similarly, the exact search leg (retrieval.py:431) reads `chunk_text` from
MySQL for FULLTEXT matching but should return `original_text` in the
`LangchainDocument.page_content`:

```python
# Current:
chunk_text = row.chunk_text or ""
# After:
chunk_text = (row.chunk_metadata or {}).get("original_text") or row.chunk_text or ""
```

### What Each Stage Sees After the Fix

#### Embedding (ingestion time)
```
Input: enriched chunk_text (original + suffix)
       "The CO ordered bns to wdr [Expansions: CO=Commanding Officer; ...]"
Output: dense vector + sparse vector
```
The embedding model and SPLADE see the full enriched text. This is correct —
the expansion suffix is what makes the chunk match abbreviation queries.

#### Retrieval (query time)
```
Query: expanded query (original + appended forms)
       "battalions withdrew from position bns wdr Withdraw ..."
Search: against enriched chunk_text in Qdrant/MySQL
Hits: returns point IDs with payloads
```
The search uses the enriched text for matching. Correct.

#### LLM context (think_node, finalize_node, evaluation)
```
page_content = original_text (no suffix)
"The CO ordered bns to wdr from the position"
```
The generation LLM sees clean, continuous prose. No `[Expansions: ...]` blocks.
The LLM can read the chunk naturally, understand the context, and generate a
coherent answer. The abbreviations in the original text ("CO", "bns", "wdr")
are either understood by the LLM directly or resolved through the conversation
context.

#### Citation display (UI)
```
citation.text = original_text (no suffix)
"The CO ordered bns to wdr from the position"
```
The user sees the original document text in the citation popover. No expansion
metadata. Clean and readable.

#### Token budget / compaction
```
count_tokens(doc.page_content) → counts original_text, not enriched text
```
The compaction system (`_trim_docs_to_budget`, `_compact_if_needed`) measures
`page_content` to decide which chunks to drop when over budget. Since
`page_content` is now the original text (shorter than enriched), the token
budget is more accurate and the LLM context is smaller.

### Why the LLM Needs a Glossary (Not Suffix Chunks)

The expansion suffix exists to bridge the gap between abbreviations and full
forms for **lexical and embedding-based matching**. The generation LLM has
different needs — but it still needs help connecting abbreviations to expanded
forms.

#### The gap in a suffix-only design

If chunks passed to the LLM contain only original text (abbreviations, no
expansions) and the query is also original (not expanded for the LLM), the LLM
sees:

```
User query: battalions withdrew from position

Retrieved context:
[KB-1] (ops_report.pdf)
The CO ordered bns to wdr from the forward position. The MO reported casualties...
```

The LLM must connect "battalions" in the query to "bns" in the chunk, and
"withdrew" to "wdr". For common abbreviations (CO, MO, HQ), a modern LLM can
infer from military context. For obscure ones (wdr, adjt, recce, spt), it
cannot reliably make the connection. There is no expanded form anywhere in the
LLM's input to bridge the gap.

#### Why the expanded query doesn't solve this

The `finalize_node` prompt (agent_graph.py:1346-1348) shows the LLM two query
fields:

```python
parts = [f"User query: {query}\n\n"]
if retrieval_query and retrieval_query != query:
    parts.append(f"Resolved retrieval query: {retrieval_query}\n\n")
```

If we expand the retrieval query in suffix mode, the LLM would see:

```
Resolved retrieval query: battalions withdrew from position bns wdr Withdraw Withdrawal Withdrawing Withdrawn Withdraws Withdrew Battalion
```

This is noisy and doesn't cleanly map "bns" to "Battalions" — it just appends
tokens. The LLM might make the connection, but it's not reliable, and the
suffix noise wastes context tokens.

#### The solution: a compact abbreviation glossary

After retrieval, scan the retrieved chunks for abbreviations, look up their
expansions, and append a compact glossary block to the context string. This is
separate from the chunk suffix (which is for retrieval indexing) and separate
from the query expansion (which is for retrieval matching).

```
User query: battalions withdrew from position

Retrieved context:
[KB-1] (ops_report.pdf)
The CO ordered bns to wdr from the forward position. The MO reported casualties...

[Abbreviation Glossary]
CO = Commanding Officer
bns = Battalions
wdr = Withdraw, Withdrawal, Withdrawing, Withdrawn, Withdraws, Withdrew
MO = Medical Officer
```

This gives the LLM the mapping it needs without polluting chunk prose.

**Properties of the glossary**:
- **Compact**: one line per abbreviation, not a suffix on every chunk
- **Scoped**: only abbreviations that actually appear in the retrieved chunks,
  not all 2,193
- **Deduplicated**: if 5 chunks all contain "CO", the glossary lists it once
- **Clean**: doesn't disrupt the prose flow of the context
- **Multi-form aware**: for abbreviations with multiple meanings (e.g. DA has
  5), all forms are listed so the LLM can pick the right one from context

**Token cost estimate**: With `RETRIEVAL_TOP_K=20` (default) and after
reranking/dedup/filtering, typically 5–15 chunks reach the LLM. Based on
simulation with military text, 3 chunks contained 40 unique abbreviations,
producing a ~800-char glossary (~200 tokens) in ultra-compact form (first form
only) or ~1200 chars (~300 tokens) with all forms. For 10–15 chunks, expect
~60–80 unique abbreviations, producing ~400–600 tokens of glossary. This is
within the context budget — the compaction system (`_compact_if_needed`) will
trim chunks if the total exceeds the limit.

**Where to inject the glossary**: In `format_context_string` (utils.py:34),
after all chunk blocks but before `file_markdown`:

```python
def format_context_string(docs, file_markdown=None, abbr_glossary=None):
    pruned_docs = _prune_contiguous_overlaps(docs) if docs else docs
    parts = []
    for i, doc in enumerate(pruned_docs, 1):
        content = doc.get("page_content", "").strip()
        source = doc.get("metadata", {}).get("source", "")
        header = f"[KB-{i}]" + (f" ({source})" if source else "")
        parts.append(f"{header}\n{content}")
    if abbr_glossary:
        parts.append(f"[Abbreviation Glossary]\n{abbr_glossary}")
    if file_markdown:
        parts.append(f"[File Content]\n{file_markdown}")
    return "\n\n---\n\n".join(parts)
```

The glossary is built in `finalize_node` (agent_graph.py:1318) after retrieval
and before the LLM call:

1. Collect all `page_content` from retrieved docs (original text with
   abbreviations)
2. Scan for abbreviations using the abbreviation lookup
3. Build a compact glossary string: `{abbr} = {form1}, {form2}, ...` per line
4. Pass it to `format_context_string` as `abbr_glossary`

**What about the query?** The query shown to the LLM should remain the original
user query — NOT the expanded version. The expansion is only for retrieval
matching. The glossary provides the mapping the LLM needs to connect the
original query terms to the abbreviation-bearing chunks.

**What about multi-meaning abbreviations?** The glossary lists all forms. For
`DA`, the glossary would show:
```
DA = Daily Allowance, Defence Attache, Deputy Assistant, Direct action, Dispersal Area
```
The LLM reads the chunk context ("The DA approved the medical resupply") and
the glossary, and infers that "Deputy Assistant" or "Defence Attache" is the
correct meaning — not "Dispersal Area". This is the same disambiguation the
embedding model does, but the LLM is much better at it because it reads the
full sentence context.

---

## The Reranker Problem

### How the Reranker Works

The reranker is a cross-encoder: `Xenova/ms-marco-MiniLM-L-12-v2` (ONNX). Unlike
bi-encoders (which embed query and passage separately), a cross-encoder
concatenates query and passage into a single input:

```
[CLS] query tokens [SEP] passage tokens [SEP]
```

It then runs the full transformer over both and outputs a single relevance
score. This means the cross-encoder sees the **exact text** of both query and
passage — not embeddings, not bag-of-words. It reads them as natural language.

File: `backend/app/services/retrieval/reranker.py:98-99`:
```python
passages = [doc.page_content for doc in docs]
scores = list(encoder.rerank(query, passages))
```

The query comes from `reranking_node` (nodes.py:306):
```python
query = state.get("rewritten_query", state.get("original_query", ""))
```

The passages come from `doc.page_content` of each retrieved doc.

### The Flow Through the Pipeline

```
rewrite_query_node → rewritten_query
                          ↓
think_node → LLM decides to call rag_retrieve with query=rewritten_query
                          ↓
_run_retrieval_pass:
  state["rewritten_query"] = query  (the rewritten query)
                          ↓
  dense_retrieval_node  → uses rewritten_query to embed, searches Qdrant
  sparse_retrieval_node → uses rewritten_query to SPLADE-embed, searches Qdrant
  exact_retrieval_node  → uses rewritten_query for MySQL FULLTEXT
                          ↓
  merge_node → RRF merge of 3 legs
                          ↓
  reranking_node → rerank(query=rewritten_query, docs=merged_docs)
                   passages = [doc.page_content for doc in docs]
                          ↓
  filter_node → threshold filter on reranker scores
```

### The Three Options for What the Reranker Sees

After abbreviation integration, there are three possible configurations for
what the reranker receives:

#### Option A: Original query + original chunks (no expansion at reranker)

```
Query:    "battalions withdrew from position"
Passages: ["The CO ordered bns to wdr from the forward position..."]
```

**Behavior**: The cross-encoder reads "battalions" in the query and "bns" in
the passage. ms-marco-MiniLM is trained on web search queries, not military
text. It does not know that "bns" = "battalions" or "wdr" = "withdrew". The
relevance score will be low because the lexical overlap is minimal. Relevant
chunks may be filtered out by the threshold (-2.0 default).

**Verdict**: Bad. The reranker becomes a bottleneck that filters out exactly
the chunks we worked to retrieve.

#### Option B: Expanded query + original chunks

```
Query:    "battalions withdrew from position bns wdr Withdraw Withdrawal Withdrawing Withdrawn Withdraws Withdrew Battalion"
Passages: ["The CO ordered bns to wdr from the forward position..."]
```

**Behavior**: The cross-encoder sees "bns" in both the expanded query and the
passage. The lexical overlap increases. But the query is now noisy — it
contains "Withdraw Withdrawal Withdrawing Withdrawn Withdraws Withdrew" which
are not what the user asked. The cross-encoder may score the passage higher
due to "bns" and "wdr" overlap, but the noise tokens dilute the semantic
signal. The cross-encoder's attention mechanism splits focus between the
user's actual intent ("battalions withdrew from position") and the appended
expansion tokens.

**Verdict**: Mixed. Better recall than Option A, but the noise degrades
precision. The cross-encoder's relevance scoring is less reliable because the
query no longer represents a single intent.

#### Option C: Original query + enriched chunks (with suffix)

```
Query:    "battalions withdrew from position"
Passages: ["The CO ordered bns to wdr from the forward position [Expansions: CO=Commanding Officer; bns=Battalions; wdr=Withdraw Withdrawal Withdrawing Withdrawn Withdraws Withdrew]"]
```

**Behavior**: The cross-encoder sees "battalions" in the query and
"Battalions" in the passage suffix. The lexical overlap is clear. But the
passage now contains `[Expansions: CO=Commanding Officer; ...]` which is not
natural language. The cross-encoder (a transformer trained on natural language
pairs) may not handle this well — the `[Expansions: ...]` block is unlike
anything in its training data. It may attend to the expansion block and ignore
the actual passage content, or it may be confused by the non-natural format.

**Verdict**: Mixed. The lexical overlap helps, but the non-natural suffix
format may confuse the cross-encoder. The passage no longer looks like a
passage from a document — it looks like a passage with metadata bolted on.

### The Correct Design: Original Query + Original Chunks at Reranker

The reranker should see **original query + original chunks** — Option A — but
this requires solving the lexical gap that Option A creates.

The key insight: **the reranker's job is to score semantic relevance, not
lexical overlap**. The cross-encoder is a 12-layer transformer that
*understands* language, not just keyword matching. The problem with Option A
is not that the cross-encoder can't handle it — it's that ms-marco-MiniLM-L-12-v2
was trained on web search data and doesn't know military abbreviations.

There are two ways to solve this:

#### Solution 1: Use the glossary at the reranker (recommended)

Feed the cross-encoder a modified passage that includes a glossary line, not
the full suffix:

```
Query:    "battalions withdrew from position"
Passage:  "The CO ordered bns to wdr from the forward position. The MO reported casualties.
           [bns=Battalions; wdr=Withdraw, Withdrawal, Withdrawing, Withdrawn, Withdraws, Withdrew; CO=Commanding Officer; MO=Medical Officer]"
```

This is different from the ingestion suffix. The ingestion suffix is stored in
the DB and used for embedding. The reranker glossary is built at query time,
scoped to only the abbreviations in that specific passage, and formatted as a
compact key=value block that the cross-encoder can parse.

**Why this works**: The cross-encoder reads the passage as natural language
followed by a compact glossary. The glossary creates a lexical bridge: "bns"
appears in the passage, and "Battalions" appears in the glossary line of the
same passage. The cross-encoder's attention mechanism can connect "battalions"
in the query to "Battalions" in the glossary. The glossary is short (one line)
and doesn't overwhelm the passage content.

**Implementation**: In `reranking_node` (nodes.py:298), before calling
`rerank()`, build a per-passage glossary and append it to `page_content`:

```python
def reranking_node(state):
    query = state.get("rewritten_query", state.get("original_query", ""))
    docs = state.get("retrieved_docs", [])

    # Build per-passage glossary for the reranker
    for doc in docs:
        original = doc.get("page_content", "")
        glossary = build_glossary_for_text(original, abbr_lookup)
        if glossary:
            doc["page_content"] = f"{original}\n[{glossary}]"

    # Rerank with glossary-enriched passages
    reranked = rerank(query=query, docs=lc_docs, score_threshold=-inf)

    # Restore original page_content after reranking
    for doc in reranked:
        doc.metadata["_reranker_passage"] = doc.page_content  # keep for debugging
        doc.page_content = doc.metadata.get("original_text", doc.page_content)
```

The `page_content` is temporarily modified for the reranker call, then restored
to `original_text` for the LLM context. The reranker sees the glossary; the LLM
does not.

**Token cost**: A per-passage glossary for a chunk with 5 abbreviations is
~100 chars (~25 tokens). With 20 chunks, that's ~500 tokens of glossary across
all passages — negligible compared to the ~3000 tokens of chunk text.

#### Solution 2: Use an abbreviation-aware reranker model (not recommended)

Replace ms-marco-MiniLM with a model fine-tuned on military or technical text.
This is not practical because:
- No such model exists off-the-shelf
- Fine-tuning requires labeled query-passage pairs from this domain
- The model would need to be retrained for every new abbreviation set

#### Solution 3: Skip the reranker for abbreviation-heavy queries (not recommended)

If the query contains abbreviations, skip the reranker and use raw RRF scores.
This avoids the filtering problem but loses the precision benefit of the
cross-encoder for all other queries.

### Why Not Use the Expanded Query at the Reranker?

The expanded query (Option B) seems like the simpler fix — just pass the
expanded query to the reranker. But it has a subtle problem:

The cross-encoder scores the **pair** (query, passage). If the query is
"battalions withdrew from position bns wdr Withdraw Withdrawal...", the
cross-encoder sees a query that contains both the user's intent and a list of
expansion tokens. It scores the passage against this hybrid query, not against
the user's actual intent. A passage that mentions "wdr" would score high even
if it's about a different context (e.g. "wdr" in a different document about
water distribution reports). The cross-encoder can't distinguish between the
user's intent and the expansion noise.

With the per-passage glossary (Solution 1), the query stays clean ("battalions
withdrew from position") and the passage carries the mapping. The cross-encoder
scores the passage against the user's actual intent, with the glossary
providing the lexical bridge. This is semantically correct.

### Summary: What Each Stage Sees

| Stage | Query | Passage/Chunk | Why |
|-------|-------|---------------|-----|
| Dense retrieval | Expanded query | Enriched chunk (with suffix) | Token matching — needs all forms |
| Sparse retrieval | Expanded query | Enriched chunk (with suffix) | Token matching — needs all forms |
| Exact retrieval | Expanded query | Enriched chunk_text (MySQL FULLTEXT) | BM25 — needs all forms |
| Reranker | Original (rewritten) query | Original chunk + per-passage glossary | Cross-encoder scores semantic relevance; glossary bridges lexical gap |
| LLM context | Original user query | Original chunk + glossary block | Clean prose for generation; glossary for abbreviation mapping |
| Citation display | N/A | Original chunk text | User sees source document text |

### What About the Think Node's Full Observations?

The `think_node` receives `_observations_text(observations, full=True)` which
includes the full `page_content` of each doc (agent_graph.py:353). After the
fix, this will be `original_text` — clean prose. The think LLM can reason about
whether the retrieval answered the query without expansion noise.

### What About the Reranker?

The reranker (`reranker.py`) scores each chunk against the query using a
cross-encoder. The cross-encoder reads `page_content` (which will be
`original_text` after the fix) and the query. It does not need the expansion
suffix — it's scoring semantic relevance, not keyword overlap. The expanded
query (with appended forms) is what the reranker sees on the query side.

### Summary of Data Flow After Integration

```
INGESTION:
  chunk text → expand (suffix mode) → enriched text
  enriched text → embed (dense + sparse) → Qdrant vectors
  enriched text → store in chunk_text (MySQL FULLTEXT, Qdrant payload)
  original text → store in chunk_metadata["original_text"] (MySQL, Qdrant payload)

RETRIEVAL (3 legs):
  user query → expand (bidirectional) → expanded query
  expanded query → embed (dense + sparse) → search Qdrant (enriched chunks)
  expanded query → MySQL FULLTEXT search against chunk_text (enriched)
  hits → _qdrant_payload_to_doc → page_content = original_text

RERANKER:
  query = rewritten query (original, NOT expanded)
  passages = original_text + per-passage glossary (temporarily appended)
  cross-encoder scores (query, passage+glossary) pairs
  page_content restored to original_text after scoring

GENERATION:
  retrieved docs (page_content = original_text) → format_context_string
  scan original_text for abbreviations → build compact glossary
  context = chunks (clean prose) + [Abbreviation Glossary] block
  LLM sees: original query + clean chunks + glossary mapping
  LLM connects query terms to chunk abbreviations via the glossary

CITATION DISPLAY:
  cited docs (page_content = original_text) → frontend → cleanChunkText → user
  user sees original document text, no expansion suffixes, no glossary
```

## Files

| File | Role |
|------|------|
| `abbreviations_enhanced.csv` | Source CSV (3,090 rows, 2,193 unique abbreviations) |
| `abbr_expander.py` | Deterministic expander class with ingestion + query expansion |
| `enhance_csv.py` | Script that applies the 7 rules to produce the enhanced CSV |
| `backend/app/services/retrieval/query_expander.py` | Existing dead-code scaffolding to replace |
| `backend/app/services/ingestion/document_processor.py` | Ingestion pipeline — insertion point for chunk expansion |
| `backend/app/services/ingestion/document_qdrant.py` | Embedding + Qdrant upsert — receives enriched chunks |
| `backend/app/services/retrieval/retrieval.py` | 3-leg retrieval — `_qdrant_payload_to_doc` and `_exact_search` need to return `original_text` as `page_content` |
| `backend/app/services/agentic_rag/nodes.py` | LangGraph nodes — insertion point for query expansion |
| `backend/app/services/agentic_rag/agent_graph.py` | `format_context_string`, `_observations_text`, `finalize_node` — consume `page_content` (will be `original_text`) |
| `backend/app/services/agentic_rag/utils.py` | `format_context_string` — renders `page_content` into LLM context |
| `backend/app/models/knowledge.py` | `DocumentChunk` model — `chunk_metadata` JSON for original text |
| `backend/app/models/organisation.py` | `OrgAbbreviation` model — existing per-org table |
| `frontend/src/components/chat/answer.tsx` | Citation display — renders `citation.text` (will be `original_text`) |
| `frontend/src/lib/utils.ts` | `cleanChunkText` — light formatting for citation display |

## Related Docs

- [chunking.md](chunking.md) — chunking strategy and SPLADE token limit rationale
- [search-implementation.md](search-implementation.md) — 3-leg retrieval implementation
- [ingestion-pipeline.md](ingestion-pipeline.md) — full ingestion pipeline
