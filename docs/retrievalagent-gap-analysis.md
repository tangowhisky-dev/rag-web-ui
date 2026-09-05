# Retrievalagent Gap Analysis & Implementation Plan

> **STATUS: SUPERSEDED by the atomic-tools redesign.** This gap analysis drove the transition from the monolithic `rag_retrieve` pipeline to the current composable atomic-tools architecture. The implementation is complete. See `docs/atomic-tools-redesign.md` for the redesign plan and `docs/retrieval-pipeline.html` for the current pipeline diagram. This document is retained for historical reference.

## Comparison: our pipeline vs retrievalagent

### Pipeline topology

**retrievalagent** (`graph.py`):
```
prepare → [keyword_search ∥ synonym_search] → evaluate
  → quality_gate → [semantic_backup →] merge_rerank
  → precision_filter → route(generate|rewrite|give_up)
  → generate → [final_grade →] END|rewrite
```

**our pipeline** (`agent_graph/build.py`):
```
load_context → expand_query → rewrite_query → plan
  → route_plan(clarify_interrupt|think)
  → think → route_think(tool|reflect_final)
  → tool → route_tool(reflect|reflect_final)
  → reflect → think
  → reflect_final → route_reflect_final(think|finalize)
  → finalize → answer_scoring → save_memory → END
```

Inside `rag_retrieve` tool (called by `tool_node`):
```
_run_relaxation_ladder:
  for each level (0,1,2):
    _run_retrieval_pass:
      expand_query (abbreviation only)
      asyncio.gather(dense, sparse, exact)  ← all 3 legs at once
      merge_node (dedup + semantic dedup at 0.95)
      optional metadata sort (BEFORE rerank — bug, see Phase 0)
      reranking_node (cross-encoder, score_threshold=-inf)
      filter_node (reranker threshold)
    _llm_sufficiency_check
    if insufficient: neo4j_expansion → recheck
  if all levels insufficient:
    _rewrite_query (with top-3 snippets + filter suggestion)
    _try_rewrite_retry (re-run ladder with rewritten query)
```

### Confirmed gaps (user's observations validated)

All five observations hold. Details below.

---

## Gap 1: Lean parallel pipeline — semantic backup only when BM25 scores low

### What retrievalagent does

Three sequential stages with a quality gate:
1. **keyword_search** — pure BM25 on the original query. Fast, cheap.
2. **synonym_search** — BM25 on spell-corrected + synonym-expanded terms. Runs in parallel with keyword_search.
3. **quality_gate** — `max(keyword_score, synonym_score) >= 0.7`? If yes, skip semantic. If no, run **semantic_backup** (full hybrid vector search).
4. **merge_rerank** — dedup all pools, MMR diversity, rerank.

The key insight: **vector search is expensive** (embedding + Qdrant query). If BM25 already found high-scoring results, skip it entirely.

### What we do

`_run_retrieval_pass` runs `asyncio.gather(dense, sparse, exact)` — all three legs fire simultaneously every time. No quality gate. No conditional semantic backup.

### Impact

- **Latency**: every retrieval pays the dense-vector cost even when exact/sparse already found the answer.
- **Cost**: unnecessary embedding API calls + Qdrant queries.
- **Quality**: not negatively affected (more legs = more recall), but no efficiency gain from cheap legs.

### Recommendation: **Implement, but adapt to our 3-leg model**

Our model is different — we have SPLADE sparse (not BM25) and MySQL FTS (exact). The adaptation:

1. Run **exact + sparse** first (cheap, no embedding needed).
2. Quality gate: if `max(exact_score, sparse_score) >= threshold` and `doc_count >= 3`, skip dense.
3. Otherwise run **dense** (expensive, embedding-based).
4. Merge + rerank as usual.

This gives us the latency win without changing our retrieval model.

**Files to change:**
- `backend/app/services/agentic_rag/tools/rag_retrieve.py` — `_run_retrieval_pass` → split into `_run_cheap_pass` + `_run_dense_pass` with a `_quality_gate` check between them.
- `backend/app/services/agentic_rag/nodes.py` — `exact_retrieval_node` and `sparse_retrieval_node` need to expose a score signal (currently they return docs but no score).

**New state fields:**
- `exact_score: float` — top ranking score from MySQL FTS
- `sparse_score: float` — top Qdrant sparse score

**Threshold:** `ADAPTIVE_RETRIEVAL_FAST_ACCEPT_SCORE` (new setting, default 0.7).

---

## Gap 2: Synonym + spell-correct as a parallel BM25 term

### What retrievalagent does

`_asynonym_search_node`:
1. One LLM call (`_aresolve_synonyms`) returns `(synonyms, corrected_query, excluded_terms)`.
2. Each synonym + the corrected query becomes a separate BM25 search term.
3. All BM25 searches run in parallel via `asyncio.gather`.
4. Results are RRF-fused.
5. The original query is preserved — synonyms are additive, not replacement.

The prompt (`synonym_expansion`):
- Spell-correct obvious typos.
- Generate up to N synonym/alias terms.
- Extract negated terms ("not X" → excluded_terms).
- Skip synonyms for pure identifiers (SKUs, codes).

### What we do

- `expand_query_node` — abbreviation expansion only. No synonyms, no spell correction.
- `rewrite_query_node` — resolves pronouns/references. Explicitly **forbidden** from adding synonyms (rule 6 in `REWRITE_SYSTEM_PROMPT`).
- `_rewrite_query` (inside rag_retrieve) — only fires after retrieval fails. Rewrites the query but does not generate parallel synonym terms.

### Impact

- **Recall miss**: if the user says "bieröffner" but documents say "Flaschenöffner", our sparse/exact legs won't find it. Dense might, but vector similarity for typos is weak.
- **No spell correction**: typos in the query reduce all three legs' effectiveness.

### Recommendation: **Implement — add a synonym expansion step**

Add a new `_expand_synonyms` function that runs once per retrieval pass (not per leg). It produces:
- `corrected_query` — spell-corrected version (used as an additional search term, not replacement).
- `synonyms` — 2-5 synonym/alias terms.
- `excluded_terms` — negated concepts (see Gap 3).

These become additional search terms in the exact and sparse legs (parallel search + RRF fusion). The original query is always preserved.

**RRF fusion must be built from scratch.** Our codebase has no RRF logic — `merge_node` does exact content-hash dedup + semantic dedup at 0.95 cosine threshold, but no rank-based fusion. Phase 2 must include writing an RRF function and wiring it into `exact_search_docs` and `sparse_search_docs`.

**Files to change:**
- `backend/app/services/agentic_rag/tools/rag_retrieve.py` — add `_expand_synonyms` function, call it in `_run_retrieval_pass` before the legs fire. Pass synonyms as additional queries to `exact_search_docs` and `sparse_search_docs`.
- `backend/app/services/retrieval/retrieval.py` — `exact_search_docs` and `sparse_search_docs` accept `extra_queries: list[str]` and run them in parallel, RRF-fusing results. **A new `_rrf_fuse(list_of_ranked_lists, k=60) -> list` function must be added** — standard RRF formula: `score(doc) = sum(1 / (k + rank(doc)))` across all input lists.
- `backend/app/services/agentic_rag/prompts.py` — add `SYNONYM_EXPANSION_PROMPT`.
- `backend/app/services/agentic_rag/graph_state.py` — add `synonyms`, `corrected_query`, `excluded_terms` to AgentState (see Gap 3 for `excluded_terms` rationale).

**New settings:**
- `SYNONYM_VARIANTS` (default 3, max 5)
- `SYNONYM_CACHE_TTL` (default 300 seconds)

**Caching:** LLM synonym results cached in Redis, key `synonyms:{org_id}:{sha256(query)}`, TTL from `SYNONYM_CACHE_TTL`. LLM client uses `get_org_llm(org_id, db, role="query")` — same role as `rewrite_query_node`. Same pattern as other Redis caches in the codebase.

---

## Gap 3: Negative filter extraction

### What retrievalagent does

Two-layer extraction:
1. **Regex** (`extract_negation_terms` in `retrieval/intent_match.py`) — deterministic, fires in `_aprepare_node` before the synonym LLM. Covers **DE/EN/FR/IT** negation patterns: "not X", "but not X", "without X", "nicht X", "ohne X", "sans X", etc. (13 compiled regexes, one per negation cue.)
2. **LLM** (in `_aresolve_synonyms`) — the synonym prompt asks the LLM to extract negated terms too. Merged with regex results.

Negated terms are post-filtered in `_amerge_rerank_node`: any doc whose content or metadata contains a negated term is dropped.

### What we do

Nothing. No negation extraction anywhere.

### Impact

- User says "show me documents about networking but not Linux" → we return Linux networking docs anyway.
- For our document corpus this is lower-impact than for e-commerce, but still relevant for KB queries with exclusions.

### Recommendation: **Implement — regex-only, port only what exists**

Port only the DE/EN/FR/IT patterns that exist in retrievalagent's `intent_match.py`. Do not invent ES/NL patterns — they don't exist in the source and writing them from scratch is out of scope unless explicitly requested.

The regex extractor is deterministic and adds zero latency. The LLM extractor (from the synonym prompt) is redundant for our use case — our queries are simpler than e-commerce product searches.

### Where to extract: `rewrite_query_node`, not inside the tool

The initial plan called for extraction inside `_run_retrieval_pass` (inside the `rag_retrieve` tool). This is the wrong place. `tool_node` has no mechanism for tools to return state updates beyond `docs` and `confidence` — it extracts only those two from `Observation.result` and ignores everything else. Adding `excluded_terms` extraction inside the tool would require adding custom extraction logic in `tool_node`, creating a one-off pattern that doesn't exist today.

**Decision: extract in `rewrite_query_node`.** This is a graph node that already writes directly to AgentState. The extraction is deterministic regex (zero latency, no LLM call), so it adds no cost to the rewrite node. The tool reads `excluded_terms` from state for post-filtering. `finalize_node` reads it from state for the generation guardrail.

This also means `excluded_terms` is extracted once and applies to all relaxation ladder levels and rewrite retries — the user's exclusion intent doesn't change between levels.

**Files to change:**
- `backend/app/services/agentic_rag/nodes.py` — add `_extract_negation_terms(query) -> list[str]` function (port the 13 DE/EN/FR/IT regexes from retrievalagent's `intent_match.py`). Call it in `rewrite_query_node` and return `excluded_terms` in the state update alongside `rewritten_query` and `resolution_provenance`.
- `backend/app/services/agentic_rag/tools/rag_retrieve.py` — `_run_retrieval_pass` reads `excluded_terms` from state (passed via `ctx.state`) and post-filters merged docs after reranking.
- `backend/app/services/agentic_rag/graph_state.py` — add `excluded_terms: list[str]` to AgentState.
- `backend/app/services/agentic_rag/agent_graph/finalization.py` — inject `excluded_terms` into finalize prompt (see below).

**No new settings** — always enabled, zero cost.

### Should `excluded_terms` be in AgentState?

**Yes.** The field needs to flow from `rewrite_query_node` (where it's extracted) to two consumers:

1. **`rag_retrieve` tool** (retrieval post-filtering): reads from `ctx.state` to drop docs containing excluded terms after merge.
2. **`finalize_node`** (generation guardrail): reads from state to inject a constraint into the finalize prompt.

Without AgentState, there's no path from `rewrite_query_node` to `finalize_node` — they're separated by the entire agent loop (plan → think → tool → reflect → finalize). AgentState is the only mechanism that spans the full graph.

### Should `excluded_terms` be injected into `finalize_node`'s prompt?

**Yes — it's a cheap guardrail against parametric knowledge leakage.**

The primary mechanism is retrieval-level filtering: docs containing excluded terms are dropped before they reach `finalize_node`. By the time the finalize prompt is built, the "Retrieved context (the only citable evidence)" section doesn't contain excluded topics.

But the LLM can still mention excluded topics from its parametric knowledge. If the user says "networking but not Linux" and all Linux docs are filtered out, the LLM might still write "Linux is commonly used in networking" from training data — not as a citation, but as general knowledge. The finalize prompt says "Retrieved context (the only citable evidence)" but "don't cite" ≠ "don't mention."

The injection is 2 lines added to `_build_finalize_prompt`'s user prompt assembly, after the retrieved context and before the final instruction:

```python
if excluded_terms:
    parts.append(f"User excluded topics: {', '.join(excluded_terms)}. Do not discuss these.\n\n")
```

Cost: zero (no extra LLM call, no extra token budget worth worrying about — it's a single line). Benefit: prevents a specific failure mode where the LLM contradicts the user's explicit exclusion. The retrieval-level filter handles 95% of the case; this handles the remaining 5%.

**`_build_finalize_prompt` signature change:** add `excluded_terms: list[str]` parameter. `finalize_node` reads it from state and passes it through.

**Not injected into `think_node`.** The agent already sees the original user query ("networking but not Linux") in the think prompt. Adding `excluded_terms` separately would be redundant — the agent doesn't need to know the regex extraction happened, it just needs to see the user's intent, which is already in the query. The exclusion is enforced at retrieval (doc filtering) and generation (finalize constraint), not at the planning level.

---

## Gap 4: Tool-calling agent — separate search_hybrid, search_bm25, rerank_results

### What retrievalagent does

The LLM agent has 5 tools:
- `get_index_settings` — discover filterable/sortable fields.
- `get_filter_values` — see distinct values for a field.
- `search_hybrid` — BM25 + vector search with filters.
- `search_bm25` — pure keyword fallback.
- `rerank_results` — rerank a list of hits.

The LLM decides dynamically which tools to call and in what order. It can:
- Call `search_bm25` first (cheap), then `rerank_results`.
- Call `get_filter_values` to discover a brand name, then `search_hybrid` with a filter.
- Skip `search_hybrid` entirely if `search_bm25` already found enough.

### What we do

Everything is baked into `rag_retrieve`. The agent calls one tool with `query, kb_ids, filters, sort, legs, top_k, min_confidence`. Internally it runs all legs, reranks, filters, and returns docs. The agent cannot:
- Call rerank separately on already-retrieved docs.
- Choose to run only BM25 (exact) without dense.
- Discover filter values and then search in two separate steps (it can via `kb_metadata` + `rag_retrieve`, but the LLM has to figure out the two-step flow).

### Impact

- **Less flexibility**: the agent can't fine-tune its retrieval strategy.
- **More latency**: `rag_retrieve` always runs the full pipeline even when a cheap search would suffice.
- **Simpler prompts**: our agent prompt is simpler because there's one retrieval tool, not five.

### Recommendation: **Do NOT split rag_retrieve into separate tools**

Reasons:
1. Our `rag_retrieve` already has `legs` parameter — the agent can pass `legs=["exact"]` to run only BM25-style search. This is equivalent to `search_bm25`.
2. Our `rag_retrieve` already has `filters` and `sort` — this is equivalent to `search_hybrid` with `filter_expr`.
3. Our `kb_metadata` tool already provides `get_index_settings` and `get_filter_values` functionality.
4. Splitting would require major changes to the agent prompt, tool registry, frontend display, and tests — for marginal benefit.
5. Our agent loop (think → tool → reflect) already gives the LLM autonomy to decide which tools to call. The LLM can call `kb_metadata` first, then `rag_retrieve` with filters — that's the same two-step flow.

**What we should do instead:** make the `legs` parameter more prominent in the tool description and the THINK_SYSTEM_PROMPT so the agent knows it can run cheap-only searches. This is a prompt change, not a code change.

**Files to change:**
- `backend/app/services/agentic_rag/prompts.py` — update `THINK_SYSTEM_PROMPT` and `PLAN_SYSTEM_PROMPT` to explain the `legs` parameter and when to use `["exact", "sparse"]` vs `["dense", "sparse", "exact"]`.
- `backend/app/services/agentic_rag/tools/rag_retrieve.py` — update the tool description string to highlight the `legs` parameter.

---

## Gap 5: Auto-strategy — sample collection at init and tune parameters

### What retrievalagent does

`_auto_init_filters` (called at agent construction):
1. Samples 15 documents from the backend.
2. Measures max doc char length → adjusts `rerank_chars` (up to 16384).
3. Infers `name_field` and `group_field` from field names (`*_name`, `*category*`, etc.).
4. Discovers filterable attributes from the backend's index config.
5. Samples 200 more documents to build a `{field: [distinct values]}` map.
6. Caches the result keyed by schema signature (hash of field names + filterable attrs).

This runs **once at init**, not per query. The discovered values are used by the filter-intent LLM to decide whether a filter applies.

### What we do

`load_context_node` runs at every chat turn. It loads previous answer, recalls memory, resets loop state. It does **not**:
- Sample the KB to discover field values.
- Adjust any retrieval parameters based on corpus characteristics.
- Cache any schema/corpus metadata.

Our `kb_metadata` tool provides `list_fields`, `unique_values`, `date_range`, `list_documents` — but the agent has to call it explicitly. There's no pre-computed cache.

### Impact

- **First query latency**: the agent has to call `kb_metadata` to discover what's in the KB, adding a round-trip.
- **No parameter tuning**: `rerank_chars`, `top_k`, retrieval thresholds are static regardless of corpus size or document length.

### Recommendation: **Implement — KB profiling at chat open**

Add a `_profile_kb` step that runs once when a chat is opened (or when KBs are linked to a chat). It:
1. Queries the `Document` table for metadata: distinct titles, content types, date range, doc count.
2. Samples 5-10 chunks from Qdrant to measure average chunk length.
3. Adjusts `rerank_chars` if chunks are longer than the default.
4. Caches the profile in Redis keyed by `(org_id, kb_ids_hash)`.

This profile is loaded into state at `load_context_node` and made available to the agent as context (not as a tool call). The agent can see "This KB has 47 documents, titles range from X to Y, content types are PDF/DOCX" without calling `kb_metadata`.

### KB profile heterogeneity: per-KB, not merged

When multiple KBs are linked to a chat, their schemas may differ (e.g., one KB of PDFs with titles, one KB of code files with no titles). Averaging chunk lengths or merging field lists across heterogeneous KBs produces useless recommendations.

**Decision: cache per-KB, merge at read time.**

- Redis key: `kb_profile:{org_id}:{kb_id}` (one key per KB, not per KB-set).
- `load_context_node` reads each KB's profile individually and merges them into a single `kb_profile` dict in AgentState.
- The merge is a simple union: `fields = union of all KB fields`, `content_types = union of all KB content_types`, `avg_chunk_length = weighted average by doc count`.
- Field availability is tracked per-KB: `{"title_contains": [kb_id_1, kb_id_3], "file_name_contains": [kb_id_1, kb_id_2]}`. This way `extract_intent` (folded into `rewrite_query_node`, see Phase 6) knows which fields are available and only suggests filters for fields that exist in all queried KBs. If a field exists in only some KBs, the suggestion is still valid — `rag_retrieve` applies filters via MySQL `doc_ids` resolution, which naturally scopes to KBs that have the field.

**Files to change:**
- `backend/app/services/agentic_rag/agent_graph/load_context.py` — add `_load_kb_profile` call that reads from Redis cache (or computes + caches if missing).
- `backend/app/services/agentic_rag/kb_profile.py` (new file) — `profile_kb(org_id, kb_id, db, redis) -> dict` function. Called per-KB.
- `backend/app/services/agentic_rag/graph_state.py` — add `kb_profile: dict` to AgentState.
- `backend/app/services/agentic_rag/agent_graph/planning.py` — include `kb_profile` summary in the plan prompt.
- `backend/app/services/agentic_rag/agent_graph/thinking.py` — include `kb_profile` summary in the think prompt.

**New settings:**
- `KB_PROFILE_CACHE_TTL` (default 3600 seconds)

### kb_metadata fallback when profile cache is cold

If the KB profile cache is cold (first chat open, Redis flushed, new KB linked), `load_context_node` computes the profile synchronously — it's a fast MySQL query (doc count, distinct titles, content types, date range) plus a small Qdrant sample (5-10 points). This adds ~50-100ms to the first turn.

`extract_intent` (folded into `rewrite_query_node`, see Phase 6) reads `kb_profile` from state. If `kb_profile` is empty (e.g., profiling failed, KB has no documents), intent extraction simply skips filter suggestions — it falls back to "no filters suggested" and the agent can still call `kb_metadata` explicitly if needed. There is no automatic fallback to calling `kb_metadata` from inside `rewrite_query_node` — that would add an LLM-tool-call round-trip inside a node that's supposed to be a single LLM call. The agent loop already handles this: if the plan/think LLM realizes it needs metadata, it calls `kb_metadata` as a tool.

---

## Gap 6: Answer evaluation → suggestion + retry_strategy for user

### What retrievalagent does

`_afinal_grade` (post-generation LLM check):
- Returns `AnswerGrade` with `sufficient`, `confidence`, `reason`, `suggestion`, `retry_strategy` (widen/narrow/pinpoint), `pinpoint_code`, `memory_worth_storing`, `memory_fact`.
- If `sufficient=false` and budget remains → rewrite → retry retrieval → regenerate.
- The `suggestion` and `retry_strategy` guide the rewrite.

### What we do

`answer_scoring_node` calls `evaluate_answer` which returns `AnswerEvaluation` with `faithfulness`, `completeness`, `confidence_match`, `flags`. No `suggestion`, no `retry_strategy`. The score is informational only — it's saved to the DB and displayed in the UI but never feeds back into the loop.

### Recommendation: **Implement — ephemeral follow-up suggestions, no DB persistence**

This is a user-generated retry: the agent suggests follow-up queries and the user clicks one to retry.

**Why no DB persistence:**

The three fields serve different purposes, and none require persistence:

1. **`follow_up_queries: list[str]`** — suggested follow-up questions (e.g., "What are the prerequisites for Chapter 3?", "How does this compare to TCP/IP?"). These are **ephemeral UI affordances** — they appear below the answer as clickable chips, the user clicks one, it's sent as the next user message, and the cycle continues. There is no reason to display them again when the chat is reloaded. The user has either clicked one (in which case it became a real message) or moved on. Persisting them adds a DB column, a migration, and a schema field for zero functional benefit.

2. **`suggestion: str`** — a one-line reasoning about the answer quality (e.g., "The answer covers the main concepts but may be missing edge cases from Chapter 4."). This is **contextual to the moment of generation** — it explains why the follow-up queries were suggested. Showing it on reload without the follow-up queries would be confusing. It belongs in the `ConfidenceCollapsible` section, shown only during the current streaming session.

3. **`retry_strategy: str`** — a label for the suggestion row ("Try a broader search:" / "Try a narrower search:" / "Look up this exact ID:"). This is **pure UI chrome** — it prefixes the follow-up chips. It has no meaning without the chips. No persistence needed.

**All three fields are already ephemeral in the existing pipeline.** The `last_answer` SSE event already carries `last_answer_object` (which includes `followups: list[str]`) to the frontend during streaming. The frontend stores it in the message's `lastAnswerObject` prop. On chat reload, `lastAnswerObject` is fetched from the DB (`Message.last_answer_object` column) — but the followups inside it are never rendered. We're not adding new persistence; we're rendering data that already flows to the frontend.

### Single implementation location: extend `LastAnswerObject`, not `AnswerEvaluation`

**The `followups` field already exists in `LastAnswerObject`** (`schemas.py` line 70) and is already extracted by `LAST_ANSWER_EXTRACT_PROMPT` (`prompts.py` line 660). The SSE `last_answer` event already carries it to the frontend. The frontend already receives it in `message.lastAnswerObject`. But **it's never rendered** — `onFollowUp` is only wired to `SelectionActions` (text selection toolbar), not to suggestion chips.

**Decision: implement follow-up suggestions entirely through the existing `LastAnswerObject.followups` path. Do not touch `AnswerEvaluation`.**

- `LAST_ANSWER_EXTRACT_PROMPT` already asks for `followups` — keep it there. This is the single source of truth.
- `suggestion` and `retry_strategy` are new fields on `LastAnswerObject` — add them to the schema and the extraction prompt. They ride the same SSE event, the same `lastAnswerObject` prop, the same ephemeral lifecycle.
- `AnswerEvaluation` stays as-is — it's for scoring (faithfulness/completeness), not for user-facing suggestions. Mixing the two concerns would create the duplication the user flagged.

**Backend changes:**
- `backend/app/services/agentic_rag/schemas.py` — extend `LastAnswerObject` with `suggestion: str = ""` and `retry_strategy: str = ""`.
- `backend/app/services/agentic_rag/prompts.py` — extend `LAST_ANSWER_EXTRACT_PROMPT` to ask for `suggestion` and `retry_strategy` alongside the existing `followups`.

**Current `LAST_ANSWER_EXTRACT_PROMPT`** (`prompts.py` lines 652-667):
```
Extract a structured summary from the assistant answer below. Return valid JSON only matching this schema:
{{
  "summary": "2-3 sentences",
  "key_points": ["..."],
  "data": [{{"label": "...", "value": 123, "unit": "...", "context": "..."}}],
  "citations": [{{"document_id": 1, "chunk_index": 0}}],
  "chart_option": null or {{ ... }},
  "followups": ["..."]
}}

If the answer contains no numbers, set data to []. If no chart, set chart_option to null. Keep key_points to at most 8 bullets.

Answer:
{answer}
```

**Updated `LAST_ANSWER_EXTRACT_PROMPT`:**
```
Extract a structured summary from the assistant answer below. Return valid JSON only matching this schema:
{{
  "summary": "2-3 sentences",
  "key_points": ["..."],
  "data": [{{"label": "...", "value": 123, "unit": "...", "context": "..."}}],
  "citations": [{{"document_id": 1, "chunk_index": 0}}],
  "chart_option": null or {{ ... }},
  "followups": ["..."],
  "suggestion": "one-line assessment of answer completeness, or empty string",
  "retry_strategy": "widen|narrow|pinpoint|"
}}

If the answer contains no numbers, set data to []. If no chart, set chart_option to null. Keep key_points to at most 8 bullets.

For followups: generate 1-3 specific follow-up questions the user might ask next based on the answer. Each should be a self-contained question. Empty list if the answer is definitive.

For suggestion: one sentence assessing whether the answer fully addresses the query, and what might be missing. Empty string if the answer is complete.

For retry_strategy: "widen" if the answer is too narrow and a broader search would help, "narrow" if the answer is too broad and the user should search more specifically, "pinpoint" if the user should look up an exact identifier, or empty string if no retry is needed.

Answer:
{answer}
```

- No changes to `AnswerEvaluation`, `evaluator.py`, `answer_scoring_node`, or `graph_state.py`.
- No changes to `AnswerEvaluation`, `evaluator.py`, `answer_scoring_node`, or `graph_state.py`.
- No changes to `Message` model, no Alembic migration, no `MessageResponse` schema changes. The data rides inside `last_answer_object` which is already a JSON column.

**Frontend changes:**
- `frontend/src/components/chat/answer.tsx` — render `lastAnswerObject.followups` as clickable `Suggestion` chips using the existing `Suggestions`/`Suggestion` components from `ai-elements/suggestion.tsx` (the same components used on the search home page). Place them **below** the bottom bar (copy/export buttons + confidence collapsible) — after every other component. Use the same visual style as the search home page suggestions: `Suggestions` wrapper (horizontal scroll) + `Suggestion` chips (rounded-full, outline variant).
- Wire `Suggestion.onClick` to `onFollowUp` (already passed as prop, already calls `handleSubmit(query)` which sends it as the next user message — same flow as manual typing).
- Show `retry_strategy` as a label prefix above the chips: "Try a broader search:" / "Try a narrower search:" / "Look up this exact ID:". Use the same `text-xs text-muted-foreground` style as the search home page's "Suggested searches" label.
- Show `suggestion` text inside the existing `ConfidenceCollapsible` section (it already has a `suggestion` prop — wire it to `lastAnswerObject.suggestion` instead of the current `suggestion` prop which comes from a different source).
- Only render when `!isStreaming` and `followups.length > 0`.

**No new settings** — no feature flags, always on.

---

## Additional differences found (not in user's list)

### Diff A: MMR diversity before reranking

**retrievalagent:** `_mmr_diverse(unique, lam=0.7, top_k=pool_size)` runs BEFORE reranking to remove near-duplicate docs from the candidate pool. Uses bag-of-words Jaccard similarity on `doc.page_content[:400]`.

**our pipeline:** `merge_node` does exact dedup (content_hash) + semantic dedup (0.95 threshold). No MMR diversity pass. The reranker sees all merged docs including near-duplicates.

**Recommendation: Skip.** Our semantic dedup at 0.95 already handles near-duplicates. MMR adds complexity for marginal gain in our document corpus (we're not ranking 1000 product variants).

### Diff B: Precision filter (result-aware second-pass filter detection)

**retrievalagent:** `_aprecision_filter_node` runs after merge_rerank. It samples the actual returned docs' metadata, asks the LLM "does any of these field values map to a constraint in the question?", and if so, re-runs retrieval with that filter.

**our pipeline:** No equivalent. The agent can call `kb_metadata` + `rag_retrieve` with filters, but this is a two-step manual process, not an automatic second pass.

**Recommendation: Skip for now.** Our `kb_metadata` tool + agent loop already provides this capability. The precision filter is a retrievalagent-specific optimization for e-commerce where filter fields are unknown at query time. Our KBs have known schemas.

### Diff C: Tier-0 ID fast-track

**retrievalagent:** `_aprepare_node` checks if the query is a single identifier (SKU, article number). If so, it does a direct lookup and skips the entire retrieval pipeline.

**our pipeline:** No equivalent. A query like "document 42" goes through the full dense+sparse+exact pipeline.

**Recommendation: Skip.** Our `kb_metadata` tool with `list_documents` action + `rag_retrieve` with `filters: {"document_ids": [42]}` already handles this. The agent can figure out the ID lookup from the plan.

### Diff D: Long-term search memory (cross-conversation)

**retrievalagent:** Stores "search facts" (synonym mappings, alias learnings) in Mem0/LangGraph store. On future queries, recalls these as BM25 hints.

**our pipeline:** Redis-based conversation memory (last 3 turns). No cross-conversation search fact storage.

**Recommendation: Skip.** This is a nice-to-have for high-volume search deployments. Our use case is document Q&A, not repeated product searches. The complexity of maintaining a search fact store is not justified.

### Diff E: Custom instructions per collection

**retrievalagent:** `custom_instructions` field appended to retrieval prompts (preprocess, filter-intent, grader). Set globally or per collection.

**our pipeline:** No equivalent. All prompts are static.

**Recommendation: Skip for now.** Can be added later as an org-level setting if domain-specific tuning is needed.

---

## Implementation plan (sorted by impact/effort ratio)

### Phase 0: Fix sort/rerank ordering bug (prerequisite)
**Effort:** Trivial. **Impact:** High (fixes broken sort behavior). **Risk:** Zero.

This is a current production bug, not a new feature. The sort step is applied after merge but before reranking, so the reranker overrides the user's explicit sort order. A query with `sort={"field":"created_at","direction":"desc"}` gets reordered by relevance, destroying the chronological order the user requested.

**Current (buggy) flow in `_run_retrieval_pass`:**
```
merge_node → sort → reranking_node → filter_node
```
Sort is applied at lines 498-500, then reranking at line 501 overrides it.

**Fixed flow:**
```
merge_node → reranking_node → filter_node → sort (if provided)
```

1. Move the sort step from before `reranking_node` to after `filter_node`.
2. The reranker scores all docs (quality gate — always runs). The filter removes irrelevant docs. Then sort re-orders the surviving docs.
3. User's explicit sort wins over reranker relevance order. If no sort, reranker score order is preserved (current behavior).

**Skip condition:** if `legs == ["exact"]` and `sort` is provided, skip `reranking_node` + `filter_node` entirely — just sort the merged docs. Exact FTS matches are already high-precision; the quality gate is safely skippable here.

**Files:**
- `backend/app/services/agentic_rag/tools/rag_retrieve.py` — `_run_retrieval_pass`: move sort after filter, add exact-only skip condition.

**No new settings.**

**Tests:**
- `backend/tests/test_sort_rerank_order.py` (new):
  - Test 1: query with `sort={"field":"created_at","direction":"desc"}` → verify returned docs are in descending `created_at` order (not reranker score order).
  - Test 2: query with no sort → verify returned docs are in reranker score order (current behavior preserved).
  - Test 3: query with `sort` and `legs=["exact"]` → verify reranking_node and filter_node are skipped, docs are in sort order.
  - Test 4: query with `sort` where all docs are below reranker threshold → verify 0 docs returned (quality gate worked), sort is not applied to empty set.

### Phase 1: Negative filter extraction (Gap 3)
**Effort:** Small. **Impact:** Medium. **Risk:** Zero (deterministic, no LLM).

1. Port `extract_negation_terms` regex from retrievalagent's `retrieval/intent_match.py` — **only the 13 DE/EN/FR/IT patterns that exist**. Do not invent ES/NL patterns.
2. Add `_extract_negation_terms(query) -> list[str]` to `nodes.py`.
3. Call it in `rewrite_query_node` — return `excluded_terms` in the state update alongside `rewritten_query` and `resolution_provenance`.
4. Add `excluded_terms: list[str]` to `AgentState`.
5. In `_run_retrieval_pass` (inside `rag_retrieve` tool), read `excluded_terms` from `ctx.state` and post-filter merged docs after reranking — drop docs whose `page_content` or `title` contains any excluded term.
6. In `_build_finalize_prompt`, add `excluded_terms: list[str]` parameter. Inject after retrieved context: `"User excluded topics: {excluded_terms}. Do not discuss these."` `finalize_node` reads from state and passes through.
7. Unit test: "networking but not Linux" → excluded_terms=["Linux"], docs containing "Linux" are dropped, finalize prompt contains exclusion constraint.

**Files:**
- `backend/app/services/agentic_rag/nodes.py` (add `_extract_negation_terms`, call in `rewrite_query_node`)
- `backend/app/services/agentic_rag/tools/rag_retrieve.py` (read `excluded_terms` from state, post-filter)
- `backend/app/services/agentic_rag/graph_state.py` (add `excluded_terms` field)
- `backend/app/services/agentic_rag/agent_graph/finalization.py` (extend `_build_finalize_prompt` signature, inject constraint)
- `backend/tests/test_negation_filter.py` (new)

### Phase 2: Synonym + spell-correct expansion (Gap 2)
**Effort:** Medium-High (RRF fusion must be built from scratch). **Impact:** High (recall improvement). **Risk:** Low (additive, original query preserved).

1. Add `SYNONYM_EXPANSION_PROMPT` to `prompts.py` — asks LLM for corrected_query, synonyms (2-5), and negated terms.
2. Add `_expand_synonyms(query, ctx) -> (corrected, synonyms, excluded)` to `rag_retrieve.py` — cached in Redis. Uses the same `query` LLM role as `rewrite_query_node` (via `get_org_llm(ctx.org_id, ctx.db, role="query")`). This is the right role because synonym expansion is a query-understanding task, not a generation or retrieval task — it uses the same cheap/fast model that pronoun resolution uses.
3. **Write `_rrf_fuse(list_of_ranked_lists, k=60) -> list`** — standard Reciprocal Rank Fusion. This is new code, not a reuse of existing logic. Our `merge_node` does dedup, not rank fusion. Formula: `score(doc) = sum(1 / (k + rank(doc)))` across all input lists, then sort by fused score.
4. Modify `exact_search_docs` and `sparse_search_docs` in `retrieval.py` to accept `extra_queries: list[str]` — run one search per query in parallel and RRF-fuse results.
5. In `_run_retrieval_pass`, call `_expand_synonyms` once, pass synonyms as `extra_queries` to exact and sparse legs only. Dense leg gets the original query only (dense embeddings already handle semantic similarity — synonym fan-out doesn't help and wastes embedding API calls).
6. Merge negated terms from `_expand_synonyms` with regex-extracted terms from Phase 1.
7. Unit test: "bieröffner" → corrected="Flaschenöffner", synonyms=["Flaschenöffner"], both terms searched, RRF-fused.

**Files:**
- `backend/app/services/agentic_rag/prompts.py`
- `backend/app/services/agentic_rag/tools/rag_retrieve.py`
- `backend/app/services/retrieval/retrieval.py` (add `_rrf_fuse`, modify `exact_search_docs`/`sparse_search_docs`)
- `backend/app/services/agentic_rag/graph_state.py`
- `backend/tests/test_synonym_expansion.py` (new)
- `backend/tests/test_rrf_fusion.py` (new — test RRF in isolation)

**New settings:**
- `SYNONYM_VARIANTS` (default 3)
- `SYNONYM_CACHE_TTL` (default 300 seconds)

**Cache spec:** Redis key `synonyms:{org_id}:{sha256(query)}`, value is JSON `{"corrected": str|null, "synonyms": list[str], "excluded": list[str]}`, TTL from `SYNONYM_CACHE_TTL`. LLM client uses `get_org_llm(org_id, db, role="query")` — same role as `rewrite_query_node`.

### Phase 3: Lean parallel pipeline — conditional dense leg (Gap 1)
**Effort:** Medium. **Impact:** High (latency reduction). **Risk:** Medium (could miss docs if threshold too high).

1. Split `_run_retrieval_pass` into two stages:
   - Stage 1: run `exact + sparse` legs in parallel. Collect scores.
   - Stage 2: `_quality_gate(max(exact_score, sparse_score))` — if `>= threshold` and `doc_count >= 3`, skip dense. Otherwise run dense.
2. Add `exact_score` and `sparse_score` to the return values of `exact_retrieval_node` and `sparse_retrieval_node`.
3. Merge all available docs, rerank, filter as usual.
4. The relaxation ladder still applies — at level 0, the gate might skip dense; at level 2 (max relaxation), always run all legs.

**Per-level leg selection:** `_run_relaxation_ladder` currently passes the same `legs` list to every level. The quality gate runs inside `_run_retrieval_pass` and decides whether to include dense for that specific level. At level 0, if the gate says "skip dense", `_run_retrieval_pass` runs only exact+sparse. At level 2 (max relaxation), the ladder overrides the gate and forces all legs — this is a parameter passed to `_run_retrieval_pass` (e.g., `force_all_legs=True` for the final relaxation level). This way the gate is per-level, not global.

5. Unit test: query with high exact score → dense skipped. Query with low scores → dense runs.

**Files:**
- `backend/app/services/agentic_rag/tools/rag_retrieve.py`
- `backend/app/services/agentic_rag/nodes.py` (exact/sparse nodes return scores)
- `backend/app/services/agentic_rag/graph_state.py`
- `backend/tests/test_quality_gate.py` (new)

**New settings:**
- `ADAPTIVE_RETRIEVAL_FAST_ACCEPT_SCORE` (default 0.7)

### Phase 4: KB profiling at chat open (Gap 5)
**Effort:** Medium. **Impact:** Medium (faster first query, better agent context). **Risk:** Low (cached, non-blocking).

1. Create `backend/app/services/agentic_rag/kb_profile.py` with `profile_kb(org_id, kb_id, db, redis) -> dict` — called **per-KB**, not per KB-set.
2. Profile includes: doc count, distinct titles (top 20), content types, date range, avg chunk length, rerank_chars adjustment, field availability per KB.
3. Cache in Redis with TTL (default 1 hour), keyed by `kb_profile:{org_id}:{kb_id}` (one key per KB).
4. In `load_context_node`, call `_load_kb_profile` for each linked KB — read from cache or compute synchronously (fast MySQL query + small Qdrant sample). Merge per-KB profiles into a single `kb_profile` dict in AgentState (union of fields, weighted average of chunk lengths, field availability tracked per-KB).
5. Include profile summary in plan and think prompts.
6. If profile is empty (KB has no docs, profiling failed), skip — the agent can call `kb_metadata` explicitly.
7. E2E test: open chat with linked KB → profile is computed and cached → second chat open uses cache.

**Files:**
- `backend/app/services/agentic_rag/kb_profile.py` (new)
- `backend/app/services/agentic_rag/agent_graph/load_context.py`
- `backend/app/services/agentic_rag/graph_state.py`
- `backend/app/services/agentic_rag/agent_graph/planning.py`
- `backend/app/services/agentic_rag/agent_graph/thinking.py`
- `backend/tests/test_kb_profile.py` (new)

**New settings:**
- `KB_PROFILE_CACHE_TTL` (default 3600)

### Phase 5: Follow-up suggestions (Gap 6)
**Effort:** Small (backend) + Small (frontend). **Impact:** High (user-facing improvement). **Risk:** Zero (rendering existing data, no new pipeline).

This phase is smaller than it looks because the infrastructure already exists:
- `LastAnswerObject.followups` is already in the schema (`schemas.py` line 70).
- `LAST_ANSWER_EXTRACT_PROMPT` already asks for `followups` (`prompts.py` line 660).
- The `last_answer` SSE event already carries `lastAnswerObject` to the frontend.
- The frontend already stores it in `message.lastAnswerObject`.
- `onFollowUp` is already wired to `handleSubmit(query)` which sends it as the next user message.
- The `Suggestions`/`Suggestion` components already exist (`ai-elements/suggestion.tsx`).

The only work is:
1. **Backend:** extend `LastAnswerObject` with `suggestion: str = ""` and `retry_strategy: str = ""`. Extend `LAST_ANSWER_EXTRACT_PROMPT` to ask for these two fields alongside the existing `followups`. No changes to `AnswerEvaluation` — follow-ups live in `LastAnswerObject`, scoring lives in `AnswerEvaluation`. No duplication.
2. **Frontend:** render `lastAnswerObject.followups` as `Suggestion` chips inside a `Suggestions` wrapper, placed **below** the bottom bar (after copy/export buttons and confidence collapsible). Use the same style as the search home page: `Suggestions` (horizontal scroll) + `Suggestion` chips (rounded-full, outline variant). Show `retry_strategy` as a label prefix above the chips. Show `suggestion` in the `ConfidenceCollapsible`. Wire `onClick` to `onFollowUp`.
3. Only render when `!isStreaming` and `followups.length > 0`.

**Files:**
- `backend/app/services/agentic_rag/schemas.py` (add `suggestion`, `retry_strategy` to `LastAnswerObject`)
- `backend/app/services/agentic_rag/prompts.py` (extend `LAST_ANSWER_EXTRACT_PROMPT`)
- `frontend/src/components/chat/answer.tsx` (render followups as Suggestion chips below bottom bar)
- `backend/tests/test_followup_suggestions.py` (new — verify extraction prompt produces followups)

**No new settings. No DB migration. No new API endpoints.**

### Phase 6: Query intent extraction — fold into rewrite_query_node (Gap 6)
**Effort:** Medium. **Impact:** High (this is what makes filters/sort actually get used). **Risk:** Low (suggestions are non-binding).

#### The problem

Currently there is no intent extraction step. The flow is:

```
User: "what all is included in latest weekly update"
  → expand_query_node: abbreviation expansion only
  → rewrite_query_node: resolves pronouns, forbidden from adding metadata terms
  → plan_node: creates Plan{intent="rag", subtasks=[{tool_hint:"rag_retrieve"}]}
  → think_node: LLM sees rag_retrieve schema (filters/sort fields auto-rendered)
              → LLM calls rag_retrieve(query="latest weekly update")
              → NO filters, NO sort — nothing told it to use them
```

The think_node LLM discovers `filters` and `sort` fields exist from the auto-rendered schema, but neither `PLAN_SYSTEM_PROMPT` nor `THINK_SYSTEM_PROMPT` mentions them. The LLM has to infer from field descriptions alone that "latest weekly update" implies `filters={"title_contains":"Weekly Update"}` and `sort={"field":"created_at","direction":"desc"}`. This is unreliable.

#### Why NOT a separate node

A separate `extract_intent_node` adds one LLM call to **every** query before planning. That's +200-500ms per turn for unconditional latency. The doc previously proposed this — it's the wrong approach.

The `rewrite_query_node` already makes one LLM call per turn (when the query needs pronoun/reference resolution). It already has access to conversation history, the previous answer object, and the expanded query. Folding intent extraction into this existing call adds zero latency — the LLM does both jobs in one response.

#### What changes in rewrite_query_node

`rewrite_query_node` currently:
- Takes the expanded query + history.
- Resolves pronouns/references.
- Returns `rewritten_query` + `resolution_provenance`.

After this phase, it also:
- Takes the KB profile (from Phase 4) as additional context.
- Extracts suggested filters, sort, and legs from the query.
- Returns `rewritten_query` + `resolution_provenance` + `query_intent`.

The `REWRITE_SYSTEM_PROMPT` is extended with a second section:

```
# Query Intent Extraction

If a [KB Profile] section is provided, also extract search intent:
1. Suggest filters ONLY when the query clearly implies a metadata constraint:
   - "latest weekly update" → filters={"title_contains":"Weekly Update"}, sort={"field":"created_at","direction":"desc"}
   - "PDF documents about networking" → filters={"content_type":"application/pdf"}
   - "documents from June" → filters={"created_after":"2026-06-01","created_before":"2026-06-30"}
2. Suggest sort ONLY when the query implies ordering (latest, newest, oldest, most recent).
3. Suggest legs=["exact","sparse"] for literal lookups (filenames, IDs, exact titles). Use null for conceptual queries.
4. If no filters/sort/legs are implied, return null for all.
5. Do NOT invent field names — use only the fields listed in [KB Profile].

Output the rewritten query on the first line, then a JSON object on the second line:
{query}
{"suggested_filters": {...}|null, "suggested_sort": {...}|null, "suggested_legs": [...]|null, "reasoning": "..."}
```

#### Why this works despite the "no synonyms" rule

The existing rule 6 in `REWRITE_SYSTEM_PROMPT` says "Do NOT introduce new entities, concepts, or relationships." This applies to the **rewritten query string** — the query itself stays clean. Intent extraction is a separate output (the JSON object on the second line) that suggests **filters and sort**, not query terms. The rewritten query is still just a pronoun-resolved version of the original. The two outputs are independent.

#### What if rewrite_query_node skips the LLM call?

`rewrite_query_node` has a fast path: if the query is already self-contained (no pronouns, no references), it returns the query as-is without an LLM call (provenance reason `self_contained`). In this case, intent extraction is also skipped — `query_intent` is null. The agent falls back to inferring filters from the schema, same as today.

This is acceptable because:
- Self-contained queries that imply filters ("latest weekly update") are a minority.
- When it does happen, the agent can still infer from the schema — it's just less reliable, same as today.
- Adding an LLM call just for intent extraction on self-contained queries would reintroduce the unconditional latency we're trying to avoid.

If this proves to be a problem in practice, the fallback is to make the self-contained check also consider whether the query contains metadata-related keywords ("latest", "newest", "PDF", "document titled", date references). If it does, force the LLM call even for self-contained queries. This is a tuning decision, not an architecture decision.

#### Error handling for malformed output

The LLM may not produce the expected two-line format (query + JSON). It may:
- Return only the query line (no JSON second line).
- Return malformed JSON (truncated, extra text, wrong keys).
- Return the JSON inline with the query instead of on a separate line.
- Wrap the output in markdown fences.

**Retry strategy: one retry with corrective instruction, then default.**

1. **Parse attempt 1:** Split on newline. First line is the rewritten query. Try `json.loads(second_line)`. If it succeeds and has valid `QueryIntent` shape, done.
2. **If parse fails:** Retry the LLM call once with an appended corrective instruction: "Your previous response was malformed. Output the rewritten query on the first line, then a valid JSON object on the second line. Do not wrap in markdown fences. Do not add any text after the JSON."
3. **If retry also fails:** Set `query_intent = None` and proceed with just the rewritten query. Log a warning. The agent falls back to inferring filters from the schema — same as today, same as the self-contained fast path. This is the sensible default: a malformed intent extraction should not block the retrieval pipeline.

The retry adds one extra LLM call only when the first call produces malformed output — which should be rare with a capable model. The cost is ~200-500ms in the rare case, zero in the common case.

**Edge case — query line itself is malformed:** If the first line is empty or the entire output is JSON (no query line), treat the entire output as the rewritten query (pass through unchanged) and skip intent extraction. The rewrite node's existing provenance-rejection logic already handles "the LLM produced something weird" — this extends it to the intent extraction case.

#### QueryIntent schema

```python
class QueryIntent(BaseModel):
    """Suggested filters/sort extracted from the query."""
    suggested_filters: Optional[dict] = Field(
        default=None,
        description="Metadata filters for rag_retrieve. Same schema: title_contains, file_name_contains, content_type, created_after, created_before, document_ids.",
    )
    suggested_sort: Optional[dict] = Field(
        default=None,
        description='Sort for rag_retrieve. Example: {"field":"created_at","direction":"desc"}.',
    )
    suggested_legs: Optional[list[str]] = Field(
        default=None,
        description='Suggested legs: ["exact","sparse"] for literal lookups, null for default.',
    )
    reasoning: str = Field(default="", description="Why these suggestions were made.")
```

#### Graph topology: no change

```
Current:  load_context → expand_query → rewrite_query → plan → ...
New:      load_context → expand_query → rewrite_query → plan → ...
```

Same topology. `rewrite_query_node` just returns more state. No new node, no new edge.

#### Where the suggestions go

1. `query_intent` is stored in `AgentState`.
2. `plan_node` includes `query_intent` suggestions in the plan user prompt: "Suggested filters: {...}. Suggested sort: {...}."
3. `think_node` includes `query_intent` suggestions in the think user prompt: "Suggested filters: {...}. Suggested sort: {...}. Use these when calling rag_retrieve unless you have a reason not to."
4. The think_node LLM still decides whether to use them — suggestions are non-binding.

#### Interaction with other phases

- **Phase 4 (KB profiling):** the KB profile provides the available fields and sample values that `rewrite_query_node` shows to the LLM. Without Phase 4, intent extraction has no field list and can't suggest filters. If `kb_profile` is empty, intent extraction is skipped.
- **Phase 2 (synonyms):** synonym expansion happens inside rag_retrieve, after the agent decides to call it. Intent extraction happens before the agent decides. They're independent.
- **Phase 3 (conditional dense):** `suggested_legs` from intent extraction feeds directly into the `legs` parameter. If intent extraction says `["exact","sparse"]`, the conditional dense leg is naturally skipped.

#### Files to change

- `backend/app/services/agentic_rag/schemas.py` — add `QueryIntent` model.
- `backend/app/services/agentic_rag/prompts.py` — extend `REWRITE_SYSTEM_PROMPT` with intent extraction section.
- `backend/app/services/agentic_rag/nodes.py` — `rewrite_query_node` parses the JSON second line, returns `query_intent` in state.
- `backend/app/services/agentic_rag/graph_state.py` — add `query_intent` to AgentState.
- `backend/app/services/agentic_rag/agent_graph/planning.py` — include `query_intent` suggestions in plan user prompt.
- `backend/app/services/agentic_rag/agent_graph/thinking.py` — include `query_intent` suggestions in think user prompt.
- `backend/tests/test_query_intent.py` (new) — test intent extraction for recency, title, content_type, and no-filter cases.

**No new settings.** No feature flag. No `INTENT_EXTRACTION_CACHE_TTL` — the intent extraction is part of the rewrite LLM call, not a separate call, so it's cached with the rewrite result (if rewrite caching exists).

---

### Phase 7: Prompt updates — legs parameter awareness (Gap 4)
**Effort:** Small. **Impact:** Low-Medium. **Risk:** Zero (prompt-only change).

1. Update `THINK_SYSTEM_PROMPT` to explain when to use `legs=["exact", "sparse"]` (fast, cheap) vs `legs=["dense", "sparse", "exact"]` (thorough, expensive).
2. Update `PLAN_SYSTEM_PROMPT` similarly.
3. Update `rag_retrieve` tool description string.
4. No code changes, no tests needed (prompt changes are validated by existing tests).

**Files:**
- `backend/app/services/agentic_rag/prompts.py`
- `backend/app/services/agentic_rag/tools/rag_retrieve.py`

---

## Implementation order (dependency-safe)

1. **Phase 0** (sort/rerank bug fix) — no dependencies, ship first
2. **Phase 1** (negation extraction) — no dependencies
3. **Phase 4** (KB profiling) — no dependencies, needed by Phase 6
4. **Phase 6** (query intent extraction, folded into rewrite_query_node) — depends on Phase 4 for KB profile
5. **Phase 2** (synonym expansion) — builds on Phase 1 (merges negated terms), requires new RRF fusion code
6. **Phase 3** (conditional dense leg) — independent, benefits from Phases 2 + 6
7. **Phase 7** (prompt updates) — independent, do anytime
8. **Phase 5** (follow-up suggestions) — independent, do last because it touches frontend

## What is NOT in scope

- Phase 5 from original plan (answer-grade retry loop / automatic retry) — still skipped per user request
- MMR diversity pass — our semantic dedup at 0.95 is sufficient
- Precision filter (result-aware second-pass filter detection) — kb_metadata + agent loop covers this
- Tier-0 ID fast-track — kb_metadata + rag_retrieve with document_ids covers this
- Long-term cross-conversation search memory — not justified for document Q&A
- HyDE — query rewriting handles vague queries
- Splitting rag_retrieve into separate search_hybrid/search_bm25/rerank_results tools — legs parameter + kb_metadata already provide this flexibility
- Custom instructions per collection — can be added later if needed
- ES/NL negation patterns — don't exist in retrievalagent, not writing from scratch
- Feature flags / `*_ENABLED` settings — we're in development, no need for gradual rollout
- DB persistence for `suggestion` / `retry_strategy` / `follow_up_queries` — ephemeral, rides existing `last_answer_object` JSON column

## Verification

For each phase:
1. Write unit tests first (failing).
2. Implement the change.
3. Run unit tests — must pass.
4. Run relevant backend test suite — must not regress.
5. Run frontend suite (Phase 5 only) — must pass.
6. Run frontend build with `NODE_ENV=production` (Phase 5 only) — must succeed.
7. Check cyclomatic complexity — refactor any function above 15.
8. E2E test against live MySQL/Qdrant environment.

### Integration test plan

Cross-phase E2E scenarios that exercise multiple features together:

**Scenario 1: Negation + intent extraction + conditional dense**
- Query: "latest weekly update but not Linux"
- Expected:
  - `rewrite_query_node` does both in one LLM call:
    - Rewrites query (pronoun resolution if needed)
    - Extracts `query_intent`: `suggested_filters={"title_contains":"Weekly Update"}`, `suggested_sort={"field":"created_at","direction":"desc"}`, `suggested_legs=null`
    - Extracts `excluded_terms=["Linux"]` via `_extract_negation_terms` regex (runs alongside the LLM call, not inside it)
  - Agent calls `rag_retrieve` with filters and sort (from `query_intent` suggestions in think prompt)
  - `_run_retrieval_pass` reads `excluded_terms` from `ctx.state`, runs exact + sparse first (with `title_contains` filter applied via `doc_ids` resolution)
  - Quality gate: if exact/sparse scores high enough, dense is skipped
  - After merge + rerank + filter, docs containing "Linux" are dropped (post-filter using `excluded_terms` from state)
  - `finalize_node` reads `excluded_terms` from state, injects "User excluded topics: Linux. Do not discuss these." into finalize prompt
  - Answer does not mention Linux

**Scenario 2: Synonym expansion + RRF fusion**
- Query: "bieröffner" (typo of "Flaschenöffner")
- Expected:
  - `_expand_synonyms` returns `corrected="Flaschenöffner"`, `synonyms=["Flaschenöffner"]`
  - `exact_search_docs` runs two parallel searches: "bieröffner" and "Flaschenöffner", RRF-fuses results
  - `sparse_search_docs` runs two parallel searches, RRF-fuses results
  - `dense_search_docs` runs once with "bieröffner" only (no synonym fan-out for dense)
  - Merged results include docs that contain "Flaschenöffner" but not "bieröffner"
  - Answer correctly discusses bottle openers

**Scenario 3: KB profile + intent extraction + sort preserved**
- Setup: KB with 47 PDF documents, titles like "Weekly Update 2026-W01", "Weekly Update 2026-W02", etc.
- Query: "what's in the newest weekly update"
- Expected:
  - `load_context_node` loads KB profile from Redis (or computes + caches): 47 docs, content_type=PDF, date range, field availability
  - `rewrite_query_node` extracts `query_intent`: `suggested_filters={"title_contains":"Weekly Update"}`, `suggested_sort={"field":"created_at","direction":"desc"}`
  - Agent calls `rag_retrieve` with filters and sort
  - `_run_retrieval_pass`: merge → rerank → filter → **sort (after filter, not before)**
  - Top result is the most recent weekly update
  - Sort order is preserved (not overridden by reranker)

**Scenario 4: Follow-up suggestions rendering**
- Query: "explain the OSI model"
- Expected:
  - `finalize_node` generates answer, then `_build_last_answer_object` extracts `followups=["What's the difference between TCP and UDP?", "How does the transport layer work?"]`, `suggestion="Covers all 7 layers but transport layer examples are thin"`, `retry_strategy="widen"`
  - `last_answer` SSE event carries `lastAnswerObject` with followups to frontend
  - Frontend renders Suggestion chips below the bottom bar: "Try a broader search:" label + two chips
  - `ConfidenceCollapsible` shows the suggestion text
  - User clicks "What's the difference between TCP and UDP?" → `onFollowUp` → `handleSubmit("What's the difference between TCP and UDP?")` → sent as next user message
  - Next turn processes it normally

**Scenario 5: Self-contained query skips intent extraction**
- Query: "what is a mutex" (no pronouns, no metadata keywords)
- Expected:
  - `rewrite_query_node` fast path: query is self-contained, no LLM call
  - `query_intent` is null
  - Agent calls `rag_retrieve` with no filters, no sort, default legs
  - Works exactly as today — no regression

**Scenario 6: Heterogeneous KBs**
- Setup: Chat linked to KB-A (PDFs with titles) and KB-B (code files, no titles)
- Query: "find the authentication module"
- Expected:
  - `load_context_node` loads both KB profiles, merges: `fields={"title_contains":[KB-A], "file_name_contains":[KB-A, KB-B], "content_type":[KB-A, KB-B]}`
  - `rewrite_query_node` sees merged profile, suggests `filters={"file_name_contains":"auth"}` (file_name_contains is available in both KBs, title_contains is not)
  - Agent calls `rag_retrieve` with filter
  - `doc_ids` resolution scopes to docs in both KBs that match the filter

---

## Agent awareness of new functionality

### How the agent discovers tools today

Two channels, both must stay in sync:

1. **Automatic (schema-driven):** `_tool_descriptions_text(tools)` in `agent_graph/observations.py` iterates `applicable_tools(ctx)`, reads each tool's `args_schema.model_json_schema()`, and renders `name + description + field list` into the think prompt. When we add a field to `RagRetrieveInput` (e.g. `filters`, `sort`), the schema automatically appears in the think prompt. **No manual prompt edit needed for schema changes.**

2. **Manual (prompt-driven):** `PLAN_SYSTEM_PROMPT` and `THINK_SYSTEM_PROMPT` in `prompts.py` contain hand-written tool lists and usage rules. These do NOT auto-update. When we add a new tool or change the semantics of an existing tool, these prompts must be updated manually.

### What needs prompt changes per phase

| Phase | New tool? | Schema change? | Prompt change needed? |
|-------|-----------|----------------|----------------------|
| 0 (sort bug) | No | No | No — internal fix |
| 1 (negation) | No | No | No — extraction runs in `rewrite_query_node` (regex, no LLM). `finalize_node` prompt gets `excluded_terms` injected as a 2-line constraint. |
| 2 (synonyms) | No | No | Update `THINK_SYSTEM_PROMPT` rag_retrieve query rules: remove "Do NOT add synonyms" rule since rag_retrieve now does synonym expansion internally. The agent should still not add synonyms manually (the tool does it), but the rule's rationale changes. |
| 3 (conditional dense) | No | No (legs already exists) | **Yes** — update `THINK_SYSTEM_PROMPT` and `PLAN_SYSTEM_PROMPT` to explain when to use `legs=["exact","sparse"]` (fast, cheap, no embedding) vs `legs=["dense","sparse","exact"]` (thorough). Add guidance: "For exact-match queries (filenames, IDs, titles), use legs=['exact','sparse']. For conceptual queries, use all legs." |
| 4 (KB profiling) | No | No | No — profile is injected as context text into plan/think prompts, not as a tool. The agent sees it passively. |
| 5 (follow-up suggestions) | No | No | No — `LAST_ANSWER_EXTRACT_PROMPT` changes but that's not an agent-facing prompt. |
| 6 (query intent extraction) | No (folded into rewrite) | Yes — new `QueryIntent` model | **Yes** — `REWRITE_SYSTEM_PROMPT` extended with intent extraction section. Plan and think prompts include intent suggestions. |
| 7 (prompt updates) | No | No | **Yes** — this IS the prompt change. |

### Specific prompt edits

**`THINK_SYSTEM_PROMPT` changes (Phases 2 + 3 + 7):**

Current rule:
```
rag_retrieve query rules:
- Reuse the rewritten query verbatim as the "query" argument. Do NOT add synonyms, related terms, or extra keywords beyond what the user or the rewriter already provided.
```

Updated rule:
```
rag_retrieve query rules:
- Reuse the rewritten query verbatim as the "query" argument. rag_retrieve now performs synonym expansion and spell correction internally — do NOT add synonyms yourself.
- rag_retrieve accepts a "legs" parameter: ["dense","sparse","exact"] (default, thorough) or ["exact","sparse"] (fast, no embedding, for exact-match queries like filenames/IDs/titles). Use the shorter set when the query is a literal lookup; use all legs for conceptual questions.
- When the query implies recency or ordering ("latest", "most recent", "newest"), pass sort={"field":"created_at","direction":"desc"}.
- When the query names a specific document title or filename pattern, pass filters={"title_contains":"..."} or {"file_name_contains":"..."}.
- When intent extraction (from rewrite_query) provided suggested filters/sort, use them unless you have a reason not to.
```

**`PLAN_SYSTEM_PROMPT` changes (Phase 3 + 7):**

Add to the "Available tools" section:
```
- rag_retrieve: search the knowledge base. Pass legs=["exact","sparse"] for literal lookups (faster), or legs=["dense","sparse","exact"] for conceptual queries. Pass filters for metadata constraints (title_contains, file_name_contains, content_type, created_after, created_before, document_ids). Pass sort={"field":"created_at","direction":"desc"} for recency queries. When intent extraction provided suggested filters/sort, include them in the plan.
```

**`REWRITE_SYSTEM_PROMPT` changes (Phase 6):**

Add after the existing rules:
```
# Query Intent Extraction

If a [KB Profile] section is provided, also extract search intent:
1. Suggest filters ONLY when the query clearly implies a metadata constraint:
   - "latest weekly update" → filters={"title_contains":"Weekly Update"}, sort={"field":"created_at","direction":"desc"}
   - "PDF documents about networking" → filters={"content_type":"application/pdf"}
   - "documents from June" → filters={"created_after":"2026-06-01","created_before":"2026-06-30"}
2. Suggest sort ONLY when the query implies ordering (latest, newest, oldest, most recent).
3. Suggest legs=["exact","sparse"] for literal lookups (filenames, IDs, exact titles). Use null for conceptual queries.
4. If no filters/sort/legs are implied, return null for all.
5. Do NOT invent field names — use only the fields listed in [KB Profile].

Output the rewritten query on the first line, then a JSON object on the second line:
{query}
{"suggested_filters": {...}|null, "suggested_sort": {...}|null, "suggested_legs": [...]|null, "reasoning": "..."}
```

**`rag_retrieve` tool description (Phase 3 + 7):**

Update the `description` field on `_RagRetrieveTool`:
```python
description: str = (
    "Search the attached knowledge bases. Returns ranked document chunks, "
    "confidence, and sufficiency. Use when the user needs facts from documents. "
    "Pass legs=['exact','sparse'] for literal lookups (filenames, IDs, titles) — "
    "faster, no embedding. Pass legs=['dense','sparse','exact'] (default) for "
    "conceptual queries. Pass filters for metadata constraints and sort for "
    "recency/ordering queries."
)
```

---

## Reranker: separate tool vs built-in

### Why retrievalagent separates rerank_results

retrievalagent exposes `rerank_results` as a separate tool for three reasons:

1. **Cost control:** the LLM can skip reranking when `search_bm25` already returned 1-3 high-confidence exact matches. A Cohere rerank call costs money and adds ~200ms latency. For a query like "SKU 12345" that returns 1 doc, reranking is pointless.

2. **Rerank with a different query:** the LLM can search with keywords ("Flaschenöffner Bier") but rerank with the full natural-language question ("What's the best bottle opener for beer?"). This decouples search terms from relevance scoring.

3. **Rerank merged results from multiple searches:** the LLM can call `search_bm25` twice with different terms, then `rerank_results` on the combined set.

### When reranker doesn't help

1. **Single-result exact matches:** if exact search returns 1 doc, reranking 1 doc is a no-op with latency cost.

2. **Multi-chunk synthesis queries:** "Summarize the key points across all weekly updates from Q3" — no single chunk answers this. The reranker ranks individual chunks by relevance to the query, but the answer requires combining chunks. Reranking doesn't hurt (it still surfaces the most relevant chunks first), but it doesn't solve the synthesis problem. The answer generation step handles synthesis, not the reranker.

3. **Already-sorted results:** when `sort={"field":"created_at","direction":"desc"}` is applied, the user explicitly wants chronological order. Reranking by relevance overrides that order. This is the bug fixed in Phase 0.

4. **Very small result sets (≤3 docs):** reranking 3 docs adds latency without meaningful reordering.

### Our pipeline's reranker behavior

Our `reranking_node` always runs with `score_threshold=-inf` (scores everything, filters later). There's no skip logic. Every `rag_retrieve` call pays the cross-encoder cost even for 1-doc results.

### Recommendation: **Split reranker into score+filter and ordering, NOT a separate tool**

Making rerank a separate tool would require: new tool class, registry entry, frontend icon, prompt updates, and the agent would need to learn a two-step search-then-rerank pattern. The complexity isn't justified.

The reranker does two jobs that should be separated:

1. **Quality gate (always run):** score every doc, filter out docs below `RERANKER_SCORE_THRESHOLD`. This removes irrelevant docs regardless of sort preference.
2. **Reordering (conditional):** reorder by reranker score OR preserve user's explicit sort.

This is exactly what Phase 0 fixes. See Phase 0 above for the implementation.

### Skip conditions for the scoring step

Scoring is cheap when the result set is small, but the cross-encoder adds ~50-200ms per call. Skip scoring only when it's provably useless:

1. **0 docs:** skip (nothing to score).
2. **1 doc:** score it (needed for the quality gate — is this 1 doc actually relevant?).
3. **Sort by metadata field + all docs from exact leg only:** if the user passed `legs=["exact"]` with a sort, they want exact matches in sorted order. The exact leg already has high precision. Skip scoring, just sort. This is the only case where skipping the quality gate is safe.

No other skip conditions. The quality gate should run in all other cases — removing it risks returning irrelevant docs.

---

## Synonym expansion: parallel queries vs appended to main query

### Why retrievalagent uses parallel BM25

retrievalagent runs each synonym as a **separate BM25 search**, then RRF-fuses the results. The original query is always one of the search arms. This is better than appending synonyms to the main query for three reasons:

1. **BM25 term-frequency dilution:** if you append "Flaschenöffner bieröffner" as one query, BM25 scores docs that match ANY term, but the IDF weighting changes. A doc matching "bieröffner" (rare) gets boosted over a doc matching "Flaschenöffner" (common), even though "Flaschenöffner" is the correct term. Parallel searches preserve each term's IDF context.

2. **RRF robustness:** each synonym search produces its own ranking. RRF combines rankings by position, not score. A doc ranked #1 in the "Flaschenöffner" search and #5 in the "bieröffner" search gets a better fused rank than a doc ranked #3 in both. This is more robust than a single search where score calibration across terms is unreliable.

3. **Spell correction safety:** the corrected query is a separate search arm, not a replacement. If the correction is wrong (e.g. "bieröffner" corrected to "Bieröffner" which doesn't exist), the original query's results are still in the pool. Appending would mix the wrong correction into the main query and degrade all results.

### What this means for our pipeline

Our three legs have different relationships with synonyms:

| Leg | Benefit from synonyms? | How |
|-----|----------------------|-----|
| **Exact (MySQL FTS)** | **Yes, high** | FTS is pure keyword matching. Synonyms as parallel queries + RRF fusion is exactly the right approach. |
| **Sparse (SPLADE)** | **Partial** | SPLADE already learns term expansions internally. But SPLADE's learned expansions are corpus-wide, not query-specific. A synonym like "bieröffner → Flaschenöffner" that SPLADE hasn't seen won't be expanded. Parallel sparse queries with synonyms add query-specific expansion that SPLADE misses. |
| **Dense (vector)** | **No** | The embedding captures semantic similarity. "bieröffner" and "Flaschenöffner" have similar embeddings regardless of exact spelling. Adding synonyms to dense search doesn't help — the vector already handles it. |

### Recommendation: **Parallel queries for exact + sparse, NOT for dense. NOT appended to main query.**

Implementation:
1. `_expand_synonyms(query)` returns `(corrected_query, synonyms, excluded_terms)`.
2. Build `search_terms = [query] + ([corrected_query] if corrected else []) + synonyms` — original query always first, always preserved.
3. For **exact** leg: run `exact_search_docs(term)` for each term in parallel, RRF-fuse results.
4. For **sparse** leg: same — run `sparse_search_docs(term)` for each term in parallel, RRF-fuse.
5. For **dense** leg: run `dense_search_docs(query)` once with the original query only. No synonym fan-out.
6. Merge all leg results as usual.

This gives us the recall benefit of synonyms on the keyword legs without paying the embedding cost on the dense leg (which doesn't need it).

**RRF fusion:** must be built from scratch — see Phase 2. The `_rrf_fuse` function takes a list of ranked doc lists and returns a single fused list. Standard formula: `score(doc) = sum(1 / (k + rank(doc)))` across all input lists, where `k=60` (standard constant). Sort by fused score descending.

**Files to change (refines Phase 2):**
- `backend/app/services/retrieval/retrieval.py` — `exact_search_docs` and `sparse_search_docs` accept `extra_queries: list[str]`. When provided, run one search per query in parallel and RRF-fuse. When not provided (backward compat), run a single search as today.
- `backend/app/services/agentic_rag/tools/rag_retrieve.py` — `_run_retrieval_pass` calls `_expand_synonyms` once, builds `search_terms`, passes them as `extra_queries` to exact and sparse legs only. Dense leg gets the original query only.

**No change to dense search** — it doesn't benefit from synonyms.
