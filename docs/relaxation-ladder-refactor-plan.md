# Relaxation Ladder Refactor: Embed-Once, Filter-in-Memory

## Problem

The current `rag_retrieve` implementation re-runs the entire retrieval pipeline (embedding, Qdrant, MySQL FTS, cross-encoder reranker, Neo4j graph expansion) at each of 3 relaxation levels. The query text is identical across levels — only the score thresholds change. This wastes 2/3 of all remote calls.

### Evidence from e2e logs (9 rag_retrieve calls, 11 user turns)

| Operation | Calls observed | Expected (with fix) |
|-----------|---------------|---------------------|
| Dense embedding API | 28 | 9 |
| SPLADE embedding | 28 | 9 |
| Qdrant dense query | 28 | 9 |
| Qdrant sparse query | 28 | 9 |
| MySQL FTS query | 28 | 9 |
| Cross-encoder rerank | 28 | 9 (+N for graph docs) |
| Neo4j graph expansion | 15 | 9 |
| **Total remote calls** | **183** | **~63** |

### Root cause

`_run_retrieval_pass()` in `rag_retrieve.py` is called in a loop, once per relaxation level. Each call re-embeds the same query, re-queries Qdrant/FTS, re-runs the cross-encoder on a growing pool, and re-runs graph expansion. Only the `min_score` and `rerank_threshold` differ between levels — but those are post-fetch filters, not search parameters.

## Architecture: Current vs Proposed

### Current flow (per rag_retrieve call, 3 levels)

```
for each level (0, 1, 2):
    dense_retrieval_node  → embed query → Qdrant query → filter by min_cosine
    sparse_retrieval_node → SPLADE embed → Qdrant query → filter by min_score
    exact_retrieval_node  → MySQL FTS query → filter by min_score
    merge_node            → dedup
    reranking_node        → cross-encoder score ALL merged docs (threshold=-inf)
    filter_node           → filter by reranker threshold
    if insufficient:
        neo4j_expansion   → graph query → append unscored docs
```

### Proposed flow (per rag_retrieve call)

```
1. FETCH: embed once → query Qdrant/FTS once with min_score=0.0 (no leg-level filtering)
2. MERGE: dedup all candidates
3. RERANK: cross-encoder score ALL candidates once (threshold=-inf)
4. GRAPH: if graph_expand, run neo4j expansion once on the full candidate set
          → score new graph docs with cross-encoder → append to scored pool
5. FILTER LOOP: for each threshold level (tight → loose):
     filter the scored pool in memory by reranker threshold
     compute confidence on the filtered subset
     if sufficient → break
```

## Query Rewording: How It Interacts

Query rewording happens at **two separate levels**, and this refactor only affects one of them.

### Level 1: Query rewrite node (before the agent loop)

`rewrite_query_node` in `nodes.py` runs once per user message. It resolves pronouns and references ("How does **it** handle mispredictions?" → "How does **branch prediction** handle mispredictions?"). This happens before the agent loop starts and is completely outside `rag_retrieve`. **The refactor does not touch this.**

### Level 2: Agent loop retry with reworded queries

This is the RISC/CISC pattern observed in e2e logs:

```
User: "What is the difference between RISC and CISC architectures?"
  ↓ rewrite_query_node (passes through, no rewrite needed)
  ↓ plan_node
  ↓ think_node → calls rag_retrieve("What is the difference between RISC and CISC architectures?")
  ↓ rag_retrieve: 3 levels, all return 0 docs, sufficient=False
  ↓ tool_node → plan satisfied, routes to reflect_final
  ↓ reflect_final_node: ready=False "Retrieval returned 0 documents; another query may help."
  ↓ route_reflect_final: not ready → back to "think"
  ↓ think_node: sees reflection feedback + tried_queries, generates new query
  ↓ think_node → calls rag_retrieve("RISC vs CISC architecture differences characteristics advantages")
  ↓ rag_retrieve: 3 levels, all return 0 docs, sufficient=False
  ↓ reflect_final_node: ready=False "Retrieval returned 0 documents; another query may help."
  ↓ route_reflect_final: not ready → back to "think"
  ↓ think_node → calls rag_retrieve("RISC vs CISC architecture comparison instruction set complexity")
  ↓ rag_retrieve: 3 levels, level 2 returns 21 docs
  ↓ reflect_final_node: ready=True
  ↓ finalize
```

The rewording is done by the **think_node** (an LLM call) based on:
1. `reflection_text` — the verification feedback ("Retrieval returned 0 documents; another query may help")
2. `tried_queries_text` — "Already tried (do NOT resubmit these exact strings): [...]"
3. The plan and observations context

This is the agent graph loop: `think → tool → reflect_final → think → tool → ...`. Each iteration is a **separate rag_retrieve call** with a **different query string**.

### How the refactor handles this

The refactor changes what happens **inside a single rag_retrieve call**. It does NOT change the agent loop. The flow becomes:

```
think_node → calls rag_retrieve(query_v1)
  ↓ rag_retrieve: embed once, query Qdrant/FTS once, rerank once, graph expand once
  ↓ filter at threshold level 0 (-2.0) → 0 docs
  ↓ filter at threshold level 1 (-5.0) → 0 docs
  ↓ filter at threshold level 2 (-8.0) → 0 docs
  ↓ return sufficient=False
↓ reflect_final: ready=False "another query may help"
↓ think_node → calls rag_retrieve(query_v2)  ← different query, new rag_retrieve call
  ↓ rag_retrieve: embed once, query Qdrant/FTS once, rerank once, graph expand once
  ↓ filter at threshold level 0 → 0 docs
  ↓ filter at threshold level 1 → 0 docs
  ↓ filter at threshold level 2 → 21 docs
  ↓ return sufficient=True
↓ reflect_final: ready=True
↓ finalize
```

Each rag_retrieve call still embeds once and queries once. When the agent retries with a reworded query, that's a new rag_retrieve call with a new query string — so it embeds the new query once and queries once. The rewording mechanism is untouched.

### Tried-query dedup

The think_node already tracks previously tried queries via `_tried_rag_retrieve_queries(observations)` and tells the LLM "do NOT resubmit these exact strings." This prevents the agent from re-calling rag_retrieve with the same query. The refactor doesn't affect this — it's in `agent_graph.py`, not `rag_retrieve.py`.

### Threshold rescue behavior

The current 3-level ladder sometimes "rescues" a query by loosening thresholds. In the new design, if the tight threshold returns 0 docs but the loose threshold would have returned some, we still get those docs — we just filter in memory instead of re-querying. The docs are the same because the query is the same. So the rescue behavior is preserved.

## Detailed Edit List

### File 1: `backend/app/services/agentic_rag/tools/rag_retrieve.py`

This is the only file with significant logic changes.

#### Remove `_relaxation_levels()` (lines 82-106)

No longer needed. Thresholds are just reranker filter values applied in memory, not per-level search parameters.

#### Remove `_run_retrieval_pass()` (lines 109-150)

Replaced by the new fetch-once flow inside `_rag_retrieve`.

#### Rewrite `_rag_retrieve()` (lines 153-245)

New structure:

```python
async def _rag_retrieve(ctx: ToolContext, input_obj: RagRetrieveInput) -> dict:
    t0 = time.monotonic()
    rbac = enforce_rbac(ctx, kb_ids=input_obj.kb_ids)
    kb_ids = rbac["kb_ids"]
    if not kb_ids and ctx.state is not None:
        kb_ids = ctx.state.get("kb_ids", [])
    org_id = ctx.org_id
    file_markdown = None
    if ctx.state is not None:
        file_markdown = ctx.state.get("file_markdown", None)

    legs = input_obj.legs or ["dense", "sparse", "exact"]
    from app.services.settings_service import get_setting
    min_confidence = (
        input_obj.min_confidence
        if input_obj.min_confidence is not None
        else get_setting(ctx.db, "ADAPTIVE_RETRIEVAL_THRESHOLD", ctx.org_id) / 100.0
    )

    adaptive_enabled = get_setting(ctx.db, "ADAPTIVE_RETRIEVAL_ENABLED", ctx.org_id)
    rerank_thresholds = _rerank_thresholds(ctx.db, ctx.org_id, adaptive_enabled)

    # 1. Fetch all candidates once (min_score=0.0 → no leg-level filtering)
    state: dict[str, Any] = {
        "rewritten_query": input_obj.query,
        "original_query": input_obj.query,
        "kb_ids": kb_ids,
        "org_id": org_id,
        "file_markdown": file_markdown,
    }
    coros = []
    if "dense" in legs:
        coros.append(dense_retrieval_node(state, ctx.db, kb_ids, org_id, file_markdown, min_score=0.0))
    if "sparse" in legs:
        coros.append(sparse_retrieval_node(state, ctx.db, kb_ids, org_id, file_markdown, min_score=0.0))
    if "exact" in legs:
        coros.append(exact_retrieval_node(state, ctx.db, kb_ids, org_id, file_markdown, min_score=0.0))
    leg_results = await asyncio.gather(*coros, return_exceptions=True)
    for r in leg_results:
        if isinstance(r, Exception):
            logger.warning("[rag_retrieve] leg failed: %s", r)
        else:
            state.update(r)

    # 2. Merge and rerank all candidates once
    state.update(merge_node(state, file_markdown))
    state.update(reranking_node(state))

    # 3. Graph expansion once (if enabled)
    if input_obj.graph_expand:
        try:
            neo4j = await neo4j_expansion_node(state, ctx.db, kb_ids, org_id, file_markdown)
            state.update(neo4j)
            # Score any new graph docs that don't have _reranker_score
            new_graph_docs = [
                d for d in state.get("retrieved_docs", [])
                if "_reranker_score" not in d.get("metadata", {})
            ]
            if new_graph_docs:
                _score_additional_docs(state, input_obj.query, new_graph_docs)
        except Exception as exc:
            logger.warning("[rag_retrieve] graph expansion failed: %s", exc)

    # 4. In-memory filter loop — no re-embedding, no re-querying
    all_scored = state.get("all_scored_docs", [])
    docs: list = []
    confidence = 0.0
    levels_tried = 0
    for i, threshold in enumerate(rerank_thresholds):
        levels_tried = i + 1
        docs = _filter_scored_docs(all_scored, threshold)
        confidence = _compute_confidence(docs)
        if _is_sufficient(docs, confidence, min_confidence):
            break
        logger.info(
            "[rag_retrieve] level %d insufficient (docs=%d confidence=%.2f) — %s",
            i, len(docs), confidence,
            "trying next relaxation level" if i < len(rerank_thresholds) - 1 else "no more levels, giving up",
        )

    # 5. Build result (same shape as before)
    confidence_level = "low"
    if confidence > 0.7:
        confidence_level = "high"
    elif confidence > 0.3:
        confidence_level = "medium"

    latency_ms = round((time.monotonic() - t0) * 1000)
    result_summary = {
        "doc_count": len(docs),
        "confidence": confidence,
        "confidence_level": confidence_level,
        "levels_tried": levels_tried,
    }
    write_audit(ctx, "rag_retrieve", input_obj.model_dump(), result_summary,
                tokens_in=0, tokens_out=0, status="ok", latency_ms=latency_ms)

    return {
        "ok": True,
        "result": {
            "docs": docs,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "query_used": input_obj.query,
            "legs_run": legs,
            "levels_tried": levels_tried,
            "sufficient": _is_sufficient(docs, confidence, min_confidence),
        },
        "error": None,
        "tokens": len(str(docs)) // 4,
    }
```

#### Add helper: `_rerank_thresholds()`

```python
def _rerank_thresholds(db, org_id, adaptive_enabled) -> list[float]:
    """Return the reranker filter thresholds for each relaxation level.

    Level 0 uses RERANKER_SCORE_THRESHOLD (tight).
    Level 1 uses ADAPTIVE_RETRIEVAL_RERANKER_THRESHOLD (medium).
    Level 2 uses RETRIEVAL_RELAX_LEVEL2_RERANKER_THRESHOLD (loose).
    When adaptive is disabled, only level 0 is returned.
    """
    from app.services.settings_service import get_setting
    base = get_def("RERANKER_SCORE_THRESHOLD").default
    if not adaptive_enabled:
        return [base]
    level1 = get_setting(db, "ADAPTIVE_RETRIEVAL_RERANKER_THRESHOLD", org_id)
    level2 = get_setting(db, "RETRIEVAL_RELAX_LEVEL2_RERANKER_THRESHOLD", org_id)
    return [base, level1, level2]
```

#### Add helper: `_filter_scored_docs()`

Extracted from `filter_node` logic — pure in-memory filtering:

```python
def _filter_scored_docs(all_scored_docs: list[dict], threshold: float) -> list[dict]:
    """Filter scored docs by reranker threshold (in-memory, no re-scoring)."""
    filtered = [
        d for d in all_scored_docs
        if d.get("metadata", {}).get("_reranker_score", -float("inf")) >= threshold
    ]
    filtered.sort(
        key=lambda d: d.get("metadata", {}).get("_reranker_score", -float("inf")),
        reverse=True,
    )
    logger.info("[FILTER] threshold=%.2f | input=%d | passed=%d", threshold, len(all_scored_docs), len(filtered))
    return filtered
```

#### Add helper: `_compute_confidence()`

Extracted from `reranking_node` confidence logic:

```python
def _compute_confidence(docs: list[dict]) -> float:
    """Compute retrieval confidence from filtered docs."""
    if not docs:
        return 0.0
    conf_result = score_retrieval(docs, {})
    return conf_result.score / 100.0 if conf_result else 0.0
```

#### Add helper: `_score_additional_docs()`

For graph expansion docs that come back without `_reranker_score`:

```python
def _score_additional_docs(state: dict, query: str, new_docs: list[dict]) -> None:
    """Score graph-expanded docs with the cross-encoder and merge into all_scored_docs."""
    from langchain_core.documents import Document as LangchainDocument
    from app.services.retrieval import rerank
    from app.services.infrastructure.utils import _serialise_doc

    lc_docs = [
        LangchainDocument(page_content=d.get("page_content", ""), metadata=d.get("metadata", {}))
        for d in new_docs
    ]
    scored = rerank(query=query, docs=lc_docs, score_threshold=float("-inf"))
    serialised = [_serialise_doc(d) for d in scored]
    state["all_scored_docs"] = state.get("all_scored_docs", []) + serialised
```

#### Update imports

Add:
```python
from app.core.settings_registry import get_def
from app.services.retrieval import score_retrieval
```

Remove (no longer called directly):
```python
filter_node,  # replaced by _filter_scored_docs in-memory
```

Keep (still called once each):
```python
dense_retrieval_node, exact_retrieval_node, merge_node, neo4j_expansion_node, reranking_node, sparse_retrieval_node
```

### File 2: `backend/app/services/agentic_rag/nodes.py`

#### Remove `adaptive_reranking_node` (lines 371-403)

Dead code — never called from anywhere. Confirmed via grep.

#### Keep `filter_node` (lines 338-364)

Kept but no longer called from `rag_retrieve.py`. It becomes dead code. Can be removed in a later cleanup. Keeping it minimizes the diff and avoids breaking any imports we might have missed.

#### No changes to retrieval nodes

`dense_retrieval_node`, `sparse_retrieval_node`, `exact_retrieval_node` — unchanged. They already accept `min_score=0.0` which means "no filtering". They'll be called once per rag_retrieve instead of 3×.

#### No changes to `merge_node`, `reranking_node`, `neo4j_expansion_node`

Still called once each per rag_retrieve.

### File 3: `backend/app/services/retrieval/retrieval.py`

**No changes.** `_dense_search`, `_sparse_search`, `_exact_search` already accept `min_score=0.0` (no filtering). They'll be called once with `min_score=0.0` instead of 3× with varying thresholds.

### File 4: `backend/app/services/retrieval/reranker.py`

**No changes.** `rerank()` already accepts `score_threshold=-inf` and scores all docs. It will be called once per rag_retrieve instead of 3×.

### File 5: `backend/tests/test_rag_retrieve_ladder.py`

#### Rework `_patch_pipeline` and `_run_rag_retrieve` (lines 30-110)

The current test helpers mock each node as per-level callables with a `call_index`. The new flow calls each node once, not per-level. The filter loop is now in `_rag_retrieve` itself (in-memory), so tests need to provide pre-scored docs and let the in-memory filter work.

New test structure:

```python
def _patch_pipeline(scored_docs, graph_docs=None):
    """Mock nodes to return pre-scored docs.

    scored_docs: list of dicts with _reranker_score in metadata.
    graph_docs: list of dicts WITHOUT _reranker_score (will be scored by mock).
    """
    async def fake_dense(state, db, kb_ids, org_id, file_markdown, min_score=None):
        return {}
    async def fake_sparse(state, db, kb_ids, org_id, file_markdown, min_score=None):
        return {}
    async def fake_exact(state, db, kb_ids, org_id, file_markdown, min_score=None):
        return {}
    def fake_merge(state, file_markdown):
        return {}
    def fake_rerank(state):
        return {
            "all_scored_docs": scored_docs,
            "retrieved_docs": scored_docs,
            "retrieval_confidence": _confidence_from_docs(scored_docs),
        }
    async def fake_neo4j(state, db, kb_ids, org_id, file_markdown):
        if graph_docs is None:
            return {"graph_docs": [], "retrieved_docs": scored_docs, "graph_expansion_done": True}
        # Simulate graph docs being scored (the real code calls _score_additional_docs)
        all_docs = scored_docs + graph_docs
        return {
            "graph_docs": graph_docs,
            "retrieved_docs": all_docs,
            "graph_expansion_done": True,
            "all_scored_docs": all_docs,  # _score_additional_docs would update this
        }
    return {
        "dense_retrieval_node": fake_dense,
        "sparse_retrieval_node": fake_sparse,
        "exact_retrieval_node": fake_exact,
        "merge_node": fake_merge,
        "reranking_node": fake_rerank,
        "neo4j_expansion_node": fake_neo4j,
    }
```

Note: `filter_node` is no longer in the patch dict because `_rag_retrieve` no longer calls it — filtering is in-memory via `_filter_scored_docs`.

The `_score_additional_docs` call in `_rag_retrieve` needs to be mocked too, or we need to make `fake_neo4j` return docs that already have `_reranker_score` so `_score_additional_docs` is a no-op. The cleanest approach: patch `_score_additional_docs` to be a no-op that just updates `all_scored_docs`:

```python
# In _run_rag_retrieve, add to the patch.multiple:
"_score_additional_docs": lambda state, query, new_docs: state.update(
    {"all_scored_docs": state.get("all_scored_docs", []) + new_docs}
),
```

Or simpler: make `fake_neo4j` return graph docs that already have `_reranker_score` in metadata, so the `new_graph_docs` list comprehension in `_rag_retrieve` finds nothing to score.

#### Update test cases (lines 113-157)

**`test_ladder_stops_at_level_0_when_sufficient`**:
- Provide 5 docs with `_reranker_score=5.0` (above level-0 threshold of -2.0).
- Assert `levels_tried == 1`, `sufficient == True`, `len(docs) == 5`.

**`test_ladder_escalates_through_all_levels_when_never_sufficient`**:
- Provide docs with `_reranker_score=-10.0` (below all thresholds).
- Assert `levels_tried == 3`, `sufficient == False`.

**`test_ladder_uses_graph_expansion_only_when_insufficient`**:
- Provide 0 scored docs from rerank, graph expansion adds 4 docs with `_reranker_score=5.0`.
- Assert `levels_tried == 1`, `sufficient == True`, `len(docs) == 4`.

**`test_ladder_disabled_skips_relaxation_levels`**:
- `adaptive_enabled=False`, docs with `_reranker_score=-10.0`.
- Only level-0 threshold (-2.0) applied → 0 docs pass.
- Assert `levels_tried == 1`, `sufficient == False`.

**`test_min_confidence_defaults_from_adaptive_threshold_setting`** — unchanged (tests input schema only).

#### Wall-clock routing tests (lines 176-198) — unchanged

These test `agent_graph.py` routing, not `rag_retrieve`.

## What Does NOT Change

- `retrieval.py` — search functions already accept `min_score=0.0`
- `reranker.py` — `rerank()` already accepts `score_threshold=-inf`
- `nodes.py` retrieval nodes — still called once each
- `nodes.py` `merge_node` — still called once
- `nodes.py` `reranking_node` — still called once
- `nodes.py` `neo4j_expansion_node` — still called (now once instead of 3×)
- `agent_graph.py` — agent loop, routing, reflect_final, think_node all unchanged
- Query rewrite node — unchanged (runs before agent loop)
- Agent retry with reworded queries — unchanged (separate rag_retrieve calls)
- `tried_queries` dedup in think_node — unchanged
- All other tests, frontend, docs

## Expected Performance Improvement

Per rag_retrieve call (3-level ladder, all insufficient):

| Operation | Before | After |
|-----------|--------|-------|
| Dense embedding API call | 3 | 1 |
| SPLADE embedding | 3 | 1 |
| Qdrant dense query | 3 | 1 |
| Qdrant sparse query | 3 | 1 |
| MySQL FTS query | 3 | 1 |
| Cross-encoder rerank | 3 | 1 (+1 for graph docs if any) |
| Neo4j graph expansion | 3 | 1 |
| **Total remote calls** | **21** | **7-8** |

For the e2e suite (9 rag_retrieve calls): 183 → ~63 remote calls.

Wall-clock improvement per rag_retrieve: ~30-45s → ~10-15s (embedding + reranker dominate).

## Verification Plan

1. Run `pytest tests/test_rag_retrieve_ladder.py -v` — all 8 tests pass
2. Run `pytest tests/test_rag_retrieve_ladder.py tests/test_agent_loop_budget.py tests/test_agent_state_integrity.py -v` — all pass
3. Run full backend suite: `pytest tests/ --ignore=tests/test_e2e_real_llm.py -q` — 636 pass
4. Run e2e suite: `pytest tests/test_e2e_real_llm.py -v` — 13 pass
5. Examine backend logs: confirm 1 embedding call per rag_retrieve, 1 reranker pass, 1 graph expansion
