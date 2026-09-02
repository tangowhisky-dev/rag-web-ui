# Agentic Retrieval Filtering — Implementation Plan

> **STATUS: IMPLEMENTED (Phases 1-4, 6) — SUPERSEDED by `docs/retrievalagent-gap-analysis.md`**
>
> All phases in this plan were implemented, tested, and validated:
> - Phase 1: Metadata in Qdrant payload (ingestion) — done
> - Phase 2: Filters + sort in rag_retrieve — done
> - Phase 3: kb_metadata introspection tool — done
> - Phase 4: Query rewrite with failure context — done
> - Phase 6: Frontend tool icon + display — done
> - Phase 5 (answer-grade retry loop) — intentionally skipped
>
> The follow-up plan in `docs/retrievalagent-gap-analysis.md` covers the next
> round of improvements (negation extraction, synonym expansion, conditional
> dense leg, KB profiling, query intent extraction, answer suggestions).

## Problem

The agent cannot pass metadata filters (title, date, file type) to `rag_retrieve`.
It only passes a search string, so queries like "latest weekly update" return many
low-ranked documents with no way to pick the most recent one. The agent has no tool
to inspect what metadata is available before searching.

## Scope

Phases 1-4 + 6 (retrieval filtering + ingestion metadata + frontend).
Phase 5 (answer-grade retry loop) is deferred — it modifies the agent graph
topology and is out of scope for this iteration.

## Phase 1: Add metadata to Qdrant payload (ingestion)

**Goal:** Store `created_at`, `modified_at`, `content_type`, `file_size` in the
Qdrant payload so retrieval can filter on them natively.

**Files:**
- `backend/app/services/ingestion/document_processor.py` — `_build_single_chunk`
- `backend/app/services/ingestion/document_qdrant.py` — payload construction

**Changes:**
1. In `_build_single_chunk`, add `created_at`, `modified_at`, `content_type`,
   `file_size` to `source_metadata` from the `document` object.
2. The Qdrant payload already spreads `**source_meta`, so these flow through
   automatically.

**Verification:**
- After re-ingestion, check Qdrant payload contains the new fields.
- Unit test: ingest a small document, verify payload keys.

## Phase 2: Add `filters` and `sort` to `rag_retrieve`

**Goal:** Let the agent pass metadata filters and sort order to the retrieval
pipeline. This is the core change that solves the "latest weekly update" problem.

**Files:**
- `backend/app/services/agentic_rag/tools/rag_retrieve.py` — input schema + dispatch
- `backend/app/services/retrieval/retrieval.py` — `_dense_search`, `_sparse_search`,
  `_exact_search` accept `doc_ids` filter
- `backend/app/services/agentic_rag/nodes.py` — retrieval nodes forward `doc_ids`

**Changes:**
1. `RagRetrieveInput` gets `filters: Optional[dict]` and `sort: Optional[dict]`.
2. New helper `_resolve_filter_to_doc_ids` translates metadata filters to a list
   of `document_ids` via MySQL `Document` table.
3. `_dense_search` and `_sparse_search` accept `doc_ids` and construct a Qdrant
   `Filter` on `document_id`.
4. `_exact_search` adds `AND d.id IN (...)` to the MySQL FULLTEXT query.
5. After `merge_node`, if `sort` is provided, sort merged docs by the specified
   field before `reranking_node`.

**Filter schema:**
```python
filters = {
    "title_contains": "Weekly Update",       # str, optional
    "file_name_contains": "weekly",           # str, optional
    "content_type": "application/pdf",        # str, optional
    "created_after": "2026-06-01",            # ISO date, optional
    "created_before": "2026-06-30",           # ISO date, optional
    "document_ids": [1, 2, 3]                 # list[int], optional
}
sort = {"field": "created_at", "direction": "desc"}
```

**Verification:**
- Unit test: pass filters, verify only matching docs returned.
- Unit test: pass sort, verify docs are ordered.
- Unit test: no filters → all docs (backward compatible).

## Phase 3: Add `kb_metadata` introspection tool

**Goal:** Let the agent discover what documents exist and what fields it can
filter on, before calling `rag_retrieve`. This makes the agent autonomous —
it doesn't need the user to pre-filter.

**New file:** `backend/app/services/agentic_rag/tools/kb_metadata.py`

**Actions:**
- `list_fields` → static schema dict (no DB query)
- `unique_values(field)` → `SELECT DISTINCT {field} FROM documents WHERE kb_id IN (...)`
- `date_range(field)` → `SELECT MIN({field}), MAX({field}) FROM documents WHERE kb_id IN (...)`
- `list_documents` → recent documents with title, filename, date, type

**Register in:** `backend/app/services/agentic_rag/tools/__init__.py`
- Add `KbMetadataTool` to `_TOOL_CLASSES`
- Gate on `has_kb` (same as kb_grep/kb_read/kb_outline)

**Verification:**
- Unit test: each action returns correct shape.
- RBAC test: unauthorized KBs are filtered out.

## Phase 4: Improve query rewrite with failure context

**Goal:** When `rag_retrieve` fails to find sufficient docs, the rewrite step
should analyze *why* it failed (using top-3 doc snippets) and suggest a filter
if the query implies metadata filtering.

**File:** `backend/app/services/agentic_rag/tools/rag_retrieve.py`

**Changes:**
1. `_REWRITE_PROMPT` includes top-3 doc snippets from the failed retrieval.
2. Rewrite returns JSON with `rewritten_query` + `filter_suggestion`.
3. If `filter_suggestion` is not null, merge it with existing filters and pass
   to the next relaxation ladder run.

**Verification:**
- Unit test: rewrite with failure context produces a filter suggestion.
- Unit test: no filter needed → `filter_suggestion` is null.

## Phase 6: Frontend changes

**Files:**
- `frontend/src/components/chat/agentic-progress.tsx` — add `kb_metadata` icon

**Changes:**
1. Add `kb_metadata: DatabaseIcon` to `TOOL_ICONS`.
2. Tool call arguments (including `filters` and `sort`) are already rendered by
   the existing `ToolCallPair` component — no new UI needed.

**Verification:**
- Frontend test suite passes.
- Frontend build succeeds.

## Context verification (part of retrieval plan)

**Goal:** Verify that filtered `rag_retrieve` results don't produce duplicate
chunks in the LLM context.

**Files:**
- `backend/app/services/agentic_rag/nodes.py` — `format_context_string`

**Checks:**
1. After filtering + merge + dedup, no two chunks in the context string have
   the same `content_hash`.
2. After filtering + sort, the context string preserves the sort order (most
   recent first when `sort={"field": "created_at", "direction": "desc"}`).
3. The existing `semantic_dedup` at 0.95 threshold still runs on filtered
   results — filtering doesn't bypass dedup.

**Verification:**
- End-to-end test: ingest documents, query with filter, verify context string
  has no duplicates and preserves sort order.

## Implementation order (dependency-safe)

1. Phase 1 (ingestion metadata) — no dependencies
2. Phase 3 (kb_metadata tool) — no dependencies on Phase 1 or 2
3. Phase 2 (filters + sort) — depends on Phase 1 for full benefit
4. Phase 4 (rewrite with context) — depends on Phase 2
5. Phase 6 (frontend) — do last
6. Re-ingest test documents
7. Context verification
8. End-to-end tests

## What is NOT in scope

- Phase 5 (answer-grade retry loop) — deferred
- HyDE — not needed, query rewriting handles vague queries
- Separate intent classifier — the LLM's reasoning is the classifier
- Synonym/spell-correct parallel BM25 fan-out — existing legs cover this
- Changes to relaxation ladder — filter is applied before the ladder
- Changes to MMR/dedup — already working at 0.3 MMR + 0.95 semantic dedup
