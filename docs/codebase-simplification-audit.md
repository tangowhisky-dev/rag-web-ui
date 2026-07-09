# Codebase Simplification Audit

**Date:** 2026-07-08 (verified 2026-07-08)
**Scope:** Full stack — backend (Python/FastAPI), frontend (Next.js/React), Docker, infra
**Goal:** Identify dead, redundant, inefficient, and anomalous code. Document findings with impact and proposed fixes. No code changes in this pass.
> **Last Updated**: 2026-07-08 — Line-by-line verification against live codebase. Several findings resolved by prior cleanup, some corrected, some confirmed.

---

## 1. Backend — Service Layer

### 1.1 Duplicate `_serialise_doc` implementations (HIGH)

| Location | Notes | Status |
|----------|-------|--------|
| `backend/app/services/rag_graph/helpers.py:56` | Deleted with `rag_graph/` package | **RESOLVED** |
| `backend/app/services/fast_pipeline.py:81` | Deleted | **RESOLVED** |
| `backend/app/services/agentic_rag/agentic_rag.py:186` | Identical logic, imports from `app.services.utils` | **CONFIRMED** — now imports from `utils.py` |
| `backend/app/services/historical_memory.py:63` | Imports from `app.services.utils` | **RESOLVED** — now uses shared import |
| `backend/app/services/export_service.py` | Inline in `_strip_markdown` (different function, not `_serialise_doc`) | **NOTED** — not a `_serialise_doc` copy |

**Impact:** Two remaining callers (`agentic_rag.py`, `historical_memory.py`) now share a single canonical `_serialise_doc` in `utils.py`. This was a prior improvement.

**Status**: **RESOLVED** — Extracted to `app/services/utils.py`. Both consumers import from there.

---

### 1.2 Duplicate datastore resolution logic (HIGH)

The pattern:

```python
datastore_links = db.query(KnowledgeBaseDataStore.data_store_id) \
    .filter(...).distinct().all()
datastore_ids = [row.data_store_id for row in datastore_links]
# then org-linked datastores
org_ds_links = db.query(OrganizationDataStore.data_store_id) \
    .filter(...).distinct().all()
```

This exact code block appears in **four** places (two resolved by deletion):

| Location | Lines | Status |
|----------|-------|--------|
| `fast_pipeline.py:172-196` | Deleted | **RESOLVED** |
| `rag_graph/nodes.py:425-452` | Deleted | **RESOLVED** |
| `agentic_rag/agentic_rag.py:209-231` | Uses `get_effective_datastore_ids` from `retrieval` | **RESOLVED** — now imports shared function |
| `builtin_tools.py:69-79` (search_documents tool) | Uses `get_effective_datastore_ids` from `retrieval` | **RESOLVED** — now imports shared function |
| `builtin_tools.py:266-276` (synthesize_documents tool) | Uses `get_effective_datastore_ids` from `retrieval` | **RESOLVED** — now imports shared function |

**Impact:** **RESOLVED** — All 4 remaining copies have been consolidated into a single `get_effective_datastore_ids(kb_ids, org_id, db)` function in `retrieval.py`.

---

### 1.3 `ChunkRecord` and `DataStoreChunkRecord` are dead code (HIGH)

Both classes (`chunk_record.py` and `datastore_chunk_record.py`) create their own `create_engine()` instances and manage sessions independently.

| File | Status |
|------|--------|
| `backend/app/services/chunk_record.py` | **RESOLVED** — file no longer exists |
| `backend/app/services/datastore_chunk_record.py` | **RESOLVED** — file no longer exists |

**Impact:** **RESOLVED** — 143 lines of dead code deleted. The imports in `document_processor.py` are also gone.

---

### 1.4 `discovery_engine.py` — `discover_all()` creates its own session but caller already has one (LOW)

`discover_all(db: Session)` receives a session but then calls `discover_datastore()` which creates its own `SessionLocal()` per datastore. The caller's session is only used to fetch active IDs. This is fine for read-only but confusing — the API suggests the caller's session is used.

**Status**: **CONFIRMED** — Still present.

**Impact:** Minor clarity issue. No functional bug.

**Fix:** Rename to `get_active_datastore_ids()` to make the actual usage clear.

---

### 1.6 `export_service.py` — 867 lines, massive switch-like `_apply_config_to_chart` (HIGH)

The `_apply_config_to_chart` function has 30+ chart-type branches, each handling a different pyecharts chart type. This is a maintenance burden. The function is only called during PDF/Word/image export.

**Status**: **CONFIRMED** — 867 lines unchanged.

**Impact:** Adding a new chart type requires editing this monolithic function. Difficult to test.

**Fix:** Consider a registry pattern or delegate to a dedicated chart renderer module. Alternatively, if chart export is rarely used, keep it but add comprehensive tests.

---

### 1.7 `agentic_rag/` — partially implemented feature (MEDIUM)

The `agentic_rag/` directory contains:
- `agentic_rag.py` — Full autonomous agent pipeline (747 lines)
- `context_manager.py` — Token budgeting (286 lines)
- `user_profile.py` — User preference store (261 lines)
- `tools/db_query_tool.py` — Safe DB query (156 lines)
- `tools/graph_query_tool.py` — Neo4j traversal (167 lines)

This is a **live feature**, gated behind `AGENT_ENABLED`. The fast/thinking pipelines (`fast_pipeline.py`) and LangGraph pipeline (`rag_graph/`) have been removed as dead code. The agentic agent is now the primary parallel pipeline.

**Status**: **CONFIRMED** — Line counts adjusted (747 lines for `agentic_rag.py`).

**Impact:** Duplicates significant retrieval logic that has been partially deduplicated via `get_effective_datastore_ids` (see 1.2).

**Fix:** Further extract shared retrieval utilities. The `_serialise_doc` deduplication (see 1.1) is done.

---

### 1.8 `graph_service.py` — 975 lines, multiple responsibilities (MEDIUM)

This file handles:
- Neo4j driver singleton
- LLM pipeline construction (neo4j-graphrag)
- Chunk batch building with overlap stripping
- LLM extraction with retry logic
- Document ingestion
- Graph expansion for retrieval
- Graph enrichment for answers
- Deletion/cleanup

**Status**: **CONFIRMED** — 975 lines, multiple responsibilities.

**Impact:** Violates single responsibility. Hard to test individual pieces. The `_get_llm_pipeline()` function has a nested class `_SafeNeo4jWriter` that overrides methods.

**Fix:** Split into: `graph_driver.py` (singleton), `graph_extraction.py` (LLM pipeline + batch building), `graph_retrieval.py` (expansion + enrichment), `graph_cleanup.py` (deletion).

---

### 1.9 `rag_graph/nodes.py` — 1382 lines — RESOLVED

This file was deleted along with the entire `rag_graph/` package.

**Status**: **RESOLVED** — Confirmed deleted. The agentic agent now lives in `agentic_rag/agentic_rag.py` (747 lines), the sole production RAG pipeline.

---

### 1.10 `retrieval.py` — `_dense_search` and `_sparse_search` are nearly identical (LOW)

`retrieval.py` is now 692 lines. `_dense_search` and `_sparse_search` share the same structure: iterate KB collections, iterate DS collections, hash dedup, populate candidates. The only difference is the embedding method and Qdrant `using` parameter.

**Status**: **CONFIRMED** — Still present in 692-line `retrieval.py`.

**Impact:** Code duplication makes it error-prone to add a new collection type.

**Fix:** Extract a `_search_collection()` helper that takes the embedding function and Qdrant `using` parameter as arguments.

---

### 1.11 `startup_recovery_service.py` — 602 lines with nested try/except/finally chains (MEDIUM)

The `_discovery_pipeline_worker` method has three levels of try/finally for database sessions (`db`, `db2`, `db3`). The `_handle_deletion_records` method also has a separate DB session.

**Status**: **CONFIRMED** — 602 lines, nested try/finally patterns still present.

**Impact:** Hard to follow. The nested session management makes it easy to miss an edge case.

**Fix:** Extract session-scoped helpers. Not urgent — the logic is correct but dense.

---

### 1.12 `config.py` — 320 lines, 40+ settings (LOW)

The Settings class has grown organically. It mixes environment variables, computed properties, and hardcoded prompt text.

**Status**: **CONFIRMED** — 320 lines unchanged.

**Impact:** Difficult to audit which settings are actually used.

**Fix:** Move prompt templates to `app/prompts/` files. Consider splitting settings into categories (retrieval, model, ingestion, agentic).

---

### 1.13 `datastore_watcher/watcher.py` — 1190 lines (NEW)

A new large file created in the recent cleanup: `datastore_watcher/watcher.py` at 1,190 lines. The original `datastore_watcher/handler.py` was refactored into separate `watcher.py` and `handler.py` modules.

**Status**: **NEW FINDING** — This is a result of the `c1d2f7f` commit that extracted `datastore_watcher` into a package.

**Impact:** Similar monolith concerns as `graph_service.py` — multiple responsibilities in one file.

**Fix:** Consider splitting into `watcher.py` (scan lifecycle), `handler.py` (file event processing), `recovery.py` (stale scan cleanup).

---

## 2. Backend — API Layer

### 2.1 `api.py` — Only 27 lines, well-structured (OK)

No issues. Clean router aggregation.

---

### 2.2 `chat.py` endpoint — delegates to `chat_service.py` (OK)

Clean separation. No issues found.

### 2.3 `datastore_scan.py` — new file (MEDIUM)

`api/api_v1/datastore_scan.py` is a new file that moved from `services/datastore_scan.py`. It contains extensive async scan logic with SSE streaming support.

**Status**: **NEW FINDING** — ~638 lines with nested error handling (8+ `except Exception` occurrences).

**Impact:** Dense error handling makes it hard to verify all failure paths.

**Fix:** Consider extracting error handling patterns.

---

### 2.4 `datastore_recovery.py` — new file (MEDIUM)

`api/api_v1/datastore_recovery.py` — 362 lines of recovery logic.

**Status**: **NEW FINDING** — New file from `c1d2f7f` commit.

**Impact:** Moderate size, needs review for error handling completeness.

---

### 2.5 `document_converter.py` — new file (LOW)

New file at 169 lines. Contains `MAX_FILE_SIZE`, `SUPPORTED_EXTENSIONS`, and `_convert_to_markdown`.

**Status**: **NEW FINDING** — Now the canonical source for file handling constants.

**Impact:** Positive — centralizes what was previously duplicated.

---

## 3. Backend — Models

### 3.1 `models/__init__.py` — exports all models (OK)

Standard pattern. No issues.

---

### 3.2 `models/base.py` — Base model with `id`, `created_at`, `updated_at` (OK)

Standard SQLAlchemy pattern. **RESOLVED**: Now uses `from sqlalchemy.orm import declarative_base`.

---

## 4. Backend — Misc Files

### 4.1 `test_imports.py` — diagnostic script (LOW)

A file named `test_imports.py` in the app root (not in `tests/`). Still present with hardcoded macOS path and `print()` calls.

**Status**: **CONFIRMED** — Diagnostic script, not a pytest test.

**Impact:** Dead code unless maintained.

**Fix:** Delete or move to `scripts/`.

---

### 4.2 `conftest_debug.py` and `rootconftest.py` — test fixtures outside `tests/` (LOW)

These files are outside the standard `tests/` directory.

**Status**: **CONFIRMED** — Both present in `backend/` root.

**Impact:** May confuse new developers about test structure.

**Fix:** Move into `tests/` or confirm they're imported by test configs.

---

### 4.3 `benchmark_vllm.py` — one-off benchmark script (LOW)

In the project root, not in `tests/`.

**Status**: **CONFIRMED** — Present, 16KB file.

**Impact:** Dead code unless maintained.

**Fix:** Move to `scripts/` or delete.

---

### 4.4 `download_assets.py` — asset download script (LOW)

In project root. Downloads model assets.

**Status**: **CONFIRMED** — Present.

---

### 4.5 `debug_pipeline.py` — debugging script (LOW)

In `backend/`. Debugging script, not part of production code.

**Status**: **CONFIRMED** — Present.

**Fix:** Move to `scripts/` or delete.

---

## 5. Frontend

### 5.1 `answer.tsx` — 1,055 lines, multiple inline components (HIGH)

This file contains 15+ nested FCs. UI polish reduced it from 1,056 to 1,055 (1 line change).

**Status**: **CONFIRMED** — 1,055 lines still present.

**Impact:** Each sub-component could be tested independently but they're all coupled in one file.

**Fix:** Extract each sub-component to its own file in `components/chat/answer/` or keep as internal components but split the file.

---

### 5.2 `agent-timeline.tsx` — 513 lines, large `NODE_META` map (MEDIUM)

The `NODE_META` dictionary maps 14 node names to display metadata.

**Status**: **CONFIRMED** — 513 lines, unchanged.

**Fix:** Extract step detail rendering into a registry pattern or separate component per step type.

---

### 5.3 `api.ts` — `fetchApi` handles 401 with `window.location.href` (MEDIUM)

`lib/api.ts` (106 lines) still does `window.location.href = '/'` on 401.

**Status**: **CONFIRMED** — 106 lines, pattern unchanged.

**Fix:** Use Next.js router for navigation.

---

### 5.4 `api.ts` — template literal syntax error (CRITICAL)

Line 27: `Authorization: *** ${token}` — this was a sanitization artifact in the audit output. The actual code has `Authorization: \`Bearer ${token}\``.

**Status**: **FALSE POSITIVE** — Sanitization artifact only.

---

### 5.5 `chat-input.tsx` — KB selector checkbox is non-functional (LOW)

`type="checkbox"` with `onChange={() => {}}` present.

**Status**: **CONFIRMED** — 279 lines, checkbox no-op still present.

**Fix:** Either remove the checkbox and use a checkmark icon, or wire it up properly.

---

### 5.6 `echarts-diagram.tsx` and `mermaid-diagram.tsx` — well-structured (OK)

Both are clean, focused components. Good use of dynamic import for SSR. No issues.

---

### 5.7 `chat-sidebar.tsx` — 499 lines (MEDIUM)

New finding from the `7c73b4f` UI polish commit. 499 lines with 3x `console.debug()` calls leaking query text, result counts, and DnD events.

**Status**: **NEW FINDING** — `chat-sidebar.tsx` grew significantly.

**Impact:** Console.debug() leaks sensitive data to browser console.

**Fix:** Gate behind `process.env.NODE_ENV === 'development'`.

---

### 5.8 Test files scattered across `__tests__/` directories (LOW)

Tests exist in:
- `frontend/src/components/chat/__tests__/` (6 files)
- `frontend/src/app/dashboard/chat/__tests__/` (1 file)
- `frontend/src/__tests__/` (1 file)
- `backend/tests/` (29+ files, grew from 29)

**Status**: **CONFIRMED** — Multiple test directories.

**Fix:** Consider consolidating under `__tests__/` at the component level for consistency.

---

## 6. Docker / Infrastructure

### 6.1 `docker-compose.dev.yml` and `docker-compose.yml` — two compose files (LOW)

Standard pattern for dev/prod separation. No issues.

---

### 6.2 `entrypoint.sh` — startup hook for stuck task recovery (OK)

Mentioned in memory as a fix for reload crashes.

---

## 7. Cross-Cutting Concerns

### 7.1 Multiple LLM client instantiation patterns (MEDIUM)

The codebase uses three different patterns for creating LLM clients:

| Pattern | Location | Status |
|---------|----------|--------|
| **Lazy singleton** | `utils.py`: `get_qdrant_client`, `get_openai_client`, `get_sparse_embedder`, `content_hash` | **CONFIRMED** — 4 utility functions |
| **Per-call instantiation** | `confidence.py`, `agentic_rag.py` | **CONFIRMED** |
| **LangChain ChatOpenAI** | `context_manager.py` (wrapped differently) | **CONFIRMED** |

**Impact:** Inconsistent resource management.

**Fix:** Standardize on a single client creation pattern. The singleton approach in `utils.py` is the most efficient.

---

### 7.2 `content_hash` usage (LOW)

`content_hash` in `utils.py` uses SHA-256. It's now used consistently across `retrieval.py`, `document_qdrant.py`, and other files. The `builtin_tools.py:synthesize_documents` still uses `hashlib.md5()` for deduplication.

**Status**: **PARTIALLY RESOLVED** — `content_hash` is now the standard but MD5 is still used in `builtin_tools.py`.

**Impact:** Inconsistent deduplication. MD5 is weaker and operates on truncated content.

**Fix:** Use `content_hash` from `utils.py` consistently.

---

### 7.3 `_get_llm` defined in three places — partially resolved

| Location | Notes | Status |
|----------|-------|--------|
| `rag_graph/helpers.py:21` | Deleted with `rag_graph/` | **RESOLVED** |
| `fast_pipeline.py:69` | Deleted | **RESOLVED** |
| `agentic_rag/agentic_rag.py:175` | Inlined, still live | **CONFIRMED** |
| `graph_service.py` | `_get_llm_pipeline()` — different function | **CONFIRMED** |
| `entity_extractor.py` | `_get_llm_client()` — different function | **NEW** |
| `context_manager.py` | `ChatOpenAI(...)` wrapped differently | **CONFIRMED** |

**Impact:** Three remaining LLM client creation patterns. All have different signatures.

**Fix:** Move the ChatOpenAI variant to `utils.py`. Create a unified `_get_llm_client()` utility.

---

### 7.4 Hardcoded prompt text in config vs. prompt files (LOW)

`config.py` contains prompt strings. `prompts/loader.py` loads chart instructions from `prompts/charts-documentation.md`.

**Status**: **CONFIRMED** — `prompts/` package exists but prompts are still mixed between files and config.

**Fix:** Move all prompts to files under `prompts/`.

---

### 7.5 `strip_reasoning_tags` called in multiple places (OK)

Called from `chat_service.py`, `agentic_rag/agentic_rag.py`, `document_converter.py`, `export_service.py`. This is correct — reasoning tags can appear in any LLM output.

**Status**: **CONFIRMED** — All callers still present.

---

## 8. Summary

| Category | High | Medium | Low | OK | Resolved |
|----------|------|--------|-----|-----|----------|
| Backend services | 2 | 5 | 2 | 2 | 4 |
| Backend API | 0 | 2 | 0 | 2 | 0 |
| Backend models | 0 | 0 | 0 | 2 | 1 |
| Frontend | 1 | 3 | 2 | 2 | 0 |
| Docker/Infra | 0 | 0 | 0 | 2 | 0 |
| Cross-cutting | 1 | 2 | 1 | 0 | 0 |
| **Total** | **4** | **12** | **7** | **10** | **5** |

### Top priorities for simplification:

1. ~~**Eliminate duplicate `_serialise_doc`** — **RESOLVED** via `utils.py`~~
2. ~~**Eliminate duplicate datastore resolution** — **RESOLVED** via `retrieval.py`'s `get_effective_datastore_ids()`~~
3. ~~**Delete `ChunkRecord` and `DataStoreChunkRecord`** — **RESOLVED** — files deleted~~
4. **Split `export_service.py`** — 867 lines, massive chart switch.
5. **Split `graph_service.py`** — 975 lines, multiple responsibilities.
6. **Split `datastore_watcher/watcher.py`** — 1,190 lines, new monolith.
7. **Split `answer.tsx`** — 1,055 lines, 15+ components.
8. **Fix `api.ts` 401 handling** — use router instead of `window.location.href`.
9. **Fix checkbox in `chat-input.tsx`** — non-functional UI element.
10. **Standardize LLM client creation** — consolidate `_get_llm`, `_get_llm_pipeline`, `_get_llm_client` to unified utility.
