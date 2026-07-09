# Logical Code Audit

**Date:** 2026-07-08 (verified 2026-07-08)
**Scope:** File placement, naming conventions, module boundaries, dependency graph, internal structure
**Goal:** Improve readability and code understanding through logical reorganization. No functionality changes.
> **Last Updated**: 2026-07-08 — Line-by-line verification against live codebase. Many findings resolved by prior cleanup.

---

## 1. Dead / Unreachable Code (blocking cleanup)

### 1.1 `rag_graph/` directory — RESOLVED (was HIGH)

The entire `rag_graph/` directory has been **deleted** along with the `rag_graph.py` service file. This cleanup was done in commit `c1d2f7f`.

**Status**: **RESOLVED** — The `rag_graph/` package and `rag_graph.py` file no longer exist.

---

### 1.2 `fast_pipeline.py` — RESOLVED (was HIGH)

`fast_pipeline.py` has been **deleted** in the same cleanup (commit `c1d2f7f`).

**Status**: **RESOLVED** — 620 lines of dead code removed.

---

### 1.3 `builtin_tools.py` and `tool_registry.py` — DEPRECATED (was MEDIUM)

`builtin_tools.py` (309 lines, 4 agentic tools) and `tool_registry.py` (158 lines, tool registry) remain at the `services/` root level. **Neither file is imported by any production code.** References to `builtin_tools`, `tool_registry`, `list_tools()`, or `execute_tool()` exist only in these two files and test files:
- `backend/tests/test_builtin_tools.py`
- `backend/tests/test_synthesis.py`
- `backend/tests/test_tool_registry.py`

Neither file imports the other from production code either (they only reference each other internally). The `@register_tool` decorators never execute at startup, so tools are never registered.

**Impact**: 467 lines of dead code. The tool registry pattern is defined but never wired into the agentic pipeline.

**Status**: **CONFIRMED** — Still present.

**Fix**: Delete both files. If tool calling is needed in the future, implement within `agentic_rag/` and wire into `run_agentic_rag`.

---

### 1.4 Misplaced scripts — RESOLVED (was LOW)

Previously flagged: `test_imports.py`, `conftest_debug.py`, `rootconftest.py`, `benchmark_vllm.py`, `debug_pipeline.py` — in `backend/` or `backend/app/` instead of `scripts/` or `tests/`.

**Status**: **PARTIALLY RESOLVED**
- ✅ `conftest_debug.py` — **DELETED**
- ✅ `rootconftest.py` — **DELETED**
- ❌ `test_imports.py` — still present in `backend/`
- ❌ `benchmark_vllm.py` — still present in project root
- ❌ `debug_pipeline.py` — still present in `backend/`

**Fix**: Move scripts to `scripts/`, or delete if unused.

---

## 2. Naming Inconsistencies

### 2.1 `agentic_rag/` directory — `__init__.py` re-exports (LOW)

The directory `agentic_rag/` contains `agentic_rag.py` which is a tautology: `agentic_rag.agentic_rag`. However, the `__init__.py` re-exports `run_agentic_rag`, so the shorter import path `from app.services.agentic_rag import run_agentic_rag` works. The line count is 747 lines (down from original claim of 776).

**Status**: **CONFIRMED** — Still present but less impactful since `__init__.py` provides clean imports.

**Fix**: Rename `agentic_rag/agentic_rag.py` → `agentic_rag/pipeline.py` (or `agentic_rag/agent.py`). Update the `__all__` in `__init__.py`.

---

### 2.2 `DataStore` vs `Datastore` — inconsistent casing (MEDIUM → LOW)

The codebase previously used `DataStore` (camelCase S) in models and `Datastore` (lowercase s) in `datastore_watcher/handler.py`.

**Status**: **RESOLVED** — The handler was refactored in commit `c1d2f7f` and renamed to `DatastoreFileEventHandler` → but actually it's **still** `DatastoreFileEventHandler` (lowercase s) in `handler.py`, `watcher.py`, and `__init__.py`. All references use consistent lowercase `s` within the `datastore_watcher/` package. The inconsistency is only between the package (`datastore_watcher`) and model names (`DataStore`, `KnowledgeBaseDataStore`).

**Impact**: Low — the naming is consistent within the package itself. Cross-package difference (lowercase `datastore` in the watcher path, camelCase `DataStore` in models) is a stylistic nit.

**Fix**: Consider renaming `DatastoreFileEventHandler` → `DataStoreFileEventHandler` for cross-package consistency, or accept the package-level naming convention.

---

### 2.3 `progress_timeout.py` vs `ProgressTimeout` — naming (OK)

The file is `progress_timeout.py` and the class is `ProgressTimeout`. This follows Python convention. No issues.

**Status**: **OK** — No changes needed.

---

### 2.4 `cancel_registry.py` naming — NO ISSUE (was LOW)

The file defines `set_cancel_token`, `get_cancel_token`, `clear_cancel_token`, `is_cancelled` — all functions, no class. The name is still accurate (it's a cancellation token registry).

**Status**: **OK** — No changes needed.

---

### 2.5 `_get_llm` in `agentic_rag/` (OK)

`agentic_rag/agentic_rag.py` defines `_get_llm` locally at line 175. It does NOT import from `rag_graph/helpers.py` (the `rag_graph/` package was deleted).

**Status**: **OK** — No issue.

---

## 3. File Placement — services/ directory organization

### 3.1 23 Python files at `services/` root (HIGH → MEDIUM)

The `backend/app/services/` directory has 23 `.py` files at its root level (not counting 3 subdirectories: `agentic_rag/`, `datastore_watcher/`, `prompts/`). No grouping by concern exists.

**Files at root**: `builtin_tools.py`, `tool_registry.py`, `chat_service.py`, `retrieval.py`, `confidence.py`, `document_processor.py`, `document_converter.py`, `document_qdrant.py`, `graph_service.py`, `deletion_service.py`, `export_service.py`, `historical_memory.py`, `startup_recovery_service.py`, `discovery_engine.py`, `entity_extractor.py`, `reranker.py`, `query_expander.py`, `markdown_cleaner.py`, `reasoning_tags.py`, `utils.py`, `progress_timeout.py`, `cancel_registry.py`

**Impact**: Finding a specific service requires scanning 23 filenames. New developers can't tell which files belong to which subsystem.

**Status**: **CONFIRMED** — No reorganization has been done since the original audit.

**Fix**: Group files into subdirectories by concern:

```
services/
  agentic_rag/          # (already exists)
  datastore_watcher/    # (already exists)
  prompts/              # (already exists)
  ingestion/            # document_processor.py, document_converter.py, document_qdrant.py, markdown_cleaner.py
  retrieval/            # retrieval.py, reranker.py, query_expander.py, confidence.py
  chat/                 # chat_service.py, historical_memory.py
  graph/                # graph_service.py, entity_extractor.py
  export/               # export_service.py
  cleanup/              # deletion_service.py
  discovery/            # discovery_engine.py, startup_recovery_service.py
  infrastructure/       # utils.py, progress_timeout.py, cancel_registry.py, reasoning_tags.py
```

After cleanup, remove `builtin_tools.py` and `tool_registry.py` first (dead code).

---

## 4. Monolithic Files

### 4.1 `graph_service.py` — 975 lines (HIGH)

This file handles 5 distinct responsibilities:
1. Neo4j driver singleton (`_get_driver`)
2. LLM pipeline construction (`_get_llm_pipeline`, `_SafeNeo4jWriter`)
3. Document ingestion and extraction (`build_graph_for_document`, `_extract_with_llm`, `_build_extraction_batches`)
4. Graph retrieval operations (`expand_docs_via_graph`, `enrich_docs_with_graph`)
5. Deletion/cleanup (`delete_graph_for_document`, `delete_graph_for_kb`, `purge_stale_graph_data`)

**Status**: **CONFIRMED** — 975 lines, unchanged. Still handles 5 responsibilities in one file.

**Fix**: Split into `graph/driver.py`, `graph/extraction.py`, `graph/retrieval.py`, `graph/cleanup.py` (see section 3.1 groupings).

---

### 4.2 `export_service.py` — 867 lines (HIGH)

This file handles 3 export formats (PDF, Word, image) plus ECharts chart rendering. The `_apply_config_to_chart` function has 30+ chart-type branches (~400+ lines of switch logic).

**Status**: **CONFIRMED** — 867 lines, unchanged.

**Fix**: Split into `export/base.py`, `export/pdf.py`, `export/word.py`, `export/image.py` (see section 3.1 groupings).

---

### 4.3 `chat_service.py` — 729 lines (MEDIUM)

This file handles:
1. Query rewriting (`_rewrite_query`)
2. Query classification (`classify_query`)
3. Message summarization (`_summarise_older_messages`, `_maybe_update_summary`)
4. Main chat orchestration (`generate_response`) — delegates to `run_agentic_rag`
5. Chat file handling (references `ChatFile` model)

**Status**: **CONFIRMED** — 729 lines, still mixes domain logic with orchestration.

**Fix**: Split into `chat/pipeline.py`, `chat/summarization.py`, `chat/classification.py` (see section 3.1 groupings).

---

### 4.4 `agentic_rag/agentic_rag.py` — 747 lines (MEDIUM)

This single file contains:
1. Constants and prompts (`_ANSWER_SYSTEM_PROMPT`, `_THINKING_KEYWORDS`)
2. Query complexity detection (`_is_complex_query`)
3. Model selection (`_select_model`)
4. LLM client creation (`_get_llm`)
5. Simple query path (`_direct_answer`)
6. Complex query path with subtask decomposition
7. Synthesis logic
8. Tool calling loop
9. Streaming event protocol

**Status**: **CONFIRMED** — 747 lines (down from ~776 originally). Still a monolith.

**Fix**: Split into `agentic_rag/pipeline.py`, `agentic_rag/decision.py`, `agentic_rag/tools.py`, `agentic_rag/synthesis.py`, `agentic_rag/streaming.py`, `agentic_rag/prompts.py`.

---

### 4.5 `document_processor.py` — 544 lines (MEDIUM)

This file handles:
1. Upload (`upload_document`)
2. Preview (`preview_document`)
3. Background processing (`process_document_background`) — 370 lines with 9 steps
4. Neo4j graph building (embedded in step 9)

**Status**: **CONFIRMED** — 544 lines, still a monolith for document processing.

**Fix**: Keep `upload_document` and `preview_document` in `document_processor.py`. Extract `process_document_background` into `ingestion/pipeline.py` with step functions to `ingestion/steps.py`.

---

### 4.6 `retrieval.py` — 692 lines (LOW)

This file handles:
1. Datastore ID resolution (`get_effective_datastore_ids`)
2. Retrieval config presets (`get_retrieval_config`)
3. Three search legs (`_dense_search`, `_sparse_search`, `_exact_search`) — each ~70 lines
4. RRF merge (`_rrf_merge_candidates`)
5. Public API (`hybrid_search_with_legs`) — 100+ lines

**Status**: **PARTIALLY CORRECT** — The structure is actually reasonable. Each leg is a function, the merge is a function, the public API composes them. The `_exact_search` leg was added since the original audit (not present before). The `_dense_search` and `_sparse_search` are nearly identical in structure, and `_exact_search` follows the same pattern.

**Fix**: Extract a `_search_leg` helper that takes the embedding function and search parameters as arguments. This would reduce duplication across all 3 legs.

---

### 4.7 `startup_recovery_service.py` — 602 lines (LOW)

Dense with nested try/finally chains for DB session management. Structure is correct but hard to follow.

**Status**: **CONFIRMED** — 602 lines, nested session management unchanged.

**Fix**: Extract session-scoped helpers. Not urgent — the logic is correct but dense.

---

## 5. Dependency Graph Issues

### 5.1 `retrieval.py` imports `QueryType` from `app.schemas.chat` (MEDIUM → HIGH)

`retrieval.py:49` imports `QueryType` from `app.schemas.chat`. The `schemas/` directory defines Pydantic models for API request/response validation. The `retrieval/` service layer should not depend on API schemas — this creates a dependency from business logic to the presentation layer.

**However**, `chat_service.py` ALSO imports `QueryType` from `app.schemas.chat` (it uses `QueryClassification`, `QueryType` at line for classification logic). Both the service layer (`retrieval.py` and `chat_service.py`) depend on API schemas.

**Status**: **CONFIRMED** — Both `retrieval.py` and `chat_service.py` import from `schemas/chat.py`.

**Impact**: If the schema changes, it affects service layer logic. The schema layer should be above the service layer, not below it.

**Fix**: Move `QueryType` and `QueryClassification` from `schemas/chat.py` to `models/` or `core/`. Or define a local enum in `retrieval.py`.

---

### 5.2 `builtin_tools.py` dead imports (LOW → RESOLVED)

`builtin_tools.py:61` imports `hybrid_search_with_legs` and `get_effective_datastore_ids` from `retrieval.py`. Since `builtin_tools.py` is dead code (finding 1.3), this is dead code importing from live code.

**Status**: **CONFIRMED** — But fixing this is contingent on deleting `builtin_tools.py` (finding 1.3).

**Fix**: Delete `builtin_tools.py` (finding 1.3).

---

### 5.3 `rag_graph/helpers.py` dependency — RESOLVED (was MEDIUM)

Previously claimed `agentic_rag/agentic_rag.py:175` imports `_get_llm` from `rag_graph/helpers.py`.

**Status**: **RESOLVED** — The `rag_graph/` package was deleted. `_get_llm` is now defined locally in `agentic_rag/agentic_rag.py`. No external dependency exists.

---

### 5.4 `datastore_watcher/handler.py` imports from `document_processor.py` (LOW)

`datastore_watcher/handler.py:31` imports `process_document_background` from `document_processor.py`. This creates a dependency from the file watcher → document processor. The handler is a subsystem that triggers ingestion.

**Status**: **CONFIRMED** — This cross-package dependency is still present between `datastore_watcher/` and `document_processor.py`.

**Impact**: Minor coupling. If `process_document_background` signature changes, both the handler and caller must update. Acceptable for now.

---

### 5.5 `document_qdrant.py` imports from `utils.py` (OK)

`document_qdrant.py:29` imports `content_hash`, `get_qdrant_client`, `get_sparse_embedder` from `utils.py`. This is correct — `utils.py` provides shared infrastructure.

**Status**: **OK** — Good separation.

---

## 6. Frontend Structure

### 6.1 `answer.tsx` — 1,055 lines, 10 inline components (HIGH)

The file contains 10 components at the module level:
- `useDebouncedValue` (hook)
- `ThinkBlock` (collapsible thinking block)
- `RewrittenQueryBlock` (collapsible query display)
- `RetrievedContextBlock` (collapsible context display)
- `RetrievedGraphBlock` (collapsible graph display)
- `QueryClassificationBlock` (classification badge)
- `ToolTraceBlock` (tool call timeline)
- `FailedLegsWarning` (warning banner)
- `ConfidenceCollapsible` (confidence score)
- `Answer` (main component)

**Status**: **CONFIRMED** — 1,055 lines, unchanged. Each component could be tested independently.

**Fix**: Extract each sub-component to its own file in `components/chat/answer/` or keep as internal components but split the file.

---

### 6.2 `agent-timeline.tsx` — 513 lines (MEDIUM)

The `NODE_META` dictionary maps 14 node names to display metadata. The `detailText` switch statement has 15+ cases.

**Status**: **CONFIRMED** — 513 lines, unchanged.

**Fix**: Extract step detail rendering into a registry pattern or separate component per step type.

---

### 6.3 `__tests__/` directories scattered (LOW)

Tests exist in:
- `frontend/src/components/chat/__tests__/` (6 files)
- `frontend/src/app/dashboard/chat/__tests__/` (1 file)
- `frontend/src/__tests__/` (1 file)

**Status**: **CONFIRMED** — Inconsistent, no changes since audit.

**Fix**: Consolidate under `__tests__/` at the component level for consistency.

---

## 7. Internal Structure — Code Placement Within Files

### 7.1 `document_processor.py` — step functions should be extracted (MEDIUM)

The `process_document_background` function (370 lines) has 9 numbered steps, each with inline logic. Steps like "Step 9: Build Neo4j knowledge graph" are nested inside the function as inline async functions.

**Status**: **CONFIRMED** — 370-line monolithic function with 9 steps, nested try/except, and inline graph building.

**Fix**: Extract each step to a named function: `_step_parse()`, `_step_chunk()`, `_step_qdrant_collection()`, `_step_move_file()`, `_step_create_document()`, `_step_delete_old_chunks()`, `_step_build_chunks()`, `_step_upsert_qdrant()`, `_step_build_graph()`.

---

### 7.2 `graph_service.py` — `_get_llm_pipeline` contains nested class (MEDIUM)

`graph_service.py:131` defines `_SafeNeo4jWriter` as a nested class inside the module. This class overrides `_upsert_nodes` and `run` methods. It's a 40-line class that exists solely to clean embedding properties before writing to Neo4j.

**Status**: **CONFIRMED** — Still present at the same location.

**Fix**: Move `_SafeNeo4jWriter` to its own file or keep as a module-level class (not nested).

---

### 7.3 `export_service.py` — chart type mapping is a giant dict (LOW)

`export_service.py:74-106` has a 30+ entry `chart_map` dict mapping ECharts chart types to pyecharts classes.

**Status**: **CONFIRMED** — Still present, 30+ entries.

**Fix**: Consider a registry pattern: `@register_chart("bar", Bar)` decorator that builds the map.

---

### 7.4 `agentic_rag/agentic_rag.py` — prompts embedded in code (MEDIUM)

`agentic_rag/agentic_rag.py:52-70` embeds `_ANSWER_SYSTEM_PROMPT` as a multi-line string. The `prompts/` directory already exists with `prompts/loader.py` for prompt loading.

**Status**: **CONFIRMED** — Prompts still embedded in code. The `prompts/` package exists but prompts are not loaded from there for the agentic pipeline.

**Fix**: Move prompts to `prompts/` files. Use `prompts/loader.py` to load them.

---

## 8. Circular Dependency Check

### 8.1 No circular dependencies found (GOOD)

Traced the full import graph:
- `services/` → imports from `models/`, `schemas/`, `core/`, `db/`
- `models/` → imports from `base.py` only (no cycles)
- `schemas/` → imports from `models/` only (no cycles)
- `api/` → imports from `services/`, `models/`, `schemas/`
- `agentic_rag/` → imports from `services/retrieval.py`, `services/confidence.py`, `services/reasoning_tags.py`, `services/prompts/loader.py`, `services/utils.py`
- `builtin_tools.py` → imports from `tool_registry.py`, `retrieval.py`, `entity_extractor.py` (dead code, see 1.3)

No circular dependencies detected. The dependency graph is a clean DAG.

**Status**: **CONFIRMED** — No changes to the import graph since the original audit.

---

## 9. Summary

| Category | High | Medium | Low | OK | Resolved |
|----------|------|--------|-----|-----|----------|
| Dead code | 1 | 1 | 1 | 0 | 3 |
| Naming | 0 | 1 | 1 | 3 | 1 |
| File placement | 0 | 1 | 0 | 0 | 0 |
| Monolithic files | 2 | 4 | 1 | 0 | 1 |
| Dependencies | 0 | 1 | 1 | 0 | 1 |
| Frontend | 1 | 1 | 1 | 0 | 0 |
| Internal structure | 0 | 3 | 1 | 0 | 0 |
| Circular deps | 0 | 0 | 0 | 1 | 0 |
| **Total** | **4** | **12** | **6** | **4** | **6** |

### Recommended order of operations:

1. **Delete `builtin_tools.py` and `tool_registry.py`** (findings 1.3) — removes 467 lines of dead code. Zero risk, highest impact.

2. **Group `services/` into subdirectories** (finding 3.1) — ingestion/, retrieval/, chat/, graph/, export/, cleanup/, discovery/, infrastructure/. Do this after deleting dead code to reduce the scope.

3. **Split monolithic files** (findings 4.1-4.7) — start with `graph_service.py` (975 lines) and `export_service.py` (867 lines).

4. **Fix `QueryType` dependency** (finding 5.1) — move out of `schemas/` to `models/` or `core/`.

5. **Move misplaced scripts** (finding 1.4) — `test_imports.py`, `benchmark_vllm.py`, `debug_pipeline.py` to `scripts/`.

6. **Fix naming** (finding 2.1) — rename `agentic_rag.py` → `pipeline.py` to reduce import verbosity.

7. **Split frontend `answer.tsx`** (finding 6.1) — extract 10 sub-components.
