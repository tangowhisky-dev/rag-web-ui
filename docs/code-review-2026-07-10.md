# Code Review: Errors, Dead Code, Redundancy, and Complexity

**Date:** 2026-07-10
**Scope:** Full codebase — backend (Python/FastAPI/LangGraph) + frontend (Next.js/React/TypeScript)
**Last updated:** 2026-07-10 (full pipeline audit)

---

## Summary of Changes Applied

### ✅ Completed

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1.1 | Missing `json` import in `context_manager.py` | CRITICAL | ✅ Fixed |
| 1.3 | DB session leak risk in `chat_service.py` | HIGH | ✅ Fixed |
| 3.1 | Context string building — 4 copies | HIGH | ✅ Unified via `utils.format_context_string()` |
| 3.2 | `_generate_streaming()` duplicates `generate_node()` | HIGH | ✅ Eliminated — `generate_node` removed, `graph_runner.py` uses `astream_events` |
| 3.3 | `_rewrite_query` in `chat_service.py` duplicates `nodes.py` | MEDIUM | ✅ Delegates to shared `utils.rewrite_query()` |
| 2.4 | `_heuristic_classify` dead code | LOW | ✅ Removed, replaced with `[rewritten]` fallback |
| 2.6 | "Unused" nodes in `nodes.py` | LOW | ✅ Restored as part of LangGraph StateGraph |
| 2.7 | `LangGraphCallbackHandler` unused | LOW | ✅ Removed |
| Bug | `generate_node()` syntax error (return+yield) | CRITICAL | ✅ Fixed, then removed as dead code |
| Bug | `graph.py` wiring — `orchestrator_node`/`direct_retrieval_node` missing deps | CRITICAL | ✅ Fixed with `functools.partial` |
| Bug | `graph_runner.py` non-LangGraph patterns | HIGH | ✅ Rewritten to use `astream_events` properly |
| New | LangGraph migration incomplete | HIGH | ✅ Completed — `graph.py` compiled StateGraph, `graph_runner.py` wired |

### ⏭️ Deferred to later sessions

| # | Issue | Severity | Reason |
|---|-------|----------|--------|
| 2.1 | `Retriever`/`Generator` dead classes in `retry.py` | LOW | Low priority |
| 2.2 | `ContextManager` 240-line unused class in `context_manager.py` | HIGH | Keep for potential future LangGraph wiring |
| 2.3 | `evaluator.py` unused | MEDIUM | Requires removing `ANSWER_QUALITY_GRADING_ENABLED` from config |
| 2.5 | `hybrid_search()` duplicate in `retrieval.py` | LOW | Used by `knowledge_base.py`, needs careful handling |
| 3.4 | `_ANSWER_SYSTEM_PROMPT` vs inline in `chat_service.py` | LOW | Different purposes (LangGraph vs legacy endpoint) |
| 3.6 | `_summarise_older_messages` three implementations | MEDIUM | Cross-module refactoring |
| 4.x | Complexity: Answer component, graph_runner, chat_service, config | Medium-High | Large scope, warrants separate sessions |
| 5.x | Code quality: magic numbers, error handling consistency | Low-Medium | Cosmetic improvements |

**All 329 existing tests pass after changes.**

---

## 1. Bugs and Correctness Issues

### 1.1 Missing `json` import in `context_manager.py`

| Field | Value |
|-------|-------|
| File | `backend/app/services/agentic_rag/context_manager.py` |
| Line | 190 |
| Severity | **CRITICAL** — will crash at runtime when `truncate_tool_output` is called |

**Issue:** The file uses `json.dumps(tool_output)` on line 190 but never imports `json`.

```python
# context_manager.py — missing import
def truncate_tool_output(self, tool_output: dict, ...) -> dict:
    ...
    sample_text = json.dumps(output[0])  # NameError
```

**Impact:** Any agentic agent cycle that calls `ContextManager.truncate_tool_output()` will raise `NameError: name 'json' is not defined`.

**Remedy:** Add `import json` at the top of `context_manager.py`.

---

### 1.2 `hybrid_search_with_legs` missing `return` in the non-async path

| Field | Value |
|-------|-------|
| File | `backend/app/services/retrieval/retrieval.py` |
| Line | ~340 (the `_rrf_merge_candidates` call) |
| Severity | **MEDIUM** — function returns `None` instead of expected dict |

**Issue:** The call to `_rrf_merge_candidates()` on line ~339 computes `candidates` but the code proceeds to annotate and return — the `candidates` variable is correctly used. However, examining the full flow, the `candidates` list is consumed immediately. **Actually this is correct — re-examining, this is fine.**

*Skip — re-examined, not a bug.*

---

### 1.3 Session reuse across streaming — DB session leak risk

| Field | Value |
|-------|-------|
| File | `backend/app/services/chat/chat_service.py` — `generate_response()` |
| Line | ~100-280 |
| Severity | **HIGH** — potential session exhaustion |

**Issue:** The `generate_response()` async generator holds `db` (a SQLAlchemy session) open for the entire duration of streaming, which can be 30–120 seconds. The session is only closed in the `finally` block. If the generator is cancelled (user disconnects, timeout), the `finally` block runs, but there's a window where the session is left in an inconsistent state.

**Impact:** Under load, this can exhaust the MySQL connection pool, causing `Too many connections` errors across all endpoints.

**Remedy:** Either use a shorter-lived session pattern (commit frequently, close early, reopen for final writes), or rely on connection pooling with proper max-size settings. The `query.py` endpoint already does this correctly by calling `db.close()` before the LLM call — apply the same pattern here.

---

## 2. Dead Code

### 2.1 Unused `HybridGraphService` — dead service class

| Field | Value |
|-------|-------|
| File | `backend/app/services/agentic_rag/retry.py` |
| Severity | **LOW** — dead code, adds confusion |

**Issue:** The `Retriever` and `Generator` classes in `retry.py` define complex retry logic with `yield_callback` patterns, but neither is actually called from anywhere in the agentic_rag pipeline. The pipeline (`graph_runner.py`) implements its own inline retry logic.

**Impact:** Dead code adds cognitive load and maintenance burden.

**Remedy:** Delete `Retriever` and `Generator` classes from `retry.py`, keep only `with_retry`, `RetryConfig`, `RetryResult`, and `RetryExhaustedError`.

---

### 2.2 Unused `ContextManager` class

| Field | Value |
|-------|-------|
| File | `backend/app/services/agentic_rag/context_manager.py` |
| Severity | **HIGH** — entire 240-line class is unused |

**Issue:** The `ContextManager` class and `TokenBudget` dataclass are never imported or called from any other module. The agentic pipeline uses a simple inline approach for context management via `estimate_messages_tokens()` in `nodes.py`'s `should_compress_context()` and `compress_context_node()`.

**Impact:** 240+ lines of dead code that are never executed. This module gives a false impression that sophisticated token budgeting is active when it is not.

**Remedy:** Delete `context_manager.py` entirely, or if the context manager logic is genuinely needed in the future, move it to `core/`.

---

### 2.3 `evaluator.py` — evaluated but not integrated

| Field | Value |
|-------|-------|
| File | `backend/app/services/agentic_rag/evaluator.py` |
| Severity | **MEDIUM** — dead code, costs on import |

**Issue:** The `evaluator.py` module defines `AnswerEvaluation`, `evaluate_answer()`, and `summarize_evaluation()` but is never imported from the pipeline. The `ANSWER_QUALITY_GRADING_ENABLED` config flag exists in `config.py` but nothing in the pipeline actually calls these functions.

**Impact:** Dead code with a config flag that does nothing. Confusing for maintainers who see the flag but never see results.

**Remedy:** Either integrate the evaluator into the pipeline, or delete the module + remove `ANSWER_QUALITY_GRADING_ENABLED` from config.

---

### 2.4 `_heuristic_classify` function

| Field | Value |
|-------|-------|
| File | `backend/app/services/agentic_rag/nodes.py` |
| Line | ~140-160 |
| Severity | **LOW** — fallback path that's only hit when LLM structured output fails |

**Issue:** The `_heuristic_classify` function always returns `[rewritten]` regardless of input. Every code path returns the same value, making the function effectively:

```python
def _heuristic_classify(query: str, rewritten: str) -> List[str]:
    return [rewritten]
```

This is called from `classify_query_node()` when structured output fails. The regex patterns for multi-part detection and question counting are dead logic.

**Remedy:** Replace with a one-liner or remove if the structured output path is reliable enough.

---

### 2.5 `hybrid_search()` — partially dead (used in `knowledge_base.py` only)

| Field | Value |
|-------|-------|
| File | `backend/app/services/retrieval/retrieval.py` |
| Line | ~410 |
| Severity | **LOW** — used in only one place, duplicated logic |

**Issue:** The `hybrid_search()` function is a synchronous wrapper around the same search legs. It IS used by `knowledge_base.py` (line 714-715), but both functions share identical search leg logic (`_dense_search`, `_sparse_search`, `_exact_search`, `_rrf_merge_candidates`). Any change to one leg must be reflected in both.

**Remedy:** Refactor `hybrid_search()` to call `hybrid_search_with_legs()` internally, eliminating the duplicated leg invocation logic. Or better: remove `hybrid_search()` and update the single caller in `knowledge_base.py`.

---

### 2.6 `orchestrator_node` and related nodes in `nodes.py` ⚠️ MISIDENTIFIED — NOT DEAD CODE

| Field | Value |
|-------|-------|
| File | `backend/app/services/agentic_rag/nodes.py` |
| Severity | **N/A** — was misidentified as dead code |

**Correction:** These nodes were **not dead code**. They were built for Phase 3 (graph compilation) of the LangGraph migration which was never completed. The current `graph_runner.py` originally used a flat async generator that mimicked the LangGraph flow but didn't actually use LangGraph's `StateGraph`.

**Status:** ✅ **Completed.** A compiled `StateGraph` is now defined in `graph.py` that wires all nodes:
- **Main graph:** `START → rewrite → classify → [direct_retrieval | agent_subgraph] → synthesize → END`
- **Agent subgraph:** `orchestrator → direct_retrieval → should_compress → orchestrator → collect/fallback`
- `graph_runner.py` now calls the compiled graph via `astream_events()` and preserves the SSE event protocol.

**Remedy:** No deletion needed — these nodes are now wired into the compiled graph.

---

### 2.7 `LangGraphCallbackHandler` in `callbacks.py`

| Field | Value |
|-------|-------|
| File | `backend/app/services/agentic_rag/callbacks.py` |
| Severity | **LOW** — unused LangChain handler |

**Issue:** The `LangGraphCallbackHandler` class is defined as a LangChain callback handler but never instantiated or passed to any LangChain operation. Only `SSEEventEmitter` is used.

**Remedy:** Delete `LangGraphCallbackHandler`.

---

## 3. Redundant Code

### 3.1 Context string building — 4 copies of the same pattern ✅ FIXED

| Field | Value |
|-------|-------|
| Files | `nodes.py`, `graph_runner.py`, `chat_service.py` |
| Severity | **HIGH** → ✅ **Resolved** |

**Issue:** The exact same loop that builds `"[KB-N] (source)\ncontent\n\n---\n\n"` context strings appeared at least 4 times.

**Status:** ✅ **Fixed.** A shared `format_context_string(docs, file_markdown)` utility was added to `utils.py` and all callers now delegate to it:
- `nodes.py` → `direct_retrieval_node()`
- `graph_runner.py` → complex and simple paths
- `chat_service.py` → context building in `generate_response()`

**Result:** One source of truth for context formatting.

---

### 3.2 `_generate_streaming()` duplicates `generate_node()`

| Field | Value |
|-------|-------|
| Files | `nodes.py` (`generate_node`, ~120 lines) and `graph_runner.py` (`_generate_streaming`, ~90 lines) |
| Severity | **HIGH** — near-identical implementations |

**Issue:** Both functions:
1. Build the same system prompt + context messages
2. Call `model.astream(messages)`
3. Strip reasoning tags with identical regex
4. Normalise citations with `r'\[(\d+)\](?!\()'`
5. Validate chart JSON with `r'\[chart\](.*?)\[/chart\]'`

The only differences: one returns a dict, the other yields SSE events. They share ~80% of their logic.

**Impact:** A bug fix in one won't propagate to the other. The chart validation logic, citation normalization, and reasoning tag stripping all need to be maintained in two places.

**Remedy:** Extract the shared logic into a common function (e.g., `build_generation_messages()`, `normalise_answer()`, `validate_charts()`) in a shared module. Let both callers delegate to these utilities.

---

### 3.3 `_rewrite_query` in `chat_service.py` duplicates `rewrite_query_node` in `nodes.py` ✅ FIXED

| Field | Value |
|-------|-------|
| Files | `chat_service.py`, `nodes.py` |
| Severity | **MEDIUM** → ✅ **Resolved** |

**Issue:** Both implemented query rewriting with nearly identical logic and prompts.

**Status:** ✅ **Fixed.** A shared `rewrite_query()` utility was added to `utils.py`. Both `rewrite_query_node()` in `nodes.py` and `_rewrite_query()` in `chat_service.py` now delegate to this shared function.

**Result:** One implementation of query rewriting logic.

---

### 3.4 `_ANSWER_SYSTEM_PROMPT` in `nodes.py` vs inline system prompt in `chat_service.py`

| Field | Value |
|-------|-------|
| Files | `nodes.py` (system prompt, ~20 lines) and `chat_service.py` (inline system prompt, ~15 lines) |
| Severity | **LOW** — similar but not identical |

**Issue:** The answer prompt in `nodes.py` has detailed formatting/citation rules. The inline prompt in `chat_service.py` (used by the legacy `/api/query` endpoint) has a much simpler version. Both serve the same role — instructing the LLM to answer from context.

**Remedy:** Standardize on one system prompt, parameterized for different usage modes.

---

### 3.5 `_is_identity_question` logic duplicated across multiple paths

| Field | Value |
|-------|-------|
| Files | `chat_service.py` (`_is_identity_question`) |
| Severity | **LOW** — only one copy, but worth noting |

*Actually this is only in one place. Skip.*

---

### 3.6 `_summarise_older_messages` duplicates summarization logic

| Field | Value |
|-------|-------|
| Files | `chat_service.py` (`_summarise_older_messages`), `context_manager.py` (`_summarize_messages`), `nodes.py` (`summarize_history_node`) |
| Severity | **MEDIUM** — three implementations of conversation summarization |

**Issue:** All three functions:
1. Take a list of messages
2. Build a summarization prompt
3. Call the LLM to produce a summary
4. Return the summary text

They differ in formatting but the core logic is identical.

**Remedy:** Extract a single `_summarize_messages(messages, existing_summary)` utility in `infrastructure/`.

---

## 4. Complexity Issues

### 4.1 `Answer` component — 600+ lines, too many responsibilities

| Field | Value |
|-------|-------|
| File | `frontend/src/components/chat/answer.tsx` |
| Lines | ~700 |
| Severity | **HIGH** — violates single responsibility, hard to test/maintain |

**Issues:**
- Handles: thinking blocks, rewritten queries, retrieved context, graph context, query classification, tool traces, citations, confidence, export, copy, delete, markdown rendering, code blocks (mermaid/echarts)
- Defines 10+ inner components (`ThinkBlock`, `RewrittenQueryBlock`, `RetrievedContextBlock`, `RetrievedGraphBlock`, `QueryClassificationBlock`, `ToolTraceBlock`, `FailedLegsWarning`, `ConfidenceCollapsible`, `CitationLink`, `CodeBlock`)
- Complex regex parsing for reasoning tags (100+ lines of regex)
- Citation fetching with debouncing + ref indirection
- Export, copy, delete logic embedded

**Remedy:** Split into separate components:
```
components/chat/
  answer.tsx              → orchestrator, thin wrapper
  think-block.tsx         → reasoning visualization
  context-block.tsx       → retrieved context display
  citation-popover.tsx    → citation link + popover
  confidence-bar.tsx      → confidence visualization
  answer-actions.tsx      → copy/export/delete buttons
```

---

### 4.2 `graph_runner.py` — ~350 lines, monolithic pipeline

| Field | Value |
|-------|-------|
| File | `backend/app/services/agentic_rag/graph_runner.py` |
| Lines | ~400 |
| Severity | **MEDIUM** — complex control flow, hard to follow |

**Issues:**
- Both the "simple path" and "complex path" are implemented inline with deep nesting
- SSE event emission is duplicated per path
- The `for idx, subtask in enumerate(subtasks)` loop handles retrieval, generation, and event emission all in one block
- Hardcoded limits (8 iterations, 20 tool calls, 50-char chunks) are magic numbers

**Remedy:** Split into separate functions:
- `run_simple_path()` — direct retrieval + generate
- `run_complex_path()` — subtask loop with inner functions for retrieval/generation
- Extract `_stream_generate()` as a shared inner function

---

### 4.3 `chat_service.py` — ~500 lines, mixed concerns

| Field | Value |
|-------|-------|
| File | `backend/app/services/chat/chat_service.py` |
| Lines | ~500 |
| Severity | **MEDIUM** — identity check, summarization, classification, response generation all mixed |

**Issues:**
- Contains: LLM config, identity check, synthesis detection, query rewrite, query classification, main response generation, and summarization
- The `generate_response()` function is an async generator that handles user message persistence, file context injection, identity shortcut, knowledge base validation, sliding window, agentic pipeline orchestration, citation persistence, confidence persistence, cancellation handling, and background summarization — all in one function

**Remedy:** Split into:
- `llm_config.py` — `get_effective_llm_config()`
- `classification.py` — `_is_identity_question`, `_is_synthesis_query`, `classify_query()`
- `rewrite.py` — `_rewrite_query()`
- `summarize.py` — `_summarise_older_messages`, `_maybe_update_summary()`
- `response.py` — `generate_response()`

---

### 4.4 `config.py` — 150+ settings, no grouping

| Field | Value |
|-------|-------|
| File | `backend/app/core/config.py` |
| Lines | ~250 |
| Severity | **LOW-MEDIUM** — all settings as one class |

**Issue:** Every setting is a flat attribute. There's no grouping or hierarchy. With 60+ settings, finding a specific one requires scrolling through the entire file.

**Remedy:** Group related settings using nested dataclasses or separate config sections (e.g., `@dataclass class LLMConfig`, `@dataclass class RetrievalConfig`). This also enables per-section validation and better IDE autocomplete.

---

### 4.5 `graph_service.py` — 500+ lines, Neo4j operations spread across too many functions

| Field | Value |
|-------|-------|
| File | `backend/app/services/graph/graph_service.py` |
| Lines | ~550 |
| Severity | **MEDIUM** — deletion logic is especially convoluted |

**Issues:**
- `delete_graph_for_kb()` has 4 sequential Neo4j transactions plus a conditional second orphan sweep (steps 3-4)
- `purge_stale_graph_data()` has batched delete loops for chunks and entities
- Multiple Neo4j label queries use `__Entity__`, `Entity`, `__KGBuilder__` — three different labels for the same concept
- `_SafeNeo4jWriter` inner class adds unnecessary indirection

**Remedy:** Consolidate deletion logic into a single `purge_graph_for_kb()` function with clear sub-steps. Standardize on one entity label (the canonical one) with fallbacks as a last resort.

---

## 5. Code Quality and Maintainability

### 5.1 Magic numbers throughout

| Location | Value | Context |
|----------|-------|---------|
| `nodes.py` | `400` | Truncate AI messages to 400 chars for rewrite context |
| `nodes.py` | `8` | Orchestrator iteration limit |
| `nodes.py` | `20` | Tool call limit |
| `graph_runner.py` | `50` | Chunk size for streaming answer chars |
| `graph_runner.py` | `4` | Pool multiplier (top_k * 4) |
| `chat_service.py` | `3` | Sliding window pairs |
| `chat_service.py` | `20` | Max messages for summarization |
| `context_manager.py` | `15` | Max docs |
| `context_manager.py` | `10` | Min docs for greedy selection |
| `callbacks.py` | `2` | Poll interval (from config) |

**Remedy:** Extract named constants in a shared `constants.py` or at module level with descriptive names.

---

### 5.2 Error handling inconsistency

| Location | Issue |
|----------|-------|
| `nodes.py` `generate_node()` | Catches all exceptions, returns a fallback message string |
| `graph_runner.py` `_generate_streaming()` | Catches all exceptions, yields an error token |
| `retrieval.py` legs | Catches all exceptions, returns empty dict with error log |
| `graph_service.py` `enrich_docs_with_graph()` | Catches per-doc, returns doc unchanged |
| Various | Bare `except Exception` without context or re-raise |

**Issue:** Some nodes swallow exceptions silently, some return error messages inline, some return empty results. No consistent error taxonomy or handling policy.

**Remedy:** Define error types (e.g., `RetrievalError`, `GenerationError`) and a consistent pattern: log at warning level, return a structured error dict, let the caller decide how to present it.

---

### 5.3 Unused imports and stale comments

| File | Issue |
|------|-------|
| `main.py` | Imports `ProcessingTask` but only uses it for stuck-task reset; also imports `Organisation` but only in seeding |
| `graph_runner.py` | Comment references "StateState" (typo: "State") |
| Various | Comments like "TODO: things that look reasonable but will break this project" from AGENTS.md carry over |
| `chat_service.py` | `_SYNTHESIS_SYSTEM_PROMPT` references tool calls (`synthesize_documents`, `extract_entities`, `summarize_chunks`) that don't exist as LangChain tools in the current pipeline |

---

### 5.4 Frontend: `Answer` component excessive `useMemo`/`useCallback` nesting

| File | Issue |
|------|-------|
| `frontend/src/components/chat/answer.tsx` | `CitationLink` uses `useCallback` with empty deps `[]` and reads from refs — this is correct but hard to follow. Combined with the ref cascade pattern (`citationsRef` → `citationInfoMapRef`), the data flow is difficult to trace. |

**Remedy:** Use a context provider or Zustand store for citation state to eliminate the ref cascade.

---

### 5.5 Frontend: API calls in render loop (citation fetching)

| File | Issue |
|------|-------|
| `frontend/src/components/chat/answer.tsx` — `fetchCitationInfo` | Uses `useEffect` with `debouncedCitations` as dependency. Each render triggers new `Promise.all` calls. If streaming produces many rapid context events, this can fire multiple parallel fetch cycles. |

**Remedy:** Track which citation keys have already been fetched and skip duplicates. Use a debounce/lock pattern to prevent concurrent fetch cycles.

---

## 6. Proposed Simplification Plan

### Phase 1: Bug fixes and dead code removal (quick wins) — ✅ COMPLETE

| # | Action | Files | Status |
|---|--------|-------|--------|
| 1 | Add `import json` to `context_manager.py` | `context_manager.py` | ✅ Done |
| 2 | Keep `context_manager.py` for now (potential future LangGraph wiring) | `context_manager.py` | ⏭️ Deferred |
| 3 | Delete `Retriever`/`Generator` classes from `retry.py` | `retry.py` | ⏭️ Deferred |
| 4 | Delete `evaluator.py` + remove `ANSWER_QUALITY_GRADING_ENABLED` | `evaluator.py`, `config.py` | ⏭️ Deferred |
| 5 | Refactor `hybrid_search()` → delegate to `hybrid_search_with_legs()` | `retrieval.py`, `knowledge_base.py` | ⏭️ Deferred |
| 6 | Fix `_heuristic_classify` (always returned `[rewritten]`) | `nodes.py` | ✅ Done — removed, replaced with inline `[rewritten]` |
| 7 | Delete `LangGraphCallbackHandler` from `callbacks.py` | `callbacks.py` | ✅ Done |
| 10 | Remove dead `generate_node()` from `nodes.py` | `nodes.py` | ✅ Done |
| 10b | Remove dead `_select_model`, `_REWRITE_SYSTEM`, `_THINKING_KEYWORDS` | `nodes.py` | ✅ Done |
| 11 | Remove dead `summarize_history_node` from `nodes.py` | `nodes.py` | ✅ Done |
| 12 | Remove dead `LangGraphCallbackHandler`, unused imports from `callbacks.py` | `callbacks.py` | ✅ Done |
| 13 | Remove dead functions from `utils.py` (`extract_chat_id`, `truncate_to_words`, `safe_json_parse`, `extract_thinking_content`, `build_task_list_events`) | `utils.py` | ✅ Done |

### Phase 2: Deduplicate context building — ✅ COMPLETE

| # | Action | Files | Status |
|---|--------|-------|--------|
| 9 | Enhance `format_context_string(docs, file_markdown=None)` and use everywhere | `utils.py`, `nodes.py`, `graph_runner.py` | ✅ Done — 4 copies eliminated |
| 11 | Shared query rewrite via `utils.rewrite_query()` | `utils.py`, `chat_service.py`, `nodes.py` | ✅ Done — both `rewrite_query_node()` and `_rewrite_query()` now delegate |

### Phase 2b: LangGraph Migration — Phase 3 (Graph Compilation) — ✅ COMPLETE

| # | Action | Files | Status |
|---|--------|-------|--------|
| 20 | Create compiled `StateGraph` with two-level architecture | `graph.py` (new) | ✅ Done |
| 21 | Fix `orchestrator_node`/`direct_retrieval_node` missing deps in subgraph | `graph.py` | ✅ Done — `functools.partial` injection |
| 22 | Rewrite `graph_runner.py` to use `astream_events()` (LangGraph conventions) | `graph_runner.py` | ✅ Done — removed hacky `final_answer` accumulation, dead `__pregel_pull` handling, `yield_from_list` |
| 23 | Verify all nodes properly wired, no dead code remaining | `nodes.py`, `graph.py`, `graph_runner.py` | ✅ Done — all nodes used, all imports correct |

### Phase 3: Split large files (medium effort, high maintainability) — ⏭️ Deferred to later session

### Phase 4: Code quality improvements (lower priority) — ⏭️ Deferred to later session

---

## 7. Summary of Impact

### Changes Made (This Session)

| Change | Impact | Lines |
|--------|--------|-------|
| Bug fix: `import json` in `context_manager.py` | Prevents crash on `truncate_tool_output()` call | +1 |
| Bug fix: DB session closed before streaming | Prevents connection pool exhaustion under load | ~15 net added |
| Context string dedup: `format_context_string()` used everywhere | 4 copies removed, single source of truth | ~40 lines deleted |
| Query rewrite dedup: shared `rewrite_query()` in `utils.py` | 2 copies removed, single prompt/API call | ~80 lines deleted |
| Removed dead `generate_node()` from `nodes.py` | Never called — LangGraph graph never compiled | ~50 lines deleted |
| Removed dead summarization nodes from `nodes.py` | 3 functions never called (same LangGraph reason) | ~55 lines deleted |
| Removed unused imports from `nodes.py` | Clean dependency graph | 5 imports |

### Remaining Work (Deferred)

| Category | Count | Severity | Effort |
|----------|-------|----------|--------|
| Dead code | 4 modules/classes still unused | Medium-High | 30 min |
| Redundant implementations | ~3 (hybrid_search, summarization, system prompts) | Medium | 1 hour |
| Complexity / maintainability | 5 areas (Answer component, chat_service, config, graph_service) | Medium | ~4 hours |
| Code quality | Magic numbers, error handling consistency | Low | ~1 hour |

---

## Completed This Session — Full Change Log

### Bug Fixes
| Fix | Files | Impact |
|-----|-------|--------|
| Missing `json` import | `context_manager.py` | Prevents `NameError` at runtime |
| DB session closed before streaming | `chat_service.py` | Prevents connection pool exhaustion |
| `generate_node()` return-with-value syntax error | `nodes.py` | Fixes `SyntaxError: 'return' with value in async generator` |
| `orchestrator_node`/`direct_retrieval_node` missing deps in subgraph | `graph.py` | Fixes runtime `TypeError` when subgraph calls these nodes without required `llm`/`db` args |

### LangGraph Migration — Full Completion
| Change | Files | Impact |
|--------|-------|--------|
| Created compiled `StateGraph` with `functools.partial` dep injection | `graph.py` (new, ~180 lines) | Two-level architecture: main graph + agent subgraph, proper LangGraph conventions |
| Rewrote `graph_runner.py` to use `astream_events()` (v2) | `graph_runner.py` | No more hacky `final_answer` accumulation, dead `yield_from_list`, `__pregel_pull` handling |
| Removed `generate_node`, `summarize_history_node` from `nodes.py` | `nodes.py` | Dead code eliminated — not wired into any graph |
| Removed `_heuristic_classify`, `_select_model`, `_REWRITE_SYSTEM`, `_THINKING_KEYWORDS` | `nodes.py` | Dead code eliminated |
| Removed `LangGraphCallbackHandler` from `callbacks.py` | `callbacks.py` | Dead code eliminated |
| Removed dead functions from `utils.py` | `utils.py` | 5 unused functions removed |

### Redundancy Elimination
| Change | Files | Lines affected |
|--------|-------|---------------|
| Unified context building via `format_context_string()` | `utils.py`, `nodes.py`, `graph_runner.py` | ~40 lines deleted |
| Unified query rewrite via `utils.rewrite_query()` | `utils.py`, `nodes.py`, `chat_service.py` | ~80 lines deleted |
| Eliminated `_generate_streaming()` duplication | `graph_runner.py` | ~90 lines removed (replaced by `astream_events` token capture) |

### Verification
| Metric | Value |
|--------|-------|
| Tests passed | **329/329** |
| Tests skipped | 1 |
| Test failures | 0 |
| Syntax errors | 0 |
| Import errors | 0 |
| Dead code in `nodes.py` | 0 (all nodes either wired or removed) |
| Dead code in `callbacks.py` | 0 (LangGraphCallbackHandler removed) |
| Dead code in `utils.py` | 0 (5 unused functions removed) |

### What's Left (Deferred)
| Category | Count | Severity | Effort |
|----------|-------|----------|--------|
| Dead code — `Retriever`/`Generator` in `retry.py` | 2 classes | Low | 15 min |
| Dead code — `ContextManager` in `context_manager.py` | 1 class (240 lines) | Medium | 15 min |
| Dead code — `evaluator.py` unused | 1 module | Medium | 15 min |
| Dead code — `hybrid_search()` in `retrieval.py` | 1 function | Low | 15 min |
| Redundancy — `_summarise_older_messages` (3 implementations) | 3 functions | Medium | 1 hour |
| Complexity — Answer component (~700 lines), chat_service (~500 lines) | 2 areas | Medium | ~3 hours |
| Code quality — magic numbers, error handling consistency | Several | Low | ~1 hour |

**Total this session:** ~500 lines deleted, ~200 lines added (net ~-300 lines), 4 bug fixes, 11 dead code removals, full LangGraph migration, 329 tests passing.
**Remaining estimated effort:** ~4-5 hours for full cleanup.
