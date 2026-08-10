# Behavioural Transcript Suite — Running Log

A chronological record of changes made while building and iterating on the
behavioural transcript test suite for the enterprise agent.

---

## Session 1: Initial Build

### Goal
Build a behavioural transcript suite that runs the full agent graph with
mocked LLMs to prove multi-turn conversation quality (entity-addition rate,
topic carryover, unsupported-citation rate, clarification flow, multi-tool
plans, code execution + chart generation).

### Architecture Decision
- Run the **full agent graph** (`build_agent_graph`) with mocked LLMs
- LLM outputs are scripted per-node (plan, think, finalize, extract)
- Graph mechanics (routing, state propagation, reference resolution, tool
  dispatch, citation normalisation) are real
- `rag_retrieve` is mocked to return scripted docs per query
- `MemorySaver` checkpointer used for multi-turn state persistence
- Each transcript is a sequence of turns; assertions on final state after
  each turn

### Changes Made

#### 1. `_call_rewriter` switched to AsyncOpenAI
**File:** `backend/app/services/agentic_rag/utils.py`
- `resolve_retrieval_query` and `_call_rewriter` changed from `def` to `async def`
- `from openai import OpenAI` → `from openai import AsyncOpenAI`
- `client.chat.completions.create(...)` → `await client.chat.completions.create(...)`
- `rewrite_query_node` in `nodes.py` now `await`s `resolve_retrieval_query`
- 3 tests in `test_agent_state_integrity.py` updated to use `asyncio.run()`
  and `async def _boom` for the mock

**Why:** The synchronous OpenAI client was blocking the FastAPI event loop
inside an async node — same bug class as the compaction fix (C10) that was
already applied elsewhere.

#### 2. Removed assistant-turn truncation in `select_recent_history`
**File:** `backend/app/services/agentic_rag/nodes.py`
- Removed `assistant_cap = settings.COMPACTION_ASSISTANT_MAX_CHARS` and the
  `str(m.content)[:assistant_cap]` truncation
- Docstring updated to explain why: the resolver, think and finalize nodes
  need the full prior answer to resolve references and compare approaches
- `COMPACTION_ASSISTANT_MAX_CHARS` removed from `config.py` (orphaned)

**Why:** The compaction path was explicitly fixed to not truncate because
"the summarizer cannot preserve facts removed before it saw them." The same
logic applies to the resolver and think/finalize nodes.

#### 3. Removed dead GraphInterrupt handler in `agent_runner`
**File:** `backend/app/services/agentic_rag/agent_runner.py`
- Removed the entire `try/except` wrapper around the `async for` loop
- The `except` block's `GraphInterrupt` handler was dead code — `astream`
  emits `__interrupt__` updates, doesn't raise
- Body un-indented by 4 spaces

**Why:** Dead code that could confuse future readers into thinking `astream`
can raise `GraphInterrupt`.

#### 4. Cleaned up `generate_response` signature
**Files:** `backend/app/services/chat/chat_service.py`, `backend/app/api/api_v1/chat.py`
- Removed `temperature`, `model_name`, `api_base`, `query_model` from the
  function signature
- Removed the `api_base` log line
- Removed `temperature`/`model_name` extraction from request dict in `chat.py`
- Removed both `llm_cfg = get_effective_llm_config(...)` calls and the
  `api_base`/`query_model` args from both call sites
- Removed the now-unused `get_effective_llm_config` import from `chat.py`

**Why:** These parameters were accepted but never forwarded to
`run_agentic_rag` — silently ignored. The implementation doc's claim that
they were "removed and propagated through" was slightly overstated.

#### 5. Built behavioural transcript suite
**File:** `backend/tests/test_behavioural_transcripts.py` (new, 11 test cases)

**Test scenarios:**
1. `TestMultiTurnReferenceResolution` — pronoun "its" resolves to entity
   from prior turn
2. `TestTopicCarryover` — topics don't conflate across 3 turns
3. `TestClarificationInterruptResume` — interrupt → resume end-to-end
4. `TestMultiToolPlan` — 2 rag_retrieve subtasks both execute
5. `TestCodeExecuteChartGenerate` — code_execute then chart_generate
6. `TestEntityAdditionRate` — 3 turns accumulate correctly
7. `TestUnsupportedCitationRejection` — out-of-range citations stripped
   (3 sub-tests: unit, no-docs, end-to-end)
8. `TestObservationNonDuplicationAcrossTurns` — observations reset between
   turns
9. `TestPreviousAnswerAction` — summarize_answer uses last_answer_object

**Key infrastructure:**
- `_ScriptedLLM` — mocks LLM with per-role script queues (plan, think,
  finalize, extract). Detects role from system prompt content. Streams
  finalize responses word-by-word.
- `_setup_graph` — wires up full agent graph with mocked LLMs, mocked
  rag_retrieve, mocked answer_scoring, and MemorySaver checkpointer
- `_run_turn` / `_get_state` — helpers for running turns and inspecting
  checkpointed state

### Issues Found & Fixed During Iteration

#### Issue 1: No checkpointer set
**Problem:** `build_agent_graph` uses `ctx.redis_memory.checkpointer` which
was None in tests → `aget_state` raised "No checkpointer set".
**Fix:** Inject a `MemorySaver` via a mock `_MockRedisMemory` class on the
`ToolContext`.

#### Issue 2: LLM role detection failed
**Problem:** `_ScriptedLLM.ainvoke` checked for "THINK_SYSTEM_PROMPT" and
"Emit either" in the system prompt, but the system prompt contains the
*value* of the prompt, not the variable name. "Emit either" is in the user
prompt, not the system prompt.
**Fix:** Changed detection strings to match actual prompt content:
- Plan: `"Produce a plan JSON" in sys_content`
- Think: `"You are the acting module" in sys_content`
- Extract: `"Extract a structured summary" in sys_content`
- Finalize: fallback

#### Issue 3: Think queue leftovers across turns
**Problem:** The pre-think sufficiency check (`_verify_execution`) skips
the LLM call when the plan is already satisfied. This means the scripted
`{"final_answer": true}` think response is never consumed, leaving it in
the queue for the next turn's first think call — causing the wrong response
to be returned.
**Fix:** Removed all `{"final_answer": true}` think scripts — the pre-think
check handles plan completion without an LLM call. Also made `_pop("think")`
return `{"final_answer": true}` as default when the queue is empty, so any
stray think call finalizes instead of looping.

#### Issue 4: Content hash collision in mock docs
**Problem:** `_mock_docs` used `f"hash-{i}"` where `i` was the index within
each call. All single-doc calls produced `content_hash: "hash-0"`, causing
the `tool_node`'s content_hash-based deduplication to treat docs from
different rag_retrieve calls as duplicates — only 1 doc survived instead of 2.
**Fix:** Changed `_mock_docs` to use `hashlib.md5(content.encode()).hexdigest()[:12]`
as the content hash, making it unique per content.

#### Issue 5: Clarification test lost checkpointer
**Problem:** The clarification test called `_setup_graph` twice (once for
the initial turn, once for the resume), creating a new `MemorySaver` each
time. The resume couldn't find the checkpointed state from the first call.
**Fix:** Modified `_setup_graph` to reuse the existing checkpointer if
`ctx.redis_memory` already has one, instead of always creating a new one.

#### Issue 6: chart_generate not available without data
**Problem:** `applicable_tools` gates `chart_generate` behind
`has_data` (retrieved_docs or last_answer_object.data). In the
code_execute + chart test, there are no retrieved_docs and no
last_answer_object, so `chart_generate` was filtered out → "Tool not
available" error.
**Fix:** Mocked `applicable_tools` to return all tools (`build_tools(ctx)`)
for this test, since it verifies tool execution flow, not tool availability
gating. This also revealed a real issue: `applicable_tools` doesn't check
code_execute results, so `chart_generate` is unavailable after code_execute
produces data — a separate issue to address.

### Final Results
- **451 tests pass** (440 existing + 11 new behavioural transcript tests)
- 0 failures
- All changes verified inside the `rag-web-ui-backend-1` Docker container

### What the Tests Prove
1. **Entity-addition rate:** Conversation history accumulates exactly 1 user
   + 1 assistant message per turn, with no duplicates or losses across 3 turns
2. **Topic carryover:** Reference resolution doesn't leak entities from
   unrelated turns into the retrieval query
3. **Unsupported-citation rate:** `normalize_citations` strips citations
   outside the docs range, both in unit tests and end-to-end through
   `finalize_node`
4. **Clarification flow:** The interrupt → resume cycle works end-to-end,
   with `clarification_count` incremented and the answer using the
   clarification
5. **Multi-tool plans:** A plan with 2 rag_retrieve subtasks executes both,
   with docs from both calls merged into `retrieved_docs`
6. **Code execution + chart:** `code_execute` and `chart_generate` produce
   observations with valid results, and the chart_option is available in
   the final state
7. **Observation reset:** `load_context_node` correctly resets observations
   between turns — no leakage from prior turns
8. **Previous answer action:** `last_answer_object` from Turn 1 is available
   to Turn 2, and `summarize_answer` uses it

### Follow-ups Identified
- **`applicable_tools` doesn't check code_execute results** — after
  `code_execute` produces data, `chart_generate` is still unavailable
  because `has_data` only checks `retrieved_docs` and
  `last_answer_object.data`. This is a real issue that should be fixed
  separately.
- **Behavioural tests use scripted LLM responses** — they prove graph
  mechanics are correct but don't test LLM reasoning quality. A separate
  eval harness with a real LLM would be needed for that.
