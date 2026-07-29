# Agent Loop Pipeline Analysis

Complete topology, node-by-node detail, branches, and iterations of the enterprise agent loop.

## Complete Pipeline

```
START
  │
  ▼
┌─────────────────┐
│  load_context    │  Deterministic (DB + Redis queries)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  rewrite_query   │  LLM call (query model, temp=0)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  compaction      │  Deterministic (token-count trigger) + LLM call (if triggered)
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌────────────────────┐
│  plan            │────▶│  clarify_interrupt  │──┐
│  (LLM call)      │     │  (interrupt())      │  │
└────────┬────────┘     └────────────────────┘  │
         │                       ▲              │
         │ needs_clarification?  │              │
         │ No                    │              │
         ▼                       │              │
┌─────────────────┐              │              │
│  think           │◀─────────────┘──────────────┘
│  (LLM call)      │◀──────────────────────────────────┐
└────────┬────────┘                                    │
         │                                             │
         │ route_think                                 │
         ├─── tool_calls? ──▶ ┌─────────┐              │
         │                    │  tool    │             │
         │                    │ (dispatch)│             │
         │                    └────┬────┘              │
         │                         │                   │
         │                         ▼                   │
         │                    ┌─────────┐              │
         │                    │ reflect  │              │
         │                    │(every K) │              │
         │                    └────┬────┘              │
         │                         │                   │
         │                         └──────────────────▶│
         │                                             │ (loop back to think)
         │                                             │
         │ no tool_calls OR iteration >= MAX           │
         ▼                                             │
┌─────────────────────┐                                │
│  reflect_final       │◀──────────────────────────────┘
│  (DETERMINISTIC)     │
└────────┬────────────┘
         │
         │ route_reflect_final
         ├─── ready=false AND iteration < MAX ──▶ think (loop back)
         │
         │ ready=true OR iteration >= MAX
         ▼
┌─────────────────┐
│  finalize        │  LLM call (chat model, temp=0.7, streaming=True)
│  (stream tokens) │  + LLM call (query model, extract LastAnswerObject)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  answer_scoring  │  LLM call (query model, evaluate faithfulness/completeness)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  save_memory     │  Deterministic (DB writes)
└────────┬────────┘
         │
         ▼
       END
```

## Node-by-node detail

### 1. load_context — Deterministic

**Purpose:** Load prior context into state before planning.

**Inputs (from state):** `original_query`, `chat_id`, `message_id`, `user_id`, `org_id`

**What it does:**
- Queries DB for the previous assistant message's `last_answer_object` (structured summary of the last answer — used for "summarize it" / "chart it" follow-ups)
- Queries Redis long-term memory store for top-3 semantically related past turns
- Queries DB for attached file metadata (names, types — not content)

**Outputs (to state):** `last_answer_object`, `retrieved_docs` (recalled memory snippets), `org_id`, `user_id`, `chat_id`, `message_id`

**SSE events:** `4: agent_step` (start/done)

---

### 2. rewrite_query — LLM call (query model, temp=0)

**Purpose:** Resolve pronouns and references from chat history into a self-contained search query.

**Inputs:** `original_query`, `messages` (recent history, max 3 pairs)

**What it does:**
- Calls the query-tier LLM with a strict rewrite prompt (8 rules)
- Rule 8: if the query is already self-contained, return it EXACTLY as-is
- Rule 6: do NOT add synonyms, related terms, or background concepts the user didn't mention
- Max 60 output tokens

**Outputs:** `rewritten_query`

**SSE events:** `1: rewritten_query`, `4: agent_step`

---

### 3. compaction — Deterministic trigger + LLM call (if triggered)

**Purpose:** Prevent context window overflow by summarizing old messages.

**Inputs:** `messages` (full conversation history)

**What it does:**
- Counts tokens across all messages using the HF tokenizer
- If total exceeds `CONTEXT_WINDOW - RESERVED_FOR_GENERATION - TOOL_BUDGET`, triggers compaction
- Compaction: LLM summarizes oldest messages into a structured summary (Goal, Topics, Decisions, Retrieved Docs, Progress, Critical Context, Next Steps)
- Recent messages (sliding window) are preserved as-is

**Outputs:** `messages` (compacted), `compaction_summary`

**SSE events:** `4: agent_step`

---

### 4. plan — LLM call (query model, temp=0, structured output)

**Purpose:** Classify intent and decompose the query into subtasks. Replaces the old one-shot classifier.

**Inputs:** `original_query`, `rewritten_query`, `last_answer_object` (summary), recalled memory, attached file metadata

**What it does:**
- Calls LLM with `PLAN_SYSTEM_PROMPT` + `AGENT_SYSTEM_PROMPT`
- Uses structured output (`with_structured_output(Plan, method="json_schema")`) with JSON parse fallback
- Produces a `Plan` object: `intent` (rag/file_action/previous_answer_action/computation/chart/conversation/mixed), `subtasks` (each with id, description, tool_hint, depends_on, expected_output), `needs_clarification`, `clarification_question`

**Outputs:** `plan` (Plan object), `needs_clarification`

**SSE events:** `pl: plan`, `4: agent_step`

**Routing after plan (`route_plan`):**
- `needs_clarification=true` → `clarify_interrupt` (interrupts, waits for user, then back to `plan`)
- `needs_clarification=false` → `think`

---

### 5. clarify_interrupt — Deterministic (LangGraph interrupt)

**Purpose:** Ask the user a clarifying question when intent is genuinely ambiguous.

**Inputs:** `plan.clarification_question`

**What it does:**
- Calls `interrupt()` with the clarification question
- Graph pauses, frontend polls for pending clarifications
- When user responds, graph resumes with the user's answer appended to messages

**Outputs:** `messages` (with user's clarification response), `needs_clarification=false`

**SSE events:** `interrupt`

---

### 6. think — LLM call (chat model, temp=0.7) — THE CENTRAL DECISION NODE

**Purpose:** Decide the next action — call a tool or signal readiness to answer.

**Inputs (from state):**
- `original_query`, `rewritten_query`
- `plan` (the structured plan from plan_node)
- `observations` (all tool results so far, formatted with `_observations_text(full=True)`)
- `messages` (recent 3 turns for multi-turn context)
- `last_answer_object` (for "summarize it" / "chart it" follow-ups)
- `reflection_final` (if reflect_final sent us back — includes reasoning about what's missing)
- `iteration` (current loop count), `max_iter` (AGENT_MAX_ITERATIONS, default 8)
- Available tools (filtered by context — no file tools if no file attached)

**What it does:**
1. Increments iteration counter
2. If `force_finalize` is set → return empty tool_calls and empty precomputed (routes to reflect_final)
3. If `precomputed_tool_calls` exist (from reflect_node's deterministic rules) → return them directly, skip LLM call
4. Otherwise, calls LLM with `THINK_SYSTEM_PROMPT` + `AGENT_SYSTEM_PROMPT`
   - Tries native function-calling first (`bind_tools(tools).ainvoke()`)
   - Falls back to JSON-text parsing (`{"tool_calls": [...]}` or `{"final_answer": true}`)
5. Parses response via `parse_think_response()`:
   - **Tier 1:** Native `tool_calls` in the response → use directly
   - **Tier 2:** JSON text with `{"tool_calls": [...]}` or `{"final_answer": true}` → parse
   - **Tier 3:** Plain text (no JSON, no tool calls) → treat as final answer text (fallback for gateways that don't support function-calling or JSON mode)

**Outputs (to state):**
- If tool calls: `{"iteration": N, "tool_calls": [...]}`
- If final_answer=true (boolean signal): `{"iteration": N, "tool_calls": [], "precomputed_answer": ""}` — finalize will generate the answer with streaming
- If final_answer is a string (Tier 3 fallback): `{"iteration": N, "tool_calls": [], "precomputed_answer": "the text"}` — finalize uses this text directly (no streaming)
- If LLM error: `{"iteration": N, "tool_calls": [], "precomputed_answer": "LLM error: ..."}`

**SSE events:** `4: agent_step`

**Routing after think (`route_think`):**
- `iteration >= AGENT_MAX_ITERATIONS` → `reflect_final` (forced)
- `tool_calls` non-empty → `tool`
- `tool_calls` empty (final_answer signal) → `reflect_final`

---

### 7. tool — Deterministic dispatcher (tools may contain LLM calls internally)

**Purpose:** Execute the tool calls from think_node, collect observations.

**Inputs:** `tool_calls` (list of {tool, arguments}), `tool_call_count` (per-tool usage counters), `observations` (accumulated)

**What it does:**
1. For each tool call:
   - Check per-tool budget (`AGENT_MAX_RETRIEVALS=3`, `AGENT_MAX_CODE_EXEC=3`) — if exceeded, return error observation
   - Stream `tc:` event (tool name + args)
   - Execute the tool asynchronously (tools run in parallel via `asyncio.gather`)
   - Each tool internally: enforce RBAC → execute → write audit row → return structured result
2. Collect all results as `Observation` objects
3. Update `tool_call_count` per tool

**Available tools:**
- `rag_retrieve` — 3-leg hybrid retrieval (dense/sparse/exact) + reranking + filter + Neo4j graph expansion + adaptive reranking. Returns `{docs, confidence, confidence_level, sufficient}`. Internally calls embedding API, Qdrant, MySQL FTS, reranker model, Neo4j.
- `file_read` — Read attached file markdown, optionally a section
- `file_summarize` — Map-reduce chunked summarization (LLM call)
- `file_extract_table` — Extract tables from CSV/Excel/HTML
- `code_execute` — RestrictedPython sandbox
- `chart_generate` — ECharts option builder (deterministic)
- `summarize_answer` — Summarize last_answer_object or cited prior turn (LLM call)
- `extract_data` — Extract numbers/stats from last_answer/retrieved_docs/file (LLM call)

**Outputs:** `observations` (accumulated list), `tool_calls` (cleared), `tool_call_count` (updated), `retrieved_docs` (if rag_retrieve ran)

**SSE events:** `tc: tool_call`, `to: tool_observation`, `2: context` (retrieved docs), `4: agent_step`

---

### 8. reflect — LLM call (query model, temp=0) + deterministic rules

**Purpose:** Mid-loop recovery. Runs every `AGENT_REFLECT_EVERY` iterations (default 2).

**Inputs:** `iteration`, `observations`, `tool_call_count`, `rewritten_query`, `original_query`, `plan`

**What it does:**
1. **Deterministic check:** if `iteration == 0` or `iteration % AGENT_REFLECT_EVERY != 0` → return empty (skip)
2. **Concrete replanning rules (deterministic):**
   - `rag_retrieve` returned 0 docs + retrieval budget remaining → precompute a retry with lower `min_confidence`
   - `chart_generate` errored → precompute `extract_data` call
   - `code_execute` errored → precompute `extract_data` call
3. **If deterministic rules triggered:** return `precomputed_tool_calls` (think_node will execute them without an LLM call)
4. **If no deterministic rule triggered:** LLM call with `REFLECT_SYSTEM_PROMPT` to decide continue vs finalize

**Outputs:** `precomputed_tool_calls` (if deterministic rules triggered), or `reflection` dict (if LLM call made)

**SSE events:** `4: agent_step`

**Edge:** Always routes back to `think`

---

### 9. reflect_final — DETERMINISTIC (no LLM call)

**Purpose:** Final pre-finalize verification. Checks structural execution completeness — did the agent do what it planned to do?

**Inputs (from state):** `plan` (subtasks), `observations` (tool results), `tool_call_count`, `iteration`, `original_query`

**What it does:**

1. **`_build_execution_summary(state)`** — builds a structured summary:
```json
{
  "user_goal": "what is mutex?",
  "intent": "rag",
  "subtasks": [
    {"id": "a", "description": "Explain mutex", "tool_hint": "rag_retrieve", "completed": true}
  ],
  "retrieval": {"queries": 1, "documents": 29},
  "tool_failures": [],
  "remaining_budget": {"retrieval": 2, "iterations": 7}
}
```

2. **`_verify_execution(summary)`** — applies deterministic checks:
   - **Check 1:** No observations + non-conversation intent → not ready ("No tool calls were made")
   - **Check 2:** Uncompleted plan subtasks (subtask's tool_hint has no matching successful observation) → not ready
   - **Check 3:** Retrieval returned 0 docs + retrieval budget remaining → not ready
   - **Check 4:** Tool failures (non-budget-exceeded) → not ready
   - All checks pass → ready=True, "All planned steps have supporting tool results."

3. **Force finalize:** if `not ready AND iteration >= AGENT_MAX_ITERATIONS` → ready=True (can't retry, ship what we have)

**Outputs:** `reflection_final: {ready: bool, reasoning: str}`

**SSE events:** `p: progress` (phase=reflect_final, ready, reasoning)

**Routing (`route_reflect_final`):**
- `ready=true` OR `iteration >= MAX` → `finalize`
- `ready=false` AND `iteration < MAX` → `think` (agent gets another chance with the reasoning injected into its prompt)

---

### 10. finalize — LLM call (chat model, temp=0.7, streaming=True) + LLM call (query model, extract LastAnswerObject)

**Purpose:** Generate the final answer and stream it token-by-token to the frontend.

**Inputs:** `precomputed_answer` (if set by think Tier 3 fallback), `original_query`, `retrieved_docs`, `file_markdown`, `observations`

**What it does:**
1. If `precomputed_answer` is non-empty → use it directly (no streaming — this is the Tier 3 fallback or error path)
2. Otherwise → call chat LLM with `streaming=True`:
   - System: `AGENT_SYSTEM_PROMPT` + `ANSWER_SYSTEM_PROMPT_BASE` + "You are the final answer synthesizer"
   - User: query + retrieved context + tool observations
   - Stream each chunk via `writer({"event": "token", "content": content})` → frontend sees word-by-word
3. After generation, extract `LastAnswerObject` via a second LLM call (query model, temp=0):
   - Structured summary: summary, key_points, data, citations, chart_option, followups
   - 2 attempts with JSON parse, then falls back to rule-based extraction
4. If a `chart_generate` observation exists, preserve its `chart_option` in the LastAnswerObject

**Outputs:** `final_answer`, `answer`, `last_answer_object`, `retrieved_docs`

**SSE events:** `0: token` (word-by-word streaming), `la: last_answer`, `4: agent_step`

---

### 11. answer_scoring — LLM call (query model)

**Purpose:** Evaluate answer quality for the confidence UI.

**Inputs:** `answer` (final answer text), `original_query`, `retrieved_docs`, `retrieval_confidence`

**What it does:**
- Calls `evaluate_answer()` which asks the LLM to score:
  - `faithfulness` (0-100): is every claim supported by retrieved docs?
  - `completeness` (0-100): does the answer address all parts of the query?
- Computes `final_confidence = 0.4 * retrieval_score + 0.3 * faithfulness + 0.3 * completeness`
- Maps to confidence_level: very_high/high/medium/low/none

**Outputs:** `final_confidence`, `confidence_level`, `faithfulness`, `completeness`, `retrieval_score`

**SSE events:** `evaluation`, `4: agent_step`

---

### 12. save_memory — Deterministic (DB writes)

**Purpose:** Persist the answer and metadata to the database.

**Inputs:** `message_id`, `final_answer`, `plan`, `last_answer_object`, `observations`, confidence fields

**What it does:**
- Updates the Message row with: content, plan, last_answer_object, tool_calls (observations), final_confidence, confidence_level, faithfulness, completeness, retrieval_score
- Commits to DB

**Outputs:** `{}` (no state changes)

**SSE events:** `4: agent_step`

---

## Branches and iterations

### Branch 1: Clarification loop
```
plan → (needs_clarification=true) → clarify_interrupt → plan → (needs_clarification=false) → think
```
The plan node detects ambiguous intent (e.g., two files attached + "summarize this"). It interrupts the graph, the frontend shows a clarification dialog, the user responds, and plan re-runs with the clarification. This is a single retry — plan either gets a clear intent or proceeds with best guess.

### Branch 2: Tool loop (the main ReAct loop)
```
think → (tool_calls) → tool → reflect → think → (tool_calls) → tool → reflect → think → ...
```
This is the core agent loop. Each iteration:
1. think decides which tool to call
2. tool executes it and collects the observation
3. reflect runs every K iterations (default 2) and may inject deterministic recovery rules
4. think sees the new observation and decides the next action

The loop is bounded by `AGENT_MAX_ITERATIONS=8`. Per-tool budgets also apply:
- `AGENT_MAX_RETRIEVALS=3` — max rag_retrieve calls
- `AGENT_MAX_CODE_EXEC=3` — max code_execute calls

When a budget is exceeded, the tool_node returns an error observation ("Budget exceeded") instead of executing. The think LLM sees this and should stop calling that tool.

### Branch 3: Final answer → verification loop
```
think → (final_answer=true) → reflect_final → (ready=false) → think → (final_answer=true) → reflect_final → (ready=true) → finalize
```
When think emits `final_answer=true`, it's not done yet — reflect_final checks structural completeness. If a subtask has no tool result, or a tool failed, or retrieval returned nothing, reflect_final sends the agent back to think. The reasoning is injected into think's prompt so the LLM knows what's missing.

This loop is bounded by the same `AGENT_MAX_ITERATIONS` counter. When the cap is hit, reflect_final forces `ready=true` and the agent ships the best answer it has.

### Branch 4: Forced finalize at iteration cap
```
think → (iteration >= MAX) → reflect_final → (forced ready=true) → finalize
```
Regardless of what think emits (tool calls or final answer), if `iteration >= AGENT_MAX_ITERATIONS`, `route_think` routes to `reflect_final`. reflect_final sees the cap and forces `ready=true`. The answer is generated from whatever observations exist.

### Branch 5: Precomputed tool calls (from reflect_node)
```
tool → reflect → (deterministic rule triggers) → think → (sees precomputed_tool_calls) → tool → reflect → think
```
When reflect_node's deterministic rules trigger (e.g., empty retrieval), it sets `precomputed_tool_calls` in state. The next think_node iteration sees these and returns them directly without an LLM call. This is a fast path — the recovery action is deterministic, so no need to ask the LLM what to do.

### Branch 6: Tier 3 fallback (no streaming)
```
think → (LLM writes plain text, no JSON, no tool calls) → reflect_final → finalize (uses precomputed text, no streaming)
```
When the LLM gateway doesn't support function-calling or JSON mode, the think LLM's raw text response is treated as the final answer. `parse_think_response` Tier 3 returns the text as `final_answer`. think_node sets `precomputed_answer` to this text. finalize_node sees `precomputed_answer` is non-empty and uses it directly — no streaming LLM call. The answer appears all at once via the `r: answer_rewrite` event.

This is the fallback path. The normal path (Tier 1 or 2) emits `final_answer=true` (boolean), precomputed_answer stays empty, and finalize generates with streaming.

---

## The think_node's role in detail

The think_node is the central decision-maker. Every iteration, it answers one question: **"Given what I know now, what should I do next?"**

Its prompt contains:
- The user's original query
- The plan (intent + subtasks with tool hints)
- All observations so far (formatted by `_observations_text`)
- Recent conversation history (3 turns)
- The last_answer_object (for follow-up queries)
- Verification feedback (if reflect_final sent it back)
- Available tools (filtered by context)
- Iteration counter (e.g., "Iteration: 3/8")

The LLM's output is either:
- `{"tool_calls": [{"tool": "rag_retrieve", "arguments": {...}}]}` — call a tool
- `{"final_answer": true}` — signal readiness to answer (finalize will generate)

### Impact of `_observations_text(full=True)` with full doc content + cross-observation dedup

**Before the fix:** observations were truncated to 500 chars of raw JSON. The think LLM saw:
```
Observation 1: tool=rag_retrieve args={'query': 'what is mutex?'}
  result: {"docs": [{"page_content": "A mutex (short for mutual exclusion) is a synchronization
```
— just the first 500 chars of a JSON blob containing 29 docs. The LLM couldn't tell if the docs actually answered the question. It would call rag_retrieve again, hitting the retrieval cap, then emit final_answer with low confidence.

**After the fix:** observations include the complete page_content of all docs per observation, deduplicated across observations by content_hash:
```
Observation 1: tool=rag_retrieve args={'query': 'what is mutex?'}
  doc_count=29 unique_so_far=29 confidence=0.71
  doc_1: A mutex (short for mutual exclusion) is a synchronization primitive used to protect shared resources in concurrent programming. It ensures that only one thread can access a critical section at a time. The mutex has two operations: acquire (lock) and release (unlock). When a thread acquires the mutex, other threads that try to acquire it are blocked until it is released. Mutexes can be implemented at the hardware level using atomic instructions or at the OS level using system calls...
  doc_2: In operating systems, a mutex lock is implemented as a binary variable that can be in one of two states: locked or unlocked. The Pthreads library provides pthread_mutex_lock and pthread_mutex_unlock operations...
  ...
  doc_29: ...

Observation 2: tool=rag_retrieve args={'query': 'mutex semaphore differences'}
  doc_count=22 unique_so_far=35 confidence=0.65
  doc_30: The key difference between a mutex and a semaphore is that a mutex...
  doc_31: Semaphores use a counter to allow multiple threads to access a resource...
  ...
```

The LLM sees all retrieved chunks — full content, no truncation, no duplicates. It can judge: "Does this explain mutex? Yes — doc_1 defines it, doc_2 explains the implementation." It emits `final_answer=true` on the first iteration instead of retrying.

**Why full docs and not truncated previews:** The ingestion pipeline uses `CHUNK_SIZE=1500` characters. At 1500 chars per chunk, 29 docs = 43,500 chars ≈ ~10,900 tokens — well within the 131k context window. An earlier version capped previews at 800 chars based on an incorrect assumption of ~5000 chars per doc; that was wrong by 3.3x. At 1500 chars, the 800-char cap was truncating each chunk at 53% of its content, cutting off the LLM mid-definition. Passing the full 1500-char chunk gives the LLM the complete unit of text that was indexed and reranked — no information loss.

**Why all docs and not a 10-doc cap:** With cross-observation dedup, the worst case (3 rag_retrieve calls returning the same 29 docs) is 29 unique docs = ~10.9k tokens. If the agent uses different queries across calls (the intended use case), each call may return different chunks — capping at 10 would discard relevant results from later calls. The `_compact_if_needed` helper handles overflow if unique docs accumulate beyond the context budget.

**Cross-observation dedup:** Each rag_retrieve call already deduplicates within its own results (merge_node deduplicates across dense/sparse/exact legs by content_hash). But when the agent makes multiple rag_retrieve calls, the same chunks can appear in multiple observations — especially if the queries are similar. Without cross-observation dedup, the think LLM sees the same chunk 2-3 times, wasting context tokens and potentially skewing its judgment. The `_observations_text(full=True)` function now deduplicates by content_hash across all observations, showing each unique chunk only once with a running `unique_so_far` counter.

**Cross-observation dedup in tool_node:** The tool_node also now merges and deduplicates all rag_retrieve observations into `retrieved_docs` (graph state), so finalize_node, answer_scoring, extract_data, and the citations payload see the full deduplicated set — not just the first call's docs (which was a bug: the old code used `break` after the first rag_retrieve observation, discarding all subsequent calls' docs).

**Impact on the loop:**
- "what is mutex?" — was 3 retrievals + 8 iterations + iteration cap forced finalize. Now: 1 retrieval + 1 think iteration + 1 reflect_final (ready=True) + finalize. Completeness improved from 60 to 85, confidence from 0.524 to 0.689.
- Complex queries with format requirements ("explain mutex and semaphore, give 5 differences") — the LLM can see whether the docs contain comparison content and decide to retrieve again with a different query or proceed to answer.
