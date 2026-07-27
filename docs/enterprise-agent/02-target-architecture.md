# 02 — Target Architecture

The agent loop that replaces the rigid pipeline. This doc defines topology, intent routing, the tool registry, memory/context model, and SSE protocol v4. Tool contracts are in `03-tool-specifications.md`; file-level changes are in `04-implementation-plan.md`.

**Design principle (from `agentic-architecture-v2.md`, retained): wrap, don't rewrite.** The retrieval, reranking, memory, evaluation, and streaming infrastructure stays. The graph topology is what changes.

---

## 1. Agent loop topology

### 1.1 Current (rigid sequence)

```
START → rewrite → compaction → classify → route_by_dependencies
      → [agent_subgraph | chat_subgraph | file_context_subgraph | sequential_loop]
      → prepare_final_context → generating → [chart_validation] → answer_evaluation
      → finalize → save_memory → END
```

The classifier commits to a plan up front. Retrieval runs once per subtask. Generation runs once. Evaluation can trigger one retry. There is no observe-and-replan.

### 1.2 Target (tool-calling loop)

```
START → load_context → plan → agent_loop → finalize → save_memory → END
                       ↑   ↓
                    (max N iterations)
```

Where `agent_loop` is:

```
agent_loop:
  1. think   — LLM sees: AGENT_SYSTEM_PROMPT (guardrail), conversation context, plan,
              tool registry, observations so far. Emits either:
                (a) one or more tool calls (structured), or
                (b) a "final_answer" signal.
              Tool-call parsing tries native function-calling first; on failure
              falls back to constrained-JSON-text parsing (see §3.2).
  2. dispatch — route each tool call to its tool node. Independent calls run in
              parallel (see §6). Stream progress/thinking + tc: events.
  3. observe  — each tool returns a structured result; appended to observations.
              Per-tool RBAC re-check runs before execution (see §5). Audit row
              written for every call (see §5).
  4. reflect  — (every K iterations, and as the final pre-finalize pass) LLM
              evaluates: is the plan on track? Did the last tool fail? Apply
              concrete replanning rules (see §7). Emits plan patch or continue.
  → loop back to think.
```

**Budgets** (configurable, env-driven):
- `AGENT_MAX_ITERATIONS` (default 8) — hard cap on think-act-observe cycles.
- `AGENT_MAX_RETRIEVALS` (default 3) — cap on `rag_retrieve` calls per turn (prevents retrieval loops).
- `AGENT_MAX_CODE_EXEC` (default 3) — cap on `code_execute` calls.
- `AGENT_MAX_REFLECTIONS` (default 2) — cap on explicit reflect steps.
- Per-tool token budget on observations (see `05-context-memory.md`).

**Why a loop, not a bigger DAG:** the whole point of agency is that the next action depends on the last observation. A DAG encodes that as ever-more-conditional edges until the graph is unreadable. A loop with a tool-calling LLM is the standard pattern (ReAct, OpenAI function-calling, LangGraph `ToolNode`).

### 1.3 LangGraph implementation

Use LangGraph's native tool-calling pattern, not a hand-rolled loop:

- `AgentState` (extended from current `graph_state.py`) adds: `plan`, `observations`, `tool_call_count`, `iteration`, `last_answer_object`.
- `think_node` — calls `ChatOpenAI.bind_tools(tool_registry).astream(...)`, parses the tool-call or final-answer signal.
- `tool_node` — a `ToolNode`-style dispatcher that runs the called tool and writes the result to `observations`.
- `reflect_node` — runs every K iterations (conditional edge on `iteration % K == 0`).
- Conditional edge after `think_node`: if tool call → `tool_node`; if final answer → `finalize`; if `iteration >= AGENT_MAX_ITERATIONS` → `finalize` (force).
- `interrupt()` preserved for clarification (now callable from any tool, not just the classifier).

The existing retrieval nodes (`dense_retrieval_node`, `sparse_retrieval_node`, `exact_retrieval_node`, `merge_node`, `reranking_node`, `filter_node`, `neo4j_expansion_node`) become the *internals* of the `rag_retrieve` tool, not top-level graph nodes. The `agent_subgraph` and `sequential_subtask_loop` are deleted; subtask decomposition becomes a planning step inside `plan_node`, and subtasks are executed as sequential tool calls within the loop.

---

## 2. Intent router (replaces one-shot classifier)

The current `classify_query_node` runs once and commits. The new `plan_node` runs once at turn start *and* can be re-invoked by `reflect_node`. It produces a `Plan`:

```python
class Plan(BaseModel):
    intent: Literal["rag", "file_action", "previous_answer_action",
                    "computation", "chart", "conversation", "mixed"]
    subtasks: list[Subtask]
    needs_clarification: bool
    clarification_question: str | None

class Subtask(BaseModel):
    id: str
    description: str
    tool_hint: Literal["rag_retrieve", "file_read", "file_summarize",
                       "file_extract_table", "code_execute", "chart_generate",
                       "summarize_answer", "extract_data", "any"]
    depends_on: list[str]          # subtask ids whose results this needs
    expected_output: str           # what the agent expects to get back
```

**Inputs to the planner** (all injected into the planning prompt):
1. Current user query.
2. Rewritten standalone query (existing `rewrite_query_node` stays — it is a cheap, high-value step).
3. Last `last_answer_object` (see §4) — the structured previous answer.
4. Recent conversation turns (sliding window, see `05-context-memory.md`).
5. Attached file metadata (names, types, sizes, token counts) — *not* full content.
6. Available tools (registry summary).
7. Long-term memory recall (top 3 semantically related past turns from Redis store).

**Intent disambiguation rules** (encoded in the planning prompt, not hard-coded):
- "it" / "this" / "that" + no attached file + last answer exists → `previous_answer_action`.
- "this file" / "the document" + attached file exists → `file_action`.
- Numbers/statistics requested + KB has structured data → `computation` + `rag_retrieve`.
- "chart" / "pie" / "bar" / "visualize" + prior data in conversation → `chart` referencing last answer's data.
- Multiple distinct questions → `mixed` with multiple subtasks.

The planner is allowed to emit `needs_clarification=True` when intent is genuinely ambiguous (e.g., two files attached and user says "summarise this"). This triggers `interrupt()` exactly as today.

---

## 3. Tool registry, tool-calling, and guardrails

### 3.1 Tool registry

All tools are offline. Each is a LangChain `BaseTool` subclass with `arun` (async), structured input schema, SSE progress emission, per-tool RBAC re-check, and audit logging. Full contracts in `03-tool-specifications.md`.

| Tool | Wraps / adds | When the agent calls it |
|---|---|---|
| `rag_retrieve` | Existing 3-leg retrieval + reranking + confidence + Neo4j expansion (graph_expand flag) | Needs facts/evidence from KB; entity/relationship questions |
| `file_read` | New: read attached file markdown, optionally a section | "Summarise this file", "show me section 3" |
| `file_summarize` | New: map-reduce chunked summarization | Large file summarization (overflows 25% budget) |
| `file_extract_table` | New: extract tables from CSV/Excel/HTML in attached file | "Give me the data in the table" |
| `code_execute` | New: RestrictedPython sandbox | Computation, data transform, stats |
| `chart_generate` | New: data → ECharts option builder (deterministic) | "Make it a pie chart" with data in hand |
| `summarize_answer` | New: summarize the `last_answer_object` or a cited prior turn | "Summarise it in 10 points" |
| `extract_data` | New: pull numbers/stats from `last_answer_object`, fresh retrieved docs, or file content | "Give me key statistics" → feed into `chart_generate` |
| `clarify` | Existing `interrupt()` mechanism | Genuinely ambiguous mid-execution |

**Pruned from earlier draft:** `graph_retrieve` (merged into `rag_retrieve` via `graph_expand`), `memory_recall` (proactive recall in `load_context_node` covers multi-turn; explicit tool added selection ambiguity without changing outcomes), `table_generate` (LLM-emitted markdown tables cover the example; add back only if LLM tables prove unreliable in practice).

**Tools explicitly NOT added (offline constraint):** web search, web fetch, email send, any external API call. The agent operates only on KB + attached files + conversation history + local compute.

**Tool selection by the LLM:** the planner emits `tool_hint` per subtask, but the `think_node` LLM makes the final call each iteration. The hint is guidance, not a constraint — the agent may call a different tool if observations suggest it. Tools not applicable to a turn (e.g., no file attached → `file_*` tools) are filtered out before binding so the LLM cannot call them.

### 3.2 Tool-calling with offline-LLM fallback

`think_node` parses the LLM's response into tool calls. Native OpenAI function-calling (`ChatOpenAI.bind_tools()`) is tried first, but many local gateways/models do not support the `tools` parameter reliably. The parser has three tiers:

1. **Native function-calling** — if the gateway returns `tool_calls` in the assistant message, use them directly. Fast path when supported (LM Studio with a capable model, some vLLM configs).
2. **JSON-text fallback** — if no native tool calls, parse the assistant's text content for a JSON object matching `{"tool": "<name>", "arguments": {...}}` or `{"final_answer": "..."}`. The `THINK_SYSTEM_PROMPT` instructs the model to emit this format when native tool-calling is unavailable. Use JSON mode (`response_format={"type": "json_object"}`) if the gateway supports it; else extract the first JSON block from the text and parse with retry on failure.
3. **Final-answer default** — if neither tool call nor final-answer signal is parsed, treat the assistant text as the final answer (the LLM chose to answer directly). This is the safe fallback — it never leaves the agent stuck.

A `TOOL_CALL_MODE` env var (`auto` | `native` | `json_text`) lets the operator force a mode if one is known to work for their gateway. `auto` (default) tries native then falls back. The chosen mode is logged per turn for observability.

### 3.3 Unified agent guardrail prompt

A single `AGENT_SYSTEM_PROMPT` is prepended to every `think_node` (and `plan_node`, `reflect_node`) call. It enforces:

- **Offline-only:** "You have no internet access. You cannot search the web, fetch URLs, or claim information from outside the knowledge base, attached files, or this conversation. If you do not know and cannot find it in your tools, say so."
- **Cite-or-refuse:** "If you state a fact from the knowledge base, it must be traceable to a retrieved chunk. If retrieval returned nothing relevant, say the KB does not contain the answer — do not fabricate."
- **Tool-use bias:** "When the user asks for facts, statistics, file content, charts, or computations, use the appropriate tool rather than answering from memory. Your memory is the conversation, not the world."
- **Bounds:** "You may only read files the user attached to this chat and knowledge bases the user's organization is entitled to. You may only execute code in the local sandbox. You cannot send email, make payments, or call external services."
- **Instruction-following:** "If the user specifies a format (e.g., '10 points', 'as a table'), deliver exactly that format or explain why you cannot."

This is the top-level behavior contract. Task prompts (`PLAN_SYSTEM_PROMPT`, `THINK_SYSTEM_PROMPT`, `REFLECT_SYSTEM_PROMPT`, `LAST_ANSWER_EXTRACT_PROMPT`) are appended after it, not instead of it.

---

## 4. Memory and context model

Detailed in `05-context-memory.md`. Summary here:

### 4.1 `last_answer_object` (new, first-class)

After every `finalize`, the agent extracts a structured object from its own answer:

```python
class LastAnswerObject(BaseModel):
    summary: str                    # 2-3 sentence summary
    key_points: list[str]           # bullet points
    data: list[DataPoint] | None    # any numbers/stats mentioned, with labels
    citations: list[CitationRef]    # chunk refs
    chart_option: dict | None       # if a chart was produced, the ECharts JSON
    followups: list[str]            # suggested follow-ups
```

Stored in `AgentState.last_answer_object` and persisted on the `Message` row (new JSON column). This is what "summarise it", "give me the stats in it", "make it a chart" operate on — *not* the raw truncated message text.

**Extraction robustness:** the extraction LLM call is wrapped in Pydantic validation + one retry ("you returned invalid JSON, fix it"). If both fail, a rule-based fallback populates a partial object: `key_points` from sentence splitting, `data` from a regex sweep for `<number> <unit?>` patterns near label words, `summary` from the first 2 sentences. A partial object is better than a missing one — "summarise it" still works on `key_points`.

### 4.2 Token-based compaction (replaces count-based)

- Compute token count per message using the deployed model's tokenizer (`tiktoken` for OpenAI-family, `transformers` tokenizer for local models).
- Maintain a rolling budget: `CONTEXT_WINDOW - RESERVED_FOR_GENERATION - TOOL_BUDGET`.
- When the window overflows, compact the oldest messages into a structured summary (existing compaction prompt, but triggered by tokens not count).
- Tool observations are compacted separately and more aggressively (they are large and lower-density than conversation).

### 4.3 Long-term recall (wire up the unused store)

`load_context_node` proactively recalls top-3 past turns semantically related to the rewritten query via `RedisMemory.search_memory()` and injects them as `<recalled_memory>` context. This is the fix for the no-op `load_subtask_memory_node` (`nodes.py:1080-1087`). No explicit `memory_recall` tool — proactive recall is sufficient for multi-turn; an on-demand tool added selection ambiguity without changing outcomes.

---

## 5. RBAC and audit (enterprise constraints)

### 5.1 Per-tool RBAC re-check

Every tool receives a `ToolContext` containing the authenticated `user_id`, `org_id`, and a DB session. Before executing, each tool re-validates that the user/org is entitled to the resources in its arguments:

- `rag_retrieve`: confirm every `kb_id` belongs to the user (or the user's org hierarchy, for admin/super_admin) via the existing `rbac.py` filters. Drop any ids the user cannot access; log dropped ids.
- `file_read` / `file_summarize` / `file_extract_table`: confirm `file_id` belongs to a `ChatFile` in a chat owned by `user_id`. No cross-user file access.
- `code_execute`: no resource check, but the sandbox denylist (no network, no filesystem writes outside the scratch dir) is enforced regardless of arguments.
- `summarize_answer` / `extract_data`: if `message_id` is specified, confirm it belongs to a chat owned by `user_id`.

The planner LLM is **not trusted** to scope resources correctly — a prompt-injected query could ask the planner to pass another org's `kb_id`. The tool is the enforcement boundary. If entitlements fail, the tool returns `{"ok": false, "error": "not entitled to kb_id ..."}` as an observation; the agent sees this and proceeds without that data (or asks for clarification).

### 5.2 Tool-call audit log

Every tool call writes a row to a new `tool_call_audit` table:

| Column | Type | Notes |
|---|---|---|
| `id` | uuid | |
| `chat_id` | uuid | |
| `message_id` | uuid | the assistant message being generated |
| `iteration` | int | loop iteration |
| `tool_name` | str | |
| `arguments` | json | the validated args |
| `result_summary` | json | truncated result (docs count, confidence, error string) — not full payload |
| `tokens_in` | int | args + context tokens |
| `tokens_out` | int | result tokens |
| `latency_ms` | int | |
| `status` | enum | `ok`, `error`, `denied` (RBAC), `timeout`, `budget_exceeded` |
| `created_at` | timestamp | |

This is the enterprise audit trail: every action the autonomous agent took, per turn, inspectable. Surfaced in a future admin "agent activity" view (out of scope for v1, but the data is captured from day one).

---

## 6. Parallel execution of independent subtasks

The loop is not strictly sequential. The `think_node` LLM may emit **multiple tool calls in a single assistant message** (native function-calling supports this; the JSON-text fallback supports `{"tool_calls": [...]}`). Independent calls run concurrently; the loop waits for all to complete before the next think step.

**Dependency enforcement:** the `Plan` declares `depends_on` per subtask. The `think_node` is instructed (via `THINK_SYSTEM_PROMPT`) to emit only independent tool calls in one message — calls that depend on a prior call's output must wait for a later iteration. The dispatcher additionally checks: if a tool call's `depends_on` subtask has no completed observation yet, defer it to the next iteration. This is a safety net in case the LLM emits a dependent call prematurely.

**Implementation:** LangGraph's `ToolNode` supports parallel tool execution when multiple tool calls arrive in one message. For the JSON-text fallback, the dispatcher fans out calls via `asyncio.gather`. Either path produces one observation per call, all appended to `observations` before the next think.

**Why this matters:** "Give me key findings on X, Y, and Z" is three independent retrievals. Sequential = 3× latency. Parallel ≈ 1× latency. For complex multi-part queries this is the difference between usable and too slow.

---

## 7. Reflect node — concrete replanning rules

The reflect node runs every `AGENT_REFLECT_EVERY` iterations (default 2) and as the final pre-finalize pass. It is not just "are we done?" — it applies explicit recovery rules when the last tool failed or returned insufficient results:

| Last observation | Reflect action |
|---|---|
| `rag_retrieve` returned 0 docs or `sufficient=False` | Rewrite the query (synonyms, broader terms, remove over-specific qualifiers) and re-retrieve. Counts against `AGENT_MAX_RETRIEVALS`. If cap reached, proceed to finalize with a "KB does not contain this" answer. |
| `rag_retrieve` returned docs but they don't answer the question (reflect LLM judges) | Reformulate and re-retrieve, OR call `extract_data` on the retrieved docs to see if the answer is in there in structured form. |
| `file_read` returned `truncated=True` and the user wanted the whole file | Switch to `file_summarize` for the rest. |
| `file_summarize` returned a summary but user wanted specific stats | Call `extract_data` on the file content with a focused `what` parameter. |
| `code_execute` returned stderr | The agent sees the error in the observation and can retry with fixed code (counts against `AGENT_MAX_CODE_EXEC`). Reflect prompts: "the code failed with X; fix it." |
| `chart_generate` returned `valid=False` | The builder already attempted a fix internally. If still invalid, reflect instructs the agent to re-extract the data (`extract_data`) and re-call `chart_generate`. No LLM-generated JSON retry — the deterministic builder is the path. |
| `extract_data` returned empty | The source text has no extractable numbers. Reflect instructs the agent to tell the user "no statistics found in the source." |
| All subtasks complete and instruction satisfied | Emit final-answer signal. |
| Instruction not satisfied (e.g., user asked for 10 points, we have 7) | Reflect instructs another retrieval or summarization pass to fill the gap before finalizing. |

The reflect LLM gets the plan, all observations, and the user's original instruction, and emits either `{"action": "continue"}` with optional plan patch, `{"action": "tool_call", ...}`, or `{"action": "final_answer"}`. The replanning rules above are encoded in `REFLECT_SYSTEM_PROMPT` as explicit instructions, not left to the LLM's discretion alone.

---

## 8. SSE protocol v4

Extends the current v3 protocol (`streaming.py`). New event types for the agent loop. Existing event types preserved for backward compatibility.

| Prefix | Event | New? | Purpose |
|---|---|---|---|
| `p:` | progress | existing | Phase status |
| `t:` | task_list | existing | Subtask checklist (now from planner, updated as loop progresses) |
| `th:` | thinking | existing | LLM chain-of-thought |
| `0:` | token | existing | Final answer streaming |
| `1:` | rewritten_query | existing | Standalone query |
| `2:` | context | existing | Retrieved docs (from `rag_retrieve` tool) |
| `3:` | error | existing | Errors |
| `d:` | done | existing | Finish + usage |
| `4:` | agent_step | existing | Node start/finish |
| `pl:` | plan | **new** | Plan object (intent, subtasks, dependencies) |
| `tc:` | tool_call | **new** | Tool name + args (from think_node) |
| `to:` | tool_observation | **new** | Tool result summary (not full payload; full payload in `2:` for retrieval) |
| `la:` | last_answer | **new** | `LastAnswerObject` after finalize |
| `interrupt` | interrupt | existing | Clarification |
| `evaluation` | evaluation | existing | Answer quality |

`tc:` and `to:` are what the frontend uses to render the agent's tool calls and results inline (like ChatGPT's "Searched the web" / "Analyzed data" cards). `pl:` drives a plan panel. `la:` lets the frontend cache the structured previous answer for "summarise it" UX. For v1 the frontend renders these as plain text/chips; a dedicated plan panel and tool-call cards are a later polish phase.

---

## 9. Multi-tenancy and per-org LLM config (fix)

`get_effective_llm_config()` (`services/chat/chat_service.py:36-59`) is called at the start of every turn. The returned `api_base`/`model_name`/`query_model` are passed into the `ChatOpenAI` instances used by `think_node`, `plan_node`, `reflect_node`, and every tool that calls an LLM. This closes the gap where `OrgLLMConfig` is stored but ignored. **Scope note:** only the existing `api_base`/`model_name`/`query_model` columns are wired now — adding `reasoning_model`/`vision_model`/`embeddings_model`/`graphrag_model` columns is deferred (not needed for the stated goals; the env vars remain the source for those).

---

## 10. What is deleted

To keep the diff reviewable and avoid two parallel pipelines, the rigid-path nodes are removed once the loop is verified:

- `route_by_dependencies`, `agent_subgraph`, `chat_subgraph`, `file_context_subgraph`, `sequential_subtask_loop` (`graph.py:84-261`).
- `classify_query_node` (replaced by `plan_node`).
- `request_clarification_node` as a top-level node (clarification becomes a tool).
- `prepare_final_context_node` (the loop accumulates context in `observations`).
- `chart_validation_node` as a separate node (validation moves into `chart_generate` tool).
- `answer_evaluation_node` becomes `reflect_node`'s final pass + a post-finalize scoring step (kept for the confidence UI).
- `load_subtask_memory_node` (already a no-op; replaced by proactive recall in `load_context_node`).
- `enrich_subtask_query_node` (subtask enrichment happens via tool dependencies in the loop).

**Retrieved and kept as tool internals:** `dense_retrieval_node`, `sparse_retrieval_node`, `exact_retrieval_node`, `merge_node`, `neo4j_expansion_node`, `reranking_node`, `filter_node`, `sufficiency_check_node`, `adaptive_reranking_node` — these become the body of `rag_retrieve` (the `graph_expand` flag controls whether `neo4j_expansion` runs).

**Kept as-is:** `rewrite_query_node` (cheap, valuable, runs before `plan_node`), `compaction_node` (upgraded to token-based, see `05`), `save_memory_node`, `finalize_answer_node` (extended to emit `last_answer_object`).

---

## 11. Failure modes and how the loop handles them

| Failure | Loop behavior |
|---|---|
| Tool throws | Observation = `{"ok": false, "error": str(e)}`. Reflect node applies recovery rules (§7). |
| Retrieval returns nothing | Observation = empty results. Reflect rewrites query and re-retrieves (counts against `AGENT_MAX_RETRIEVALS`); if cap reached, finalize with "KB does not contain this." |
| Code execution errors | Sandbox returns stderr. Agent can retry with fixed code (counts against `AGENT_MAX_CODE_EXEC`). |
| Chart JSON invalid | `chart_generate` validates internally and returns `valid=False`; reflect instructs re-extract data + re-call builder. No LLM-JSON retry. |
| Iteration budget exhausted | `finalize` is forced. Agent emits best-effort answer with a note that it hit the budget. |
| LLM emits no tool call and no final answer | Treated as final answer (the LLM's text becomes the answer). |
| Native tool-calling unsupported by gateway | `think_node` falls back to JSON-text parsing (§3.2). If that also fails, final-answer default. |
| LLM emits malformed JSON in fallback mode | One retry with "you returned invalid JSON, fix it." If still invalid, final-answer default. |
| RBAC denies a tool call | Observation = `{"ok": false, "error": "not entitled", "status": "denied"}`. Audit row written with `status=denied`. Agent proceeds without that data. |
| Clarification needed mid-loop | `clarify` tool calls `interrupt()`. Graph resumes when user responds. |

---

## 12. Why this satisfies the five requirements

| Req | How the architecture meets it |
|---|---|
| 1. KB Q&A | `rag_retrieve` (with `graph_expand`) wraps the existing strong retrieval. Iterative retrieval (loop + reflect replanning) fixes the one-shot gap. `code_execute` + `file_extract_table` handle structured KB data. `extract_data` works on fresh retrieved docs, not just previous answers. |
| 2. Multi-turn intent | `plan_node` runs each turn with `last_answer_object`, file metadata, and recalled memory as inputs. Intent is a planner output, not a regex. `summarize_answer` / `extract_data` / `chart_generate` are explicit act-on-previous-answer tools. |
| 3. Complex multi-subtask | `plan_node` emits subtasks with `depends_on` and `tool_hint`. Independent subtasks dispatch in parallel (§6); dependent ones run sequentially with outputs as later inputs. Re-planning via `reflect_node` (§7). |
| 4. Summarize files/answers | `file_summarize` (map-reduce, chunked) handles large files. `summarize_answer` handles previous answers. Instruction-following is checked in `reflect_node`'s final pass ("did we deliver the 10 points requested?"). |
| 5. Context management | Token-based compaction, sliding window with importance, `last_answer_object` as a compact referent, proactive long-term recall. See `05-context-memory.md`. |
| 6. Charts | `chart_generate` builds ECharts JSON deterministically from data (from `code_execute` or `extract_data`), not from LLM free-form JSON. |
| Enterprise constraints | Per-tool RBAC re-checks (§5.1), tool-call audit log (§5.2), unified guardrail prompt (§3.3), offline tool-calling fallback (§3.2). |
