# Agentic Retrieval Pipeline — Structural Review

Date: 2026-08-10
Scope: `backend/app/services/agentic_rag/*`, plus its entry points in
`chat_service.py`, `pipeline.py`, `api/api_v1/chat.py`.
Method: code read + empirical verification of LangGraph 1.2.10 semantics inside
`rag-web-ui-backend-1`.

---

## 1. Executive summary

The architecture is sound. The pipeline does not need to be replaced or
re-framed. Every symptom you described traces to five concrete defects, four of
which are single-function fixes.

**The single root cause of multi-turn drift: the graph never stores assistant
turns.** No node ever writes an `AIMessage` into `state["messages"]`.
`agent_runner.run_agent_loop` seeds the thread with one `HumanMessage`,
`chat_service` deliberately stops passing history ("delegating history to Redis
checkpoint"), and nothing puts the answer back. The checkpointed conversation is
therefore a list of **user questions only**. Every downstream consumer of
history — the query rewriter, `think_node`, compaction — is reading half a
conversation. A rewriter that must resolve "its limitations" while being shown
only prior *questions* has no choice but to invent the missing referent. That is
your term injection.

**Verdict on query rewriting: keep it, but make it conditional and
provenance-bound.** Modern LLMs do resolve references from chat history — but
only in the model call that receives that history. The retrieval backends
(`dense_search_docs`, `sparse_search_docs`, `exact_search_docs`, the
cross-encoder) receive a single string and no history. A standalone retrieval
string is therefore genuinely required, but only for messages that actually
contain a cross-turn reference. Details in §6.

**Verdict on restructuring: no.** Keep the graph, the tool loop, the relaxation
ladder in `rag_retrieve`, the deterministic `_verify_execution` gate, and
`last_answer_object`. Change five things (§8).

**Compaction status: it currently cannot work.** Three independent reasons —
the message reducer appends instead of replacing, nothing compacts
`retrieved_docs` (the largest contributor to the finalize prompt), and the
pre-plan `compaction_node` threshold is ~101k tokens of pure chat history, which
this pipeline can never reach because it only stores user messages. Do not
enable it in production until §4 is fixed.

Priority order:

1. P0-1 Persist assistant turns in graph state.
2. P0-2 Stop duplicating observations (`accumulate` reducer misuse).
3. P0-3 Declare the four undeclared state keys (silently dropped today).
4. P0-4 Fix the clarification flow (interrupt is swallowed; the user's answer is discarded).
5. P1-1 Make compaction actually replace, and make it target `retrieved_docs`.
6. P1-2 Conditional, provenance-checked rewriting.
7. P1-3 Separate recalled memory from citable evidence.

---

## 2. How context is actually built, stage by stage

| Stage | Conversation history | Prior answer | Retrieved evidence | Tool results |
|---|---|---|---|---|
| `load_context` | — | `last_answer_object` from DB (previous assistant `Message`) | writes Redis memory hits into `retrieved_docs` | — |
| `rewrite_query` | `select_recent_history(max_pairs=3)` → **user messages only** | not passed (`memory_context=""`) | — | — |
| `compaction` | full `messages` (user-only) | — | — | — |
| `plan` | **none** (never reads `messages`) | `lao.summary` (full) | first 3 recalled memory docs, raw text | — |
| `think` | `select_recent_history(3)` → user-only | `lao.summary[:300]` + 5 key points | metadata only (`doc_count`, `confidence`) | full JSON for non-RAG tools |
| `tool` | — | — | merges memory docs + all `rag_retrieve` docs, dedup by `content_hash` | — |
| `finalize` | **none** | **none** | `format_context_string(retrieved_docs)` — full text, overlap-pruned | `_non_retrieval_observations_text` |
| `answer_scoring` | — | — | full context text again (second full copy, separate LLM call) | — |

Two observations from this table:

- There is no single authoritative context object. Four stages each build their
  own view, and the rewritten query — produced from the *weakest* view — becomes
  the authoritative query for all later stages, including the answer.
- `finalize_node` gets no conversation context at all. This is documented as
  intentional in [prompts.py](backend/app/services/agentic_rag/prompts.py), but
  it means "compare that to what you said before" cannot work even when the
  retrieval is perfect.

Per-turn LLM calls today: rewrite (1) + plan (1) + think (N≥1) + finalize (1) +
`last_answer_object` extraction (1–2) + evaluation (1) = **6–8 calls**, of which
rewrite/plan and think overlap substantially in purpose.

---

## 3. P0 defects

### P0-1 — Assistant turns are never written to graph state

Evidence:

- [agent_runner.py](backend/app/services/agentic_rag/agent_runner.py#L67) seeds
  `messages=[HumanMessage(content=query)]`.
- [chat_service.py](backend/app/services/chat/chat_service.py#L290) computes
  `prior_messages` and then only logs it — history is delegated to the
  checkpoint.
- No node in `agent_graph.py` or `nodes.py` constructs an `AIMessage` for
  `state["messages"]`. `finalize_node` returns `final_answer`/`answer`;
  `save_memory_node` writes to the MySQL `Message` row only.

Consequences:

- `select_recent_history` in
  [nodes.py](backend/app/services/agentic_rag/nodes.py#L43) has an `AIMessage`
  branch that is dead in production.
- The rewriter sees prior *questions*, not prior *answers*. "Summarise your
  second point", "the one you mentioned first", "expand that table" are
  unresolvable, and the model fills the gap by inventing terms.
- Compaction summarises a user-question log, not a conversation.
- The only cross-turn assistant context the agent has is `lao.summary[:300]`.

Fix: append `AIMessage(content=final_answer)` from `finalize_node` (return
`{"messages": [AIMessage(...)]}` — the `add_messages` reducer appends). Also
append the current `HumanMessage` only once per turn (it already is). Guard
against replay: on resume-after-interrupt the human turn must not be re-added.

### P0-2 — Observations are duplicated on every tool round

`observations` uses the `accumulate` reducer
([graph_state.py](backend/app/services/agentic_rag/graph_state.py#L139)), but
[tool_node](backend/app/services/agentic_rag/agent_graph.py#L933) returns the
**entire** list (`prior + new`). The reducer then computes `existing + returned`.

Growth is `len(n) = 2·len(n-1) + 1`: 1 → 3 → 7 → 15 observations for 4 tool
rounds.

Second instance of the same bug:
[`_compact_if_needed`](backend/app/services/agentic_rag/agent_graph.py#L530)
returns `updates["observations"] = compacted_obs` — a *full replacement* list
which the reducer appends. Compaction therefore roughly doubles the observation
list while claiming to have saved tokens.

Impact: inflated `think` prompts, wrong `total_docs` in
`_build_execution_summary` (which drives the deterministic completion gate),
duplicated `msg.tool_calls` persisted to MySQL, and wasted `_compact_observations`
work. Merged docs are unaffected (hash dedup).

Fix: `tool_node` returns only the observations created in this invocation. Keep
`prior_signatures` locally for the idempotency check. For compaction, either
give `observations` a reducer that supports replacement (extend the existing
`__reset__` marker convention) or emit `[{"__reset__": True}, *compacted]`.

### P0-3 — Four state keys are undeclared and silently dropped

Verified empirically on langgraph 1.2.10: a node returning a key absent from the
state schema has that key **silently discarded** (no error, no warning).

`AgentState` does not declare `started_at`, `force_finalize`,
`precomputed_tool_calls`, or `reflection`. Therefore:

| Key | Written at | Read at | Actual behaviour |
|---|---|---|---|
| `started_at` | [agent_graph.py#L588](backend/app/services/agentic_rag/agent_graph.py#L588) | [#L811](backend/app/services/agentic_rag/agent_graph.py#L811), [#L1308](backend/app/services/agentic_rag/agent_graph.py#L1308) | `_wall_clock_exceeded()` always `False` → **`AGENT_MAX_WALL_SECONDS` is dead**; `remaining_budget.seconds` always reports the full budget |
| `force_finalize` | [#L596](backend/app/services/agentic_rag/agent_graph.py#L596), [#L979](backend/app/services/agentic_rag/agent_graph.py#L979) | [#L678](backend/app/services/agentic_rag/agent_graph.py#L678), [#L828](backend/app/services/agentic_rag/agent_graph.py#L828) | `route_tool` never short-circuits; masked only because `think_node` re-runs the same `_verify_execution` check |
| `precomputed_tool_calls` | [#L1224](backend/app/services/agentic_rag/agent_graph.py#L1224) | [#L689](backend/app/services/agentic_rag/agent_graph.py#L689) | **`reflect_node`'s recovery rules never fire** — the node is entirely inert |
| `reflection` | [#L1223](backend/app/services/agentic_rag/agent_graph.py#L1223) | nowhere | inert |

Fix: add the three live keys to `AgentState` with `_last_value` reducers, then
re-test the wall-clock and reflect paths — they have never actually run. Delete
`reflection`. Also delete the duplicated `clarification_question` declaration
([graph_state.py#L147](backend/app/services/agentic_rag/graph_state.py#L147) and
[#L149](backend/app/services/agentic_rag/graph_state.py#L149)).

### P0-4 — The clarification flow is broken in three places

1. **The interrupt is swallowed.**
   [`clarify_interrupt_node`](backend/app/services/agentic_rag/agent_graph.py#L1248)
   wraps `interrupt()` in `try/except Exception`. Verified:
   `GraphInterrupt` → `GraphBubbleUp` → `Exception`. The control-flow signal is
   caught, logged as "interrupt not supported or failed", and execution
   continues with `user_response = ""`. The `GraphInterrupt` handler in
   `agent_runner` can only fire for interrupts raised elsewhere.
2. **The clarification answer is discarded.** After resume, the edge is
   `clarify_interrupt → plan`. `plan_node` reads `rewritten_query` (computed
   from the *original* ambiguous query) and never reads `messages`.
   `rewrite_query` does not re-run. The appended clarification message is the
   *last* message, and `select_recent_history` explicitly drops the last message
   as "the current query" — so `think_node` does not see it either. The user's
   answer influences nothing.
3. **No clarification budget.** `plan → clarify_interrupt → plan` has no
   counter. A model that keeps setting `needs_clarification=true` loops until
   the recursion limit.

Additional: `clarify_interrupt_node` returns `list(state["messages"]) + [new]`.
It should return `[new]` only — `add_messages` appends, and re-emitting the full
list only works because the stored messages already carry ids.

Fix: (a) catch nothing around `interrupt()`, or catch only non-`GraphBubbleUp`
exceptions; (b) route `clarify_interrupt → rewrite_query → plan`, or merge the
clarification answer into `original_query`/`retrieval_query` explicitly;
(c) add `clarification_count` with a cap of 1 and force `needs_clarification=False`
beyond it.

---

## 4. P1 — Context compaction and management

Compaction is untested because, as written, it cannot reduce context. Three
independent reasons:

### 4.1 The message reducer appends; it never replaces

`AgentState` extends `MessagesState`, whose `messages` channel uses
`add_messages`. Verified on langgraph 1.2.10 — a node returning
`[summary] + recent` produced:

```
[q1, a0, q2, a1, SUMMARY]      # old messages retained, summary appended last
```

So [`_compact_if_needed` stage 2](backend/app/services/agentic_rag/agent_graph.py#L542)
**increases** the checkpoint size, and the summary lands at the end of the
history rather than at the front.

Fix: emit `RemoveMessage(id=m.id)` for each dropped message (or
`RemoveMessage(id=REMOVE_ALL_MESSAGES)` followed by the retained set), and give
the summary message a stable id so repeated compactions replace it instead of
stacking.

### 4.2 Nothing compacts the actual dominant context

`finalize_node` builds its prompt from `format_context_string(retrieved_docs)`.
`_compact_if_needed` touches `observations` and `messages` — never
`retrieved_docs`. Worse, stage-1 savings are estimated as
`count_tokens(json.dumps(obs.result))` before/after, i.e. tokens that were never
in the prompt at all: `think_node` renders observations with
`_observations_metadata_text` (counts and confidence only, no chunk text).

Net effect: the budget check reports a large "saving", concludes stage 2 is
unnecessary, and the finalize prompt is unchanged. An over-budget finalize call
stays over budget.

Fix: compaction must act on the same objects that the prompt is built from.
For `finalize`, that means trimming `retrieved_docs` (drop lowest
`_reranker_score` chunks until the context fits) rather than trimming
observations. Compute savings by rebuilding the prompt, not by estimating.

### 4.3 The pre-plan `compaction_node` can never trigger

[`compaction_node`](backend/app/services/agentic_rag/nodes.py#L82) builds a
`ContextBudget` from conversation text + fixed system-prompt overhead only.
With defaults (`131072 − 4096 − 8192 = 118784`, ratio `0.85`) the threshold is
**~100,966 tokens of chat history**. Since the history holds only user questions
(P0-1), this is unreachable. The node is a no-op that runs on every turn.

Additional problems in that node if it ever does fire:

- `llm.invoke(...)` — a **synchronous** call inside an `async` node
  ([nodes.py#L151](backend/app/services/agentic_rag/nodes.py#L151), and again at
  [agent_graph.py#L466](backend/app/services/agentic_rag/agent_graph.py#L466)).
  This blocks the FastAPI event loop for the duration of the summarisation and
  will stall every concurrent chat stream. Use `ainvoke`.
- It uses `_get_llm(settings.effective_query_model)` with global
  `OPENAI_API_BASE`, bypassing the per-org LLM config used everywhere else
  (`build_chat_llm`). `ContextBudget` likewise uses the global
  `OPENAI_MODEL_CONTEXT_SIZE`, not the org's model window.
- `compaction_summary` is written to state and **never read** by any prompt.

Fix: keep exactly one compactor. Run it after `save_memory` (so the next turn
starts from a stable checkpoint) plus as an emergency pre-call guard that shares
the same implementation. Feed `compaction_summary` into the rewriter and
`think`. Delete `COMPACTION_HISTORY_THRESHOLD` and
`COMPACTION_ASSISTANT_MAX_CHARS` — neither is used on this path.

---

## 5. P1/P2 — Remaining findings

### P1-3 Recalled memory is treated as citable evidence

[`load_context_node`](backend/app/services/agentic_rag/agent_graph.py#L583)
writes `search_memory()` results into `retrieved_docs`. `tool_node` seeds its
merged doc list with them; `finalize_node` renders them as `[KB-n]` chunks;
`normalize_citations` will happily cite them; `answer_evaluation_node` scores
faithfulness against them. A previous model answer can therefore become a
citation source for a new answer — a self-reinforcing hallucination path.

Two further notes: `RedisMemory.save_turn` is **never called anywhere in the
codebase**, so the long-term store is always empty and this path is currently
latent rather than active. And whichever memory doc lands at index 0 shifts
every `[KB-n]` label.

Fix: separate `recalled_memories` state field. It may inform intent resolution
and planning; it must never enter `retrieved_docs`, citations, retrieval
confidence, or faithfulness scoring. Decide explicitly whether `save_turn`
should be wired in at all — an unused semantic store is one less moving part.

### P1-4 Per-turn accumulators are not reset

`load_context_node` resets `observations` (via `__reset__`) and several scalars,
but not `dense_docs`, `sparse_docs`, `exact_docs`, `graph_docs`, `failed_legs`,
`leg_results`, `leg_doc_counts`, `retrieval_keys`, `subtask_contexts`,
`subtask_answers`, or `artifacts`. Most are written only by the retrieval nodes
when invoked through `rag_retrieve` with a plain dict state (not graph state),
so they mostly stay empty — but `artifacts` and `subtask_contexts` are real
graph-state accumulators and will grow across a chat's lifetime inside the Redis
checkpoint.

Fix: delete the fields that belong to the retired subtask graph; add
`__reset__`-style clearing for the rest in `load_context_node`.

### P2-1 `_verify_execution` cannot distinguish subtasks

[`_build_execution_summary`](backend/app/services/agentic_rag/agent_graph.py#L1268)
marks a subtask complete if **any** successful observation used its `tool_hint`.
A three-part question planned as three `rag_retrieve` subtasks is declared
complete after the first retrieval, and `think_node`'s pre-check then
short-circuits to finalize. This silently caps multi-hop questions at one
retrieval.

Fix: record the subtask id on each tool call (`tool_calls` already flow through
`think_node`) and match observations to subtasks by id, not by tool name.

### P2-2 `finalize_node` receives no conversation context

Documented as deliberate, but it makes conversational instructions
("shorter", "in a table", "compare with your last answer") depend entirely on
`summarize_answer` being selected. Once P0-1 lands, pass a token-budgeted
conversation block plus `last_answer_object` to finalize, with the priority
order stated explicitly: retrieved documents are evidence, conversation is
intent.

### P2-3 Non-determinism where determinism is wanted

`think_node` selects tools at `temperature=0.7`
([#L777](backend/app/services/agentic_rag/agent_graph.py#L777),
[#L781](backend/app/services/agentic_rag/agent_graph.py#L781)). Tool selection
is a classification decision; use `0.0`. Keep `0.7` for `finalize_node` prose.

### P2-4 Reported token usage is fabricated

`agent_runner` reconstructs `promptTokens` by re-counting prompt fragments
instead of reading provider usage. It counts `think_sys_tokens × iterations` but
omits observation and history text actually sent, and counts all
`retrieved_docs` once regardless of how many calls saw them. `AgentState.answer_usage`
exists and is never populated. Either capture real usage from the LLM responses
or label the number as an estimate in the UI.

### P2-5 Dead / unreachable code

- `_observations_text(full=True)` — referenced only by
  `backend/tests/test_agent_loop.py`.
- `reflect_node` — inert (P0-3); the graph still pays an extra hop
  `tool → reflect → think` on every round.
- `run_agent_loop` accepts `temperature`, `model_name`, `api_base`, and
  `query_model` and uses none of them; all LLMs come from `build_chat_llm` or
  global settings.
- `AgentState` retains ~20 fields from the retired subtask graph
  (`sufficiency_*`, `needs_graph_expansion`, `adaptive_rerunning`,
  `subtask_answers`, `chart_retries`, `_task_list`, `_confidence`, …).

### P2-6 `_prune_contiguous_overlaps` uses `list.index` on dicts

`prev_idx = group[group.index(doc) - 1]...` resolves position by dict equality,
which is O(n²) and returns the wrong neighbour if two chunk dicts compare equal.
Use `enumerate`.

---

## 6. Is query rewriting actually required?

**Yes — but only as a retrieval-query resolver, not as an unconditional
paraphraser.**

The argument for removing it is that modern chat models resolve references from
history on their own. That is true *for the model call that receives the
history*. It does not hold here, because the components that consume the query
are not LLMs:

- `dense_search_docs` embeds one string; "its limitations" embeds to nothing useful.
- `sparse_search_docs` / SPLADE expands one string's terms.
- `exact_search_docs` runs MySQL FTS on one string.
- The cross-encoder reranker scores `(query, chunk)` pairs — a pronoun-only
  query poisons the ranking for every chunk equally.

So the retrieval leg needs a self-contained string. What it does **not** need is
a rewrite on every turn.

Why the current implementation injects terms:

1. It runs unconditionally, including on already self-contained queries. Rule 8
   of `REWRITE_SYSTEM_PROMPT` says "return it EXACTLY as-is" — that is a soft
   instruction with no enforcement. Every unnecessary invocation is a chance to
   drift.
2. It is given a **user-questions-only** history (P0-1). Asked to resolve a
   reference against a half-conversation, the model extrapolates.
3. Its output is accepted as free text. The only validation is a regex for
   answer-like phrasing. Nothing checks that introduced entities came from
   somewhere real.
4. `max_tokens=60` truncation can silently cut the rewrite mid-phrase, and the
   meta-commentary stripper (`rsplit(":", 1)`) can mangle a legitimate query
   containing a colon.

Recommended policy — "resolve retrieval intent when needed", not "rewrite
always":

- Emit a structured object, not free text:
  `{needs_resolution: bool, retrieval_query: str, resolved_from: [message_id|"last_answer_object.key_points[2]"]}`.
- `needs_resolution=false` → use the original query **byte-for-byte**.
- `needs_resolution=true` → accept the rewrite only if every content token added
  relative to the original appears in the cited source. A pure "no new words"
  rule is wrong (resolving "it" must add the entity); provenance is the correct
  invariant.
- Any parse failure, validation failure, or timeout → fall back to the original
  query. Never let a free-form rewrite become authoritative.
- Keep `original_query` and `retrieval_query` separate. The **answer** model
  should always see the original message; only retrieval sees the resolved one.
- Merge this decision into `plan_node`. Planning already classifies intent and
  decides clarification; resolution is the same judgement over the same context.
  That removes one LLM call per turn and removes the possibility of the planner
  and the rewriter disagreeing.

Order of operations matters: fix P0-1 first. Roughly half the observed term
injection should disappear once the rewriter can see the answers it is being
asked to resolve references against.

---

## 7. Duplication and redundancy inventory

| # | Duplication | Where | Action |
|---|---|---|---|
| 1 | Observations appended twice per round (2ⁿ growth) | `tool_node`, `_compact_if_needed` | Fix (P0-2) |
| 2 | Two independent compaction implementations | `nodes.compaction_node`, `agent_graph._compact_if_needed` | Keep one |
| 3 | Two conversation-history builders with different rules | `select_recent_history` (rewrite) vs. inline loop in `think_node` | One context object |
| 4 | `lao.summary` rendered in both `plan` and `think` prompts | `plan_node`, `think_node` | Acceptable; cap once, centrally |
| 5 | Recalled memory in the `plan` prompt **and** in `retrieved_docs` | `load_context_node` | Fix (P1-3) |
| 6 | Full context text rendered twice: `finalize` prompt + `answer_evaluation` prompt | `finalize_node`, `answer_evaluation_node` | Evaluate against cited chunks only, not all docs |
| 7 | `last_answer_object` extraction is a separate LLM call (up to 2 attempts) over the answer just generated | `finalize_node` | Fold into finalize via structured output, or make it lazy/background |
| 8 | History rebuilt from scratch when compaction fires mid-`think` | `think_node` | Falls out of #3 |
| 9 | `plan` and `think` both decide which tools to run | `plan_node`, `think_node` | Plan = intent + clarification + resolution only |
| 10 | Chunk-overlap pruning applied in both `_observations_text` and `format_context_string` | `agent_graph`, `utils` | Harmless; only `format_context_string` is live |

Note what is **not** duplicated and should stay as-is: `think_node` correctly
sends observation *metadata* only, `tool_node` deduplicates docs by
`content_hash`, and `format_context_string` prunes contiguous chunk overlap.
Those three are the right decisions.

---

## 8. Recommended target structure

Minimal diff — same graph, same tools, five changes:

```text
load_context          reset per-turn state; load last_answer_object;
                      recalled memory -> recalled_memories (NOT retrieved_docs)
   |
resolve_intent        replaces rewrite_query + merges into plan.
   |                  one structured call ->
   |                  {intent, needs_resolution, retrieval_query, provenance,
   |                   subtasks, needs_clarification}
   |                  needs_resolution=false -> retrieval_query = original_query
   |
   +-- needs_clarification (max 1 per turn) --> interrupt --> resolve_intent
   |
think  <-> tool        think: metadata-only observations + real history
   |                   tool: returns NEW observations only
   |
reflect_final          deterministic gate, per-subtask matching
   |
finalize               original_query + conversation context + evidence;
                       compaction trims retrieved_docs by reranker score
   |
answer_scoring         scores cited chunks only
   |
save_memory            persist answer; append AIMessage to state["messages"]
   |
compact                RemoveMessage-based; stable summary id; async LLM call
```

Change list:

1. `finalize`/`save_memory` append the `AIMessage`; `compaction` moves to the
   end of the turn.
2. `tool_node` returns new observations only; `observations` gains explicit
   replacement support for compaction.
3. `AgentState` declares `started_at`, `force_finalize`,
   `precomputed_tool_calls`; drop the dead subtask-graph fields and the
   duplicate `clarification_question`.
4. `rewrite_query` folds into `plan` as structured intent resolution with
   provenance validation and byte-exact passthrough.
5. Compaction: one implementation, `RemoveMessage`-based, `ainvoke`, org-aware
   LLM config, and doc-trimming for the finalize prompt.

Everything else — the relaxation ladder, `_verify_execution`, tool budgets,
citation normalisation, streaming — stays.

---

## 9. Tests required before this is trustworthy

Structural (deterministic fakes, no live model):

1. Three tool rounds produce exactly three observations.
2. Turn 2 of a chat sees turn 1's `AIMessage` in `state["messages"]`.
3. Compaction over two consecutive triggers: old messages absent from the
   persisted checkpoint, exactly one summary message, token count monotonically
   bounded.
4. `_wall_clock_exceeded` returns `True` once `AGENT_MAX_WALL_SECONDS` elapses
   (this has never been exercised — see P0-3).
5. `reflect_node`'s `precomputed_tool_calls` actually reach `think_node`.
6. `clarify_interrupt` raises `GraphInterrupt` out of the graph; after
   `Command(resume=...)` the clarification text appears in the retrieval query.
7. A second `needs_clarification` in one turn is refused.
8. No per-turn accumulator survives into the next turn.
9. Recalled memory never appears in `retrieved_docs`, citations, or
   faithfulness scoring.
10. Rewrite/plan/compaction LLM failure → original query preserved, raw recent
    messages preserved.

Behavioural (small fixed multi-turn transcript set, measured not eyeballed):

- Self-contained follow-up → `retrieval_query == original_query` byte-for-byte.
- Pronoun follow-up → referent resolved, provenance recorded.
- Topic switch → zero terms carried over from the previous topic.
- Long-distance reference (older than the recent window) → resolved from the
  compaction summary.
- Retrieval recall on turn N vs. turn 1 for the same question — drift shows up
  here before it shows up in answer quality.

Do not use answer quality as the rewriting metric. A fluent answer routinely
hides a corrupted retrieval query.

---

## 10. Changes explicitly not recommended

- Do not replace LangGraph or restructure the tool loop.
- Do not remove query rewriting entirely — retrieval backends need a standalone
  string.
- Do not add a second post-retrieval rewrite pass. Fix the single resolution
  boundary.
- Do not widen the history window as the primary fix for drift; it raises cost
  and delays the failure rather than removing it.
- Do not add importance scoring, pinned messages, or a memory-recall tool until
  §3 and §4 are fixed and measured.
- Do not enable `COMPACTION_ENABLED` in production before §4.
