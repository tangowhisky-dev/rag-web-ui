# Enterprise Agent Review — Cross-Validation & Implementation Report

Date: 2026-08-10
Reviews assessed: `docs/enterprise-agent-review-opus.md`, `docs/enterprise-agent-review-sol.md`
Scope: `backend/app/services/agentic_rag/*`, plus entry points in `chat_service.py`, `pipeline.py`, `api/api_v1/chat.py`, `api/api_v1/query.py`
Verification: code read + empirical probes against langgraph 1.2.10 / langgraph-checkpoint 4.1.1 / langchain-core 1.5.2 inside `rag-web-ui-backend-1`

---

## 1. Verdict summary

Both reviews are substantially accurate. Of the 24 distinct claims examined, **21 were confirmed**, **1 was materially wrong**, **1 was overstated**, and **1 was correct but incomplete** (both reviews missed the actual mechanism, which turned out to matter for the fix).

Nothing in either review recommended replacing the retrieval stack, and I agree with that. The changes below are a repair of conversation-state and context ownership, not a re-architecture.

Test suite: **440 passed** (413 pre-existing + 27 new structural tests), 0 failures.

---

## 2. Cross-validation results

### 2.1 Confirmed — verified empirically

| # | Claim | Verification |
|---|---|---|
| C1 | Undeclared state keys are silently dropped by LangGraph | Probe graph on langgraph 1.2.10: node returned `{"known": ..., "undeclared": 123}`; downstream node saw `None`, final state keys were `['known', 'messages']`. **No error, no warning.** |
| C2 | `started_at`, `force_finalize`, `precomputed_tool_calls`, `reflection` were undeclared in `AgentState` | Confirmed by reading `graph_state.py`. Consequence: `AGENT_MAX_WALL_SECONDS` was dead, `route_tool` never short-circuited, `reflect_node` was entirely inert. |
| C3 | `clarification_question` declared twice | Confirmed — `graph_state.py` lines 147 and 149. |
| C4 | `GraphInterrupt` subclasses `Exception`, so `except Exception` swallows the pause | Probe: MRO is `GraphInterrupt → GraphBubbleUp → Exception → BaseException`. `clarify_interrupt_node` caught it, logged "interrupt not supported or failed", and continued with `user_response = ""`. |
| C5 | Observations grow `2n+1` per tool round | Confirmed by reading: `tool_node` returned `prior + new` through the append-style `accumulate` reducer. 1 → 3 → 7 → 15 across four rounds. |
| C6 | `_compact_if_needed` repeated the same reducer mistake | Confirmed — returned a full replacement list into an append-only channel. |
| C7 | `add_messages` appends; `[summary] + recent` grows the checkpoint | Probe: input `[q1, a1, q2]` + return `[SUMMARY, q2]` produced `[q1, a1, q2, SUMMARY]`. Old messages retained, summary appended last. |
| C8 | No node ever appended an `AIMessage` | Confirmed. `finalize_node` returned `answer`/`final_answer`; `save_memory_node` wrote MySQL only. The checkpointed thread held user questions only. |
| C9 | The pre-plan `compaction_node` could never fire | Confirmed arithmetic: `(131072 − 4096 − 8192) × 0.85 ≈ 100,966` tokens of chat history required, against a history that only ever held user questions. |
| C10 | `compaction_node` and `_compact_messages_llm` used synchronous `llm.invoke` inside `async def` | Confirmed. Blocks the FastAPI event loop for the whole summarisation. |
| C11 | `compaction_node` bypassed the per-org LLM config | Confirmed — used `_get_llm(settings.effective_query_model)` with global `OPENAI_API_BASE`, unlike every other node. |
| C12 | `compaction_summary` was written and never read | Confirmed — no prompt consumed it. |
| C13 | Stage-1 "savings" were measured against tokens that were never in the prompt | Confirmed — savings computed from `json.dumps(obs.result)` while `think_node` renders observations via `_observations_metadata_text` (counts only). |
| C14 | Nothing compacted `retrieved_docs`, which dominates the finalize prompt | Confirmed. |
| C15 | Recalled memory was written into `retrieved_docs` and became citable | Confirmed — `load_context_node` returned `"retrieved_docs": recalled`; `tool_node` seeded its merge with them; `format_context_string` rendered them as `[KB-n]`; `answer_evaluation_node` scored faithfulness against them. |
| C16 | `RedisMemory.save_turn` has no caller | Confirmed — single grep hit is the definition itself. The path is latent, not active. |
| C17 | `_build_execution_summary` marked a subtask complete on *any* matching observation | Confirmed — three `rag_retrieve` subtasks were declared complete after one retrieval, and `think_node`'s pre-check then short-circuited to finalize. |
| C18 | `think_node` selected tools at `temperature=0.7` | Confirmed, both branches. |
| C19 | Reported token usage is reconstructed, not measured; `answer_usage` was never populated | Confirmed — only read (in the clarification-resume endpoint), never written. |
| C20 | `_prune_contiguous_overlaps` used `list.index` on dicts | Confirmed — O(n²) and returns the wrong neighbour when two chunk dicts compare equal. |
| C21 | `run_agent_loop` accepted `temperature`, `model_name`, `api_base`, `query_model` and used none of them | Confirmed. |

### 2.2 False positive

**Opus §5 / P2-5: "`AgentState` retains ~20 fields from the retired subtask graph."**

The *count* is right but the framing implies they were merely unused. Several were live: `all_scored_docs`, `leg_results`, `failed_legs`, `leg_doc_counts`, `dense_docs`/`sparse_docs`/`exact_docs`/`graph_docs`, `graph_expansion_done` and `adaptive_reran` are all written by the retrieval nodes invoked from the `rag_retrieve` tool. Deleting them wholesale, as the review implies, would have broken retrieval.

Verified per-field usage counts outside `graph_state.py`. Genuinely dead (0 references): `subtask_answers`, `subtask_contexts`, `artifacts`, `retrieval_keys`, `retrieval_iterations`, `sufficiency_met`, `sufficiency_message`, `needs_graph_expansion`, `adaptive_rerunning`, `needs_retry`, `chart_retries`, `thinking_chunks`, `is_chart_query`, `chart_data`, `_task_list`, `_confidence`. Those 16 were removed; the live ones were kept.

Related: Opus P1-4 warns that `artifacts` and `subtask_contexts` "are real graph-state accumulators and will grow across a chat's lifetime inside the Redis checkpoint." They cannot — nothing writes them. The correct fix was deletion, not per-turn reset.

### 2.3 Overstated

**Opus P1-3 / Sol P1-2: recalled memory as an active "self-reinforcing hallucination path."**

The state boundary is genuinely wrong and worth fixing, but the risk is currently *zero*, not merely "latent": `save_turn` has no caller, so `search_memory` always returns an empty list. Both reviews do note this, but the P0-adjacent framing overstates present impact. Fixed anyway — the boundary is what matters.

### 2.4 Correct conclusion, wrong mechanism

**Sol P0-3 #2 and Opus P0-4 #1** both correctly conclude the clarification flow is broken, but neither identified the decisive mechanism.

Probe result: **`graph.astream()` does not raise `GraphInterrupt`.** It emits an `('updates', {'__interrupt__': (Interrupt(value=...),)})` chunk *after* persisting the interrupt checkpoint. Therefore:

- `agent_runner`'s `except Exception` / `exc_name == "GraphInterrupt"` handler was **dead code on the streaming path** — it could never fire.
- The only reason the UI ever saw a clarification prompt was the custom `writer({"event": "interrupt"})` emitted *before* `interrupt()` was called.
- `chat_service` breaks its stream on that custom event, closing the generator and abandoning the graph run.
- Meanwhile `interrupt()` raised inside the node and was swallowed by `except Exception`, so the graph never actually paused and no interrupt checkpoint was ever written.

Net effect: the user was asked a question, the graph continued in the background with an empty answer until the abandoned generator was closed, and `Command(resume=...)` on `/api/clarification` had nothing to resume. This is worse than either review described, and the fix has to move event emission *after* the `__interrupt__` update rather than merely removing the `try/except`.

### 2.5 Incidental finding (not in either review)

`api/api_v1/query.py` called `run_agentic_rag(kb_ids=..., generate_answer=...)`, but the function's parameter is `knowledge_base_ids` and it never forwarded `generate_answer`. The endpoint would have raised `TypeError` at runtime; the test passed only because its mock replicated the wrong signature. Fixed as part of the signature cleanup (C21).

---

## 3. What was implemented

### P0-1 — Assistant turns are persisted in graph state
`agent_graph.finalize_node`, `graph_state.py`

`finalize_node` now returns `{"messages": [AIMessage(content=final, id=f"assistant-{message_id}")]}`. The `add_messages` reducer appends; the stable per-message id makes a resume-after-interrupt replay replace the turn instead of duplicating it. `cited_doc_indices` and provider `answer_usage` are returned alongside.

This is the root-cause fix. Every downstream consumer of history — query resolution, `think_node`, compaction, `finalize_node` — now reads a whole conversation.

### P0-2 — Observation duplication removed
`agent_graph.tool_node`, `agent_graph._compact_if_needed`, `graph_state.accumulate`

`tool_node` keeps `prior_observations` locally for the idempotency check and doc merge, but returns only `new_observations`. Compaction sends replacements through the existing `__reset__` marker contract (`[{"__reset__": True}, *compacted]`). `accumulate` was also hardened: its reset branch called `.get()` on every item, which would have raised on a list containing `Observation` objects.

### P0-3 — State keys declared; dead fields deleted
`graph_state.py`

Declared `started_at`, `force_finalize`, `precomputed_tool_calls`, plus new `recalled_memories`, `clarification_count`, `clarification_response`, `resolution_provenance`. Removed the duplicate `clarification_question`, the never-read `reflection` write in `reflect_node`, and the 16 genuinely dead fields listed in §2.2.

Fallout found while testing: `_wall_clock_exceeded` used `if not started_at`, so a `started_at` of `0.0` disabled the check. Now `if started_at is None`.

`AGENT_MAX_WALL_SECONDS` and `reflect_node`'s recovery rules are now live for the first time. Both are covered by new tests.

### P0-4 — Clarification interrupt and resume repaired
`agent_graph.clarify_interrupt_node`, `agent_graph.plan_node`, `agent_runner.py`, graph wiring

1. `interrupt()` is called with no exception handler and no pre-emitted custom event.
2. `agent_runner` detects the `__interrupt__` update and yields `{"event": "interrupt", "question", "thread_id"}` — after LangGraph has persisted the checkpoint, so `Command(resume=...)` has something to resume.
3. The node returns only the new `HumanMessage`, plus `clarification_response` and an incremented `clarification_count`.
4. The edge is now `clarify_interrupt → rewrite_query → plan`, and `rewrite_query_node` folds `clarification_response` into the text it resolves. The clarification answer now reaches the retrieval query and the plan; previously it influenced nothing.
5. `plan_node` enforces `AGENT_MAX_CLARIFICATIONS` (default 1) and ignores `needs_clarification` past the cap.

### P1 — One compaction implementation, and it actually reduces context
`agent_graph._compact_if_needed`, `_compact_messages_llm`, `_trim_docs_to_budget`, `_build_compaction_llm`; `nodes.compaction_node` deleted; graph wiring

- The unreachable pre-plan `compaction_node` is gone. One implementation remains: a budget guard invoked immediately before the `think` and `finalize` LLM calls, i.e. against the actual next request.
- Message replacement uses `RemoveMessage(REMOVE_ALL_MESSAGES)` plus a stable summary id (`_COMPACTION_SUMMARY_ID`), so old turns leave the checkpoint and repeated compactions replace the summary instead of stacking.
- `ainvoke` instead of `invoke` — no more event-loop blocking.
- Org-aware LLM via `build_chat_llm`, with a global-config fallback.
- New stage 2 for `finalize` only: `_trim_docs_to_budget` drops the lowest `_reranker_score` chunks until the overflow is covered, never below one chunk. Conversation summarisation cannot fix an evidence-payload overflow.
- `_messages_to_conversation_text` no longer truncates turns to 500/800 chars before summarising — a summariser cannot preserve facts removed before it saw them.
- `_compact_if_needed` returns `(graph_updates, local_view)`. The graph update speaks reducer contracts (`__reset__`, `RemoveMessage`); the local view is pre-resolved so callers rebuild prompts from real data rather than reducer markers.
- `compaction_summary` is now consumed by `think_node` and `finalize_node`.
- `COMPACTION_HISTORY_THRESHOLD` deleted (unused on this path). `COMPACTION_ASSISTANT_MAX_CHARS` is now actually used, as the assistant-turn cap in `select_recent_history` (previously a hardcoded 400).

### P1-3 — Recalled memory separated from citable evidence
`graph_state.py`, `load_context_node`, `plan_node`, `tool_node`

`load_context_node` writes memory hits to `recalled_memories` and resets `retrieved_docs` to `[]`. `plan_node` reads memory from the new field and labels it "context only, not evidence" in the prompt. `tool_node` no longer seeds its merged doc list with it. Recalled memory can therefore no longer be rendered as a `[KB-n]` chunk, cited, counted toward retrieval confidence, or scored for faithfulness.

### P1-4 — Per-turn accumulators reset
`load_context_node`

Added resets for `retrieved_docs`, `cited_doc_indices`, `answer_usage`, `final_answer`, `answer`, `precomputed_tool_calls`, `clarification_count`, `clarification_response`, `needs_clarification`, `resolution_provenance`. The fields the reviews wanted reset but that no code writes (`artifacts`, `subtask_contexts`, `retrieval_keys`) were deleted instead.

### Conditional, provenance-bound query resolution
`utils.resolve_retrieval_query`, `needs_reference_resolution`, `validate_resolution_provenance`, `_clean_rewrite`, `_call_rewriter`; `nodes.rewrite_query_node`

Both reviews' core recommendation, implemented without merging into `plan_node` (see §4):

- **Conditional.** `needs_reference_resolution` gates the LLM call on anaphora/ordinal/ellipsis markers plus the presence of history. Self-contained messages pass through byte-for-byte with no call, no cost, and no drift opportunity.
- **Provenance-bound.** `validate_resolution_provenance` allows added content tokens only if they appear in the original query, the recent verbatim turns, the compaction summary, `LastAnswerObject.summary`/`key_points`, recalled memory, or the clarification text. A "no new words" rule would be wrong — resolving "it" must add an entity — so the invariant is traceability, not novelty. Unsupported terms mean the original query is used and the rejection is logged with the offending tokens.
- **Fail-safe.** Parse failure, provider error, timeout, an answer-echo, or an unchanged rewrite all fall back to the original query. A free-form rewrite can never become authoritative unchecked.
- **Bounded blast radius.** `original_query` is now authoritative for `plan_node`, `finalize_node` and `answer_evaluation_node`; `rewritten_query` reaches retrieval and reranking only. `resolution_provenance` records what happened for audit.
- `max_tokens` raised 60 → 160 (60 truncated rewrites mid-phrase against a 30-word prompt cap), and the meta-commentary stripper no longer does a blind `rsplit(":", 1)` that mangled legitimate queries containing a colon.

### P2 items implemented

| Item | Change |
|---|---|
| P2-1 subtask identity | `_build_execution_summary` matches observations to subtasks **by count**, not by presence. Three `rag_retrieve` subtasks now require three successful retrievals. Chosen over threading `subtask_id` through tool calls because it is deterministic and needs no prompt change (see §4). |
| P2-2 finalize context | `finalize_node` now receives `original_query`, the bounded recent conversation, and the compaction summary, with explicit priority stated in the prompt: retrieved documents are evidence, conversation is intent. |
| P2-3 determinism | `think_node` tool selection at `temperature=0.0`. `finalize_node` prose stays at `0.7`. |
| P2-4 token usage | `finalize_node` captures `usage_metadata` from the streaming response into `answer_usage`; `agent_runner` uses it when present and otherwise marks the reconstructed figure `"estimated": true`. |
| P2-5 dead code | `compaction_node` deleted; `reflection` write deleted; unused `run_agent_loop` parameters removed and propagated through `pipeline.py`, `chat_service.py`, `query.py`; 16 dead `AgentState` fields removed. |
| P2-6 overlap pruning | `_prune_contiguous_overlaps` tracks the previous chunk index by enumeration order instead of `list.index(doc)`. |
| Duplication #3 | `select_recent_history` + new `history_to_text` are now the single history projection used by resolution, `think` and `finalize`. `think_node`'s bespoke inline loop is gone. |

### Tests
`backend/tests/test_agent_state_integrity.py` — 27 new tests, one per repaired defect:

- assistant turn appended once; turn 2 sees turn 1's answer exactly once; replay does not duplicate
- `started_at` / `force_finalize` / `precomputed_tool_calls` survive a node hop; wall-clock budget terminates the graph; `clarification_question` declared once
- three tool rounds persist exactly three observations; compaction replaces observations instead of appending
- recalled memory stays out of `retrieved_docs` in both `load_context_node` and `tool_node`
- one retrieval does not complete three subtasks; three do
- `interrupt()` produces a persisted checkpoint; `Command(resume=...)` reaches state; only the new message is appended; the clarification budget caps at one
- compaction removes old messages, keeps one summary, and a second compaction replaces rather than stacks
- evidence trimming drops lowest-score first and never empties the context
- self-contained passthrough, no-history passthrough, provenance rejection, provenance acceptance, resolver-failure fallback
- identical chunk dicts do not confuse the neighbour lookup

---

## 4. What was left out, and why

### Deferred by design

**Merging query resolution into `plan_node` (Sol P1-1 "cleanest minimal implementation", Opus §6).**
Not done. Both reviews present this as an *optimisation* — it removes one LLM call and one disagreement surface — and Opus explicitly orders it after the state-correctness work. The behavioural requirements (conditional, structured, provenance-bound, retrieval-scoped) are all implemented in `rewrite_query_node`; merging is now a pure call-count change. Doing it in the same pass would have entangled the plan schema, the `Plan` Pydantic model, `PLAN_SYSTEM_PROMPT`, and the clarification loop with the P0 fixes, making a regression impossible to attribute. Worth doing once the P0 fixes have real traffic behind them.

**Threading `subtask_id` through tool calls (Opus P2-1, Sol #3).**
Implemented by observation *counting* instead. Carrying `subtask_id` requires the acting LLM to emit it on every tool call — a prompt-compliance dependency on exactly the small/local models the codebase already works around (see the `force_finalize` short-circuit comment in `tool_node`). Counting is deterministic and closes the reported failure: a three-part plan can no longer finalize after one retrieval. If subtasks with the same hint ever need *distinguishing* rather than merely *counting*, the id-threading version becomes necessary.

**Scoring only cited chunks in `answer_evaluation_node` (Opus duplication #6, Sol duplication table).**
Not done. Faithfulness scored against only the cited subset cannot detect the failure mode it exists to catch: an answer that asserts something no retrieved chunk supports and simply omits the citation. The duplicated cost is real, but the reviews' proposed fix trades a correctness signal for a token saving. Making evaluation sampled or configurable — Sol's alternative — is a better direction and is not blocked by anything here.

**Folding `LastAnswerObject` extraction into `finalize` (Opus duplication #7).**
Not done. Sol explicitly classifies it as "not a correctness blocker; defer optimization until P0 fixes are complete". Structured output for `LastAnswerObject` would also have to survive the streaming path, which currently streams raw text; that is a separate design change.

**Removing the `tool → reflect → think` hop (Opus P2-5, Sol duplication table).**
Not done — and the premise no longer holds. `reflect_node` was inert *because* `precomputed_tool_calls` was undeclared (P0-3). With the key declared, its recovery rules fire for the first time. Sol's condition — "keep only if recovery rules are active and state fields are declared" — is now satisfied. The rules have never actually executed in production, so removing the hop now would be deleting untested-but-live behaviour.

**Wiring or removing `RedisMemory.save_turn` (both reviews).**
Not done. Both reviews frame this as a decision requiring product input: wiring it needs retention and privacy rules, removing it discards a built feature. The state boundary — the actual defect — is fixed regardless: recalled memory can no longer become evidence whether or not the store is ever populated. Flagging for a product decision rather than making it unilaterally.

**Emitting the clarification event from graph interrupt metadata rather than a custom event (Sol P0-3 #2).**
Done in substance, differently in form. The custom pre-interrupt event is removed and the event is now derived from the graph's `__interrupt__` update in `agent_runner`. The `/api/clarification` resume endpoint already used `resumed_stream.interrupted()` / `.interrupts()`, which is the same mechanism, so it needed no change — it just now actually has a persisted interrupt to find.

### Deferred as out of scope

**The full verification suites (Opus §9 behavioural set, Sol "Behavior transcript set").**
The structural half is implemented (27 tests, all deterministic, no live model). The behavioural half — fixed multi-turn transcripts measuring retrieval recall, entity-addition rate, topic-carryover rate, clarification success rate, unsupported-citation rate — needs a corpus, a fixture harness, and baseline numbers to compare against. It belongs with the existing `eval/` harness, not in `backend/tests/`. This is the single largest remaining gap: the fixes are structurally verified but not yet *behaviourally measured*, and both reviews are right that a fluent answer routinely hides a corrupted retrieval query.

**`COMPACTION_ENABLED` default.**
Left at `true`. Opus recommends not enabling compaction in production before §4 is fixed; §4 is now fixed and covered by tests, so flipping the default off would be a regression from the reviews' own criterion. It should still be watched on the first long conversation that actually triggers stage 3.

**Historical-memory retrieval (`HISTORICAL_MEMORY_*`).**
Untouched. Neither review examined it, and it is a separate path from `recalled_memories`.

### Deliberately not done

**Widening the history window as a drift fix; adding a second post-retrieval rewrite; importance scoring, pinned messages, or a memory-recall tool.**
Opus §10 lists these as explicitly not recommended, and I agree. None were added.

---

## 5. Changed files

| File | Nature of change |
|---|---|
| `backend/app/services/agentic_rag/graph_state.py` | Declared 3 live keys + 4 new ones; removed duplicate and 16 dead fields; hardened `accumulate` |
| `backend/app/services/agentic_rag/agent_graph.py` | Assistant-turn persistence, observation reducer fix, memory/evidence split, single compaction implementation with doc trimming, clarification repair, subtask counting, temperature, overlap pruning, wall-clock guard, graph rewiring |
| `backend/app/services/agentic_rag/nodes.py` | Deleted `compaction_node`; conditional provenance-bound `rewrite_query_node`; shared history projection; evaluation against `original_query` |
| `backend/app/services/agentic_rag/utils.py` | Replaced `rewrite_query` with `resolve_retrieval_query` + provenance validation + reference detection |
| `backend/app/services/agentic_rag/agent_runner.py` | `__interrupt__` handling, provider usage, `estimated` flag, parameter cleanup |
| `backend/app/services/agentic_rag/pipeline.py` | Parameter cleanup |
| `backend/app/services/agentic_rag/__init__.py` | Docstring updated for the removed node |
| `backend/app/services/chat/chat_service.py` | Removed the orphaned `_rewrite_query` wrapper; updated the call site |
| `backend/app/api/api_v1/query.py` | Fixed the broken `run_agentic_rag` call |
| `backend/app/core/config.py` | `+AGENT_MAX_CLARIFICATIONS`, `+AGENT_HISTORY_PAIRS`, `−COMPACTION_HISTORY_THRESHOLD` |
| `backend/tests/test_agent_state_integrity.py` | New — 27 structural tests |
| `backend/tests/test_query.py` | Mock signature corrected to the real API |
| `docs/FEATURES.md` | Compaction section updated to describe the budget-guard behaviour |

---

## 6. Follow-ups, ordered

1. Build the behavioural transcript suite in `eval/`. Everything here is structurally verified and behaviourally unmeasured.
2. Watch the first production conversation long enough to trigger compaction stage 3, and confirm the rendered prompt actually shrinks.
3. Decide on `RedisMemory.save_turn`: wire it with retention rules, or delete it.
4. Once 1–3 are settled, merge query resolution into `plan_node` to remove one LLM call per turn.
5. Make `answer_evaluation_node` sampled or configurable rather than re-sending the full evidence context.
