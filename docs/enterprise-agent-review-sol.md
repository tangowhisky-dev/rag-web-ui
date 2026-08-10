# Enterprise Agentic Retrieval Pipeline Review

Date: 2026-08-10

Scope: the active agent loop under `backend/app/services/agentic_rag/`, its chat and clarification entry points, retrieval tool, Redis checkpointer, prompts, state reducers, and current tests.

## Executive conclusion

The retrieval stack itself does not need replacement. The dense, sparse, exact, reranking, graph-expansion, relaxation-ladder, citation-normalization, and deterministic completion pieces are reasonable. The main defects are in conversation state and context ownership around that stack.

The primary cause of multi-turn drift is that the checkpointed `messages` channel receives user messages but never receives the generated assistant answer. The query rewriter, planner, actor, and compactor therefore operate on different and incomplete views of the conversation. A follow-up such as "compare it with the first approach" cannot be resolved reliably when the graph contains the prior user question but not the answer that introduced those approaches.

Query rewriting is still required for context-dependent retrieval queries. Modern chat LLMs can understand a follow-up from prior messages only when those messages are included in that model call. The dense embedder, sparse search, exact search, and cross-encoder each receive one query string and do not see chat history. However, rewriting should not run as an unconditional free-form paraphrase. Self-contained messages should pass through byte-for-byte; only references such as "it", "that section", or "the second option" need resolution.

Recommended scope: keep the graph and retrieval architecture, but repair the six issues below. Do not undertake a broad agent rewrite.

## Current structure and context flow

```text
chat_service
  -> run_agent_loop (one HumanMessage added to checkpointed thread)
  -> load_context
       previous LastAnswerObject from MySQL
       semantic memory hits placed in retrieved_docs
       reset selected per-turn fields
  -> rewrite_query
       original query + up to 3 recent pairs from messages
  -> compaction
       may summarize old messages into compaction_summary
  -> plan
       rewritten query + LastAnswerObject summary + semantic memory + file metadata
  -> clarify_interrupt OR think
  -> tool <-> reflect <-> think
       rag_retrieve runs dense + sparse + exact + rerank + optional graph expansion
  -> reflect_final
  -> finalize
       rewritten query + retrieved_docs + non-retrieval tool results
  -> answer_scoring
  -> save_memory
       answer and metadata saved to MySQL, but not to graph messages
```

| Stage | Current conversation input | Important omission or duplication |
|---|---|---|
| `rewrite_query_node` | Last three user/assistant pairs from `messages` | In production, assistant messages are never appended; assistant text is also truncated to 400 characters |
| `compaction_node` | Full `messages` channel | Produces `compaction_summary`, but no later prompt reads it and messages are not replaced |
| `plan_node` | Rewritten query, prior `LastAnswerObject.summary`, recalled memory, files | Does not receive normal chat history; query and prior answer are separate fragments |
| `think_node` | Rewritten query, recent history, truncated prior answer, plan, observation metadata | Rebuilds a second conversation view with different limits |
| `finalize_node` | Rewritten query, retrieved documents, non-RAG tool outputs | Receives neither the original user wording nor normal conversation history nor prior-answer context |
| `answer_evaluation_node` | Rewritten query and all retrieved documents | Re-sends the full evidence context and evaluates against the rewrite rather than the user's exact request |

There is no single authoritative turn context. The rewritten query, rather than the original user message plus a separately resolved retrieval query, becomes the input to planning, acting, final generation, reranking, and evaluation.

## Required findings and fixes

### P0-1: Assistant turns are absent from checkpointed conversation state

Evidence:

- `agent_runner.run_agent_loop` initializes each turn with one `HumanMessage`.
- `chat_service` deliberately delegates history to the Redis/LangGraph checkpoint.
- No active node appends an `AIMessage`; `finalize_node` returns `answer` and `final_answer`, while `save_memory_node` only updates the MySQL row.
- The only `AIMessage(...)` construction in the agentic service is inside `select_recent_history`, where it copies messages that would already need to exist.

Impact:

- Reference resolution sees prior questions without prior answers.
- Topic switches and ordinal references are easy to misresolve.
- Compaction summarizes an incomplete conversation.
- `last_answer_object.summary[:300]` becomes an accidental substitute for real history, but cannot preserve exact lists, formatting, entities, or long answers.

Required fix:

1. Append `AIMessage(content=final_answer)` to `messages` once the final answer is stable.
2. Keep MySQL as the durable display/audit record and the LangGraph checkpoint as short-term conversational state; both must contain the completed turn.
3. Add a two-turn test that inspects the turn-two checkpoint and proves it contains `HumanMessage(turn 1)`, `AIMessage(turn 1)`, and `HumanMessage(turn 2)` exactly once.

### P0-2: Observation reducers duplicate state exponentially

`AgentState.observations` uses the append-style `accumulate` reducer. `tool_node` reads all existing observations, appends the new ones locally, and returns the entire list. LangGraph then appends that full returned list to the existing channel.

The resulting size follows `n_next = 2 * n_previous + new`, so one observation per round produces 1, 3, 7, 15 rather than 1, 2, 3, 4. `_compact_if_needed` repeats the same reducer mistake when it returns a full compacted observation list as though it were a replacement.

Impact:

- Inflated prompts and persisted tool-call records.
- Incorrect document totals in `_build_execution_summary`.
- Repeated idempotency matches and misleading token accounting.
- "Compaction" can increase state size.

Required fix:

- Make `tool_node` return only observations created in the current invocation.
- For replacement during compaction, use the existing reset marker contract: return `[{'__reset__': True}, *compacted_observations]`, or define a dedicated replacement channel. Do not return a replacement list through an append-only reducer.

### P0-3: Clarification interrupt and resume do not preserve the clarified intent

The current flow has four defects:

1. `clarify_interrupt_node` catches `Exception` around `interrupt()`. LangGraph's interrupt is a control-flow exception, so broad catching can swallow the pause and continue with an empty response.
2. The node emits a custom `interrupt` event before calling `interrupt()`. `chat_service` breaks its stream when it sees that custom event, which can close the graph stream before the node reaches a persisted interrupt checkpoint.
3. After resume, the edge goes directly from `clarify_interrupt` to `plan`. The original rewrite is not recomputed, and `plan_node` does not consume `messages`; therefore the clarification answer may not affect the plan or retrieval query.
4. There is no graph-level clarification counter. The API advertises two attempts, but the graph can continue `plan -> clarify -> plan` independently of that UI limit.

The resume endpoint does use `Command(resume=body.response)` with the correct chat-derived thread id, but that cannot repair the stale rewritten query or a pause that was never checkpointed.

Required fix:

- Call `interrupt()` without a broad exception handler and let LangGraph persist and expose the interrupt.
- Emit the UI clarification event from the graph's actual interrupt metadata, not a custom event before the interrupt.
- On resume, append only the new clarification `HumanMessage`, combine it explicitly with the unresolved user request, and route back through intent/query resolution before planning.
- Add `clarification_count` to state and enforce one clarification round by default. If ambiguity remains, answer with the limitation instead of looping.
- Do not separately store the clarification response as a normal independent user turn unless the graph/message persistence contract guarantees it will not be duplicated.

### P0-4: Compaction has two implementations, neither controls the real prompt size

There are two separate mechanisms:

- `nodes.compaction_node`, always placed before planning.
- `agent_graph._compact_if_needed`, called inside think and finalize.

They have different behavior and neither is reliable:

- `compaction_node` writes only `compaction_summary`; no downstream prompt reads it and it does not remove old messages.
- `_compact_messages_llm` returns `[summary] + recent` through the `MessagesState` add-message reducer. That is not a replacement operation; old messages can remain while a summary is appended.
- `_compact_if_needed` estimates savings by shrinking observation payloads. `think_node` already includes only retrieval metadata, while `finalize_node` is dominated by `retrieved_docs`, which this compactor does not trim.
- Both summarizers call synchronous `llm.invoke` inside async functions.
- The pre-plan threshold is based on global model settings and cumulative system prompts, not the actual next model request or per-organization model context window.
- `_messages_to_conversation_text` truncates individual turns before summarization, so the summary cannot preserve facts that were removed first.

Required fix:

1. Keep one context-budget service invoked with the exact messages for the next LLM call.
2. Compact completed conversation turns after the answer is appended, using LangGraph message removal/replacement semantics and one stable summary message id.
3. Keep a small recent verbatim window plus one structured summary. Feed both into intent resolution, planning/acting where needed, and finalization.
4. For finalize overflow, trim or pack `retrieved_docs` by reranker score and diversity until the actual rendered prompt fits. Conversation summarization cannot solve an evidence-payload overflow.
5. Use `ainvoke` and the organization-specific model configuration and context window.
6. Keep emergency pre-call compaction as a guard, but share the same implementation and verify the rebuilt prompt after each reduction.

Until these changes have deterministic checkpoint tests, compaction should be treated as unverified, not as a production safety mechanism.

### P1-1: Query rewriting is unconditional and becomes authoritative beyond retrieval

`rewrite_query` correctly instructs the model not to introduce entities, but this is only a prompt rule. The output is unstructured free text and the runtime validates only a few answer-like phrases. Every self-contained query still incurs an unnecessary model call and a chance to drift.

The larger structural problem is that `rewritten_query` is used not only by retrieval, but also by planning, final answer generation, reranking, and answer evaluation. A bad rewrite can therefore alter both what is searched and what is answered.

Required policy:

- Preserve three distinct fields:
  - `original_query`: exact user wording, authoritative for planning, clarification, finalization, and evaluation.
  - `retrieval_query`: standalone text used only by retrieval and reranking.
  - `resolution_provenance`: message ids or prior-answer fields used to resolve references.
- Return the original query byte-for-byte for self-contained turns.
- Resolve only context-dependent references. Any new named entity in `retrieval_query` must be traceable to the original query, recent verbatim turns, the conversation summary, or `LastAnswerObject`.
- On timeout, parse failure, or failed provenance validation, use the original query and let retrieval insufficiency or clarification handle the result.

The cleanest minimal implementation is to merge resolution into the existing structured `plan_node` call. Planning already decides intent and clarification from the same context. Its schema can return `needs_resolution`, `retrieval_query`, and `resolution_provenance` alongside the plan, removing one LLM call and preventing planner/rewriter disagreement.

Do not remove query resolution entirely. A modern answer LLM can understand history, but the retrieval components cannot because they receive only one string.

### P1-2: Recalled model memory is mixed with citable document evidence

`load_context_node` stores semantic memory search results in `retrieved_docs`. `tool_node` preserves those entries when merging actual retrieval results. Finalization then formats all of them as document context, citation normalization can cite them, and answer scoring can treat them as evidence.

This is a provenance error: a prior model answer is conversational memory, not a knowledge-base source. It can create a self-reinforcing hallucination loop. At present the risk is mostly latent because `RedisMemory.save_turn` has no caller, but the state boundary is still wrong.

Required fix:

- Add a separate `recalled_memories` field.
- Permit recalled memory to inform reference resolution or planning only.
- Never include recalled memory in `retrieved_docs`, citations, retrieval confidence, or faithfulness evaluation.
- Either intentionally wire `save_turn` with retention/privacy rules or remove the unused semantic-memory path. Do not leave it half-active.

## Additional structural bugs worth fixing in the same pass

These are smaller but directly affect correctness:

1. `AgentState` does not declare `started_at`, `force_finalize`, `precomputed_tool_calls`, or `reflection`, although nodes write them. LangGraph state schemas may discard undeclared updates. Declare the three live fields and delete `reflection` if it remains unread. Add tests for wall-clock termination and reflect recovery.
2. `clarification_question` is declared twice in `AgentState`; remove the duplicate.
3. `_build_execution_summary` considers every subtask with the same `tool_hint` complete after one matching observation. A multi-part plan with three `rag_retrieve` subtasks can finalize after one retrieval. Carry `subtask_id` through tool calls and observations and verify each subtask independently.
4. `think_node` uses temperature `0.7` for tool selection. Tool choice and JSON generation should use `0.0`; keep creative temperature only for final prose if desired.
5. `run_agent_loop` accepts `temperature`, `model_name`, `api_base`, and `query_model`, but does not consistently apply them. Remove unused parameters or propagate them through the organization-aware LLM factory.
6. Reported prompt-token usage is reconstructed from incomplete fragments rather than provider usage. Capture actual response usage where supported; otherwise label it explicitly as an estimate.
7. Several accumulator fields from older retrieval/subtask designs are not reset per turn. Remove dead fields, then reset every remaining per-turn accumulator in `load_context_node`.

## Duplication and redundancy assessment

| Duplication | Decision |
|---|---|
| Separate rewrite and plan calls over nearly the same intent context | Merge into one structured intent-resolution/planning call |
| Two compaction implementations | Replace with one prompt-aware context-budget service |
| Different recent-history assembly in rewrite and think | Build one `TurnContext` projection with explicit per-node views |
| `LastAnswerObject` repeated in plan and think | Keep as a compact action artifact, but derive and cap it once |
| Recalled memory included in plan and evidence | Keep only in intent context; remove from evidence |
| Full evidence sent to finalize and again to evaluation | Evaluate cited/used chunks only, or make evaluation sampled/configurable |
| Separate LLM extraction of `LastAnswerObject` after generation | Not a correctness blocker; defer optimization until P0 fixes are complete |
| Reflection hop on every tool round | Keep only if recovery rules are active and state fields are declared; otherwise remove the inert hop |

Not redundant and worth retaining:

- Retrieval's internal graduated relaxation ladder.
- Metadata-only retrieval observations in `think_node`.
- Content-hash deduplication and contiguous chunk overlap pruning.
- Deterministic tool budgets and final execution verification.
- Original-query and retrieved-evidence audit records.

## Minimal target structure

```text
load_context
  - reset all per-turn state
  - load previous answer artifact
  - load recalled memory into a non-evidence channel
        |
resolve_and_plan (one structured, temperature-0 call)
  - original query remains immutable
  - exact passthrough or provenance-bound retrieval query
  - plan + clarification decision
        |
        +-- clarify once --> interrupt --> resolve_and_plan
        |
think <--> tool
  - one shared conversation projection
  - tool returns only new observations
  - observations carry subtask ids
        |
reflect_final
        |
finalize
  - original query + bounded recent conversation/summary
  - score/diversity-packed retrieved evidence
        |
answer_scoring (cited evidence only)
        |
save answer + append AIMessage
        |
compact completed conversation if required
```

This is a restructuring of context ownership, not a replacement of the retrieval pipeline.

## Implementation order

1. Persist assistant messages and add a two-turn checkpoint test.
2. Fix append-versus-replace reducer usage for observations and messages.
3. Repair clarification interrupt/resume and cap clarification rounds.
4. Separate `original_query` from `retrieval_query`; merge conditional resolution into planning.
5. Separate recalled memory from document evidence.
6. Replace both compactors with one prompt-aware implementation, then test it against real checkpoint semantics.
7. Add per-subtask observation identity and clean up undeclared/dead state.

Do not optimize LLM call count, remove reflection, or fold `LastAnswerObject` extraction into generation before the state correctness work above. Those are cost/latency improvements, not causes of the reported drift.

## Required verification suite

Use deterministic fake models for structural tests and a fixed transcript set for behavior tests.

Structural tests:

1. Turn two sees turn one's assistant answer exactly once.
2. Three tool rounds persist exactly three new observations.
3. Two consecutive compactions leave one summary, preserve the recent verbatim window, remove old messages, and reduce the actual rendered prompt token count.
4. Finalize evidence packing always fits the configured model window including reserved output tokens.
5. `interrupt()` produces a persisted checkpoint; `Command(resume=...)` changes the resolved retrieval query and plan.
6. A second clarification request is refused or converted into a bounded limitation response.
7. Wall-clock and tool-call caps terminate the graph.
8. Three same-tool subtasks require three matching subtask observations.
9. Recalled memories never appear in citations or answer-evaluation evidence.

Behavior transcript set:

- Self-contained query after an unrelated topic: exact query passthrough and no prior-topic terms.
- Pronoun follow-up: correct referent and recorded provenance.
- Ordinal follow-up such as "expand the second point": resolves against the assistant answer.
- Long-distance reference after compaction: resolves from the structured summary.
- Ambiguous reference with two plausible antecedents: one clarification, then correct retrieval.
- Clarification answer that changes topic: no stale terms from the original rewrite.
- Multi-part question: each planned part has independent retrieval/evidence coverage.

Track at least retrieval recall, rewrite entity-addition rate, topic-carryover rate, clarification success rate, unsupported citation rate, total prompt tokens, and end-to-end latency. A lower entity-addition rate alone is not sufficient if pronoun-resolution recall also falls.

## Final recommendation

Keep query resolution, but make it conditional, structured, and restricted to retrieval. Modern LLM context understanding does not remove the need to provide a standalone string to non-chat retrieval components. The current drift is primarily a missing-assistant-history and context-ownership problem, not evidence that agentic retrieval itself is the wrong architecture.

The required redesign is narrow: one complete conversation state, one immutable original query, one provenance-bound retrieval query, one compaction mechanism, and strict separation between conversational memory and citable evidence.
