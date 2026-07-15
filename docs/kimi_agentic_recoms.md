# Recommendation: Autonomous Enterprise Assistant Architecture

**Date:** 2026-07-13  
**Scope:** Consolidate rag-web-ui into a single autonomous agentic RAG pipeline.  
**Decision:** Keep LangGraph as the orchestration backbone. Do **not** migrate the core pipeline to LangChain Agent or DeepAgents.

---

## 1. Executive Recommendation

**Use LangGraph (the current stack) as the backbone.** Wrap existing retrieval, generation, evaluation, and chart capabilities as LangGraph nodes/tools. Use LangChain `create_agent` only inside individual nodes where dynamic tool selection adds value. Do **not** adopt DeepAgents as the primary framework.

This is the only option that satisfies all three constraints simultaneously:

1. **Preserves the exact pipeline** you specified (rewrite → decompose → parallel/sequential subtasks → per-subtask retrieval with sufficiency checks → synthesis → evaluation).
2. **Keeps the codebase simple** by reusing the existing LangGraph graph, state schema, streaming events, and retrieval infrastructure.
3. **Avoids a high-risk rewrite** of a working system into a newer, less mature abstraction.

---

## 2. Why LangChain Agent Is the Wrong Choice

LangChain `create_agent` / `AgentExecutor` is a single ReAct-style tool-calling loop. It is designed for agents that:

- Receive one user query.
- Decide whether to call a tool or answer.
- Loop until done.

It does **not** natively support:

- Explicit query decomposition into a task list.
- Parallel vs. sequential subtask dispatch based on dependency analysis.
- A fixed synthesis stage after all subtasks complete.
- Conditional chart validation and retry.
- Per-subtask retrieval legs with RRF merging and sufficiency checks.

You could build all of that on top of `AgentExecutor`, but you would essentially be re-implementing LangGraph inside it. The current codebase already has LangGraph.

**Verdict:** LangChain Agent is too simple for this pipeline. Skip it.

---

## 3. Why DeepAgents Is Not the Right Choice Either

DeepAgents (`deepagents.create_deep_agent`) is an opinionated harness built on top of LangChain `create_agent`, which itself runs on LangGraph. It targets "deep research" agents that autonomously plan, spawn subagents, and iterate.

### What DeepAgents offers that matches your goal

- Subagent delegation.
- Autonomous planning and iteration.
- Context management.
- Built-in memory and checkpointer support.

### Why it still does not fit this codebase

| Concern | Impact |
|--------|--------|
| **Black-box orchestration** | DeepAgents decides how to plan and delegate. Your pipeline has *explicit* stages (sufficiency check → adaptive reranking → chart validation → answer evaluation). Mapping those exactly into DeepAgents' harness requires fighting the framework. |
| **Immature / rapidly changing** | It is a newer package with a smaller ecosystem than LangGraph. Adopting it as the core engine introduces dependency and API-stability risk. |
| **Not in the current dependency tree** | `requirements.txt` has `langchain` and `langgraph`, but not `deepagents`. Adding it is easy; maintaining a migration is not. |
| **Loss of fine-grained streaming control** | The frontend relies on typed SSE events (`p:`, `t:`, `th:`, `0:`, `2:`, `d:`) emitted from specific LangGraph nodes. DeepAgents would require significant work to produce the same granular event stream. |
| **Existing investment** | `backend/app/services/agentic_rag/` already implements the requested graph: `summarize_history` → `rewrite_query` → `classify_query` → `Send(agent_subgraph, ...)` → `synthesize`. Throwing it away is wasted effort. |

**Verdict:** DeepAgents is closer to the autonomous goal than LangChain Agent, but it is the wrong tool for a deterministic enterprise pipeline. Skip it as the primary framework.

---

## 4. What the Codebase Already Has

The current implementation in `backend/app/services/agentic_rag/` already covers most of your requirements:

| Requirement | Current Implementation |
|------------|------------------------|
| Rewrite query using history | `rewrite_query_node` in `nodes.py` |
| Decomposition + independence check | `classify_query_node` + `SubtaskIndependence` schema |
| Parallel/sequential subtask dispatch | `route_after_classify` in `graph.py` using `Send()` |
| Per-subtask retrieval | `dense_retrieval_node`, `sparse_retrieval_node`, `exact_retrieval_node` |
| Graph expansion | `graph_expansion` triggered by `sufficiency_check_node` |
| Reranking | `rerank` from `retrieval/reranker.py` |
| Adaptive reranking | `adaptive_reranking_node` |
| Answer generation | `generating_node` |
| Chart validation | `chart_validation_node` |
| Answer evaluation | `answer_evaluation_node` + `evaluator.py` |
| Synthesis | `synthesize_node` |
| Streaming events | `callbacks.py` / `graph_runner.py` |
| Checkpointing / memory | `SqliteSaver` with `thread_id=chat_id` |

The main gaps are not framework gaps; they are **completeness and cleanup** gaps:

1. Multiple legacy pipeline references in docs and possibly frontend UI still mention fast/thinking/agentic modes.
2. Retry logic is inconsistent (some nodes retry twice, some not at all).
3. `answer_evaluation` is present but its retry loop is basic.
4. Tool execution retries are not uniformly applied.
5. Historical memory and user profile modules exist but are not wired into the active graph.

---

## 5. Proposed Architecture

Keep the existing two-level LangGraph architecture, but simplify and harden it so it is the *only* pipeline.

```
Main Graph:
  START
    → summarize_history
    → rewrite_query  (always)
    → classify_query (always: clarity + subtasks + independence)
    → [clarification interrupt | Send(agent_subgraph, subtask)...]
    → synthesize
    → END

Agent Subgraph (per subtask):
  START
    → rewrite_subtask_query
    → orchestrator (decides retrieval strategy / tools)
    → [tool calls: keyword_search, dense_search, sparse_search, graph_expansion]
    → rrf_merge
    → sufficiency_check
    → [graph_expansion if needed]
    → rerank
    → adaptive_reranking (if sufficiency still low)
    → generate_answer
    → [chart_validation if chart query]
    → answer_evaluation
    → [retry if evaluation fails, max 3 total attempts]
    → collect_answer
    → END
```

### Key design rules

1. **One pipeline only.** Delete or disable all non-agentic RAG paths. Remove frontend mode selectors.
2. **Every tool call retried 3 times** with exponential backoff before failing the subtask.
3. **Every subtask retried up to 3 times** if answer evaluation fails.
4. **All state in `AgentState`.** No hidden local variables in nodes.
5. **Streaming is first-class.** Every node emits progress events; generation streams tokens.
6. **Graceful degradation.** If retrieval fails entirely, the agent answers from history or says it cannot find the answer rather than crashing.

---

## 6. Implementation Plan

### Phase 1: Consolidate to one pipeline

**Goal:** Ensure `run_agentic_rag` is the only active RAG path.

- Remove frontend references to fast/thinking/agentic modes (radio buttons, query params, state).
- Remove or deprecate any backend endpoints that expose non-agentic RAG.
- Update `docs/` to mark old pipeline documents as superseded.
- Verify `generate_response` in `chat_service.py` always calls `run_agentic_rag`.

### Phase 2: Standardize retries

**Goal:** Every external call retries 3 times.

- Create a shared `with_retry(fn, max_attempts=3)` decorator using `tenacity` (already in dependencies).
- Apply it to:
  - `hybrid_search_with_legs` and individual retrieval legs.
  - `rerank`.
  - `graph_service.expand_docs_via_graph`.
  - `db_query_tool` and `graph_query_tool`.
  - LLM calls in rewrite, classify, generate, evaluate nodes.
- Add a retry budget to `AgentState` so the graph does not loop forever.

### Phase 3: Harden per-subtask flow

**Goal:** Implement the exact per-subtask pipeline you described.

1. **Rewrite subtask query** — create `rewrite_subtask_query_node` that makes the subtask self-contained.
2. **Orchestrator** — update `orchestrator_node` to choose which retrieval legs/tools to call based on subtask type, instead of always calling all legs.
3. **Sufficiency check** — strengthen `sufficiency_check_node` to use both confidence score and doc count.
4. **Adaptive reranking** — keep `adaptive_reranking_node` but make it trigger only once per subtask.
5. **Chart validation** — replace heuristic with a real ECharts JSON validation tool; retry up to 3 times on failure.
6. **Answer evaluation** — expand `answer_evaluation_node` to grade faithfulness, completeness, citation quality; route back to retrieval if below threshold.

### Phase 4: Wire memory and context management

**Goal:** Make the agent truly aware of long conversation history and user preferences.

- Wire `historical_memory.retrieve_historical_memory` into the main graph before retrieval.
- Wire `user_profile` lookups into `rewrite_query_node`.
- Use `compress_context_node` between iterations when retrieved context exceeds a token budget.

### Phase 5: Evaluation and cleanup

**Goal:** Verify the single pipeline works end-to-end.

- Run the existing `eval/` harness against the consolidated pipeline.
- Add tests for:
  - Retry exhaustion.
  - Parallel vs. sequential subtask routing.
  - Clarification interrupt.
  - Chart validation failure and retry.
- Remove dead code identified in `docs/codebase-simplification-audit.md` and `docs/final-audit-report.md` that becomes orphaned by the consolidation.

---

## 7. Where LangChain Agent Fits

Although LangChain Agent is wrong for the top-level orchestration, it is useful inside individual LangGraph nodes for **dynamic tool selection**.

Use `langchain.agents.create_agent` inside a node when:

- A subtask requires the LLM to choose among many tools dynamically.
- The exact sequence of tool calls is not known in advance.
- You want a ReAct loop for a contained subproblem.

Example:

```python
from langchain.agents import create_agent

research_agent = create_agent(
    model="openai:" + settings.OPENAI_MODEL,
    tools=[db_query_tool, graph_query_tool, web_search_tool],
    system_prompt="You are a research subagent. Call tools to gather evidence.",
)
```

This agent can be invoked from within a LangGraph node, and its result returned to the graph state. The graph still controls the overall pipeline.

**Do not** use `AgentExecutor` as the top-level orchestrator.

---

## 8. Where DeepAgents Does Not Fit

DeepAgents is designed for open-ended research agents, not for deterministic enterprise RAG pipelines. Do not use it for:

- Replacing the existing LangGraph StateGraph.
- Implementing the exact per-subtask retrieval flow.
- Driving the SSE event stream consumed by the frontend.

Reconsider DeepAgents only if the product later needs an open-ended "research mode" that is separate from the structured RAG pipeline.

---

## 9. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Stakeholder expects DeepAgents/LangChain Agent | Show this document. The requirement is "autonomous enterprise assistant," not "use framework X." LangGraph already delivers the autonomy with more control. |
| Existing graph has bugs | Fix them within the current framework; do not rewrite. The audit found real issues but no framework issues. |
| Frontend depends on agent step events | Keep LangGraph; only it currently emits the `4:` agent step events the timeline UI consumes. |
| Retries increase latency | Cap retries at 3, use exponential backoff, and make retries visible in progress events. |

---

## 10. Decisions / Clarifications

| # | Question | Decision |
|---|----------|----------|
| 1 | Framework choice | **Keep LangGraph as the backbone.** LangChain Agent / DeepAgents are not required. |
| 2 | Frontend modes | **Always use the agentic pipeline.** Remove fast/thinking/agentic selectors from the UI. |
| 3 | External tools | **Existing KB retrieval tools only.** This is an offline application; no web search, API calls, or DB writes. Allowed tools: `keyword_search`, `dense_search`, `sparse_search`, `graph_expansion`. |
| 4 | Chart validation | **Validate ECharts JSON schema only.** No rendering pass. |
| 5 | Retry scope | **Per tool call and per LLM generation.** Each tool invocation and each LLM call may retry up to 3 times. Subtask-level retries are not required; failure after retries surfaces as low-confidence output. |
| 6 | Answer evaluation failure | **Produce a low-confidence final answer.** If retrieval, adaptive reranking, and all tool calls have been exhausted and the answer still fails evaluation, do not loop indefinitely. Emit the best answer with a low-confidence flag. |
| 7 | Historical memory | **Current chat only.** Retrieve relevant past assistant answers only from the current conversation history. |

---

## 11. Updated Implementation Plan

With the decisions above, the implementation becomes a focused hardening of the existing LangGraph pipeline rather than a migration.

### Phase 1: Consolidate to one pipeline

**Goal:** Make the agentic pipeline the only RAG path in the UI and backend.

- Remove fast/thinking/agentic radio buttons, query params, and related state from the frontend chat components.
- Delete any backend endpoints or service branches that still expose a non-agentic RAG mode.
- Update `chat_service.generate_response` so it always calls `run_agentic_rag` and never short-circuits to a legacy path.
- Mark old pipeline docs (`AGENTIC_AGENT_REDESIGN.md`, `AGENTIC_RAG_PROPOSAL.md`, etc.) as superseded in their headers.

### Phase 2: Standardize retries

**Goal:** Every external call and every LLM generation retries up to 3 times.

- Add a shared `with_retry(max_attempts=3)` utility using `tenacity`.
- Apply retries to:
  - Individual retrieval legs (`_dense_search`, `_sparse_search`, `_exact_search`).
  - `hybrid_search_with_legs`.
  - `rerank`.
  - `graph_service.expand_docs_via_graph`.
  - `db_query_tool` / `graph_query_tool`.
  - All LLM calls: rewrite, classify, generate, evaluate.
- Track retry exhaustion in `AgentState` so nodes do not loop forever.
- Emit progress events when a retry happens so the UI can show "retrying retrieval...".

### Phase 3: Harden per-subtask flow

**Goal:** Implement the exact per-subtask pipeline with the clarified scope.

1. **Rewrite subtask query** — add `rewrite_subtask_query_node` that makes each subtask self-contained.
2. **Orchestrator** — restrict `orchestrator_node` to the four allowed KB tools. It may decide which tools to call per subtask but cannot invent external tools.
3. **Retrieval legs** — keep dense, sparse, exact, and graph legs. Call them as tools from the orchestrator.
4. **Sufficiency check** — route to graph expansion if confidence < threshold or doc count is low. After graph expansion, run reranking.
5. **Adaptive reranking** — trigger at most once per subtask when confidence is still low after the first reranking pass.
6. **Answer generation** — LLM call with retry; stream tokens to the frontend.
7. **Chart validation** — JSON-schema validation of ECharts options only. Retry generation up to 3 times on schema failure; if still failing, mark subtask as low-confidence and continue.
8. **Answer evaluation** — grade faithfulness, completeness, citation quality. If evaluation fails after all retrieval and retries, mark the subtask as low-confidence and continue to synthesis.

### Phase 4: Wire historical memory and context management

**Goal:** Use current-chat history without adding external memory sources.

- Wire `historical_memory.retrieve_historical_memory` to search only the current chat's prior assistant answers.
- Use the existing rolling summary and sliding window for recent context.
- Use `compress_context_node` when retrieved context exceeds the configured token budget.
- Do not wire `user_profile` unless product requirements change; keep it out of scope for now to maintain simplicity.

### Phase 5: Synthesis and final confidence

**Goal:** Combine subtask answers and surface confidence honestly.

- `synthesize_node` merges subtask answers.
- Aggregate confidence from subtask evaluations:
  - If any subtask is low-confidence, the final answer is low-confidence.
  - If all subtasks pass evaluation, the final answer is high-confidence.
- Emit a final `context` event with confidence level and breakdown.
- Do not retry the whole query if synthesis produces a low-confidence answer; the pipeline is done.

### Phase 6: Evaluation and cleanup

**Goal:** Verify the single pipeline works end-to-end and remove dead code.

- Run `pytest` for existing agentic RAG tests.
- Run the `eval/eval_harness.py` end-to-end on a small dataset.
- Add focused tests for:
  - Retry exhaustion on a failing retrieval leg.
  - Chart JSON-schema validation failure.
  - Parallel vs. sequential subtask routing.
  - Low-confidence final answer path.
- Remove dead code identified in `docs/codebase-simplification-audit.md` and `docs/final-audit-report.md` that becomes orphaned by the consolidation.

---

## 12. Updated Design Rules

1. **One pipeline only.** The UI always uses `run_agentic_rag`.
2. **No external tools.** Only KB retrieval tools are allowed.
3. **Retries are per call/generation, max 3.** No subtask-level retries.
4. **Low-confidence final answer is the terminal fallback.** Never loop forever.
5. **Historical memory is current-chat only.** No cross-chat retrieval.
6. **Chart validation is JSON-schema only.** No rendering.
7. **Streaming is first-class.** Every node emits progress; generation streams tokens.

---

## 13. Final Verdict

**Do not migrate to LangChain Agent or DeepAgents.** The codebase already has the right framework in LangGraph. Invest the implementation effort in:

1. Removing legacy pipeline remnants.
2. Standardizing retries across all nodes and tools.
3. Hardening the per-subtask flow.
4. Wiring historical memory and user profile.
5. Evaluating end-to-end.

This delivers the autonomous enterprise assistant goal with the least complexity and the lowest risk.
