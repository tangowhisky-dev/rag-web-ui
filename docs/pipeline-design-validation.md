# Pipeline Design Validation: Design Spec vs Actual Implementation

**Date:** 2026-07-10  
**Scope:** LangGraph-based agentic RAG pipeline — nodes, edges, routing, retries, conditional branching  
**Reference:** GiovanniPasq/agentic-rag-for-dummies (LangGraph agent pattern for comparison)

---

## 1. Simple Query Execution Order — Design vs Actual

| # | Step | Design Spec | Actual Implementation | Match? |
|---|------|------------|----------------------|--------|
| 1 | rewriting | always | `rewrite_query` node runs after START | ✅ |
| 2 | keyword_search | if exact leg enabled (default: yes) | Bundled in `hybrid_search_with_legs()` inside `direct_retrieval_node`; controlled by `use_exact` param | ✅ (bundled) |
| 3 | dense_search | if dense leg enabled (default: yes) | Bundled in `hybrid_search_with_legs()`; controlled by `use_dense` param | ✅ (bundled) |
| 4 | sparse_search | if sparse leg enabled (default: yes) | Bundled in `hybrid_search_with_legs()`; controlled by `use_sparse` param | ✅ (bundled) |
| 5 | sufficiency_check | before graph expansion | `sufficiency_check_node` routes to `graph_expansion` when `needs_graph_expansion=True` | ✅ |
| 6 | graph_expansion | if graph found new chunks (conditional) | Runs unconditionally when sufficiency fails — no check for "new chunks" | ⚠️ Gap |
| 7 | reranking | always | Always runs (after sufficiency or graph_expansion) | ✅ |
| 8 | adaptive_reranking | conditional (if confidence low) | Runs always but early-returns if confidence >= 0.3. Functionally conditional. | ✅ |
| 9 | generating | always | Always runs (unless budget exceeded) | ✅ |
| 10 | answer_evaluation | always, after generation, before done | **NOT IMPLEMENTED** — no `answer_evaluation` node exists | ❌ Missing |
| 11 | chart_validation | conditional (if charts) | Runs conditionally after generating when `is_chart_query=True` | ✅ |

---

## 2. Complex Query Execution Order — Design vs Actual

| # | Step | Design Spec | Actual Implementation | Match? |
|---|------|------------|----------------------|--------|
| 1 | rewriting | always | Same as simple path | ✅ |
| 2 | decomposition | if complex | LLM extracts questions via `QueryAnalysis` schema → stored as `subtasks`. No explicit decomposition node. | ⚠️ Partial |
| 3 | [task list] | shown to user | **NOT IMPLEMENTED** — `subtasks` stored in state but never surfaced to user | ❌ Missing |
| 4 | per subtask | a. rewriting through j. chart_validation | Processed sequentially by orchestrator index (0, 1, 2...). No parallel execution. | ❌ Wrong model |
| 5 | synthesizing | if multiple tasks | `synthesize_node` exists but only receives 1 item (see gap #3 below) | ⚠️ Broken |
| 6 | answer_evaluation | always, after all tasks | **NOT IMPLEMENTED** | ❌ Missing |
| 7 | done | — | END node reached | ✅ |

---

## 3. Critical Architecture Gaps

### 3.1 Missing `answer_evaluation` Node (Must Fix)

The design spec mandates an answer evaluation step after generation and before completion. This should evaluate whether the generated answer meets quality thresholds, can cite sources, and is self-consistent. No such node exists in `backend/app/services/agentic_rag/nodes.py` or `graph.py`.

**Impact:** Both simple and complex query paths skip answer quality verification.

### 3.2 No Parallel Execution — Sequential Instead of Simultaneous (Must Fix)

The reference repo uses LangGraph's `Send()` API for true parallel subgraph execution:

```python
# Reference repo: edges.py — parallel Send() calls
def route_after_rewrite(state: State) -> Literal["request_clarification", "agent"]:
    if not state.get("questionIsClear", False):
        decision = "request_clarification"
    else:
        decision = [
            Send("agent", {"question": query, "question_index": idx, "messages": []})
            for idx, query in enumerate(state["rewrittenQuestions"])
        ]
    return decision
```

Our `route_after_classify` routes only once:

```python
# Our implementation: graph.py
def route_after_classify(state: AgentState) -> str:
    if not state.get("question_is_clear", True):
        return "request_clarification"
    if state.get("is_complex", False) and len(state.get("subtasks", [])) > 1:
        return "agent_subgraph"       # Single path, not parallel
    return "direct_retrieval"
```

The `agent_subgraph` processes subtasks sequentially via an index counter in the orchestrator. Independent subtasks are **not** executed simultaneously.

**Impact:** Complex queries take linearly longer than necessary. The design spec explicitly calls for "simultaneous agents for independent sub tasks."

### 3.3 Synthesis Broken by collect_answer (Must Fix)

`collect_answer_node` always returns a single-answer list, which means multi-task synthesis never works:

```python
# Our implementation: nodes.py
def collect_answer_node(state: AgentState) -> dict:
    """Collect the answer from the agent subgraph."""
    answer = state.get("answer", "")
    return {
        "subtask_answers": [{"answer": answer}],   # Always exactly 1 item
    }
```

`synthesize_node` then receives exactly 1 subtask answer and treats it as a simple single answer:

```python
# nodes.py — synthesize_node
def synthesize_node(state: AgentState) -> dict:
    subtask_answers = state.get("subtask_answers", [])
    if len(subtask_answers) > 1:
        # Multi-task synthesis path — never reached
        ...
    elif subtask_answers:
        first = subtask_answers[0]
        final_answer = first.get("answer", first)  # Single answer returned as-is
    ...
```

The `agent_subgraph` compiles and returns a single answer to the main graph's `synthesize` node. Even if the orchestrator processes multiple subtasks internally, only the last one is collected.

**Impact:** Multi-task complex queries return only the final subtask answer with no synthesis.

### 3.4 No Task List Display (Should Fix)

The design spec shows the task list to the user before processing subtasks:

```
3. [task list shown]
```

Our `classify_query_node` extracts subtasks from the LLM's structured output but never creates a user-facing "I'll answer these X questions" message. The subtasks are stored in state and consumed by the routing logic, but the user never sees them.

---

## 4. Reference Repo Concepts — Coverage Analysis

| Feature | Reference Repo | Our Implementation | Match? |
|---------|---------------|-------------------|--------|
| **Hierarchical Indexing** (parent/child chunks) | Parent chunks (markdown sections) + child chunks (small pieces); retrieve child for precision, fetch parent for context | Parent-child exists in datastore layer but not in the pipeline nodes. `direct_retrieval_node` doesn't have parent-fetch logic. | ⚠️ Partial |
| **Conversation Memory** (rolling summary + bounded history) | `summarize_history` node creates rolling summary; bounded message history | No conversation summary in our state. Messages passed but no summarization. | ❌ Missing |
| **Query Clarification** | LLM decides if query is clear; pauses for clarification | `request_clarification_node` exists; loops back to classify. Clarification response appended to messages without special handling. | ⚠️ Partial |
| **Agent Orchestration** (LangGraph subgraph) | Full agent subgraph with tool-based retrieval (LMM calls `search_child_chunks` and `retrieve_parent_chunks` tools) | Our agent subgraph with orchestrator → retrieval → sufficiency → reranking → generating. Orchestrator doesn't call tools — it just sets state and routes back. | ⚠️ Different pattern |
| **Multi-Agent Map-Reduce** (parallel sub-queries) | `Send()` spawns N parallel agent subgraphs | Sequential processing via index. No Send(). | ❌ Missing |
| **Self-Correction** (re-queries if insufficient) | Orchestrator sees empty/wrong results and calls tools again. `should_compress_context` routes back to orchestrator on budget overflow. | `sufficiency_check` only routes to `graph_expansion`, not back to `direct_retrieval` for retry. No feedback loop. | ❌ Missing |
| **Context Compression** | `compress_context` node compresses full agent conversation into summary, preserving retrieval keys to avoid repeats | `compress_context_node` exists but is **dead code** — defined in `nodes.py`, imported in `graph.py`, but never added as a node or edge in any graph. | ❌ Dead code |
| **Observability** (Langfuse) | Built-in execution logging, Langfuse integration | Not present in our pipeline code. | ❌ Missing |
| **Dual Retrieval Legs** (dense + sparse/hybrid) | Single `search_child_chunks` tool (dense only) | Three search legs: keyword, dense, sparse — more sophisticated. | ✅ Exceeds |
| **Adaptive Reranking** | Not present | `adaptive_reranking_node` re-runs retrieval with `return_full_pool=True` when confidence < 0.3. | ✅ Exceeds |
| **Fallback Response** | `fallback_response` node generates best-effort answer from partial data | `fallback_response_node` returns generic "couldn't find enough info" (no generation attempt). | ⚠️ Partial |
| **Evaluation** (RAGAS metrics) | RAGAS evaluation notebook with answer/retrieval scoring | `evaluator.py` exists in `agentic_rag/` but not integrated into pipeline. | ⚠️ Not integrated |

---

## 5. Architecture Diagram

### Our Current Graph Structure

```
Main Graph:
  START → rewrite_query → classify_query
    ├─→ request_clarification → classify_query (loop)
    ├─→ direct_retrieval → synthesize → END          (simple path)
    └─→ agent_subgraph → synthesize → END            (complex path)

Agent Subgraph:
  START → orchestrator
    ├─→ direct_retrieval → sufficiency_check
    │   ├─→ graph_expansion → reranking → adaptive_reranking → generating
    │   │   └─→ chart_validation → collect_answer → END  (conditional)
    │   └─→ reranking → adaptive_reranking → generating → collect_answer → END
    └─→ fallback_response → collect_answer → END

Dead Code (defined but wired into no graph):
  compress_context_node — imported but never added as node or edge
  should_compress_context — routing function never called
```

### Reference Repo Graph Structure

```
Main Graph:
  START → summarize_history → rewrite_query
    ├─→ request_clarification → rewrite_query (interrupt_before, user responds, then loops)
    └─→ [Send()] → N parallel agent subgraphs → aggregate_answers → END

Agent Subgraph:
  START → orchestrator (LLM calls tools: search_child_chunks, retrieve_parent_chunks)
    ├─→ tools (LangGraph ToolNode) → should_compress_context
    │   ├─→ compress_context → orchestrator (loop for self-correction)
    │   └─→ orchestrator (loop)
    └─→ fallback_response → collect_answer → END
    └─→ collect_answer → END (when no tool calls)
```

---

## 6. Summary of Gaps by Severity

### Must Fix

| # | Gap | Description |
|---|-----|------------|
| 1 | **Answer evaluation missing** | Step 9 in simple path (after generation, before done) and Step 6 in complex path are completely absent from the code. |
| 2 | **No parallel execution** | Complex queries process subtasks sequentially instead of simultaneously (no LangGraph `Send()`). |
| 3 | **Synthesis broken** | `collect_answer_node` always returns a single-answer list; multi-task synthesis never works. |
| 4 | **No task list display** | Design spec shows the subtask list to the user before processing. |

### Should Fix

| # | Gap | Description |
|---|-----|------------|
| 5 | **Dead code** | `compress_context_node` and `should_compress_context` are defined but never wired into any graph. |
| 6 | **No self-correction loop** | If initial retrieval fails, there's no retry back to `direct_retrieval`; only routes forward to `graph_expansion`. |
| 7 | **graph_expansion has no new_chunks check** | Design says "if graph found new chunks", but implementation runs unconditionally when sufficiency fails. |

### Nice to Have

| # | Gap | Description |
|---|-----|------------|
| 8 | **Conversation summarization** | No rolling summary like reference repo. |
| 9 | **Evaluation integration** | RAGAS-style answer evaluation not integrated into pipeline. |
| 10 | **Hierarchical parent chunk fetching** | Pipeline doesn't fetch parent chunks for richer context. |

---

## 7. Source Files Referenced

| File | Purpose |
|------|---------|
| `backend/app/services/agentic_rag/graph.py` | Main graph and agent subgraph builders, routing functions |
| `backend/app/services/agentic_rag/nodes.py` | All node implementations (rewrite, classify, retrieval, generating, etc.) |
| `backend/app/services/agentic_rag/schemas.py` | Pydantic models for structured output |
| `backend/app/services/agentic_rag/graph_state.py` | AgentState definition |
| `backend/app/services/agentic_rag/utils.py` | Query rewriting, token estimation, context formatting |
| `backend/app/services/retrieval/__init__.py` | `hybrid_search_with_legs`, `score_retrieval`, `rerank` |
| `backend/app/services/agentic_rag/evaluator.py` | Answer evaluation (exists but not integrated) |

---

## 8. Reference Repo Files (for comparison)

| File (reference repo) | Purpose |
|-----------------------|---------|
| `rag_agent/graph.py` | Main graph + agent subgraph with `Send()` parallelism |
| `rag_agent/nodes.py` | All agent nodes including orchestrator, tools, compression |
| `rag_agent/edges.py` | Routing functions including parallel `Send()` after rewrite |
| `rag_agent/tools.py` | `ToolFactory` with `search_child_chunks` and `retrieve_parent_chunks` |
| `rag_agent/schemas.py` | `QueryAnalysis` Pydantic model |
| `rag_agent/graph_state.py` | State definitions |
| `config.py` | Configuration constants (MAX_ITERATIONS, RETRIEVAL_SCORE_THRESHOLD, etc.) |
