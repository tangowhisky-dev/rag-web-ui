# Autonomous Agentic RAG — Final Architecture Proposal

> **⚠️ OUTDATED — Superseded by live implementation.**
> This proposal describes the vision for the agentic agent. The implementation now lives at
> `backend/app/services/agentic_rag/agentic_rag.py`. The former Fast/Thinking pipeline
> (`fast_pipeline.py`) and Agentic LangGraph (`rag_graph/`) have been removed as dead code.
> The live code follows this proposal's general direction with a simpler architecture (simple/complex branching).
> See `architecture.md` for the current state.

---

## 1. Executive Summary

Transform rag-web-ui from a *pipeline selector* (user picks fast/thinking/agentic) into a *fully autonomous enterprise assistant* that evaluates the user query, decomposes it into subtasks, selects the optimal retrieval strategy for each subtask, and iterates until confident the answer is complete.

**Key insight:** The existing agentic pipeline (LangGraph with decomposition → retrieval → grading → retry) is already 70% of what's needed. The missing piece is a *meta-controller* — an LLM supervisor that decides *which* pipeline (fast/thinking/agentic), *which* retrieval leg (dense/sparse/exact/graph), and *which* capability (chart generation, tool calling, synthesis) to deploy for each subtask, instead of following a fixed linear graph.

## 2. Current State Analysis

### 2.1 What already exists

| Capability | Module | Notes |
|---|---|---|
| **Fast pipeline** | `fast_pipeline.py` | Rewrite → hybrid search → stream. ~300ms on good data. |
| **Thinking pipeline** | `fast_pipeline.py` + `REASONING_MODEL` | Same flow, uses reasoning model (CoT). |
| **Agentic pipeline (v1)** | `rag_graph/` (LangGraph) | 11-node graph: rewrite → router → decompose → parallel_retrieval → extract_file_sections → draft → grade_coverage → [conditional: widen/keyword/generate]. 3-level retry loop. |
| **3-leg hybrid retrieval** | `retrieval.py` | Qdrant dense (OpenAI embeddings), Qdrant sparse (SPLADE), MySQL FULLTEXT. RRF fusion. |
| **Graph expansion** | `graph_service.py` | Neo4j 2-hop traversal: find entity-connected chunks not in vector results. |
| **Graph enrichment** | `graph_service.py` | Append entity/relationship triples to chunk text for LLM context. |
| **Cross-encoder reranker** | `reranker.py` | fastembed TextCrossEncoder. Configurable threshold. |
| **Confidence scoring** | `confidence.py` | 3-signal (top score, evidence count, mean score) → 0-100 → level. |
| **Historical memory** | `historical_memory.py` | Query past assistant messages from MySQL, rerank against current query. |
| **Entity-aware retrieval** | `entity_extractor.py` | LLM NER → Neo4j 1-hop neighbor expansion → score boost. |
| **Abbreviation expansion** | `query_expander.py` | Org-specific abbreviation dictionary. |
| **Tool calling framework** | `tool_registry.py` | OpenAI-compatible tool schema + execution loop. Currently exists but not integrated into agentic pipeline. |
| **Query classification** | `chat_service.py` | FACTUAL / MULTI_PART / AMBIGUOUS / ENTITY_CENTRIC. |
| **Adaptive retrieval** | `fast_pipeline.py` | Confidence-based threshold relaxation. |
| **Chart generation** | `prompts/loader.py` (append_chart_instructions) + `export_service.py` | ECharts JSON in answer → pyecharts rendering. |
| **File section extraction** | `rag_graph/nodes.py` | LLM selects relevant sections from large attached files. |
| **SSE event streaming** | `rag_graph/nodes.py::run_stream` | Agent timeline UI already renders node progress. |
| **Multi-tenant** | `enterprise-multitenancy` branch | Org-scoped KBs, data stores, per-org LLM config. |

### 2.2 What's missing for full autonomy

| Gap | Why it matters |
|---|---|
| **No meta-controller** | The current agentic graph is a fixed linear pipeline. The LLM never chooses *which* pipeline to use — it always goes through the same 11 nodes. A true autonomous agent needs to decide: "this is a simple lookup → use fast", "this needs reasoning → use thinking", "this needs multi-source synthesis → use agentic". |
| **Tool calling not integrated** | `tool_registry.py` exists but the agentic pipeline never uses it. The agent should be able to call tools (DB queries, API calls, file operations) as part of its reasoning loop. |
| **No self-evaluation loop** | The current grade_coverage node only checks if sub-queries are answered. It doesn't evaluate answer *quality* (faithfulness, completeness, coherence) or decide to regenerate. |
| **No dynamic pipeline selection** | User currently picks fast/thinking/agentic. The agent should pick automatically based on query complexity. |
| **No cross-pipeline orchestration** | A single user query might need: fast retrieval for factual lookup + agentic for synthesis + chart generation for visualization. No existing code handles mixing pipelines in one response. |
| **UI doesn't show meta-reasoning** | The agent timeline shows node progress but not the supervisor's decision-making (why it chose a pipeline, what subtasks it planned, why it retried). |

## 3. SOTA Research Summary

### 3.1 What the leading systems do

| System | Approach | Relevance |
|---|---|---|
| **ChatGPT Deep Research** | Meta-planner decomposes query → spawns parallel research agents (each does search → read → summarize) → synthesis agent combines results → quality check → iterate | Direct model for our "supervisor → workers" pattern. |
| **Claude (Opus/sonnet)** | ReAct loop: observe → think → act (tool call) → observe → refine. No fixed graph — LLM decides each step. | Shows that fixed graphs are inferior to dynamic reasoning. |
| **LangGraph (official)** | State machine with *conditional edges* where the LLM itself decides the next node. Supports *supervisor pattern* (one LLM routes to workers). | Our framework of choice — already in dependencies. |
| **DSPy** | Treats prompting as code. Optimizes prompts and retrieval strategies via compilation. | Useful for the meta-controller's prompt engineering. |
| **AutoGen (Microsoft)** | Multi-agent conversation: supervisor delegates to specialized agents, agents can talk to each other. | Relevant for multi-worker orchestration. |
| **CrewAI** | Role-based agents with task delegation. | Less relevant — too opinionated, we want a single supervisor. |
| **Microsoft Semantic Kernel** | Planner generates plan steps, executor runs them, critic evaluates. | Similar to our proposed supervisor → worker → critic pattern. |

### 3.2 Key architectural patterns from SOTA

1. **Supervisor/Worker (Router-Worker)** — A central LLM (supervisor) decomposes the query, assigns subtasks to specialized workers, collects results, and decides whether to iterate. This is the dominant pattern in ChatGPT Deep Research, LangGraph's official examples, and AutoGen.

2. **ReAct (Reason + Act)** — The LLM interleaves reasoning with tool execution. Each tool call is a "step" the LLM chooses to take. No fixed graph.

3. **Self-Reflection / Critic** — After generating an answer, a separate evaluation step checks quality (faithfulness, completeness, coherence) and decides whether to accept or retry.

4. **Dynamic Pipeline Selection** — Instead of fixed pipelines, the supervisor chooses the right tool/pipeline for each subtask based on query characteristics.

5. **Convergent Iteration** — The agent iterates until confidence exceeds a threshold. Each iteration can change strategy (widen search, try different retrieval leg, call different tool).

### 3.3 Framework comparison

| Framework | Pros | Cons | Verdict |
|---|---|---|---|
| **LangGraph** (our current) | Already installed, 11-node graph works, streaming support, state machine | Requires building the supervisor pattern on top | **USE — extend existing** |
| **LangChain Agents** | Built-in ReAct, tool calling | Heavier, less control over flow, older API | Skip |
| **Haystack** | Pipeline DSL, good for retrieval | Not designed for agentic loops, different paradigm | Skip |
| **Dify** | Visual workflow builder, ready-made | Opaque, hard to customize, external dependency | Skip |
| **AutoGen** | Multi-agent, conversation-based | Heavy, Microsoft-specific, overkill | Skip |
| **CrewAI** | Role-based agents | Opinionated, less flexible | Skip |
| **DSPy** | Programmatic optimization | Not an orchestration framework | Use for prompt tuning only |

**Decision: Extend the existing LangGraph implementation.** It's already in the stack, the 11-node graph provides all the building blocks, and we can add a supervisor layer on top without rewriting anything.

## 4. Proposed Architecture

### 4.1 High-level flow

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  SUPERVISOR NODE (LLM)                              │
│  - Classify query type & complexity                 │
│  - Decompose into subtasks (if needed)              │
│  - For each subtask, select:                        │
│      • Pipeline: fast / thinking / agentic          │
│      • Retrieval legs: dense, sparse, exact, graph  │
│      • Reranker threshold: tight / relaxed          │
│      • Tools to call: chart, DB query, etc.         │
│  - Set iteration budget & confidence threshold      │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  WORKER NODES (parallel or sequential)              │
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ FAST     │  │ THINKING │  │ AGENTIC (multi)  │  │
│  │ WORKER   │  │ WORKER   │  │ WORKER           │  │
│  │          │  │          │  │                  │  │
│  │ rewrite  │  │ rewrite  │  │ decompose        │  │
│  │ hybrid   │  │ hybrid   │  │ retrieve         │  │
│  │ search   │  │ search   │  │ grade + retry    │  │
│  │ stream   │  │ stream   │  │ stream           │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ CHART    │  │ TOOL     │  │ SYNTHESIS        │  │
│  │ WORKER   │  │ WORKER   │  │ WORKER           │  │
│  │          │  │          │  │                  │  │
│  │ extract  │  │ execute  │  │ gather all docs  │  │
│  │ data     │  │ tool     │  │ write report     │  │
│  │ render   │  │ (DB/API) │  │ with themes      │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  CRITIC / REFLECTION NODE (LLM)                     │
│  - Evaluate answer quality:                         │
│      • Faithfulness (grounded in retrieved docs?)   │
│      • Completeness (all subtasks answered?)        │
│      • Coherence (well-structured, no contradictions?)│
│      • Confidence (retrieval score sufficient?)     │
│  - Decision: ACCEPT → generate final answer         │
│            OR RETRY → feed feedback to supervisor   │
└─────────────────────────────────────────────────────┘
    │
    ├─ ACCEPT ──► FINAL ANSWER (streamed to UI)
    │
    └─ RETRY ──► Back to SUPERVISOR (with feedback)
         (max N iterations, default 3)
```

### 4.2 State machine design

```
                    ┌──────────────────────────────────┐
                    │          INITIAL STATE           │
                    │  query, chat_id, org_id, kb_ids  │
                    └──────────────┬───────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────────┐
                    │  SUPERVISOR (LLM decision node)  │
                    │  - Classify query                │
                    │  - Decompose into tasks          │
                    │  - Select pipeline per task      │
                    │  - Output: task_plan             │
                    └──────────────┬───────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────────┐
                    │  TASK EXECUTOR (parallel/seq)    │
                    │  - Execute each task             │
                    │  - Collect results               │
                    │  - Output: task_results          │
                    └──────────────┬───────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────────┐
                    │  CRITIC (LLM evaluation node)    │
                    │  - Evaluate quality              │
                    │  - Output: verdict + feedback    │
                    └──────────────┬───────────────────┘
                              ┌────┴────┐
                              │         │
                         ACCEPT      RETRY
                              │         │
                              ▼         ▼
                    ┌──────────────┐  ┌──────────────────┐
                    │  FINAL       │  │  SUPERVISOR      │
                    │  ANSWER      │  │  (with feedback) │
                    │  (stream)    │  │  → new plan      │
                    └──────────────┘  └──────────────────┘
```

### 4.3 New modules to create

```
backend/app/services/agentic_agent/
├── __init__.py              # Public API: run_autonomous_agent()
├── state.py                 # AgenticAgentState TypedDict
├── supervisor.py            # LLM supervisor: classify, decompose, plan
├── executor.py              # Task executor: dispatch to fast/thinking/agentic workers
├── critic.py                # Quality evaluation: faithfulness, completeness, coherence
├── workers/
│   ├── __init__.py
│   ├── fast_worker.py       # Wrapper around fast_pipeline.fast_stream()
│   ├── thinking_worker.py   # Wrapper around fast_pipeline (reasoning model)
│   ├── agentic_worker.py    # Wrapper around rag_graph.run_stream()
│   └── chart_worker.py      # Data extraction + ECharts generation
├── tools/
│   ├── __init__.py
│   ├── db_query_tool.py     # MySQL exact search as a tool
│   └── graph_query_tool.py  # Neo4j traversal as a tool
└── graph_builder.py         # LangGraph assembly
```

### 4.4 State schema

```python
class AgenticAgentState(TypedDict):
    # ── Input ──────────────────────────────────────────────────
    query: str
    chat_id: int
    knowledge_base_ids: List[int]
    recent_lc_history: list
    existing_summary: Optional[str]
    file_markdown: Optional[str]
    org_id: Optional[int]
    _db: Any

    # ── Supervisor output ──────────────────────────────────────
    query_classification: str          # FACTUAL / MULTI_PART / AMBIGUOUS / ENTITY_CENTRIC
    task_plan: List[dict]              # [{subtask, pipeline, legs, threshold, tools}]
    iteration: int                     # Current iteration count

    # ── Worker results ─────────────────────────────────────────
    task_results: List[dict]           # [{subtask, pipeline_used, docs, answer, tool_output}]
    all_docs: List[dict]               # Accumulated retrieved docs
    chart_config: Optional[dict]       # ECharts config if chart generated

    # ── Critic output ──────────────────────────────────────────
    quality_score: int                 # 0-100
    quality_feedback: str              # Why not accepted, what to improve
    verdict: str                       # ACCEPT or RETRY

    # ── Final output ───────────────────────────────────────────
    answer: str
    confidence: str                    # confidence level
    agent_steps: List[dict]            # For UI timeline
    _usage: dict
```

### 4.5 Supervisor prompt design

The supervisor is the critical piece. Its system prompt determines the agent's behavior:

```
You are an autonomous enterprise research assistant. Given a user query,
your job is to plan and execute the optimal strategy to answer it.

## Your workflow:
1. CLASSIFY the query type (FACTUAL, MULTI_PART, AMBIGUOUS, ENTITY_CENTRIC)
2. DECOMPOSE into subtasks (1-5) — only if the query has multiple distinct parts
3. For each subtask, SELECT the best approach:

   Pipeline choices:
   - "fast"     → Simple factual lookup, single document retrieval
   - "thinking" → Requires reasoning, comparison, or analysis
   - "agentic"  → Needs multi-source synthesis, iterative retrieval, or chart generation

   Retrieval legs (per pipeline):
   - "dense"    → Semantic similarity (vector search)
   - "sparse"   → Learned sparse terms (SPLADE)
   - "exact"    → Exact keyword match (MySQL FULLTEXT)
   - "graph"    → Neo4j entity traversal

   Reranker threshold:
   - "tight" (default 2.0) → Precise, high-precision retrieval
   - "relaxed" (-5.0) → Broad, high-recall retrieval (use when tight fails)

   Tools:
   - "chart" → Generate ECharts visualization
   - "db_query" → Direct MySQL query
   - "graph_query" → Neo4j graph traversal

4. OUTPUT a structured task_plan

## Rules:
- Start with the simplest approach. Escalate complexity only if needed.
- For simple factual questions: 1 fast task, tight threshold.
- For comparison/analysis: 1 thinking task with dense+sparse legs.
- For multi-part questions: decompose → fast/thinking per part.
- For "show me data/visualize": include chart task.
- Maximum 5 subtasks, maximum 3 iterations.
- If a subtask has no answer after 1 attempt, retry with relaxed threshold.
```

### 4.6 Critic prompt design

```
You are an answer quality critic. Evaluate the generated answer against
the original query and retrieved context.

## Evaluate on 4 dimensions (0-25 each, total 100):
1. FAITHFULNESS: Is every claim grounded in the retrieved context?
2. COMPLETENESS: Are all subtasks from the plan addressed?
3. COHERENCE: Is the answer well-structured with no contradictions?
4. CONFIDENCE: Does the retrieval evidence support the answer?

## Output:
{
  "quality_score": <0-100>,
  "faithfulness": <0-25>,
  "completeness": <0-25>,
  "coherence": <0-25>,
  "confidence": <0-25>,
  "verdict": "ACCEPT" or "RETRY",
  "feedback": "<specific instructions for improvement>"
}

## Acceptance threshold:
- ACCEPT if quality_score >= 70 AND completeness >= 20
- RETRY otherwise, with specific feedback on what to improve
```

## 5. UI/UX Design

### 5.1 Principles (ChatGPT / Claude level)

1. **Progressive disclosure** — Show the agent's thinking as it happens, but keep it collapsible. The default view is the answer. The "how I got here" is one click away.

2. **Real-time transparency** — Each supervisor decision, worker execution, and critic evaluation appears as a live step in the timeline. Not as a wall of text — as structured, expandable cards.

3. **No pipeline selector** — Remove the fast/thinking/agentic toggle from the UI. The agent decides. (Keep it as an advanced setting behind a "force mode" dropdown for power users.)

4. **Confidence visualization** — Show retrieval confidence as a subtle indicator (dot color + percentage) next to each answer, similar to ChatGPT's "sources" indicator.

5. **Iteration feedback** — When the agent retries, show "Expanding search..." or "Gathering more context..." — not "Retrying..." which sounds like an error.

### 5.2 Timeline redesign

Current timeline shows LangGraph nodes. New timeline shows *agent decisions*:

```
┌─────────────────────────────────────────────────────────┐
│ 🔍 Query: "Compare Q3 revenue vs Q2 for both divisions" │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ ▶ [1] Analyzing query...              45ms             │
│   → Classified as: MULTI_PART                            │
│   → Plan: 2 subtasks (Q3 revenue, Q2 revenue)           │
│                                                         │
│ ▶ [2] Retrieving Q3 revenue data...      1.2s          │
│   → Pipeline: thinking (requires comparison)            │
│   → Legs: dense + exact + graph                         │
│   → 8 docs retrieved, confidence: high                   │
│                                                         │
│ ▶ [3] Retrieving Q2 revenue data...      0.9s          │
│   → Pipeline: fast (simple lookup)                      │
│   → Legs: dense + exact                                 │
│   → 5 docs retrieved, confidence: high                   │
│                                                         │
│ ▶ [4] Evaluating answer quality...     0.6s            │
│   → Faithfulness: 24/25  Completeness: 22/25           │
│   → Coherence: 23/25     Confidence: 20/25             │
│   → Score: 89/100 → ACCEPT                             │
│                                                         │
│ ▶ [5] Generating answer...           2.3s              │
│                                                         │
│ ─────────────────────────────────────────────────────── │
│ [Answer rendered here with citations]                   │
└─────────────────────────────────────────────────────────┘
```

Each step is a card with:
- **Icon** indicating the type (🔍 analyze, 📊 retrieve, ⚖️ evaluate, ✍️ generate)
- **Status** (running spinner → checkmark)
- **Expandable detail** (the LLM's reasoning, docs retrieved, scores)
- **Latency** in the corner

### 5.3 New UI components needed

| Component | Purpose | Reuses existing? |
|---|---|---|
| `AgentSupervisorCard` | Shows supervisor's classification + task plan | Extends `AgentTimeline` |
| `WorkerExecutionCard` | Shows each worker's execution (pipeline, legs, results) | Extends `AgentTimeline` |
| `CriticEvaluationCard` | Shows quality scores and verdict | New |
| `ConfidenceIndicator` | Small dot + percentage next to answer | New (simple) |
| `IterationBadge` | Shows "Round 2 of 3" during retries | New (simple) |
| `ForceModeDropdown` | Advanced: override pipeline selection | New (small) |

### 5.4 Backend event protocol (SSE)

Extend the existing event protocol to include supervisor/worker/critic events:

```python
# Existing events (unchanged):
{"event": "agent_step", "node": "...", "status": "active|done", ...}
{"event": "token", "content": "..."}
{"event": "done", "full_response": "..."}

# New events:
{"event": "supervisor_plan", "classification": "...", "task_plan": [...]}
{"event": "worker_start", "worker": "fast|thinking|agentic|chart", "subtask": "..."}
{"event": "worker_done", "worker": "...", "docs_found": N, "pipeline_used": "..."}
{"event": "critic_eval", "quality_score": 89, "verdict": "ACCEPT|RETRY", "feedback": "..."}
{"event": "iteration", "current": 2, "max": 3, "reason": "low completeness"}
```

The frontend `chat-context.tsx` already handles arbitrary event types via the existing `agentSteps` mechanism. New events map to new `agent_step` node types.

## 6. Implementation Plan

### Phase 1: Core Agent (Week 1-2)

1. **Create `agentic_agent/` package** with state schema and LangGraph builder
2. **Implement supervisor node** — LLM-based classification + task planning
3. **Implement worker wrappers** — thin wrappers around existing `fast_stream()`, `rag_graph.run_stream()`, and a new thinking worker
4. **Implement critic node** — LLM-based quality evaluation
5. **Wire up the LangGraph** — supervisor → executor → critic → (retry or accept)
6. **Add new SSE events** — supervisor_plan, worker_start/done, critic_eval, iteration

### Phase 2: Tool Integration (Week 2-3)

7. **Integrate tool registry** — connect `tool_registry.py` to the agentic loop
8. **Implement DB query tool** — MySQL exact search as an LLM-callable tool
9. **Implement graph query tool** — Neo4j traversal as an LLM-callable tool
10. **Implement chart worker** — data extraction + ECharts generation

### Phase 3: UI/UX (Week 3-4)

11. **Redesign agent timeline** — supervisor/worker/critic cards
12. **Add confidence indicator** — next to answers
13. **Add force mode dropdown** — advanced override
14. **Remove pipeline selector** — from main chat UI (move to settings)
15. **Update chat settings page** — new options for agent behavior

### Phase 4: Hardening (Week 4-5)

16. **Add iteration budget enforcement** — max 3 iterations, timeout protection
17. **Add error recovery** — fallback to fast pipeline if supervisor fails
18. **Add streaming for supervisor decisions** — show plan as it's being generated
19. **Add caching** — cache supervisor decisions for repeated queries
20. **Write tests** — unit tests for supervisor, critic, workers

## 7. Key Technical Decisions

### 7.1 Why extend LangGraph, not replace it

- Already installed and working (11-node graph)
- `run_stream()` already handles SSE streaming, token emission, state management
- The existing graph's nodes (rewrite, decompose, grade_coverage, widened_retrieval, keyword_search_loop) are all useful *as workers* within the new architecture
- Minimal migration cost: wrap existing pipelines as workers, add supervisor/critic on top

### 7.2 Why not use LangChain's built-in agents

- LangChain's `create_react_agent` and similar are single-purpose (ReAct with tools). They don't handle the multi-pipeline orchestration we need.
- LangGraph gives us explicit control over the state machine, which is essential for the supervisor → worker → critic pattern.

### 7.3 Why a single supervisor, not multiple specialized agents

- A single supervisor with a structured prompt is cheaper (1 LLM call per iteration vs N)
- Easier to maintain and debug
- The supervisor can dynamically adjust its strategy based on what it learns
- Multi-agent conversation overhead (message passing, coordination) isn't needed for this use case

### 7.4 Pipeline selection strategy

The supervisor picks pipelines based on query classification:

| Classification | Default Pipeline | Escalation |
|---|---|---|
| FACTUAL (simple) | fast | thinking (if confidence < 50) |
| FACTUAL (complex) | thinking | agentic (if confidence < 50) |
| MULTI_PART | agentic (decomposed) | parallel fast/thinking per subtask |
| AMBIGUOUS | fast (clarify first) | agentic (if clarification needed) |
| ENTITY_CENTRIC | agentic (with graph) | thinking (if analysis needed) |

### 7.5 Handling the existing "answering_mode" parameter

The `answering_mode` parameter in `chat.py` (fast/thinking/agentic) becomes a *hint* rather than a directive:

- If user selects "fast" → supervisor is constrained to use fast pipeline for all tasks
- If user selects "thinking" → supervisor is constrained to thinking/agentic
- If user selects "agentic" → supervisor has full freedom
- If no mode selected (default) → supervisor decides freely

This preserves backward compatibility while enabling full autonomy by default.

## 8. Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| Supervisor adds latency (extra LLM call) | High | Supervisor runs in parallel with initial retrieval; results cached |
| Supervisor makes bad pipeline choices | Medium | Critic catches bad choices; fallback to default pipeline |
| Infinite retry loops | Low | Hard cap at 3 iterations + timeout |
| Context window overflow (accumulated docs) | Medium | Supervisor caps total docs; critic can request pruning |
| LLM cost increase (3x-5x more LLM calls) | High | Supervisor is fast model; only critic uses strong model; caching |
| Breaking existing fast/thinking pipelines | Low | New agent is a separate entry point; existing pipelines unchanged |

## 9. Cost Implications

| Operation | Current (agentic mode) | Proposed (autonomous) | Delta |
|---|---|---|---|
| Query classification | 1 call (chat_service) | 1 call (supervisor) | Same |
| Query rewrite | 1 call | 0 calls (supervisor does it) | -1 |
| Retrieval per subtask | 1-3 calls (retry loop) | 1-3 calls (same) | Same |
| Answer generation | 1 call | 1 call | Same |
| Coverage grading | 1 call | 1 call (critic replaces) | Same |
| **Total per query** | ~5-7 calls | ~5-8 calls | +1 (supervisor) |

Net cost increase: ~15-20% per query. Acceptable for the quality improvement.

## 10. Migration Path

1. **Branch**: `feature/agentic-agent`
2. **New code**: All new code in `backend/app/services/agentic_agent/`
3. **Existing code**: Unchanged — `fast_pipeline.py`, `rag_graph/`, `tool_registry.py` all stay as-is
4. **API**: New endpoint `POST /api/chat/autonomous` (or reuse existing with `answering_mode="autonomous"`)
5. **Frontend**: New event types are additive — existing timeline renders unknown nodes gracefully
6. **Config**: New env vars:
   - `AGENT_MAX_ITERATIONS` (default 3)
   - `AGENT_QUALITY_THRESHOLD` (default 70)
   - `AGENT_SUPERVISOR_MODEL` (optional, falls back to OPENAI_MODEL)
   - `AGENT_FORCE_MODE` (optional, overrides supervisor)

## 11. Future Extensions (Out of Scope for V1)

- **Memory persistence**: Agent remembers patterns across sessions (e.g., "user always wants charts for revenue queries")
- **Self-improvement**: Critic's feedback is logged and used to fine-tune the supervisor prompt
- **Multi-user collaboration**: Supervisor delegates to domain-specific agents (finance agent, technical agent, etc.)
- **Human-in-the-loop**: Supervisor can ask the user clarifying questions before proceeding
- **Plugin system**: Users can register custom tools that the supervisor can call
