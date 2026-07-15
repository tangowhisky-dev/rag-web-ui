# Agentic Agent Redesign — Implementation Plan

> **⚠️ SUPERCEDED — This document is kept for historical context only.**
> The consolidated autonomous enterprise assistant plan is now in [`kimi_agentic_recoms.md`](kimi_agentic_recoms.md).
> This design document describes the planned agentic agent. The implementation now lives at
> `backend/app/services/agentic_rag/agentic_rag.py`. The former Fast/Thinking pipeline
> (`fast_pipeline.py`) and Agentic LangGraph (`rag_graph/`) have been removed. The live
> code follows this plan's architecture with minor variations. See `architecture.md` for the current state.

---

## Problem Statement

The current agentic agent has two critical UX bugs and an over-engineered architecture:

1. **Rewritten query leaks into response text** — The first line of the user-visible output is the rewritten query, not the answer.
2. **No streaming during execution** — The entire response is held back until the critic accepts, so users stare at a blank screen for 15-20s then get everything at once.
3. **Supervisor/worker/critic loop is over-engineered** — The supervisor classifies and delegates to workers (fast/thinking/agentic), but all workers are thin wrappers around `fast_stream()`. The critic then re-evaluates and may reject, causing the user to see nothing until retry. This adds latency with no visible benefit.

## Design Goals

A single autonomous agent that:
- Is transparent: users see progress in real-time
- Is efficient: no redundant LLM calls for classification
- Is adaptive: auto-selects model type and retrieval depth based on query complexity
- Streams everything: tokens, progress, thinking traces
- Falls back gracefully: if a subtask fails, continue with others

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────┐
│  1. QUERY UNDERSTANDING                        │
│  - Rewrite query using chat history            │
│  - Decide: simple (direct answer) or complex   │
│    (needs subtask breakdown)                    │
│  - Emit: progress events                       │
│  - Emit: rewritten query (NOT in response)     │
└────────────────┬────────────────────────────────┘
                 │
      ┌──────────┴──────────┐
      │                     │
  SIMPLE              COMPLEX
      │                     │
      ▼                     ▼
┌─────────────┐   ┌──────────────────────────────┐
│  Direct     │   │  2. SUBTASK DECOMPOSITION    │
│  Answer     │   │  - Split query into N tasks  │
│  Pipeline:  │   │  - Display hanging task list │
│  rewrite    │   │  - Update as each completes  │
│  → search   │   │                                │
│  → rerank   │   │  3. FOR EACH SUBTASK:        │
│  → stream   │   │  - Rewrite subtask query     │
│  → done     │   │  - Search (keyword+dense+    │
│             │   │    sparse+exact+neo4j)        │
│             │   │  - Progress: "found N chunks"│
│             │   │  - Rerank → filter bloat     │
│             │   │  - Progress: "reranked"      │
│             │   │  - Select model (fast/think) │
│             │   │  - Stream answer to user     │
│             │   │  - If thinking model:        │
│             │   │    stream thinking traces    │
│             │   │  - Update task list ✓        │
│             │   │                                │
│             │   │  4. FINAL SYNTHESIS          │
│             │   │  - Combine all subtask answers│
│             │   │  - Grade & confidence score  │
│             │   │  - Stream final summary      │
│             │   │                                │
│             │   │  5. REVIEW (lightweight)     │
│             │   │  - Self-check: is answer     │
│             │   │    complete & faithful?      │
│             │   │  - If not: retry 1 subtask   │
│             │   │  - If yes: done              │
└─────────────┘   └──────────────────────────────┘
```

## Event Protocol

All events stream to the frontend via SSE. Each event type maps to a UI component.

### Event Types

| Type | SSE Prefix | Purpose | UI |
|------|-----------|---------|----|
| `progress` | `p:` | Transient status messages | Inline progress indicator |
| `task_list` | `t:` | Subtask list with status | Hanging task list sidebar |
| `thinking` | `th:` | Thinking model reasoning | Collapsible thinking trace |
| `token` | `0:` | Answer text tokens | Streaming answer |
| `context` | `2:` | Retrieved documents | Citation panel |
| `done` | `d:` | Completion | Finalize message |

### Progress Event Schema

```json
{
  "event": "progress",
  "phase": "rewriting|searching|reranking|generating|synthesizing|reviewing",
  "message": "Searching knowledge base...",
  "details": {
    "subtask_index": 0,
    "subtask_total": 3,
    "chunks_found": 20,
    "reranked": 5,
    "model": "qwen3.5-9b-nothink",
    "model_type": "fast|thinking"
  }
}
```

### Task List Event Schema

```json
{
  "event": "task_list",
  "tasks": [
    {
      "id": 0,
      "text": "Explain the concept of PCB",
      "status": "pending|running|done|error",
      "progress": null
    },
    {
      "id": 1,
      "text": "Describe PCB structure",
      "status": "pending",
      "progress": null
    },
    {
      "id": 2,
      "text": "Explain PCB role in scheduling",
      "status": "pending",
      "progress": null
    }
  ]
}
```

### Thinking Event Schema

```json
{
  "event": "thinking",
  "content": "Chain of thought reasoning here...",
  "done": false
}
```

## Implementation Steps

### Step 1: Create `agentic_agent.py` — the new single agent

**File**: `backend/app/services/agentic_rag/agentic_agent.py`

This replaces the entire `supervisor → worker → critic` loop.

#### Core structure:

```python
async def run_agentic_agent(...) -> AsyncGenerator[dict, None]:
    """
    Single autonomous agent. No supervisor, no workers, no critic loop.
    Streams everything in real-time.
    """
    # 1. Rewrite query
    rewritten = await rewrite_query(query, history)
    yield progress("rewriting", "Rewriting query...")
    yield rewritten_query(rewritten)

    # 2. Decide: simple or complex?
    complexity = await classify_complexity(query, rewritten, history)
    # Returns: {"complex": bool, "subtasks": List[str] | None}

    if not complexity["complex"]:
        # Simple path: direct answer
        yield from _direct_answer(rewritten, ...)
    else:
        # Complex path: subtask decomposition + iterative
        yield from _complex_answer(rewritten, complexity["subtasks"], ...)

    # 3. Final review (lightweight, non-blocking)
    #    Only re-generate if the answer is clearly wrong.
    #    Don't hold back streaming — this is a safety net.
```

#### Key methods:

- `_rewrite_query()` — reuse existing `_rewrite_query()` from `chat_service.py`
- `_classify_complexity()` — lightweight LLM call to decide if query needs breakdown. Not a full supervisor. Single yes/no + optional subtasks.
- `_direct_answer()` — rewrite → search → rerank → stream answer (like current fast_pipeline but with progress events)
- `_complex_answer()` — decompose → loop over subtasks → synthesize → review
- `_search_and_rerank()` — reuse existing `hybrid_search_with_legs()` + reranker, emit progress events
- `_generate_answer()` — stream tokens to user, with thinking trace support

#### Model selection logic:

```python
def _select_model(subtask_text: str, is_complex: bool) -> str:
    """Auto-select model based on query nature."""
    thinking_keywords = [
        "compare", "contrast", "analyze", "evaluate", "design",
        "reason", "deduce", "infer", "explain why", "explain how",
        "tradeoff", "pros and cons", "architect", "implement"
    ]
    is_thinking = any(kw in subtask_text.lower() for kw in thinking_keywords)
    
    if is_thinking or is_complex:
        return settings.REASONING_MODEL or settings.OPENAI_MODEL
    return settings.OPENAI_MODEL
```

### Step 2: Fix rewritten query leak

**Root cause**: The `rewritten_query` event (prefix `1:`) is being parsed by the frontend and stored on the message object. But when the answer is generated, the LLM receives the rewritten query in its context and may echo it.

**Fix**: 
1. The rewritten query event is internal — it should NOT be stored on the message or shown in the answer context.
2. In `agentic_agent.py`, yield the rewritten query as a `progress` event, not a separate SSE event that gets stored.
3. The LLM context should only contain `[Rewritten Query]` as an internal marker, not as part of the answer.

### Step 3: Enable real-time streaming

**Root cause**: The current `run_autonomous_agent` collects all worker results, runs critic, and only then streams the final answer. Everything is buffered.

**Fix**:
- Stream tokens as they're generated, not after critic acceptance.
- The critic becomes a lightweight post-check, not a gate.
- If the critic rejects, the agent retries and appends corrections to the already-streamed answer.

### Step 4: Add progress events

Every pipeline stage emits progress events:

```
p: {"phase":"rewriting","message":"Rewriting query..."}
p: {"phase":"decomposition","message":"Breaking down into 3 subtasks"}
p: {"phase":"searching","message":"Searching knowledge base...","details":{"subtask_index":0,"subtask_total":3}}
p: {"phase":"searching","message":"Found 20 relevant chunks","details":{"chunks_found":20}}
p: {"phase":"reranking","message":"Reranking for relevance..."}
p: {"phase":"reranking","message":"Shortlisted 5 chunks","details":{"reranked":5}}
p: {"phase":"generating","message":"Generating answer...","details":{"model":"qwen3.5-9b-nothink","model_type":"fast"}}
th: {"content":"Let me think about this step by step...","done":false}  # if thinking model
0: {"content":"The Process Control Block..."}  # streaming tokens
p: {"phase":"generating","message":"Answer generated","details":{"tokens":586}}
t: {"tasks":[...]}  # update task list
```

### Step 5: Update frontend

**File**: `frontend/src/app/dashboard/chat/[id]/page.tsx`

Add handlers for new event types:
- `p:` (progress) — show transient inline progress indicator
- `t:` (task_list) — render hanging subtask list with status icons
- `th:` (thinking) — render collapsible thinking trace block

The existing `0:` (token), `2:` (context), `d:` (done) handlers remain unchanged.

### Step 6: Update `chat_service.py`

- Replace `run_autonomous_agent` import with `run_agentic_agent`
- Remove handling of old event types (`supervisor_plan`, `worker_start`, `worker_done`, `critic_eval`, `iteration`)
- Add handling for new event types (`progress`, `task_list`, `thinking`)
- Remove `rewritten_query` event forwarding (it's now a progress event internally)

### Step 7: Cleanup

- Delete `backend/app/services/agentic_rag/supervisor.py`
- Delete `backend/app/services/agentic_rag/workers/` directory
- Delete `backend/app/services/agentic_rag/graph_builder.py` (deprecated)
- Delete `backend/app/services/agentic_rag/state.py` (if no longer needed)
- Update `backend/app/services/agentic_rag/__init__.py` to export only `run_agentic_agent`

## Files Modified

| File | Action |
|------|--------|
| `backend/app/services/agentic_rag/agentic_agent.py` | **NEW** — the single autonomous agent |
| `backend/app/services/chat_service.py` | Modified — replace agent import, update event handling |
| `frontend/src/app/dashboard/chat/[id]/page.tsx` | Modified — add progress/task_list/thinking handlers |
| `backend/app/services/agentic_rag/__init__.py` | Modified — replace exports |
| `backend/app/services/agentic_rag/supervisor.py` | **DELETE** |
| `backend/app/services/agentic_rag/workers/` | **DELETE** |
| `backend/app/services/agentic_rag/graph_builder.py` | **DELETE** (already deprecated) |
| `backend/app/services/agentic_rag/state.py` | **DELETE** (if unused) |

## Key Design Decisions

1. **No more supervisor classification** — We skip the LLM call that classifies query type. Instead, a lightweight complexity check happens during rewrite. This saves ~6s per query.

2. **Critic becomes lightweight post-check** — Not a gate that blocks streaming. If it finds issues, the agent appends corrections.

3. **Model selection is heuristic, not LLM-decided** — Keyword matching on subtask text. No extra LLM call needed.

4. **Progress events are transient** — They appear briefly then fade. They're not stored in the message history.

5. **Subtask list updates in real-time** — Each task shows: pending (○), running (◐), done (●), error (✗).

6. **Thinking traces are collapsible** — Only shown when using a reasoning model. User can expand/collapse.

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Complexity classifier misclassifies | Fallback to complex path (safer, just slightly slower) |
| Subtask decomposition produces bad splits | Limit to 5 subtasks, use existing prompt patterns |
| Progress events overwhelm UI | Show only one progress indicator at a time, auto-dismiss after 3s |
| Thinking traces too verbose | Cap at 2000 chars, collapse by default |
| Breaking existing chat functionality | Keep old event types as no-ops for backward compatibility |
