# Agentic Architecture v2 — Tool-Use Agent Wrapping Existing Pipeline

> **Status**: Design document. Implementation pending.
> **Principle**: Wrap, don't rewrite. The retrieval, reranking, memory, evaluation, and streaming infrastructure is solid. The agent loop is the only thing that changes.
> **Constraints**: Offline (no web search). Local model gateway. Attached files supported.

---

## 1. What Stays (Reused As-Is)

These components are production-ready and should be reused without modification:

| Component | Location | Purpose |
|-----------|----------|---------|
| **3-leg retrieval** | `retrieval/retrieval.py` | Dense + sparse + exact with RRF merge |
| **Reranking** | `retrieval/reranker.py` | fastembed ONNX cross-encoder |
| **Confidence scoring** | `retrieval/confidence.py` | 3-signal scoring (top score 60%, evidence 10%, mean 30%) |
| **Neo4j expansion** | `graph/expand.py`, `graph/enrich.py` | Graph-based document expansion |
| **Redis memory** | `agentic_rag/redis_memory.py` | Checkpointer + semantic long-term store |
| **Streaming transformer** | `agentic_rag/streaming.py` | v3 protocol → SSE events |
| **Query rewriting** | `agentic_rag/utils.py` `rewrite_query()` | Pronoun/reference resolution |
| **Answer evaluation** | `agentic_rag/evaluator.py` | Faithfulness, completeness, citation quality |
| **Retry helpers** | `agentic_rag/retry.py` | Tenacity-based retry with backoff |
| **File handling** | `api/chat_files.py` | Upload, markdown conversion, token budgeting |
| **Context formatting** | `agentic_rag/utils.py` `format_context_string()` | Doc → context string for LLM |
| **SSE event protocol** | `agentic_rag/streaming.py` | p:/t:/th:/0:/2:/d: event types |
| **Clarification** | `agentic_rag/nodes.py` `request_clarification_node()` | Human-in-the-loop when query is unclear |

**Key insight**: The existing pipeline nodes are the *tools* the agent will call. We're not replacing them — we're replacing the rigid graph topology that forces them into a fixed sequence.

---

## 2. What Changes (Minimal Delta)

### 2.1 Graph Topology

**Current** (rigid sequence):
```
START → rewrite → classify → [subtasks → agent_subgraph] → generate → evaluate → finalize
```

**New** (agent decides each step):
```
START → rewrite → [agent_loop (max 5 iterations)] → generate → evaluate → [if gaps → agent_loop] → finalize
```

The `agent_loop` wraps the existing retrieval pipeline as a tool call. Instead of nodes executing in a predetermined order, the agent decides which tool to call next based on what it knows.

### 2.2 Tool Definitions

The agent has access to these tools (wrapping existing infrastructure):

#### `rag_retrieve` — Existing retrieval pipeline
Wraps the same leg functions + merge + rerank + confidence scoring that the current agentic nodes already use:
`dense_search_docs()`, `sparse_search_docs()`, `exact_search_docs()`, `_merge_docs()`, `rerank()`, `score_retrieval()`.

**Parameters** (agent controls these):
```python
{
    "query": str,                        # Search query
    "score_threshold": float,           # -2.0 (strict) or -5.0 (loose)
    "legs": ["dense", "sparse", "exact"],  # Which legs to use
    "top_k": int,                       # Default 10
    "kb_ids": List[int],
    "file_markdown": Optional[str],     # Attached file content
}
```

**Returns**:
```python
{
    "docs": [...],                      # Retrieved + reranked docs
    "confidence": float,               # 0-1 score from score_retrieval()
    "leg_results": dict,               # Per-leg stats
    "suggestion": Optional[str],       # From confidence scoring
}
```

**Reuse**: Calls `dense_search_docs()`, `sparse_search_docs()`, `exact_search_docs()`, `_merge_docs()`, `rerank()`, and `score_retrieval()` — all existing functions. (Note: `hybrid_search_with_legs()` is NOT used by the agentic pipeline; it's only used by `builtin_tools.py` and tests.)

#### `code_execute` — New sandboxed Python
For computations, data analysis, transformations that can't be done with text retrieval alone.

**Parameters**:
```python
{
    "code": str,                        # Python code
    "description": str,                # What it does (logging)
}
```

**Returns**:
```python
{
    "stdout": str,                      # Printed output
    "result": Any,                      # Last expression value
    "error": Optional[str],            # Exception if failed
}
```

**Implementation**: Use `exec()` with restricted globals, or `restrictedpython` for proper isolation. Timeout: 5 seconds. No filesystem or network access.

**Use cases**: Math, statistics, date arithmetic, unit conversions, data parsing, generating ECharts JSON, text processing.

#### `self_reflect` — New internal reasoning
Agent pauses to think about what it knows, what it's missing, and what to do next.

**Parameters**:
```python
{
    "reflection_prompt": str,          # Custom prompt
}
```

**Returns**:
```python
{
    "analysis": str,                   # What the agent figured out
    "confidence": float,              # Updated confidence
    "missing_info": List[str],        # Information gaps
    "next_action": str,               # Suggested next tool
}
```

**Reuse**: Uses the same LLM call pattern as `classify_query_node()` — just a different prompt.

#### `generate_answer` — Existing generating_node
Wraps the existing `generating_node` logic (streaming answer generation).

**Parameters**:
```python
{
    "context": str,                    # Retrieved docs context
    "file_markdown": Optional[str],   # Attached files
    "memory_context": str,            # Historical memory
    "is_chart_query": bool,
}
```

**Returns**:
```python
{
    "answer": str,
    "thinking": str,                   # Chain-of-thought
    "is_chart_query": bool,
    "chart_data": Optional[dict],
}
```

**Reuse**: Uses `_build_generation_messages()`, `_get_llm()`, and the streaming token generation from `generating_node`.

---

## 3. Agent Loop (The Core Change)

```python
async def agent_loop(state: AgentState, config: dict) -> dict:
    """Think → Act → Observe cycle. Wraps existing pipeline as tools."""
    max_iterations = 5
    budget = {
        "rag_retrieve": {"used": 0, "max": 3},
        "code_execute": {"used": 0, "max": 2},
        "self_reflect": {"used": 0, "max": 2},
    }
    
    for iteration in range(max_iterations):
        state.iteration = iteration
        
        # 1. PLAN: Agent decides next action
        plan = await _plan_next_action(state, budget)
        state.plan = plan
        
        # 2. EXECUTE: Call the chosen tool
        tool_name = plan["action"]
        tool_result = await _execute_tool(tool_name, plan["params"], state)
        
        # 3. OBSERVE: Record result
        state.tool_calls.append({
            "tool": tool_name,
            "params": plan["params"],
            "result": tool_result,
            "iteration": iteration,
        })
        budget[tool_name]["used"] += 1
        
        # 4. CHECK: Should we stop?
        if await _should_stop(state, tool_result):
            break
    
    return state
```

### Planning Prompt (reuses existing LLM patterns)

```
You are an autonomous research agent. You have access to these tools:
- rag_retrieve: Search documents (uses existing 3-leg retrieval + reranking)
- code_execute: Run Python for computations
- self_reflect: Think about what you know and need
- generate_answer: Produce the final answer

Current state:
- Query: {original_query}
- Plan so far: {plan}
- Last action: {last_tool}
- Retrieved: {doc_count} docs, confidence: {confidence}
- Budget: {budget_remaining}

Decide the next step. Respond with JSON:
{
    "action": "rag_retrieve|code_execute|self_reflect|generate_answer",
    "params": {...},
    "reasoning": "Why this action"
}

Rules:
- If you just retrieved, vary your approach next time (loosen threshold, change legs, rephrase)
- If you have sufficient info, call generate_answer
- If stuck, call self_reflect
- Max 3 retrievals, 2 code executions, 2 reflections
```

### Memory-Aware Tool Selection

Before choosing `rag_retrieve`, the agent checks `state.tool_calls`:

```python
def _should_loosen_criteria(state: AgentState) -> bool:
    """Detect if we've already tried retrieval and should adjust."""
    recent = state.tool_calls[-3:]
    retrievals = [tc for tc in recent if tc["tool"] == "rag_retrieve"]
    if len(retrievals) >= 2:
        # Agent should loosen threshold or change legs, not repeat
        return True
    return False
```

---

## 4. Self-Evaluation & Self-Correction (Fixing Dead Code)

The existing `answer_evaluation_node` and `evaluator.py` are good — they just need to be wired up.

**Current bug**: `needs_retry` is hardcoded `False` in `graph_state.py:124`, and `answer_evaluation_node` always returns `needs_retry: False`.

**Fix**:
1. Remove hardcoded `False` from `graph_state.py`
2. Update `answer_evaluation_node` to set `needs_retry` based on scores:
   - `faithfulness < 60` → needs_retry = True (answer not grounded)
   - `completeness < 50` → needs_retry = True (missed key parts)
   - `citation_quality < 40` → needs_retry = True (bad citations)
3. Wire `route_after_answer_evaluation` to actually route back to `generating` when `needs_retry=True`

This reuses the existing evaluator — just fixes the wiring.

### Self-Reflection Loop

After evaluation, if gaps are found:

```python
async def self_evaluate(state: AgentState) -> dict:
    """Critique answer and decide if we need more retrieval."""
    evaluation = await evaluate_answer(
        query=state.original_query,
        answer=state.answer,
        context=format_context_string(state.retrieved_docs),
    )
    
    gaps = []
    if evaluation.faithfulness < 70:
        gaps.append("Answer not well-supported by retrieved context")
    if evaluation.completeness < 70:
        gaps.append("Answer doesn't fully address the query")
    if evaluation.citation_quality < 60:
        gaps.append("Citations are weak or missing")
    
    needs_retry = len(gaps) > 0 and state.iteration < 4
    
    return {
        "evaluation": evaluation,
        "gaps": gaps,
        "needs_retry": needs_retry,
    }
```

If `needs_retry`, the agent loops back with the gaps communicated as context. It should:
- Identify which parts of the query are unanswered
- Choose a different retrieval strategy (new query, different legs, lower threshold)

---

## 5. Attached Files (Enhanced)

### Current State
Files uploaded via `POST /api/chat/{chat_id}/files` are:
1. Stored on disk (`ephemeral/{chat_id}/filename`)
2. Converted to markdown asynchronously
3. Injected as `file_markdown` string appended to context

### Enhancement: File Query Tool

Instead of dumping all file content into context (wastes tokens, can't selectively use), treat files as a tool:

```python
# New tool: file_query
{
    "file_id": int,                    # Which uploaded file
    "query": str,                      # What to look for in the file
    "max_tokens": int,                # Max tokens to return
}
```

**How it works**:
1. Agent calls `file_query` with a specific search query
2. Tool searches the file's markdown content using dense retrieval (reuses `dense_search_docs()` against file chunks)
3. Only relevant excerpts returned, not entire file

**Fallback**: If agent doesn't call `file_query`, file content is still appended to context (current behavior preserved).

**State field**:
```python
state.attached_files = [
    {"id": 1, "name": "paper.pdf", "token_count": 5000, "status": "ready"},
]
```

Agent sees names + token counts but not content. Must explicitly query files.

---

## 6. Multi-Turn Conversation & Loosened Criteria

### Loosened Retrieval (3-Tier Strategy)

When first retrieval returns insufficient results:

```
Iteration 1: threshold -2.0, all legs (strict)
Iteration 2: threshold -5.0, all legs (loose)
Iteration 3: threshold -5.0, dense + sparse only, higher top_k (broadest)
```

The agent decides when to loosen based on:
- Doc count after filter
- Confidence score
- Whether any leg returned results

### Follow-Up Awareness

In multi-turn conversations:
- Agent checks: "Is this answerable from previous answer + new context?"
- If partially answerable: fill gaps with targeted retrieval
- If fully answerable: skip retrieval, generate from context

### Clarification Enhancement

Current `request_clarification` requires user input. New behavior:
1. If query is unclear, agent provides best-guess answer first
2. Appends: "I answered based on assumption X. If you meant Y, please clarify."
3. If user provides clarification, agent restarts with clarified query
4. If confidence is below threshold AND agent can't improve, offers to escalate

---

## 6a. Structured Context Pipeline (Multi-Turn Continuity)

> **Inspired by**: [pi coding-agent](https://github.com/earendil-works/pi/tree/main/packages/coding-agent) — specifically its compaction, branch summarization, and file-operation tracking systems.

### The Problem (Current State)

Our multi-turn context pipeline has three structural weaknesses:

**1. No compaction — unbounded growth with truncation as a band-aid.**
The Redis checkpointer accumulates the full message history. Assistant responses are truncated to 200 chars in `_build_generation_messages()`, which throws away the actual answer content. The LLM generating the next turn can't see what was previously decided, implemented, or discussed.

**2. Historical memory is a separate retrieval pass, not integrated into the message flow.**
`load_historical_memory_node` does a semantic search on stored Q&A pairs and injects results as text blocks. The LLM sees isolated Q&A snippets, not a conversation flow. "The thing about the database connection" won't match a past turn titled "PostgreSQL setup".

**3. No file/state tracking across turns.**
If a user says "now fix the error in that file", the agent has no idea which file. pi tracks read/write/edit operations and preserves them across compactions.

### What Pi Does Differently

pi uses a structured compaction system:

- **JSONL session file** with typed entries (messages, compaction summaries, branch summaries, custom messages)
- **Compaction triggers** when context nears the window limit — finds a valid cut point (never splits tool calls from results)
- **LLM-generated structured summary** with sections: Goal, Constraints, Progress, Key Decisions, Next Steps, Critical Context
- **File operation tracking** — read/write/edit sets preserved across compactions
- **Branch summaries** — when returning from a forked conversation, a `branchSummary` message is injected

The structured summary format:
```
## Goal
[What is the user trying to accomplish?]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)"]
```

### Our Implementation: Structured Compaction

**6a.1 Compaction node**

Add a `compact_context_node` that runs before `load_historical_memory` when the message history exceeds a threshold:

```python
async def compact_context_node(state: AgentState) -> dict:
    """Generate a structured summary when context is approaching limits."""
    messages = state.get("messages", [])
    token_estimate = estimate_messages_tokens(messages)
    
    if token_estimate < COMPACTION_THRESHOLD:
        return {}  # No compaction needed
    
    # Extract file operations from tool calls
    file_ops = extract_file_operations(messages)
    
    # Serialize conversation for summarization
    serialized = serialize_conversation(messages)
    
    # LLM call to generate structured summary
    summary = await generate_compaction_summary(
        serialized=serialized,
        file_ops=file_ops,
    )
    
    # Return summary for injection, mark messages for truncation
    return {
        "compaction_summary": summary,
        "file_operations": file_ops,
        "compaction_triggered": True,
    }
```

**6a.2 Structured memory storage**

Replace the current raw Q&A storage in `redis_memory.save_turn()` with structured entries:

```python
# Current (raw Q&A):
value = {"text": f"User: {query}\nAssistant: {answer}"}

# New (structured):
value = {
    "type": "turn",
    "query": query,
    "answer": answer,
    "summary": {
        "goal": "...",
        "decisions": [...],
        "next_steps": [...],
    },
    "file_ops": {"read": [...], "written": [...], "edited": [...]},
}
```

**6a.3 Merged context injection**

In `_build_generation_messages()`, replace the current two-part approach (raw checkpoint messages + separate memory docs) with a unified structured context:

```python
def _build_generation_messages(state: AgentState) -> list[dict]:
    system_parts = [_ANSWER_SYSTEM_PROMPT]
    
    # 1. Compaction summary (if triggered)
    if state.get("compaction_summary"):
        system_parts.append(f"PREVIOUS SESSION CONTEXT:\n{state['compaction_summary']}")
    
    # 2. File operations context (if any)
    if state.get("file_operations"):
        system_parts.append(f"FILE STATE:\n{format_file_operations(state['file_operations'])}")
    
    # 3. Historical memory (semantic search supplement for cross-turn recall)
    mem_docs = state.get("historical_memory_docs", [])
    if mem_docs:
        system_parts.append(f"PAST CONVERSATION REFERENCES:\n{_build_memory_context(mem_docs)}")
    
    # 4. Recent checkpoint messages (full, untruncated — compaction handles size)
    for m in state.get("messages", []):
        # No truncation — compaction keeps context bounded
        messages.append({"role": "user" if isinstance(m, HumanMessage) else "assistant", 
                        "content": str(m.content)})
    
    return messages
```

**6a.4 Subtask context sharing**

Currently each subagent gets only 2 message pairs and no memory docs. Update `route_after_classify` to pass richer context:

```python
def _subgraph_send_kwargs(state: AgentState) -> dict:
    return {
        "chat_id": state.get("chat_id"),
        "user_id": state.get("user_id"),
        "kb_ids": state.get("kb_ids", []),
        "org_id": state.get("org_id"),
        "file_markdown": state.get("file_markdown"),
        "historical_memory_docs": state.get("historical_memory_docs", []),
        # NEW: pass compaction summary to subagents
        "compaction_summary": state.get("compaction_summary", ""),
        # NEW: pass file operations to subagents
        "file_operations": state.get("file_operations", {}),
        # NEW: pass more recent history (not just 2 pairs)
        "messages": select_recent_history(state.get("messages", []), max_pairs=5),
    }
```

### Implementation Order

1. **Compaction utility functions** — `extract_file_operations()`, `serialize_conversation()`, `generate_compaction_summary()`, `format_file_operations()` in a new `context_manager.py`
2. **Compaction node** — `compact_context_node` in `nodes.py`, wired between `load_historical_memory` and `rewrite_query`
3. **Structured memory storage** — update `save_turn()` and `search_memory()` in `redis_memory.py`
4. **Unified context injection** — update `_build_generation_messages()` in `nodes.py`
5. **Subtask context sharing** — update `route_after_classify()` in `graph.py`

---

## 7. Structured Reasoning Traces

Agent's chain-of-thought is exposed to UI via `thinking` events (prefix `th:`):

```python
# In planning step
state.thinking_chunks.append(f"I should call {tool_name} because: {plan.reasoning}")

# In execution step
state.thinking_chunks.append(f"{tool_name} returned {result_summary}")

# In reflection step
state.thinking_chunks.append(f"After reflection: {reflection.analysis}")
```

Displayed as collapsible blocks in UI.

---

## 8. New Event Types

Existing events preserved. Additions:

| Event | Prefix | Purpose |
|-------|--------|---------|
| `tool_call` | `tc:` | Tool execution status: `{"tool": "rag_retrieve", "status": "started"|"done"|"error"}` |
| `progress` | `p:` | Enhanced with `action: "planning"|"retrieving"|"executing"|"reflecting"|"generating"` |
| `thinking` | `th:` | Enhanced with `step: "plan"|"execute"|"observe"|"reflect"` |

Existing events unchanged: `token` (0:), `context` (2:), `done` (d:), `task_list` (t:), `rewritten_query` (1:).

---

## 9. Implementation Plan

### Phase 1: Foundation (Week 1)

**1.1 Fix existing bugs (prerequisite)**
- Remove hardcoded `needs_retry=False` from `graph_state.py:124`
- Update `answer_evaluation_node` to compute `needs_retry` from evaluation scores
- Wire `route_after_answer_evaluation` to actually route back to `generating` when `needs_retry=True`

**1.2 Create tool wrapper layer**
- `backend/app/services/agentic_rag/tools/__init__.py` — Tool registry
- `backend/app/services/agentic_rag/tools/base.py` — Base tool class with schema validation
- `backend/app/services/agentic_rag/tools/rag_tool.py` — Wraps `retrieval/hybrid_search_with_legs()` + `rerank()` + `score_retrieval()`
- `backend/app/services/agentic_rag/tools/reflect_tool.py` — LLM call for self-reflection

**1.3 Update AgentState**
- Add: `tool_calls`, `tool_budget`, `plan`, `iteration`, `last_tool`, `attached_files`
- Keep all existing fields (retrieval docs, memory, evaluation, etc.)

### Phase 2: Structured Context Pipeline (Week 2)

> **Goal**: Replace unbounded truncation with structured compaction so the agent retains continuity across turns.

**2.1 Compaction utility functions**
- `backend/app/services/agentic_rag/context_manager.py` — NEW
  - `extract_file_operations(messages)` — parse tool calls for read/write/edit sets
  - `serialize_conversation(messages)` — convert messages to text for summarization
  - `generate_compaction_summary(serialized, file_ops)` — LLM call for structured summary
  - `format_file_operations(file_ops)` — XML-formatted file state
  - `estimate_messages_tokens(messages)` — token estimation (reuse from pi pattern)

**2.2 Compaction node**
- Add `compact_context_node` in `nodes.py`, wired between `load_historical_memory` and `rewrite_query`
- Triggers when `estimate_messages_tokens(messages) > COMPACTION_THRESHOLD`
- Inserts `compaction_summary` into state; triggers message truncation after summary

**2.3 Structured memory storage**
- Update `redis_memory.save_turn()` to store structured entries (goal, decisions, next_steps, file_ops)
- Update `redis_memory.search_memory()` to handle structured entry types

**2.4 Unified context injection**
- Update `_build_generation_messages()` in `nodes.py` — replace raw truncation with structured context:
  1. Compaction summary (if triggered)
  2. File operations context
  3. Historical memory (semantic search supplement)
  4. Recent checkpoint messages (full, untruncated)

**2.5 Subtask context sharing**
- Update `route_after_classify()` in `graph.py` — pass compaction summary, file ops, and expanded history (5 pairs) to subagents

### Phase 3: Agent Loop (Week 3)

**3.1 Implement agent loop**
- `backend/app/services/agentic_rag/agent_loop.py` — Main think-act-observe cycle
- `backend/app/services/agentic_rag/planner.py` — Planning LLM call (reuses existing LLM patterns)
- `backend/app/services/agentic_rag/tool_executor.py` — Dispatches tool calls with budget enforcement

**3.2 Implement self-evaluation with retry**
- Extend `evaluator.py` to return `needs_retry` and `gaps`
- Wire evaluation feedback into agent loop

**3.3 Implement memory-aware tool selection**
- `backend/app/services/agentic_rag/tool_memory.py` — Tracks tool usage per query
- Prevents redundant retrievals

### Phase 4: Code Execution & Files (Week 3)

**4.1 Code execution tool**
- `backend/app/services/agentic_rag/tools/code_tool.py` — Sandboxed Python execution
- Restricted globals, 5s timeout, no filesystem/network

**4.2 File query tool**
- `backend/app/services/agentic_rag/tools/file_tool.py` — Query attached files
- Uses dense retrieval against file markdown content

**4.3 Update API**
- Pass `attached_files` from `chat_files.py` to agent state

### Phase 5: Multi-Turn & Clarification (Week 4)

**5.1 Enhanced clarification**
- Agent provides best-guess + clarification request
- Handles follow-ups with context awareness

**5.2 Loosened retrieval**
- Agent adjusts retrieval parameters (threshold, legs, top_k)

### Phase 6: Streaming & UI Events (Week 4)

**6.1 New event types**
- `tool_call` events for each tool execution
- Enhanced `progress` and `thinking` events
- `compaction` event — signals when compaction triggered

**6.2 Update streaming transformer**
- `streaming.py` — Add handlers for new event types

### Phase 7: Testing & Polish (Week 4)

**7.1 Unit tests**
- Tool execution tests
- Agent loop budget tests
- Memory-aware selection tests
- File query tests
- Compaction utility tests (serialization, file ops extraction, token estimation)

**7.2 Integration tests**
- End-to-end agentic flow
- Multi-turn conversation with compaction
- Chart generation
- Clarification flow
- Subtask context sharing

**7.3 Performance testing**
- Latency comparison
- Token usage analysis
- Compaction trigger frequency and summary size

---

## 10. Files to Create/Modify

### New Files
```
backend/app/services/agentic_rag/
  agent_loop.py                    # NEW — main think-act-observe loop
  planner.py                       # NEW — planning LLM call
  tool_executor.py                 # NEW — tool dispatch
  tool_memory.py                   # NEW — memory-aware selection
  context_manager.py               # NEW — compaction utilities (extract_file_operations,
                                   #                serialize_conversation, generate_compaction_summary,
                                   #                format_file_operations, estimate_messages_tokens)
  tools/
    __init__.py                    # NEW — tool registry
    base.py                        # NEW — base tool class
    rag_tool.py                    # NEW — rag_retrieve wrapper
    code_tool.py                   # NEW — code_execute tool
    reflect_tool.py                # NEW — self_reflect tool
    file_tool.py                   # NEW — file_query tool
```

### Modified Files
```
backend/app/services/agentic_rag/
  graph_state.py                   # Add new state fields, fix needs_retry, add compaction fields
  evaluator.py                     # Add needs_retry + gaps to output
  streaming.py                     # Add tool_call event handling
  graph.py                         # Replace rigid graph with agent_loop, update subgraph context
  graph_runner.py                  # Update runner to use agent_loop
  nodes.py                         # Convert nodes to tool wrappers, add compact_context_node
  redis_memory.py                  # Structured memory storage (save_turn, search_memory)
  utils.py                         # Update format_context_string for files
```

### Config
```
backend/app/core/config.py         # Add TOOL_BUDGET limits, COMPACTION_THRESHOLD
```

---

## 11. Branch Strategy

Work on `feature/agentic-v2`. Completely replace v1 pipeline — no feature flag.

After implementation is complete and tested:
1. Merge `feature/agentic-v2` into `feature/langgraph-migration`
2. Delete v1 files during merge if any dead code remains

---

## 12. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Agent loops infinitely | High | Hard budget (max 5 iterations, per-tool limits) |
| More LLM calls = higher cost | Medium | Budget enforcement, caching retrieval results |
| Slower than rigid pipeline | Medium | Streaming progress, parallel tool execution where possible |
| Hallucinated tool parameters | Medium | Pydantic validation on all tool inputs |
| Code execution security | High | Sandboxed execution with restrictions |
| Complex debugging | Medium | Structured reasoning traces, tool_call events |
| Compaction loses critical context | High | Structured summary format preserves Goal, Decisions, Next Steps; review summaries during QA |
| Compaction triggers too aggressively | Medium | Threshold-based trigger with configurable margin; only triggers when token estimate exceeds threshold |
| Structured memory degrades search quality | Medium | Keep raw Q&A alongside structured data; semantic search works on both formats |

---

## 13. Tool Budget Configuration

```python
# In config.py
TOOL_BUDGET: dict = {
    "rag_retrieve": {"max": 3, "default_threshold": -2.0, "adaptive_threshold": -5.0},
    "code_execute": {"max": 2, "timeout_seconds": 5},
    "self_reflect": {"max": 2},
    "generate_answer": {"max": 1},
}

AGENT_MAX_ITERATIONS: int = 5
```

---

## 14. Appendix: Reuse Mapping

This table shows exactly what existing code maps to what in the new architecture:

| New Tool/Component | Existing Code Reused |
|-------------------|---------------------|
| `rag_retrieve` | `dense_search_docs()`, `sparse_search_docs()`, `exact_search_docs()`, `_merge_docs()`, `rerank()`, `score_retrieval()` |
| `generate_answer` | `generating_node()`, `_build_generation_messages()`, `_get_llm()` |
| `self_reflect` | Same LLM call pattern as `classify_query_node()` |
| `self_evaluate` | `evaluate_answer()` from `evaluator.py` |
| Query rewriting | `rewrite_query()` from `utils.py` |
| Memory loading | `load_historical_memory_node()` from `nodes.py` |
| Memory saving | `save_memory_node()` from `nodes.py` |
| Streaming | `AgenticRAGTransformer` from `streaming.py` |
| Retry | `with_retry()` / `with_retry_sync()` from `retry.py` |
| File upload | `upload_chat_file()` from `api/chat_files.py` |
| Clarification | `request_clarification_node()` from `nodes.py` |
| Chart validation | `chart_validation_node()` from `nodes.py` |
| Confidence scoring | `score_retrieval()` from `retrieval/confidence.py` |
| Reranking | `rerank()` from `retrieval/reranker.py` |
| Redis memory | `get_redis_memory()` from `redis_memory.py` |
| Graph expansion | `expand_docs_via_graph()` from `graph/` |
| Compaction summary | LLM call with structured prompt (same pattern as `classify_query_node()`) |
| Token estimation | `estimate_messages_tokens()` from `utils.py` (existing, reused) |
