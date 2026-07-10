# LangGraph Migration Plan: Agentic RAG Pipeline

> **Date:** 2026-07-10
> **Status:** Design
> **Reference:** [GiovanniPasq/agentic-rag-for-dummies](https://github.com/GiovanniPasq/agentic-rag-for-dummies)

---

## 1. Current State

### Our Architecture

```
run_agentic_rag() → query rewrite → heuristic classification → simple | complex
                                              ├─ simple: _direct_answer()
                                              │           rewrite → search → rerank → stream
                                              │
                                              └─ complex: _complex_answer()
                                                          rewrite → search → rerank → stream  (per subtask)
                                                          ↓ (loop)
                                                          synthesis → stream
```

**Current files:**
- `backend/app/services/agentic_rag/pipeline.py` — main pipeline entry point, generator-based
- `backend/app/services/agentic_rag/retry.py` — retry wrappers (Retriever, Generator)
- `backend/app/services/agentic_rag/evaluator.py` — post-generation quality evaluation
- `backend/app/services/agentic_rag/context_manager.py` — token budgeting, document pruning
- `backend/app/services/agentic_rag/prompts.py` — system prompts
- `frontend/src/components/chat/agent-timeline.tsx` — UI displaying progress phases

**Event protocol (SSE prefixes):**

| Prefix | Meaning |
|--------|---------|
| `p:` | progress — transient status messages |
| `t:` | task_list — subtask list with status |
| `th:` | thinking — reasoning model chain-of-thought |
| `0:` | token — streaming answer text |
| `1:` | rewritten_query — standalone query (internal) |
| `2:` | context — retrieved documents |
| `3:` | error — exception message |
| `d:` | done — finish reason + usage |

**Dependencies already present:** `langgraph>=0.2.0`, `langchain-core>=0.3.0`, `langchain-openai>=0.2.0`

---

## 2. Reference Architecture

The reference repo uses a **two-level StateGraph** architecture:

```
Main Graph:                     Agent Subgraph:
─────────────────               ─────────────────
START → summarize_history ─→    START → orchestrator ─→ tools ─→ should_compress
      → rewrite_query ─→        → compress_context ─→ (back to orchestrator)
              ├→ request_clarification  → fallback_response
              └→ Send(agent, ...)       → collect_answer → END
                        ↓
                 aggregate_answers → END
```

### Key Patterns from the Reference

| Pattern | Benefit |
|---------|---------|
| **Nested StateGraph** | Main graph contains an agent subgraph with independent lifecycle |
| **State accumulation** | `Annotated[List, operator.add]` for automatic accumulation across nodes |
| **`Command` object** | Conditional routing with state updates |
| **`Send` API** | Parallel fan-out to multiple agent instances |
| **Checkpointing** | `InMemorySaver()` at every node transition |
| **`interrupt()`** | Human-in-the-loop pause/continue |
| **ToolNode** | Built-in tool execution node |
| **Structured output** | `llm.with_structured_output(QueryAnalysis)` for query classification |

---

## 3. Lessons Learned from the Reference

### What Works (copy these)

| Pattern | Benefit | Applicable? |
|---------|---------|-------------|
| **Bounded conversation memory** | Rolling summaries prevent context bloat in long conversations | **YES** — add to pipeline |
| **Query clarification** | LLM classifies if query is answerable; interrupts for more info | **YES** — add clarification node |
| **Context compression** | Compresses accumulated retrieval context between iterations | **YES** — add compression node |
| **Structured query analysis** | `QueryAnalysis` Pydantic model for clarity, questions, clarifications | **YES** — replace `_classify_complexity` heuristic |
| **Fallback response** | Dedicated node with special "do your best" prompt | **YES** — add fallback |
| **Accumulator state** | `Annotated[List, operator.add]` for automatic accumulation | **YES** — for messages, answers, contexts |
| **Tool abstraction** | Separate tool functions with clear docstrings that guide the LLM | **YES** — wrap retriever as LangChain tool |
| **Message naming** | Tag subgraph-only messages so they don't leak into chat history | **MAYBE** — useful if we preserve messages |
| **Reduction-based summarization** | Merge existing summary + new messages (not re-summarize from scratch) | **YES** |
| **Token threshold gating** | `should_compress_context` checks token budget before compressing | **YES** — we have budget tracking already |
| **Retrieval keys** | Track what was already retrieved to avoid repetition | **YES** — add to state |
| **RAGAS evaluation** | Evaluate contexts the agent actually consumed, not just final answer | **YES** — add eval module |

### What Doesn't Apply (skip these)

| Pattern | Why Not |
|---------|---------|
| **Parallel sub-agents via `Send`** | We don't need true fan-out; sequential subtask loop is fine |
| **Multi-agent supervision** | One agent making decisions is correct for our problem space |
| **Agent-to-agent handoffs** | No specialist agents (writer, coder, analyst) in our use case |
| **Graphene query tools** | Their `graph_query` tool is domain-specific |

### What We Should Add That They Don't Have

| Addition | Rationale |
|----------|-----------|
| **Context sufficiency loop** | Our existing 2-iteration widening strategy is more sophisticated than their single-pass retrieval |
| **Streaming-first design** | They aggregate at the end; we stream in real-time — our competitive advantage |
| **Chart validation** | We validate ECharts JSON in answers; they don't have this |
| **Thinking model auto-select** | We auto-select reasoning model based on query keywords; they use one model |
| **Hierarchical indexing** (parent/child chunks) | Reference uses this; we should adopt it for better retrieval granularity |
| **Retry with threshold widening** | Our `RetryConfig` with progressive threshold lowering is more sophisticated |

---

## 4. Migration Strategy

### Phased Approach

We migrate the pipeline incrementally — each phase ships working code, none is "all-or-nothing".

```
Phase 1: State Layer          → Define State, move data from local variables
Phase 2: Node Wrappers        → Wrap existing functions as LangGraph nodes
Phase 3: Graph Compilation    → Connect nodes into a StateGraph
Phase 4: Enhancements         → Clarification, compression, fallback, tools
```

---

### Phase 1: State Layer

**Goal:** Define LangGraph-compatible `State` with all data flowing through the graph.

**New file:** `backend/app/services/agentic_rag/graph_state.py`

```python
from typing import List, Annotated, Literal, Optional
from langgraph.graph import MessagesState
import operator

def accumulate(existing: List, new: List) -> List:
    """Default reducer: append new items to existing list."""
    if new and any(isinstance(item, dict) and item.get('__reset__') for item in new):
        return []
    return existing + new

def set_union(a: set, b: set) -> set:
    return a | b

class AgentState(MessagesState):
    """State for the agent graph. Extends MessagesState with agentic fields."""
    # ── Query state ─────────────────────────────────────────────────────
    original_query: str = ""
    rewritten_query: str = ""
    is_complex: bool = False
    subtasks: List[str] = []
    current_subtask_index: int = 0
    
    # ── Clarification state ─────────────────────────────────────────────
    question_is_clear: bool = True
    pending_query: str = ""
    clarification_questions: List[str] = []
    
    # ── Retrieval state ─────────────────────────────────────────────────
    retrieved_docs: Annotated[List[dict], accumulate] = []
    retrieved_contexts: Annotated[List[str], accumulate] = []
    retrieval_keys: Annotated[set, set_union] = set()  # Track what we've already retrieved
    retrieval_iterations: int = 0
    retrieval_confidence: float = 0.0
    
    # ── Generation state ────────────────────────────────────────────────
    answer: str = ""
    thinking_chunks: List[str] = []
    
    # ── Synthesis state ─────────────────────────────────────────────────
    subtask_answers: Annotated[List[dict], accumulate] = []  # {subtask, answer, docs}
    final_answer: str = ""
    
    # ── Configuration ───────────────────────────────────────────────────
    kb_ids: List[int] = []
    org_id: Optional[int] = None
    file_markdown: Optional[str] = None
    existing_summary: str = ""
    
    # ── Metadata ────────────────────────────────────────────────────────
    latency_ms: int = 0
    model_used: str = ""
```

**Changes to existing files:**
- `pipeline.py` — remove all local variables that become state fields
- No code logic changes yet — state just flows through the graph

---

### Phase 2: Node Wrappers

**Goal:** Wrap existing pipeline functions as LangGraph nodes with proper signatures.

**New file:** `backend/app/services/agentic_rag/nodes.py`

Map our existing functions to nodes:

| Our Function | LangGraph Node | Changes Required |
|--------------|---------------|------------------|
| `_rewrite_query()` | `rewrite_query_node(state, llm)` | Wrap in node signature, emit events via callback |
| `_classify_complexity()` | `classify_query_node(state, llm)` | Replace heuristic with structured LLM output |
| `_search_and_rerank()` | `retrieval_node(state, db, kb_ids)` | Wrap as node, update state with docs |
| `_generate_answer()` | `generate_node(state, llm)` | Stream from node, update state.answer |
| `_complex_answer()` | No single node | Decompose into orchestrator → loop → synthesize |

```python
from langgraph.types import Command

def rewrite_query_node(state: AgentState, llm) -> Command:
    """Rewrite query using chat history. Same logic as _rewrite_query."""
    from openai import AsyncOpenAI
    from app.core.config import settings
    
    # ... existing rewrite logic ...
    
    # Emit events via callback (see Section 5)
    emit_event(state, "progress", "rewriting", "Rewriting query...")
    emit_event(state, "rewritten_query", state["rewritten_query"])
    
    return {"rewritten_query": rewritten}

def classify_query_node(state: AgentState, llm) -> Command:
    """Classify query using structured LLM output (replaces _classify_complexity).
    
    Uses the reference repo's QueryAnalysis model.
    """
    from .schemas import QueryAnalysis  # New file
    llm_structured = llm.with_structured_output(QueryAnalysis)
    
    response = llm_structured.invoke(...)
    
    state["question_is_clear"] = response.is_clear
    state["subtasks"] = response.questions
    state["is_complex"] = len(response.questions) > 1
    
    # Route based on complexity
    if not response.is_clear:
        return Command(
            update={"question_is_clear": False, "pending_query": state["rewritten_query"]},
            goto="request_clarification",
        )
    elif len(response.questions) > 1:
        return Command(goto="agent_loop")  # subtask path
    else:
        return Command(goto="direct_retrieval")  # simple path
```

---

### Phase 3: Graph Compilation

**Goal:** Wire nodes into a StateGraph with proper edges and conditional routing.

**New file:** `backend/app/services/agentic_rag/graph.py`

```python
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode

from .graph_state import AgentState
from .nodes import (
    rewrite_query_node,
    classify_query_node,
    request_clarification_node,
    direct_retrieval_node,
    agent_loop_orchestrator,
    agent_tools,
    should_compress_context,
    compress_context_node,
    fallback_response_node,
    collect_answer_node,
    synthesize_node,
)
from .edges import (
    route_after_rewrite,
    route_after_clarification,
    route_after_orchestrator,
    route_after_should_compress,
)

def create_agent_graph(llm, tools_list=None, checkpointer=None):
    """Create the compiled LangGraph agent graph."""
    
    # ── Agent subgraph (self-correcting retrieval loop) ─────────────────
    agent_builder = StateGraph(AgentState)
    agent_builder.add_node("orchestrator", agent_loop_orchestrator)
    agent_builder.add_node("tools", ToolNode(tools_list or []))
    agent_builder.add_node("compress", compress_context_node)
    agent_builder.add_node("fallback", fallback_response_node)
    agent_builder.add_node("collect", collect_answer_node)
    agent_builder.add_node("should_compress", should_compress_context)
    
    agent_builder.add_edge(START, "orchestrator")
    agent_builder.add_conditional_edges(
        "orchestrator", route_after_orchestrator,
        {"tools": "tools", "fallback": "fallback", "collect": "collect"}
    )
    agent_builder.add_edge("tools", "should_compress")
    agent_builder.add_edge("compress", "orchestrator")  # self-correcting loop
    agent_builder.add_edge("fallback", "collect")
    agent_builder.add_edge("collect", END)
    
    agent_subgraph = agent_builder.compile()
    
    # ── Main graph ──────────────────────────────────────────────────────
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("rewrite", rewrite_query_node)
    graph_builder.add_node("classify", classify_query_node)
    graph_builder.add_node("clarification", request_clarification_node)
    graph_builder.add_node("direct_retrieval", direct_retrieval_node)
    graph_builder.add_node("agent", agent_subgraph)  # nested subgraph
    graph_builder.add_node("synthesize", synthesize_node)
    
    graph_builder.add_edge(START, "rewrite")
    graph_builder.add_edge("rewrite", "classify")
    graph_builder.add_conditional_edges("classify", route_after_classification,
        {"clarification": "clarification", "direct": "direct_retrieval", "agent": "agent"}
    )
    graph_builder.add_edge("clarification", "classify")  # loop back after clarification
    graph_builder.add_edge("direct_retrieval", "synthesize")
    graph_builder.add_edge("agent", "synthesize")
    graph_builder.add_edge("synthesize", END)
    
    main_graph = graph_builder.compile(checkpointer=checkpointer)
    return main_graph
```

**New file:** `backend/app/services/agentic_rag/edges.py`

```python
from langgraph.types import Command

def route_after_classification(state: AgentState) -> Command:
    """Route to clarification, direct retrieval, or agent subgraph."""
    if not state["question_is_clear"]:
        return Command(goto="clarification")
    elif state["is_complex"]:
        return Command(goto="agent")  # enter subgraph
    else:
        return Command(goto="direct_retrieval")

def route_after_clarification(state: AgentState) -> Command:
    """After clarification, go back to classification."""
    return Command(goto="classify")

def route_after_orchestrator(state: AgentState) -> Command:
    """Route from orchestrator based on tool calls or answer."""
    tool_calls = getattr(state["messages"][-1], "tool_calls", []) or []
    if not tool_calls:
        return Command(goto="collect")  # answer generated
    
    # Check iteration/tool call budgets
    if state.get("iteration_count", 0) >= 8 or state.get("tool_call_count", 0) >= 20:
        return Command(goto="fallback")  # budget exceeded
    
    return Command(goto="tools")

def route_after_should_compress(state: AgentState) -> Command:
    """Route based on whether token budget is exceeded."""
    from .utils import estimate_context_tokens
    from app.core.config import settings
    
    current_tokens = estimate_context_tokens(state["messages"])
    max_tokens = settings.OPENAI_MODEL_CONTEXT_SIZE * 0.8
    
    if current_tokens > max_tokens:
        return Command(goto="compress")
    return Command(goto="orchestrator")
```

---

### Phase 4: Enhancements

High-value features from the reference repo added on top of the migrated graph.

#### 4a. Bounded Conversation Memory

**File:** `backend/app/services/agentic_rag/nodes.py` — `summarize_history` function

```python
def summarize_history_node(state: AgentState, llm) -> dict:
    """Reduce conversation history using rolling summaries.
    
    Pattern from reference repo: merge existing summary with new messages,
    keep the last N turn pairs intact.
    """
    messages = state.get("messages", [])
    
    # Keep last 2 turn pairs (4 messages) intact
    plain_msgs = [m for m in messages if not getattr(m, "tool_calls", None)]
    keep_count = 4
    keep_ids = {getattr(m, "id", None) for m in plain_msgs[-keep_count:]}
    
    # Summarize older messages
    older = plain_msgs[:-keep_count] if len(plain_msgs) > keep_count else []
    if older:
        existing_summary = state.get("existing_summary", "").strip()
        conversation = f"Existing summary:\n{existing_summary or '(none)'}\n\n"
        conversation += "New messages:\n" + "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content[:200]}"
            for m in older
        )
        
        summary_response = llm.invoke([
            {"role": "system", "content": SUMMARIZE_PROMPT},
            {"role": "user", "content": conversation},
        ])
        
        return {"existing_summary": summary_response.content.strip()}
    
    return {}
```

**Integration:** Add `summarize_history_node` as the first node after START.

#### 4b. Context Compression Between Retrieval Iterations

**File:** `backend/app/services/agentic_rag/nodes.py` — `compress_context_node`

```python
def compress_context_node(state: AgentState, llm) -> dict:
    """Compress accumulated retrieval context to free token budget.
    
    Pattern from reference repo: summarize conversation history (retrieval
    messages, tool results, answers) into a condensed context. Track what
    has already been retrieved to avoid repeating searches.
    """
    messages = state.get("messages", [])
    existing_summary = state.get("retrieved_contexts", "")
    
    conversation = f"USER QUESTION: {state['rewritten_query']}\n\n"
    if existing_summary:
        conversation += f"COMPRESSED CONTEXT (from prior iterations):\n{existing_summary}\n\n"
    
    for msg in messages[1:]:  # skip the human question
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            calls = ", ".join(f"{tc['name']}({json.dumps(tc['args'])})" for tc in msg.tool_calls)
            conversation += f"ASSISTANT (tool calls: {calls}): {msg.content or '(tool call only)'}\n\n"
        elif isinstance(msg, ToolMessage):
            conversation += f"TOOL RESULT ({getattr(msg, 'name', 'tool')}):\n{msg.content}\n\n"
    
    response = llm.invoke([
        {"role": "system", "content": COMPRESSION_PROMPT},
        {"role": "user", "content": conversation},
    ])
    
    new_summary = response.content
    
    # Track what was already retrieved to avoid repetition
    retrieval_keys = set(state.get("retrieval_keys", set()) or set())
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                retrieval_keys.add(f"{tc['name']}:{json.dumps(tc['args'], sort_keys=True)}")
    
    return {
        "retrieved_contexts": [new_summary],
        "retrieval_keys": list(retrieval_keys),
    }
```

**Integration:** Called from `should_compress_context` routing when token budget is exceeded.

#### 4c. Query Clarification

**File:** `backend/app/services/agentic_rag/nodes.py` — `request_clarification_node`

```python
def request_clarification_node(state: AgentState, llm) -> dict:
    """Ask the user for clarification when the query is unclear.
    
    In LangGraph, this is where interrupt() would pause execution.
    For now, we emit a message that gets added to the conversation.
    """
    pending = state.get("pending_query", "")
    clarifications = state.get("clarification_questions", [])
    
    if clarifications:
        context = "\n".join(f"{i+1}. {c}" for i, c in enumerate(clarifications))
        clarification_msg = (
            f"I need more information to answer your question about: '{pending}'\n\n"
            f"Please clarify:\n{context}"
        )
    else:
        clarification_msg = f"I need more information to understand your question: '{pending}'"
    
    return {
        "messages": [AIMessage(content=clarification_msg, name="clarification")],
    }
```

**Integration:** The main graph routes to `clarification` node, which adds a clarification message to the conversation, then routes back to `classify` which re-evaluates the now-clarified query.

#### 4d. Fallback/Last-Resort Prompt

**File:** `backend/app/services/agentic_rag/prompts.py` — add fallback prompt

```python
FALLBACK_RESPONSE_PROMPT = """\
You are a helpful assistant. The user asked a question but insufficient data was found in the documents.

DO NOT make up information. If you truly have no relevant information:
1. Acknowledge that the documents don't contain enough information to answer fully.
2. Provide any partial answer you can based on what was found.
3. Suggest what the user might look for instead.

The user's question was: {question}

Retrieved context (may be partially relevant):
{context}

Be honest about limitations but still be helpful.
"""
```

**Integration:** `fallback_response_node` is called when the orchestrator hits the iteration/tool-call budget without producing an answer.

#### 4e. Hierarchical Retrieval Tools

**File:** `backend/app/services/agentic_rag/tools.py`

```python
"""LangChain tool definitions for the agent graph.

Wraps our existing retrieval infrastructure as LangChain tools that the
orchestrator can call during its self-correction loop.
"""

from langchain_core.tools import tool

@tool
def search_child_chunks(query: str, limit: int = 10) -> str:
    """Search document excerpts for evidence related to the user question.
    
    Use this as the first retrieval step. Results include parent IDs, file
    names, and short child-chunk excerpts. If excerpts are relevant but too
    fragmented to answer confidently, call retrieve_parent_chunks with the
    returned parent_id.
    
    Args:
        query: Focused search query with concrete keywords from the question.
        limit: Maximum number of child chunks to return (default: 10).
    """
    # Wrap existing hybrid_search_with_legs
    ...

@tool
def retrieve_parent_chunks(parent_id: str) -> str:
    """Retrieve the full parent chunk for a relevant child search result.
    
    Use this only after search_child_chunks returns a relevant parent_id and
    the child excerpt needs more surrounding context.
    
    Args:
        parent_id: Parent chunk ID returned by search_child_chunks.
    """
    # Wrap existing parent chunk retrieval
    ...
```

**Integration:** `ToolNode` in the agent subgraph executes these tools when the orchestrator's LLM response includes tool calls.

---

## 5. Event Emission Strategy

The challenge: in our current pipeline, progress events flow directly from generator yields to SSE. In LangGraph, nodes return state updates — they don't yield.

### Strategy: Custom Callback Handler

```python
# In callbacks.py
from langchain_core.callbacks import BaseCallbackHandler

class SSEEventHandler(BaseCallbackHandler):
    """Emit SSE events from graph node transitions."""
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        node_name = serialized.get("name", "")
        yield {"event": "progress", "phase": node_name, "message": f"Starting {node_name}..."}
        yield {"event": "agent_step", "node": node_name, "status": "active"}
    
    def on_chain_end(self, outputs, **kwargs):
        node_name = ...
        yield {"event": "progress", "phase": node_name, "message": f"Finished {node_name}"}
        yield {"event": "agent_step", "node": node_name, "status": "done"}
```

This preserves our existing `p:`/`t:`/`th:`/`0:` event protocol without changes to the frontend.

---

## 6. Streaming Impact Analysis

| Aspect | Current Pipeline | With LangGraph | Mitigation |
|--------|-----------------|----------------|------------|
| **Token streaming** | Zero overhead — direct `ChatOpenAI.astream()` → yield | Node returns after stream completes — tokens appear at end | Stream from within nodes using `astream` + yield via callback |
| **Progress events** | Direct yield → SSE | Node completion → callback → SSE | `on_chain_start`/`on_chain_end` callbacks handle this |
| **Subtask task_list** | Yield in loop → SSE | Need to emit from within nested subgraph | Subgraph callbacks propagate events to parent |
| **Thinking blocks** | Detected via reasoning tag extraction → yield | Same extraction happens inside `generate` node | Works identically — thinking detection is in `_generate_answer` |
| **Latency** | ~0ms overhead | ~5-20ms per checkpoint write (InMemorySaver) | Use `checkpointer=None` for now; add later if needed |
| **Memory** | O(tokens) per generation | O(tokens * nodes) — state persisted at each node | Clear checkpoint between sessions; use `InMemorySaver` |

**Critical:** We must keep streaming alive. LangGraph's `astream()` yields full node completions, not individual tokens. To preserve real-time streaming, we:

1. Stream from within each node (the node returns after the stream completes)
2. Use `astream_events()` with `subgraphs=True` to surface subgraph events
3. Apply a token filter that yields individual token events to the client

---

## 7. File Changes Summary

### New Files

| File | Purpose |
|------|---------|
| `graph_state.py` | State definitions with accumulators |
| `schemas.py` | Pydantic models (QueryAnalysis) |
| `prompts.py` | System prompts (rewrite, classification, compression, fallback, aggregation) |
| `tools.py` | LangChain tool definitions |
| `graph.py` | Main graph compilation |
| `edges.py` | Conditional routing logic |
| `nodes.py` | Node implementations (wrapping existing functions) |
| `utils.py` | Helper functions (token estimation, message formatting) |
| `callbacks.py` | SSE event callback handlers |

### Modified Files

| File | Changes |
|------|---------|
| `pipeline.py` | Entry point calls `graph.astream()`, bridges events to SSE |
| `__init__.py` | Updated to import from new structure |
| `retry.py` | Minimal changes — retry still wraps individual function calls |
| `evaluator.py` | Minimal changes — evaluation happens post-streaming |
| `context_manager.py` | Token budget moved into state; budget checks in edges |
| `chat_service.py` | No changes — still calls `run_agentic_rag()` |
| `agent-timeline.tsx` | No changes — same event protocol (p: t: th: 0: 1: 2: 3: d:) |

---

## 8. Testing Strategy

### Unit Tests

| Test | What it validates |
|------|-------------------|
| `test_graph_state.py` | State schema, accumulator reducers, union operators |
| `test_schemas.py` | QueryAnalysis Pydantic model parsing |
| `test_edges.py` | All routing functions (classify, clarify, orchestrator, compress) |
| `test_prompts.py` | Prompt string formatting |
| `test_tools.py` | Tool definitions, docstrings, args parsing |

### Integration Tests

| Test | What it validates |
|------|-------------------|
| `test_simple_path.py` | Full graph execution for simple queries |
| `test_complex_path.py` | Full graph execution for complex queries (subgraph → synthesize) |
| `test_clarification.py` | Graph with unclear query → clarification → re-classification → agent |
| `test_context_compression.py` | Agent subgraph with >MAX tokens → compress → continue |
| `test_fallback.py` | Agent that hits iteration budget → fallback response |
| `test_retry_integration.py` | Retry wrapper works with LangGraph nodes |

### Migration Validation

Run against the existing test suite (329 tests):
- All existing pipeline tests should pass with minimal changes
- Add `langgraph_graph` as a test fixture alongside existing pipeline fixtures
- Compare output of LangGraph path vs generator path for same input

---

## 9. Migration Timeline

| Phase | Scope | Est. Effort | Risk |
|-------|-------|-------------|------|
| **Phase 1: State Layer** | Define `AgentState`, wire through existing code | 2 days | Low — no behavior changes |
| **Phase 2: Node Wrappers** | Wrap existing functions as nodes, keep generator flow | 3 days | Medium — callback system needed |
| **Phase 3: Graph Compilation** | Connect nodes, compile graph, test paths | 3 days | High — graph routing bugs possible |
| **Phase 4: Enhancements** | Clarification, compression, fallback, tools | 4 days | Medium — new features, new tests |
| **Phase 5: Validation** | Test suite migration, performance benchmark, bug fixes | 3 days | Low — regression only |
| **Total** | | **~15 days** | |

### Risk Mitigation

1. **Don't ship Phase 1-3 without Phase 0** — keep generator pipeline running alongside LangGraph graph. Feature-flag the graph:

```python
if settings.USE_LANGGRAPH:
    return run_via_graph(query, ...)
else:
    return run_via_generator(query, ...)  # existing code
```

2. **A/B compare** — for the first 2 weeks after Phase 3, log both graph and generator outputs for same queries and compare.

3. **Streaming test first** — verify that token streaming latency hasn't increased by >20% before proceeding with Phase 4.

4. **Rollback plan** — if Phase 3 breaks critical functionality, revert to generator pipeline. They coexist behind the flag.

---

## 10. When NOT to Migrate

If any of these become true, **do not migrate** and instead add features individually:

1. Our agent is truly simple (one query → one answer) — no need for graph orchestration
2. Streaming latency increases by >50% — unacceptable for user experience
3. The graph introduces more bugs than it fixes — counterproductive
4. We only need bounded memory or context compression — single-function additions suffice

### Add These Features Without Migration

| Feature | What to Add | Cost |
|---------|-------------|------|
| Bounded conversation memory | `summarize_history()` function in `context_manager.py` | 1 day |
| Context compression | `compress_context()` function in `context_manager.py` | 0.5 days |
| Fallback prompt | Add prompt string to `prompts.py` | 0.25 days |
| Query clarification | Enhanced `_classify_complexity` with structured output | 1 day |
| Hierarchical indexing | Add parent/child chunk retrieval to ingestion pipeline | 2 days |
| RAGAS evaluation | Add eval module alongside pipeline | 2 days |

These deliver 80% of the value with 20% of the migration cost.

---

## 11. Decision Criteria

### Proceed with Migration When:
- [ ] We need **human-in-the-loop clarification** that pauses execution (LangGraph's `interrupt()` is the right tool)
- [ ] We need **production debugging** with Langfuse traces per node
- [ ] We need **state persistence** across server restarts (checkpoint recovery)
- [ ] We add a **second specialized agent** (tool-calling agent alongside retriever agent)
- [ ] We have **3+ agents** communicating (orchestrator → worker → critic)

### Defer Migration Until:
- [ ] We have a concrete blocker that requires checkpointing
- [ ] We have user complaints about missing clarification flow
- [ ] We've validated that individual additions (bounded memory, compression) don't solve our problems
- [ ] We're ready to invest in Langfuse observability

---

## 12. Key Code Snippets

### QueryAnalysis Schema (from reference repo)

```python
# schemas.py
from pydantic import BaseModel, Field

class QueryAnalysis(BaseModel):
    """Structured output for query classification."""
    is_clear: bool = Field(
        description="Whether the user's question is clear and answerable from the knowledge base."
    )
    questions: List[str] = Field(
        description="List of rewritten, self-contained questions extracted from the query."
    )
    clarification_needed: str = Field(
        description="Explanation of what additional information is needed, or empty string if none."
    )
```

### should_compress_context (from reference repo)

```python
def should_compress_context(state: AgentState) -> Command:
    """Decide whether to compress context based on token budget.
    
    Returns Command(goto='compress') if tokens exceed threshold,
    otherwise Command(goto='orchestrator') to continue the loop.
    """
    current_tokens = estimate_context_tokens(state["messages"])
    max_allowed = settings.OPENAI_MODEL_CONTEXT_SIZE * 0.8
    
    if current_tokens > max_allowed:
        return Command(goto="compress")
    return Command(goto="orchestrator")
```

### Command Object (for routing with state updates)

```python
from langgraph.types import Command

# Route with state updates (replaces simple goto)
return Command(
    update={
        "rewritten_query": "rewritten text",
        "question_is_clear": True,
    },
    goto="direct_retrieval",
)

# Route with state updates AND subgraph spawn
return Command(
    update={"pending_query": "query text"},
    goto="agent",  # enter nested subgraph
)
```

---

## 13. Appendix: Reference Pattern Mapping

| Reference Node | Our Equivalent | Migration Path |
|---------------|----------------|----------------|
| `summarize_history` | `context_manager.compress_history` | Extract into node |
| `rewrite_query` | `_rewrite_query` | Wrap as node |
| `request_clarification` | _nothing_ | Add new node |
| `orchestrator` (subgraph) | _complex_answer logic_ | Decompose into orchestrator → tools → loop |
| `tools` | `_search_and_rerank` | Wrap as LangChain ToolNode |
| `compress_context` | `context_manager.compress_history` | Extract into node |
| `should_compress_context` | _nothing_ | Add new function |
| `fallback_response` | _nothing_ | Add new node |
| `collect_answer` | _nothing_ | Add new node |
| `aggregate_answers` | `_complex_answer` synthesis | Keep as node, add LangGraph integration |
