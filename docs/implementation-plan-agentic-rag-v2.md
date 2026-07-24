# Agentic RAG Pipeline v2 — Implementation Plan

## Overview

This document describes the comprehensive redesign of the agentic RAG pipeline to support:

1. **Multi-turn conversation fluency** — model understands it's continuing a session, not repeating itself
2. **Structured conversation compaction** — long conversations are summarized into checkpoints (inspired by [pi](https://github.com/earendil-works/pi))
3. **RAG vs Chat-only routing** — decision node determines whether a query needs document retrieval or can be answered from conversation
4. **File attachment handling** — uploaded files participate in conversation flow, not just retrieval context
5. **Parallel and sequential subtask execution** — independent subtasks run in parallel; dependent subtasks run sequentially with context passing

## Current Architecture

```
START → rewrite_query → classify_query → Send(agent_subgraph, ...) → prepare_final_context → generating → answer_evaluation → finalize_answer → save_memory → END
```

### Limitations

- **No conversation fluency:** System prompt says "answer using the provided context" — model treats each turn as a fresh retrieval task
- **No compaction:** Message history grows unbounded; when context window is exceeded, the call fails
- **No routing:** Every query runs the full retrieval pipeline (dense + sparse + exact + reranking), even for chat-only queries like "summarize what we discussed"
- **No file awareness:** Files are prepended to the query on every turn but never stored in conversation history; chat-only path can't answer questions about files
- **No subtask dependencies:** Dependent subtasks are concatenated into one query string; reference resolution ("how does that compare") is fragile

---

## Proposed Architecture

```
                    ┌─ parallel independent subtasks (Send) ─┐
START → rewrite → compaction → classify → route_by_dependencies → collect_context → prepare_final_context → generating → answer_evaluation → finalize_answer → save_memory → END
                    └─ sequential dependent subtasks (loop) ──┘
```

### Key Components

| Component | Responsibility |
|-----------|---------------|
| `compaction_node` | Summarizes older conversation turns into a structured checkpoint when message count exceeds threshold |
| `classify_query_node` | Extracts subtasks, dependency graph, and routing decision (needs_retrieval, needs_file_content, needs_file_metadata) |
| `route_by_dependencies` | Groups subtasks by dependency level; routes independent subtasks via `Send()`, dependent subtasks via sequential loop |
| `sequential_subtask_loop` | Executes dependent subtasks one-by-one, enriching each query with context from its dependencies |
| `generating_node` | Assembles the final prompt based on routing decision and available context |

---

## 1. Multi-Turn Conversation Fluency

### 1.1 System Prompt Rewrite

**Current:**
```
You are a helpful assistant. Answer the user's question using ONLY the provided context.
```

**New:**
```
You are a helpful assistant operating within an ongoing conversation session.
You answer the user's questions using the provided document context.

## Session Awareness
This is a continuing session — you have previous conversation history below.
- You CAN and SHOULD reference your own prior answers when relevant
  ("as I mentioned above", "to continue from the last point", "building on that").
- The user may ask follow-up questions that reference earlier discussion.
  Use the conversation history to understand what "that", "the second one",
  "similar to above", etc. refer to.
- Do NOT repeat your entire previous answer when the user asks a follow-up.
  Only address what is new or different in the follow-up.
- If a follow-up question can be fully answered from your prior response
  without needing new document retrieval, answer it directly.

## Document Context
Answer the user's question using ONLY the provided document context below.
If the context is insufficient or irrelevant, say so clearly.
When you need to answer from both documents and general knowledge,
prioritize the documents and note when you are supplementing with general knowledge.

[... formatting and citation rules unchanged ...]
```

### 1.2 History Inclusion in Generation

**Current:** Only prior user messages are included in generation. Assistant responses are excluded to avoid "context poisoning" (model regurgitating previous long answers).

**New:** Recent assistant responses (capped at 300 chars) are included alongside user messages. The cap prevents poisoning while preserving conversational continuity.

**Generation prompt structure:**
```
system: [session-aware prompt + document context if retrieval]
user: <conversation_checkpoint>\n[compaction summary if exists]
user: [recent user messages]
user: [recent assistant responses, 300 chars each]
user: [current query]
```

**When compaction has triggered:** The checkpoint summary replaces raw history. The model sees a structured checkpoint of older turns plus recent raw messages.

**When compaction has NOT triggered (early turns):** The model sees recent raw user/assistant messages directly.

---

## 2. Conversation Compaction

### 2.1 Trigger Conditions

Compaction runs after `rewrite_query` but before `classify_query`. It triggers when:

```
len(messages) > COMPACTION_HISTORY_THRESHOLD  (default: 20)
```

### 2.2 Compaction Process

1. **Split messages:** Keep recent `COMPACTION_KEEP_RECENT` (default: 10) messages; summarize the rest
2. **Build conversation text:** Convert old messages to readable text (user/assistant turns, each capped)
3. **Call LLM:** Use query model for cheap summarization with structured prompt
4. **Store result:** Save structured summary to `state.compaction_summary`

### 2.3 Structured Summary Format (RAG-specific, pi-inspired)

The summary format is adapted from [pi's compaction](https://github.com/earendil-works/pi) to capture RAG-specific context:

```markdown
## Goal
[What is the user trying to accomplish? What topic(s) is the conversation about?]

## Topics Covered
- [Brief description of each topic/area that has been discussed]
- [Include what documents or knowledge bases were consulted]

## Key Decisions & Findings
- [Important conclusions, answers, or decisions made]
- [Specific facts, numbers, or details that were established]

## Retrieved Documents
- [List of document sources or knowledge bases consulted]
- [Key topics each document covered, if relevant]

## Progress
### Completed
- [What has been fully answered or resolved]

### In Progress
- [What is still being worked on or needs more information]

## Critical Context
- [Specific file paths, function names, error messages, or data that must be preserved]
- [Any constraints or preferences the user has mentioned]

## Next Steps
1. [What the user is likely to ask next or what work remains]
2. [Any open questions or incomplete topics]
```

### 2.4 Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `COMPACTION_ENABLED` | `true` | Enable/disable compaction |
| `COMPACTION_HISTORY_THRESHOLD` | `20` | Messages after which compaction triggers |
| `COMPACTION_KEEP_RECENT` | `10` | Recent messages to keep after compaction |
| `COMPACTION_SUMMARY_MAX_CHARS` | `2000` | Max chars for the summary |
| `COMPACTION_ASSISTANT_MAX_CHARS` | `300` | Max chars per assistant response in generation |

---

## 3. RAG vs Chat-Only Routing

### 3.1 Decision Node

A lightweight classifier determines what each query needs:

```python
class QueryDecision(TypedDict):
    needs_retrieval: bool        # Does this need document search?
    needs_file_content: bool     # Does this need attached file content?
    needs_file_metadata: bool    # Does this need file names/descriptions?
```

### 3.2 Classifier Prompt

```
You are a query router for a RAG system. Decide what this query needs.

Query: {rewritten_query}

Respond with ONLY a JSON object with keys:
- needs_retrieval: true if the query asks about facts, definitions, comparisons, or information that should be in documents
- needs_file_content: true if the query asks about the content of an attached file
- needs_file_metadata: true if the query asks about file names or descriptions

Rules:
- "needs_retrieval=true" for: factual questions, comparisons, definitions, analysis
- "needs_retrieval=false" for: summarizing conversation, "what did I say", "explain what you mentioned"
- "needs_file_content=true" when query references attached file content
- "needs_file_metadata=true" when query references file names/descriptions
- When in doubt, set needs_retrieval=true (better to retrieve unnecessarily than hallucinate)
```

### 3.3 Prompt Assembly by Routing Decision

**RAG path (needs_retrieval=true):**
```
system: [session-aware prompt + document context]
user: <conversation_checkpoint>\n[compaction summary if exists]
user: [recent user messages]
user: [recent assistant responses, 300 chars]
user: [current query]
```

**Chat-only path (needs_retrieval=false, no file):**
```
system: [session-aware prompt — chat-only mode]
user: <conversation_checkpoint>\n[compaction summary if exists]
user: [recent user messages]
user: [current query]
```

**Chat-only with file content (needs_retrieval=false, needs_file_content=true):**
```
system: [session-aware prompt — chat-only mode]
user: <conversation_checkpoint>\n[compaction summary if exists]
user: [file content]
user: [recent user messages]
user: [current query]
```

**Chat-only with file metadata (needs_retrieval=false, needs_file_metadata=true):**
```
system: [session-aware prompt — chat-only mode]
user: <conversation_checkpoint>\n[compaction summary if exists]
user: [file names and descriptions]
user: [recent user messages]
user: [current query]
```

---

## 4. File Attachment Handling

### 4.1 Current Behavior

Files are uploaded → converted to markdown → stored in `ChatFile.markdown_content`. On each query:
1. Current file content is prepended to the query text
2. All prior file content from the same chat is also prepended
3. `file_markdown` (current file only) is passed to nodes and included in `format_context_string()`

**Problem:** Files are injected into the query but never stored in conversation messages. The chat-only path has no way to know what files were attached.

### 4.2 New Behavior

Files participate in conversation flow:

1. **On upload:** Store file content as a special message in the conversation:
   ```
   {
     "role": "system",
     "content": "[File Attached: filename.ext]\n\n{markdown_content}"
   }
   ```

2. **On each query:** The decision node checks if the query references files:
   - `needs_file_content=true`: Include file content in prompt
   - `needs_file_metadata=true`: Include file names/descriptions in prompt
   - `needs_retrieval=true`: Include file content in retrieval context (unchanged)

3. **During compaction:** File-related discussion is summarized into the checkpoint. The checkpoint preserves:
   - Which files were attached
   - What topics were discussed about each file
   - Key findings extracted from files

### 4.3 File Content in Compaction

When compaction runs, it summarizes file-related conversation:

```markdown
## Retrieved Documents
- [Knowledge base documents consulted]
- Attached file: document.pdf (discussed: section 3, performance metrics)
- Attached file: architecture.md (discussed: microservice design)
```

---

## 5. Subtask Execution: Parallel and Sequential

### 5.1 Classification Output

The classifier extracts subtasks and their dependency graph:

```python
class QueryClassification:
    type: QueryType
    questions: List[str]          # subtasks
    subtask_dependencies: List[List[int]]  # dependencies[i] = [indices of subtasks i depends on]
    clarification_needed: str
    needs_retrieval: bool
    needs_file_content: bool
    needs_file_metadata: bool
```

### 5.2 Example

**Query:** "What's the difference between semaphores and mutexes? How does that compare to condition variables?"

**Output:**
```python
questions = [
    "What's the difference between semaphores and mutexes?",
    "How does that compare to condition variables?"
]

subtask_dependencies = [
    [],           # subtask 0: no dependencies
    [0]           # subtask 1: depends on subtask 0 ("that" = semaphores/mutexes)
]
```

### 5.3 Routing Logic

```python
def route_by_dependencies(state: AgentState) -> list[Send] | str:
    subtasks = state.get("subtasks", [])
    dependencies = state.get("subtask_dependencies", [[] for _ in subtasks])
    
    # Check if any subtask has dependencies
    has_dependencies = any(len(deps) > 0 for deps in dependencies)
    
    if not has_dependencies:
        # All independent — run in parallel via Send()
        return [Send("agent_subgraph", {...}) for _ in subtasks]
    else:
        # Has dependencies — run sequentially
        return "sequential_subtask_loop"
```

### 5.4 Parallel Independent Subtasks (Send)

```python
# Each independent subtask runs in parallel with its own context
for i, subtask in enumerate(independent_subtasks):
    sends.append(Send("agent_subgraph", {
        "original_query": subtask,
        "rewritten_query": subtask,
        "messages": state.get("messages", []),
        "compaction_summary": state.get("compaction_summary"),
        "subtasks": [subtask],
        "is_complex": False,
        "current_subtask_index": 0,
    }))
```

### 5.5 Sequential Dependent Subtasks (Loop)

```
current_index: 0 → agent_subgraph(enriched_query) → increment_index → check_done →
  if not done: loop back to agent_subgraph with enriched query
  if done: return all subtask contexts to main graph
```

**Enrichment:** Each dependent subtask's query is enriched with context from its dependencies:

```python
def enrich_query(subtask: str, dependencies: List[int], prior_contexts: Dict[int, str]) -> str:
    query = subtask
    for dep_idx in dependencies:
        if dep_idx in prior_contexts:
            query += f"\n\n[Reference to previous subtask {dep_idx}:\n{prior_contexts[dep_idx]}]"
    return query
```

### 5.6 Convergence

Both paths converge at `collect_context`:
- Parallel paths: LangGraph merges results from all `agent_subgraph` invocations
- Sequential path: The loop collects results and returns them to the main graph

---

## 6. Graph Structure

### 6.1 Main Graph

```
START → rewrite_query → compaction → classify_query → route_by_dependencies →
  ├─ Send(agent_subgraph, ...) → collect_context → prepare_final_context → generating → answer_evaluation → finalize_answer → save_memory → END
  └─ sequential_subtask_loop → collect_context → prepare_final_context → generating → answer_evaluation → finalize_answer → save_memory → END
```

### 6.2 State Fields

New fields in `AgentState`:

```python
# Compaction
compaction_summary: Optional[str] = None
compaction_triggered: bool = False

# Routing
needs_retrieval: bool = True
needs_file_content: bool = False
needs_file_metadata: bool = False

# Subtask dependencies
subtask_dependencies: List[List[int]] = []
current_subtask_index: int = 0  # For sequential loop
subtask_contexts: List[dict] = []  # Accumulated contexts from all subtasks
```

### 6.3 Edge Map

```
START → rewrite_query
rewrite_query → compaction
compaction → classify_query
classify_query → route_by_dependencies (conditional)
route_by_dependencies → agent_subgraph (via Send, for parallel)
route_by_dependencies → sequential_subtask_loop (for sequential)
agent_subgraph → collect_context
sequential_subtask_loop → collect_context
collect_context → prepare_final_context
prepare_final_context → generating
generating → answer_evaluation (conditional: chart_validation if chart query)
chart_validation → answer_evaluation (conditional: generating if retry)
answer_evaluation → finalize_answer
finalize_answer → save_memory
save_memory → END
```

---

## 7. Configuration

### 7.1 New Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `COMPACTION_ENABLED` | `true` | Enable/disable compaction |
| `COMPACTION_HISTORY_THRESHOLD` | `20` | Messages after which compaction triggers |
| `COMPACTION_KEEP_RECENT` | `10` | Recent messages to keep after compaction |
| `COMPACTION_SUMMARY_MAX_CHARS` | `2000` | Max chars for the summary |
| `COMPACTION_ASSISTANT_MAX_CHARS` | `300` | Max chars per assistant response in generation |

### 7.2 Existing Settings (unchanged)

- `OPENAI_MODEL` — main generation model
- `QUERY_MODEL` — query rewriting and summarization model
- `RETRIEVAL_*` — retrieval configuration
- `RERANKER_*` — reranker configuration

---

## 8. Implementation Order

### Phase 1: Conversation Fluency
1. Update `_ANSWER_SYSTEM_PROMPT` with session-aware framing
2. Update `_build_generation_messages` to include recent assistant responses (capped)
3. Add `compaction_summary` field to `AgentState`

### Phase 2: Compaction
4. Add compaction config settings to `config.py`
5. Implement `compaction_node` with structured summary format
6. Wire compaction into graph (after rewrite_query, before classify_query)
7. Update `_build_generation_messages` to include compaction summary in prompt

### Phase 3: Routing
8. Extend `QueryClassification` with routing fields (`needs_retrieval`, `needs_file_content`, `needs_file_metadata`)
9. Implement routing classifier in `classify_query_node`
10. Implement `route_by_dependencies` routing function
11. Update `generating_node` to assemble prompt based on routing decision

### Phase 4: File Handling
12. Store file attachments as conversation messages (system role with file content)
13. Update decision node to detect file references
14. Update prompt assembly to include file content/metadata based on routing decision
15. Update compaction to summarize file-related conversation

### Phase 5: Subtask Dependencies
16. Extend `QueryClassification` with `subtask_dependencies`
17. Update classifier prompt to output dependency graph
18. Implement `route_by_dependencies` to handle both parallel and sequential
19. Implement `sequential_subtask_loop` with query enrichment
20. Wire both paths to converge at `collect_context`

---

## 9. Reference: pi's Approach

The [pi coding agent](https://github.com/earendil-works/pi) handles conversation context through:

1. **Session files** — JSONL format, each line is a session entry (message, compaction summary, branch summary)
2. **Compaction** — When context nears the model's window, an LLM summarizes older messages into a structured checkpoint
3. **Branch summarization** — When navigating conversation branches, summaries preserve abandoned context
4. **Structured summaries** — Preserve goals, decisions, file operations, progress, and next steps

Our approach adapts this for RAG:
- Use LangGraph's Redis checkpointer instead of file-based sessions
- Adapt the summary format for RAG-specific context (knowledge bases, retrieved documents, file attachments)
- Integrate compaction into the graph pipeline rather than post-hoc
- Add RAG-specific routing (retrieval vs chat-only) on top of the conversation management

---

## 10. Testing Strategy

### Unit Tests
- `compaction_node`: Verify summary format, threshold logic, truncation
- `classify_query_node`: Verify routing decisions for various query types
- `route_by_dependencies`: Verify parallel vs sequential routing
- `enrich_query`: Verify dependency context injection
- `_build_generation_messages`: Verify prompt assembly for all routing paths

### Integration Tests
- Single-turn query with retrieval
- Multi-turn query with compaction trigger
- Chat-only query (no retrieval)
- Query with file attachment
- Multi-subtask query (independent)
- Multi-subtask query (dependent)
- Multi-subtask query (mixed independent + dependent)

### Edge Cases
- Compaction fails (LLM unavailable) — should continue without summary
- Empty subtasks — should route to single subtask path
- Circular dependencies — should detect and error
- Very long file content — should truncate in prompt
- Compaction summary too long — should truncate with marker
