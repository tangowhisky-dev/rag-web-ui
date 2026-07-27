# 04 — Implementation Plan

File- and function-level changes. Ordered by dependency. Backend first, then frontend, then migration/cutover. Each section lists what to add, modify, and delete, with the target file path.

---

## Part A — Backend

### A1. New tool package: `backend/app/services/agentic_rag/tools/`

Currently this directory has `__init__.py`, `db_query_tool.py`, `graph_query_tool.py` (unused). Replace its contents with the tool registry from `03-tool-specifications.md`.

**Add files:**
- `tools/rag_retrieve.py` — wraps existing retrieval nodes (incl. `neo4j_expansion` via `graph_expand` flag) into one tool. Absorbs the former `graph_retrieve`.
- `tools/file_read.py` — section-level file reader.
- `tools/file_summarize.py` — map-reduce summarizer.
- `tools/file_extract_table.py` — CSV/Excel/HTML table extractor.
- `tools/code_execute.py` — sandboxed execution entry.
- `tools/sandbox.py` — sandbox abstraction (RestrictedPython v1; nsjail pluggable for later hardening).
- `tools/chart_generate.py` — ECharts option builder + validator.
- `tools/echarts_builder.py` — per-chart-type templates.
- `tools/summarize_answer.py` — previous-answer summarizer.
- `tools/extract_data.py` — generalized structured-data extractor (source: last_answer | retrieved_docs | file | specified).
- `tools/clarify.py` — interrupt wrapper.
- `tools/tool_context.py` — `ToolContext` dataclass (db session, user_id, org_id, qdrant client, redis memory, org llm config, agent state ref) + `enforce_rbac` helper + `write_audit` helper.
- `tools/__init__.py` — exports `ALL_TOOLS` + `applicable_tools(state)` filter.

**Delete:** `tools/db_query_tool.py`, `tools/graph_query_tool.py` (dead code; superseded by `rag_retrieve`).

**Pruned from earlier draft:** `tools/graph_retrieve.py` (merged into `rag_retrieve`), `tools/memory_recall.py` (proactive recall in `load_context_node` covers it), `tools/table_generate.py` (LLM markdown tables cover the example).

**Each tool** subclasses `langchain_core.tools.BaseTool`, defines `args_schema` (Pydantic), `name`, `description`, and async `_arun`. Tools receive a `ToolContext` via `functools.partial` binding at graph compile time — same injection pattern the current nodes use (`graph.py` `partial(...)`). Every tool calls `enforce_rbac(ctx, ...)` before executing and `write_audit(ctx, ...)` after.

### A2. New graph: `backend/app/services/agentic_rag/agent_graph.py`

Replaces `graph.py` as the compiled graph. Keep `graph.py` during migration (see §C), remove after cutover.

**Nodes:**
- `load_context_node` — loads `last_answer_object`, recalled memory (top 3), attached file metadata, recent turns (sliding window). Replaces `load_subtask_memory_node` + the implicit context loading in `_build_generation_messages`.
- `rewrite_query_node` — **kept as-is** from `nodes.py:204-231`.
- `compaction_node` — **modified**: trigger on token count, not message count (see `05-context-memory.md`). Same file `nodes.py:87-158`.
- `plan_node` — new. Calls LLM with planning prompt → `Plan` structured output. Replaces `classify_query_node`.
- `think_node` — new. Calls the LLM with `AGENT_SYSTEM_PROMPT` + `THINK_SYSTEM_PROMPT` + applicable tools. Parses tool call vs final answer via the tool-call parser (A2b). Supports multiple tool calls in one message for parallel dispatch of independent subtasks.
- `tool_node` — new. Dispatches each called tool (in parallel when multiple arrive), enforces RBAC, writes audit row, writes observation. Streams `tc:`/`to:` events.
- `reflect_node` — new. Runs every `AGENT_REFLECT_EVERY` iterations (default 2) and as the final pre-finalize pass. Applies concrete replanning rules (`02` §7) + instruction satisfaction check.
- `finalize_answer_node` — **modified** from `nodes.py:1406-1412`: also extracts `last_answer_object` via an LLM call (query model).
- `save_memory_node` — **kept** from `nodes.py:1415-1440`: also persists `last_answer_object` to the `Message` row.
- `answer_scoring_node` — **kept** from `answer_evaluation_node` (`nodes.py:1447-1519`), moved post-finalize.

**Edges:**
```
START → load_context → rewrite → compaction → plan → think
think → (one or more tool_calls) tool [parallel dispatch] → think   [loop, increment iteration]
think → (final_answer) reflect_final → finalize → answer_scoring → save_memory → END
think → (iteration >= MAX) reflect_final → finalize → ... → END
every Kth iteration: tool → reflect → think
plan → (needs_clarification) clarify_interrupt → plan   [resume after user]
```

**State changes** (`graph_state.py`):
- Add: `plan: Plan`, `observations: list[Observation]` (annotated with reducer), `tool_call_count: dict[str,int]`, `iteration: int`, `last_answer_object: LastAnswerObject | None`, `artifacts: list[Artifact]` (plots, generated files).
- Remove: `subtasks`, `subtask_dependencies`, `subtask_routing`, `current_subtask_index`, `historical_memory_docs`, `subgraph_history`, `is_complex`, `question_is_clear` (now in `plan`).
- Keep: `messages`, `rewritten_query`, `original_query`, `chat_id`, `user_id`, `kb_ids`, `org_id`, `file_markdown`, `compaction_summary`, `final_answer`, `citations`, `confidence_*`.

### A2b. Tool-call parser: `backend/app/services/agentic_rag/tool_call_parser.py` (new)

Parses the `think_node` LLM response into tool calls or a final-answer signal. Three tiers per `02` §3.2:
1. Native function-calling (`response.tool_calls`) — fast path.
2. JSON-text fallback — extract `{"tool": "...", "arguments": {...}}` or `{"tool_calls": [...]}` or `{"final_answer": "..."}` from `response.content`. Use gateway JSON mode if available; else regex-extract the first JSON block + `json.loads` with one retry.
3. Final-answer default — if neither parses, treat `response.content` as the final answer.

`TOOL_CALL_MODE` env (`auto` | `native` | `json_text`) forces a mode. `auto` tries native then falls back. Chosen mode logged per turn.

### A3. New schemas: `backend/app/services/agentic_rag/schemas.py` (extend)

Add `Plan`, `Subtask`, `LastAnswerObject`, `DataPoint`, `CitationRef`, `Observation`, `Artifact`. Keep existing `QueryAnalysis` during migration, delete after.

### A4. New prompts: `backend/app/services/agentic_rag/prompts.py` (extend)

Add:
- `AGENT_SYSTEM_PROMPT` — the unified guardrail (offline-only, cite-or-refuse, tool-use bias, bounds, instruction-following). Prepended to every `plan`/`think`/`reflect` call. See `02` §3.3.
- `PLAN_SYSTEM_PROMPT` — intent classification + subtask decomposition + tool hint. Inputs: query, rewritten query, last_answer_object summary, file metadata, recalled memory, tool list.
- `THINK_SYSTEM_PROMPT` — ReAct-style: "you have these tools and observations; emit one or more tool calls (JSON format if native tool-calling is unavailable) or a final answer." Includes the JSON-text fallback format instructions.
- `REFLECT_SYSTEM_PROMPT` — "given plan + observations, are we done? Any gaps? Did the last tool fail? Apply these recovery rules: ... Did we satisfy the user's explicit instructions (e.g., 10 points)?"
- `LAST_ANSWER_EXTRACT_PROMPT` — extract `LastAnswerObject` from final answer text. Includes "return valid JSON" instruction.

Keep `COMPACTION_SYSTEM_PROMPT`, `COMPACTION_USER_PROMPT`, `ANSWER_SYSTEM_PROMPT_BASE` (chart instructions still appended). Delete `CLASSIFY_SYSTEM_PROMPT` after migration.

### A5. Token budgeting: `backend/app/services/agentic_rag/token_budget.py` (new)

- `get_tokenizer(model_name)` — returns `tiktoken` for OpenAI-family, `transformers.AutoTokenizer` for local models (cached).
- `count_tokens(text, model)` — accurate count.
- `ContextBudget` — computes `available = CONTEXT_WINDOW - RESERVED_GENERATION - TOOL_BUDGET`, tracks `used` across messages + observations, triggers compaction when `used > available * 0.85`.

Replaces the character-heuristic token estimation in `utils.py`.

### A6. Per-org LLM config wiring: `backend/app/services/agentic_rag/llm_factory.py` (new)

- `get_org_llm(org_id, role)` — reads `OrgLLMConfig` (existing `api_base`/`model_name`/`query_model` columns only), falls back to `settings`.
- `build_chat_llm(org_id, role, **kwargs)` — returns `ChatOpenAI` with the right `base_url`/`model`/`api_key`.
- Every node and tool that creates a `ChatOpenAI` calls this factory instead of reading `settings` directly.

Closes the `OrgLLMConfig` not-consumed gap. **Scope note:** only the existing columns are wired. Adding `reasoning_model`/`vision_model`/`embeddings_model`/`graphrag_model` columns is deferred — those remain env-var-driven, which is sufficient for the stated goals.

### A7. SSE protocol v4: `backend/app/services/agentic_rag/streaming.py` (extend)

Add `pl:`, `tc:`, `to:`, `la:` event emitters. Extend `AgenticRAGTransformer` to map the new graph nodes (`think`, `tool`, `reflect`, `plan`) to `4: agent_step` entries. Keep all existing event types.

### A8. Retrieval tool internal pipeline: `backend/app/services/agentic_rag/tools/rag_retrieve.py`

The existing nodes `dense_retrieval_node` ... `adaptive_reranking_node` are refactored from graph nodes into plain async functions in a new `retrieval_pipeline.py` (or kept in `nodes.py` as functions the tool calls). The tool orchestrates them. No retrieval logic is rewritten — the functions are lifted out of the node wrappers.

### A9. Database migration

New Alembic migration:
- `messages.last_answer_object` — JSON column, nullable.
- `messages.plan` — JSON column, nullable (for debugging/replay).
- `messages.tool_calls` — JSON column, nullable (array of `{tool, args, observation_summary}`).
- `tool_call_audit` table — `id` (uuid), `chat_id`, `message_id`, `iteration` (int), `tool_name` (str), `arguments` (json), `result_summary` (json), `tokens_in` (int), `tokens_out` (int), `latency_ms` (int), `status` (enum: ok/error/denied/timeout/budget_exceeded), `created_at` (timestamp). Index on `(chat_id, created_at)`.

**Not added (deferred):** `org_llm_configs` column expansion (`reasoning_model`/`vision_model`/etc.) — only the existing columns are wired in A6.

### A10. Config additions: `backend/app/core/config.py`

Add:
- `AGENT_MAX_ITERATIONS=8`, `AGENT_MAX_RETRIEVALS=3`, `AGENT_MAX_CODE_EXEC=3`, `AGENT_MAX_REFLECTIONS=2`, `AGENT_REFLECT_EVERY=2`.
- `TOOL_CALL_MODE="auto"` (`auto` | `native` | `json_text`).
- `SANDBOX_BACKEND="restrictedpython"`.
- `SANDBOX_TIMEOUT_S=10`.
- `CONTEXT_RESERVED_GENERATION=4096`, `CONTEXT_TOOL_BUDGET=8192`.
- `TOKENIZER_MODEL` (for token counting; defaults to `OPENAI_MODEL`).

### A11. Requirements: `backend/requirements.txt`

Add: `RestrictedPython>=7.0`, `pandas>=2.0`, `openpyxl>=3.1`, `lxml>=5.0` (for `pandas.read_html`), `tiktoken>=0.7`, `matplotlib>=3.7` (already in eval, add to backend), `numpy>=1.26`. Pin `langgraph>=0.2.50`. (`nsjail` is not a pip dep — it's a later Docker-image addition for hardened prod, not v1.)

### A12. Tests: `backend/tests/`

Add:
- `test_tools_rag_retrieve.py` — wraps existing retrieval tests; verifies tool output schema + RBAC (denied kb_id dropped).
- `test_tools_code_execute.py` — sandbox denylist, timeout, data injection, plot capture.
- `test_tools_chart_generate.py` — each chart type, validation, data-sanity checks.
- `test_tools_file_*.py` — read, summarize (map-reduce), extract_table.
- `test_tools_extract_data.py` — extraction from last_answer / retrieved_docs / file; malformed-JSON retry + regex fallback.
- `test_tool_call_parser.py` — native parsing, JSON-text fallback, final-answer default, malformed JSON retry.
- `test_agent_loop.py` — full loop: plan → think → tool → observe → finalize. Mock LLM with scripted tool calls. Parallel dispatch of multiple tool calls.
- `test_agent_loop_budget.py` — iteration caps, per-tool caps, forced finalize.
- `test_agent_loop_rbac.py` — prompt-injection attempt to pass another org's kb_id is denied by the tool.
- `test_agent_loop_audit.py` — every tool call writes an audit row with correct status.
- `test_token_budget.py` — compaction trigger on tokens, sliding window.
- `test_org_llm_wiring.py` — per-org config overrides take effect (existing columns only).
- `test_last_answer_object.py` — extraction + Pydantic validation + retry + rule-based fallback + persistence + use in next turn.

Run inside container: `docker exec rag-web-ui-backend-1 pytest tests/test_agent_loop.py` (per AGENTS.md §12).

---

## Part B — Frontend

### B1. Agent-turn state: `frontend/src/app/dashboard/chat/[id]/page.tsx` (modify)

Add a `useReducer` for the active turn's agent state: `plan`, `toolCalls[]`, `observations[]`, `lastAnswer`, `iteration`. No new state library for v1 — a reducer in the chat page handles the new SSE events without re-rendering the whole chat tree. Existing `chat-context.tsx` stays for chat/KB list state. If re-renders become a measured problem later, extract to `zustand`.

### B2. SSE handler: `frontend/src/app/dashboard/chat/[id]/page.tsx` (modify)

Add handlers for `pl:`, `tc:`, `to:`, `la:` events → dispatch to the agent-turn reducer. Existing handlers for `content_delta`, `citation`, `agent_step`, `progress_message`, `task_list`, `thinking`, `done`, `error` stay.

### B3. Minimal agent-event rendering: `frontend/src/components/chat/agent-events.tsx` (new, v1)

Renders the new events as plain text/chips above the answer, no fancy panels:
- `pl:` → one-line "Plan: <intent> — N subtasks" + collapsible subtask list (text).
- `tc:` → chip "Calling <tool>…".
- `to:` → chip "<tool>: <result summary>" (e.g., "Searched KB — 12 chunks, confidence high").
- `la:` → stored in reducer; no visible rendering in v1 (used internally for follow-up hints).

A dedicated `plan-panel.tsx` and rich `tool-call-card.tsx` (per-tool-type cards with expandable code/docs/chart previews) are a **later polish phase**, not v1. The behavior does not depend on them.

### B4. Chart rendering: `frontend/src/components/chat/echarts-diagram.tsx` (kept)

Already works for ````echarts` blocks in the final answer. No change for v1 — charts produced by `chart_generate` appear in the final answer's markdown. Inline rendering inside a tool-call card is part of the later polish phase. "Edit chart" = user sends "make it a bar chart" as a follow-up; the agent re-calls `chart_generate` with the same data and a different `chart_type`.

### B5. File-action affordances: `frontend/src/components/chat/file-attachment.tsx` (optional, defer)

Quick-action chips ("Summarize", "Extract tables") are a UX nicety. The agent works without them — the user just types the request. Defer to the polish phase.

### B6. Clarification dialog: `frontend/src/components/chat/clarification-dialog.tsx` (kept)

Already exists for the `interrupt` event. No change needed — the new `clarify` tool emits the same event.

### B7. Dependencies: `frontend/package.json`

No new deps for v1. Everything needed (echarts, mermaid, react-markdown, react-dropzone) is already present. `zustand` is deferred with the polish phase.

---

## Part C — Cutover

The agent loop replaces the rigid pipeline entirely. There is no feature flag or parallel-run period — `pipeline.py` dispatches directly to `run_agent_loop`.

### C1. Cutover strategy

1. Verify the agent loop passes the full test suite (`pytest` and eval harness).
2. Deploy the new pipeline as the sole path. `chat_service.generate_response` calls `run_agent_loop` unconditionally.
3. Delete `graph.py` rigid path, `classify_query_node`, `route_by_dependencies`, subgraphs, `CLASSIFY_SYSTEM_PROMPT`, `QueryAnalysis` schema in a separate cleanup PR.

### C2. Backward compatibility

- Existing chats: `Message` rows without `last_answer_object`/`plan`/`tool_calls` render normally (columns nullable). The first turn under the new loop populates `last_answer_object` going forward.
- SSE: v4 events are additive; a frontend that ignores `pl:`/`tc:`/`to:`/`la:` still works (just no tool-call cards).
- Compaction: token-based trigger is strictly better; no migration of existing summaries needed.

---

## Part D — What is explicitly NOT changed

- Ingestion pipeline (`services/ingestion/`) — chunking, embedding, Qdrant upsert, Neo4j graph build. Unchanged.
- Retrieval primitives (`services/retrieval/retrieval.py`, `reranker.py`, `confidence.py`) — lifted into `rag_retrieve` tool, logic unchanged.
- Auth, RBAC, multi-tenancy (`core/security.py`, `api/rbac.py`) — unchanged.
- Knowledge base / datastore / watcher / recovery services — unchanged.
- Export service — unchanged.
- Frontend chat list, folders, KB management, admin pages — unchanged.
- Docker Compose topology — unchanged (nsjail sandbox is an optional add service, not required for v1).

This keeps the diff focused on the agency layer, matching the "wrap, don't rewrite" principle.
