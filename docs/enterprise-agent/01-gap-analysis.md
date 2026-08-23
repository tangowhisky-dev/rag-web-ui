# 01 — Gap Analysis

Evidence-based assessment of the current codebase against the five required capabilities. Every claim cites a file and line range. "Works" means production-usable today; "Gap" means missing or insufficient.

---

## Requirement 1: Handle all user queries related to accessible data store / KB

### What works

- **3-leg hybrid retrieval** with native Qdrant MMR diversity and recency-aware dedup (exact content_hash + semantic): dense (Qdrant cosine + MMR), sparse (SPLADE via Qdrant + MMR), exact (MySQL FULLTEXT). `backend/app/services/retrieval/retrieval.py`.
- **Cross-encoder reranking** with configurable threshold. `backend/app/services/retrieval/reranker.py`.
- **Neo4j graph expansion** (2-hop traversal) and entity enrichment. `backend/app/services/graph/graph_service.py`; called from `agentic_rag/nodes.py:449-497`.
- **Confidence scoring** (4-signal: coverage, cross-leg agreement, volume, diversity). `backend/app/services/retrieval/confidence.py`.
- **Adaptive retrieval**: sufficiency check + lower-threshold re-filter. `agentic_rag/nodes.py:577-641`.
- **Multi-KB scoping**: `kb_ids` and `datastore_ids` threaded through every retrieval call. `agentic_rag/graph_state.py` + `graph.py:73-81`.

### Gaps

1. **No iterative retrieval.** The pipeline retrieves once per subtask. If the first retrieval misses the answer, the only recovery is `adaptive_reranking` (lowers the score threshold on the *same* candidate set — it does not issue a new query). A real agent needs to reformulate and re-retrieve. `agentic_rag/nodes.py:609-641`.
2. **No tool to query structured data.** KBs containing CSV/Excel/JSON are chunked as text; the agent cannot run a filter/aggregation against the raw table. `ingestion/document_processor.py:114-118` chunks everything via `RecursiveCharacterTextSplitter`.
3. **Subtask decomposition is one-shot.** `classify_query_node` (`nodes.py:278-394`) emits the subtask list once at the start. The agent cannot add subtasks mid-execution after seeing retrieval results.
4. **OrgLLMConfig is stored but not consumed.** Per-org `api_base`/`model_name`/`query_model` exist in `models/org_llm_config.py` and an upsert endpoint exists, but services read from `settings` instead. `services/chat/chat_service.py:36-59` has `get_effective_llm_config()` but the agentic pipeline does not call it. Multi-tenant LLM routing is broken in practice.

---

## Requirement 2: Multi-turn conversation, intent understanding (RAG vs file action vs previous-answer action)

### What works

- **Query rewriting** resolves pronouns/references against recent history. `agentic_rag/nodes.py:204-231`.
- **Compaction** summarizes old turns when history exceeds a threshold. `nodes.py:87-158`.
- **Routing flags** `needs_retrieval`, `needs_file_content`, `needs_file_metadata` exist in the classifier schema. `agentic_rag/schemas.py:10-25`.
- **File markdown injection**: attached files are converted via markitdown and passed as `file_markdown`. `api/api_v1/chat.py:461-481`; injected in `nodes.py:701-710`.

### Gaps

1. **Intent classification is a one-shot gate, not a planner.** `classify_query_node` runs once, emits subtasks + flags, and the graph commits to that plan. It cannot revise intent after observing a tool result. "Make it a pie chart" (reference to previous answer + chart intent) is detected by a regex on the *original query string* (`nodes.py:937`), not by understanding the conversational referent.
2. **No structured representation of the previous answer.** "Summarise it in 10 points" relies on the LLM seeing the prior assistant message in the context window, capped at `COMPACTION_ASSISTANT_MAX_CHARS` (`nodes.py:718-732`). If the prior answer was long, it is truncated and the summarization is lossy. There is no extracted "key points / data / citations" object the agent can operate on.
3. **No "act on previous answer" tool.** The agent cannot invoke "summarize the last answer", "extract statistics from the last answer", "chart the data in the last answer" as discrete operations. Everything is implicit, delegated to the generation LLM's context window.
4. **File actions are limited to summarization.** The pipeline injects `file_markdown` into the generation prompt; the LLM is expected to summarize/answer from it. There is no tool to extract tables, query structured regions, or transform the file. `nodes.py:701-710`.
5. **Chart detection is a regex.** `re.search(r"\b(chart|graph|plot|visuali[zs]|trend|distribution)\b", original_query.lower())` at `nodes.py:937`. "Make it a pie chart" matches `chart`, but "visualize this" matches `visuali`, while "draw the breakdown" matches nothing. Fragile.
6. **Clarification is graph-level interrupt only.** `request_clarification_node` (`nodes.py:401-428`) fires when `question_is_clear=False` from the classifier. The agent cannot ask for clarification mid-execution (e.g., "which file do you mean?" when two are attached).

---

## Requirement 3: Complex queries → multiple subtasks → complete all

### What works

- **Subtask decomposition** with `subtask_dependencies` (list of lists). `schemas.py:52-55`.
- **Parallel fan-out** via `Send()` for independent subtasks. `graph.py:84-176`.
- **Sequential loop** for dependent subtasks with context enrichment. `graph.py:244-261`.
- **Circular dependency detection** with parallel fallback. `graph.py:179-199`.
- **Reinforced scoring**: chunks retrieved by multiple sub-queries accumulate score. `search-implementation.md` documents this.

### Gaps

1. **Subtasks are retrieval-only.** Each subtask routes to `agent_subgraph` (retrieval), `chat_subgraph` (no retrieval), or `file_context_subgraph` (file content). There is no subtask type for "compute", "chart", "transform table", "summarize previous answer". `graph.py:154-158`.
2. **No subtask can depend on the *output* of another subtask's tool call.** Dependencies are declared upfront by index. If subtask 2 needs "the statistics from subtask 1's answer", that is not expressible — subtask 1 returns retrieved *contexts*, not computed *results*. `schemas.py:52-55`.
3. **No re-planning.** If a subtask's retrieval returns nothing relevant, the pipeline proceeds to `prepare_final_context` with a gap; it does not spawn a new subtask with a reformulated query.
4. **Single final generation.** All subtask contexts are merged and one generation pass produces the answer. There is no per-subtask answer that can itself be a tool input to a later subtask. `nodes.py:1302-1399` → `nodes.py:649-948`.

---

## Requirement 4: Summarize attached documents or previous answers; follow user instructions

### What works

- **File upload + markdown conversion** via markitdown (PDF, DOCX, XLSX, images+OCR, etc.). `services/ingestion/document_converter.py`.
- **Chat-scoped files** with async conversion and status polling. `api/api_v1/chat_files.py`; frontend `components/chat/file-attachment.tsx`.
- **Token budgeting**: file content capped at 25% of `OPENAI_MODEL_CONTEXT_SIZE`. `chat.py:461-481`.

### Gaps

1. **Summarization is a prompt-level instruction, not a tool.** "Summarise this file" sets `needs_file_content=True`, injects the markdown, and hopes the generation LLM summarizes. For a 200-page document this overflows the 25% budget and silently truncates. There is no chunked/map-reduce summarization tool.
2. **No file-section tool.** The agent cannot ask for "section 3" or "the table on page 12" of an attached file. The whole markdown is one blob.
3. **No previous-answer summarization tool.** Same as Req 2 gap 2/3 — there is no `summarize_last_answer` tool; it is implicit context-window work.
4. **Instruction following is unbounded.** The generation prompt (`prompts.py:91` `ANSWER_SYSTEM_PROMPT_BASE`) instructs the model to follow user instructions, but there is no enforcement that the instruction was actually satisfied. `answer_evaluation_node` (`nodes.py:1447-1519`) scores faithfulness/completeness but does not check "did the user ask for 10 points and did we deliver 10?"

---

## Requirement 5: Automatic context-window management and compaction

### What works

- **Compaction node** triggers when `len(messages) > COMPACTION_HISTORY_THRESHOLD`. `nodes.py:87-158`.
- **Structured compaction summary** with sections (Goal, Topics, Decisions, Retrieved Docs, Progress, Critical Context, Next Steps). `prompts.py:13-68`.
- **Redis checkpointer** persists full state per thread. `agentic_rag/redis_memory.py`.
- **Long-term semantic store** for cross-thread recall. `redis_memory.py:90-174`.

### Gaps

1. **Compaction triggers on message count, not token count.** A turn with a 50k-token file attachment counts as one message. The pipeline can blow the context window long before the count threshold fires. `nodes.py:87-158` uses `len(messages)`.
2. **No token-accurate budgeting.** Token estimates are character-based heuristics. `agentic_rag/utils.py` token estimation. With local models whose tokenizers differ from the heuristic, budgets drift.
3. **No sliding window with importance.** Compaction summarizes everything older than `COMPACTION_KEEP_RECENT` uniformly. A critical earlier answer and a throwaway earlier turn get the same treatment. No importance scoring, no selective retention.
4. **No "last answer" fast path.** The most recent assistant answer — the single most likely referent for "summarise it" — is not stored as a first-class object. It is reconstructed from the message list and truncated.
5. **Long-term memory is not recalled into context.** `search_memory` exists (`redis_memory.py`) but is not called in the current pipeline (the `load_subtask_memory_node` is a no-op, `nodes.py:1080-1087`). Cross-thread recall is wired but unused.
6. **No compaction of tool results.** When the agent loop is added, tool outputs (retrieved docs, code results) will accumulate fast. The current compaction only handles user/assistant messages.

---

## Requirement 6 (implicit): Chart generation via echarts

### What works

- **ECharts rendering** in the frontend. `components/chat/echarts-diagram.tsx`; rendered from ````echarts` JSON code blocks in `answer.tsx:32-35`.
- **Chart validation** checks JSON structure and required keys. `nodes.py:1526-1563`.
- **Chart retry loop**: invalid chart → re-generate (max 3). `graph.py:216-226`.
- **ECharts documentation** injected into the generation prompt. `services/prompts/loader.py` `append_chart_instructions`.

### Gaps

1. **Charts are LLM-generated JSON, not data-driven.** The generation LLM emits an ECharts option object from retrieved text. For "give me key statistics" → "make it a pie chart", the LLM must (a) extract numbers from text, (b) construct a valid ECharts series, (c) get the JSON right. Each step is failure-prone; the 3-retry loop papers over it.
2. **No chart-from-data tool.** The agent cannot say "I have this DataFrame, render it as a pie chart" and have a deterministic builder produce the ECharts JSON. The `code_execute` + `chart_generate` tool pair (planned in v2 doc, unimplemented) would fix this.
3. **No chart editing.** "Make it a bar chart instead" requires regenerating the whole answer. There is no chart-option mutation tool that takes the prior chart JSON and swaps the series type.
4. **Validation is structural only.** `validate_echarts_json` checks `series` and `xAxis`/`yAxis` presence, not data sanity (empty series, NaN values, type mismatch). `nodes.py:1526-1563`.

---

## Cross-cutting gaps

1. **`tool_registry.py` and `builtin_tools.py` exist but are not wired into the graph.** `services/tool_registry.py`, `services/builtin_tools.py` define `search_documents` and `extract_entities` tools, but no node calls them. `TOOL_CALLING_ENABLED` and `MAX_TOOL_ITERATIONS` env vars exist but have no effect in the current pipeline. `config.py`.
2. **`db_query_tool.py` and `graph_query_tool.py` exist but are unused.** Found by the exploration subagent in `agentic_rag/tools/`. Dead code waiting for the loop.
3. **`user_profile.py` model exists but is not integrated.** `models/user_profile.py`. No prompt injection, no routing use.
4. **`REASONING_MODEL` env var is reserved but unused.** `config.py`. The pipeline uses `OPENAI_MODEL` for everything including reasoning-heavy steps.
5. **SSE protocol is v3.** Adding tool calls, plan updates, and structured observations requires a v4 extension (new event types). `agentic_rag/streaming.py:20-44`.
6. **No tool-calling fallback for offline LLMs.** Any agent loop will use `ChatOpenAI.bind_tools()`, which assumes the OpenAI function-calling format. LM Studio supports it for some models, Ollama has its own format, vLLM support is spotty, and many local models ignore the `tools` parameter or emit malformed calls. An offline-first plan that assumes reliable native function-calling is fragile. A constrained-JSON-text fallback (LLM emits `{"tool": "...", "arguments": {...}}` parsed by us) is required.
7. **No unified agent guardrail prompt.** The current prompts (`prompts.py`) are task-specific (compaction, classify, answer). There is no top-level behavior contract enforcing: offline-only (no claiming web results), cite-or-refuse when the KB has nothing, agent bounds (what it may and may not do), and when to use a tool vs. answer directly. An autonomous agent without this will hallucinate answers instead of calling tools.
8. **No per-tool RBAC re-check or audit log.** `kb_ids`, `datastore_ids`, and `org_id` are threaded through `AgentState` but never re-validated inside the retrieval/file/code paths. The pipeline trusts that the caller scoped them correctly. For an "enterprise autonomous agent," each tool must re-check entitlements against the authenticated user before executing (the planner LLM could be prompt-injected into passing another org's ids), and every tool call must be written to an audit table (tool name, args, result summary, tokens, latency, success/failure).
9. **No structured extraction from fresh retrieval results.** An extraction tool that targets only the previous answer (the earlier `extract_answer_data` concept) is too narrow. "Give me key statistics in these findings" after a RAG turn needs structured extraction from *fresh retrieved chunks*, not just the previous answer. Without this, the agent cannot feed retrieved text into `chart_generate` or `code_execute` reliably — it would have to ask the LLM to emit JSON from prose, which is the failure mode the deterministic chart builder was meant to eliminate. The generalized `extract_data` tool (source: last_answer | retrieved_docs | file) fixes this.
10. **No explicit tool-failure replanning.** When retrieval returns nothing, `file_read` overshoots the token budget, or `chart_generate` produces invalid output, the current pipeline proceeds to finalize with a gap. The reflect step needs concrete recovery rules (empty retrieval → rewrite → re-retrieve; file over budget → `file_summarize`; chart invalid → builder fixes, not LLM retry), not just "the agent decides."



---

## Summary table

| Requirement | Coverage | Primary gap |
|---|---|---|
| 1. KB-grounded Q&A | Strong | No iterative retrieval; no structured-data tool; no extraction from fresh retrieval results; OrgLLMConfig not consumed |
| 2. Multi-turn intent | Partial | One-shot classifier; no structured previous-answer object; no act-on-previous-answer tool; regex chart detection |
| 3. Complex multi-subtask | Partial | Subtasks are retrieval-only; no output-dependencies; no re-planning; no parallel dispatch of independent subtasks |
| 4. Summarize files/answers | Partial | Summarization is prompt-level, not a tool; no chunked summarization; no instruction-following verification |
| 5. Context management | Partial | Count-based compaction; no token budgeting; no importance scoring; long-term recall unused |
| 6. Charts | Partial | LLM-generated JSON, not data-driven; no chart-from-data tool |
| Cross-cutting (enterprise) | Missing | No tool-calling fallback for offline LLMs; no unified guardrail prompt; no per-tool RBAC/audit; no tool-failure replanning |

The pattern is consistent: **the infrastructure is built but the agency layer that would orchestrate it is missing.** The plan in `02-target-architecture.md` adds that layer.
