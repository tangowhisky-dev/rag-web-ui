# 03 — Tool Specifications

Contract for every tool in the registry. Each tool is a LangChain `BaseTool` subclass in `backend/app/services/agentic_rag/tools/`. All tools are async (`arun`), emit SSE progress via the shared callback bridge, and return a structured observation dict that the agent loop appends to `observations`.

Conventions:
- **Input**: Pydantic schema, validated before execution.
- **Output**: `{"ok": bool, "result": <tool-specific>, "error": str | None, "status": "ok"|"error"|"denied"|"timeout"|"budget_exceeded", "tokens": int}`. `tokens` is the observation's token cost (for context budgeting).
- **Progress**: tools emit `p:` events with `phase = tool_name` and human-readable messages.
- **Streaming**: long-running tools (retrieval, summarization, code) emit `th:` thinking and intermediate `to:` observation summaries.
- **Budget**: every tool checks `AgentState.tool_call_count[tool_name]` against its per-turn cap and refuses with `status="budget_exceeded"` if exceeded.
- **RBAC**: every tool receives a `ToolContext` (db session, user_id, org_id, qdrant client, redis memory, org llm config, agent state ref). Before executing, the tool re-validates entitlements against the authenticated user (see `02` §5.1). Denied calls return `status="denied"` and write an audit row.
- **Audit**: every tool call writes a `tool_call_audit` row (chat_id, message_id, iteration, tool_name, arguments, result_summary, tokens_in, tokens_out, latency_ms, status) — see `02` §5.2.

---

## 1. `rag_retrieve`

Wraps the existing 3-leg hybrid retrieval + reranking + confidence + optional Neo4j graph expansion. The retrieval nodes (`dense_retrieval_node`, `sparse_retrieval_node`, `exact_retrieval_node`, `merge_node`, `neo4j_expansion_node`, `reranking_node`, `filter_node`, `sufficiency_check_node`, `adaptive_reranking_node`) become the internal pipeline of this tool. The `graph_expand` flag controls whether `neo4j_expansion` runs — this absorbs the former separate `graph_retrieve` tool.

**Input**
```python
class RagRetrieveInput(BaseModel):
    query: str                       # search query (already rewritten if needed)
    kb_ids: list[str] | None = None  # override state kb_ids; default uses state
    datastore_ids: list[str] | None = None
    top_k: int | None = None         # default RETRIEVAL_TOP_K
    legs: list[Literal["dense","sparse","exact"]] | None = None  # default all enabled
    graph_expand: bool = True        # include Neo4j expansion
    min_confidence: float = 0.3      # sufficiency threshold
```

**Output `result`**
```python
{
    "docs": [{"chunk_id","file_name","chunk_text","score","reranker_score","leg","entities"}],
    "confidence": float,
    "confidence_level": str,
    "confidence_breakdown": dict,
    "query_used": str,
    "legs_run": list[str],
    "sufficient": bool,
}
```

**Behavior**
1. Resolve `kb_ids`/`datastore_ids` from `AgentState` if not provided.
2. **RBAC**: re-validate every `kb_id`/`datastore_id` against the authenticated user's entitlements via `rbac.py` filters. Drop denied ids; log them.
3. Run enabled legs in parallel (existing functions).
4. Merge via RRF (existing `_merge_docs`).
5. Rerank (existing `rerank`), filter (existing `filter`).
6. If `graph_expand`, run `neo4j_expansion` and append.
7. Score confidence (existing `score_retrieval`).
8. If `confidence < min_confidence` and `adaptive` not yet tried, lower threshold and re-filter (existing `adaptive_reranking` logic).
9. Emit `2: context` SSE event with docs + confidence (existing protocol).
10. Return result. If `docs` empty, `sufficient=False` — agent will see this and may reformulate.

**Cap**: `AGENT_MAX_RETRIEVALS` (default 3) per turn.

**Reuses**: `services/retrieval/retrieval.py`, `services/retrieval/reranker.py`, `services/retrieval/confidence.py`, `services/graph/graph_service.py`. No new retrieval code.

---

## 2. `file_read`

Reads an attached chat file's markdown. Supports section-level retrieval so the agent doesn't load a 200-page document into context.

**Input**
```python
class FileReadInput(BaseModel):
    file_id: str | None = None       # specific file; default most recent attached
    section: str | None = None       # heading text or "page:N" or "chunk:I"
    max_tokens: int = 4000           # cap returned content
```

**Output `result`**: `{"file_name","section","content","total_tokens","truncated"}`.

**Behavior**
1. Load `ChatFile.markdown_content` from DB (existing model).
2. If `section` given, find heading (markdown `#`/`##`) or page marker; slice.
3. Truncate to `max_tokens` using tokenizer; set `truncated=True` if cut.
4. If no `file_id`, use most recent attached file in the chat.

**New code**: section slicer in `tools/file_read.py`. Markdown heading parsing via `markdown-it-py` (already a markitdown transitive dep) or simple regex.

**RBAC**: confirm `file_id` belongs to a `ChatFile` in a chat owned by `ctx.user_id`.

---

## 3. `file_summarize`

Map-reduce summarization for large attached files. Fixes the "25% budget silently truncates" gap.

**Input**
```python
class FileSummarizeInput(BaseModel):
    file_id: str | None = None
    focus: str | None = None         # "key findings", "financials", "risks" — guides the reduce prompt
    max_points: int | None = None    # target bullet count
    chunk_size: int = 4000           # tokens per map chunk
```

**Output `result`**: `{"summary","key_points":[...],"file_name","chunks_processed"}`.

**Behavior**
1. Load file markdown.
2. Split into `chunk_size`-token pieces.
3. **Map**: summarize each chunk in parallel (query model, cheap).
4. **Reduce**: combine chunk summaries into final summary + key points, guided by `focus`.
5. Stream progress per chunk (`p:` events).

**New code**: `tools/file_summarize.py`. Uses `QUERY_MODEL` (or per-org equivalent) for map step, `OPENAI_MODEL` for reduce.

**RBAC**: confirm `file_id` belongs to a chat owned by `ctx.user_id`.

---

## 4. `file_extract_table`

Extracts structured tables from CSV/Excel/HTML-in-markdown attached files. Returns a JSON table the agent can pass to `code_execute` or `chart_generate`.

**Input**
```python
class FileExtractTableInput(BaseModel):
    file_id: str | None = None
    table_index: int = 0             # which table if multiple
    filter: str | None = None        # optional row filter expression (evaluated in code_execute)
```

**Output `result`**: `{"columns":[...],"rows":[[...],...],"row_count","file_name"}`.

**Behavior**
1. Detect file type from `ChatFile.content_type`.
2. CSV/Excel: parse with `pandas` (`openpyxl` for xlsx).
3. HTML/markdown: extract `<table>` via `pandas.read_html`.
4. Apply optional filter via `pandas.query` (sanitized).
5. Return JSON-serializable table.

**New code**: `tools/file_extract_table.py`. Adds `pandas`, `openpyxl` to requirements (pandas already transitive via pyarrow).

**RBAC**: confirm `file_id` belongs to a chat owned by `ctx.user_id`.

---

## 5. `code_execute`

Local Python sandbox for computation, data transform, statistics. **Offline only — no network imports.**

**Input**
```python
class CodeExecuteInput(BaseModel):
    code: str
    data: dict | None = None         # variables to inject (e.g., a table from file_extract_table)
    timeout_s: int = 10
```

**Output `result`**: `{"stdout","stderr","result","plots":[...],"error"}`.

**Behavior**
1. Validate code against a denylist: no `import socket`, `subprocess`, `os.system`, `urllib`, `requests`, `http`, `open` (network), `__import__` of banned modules.
2. Inject `data` as local variables. Inject `pd` (pandas), `np` (numpy), `plt` (matplotlib, Agg backend).
3. Execute via `RestrictedPython` (v1) or `nsjail`-wrapped subprocess (hardened). Capture stdout/stderr.
4. If `plt` figures exist, render to PNG bytes, store in `AgentState.artifacts`, return refs.
5. `result` = value of a `result` variable if defined, else last expression.
6. Enforce `timeout_s` (signal/alarm in RestrictedPython, timeout flag in nsjail).

**Sandbox choice**:
- **v1 (recommended for first ship)**: `RestrictedPython`. Pure Python, pip-installable offline, compiles code to a restricted AST. Sufficient for an enterprise internal tool where the threat model is "user mistakes" not "malicious user". Add a subprocess timeout wrapper.
- **Hardened (prod, multi-tenant hostile)**: `nsjail` binary wrapping `python -c`. Needs an nsjail image in Docker Compose. Stronger isolation (filesystem, network, syscall). Recommend adding as a separate `sandbox` service in docker-compose for prod deployments.

**New code**: `tools/code_execute.py` + `tools/sandbox.py` (sandbox abstraction so v1→nsjail swap is config).

**Cap**: `AGENT_MAX_CODE_EXEC` (default 3) per turn.

**RBAC**: no resource check (sandbox is local), but the denylist is enforced regardless of arguments.

---

## 6. `chart_generate`

Deterministic ECharts option builder. Takes data + chart spec, returns validated ECharts JSON. Replaces the LLM-emits-JSON approach.

**Input**
```python
class ChartGenerateInput(BaseModel):
    data: list[dict] | dict          # rows or series
    chart_type: Literal["pie","bar","line","scatter","area"]
    title: str | None = None
    x_field: str | None = None
    y_field: str | None = None
    name_field: str | None = None    # for pie
    value_field: str | None = None   # for pie
```

**Output `result`**: `{"echarts_option": dict, "valid": bool, "error": str | None}`.

**Behavior**
1. Build option from `data` + fields using a fixed template per `chart_type`.
2. Validate via existing `validate_echarts_json` (extended with data-sanity checks: non-empty series, no NaN, type match).
3. Return option. The agent includes it in the final answer as an ````echarts` block.

**New code**: `tools/chart_generate.py` + `tools/echarts_builder.py` (templates per chart type).

**Why deterministic**: the current failure mode is the LLM producing malformed JSON or wrong data. A builder that takes structured data and emits valid ECharts is reliable and testable. The LLM's job is to get the *data* right (via `code_execute` / `extract_data`), not the chart JSON.

**Chart editing** ("make it a bar chart instead"): handled by the agent re-calling `chart_generate` with the same data and a different `chart_type`. No `from_option`/`mutation` parameters — keeping the tool surface minimal. If a future need arises for in-place option mutation, add it then.

---

## 7. `summarize_answer`

Summarizes the `last_answer_object` or a cited prior turn. This is the explicit "summarise it in 10 points" tool.

**Input**
```python
class SummarizeAnswerInput(BaseModel):
    target: Literal["last","cited","specified"] = "last"
    message_id: str | None = None    # for "specified"
    focus: str | None = None
    max_points: int | None = None
    format: Literal["bullets","paragraph"] = "bullets"
```

**Output `result`**: `{"summary","key_points":[...],"format"}`.

**Behavior**
1. Load target: `last_answer_object` from state, or `Message` by id, or recalled turn.
2. If `max_points` given, instruct the LLM to produce exactly that many points.
3. Use query model (cheap) for the summarization.

**New code**: `tools/summarize_answer.py`.

**RBAC**: if `message_id` specified, confirm it belongs to a chat owned by `ctx.user_id`.

---

## 8. `extract_data`

Pulls numbers/stats from a text source so they can be fed to `chart_generate` or `code_execute`. Fixes "give me key statistics" → "make it a pie chart". Generalized to work on the previous answer **or** fresh retrieved docs **or** file content — not only `last_answer_object`.

**Input**
```python
class ExtractDataInput(BaseModel):
    source: Literal["last_answer","retrieved_docs","file","specified"] = "last_answer"
    message_id: str | None = None       # for "specified"
    file_id: str | None = None          # for "file"
    what: str                           # "statistics", "financials", "all numbers", "key metrics"
    # for "retrieved_docs": uses the most recent rag_retrieve observation in state
```

**Output `result`**: `{"data": list[{"label","value","unit","context"}], "source": str, "source_ref": str}`.

**Behavior**
1. Load source text:
   - `last_answer`: from `last_answer_object.data` if populated, else from the last assistant message text.
   - `retrieved_docs`: concatenated chunk text from the most recent `rag_retrieve` observation.
   - `file`: from `ChatFile.markdown_content` (or a section if `file_read` was just called).
   - `specified`: from `Message` by `message_id`.
2. LLM extraction (query model) with structured output: list of `{label, value, unit, context}`. Pydantic-validated; one retry on malformed JSON; rule-based regex fallback (sweep for `<number> <unit?>` near label words) if retry fails.
3. Return structured data — agent passes it to `chart_generate` or `code_execute`.

**New code**: `tools/extract_data.py`.

**RBAC**: if `message_id` or `file_id` specified, confirm ownership.

**Why generalized**: the earlier `extract_answer_data` only targeted the previous answer. But "give me key statistics in these findings" after a RAG turn needs extraction from *fresh retrieved chunks*. One tool, three sources, same extraction logic.

---

## 9. `clarify`

Mid-loop clarification. Wraps LangGraph `interrupt()`.

**Input**
```python
class ClarifyInput(BaseModel):
    question: str
    options: list[str] | None = None
```

**Output `result`**: `{"user_response": str}` (after graph resumes).

**Behavior**
1. Call `interrupt({"question": ..., "options": ...})`.
2. Graph pauses, SSE emits `interrupt` event.
3. Frontend shows clarification dialog (existing component).
4. User responds; graph resumes; tool returns the response as observation.

**Reuses**: existing `ClarificationRequest` model + `clarification-dialog.tsx`.

---

## Tool registry wiring

`services/agentic_rag/tools/__init__.py` exports `ALL_TOOLS = [rag_retrieve, file_read, file_summarize, file_extract_table, code_execute, chart_generate, summarize_answer, extract_data, clarify]`.

`think_node` binds the applicable subset via `ChatOpenAI(...).bind_tools(applicable_tools(state))` (native mode) or includes them in the `THINK_SYSTEM_PROMPT` tool list (JSON-text fallback mode). The tool descriptions (LangChain `description` field) are the LLM's selection signal — written carefully to disambiguate (e.g., `file_read` vs `file_summarize` vs `file_extract_table`).

Tools that are not applicable to a turn (e.g., no file attached → `file_*` tools) are filtered out before binding, so the LLM cannot call them. This is cheaper than relying on the LLM to refuse.

**Pruned tools (not in v1):** `graph_retrieve` (merged into `rag_retrieve` via `graph_expand`), `memory_recall` (proactive recall in `load_context_node` covers it), `table_generate` (LLM-emitted markdown tables cover the example; add back only if unreliable in practice).
