# 03 — Tool Specifications

Contract for every tool in the registry. Each tool is a LangChain `BaseTool` subclass in `backend/app/services/agentic_rag/tools/`. All tools are async (`arun`), emit SSE progress via the shared callback bridge, and return a structured observation dict that the agent loop appends to `observations`.

Conventions:
- **Input**: Pydantic schema, validated before execution.
- **Output**: `{"ok": bool, "result": <tool-specific>, "error": str | None, "tokens": int}`. `tokens` is the observation's token cost (for context budgeting).
- **Progress**: tools emit `p:` events with `phase = tool_name` and human-readable messages.
- **Budget**: every tool checks its per-turn cap via `_tool_call_budget()` and refuses if exceeded.
- **RBAC**: every tool receives a `ToolContext` (db session, user_id, org_id, qdrant client, redis memory, org llm config, agent state ref). Before executing, the tool re-validates entitlements via `enforce_rbac()` (see `tool_context.py`). Denied calls return `ok=False` with an error message.
- **Audit**: every tool call writes a `tool_call_audit` row (chat_id, message_id, iteration, tool_name, arguments, result_summary, tokens_in, tokens_out, latency_ms, status).

---

## 1. `rag_retrieve`

3-leg hybrid retrieval (dense + sparse + exact) with cross-encoder reranking, LLM-based sufficiency checking, internal query rewriting, and optional Neo4j graph expansion.

**Input**
```python
class RagRetrieveInput(BaseModel):
    query: str                       # search query
    kb_ids: list[int] | None = None  # override state kb_ids; default uses state
    top_k: int | None = None         # default RETRIEVAL_TOP_K (20)
    graph_expand: bool = True        # include Neo4j expansion
    min_confidence: float = 0.3      # sufficiency threshold
```

**Output `result`**
```python
{
    "docs": [{"chunk_id","file_name","chunk_text","score","reranker_score","leg","entities"}],
    "confidence": float,
    "confidence_level": str,
    "query_used": str,
    "original_query": str,
    "query_rewritten": bool,
    "sufficient": bool,
    "missing": str,           # description of what's missing if insufficient
    "legs_run": list[str],
    "levels_tried": list[int],
}
```

**Behavior**
1. Resolve `kb_ids` from `AgentState` if not provided. RBAC-filter via `enforce_rbac()`.
2. Run relaxation ladder: dense → sparse → exact → merge → rerank → filter, at progressively looser thresholds.
3. After each level, run `_llm_sufficiency_check()` — evaluates whether retrieved chunks collectively answer the query. Falls back to heuristic (≥3 docs, confidence ≥ min) if LLM unavailable.
4. If insufficient and `missing` is non-empty, rewrite the query via `_rewrite_query()` and run a second ladder pass.
5. If still insufficient and `graph_expand`, run Neo4j expansion and re-check sufficiency.
6. Return metadata including `sufficient`, `missing`, `query_rewritten`, `query_used`.

**Cap**: `AGENT_MAX_RETRIEVALS` (default 3) per turn.

**Source**: `tools/rag_retrieve.py`. Reuses `services/retrieval/retrieval.py`, `services/retrieval/reranker.py`, `services/graph/graph_service.py`.

---

## 2. `kb_grep`

Regex/keyword search across all authorized KB documents' converted markdown. Last-resort tool for finding exact terms that vector search missed.

**Input**
```python
class KbGrepInput(BaseModel):
    pattern: str                           # search term or regex
    kb_ids: list[int] | None = None        # default all authorized KBs
    document_ids: list[int] | None = None  # restrict to specific documents
    max_results: int = 50                  # max matching lines (1-200)
    case_insensitive: bool = True
```

**Output `result`**
```python
{
    "matches": [{"document_id","title","file_name","line_number","line_text"}],
    "total_matches": int,
    "documents_searched": int,
    "pattern": str,
}
```

**Behavior**
1. RBAC-filter `kb_ids` via `enforce_rbac()`. Resolve datastore IDs via `get_effective_datastore_ids()`.
2. Query `Document` rows in authorized KBs/datastores, optionally filtered by `document_ids`.
3. For each document, search `converted_markdown` line-by-line with `re.search()`.
4. Collect matches (line text truncated to 200 chars), up to `max_results`.

**Cap**: `AGENT_MAX_KB_GREP` (default 5) per turn.

**Source**: `tools/kb_grep.py`.

---

## 3. `kb_outline`

Returns the heading structure (table of contents) of a KB document. Pure regex parse of `converted_markdown` — no LLM call.

**Input**
```python
class KbOutlineInput(BaseModel):
    document_id: int
```

**Output `result`**
```python
{
    "document_id": int,
    "title": str,
    "file_name": str,
    "headings": [{"level": int, "text": str, "char_offset": int}],
    "total_chars": int,
}
```

**Behavior**
1. Load document and verify RBAC via `_load_authorized_document()` (shared with `kb_read`).
2. Parse `converted_markdown` with `re.finditer(r"^(#{1,6})\s+(.+)$", markdown, re.MULTILINE)`.
3. Return heading level, text, and character offset for each heading.

**Cap**: `AGENT_MAX_KB_READ` (default 10, shared with `kb_read`) per turn.

**Source**: `tools/kb_outline.py`.

---

## 4. `kb_read`

Reads a specific section (by heading name) or character range of a KB document's converted markdown. Last-resort tool for reading content that chunk retrieval missed.

**Input**
```python
class KbReadInput(BaseModel):
    document_id: int
    section: str | None = None       # heading text; reads until next heading of same/higher level
    start_char: int | None = None    # character offset (from kb_outline or kb_grep)
    end_char: int | None = None
    max_tokens: int = 4000           # token budget (500-16000)
```

**Output `result`**
```python
{
    "document_id": int,
    "title": str,
    "file_name": str,
    "section": str | None,
    "content": str,
    "total_tokens": int,
    "truncated": bool,
    "char_range": [int, int],
}
```

**Behavior**
1. Load document and verify RBAC via `_load_authorized_document()`.
2. If `section` given, find heading by regex, extract content until next heading of same/higher level.
3. If `start_char`/`end_char` given, slice the markdown.
4. If neither, return full document.
5. Token-truncate to `max_tokens` using `count_tokens()`.
6. If section not found, fall back to full document.

**Cap**: `AGENT_MAX_KB_READ` (default 10, shared with `kb_outline`) per turn.

**Source**: `tools/kb_read.py`.

---

## 5. `file_read`

Reads an attached chat file's markdown. Supports section-level retrieval.

**Input**
```python
class FileReadInput(BaseModel):
    file_id: int | None = None       # specific file; default most recent attached
    section: str | None = None       # heading text, "page:N", or "chunk:I"
    max_tokens: int = 4000           # cap returned content (500-16000)
```

**Output `result`**: `{"file_name","section","content","total_tokens","truncated"}`.

**RBAC**: confirm `file_id` belongs to a `ChatFile` in a chat owned by `ctx.user_id` via `enforce_rbac()`.

**Source**: `tools/file_read.py`.

---

## 6. `file_summarize`

Map-reduce summarization for large attached files.

**Input**
```python
class FileSummarizeInput(BaseModel):
    file_id: int | None = None
    focus: str | None = None         # "key findings", "financials", "risks"
    max_points: int | None = None
    chunk_size: int = 4000           # tokens per map chunk
```

**Output `result`**: `{"summary","key_points":[...],"file_name","chunks_processed"}`.

**Source**: `tools/file_summarize.py`.

---

## 7. `file_extract_table`

Extracts structured tables from CSV/Excel/HTML-in-markdown attached files.

**Input**
```python
class FileExtractTableInput(BaseModel):
    file_id: int | None = None
    table_index: int = 0
    filter: str | None = None
```

**Output `result`**: `{"columns":[...],"rows":[[...],...],"row_count","file_name"}`.

**Source**: `tools/file_extract_table.py`.

---

## 8. `code_execute`

Local Python sandbox for computation, data transform, statistics. Offline only — no network imports.

**Input**
```python
class CodeExecuteInput(BaseModel):
    code: str
    data: dict | None = None         # variables to inject
    timeout_s: int = 10
```

**Output `result`**: `{"stdout","stderr","result","plots":[...],"error"}`.

**Cap**: `AGENT_MAX_CODE_EXEC` (default 3) per turn.

**Source**: `tools/code_execute.py`.

---

## 9. `chart_generate`

Deterministic ECharts option builder. Takes structured data + chart spec, returns validated ECharts JSON.

**Input**
```python
class ChartGenerateInput(BaseModel):
    data: list[dict] | dict
    chart_type: Literal["pie","bar","line","scatter","area"]
    title: str | None = None
    x_field: str | None = None
    y_field: str | None = None
    name_field: str | None = None
    value_field: str | None = None
```

**Output `result`**: `{"echarts_option": dict, "valid": bool, "error": str | None}`.

**Source**: `tools/chart_generate.py`.

---

## 10. `summarize_answer`

Summarizes the `last_answer_object` or a cited prior turn.

**Input**
```python
class SummarizeAnswerInput(BaseModel):
    target: Literal["last","cited","specified"] = "last"
    message_id: int | None = None
    focus: str | None = None
    max_points: int | None = None
    format: Literal["bullets","paragraph"] = "bullets"
```

**Output `result`**: `{"summary","key_points":[...],"format"}`.

**Source**: `tools/summarize_answer.py`.

---

## 11. `extract_data`

Pulls numbers/stats from a text source so they can be fed to `chart_generate` or `code_execute`.

**Input**
```python
class ExtractDataInput(BaseModel):
    source: Literal["last_answer","retrieved_docs","file","specified"] = "last_answer"
    message_id: int | None = None
    file_id: int | None = None
    what: str
```

**Output `result`**: `{"data": list[{"label","value","unit","context"}], "source": str, "source_ref": str}`.

**Source**: `tools/extract_data.py`.

---

## Tool registry wiring

`services/agentic_rag/tools/__init__.py` exports `_TOOL_CLASSES` — the list of all 11 tool classes. `build_tools(ctx)` instantiates them with a `ToolContext`. `applicable_tools(ctx)` filters based on current state:

- File tools (`file_read`, `file_summarize`, `file_extract_table`) only if a file is attached.
- Data tools (`chart_generate`, `extract_data`) only if there is data to chart.
- KB tools (`kb_grep`, `kb_read`, `kb_outline`) only if the chat has KBs linked (`state["kb_ids"]` non-empty).

`think_node` binds the applicable subset via `ChatOpenAI(...).bind_tools()` (native mode) or includes them in the `THINK_SYSTEM_PROMPT` tool list (JSON-text fallback mode). Tool descriptions (LangChain `description` field) are the LLM's selection signal.

Per-turn caps are configured in `_tool_call_budget()` and enforced in `tool_node`. Exceeded calls return a budget_exceeded error without executing.
