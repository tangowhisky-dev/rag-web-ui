# Atomic Tools Redesign — Implementation Plan

> **Status**: Implemented. All atomic tools, provenance validation, confidence scoring, and sufficiency checks are live.
> **Branch**: `enterprise-agent`
> **Date**: 2026-09-04
> **Breaking change**: Yes. No compatibility layers, no legacy paths.

---

## 1. Motivation

The current agent has a single composite retrieval tool (`rag_retrieve`) that internally owns query expansion, leg selection (dense/sparse/exact), conditional dense fast-accept, merge/dedup/RRF, cross-encoder reranking, graph expansion, sufficiency checking, relaxation ladder, and query rewrite retry. The agent never sees intermediate results — it gets a final blob of docs and a `sufficient: bool` flag.

This creates three problems:

1. **Strategy rigidity.** The agent cannot choose between searching titles vs chunk content, searching within a specific document vs across all documents, or choosing lexical vs semantic search based on query intent. The composite tool makes these decisions internally with hardcoded heuristics.

2. **No partial reads.** The agent cannot inspect intermediate search results and adjust strategy. If sparse+exact returns 3 high-confidence hits, the agent should skip dense — but today this decision is buried inside `_run_retrieval_pass` (`rag_retrieve.py:562`), not visible to the agent.

3. **Citation model is chunk-only.** `CitationRef(document_id, chunk_index)` cannot represent whole-file reads, section reads, character ranges, grep matches, or graph-expanded chunks. The `[KB-N](N)` labeling assumes all evidence comes from `rag_retrieve` chunks.

The `retrievalagent` codebase (`~/code/retrievalagent`) demonstrates a working atomic pattern: `search_bm25`, `search_hybrid`, `rerank_results`, `get_index_settings`, `get_filter_values` — each a separate tool, the LLM composes them. The Pi agent (`~/code/pi`) contributes execution patterns: `isError` results for retries, `terminate` hint, deferred tool loading, `prepareArguments`, structured compaction.

---

## 2. Architecture

### 2.1 Design principle

The agent decides WHAT to search and HOW MUCH to read. The search tools decide HOW to search (internally: synonym expansion, filter resolution, min-score thresholds). The reranker is a separate tool that accepts hits from any search and dedups/scores them.

### 2.2 Layer diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Agent Loop (LangGraph)                                       │
│                                                               │
│  load_context → plan → think → tool → sufficiency_check      │
│                                        ↓                      │
│  save_memory ← answer_scoring ← finalize                     │
│                                                               │
│  Tools available to the LLM (think node):                     │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ Discovery: kb_metadata, kb_search_documents,          │    │
│  │            kb_outline, current_datetime               │    │
│  │ Search:    search_exact, search_sparse, search_dense  │    │
│  │            kb_grep                                    │    │
│  │ Post-search: rerank_results, graph_expand             │    │
│  │ Read:      kb_read, file_read                         │    │
│  │ Processing: extract_data, chart_generate,             │    │
│  │             code_execute, summarize_answer            │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                               │
│  Internal (not LLM-callable, inside search tools):            │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ Synonym expansion (Redis-cached LLM)                  │    │
│  │ Filter resolution (MySQL metadata → document_ids)     │    │
│  │ Per-leg min-score filtering                           │    │
│  │ Content-hash dedup within a single search             │    │
│  │ Cross-encoder scoring (inside rerank_results)         │    │
│  │ Semantic dedup (inside rerank_results)                │    │
│  │ Threshold/elbow cut (inside rerank_results)           │    │
│  │ Graph traversal (inside graph_expand)                 │    │
│  └──────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 What the agent sees

Each search tool returns hits as a list of dicts:

```python
{
    "ok": True,
    "result": {
        "hits": [
            {
                "document_id": 42,
                "chunk_index": 5,
                "page": 3,
                "title": "Weekly Update 21-28 Aug 2026.pdf",
                "file_name": "Weekly_Update_21-28_Aug_2026.pdf",
                "content": "The weekly update covers 4 topics...",
                "score": 0.87,
                "content_hash": "abc123...",
                "qdrant_point_id": "uuid-...",
                "citation_ref": {
                    "document_id": 42,
                    "citation_kind": "chunk",
                    "chunk_index": 5,
                    "page": 3,
                    "quoted_text": "The weekly update covers 4 topics...",
                    "source_tool": "search_dense",
                    "citation_id": "E1"
                }
            },
            ...
        ],
        "query_used": "weekly update topics",
        "search_type": "dense",
        "count": 8
    },
    "error": None,
    "tokens": 120
}
```

The agent inspects hits, decides whether to rerank, search again with a different tool, read a full document, or proceed to answer.

### 2.4 What gets removed

| Removed | Replaced by |
|---|---|
| `rag_retrieve` tool | `search_exact` + `search_sparse` + `search_dense` + `rerank_results` + `graph_expand` |
| `rag_retrieve`'s `legs` parameter | LLM chooses which search tool(s) to call |
| `rag_retrieve`'s `graph_expand` parameter | `graph_expand` is a separate tool |
| `rag_retrieve`'s `min_confidence` parameter | LLM decides sufficiency in think loop |
| `rag_retrieve`'s internal relaxation ladder | LLM adjusts query/strategy on retry |
| `rag_retrieve`'s internal query rewrite | LLM does this naturally in think loop |
| `rag_retrieve`'s internal sufficiency check | `sufficiency_check` graph node |
| `rag_retrieve`'s conditional dense fast-accept | LLM decides whether to call `search_dense` |
| `expand_query` graph node | Query expansion moves inside search tools |
| `rewrite_query` graph node | LLM does this in think loop |
| `reflect` graph node | Replaced by `sufficiency_check` |
| `reflect_final` graph node | Replaced by `sufficiency_check` |
| `suggested_legs` in `Subtask` schema | Removed |
| `QueryIntent` schema | Removed (LLM chooses search tools directly) |
| `REWRITE_INTENT_SUFFIX` prompt | Removed |
| Correction-LLM retry (`_correct_tool_args`) | Pi-style `isError` return to LLM |
| `dense_docs`, `sparse_docs`, `exact_docs` state keys | Each search tool returns hits as observations |
| `all_scored_docs` state key | Replaced by `evidence` list |
| `graph_docs` state key | `graph_expand` returns hits as observations |
| `retrieval_confidence` state key | LLM assesses confidence, not a computed score |
| `leg_results`, `failed_legs`, `leg_doc_counts` state keys | Removed (no per-leg state) |
| `adaptive_reran`, `graph_expansion_done` state keys | Removed |
| `_tried_rag_retrieve_queries` helper | Replaced by `_tried_search_queries` (generic over all search tools) |
| `_correct_tool_args` function | Replaced by `isError` pattern |
| `_correction_hints` function | Removed |
| `route_tool` function | Removed (tool node has fixed edge to `sufficiency_check`) |
| `route_reflect_final` function | Removed (no `reflect_final` node to route from) |
| `merge_node` in `nodes.py` | Dead (only called from `rag_retrieve.py`); RRF/dedup moves into `rerank_results` |
| `collapse_same_title_versions` in `nodes.py` | Dead (only called from `merge_node`); `kb_search_documents` has its own independent same-title dedup |
| `_elbow_cut`, `filter_node` in `nodes.py` | Dead (only called from `rag_retrieve.py` pipeline); threshold/elbow cut moves into `rerank_results` |
| `dense_retrieval_node`, `sparse_retrieval_node`, `exact_retrieval_node` in `nodes.py` | Dead (only called from `rag_retrieve.py`); logic moves into atomic search tools |
| `reranking_node` in `nodes.py` | Dead (only called from `rag_retrieve.py`); logic moves into `rerank_results` tool |
| `neo4j_expansion_node` in `nodes.py` | Dead (only called from `rag_retrieve.py`); logic moves into `graph_expand` tool |
| `answer_evaluation_node` in `nodes.py` | Kept but updated to use `evidence` instead of `retrieved_docs` for cited-evidence-only scoring |
| `_enrich_with_modified_at` in `nodes.py` | Dead (only called from retrieval legs); file timestamps are read directly by search tools |
| `_retrieval_confidence_level`, `_resolve_eval_kwargs`, `_final_confidence_level` in `nodes.py` | Review: if only called from dead retrieval legs, remove; if called from `answer_evaluation_node`, keep |
| `ADAPTIVE_RETRIEVAL_*` settings | Dead (only used by `rag_retrieve.py`) |
| `SYNONYM_VARIANTS`, `SYNONYM_CACHE_TTL`, `PRE_FUSION_MIN_DOCS` settings | Dead or moved into search tool internals |
| `COLLAPSE_SAME_TITLE_VERSIONS`, `RRF_FUSION_ENABLED`, `MERGE_MMR_LAMBDA` settings | Dead (only used by `merge_node`) |
| `rewritten_query` SSE event (`1:` prefix) | Removed (no `rewrite_query` node to emit it) |
| `expanded_query` SSE event (`eq:` prefix) | Removed (no `expand_query` node to emit it) |
| `Message.rewritten_query` DB column | Kept as nullable; no longer written by the agent pipeline. Historical data remains readable. |
| `Message.expanded_query` DB column | Kept as nullable; no longer written by the agent pipeline. Historical data remains readable. |

### 2.5 What stays the same

| Kept | Why |
|---|---|
| `dense_search_docs`, `sparse_search_docs`, `exact_search_docs` in `retrieval.py` | Low-level implementations, called by the new search tools |
| `dedup_by_content_hash`, `semantic_dedup` in `retrieval.py` | Called by `rerank_results` |
| `rerank()` in `reranker.py` | Called by `rerank_results` tool |
| `expand_docs_via_graph()` in `graph/expand.py` | Called by `graph_expand` tool |
| `ToolContext`, `enforce_rbac`, `write_audit` | Tool-agnostic infrastructure |
| `BaseAgentTool` dispatch pattern | Enhanced with `prepareArguments` and `terminate` |
| Post-generation answer scoring and suggestions | Unchanged |
| Memory/save behavior | Unchanged |
| `accumulated_data` for extract_data → chart_generate | Unchanged |
| `kb_metadata`, `kb_search_documents`, `kb_outline`, `kb_read`, `kb_grep` tools | Existing, enhanced with `CitationRef` |
| `file_read`, `file_summarize`, `file_extract_table` tools | Existing, enhanced with `CitationRef` |
| `code_execute`, `chart_generate`, `extract_data`, `summarize_answer` tools | Existing |
| `current_datetime` tool | Existing |
| Compaction logic | Existing, enhanced to preserve `CitationRef` metadata |
| `_verify_execution`, `_build_execution_summary` | Moved to `execution_check.py`, rewritten for atomic tools, but the verification concept is retained |
| `answer_scoring_node`, `clarify_interrupt_node` | Stay in `reflection.py` (only `reflect_node`/`reflect_final_node` are removed) |
| `answer_evaluation_node` | Stays in `nodes.py`, updated to read `evidence` for cited-evidence-only scoring |
| `_agent_step`, `history_to_text`, `select_recent_history`, `_messages_to_conversation_text`, `_get_llm` | Stay in `nodes.py` (shared helpers used by think/tool/finalize/compaction) |
| `_safe_writer` | Stays in `nodes.py` (used by search tools for progress events) |
| `get_effective_datastore_ids` from `retrieval.py` | Called by search tools to resolve datastore scope from KB IDs |

---

## 3. Tool Inventory

### 3.1 New search tools

#### `search_exact`

MySQL FULLTEXT search across `document_chunks.chunk_text` and `documents.title`. Title matches weighted 2×. Fast, good for exact terms, code identifiers, title lookups.

**File**: `backend/app/services/agentic_rag/tools/search_exact.py`

**Schema** (`SearchExactInput`):

```python
class SearchExactInput(BaseModel):
    query: str = Field(description="Search query — exact terms, code, identifiers, or title fragments.")
    kb_ids: List[int] = Field(default_factory=list, description="Knowledge base IDs to search.")
    document_ids: Optional[List[int]] = Field(default=None, description="Restrict to these document IDs.")
    filters: Optional[dict] = Field(default=None, description="Metadata filters: title_contains, file_name_contains, content_type, file_modified_after, file_modified_before, file_created_after, file_created_before.")
    top_k: int = Field(default=20, description="Maximum hits to return. Increase for aggregate queries.")
```

**Internal behavior**:
1. Resolve `filters` to `document_ids` via `_resolve_filter_to_doc_ids` (same as current `rag_retrieve`).
2. Call `exact_search_docs()` from `retrieval.py`.
3. Internally run synonym expansion (Redis-cached, same as current `_expand_synonyms`).
4. Apply `EXACT_MIN_SCORE` threshold.
5. Return hits with `CitationRef(citation_kind="chunk", source_tool="search_exact")`.

**Returns**:
```python
{
    "ok": True,
    "result": {"hits": [...], "query_used": "...", "search_type": "exact", "count": N},
    "error": None,
    "tokens": estimated_tokens
}
```

#### `search_sparse`

Qdrant SPLADE sparse vector search. Good for keyword matching across long documents.

**File**: `backend/app/services/agentic_rag/tools/search_sparse.py`

**Schema** (`SearchSparseInput`): same fields as `SearchExactInput`.

**Internal behavior**:
1. Resolve `filters` to `document_ids`.
2. Call `sparse_search_docs()` from `retrieval.py` (includes synonym RRF fusion).
3. Apply `SPARSE_MIN_SCORE` threshold.
4. Return hits with `CitationRef(citation_kind="chunk", source_tool="search_sparse")`.

#### `search_dense`

Qdrant dense vector search. Good for semantic/conceptual matching.

**File**: `backend/app/services/agentic_rag/tools/search_dense.py`

**Schema** (`SearchDenseInput`): same fields as `SearchExactInput`.

**Internal behavior**:
1. Resolve `filters` to `document_ids`.
2. Call `dense_search_docs()` from `retrieval.py`.
3. Apply `DENSE_MIN_SCORE` threshold.
4. Return hits with `CitationRef(citation_kind="chunk", source_tool="search_dense")`.

#### `rerank_results`

Cross-encoder reranker. Accepts hits from any search tool(s), dedups by content hash, runs semantic dedup, scores with cross-encoder, applies threshold/elbow cut.

**File**: `backend/app/services/agentic_rag/tools/rerank_results.py`

**Schema** (`RerankResultsInput`):

```python
class RerankResultsInput(BaseModel):
    query: str = Field(description="The search query to rerank against.")
    hits: List[dict] = Field(description="Hits from one or more search tools. Each hit is a dict with 'content', 'document_id', 'chunk_index', 'title', 'content_hash', etc.")
    top_n: Optional[int] = Field(default=None, description="Maximum hits after reranking. If None, all hits passing the threshold are returned (no hard cap).")
```

**Internal behavior**:
1. Convert `hits` (list of dicts) to `LangchainDocument` objects.
2. Dedup by content hash (`dedup_by_content_hash`).
3. Semantic dedup (`semantic_dedup`).
4. Score with `rerank()` from `reranker.py` (threshold from `RERANKER_SCORE_THRESHOLD` setting).
5. If `top_n` is provided, cap to `top_n`. If `top_n` is None, return all hits passing threshold (no hard cap — per user's explicit requirement).
6. Preserve `CitationRef` from source hits, update `source_tool` to `"rerank_results"`.
7. Set `_reranker_score` on each hit.

**Returns**:
```python
{
    "ok": True,
    "result": {
        "hits": [...],  # reranked, deduped, threshold-filtered
        "query_used": "...",
        "input_count": N,
        "output_count": M,
        "best_score": float,
        "threshold": float
    },
    "error": None,
    "tokens": estimated_tokens
}
```

#### `graph_expand`

Neo4j graph expansion from seed chunks. Takes seed document IDs or chunk point IDs, returns related chunks via entity graph traversal.

**File**: `backend/app/services/agentic_rag/tools/graph_expand.py`

**Schema** (`GraphExpandInput`):

```python
class GraphExpandInput(BaseModel):
    kb_ids: List[int] = Field(default_factory=list, description="Knowledge base IDs to search within.")
    seed_document_ids: Optional[List[int]] = Field(default=None, description="Document IDs to use as seeds. The tool will find their chunks and traverse the graph.")
    seed_chunk_ids: Optional[List[str]] = Field(default=None, description="Qdrant point UUIDs to use as seeds directly.")
    top_k: int = Field(default=10, description="Maximum expanded chunks to return.")
```

**Internal behavior**:
1. If `seed_document_ids` provided, fetch their `qdrant_point_id` from Qdrant.
2. If `seed_chunk_ids` provided, use them directly.
3. Call `expand_docs_via_graph()` from `graph/expand.py`.
4. Return expanded chunks with `CitationRef(citation_kind="chunk", source_tool="graph_expand")`.
5. Non-fatal: returns empty list on any failure.

### 3.2 Existing tools — enhancements only

#### `kb_search_documents`

**Enhancement**: Attach `CitationRef(citation_kind="file")` to each returned document.

#### `kb_read`

**Enhancement**: Attach `CitationRef` with kind based on read mode:
- Full file read → `citation_kind="file"`
- Section read → `citation_kind="section"`, populate `section`, `start_char`, `end_char`
- Range read → `citation_kind="range"`, populate `start_char`, `end_char`, `start_line`, `end_line`

**Enhancement**: Return line numbers alongside content for range reads, so citations can point to specific lines.

#### `kb_grep`

**Enhancement**: Attach `CitationRef(citation_kind="grep")` with `match_line` and `quoted_text` for each match.

#### `kb_outline`

**Enhancement**: Attach `CitationRef(citation_kind="outline")`.

#### `extract_data`

**Enhancement**: Attach `CitationRef(citation_kind="table")` with `section` or `start_char`/`end_char` from source documents.

#### `kb_metadata`, `current_datetime`, `file_read`, `file_summarize`, `file_extract_table`, `code_execute`, `chart_generate`, `summarize_answer`

No changes to schemas. These tools don't produce citable evidence (or already produce non-chunk evidence that doesn't need `CitationRef`).

### 3.3 Tool registry update

**File**: `backend/app/services/agentic_rag/tools/__init__.py`

New `_TOOL_CLASSES`:

```python
_TOOL_CLASSES = [
    # Search (atomic, replaces rag_retrieve)
    SearchExactTool,
    SearchSparseTool,
    SearchDenseTool,
    RerankResultsTool,
    GraphExpandTool,
    # Discovery
    KbSearchDocumentsTool,
    KbMetadataTool,
    KbOutlineTool,
    CurrentDatetimeTool,
    # Read
    KbReadTool,
    FileReadTool,
    FileSummarizeTool,
    FileExtractTableTool,
    # Processing
    CodeExecuteTool,
    ChartGenerateTool,
    SummarizeAnswerTool,
    ExtractDataTool,
    KbGrepTool,
]
```

### 3.4 Deferred tool gating (`applicable_tools`)

Tools become available only after prior tool use, using Pi's `addedToolNames` pattern:

| Tool | Available after |
|---|---|
| `rerank_results` | At least one search tool (`search_exact`, `search_sparse`, `search_dense`) has been called |
| `graph_expand` | At least one search tool has been called |
| `chart_generate` | `extract_data` or `code_execute` has been called (existing behavior) |
| `extract_data` | `kb_read`, `kb_search_documents`, or a search tool has been called |

**Implementation**: Track which tools have been called in `state["tool_call_counts"]`. In `applicable_tools()`, check the counts before including deferred tools.

```python
def applicable_tools(ctx: "ToolContext") -> list:
    tools = build_tools(ctx)
    state = ctx.state
    counts = state.get("tool_call_counts", {}) if state else {}
    has_file = bool(state.get("file_markdown")) if state else False
    has_data = _has_chart_data(state)
    has_search = any(counts.get(t, 0) > 0 for t in ("search_exact", "search_sparse", "search_dense"))
    has_read = any(counts.get(t, 0) > 0 for t in ("kb_read", "kb_search_documents")) or has_search

    if not has_file:
        tools = _filter_tools_by_name(tools, ("file_read", "file_summarize", "file_extract_table"))
    if not has_data:
        tools = _filter_tools_by_name(tools, ("chart_generate",))
    if not has_search:
        tools = _filter_tools_by_name(tools, ("rerank_results", "graph_expand"))
    if not has_read:
        tools = _filter_tools_by_name(tools, ("extract_data",))

    return tools
```

Note: `tool_call_counts` is a new state key (see §6). The current `tool_call_count` dict is renamed to `tool_call_counts` for clarity.

---

## 4. Citation Model

### 4.1 New `CitationRef` schema

Replaces the current `CitationRef(document_id, chunk_index)` in `schemas.py`.

**File**: `backend/app/services/agentic_rag/schemas.py`

```python
class CitationRef(BaseModel):
    """Reference to a piece of evidence cited in the answer."""

    document_id: int
    citation_kind: Literal["chunk", "file", "section", "range", "grep", "table", "outline"]
    chunk_index: Optional[int] = None
    section: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    page: Optional[int] = None
    match_line: Optional[int] = None
    quoted_text: Optional[str] = None
    source_tool: Optional[str] = None
    citation_id: str = ""

    @field_validator("document_id", mode="before")
    @classmethod
    def _coerce_kb_label(cls, v):
        if isinstance(v, str):
            digits = "".join(ch for ch in v if ch.isdigit())
            if digits:
                return int(digits)
        return v
```

### 4.2 Citation per tool

| Tool | `citation_kind` | Populated fields |
|---|---|---|
| `search_exact` | `chunk` | `document_id`, `chunk_index`, `page`, `quoted_text`, `source_tool` |
| `search_sparse` | `chunk` | `document_id`, `chunk_index`, `page`, `quoted_text`, `source_tool` |
| `search_dense` | `chunk` | `document_id`, `chunk_index`, `page`, `quoted_text`, `source_tool` |
| `rerank_results` | `chunk` (passes through from source hits) | Same as source + `source_tool="rerank_results"` |
| `graph_expand` | `chunk` | `document_id`, `chunk_index`, `page`, `source_tool="graph_expand"` |
| `kb_search_documents` | `file` | `document_id`, `quoted_text` (first 200 chars), `source_tool="kb_search_documents"` |
| `kb_read` (full file) | `file` | `document_id`, `quoted_text`, `source_tool="kb_read"` |
| `kb_read` (section) | `section` | `document_id`, `section`, `start_char`, `end_char`, `source_tool="kb_read"` |
| `kb_read` (range) | `range` | `document_id`, `start_char`, `end_char`, `start_line`, `end_line`, `source_tool="kb_read"` |
| `kb_grep` | `grep` | `document_id`, `match_line`, `quoted_text`, `source_tool="kb_grep"` |
| `kb_outline` | `outline` | `document_id`, `source_tool="kb_outline"` |
| `extract_data` | `table` | `document_id`, `section` or `start_char`/`end_char`, `source_tool="extract_data"` |

### 4.3 Citation rendering in generation prompt

**File**: `backend/app/services/agentic_rag/utils.py` — `format_context_string()`

Replace the current `[KB-N]` labeling with `[E1]`, `[E2]`, ... evidence IDs. Each evidence item gets a stable `citation_id` assigned at finalize time.

New `format_context_string` output:

```
[E1] document="Weekly Update 21-28 Aug 2026.pdf", kind=chunk, chunk=5, page=3, source=search_dense
     "The weekly update covers 4 topics: ..."

[E2] document="Weekly Update 21-28 Aug 2026.pdf", kind=section, section="Topics Covered", source=kb_read
     "1. API Gateway migration 2. Database sharding ..."

[E3] document="Weekly Update 1-7 Aug 2026.pdf", kind=file, source=kb_search_documents
     "Weekly Update for Aug 1-7, 2026. Topics: ..."
```

Implementation:

```python
def format_context_string(
    docs: list[dict],
    file_markdown: str | None = None,
    db: Any = None,
    org_id: Any = None,
    query_glossary: str = "",
) -> str:
    from app.services.agentic_rag.agent_graph import _prune_contiguous_overlaps

    pruned_docs = _prune_contiguous_overlaps(docs) if docs else docs
    parts: list[str] = []
    for i, doc in enumerate(pruned_docs, 1):
        metadata = doc.get("metadata", {})
        content = metadata.get("original_text", doc.get("page_content", "")).strip()
        citation_ref = metadata.get("citation_ref", {})
        citation_id = citation_ref.get("citation_id") or f"E{i}"
        # Build the evidence header
        kind = citation_ref.get("citation_kind", "chunk")
        header_parts = [f'document="{metadata.get("title", metadata.get("file_name", ""))}"',
                        f"kind={kind}"]
        if citation_ref.get("chunk_index") is not None:
            header_parts.append(f"chunk={citation_ref['chunk_index']}")
        if citation_ref.get("page") is not None:
            header_parts.append(f"page={citation_ref['page']}")
        if citation_ref.get("section"):
            header_parts.append(f"section={citation_ref['section']}")
        if citation_ref.get("start_line") is not None and citation_ref.get("end_line") is not None:
            header_parts.append(f"lines={citation_ref['start_line']}-{citation_ref['end_line']}")
        if citation_ref.get("match_line") is not None:
            header_parts.append(f"line={citation_ref['match_line']}")
        if citation_ref.get("source_tool"):
            header_parts.append(f"source={citation_ref['source_tool']}")
        header = f"[{citation_id}] " + ", ".join(header_parts)
        parts.append(f"{header}\n     \"{content}\"")
    if file_markdown:
        parts.append(f"[File Content]\n{file_markdown}")
    glossary = _build_glossary_section(db, org_id, query_glossary, pruned_docs)
    if glossary:
        parts.append(glossary)
    return "\n\n---\n\n".join(parts)
```

### 4.4 Citation normalization

**File**: `backend/app/services/agentic_rag/utils.py` — `normalize_citations()`

Replace the current `[N](N)` normalization with `[E1]`, `[E2]` handling.

New behavior:
1. Accept `[E1]`, `[E2]`, ... markers (case-insensitive).
2. Map each `E` marker to the corresponding `CitationRef` from the evidence list.
3. Renumber in first-appearance order (e.g., if the answer cites `[E3]` first, then `[E1]`, the output uses `[1]` for E3 and `[2]` for E1).
4. Strip out-of-range citations (e.g., `[E99]` when only 5 evidence items exist).
5. Protect code blocks (same as current: extract before processing, restore after).
6. Strip citations from reasoning sections (same as current).
7. Return: `(rewritten_answer, list_of_citation_refs_in_display_order)`.

```python
def normalize_citations(answer: str, evidence: list[dict]) -> tuple[str, list[dict]]:
    """Validate, deduplicate, and renumber [E1], [E2] citations.

    Returns (rewritten_answer, cited_evidence_in_display_order).
    Each item in cited_evidence is the evidence dict with its CitationRef.
    """
    if not answer:
        return answer or "", []
    if not evidence:
        # Strip all [E\d+] markers
        cleaned = re.sub(r"\[E\d+\]", "", answer, flags=re.IGNORECASE)
        return cleaned.strip(), []

    # Build E-number → evidence index map (1-based)
    max_e = len(evidence)

    # Split out code blocks
    _code_segments: list[str] = []
    def _extract_code(m):
        _code_segments.append(m.group(0))
        return f"\x00CODE{len(_code_segments) - 1}\x00"
    answer = re.sub(r"```[\s\S]*?```", _extract_code, answer)
    answer = re.sub(r"`[^`]*`", _extract_code, answer)

    # Split out reasoning sections
    _reasoning_segments: list[str] = []
    def _extract_reasoning(m):
        _reasoning_segments.append(m.group(0))
        return f"\x00REASONING{len(_reasoning_segments) - 1}\x00"
    _full_patterns, _ = _build_reasoning_patterns()
    for pat in _full_patterns:
        answer = pat.sub(_extract_reasoning, answer)

    # Collect unique E-numbers in first-appearance order
    valid_cited: list[int] = []
    seen: set[int] = set()
    for match in re.finditer(r"\[E(\d+)\]", answer, re.IGNORECASE):
        n = int(match.group(1))
        if 1 <= n <= max_e and n not in seen:
            valid_cited.append(n)
            seen.add(n)

    # Renumber: first cited → [1], second → [2], etc.
    index_map = {orig: new for new, orig in enumerate(valid_cited, start=1)}

    def _replace_marker(match):
        n = int(match.group(1))
        if n in index_map:
            return f"[{index_map[n]}]"
        return ""
    normalized = re.sub(r"\[E(\d+)\]", _replace_marker, answer, flags=re.IGNORECASE)

    # Restore code blocks
    normalized = re.sub(r"\x00CODE(\d+)\x00", lambda m: _code_segments[int(m.group(1))], normalized)

    # Restore reasoning sections with citations stripped
    def _strip_reasoning_citations(text):
        return re.sub(r"\[E\d+\]", "", text, flags=re.IGNORECASE)
    normalized = re.sub(r"\x00REASONING(\d+)\x00",
                        lambda m: _strip_reasoning_citations(_reasoning_segments[int(m.group(1))]),
                        normalized)

    cited_evidence = [evidence[i - 1] for i in valid_cited]
    return normalized, cited_evidence
```

### 4.5 Citation survival across compaction

When compaction trims `retrieved_docs`, it must preserve the `citation_ref` metadata. The compaction logic in `agent_graph/compaction.py` trims `page_content` but keeps `metadata` intact. Since `citation_ref` lives in `metadata`, it survives compaction automatically.

For `accumulated_data` (structured extraction results), each entry already carries `document_id` and `title`. The `CitationRef` is attached to the source document in `retrieved_docs`, not to the accumulated data itself. When `chart_generate` produces a chart from accumulated data, the chart's citation points to the source documents, not to the accumulated data rows.

### 4.6 Citation in `LastAnswerObject`

**File**: `backend/app/services/agentic_rag/schemas.py`

Update `LastAnswerObject.citations` to use the new `CitationRef`:

```python
class LastAnswerObject(BaseModel):
    ...
    citations: List[CitationRef] = Field(default_factory=list, description="Evidence cited in the answer.")
    ...
```

### 4.7 Citation in `finalize_node`

**File**: `backend/app/services/agentic_rag/agent_graph/finalization.py`

Update `finalize_node`:

1. Gather evidence from `retrieved_docs` (which now carry `citation_ref` in metadata).
2. Assign `citation_id` (`E1`, `E2`, ...) to each evidence item in `group_docs_by_document` order.
3. Build context string with `format_context_string` (uses `[E1]` labels).
4. Stream final answer.
5. Call `normalize_citations(final, evidence)` → get `(rewritten_answer, cited_evidence)`.
6. Emit `answer_rewrite` with `citations: cited_evidence` (list of dicts with `CitationRef`).
7. Build `LastAnswerObject` with `citations` as list of `CitationRef` objects.

### 4.8 Frontend citation rendering

**File**: `frontend/src/components/chat/answer.tsx`

Update the `Citation` interface to include `citation_kind` and new fields:

```typescript
interface CitationMetadata {
  kb_id?: number;
  document_id?: number;
  source?: string;
  citation_kind?: string;  // NEW: "chunk" | "file" | "section" | "range" | "grep" | "table" | "outline"
  chunk_index?: number;
  section?: string;         // NEW
  start_char?: number;      // NEW
  end_char?: number;        // NEW
  start_line?: number;      // NEW
  end_line?: number;        // NEW
  page?: number;            // NEW
  match_line?: number;      // NEW
  quoted_text?: string;     // NEW
  source_tool?: string;     // NEW
  [key: string]: unknown;
}
```

Rendering per `citation_kind`:
- `chunk`: existing behavior — link to document + chunk highlight.
- `file`: link to document, no chunk highlight. Tooltip shows "Full document".
- `section`: link to document, scroll to section. Tooltip shows section name.
- `range`: link to document, highlight character/line range. Tooltip shows line range.
- `grep`: link to document, highlight matching line. Tooltip shows matched text.
- `table`: link to document, scroll to section. Tooltip shows "Extracted table".
- `outline`: link to document. Tooltip shows "Document outline".

The citation marker in the answer text changes from `[N](N)` to `[N]`. The frontend renders `[N]` as a clickable superscript link.

---

## 5. Execution Layer Changes

### 5.1 Replace correction-LLM with Pi-style `isError` pattern

**File**: `backend/app/services/agentic_rag/agent_graph/tooling.py`

**Current behavior**: Failed tool calls with argument errors call `_correct_tool_args()` which invokes a correction LLM to produce fixed arguments. This adds latency and an extra LLM call per failure.

**New behavior**: Failed tool calls return as error observations. The LLM sees the error in the next think turn and decides whether to retry with adjusted arguments. This is Pi's pattern: `isError` results go back to the model.

**Changes**:
1. Remove `_correct_tool_args()` function.
2. Remove `TOOL_CORRECTION_PROMPT` from `prompts.py`.
3. Simplify `_retry_failed_calls()`: only retry transient errors (network, timeout) with backoff. Argument errors are NOT retried — they go back to the LLM as `isError` observations.
4. Remove `_correction_hints()` from `helpers.py`.

New `_retry_failed_calls`:

```python
async def _retry_failed_calls(
    new_observations: list[Observation],
    tool_calls: list[dict],
    tools: dict,
    max_retries: int,
    ctx: "ToolContext",
) -> None:
    """Retry transient failures only. Argument errors go back to the LLM."""
    writer = _writer()
    if max_retries <= 0:
        return
    for idx, obs in enumerate(new_observations):
        if obs.error is None:
            continue
        if not _is_transient_error(obs.error):
            continue  # LLM will see the error and decide what to do
        tool_name = obs.tool
        tool = tools.get(tool_name)
        if tool is None:
            continue
        for attempt in range(max_retries):
            await asyncio.sleep(get_setting(ctx.db, "AGENT_RETRY_BACKOFF_BASE", ctx.org_id) * (2 ** attempt))
            retry_result = await _run_tool(tool, tool_name, obs.arguments)
            retry_obs = Observation(
                tool=retry_result["tool"],
                arguments=retry_result["arguments"],
                result=retry_result.get("result", {}),
                error=retry_result.get("error"),
                tokens=retry_result.get("tokens", 0),
            )
            writer({
                "event": "tool_retry",
                "tool": tool_name,
                "attempt": attempt + 1,
                "max_retries": max_retries,
                "success": retry_obs.error is None,
                "error": retry_obs.error,
            })
            if retry_obs.error is None:
                new_observations[idx] = retry_obs
                break
            if not _is_transient_error(retry_obs.error):
                break  # Became a non-transient error, stop retrying
```

### 5.2 Total tool-call budget

**File**: `backend/app/services/agentic_rag/agent_graph/helpers.py`, `backend/app/core/settings_registry.py`

New setting: `AGENT_TOTAL_TOOL_BUDGET` (default: 20, org-overridable).

> **Note**: An earlier conversation summary mentioned a default of 15. The canonical value is **20**, consistent across all sections of this document. The total budget must be higher than the sum of per-tool budgets to allow the LLM to compose multiple search strategies (e.g. `search_sparse` + `search_dense` + `rerank_results` + `kb_read`).

Track total tool calls across all tools in `state["total_tool_calls"]`. When exceeded, force finalize.

In `tool_node`:

```python
total = state.get("total_tool_calls", 0) + len(new_observations)
if total >= get_setting(ctx.db, "AGENT_TOTAL_TOOL_BUDGET", ctx.org_id):
    logger.info("[tool_node] total tool-call budget (%d) exceeded, forcing finalize", total)
    state_update["force_finalize"] = True
state_update["total_tool_calls"] = total
```

### 5.3 Per-tool budgets (updated)

**File**: `backend/app/services/agentic_rag/agent_graph/helpers.py`

New `_tool_call_budget`:

```python
def _tool_call_budget(db, org_id) -> dict:
    return {
        "search_exact": get_setting(db, "AGENT_MAX_SEARCH_EXACT", org_id),
        "search_sparse": get_setting(db, "AGENT_MAX_SEARCH_SPARSE", org_id),
        "search_dense": get_setting(db, "AGENT_MAX_SEARCH_DENSE", org_id),
        "rerank_results": get_setting(db, "AGENT_MAX_RERANK", org_id),
        "graph_expand": get_setting(db, "AGENT_MAX_GRAPH_EXPAND", org_id),
        "code_execute": get_setting(db, "AGENT_MAX_CODE_EXEC", org_id),
        "kb_grep": get_setting(db, "AGENT_MAX_KB_GREP", org_id),
        "kb_read": get_setting(db, "AGENT_MAX_KB_READ", org_id),
        "kb_outline": get_setting(db, "AGENT_MAX_KB_READ", org_id),
        "kb_search_documents": get_setting(db, "AGENT_MAX_KB_SEARCH", org_id),
        "extract_data": get_setting(db, "AGENT_MAX_EXTRACT_DATA", org_id),
        "chart_generate": get_setting(db, "AGENT_MAX_CHART_GENERATE", org_id),
    }
```

New settings in `settings_registry.py`:

```python
SettingDef("AGENT_TOTAL_TOOL_BUDGET", "Agentic", "Total tool-call budget", 20, ...),
SettingDef("AGENT_MAX_SEARCH_EXACT", "Agentic", "Max search_exact calls", 5, ...),
SettingDef("AGENT_MAX_SEARCH_SPARSE", "Agentic", "Max search_sparse calls", 5, ...),
SettingDef("AGENT_MAX_SEARCH_DENSE", "Agentic", "Max search_dense calls", 5, ...),
SettingDef("AGENT_MAX_RERANK", "Agentic", "Max rerank_results calls", 5, ...),
SettingDef("AGENT_MAX_GRAPH_EXPAND", "Agentic", "Max graph_expand calls", 3, ...),
SettingDef("AGENT_MAX_KB_SEARCH", "Agentic", "Max kb_search_documents calls", 10, ...),
SettingDef("AGENT_MAX_EXTRACT_DATA", "Agentic", "Max extract_data calls", 5, ...),
SettingDef("AGENT_MAX_CHART_GENERATE", "Agentic", "Max chart_generate calls", 3, ...),
```

Remove old settings: `AGENT_MAX_RETRIEVALS` (replaced by per-search-tool caps).

**Critical ordering constraint**: `get_setting()` falls back to `getattr(env_settings, key, None)` for unregistered keys, returning `None`. If new settings (`AGENT_TOTAL_TOOL_BUDGET`, `AGENT_MAX_SEARCH_*`) are used in code before they are added to `settings_registry.py`, arithmetic like `total >= budget` will raise `TypeError: '>=' not supported between int and NoneType`. Similarly, if `AGENT_MAX_RETRIEVALS` is removed from the registry while `_build_execution_summary` or `_tool_call_budget` still call `get_setting(..., "AGENT_MAX_RETRIEVALS", ...)`, the subtraction `max_retrievals - retrieval_queries` will raise `TypeError`.

**Implementation order**:
1. Add all new settings to `settings_registry.py` first.
2. Update all code that reads the new settings.
3. Only then remove `AGENT_MAX_RETRIEVALS` from the registry, after confirming no code reads it.

Also remove these orphaned settings (only used by `rag_retrieve.py`):
- `ADAPTIVE_RETRIEVAL_ENABLED`, `ADAPTIVE_RETRIEVAL_THRESHOLD`, `ADAPTIVE_RETRIEVAL_RERANKER_THRESHOLD`, `ADAPTIVE_RETRIEVAL_FAST_ACCEPT_SCORE`
- `SYNONYM_VARIANTS`, `SYNONYM_CACHE_TTL` (synonym expansion moves into search tools, which use the same Redis cache but read these settings internally — keep if search tools use them, remove otherwise)
- `PRE_FUSION_MIN_DOCS`
- `COLLAPSE_SAME_TITLE_VERSIONS`, `RRF_FUSION_ENABLED`, `MERGE_MMR_LAMBDA` (only used by `merge_node` in `nodes.py`, which becomes dead when `rag_retrieve` is removed)

**Keep** these settings (used by low-level retrieval or other code):
- `RETRIEVAL_TOP_K`, `DENSE_MIN_SCORE`, `SPARSE_MIN_SCORE`, `EXACT_MIN_SCORE`, `QDRANT_MMR_DIVERSITY`, `DEDUP_SEMANTIC_THRESHOLD`
- `RERANKER_SCORE_THRESHOLD`, `ELBOW_CUT_ENABLED`, `RERANKER_CONFIDENCE_THRESHOLD`, `RERANKER_CONFIDENCE_GAP`
- `GRAPHRAG_RETRIEVAL_HOPS`, `GRAPHRAG_RETRIEVAL_LIMIT`, `GRAPHRAG_ENTITY_FANOUT_CAP`
- `ENTITY_BOOST_FACTOR`
- `ABBREVIATION_EXPANSION_ENABLED` (used by `abbreviation_service.py` independently)

### 5.4 `terminate` hint on tool results

**File**: `backend/app/services/agentic_rag/tools/base.py`

Add optional `terminate` field to tool return envelopes:

```python
class BaseAgentTool(BaseTool):
    ctx: Optional[ToolContext] = Field(default=None, exclude=True)
    ui_label: str = "Running tool"

    async def _arun(self, *args, **kwargs) -> Any:
        kwargs.pop("run_manager", None)
        if args and isinstance(args[0], dict):
            kwargs = args[0]
        input_obj = self.args_schema(**kwargs)
        return await self._execute(input_obj)

    async def _execute(self, input_obj: BaseModel) -> dict:
        raise NotImplementedError
```

Tools return `{"ok": bool, "result": dict, "error": str|None, "tokens": int, "terminate": bool}`. The `terminate` field defaults to `False`. When `True`, the tool node sets `force_finalize = True`.

Used by: `sufficiency_check` node (not a tool — it's a graph node that can set `terminate`), and potentially by `current_datetime` (single-use).

### 5.5 `prepareArguments` pattern from Pi

**File**: `backend/app/services/agentic_rag/tools/base.py`

Add optional `prepare_arguments` method to `BaseAgentTool`:

```python
class BaseAgentTool(BaseTool):
    ...
    def prepare_arguments(self, args: dict) -> dict:
        """Normalize/validate arguments before execution. Override in subclasses."""
        return args

    async def _arun(self, *args, **kwargs) -> Any:
        kwargs.pop("run_manager", None)
        if args and isinstance(args[0], dict):
            kwargs = args[0]
        kwargs = self.prepare_arguments(kwargs)
        input_obj = self.args_schema(**kwargs)
        return await self._execute(input_obj)
```

Use cases:
- `rerank_results`: coerce `hits` from list-of-dicts with varying shapes to a consistent format.
- `search_*`: normalize `kb_ids` to list of ints, resolve `document_ids` from filters.
- `kb_read`: normalize `section` to string, `start_char`/`end_char` to ints.

### 5.6 `_merge_retrieved_docs` update

**File**: `backend/app/services/agentic_rag/agent_graph/tooling.py`

Update `_merge_observation_docs` to handle hits from the new search tools:

```python
def _merge_observation_docs(all_observations, seen_hashes, merged_docs):
    from app.services.infrastructure import content_hash as _ch
    best_confidence = 0.0
    SEARCH_TOOLS = {"search_exact", "search_sparse", "search_dense", "rerank_results", "graph_expand"}
    for obs in all_observations:
        if obs.tool in SEARCH_TOOLS and not obs.error:
            hits = obs.result.get("hits")
            if isinstance(hits, list):
                for hit in hits:
                    if not isinstance(hit, dict):
                        continue
                    content = hit.get("content", "")
                    h = hit.get("content_hash") or _ch(content)
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        # Convert to the doc dict shape expected by finalize
                        doc_dict = {
                            "page_content": content,
                            "metadata": {
                                "document_id": hit.get("document_id"),
                                "title": hit.get("title"),
                                "file_name": hit.get("file_name"),
                                "chunk_index": hit.get("chunk_index"),
                                "page": hit.get("page"),
                                "content_hash": h,
                                "qdrant_point_id": hit.get("qdrant_point_id"),
                                "source": hit.get("source_tool", obs.tool),
                                "_reranker_score": hit.get("_reranker_score", hit.get("score", 0.0)),
                                "citation_ref": hit.get("citation_ref", {}),
                            },
                        }
                        merged_docs.append(doc_dict)
                # Update confidence from reranker scores if present
                scores = [h.get("_reranker_score", 0) for h in hits if h.get("_reranker_score") is not None]
                if scores and max(scores) > best_confidence:
                    best_confidence = max(scores)
        elif obs.tool == "kb_search_documents" and not obs.error:
            # Same as current: document-level matches
            docs = obs.result.get("docs")
            if isinstance(docs, list):
                for doc in docs:
                    if not isinstance(doc, dict):
                        continue
                    h = doc.get("metadata", {}).get("content_hash") or _ch(doc.get("page_content", ""))
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        merged_docs.append(doc)
                if best_confidence < 0.9:
                    best_confidence = 0.9
        elif obs.tool == "kb_read" and not obs.error:
            # Same as current: single document content
            content = obs.result.get("content", "")
            if content:
                doc_dict = {
                    "page_content": content,
                    "metadata": {
                        "document_id": obs.result.get("document_id"),
                        "title": obs.result.get("title") or obs.result.get("file_name"),
                        "file_name": obs.result.get("file_name"),
                        "section": obs.result.get("section"),
                        "source": "kb_read",
                        "_reranker_score": 1.0,
                        "truncated": obs.result.get("truncated", False),
                        "citation_ref": obs.result.get("citation_ref", {}),
                    },
                }
                h = _ch(content)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    merged_docs.append(doc_dict)
                if best_confidence < 0.9:
                    best_confidence = 0.9
    return best_confidence
```

---

## 6. Graph Changes

### 6.1 New graph topology

**Current**:
```
load_context → expand_query → rewrite_query → plan → clarify_interrupt → think → tool → reflect → reflect_final → finalize → answer_scoring → save_memory
```

**New**:
```
load_context → plan → clarify_interrupt → think → tool → sufficiency_check → finalize → answer_scoring → save_memory
```

### 6.2 Removed nodes and retained helpers

| Node | File | Reason |
|---|---|---|
| `expand_query` | `nodes.py:expand_query_node` | Query expansion moves inside search tools (synonym expansion is per-search, not per-turn) |
| `rewrite_query` | `nodes.py:rewrite_query_node` | LLM does query rewriting naturally in the think loop; no need for a separate node |
| `reflect` | `agent_graph/reflection.py:reflect_node` | Replaced by `sufficiency_check` (simpler, more direct) |
| `reflect_final` | `agent_graph/reflection.py:reflect_final_node` | Replaced by `sufficiency_check` |

**Critical: helpers that must be retained.** `_verify_execution` and `_build_execution_summary` are defined in `reflection.py` (lines 168-255) but are imported and called by `tooling.py` (line 38, 382) and `thinking.py` (line 30, 108) — not just by `reflect_node`/`reflect_final_node`. Deleting them would break the tool and think nodes.

**Action**: Move `_verify_execution`, `_build_execution_summary`, and their sub-helpers (`_count_successful_by_tool`, `_build_subtask_status`, `_retrieval_doc_count`, `_collect_tool_failures`) into a new shared module `backend/app/services/agentic_rag/agent_graph/execution_check.py`. Update imports in `tooling.py`, `thinking.py`, and `agent_graph/__init__.py`. Rewrite `_build_execution_summary` to count atomic search tools instead of `rag_retrieve` and to read `AGENT_TOTAL_TOOL_BUDGET` instead of `AGENT_MAX_RETRIEVALS`.

**Critical: `agent_graph/__init__.py` exports.** The package `__init__.py` re-exports `expand_query_node`, `rewrite_query_node`, `reflect_node`, `reflect_final_node`, `_tried_rag_retrieve_queries`, and includes them in `__all__`. Removing the node functions without updating `__init__.py` will break the entire `agent_graph` package import, which breaks `agent_runner.py`, all API endpoints, and every test that imports from `agent_graph`.

**Action**: Remove these names from `__init__.py` imports and `__all__`. Add imports for `sufficiency_check_node`, `route_sufficiency` from the new `sufficiency.py`. Add imports for `_verify_execution`, `_build_execution_summary` from the new `execution_check.py`.

### 6.3 New node: `sufficiency_check`

**File**: `backend/app/services/agentic_rag/agent_graph/sufficiency.py` (new file)

**Purpose**: After each tool round, check whether the agent has enough evidence to answer. If yes → route to `finalize`. If no → route back to `think`.

**Implementation**:

```python
async def sufficiency_check_node(state, ctx) -> dict:
    """Check if the agent has sufficient evidence to answer.

    Two-tier check:
    1. Deterministic: if total_tool_calls >= AGENT_TOTAL_TOOL_BUDGET, force finalize.
    2. Deterministic: if force_finalize is already set, pass through.
    3. LLM-based: ask a lightweight LLM whether the evidence is sufficient.
    """
    with _agent_step("sufficiency_check"):
        # Tier 1: budget exhausted
        total = state.get("total_tool_calls", 0)
        budget = get_setting(ctx.db, "AGENT_TOTAL_TOOL_BUDGET", ctx.org_id)
        if total >= budget:
            return {"sufficient": True, "force_finalize": True}

        # Tier 2: already forced
        if state.get("force_finalize"):
            return {"sufficient": True}

        # Tier 3: LLM sufficiency check
        # Only run if we have some evidence
        docs = state.get("retrieved_docs", [])
        if not docs:
            return {"sufficient": False}

        query = state.get("original_query", "") or state.get("rewritten_query", "")
        previews = _build_evidence_previews(docs, max_chars=2000)

        llm = build_chat_llm(ctx.org_id, ctx.db, role="query", temperature=0.0)
        prompt = SUFFICIENCY_CHECK_PROMPT
        user = SUFFICIENCY_CHECK_USER_PROMPT.format(query=query, previews=previews)
        try:
            response = await llm.ainvoke([
                {"role": "system", "content": prompt},
                {"role": "user", "content": user},
            ])
            block = _extract_json_block(str(response.content))
            if block:
                result = json.loads(block)
                sufficient = result.get("sufficient", False)
                missing = result.get("missing", "")
                if sufficient:
                    return {"sufficient": True, "force_finalize": True}
                return {"sufficient": False}
        except Exception as exc:
            logger.warning("[sufficiency_check] LLM check failed: %s", exc)

        # Fallback: if we have docs and the reranker is confident, finalize
        if _reranker_confident(docs, ctx):
            return {"sufficient": True, "force_finalize": True}
        return {"sufficient": False}
```

**Routing**:

```python
def route_sufficiency(state) -> str:
    if state.get("sufficient") or state.get("force_finalize"):
        return "finalize"
    # Check iteration limit
    iteration = state.get("iteration", 0)
    # ... (wall clock check)
    if iteration >= max_iter or _wall_clock_exceeded(state):
        return "finalize"
    return "think"
```

### 6.4 Updated `build_agent_graph`

**File**: `backend/app/services/agentic_rag/agent_graph/build.py`

```python
def build_agent_graph(ctx):
    graph = StateGraph(AgentState)

    graph.add_node("load_context", partial(load_context_node, ctx=ctx))
    graph.add_node("plan", partial(plan_node, ctx=ctx))
    graph.add_node("clarify_interrupt", clarify_interrupt_node)
    graph.add_node("think", partial(think_node, ctx=ctx))
    graph.add_node("tool", partial(tool_node, ctx=ctx))
    graph.add_node("sufficiency_check", partial(sufficiency_check_node, ctx=ctx))
    graph.add_node("finalize", partial(finalize_node, ctx=ctx))
    graph.add_node("answer_scoring", partial(answer_scoring_node, ctx=ctx))
    graph.add_node("save_memory", partial(save_memory_node, ctx=ctx))

    graph.set_entry_point("load_context")
    graph.add_edge("load_context", "plan")
    graph.add_conditional_edges("plan", route_plan)
    graph.add_edge("clarify_interrupt", "plan")  # Changed: go back to plan, not expand_query
    graph.add_conditional_edges("think", route_think)
    graph.add_edge("tool", "sufficiency_check")
    graph.add_conditional_edges("sufficiency_check", route_sufficiency)
    graph.add_edge("finalize", "answer_scoring")
    graph.add_edge("answer_scoring", "save_memory")
    graph.add_edge("save_memory", END)

    checkpointer = getattr(ctx.redis_memory, "checkpointer", None) if ctx.redis_memory else None
    return graph.compile(checkpointer=checkpointer)
```

### 6.5 `route_think` update

**File**: `backend/app/services/agentic_rag/agent_graph/thinking.py`

`route_think` currently returns `"reflect_final"` when there are no tool calls or when the iteration/wall-clock limit is hit. This must change:

```python
def route_think(state) -> str:
    iteration = state.get("iteration", 0)
    from app.db.session import SessionLocal
    org_id = state.get("org_id")
    _db = SessionLocal()
    try:
        max_iter = get_setting(_db, "AGENT_MAX_ITERATIONS", org_id)
    finally:
        _db.close()
    if iteration >= max_iter or _wall_clock_exceeded(state):
        return "finalize"  # was "reflect_final"
    if state.get("tool_calls"):
        return "tool"
    return "finalize"  # was "reflect_final" — LLM emitted {final_answer: true}
```

The think → tool → sufficiency_check → (think | finalize) loop is preserved. When the LLM emits no tool calls (either `{final_answer: true}` or a Tier 3 plain-text fallback), the graph goes directly to `finalize`.

### 6.6 `route_tool` removal

**File**: `backend/app/services/agentic_rag/agent_graph/tooling.py`

The current `route_tool` function (line 43) returns `"reflect"` or `"reflect_final"`. Both destination nodes are removed. The function is deleted entirely. The graph edge changes from `tool → route_tool → (reflect | reflect_final)` to a fixed edge `tool → sufficiency_check`.

Also remove `route_reflect_final` (line 50) — it routes after `reflect_final`, which no longer exists.

### 6.7 `load_context_node` update

**File**: `backend/app/services/agentic_rag/agent_graph/load_context.py`

Remove the `expand_query` and `rewrite_query` calls from `load_context_node`. The node now only loads KB profile, conversation history, and file metadata. Query expansion and rewriting happen inside the search tools.

The current `load_context_node` (lines 80-103) resets a long list of state keys at the start of each turn. Update the reset list:

**Remove from reset list** (state keys no longer exist):
- `expanded_query`, `abbreviation_glossary`, `rewritten_query`, `resolution_provenance`, `query_intent`, `excluded_terms`
- `dense_docs`, `sparse_docs`, `exact_docs`, `graph_docs`, `leg_results`, `failed_legs`, `leg_doc_counts`
- `all_scored_docs`, `retrieval_confidence`, `adaptive_reran`, `graph_expansion_done`
- `reflection_final`, `tool_call_count`, `precomputed_tool_calls`

**Add to reset list** (new state keys):
- `total_tool_calls: 0`
- `tool_call_counts: {}`
- `sufficient: False`
- `evidence: []`

**Keep in reset list** (unchanged):
- `force_finalize: False`, `iteration: 0`, `observations: []`, `tool_calls: []`, `retrieved_docs: []`
- `plan: None`, `needs_clarification: False`, `clarification_question: None`
- `accumulated_data: []`, `cited_doc_indices: []`

---

## 7. State Changes

### 7.1 `AgentState` — additions

**File**: `backend/app/services/agentic_rag/graph_state.py`

```python
class AgentState(MessagesState):
    # ... existing fields kept (see §7.3) ...

    # ── NEW: Evidence with CitationRef ────────────────────────────
    # Each item: {"citation_id": "E1", "citation_ref": CitationRef, "content": str, "metadata": dict}
    # Populated by tool_node from search/rerank/read observations.
    # Used by finalize_node to build the evidence block and normalize citations.
    evidence: Annotated[List[dict], accumulate] = []

    # ── NEW: Total tool calls across all tools ────────────────────
    # Incremented by tool_node after each tool round. When it reaches
    # AGENT_TOTAL_TOOL_BUDGET, sufficiency_check forces finalize.
    total_tool_calls: Annotated[int, _last_value] = 0

    # ── NEW: Per-tool call counts (replaces tool_call_count) ──────
    # Used by applicable_tools() for deferred tool gating and by
    # _tool_call_budget() for per-tool caps.
    tool_call_counts: Annotated[dict, _last_value] = {}

    # ── NEW: Sufficiency flag ─────────────────────────────────────
    # Set by sufficiency_check_node. When True, route_sufficiency
    # sends the graph to finalize.
    sufficient: Annotated[bool, _last_value] = False
```

### 7.2 `AgentState` — removals

Remove these fields from `AgentState`:

```python
# Per-leg retrieval state — no longer needed (each search tool returns its own hits)
dense_docs: ...           # REMOVE
sparse_docs: ...          # REMOVE
exact_docs: ...           # REMOVE
graph_docs: ...           # REMOVE
leg_results: ...          # REMOVE
failed_legs: ...          # REMOVE
leg_doc_counts: ...       # REMOVE

# Merged retrieval state — replaced by evidence
all_scored_docs: ...      # REMOVE
retrieval_confidence: ... # REMOVE

# Query state — expand/rewrite nodes removed
expanded_query: ...       # REMOVE
abbreviation_glossary: ...  # REMOVE (built per-search inside each tool)
rewritten_query: ...      # REMOVE (LLM uses original_query directly)
resolution_provenance: ...  # REMOVE
query_intent: ...         # REMOVE
excluded_terms: ...       # REMOVE (handled inside search tools)

# Retry budget state
adaptive_reran: ...       # REMOVE
graph_expansion_done: ... # REMOVE

# Reflection state
reflection_final: ...     # REMOVE

# Old tool call count (renamed to tool_call_counts)
tool_call_count: ...      # REMOVE (renamed to tool_call_counts)

# Precomputed tool calls (recovery from reflect — reflect removed)
precomputed_tool_calls: ...  # REMOVE
```

### 7.3 `AgentState` — kept (unchanged)

```python
original_query: ...       # KEPT — the user's exact wording
kb_profile: ...           # KEPT — loaded by load_context_node
recalled_memories: ...    # KEPT — long-term memory hits
compaction_summary: ...   # KEPT
compaction_triggered: ... # KEPT
answer: ...               # KEPT
answer_usage: ...         # KEPT
cited_doc_indices: ...    # KEPT (but now indices into evidence, not docs)
final_answer: ...         # KEPT
final_confidence: ...     # KEPT
confidence_level: ...     # KEPT
faithfulness: ...         # KEPT
completeness: ...         # KEPT
retrieval_score: ...      # KEPT
confidence_match: ...     # KEPT
evaluation_flags: ...     # KEPT
kb_ids: ...               # KEPT
org_id: ...               # KEPT
chat_id: ...              # KEPT
user_id: ...              # KEPT
message_id: ...           # KEPT
file_markdown: ...        # KEPT
generate_answer: ...      # KEPT
plan: ...                 # KEPT
observations: ...         # KEPT
iteration: ...            # KEPT
tool_calls: ...           # KEPT
last_answer_object: ...   # KEPT (citations field uses new CitationRef)
needs_clarification: ...  # KEPT
clarification_question: ...  # KEPT
accumulated_data: ...     # KEPT
started_at: ...           # KEPT
force_finalize: ...       # KEPT
clarification_count: ...  # KEPT
clarification_response: ...  # KEPT
latency_ms: ...           # KEPT
model_used: ...           # KEPT
retrieved_docs: ...       # KEPT — still used by finalize, populated from observations
```

### 7.4 `Subtask` schema changes

**File**: `backend/app/services/agentic_rag/schemas.py`

Remove `suggested_legs`:

```python
class Subtask(BaseModel):
    id: str = Field(...)
    description: str = Field(...)
    tool_hint: str = Field(default="any", description="Tool name or 'any'.")
    depends_on: List[str] = Field(default_factory=list)
    expected_output: str = Field(default="")
    suggested_filters: Optional[dict] = Field(default=None, ...)
    suggested_sort: Optional[dict] = Field(default=None, ...)
    # suggested_legs: REMOVED
    suggested_query: Optional[str] = Field(default=None, ...)
    suggested_top_n: Optional[int] = Field(default=None, ...)
    suggested_metadata_only: Optional[bool] = Field(default=None, ...)
```

### 7.5 `QueryIntent` schema — removed

**File**: `backend/app/services/agentic_rag/schemas.py`

Remove the `QueryIntent` class entirely. The LLM chooses search tools directly in the think loop; there's no need for a separate intent extraction step.

### 7.6 `Observation` schema — unchanged

The `Observation` schema stays the same. Each tool call produces an `Observation(tool, arguments, result, error, tokens)`. The `result` field now contains `{"hits": [...]}` for search tools, `{"content": "..."}` for read tools, etc.

---

## 8. Prompt Changes

### 8.1 `PLAN_SYSTEM_PROMPT` (rewrite)

**File**: `backend/app/services/agentic_rag/prompts.py`

```python
PLAN_SYSTEM_PROMPT: str = """\
You are the planning module for an autonomous knowledge assistant. Given the user's query, the conversation context, the previous answer summary, attached file metadata, and the available tools, produce a plan.

Available tools:
- current_datetime: returns the current UTC date and time. Call this FIRST when the query involves "latest", "most recent", "newest", "this week", "last month", or any temporal reasoning.
- search_exact: MySQL fulltext search across chunk text and document titles. Fast. Best for exact terms, code identifiers, title fragments. Supports filters and document_ids.
- search_sparse: SPLADE sparse vector search. Best for keyword matching across document content. Supports filters and document_ids.
- search_dense: dense vector search. Best for semantic/conceptual matching. Supports filters and document_ids.
- rerank_results: cross-encoder reranker. Call AFTER one or more search tools when you have multiple hits and need to prioritize. Pass the hits from your search calls. No hard top_n cap — all hits passing the threshold are returned.
- graph_expand: expand from seed documents/chunks via Neo4j graph. Call when initial search results are insufficient and the KB has graph data.
- kb_search_documents: document-level retrieval by title, filename, content type, or date range. Returns full converted markdown. Use for named-document queries. Supports metadata_only=true for discovery.
- kb_metadata: inspect KB document metadata. Actions: list_fields, unique_values, date_range, list_documents, count_only.
- kb_outline: get heading structure of a KB document.
- kb_read: read a specific section or character range of a KB document.
- kb_grep: regex search across all KB document contents. Returns matching lines with line numbers.
- file_read: read a section of an attached file.
- file_summarize: map-reduce summarization of a large attached file.
- file_extract_table: extract a table from CSV/Excel/HTML in a file.
- code_execute: run Python for computation or data transformation.
- chart_generate: build an ECharts option from structured data. Reads from accumulated_data if no data argument.
- summarize_answer: summarize the previous answer.
- extract_data: pull structured data from retrieved docs, accumulated data, or a file. Results accumulate in state.

Query classification — choose the primary strategy:

1. NAMED-DOCUMENT (user wants a specific document by title/filename):
   → kb_search_documents with title_contains to find it, then kb_read to read it.
   → Do NOT use chunk search tools (search_exact/sparse/dense) as the first call.

2. CONCEPTUAL (user wants information about a topic):
   → Choose search tool based on query type:
     - Exact terms/code/identifiers → search_exact
     - Keyword matching → search_sparse
     - Semantic/conceptual → search_dense
   → After search, call rerank_results if you got more than 10 hits.
   → Call graph_expand if results are insufficient and KB has graph data.

3. AGGREGATE (user wants summary/count/table/chart across many documents):
   → kb_metadata (count_only or list_documents) first to discover scope.
   → kb_search_documents (metadata_only=true) to get document list.
   → Batch: kb_read or extract_data per document → accumulate → chart_generate.

4. EXACT-LOOKUP (user wants an exact term, code, or identifier):
   → kb_grep first (fastest for exact matches).
   → Fall back to search_exact if grep is insufficient.

5. TEMPORAL (user wants latest/oldest/by-date):
   → current_datetime first.
   → kb_search_documents with date filters and sort by file_modified_at.

Output a JSON object with this structure:
{{
  "intent": "rag|file_action|previous_answer_action|computation|chart|conversation|mixed",
  "subtasks": [
    {{
      "id": "a",
      "description": "...",
      "tool_hint": "search_dense|search_exact|search_sparse|kb_search_documents|kb_metadata|current_datetime|kb_read|kb_grep|rerank_results|graph_expand|extract_data|chart_generate|...|any",
      "depends_on": [],
      "expected_output": "...",
      "suggested_filters": null,
      "suggested_sort": null,
      "suggested_query": null,
      "suggested_top_n": null,
      "suggested_metadata_only": null
    }}
  ],
  "needs_clarification": false,
  "clarification_question": null
}}

Per-subtask parameters:
- suggested_filters: {{"title_contains":"..."}} for named documents, {{"content_type":"application/pdf"}} for file types, {{"file_modified_after":"2026-01-01"}} for date ranges.
- suggested_sort: {{"field":"file_modified_at","direction":"desc"}} for recency.
- suggested_query: Set when the subtask targets a specific aspect of a multi-part query.
- suggested_top_n: For kb_search_documents. 3 for "latest", 20-50+ for aggregate queries.
- suggested_metadata_only: true for discovery subtasks.
- Independent subtasks (no depends_on) dispatch in parallel. Dependent subtasks wait.

Parallel multi-search: for conceptual queries that benefit from both lexical and semantic matching, create two independent subtasks — one with search_sparse, one with search_dense — then a dependent subtask with rerank_results that combines their hits.

Rules for needs_clarification:
- Set true ONLY if the query is genuinely ambiguous or under-specified.
- Never set true because the topic seems "already covered".
- Default to false and let the acting module retrieve and answer.
"""
```

### 8.2 `THINK_SYSTEM_PROMPT` (rewrite)

```python
THINK_SYSTEM_PROMPT: str = """\
You are the acting module. You have a plan, a list of previous tool observations, and a set of tools. Decide the next action.

If the gateway supports function-calling, emit native tool calls. If it does not, emit a JSON block:
{ "tool_calls": [{"tool": "<name>", "arguments": {...}}] }
or for a single call:
{ "tool": "<name>", "arguments": {...} }
or to finish:
{ "final_answer": true }

Do NOT write the answer text. Emit the next tool call needed to advance the plan, or { "final_answer": true } if you have nothing left to do. Only call independent tools in one message; dependent calls must wait for their observations.

Search tool selection:
- search_exact: use for exact terms, code identifiers, title fragments. Fastest search.
- search_sparse: use for keyword matching across content. Good when the query has specific terms.
- search_dense: use for semantic/conceptual matching. Good when the query is about a concept, not specific terms.
- You can call multiple search tools in parallel (e.g. search_sparse + search_dense) then rerank_results with the combined hits.
- After any search, inspect the hits. If you have more than 10 hits, call rerank_results to prioritize.
- If search returns too few hits, try a different search tool or adjust your query.
- Never repeat a search call with the same query — it will return identical results.

Reranking:
- rerank_results accepts hits from any search tool(s). Pass the hits you received.
- rerank_results deduplicates by content hash and scores with a cross-encoder.
- No hard top_n cap — all hits passing the threshold are returned. Set top_n only if you want to limit.

Graph expansion:
- graph_expand takes seed document_ids or chunk_ids from prior search results.
- Call it when initial search is insufficient and the KB has graph data.

Document-specific queries (named documents like "weekly update", "Q3 report"):
- FIRST CHOICE: kb_search_documents with title_contains to get the full document.
- If the document is too large: kb_outline to see structure, then kb_read for specific sections.
- If kb_search_documents finds nothing: fall back to search_exact with filters={{"title_contains":"..."}}.

Aggregate/analysis queries (counting, summarizing across many documents, trends, tables, charts):
- Use kb_search_documents with metadata_only=true first to discover all matching documents.
- Then read specific documents in batches: kb_search_documents with document_ids for 5-10 at a time.
- After each batch, call extract_data with source="retrieved_docs" and document_ids=[...].
- After all batches: chart_generate with no data argument (reads from accumulated_data).
- Pattern: discover → read batch 1 → extract_data(batch 1) → read batch 2 → extract_data(batch 2) → ... → chart_generate() → final_answer.

Temporal reasoning — deciding which document is "latest":
- Call current_datetime FIRST to learn today's date.
- Do NOT blindly trust file_modified_at — a user may accidentally modify an old file.
- Compare dates in TITLES and CONTENT to determine which is truly latest.
- "Weekly Update 21-28 Aug 2026" is newer than "Weekly Update 1-7 Aug 2026" regardless of file_modified_at.
- Only fall back to file_modified_at when title/content dates are ambiguous.

Citation rules:
- The finalize node will format evidence as [E1], [E2], etc.
- You do not need to manage citation IDs — just gather the best evidence.
- Every search/read tool attaches citation metadata automatically.

Error handling:
- If a tool fails with an error, read the error message and adjust your arguments.
- Do NOT retry the same call with the same arguments — it will fail again.
- Try a different tool or different arguments based on the error.

You have a total budget of {total_budget} tool calls. Use them wisely.
"""
```

### 8.3 `FINALIZE_ANSWER_PROMPT` (rewrite)

```python
FINALIZE_ANSWER_PROMPT: str = """\
# Role

You are a helpful AI assistant. Your primary responsibility is to answer the user's questions accurately using the gathered evidence.

---

# Knowledge Source Priority

1. Gathered evidence (the only citable source)
2. General knowledge (only when necessary and clearly identified)

---

# Evidence

The evidence consists of items labeled [E1], [E2], etc. Each item shows:
- The document name
- The citation kind (chunk, file, section, range, grep, table)
- Relevant metadata (chunk index, page, section, line range)
- The evidence text

These are the authoritative source for document-specific information.

---

# Citation Rules

Every factual statement derived from the evidence should cite at least one evidence item.

Use this format: [N]

where N is the number of the evidence item (e.g., [1] for E1, [2] for E2).

Examples:
Process scheduling saves the CPU state before switching tasks [1].
The Banker algorithm avoids deadlock by checking resource availability [2] [3].

Rules:
- Cite only evidence you actually used.
- Never invent citations.
- A sentence supported by multiple evidence items may include multiple citations.
- The number MUST correspond to an evidence item listed above.

---

# Formatting Rules

Adapt structure to complexity:
- Simple questions: concise natural prose.
- Multi-part/technical: ### headings, numbered lists, bullet lists, **bold**, inline code.

Avoid unnecessary verbosity.

---

# Critical Rules

- Answer directly without repeating the user's question.
- Prefer evidence over general knowledge.
- Always use [N] citation format.
- If the evidence is insufficient, say so explicitly.
"""
```

### 8.4 `FINALIZE_GUARDRAIL_PROMPT` (update)

Remove the reference to "retrieved document chunks" — replace with "gathered evidence":

```python
FINALIZE_GUARDRAIL_PROMPT: str = """\
You are an autonomous enterprise knowledge assistant. You have no internet access. You operate only on:
1. The attached knowledge bases / data stores.
2. Files uploaded to this chat.
3. The current conversation history.

Critical rules:
- If you cannot find the answer in the provided evidence, say so. Do not fabricate.
- Cite the evidence items that support each factual claim.
- Be concise and follow the user's formatting instructions exactly.
"""
```

### 8.5 Removed prompts

| Prompt | Reason |
|---|---|
| `REWRITE_INTENT_SUFFIX` | `rewrite_query` node removed |
| `RETRIEVAL_REWRITE_PROMPT` | Query rewrite is now LLM-driven in think loop |
| `SUFFICIENCY_CHECK_PROMPT` | Kept but moved to `sufficiency.py` (used by the new node) |
| `SUFFICIENCY_CHECK_USER_PROMPT` | Same |
| `TOOL_CORRECTION_PROMPT` | Correction LLM removed |

### 8.6 `planning.py` precomputed tool calls (update)

**File**: `backend/app/services/agentic_rag/agent_graph/planning.py`

The current `plan_node` (lines 187-309) precomputes `rag_retrieve` tool calls based on `query_intent["suggested_filters"]["title_contains"]`, `suggested_legs`, `rewritten_query`, and `abbreviation_glossary`. All of these inputs are removed.

**Changes**:
1. Remove the fast-track branch that precomputes `rag_retrieve` calls (lines 287-288, 296-305).
2. Remove reading of `query_intent`, `suggested_legs`, `rewritten_query`, `abbreviation_glossary`.
3. The plan node now only produces a plan (subtasks with `tool_hint` and `suggested_filters`/`suggested_sort`/`suggested_query`). The think node dispatches actual tool calls based on the plan.
4. If precomputed tool calls are still desired for fast-tracking, precompute `search_exact`/`search_sparse`/`search_dense`/`kb_search_documents` calls based on `tool_hint` and `suggested_filters`. But this is optional — the think node can dispatch from the plan alone.

### 8.7 `finalization.py` dependencies (update)

**File**: `backend/app/services/agentic_rag/agent_graph/finalization.py`

The current `finalize_node` (lines 237-289) uses:
- `state["rewritten_query"]` (line 237-238) — for the generation prompt's query field
- `state["abbreviation_glossary"]` (line 268, 288) — passed to `format_context_string`
- `state["excluded_terms"]` (line 269, 289) — for the guardrail prompt

**Changes**:
1. Replace `rewritten_query` with `original_query` in the generation prompt.
2. Remove `abbreviation_glossary` from `format_context_string` calls. Abbreviation expansion now happens inside search tools (synonym expansion), not at the turn level. The glossary section in the context string is no longer needed.
3. Remove `excluded_terms` from the guardrail prompt. Negation handling is now LLM-driven in the think loop (if the user says "but not Linux", the LLM should not cite Linux evidence).

### 8.8 `kb_profile.py` update

**File**: `backend/app/services/agentic_rag/kb_profile.py`

The KB profile builder references `rag_retrieve` in its filter field descriptions. Update to reference the new search tools' filter parameters (`search_exact`/`search_sparse`/`search_dense` all accept the same `filters` dict).

### 8.9 `streaming.py` update

**File**: `backend/app/services/agentic_rag/streaming.py`

References `rag_retrieve` in event handling. Update to handle new search tool names in progress events. The `AgenticRAGTransformer` (used for clarification path) references `rewritten_query` and `expanded_query` — remove or make optional.

### 8.10 `confidence.py` update

**File**: `backend/app/services/retrieval/confidence.py`

References `rag_retrieve` in confidence level computation. If confidence is now LLM-assessed (not computed), this module may become dead. Audit and either update or remove.

### 8.11 `chat_service.py` update

**File**: `backend/app/services/chat/chat_service.py`

- `_handle_rewritten_query` (line 133) handles `1:` SSE events — remove or make no-op (the event is no longer emitted).
- `_handle_expanded_query` (line 140) handles `eq:` SSE events — remove or make no-op.
- `_handle_answer_rewrite` (lines 174-196) — update to read new `CitationRef` fields from citation metadata.
- `_persist_citations` (lines 381-405) — update to persist new `CitationRef` fields to `MessageCitation` rows.

### 8.12 `api/api_v1/chat/branching.py` and `exports.py` updates

**File**: `backend/app/api/api_v1/chat/branching.py`
- Line 270-271: references `rewritten_query` and `expanded_query` in event mapping. Remove or make optional.

**File**: `backend/app/api/api_v1/chat/exports.py`
- Line 108-109: references `msg.rewritten_query` in markdown export. Keep reading the DB column (it's nullable and historical data has it), but don't fail if it's NULL.

### 8.13 `api/api_v1/search.py` update

**File**: `backend/app/api/api_v1/search.py`

The standalone search endpoint uses `expanded_query` (lines 62, 71-72, 126-132, 147-148, 177-178, 186-187). This endpoint calls `expand_query_node` directly. If `expand_query_node` is removed, this endpoint must either:
- Call `expand_query_suffix` from `abbreviation_service.py` directly (as a replacement), or
- Remove abbreviation expansion from the standalone search path.

**Recommendation**: Replace `expand_query_node` with a direct call to `expand_query_suffix` from `app.services.abbreviation_service`. This is the same pattern the standalone search already uses for abbreviation expansion.

### 8.14 `LAST_ANSWER_EXTRACT_PROMPT` (update)

Update the `citations` field in the extraction prompt to match the new `CitationRef`:

```python
LAST_ANSWER_EXTRACT_PROMPT: str = """\
Extract a structured summary from the assistant answer below. Return valid JSON only matching this schema:
{{
  "summary": "2-3 sentences",
  "key_points": ["..."],
  "data": [{{"label": "...", "value": 123, "unit": "...", "context": "..."}}],
  "citations": [{{"document_id": 1, "citation_kind": "chunk", "chunk_index": 0, "source_tool": "search_dense"}}],
  "chart_option": null or {{ ... }},
  "followups": ["..."],
  "suggestion": "one-line assessment of answer completeness, or empty string",
  "retry_strategy": "widen|narrow|pinpoint|"
}}

For citations: extract the document_id and citation_kind from each [N] citation in the answer. The citation metadata is in the evidence block above the answer.

If the answer contains no numbers, set data to []. If no chart, set chart_option to null. Keep key_points to at most 8 bullets.

Answer:
{answer}
"""
```

---

## 9. Implementation Phases

### Phase 1: Atomic Search Tools (files: 6 new, 1 deleted, 2 modified)

**New files**:
1. `backend/app/services/agentic_rag/tools/search_exact.py`
2. `backend/app/services/agentic_rag/tools/search_sparse.py`
3. `backend/app/services/agentic_rag/tools/search_dense.py`
4. `backend/app/services/agentic_rag/tools/rerank_results.py`
5. `backend/app/services/agentic_rag/tools/graph_expand.py`
6. `backend/app/services/agentic_rag/tools/_search_helpers.py` — shared helpers extracted from `rag_retrieve.py`: `_resolve_filters`, `_expand_synonyms`, `_apply_excluded_terms_filter`, `_pin_filter_matches`

**Deleted files**:
7. `backend/app/services/agentic_rag/tools/rag_retrieve.py`

**Modified files**:
8. `backend/app/services/agentic_rag/tools/__init__.py` — update registry (remove `RagRetrieveTool`, add 5 new tool classes)
9. `backend/app/services/agentic_rag/tools/base.py` — add `prepare_arguments`, `terminate` field

**Note**: Do NOT delete `rag_retrieve.py` until Phase 4 is complete and the graph no longer imports anything from it. During Phase 1-3, `rag_retrieve.py` can exist but be unregistered. The graph still wires `expand_query`/`rewrite_query`/`reflect`/`reflect_final` which internally call functions in `nodes.py` that `rag_retrieve.py` also calls. Only delete `rag_retrieve.py` when no code imports from it.

**Implementation details for each search tool**:

Each search tool follows this pattern (showing `search_dense.py` as example):

```python
"""Dense vector search tool — semantic/conceptual chunk retrieval."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from app.services.agentic_rag.tool_context import ToolContext
from app.services.agentic_rag.tools.base import BaseAgentTool
from app.services.agentic_rag.schemas import CitationRef
from app.services.retrieval.retrieval import dense_search_docs
from app.services.settings_service import get_setting

logger = logging.getLogger(__name__)


class SearchDenseInput(BaseModel):
    query: str = Field(description="Search query for semantic/conceptual matching.")
    kb_ids: List[int] = Field(default_factory=list, description="Knowledge base IDs to search.")
    document_ids: Optional[List[int]] = Field(default=None, description="Restrict to these document IDs.")
    filters: Optional[dict] = Field(default=None, description="Metadata filters: title_contains, file_name_contains, content_type, file_modified_after, file_modified_before, file_created_after, file_created_before.")
    top_k: int = Field(default=20, description="Maximum hits to return.")


class SearchDenseTool(BaseAgentTool):
    name: str = "search_dense"
    description: str = "Dense vector search. Best for semantic/conceptual matching. Returns ranked chunks."
    args_schema: type = SearchDenseInput
    ui_label: str = "Searching (dense)"

    def prepare_arguments(self, args: dict) -> dict:
        """Normalize kb_ids to list of ints."""
        kb_ids = args.get("kb_ids", [])
        if isinstance(kb_ids, (str, int)):
            kb_ids = [int(kb_ids)]
        args["kb_ids"] = [int(k) for k in kb_ids]
        return args

    async def _execute(self, input_obj: SearchDenseInput) -> dict:
        ctx = self.ctx
        if ctx is None:
            return {"ok": False, "result": {}, "error": "No context", "tokens": 0}

        # RBAC: filter kb_ids to those attached to the chat
        from app.services.agentic_rag.tool_context import enforce_rbac
        rbac = enforce_rbac(ctx, kb_ids=input_obj.kb_ids)
        kb_ids = rbac["kb_ids"]
        if not kb_ids and ctx.state is not None:
            kb_ids = ctx.state.get("kb_ids", [])
        if not kb_ids:
            return {"ok": True, "result": {"hits": [], "count": 0}, "error": None, "tokens": 0}

        # Resolve datastore_ids from kb_ids (ToolContext does not carry datastore_ids)
        from app.services.retrieval import get_effective_datastore_ids
        datastore_ids = get_effective_datastore_ids(kb_ids, ctx.org_id, ctx.db) if ctx.db else []

        # Resolve filters to document_ids
        doc_ids = input_obj.document_ids
        if input_obj.filters:
            doc_ids = _resolve_filters(ctx, input_obj.filters, doc_ids)

        # Run synonym expansion (Redis-cached)
        query, extra_queries = await _expand_synonyms(ctx, input_obj.query)

        # Call the low-level retrieval function
        try:
            docs = dense_search_docs(
                query=query,
                kb_ids=kb_ids,
                datastore_ids=datastore_ids,
                db=ctx.db,
                org_id=ctx.org_id,
                top_k=input_obj.top_k,
                doc_ids=doc_ids,
            )
        except Exception as exc:
            logger.warning("[search_dense] failed: %s", exc)
            return {"ok": False, "result": {}, "error": str(exc), "tokens": 0}

        # Convert to hit dicts with CitationRef
        hits = []
        for doc in docs:
            meta = doc.metadata or {}
            hit = {
                "document_id": meta.get("document_id"),
                "chunk_index": meta.get("chunk_index"),
                "page": meta.get("page"),
                "title": meta.get("title", ""),
                "file_name": meta.get("file_name", ""),
                "content": doc.page_content,
                "score": meta.get("score", 0.0),
                "content_hash": meta.get("content_hash", ""),
                "qdrant_point_id": meta.get("qdrant_point_id", ""),
                "citation_ref": {
                    "document_id": meta.get("document_id"),
                    "citation_kind": "chunk",
                    "chunk_index": meta.get("chunk_index"),
                    "page": meta.get("page"),
                    "quoted_text": doc.page_content[:200],
                    "source_tool": "search_dense",
                    "citation_id": "",  # Assigned at finalize time
                },
            }
            hits.append(hit)

        return {
            "ok": True,
            "result": {
                "hits": hits,
                "query_used": query,
                "search_type": "dense",
                "count": len(hits),
            },
            "error": None,
            "tokens": sum(len(h["content"]) for h in hits) // 4,
        }
```

`search_exact.py` and `search_sparse.py` follow the same pattern, calling `exact_search_docs()` and `sparse_search_docs()` respectively.

`rerank_results.py`:

```python
class RerankResultsInput(BaseModel):
    query: str = Field(description="The search query to rerank against.")
    hits: List[dict] = Field(description="Hits from one or more search tools.")
    top_n: Optional[int] = Field(default=None, description="Max hits after reranking. None = no cap.")


class RerankResultsTool(BaseAgentTool):
    name: str = "rerank_results"
    description: str = "Cross-encoder reranker. Deduplicates and reranks hits from search tools."
    args_schema: type = RerankResultsInput
    ui_label: str = "Reranking results"

    async def _execute(self, input_obj: RerankResultsInput) -> dict:
        ctx = self.ctx
        if ctx is None:
            return {"ok": False, "result": {}, "error": "No context", "tokens": 0}
        if not input_obj.hits:
            return {"ok": True, "result": {"hits": [], "count": 0}, "error": None, "tokens": 0}

        from langchain_core.documents import Document as LangchainDocument
        from app.services.retrieval.retrieval import dedup_by_content_hash, semantic_dedup
        from app.services.retrieval.reranker import rerank

        # Convert hits to LangchainDocuments
        docs = []
        for hit in input_obj.hits:
            docs.append(LangchainDocument(
                page_content=hit.get("content", ""),
                metadata={
                    "document_id": hit.get("document_id"),
                    "chunk_index": hit.get("chunk_index"),
                    "page": hit.get("page"),
                    "title": hit.get("title", ""),
                    "file_name": hit.get("file_name", ""),
                    "content_hash": hit.get("content_hash", ""),
                    "qdrant_point_id": hit.get("qdrant_point_id", ""),
                    "citation_ref": hit.get("citation_ref", {}),
                },
            ))

        # Dedup
        docs = dedup_by_content_hash(docs)
        docs = semantic_dedup(docs, threshold=0.95)

        # Rerank
        reranked = rerank(
            query=input_obj.query,
            docs=docs,
            score_threshold=None,  # Use default from settings
            db=ctx.db,
            org_id=ctx.org_id,
        )

        # Apply top_n if specified (no hard cap otherwise)
        if input_obj.top_n is not None:
            reranked = reranked[:input_obj.top_n]

        # Convert back to hit dicts with updated CitationRef
        hits = []
        for doc in reranked:
            meta = doc.metadata or {}
            citation_ref = meta.get("citation_ref", {})
            citation_ref["source_tool"] = "rerank_results"
            hit = {
                "document_id": meta.get("document_id"),
                "chunk_index": meta.get("chunk_index"),
                "page": meta.get("page"),
                "title": meta.get("title", ""),
                "file_name": meta.get("file_name", ""),
                "content": doc.page_content,
                "_reranker_score": meta.get("_reranker_score", 0.0),
                "content_hash": meta.get("content_hash", ""),
                "qdrant_point_id": meta.get("qdrant_point_id", ""),
                "citation_ref": citation_ref,
            }
            hits.append(hit)

        scores = [h.get("_reranker_score", 0) for h in hits]
        return {
            "ok": True,
            "result": {
                "hits": hits,
                "query_used": input_obj.query,
                "input_count": len(input_obj.hits),
                "output_count": len(hits),
                "best_score": max(scores) if scores else 0.0,
            },
            "error": None,
            "tokens": sum(len(h["content"]) for h in hits) // 4,
        }
```

`graph_expand.py`:

```python
class GraphExpandInput(BaseModel):
    kb_ids: List[int] = Field(default_factory=list, description="Knowledge base IDs.")
    seed_document_ids: Optional[List[int]] = Field(default=None, description="Document IDs to use as seeds.")
    seed_chunk_ids: Optional[List[str]] = Field(default=None, description="Qdrant point UUIDs to use as seeds.")
    top_k: int = Field(default=10, description="Maximum expanded chunks.")


class GraphExpandTool(BaseAgentTool):
    name: str = "graph_expand"
    description: str = "Graph expansion via Neo4j. Finds related chunks through entity relationships."
    args_schema: type = GraphExpandInput
    ui_label: str = "Expanding via graph"

    async def _execute(self, input_obj: GraphExpandInput) -> dict:
        ctx = self.ctx
        if ctx is None:
            return {"ok": False, "result": {}, "error": "No context", "tokens": 0}

        from app.services.graph.expand import expand_docs_via_graph
        from langchain_core.documents import Document as LangchainDocument

        # Build seed docs from seed_document_ids or seed_chunk_ids
        seed_docs = _build_seed_docs(ctx, input_obj.seed_document_ids, input_obj.seed_chunk_ids)
        if not seed_docs:
            return {"ok": True, "result": {"hits": [], "count": 0}, "error": None, "tokens": 0}

        try:
            from app.services.retrieval import get_effective_datastore_ids
            datastore_ids = get_effective_datastore_ids(input_obj.kb_ids, ctx.org_id, ctx.db) if ctx.db else []
            expanded = expand_docs_via_graph(
                docs=seed_docs,
                kb_ids=input_obj.kb_ids,
                db=ctx.db,
                org_id=ctx.org_id,
                datastore_ids=datastore_ids,
            )
        except Exception as exc:
            logger.warning("[graph_expand] failed: %s", exc)
            return {"ok": True, "result": {"hits": [], "count": 0}, "error": None, "tokens": 0}

        hits = []
        for doc in expanded[:input_obj.top_k]:
            meta = doc.metadata or {}
            hit = {
                "document_id": meta.get("document_id"),
                "chunk_index": meta.get("chunk_index"),
                "page": meta.get("page"),
                "title": meta.get("title", ""),
                "content": doc.page_content,
                "content_hash": meta.get("content_hash", ""),
                "qdrant_point_id": meta.get("qdrant_point_id", ""),
                "citation_ref": {
                    "document_id": meta.get("document_id"),
                    "citation_kind": "chunk",
                    "chunk_index": meta.get("chunk_index"),
                    "page": meta.get("page"),
                    "quoted_text": doc.page_content[:200],
                    "source_tool": "graph_expand",
                    "citation_id": "",
                },
            }
            hits.append(hit)

        return {
            "ok": True,
            "result": {"hits": hits, "count": len(hits)},
            "error": None,
            "tokens": sum(len(h["content"]) for h in hits) // 4,
        }
```

**Shared helper** (`_resolve_filters`, `_expand_synonyms`): extract these from the current `rag_retrieve.py` into a shared module `backend/app/services/agentic_rag/tools/_search_helpers.py` so all search tools can use them.

### Phase 2: Citation Pipeline (files: 7 modified, 1 new)

**Modified files**:
1. `backend/app/services/agentic_rag/schemas.py` — new `CitationRef`, updated `LastAnswerObject`
2. `backend/app/services/agentic_rag/utils.py` — new `format_context_string`, new `normalize_citations`
3. `backend/app/services/agentic_rag/agent_graph/finalization.py` — evidence block, citation normalization, remove `rewritten_query`/`abbreviation_glossary`/`excluded_terms` dependencies
4. `backend/app/services/chat/chat_service.py` — update `_handle_answer_rewrite` to extract citation fields from new `CitationRef` shape (currently reads `doc["metadata"]["document_id"]` and `doc["metadata"]["chunk_index"]`; must also read `citation_kind`, `section`, etc.)
5. `backend/app/api/api_v1/chat/messages.py` — update `_serialize_messages_with_citations` to include new `CitationRef` fields in the citation dict; also fix siblings endpoint (line 567-570) to include citations (pre-existing bug: branch navigation drops citations)
6. `backend/app/models/chat.py` — `MessageCitation` model: add `citation_kind` column (nullable, for backward compat), add `section`, `start_char`, `end_char`, `start_line`, `end_line`, `match_line`, `source_tool` columns (all nullable). Requires Alembic migration.
7. `frontend/src/components/chat/answer.tsx` — citation rendering per kind

**New file**:
8. `frontend/src/components/chat/__tests__/citation-ref.test.tsx` — citation rendering tests

**Alembic migration**: `backend/alembic/versions/XXXX_add_citation_ref_fields.py` — add nullable columns to `message_citations` table. Old rows have NULL for new columns, which the frontend handles via defensive parsing.

### Phase 3: Execution Layer (files: 6 modified)

**Modified files**:
1. `backend/app/services/agentic_rag/agent_graph/tooling.py` — `isError` pattern, `_merge_observation_docs` (add search tools to `SEARCH_TOOLS` set), total budget, remove `route_tool`/`route_reflect_final`
2. `backend/app/services/agentic_rag/agent_graph/helpers.py` — new `_tool_call_budget`, remove `_correction_hints`
3. `backend/app/core/settings_registry.py` — add new settings FIRST (before any code reads them), then remove orphaned settings
4. `backend/app/services/agentic_rag/tools/__init__.py` — deferred tool gating
5. `backend/app/services/agentic_rag/agent_graph/observations.py` — rename `_tried_rag_retrieve_queries` → `_tried_search_queries`, update skip lists and compaction
6. `backend/app/services/agentic_rag/prompts.py` — remove `TOOL_CORRECTION_PROMPT` (note: full prompt rewrite is Phase 6, but this prompt must be removed here since `_correct_tool_args` is deleted in this phase)

### Phase 4: Graph Changes (files: 5 modified, 2 new, code deleted in 3 files)

**New files**:
1. `backend/app/services/agentic_rag/agent_graph/sufficiency.py` — `sufficiency_check_node`, `route_sufficiency`
2. `backend/app/services/agentic_rag/agent_graph/execution_check.py` — moved `_verify_execution`, `_build_execution_summary` and sub-helpers (rewritten for atomic tools)

**Modified files**:
3. `backend/app/services/agentic_rag/agent_graph/build.py` — new graph topology, remove expand/rewrite/reflect/reflect_final nodes and edges
4. `backend/app/services/agentic_rag/agent_graph/load_context.py` — remove expand/rewrite calls, update reset list
5. `backend/app/services/agentic_rag/agent_graph/__init__.py` — remove deleted re-exports, add new ones
6. `backend/app/services/agentic_rag/agent_graph/thinking.py` — update `route_think` to return `"finalize"` instead of `"reflect_final"`
7. `backend/app/services/agentic_rag/agent_graph/tooling.py` — remove `route_tool`, `route_reflect_final`; update imports of `_verify_execution`/`_build_execution_summary` to use `execution_check.py`

**Deleted code** (functions within files, not entire files):
8. `backend/app/services/agentic_rag/agent_graph/reflection.py` — remove `reflect_node`, `reflect_final_node`, `route_reflect_final` (keep `answer_scoring_node`, `clarify_interrupt_node`; move `_verify_execution`/`_build_execution_summary` and sub-helpers to `execution_check.py`)
9. `backend/app/services/agentic_rag/nodes.py` — remove `expand_query_node`, `rewrite_query_node`, `reranking_node`, `filter_node`, `_elbow_cut`, `dense_retrieval_node`, `sparse_retrieval_node`, `exact_retrieval_node`, `collapse_same_title_versions`, `_rrf_fuse_legs`, `_bow_jaccard`, `_mmr_diverse`, `merge_node`, `neo4j_expansion_node`, `_enrich_with_modified_at`, `_retrieval_confidence_level`, `_resolve_eval_kwargs`, `_final_confidence_level` (keep `_agent_step`, `history_to_text`, `select_recent_history`, `_messages_to_conversation_text`, `_get_llm`, `_safe_writer`, `_collect_provenance_sources`, `_lookup_cited_titles`, `_extract_negation_terms`, `_content_contains_exclusion`, `answer_evaluation_node`)
10. `backend/app/services/agentic_rag/agent_graph/observations.py` — rename `_tried_rag_retrieve_queries` to `_tried_search_queries` (generic over all search tools), update `_non_retrieval_observations_text` skip list to include `search_exact`/`search_sparse`/`search_dense`/`rerank_results`/`graph_expand`, update `_compact_observations` to compact docs from any search tool

### Phase 5: State Changes (files: 2 modified)

**Modified files**:
1. `backend/app/services/agentic_rag/graph_state.py` — add/remove fields
2. `backend/app/services/agentic_rag/schemas.py` — remove `suggested_legs`, `QueryIntent`

### Phase 6: Prompts and Backend Linkages (files: 12+ modified)

**Modified files**:
1. `backend/app/services/agentic_rag/prompts.py` — rewrite `PLAN_SYSTEM_PROMPT`, `THINK_SYSTEM_PROMPT`, `FINALIZE_ANSWER_PROMPT`, `FINALIZE_GUARDRAIL_PROMPT`, `LAST_ANSWER_EXTRACT_PROMPT`; remove `REWRITE_INTENT_SUFFIX`, `RETRIEVAL_REWRITE_PROMPT`
2. `backend/app/services/agentic_rag/agent_graph/planning.py` — remove precomputed `rag_retrieve` calls, remove `query_intent`/`suggested_legs` deps
3. `backend/app/services/agentic_rag/kb_profile.py` — update filter field descriptions
4. `backend/app/services/agentic_rag/streaming.py` — remove `rewritten_query`/`expanded_query` refs, update tool names
5. `backend/app/services/retrieval/confidence.py` — audit and update or remove
6. `backend/app/services/chat/chat_service.py` — remove `1:`/`eq:` handlers, update `_handle_answer_rewrite`
7. `backend/app/api/api_v1/chat/branching.py` — remove `rewritten_query`/`expanded_query` refs, fix `done:` prefix
8. `backend/app/api/api_v1/chat/exports.py` — handle nullable `rewritten_query`
9. `backend/app/api/api_v1/search.py` — replace `expand_query_node` with `expand_query_suffix`
10. `backend/app/services/agentic_rag/__init__.py` — update module docstring
11. `.env.example` — remove old settings, add new settings
12. `docs/FEATURES.md` and `docs/enterprise-agent/*.md` — update references

### Phase 7: Tests (files: 16+ modified, 2 new)

**Modified test files** (all reference `rag_retrieve`, removed state keys, or removed nodes):
1. `backend/tests/test_agent_loop.py` — update tool registry, remove rag_retrieve references, update `TestTriedRagRetrieveQueries`
2. `backend/tests/test_agent_loop_budget.py` — update budget assertions for new settings
3. `backend/tests/test_agent_state_integrity.py` — update state field assertions
4. `backend/tests/test_retrieval_filtering.py` — update tool references
5. `backend/tests/test_retrieval_filtering_e2e.py` — update tool references
6. `backend/tests/test_rag_retrieve_ladder.py` — delete or rewrite for atomic tools (relaxation ladder no longer exists)
7. `backend/tests/test_conditional_dense.py` — update for atomic tools (conditional dense fast-accept is LLM-driven now)
8. `backend/tests/test_sort_rerank_order.py` — update for `rerank_results` tool
9. `backend/tests/test_synonym_rrf.py` — update (synonym RRF moves into search tools)
10. `backend/tests/test_negation_filter.py` — update (negation handling moves to search tools or LLM)
11. `backend/tests/test_behavioural_transcripts.py` — update state field and node references
12. `backend/tests/test_query_intent.py` — delete (QueryIntent schema removed)
13. `backend/tests/test_abbr_e2e.py` — update (expand_query_node removed; abbreviation expansion moves into search tools)
14. `backend/tests/test_abbr_quality.py` — update (same)
15. `backend/tests/test_search_abbr_isolation.py` — update (same)
16. `backend/tests/test_settings_phase5.py` — update settings references
17. `backend/tests/test_kb_tools.py` — update patch data for new tool names
18. `frontend/src/components/chat/__tests__/answer.test.tsx` — update citation tests

**New files**:
19. `backend/tests/test_atomic_search_tools.py` — tests for search_exact, search_sparse, search_dense, rerank_results, graph_expand
20. `backend/tests/test_citation_ref.py` — tests for new CitationRef schema and normalize_citations

### Phase 8: Frontend (files: 5 modified, 1 new)

**Modified files**:
1. `frontend/src/components/chat/answer.tsx` — citation rendering per `citation_kind`, update `Citation`/`CitationMetadata` interfaces, update `CitationLink` to handle `[N]` bare-bracket format (currently only handles `[N](N)` markdown links), update `buildFetchPairs`/`fetchKbBatch`/`fetchGenericBatch` if citation field names change
2. `frontend/src/app/dashboard/chat/[id]/page.tsx` — update `Message`/`Citation` interfaces (remove `rewrittenQuery`/`expandedQuery` if desired, add `citation_kind`), update `r:` event handler to map new citation shape, remove `1:`/`eq:` event handlers (or keep for backward compat with historical messages)
3. `frontend/src/components/chat/agentic-progress.tsx` — update `TOOL_ICONS` map (remove `rag_retrieve`, add `search_exact`/`search_sparse`/`search_dense`/`rerank_results`/`graph_expand`), update `NODE_PHASE` map (remove `rewrite_query`/`reflect`/`reflect_final`, add `sufficiency_check`), update result summary logic for new `hits` shape
4. `frontend/src/components/chat/__tests__/answer.test.tsx` — update citation test mocks and assertions for new format
5. `frontend/src/lib/utils.ts` — update `cleanChunkText` if citation text field name changes

**New file**:
6. `frontend/src/types/chat.ts` — shared SSE event type definitions (currently all inline casts in `processStreamLine`; create a single typed interface for `r:`, `tc:`, `to:`, `la:`, `p:`, `4:` events)

**Frontend rendering change for `[N]` bare brackets**: The current `CitationLink` component is invoked by `react-markdown` only for `<a>` elements (markdown links `[N](N)`). Bare `[N]` renders as plain text. Two options:
- **Option A (recommended)**: Pre-process `parsedContent.answerText` in `answer.tsx` to convert `[N]` to `[N](N)` markdown links before passing to `react-markdown`. This preserves the existing `CitationLink` component.
- **Option B**: Replace `react-markdown` rendering with a custom parser that turns `[N]` into `CitationLink` buttons directly. More invasive.

Choose Option A for minimal diff. The pre-processing step is a regex replacement: `text.replace(/\[(\d+)\](?!\()/g, '[$1]($1)')`.

---

## 10. Testing

### 10.1 Test strategy

Each phase has its own verification step. Tests run inside the Docker container (`docker exec rag-web-ui-backend-1 pytest`) per AGENTS.md.

**Order of testing**:
1. Unit tests for each new search tool (Phase 1)
2. Unit tests for `CitationRef` schema and `normalize_citations` (Phase 2)
3. Unit tests for execution layer changes (Phase 3)
4. Integration tests for graph topology (Phase 4)
5. Full agent loop tests (after all phases)
6. Frontend citation rendering tests (Phase 8)

### 10.2 Phase 1 tests — `test_atomic_search_tools.py`

**File**: `backend/tests/test_atomic_search_tools.py` (new)

```python
"""Tests for atomic search tools (search_exact, search_sparse, search_dense, rerank_results, graph_expand)."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from app.services.agentic_rag.tools.search_exact import SearchExactTool, SearchExactInput
from app.services.agentic_rag.tools.search_sparse import SearchSparseTool, SearchSparseInput
from app.services.agentic_rag.tools.search_dense import SearchDenseTool, SearchDenseInput
from app.services.agentic_rag.tools.rerank_results import RerankResultsTool, RerankResultsInput
from app.services.agentic_rag.tools.graph_expand import GraphExpandTool, GraphExpandInput


class TestSearchExactTool:
    def test_schema_has_required_fields(self):
        schema = SearchExactInput.model_json_schema()
        assert "query" in schema["required"]
        assert "kb_ids" in schema["properties"]
        assert "document_ids" in schema["properties"]
        assert "filters" in schema["properties"]
        assert "top_k" in schema["properties"]

    def test_returns_empty_when_no_kb_ids(self):
        ctx = MagicMock()
        ctx.enforce_rbac.return_value = []
        tool = SearchExactTool()
        tool.ctx = ctx
        import asyncio
        result = asyncio.run(tool.arun({"query": "test", "kb_ids": []}))
        assert result["ok"] is True
        assert result["result"]["hits"] == []
        assert result["result"]["count"] == 0

    @patch("app.services.agentic_rag.tools.search_exact.exact_search_docs")
    def test_returns_hits_with_citation_ref(self, mock_search):
        from langchain_core.documents import Document
        mock_search.return_value = [
            Document(page_content="test content", metadata={
                "document_id": 1, "chunk_index": 0, "page": 1,
                "title": "Test Doc", "file_name": "test.pdf",
                "content_hash": "abc123", "qdrant_point_id": "uuid-1",
            })
        ]
        ctx = MagicMock()
        ctx.enforce_rbac.return_value = [1]
        ctx.datastore_ids = []
        ctx.db = MagicMock()
        ctx.org_id = 1
        tool = SearchExactTool()
        tool.ctx = ctx
        import asyncio
        result = asyncio.run(tool.arun({"query": "test", "kb_ids": [1]}))
        assert result["ok"] is True
        assert len(result["result"]["hits"]) == 1
        hit = result["result"]["hits"][0]
        assert hit["document_id"] == 1
        assert hit["citation_ref"]["citation_kind"] == "chunk"
        assert hit["citation_ref"]["source_tool"] == "search_exact"
        assert hit["citation_ref"]["document_id"] == 1


class TestSearchSparseTool:
    def test_schema_matches_dense(self):
        sparse_schema = SearchSparseInput.model_json_schema()
        dense_schema = SearchDenseInput.model_json_schema()
        assert set(sparse_schema["properties"].keys()) == set(dense_schema["properties"].keys())


class TestSearchDenseTool:
    def test_prepare_arguments_normalizes_kb_ids(self):
        tool = SearchDenseTool()
        result = tool.prepare_arguments({"kb_ids": "5", "query": "test"})
        assert result["kb_ids"] == [5]

    def test_prepare_arguments_handles_int(self):
        tool = SearchDenseTool()
        result = tool.prepare_arguments({"kb_ids": 5, "query": "test"})
        assert result["kb_ids"] == [5]


class TestRerankResultsTool:
    def test_empty_hits_returns_empty(self):
        tool = RerankResultsTool()
        import asyncio
        result = asyncio.run(tool.arun({"query": "test", "hits": []}))
        assert result["ok"] is True
        assert result["result"]["hits"] == []

    def test_no_top_n_cap(self):
        """Verify that top_n=None returns all hits passing threshold."""
        # This test verifies the user's explicit requirement:
        # "no hard top-k cap before generation is authorized"
        # ... (mock reranker to return all 15 hits)
        pass

    def test_top_n_caps_results(self):
        """When top_n is specified, results are capped."""
        # ... (mock reranker, pass 15 hits, top_n=5, verify 5 returned)
        pass

    def test_citation_ref_source_tool_updated(self):
        """Reranked hits should have citation_ref.source_tool = 'rerank_results'."""
        pass


class TestGraphExpandTool:
    def test_no_seeds_returns_empty(self):
        tool = GraphExpandTool()
        import asyncio
        result = asyncio.run(tool.arun({"kb_ids": [1], "seed_document_ids": None, "seed_chunk_ids": None}))
        assert result["ok"] is True
        assert result["result"]["hits"] == []

    def test_failure_is_non_fatal(self):
        """Graph expansion failures return empty hits, not errors."""
        pass


class TestToolRegistry:
    def test_build_tools_returns_atomic_search_tools(self):
        from app.services.agentic_rag.tools import build_tools
        ctx = MagicMock()
        ctx.state = {}
        tools = build_tools(ctx)
        names = {t.name for t in tools}
        assert "search_exact" in names
        assert "search_sparse" in names
        assert "search_dense" in names
        assert "rerank_results" in names
        assert "graph_expand" in names
        assert "rag_retrieve" not in names

    def test_applicable_tools_excludes_rerank_without_search(self):
        from app.services.agentic_rag.tools import applicable_tools
        ctx = MagicMock()
        ctx.state = {"tool_call_counts": {}}
        tools = applicable_tools(ctx)
        names = {t.name for t in tools}
        assert "rerank_results" not in names
        assert "graph_expand" not in names

    def test_applicable_tools_includes_rerank_after_search(self):
        from app.services.agentic_rag.tools import applicable_tools
        ctx = MagicMock()
        ctx.state = {"tool_call_counts": {"search_dense": 1}}
        tools = applicable_tools(ctx)
        names = {t.name for t in tools}
        assert "rerank_results" in names
        assert "graph_expand" in names
```

### 10.3 Phase 2 tests — `test_citation_ref.py`

**File**: `backend/tests/test_citation_ref.py` (new)

```python
"""Tests for the new CitationRef schema and normalize_citations."""

import pytest
from app.services.agentic_rag.schemas import CitationRef
from app.services.agentic_rag.utils import normalize_citations


class TestCitationRefSchema:
    def test_chunk_citation(self):
        ref = CitationRef(
            document_id=1,
            citation_kind="chunk",
            chunk_index=5,
            page=3,
            quoted_text="test",
            source_tool="search_dense",
        )
        assert ref.citation_kind == "chunk"
        assert ref.chunk_index == 5

    def test_file_citation(self):
        ref = CitationRef(
            document_id=1,
            citation_kind="file",
            source_tool="kb_read",
        )
        assert ref.citation_kind == "file"
        assert ref.chunk_index is None

    def test_section_citation(self):
        ref = CitationRef(
            document_id=1,
            citation_kind="section",
            section="Topics Covered",
            start_char=100,
            end_char=500,
            source_tool="kb_read",
        )
        assert ref.citation_kind == "section"
        assert ref.section == "Topics Covered"

    def test_range_citation(self):
        ref = CitationRef(
            document_id=1,
            citation_kind="range",
            start_char=100,
            end_char=200,
            start_line=5,
            end_line=10,
            source_tool="kb_read",
        )
        assert ref.citation_kind == "range"
        assert ref.start_line == 5

    def test_grep_citation(self):
        ref = CitationRef(
            document_id=1,
            citation_kind="grep",
            match_line=42,
            quoted_text="matched text",
            source_tool="kb_grep",
        )
        assert ref.citation_kind == "grep"
        assert ref.match_line == 42

    def test_coerce_kb_label(self):
        """CitationRef should coerce 'KB-2' to 2."""
        ref = CitationRef(document_id="KB-2", citation_kind="chunk", chunk_index=0)
        assert ref.document_id == 2


class TestNormalizeCitations:
    def test_empty_answer(self):
        answer, cited = normalize_citations("", [])
        assert answer == ""
        assert cited == []

    def test_no_evidence_strips_citations(self):
        answer, cited = normalize_citations("Hello [E1] world", [])
        assert "[E1]" not in answer
        assert cited == []

    def test_renumbers_by_first_appearance(self):
        evidence = [
            {"citation_id": "E1", "citation_ref": {"document_id": 1}},
            {"citation_id": "E2", "citation_ref": {"document_id": 2}},
            {"citation_id": "E3", "citation_ref": {"document_id": 3}},
        ]
        answer = "Foo [E3] bar [E1] baz [E2]"
        result, cited = normalize_citations(answer, evidence)
        # E3 cited first → [1], E1 cited second → [2], E2 cited third → [3]
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result
        assert len(cited) == 3
        assert cited[0]["citation_ref"]["document_id"] == 3  # E3
        assert cited[1]["citation_ref"]["document_id"] == 1  # E1
        assert cited[2]["citation_ref"]["document_id"] == 2  # E2

    def test_strips_out_of_range(self):
        evidence = [{"citation_id": "E1", "citation_ref": {"document_id": 1}}]
        answer = "Foo [E1] bar [E99] baz"
        result, cited = normalize_citations(answer, evidence)
        assert "[E99]" not in result
        assert len(cited) == 1

    def test_protects_code_blocks(self):
        evidence = [{"citation_id": "E1", "citation_ref": {"document_id": 1}}]
        answer = "Code: `arr[E1]` and text [E1]"
        result, cited = normalize_citations(answer, evidence)
        # The [E1] inside backticks should be preserved as-is
        assert "`arr[E1]`" in result or "`arr[1]`" in result
        # The [E1] in text should be renumbered
        assert len(cited) == 1

    def test_strips_reasoning_citations(self):
        evidence = [{"citation_id": "E1", "citation_ref": {"document_id": 1}}]
        answer = "<think>Let me check [E1]</think>\nAnswer: foo [E1]"
        result, cited = normalize_citations(answer, evidence)
        # Citation in reasoning should be stripped
        assert "<think>" in result
        # Citation in answer should be preserved
        assert len(cited) == 1

    def test_multiple_citations_same_evidence(self):
        evidence = [{"citation_id": "E1", "citation_ref": {"document_id": 1}}]
        answer = "Foo [E1]. Bar [E1]. Baz [E1]."
        result, cited = normalize_citations(answer, evidence)
        # Same evidence cited 3 times → still only 1 in cited list
        assert len(cited) == 1
        # All three [E1] should become [1]
        assert result.count("[1]") == 3
```

### 10.4 Phase 3 tests — execution layer

Update `test_agent_loop.py`:

```python
class TestExecutionLayer:
    def test_iserror_pattern_no_correction_llm(self):
        """Failed tool calls return errors to the LLM, not to a correction LLM."""
        # Verify _correct_tool_args is no longer called
        pass

    def test_total_tool_budget_forces_finalize(self):
        """When total_tool_calls reaches AGENT_TOTAL_TOOL_BUDGET, force_finalize is set."""
        pass

    def test_transient_error_retries(self):
        """Transient errors (timeout, network) retry with backoff."""
        pass

    def test_argument_error_no_retry(self):
        """Argument errors do NOT retry — they go back to the LLM."""
        pass


class TestToolCallBudget:
    def test_new_budget_includes_search_tools(self):
        from app.services.agentic_rag.agent_graph.helpers import _tool_call_budget
        db = MagicMock()
        budget = _tool_call_budget(db, org_id=1)
        assert "search_exact" in budget
        assert "search_sparse" in budget
        assert "search_dense" in budget
        assert "rerank_results" in budget
        assert "graph_expand" in budget
        assert "rag_retrieve" not in budget
```

### 10.5 Phase 4 tests — graph topology

```python
class TestGraphTopology:
    def test_graph_has_sufficiency_check_node(self):
        from app.services.agentic_rag.agent_graph.build import build_agent_graph
        ctx = MagicMock()
        ctx.redis_memory = None
        graph = build_agent_graph(ctx)
        # Verify sufficiency_check is in the graph
        assert "sufficiency_check" in graph.nodes

    def test_graph_does_not_have_expand_query(self):
        from app.services.agentic_rag.agent_graph.build import build_agent_graph
        ctx = MagicMock()
        ctx.redis_memory = None
        graph = build_agent_graph(ctx)
        assert "expand_query" not in graph.nodes

    def test_graph_does_not_have_rewrite_query(self):
        # ... same pattern
        pass

    def test_graph_does_not_have_reflect(self):
        # ... same pattern
        pass

    def test_route_sufficiency_to_finalize(self):
        from app.services.agentic_rag.agent_graph.sufficiency import route_sufficiency
        state = {"sufficient": True}
        assert route_sufficiency(state) == "finalize"

    def test_route_sufficiency_to_think(self):
        from app.services.agentic_rag.agent_graph.sufficiency import route_sufficiency
        state = {"sufficient": False, "iteration": 1}
        # Mock max_iter and wall clock
        assert route_sufficiency(state) == "think"
```

### 10.6 Phase 7 tests — full agent loop

After all phases, run the full test suite:

```bash
docker exec rag-web-ui-backend-1 pytest tests/test_agent_loop.py -v
docker exec rag-web-ui-backend-1 pytest tests/test_atomic_search_tools.py -v
docker exec rag-web-ui-backend-1 pytest tests/test_citation_ref.py -v
docker exec rag-web-ui-backend-1 pytest tests/test_retrieval_filtering.py -v
docker exec rag-web-ui-backend-1 pytest tests/test_retrieval_filtering_e2e.py -v
```

Update `test_agent_loop.py::TestToolRegistry::test_build_tools_returns_all_tools`:

```python
def test_build_tools_returns_all_tools(self):
    ctx = MagicMock()
    ctx.state = {}
    tools = build_tools(ctx)
    names = {t.name for t in tools}
    expected = {
        "search_exact", "search_sparse", "search_dense",
        "rerank_results", "graph_expand",
        "kb_search_documents", "kb_metadata", "kb_outline",
        "current_datetime", "kb_read", "kb_grep",
        "file_read", "file_summarize", "file_extract_table",
        "code_execute", "chart_generate", "summarize_answer", "extract_data",
    }
    assert names == expected
```

Update `test_agent_loop.py::TestTriedRagRetrieveQueries` — remove or rename since `rag_retrieve` no longer exists:

```python
class TestTriedSearchQueries:
    def test_dedups_and_preserves_order(self):
        observations = [
            Observation(tool="search_dense", arguments={"query": "race condition"}, result={"hits": [], "count": 0}),
            Observation(tool="search_dense", arguments={"query": "mutual exclusion"}, result={"hits": [], "count": 0}),
            Observation(tool="search_dense", arguments={"query": "race condition"}, result={"hits": [], "count": 0}),
        ]
        # ... verify dedup
```

### 10.7 Frontend tests

Update `frontend/src/components/chat/__tests__/answer.test.tsx`:

```typescript
describe("CitationRef rendering", () => {
  it("renders chunk citation with chunk highlight", () => {
    // ... test chunk kind
  });

  it("renders file citation without chunk highlight", () => {
    // ... test file kind
  });

  it("renders section citation with section name tooltip", () => {
    // ... test section kind
  });

  it("renders range citation with line range", () => {
    // ... test range kind
  });

  it("renders grep citation with line number", () => {
    // ... test grep kind
  });
});
```

---

## 11. Validation

### 11.1 Pre-implementation validation

Before starting implementation, verify the current test suite passes:

```bash
docker exec rag-web-ui-backend-1 pytest tests/test_agent_loop.py -v
```

Expected: `67 passed, 1 warning` (current baseline).

### 11.2 Per-phase validation

After each phase, run the relevant tests:

| Phase | Test command | Expected result |
|---|---|---|
| 1 (search tools) | `pytest tests/test_atomic_search_tools.py -v` | All new tests pass |
| 2 (citations) | `pytest tests/test_citation_ref.py -v` | All new tests pass |
| 3 (execution) | `pytest tests/test_agent_loop.py::TestExecutionLayer -v` | All new tests pass |
| 4 (graph) | `pytest tests/test_agent_loop.py::TestGraphTopology -v` | All new tests pass |
| 5 (state) | `pytest tests/test_agent_loop.py -v` | Updated tests pass |
| 6 (prompts) | `pytest tests/test_agent_loop.py -v` | All tests pass |
| 7 (full suite) | `pytest tests/ -v` | All tests pass |
| 8 (frontend) | `docker exec rag-web-ui-frontend-1 npm run test:ci` | All tests pass |

### 11.3 End-to-end validation

After all phases, run end-to-end tests with real queries:

1. **Named-document query**: "What is in the latest weekly update?"
   - Expected: agent calls `kb_search_documents` → `kb_read` (not `search_dense`)
   - Verify citation shows `citation_kind="file"` or `"section"`

2. **Conceptual query**: "How does the API gateway handle authentication?"
   - Expected: agent calls `search_dense` → `rerank_results` → finalize
   - Verify citation shows `citation_kind="chunk"`

3. **Exact-lookup query**: "Find all references to CONFIG_REDIS_URL"
   - Expected: agent calls `kb_grep` → finalize
   - Verify citation shows `citation_kind="grep"`

4. **Aggregate query**: "How many weekly updates were prepared this year? Table by month."
   - Expected: agent calls `kb_metadata` → `kb_search_documents` (metadata_only) → batch `kb_read`/`extract_data` → `chart_generate`
   - Verify chart is produced from accumulated data

5. **Multi-search query**: "Compare encryption methods in satellite and fiber optic communications"
   - Expected: agent calls `search_dense` (satellite) + `search_dense` (fiber) in parallel → `rerank_results` → finalize
   - Verify citations from both sub-queries appear

6. **Temporal query**: "What is the latest weekly update?"
   - Expected: agent calls `current_datetime` → `kb_search_documents` with date sort → compares title dates → selects latest
   - Verify the correct (latest by content date, not file mtime) document is cited

### 11.4 Performance validation

Measure before and after:

| Metric | Measurement method |
|---|---|
| Latency (simple query) | Time from user send to first token |
| Latency (complex query) | Time from user send to first token |
| Tool calls per query | Count from `tool_call_counts` state |
| Token cost per query | Sum of `tokens` from all observations |
| Failure rate | Percentage of queries that error or produce empty answers |

Run 10 queries of each type (named-document, conceptual, exact, aggregate, multi-search, temporal) and compare.

### 11.5 Citation correctness validation

For each end-to-end query, verify:
1. Every `[N]` citation in the answer maps to a real evidence item.
2. The `CitationRef` for each cited evidence has the correct `document_id`.
3. The `citation_kind` matches the tool that produced the evidence.
4. No fabricated citations (citations to evidence not in the list).
5. No missing citations (factual statements without citations).

---

## 12. Benchmarks

### 12.1 Benchmark query set

Create a benchmark file: `backend/tests/benchmark_queries.py`

```python
BENCHMARK_QUERIES = [
    # Named-document queries
    {"id": "nd1", "query": "What is in the latest weekly update?", "type": "named_document"},
    {"id": "nd2", "query": "Show me the Q3 2026 financial report", "type": "named_document"},

    # Conceptual queries
    {"id": "c1", "query": "How does the API gateway handle authentication?", "type": "conceptual"},
    {"id": "c2", "query": "What are the main security risks in microservices?", "type": "conceptual"},

    # Exact-lookup queries
    {"id": "el1", "query": "Find all references to CONFIG_REDIS_URL", "type": "exact_lookup"},
    {"id": "el2", "query": "Where is the function calculate_risk_score defined?", "type": "exact_lookup"},

    # Multi-part comparisons
    {"id": "mc1", "query": "Compare encryption methods in satellite and fiber optic communications", "type": "comparison"},
    {"id": "mc2", "query": "What is the difference between Q3 and Q4 revenue?", "type": "comparison"},

    # Aggregate queries
    {"id": "a1", "query": "How many weekly updates were prepared this year? Table by month with topics.", "type": "aggregate"},
    {"id": "a2", "query": "Count all documents by content type and show as a chart.", "type": "aggregate"},

    # Long-document queries
    {"id": "ld1", "query": "What are the key risks listed in the audit report?", "type": "long_document"},
    {"id": "ld2", "query": "Summarize the compliance section of the security policy.", "type": "long_document"},

    # Temporal queries
    {"id": "t1", "query": "What is the most recent weekly update?", "type": "temporal"},
    {"id": "t2", "query": "Show me documents from last month", "type": "temporal"},
]
```

### 12.2 Benchmark metrics

For each benchmark query, record:

```python
{
    "query_id": "nd1",
    "query": "What is in the latest weekly update?",
    "type": "named_document",
    "tool_calls": ["kb_search_documents", "kb_read"],
    "tool_call_count": 2,
    "total_tokens": 1500,
    "latency_ms": 3200,
    "answer_length": 450,
    "citation_count": 2,
    "citation_kinds": ["file", "section"],
    "correct_document_selected": True,  # Manual verification
    "answer_quality_score": 85,  # From answer_scoring
    "errors": [],
}
```

### 12.3 Before/after comparison

Run the benchmark suite on the current `rag_retrieve`-based system (before implementation) and on the new atomic-tools system (after implementation). Compare:

| Metric | Before (composite) | After (atomic) | Delta |
|---|---|---|---|
| Named-document accuracy | TBD | TBD | |
| Conceptual accuracy | TBD | TBD | |
| Exact-lookup accuracy | TBD | TBD | |
| Aggregate accuracy | TBD | TBD | |
| Citation correctness | TBD | TBD | |
| Avg latency (simple) | TBD | TBD | |
| Avg latency (complex) | TBD | TBD | |
| Avg tool calls | TBD | TBD | |
| Avg token cost | TBD | TBD | |
| Failure rate | TBD | TBD | |

---

## 13. Migration

### 13.1 Breaking change declaration

This is a breaking change. No compatibility layers, no legacy paths. The following are removed without deprecation:

- `rag_retrieve` tool
- `expand_query` graph node
- `rewrite_query` graph node
- `reflect` graph node
- `reflect_final` graph node
- `QueryIntent` schema
- `suggested_legs` field in `Subtask`
- `TOOL_CORRECTION_PROMPT`
- `REWRITE_INTENT_SUFFIX`
- `RETRIEVAL_REWRITE_PROMPT`
- `_correct_tool_args` function
- `_correction_hints` function
- `AGENT_MAX_RETRIEVALS` setting
- State keys: `dense_docs`, `sparse_docs`, `exact_docs`, `graph_docs`, `leg_results`, `failed_legs`, `leg_doc_counts`, `all_scored_docs`, `retrieval_confidence`, `expanded_query`, `abbreviation_glossary`, `rewritten_query`, `resolution_provenance`, `query_intent`, `excluded_terms`, `adaptive_reran`, `graph_expansion_done`, `reflection_final`, `tool_call_count`, `precomputed_tool_calls`

### 13.2 Database migration

**`messages` table**: No schema changes to `messages` itself. The `tool_calls` JSON column stores new tool names (`search_exact`, `search_sparse`, etc.) as different JSON values. Old messages with `rag_retrieve` tool calls remain readable but won't be re-executed. The `rewritten_query` and `expanded_query` columns remain (nullable) — no longer written by the agent pipeline, but historical data is preserved.

**`message_citations` table**: Add nullable columns for new `CitationRef` fields:
- `citation_kind VARCHAR(32)` (default `'chunk'` for old rows)
- `section VARCHAR(255)`
- `start_char INT`
- `end_char INT`
- `start_line INT`
- `end_line INT`
- `match_line INT`
- `source_tool VARCHAR(64)`

**Alembic migration**: `backend/alembic/versions/XXXX_add_citation_ref_fields.py`

```python
def upgrade():
    op.add_column('message_citations', sa.Column('citation_kind', sa.String(32), nullable=True))
    op.add_column('message_citations', sa.Column('section', sa.String(255), nullable=True))
    op.add_column('message_citations', sa.Column('start_char', sa.Integer, nullable=True))
    op.add_column('message_citations', sa.Column('end_char', sa.Integer, nullable=True))
    op.add_column('message_citations', sa.Column('start_line', sa.Integer, nullable=True))
    op.add_column('message_citations', sa.Column('end_line', sa.Integer, nullable=True))
    op.add_column('message_citations', sa.Column('match_line', sa.Integer, nullable=True))
    op.add_column('message_citations', sa.Column('source_tool', sa.String(64), nullable=True))
    # Backfill old rows
    op.execute("UPDATE message_citations SET citation_kind = 'chunk' WHERE citation_kind IS NULL")

def downgrade():
    op.drop_column('message_citations', 'source_tool')
    op.drop_column('message_citations', 'match_line')
    op.drop_column('message_citations', 'end_line')
    op.drop_column('message_citations', 'start_line')
    op.drop_column('message_citations', 'end_char')
    op.drop_column('message_citations', 'start_char')
    op.drop_column('message_citations', 'section')
    op.drop_column('message_citations', 'citation_kind')
```

The `last_answer_object` JSON in stored messages will have the old `CitationRef(document_id, chunk_index)` shape. The frontend should handle both shapes during the transition period (old messages with old citations, new messages with new citations). This is not a "compatibility layer" — it's defensive parsing of historical data.

### 13.2.1 SSE event migration

**Removed SSE events** (no longer emitted by the agent):
- `1:` (rewritten_query) — `chat_service.py:_handle_rewritten_query` becomes a no-op or is removed. Frontend `page.tsx` handler for `1:` can be removed or kept as a no-op for backward compat.
- `eq:` (expanded_query) — same treatment.

**Modified SSE events**:
- `r:` (answer_rewrite) — citation payload shape changes from `{page_content, metadata: {document_id, chunk_index, ...}}` to `{page_content, metadata: {citation_ref: {document_id, citation_kind, chunk_index, section, ...}, ...}}`. Frontend `page.tsx` `r:` handler must map the new shape.
- `tc:`/`to:`/`tr:` (tool_call/tool_observation/tool_retry) — tool names change. Frontend `agentic-progress.tsx` `TOOL_ICONS` and result summary logic must handle new tool names and `hits` result shape.
- `p:` (progress) — phase names change (remove `dense_retrieval`/`sparse_retrieval`/`exact_retrieval`/`reflect_final`/`sufficiency_check` from `rag_retrieve`; add `search_exact`/`search_sparse`/`search_dense`/`rerank_results`/`graph_expand`/`sufficiency_check`).

**Pre-existing bug to fix**: `branching.py:353` emits `done:` prefix but frontend only handles `d:`. Either change backend to emit `d:` or add `done:` handling in frontend.

### 13.2.2 `.env.example` and docs update

Update `.env.example`:
- Remove `AGENT_MAX_RETRIEVALS`, `ADAPTIVE_RETRIEVAL_*`, `SYNONYM_VARIANTS`, `SYNONYM_CACHE_TTL`, `PRE_FUSION_MIN_DOCS`, `COLLAPSE_SAME_TITLE_VERSIONS`, `RRF_FUSION_ENABLED`, `MERGE_MMR_LAMBDA`
- Add `AGENT_TOTAL_TOOL_BUDGET=20`, `AGENT_MAX_SEARCH_EXACT=5`, `AGENT_MAX_SEARCH_SPARSE=5`, `AGENT_MAX_SEARCH_DENSE=5`, `AGENT_MAX_RERANK=5`, `AGENT_MAX_GRAPH_EXPAND=3`, `AGENT_MAX_KB_SEARCH=10`, `AGENT_MAX_EXTRACT_DATA=5`, `AGENT_MAX_CHART_GENERATE=3`

Update docs:
- `docs/FEATURES.md` — remove `rag_retrieve` references, update settings list
- `docs/enterprise-agent/02-target-architecture.md` — update architecture description
- `docs/enterprise-agent/03-tool-specifications.md` — update tool list
- `docs/enterprise-agent/pipeline-analysis.md` — update pipeline description
- `backend/app/services/agentic_rag/__init__.py` module docstring — update pipeline description
- `backend/app/services/agentic_rag/agent_graph/__init__.py` module docstring — update node list

### 13.3 Settings migration

Old settings that are removed:
- `AGENT_MAX_RETRIEVALS` — replaced by `AGENT_MAX_SEARCH_EXACT`, `AGENT_MAX_SEARCH_SPARSE`, `AGENT_MAX_SEARCH_DENSE`

New settings:
- `AGENT_TOTAL_TOOL_BUDGET` (default: 20)
- `AGENT_MAX_SEARCH_EXACT` (default: 5)
- `AGENT_MAX_SEARCH_SPARSE` (default: 5)
- `AGENT_MAX_SEARCH_DENSE` (default: 5)
- `AGENT_MAX_RERANK` (default: 5)
- `AGENT_MAX_GRAPH_EXPAND` (default: 3)
- `AGENT_MAX_KB_SEARCH` (default: 10)
- `AGENT_MAX_EXTRACT_DATA` (default: 5)
- `AGENT_MAX_CHART_GENERATE` (default: 3)

No Alembic migration needed — settings are stored in the `settings` table with org override. Old settings remain in the table but are unused. New settings get default values on first access.

### 13.4 Frontend migration

The frontend `Citation` interface needs to handle both old and new citation shapes:
- Old: `{document_id, chunk_index, metadata: {kb_id, document_id, source, ...}}`
- New: `{document_id, citation_kind, chunk_index, section, start_char, end_char, ..., metadata: {citation_ref: {...}, ...}}`

This is defensive parsing, not a compatibility layer. Old messages are read-only (displayed from stored data); new messages use the new shape.

### 13.5 Commit strategy

Commit per phase, with tests passing at each commit:

1. `feat: add atomic search tools (search_exact, search_sparse, search_dense, rerank_results, graph_expand)`
2. `feat: add expanded CitationRef schema with chunk/file/section/range/grep/table kinds`
3. `feat: replace correction-LLM with isError pattern, add total tool-call budget`
4. `feat: replace reflect/reflect_final with sufficiency_check node, remove expand_query/rewrite_query`
5. `feat: update AgentState for atomic tools, remove per-leg state`
6. `feat: rewrite planner/think/finalize prompts for atomic tool selection`
7. `test: update test suite for atomic tools and new citation model`
8. `feat: update frontend citation rendering for citation_kind`

### 13.6 Rollback plan

If the atomic tools approach proves worse than the composite approach in benchmarks:

1. Revert to the pre-implementation commit.
2. The old `rag_retrieve` code is preserved in git history.
3. No database migration to reverse.
4. No settings migration to reverse (old settings still in the table).

The breaking-change nature of this redesign makes rollback a simple git revert — no data migration concerns.

---

## 14. What Does NOT Change

- Low-level retrieval implementations in `app/services/retrieval/retrieval.py`:
  - `dense_search_docs()`, `sparse_search_docs()`, `exact_search_docs()`
  - `dedup_by_content_hash()`, `semantic_dedup()`
  - `_rrf_fuse()` (used inside `sparse_search_docs` and `exact_search_docs` for synonym fusion)
- Reranker in `app/services/retrieval/reranker.py`: `rerank()` function
- Graph expansion in `app/services/graph/expand.py`: `expand_docs_via_graph()` function
- `ToolContext`, `enforce_rbac`, `write_audit`
- `BaseAgentTool` dispatch pattern (enhanced with `prepare_arguments` and `terminate`)
- Post-generation answer scoring (`answer_scoring_node`)
- Memory/save behavior (`save_memory_node`)
- `accumulated_data` for extract_data → chart_generate flow
- Compaction logic (enhanced to preserve `citation_ref` metadata, but the algorithm is unchanged)
- Token budget management (`token_budget.py`)
- `kb_metadata`, `kb_search_documents`, `kb_outline`, `kb_read`, `kb_grep` tools (enhanced with `CitationRef`, but core logic unchanged)
- `file_read`, `file_summarize`, `file_extract_table` tools (unchanged)
- `code_execute`, `chart_generate`, `extract_data`, `summarize_answer` tools (unchanged)
- `current_datetime` tool (unchanged)
- Abbreviation service (`app/services/abbreviation_service.py`)
- Redis caching for synonym expansion
- Qdrant, Neo4j, MySQL infrastructure
- `_verify_execution` / `_build_execution_summary` (moved to `execution_check.py`, rewritten for atomic tools, but the verification concept is retained)
- `answer_scoring_node`, `clarify_interrupt_node` (stay in `reflection.py`)
- `answer_evaluation_node` (stays in `nodes.py`, updated to use `evidence`)
- `_agent_step`, `history_to_text`, `select_recent_history`, `_messages_to_conversation_text`, `_get_llm`, `_safe_writer` (stay in `nodes.py`)
- `get_effective_datastore_ids` (called by search tools to resolve datastore scope)

---

## 15. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| LLM chooses wrong search tool | Medium | Medium | Prompt guidance + fallback (if one search fails, LLM tries another) |
| LLM forgets to rerank | Medium | Low | Prompt says "call rerank_results if >10 hits"; rerank is optional, not critical |
| More tool calls = more latency | High | Medium | Total budget caps calls; deferred tool gating prevents unnecessary calls |
| LLM loops on failed searches | Medium | High | Total tool-call budget forces finalize; idempotency guard prevents duplicate calls |
| Citation metadata lost across search → rerank → read | Low | High | `CitationRef` travels with every hit; `rerank_results` preserves and updates `source_tool` |
| Frontend breaks on old citations | Low | Low | Defensive parsing handles both old and new citation shapes |
| Graph expansion not called when needed | Medium | Low | Prompt instructs to call `graph_expand` when search is insufficient; it's a separate tool the LLM can choose |
| Sufficiency check too aggressive | Medium | Medium | Three-tier check (budget, force_finalize, LLM); LLM check has fallback to reranker confidence |
| Sufficiency check not aggressive enough | Low | Medium | Total budget is a hard cap; wall-clock limit is a hard cap |

---

## 16. Open Questions

These need to be resolved during implementation, not before:

1. **Should `search_exact` search document titles as well as chunk text?**
   Current `rag_retrieve`'s exact leg searches both `document_chunks.chunk_text` and `documents.title` (title weighted 2×). The new `search_exact` should do the same. But should there be a separate `search_titles` tool for title-only search? Probably not — `kb_search_documents` already handles title search. `search_exact` should search both titles and chunks.

2. **Should `rerank_results` apply the excluded-terms filter?**
   Currently `_apply_excluded_terms_filter` runs inside `rag_retrieve` after reranking. In the new architecture, negation handling is less clear since there's no `rewrite_query` node to extract negated terms. Options:
   - Each search tool applies excluded terms internally (requires the LLM to pass `excluded_terms` as a parameter).
   - `rerank_results` applies excluded terms after reranking.
   - The finalize prompt instructs the LLM to ignore excluded topics.
   Recommendation: let the LLM handle this in the think loop — if the user says "but not Linux", the LLM should not cite Linux-related evidence. This is simpler than passing `excluded_terms` through every tool.

3. **Should the `sufficiency_check` node use an LLM call or be deterministic only?**
   The LLM-based check adds latency (one extra LLM call per tool round). A deterministic-only check (budget + reranker confidence) is faster but less accurate. Recommendation: start with deterministic-only, add LLM check only if the deterministic check proves insufficient.

4. **Should `kb_search_documents` deduplicate same-title versions?**
   Currently it collapses same-title versions (keeps latest by `file_modified_at`). In the new architecture, the LLM may want to see all versions to compare them. Recommendation: add a `dedup_same_title` parameter (default: true) so the LLM can opt out.

---

## 17. File Summary

### New files (10)

| File | Purpose |
|---|---|
| `backend/app/services/agentic_rag/tools/search_exact.py` | MySQL FULLTEXT search tool |
| `backend/app/services/agentic_rag/tools/search_sparse.py` | SPLADE sparse vector search tool |
| `backend/app/services/agentic_rag/tools/search_dense.py` | Dense vector search tool |
| `backend/app/services/agentic_rag/tools/rerank_results.py` | Cross-encoder reranker tool |
| `backend/app/services/agentic_rag/tools/graph_expand.py` | Neo4j graph expansion tool |
| `backend/app/services/agentic_rag/tools/_search_helpers.py` | Shared helpers (filter resolution, synonym expansion, excluded terms) |
| `backend/app/services/agentic_rag/agent_graph/sufficiency.py` | Sufficiency check node |
| `backend/app/services/agentic_rag/agent_graph/execution_check.py` | Moved `_verify_execution`/`_build_execution_summary` (rewritten for atomic tools) |
| `backend/alembic/versions/XXXX_add_citation_ref_fields.py` | Alembic migration for new citation columns |
| `frontend/src/types/chat.ts` | Shared SSE event type definitions |

### Deleted files (1)

| File | Reason |
|---|---|
| `backend/app/services/agentic_rag/tools/rag_retrieve.py` | Replaced by atomic search tools |

### Modified backend files (25)

| File | Changes |
|---|---|
| `backend/app/services/agentic_rag/tools/__init__.py` | New tool registry, deferred tool gating |
| `backend/app/services/agentic_rag/tools/base.py` | `prepare_arguments`, `terminate` field |
| `backend/app/services/agentic_rag/schemas.py` | New `CitationRef`, remove `QueryIntent`, remove `suggested_legs` |
| `backend/app/services/agentic_rag/graph_state.py` | Add/remove state fields |
| `backend/app/services/agentic_rag/agent_graph/build.py` | New graph topology |
| `backend/app/services/agentic_rag/agent_graph/tooling.py` | `isError` pattern, `_merge_observation_docs`, total budget, remove `route_tool`/`route_reflect_final` |
| `backend/app/services/agentic_rag/agent_graph/helpers.py` | New `_tool_call_budget`, remove `_correction_hints` |
| `backend/app/services/agentic_rag/agent_graph/finalization.py` | Evidence block, citation normalization, remove `rewritten_query`/`abbreviation_glossary`/`excluded_terms` deps |
| `backend/app/services/agentic_rag/agent_graph/load_context.py` | Remove expand/rewrite, update reset list |
| `backend/app/services/agentic_rag/agent_graph/thinking.py` | Update `route_think` to return `"finalize"` instead of `"reflect_final"` |
| `backend/app/services/agentic_rag/agent_graph/__init__.py` | Remove deleted re-exports, add new ones |
| `backend/app/services/agentic_rag/agent_graph/reflection.py` | Remove `reflect_node`, `reflect_final_node`, `route_reflect_final`; move `_verify_execution`/`_build_execution_summary` to `execution_check.py` |
| `backend/app/services/agentic_rag/agent_graph/observations.py` | Rename `_tried_rag_retrieve_queries`, update skip lists, update compaction |
| `backend/app/services/agentic_rag/agent_graph/planning.py` | Remove precomputed `rag_retrieve` calls, remove `query_intent`/`suggested_legs` deps |
| `backend/app/services/agentic_rag/nodes.py` | Remove dead retrieval functions (keep shared helpers, `answer_evaluation_node`) |
| `backend/app/services/agentic_rag/prompts.py` | Rewrite all prompts, remove obsolete prompts |
| `backend/app/services/agentic_rag/utils.py` | New `format_context_string`, new `normalize_citations` |
| `backend/app/services/agentic_rag/kb_profile.py` | Update filter field descriptions |
| `backend/app/services/agentic_rag/streaming.py` | Remove `rewritten_query`/`expanded_query` refs, update tool names |
| `backend/app/services/agentic_rag/__init__.py` | Update module docstring |
| `backend/app/services/chat/chat_service.py` | Update `_handle_answer_rewrite`, `_persist_citations`; remove `1:`/`eq:` handlers |
| `backend/app/services/retrieval/confidence.py` | Audit: update or remove if dead |
| `backend/app/core/settings_registry.py` | Add new settings, remove orphaned settings |
| `backend/app/models/chat.py` | `MessageCitation`: add new columns |
| `backend/app/api/api_v1/chat/messages.py` | Update `_serialize_messages_with_citations`, fix siblings endpoint |
| `backend/app/api/api_v1/chat/branching.py` | Remove `rewritten_query`/`expanded_query` refs, fix `done:` prefix bug |
| `backend/app/api/api_v1/chat/exports.py` | Handle nullable `rewritten_query` |
| `backend/app/api/api_v1/search.py` | Replace `expand_query_node` with `expand_query_suffix` |

### Modified frontend files (5)

| File | Changes |
|---|---|
| `frontend/src/components/chat/answer.tsx` | Citation rendering per `citation_kind`, `[N]` bare-bracket pre-processing, update interfaces |
| `frontend/src/app/dashboard/chat/[id]/page.tsx` | Update `Message`/`Citation` interfaces, update `r:` handler, remove `1:`/`eq:` handlers |
| `frontend/src/components/chat/agentic-progress.tsx` | Update `TOOL_ICONS`, `NODE_PHASE`, result summary logic |
| `frontend/src/components/chat/__tests__/answer.test.tsx` | Update citation test mocks and assertions |
| `frontend/src/lib/utils.ts` | Update `cleanChunkText` if citation text field changes |

### New test files (2)

| File | Purpose |
|---|---|
| `backend/tests/test_atomic_search_tools.py` | Tests for search_exact, search_sparse, search_dense, rerank_results, graph_expand |
| `backend/tests/test_citation_ref.py` | Tests for CitationRef schema and normalize_citations |

### Modified test files (16+)

| File | Changes |
|---|---|
| `backend/tests/test_agent_loop.py` | Update tool registry, remove rag_retrieve references, add execution/graph tests |
| `backend/tests/test_agent_loop_budget.py` | Update budget assertions |
| `backend/tests/test_agent_state_integrity.py` | Update state field assertions |
| `backend/tests/test_retrieval_filtering.py` | Update tool references |
| `backend/tests/test_retrieval_filtering_e2e.py` | Update tool references |
| `backend/tests/test_rag_retrieve_ladder.py` | Delete or rewrite (relaxation ladder removed) |
| `backend/tests/test_conditional_dense.py` | Update for atomic tools |
| `backend/tests/test_sort_rerank_order.py` | Update for `rerank_results` tool |
| `backend/tests/test_synonym_rrf.py` | Update (synonym RRF moves into search tools) |
| `backend/tests/test_negation_filter.py` | Update (negation handling moves to search tools/LLM) |
| `backend/tests/test_behavioural_transcripts.py` | Update state field and node references |
| `backend/tests/test_query_intent.py` | Delete (QueryIntent schema removed) |
| `backend/tests/test_abbr_e2e.py` | Update (expand_query_node removed) |
| `backend/tests/test_abbr_quality.py` | Update (expand_query_node removed) |
| `backend/tests/test_search_abbr_isolation.py` | Update (expand_query_node removed) |
| `backend/tests/test_settings_phase5.py` | Update settings references |
| `backend/tests/test_kb_tools.py` | Update patch data |

### New benchmark file (1)

| File | Purpose |
|---|---|
| `backend/tests/benchmark_queries.py` | Benchmark query set for before/after comparison |

### Config/docs files (5+)

| File | Changes |
|---|---|
| `.env.example` | Remove old settings, add new settings |
| `docs/FEATURES.md` | Update tool and settings references |
| `docs/enterprise-agent/02-target-architecture.md` | Update architecture description |
| `docs/enterprise-agent/03-tool-specifications.md` | Update tool list |
| `docs/enterprise-agent/pipeline-analysis.md` | Update pipeline description |

---

## 18. Implementation Order

Execute phases sequentially. Each phase must pass its tests before moving to the next.

**Critical ordering notes**:
- Settings must be added to `settings_registry.py` in Phase 3 BEFORE any code reads them.
- `rag_retrieve.py` is not deleted until Phase 4 (when the graph no longer imports from it).
- `AGENT_MAX_RETRIEVALS` is not removed from the registry until Phase 5 (after all code stops reading it).
- The Alembic migration (Phase 2) must run before the frontend can render new citation fields.

```
Phase 1: Atomic Search Tools
  ├── Add new settings to settings_registry.py (MUST be first)
  ├── Create _search_helpers.py (extract from rag_retrieve.py)
  ├── Create search_exact.py
  ├── Create search_sparse.py
  ├── Create search_dense.py
  ├── Create rerank_results.py
  ├── Create graph_expand.py
  ├── Update tools/__init__.py (registry — add new tools, keep rag_retrieve for now)
  ├── Update tools/base.py (prepare_arguments, terminate)
  ├── Run: pytest tests/test_atomic_search_tools.py

Phase 2: Citation Pipeline
  ├── Create Alembic migration (add citation_ref columns to message_citations)
  ├── Update schemas.py (new CitationRef)
  ├── Update utils.py (format_context_string, normalize_citations)
  ├── Update finalization.py (evidence block, citation normalization, remove old deps)
  ├── Update chat_service.py (_handle_answer_rewrite, _persist_citations)
  ├── Update messages.py (_serialize_messages_with_citations, fix siblings endpoint)
  ├── Update models/chat.py (MessageCitation new columns)
  ├── Update existing tools (attach CitationRef to results)
  └── Run: pytest tests/test_citation_ref.py

Phase 3: Execution Layer
  ├── Update tooling.py (isError pattern, _merge_observation_docs, total budget)
  ├── Update helpers.py (new _tool_call_budget, remove _correction_hints)
  ├── Update observations.py (rename _tried_rag_retrieve_queries, update skip lists)
  ├── Update prompts.py (remove TOOL_CORRECTION_PROMPT only — full rewrite is Phase 6)
  ├── Update tools/__init__.py (deferred tool gating)
  └── Run: pytest tests/test_agent_loop.py::TestExecutionLayer

Phase 4: Graph Changes
  ├── Create execution_check.py (move _verify_execution/_build_execution_summary)
  ├── Create sufficiency.py (sufficiency_check_node, route_sufficiency)
  ├── Update build.py (new graph topology)
  ├── Update load_context.py (remove expand/rewrite, update reset list)
  ├── Update thinking.py (route_think returns "finalize" not "reflect_final")
  ├── Update tooling.py (remove route_tool, route_reflect_final, update imports)
  ├── Update __init__.py (remove deleted re-exports, add new ones)
  ├── Remove reflect_node, reflect_final_node from reflection.py
  ├── Remove dead functions from nodes.py
  ├── Delete rag_retrieve.py (now safe — no imports from it)
  └── Run: pytest tests/test_agent_loop.py::TestGraphTopology

Phase 5: State Changes
  ├── Update graph_state.py (add/remove fields)
  ├── Update schemas.py (remove suggested_legs, QueryIntent)
  ├── Remove AGENT_MAX_RETRIEVALS from settings_registry.py (now safe)
  ├── Remove other orphaned settings from registry
  └── Run: pytest tests/test_agent_loop.py

Phase 6: Prompts and Backend Linkages
  ├── Rewrite PLAN_SYSTEM_PROMPT, THINK_SYSTEM_PROMPT, FINALIZE_ANSWER_PROMPT
  ├── Rewrite FINALIZE_GUARDRAIL_PROMPT, LAST_ANSWER_EXTRACT_PROMPT
  ├── Remove REWRITE_INTENT_SUFFIX, RETRIEVAL_REWRITE_PROMPT
  ├── Update planning.py (remove precomputed rag_retrieve calls)
  ├── Update kb_profile.py, streaming.py, confidence.py
  ├── Update chat_service.py (remove 1:/eq: handlers)
  ├── Update branching.py, exports.py, search.py
  ├── Update .env.example, docs
  └── Run: pytest tests/test_agent_loop.py

Phase 7: Full Test Suite
  ├── Update all test files (16+ files)
  ├── Run: pytest tests/ -v (full suite)
  └── Verify: all tests pass

Phase 8: Frontend
  ├── Create frontend/src/types/chat.ts (shared SSE types)
  ├── Update answer.tsx (citation rendering, [N] pre-processing, interfaces)
  ├── Update page.tsx (interfaces, r: handler, remove 1:/eq: handlers)
  ├── Update agentic-progress.tsx (TOOL_ICONS, NODE_PHASE, result summary)
  ├── Update answer.test.tsx, utils.ts
  ├── Run: docker exec rag-web-ui-frontend-1 npm run test:ci
  └── Run: docker exec rag-web-ui-frontend-1 npm run build
```

---

## 19. Glossary

| Term | Definition |
|---|---|
| Atomic tool | A tool that does one thing (e.g. `search_dense`) rather than orchestrating multiple operations (e.g. `rag_retrieve`) |
| Composite tool | A tool that internally orchestrates multiple operations (the current `rag_retrieve`) |
| CitationRef | Structured citation metadata: document_id, citation_kind, chunk_index, section, offsets, etc. |
| Evidence | Any piece of retrieved content (chunk, file, section, grep match) that can be cited in the answer |
| Sufficiency check | Graph node that determines whether the agent has enough evidence to answer |
| Deferred tool gating | Tools that become available only after prior tool use (e.g. `rerank_results` after a search) |
| `isError` pattern | Failed tool calls return as error observations; the LLM sees the error and decides whether to retry |
| `terminate` hint | A tool result can signal "stop calling tools" to short-circuit the loop |
| Total tool-call budget | Configurable cap on total tool calls across all tools per query (default: 20) |
| RRF | Reciprocal Rank Fusion — merges multiple ranked lists into one |
| SPLADE | Sparse Lexical and Expansion model — sparse vector search |
| Cross-encoder | Reranker model that scores (query, passage) pairs jointly |
| Execution check | Moved `_verify_execution`/`_build_execution_summary` helpers, rewritten for atomic tools |

---

## 20. Cross-Validation Audit Log

This section documents the cross-validation performed against the full `rag-web-ui` codebase. Four subagents audited: (1) backend retrieval/ingestion pipeline, (2) backend agent graph and state, (3) frontend citation and SSE consumers, (4) settings and infrastructure linkages.

### 20.1 Findings addressed in this update

| # | Finding | Resolution |
|---|---|---|
| 1 | `route_think` returns `"reflect_final"`, not `"finalize"` — plan said "stays the same" | §6.5 updated with correct return values |
| 2 | `route_tool` function exists and must be explicitly removed | §6.6 updated to specify function deletion |
| 3 | `_verify_execution`/`_build_execution_summary` used by `tooling.py` and `thinking.py` — cannot be deleted with `reflect_node` | §6.2 updated: move to `execution_check.py` |
| 4 | `agent_graph/__init__.py` re-exports removed nodes — deleting without update breaks package import | §6.2 updated: must update `__init__.py` |
| 5 | Settings registry ordering: `get_setting` returns `None` for unregistered keys → `TypeError` | §5.3 updated with ordering constraint |
| 6 | `ToolContext` has no `datastore_ids` field — plan's search tool examples used `ctx.datastore_ids` | §9 Phase 1 example updated to use `get_effective_datastore_ids()` |
| 7 | `enforce_rbac` is a standalone function, not a `ToolContext` method — plan used `ctx.enforce_rbac()` | §9 Phase 1 example updated to `enforce_rbac(ctx, ...)` |
| 8 | `observations.py` hardcodes `rag_retrieve` in skip lists, `_tried_rag_retrieve_queries`, `_compact_observations` | §6.2 and Phase 4 updated to include `observations.py` changes |
| 9 | `planning.py` precomputes `rag_retrieve` calls from `query_intent`/`suggested_legs` | §8.6 added to specify planning.py changes |
| 10 | `finalization.py` uses `rewritten_query`/`abbreviation_glossary`/`excluded_terms` | §8.7 added to specify finalization.py changes |
| 11 | `Message.rewritten_query` and `Message.expanded_query` are DB columns — plan removed state keys but didn't address columns | §2.4 and §13.2 updated: columns kept as nullable |
| 12 | SSE events `1:` (rewritten_query) and `eq:` (expanded_query) — plan didn't address | §2.4 and §13.2.1 updated: events removed, handlers become no-ops |
| 13 | `chat_service.py` `_handle_answer_rewrite` and `_persist_citations` need updates for new CitationRef | §8.11 and Phase 2 updated |
| 14 | `messages.py` siblings endpoint drops citations (pre-existing bug) | Phase 2 updated: fix siblings endpoint |
| 15 | `branching.py` emits `done:` but frontend only handles `d:` (pre-existing bug) | §13.2.1 updated |
| 16 | `agentic-progress.tsx` `TOOL_ICONS` and `NODE_PHASE` maps reference `rag_retrieve`/removed nodes | Phase 8 updated to include `agentic-progress.tsx` |
| 17 | `page.tsx` `Message` interface has `rewrittenQuery`/`expandedQuery` | Phase 8 updated |
| 18 | Frontend `[N](N)` markdown link rendering — bare `[N]` won't be clickable | Phase 8 updated with Option A (pre-process to `[N](N)`) |
| 19 | 16+ test files reference `rag_retrieve` or removed state keys — plan only listed 3 | Phase 7 updated with full test file list |
| 20 | `.env.example` and docs reference removed settings | §13.2.2 added |
| 21 | `kb_profile.py`, `streaming.py`, `confidence.py`, `chat_service.py`, `branching.py`, `exports.py`, `search.py` reference `rag_retrieve` or removed state | §8.6-§8.13 added |
| 22 | Orphaned settings: `ADAPTIVE_RETRIEVAL_*`, `SYNONYM_*`, `PRE_FUSION_MIN_DOCS`, `COLLAPSE_*`, `RRF_*`, `MERGE_MMR_*` | §5.3 updated with full list |
| 23 | Dead functions in `nodes.py`: `merge_node`, `collapse_same_title_versions`, retrieval legs, `reranking_node`, etc. | §2.4 updated with full list |
| 24 | `_pin_filter_matches` and `_apply_excluded_terms_filter` need a new home | Phase 1 updated: extract to `_search_helpers.py` |
| 25 | `MessageCitation` table needs new columns for `CitationRef` fields | §13.2 and Phase 2 updated with Alembic migration |
| 26 | Budget default inconsistency (15 vs 20) | §5.2 updated with explicit note: canonical value is 20 |
| 27 | `rag_retrieve.py` deletion timing — plan said delete in Phase 1 but graph still imports from it | Phase 1 updated: don't delete until Phase 4 |
| 28 | `_is_transient_error` and `_writer` already exist in `helpers.py` | Verified — no action needed |
| 29 | `answer_evaluation_node` stays but must use `evidence` for cited-evidence-only scoring | §2.4 and §2.5 updated |
| 30 | `GroupedResultCard.tsx` uses chunk field names that might change | Phase 8: audit if search result card field names change |

### 20.2 Findings deferred to implementation

| # | Finding | Action during implementation |
|---|---|---|
| 1 | `MessageResponse` doesn't include `citations`, `tool_calls`, `plan`, `last_answer_object` | Decide whether to add these to the schema for reload support |
| 2 | `task_list` and `thinking` SSE events registered but no producer | Decide whether to remove dead handlers or implement producers |
| 3 | `context` event (`2:`) in normal chat only carries `docs`, not `confidence`/`score` | Decide whether to add confidence fields to the normal chat path |
| 4 | `exports.py` doesn't include citations or tool calls in PDF/Word exports | Decide whether to add citation rendering to exports |
| 5 | `progressMessages` and `toolTrace` in frontend are stored but never rendered | Decide whether to render them or remove the dead state |
| 6 | `confidence.py` may become dead if confidence is LLM-assessed | Audit during Phase 6 |
| 7 | `_retrieval_confidence_level`, `_resolve_eval_kwargs`, `_final_confidence_level` in `nodes.py` | Audit during Phase 4: keep if called by `answer_evaluation_node`, remove otherwise |

### 20.3 Ingestion linkage verification

The atomic retrieval plan was cross-validated against the ingestion pipeline. The following ingestion outputs are consumed by the new search tools:

| Ingestion output | Consumed by | Preserved in search results? |
|---|---|---|
| Document conversion Markdown | `kb_read`, `kb_search_documents` | Yes — `content` field |
| OCR metadata | `kb_metadata`, `kb_search_documents` | Yes — `metadata` dict |
| Page/image boundaries | `search_*` tools | Yes — `page` field in hits |
| Document title and filename | All search and discovery tools | Yes — `title`, `file_name` fields |
| `file_created_at`, `file_edited_at`, `file_modified_at` | `kb_search_documents` (date filters, sort) | Yes — filter resolution uses these |
| Content type | `kb_search_documents` (content_type filter) | Yes — filter resolution |
| Document and chunk IDs | All search tools | Yes — `document_id`, `chunk_index` |
| Content hashes | `search_*` tools, `rerank_results` | Yes — `content_hash` field, used for dedup |
| Qdrant point IDs | `search_*` tools, `graph_expand` | Yes — `qdrant_point_id` field |
| Vector and sparse indexing | `search_dense`, `search_sparse` | Yes — via `dense_search_docs`/`sparse_search_docs` |
| Graph indexing (Neo4j) | `graph_expand` | Yes — via `expand_docs_via_graph` |
| KB/datastore scope | All search tools | Yes — `kb_ids` + `get_effective_datastore_ids` |
| RBAC filtering | All search tools | Yes — `enforce_rbac(ctx, kb_ids=...)` |

No ingestion linkage gaps were found. The new search tools consume the same ingestion outputs as the current `rag_retrieve`, via the same low-level retrieval functions.
