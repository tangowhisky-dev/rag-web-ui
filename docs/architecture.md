# RAG Web UI Architecture

## Overview

A self-hosted knowledge base Q&A system with three answering modes, 3-leg hybrid retrieval, optional GraphRAG, and an agentic LangGraph pipeline for complex queries.

```
USER REQUEST → [Frontend:3000] → [Backend API:8000] → [Pipeline] → [LLM] → STREAMING RESPONSE
```

---

## Answering Modes

### Fast ⚡ and Thinking 🧠 (`fast_pipeline.py`)

Linear pipeline, low latency:

```
rewrite_query
  → hybrid_search_with_legs (dense + sparse + exact + graph in parallel)
  → stream LLM answer
```

Fast uses `OPENAI_MODEL`; Thinking uses `REASONING_MODEL` (falls back to `OPENAI_MODEL`). Thinking mode is identical in structure — only the model changes.

**Steps visible in UI:** Rewriting query → Retrieving context → (optional) Additional context from Neo4j → Generating answer.

### Agentic 🤖 (`rag_graph.py`)

Full LangGraph `StateGraph` with 10 nodes and a coverage-driven retry loop:

```
rewrite_query
  → context_router          (smart source routing: kb / file_current / file_prior / both)
  → decompose_query         (LLM splits into 2–5 atomic sub-queries)
  → parallel_retrieval      (hybrid search per sub-query via asyncio.gather; reinforced dedup)
  → extract_file_sections   (LLM selects 3–6 relevant file sections; passthrough if ≤ 12 KB)
  → draft_answer            (LLM draft keyed by sub-query, for grading only)
  → grade_coverage          (LLM grades each sub-query: covered / partially_covered / not_covered)
  → [conditional_router]
      ├─ all covered         → generate_answer (final)
      ├─ uncovered, attempt 0 → widened_retrieval (relaxed reranker threshold −5.0)
      │                            → draft_answer → grade_coverage
      ├─ uncovered, attempt 1 → keyword_search_loop (MySQL FULLTEXT: broad → narrow)
      │                            → draft_answer → grade_coverage
      └─ attempt ≥ 2          → generate_answer (partial / unable)
```

**Reinforced scoring:** a chunk retrieved by N sub-queries has its RRF score multiplied by N, making broadly relevant chunks rank higher.

**All nodes emit `active` and `done` events** — the UI shows live step labels ("Decomposing query…" → "Sub-queries: [list]") with collapsible detail panels.

---

## Data Flow

### 1. Document Ingestion Pipeline

```
Upload (PDF / DOCX / PPTX / XLSX / TXT / MD / HTML / CSV / JSON /
        XML / MSG / EML / EPUB / images (OCR) / ZIP)
    │
    ▼
document_processor.py
    ├── MarkItDown → Markdown (OCR via VISION_MODEL when set)
    ├── RecursiveCharacterTextSplitter → chunks
    ├── Dense embedding → Qdrant (per-KB collection kb_<id>)
    ├── SPLADE sparse embedding → Qdrant (named sparse vector)
    ├── chunk_text + metadata → MySQL document_chunks (for FTS)
    └── graph_service.py [GRAPHRAG_ENABLED=true]
            └── LLMEntityRelationExtractor → Neo4j Entity nodes + relationships
                → (chunk)-[:FROM_CHUNK]->(entity) keyed by qdrant_point_id
```

### 2. Fast / Thinking Pipeline (`fast_pipeline.py`)

```
User message
    │
    ▼
chat_service.generate_response()
    ├── answeringMode = "fast" or "thinking"
    └── fast_stream()
            ├── _rewrite_query()        — standalone question via QUERY_MODEL
            ├── hybrid_search_with_legs() — all 3 legs in parallel
            ├── [graph_enrichment step if Neo4j returned data]
            └── ChatOpenAI.astream()    — token streaming
```

Events emitted: `agent_step` (active + done per node), `rewritten_query`, `context`, `token`, `done`.

### 3. Agentic Pipeline (`rag_graph.py`)

```
User message
    │
    ▼
chat_service.generate_response()
    ├── answeringMode = "agentic"
    └── run_stream()
            └── _rag_graph.astream_events()
                    ├── on_chain_start  → emit agent_step {status: "active"}
                    ├── on_chain_end    → emit agent_step {status: "done", ...node data}
                    └── on_chat_model_stream → emit token events (generate_answer only)
```

#### Node-by-node detail

| Node | What it does | State written |
|---|---|---|
| `rewrite_query` | Condense with history → retrieval-friendly query | `rewritten_query` |
| `context_router` | LLM decides: kb / file_current / file_prior / both | `sources`, `file_ids_needed` |
| `decompose_query` | LLM splits into 2–5 atomic sub-queries | `sub_queries` |
| `parallel_retrieval` | `hybrid_search_with_legs` per sub-query via `asyncio.gather`; dedup with reinforced scoring | `retrieved_docs` |
| `extract_file_sections` | LLM selects 3–6 relevant sections from attached file; passthrough if ≤ 12 KB | `file_markdown` (trimmed) |
| `draft_answer` | LLM writes per-sub-query draft using current context | `draft_answer` |
| `grade_coverage` | LLM grades each sub-query as covered / partially_covered / not_covered | `coverage_result`, `uncovered_sub_queries` |
| `widened_retrieval` | Retry for uncovered sub-queries; reranker threshold relaxed to −5.0 | `retrieved_docs` (accumulated), `retrieval_attempt=1` |
| `keyword_search_loop` | MySQL FULLTEXT: broad keywords first, narrow if no results; max 3 sub-queries × 2 iterations | `retrieved_docs`, `keyword_iterations`, `retrieval_attempt=2` |
| `generate_answer` | Final streaming answer with citation normalisation; transparent partial-answer note if uncovered sub-queries remain after all retries | `answer` |

#### Conditional routing

```python
def _route_after_grade(state) -> str:
    if not uncovered:           return "generate_answer"
    if attempt == 0:            return "widened_retrieval"
    if attempt == 1:            return "keyword_search_loop"
    return "generate_answer"    # attempt >= 2: partial/unable
```

### 4. Chat File Upload Pipeline

```
POST /api/chat/{chat_id}/files
    ├── 10 MB size guard
    ├── Save to uploads/ephemeral/{chat_id}/
    ├── Background: MarkItDown → Markdown → token estimate
    │       ├── token_count > 25% of OPENAI_MODEL_CONTEXT_SIZE → status=error
    │       └── else → status=ready, markdown_content stored in MySQL
    └── Client polls /files/{file_id} for status

User sends message with file
    ├── Fast/Thinking: file_markdown passed to fast_stream() — full content, no truncation
    └── Agentic: file_markdown passed to run_stream() → extract_file_sections selects relevant sections

Chat delete → rm -rf uploads/ephemeral/{chat_id}/
```

---

## Backend Code Structure

```
backend/app/
├── api/api_v1/
│   ├── auth.py             # JWT login / register
│   ├── chat.py             # Chat endpoints; extracts answering_mode from request body
│   ├── chat_files.py       # Ephemeral file upload, status poll, download, delete
│   ├── folders.py          # Chat folder management
│   └── knowledge_base.py   # KB + document CRUD, upload, processing
├── core/
│   ├── config.py           # All settings (pydantic-settings); OPENAI_MODEL,
│   │                         QUERY_MODEL, REASONING_MODEL, VISION_MODEL,
│   │                         OPENAI_MODEL_CONTEXT_SIZE, RERANKER_*, GRAPHRAG_*
│   ├── security.py         # Password hashing, JWT
│   └── storage.py          # Local filesystem helpers
├── models/                 # SQLAlchemy ORM: User, KnowledgeBase, Document,
│                             DocumentChunk, ProcessingTask, Chat, Message, ChatFile
├── services/
│   ├── fast_pipeline.py    # Fast/Thinking: rewrite → hybrid search → stream
│   ├── rag_graph.py        # Agentic: 10-node LangGraph StateGraph
│   ├── chat_service.py     # Routes to fast_stream or run_stream by answering_mode
│   ├── retrieval.py        # 3-leg hybrid search + weighted RRF + adaptive presets
│   ├── reranker.py         # Cross-encoder reranking (score_threshold configurable)
│   ├── document_processor.py # Ingest: parse → chunk → embed → index
│   ├── graph_service.py    # Neo4j ingestion + graph expansion/enrichment
│   ├── entity_extractor.py # LLM entity extraction from queries + Neo4j score boost
│   ├── confidence.py       # 4-level retrieval confidence scoring
│   ├── export_service.py   # PDF/Word/Image export
│   ├── auto_tune.py        # Retrieval config auto-tuning
│   └── markdown_cleaner.py # Post-processing for LLM markdown output
└── startup/                # Alembic auto-migrate on startup
```

## Frontend Code Structure

```
frontend/src/
├── app/
│   ├── dashboard/
│   │   ├── chat/[id]/page.tsx    # Chat view: messages, streaming, AgentTimeline,
│   │   │                           mode selector, stop button, abort controller
│   │   ├── chat/new/page.tsx     # New chat: select KB + retrieval options
│   │   └── knowledge/            # KB management CRUD
│   └── api/chat/[id]/
│       ├── messages/route.ts           # Streaming proxy
│       ├── messages/with-file/route.ts # File+message streaming proxy
│       └── files/[fileId]/download/route.ts
├── components/chat/
│   ├── agent-timeline.tsx  # Real-time pipeline step display (active/done collapsibles)
│   │                         Nodes: rewrite_query, context_router, decompose_query,
│   │                                parallel_retrieval, extract_file_sections,
│   │                                draft_answer, grade_coverage, widened_retrieval,
│   │                                keyword_search_loop, graph_enrichment, generate_answer
│   ├── answer.tsx          # Markdown renderer with [N](N) citation link parsing
│   ├── chat-input.tsx      # Textarea + mode selector (Fast/Thinking/Agentic pills)
│   │                         + Stop button (replaces Send during generation)
│   ├── chat-sidebar.tsx    # Collapsible sidebar; drag-to-folder; action buttons
│   ├── file-attachment.tsx # Pre-send dropzone chip + post-send download chip
│   ├── branch-picker.tsx   # Chat branching (multiple answer variants)
│   └── mermaid-diagram.tsx # Mermaid diagram rendering in answers
└── middleware.ts            # Route protection (redirect to /login)
```

---

## Docker Stack

| Service | Image | Purpose |
|---|---|---|
| `backend` | custom (Python FastAPI) | API server + pipeline |
| `frontend` | custom (Next.js) | Web UI |
| `qdrant` | `qdrant/qdrant` | Vector database |
| `db` | `mysql:8` | Relational data + FULLTEXT index |
| `neo4j` | `neo4j:2026.04` | Entity/relationship graph (GraphRAG) |
| `adminer` | `adminer` | MySQL web GUI (dev compose only, port 8081) |

---

## Key Architectural Decisions

### 1. Three Answering Modes with a Single Streaming Contract
All three modes (`fast_stream`, `run_stream`) emit the same SSE event shapes: `agent_step`, `rewritten_query`, `context`, `token`, `answer_rewrite`, `done`. The frontend handles them identically regardless of mode.

### 2. Draft-Grade-Retry Loop (Agentic)
Rather than grading individual documents (which misses synthesis failures), the agentic pipeline generates a per-sub-query draft answer and grades it for coverage. This catches cases where 8 individually relevant chunks together still don't answer the compound question. Retry escalates from widened vector search to keyword search to partial-answer transparency — always showing the user what was found and what wasn't.

### 3. Reinforced Scoring (Agentic)
A chunk retrieved for N sub-queries has its RRF score accumulated across those N results. Chunks central to many aspects of the question naturally rank higher than chunks relevant to only one edge.

### 4. File Token Budget Before Pipeline
Both file size (10 MB) and token count (25% of `OPENAI_MODEL_CONTEXT_SIZE`) are enforced at upload/processing time — not at generation time. By the time a file reaches either pipeline, it has already been approved. No silent truncation at the LLM boundary.

### 5. Smart Routing Preserved in Agentic Mode
`context_router` (LLM-based, JSON-schema constrained) decides whether to search the KB, use the attached file, or both — before decomposition and retrieval. `extract_file_sections` then selects the relevant portions of the file for the sub-queries. These nodes are inherited from v1 and unchanged.

### 6. Active State Visibility via `on_chain_start`
LangGraph's `astream_events` fires `on_chain_start` before a node runs. The `run_stream` generator intercepts this and emits an `agent_step` with `status: "active"` immediately, so the UI shows "Decomposing query…" before the LLM call completes — not after.

### 7. 3-Leg Hybrid Retrieval with Adaptive Presets
Dense vectors handle paraphrases; SPLADE captures TF-IDF signal; MySQL FTS handles exact keywords. Weighted RRF fuses all three. Per-query-type presets (`RETRIEVAL_CONFIG_PRESETS`) allow different leg weights for FACTUAL vs ENTITY_CENTRIC vs MULTI_PART queries.

### 8. GraphRAG: Qdrant + Neo4j Strict Separation
Vectors live exclusively in Qdrant. Neo4j Chunk nodes are keyed by `qdrant_point_id`. Graph expansion finds chunks not in the top-K by traversing entity edges; enrichment appends entity triples to existing chunks. Neither operation requires re-embedding.

### 9. OpenAI-Compatible API Throughout
Four model roles — `OPENAI_MODEL`, `QUERY_MODEL`, `REASONING_MODEL`, `VISION_MODEL` — all point at OpenAI-compatible endpoints and can be on different servers via `OPENAI_API_BASE` / `OPENAI_VISION_API_BASE`. Switching models requires only `.env` changes, no code changes.

### 10. AbortController for Stop
The frontend holds an `AbortController` in a ref. The Stop button calls `abortControllerRef.current.abort()`. The `fetch` stream catches `AbortError` and preserves the partial message with `*(generation stopped)*` appended. Real errors show a toast and remove the placeholder.
