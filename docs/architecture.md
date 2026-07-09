# RAG Web UI Architecture

## Overview

A self-hosted knowledge base Q&A system with multi-tenant org management, agentic multi-step retrieval, 3-leg hybrid retrieval, and optional GraphRAG.

```
USER REQUEST → [Frontend:3000] → [Backend API:8000] → [Agentic Pipeline] → [LLM] → STREAMING RESPONSE
```

---

## Answering Modes

### Single Agentic Pipeline (`agentic_rag/agentic_rag.py`)

The codebase consolidates to a single agentic pipeline. The former Fast/Thinking (`fast_pipeline.py`) and Agentic LangGraph (`rag_graph/`) pipelines have been removed. The agentic agent is now the sole production pipeline.

```
User Query
  │
  ├─ 1. Rewrite query using chat history (LLM)
  ├─ 2. Classify: simple (direct) or complex (decompose)
  │
  └─ SIMPLE path:
      rewrite → hybrid search → rerank → stream answer
  │
  └─ COMPLEX path:
      rewrite → decompose into sub-queries
        └─ FOR EACH SUBTASK:
            rewrite sub-query → search → rerank → stream answer (token-by-token)
            update task list in UI
      └─ FINAL: synthesize all subtask answers → grade → stream final summary
         lightweight self-review (non-blocking)
```

**Model auto-selection:** heuristic keyword matching on subtask text (`compare`, `analyze`, `design`, etc.) selects between `OPENAI_MODEL` and `REASONING_MODEL`. No extra LLM classification call.

**Model selection for complex queries:** if the overall query matches thinking keywords, all subtasks use `REASONING_MODEL`; otherwise `OPENAI_MODEL`.

---

## Data Flow

### 1. Document Ingestion Pipeline

**Direct uploads:** PDF, DOCX, PPTX, XLSX, TXT, MD, HTML, CSV, JSON, XML, MSG, EML, EPUB, images (JPG/PNG/GIF/BMP/TIFF), ZIP

**Event-driven ingestion (DataStores):** same formats, detected via watchdog filesystem events.

**Parsing:** MarkItDown (Microsoft) — single library for all formats, producing consistent Markdown output. OCR via `markitdown-ocr` when `VISION_MODEL` is set.

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

### 2. Agentic Pipeline (`agentic_rag/agentic_rag.py`)

The sole production pipeline. `chat_service.generate_response()` delegates to `run_agentic_rag()` when `answering_mode = "agentic"` (the current default).

```
User message
    │
    ▼
chat_service.generate_response()
    ├── answeringMode = "agentic"
    └── run_agentic_rag()
            ├─ 1. Rewrite query (LLM: QUERY_MODEL)
            ├─ 2. Classify: simple or complex (heuristic, no LLM call)
            │
            ├─ SIMPLE path:
            │   → hybrid search → rerank → stream answer
            │
            └─ COMPLEX path:
                → decompose into N sub-queries (LLM)
                └─ FOR EACH SUBTASK:
                    → rewrite sub-query → search → rerank
                    → stream answer (token-by-token) with progress events
                → synthesize final answer
                → lightweight post-review (non-blocking)
```

**Event protocol (SSE prefixes):**
- `p:` progress — transient status messages
- `t:` task_list — subtask list with status
- `th:` thinking — reasoning model chain-of-thought
- `0:` token — streaming answer text
- `1:` rewritten_query — standalone rewritten query (internal)
- `2:` context — retrieved documents
- `3:` error — exception message
- `d:` done — finish reason + usage

The frontend `agent-timeline.tsx` renders `p:`, `t:`, and `th:` events as real-time progress indicators. All other events follow the same contract as the previous `rag_graph.run_stream()` format.

### 4. Chat File Upload Pipeline

```json
POST /api/chat/{chat_id}/files
    ├── 10 MB size guard
    ├── Save to uploads/ephemeral/{chat_id}/
    ├── Background: MarkItDown → Markdown → token estimate
    │       ├── token_count > 25% of OPENAI_MODEL_CONTEXT_SIZE → status=error
    │       └── else → status=ready, markdown_content stored in MySQL
    └── Client polls /files/{file_id} for status

User sends message with file
    └── Agentic: file_markdown passed to run_agentic_rag() — full content, no truncation. File section selection uses LLM-based heuristic.

Chat delete → rm -rf uploads/ephemeral/{chat_id}/
```

### 5. Multi-Tenancy Flow

```
Admin creates org
    ├── Assign users (role: user/admin/super_admin, org_id)
    ├── Assign data sources (DataStore → OrganisationDataStore)
    ├── Configure LLM settings (org LLM config, ingestion status)
    └── Configure file watchers (per-datastore local dir)

User creates chat (user_id, org_id)
    ├── Chats are user-scoped (Chat.user_id == current_user.id)
    └── Knowledge bases are filtered by org_id

Admin creates data store (folder_path, scan_pattern)
    ├── Auto-scan (event-driven, not periodic) with debouncing interval
    ├── Assign to orgs (OrganisationDataStore junction)
    └── Per-datastore file watcher picks up new files from the folder
```

---

## Backend Code Structure

```
backend/app/
├── api/api_v1/
│   ├── auth.py             # JWT login / register / rate limiting / token test
│   ├── chat.py             # Chat endpoints; extracts answering_mode from request body
│   ├── chat_files.py       # Ephemeral file upload, status poll, download, delete
│   ├── folders.py          # Chat folder management
│   ├── knowledge_base.py   # KB + document CRUD, upload, processing
│   ├── admin.py            # Org CRUD, LLM config, ingestion status, users, watchers
│   ├── datastores.py       # DataStore CRUD, assign/unassign, scan status
│   ├── watcher.py          # Per-org file watcher endpoints
│   ├── query.py            # Stateless RAG query, KB ingest status
│   └── api.py              # Router aggregation, /config endpoint
├── core/
│   ├── config.py           # All settings (pydantic-settings); OPENAI_MODEL,
│   │                         QUERY_MODEL, REASONING_MODEL, VISION_MODEL,
│   │                         OPENAI_MODEL_CONTEXT_SIZE, RERANKER_*, GRAPHRAG_*
│   ├── security.py         # Password hashing, JWT, rate limiting
│   └── storage.py          # Local filesystem helpers
├── models/                 # SQLAlchemy ORM: User, KnowledgeBase, Document,
│                             DocumentChunk, ProcessingTask, Chat, Message, ChatFile
├── services/
│   ├── agentic_rag/        # Single agentic pipeline (rewrite → classify → subtasks → stream)
│   │   ├── agentic_rag.py  # Autonomous agent: rewrite, decompose, iterate, synthesize
│   │   ├── context_manager.py  # Token budgeting
│   │   ├── user_profile.py   # User preference store
│   │   └── tools/            # Safe DB query, graph query tools
│   ├── chat_service.py     # Routes to run_agentic_rag()
│   ├── retrieval.py        # 3-leg hybrid search + weighted RRF + adaptive presets
│   ├── reranker.py         # Cross-encoder reranking (score_threshold configurable)
│   ├── document_processor.py # Ingest: parse → chunk → embed → index
│   ├── graph_service.py    # Neo4j ingestion + graph expansion/enrichment
│   ├── entity_extractor.py # LLM entity extraction from queries + Neo4j score boost
│   ├── confidence.py       # 4-level retrieval confidence scoring
│   ├── export_service.py   # PDF/Word/Image export
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
│   │   ├── admin/                # Admin panel: orgs, users, data sources, watcher
│   │   ├── knowledge/            # KB management CRUD
│   │   └── test-retrieval/[id]/  # KB retrieval test page
│   └── api/chat/[id]/
│       ├── messages/route.ts           # Streaming proxy
│       ├── messages/with-file/route.ts # File+message streaming proxy
│       └── files/[fileId]/download/route.ts
├── components/chat/
│   ├── agent-timeline.tsx  # Real-time pipeline step display (active/done collapsibles)
│   │                         Events: p: (progress), t: (task list), th: (thinking)
│   ├── answer.tsx          # Markdown renderer with [N](N) citation link parsing
│   │                         Think blocks, confidence score, query classification badge,
│   │                         tool trace, retrieved context blocks, citation popovers
│   ├── chat-input.tsx      # Textarea + mode selector (agentic mode)
│   │                         + Stop button (replaces Send during generation)
│   │                         + KB selector + file upload chip
│   ├── chat-sidebar.tsx    # Collapsible sidebar; drag-to-folder; message search
│   │                         Chat export, folder create/rename/delete
│   ├── file-attachment.tsx # Pre-send dropzone chip + post-send download chip
│   ├── branch-picker.tsx   # Chat branching (multiple answer variants) with sibling navigation
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

### 1. Single Agentic Pipeline with Simple/Complex Branching
The codebase consolidates to one production pipeline: `agentic_rag/agentic_rag.py`. The former Fast/Thinking (`fast_pipeline.py`) and Agentic LangGraph (`rag_graph/`) pipelines were removed as dead code. The agent classifies queries as simple (direct answer) or complex (subtask decomposition) using heuristic keyword matching — no extra LLM classification call.

### 2. Inline Streaming for Complex Subtasks
Rather than buffering all results, each subtask streams tokens in real-time. The UI shows a task list that updates as each subtask completes. The final synthesis is a lightweight post-review that never blocks streaming.

### 3. Reinforced Scoring (via shared `retrieval.py`)
A chunk retrieved for N sub-queries has its RRF score accumulated across those N results. Chunks central to many aspects of the question naturally rank higher than chunks relevant to only one edge. Implemented in the shared `retrieval.hybrid_search_with_legs()` and `_dedup_and_reinforce()` logic used by all retrieval paths.

### 4. File Token Budget Before Pipeline
Both file size (10 MB) and token count (25% of `OPENAI_MODEL_CONTEXT_SIZE`) are enforced at upload/processing time — not at generation time. By the time a file reaches the pipeline, it has already been approved. No silent truncation at the LLM boundary.

### 5. Simple vs Complex Branching
For simple queries, the agent takes a direct path: rewrite → search → rerank → stream. For complex queries, it decomposes into subtasks with progress events. Heuristic keyword matching (`compare`, `analyze`, `design`, etc.) determines the path. No LLM classification overhead.

### 6. Real-Time Progress Events
The agentic pipeline emits `p:` (progress), `t:` (task list), and `th:` (thinking) SSE events. The frontend `agent-timeline.tsx` renders these as live progress indicators — no LangGraph `on_chain_start` intercept needed.

### 7. 3-Leg Hybrid Retrieval with Adaptive Presets
Dense vectors handle paraphrases; SPLADE captures TF-IDF signal; MySQL FTS handles exact keywords. Weighted RRF fuses all three. Per-query-type presets (`RETRIEVAL_CONFIG_PRESETS`) allow different leg weights for FACTUAL vs ENTITY_CENTRIC vs MULTI_PART queries.

### 8. GraphRAG: Qdrant + Neo4j Strict Separation
Vectors live exclusively in Qdrant. Neo4j Chunk nodes are keyed by `qdrant_point_id`. Graph expansion finds chunks not in the top-K by traversing entity edges; enrichment appends entity triples to existing chunks. Neither operation requires re-embedding.

### 9. OpenAI-Compatible API Throughout
Four model roles — `OPENAI_MODEL`, `QUERY_MODEL`, `REASONING_MODEL`, `VISION_MODEL` — all point at OpenAI-compatible endpoints and can be on different servers via `OPENAI_API_BASE` / `OPENAI_VISION_API_BASE`. Switching models requires only `.env` changes, no code changes.

### 10. AbortController for Stop
The frontend holds an `AbortController` in a ref. The Stop button calls `abortControllerRef.current.abort()`. The `fetch` stream catches `AbortError` and preserves the partial message with `*(generation stopped)*` appended. Real errors show a toast and remove the placeholder.

### 11. Multi-Tenancy with Org-Scoped Chats
Chats are user-scoped (`Chat.user_id == current_user.id`), not org-scoped. Users can only see their own chats regardless of org membership. Knowledge bases are org-scoped, so users only see KBs in their org. Admins and super admins see across all orgs.

### 12. DataStore Event-Driven Ingestion

DataStores are separate from KnowledgeBases — they have no KB relationship. File ingestion happens via two independent paths:

**Event-driven (watchdog):**
```
File added/modified/deleted in datastore folder
    │
    ▼
DatastoreFileEventHandler
    ├── Watchdog observer (PollingObserver on macOS Docker, InotifyObserver on Linux)
    ├── _resolve_datastore() — finds which datastore's folder_path contains the event
    ├── _should_process() — per-file debounce (1s window)
    ├── _dispatch() — 1s write-completion delay via _SyntheticEvent, then debouncer
    └── _queue_change() → _process_pending_changes() → _on_changes()
            │
            ├── _handle_file() — the real entry point
            │     ├── Event = "deleted" → _handle_deletion()
            │     ├── Document exists + hash changed → _update_document()
            │     └── Document doesn't exist → _ingest_file()
            └── _refresh_file_count() — updates last_scan_total_files

**Manual scan (user clicks "Scan"):**
```
POST /datastores/{id}/scan
    │
    ▼
DataStoreWatcher.scan_single_datastore(datastore_id)
    ├── _init_scan() — assigns scan_id, counts files, sets status=running
    ├── Walk all files in datastore folder
    │     For each file:
    │       _handle_file_in_scan() — compute hash, check Document existence
    │         ├── Hash unchanged + chunks exist → skip
    │         ├── Hash unchanged + no chunks → re-ingest (ingestion likely failed)
    │         ├── Hash changed → re-ingest
    │         └── New file → ingest
    ├── Wait for all ingestion Futures (up to 1 hour each)
    └── _complete_scan() — sets status=completed or error, persists new/modified/skipped/errors
```

### 13. Rate Limiting with Exponential Backoff
The login endpoint tracks failed attempts per IP address with exponential backoff: 3 attempts trigger escalating delays (15s → 30s → 60s → 120s → 240s → 480s → 900s). Successful login resets the counter. Rate-limited responses return 429 with `Retry-After` header.

### 14. Message Pagination with Infinite Scroll
The chat page loads messages in pages of 20, using cursor-based pagination (`before_id`). An `IntersectionObserver` watches a sentinel element at the top of the message list and loads older messages when the sentinel enters the viewport. Scroll position is preserved across page loads using `useLayoutEffect`.
