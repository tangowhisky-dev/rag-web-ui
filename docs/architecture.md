# RAG Web UI Architecture

## Overview

A self-hosted knowledge base Q&A system using 3-leg hybrid retrieval (dense vector + SPLADE sparse + MySQL full-text) with any OpenAI-compatible LLM.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           RAG WEB UI ARCHITECTURE                            │
└──────────────────────────────────────────────────────────────────────────────┘

┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃  FRONTEND   ┃   BACKEND    ┃  VECTOR DB  ┃   GRAPH DB   ┃  DATABASE    ┃
┃ (Next.js)   ┃ (FastAPI)    ┃ (Qdrant)    ┃  (Neo4j)     ┃ (MySQL 8)    ┃
┗━━━━━━━━━━━━━┻━━━━━━━━━━━━━━┻━━━━━━━━━━━━━┻━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━┛

USER REQUEST → [Frontend:3000] → [Backend API:8000] → [Retrieval Engine] → [LLM] → RESPONSE
```

---

## Data Flow

### 1. Document Ingestion Pipeline

```
Upload (PDF / DOCX / DOC / PPTX / PPT / XLSX / XLS /
        TXT / MD / HTML / MHTML / CSV / JSON / XML /
        MSG / EML / EPUB / JPG / PNG / GIF / BMP / TIFF / ZIP)
    │
    ▼
document_processor.py
    ├── Convert to Markdown (MarkItDown — single unified parser for all formats; OCR via vision model when VISION_MODEL is set)
    ├── Chunk (RecursiveCharacterTextSplitter)
    ├── Embed chunks — async OpenAI-compatible API → dense vectors
    ├── Embed chunks — FastEmbed SPLADE → sparse vectors
    ├── Upsert to Qdrant (dense + sparse named vectors per collection kb_<id>; qdrant_point_id stored in payload)
    ├── Store chunk text in MySQL document_chunks (for FTS + metadata)
    └── graph_service.py — GraphRAG extraction (when GRAPHRAG_ENABLED=true)
            ├── [ReLiK mode]  POST /api/relik → NER+RE → write Entity nodes + relationships to Neo4j
            │                  └── MATCH Chunk node by qdrant_point_id → CREATE (chunk)-[:FROM_CHUNK]->(entity)
            └── [LLM mode]    LLMEntityRelationExtractor (use_structured_output=True, JSON Schema-constrained)
                               → neo4j-graphrag Pipeline → write Entity nodes + relationships to Neo4j
                               └── MATCH Chunk node by qdrant_point_id → CREATE (chunk)-[:FROM_CHUNK]->(entity)
```

### 2. Query / Chat Pipeline

```
User message
    │
    ▼
chat_service.py
    ├── Identity shortcut  (hardcoded response for "who are you?" etc.)
    ├── Sliding-window context (3 most-recent turn-pairs verbatim)
    ├── Rolling summary    (older turns folded into a summary via LLM)
    ├── Standalone question (context folded in → self-contained query; uses QUERY_MODEL if set)
    │
    ▼
classifier.py — classify_query()  [QUERY_CLASSIFIER_ENABLED]
    └── LLM-based 4-way classification: FACTUAL / ENTITY_CENTRIC / MULTI_PART / AMBIGUOUS
        → emitted in stream frame 1: alongside rewritten query
    │
    ▼
retrieval.py — hybrid_search_with_legs()
    ├── get_retrieval_config(query_type) → per-type leg weights + top-k (RETRIEVAL_CONFIG_PRESETS)
    ├── Leg 1: _dense_search()          → Qdrant cosine similarity (dense)
    ├── Leg 2: _qdrant_sparse_search()  → Qdrant SPLADE sparse vectors
    ├── Leg 3: _exact_search()          → MySQL FULLTEXT NATURAL LANGUAGE MODE
    │                   │
    │                   ▼
    │              _rrf_merge_candidates()
    │                   └── Weighted Reciprocal Rank Fusion using per-preset weights
    │                   │
    │                   ▼ (optional, GRAPHRAG_ENABLED + use_graph_rag)
    │         graph_service.py — expand_docs_via_graph() + enrich_docs_with_graph()
    │                   │
    │                   ▼ (optional, RERANKER_ENABLED)
    │         reranker.py — cross-encoder reranking
    │                   │
    │                   ▼ (ENTITY_AWARE_ENABLED + ENTITY_CENTRIC query)
    │         entity_extractor.py — extract_expand_boost()
    │             ├── extract_entities_from_query() — LLM (GRAPHRAG_LLM) extracts entities
    │             ├── expand_query_entities()       — Neo4j 1-hop CONTAINS match + expansion
    │             └── apply_entity_boost()          — additive score boost (ENTITY_BOOST_FACTOR)
    │
    ▼
confidence.py — score_retrieval()
    └── 4-level confidence: HIGH/MEDIUM/LOW/NONE + suggestion
    │
    ▼  [stream frame 2: emitted to UI — context, confidence, query_classification, tool_trace, synthesis_mode]
    │
    ▼ (TOOL_CALLING_ENABLED + non-synthesis query — Step 2.5)
tool_registry.py / builtin_tools.py
    ├── search_documents(query, kb_ids, top_k)
    ├── extract_entities(text)
    └── summarize_chunks(chunks, instruction)
    └── Loop up to MAX_TOOL_ITERATIONS — tool results fed back into conversation
    │
    ▼ (SYNTHESIS_MODE_ENABLED + MULTI_PART/AMBIGUOUS + synthesis keywords — Step 4 with synthesis prompt)
builtin_tools.py — synthesize_documents(topic, sub_queries, kb_ids, top_k_per_query)
    └── asyncio.gather N sub-queries in parallel → MD5 content-hash dedup → score-sorted
    │
    ▼
chat_service.py
    ├── QA prompt (standard) OR synthesis orchestration prompt (synthesis mode)
    ├── Stream response via AsyncOpenAI
    └── Strip <think> blocks (reasoning model support)
    │
export_service.py — generate_synthesis_report() [synthesis mode]
    └── Structured Markdown: header + answer + ## Sources from tool_trace
```

### 3. Chat File Upload Pipeline

Files attached to chat messages are processed ephemerally — not indexed in any knowledge base.

```
User attaches file to message
    │
    ▼
POST /api/chat/{chat_id}/files
    ├── Save original to uploads/ephemeral/{chat_id}/{filename} (deduplicated with _1, _2 suffixes)
    ├── Insert ChatFile row (status=processing, stored_path=<disk path>)
    └── Background task: MarkItDown → Markdown
            ├── Update ChatFile.markdown_content + token_count (status=ready)
            └── Original file kept on disk until chat delete

User sends message with file_id
    │
    ▼
POST /api/chat/{chat_id}/messages (or /messages/with-file for streaming)
    ├── Load ChatFile.markdown_content from DB
    ├── Inject into generate_response() as file_markdown (bypasses query rewrite + KB retrieval)
    └── QA system prompt receives: "## Uploaded File Context\n<markdown>"
            └── Prior-turn files in same chat also injected for multi-turn continuity

Chat delete  →  delete_ephemeral_chat_files(chat_id)  →  rm -rf uploads/ephemeral/{chat_id}/

Download
    └── GET /api/chat/{chat_id}/files/{file_id}/download  →  FileResponse(stored_path)
```

**Key design constraints:**
- File content bypasses the query rewrite step (rewriter would corrupt the query using prior "no file" history)
- Synthesis mode is disabled when a file is present (synthesis prompt has no file injection path)
- File content is capped at 25% of `OPENAI_MODEL_CONTEXT_SIZE` tokens
- `chat_files.file_name` stores the finalized on-disk filename (post-deduplication), not the raw upload name
```

#### GraphRAG architecture note

**Strict separation of concerns:**

```
Qdrant  — source of truth for all chunk TEXT and VECTORS
Neo4j   — source of truth for GRAPH TOPOLOGY (entities, relationships, chunk linkage)
```

Vectors are never stored in Neo4j. Neo4j Chunk nodes are keyed by `qdrant_point_id` (the exact UUID Qdrant uses as a point ID), enabling bidirectional lookup in a single index hit. This means graph-expanded chunks can be fetched from Qdrant by UUID — no re-embedding, no re-scoring, just a direct key lookup.

**Retrieval expansion vs enrichment** — two distinct operations:

- **Expansion** (`expand_docs_via_graph`) — finds chunks NOT in the vector search results by traversing entity connections. A query surface 5 chunks; expansion may add 3 more that are entity-connected but not in the top-K by similarity.
- **Enrichment** (`enrich_docs_with_graph`) — appends entity/relationship triples as `[Graph context]` text to every candidate chunk (seed + expanded), giving the reranker and LLM explicit graph signal.

**Entity-aware retrieval** (`entity_extractor.py`, M001):

- For ENTITY_CENTRIC queries, `extract_entities_from_query()` calls the `GRAPHRAG_LLM` model with a JSON-schema prompt to extract named entities.
- `expand_query_entities()` fuzzy-matches extracted entities against Neo4j `__Entity__` nodes (CONTAINS match), then fetches 1-hop neighbors.
- `apply_entity_boost()` adds `ENTITY_BOOST_FACTOR` to the RRF score of chunks whose text mentions any matched entity.

**Extraction backends** (controlled by `.env`):

| Mode | Trigger | Notes |
|---|---|---|
| LLM | `GRAPHRAG_LLM=<model>` | `LLMEntityRelationExtractor` + `use_structured_output=True`; JSON Schema-constrained |
| Disabled | both unset | Extraction silently skipped; retrieval graph leg also inactive |

**Explore the graph:** Neo4j Browser — http://localhost:7474/browser/ (login: `neo4j` / `ragwebui_neo4j`)


```
app/
├── main.py                    # FastAPI entry point, startup hooks
├── api/
│   └── api_v1/
│       ├── api.py             # Router registration
│       ├── auth.py            # JWT login / register
│       ├── chat.py            # Chat endpoints (create, stream, history)
│       ├── chat_files.py      # Ephemeral chat file upload, status poll, download, delete
│       └── knowledge_base.py  # KB + document CRUD, upload, processing
├── core/
│   ├── config.py              # All settings (pydantic-settings, reads .env)
│   ├── security.py            # Password hashing, JWT creation/verification
│   └── storage.py             # Local filesystem helpers (save, move, delete)
├── db/
│   └── session.py             # SQLAlchemy engine + SessionLocal
├── models/
│   ├── user.py                # User ORM model
│   ├── knowledge.py           # KnowledgeBase, Document, DocumentChunk, ProcessingTask
│   └── chat.py                # Chat, Message ORM models
├── schemas/
│   ├── user.py                # Pydantic request/response schemas
│   ├── knowledge.py
│   ├── chat.py
│   └── token.py
├── services/
│   ├── document_processor.py  # Ingestion: parse → chunk → embed → index
│   ├── retrieval.py           # 3-leg hybrid search + weighted RRF merge + adaptive config
│   ├── reranker.py            # Cross-encoder reranking (RERANKER_ENABLED)
│   ├── entity_extractor.py    # LLM entity extraction, Neo4j expansion, score boost (M001)
│   ├── tool_registry.py       # Tool registry + execute_tool() (M001)
│   ├── builtin_tools.py       # search_documents, extract_entities, summarize_chunks, synthesize_documents (M001)
│   ├── chat_service.py        # Conversation context, prompt, LLM streaming, synthesis mode
│   ├── export_service.py      # Export to PDF/Word/Image + generate_synthesis_report
│   ├── confidence.py          # 4-level retrieval confidence scoring
│   └── chunk_record.py        # MySQL chunk upsert helpers
└── startup/                   # Startup utilities (Alembic auto-migrate etc.)

alembic/                       # Database migration scripts
```

### Frontend Structure (`frontend/`)

Next.js 14 app with TypeScript, Tailwind CSS, shadcn/ui, and the Vercel AI SDK for streaming.

```
src/
├── app/
│   ├── dashboard/
│   │   ├── chat/
│   │   │   ├── [id]/page.tsx      # Chat view: messages, streaming, file chip, AgentTimeline
│   │   │   ├── new/page.tsx       # New chat: select KB + retrieval options
│   │   │   └── page.tsx           # Redirect to most recent chat or /new
│   │   ├── knowledge/             # KB management CRUD
│   │   └── test-retrieval/        # Retrieval quality tester
│   ├── api/
│   │   └── chat/[id]/
│   │       ├── messages/route.ts          # Streaming proxy (Node.js http.request)
│   │       ├── messages/with-file/route.ts # File+message streaming proxy
│   │       └── files/[fileId]/download/route.ts  # File download proxy
│   └── layout.tsx                 # ThemeProvider (dark/light/system)
├── components/
│   ├── chat/
│   │   ├── chat-sidebar.tsx       # Collapsible sidebar; localStorage persistence
│   │   ├── chat-input.tsx         # Textarea + file attachment button
│   │   ├── file-attachment.tsx    # FileUploadChip (pre-send) + MessageFileChip (post-send download)
│   │   ├── answer.tsx             # Markdown renderer with citation parsing
│   │   └── agent-timeline.tsx     # Streaming AgentTimeline (tool trace + confidence)
│   ├── layout/
│   │   ├── chat-layout.tsx        # Full-width breadcrumb bar + sidebar + main
│   │   └── dashboard-layout.tsx   # KB management layout
│   └── ui/
│       └── theme-toggle.tsx       # Dark / light / system toggle
└── middleware.ts                  # Route protection (redirect unauthenticated to /login)
```

**Chat UI layout:** A full-width breadcrumb bar sits at the top of the viewport (blurred glass effect). Below it, the collapsible sidebar and main chat area fill the remaining height. Sidebar state persists in `localStorage` under `chat-sidebar-collapsed`. The `ThemeProvider` (next-themes) is mounted at root with `attribute="class"` supporting dark/light/system modes; the CSS palette uses pure neutral greys (zero chroma) for dark mode.

### Docker Stack

| Service | Image | Purpose |
|---------|-------|---------|
| `backend` | custom (Python FastAPI) | API server; uvicorn with hot-reload in dev |
| `frontend` | custom (Next.js) | Web UI; Next.js dev server or production build |
| `qdrant` | `qdrant/qdrant` | Vector database (dense + sparse collections) |
| `db` | `mysql:8` | Relational data + FULLTEXT chunk index |
| `neo4j` | `neo4j:2026.04` | Graph DB — entity/relationship storage for GraphRAG + entity-aware retrieval; browser at http://localhost:7474/browser/ |
| `adminer` | `adminer` | MySQL web GUI (dev compose only) |

---

## Key Architectural Decisions

### 1. 3-Leg Hybrid Retrieval
No single modality dominates all query types. Dense vectors handle paraphrases; SPLADE handles technical terms; MySQL FTS handles exact keywords and product codes. Weighted RRF fuses all three without requiring scores to be on the same scale.

### 2. CPU-First Sparse Embeddings
SPLADE runs locally via FastEmbed (ONNX, CPU-optimised), avoiding any GPU dependency for retrieval while maintaining learned sparse expansion beyond raw BM25.

### 3. MarkItDown for Unified Document Parsing
All document types are converted to Markdown by [MarkItDown](https://github.com/microsoft/markitdown) before chunking. A single parser handles 20+ formats (PDF, Office, spreadsheets, email, images via OCR, archives) and produces consistent Markdown output that the splitter can break on structural boundaries. Format-specific LangChain loaders (`PyPDFLoader`, `Docx2txtLoader`) are no longer used.

When `VISION_MODEL` is set, the `markitdown-ocr` plugin is activated and embedded images in documents (scanned PDF pages, photos in DOCX/PPTX/XLSX) are sent to the vision model for OCR. Think-block traces emitted by reasoning vision models are stripped before the text is chunked. When `VISION_MODEL` is unset the behaviour is identical to before — no OCR, no external calls.

### 4. Ingestion Always Indexes All Three Stores
Per-leg retrieval can be toggled via `.env` without re-ingestion. This makes A/B testing retrieval configurations cheap — flip a flag, test, flip back.

### 5. Sliding Window + Rolling Summary for Context
Rather than truncating history or stuffing the full chat into the prompt, older turns are summarised by the LLM and folded into a rolling summary. The 3 most-recent turn-pairs are kept verbatim. Both the query-rewriting step and the summarisation step use `QUERY_MODEL` when set, falling back to `OPENAI_MODEL`.

### 6. OpenAI-Compatible API for LLM and Embeddings
Four distinct model roles are supported, all pointing at OpenAI-compatible endpoints:

| Variable | Role | Falls back to |
|---|---|---|
| `OPENAI_MODEL` | Response generation (RAG answers) | — (required) |
| `QUERY_MODEL` | Query rewriting + rolling summarisation | `OPENAI_MODEL` |
| `VISION_MODEL` | markitdown-ocr OCR during ingestion | unset = OCR disabled |
| `DENSE_EMBEDDINGS_MODEL` | Dense embeddings | — (required) |

`OPENAI_VISION_API_BASE` lets the vision model live on a different server (e.g. a separate Ollama instance for a multimodal model). When unset it falls back to `OPENAI_API_BASE`.

---

## Memory & Session Management

- Alembic migrations for MySQL schema evolution (auto-applied on backend startup)
- JWT tokens with configurable expiration (default 7 days)
- Ephemeral `SECRET_KEY` in dev — tokens are invalidated on container restart

### 7. LLM Query Classifier for Adaptive Retrieval
Rather than using the same retrieval config for every query, an LLM classifier (using `QUERY_MODEL` or `OPENAI_MODEL`) assigns each query to one of four types. Each type maps to a tunable retrieval preset (leg weights + top-k) stored in `RETRIEVAL_CONFIG_PRESETS`. This allows ENTITY_CENTRIC queries to rely more on dense search, MULTI_PART queries to blend dense + sparse, and AMBIGUOUS queries to retrieve broader candidate sets — without any retraining.

### 8. Entity-Aware Retrieval via Neo4j
For ENTITY_CENTRIC queries, named entities are extracted from the query using the `GRAPHRAG_LLM` model, matched against existing Neo4j entities with fuzzy CONTAINS matching and 1-hop expansion, and used to boost chunk scores. This integrates the graph's entity knowledge into the retrieval scoring without a separate retrieval leg.

### 9. Agentic Tool Calling Substrate
A global tool registry (`tool_registry.py`) lets any code register callable tools as structured JSON schemas. `execute_tool()` always returns a `ToolResult` — it never raises — so one tool failure never breaks the loop. The pre-answer tool-calling loop in `chat_service.py` is gated by `TOOL_CALLING_ENABLED` and skipped entirely for synthesis queries (which use a structured synthesis prompt instead, avoiding double tool execution).

### 10. Synthesis Mode for Multi-Document Queries
Synthesis queries (MULTI_PART/AMBIGUOUS + synthesis keywords) bypass the standard QA prompt and enter synthesis mode. The LLM is given a structured synthesis prompt that directs it to call `synthesize_documents` (parallel sub-query fan-out with MD5 deduplication), then produce a structured Markdown report. The `synthesize_documents` tool runs sub-queries in a separate thread (`ThreadPoolExecutor`) with `asyncio.run()` to avoid the "event loop already running" error that would occur with `loop.run_until_complete()` inside a running FastAPI context.

---

## Technology Stack Summary

| Layer | Technology |
|---|---|
| Document Parsing | MarkItDown (Microsoft) — 20+ formats to Markdown; OCR via markitdown-ocr |
| Frontend | Next.js 14, TypeScript, Tailwind CSS, shadcn/ui, Vercel AI SDK |
| Backend | Python FastAPI, LangChain, SQLAlchemy, Alembic |
| Vector DB | Qdrant (dense + sparse named vectors) |
| Graph DB | Neo4j (entity/relationship graph; neo4j-graphrag pipeline; entity-aware retrieval) |
| Sparse Embeddings | SPLADE via FastEmbed (CPU, ONNX, local) |
| Cross-Encoder Reranking | HuggingFace cross-encoder (RERANKER_MODEL) — CPU inference |
| File Storage | Local filesystem (Docker volume mount) |
| Database | MySQL 8 (ORM data + FULLTEXT index) |
| Auth | JWT (python-jose, bcrypt) |

---

## Quick Start

```bash
git clone https://github.com/tangowhisky-dev/rag-web-ui.git
cd rag-web-ui
cp .env.example .env
# Edit .env — set OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_MODEL, DENSE_EMBEDDINGS_MODEL, DENSE_EMBEDDING_DIM
# Optional: QUERY_MODEL (query rewriting), VISION_MODEL (OCR), OPENAI_VISION_API_BASE
docker compose up -d --build
```

Open **http://localhost:3000**, register an account, and start uploading documents.

See [README.md](../README.md) for full configuration reference and development setup.
