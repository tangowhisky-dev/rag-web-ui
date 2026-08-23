<div align="center">
  <img src="https://raw.githubusercontent.com/rag-web-ui/rag-web-ui/main/docs/images/github-cover-new.png" alt="RAG Web UI">
  <br />
  <p>
    <strong>Knowledge Base Management with Retrieval-Augmented Generation</strong>
  </p>
  <p>
    <a href="https://github.com/rag-web-ui/rag-web-ui/blob/main/LICENSE"><img src="https://img.shields.io/github/license/rag-web-ui/rag-web-ui" alt="License"></a>
    <a href="#"><img src="https://img.shields.io/badge/python-3.11-blue.svg" alt="Python"></a>
    <a href="#"><img src="https://img.shields.io/badge/node-%3E%3D18-green.svg" alt="Node"></a>
    <a href="#"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome"></a>
  </p>
</div>

## Introduction

RAG Web UI is a self-hosted knowledge base Q&A system with multi-tenant org management. Upload your documents, then chat with them using any **OpenAI-compatible API** — works with OpenAI, LM Studio, Ollama, or any local model server.

**Agentic RAG pipeline:**

The system uses a single LangGraph-based agentic pipeline that automatically adapts to query complexity:

- Query rewriting with chat history context
- LLM-based query classification (FACTUAL/ENTITY_CENTRIC/MULTI_PART/AMBIGUOUS)
- Automatic sub-query decomposition for complex queries
- Parallel retrieval with reinforced scoring
- Draft-grade-retry loop with widened retrieval and keyword search fallback
- Confidence scoring and partial-answer transparency

**Retrieval:** 3-leg hybrid search (dense vector via Qdrant, sparse via SPLADE, exact via MySQL FULLTEXT) with native Qdrant MMR diversity and recency-aware dedup (exact + semantic). Optional **GraphRAG** adds entity/relationship extraction into Neo4j for graph-traversal expansion.

**Multi-tenancy:** Admins create organisations, assign users and data sources to orgs, and configure org-specific LLM settings.

> **Based on:** An opinionated fork of [rag-web-ui/rag-web-ui](https://github.com/rag-web-ui/rag-web-ui). Credit to the original authors. Goal: minimal dependencies, visible RAG internals, and an agentic pipeline that genuinely improves retrieval on hard queries.

## Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, TypeScript, Tailwind CSS, shadcn/ui |
| Backend | Python 3.11, FastAPI, LangGraph, LangChain, SQLAlchemy |
| Vector DB | Qdrant v1.18 (dense + sparse vectors) |
| Graph DB | Neo4j 2026.04.0 (entity/relationship graph for GraphRAG — optional) |
| Sparse Embeddings | SPLADE via FastEmbed (CPU, local) |
| Cache/State | Redis Stack 7.4.0 (LangGraph checkpoints, response cache) |
| File Storage | Local folder mapped as Docker volume |
| Database | MySQL 8.4 |

## Quick Start

**Prerequisites:** Docker & Docker Compose v2+

```bash
git clone https://github.com/tangowhisky-dev/rag-web-ui.git
cd rag-web-ui
cp .env.example .env
# Edit .env — set OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_MODEL,
#              DENSE_EMBEDDINGS_MODEL, DENSE_EMBEDDING_DIM
docker compose up -d --build
```

Open **http://localhost:3000** — register an account and start uploading documents.

> **First run note:** The SPLADE model (~500 MB) downloads on first document ingestion. To pre-download: `python download_assets.py` (requires `pip install fastembed`).

## Configuration

Copy `.env.example` to `.env` and set these values:

### LLM & Embeddings

| Variable | Required | Description | Example |
|---|---|---|---|
| `OPENAI_API_KEY` | yes | API key for your LLM provider | `sk-...` or `lmstudio` |
| `OPENAI_API_BASE` | yes | Base URL of OpenAI-compatible API | `https://api.openai.com/v1` |
| `OPENAI_MODEL` | yes | Main response-generation model | `gpt-4o` |
| `QUERY_MODEL` | no | Model for query rewriting and rolling summarisation. Falls back to `OPENAI_MODEL`. | `gpt-4o-mini` |
| `REASONING_MODEL` | no | Reserved for reasoning/CoT support; currently unused by the active pipeline. | `o3-mini` |
| `VISION_MODEL` | no | Multimodal model for OCR of scanned PDFs and embedded images. | `gpt-4o-mini` |
| `OPENAI_VISION_API_BASE` | no | Base URL for vision model if on a different server. Falls back to `OPENAI_API_BASE`. | `http://host.docker.internal:11434/v1` |
| `DENSE_EMBEDDINGS_MODEL` | yes | Embedding model name | `text-embedding-3-small` |
| `DENSE_EMBEDDING_DIM` | yes | Output dimension of the embedding model | `1536` |
| `OPENAI_MODEL_CONTEXT_SIZE` | no | Context window size in tokens — controls file injection cap (25%). | `131072` |

**Using LM Studio / Ollama:**
```env
OPENAI_API_KEY=lmstudio
OPENAI_API_BASE=http://host.docker.internal:1234/v1
OPENAI_MODEL=your-chat-model
QUERY_MODEL=your-fast-model
# REASONING_MODEL is currently unused by the active pipeline — leave empty or remove.
# REASONING_MODEL=your-reasoning-model
VISION_MODEL=your-vision-model         # optional
DENSE_EMBEDDINGS_MODEL=your-embedding-model
DENSE_EMBEDDING_DIM=1024
```

### Agentic Pipeline

The LangGraph-based agentic pipeline automatically adapts to query complexity:

```
rewrite_query → context_router → decompose_query → parallel_retrieval
  → extract_file_sections → draft_answer → grade_coverage
  → [if uncovered, attempt 0] widened_retrieval → draft_answer → grade_coverage
  → [if still uncovered, attempt 1] keyword_search_loop → draft_answer → grade_coverage
  → generate_answer
```

All steps are streamed to the UI as collapsible timeline entries in real time.

### Pipeline Features

Beyond the basic retrieval, the pipeline includes:

- **Native Qdrant MMR** — both dense and sparse legs use Qdrant's Maximal Marginal Relevance to diversify candidates and reduce near-duplicate clustering
- **Confidence scoring** — per-query confidence levels (low/medium/high) based on coverage and chunk quality
- **Query classification** — FACTUAL, ENTITY_CENTRIC, MULTI_PART, or AMBIGUOUS with confidence and latency metrics
- **Tool trace** — collapsible timeline of tool calls during the pipeline (search, graph traversal, etc.)
- **Synthesis mode** — LLM can synthesize across multiple retrieved contexts before answering

### Chunking

| Variable | Description | Default |
|---|---|---|
| `CHUNK_SIZE` | Target chunk size in characters. Keep ≤ 1800 for SPLADE. | `1500` |
| `OVERLAP_PERCENTAGE` | Fraction of `CHUNK_SIZE` repeated at boundaries (0.0–1.0). | `0.20` |

> **Warning:** Do not change these after ingesting documents. Re-upload to re-index.

### Chat File Upload

Files attached to chat messages are processed ephemerally — **not** indexed in any knowledge base.

1. File saved to `uploads/ephemeral/{chat_id}/`.
2. MarkItDown converts to Markdown; token count estimated at ~4 chars/token.
3. **Upload guards:** 10 MB file size limit; token budget = 25% of `OPENAI_MODEL_CONTEXT_SIZE`. Files exceeding the token budget are rejected with a clear error message.
4. Markdown stored in MySQL `chat_files` table.
5. On next message, file content is injected into the pipeline.
   - Full approved content passed to the LLM (no truncation)
   - `extract_file_sections` node uses the LLM to select 3–6 most relevant sections; files ≤ 12,000 chars are passed through unchanged
6. Prior-turn files in the same chat are re-injected for continuity.
7. Files are deleted when the chat is deleted.

| Variable | Description | Default |
|---|---|---|
| `OPENAI_MODEL_CONTEXT_SIZE` | Context window tokens — controls 25% file injection cap. | `131072` |

### GraphRAG (Knowledge Graph)

GraphRAG extracts entities and relationships from ingested chunks and stores them in Neo4j. At query time, vector search results are expanded via entity-graph traversal.

**Extraction backends:**

| Mode | How to enable | Notes |
|---|---|---|
| LLM | `GRAPHRAG_ENABLED=true` and `GRAPHRAG_LLM=<model>` in `.env` | `LLMEntityRelationExtractor` with JSON-schema constrained output. No RAM requirement beyond your LLM. |
| Disabled | `GRAPHRAG_ENABLED` unset or `false` | Extraction skipped; graph retrieval leg inactive. |

**Data architecture:**
```
Qdrant  — source of truth for all chunk TEXT and VECTORS
Neo4j   — source of truth for GRAPH TOPOLOGY (entities, relationships, chunk linkage)
```
Vectors are never stored in Neo4j. Neo4j Chunk nodes cross-reference Qdrant by `qdrant_point_id`.

**Explore the graph:** http://localhost:7474/browser/ — login: `neo4j` / `ragwebui_neo4j`

```cypher
// See all entity types
MATCH (e:__Entity__) RETURN DISTINCT labels(e), count(*) ORDER BY count(*) DESC

// Find chunks linked to an entity
MATCH (c:Chunk)-[:FROM_CHUNK]-(e:__Entity__ {name: "Apple"}) RETURN c, e
```

**GraphRAG environment variables:**

| Variable | Description |
|---|---|
| `GRAPHRAG_ENABLED` | Set `false` to skip graph extraction at ingest time. |
| `GRAPHRAG_LLM` | Model name for LLM-based extraction. |
| `RETRIEVAL_GRAPH_ENABLED` | Enable/disable graph retrieval leg independently of ingestion. |
| `GRAPHRAG_RETRIEVAL_HOPS` | Relationship hops to traverse (default `2`). |

### Retrieval

| Variable | Description | Default |
|---|---|---|
| `RETRIEVAL_TOP_K` | Chunks returned per query | `10` |
| `RETRIEVAL_DENSE_ENABLED` | Enable/disable dense leg | `true` |
| `RETRIEVAL_QDRANT_SPARSE_ENABLED` | Enable/disable sparse leg | `true` |
| `RETRIEVAL_EXACT_ENABLED` | Enable/disable MySQL FTS leg | `true` |
| `QDRANT_MMR_DIVERSITY` | Qdrant native MMR diversity (0=pure relevance, 1=pure diversity) | `0.3` |
| `DEDUP_SEMANTIC_THRESHOLD` | Cosine similarity for semantic dedup (1.0=disabled) | `0.95` |
| `RERANKER_ENABLED` | Enable cross-encoder reranker | `true` |
| `RERANKER_MODEL` | HuggingFace cross-encoder model | `Xenova/ms-marco-MiniLM-L-12-v2` |
| `RERANKER_SCORE_THRESHOLD` | Minimum logit to pass reranker (default retrieval) | `-2.0` |

### Vector DB (Qdrant)

| Variable | Default |
|---|---|
| `QDRANT_HOST` | `qdrant` |
| `QDRANT_PORT` | `6333` |
| `QDRANT_GRPC_PORT` | `6334` |

### Storage & Auth

| Variable | Example |
|---|---|
| `MYSQL_SERVER` | `db` |
| `MYSQL_USER` | `ragwebui` |
| `MYSQL_PASSWORD` | `ragwebui` |
| `MYSQL_DATABASE` | `ragwebui` |
| `SECRET_KEY` | random string |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `10080` (7 days) |

## Multi-Tenancy & Admin Panel

The admin panel at `/dashboard/admin` provides org-level management, per-org LLM configuration, and data source management.

### Organisations

Admins create organisations and assign users, data sources, and LLM settings to them. Orgs support parent-child hierarchy with materialized paths.

- **Per-org LLM config:** Each org can have its own API base URL, model name, and query model via `PUT /api/admin/orgs/{org_id}/llm-config`
- **Per-org ingestion status:** Aggregated status across all KBs in an org (idle/running/completed/failed with doc counts)
- **Per-org abbreviations:** Custom short-expansion mappings for query matching

### Users

- **Create users** with role (user/admin/super_admin) and org assignment
- **Deactivate users** by setting `is_active=False` — deactivated users cannot log in
- **Permanent delete** (super admin only) — cascades to KBs, chats, and messages
- **Password change** for other users (super admin only)

### Data Sources (DataStores)

Folder-based document ingestion sources:

- **Create:** Point to a local folder, set scan pattern (e.g., `*.pdf,*.docx`), enable auto-scan with a debouncing interval
- **Auto-scan:** Event-driven file watching — new/modified files are detected via watchdog and ingested immediately after a 1-second write-completion delay; the interval only controls the minimum debounce window per datastore (prevents duplicate processing of rapid repeated events)
- **Assign to orgs:** Link data sources to specific organisations
- **Scan status:** Progress bar showing total/processed counts plus a breakdown of new/modified/skipped/error files; polling every 500ms when a scan is active
- **Manual trigger:** Force a full re-scan from the admin panel
- **Flush button:** When pending changes are queued (e.g., many files dropped rapidly), flush them immediately instead of waiting for the debounce window

#### Scan result fields

When a scan completes, the datastore record includes:

| Field | Description |
|-------|-------------|
| `last_scan_total_files` | Total files matched by the scan pattern on disk |
| `last_scan_processed` | Number of files successfully ingested |
| `last_scan_new` | Number of previously unseen files |
| `last_scan_modified` | Number of existing files with changed content (re-ingested) |
| `last_scan_skipped` | Number of files skipped (e.g., unsupported extension, pattern mismatch, hidden file) |
| `last_scan_errors` | Number of files that failed ingestion |
| `last_scan_error` | Error message if any files failed |

#### How auto-scan works

Auto-scan is **event-driven**, not periodic. When a file is added or modified in the folder, the watchdog observer detects the filesystem event and begins ingestion after a 1-second delay (to let the file write complete). The `auto_scan_interval_minutes` configures the **minimum processing interval** for the same datastore — rapid repeated events (e.g., VS Code temp-file write-and-rename) within this window are coalesced into a single processing run. This prevents duplicate ingestion of the same file.

If a manual scan is triggered while auto-scan is running, both operate independently: the manual scan walks all files, while event-driven ingestion continues to pick up new changes as they occur.

## API Reference

The OpenAPI reference is available at http://localhost:8000/redoc. Below are the key endpoints:

### Auth (prefix: `/api/auth`)

| Method | URL | Auth | Description |
|---|---|---|---|
| POST | `/register` | none | Register new user |
| POST | `/token` | none | OAuth2 login (rate-limited, exponential backoff) |
| GET | `/admin-only` | admin | Returns current admin user |
| POST | `/change-password` | user | Change own password |
| POST | `/test-token` | user | Validate token by returning current user |

### Config

| Method | URL | Auth | Description |
|---|---|---|---|
| GET | `/config` | none | Returns `chunk_size` and `chunk_overlap` settings |

### Knowledge Base (prefix: `/api/knowledge-base`)

| Method | URL | Auth | Description |
|---|---|---|---|
| POST | `/` | user | Create KB |
| GET | `/` | user | List KBs (paginated, skip/limit) |
| GET | `/{kb_id}` | user | Get KB with documents |
| PUT | `/{kb_id}` | user | Update KB |
| DELETE | `/{kb_id}` | user | Delete KB (direct uploads only) |
| POST | `/{kb_id}/documents/upload` | user | Batch upload documents |
| POST | `/{kb_id}/documents/preview` | user | Preview document chunks |
| POST | `/{kb_id}/documents/process` | user | Process uploaded docs |
| GET | `/{kb_id}/documents/tasks` | user | Get processing task statuses |
| DELETE | `/{kb_id}/documents/{doc_id}` | user | Delete a single document |
| GET | `/{kb_id}/documents/{doc_id}` | user | Get document details |
| POST | `/cleanup` | user | Clean expired temp uploads (>24h) |
| POST | `/test-retrieval` | user | Test retrieval quality for a query |
| POST | `/{kb_id}/link-datastore` | user | Link a datastore to KB |
| DELETE | `/{kb_id}/unlink-datastore/{data_store_id}` | user | Unlink a datastore from KB |

### Chat (prefix: `/api/chat`)

| Method | URL | Auth | Description |
|---|---|---|---|
| POST | `/` | user | Create chat |
| GET | `/` | user | List chats (paginated) |
| GET | `/search` | user | Full-text search across messages |
| GET | `/{chat_id}` | user | Get chat (with messages) |
| PATCH | `/{chat_id}` | user | Update chat (title, pinned, retrieval flags) |
| DELETE | `/{chat_id}` | user | Delete chat |
| POST | `/{chat_id}/cancel` | user | Cancel streaming response |
| GET | `/{chat_id}/export` | user | Export chat as Markdown |
| GET | `/{chat_id}/messages/paginated` | user | Paginated messages (cursor-based) |
| POST | `/{chat_id}/messages` | user | Send message (JSON body, streaming) |
| POST | `/{chat_id}/messages/with-file` | user | Send message with file upload (multipart) |
| DELETE | `/{chat_id}/messages/{message_id}` | user | Delete an assistant message |
| GET | `/{chat_id}/messages/{message_id}/export` | user | Export message as PDF/Word/image |
| PATCH | `/messages/{message_id}` | user | Edit message (creates branch) |
| GET | `/messages/{message_id}/siblings` | user | Get branch siblings |

### Chat Files (prefix: `/api/chat`)

| Method | URL | Auth | Description |
|---|---|---|---|
| POST | `/{chat_id}/files` | user | Upload file to chat (10MB limit, async conversion) |
| GET | `/{chat_id}/files/{file_id}` | user | Poll file processing status |
| DELETE | `/{chat_id}/files/{file_id}` | user | Delete file record |
| GET | `/{chat_id}/files/{file_id}/download` | user | Download original file |

### Folders (prefix: `/api/folders`)

| Method | URL | Auth | Description |
|---|---|---|---|
| POST | `/` | user | Create folder |
| GET | `/` | user | List folders |
| PATCH | `/{folder_id}` | user | Rename folder |
| DELETE | `/{folder_id}` | user | Delete folder |
| PATCH | `/{folder_id}/chats/{chat_id}` | user | Assign chat to folder |
| DELETE | `/{folder_id}/chats/{chat_id}` | user | Unassign chat from folder |

### Query (prefix: `/api/query`)

| Method | URL | Auth | Description |
|---|---|---|---|
| POST | `/` | user | Stateless RAG query (JSON, no SSE) |
| GET | `/kb/{kb_id}/ingest-status` | user | KB processing readiness check |

### Admin — Orgs (prefix: `/api/admin/orgs`)

| Method | URL | Auth | Description |
|---|---|---|---|
| GET | `/` | admin | List all orgs (with user_count, hierarchy) |
| POST | `/` | admin | Create org (parent_id required, auto-computes path) |
| PATCH | `/{org_id}` | admin | Update org (name, parent) |
| DELETE | `/{org_id}` | admin | Delete org (no children, no users) |
| GET | `/{org_id}/llm-config` | admin | Get org LLM config |
| PUT | `/{org_id}/llm-config` | admin | Upsert org LLM config (api_base, model_name, query_model) |
| GET | `/{org_id}/ingestion-status` | admin | Aggregated ingestion status for all KBs in org |
| POST | `/{org_id}/abbreviations` | admin | Create abbreviation |
| GET | `/{org_id}/abbreviations` | admin | List abbreviations |
| DELETE | `/{org_id}/abbreviations/{abbrev_id}` | admin | Delete abbreviation |

### Admin — Users (prefix: `/api/admin/users`)

| Method | URL | Auth | Description |
|---|---|---|---|
| GET | `/` | admin | List users (super_admin sees all; admin sees own org) |
| POST | `/` | admin | Create user (requires org_id) |
| PATCH | `/{user_id}` | admin | Update user (role, org_id, is_active) |
| POST | `/{user_id}/change-password` | admin | Change password (super_admin only) |
| DELETE | `/{user_id}` | admin | Delete user (super_admin only) |

### Admin — Data Stores (prefix: `/api/admin/datastores`)

| Method | URL | Auth | Description |
|---|---|---|---|
| GET | `/datastores` | admin | List all datastores |
| POST | `/datastores` | admin | Create datastore |
| GET | `/datastores/{id}` | admin | Get datastore details |
| PATCH | `/datastores/{id}` | admin | Update datastore |
| DELETE | `/datastores/{id}` | admin | Delete datastore (204 on success) |
| POST | `/datastores/{id}/assign` | admin | Assign datastore to orgs (body: `{"org_ids": []}`) |
| DELETE | `/datastores/{id}/assign` | admin | Unassign datastore from orgs (body: `{"org_ids": []}`) |
| GET | `/datastores/{id}/status` | admin | Get datastore scan status |
| POST | `/datastores/{id}/scan` | admin | Trigger manual scan |
| GET | `/datastores/{id}/scan-progress` | admin | Get scan progress (polling; real-time during scan, DB after completion) |
| GET | `/datastores/{id}/scan-progress-stream` | admin | SSE endpoint for real-time scan progress (rarely used — frontend uses polling) |
| POST | `/datastores/{id}/flush` | admin | Process pending changes immediately |
| POST | `/datastores/{id}/stop-scan` | admin | Cancel a running scan |

#### Scan progress response fields

| Field | Description |
|-------|-------------|
| `datastore_id` | Datastore identifier |
| `datastore_name` | Datastore name |
| `scan_id` | Integer scan ID (null when no scan is running) |
| `total_files` | Total files matched by the scan pattern |
| `processed_files` | Files processed so far during the scan |
| `new_files` | Files ingested as new documents |
| `modified_files` | Existing files with changed content that were re-ingested |
| `skipped_files` | Files skipped during the scan |
| `error_files` | Files that failed ingestion |
| `status` | `running`, `completed`, `error`, `idle`, or `cancelled` |
| `last_scan_at` | Timestamp of the last scan |
| `error_message` | Error message if the scan failed |

#### Flush response fields

| Field | Description |
|-------|-------------|
| `datastore_id` | Datastore identifier |
| `pending_processed` | Number of changes processed in the flush |
| `processing` | Whether the datastore is still processing | |

### Admin — Counts

| Method | URL | Auth | Description |
|---|---|---|---|
| GET | `/counts` | admin | Returns `{ organizations, users }` |

## Admin & Developer Tools

| Tool | URL | Purpose |
|---|---|---|
| **RAG Web UI** | http://localhost:3000 | Main application |
| **Backend API Docs** | http://localhost:8000/redoc | OpenAPI reference |
| **Qdrant Dashboard** | http://localhost:6333/dashboard | Browse collections, inspect points |
| **Neo4j Browser** | http://localhost:7474/browser/ | Explore entity/relationship graph |
| **Adminer** | http://localhost:8081 | MySQL web GUI (dev only) | Note: Adminer was moved from port 8080 to 8081 to avoid conflicts with other services.

Adminer is in `docker-compose.dev.yml`:
```bash
docker compose -f docker-compose.dev.yml up -d adminer
```
Login: System=MySQL, Server=`db`, User=`ragwebui`, Password=`ragwebui`, Database=`ragwebui`.

## Features

- Upload PDF, DOCX, PPTX, XLSX, Markdown, HTML, CSV, JSON, XML, email, EPUB, images (OCR), ZIP archives
- Optional OCR for scanned PDFs and embedded images via `markitdown-ocr` — enabled by `VISION_MODEL`
- **Agentic pipeline**: query decomposition → parallel sub-query retrieval with reinforced scoring → LLM draft-grade loop → widened retrieval retry → keyword search fallback → partial-answer transparency
- **Pipeline extras**: confidence scoring, query classification (FACTUAL/ENTITY_CENTRIC/MULTI_PART/AMBIGUOUS), tool trace, synthesis mode
- **3-leg hybrid retrieval**: dense vector + SPLADE sparse + MySQL FULLTEXT, with native Qdrant MMR diversity and recency-aware dedup (exact + semantic)
- **GraphRAG**: optional entity/relationship extraction into Neo4j with graph-traversal retrieval expansion
- **Cross-encoder reranking**: retrieved candidates re-ranked by a local cross-encoder before context assembly
- **Chat file upload**: attach any supported document; content injected directly into pipeline (not indexed); 10 MB size limit + 25% context-window token budget; smart section extraction
- **Chat features**: branching (multiple answer variants), folder organisation, message search, chat export (Markdown), message export (PDF/Word/image), pagination (infinite scroll), collapsible sidebar with localStorage persistence
- **Streaming responses** with real-time AgentTimeline showing each pipeline step (active → done with detail on click)
- **Clickable citations** `[N]` in answers — linked to source chunk with score bar and leg badge
- **Stop button** during generation (AbortController); partial message preserved with `*(generation stopped)*`
- **Rate limiting** on login: 3 failed attempts trigger exponential backoff (15s → 30s → 60s → 120s → 240s → 480s → 900s)
- **Multi-tenancy**: org-level user management, per-org LLM config, and data source assignment
- **Dark / light / system theme toggle**
- **Multi-turn chat** with rolling conversation summary

## Troubleshooting

See [docs/troubleshooting.md](docs/troubleshooting.md).

## License

[Apache-2.0](LICENSE)

---

<div align="center">If this project helps you, please give it a ⭐️</div>
