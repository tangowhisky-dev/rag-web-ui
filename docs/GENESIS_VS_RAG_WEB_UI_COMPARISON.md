# Genesis vs RAG-Web-UI — Detailed Comparison

*Generated: 2026-06-02*

---

## 1. Purpose & Design Philosophy

| Dimension | Genesis | RAG-Web-UI |
|-----------|---------|------------|
| **Primary use case** | Enterprise Office Brief Generation — evidence-constrained answers for government/intelligence staff producing formal 1-2 page briefs | General-purpose RAG chat assistant — conversational Q&A over user-uploaded knowledge bases |
| **Audience** | Organisations with a defined hierarchy (orgs → wings → sections → cells); admin-managed users | Individual users managing their own knowledge bases and chats |
| **Tone** | Formal; every statement must carry a citation; "no record found" is a valid answer | Conversational; citations encouraged but pipeline adapts to available evidence |
| **Deployment model** | On-premises, air-gapped-friendly; LLM/Embedding/Reranker served by XInference/vLLM on a local DGX node | Cloud or on-prem; LLM providers are OpenAI-compatible (LM Studio, Ollama, any OpenAI-compatible endpoint) |
| **Multi-tenancy** | First-class — org hierarchy enforced at DB and vector-store level | None — single-tenant per deployment; knowledge bases are user-scoped only |

---

## 2. Architecture Overview

### Genesis

```
Windows File Server (SMB)
        ↓
Ubuntu HP Server (256 GB RAM)
  ├── FastAPI (port 8000)
  ├── PostgreSQL 15  — metadata, chunks metadata, org hierarchy, users
  ├── Milvus 2.4.0   — dense vectors + BM25 sparse vectors + raw text
  ├── Redis 7        — caching / task queue
  └── Nginx          — reverse proxy
        ↓ (GPU inference)
NVIDIA DGX Spark (128 GB VRAM)
  └── XInference / vLLM / LM Studio / Ollama
        ├── LLM (nvidia-nemotron, etc.)
        ├── Qwen3-Embedding-0.6B
        └── Qwen3-Reranker-0.6B
```

### RAG-Web-UI

```
User Browser
     ↓
Next.js Frontend (port 3000)
     ↓
FastAPI Backend (port 8000)
  ├── MySQL 8.4      — all metadata, chat history, confidence scores
  ├── Qdrant         — dense vectors (cosine) + sparse (SPLADE)
  ├── Neo4j          — knowledge graph (optional GraphRAG leg)
  └── Any OpenAI-compatible LLM endpoint
```

**Key architectural difference**: Genesis uses Milvus as a single store for both semantic vectors and BM25 full-text search, delegating embedding computation server-side to XInference. RAG-Web-UI separates concerns — Qdrant for vectors, MySQL for full-text (exact search), Neo4j for graph — and generates embeddings client-side in the ingestion pipeline.

---

## 3. Ingestion Pipeline

| Step | Genesis | RAG-Web-UI |
|------|---------|------------|
| **Trigger** | Admin manual trigger or file-watcher (SMB/local folder polling) | User manual upload via UI |
| **Source** | Windows SMB shares, local folders; automated folder scanning with pattern matching | User-uploaded files via browser |
| **Hash detection** | SHA-256 per file — skips unchanged documents | Not implemented (re-processes on re-upload) |
| **OCR** | Three backends: Tesseract, PaddleOCR API, Vision Model (configurable at runtime) | Via `markitdown` library (no scanned-document OCR) |
| **Chunking** | Fixed 500-char segments, 50-char overlap | Configurable: semantic + paragraph-aware via `markitdown`; multiple strategies |
| **Embedding** | **Server-side in Milvus** — text is sent to Milvus which calls XInference to embed | **Client-side in pipeline** — embeddings generated in Python then uploaded to Qdrant |
| **Sparse vectors** | Milvus built-in BM25 (server-side) | SPLADE via `fastembed` (client-side) |
| **Graph extraction** | Not implemented | Entity extraction → Neo4j graph for multi-hop GraphRAG |
| **Progress tracking** | `org_ingestion_status` table, per-org status (`running`/`completed`/`failed`) | `processing_tasks` table, per-document progress with `graph_status` field |

**Notable Genesis strength**: the combination of hash-based incremental ingestion and server-side embedding makes it suitable for continuous background sync of large shared drives without re-embedding unchanged documents.

**Notable RAG-Web-UI strength**: semantic chunking and graph entity extraction provide richer context; multiple retrieval legs (dense + sparse + exact + graph) with RRF merging.

---

## 4. Retrieval Pipeline

### Genesis Retrieval Flow
```
User query
  ↓ QueryRewriter (LLM-based, resolves pronouns from chat history)
  ↓ QueryExpander (abbreviation expansion, plural/singular variants)
  ↓ Milvus hybrid_search (vector cosine + BM25, with org_id filter)
  ↓ XInference Reranker (Qwen3-Reranker-0.6B)
  ↓ RAGFlow-style weighted score:
        reranker_score × 0.3 + normalised_BM25 × 0.7
  ↓ Top-10 chunks → LLM
```

### RAG-Web-UI Retrieval Flow (fast/thinking mode)
```
User query
  ↓ _rewrite_query (LLM-based standalone query)
  ↓ hybrid_search_with_legs (parallel):
     ├── Dense leg:   Qdrant cosine similarity
     ├── Sparse leg:  Qdrant SPLADE
     ├── Exact leg:   MySQL full-text BM25
     └── Graph leg:   Neo4j multi-hop (optional)
  ↓ RRF merge + graph expansion + entity enrichment
  ↓ CrossEncoder reranker (fastembed ONNX, ms-marco-MiniLM-L-12-v2)
  ↓ Threshold filtering → LLM
```

| Dimension | Genesis | RAG-Web-UI |
|-----------|---------|------------|
| **Vector search** | Milvus cosine (dense via XInference) | Qdrant cosine (dense via fastembed) |
| **Sparse/keyword** | Milvus BM25 (server-side) | SPLADE (fastembed) + MySQL full-text |
| **Graph search** | None | Neo4j multi-hop GraphRAG (optional leg) |
| **Fusion** | Manual union + dedup, weighted score | RRF (Reciprocal Rank Fusion) across legs |
| **Reranker** | Qwen3-Reranker-0.6B via XInference (HTTP API, GPU) | ms-marco-MiniLM-L-12-v2 via fastembed ONNX (CPU, in-process) |
| **Score formula** | `keyword×0.7 + rerank×0.3` (RAGFlow-style) | Reranker logit threshold (bimodal: relevant 1–10, irrelevant −5 to −11) |
| **Org/tenant filtering** | `filter='org_id == "uuid"'` in Milvus | KB-level filtering (user owns KBs) |
| **Multi-query decomposition** | No | Yes (agentic pipeline: 2–5 sub-queries, parallel retrieval per sub-query) |
| **Coverage grading** | No | Yes (agentic: draft → LLM grade coverage → retry loop with widened retrieval or keyword fallback) |
| **Confidence scoring** | Threshold check (`top_final_score >= 0.3`) | Multi-signal: reranker top score (60%), mean score (30%), evidence count (10%) → 0–100 score + level label |

**Key difference**: Genesis uses a single weighted score formula borrowed from RAGFlow. RAG-Web-UI uses RRF for multi-leg fusion, then a separate reranker pass for quality filtering, producing a richer confidence signal.

---

## 5. Answering Pipelines

Genesis has **one pipeline** (single-stage retrieval → LLM generation).

RAG-Web-UI has **three selectable pipelines**:

| Mode | Description | Latency |
|------|-------------|---------|
| **Fast** | Rewrite → hybrid search → generate | ~2–4s |
| **Thinking** | Same as Fast, but uses a reasoning model | ~5–15s |
| **Agentic** | Full LangGraph: rewrite → route → decompose → parallel retrieval (per sub-query) → extract file sections → draft answer → grade coverage → conditional retry loop (widened retrieval → keyword search) → final answer | ~8–20s |

The agentic pipeline's iterative retrieval-grading loop has no equivalent in Genesis.

---

## 6. Conversation History & Context Management

| Dimension | Genesis | RAG-Web-UI |
|-----------|---------|------------|
| **History window** | Last 100 messages passed to LLM (per product spec) | Sliding window: last 6 messages; older messages are summarised by a rolling LLM call |
| **Rolling summary** | Not implemented | After every turn: LLM produces a condensed summary of all prior turns; injected as a system message |
| **Generator context** | Raw history messages | Summary only (raw history removed to prevent contamination) |
| **Query rewriter context** | Last 6 messages (raw, truncated) | Last 6 messages (raw, AI truncated to 400 chars) |
| **Chat history retrieval** | No | Yes (agentic): prior assistant answers scored by reranker; relevant ones injected as `[Prior Answer]` context docs |
| **Chat storage** | PostgreSQL `chats` + `messages` tables | MySQL `chats` + `messages` tables |
| **Rewritten query persistence** | No | Yes: `messages.rewritten_query` column; shown in UI and included in exports |

---

## 7. Knowledge Base & Data Model

### Genesis data model (PostgreSQL)
```
Organization (tree: orgs → wings → sections → cells)
  ├── User (role: admin/super_admin/user)
  ├── DataStore (folder config + scan status)
  ├── Document (file_path, file_hash, org_id)
  ├── Chunk (content, chunk_index, doc_id — metadata only; text stored in Milvus)
  └── Chat → Message → MessageAttachment
```

### RAG-Web-UI data model (MySQL)
```
User
  └── KnowledgeBase (multiple per user)
        ├── DocumentUpload (source file)
        └── Chunk (content, embedding metadata, processing status)
                  ↕ (also in Qdrant + Neo4j)
User
  └── Chat (per KB selection)
        └── Message (content, citations, confidence, rewritten_query)
                └── ChatFile (file attachment per turn)
```

**Notable difference**: Genesis separates chunk metadata (PostgreSQL) from chunk content (Milvus). RAG-Web-UI stores chunk content in MySQL alongside metadata (Qdrant holds only vectors; MySQL holds the full text too).

---

## 8. LLM Provider Integration

| Dimension | Genesis | RAG-Web-UI |
|-----------|---------|------------|
| **Provider abstraction** | `BaseLLMProvider` → `LMStudioProvider`, `OllamaProvider`, `XInferenceProvider`, `VLLMProvider` (pluggable) | Single OpenAI-compatible client (`openai.AsyncOpenAI`) with configurable `api_base` and `api_key` |
| **Per-org LLM config** | Yes — each organisation can have a different LLM endpoint and model | No — single system-wide LLM config |
| **Streaming** | Yes (`StreamingResponse`) | Yes (SSE with typed events: `0:` tokens, `1:` rewritten query, `2:` context, `4:` agent steps, `r:` normalisation) |
| **Embedding** | Server-side via XInference (GPU) | Client-side via `fastembed` (CPU ONNX) |
| **Reranker** | XInference HTTP API (GPU) | `fastembed` ONNX in-process (CPU) |
| **Temperature / model override** | Per-org in DB | Per-chat in UI; per-request override |

---

## 9. Citation System

| Dimension | Genesis | RAG-Web-UI |
|-----------|---------|------------|
| **Citation format (LLM output)** | `[1]`, `[2]` numeric references | `[citation:N]` → normalised to `[N](N)` markdown links |
| **Citation validation** | `CitationValidator` class — post-hoc LLM call to verify claims are grounded | No post-hoc validation; reranker threshold acts as quality gate |
| **Citation display (UI)** | Not detailed in reviewed code | Popover on hover: shows source document, KB name, reranker score, retrieval leg badge |
| **Export citations** | Not reviewed | Word/PDF/Image export per message; Markdown export includes rewritten query block |
| **Confidence score** | Simple `top_score >= 0.3` relevance flag | Multi-signal 0–100 score with `very_high/high/medium/low/none` level displayed in UI |

---

## 10. Frontend

| Dimension | Genesis | RAG-Web-UI |
|-----------|---------|------------|
| **Framework** | Next.js 14 + React + Tailwind | Next.js (App Router) + React + Tailwind + shadcn/ui |
| **Input** | Textarea (details not reviewed) | Auto-grow textarea (2–10 lines), `scrollHeight` resize pattern |
| **Chat rendering** | Markdown (details not reviewed) | `react-markdown` + `remark-gfm` + `rehype-highlight` + KaTeX math + Mermaid diagrams |
| **Streaming UI** | Yes | Yes; includes AgentTimeline (node step progress), ThinkBlock (reasoning display), citation popovers |
| **Message pagination** | Not implemented (last 100 messages loaded at once) | Cursor-based pagination: initial load of last 20 messages; `IntersectionObserver` loads older pages on scroll |
| **Admin panel** | Dedicated `/admin` route: user management, org hierarchy, ingestion control | Chat settings per session; no admin panel |
| **Themes** | Dark/Light toggle | Dark/Light toggle |
| **Export** | Word download (per spec) | Word, PDF, PNG per message; full chat Markdown export |
| **File attachment** | Per message (MessageAttachment model) | Per turn (ChatFile model); supports PDF, DOCX, images, etc. |

---

## 11. Ingestion Monitoring & Observability

| Dimension | Genesis | RAG-Web-UI |
|-----------|---------|------------|
| **Ingestion status** | `org_ingestion_status` table; admin can see per-org running/completed/failed | `processing_tasks` table with per-document progress % and graph status |
| **File-watcher** | `watcher.py` + `local_folder_ingest.py` — continuous background monitoring | None (manual upload only) |
| **Background jobs** | Separate `ingestion` Docker service (independent from API) | Background tasks in FastAPI (summary updates via `asyncio.ensure_future`) |
| **Hash deduplication** | SHA-256 per file, skips unchanged | None |
| **Error recovery** | `ProgressTimeout` pattern — only times out if no progress for N seconds (allows long OCR) | Standard exception handling + DB status flags |

---

## 12. Security & Access Control

| Dimension | Genesis | RAG-Web-UI |
|-----------|---------|------------|
| **Auth** | JWT (`user_id`, `org_id`, `role`) | JWT (`user_id`) |
| **Roles** | `admin`, `super_admin`, `user` | Single role (user); no admin panel |
| **Org isolation** | Hard org_id filter in every Milvus query | KB ownership (user_id FK); no cross-user access |
| **Rate limiting** | Not reviewed | Not implemented |
| **Cancellation** | `active_streams` dict; `/api/chat/stream/cancel/{stream_id}` endpoint | AbortController on frontend; no server-side cancellation |

---

## 13. What Each Could Learn From the Other

### Things RAG-Web-UI could adopt from Genesis
1. **Hash-based incremental ingestion** — avoid re-embedding unchanged documents on re-upload
2. **Server-side embedding via XInference/Milvus** — offload embedding computation to GPU, keeping the API service lightweight
3. **`CitationValidator`** — post-generation check that cited facts are actually grounded
4. **`QueryExpander`** — abbreviation/acronym expansion before retrieval (especially useful in domain-specific KBs)
5. **Per-org LLM configuration** — different teams using different models
6. **File-watcher + SMB ingestion** — automated document sync from network shares
7. **`ProgressTimeout` pattern** — smarter timeout for long OCR jobs
8. **Streaming cancellation endpoint** — server-side abort for in-flight generation

### Things Genesis could adopt from RAG-Web-UI
1. **Multi-pipeline answering** (Fast/Thinking/Agentic) — expose a simple mode vs. thorough mode
2. **Agentic query decomposition + coverage grading** — break complex queries into sub-questions, grade and retry
3. **SPLADE sparse vectors** — richer sparse retrieval beyond BM25
4. **Neo4j GraphRAG** — multi-hop entity relationships for complex intelligence queries (particularly valuable for genesis's domain)
5. **RRF fusion** — principled multi-leg score merging rather than manual weighted formula
6. **Rolling conversation summary** — avoid sending all 100 messages to the LLM; instead summarise older turns
7. **Summary-only generator context** — prevents history contamination ("your statement is partially correct")
8. **Chat history retrieval as a source** — use reranker to find relevant prior answers before hitting KB
9. **Richer confidence scoring** — multi-signal 0–100 score rather than a binary relevance flag
10. **Cursor-based message pagination** — load last N messages on open; scroll up for history
11. **`rewritten_query` persistence** — store and display the actual retrieval query, not just the original
12. **Reranker via fastembed ONNX** — eliminates PyTorch dependency; ~1.8× faster on CPU vs. sentence-transformers

---

## 14. Summary Scorecard

| Capability | Genesis | RAG-Web-UI |
|------------|---------|------------|
| Multi-tenancy / org hierarchy | ✅ Full | ❌ Single tenant |
| Automated ingestion (file watch / SMB) | ✅ | ❌ Manual only |
| Incremental ingestion (hash dedup) | ✅ | ❌ |
| OCR (scanned documents) | ✅ Three backends | ⚠️ Via markitdown only |
| Server-side GPU embedding | ✅ | ❌ CPU ONNX |
| Graph RAG | ❌ | ✅ Neo4j |
| Multi-leg retrieval (dense+sparse+exact+graph) | ⚠️ Dense+BM25 only | ✅ 4 legs |
| Agentic pipeline (decompose/grade/retry) | ❌ | ✅ |
| Rolling conversation summary | ❌ | ✅ |
| Per-org LLM configuration | ✅ | ❌ Single config |
| Citation validation (post-hoc) | ✅ | ❌ |
| Confidence scoring (multi-signal) | ⚠️ Binary threshold | ✅ 0–100 score |
| Message pagination (lazy load) | ❌ | ✅ Cursor-based |
| Rewritten query persistence | ❌ | ✅ |
| Admin panel | ✅ Full | ❌ |
| Word/PDF/Image export | ✅ Word | ✅ Word + PDF + PNG + Markdown |
| Streaming with typed SSE events | ⚠️ Basic | ✅ Rich typed protocol |
| AgentTimeline / step visibility | ❌ | ✅ |
