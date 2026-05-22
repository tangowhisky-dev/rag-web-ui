<div align="center">
  <img src="https://raw.githubusercontent.com/rag-web-ui/rag-web-ui/main/docs/images/github-cover-new.png" alt="RAG Web UI">
  <br />
  <p>
    <strong>Knowledge Base Management with Retrieval-Augmented Generation</strong>
  </p>
  <p>
    <a href="https://github.com/rag-web-ui/rag-web-ui/blob/main/LICENSE"><img src="https://img.shields.io/github/license/rag-web-ui/rag-web-ui" alt="License"></a>
    <a href="#"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python"></a>
    <a href="#"><img src="https://img.shields.io/badge/node-%3E%3D18-green.svg" alt="Node"></a>
    <a href="#"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome"></a>
  </p>
</div>

## Introduction

RAG Web UI is a self-hosted knowledge base Q&A system. Upload your documents, then chat with them. Uses any **OpenAI-compatible API** for LLM and embeddings — works with OpenAI, LM Studio, or any local model server.

Retrieval uses **3-leg hybrid search**: dense vector (Qdrant cosine), sparse vector (SPLADE via Qdrant), and exact keyword (MySQL full-text), combined by Reciprocal Rank Fusion (RRF) with per-query-type weight presets. An LLM **query classifier** routes each query (FACTUAL / ENTITY_CENTRIC / MULTI_PART / AMBIGUOUS) to the best retrieval config. A **cross-encoder reranker** refines the top candidates before answer generation. For complex queries the system enters **synthesis mode**, fanning out multiple sub-queries in parallel and producing a structured Markdown report. Optionally extended with **GraphRAG**: entity/relationship extraction into Neo4j enables graph-traversal expansion and **entity-aware retrieval** that boosts chunks mentioning query entities.

> **Based on:** This is an opinionated, slimmed-down fork of [rag-web-ui/rag-web-ui](https://github.com/rag-web-ui/rag-web-ui). All credit for the original design and implementation goes to the original authors. The goal of this fork is to serve as a learning resource for understanding the RAG pipeline end-to-end — keeping minimal dependencies, removing abstraction layers, and adding visibility into individual RAG components (retrieval legs, reranking, prompt construction, token flow).

## Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, TypeScript, Tailwind CSS, shadcn/ui, Vercel AI SDK |
| Backend | Python FastAPI, LangChain, SQLAlchemy |
| Vector DB | Qdrant (dense + sparse vectors) |
| Graph DB | Neo4j (entity/relationship graph for GraphRAG — optional) |
| Sparse Embeddings | SPLADE via FastEmbed (CPU, local) |
| File Storage | Local Folder mapped to Docker as Volume |
| Database | MySQL 8 |

## Quick Start

**Prerequisites:** Docker & Docker Compose v2+

```bash
git clone https://github.com/tangowhisky-dev/rag-web-ui.git
cd rag-web-ui
cp .env.example .env
# Edit .env — set OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_MODEL, DENSE_EMBEDDINGS_MODEL,
#              DENSE_EMBEDDING_DIM.  Optionally set QUERY_MODEL (query rewriting),
#              VISION_MODEL (OCR), and OPENAI_VISION_API_BASE.
docker compose up -d --build
```

Open **http://localhost:3000** — register an account and start uploading documents.

> **First run note:** The SPLADE model (~500 MB) is downloaded on first document ingestion if not pre-cached. To pre-download it into `./assets/fastembed/` see [Pre-downloading the SPLADE model](#pre-downloading-the-splade-model).

## Configuration

Copy `.env.example` to `.env` and set these values:

### LLM & Embeddings

| Variable | Required | Description | Example |
|---|---|---|---|
| `OPENAI_API_KEY` | yes | API key for your LLM provider | `sk-...` or `lmstudio` |
| `OPENAI_API_BASE` | yes | Base URL of OpenAI-compatible API | `https://api.openai.com/v1` |
| `OPENAI_MODEL` | yes | Response-generation model | `gpt-4o` |
| `QUERY_MODEL` | no | Model for query rewriting and rolling summarisation. Falls back to `OPENAI_MODEL` when unset. A smaller/faster model works well here. | `gpt-4o-mini` |
| `VISION_MODEL` | no | Multimodal model for markitdown-ocr OCR of scanned PDFs and embedded images. Leave unset to disable OCR. | `gpt-4o-mini` |
| `OPENAI_VISION_API_BASE` | no | Base URL for the vision model when it lives on a different server. Falls back to `OPENAI_API_BASE`. | `http://host.docker.internal:11434/v1` |
| `DENSE_EMBEDDINGS_MODEL` | yes | Embedding model name | `text-embedding-3-small` |
| `DENSE_EMBEDDING_DIM` | yes | Output dimension of the embedding model | `1536` for OpenAI, `1024` for qwen3-0.6b |
| `OPENAI_MODEL_CONTEXT_SIZE` | no | Total context window of `OPENAI_MODEL` in tokens. Used to limit ephemeral file content to 25% of the window. | `131072` |

**Using a local model server (e.g. LM Studio):**
```env
OPENAI_API_KEY=***
OPENAI_API_BASE=http://host.docker.internal:1234/v1
OPENAI_MODEL=your-chat-model
QUERY_MODEL=your-fast-model        # optional — reuse OPENAI_MODEL if unset
VISION_MODEL=your-vision-model     # optional — enables OCR for scanned PDFs / images
DENSE_EMBEDDINGS_MODEL=your-embedding-model
DENSE_EMBEDDING_DIM=1024
```

### Chunking

| Variable | Description | Default |
|---|---|---|
| `CHUNK_SIZE` | Target chunk size in **characters** (not tokens). Keep <= 1800 for SPLADE compatibility. | `1500` |
| `OVERLAP_PERCENTAGE` | Fraction of `CHUNK_SIZE` repeated at chunk boundaries (0.0-1.0). | `0.20` |

> **Warning:** Do not change these after ingesting documents. Re-upload existing documents to re-index with new settings. See [docs/chunking.md](docs/chunking.md).

### GraphRAG (Knowledge Graph)

GraphRAG enriches the retrieval pipeline by extracting entities and relationships from ingested chunks and storing them as a graph in Neo4j. At query time, vector search results are expanded by traversing entity connections — surfacing related chunks that similarity search alone would miss.

**Extraction backends (mutually exclusive, controlled by `.env`):**

| Mode | How to enable | Notes |
|---|---|---|
| ReLiK | `COMPOSE_PROFILES=relik` in `.env` | Local NER+RE model. Accurate, zero API cost per chunk, but requires 12–16 GB RAM to load Wikipedia-scale indexes. |
| LLM | `GRAPHRAG_LLM=<model>` in `.env` | Uses `neo4j-graphrag` Pipeline + `LLMEntityRelationExtractor` with `use_structured_output=True` (JSON Schema-constrained). Flexible, no RAM requirement beyond your LLM server. |
| Disabled | Both unset | Graph extraction silently skipped. Documents remain fully searchable via 3-leg hybrid. |

**Priority:** if `GRAPHRAG_LLM` is set, it takes precedence over ReLiK regardless of `COMPOSE_PROFILES`.

**Data architecture:**

```
Qdrant  — source of truth for all chunk TEXT and VECTORS
Neo4j   — source of truth for GRAPH TOPOLOGY (entities, relationships, chunk linkage)
```

Vectors are never stored in Neo4j. Neo4j Chunk nodes are cross-referenced to Qdrant by `qdrant_point_id` (the exact UUID Qdrant uses as point ID), enabling bidirectional lookup in a single index hit.

**Retrieval flow with GraphRAG enabled:**

```
vector search → RRF merge → graph expansion (new chunks via entity traversal)
             → graph enrichment (append triples to all candidates) → reranker → LLM
```

Graph expansion finds chunks that share entities with the seed results but weren't returned by vector similarity — a query about "Steve Jobs" may surface an Apple product philosophy chunk that never mentions Jobs by name but is entity-connected through the graph.

**Environment variables:**

| Variable | Description |
|---|---|
| `GRAPHRAG_ENABLED` | Set `false` to skip graph extraction at ingest time. Retrieval graph leg also becomes inactive. |
| `COMPOSE_PROFILES` | Set to `relik` to start the ReLiK container with the stack. |
| `GRAPHRAG_LLM` | Model name for LLM-based extraction (e.g. `gpt-4o`, `qwen/qwen3.5-4b`). |
| `RETRIEVAL_GRAPH_ENABLED` | Enable/disable graph retrieval leg independently of ingestion. |
| `GRAPHRAG_RETRIEVAL_HOPS` | Relationship hops to traverse from seed nodes (default `2`). |

**Explore the graph:** Neo4j Browser is available at **http://localhost:7474/browser/**

Log in with:
- Username: `neo4j`
- Password: `ragwebui_neo4j`

Useful Cypher queries to explore the graph:
```cypher
// See all entity types
MATCH (e:__Entity__) RETURN DISTINCT labels(e), count(*) ORDER BY count(*) DESC

// See entities and their relationships
MATCH (a:__Entity__)-[r]->(b:__Entity__) RETURN a, r, b LIMIT 50

// Find which chunks contain a given entity
MATCH (c:Chunk)-[:FROM_CHUNK]-(e:__Entity__ {name: "Apple"}) RETURN c, e
```



| Variable | Description | Default |
|---|---|---|
| `RETRIEVAL_TOP_K` | Number of chunks returned per query | `6` |
| `HYBRID_DENSE_WEIGHT` | Weight for dense vector leg | `0.5` |
| `HYBRID_QDRANT_SPARSE_WEIGHT` | Weight for SPLADE sparse leg | `0.3` |
| `HYBRID_EXACT_WEIGHT` | Weight for MySQL full-text leg | `0.2` |
| `RETRIEVAL_DENSE_ENABLED` | Enable/disable dense retrieval leg | `true` |
| `RETRIEVAL_QDRANT_SPARSE_ENABLED` | Enable/disable SPLADE sparse leg | `true` |
| `RETRIEVAL_EXACT_ENABLED` | Enable/disable MySQL FTS leg | `true` |

### Adaptive Retrieval & Agentic Features (M001)

| Variable | Description | Default |
|---|---|---|
| `QUERY_CLASSIFIER_ENABLED` | Enable LLM query classifier (FACTUAL/ENTITY_CENTRIC/MULTI_PART/AMBIGUOUS). Disable to use default hybrid config for all queries. | `true` |
| `RETRIEVAL_CONFIG_PRESETS` | JSON overrides per query type for leg weights and top-k. Leave blank to use global `HYBRID_*_WEIGHT` defaults. | *(see .env.example)* |
| `ENTITY_AWARE_ENABLED` | Enable entity extraction + Neo4j expansion + score boost for ENTITY_CENTRIC queries. Requires Neo4j + `GRAPHRAG_LLM`. | `true` |
| `ENTITY_BOOST_FACTOR` | Additive score boost per chunk per matching entity mention. | `0.1` |
| `TOOL_CALLING_ENABLED` | Enable the pre-answer agentic tool-calling loop. LLM may call `search_documents`, `extract_entities`, `summarize_chunks`. | `true` |
| `MAX_TOOL_ITERATIONS` | Maximum tool-call iterations per chat turn. | `5` |
| `SYNTHESIS_MODE_ENABLED` | For MULTI_PART queries with synthesis keywords (summarize, compare, themes…), use the synthesis prompt and `synthesize_documents` for parallel multi-query retrieval. | `true` |

### Chat File Upload

When a user attaches a file to a chat message, the file is processed ephemerally — it is **not** added to any knowledge base. Instead:

1. The file is saved to `uploads/ephemeral/{chat_id}/` (same Docker volume as KB uploads).
2. MarkItDown extracts the full text to Markdown.
3. The Markdown is stored in MySQL (`chat_files` table) alongside a `stored_path` to the original file on disk.
4. On the next message send, up to 25% of `OPENAI_MODEL_CONTEXT_SIZE` tokens of the file Markdown is injected directly into the QA system prompt as `## Uploaded File Context`.
5. Prior-turn files in the same chat are also re-injected for multi-turn context continuity.
6. The original file is kept on disk until the chat is deleted (calls `delete_ephemeral_chat_files`).

| Variable | Description | Default |
|---|---|---|
| `OPENAI_MODEL_CONTEXT_SIZE` | Total context window tokens — controls how much file content to inject (capped at 25%). | `131072` |

> Files attached to chat messages are **not** indexed in Qdrant or MySQL FTS. They exist only in the chat session and are served for download via the attachment chip in the UI.

### Vector DB (Qdrant)

| Variable | Description | Default |
|---|---|---|
| `QDRANT_HOST` | Qdrant service hostname | `qdrant` |
| `QDRANT_PORT` | Qdrant HTTP port | `6333` |
| `QDRANT_GRPC_PORT` | Qdrant gRPC port | `6334` |

### SPLADE Sparse Embedder

| Variable | Description | Default |
|---|---|---|
| `SPLADE_MODEL` | FastEmbed model name | `prithivida/Splade_PP_en_v1` |
| `FASTEMBED_CACHE_DIR` | Where to persist downloaded model | `./assets/fastembed` |

### Storage & Auth

| Variable | Description | Example |
|---|---|---|
| `MYSQL_SERVER` | MySQL host | `db` |
| `MYSQL_USER` | MySQL username | `ragwebui` |
| `MYSQL_PASSWORD` | MySQL password | `ragwebui` |
| `MYSQL_DATABASE` | MySQL database name | `ragwebui` |
| `SECRET_KEY` | JWT signing secret (auto-generated in dev if left as placeholder) | random string |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | JWT lifetime | `10080` (7 days) |

## Pre-downloading the SPLADE model

To avoid a ~500 MB download on first ingestion, pre-download the model into `./assets/fastembed/`:

```bash
pip install fastembed
python download_assets.py
```

The directory is bind-mounted into the container, so the model will be available immediately on next start.

## Admin & Developer Tools

The following web UIs are available when running locally:

| Tool | URL | Purpose |
|---|---|---|
| **RAG Web UI** | http://localhost:3000 | Main application |
| **Backend API Docs** | http://localhost:8000/redoc | OpenAPI reference |
| **Qdrant Dashboard** | http://localhost:6333/dashboard | Browse vector collections, inspect points, run queries |
| **Neo4j Browser** | http://localhost:7474/browser/ | Explore entity/relationship graph, run Cypher queries |
| **File Storage** | Local Folder mapped to Docker as Volume | Browse uploaded files |
| **Adminer** | http://localhost:8080 | MySQL web GUI (dev only — see below) |

### Adminer (MySQL web GUI)

Adminer is included in `docker-compose.dev.yml`. It's a lightweight single-container MySQL browser.

Start it:
```bash
docker compose -f docker-compose.dev.yml up -d adminer
```

Then open http://localhost:8080 and log in:

| Field | Value |
|---|---|
| System | MySQL |
| Server | `db` |
| Username | `ragwebui` |
| Password | `ragwebui` |
| Database | `ragwebui` |

### Qdrant Dashboard

Open http://localhost:6333/dashboard — no login required. From here you can:
- Browse collections and their vector counts
- Inspect individual points and their payloads
- Run search queries manually
- View collection configuration (vector dimensions, distance metric)

## Development

Hot reload for both frontend and backend:

```bash
docker compose -f docker-compose.dev.yml up -d --build
```

Services:

| Service | URL |
|---|---|
| Frontend (Next.js dev) | http://localhost:3000 |
| Backend (uvicorn reload) | http://localhost:8000 |
| API Docs | http://localhost:8000/redoc |
| Qdrant Dashboard | http://localhost:6333/dashboard |
| Neo4j Browser | http://localhost:7474/browser/ |
| Adminer (MySQL) | http://localhost:8080 |

**Stop without losing state:**
```bash
docker compose -f docker-compose.dev.yml stop
```

**Start again (no rebuild):**
```bash
docker compose -f docker-compose.dev.yml start
```

**Useful commands:**
```bash
# Restart a single service
docker compose -f docker-compose.dev.yml restart backend

# Check status
docker compose -f docker-compose.dev.yml ps

# Tail logs
docker compose -f docker-compose.dev.yml logs -f backend
docker compose -f docker-compose.dev.yml logs -f frontend

# Rebuild only when requirements.txt or Dockerfile changes
docker compose -f docker-compose.dev.yml up -d --build backend
```

## Features

- Upload PDF, DOCX, DOC, PPTX, PPT, XLSX, XLS, Markdown, plain text, HTML, CSV, JSON, XML, email (MSG/EML), EPUB, images (OCR), and ZIP archives (see [ingestion pipeline](docs/ingestion-pipeline.md) for full format list)
- Optional OCR for scanned PDFs and embedded images via `markitdown-ocr` — enabled by setting `VISION_MODEL`
- Separate model for query rewriting and summarisation via `QUERY_MODEL` (falls back to `OPENAI_MODEL`)
- Automatic chunking, embedding, and incremental updates
- **3-leg hybrid search**: dense vector + SPLADE sparse + MySQL full-text, combined by Reciprocal Rank Fusion (RRF)
- **GraphRAG** — optional entity/relationship extraction into Neo4j with graph-traversal retrieval expansion; LLM-based extraction via `GRAPHRAG_LLM`
- **Adaptive retrieval** — LLM query classifier (FACTUAL / ENTITY_CENTRIC / MULTI_PART / AMBIGUOUS) routes each query to a per-type retrieval preset (leg weights, top-k) via `RETRIEVAL_CONFIG_PRESETS`
- **Entity-aware retrieval** — for ENTITY_CENTRIC queries, entities are extracted from the query via `GRAPHRAG_LLM`, matched against the Neo4j entity graph with 1-hop expansion, and used to boost chunk scores that contain entity mentions
- **Cross-encoder reranking** — retrieved candidates re-ranked by a local cross-encoder model (`RERANKER_MODEL`) before context assembly
- **Retrieval confidence scoring** — every answer shows a 4-level confidence indicator (High/Medium/Low/None) based on score distribution, leg overlap, and entity signals; actionable suggestions when confidence is low
- **Agentic tool calling** — LLM can call `search_documents`, `extract_entities`, and `summarize_chunks` in a pre-answer loop (up to `MAX_TOOL_ITERATIONS`); tool trace visible per message
- **Multi-document synthesis** — for synthesis queries (summarize / compare / themes), the LLM uses `synthesize_documents` for parallel multi-query fan-out and produces a structured Markdown report with citations
- **Chat file upload** — attach any supported document to a chat message; content is injected into the LLM context (up to 25% of the model context window) without indexing; multi-turn re-injection keeps file context available across follow-up questions; attachment chip under user messages shows filename on hover and triggers download on click
- Multi-turn chat with source citations
- Streaming responses with think-block collapsing for reasoning models
- Query classification badge, tool trace timeline, and synthesis mode indicator in the chat UI
- Collapsible chat sidebar with localStorage persistence
- Dark / light / system theme toggle in both dashboard and chat layouts
- Retrieval quality testing UI
- Route protection: unauthenticated users are redirected to `/login`
- JWT invalidated on container restart (ephemeral secret key in dev)

## Troubleshooting

See [docs/troubleshooting.md](docs/troubleshooting.md).

## License

[Apache-2.0](LICENSE)

---

<div align="center">If this project helps you, please give it a ⭐️</div>
