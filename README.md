<div align="center">
  <img src="./docs/images/github-cover-new.png" alt="RAG Web UI">
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

## Screenshots

<div align="center">
  <img src="./docs/images/screenshot1.png" alt="Knowledge Base Management" width="800">
  <p><em>Knowledge Base Management</em></p>

  <img src="./docs/images/screenshot2.png" alt="Chat Interface" width="800">
  <p><em>Document Processing</em></p>

  <img src="./docs/images/screenshot3.png" alt="Document List" width="800">
  <p><em>Document List</em></p>

  <img src="./docs/images/screenshot4.png" alt="Chat" width="800">
  <p><em>Chat with References</em></p>

  <img src="./docs/images/screenshot5.png" alt="API Keys" width="800">
  <p><em>API Key Management</em></p>

  <img src="./docs/images/screenshot6.png" alt="API Reference" width="800">
  <p><em>OpenAPI Reference</em></p>
</div>

## Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, TypeScript, Tailwind CSS, shadcn/ui, Vercel AI SDK |
| Backend | Python FastAPI, LangChain, SQLAlchemy |
| Vector DB | ChromaDB |
| File Storage | MinIO |
| Database | MySQL 8 |

## Quick Start

**Prerequisites:** Docker & Docker Compose v2+

```bash
git clone https://github.com/rag-web-ui/rag-web-ui.git
cd rag-web-ui
cp .env.example .env
# Edit .env with your API key and model settings
docker compose up -d --build
```

Open **http://localhost:3000** — register an account and start uploading documents.

- MinIO Console: http://localhost:9001

## Configuration

Copy `.env.example` to `.env` and set these values:

| Variable | Description | Example |
|---|---|---|
| `OPENAI_API_KEY` | API key for your LLM provider | `sk-...` or `lmstudio` |
| `OPENAI_API_BASE` | Base URL of OpenAI-compatible API | `https://api.openai.com/v1` |
| `OPENAI_MODEL` | Chat model name | `gpt-4o` |
| `OPENAI_EMBEDDINGS_MODEL` | Embedding model name | `text-embedding-3-small` |
| `CHROMA_DB_HOST` | ChromaDB host (Docker service name) | `chromadb` |
| `CHROMA_DB_PORT` | ChromaDB port | `8000` |
| `MINIO_ENDPOINT` | MinIO endpoint | `minio:9000` |
| `MYSQL_SERVER` | MySQL host | `db` |
| `SECRET_KEY` | JWT signing secret | random string |

**Using a local model server (e.g. LM Studio):**
```env
OPENAI_API_KEY=lmstudio
OPENAI_API_BASE=http://host.docker.internal:1234/v1
OPENAI_MODEL=your-model-name
OPENAI_EMBEDDINGS_MODEL=your-embedding-model-name
```

## Development

Hot reload for both frontend and backend:

```bash
docker compose -f docker-compose.dev.yml up -d --build
```

- Frontend: http://localhost:3000 (Next.js dev server)
- Backend: http://localhost:8000 (uvicorn with reload)
- API Docs: http://localhost:8000/redoc

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
```

**Rebuild only when** `requirements.txt` or `Dockerfile` changes:
```bash
docker compose -f docker-compose.dev.yml up -d --build backend
```

## Features

- Upload PDF, DOCX, Markdown, and plain text files
- Automatic chunking, embedding, and incremental updates
- Multi-turn chat with source citations
- Streaming responses with think-block collapsing for reasoning models
- API key management for programmatic access (OpenAPI at `/api/openapi`)
- Retrieval quality testing UI

## Troubleshooting

See [docs/troubleshooting.md](docs/troubleshooting.md).

## License

[Apache-2.0](LICENSE)

---

<div align="center">If this project helps you, please give it a ⭐️</div>
