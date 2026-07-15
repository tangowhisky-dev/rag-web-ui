# Moving .env Parameters to Admin & SuperAdmin UIs

## Executive Summary

This document identifies which environment variables from `.env` / `.env.example` can be migrated from file-based configuration to the admin (organisation-scoped) and superAdmin (app-wide) UIs, and proposes the data model, API, and UI design for each.

**Key principle:** Anything that changes the app's operational behaviour at runtime belongs in the admin UI. Anything that changes the app's infrastructure (docker-compose topology, DB credentials, network topology) stays in `.env` — it requires a restart or a container rebuild.

---

## Current State

The `.env` file currently has **~70 parameters**. These fall into three categories:

| Category | Description | Count | Example |
|----------|-------------|-------|---------|
| **Infrastructure** | Required for the app to start (service URLs, DB credentials, JWT secrets) | ~12 | `MYSQL_PASSWORD`, `SECRET_KEY`, `NEO4J_PASSWORD` |
| **Operational / Tuning** | Change how retrieval, chunking, reasoning, and agentic features behave | ~50 | `RETRIEVAL_TOP_K`, `CHUNK_SIZE`, `TOOL_CALLING_ENABLED` |
| **Initialisation** | One-time seeding values | ~3 | `ROOT_ORG`, `SUPERADMIN_USERNAME`, `SUPERADMIN_PASSWORD` |

**Already in UI:** The Orgs page (`/dashboard/admin/orgs`) has an **LLM Config** dialog that manages `api_base`, `model_name`, and `query_model` per organisation (backed by the `OrgLLMConfig` model).

---

## Decision Matrix

### SuperAdmin Scope (App-Wide)

These parameters apply to the entire application — no organisation context.

| # | .env Variable | Current Default | Admin UI Tab | Reason for SuperAdmin Scope |
|---|---------------|-----------------|--------------|-----------------------------|
| 1 | `SECRET_KEY` | `your-secret-key-here` | **System** | JWT signing key — app-wide |
| 2 | `ACCESS_TOKEN_EXPIRE_MINUTES` | `360` | **System** | Session lifetime — app-wide |
| 3 | `TIMEOUT_SECONDS` | `30000` | **System** | Frontend request timeout — app-wide |
| 4 | `PROCESSING_TIMEOUT_SILENCE_S` | `300` | **System** | Chat processing timeout — app-wide |
| 5 | `UPLOAD_DIR` | `/app/uploads` | **System** | File storage location — app-wide |
| 6 | `WATCHER_ENABLED` | `true` | **System** | Enable/disable file watcher service — app-wide |
| 7 | `WATCH_POLL_INTERVAL` | `2` | **System** | Watcher scan interval — app-wide |
| 8 | `WATCHER_USE_INOTIFY` | `true` | **System** | inotify vs polling — app-wide |
| 9 | `ANSWER_QUALITY_GRADING_ENABLED` | `true` | **System** | Enable/disable answer quality grading — app-wide |
| 10 | `TZ` | `UTC` | **System** | Application timezone — app-wide |

### SuperAdmin Scope — Infrastructure (MUST remain in .env)

These parameters control infrastructure and are **NOT** candidates for UI migration:

| Variable | Why it must stay in .env |
|----------|--------------------------|
| `MYSQL_SERVER`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE` | Database credentials and connection — required before the app starts. Changing mid-flight breaks existing DB sessions. |
| `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_GRPC_PORT` | Vector store connection — same reason as MySQL. |
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | Graph DB credentials — same. |
| `SECRET_KEY` | JWT secret — changing this invalidates ALL existing sessions. Must stay in .env. |
| `ROOT_ORG`, `SUPERADMIN_USERNAME`, `SUPERADMIN_PASSWORD` | Initial seed values — used once at first startup. |
| `OPENAI_API_KEY` | The API key itself should stay in .env for security. The base URL and model names can go in the UI. |
| `OPENAI_API_BASE` | Base URL — this is infrastructure. However, if the backend connects to multiple LLM endpoints (which is now the case with per-org configs), it could move to UI. |

**Verdict:** Infrastructure parameters stay in `.env`. The `SECRET_KEY` is a special case — it's security-sensitive and changing it would invalidate all sessions, so it should remain a .env-only setting. A future enhancement could add a secure "rotate secret key" endpoint with session invalidation.

### Admin Scope (Organisation-Specific)

These parameters are **per-organisation** — each org in a multi-tenant setup can have its own values.

#### A. LLM Configuration (Already partially implemented)

| # | .env Variable | Current Default | Org UI (LLM Config) | Notes |
|---|---------------|-----------------|---------------------|-------|
| 1 | `OPENAI_API_BASE` | (inherited) | ✅ `api_base` | Already in UI as `api_base` |
| 2 | `OPENAI_MODEL` | (inherited) | ✅ `model_name` | Already in UI as `model_name` |
| 3 | `QUERY_MODEL` | (inherited) | ✅ `query_model` | Already in UI as `query_model` |

**Gaps in existing OrgLLMConfig:** The model stores only `api_base`, `model_name`, and `query_model`. Several LLM-related .env vars are missing:

| Missing Field | .env Variable | Suggested DB Column | Reason |
|---------------|---------------|---------------------|--------|
| Reasoning model | `REASONING_MODEL` | `reasoning_model` | "Thinking" answering mode uses this per-org |
| Vision / OCR model | `VISION_MODEL` | `vision_model` | OCR of scanned PDFs / embedded images |
| Vision API base | `OPENAI_VISION_API_BASE` | `vision_api_base` | Optional separate base for the vision model |
| Embeddings model | `DENSE_EMBEDDINGS_MODEL` | `embeddings_model` | Per-org embedding model selection |
| Dense embedding dim | `DENSE_EMBEDDING_DIM` | `embedding_dim` | Must match the embeddings model |
| GraphRAG LLM | `GRAPHRAG_LLM` | `graphrag_model` | Entity/relationship extraction model |

#### B. Retrieval Configuration (per-organisation)

| # | .env Variable | Current Default | Suggested Admin UI | Notes |
|---|---------------|-----------------|--------------------|-------|
| 1 | `RETRIEVAL_TOP_K` | `20` | **Retrieval** tab | Max chunks returned per query |
| 2 | `RETRIEVAL_MIN_RRF_SCORE` | `0.005` | **Retrieval** tab | RRF score threshold |
| 3 | `RETRIEVAL_DENSE_ENABLED` | `true` | **Retrieval** tab | Toggle dense retrieval leg |
| 4 | `RETRIEVAL_QDRANT_SPARSE_ENABLED` | `true` | **Retrieval** tab | Toggle SPLADE sparse leg |
| 5 | `RETRIEVAL_EXACT_ENABLED` | `true` | **Retrieval** tab | Toggle MySQL FTS exact leg |
| 6 | `HYBRID_DENSE_WEIGHT` | `0.5` | **Retrieval** tab | Dense leg weight |
| 7 | `HYBRID_QDRANT_SPARSE_WEIGHT` | `0.3` | **Retrieval** tab | Sparse leg weight |
| 8 | `HYBRID_EXACT_WEIGHT` | `0.2` | **Retrieval** tab | Exact leg weight |
| 9 | `RETRIEVAL_GRAPH_ENABLED` | `true` | **Retrieval** tab | Toggle graph retrieval leg |
| 10 | `ENTITY_AWARE_ENABLED` | `true` | **Retrieval** tab | Entity-aware retrieval toggle |
| 11 | `ENTITY_BOOST_FACTOR` | `0.1` | **Retrieval** tab | Per-mention score boost |
| 12 | `RETRIEVAL_CONFIG_PRESETS` | (JSON) | **Retrieval** tab | Per-query-type config presets |

#### C. Chunking Configuration (per-organisation)

| # | .env Variable | Current Default | Suggested Admin UI | Notes |
|---|---------------|-----------------|--------------------|-------|
| 1 | `CHUNK_SIZE` | `1500` | **Chunking** tab | Target chunk size in characters |
| 2 | `OVERLAP_PERCENTAGE` | `0.20` | **Chunking** tab | Overlap fraction (0.0–1.0) |

> **Warning:** Chunking settings should emit a warning if changed after documents exist. Changing them requires re-uploading all documents to re-index.

#### D. Reranker Configuration (per-organisation)

| # | .env Variable | Current Default | Suggested Admin UI | Notes |
|---|---------------|-----------------|--------------------|-------|
| 1 | `RERANKER_ENABLED` | `true` | **Reranker** tab | Enable/disable cross-encoder reranking |
| 2 | `RERANKER_MODEL` | `Xenova/ms-marco-MiniLM-L-12-v2` | **Reranker** tab | Reranker model name |
| 3 | `RERANKER_SCORE_THRESHOLD` | `-5.0` | **Reranker** tab | Min cross-encoder logit |

#### E. GraphRAG Configuration (per-organisation)

| # | .env Variable | Current Default | Suggested Admin UI | Notes |
|---|---------------|-----------------|--------------------|-------|
| 1 | `GRAPHRAG_ENABLED` | `true` | **GraphRAG** tab | Enable/disable graph extraction |
| 2 | `GRAPHRAG_RETRIEVAL_HOPS` | `1` | **GraphRAG** tab | Relationship hops |
| 3 | `GRAPHRAG_MAX_CHUNKS` | `0` | **GraphRAG** tab | Max chunks per doc for graph extraction (0 = no limit) |
| 4 | `NEO4J_LLM_CONTEXT` | `24000` | **GraphRAG** tab | Context window budget for graph LLM |

#### F. Query Classification Configuration (per-organisation)

| # | .env Variable | Current Default | Suggested Admin UI | Notes |
|---|---------------|-----------------|--------------------|-------|
| 1 | `QUERY_CLASSIFIER_ENABLED` | `true` | **Adaptive Retrieval** tab | Enable/disable query classification |
| 2 | `QUERY_CLASSIFIER_PROMPT` | (long prompt) | **Adaptive Retrieval** tab | Classification prompt text |
| 3 | `ENTITY_AWARE_ENABLED` | `true` | **Adaptive Retrieval** tab | Also appears in Retrieval — deduplicate |

#### G. Agentic Features Configuration (per-organisation)

| # | .env Variable | Current Default | Suggested Admin UI | Notes |
|---|---------------|-----------------|--------------------|-------|
| 1 | `TOOL_CALLING_ENABLED` | `true` | **Agentic** tab | Enable/disable tool calling |
| 2 | `MAX_TOOL_ITERATIONS` | `5` | **Agentic** tab | Max iterations per chat turn |
| 3 | `SYNTHESIS_MODE_ENABLED` | `true` | **Agentic** tab | Enable/disable synthesis mode |

#### H. Historical Memory Configuration (per-organisation)

| # | .env Variable | Current Default | Suggested Admin UI | Notes |
|---|---------------|-----------------|--------------------|-------|
| 1 | `HISTORICAL_MEMORY_ENABLED` | `true` | **Memory** tab | Enable/disable historical memory retrieval |
| 2 | `HISTORICAL_MEMORY_TOP_K` | `5` | **Memory** tab | Historical docs to return |
| 3 | `HISTORICAL_MEMORY_SCORE_THRESHOLD` | `2.0` | **Memory** tab | Min reranker score |

#### I. Adaptive Retrieval Configuration (per-organisation)

| # | .env Variable | Current Default | Suggested Admin UI | Notes |
|---|---------------|-----------------|--------------------|-------|
| 1 | `ADAPTIVE_RETRIEVAL_ENABLED` | `true` | **Retrieval** tab | Enable/disable two-pass retrieval |
| 2 | `ADAPTIVE_RETRIEVAL_THRESHOLD` | `55` | **Retrieval** tab | Confidence threshold for expansion |
| 3 | `ADAPTIVE_RETRIEVAL_RERANKER_THRESHOLD` | `-5.0` | **Retrieval** tab | Second-pass reranker cutoff |

#### J. Sparse Embedding / SPLADE (per-organisation)

| # | .env Variable | Current Default | Suggested Admin UI | Notes |
|---|---------------|-----------------|--------------------|-------|
| 1 | `SPLADE_MODEL` | `prithivida/Splade_PP_en_v1` | **Chunking** tab | SPLADE sparse model |
| 2 | `FASTEMBED_CACHE_DIR` | `/app/assets/fastembed` | (infrastructure) | Cache directory — stays in .env |

---

## Summary Table

| Scope | Tab | Parameters | DB Table |
|-------|-----|------------|----------|
| **SuperAdmin** | System | `SECRET_KEY`, `ACCESS_TOKEN_EXPIRE_MINUTES`, `TIMEOUT_SECONDS`, `PROCESSING_TIMEOUT_SILENCE_S`, `UPLOAD_DIR`, `WATCHER_ENABLED`, `WATCH_POLL_INTERVAL`, `WATCHER_USE_INOTIFY`, `ANSWER_QUALITY_GRADING_ENABLED`, `TZ` | `AppSettings` (new) |
| **SuperAdmin** | — | Infrastructure (MySQL, Qdrant, Neo4j, API keys) | — Stay in .env |
| **SuperAdmin** | Users | N/A (users already managed) | `users` (existing) |
| **SuperAdmin** | Data Stores | N/A (already managed) | `data_stores` (existing) |
| **Admin (per-org)** | LLM Config | `api_base`, `model_name`, `query_model`, `reasoning_model`, `vision_model`, `vision_api_base`, `embeddings_model`, `embedding_dim`, `graphrag_model` | `org_llm_configs` (extend) |
| **Admin (per-org)** | Retrieval | `retrieval_top_k`, `retrieval_min_rrf_score`, `retrieval_dense_enabled`, `retrieval_sparse_enabled`, `retrieval_exact_enabled`, `retrieval_graph_enabled`, `hybrid_dense_weight`, `hybrid_sparse_weight`, `hybrid_exact_weight`, `entity_aware_enabled`, `entity_boost_factor`, `retrieval_config_presets`, `adaptive_retrieval_enabled`, `adaptive_retrieval_threshold`, `adaptive_retrieval_reranker_threshold` | `org_retrieval_configs` (new) |
| **Admin (per-org)** | Chunking | `chunk_size`, `overlap_percentage`, `splade_model` | `org_chunking_configs` (new) |
| **Admin (per-org)** | Reranker | `reranker_enabled`, `reranker_model`, `reranker_score_threshold` | `org_reranker_configs` (new) |
| **Admin (per-org)** | GraphRAG | `graphrag_enabled`, `graphrag_retrieval_hops`, `graphrag_max_chunks`, `neo4j_llm_context` | `org_graphrag_configs` (new) |
| **Admin (per-org)** | Agentic | `tool_calling_enabled`, `max_tool_iterations`, `synthesis_mode_enabled` | `org_agentic_configs` (new) |
| **Admin (per-org)** | Query Classification | `query_classifier_enabled`, `query_classifier_prompt` | `org_query_configs` (new) |
| **Admin (per-org)** | Memory | `historical_memory_enabled`, `historical_memory_top_k`, `historical_memory_score_threshold` | `org_memory_configs` (new) |
| **Admin (per-org)** | Abbreviations | `short`, `expansion` | `org_abbreviations` (existing) |

---

## Proposed Database Schema

### 1. Extend `org_llm_configs`

```sql
ALTER TABLE org_llm_configs
  ADD COLUMN reasoning_model VARCHAR(255) NULL,
  ADD COLUMN vision_model VARCHAR(255) NULL,
  ADD COLUMN vision_api_base VARCHAR(512) NULL,
  ADD COLUMN embeddings_model VARCHAR(255) NULL,
  ADD COLUMN embedding_dim INT NULL,
  ADD COLUMN graphrag_model VARCHAR(255) NULL;
```

### 2. New tables (one per config domain)

All follow the same pattern: one row per org, nullable fields fall back to .env defaults.

```sql
CREATE TABLE org_retrieval_configs (
  org_id INT PRIMARY KEY,
  retrieval_top_k INT NULL DEFAULT 20,
  retrieval_min_rrf_score FLOAT NULL DEFAULT 0.005,
  retrieval_dense_enabled BOOLEAN NULL DEFAULT TRUE,
  retrieval_qdrant_sparse_enabled BOOLEAN NULL DEFAULT TRUE,
  retrieval_exact_enabled BOOLEAN NULL DEFAULT TRUE,
  retrieval_graph_enabled BOOLEAN NULL DEFAULT TRUE,
  hybrid_dense_weight FLOAT NULL DEFAULT 0.5,
  hybrid_qdrant_sparse_weight FLOAT NULL DEFAULT 0.3,
  hybrid_exact_weight FLOAT NULL DEFAULT 0.2,
  entity_aware_enabled BOOLEAN NULL DEFAULT TRUE,
  entity_boost_factor FLOAT NULL DEFAULT 0.1,
  retrieval_config_presets TEXT NULL,
  adaptive_retrieval_enabled BOOLEAN NULL DEFAULT TRUE,
  adaptive_retrieval_threshold FLOAT NULL DEFAULT 55.0,
  adaptive_retrieval_reranker_threshold FLOAT NULL DEFAULT -5.0,
  FOREIGN KEY (org_id) REFERENCES organisations(id) ON DELETE CASCADE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE org_chunking_configs (
  org_id INT PRIMARY KEY,
  chunk_size INT NULL DEFAULT 1500,
  overlap_percentage FLOAT NULL DEFAULT 0.2,
  splade_model VARCHAR(255) NULL DEFAULT 'prithivida/Splade_PP_en_v1',
  FOREIGN KEY (org_id) REFERENCES organisations(id) ON DELETE CASCADE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE org_reranker_configs (
  org_id INT PRIMARY KEY,
  reranker_enabled BOOLEAN NULL DEFAULT TRUE,
  reranker_model VARCHAR(255) NULL DEFAULT 'Xenova/ms-marco-MiniLM-L-12-v2',
  reranker_score_threshold FLOAT NULL DEFAULT -5.0,
  FOREIGN KEY (org_id) REFERENCES organisations(id) ON DELETE CASCADE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE org_graphrag_configs (
  org_id INT PRIMARY KEY,
  graphrag_enabled BOOLEAN NULL DEFAULT TRUE,
  graphrag_retrieval_hops INT NULL DEFAULT 2,
  graphrag_max_chunks INT NULL DEFAULT 0,
  neo4j_llm_context INT NULL DEFAULT 24000,
  FOREIGN KEY (org_id) REFERENCES organisations(id) ON DELETE CASCADE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE org_agentic_configs (
  org_id INT PRIMARY KEY,
  tool_calling_enabled BOOLEAN NULL DEFAULT TRUE,
  max_tool_iterations INT NULL DEFAULT 5,
  synthesis_mode_enabled BOOLEAN NULL DEFAULT TRUE,
  FOREIGN KEY (org_id) REFERENCES organisations(id) ON DELETE CASCADE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE org_query_configs (
  org_id INT PRIMARY KEY,
  query_classifier_enabled BOOLEAN NULL DEFAULT TRUE,
  query_classifier_prompt TEXT NULL,
  FOREIGN KEY (org_id) REFERENCES organisations(id) ON DELETE CASCADE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE org_memory_configs (
  org_id INT PRIMARY KEY,
  historical_memory_enabled BOOLEAN NULL DEFAULT TRUE,
  historical_memory_top_k INT NULL DEFAULT 5,
  historical_memory_score_threshold FLOAT NULL DEFAULT 2.0,
  FOREIGN KEY (org_id) REFERENCES organisations(id) ON DELETE CASCADE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE app_settings (
  id INT PRIMARY KEY AUTO_INCREMENT,
  secret_key VARCHAR(512) NULL,
  access_token_expire_minutes INT NULL DEFAULT 360,
  timeout_seconds INT NULL DEFAULT 30000,
  processing_timeout_silence_s INT NULL DEFAULT 300,
  upload_dir VARCHAR(1024) NULL DEFAULT '/app/uploads',
  watcher_enabled BOOLEAN NULL DEFAULT TRUE,
  watch_poll_interval INT NULL DEFAULT 2,
  watcher_use_inotify BOOLEAN NULL DEFAULT TRUE,
  answer_quality_grading_enabled BOOLEAN NULL DEFAULT TRUE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

---

## Proposed API Endpoints

### SuperAdmin — App Settings

```
GET    /api/admin/app-settings            → Current app settings (with masked secret_key)
PUT    /api/admin/app-settings            → Upsert app settings
```

> The `secret_key` field should be masked in responses (show `***-xxxx`) and only change when explicitly sent in the request body.

### Admin — Per-Org Configs (same CRUD for each domain)

Each config domain follows the same pattern:

```
GET    /api/admin/orgs/{org_id}/configs/{domain}    → Current config (null = inherit .env)
PUT    /api/admin/orgs/{org_id}/configs/{domain}    → Upsert config
DELETE /api/admin/orgs/{org_id}/configs/{domain}    → Clear all (reset to .env defaults)
```

Where `{domain}` is one of: `retrieval`, `chunking`, `reranker`, `graphrag`, `agentic`, `query`, `memory`.

For LLM config (already exists, extend with new endpoints):
```
GET    /api/admin/orgs/{org_id}/llm-config          → Existing, extend response
PUT    /api/admin/orgs/{org_id}/llm-config          → Existing, extend body
```

---

## UI Navigation

### SuperAdmin Sidebar (current + new)

| Current Items | New Items |
|---------------|-----------|
| Orgs | **Settings** (System settings tab) |
| Users | |
| Data Stores | |

The **Settings** page has tabs:
- **System** — all app-wide operational settings (watcher, timeouts, storage, quality grading)
- **Users** — already exists
- **Orgs** — already exists
- **Data Stores** — already exists

### Admin Sidebar (per-org — shown when an org is selected)

The sidebar would be context-aware. When viewing an organisation (from the Orgs page), secondary nav expands:

| Section | Tabs |
|---------|------|
| **Organisation** | Name, parent |
| **LLM Config** | API base, models (already exists as dialog) |
| **Retrieval** | Weights, top-k, legs, entity-aware, adaptive retrieval |
| **Chunking** | Chunk size, overlap, SPLADE model |
| **Reranker** | Enabled, model, threshold |
| **GraphRAG** | Enabled, hops, max chunks, context |
| **Agentic** | Tool calling, synthesis, max iterations |
| **Query** | Classifier enabled, prompt |
| **Memory** | Historical memory settings |
| **Abbreviations** | (already exists as dialog on Orgs page) |

---

## Migration Strategy

### Phase 1: Database + Backend (no UI changes)

1. Add new tables and extend `OrgLLMConfig`.
2. Add Pydantic schemas for all new config domains.
3. Add API endpoints (GET/PUT/DELETE for each domain).
4. Create a new `AppSettings` API endpoint.

### Phase 2: Config resolution layer

Modify the `Settings` class (or create a `PerOrgSettings` class) that:
1. Reads the per-org config from the database.
2. Falls back to `.env` defaults for unset/null fields.
3. Falls back to hardcoded defaults in `config.py`.

**Key design decision:** The config resolution should happen **per-request**, not at startup. This ensures changes take effect immediately without restarts.

### Phase 3: UI — SuperAdmin Settings

Build the Settings page with tabs for system-wide configuration.

### Phase 4: UI — Per-Org Tabs

Expand the Orgs page to include collapsible config sections per domain (similar to the existing LLM Config dialog, but with more fields and better visual grouping).

---

## What Stays in .env (Never)

| Category | Parameters | Why |
|----------|-----------|-----|
| **Database credentials** | `MYSQL_*` | Required at startup; changes break DB sessions |
| **Vector store** | `QDRANT_*` | Same |
| **Graph DB** | `NEO4J_*` | Same |
| **JWT secret** | `SECRET_KEY` | Changing invalidates all sessions |
| **API keys** | `OPENAI_API_KEY` | Security-sensitive; should be injected, not browsed |
| **Initial seed** | `ROOT_ORG`, `SUPERADMIN_USERNAME`, `SUPERADMIN_PASSWORD` | One-time use at first startup |
| **Cache dirs** | `FASTEMBED_CACHE_DIR`, `RERANKER_CACHE_DIR` | Infrastructure paths; mount points |
| **Docker compose profiles** | `COMPOSE_PROFILES` | Docker-level config |

**Exception — `OPENAI_API_BASE`:** This is a borderline case. The existing per-org `api_base` already allows org-specific API bases. The global default from `.env` remains a fallback. Consider whether the global default should also be configurable via the SuperAdmin Settings page, so different environments (dev, staging, prod) can have different defaults.

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Config changes affect in-flight queries** | Retrieval tuning mid-query may produce inconsistent results | Acceptable — retrieval config is inherently runtime; chunking changes should warn about needing re-ingestion |
| **Admin UI changes bypass validation** | Invalid config values could break queries | Server-side validation in API; range checks for numeric fields |
| **Multiple admin users editing simultaneously** | Lost updates | Use optimistic locking or last-write-wins with timestamps |
| **Config drift between .env and UI** | Confusion about what's actually active | UI should show "inherited from .env" for null fields; add a `/api/admin/config/snapshot` endpoint that shows the effective config |
| **Massive config table** | DB performance | One row per org is fine (< 1000 orgs); add indexes on `org_id` |

---

## Not Migrated (Too Risky / Not Applicable)

| .env Variable | Reason Not Migrated |
|---------------|---------------------|
| `MYSQL_*` | Infrastructure — restart required |
| `QDRANT_*` | Infrastructure — restart required |
| `NEO4J_URI` | Infrastructure — restart required |
| `SECRET_KEY` | Changing invalidates all sessions; security-sensitive |
| `OPENAI_API_KEY` | Never expose API keys in UI; use secure env injection |
| `UPLOAD_DIR` | Storage path; could be added to System Settings but changes require restart |
| `FASTEMBED_CACHE_DIR` | Infrastructure mount point |
| `RERANKER_CACHE_DIR` | Infrastructure mount point |
| `COMPOSE_PROFILES` | Docker-level config |
| `TZ` | Could be added to System Settings but changes require restart; low value |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Could be added but low value — session lifetime rarely changes |

---

## File Changes Required

### Backend (Python)

| File | Change |
|------|--------|
| `backend/app/models/org_llm_config.py` | Add new columns |
| `backend/app/models/` | Add new model files: `org_retrieval_config.py`, `org_chunking_config.py`, `org_reranker_config.py`, `org_graphrag_config.py`, `org_agentic_config.py`, `org_query_config.py`, `org_memory_config.py`, `app_setting.py` |
| `backend/app/schemas/organisation.py` | Add new schemas for all config domains |
| `backend/app/api/api_v1/admin.py` | Add config CRUD endpoints |
| `backend/app/api/api_v1/datastores.py` | Add `AppSettings` endpoints (or create `settings.py`) |
| `backend/app/core/config.py` | Add `PerOrgSettings` resolver class |
| `backend/app/main.py` | Wire per-org config resolution into the request pipeline |
| `backend/alembic/versions/` | Add migration scripts |

### Frontend (TypeScript/React)

| File | Change |
|------|--------|
| `frontend/src/app/dashboard/admin/page.tsx` | Add "Settings" stat card or tab |
| `frontend/src/app/dashboard/admin/settings/page.tsx` | **New** — SuperAdmin system settings page |
| `frontend/src/components/admin/admin-sidebar.tsx` | Add "Settings" nav item |
| `frontend/src/app/dashboard/admin/orgs/page.tsx` | Expand LLM Config dialog; add per-domain config dialogs/tabs |
| `frontend/src/lib/api.ts` | Add API call functions |

### Database

| File | Change |
|------|--------|
| `backend/alembic/versions/XXXX_extend_org_llm_configs.py` | Add new columns to `org_llm_configs` |
| `backend/alembic/versions/XXXX_create_org_config_tables.py` | Create all new config tables |
| `backend/alembic/versions/XXXX_create_app_settings.py` | Create `app_settings` table |
