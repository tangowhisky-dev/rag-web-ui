# Migrating `.env` Parameters to Admin & Super Admin UIs

## Overview

This report analyses every parameter currently loaded from `.env` via `backend/app/core/config.py` (the `Settings` class). It classifies each parameter by:

- **Should stay in `.env`** — infrastructure secrets, external service credentials that don't make sense in a UI, or values that need to be hot-reloadable from the OS.
- **App-wide settings** — suitable for **Super Admin** UI (affects all organisations).
- **Org-specific settings** — suitable for **Admin** UI (affects a single organisation).

The project already has a partial precedent: `OrgLLMConfig` (stored in the `org_llm_configs` DB table) lets admins override `api_base`, `model_name`, and `query_model` per organisation. This pattern should be extended.

---

## Current `.env` Parameters (from `config.py`)

| # | Parameter | Current Type | Default | Category |
|---|-----------|-------------|---------|----------|
| 1 | `PROJECT_NAME` | str | `"InsightCore"` | App-wide (UI) |
| 2 | `VERSION` | str | `"0.1.0"` | App-wide (read-only info) |
| 3 | `API_V1_STR` | str | `"/api"` | Infrastructure |
| 4 | `MYSQL_SERVER` | str | `"localhost"` | Infrastructure |
| 5 | `MYSQL_PORT` | int | `3306` | Infrastructure |
| 6 | `MYSQL_USER` | str | `"ragwebui"` | Infrastructure |
| 7 | `MYSQL_PASSWORD` | str | `"ragwebui"` | Infrastructure |
| 8 | `MYSQL_DATABASE` | str | `"ragwebui"` | Infrastructure |
| 9 | `SQLALCHEMY_DATABASE_URI` | str | `None` | Infrastructure |
| 10 | `SECRET_KEY` | str | `"your-secret-key-here"` | Infrastructure (secret) |
| 11 | `ALGORITHM` | str | `"HS256"` | Infrastructure |
| 12 | `ACCESS_TOKEN_EXPIRE_MINUTES` | int | `360` | App-wide (UI) |
| 13 | `UPLOAD_DIR` | str | `"/app/uploads"` | Infrastructure |
| 14 | `WATCH_DIR` | str | `"/app/uploads"` | Infrastructure |
| 15 | `WATCH_POLL_INTERVAL` | int | `2` | App-wide (UI) |
| 16 | `WATCHER_ENABLED` | bool | `true` | App-wide (UI) |
| 17 | `WATCHER_USE_INOTIFY` | bool | `true` | App-wide (UI) |
| 18 | `OPENAI_API_BASE` | str | `"http://localhost:1234/v1"` | App-wide (UI) or Org (already DB-backed) |
| 19 | `OPENAI_API_KEY` | str | `"lmstudio"` | Infrastructure (secret) |
| 20 | `OPENAI_MODEL` | str | `"local-model"` | App-wide (UI) or Org (already DB-backed) |
| 21 | `OPENAI_MODEL_CONTEXT_SIZE` | int | `131072` | App-wide (UI) |
| 22 | `QUERY_MODEL` | str | `None` | App-wide (UI) or Org (already DB-backed) |
| 23 | `VISION_MODEL` | str | `None` | App-wide (UI) |
| 24 | `OPENAI_VISION_API_BASE` | str | `None` | App-wide (UI) |
| 25 | `REASONING_MODEL` | str | `None` | App-wide (UI) |
| 26 | `DENSE_EMBEDDINGS_MODEL` | str | `"local-embedding-model"` | App-wide (UI) |
| 27 | `DENSE_EMBEDDING_DIM` | int | `1024` | App-wide (UI) |
| 28 | `RETRIEVAL_CONFIG_PRESETS` | JSON string | preset JSON | App-wide (UI) |
| 29 | `SPLADE_MODEL` | str | `"prithivida/Splade_PP_en_v1"` | App-wide (UI) |
| 30 | `FASTEMBED_CACHE_DIR` | str | `"/app/assets/fastembed"` | Infrastructure |
| 31 | `RETRIEVAL_TOP_K` | int | `20` | App-wide (UI) |
| 32 | `RETRIEVAL_MIN_RRF_SCORE` | float | `0.005` | App-wide (UI) |
| 33 | `RERANKER_ENABLED` | bool | `true` | App-wide (UI) |
| 34 | `RERANKER_MODEL` | str | `"Xenova/ms-marco-MiniLM-L-12-v2"` | App-wide (UI) |
| 35 | `RERANKER_CACHE_DIR` | str | `"/app/assets/reranker"` | Infrastructure |
| 36 | `RERANKER_SCORE_THRESHOLD` | float | `-2.0` | App-wide (UI) |
| 37 | `ADAPTIVE_RETRIEVAL_ENABLED` | bool | `true` | App-wide (UI) |
| 38 | `ADAPTIVE_RETRIEVAL_THRESHOLD` | float | `55` | App-wide (UI) |
| 39 | `ADAPTIVE_RETRIEVAL_RERANKER_THRESHOLD` | float | `-5.0` | App-wide (UI) |
| 40 | `HISTORICAL_MEMORY_ENABLED` | bool | `true` | App-wide (UI) |
| 41 | `HISTORICAL_MEMORY_TOP_K` | int | `5` | App-wide (UI) |
| 42 | `HISTORICAL_MEMORY_SCORE_THRESHOLD` | float | `2.0` | App-wide (UI) |
| 43 | `CHUNK_SIZE` | int | `1500` | App-wide (UI) |
| 44 | `OVERLAP_PERCENTAGE` | float | `0.20` | App-wide (UI) |
| 45 | `HYBRID_DENSE_WEIGHT` | float | `0.5` | App-wide (UI) |
| 46 | `HYBRID_QDRANT_SPARSE_WEIGHT` | float | `0.3` | App-wide (UI) |
| 47 | `HYBRID_EXACT_WEIGHT` | float | `0.2` | App-wide (UI) |
| 48 | `RETRIEVAL_DENSE_ENABLED` | bool | `true` | App-wide (UI) |
| 49 | `RETRIEVAL_QDRANT_SPARSE_ENABLED` | bool | `true` | App-wide (UI) |
| 50 | `RETRIEVAL_EXACT_ENABLED` | bool | `true` | App-wide (UI) |
| 51 | `NEO4J_URI` | str | `"bolt://neo4j:7687"` | Infrastructure |
| 52 | `NEO4J_USER` | str | `"neo4j"` | Infrastructure |
| 53 | `NEO4J_PASSWORD` | str | `"ragwebui_neo4j"` | Infrastructure |
| 54 | `GRAPHRAG_ENABLED` | bool | `true` | App-wide (UI) |
| 55 | `GRAPHRAG_LLM` | str | `None` | App-wide (UI) |
| 56 | `RETRIEVAL_GRAPH_ENABLED` | bool | `true` | App-wide (UI) |
| 57 | `GRAPHRAG_RETRIEVAL_HOPS` | int | `2` | App-wide (UI) |
| 58 | `GRAPHRAG_MAX_CHUNKS` | int | `0` | App-wide (UI) |
| 59 | `NEO4J_LLM_CONTEXT` | int | `12000` | App-wide (UI) |
| 60 | `ENTITY_AWARE_ENABLED` | bool | `true` | App-wide (UI) |
| 61 | `ENTITY_BOOST_FACTOR` | float | `0.1` | App-wide (UI) |
| 62 | `TOOL_CALLING_ENABLED` | bool | `true` | App-wide (UI) |
| 63 | `MAX_TOOL_ITERATIONS` | int | `5` | App-wide (UI) |
| 64 | `SYNTHESIS_MODE_ENABLED` | bool | `true` | App-wide (UI) |
| 65 | `PROCESSING_TIMEOUT_SILENCE_S` | int | `300` | App-wide (UI) |
| 66 | `ANSWER_QUALITY_GRADING_ENABLED` | bool | `true` | App-wide (UI) |
| 67 | `QDRANT_HOST` | str | `"qdrant"` | Infrastructure |
| 68 | `QDRANT_PORT` | int | `6333` | Infrastructure |
| 69 | `QDRANT_GRPC_PORT` | int | `6334` | Infrastructure |

---

## Classification Summary

### A. Stay in `.env` — Infrastructure & Secrets (do NOT expose in UI)

These are low-level infrastructure parameters that either: (1) are secrets, (2) are needed before the app starts to connect to the database, or (3) are file-system paths that Docker volumes control.

| Parameter | Reason |
|-----------|--------|
| `MYSQL_SERVER`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`, `SQLALCHEMY_DATABASE_URI` | Database credentials — needed at import time for Alembic migrations, `conftest.py`, and any code that imports `config.py`. Moving to DB-stored settings is circular (can't connect to the DB to read settings stored in the DB). |
| `SECRET_KEY` | JWT signing secret — must not be changeable at runtime (all existing tokens would become invalid). |
| `ALGORITHM` | JWT algorithm — rarely needs changing; tied to `SECRET_KEY`. |
| `UPLOAD_DIR` | File-system path — controlled by Docker volume mounts. |
| `FASTEMBED_CACHE_DIR` | File-system path — same as above. |
| `RERANKER_CACHE_DIR` | File-system path — same as above. |
| `OPENAI_API_KEY` | Secret credential — sensitive key that should not be stored in the database in plaintext (risk of data exfiltration, backup exposure). |
| `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_GRPC_PORT` | Vector store connection — needed before DB connection is fully up. |
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | Graph DB credentials — same as MySQL. |
| `API_V1_STR` | API routing string — rarely changed. |
| `VERSION` | App version — read from package metadata, not a runtime config. |
| `WATCH_DIR` | File-system path — controlled by Docker. |

### B. Suitable for Super Admin UI (app-wide, all organisations)

These settings affect the entire application and should be managed by Super Admins. They are **not secrets** and **do not need to exist before DB connection**. They can be stored in a new `app_settings` table.

| Group | Parameters | Notes |
|-------|-----------|-------|
| **LLM Chat Model** | `OPENAI_API_BASE`, `OPENAI_MODEL`, `OPENAI_MODEL_CONTEXT_SIZE` | Base model & endpoint for all orgs (orgs can override via existing `OrgLLMConfig`). |
| **Query/Rewrite Model** | `QUERY_MODEL` | Already in `OrgLLMConfig`; needs app-wide default. |
| **Reasoning Model** | `REASONING_MODEL` | Used in "Thinking" answering mode. |
| **Vision/OCR Model** | `VISION_MODEL`, `OPENAI_VISION_API_BASE` | Multimodal model for OCR. |
| **Embedding Models** | `DENSE_EMBEDDINGS_MODEL`, `DENSE_EMBEDDING_DIM`, `SPLADE_MODEL` | Models used for document ingestion and retrieval. Changing these requires re-indexing. |
| **Authentication** | `ACCESS_TOKEN_EXPIRE_MINUTES` | JWT token lifetime. |
| **Project Metadata** | `PROJECT_NAME` | Display name shown in UI header, email templates, etc. |

### C. Suitable for Super Admin UI (retrieval pipeline toggles & tuning)

These control the RAG pipeline behaviour globally. They are all boolean or numeric tunables that admins may want to adjust without restarting the server.

| Group | Parameters |
|-------|-----------|
| **Hybrid Weights** | `HYBRID_DENSE_WEIGHT`, `HYBRID_QDRANT_SPARSE_WEIGHT`, `HYBRID_EXACT_WEIGHT` |
| **Retrieval Legs** | `RETRIEVAL_DENSE_ENABLED`, `RETRIEVAL_QDRANT_SPARSE_ENABLED`, `RETRIEVAL_EXACT_ENABLED` |
| **Retrieval Tuning** | `RETRIEVAL_TOP_K`, `RETRIEVAL_MIN_RRF_SCORE` |
| **Reranker** | `RERANKER_ENABLED`, `RERANKER_MODEL`, `RERANKER_SCORE_THRESHOLD` |
| **Adaptive Retrieval** | `ADAPTIVE_RETRIEVAL_ENABLED`, `ADAPTIVE_RETRIEVAL_THRESHOLD`, `ADAPTIVE_RETRIEVAL_RERANKER_THRESHOLD` |
| **Chunking** | `CHUNK_SIZE`, `OVERLAP_PERCENTAGE` |
| **Query Classifier** | `QUERY_CLASSIFIER_ENABLED`, `QUERY_CLASSIFIER_PROMPT`, `RETRIEVAL_CONFIG_PRESETS` |
| **Historical Memory** | `HISTORICAL_MEMORY_ENABLED`, `HISTORICAL_MEMORY_TOP_K`, `HISTORICAL_MEMORY_SCORE_THRESHOLD` |
| **GraphRAG** | `GRAPHRAG_ENABLED`, `GRAPHRAG_LLM`, `RETRIEVAL_GRAPH_ENABLED`, `GRAPHRAG_RETRIEVAL_HOPS`, `GRAPHRAG_MAX_CHUNKS`, `NEO4J_LLM_CONTEXT` |
| **Entity-Aware Retrieval** | `ENTITY_AWARE_ENABLED`, `ENTITY_BOOST_FACTOR` |
| **Tool Calling** | `TOOL_CALLING_ENABLED`, `MAX_TOOL_ITERATIONS` |
| **Synthesis** | `SYNTHESIS_MODE_ENABLED` |
| **Answer Quality** | `ANSWER_QUALITY_GRADING_ENABLED` |
| **Processing** | `PROCESSING_TIMEOUT_SILENCE_S` |
| **Watcher** | `WATCHER_ENABLED`, `WATCH_POLL_INTERVAL`, `WATCHER_USE_INOTIFY` |

### D. Suitable for Admin UI (org-specific)

These already exist or can extend the existing `OrgLLMConfig` table.

| Existing | Parameters | Notes |
|----------|-----------|-------|
| `OrgLLMConfig` | `api_base`, `model_name`, `query_model` | Already DB-backed (stored per org). |

**Recommended: Extend `OrgLLMConfig` or create new org-level tables for:**

| Group | Parameters | Notes |
|-------|-----------|-------|
| **Per-org Retrieval Legs** | `RETRIEVAL_DENSE_ENABLED`, `RETRIEVAL_QDRANT_SPARSE_ENABLED`, `RETRIEVAL_EXACT_ENABLED` | Some orgs may want to disable specific legs. |
| **Per-org Weights** | `HYBRID_DENSE_WEIGHT`, `HYBRID_QDRANT_SPARSE_WEIGHT`, `HYBRID_EXACT_WEIGHT` | Org-specific tuning. |
| **Per-org Reranker** | `RERANKER_ENABLED`, `RERANKER_SCORE_THRESHOLD` | Enable/disable reranking per org. |
| **Per-org GraphRAG** | `RETRIEVAL_GRAPH_ENABLED`, `GRAPHRAG_RETRIEVAL_HOPS` | Some orgs may not need graph retrieval. |
| **Per-org Chunking** | `CHUNK_SIZE`, `OVERLAP_PERCENTAGE` | Different orgs may have different document types. |
| **Per-org Adaptive Retrieval** | `ADAPTIVE_RETRIEVAL_ENABLED`, `ADAPTIVE_RETRIEVAL_THRESHOLD` | Org-specific confidence thresholds. |
| **Per-org Features** | `TOOL_CALLING_ENABLED`, `SYNTHESIS_MODE_ENABLED`, `ANSWER_QUALITY_GRADING_ENABLED`, `HISTORICAL_MEMORY_ENABLED` | Feature flags per org. |

---

## Current Admin UI State

The frontend already has an admin dashboard at `/dashboard/admin/` with three pages:
- `/dashboard/admin/orgs` — Organisation CRUD
- `/dashboard/admin/users` — User CRUD
- `/dashboard/admin/data-sources` — DataStore management

The `OrgLLMConfig` is managed from the Orgs page via an "LLM Config" button (as noted in `admin-sidebar.tsx`).

No settings page currently exists for app-wide configuration.

---

## Proposed Data Model

### 1. App-wide settings table: `app_settings`

Stores Super Admin-managed settings. Uses a key-value structure for flexibility.

```python
class AppSetting(Base):
    __tablename__ = "app_settings"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(128), unique=True, nullable=False, index=True)   # e.g. "OPENAI_MODEL"
    value_type = Column(String(32), nullable=False)                      # "string", "int", "float", "bool", "json"
    value = Column(Text, nullable=True)                                  # serialized value
    description = Column(Text, nullable=True)                            # UI tooltip / label
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
```

This allows storing any parameter type without migrating the schema. Boolean values are stored as `"true"`/`"false"`. Integers and floats are stored as string representations. JSON values are stored as JSON strings.

### 2. Org-specific settings table: `org_settings`

Stores Admin-managed, organisation-specific settings.

```python
class OrgSetting(Base):
    __tablename__ = "org_settings"

    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(Integer, ForeignKey("organisations.id"), nullable=False, index=True)
    key = Column(String(128), nullable=False, index=True)
    value_type = Column(String(32), nullable=False)
    value = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("org_id", "key", name="uq_org_settings_org_key"),
    )
```

### 3. Default values

A new `AppDefaultSetting` model holds the factory defaults (mirroring current `.env` defaults). This is seed data only:

```python
class AppDefaultSetting(Base):
    """Immutable defaults synced from config.py — used for UI forms."""
    __tablename__ = "app_default_settings"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(128), unique=True, nullable=False)
    default_value = Column(Text, nullable=True)
    value_type = Column(String(32), nullable=False)
    is_secret = Column(Boolean, default=False)         # if True, hide value in UI
    is_editable = Column(Boolean, default=True)        # if False, read-only in UI
    group = Column(String(64), nullable=True)          # UI grouping
    order = Column(Integer, default=0)
```

---

## Proposed UI Structure

### Super Admin UI (app-wide settings)

```
Settings
├── General
│   ├── Project Name
│   └── JWT Token Expiry (minutes)
├── LLM Configuration
│   ├── Chat Model (OPENAI_MODEL)
│   ├── Chat Model Context Size (tokens)
│   ├── API Base URL
│   ├── Query/Rewrite Model
│   ├── Reasoning Model
│   ├── Vision Model
│   └── Vision API Base URL
├── Embedding Models
│   ├── Dense Embeddings Model
│   ├── Dense Embedding Dimension
│   └── SPLADE Sparse Model
├── Retrieval Pipeline
│   ├── Hybrid Weights (dense / sparse / exact)
│   ├── Retrieval Legs (toggle dense, sparse, exact)
│   ├── Top-K Results
│   └── Minimum RRF Score
├── Reranker
│   ├── Enabled
│   ├── Model
│   └── Score Threshold
├── Adaptive Retrieval
│   ├── Enabled
│   ├── Confidence Threshold
│   └── Reranker Threshold
├── Chunking
│   ├── Chunk Size
│   └── Overlap Percentage
├── Query Classifier
│   ├── Enabled
│   ├── Classification Prompt
│   └── Retrieval Config Presets (JSON editor)
├── GraphRAG
│   ├── Enabled
│   ├── Graph LLM
│   ├── Graph Retrieval Enabled
│   ├── Retrieval Hops
│   ├── Max Chunks per Document
│   └── LLM Context Size
├── Entity-Aware Retrieval
│   ├── Enabled
│   └── Boost Factor
├── Feature Flags
│   ├── Historical Memory (enabled, top-k, score threshold)
│   ├── Tool Calling (enabled, max iterations)
│   ├── Synthesis Mode
│   ├── Answer Quality Grading
│   ├── Watcher (enabled, poll interval, use inotify)
│   └── Processing Timeout
```

### Admin UI (org-specific settings, per org)

From the existing `/dashboard/admin/orgs` page, each organisation row gets an "Settings" button alongside "LLM Config". This opens an org-specific settings panel:

```
Organisation: Acme Corp — Settings
├── Retrieval Legs (toggle dense, sparse, exact)
├── Hybrid Weights (dense / sparse / exact)
├── Top-K Results
├── Reranker (enabled, score threshold)
├── Adaptive Retrieval (enabled, threshold)
├── Chunking (chunk size, overlap %)
├── Graph Retrieval (enabled, hops)
└── Feature Flags (tool calling, synthesis, quality grading, historical memory)
```

Each org setting page has a **"Reset to app defaults"** button that deletes the `OrgSetting` row, falling back to the app-wide `AppSetting` value.

---

## Migration Plan

### Phase 1: Backend data model

1. Create `app_settings` and `org_settings` tables (alembic migration).
2. Create seed script that reads all current `.env` values from `Settings` and inserts them as defaults into `app_default_settings`.
3. Create a new `AppSetting` model with a method `get_value(key)` that reads from DB, falling back to `config.py` defaults (preserving `.env` override as final fallback — allows gradual migration).
4. Create a `OrgSetting` model with `get_value(key, org_id)` that reads from DB, falling back to `AppSetting` value.

### Phase 2: Backend settings resolution layer

5. Create `app/core/setting_manager.py` — a singleton `SettingManager` that:
   - Reads `app_settings` from DB at startup (warm cache with TTL).
   - Provides `get(key: str)` and `set(key: str, value: Any)` methods.
   - Maints backward compatibility: if the DB has no row, falls back to `config.settings.<KEY>`.
6. Update key services to use `SettingManager` instead of `settings.<KEY>`:
   - `chat_service.py` — `QUERY_CLASSIFIER_ENABLED`, `QUERY_CLASSIFIER_PROMPT`, `OPENAI_MODEL`, `OPENAI_API_BASE`, `effective_query_model`
   - `retrieval.py` — all hybrid weights, leg enablement flags, retrieval top-k, min RRF score
   - `fast_pipeline.py` — adaptive retrieval, answer quality grading, historical memory, reranker
   - `rag_graph.py` — reranker, graphRAG
   - `entity_extractor.py` — entity aware, graphRAG
   - `document_processor.py` — processing timeout
   - `datastore_watcher.py` — watcher settings
   - `security.py` — `ACCESS_TOKEN_EXPIRE_MINUTES`, `SECRET_KEY` (keep SECRET_KEY in env)
   - `builtin_tools.py` — OPENAI_API_KEY (keep in env)

### Phase 3: Backend API endpoints

7. Create `backend/app/api/api_v1/settings.py`:
   - `GET /api/admin/settings` — list all app-wide settings (grouped, with descriptions) — Super Admin only
   - `PUT /api/admin/settings/{key}` — update a single app-wide setting — Super Admin only
   - `PATCH /api/admin/settings` — batch update multiple settings — Super Admin only
   - `GET /api/admin/settings/{key}/reset` — reset to factory default — Super Admin only
   - `GET /api/admin/orgs/{org_id}/settings` — list all org settings — Admin (of that org) or Super Admin
   - `PUT /api/admin/orgs/{org_id}/settings/{key}` — update an org setting — Admin or Super Admin
   - `DELETE /api/admin/orgs/{org_id}/settings/{key}` — reset org setting to app default — Admin or Super Admin

### Phase 4: Frontend UI

8. Add settings page to admin sidebar (`Settings` nav item).
9. Implement Super Admin settings pages:
   - `/dashboard/admin/settings` — tabbed interface matching the structure above
   - Each setting rendered as a form field with validation, tooltips (from `description`), and type-appropriate controls (toggle for bools, number input for ints/floats, text for strings, JSON editor for presets)
10. Extend Org detail view in `/dashboard/admin/orgs/[id]`:
    - Add "Settings" tab alongside existing info tabs
    - Reuse the same setting components but scoped to org_id
    - Show "Reset to default" badge on overridden settings

### Phase 5: Hot-reload & validation

11. Add a mechanism to reload affected services without full restart:
    - LLM settings: need restart (models loaded at service init)
    - Retriever feature flags: can be hot-reloaded via `SettingManager` cache invalidation (re-read `settings` object)
12. Add validation hooks:
    - Ensure `HYBRID_DENSE_WEIGHT + HYBRID_QDRANT_SPARSE_WEIGHT + HYBRID_EXACT_WEIGHT` is reasonable
    - Ensure `RERANKER_SCORE_THRESHOLD` is less than `ADAPTIVE_RETRIEVAL_RERANKER_THRESHOLD`
    - Ensure `ADAPTIVE_RETRIEVAL_THRESHOLD` is between 0 and 100
    - Validate model name format (non-empty string)

---

## Key Design Decisions

### Decision 1: Keep secrets in `.env`, not in DB
Secrets (`SECRET_KEY`, `OPENAI_API_KEY`, `MYSQL_PASSWORD`, `NEO4J_PASSWORD`) should remain in `.env` / the OS environment. Storing credentials in a database table increases the blast radius of any data breach. If org-specific API keys are needed in the future, they should be encrypted at rest using a key from `.env` (KMS or similar).

### Decision 2: Gradual migration with fallback chain
The setting resolution should follow this precedence (highest to lowest):
1. `.env` override (explicit env var takes priority for backward compatibility)
2. DB-stored value (`app_settings` / `org_settings`)
3. Factory default (from `app_default_settings` / `config.py`)

This allows a staged rollout: deploy the infrastructure first, then gradually migrate individual settings. During the transition, `.env` values still work.

### Decision 3: Super admin sees all settings
Super admins see all app-wide settings and all org-specific settings. Regular admins see only their org's settings and read-only app-wide defaults. This matches the existing RBAC pattern (`require_admin` / `require_super_admin`).

### Decision 4: Settings that require restart
Some settings (model names, cache directories, embedding models) take effect only at service startup because models are loaded into memory once. The UI should clearly label these as "requires restart" so admins know to restart the container. Feature-flag settings (enabled/disabled booleans) can be hot-reloaded.

### Decision 5: Chunking and embedding model changes require re-indexing
Changing `CHUNK_SIZE`, `OVERLAP_PERCENTAGE`, `DENSE_EMBEDDINGS_MODEL`, or `SPLADE_MODEL` invalidates existing indexed chunks. The UI should display a prominent warning when these settings are modified.

---

## Parameters Not Recommended for UI Migration

| Parameter | Reason |
|-----------|--------|
| `MYSQL_*` variables | Needed at import time for migrations; circular dependency. |
| `SECRET_KEY` | Changing invalidates all active JWT sessions; security risk in DB. |
| `OPENAI_API_KEY` | Secret credential; data exfiltration risk. |
| `UPLOAD_DIR`, `FASTEMBED_CACHE_DIR`, `RERANKER_CACHE_DIR` | File-system paths controlled by Docker volumes. |
| `QDRANT_*` variables | Vector store connection; needed before full app init. |
| `NEO4J_*` variables | Graph DB credentials; same as MySQL. |
| `ALGORITHM` | Rarely changes; tied to JWT signing. |
| `API_V1_STR`, `VERSION` | Infrastructure metadata. |

If in the future these need UI-manageability, they should go into a separate `__init__.py`-level config loader that is NOT imported by models/migrations, and stored encrypted in the DB.
