# Admin & Super Admin UI Settings Migration

## Overview

This document analyses every parameter currently controlled via `.env` in `backend/app/core/config.py` and determines which can be migrated to the Admin & Super Admin UIs, how they should be organized, and the architectural implications.

The existing codebase already has two role tiers:

- **Super Admin** — app-wide control (all organisations, system settings, infrastructure)
- **Admin** — organisation-scoped control (their org's users, LLM config, data stores)

Both are protected by `require_admin()` in `backend/app/core/security.py`. Super Admin additionally guards `require_super_admin()` endpoints.

The existing OrgLLMConfig model (`backend/app/models/org_llm_config.py`) stores per-org `api_base`, `model_name`, `query_model` but these values are stored in the DB and never consumed — the services still read `settings.OPENAI_API_BASE`, `settings.OPENAI_MODEL`, and `settings.QUERY_MODEL` from `config.py`. Moving settings to the UI requires wiring these stored values into the services.

---

## Classification of .env Parameters

### Category A — Do NOT move to UI (infrastructure / security)

These must remain in `.env` (or equivalent secrets manager). They are either credentials, infrastructure endpoints, or one-time setup values that do not change during normal operations.

| Parameter | Reason |
|---|---|
| `OPENAI_API_KEY` | Secret — must never be stored in cleartext DB. Use a secrets manager. |
| `SECRET_KEY` | JWT signing key. Changing it invalidates all sessions. Must stay in `.env`. |
| `MYSQL_*` (server, port, user, password, database) | Infrastructure — database connection details. |
| `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_GRPC_PORT` | Infrastructure — vector store endpoint. |
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | Infrastructure — graph database credentials. |
| `UPLOAD_DIR` | Filesystem path — deployment concern. |
| `FASTEMBED_CACHE_DIR` | Filesystem path. |
| `RERANKER_CACHE_DIR` | Filesystem path. |
| `TZ` | Container timezone — deployment concern. |
| `ROOT_ORG`, `SUPERADMIN_USERNAME`, `SUPERADMIN_PASSWORD` | One-time init seed. Not runtime-configurable. |
| `COMPOSE_PROFILES` | Docker deployment directive. |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Security-sensitive (controls token lifetime). Could be debated but best kept in `.env`. |
| `TIMEOUT_SECONDS` | Process-level timeout (used by celery/async workers). |

### Category B — Candidates for Super Admin UI (app-wide settings)

These control system-wide behaviour. All users/orgs share the same values. Changing them affects the entire instance.

| Parameter | Current Default | Type | Impact of Change |
|---|---|---|---|
| `OPENAI_MODEL` | `qwen/qwen3.5-9b` | String | Default model for all orgs. Can be overridden per-org. |
| `OPENAI_MODEL_CONTEXT_SIZE` | 131072 | Integer | Context window budget. |
| `OPENAI_API_BASE` | `http://localhost:1234/v1` | String | Base URL for all LLM calls. Can be overridden per-org. |
| `VISION_MODEL` | `qwen/qwen3.5-4b` | Optional String | OCR model. |
| `OPENAI_VISION_API_BASE` | (fallback to OPENAI_API_BASE) | Optional String | Separate base URL for vision/OCR calls. |
| `DENSE_EMBEDDINGS_MODEL` | `qwen/qwen3-embedding-0.6b` | String | Default embeddings model. |
| `DENSE_EMBEDDING_DIM` | 1024 | Integer | Must match the model's output dim. |
| `SPLADE_MODEL` | `prithivida/Splade_PP_en_v1` | String | Sparse embedding model. |
| `QUERY_MODEL` | (fallback to OPENAI_MODEL) | Optional String | Query rewriting / summarisation model. |
| `REASONING_MODEL` | (fallback to OPENAI_MODEL) | Optional String | "Thinking" answering mode model. |
| `RETRIEVAL_TOP_K` | 20 | Integer | Number of chunks returned per query. |
| `RETRIEVAL_MIN_RRF_SCORE` | 0.005 | Float | RRF threshold to drop chunks. |
| `RERANKER_ENABLED` | true | Boolean | Cross-encoder reranker on/off. |
| `RERANKER_MODEL` | `Xenova/ms-marco-MiniLM-L-12-v2` | String | Reranker model name. |
| `RERANKER_SCORE_THRESHOLD` | -5.0 | Float | Reranker logit cutoff. |
| `HYBRID_DENSE_WEIGHT` | 0.5 | Float | Dense leg RRF weight. |
| `HYBRID_QDRANT_SPARSE_WEIGHT` | 0.3 | Float | Sparse leg RRF weight. |
| `HYBRID_EXACT_WEIGHT` | 0.2 | Float | Exact (FTS) leg RRF weight. |
| `RETRIEVAL_DENSE_ENABLED` | true | Boolean | Enable/disable dense leg. |
| `RETRIEVAL_QDRANT_SPARSE_ENABLED` | true | Boolean | Enable/disable sparse leg. |
| `RETRIEVAL_EXACT_ENABLED` | true | Boolean | Enable/disable exact leg. |
| `CHUNK_SIZE` | 1500 | Integer | Chunk size in characters. |
| `OVERLAP_PERCENTAGE` | 0.20 | Float | Overlap between chunks. |
| `ADAPTIVE_RETRIEVAL_ENABLED` | true | Boolean | Two-pass retrieval on/off. |
| `ADAPTIVE_RETRIEVAL_THRESHOLD` | 55 | Float | Confidence threshold for expansion. |
| `ADAPTIVE_RETRIEVAL_RERANKER_THRESHOLD` | -5.0 | Float | Reranker threshold for adaptive pass. |
| `HISTORICAL_MEMORY_ENABLED` | true | Boolean | Past message retrieval on/off. |
| `HISTORICAL_MEMORY_TOP_K` | 5 | Integer | Historical memory docs returned. |
| `HISTORICAL_MEMORY_SCORE_THRESHOLD` | 2.0 | Float | Score cutoff for historical memory. |
| `PROCESSING_TIMEOUT_SILENCE_S` | 300 | Integer | Silent period before timeout warning. |
| `ANSWER_QUALITY_GRADING_ENABLED` | true | Boolean | Post-answer grading on/off. |

### Category C — Candidates for Admin UI (organisation-specific settings)

These control retrieval and processing behaviour at the org level. The rationale for making them per-org: different orgs may ingest different document types (legal docs vs. short notes), have different RAM/CPU budgets, and want different retrieval quality/latency tradeoffs.

| Parameter | Current Default | Type | Impact |
|---|---|---|---|
| `GRAPHRAG_ENABLED` | true | Boolean | Graph extraction on/off per org. |
| `GRAPHRAG_LLM` | (fallback to OPENAI_MODEL) | Optional String | LLM used for entity/relationship extraction. |
| `GRAPHRAG_RETRIEVAL_HOPS` | 2 | Integer | Graph hops at query time per org. |
| `GRAPHRAG_MAX_CHUNKS` | 0 (unlimited) | Integer | Max chunks for graph extraction per document. |
| `NEO4J_LLM_CONTEXT` | 12000 | Integer | Context budget for graph extraction LLM. |
| `RETRIEVAL_GRAPH_ENABLED` | true | Boolean | Graph retrieval leg on/off per org. |
| `ENTITY_AWARE_ENABLED` | true | Boolean | Entity-aware retrieval boost per org. |
| `ENTITY_BOOST_FACTOR` | 0.1 | Float | Score boost per matching entity per org. |
| `TOOL_CALLING_ENABLED` | true | Boolean | Agentic tool calling loop on/off per org. |
| `MAX_TOOL_ITERATIONS` | 5 | Integer | Max tool call iterations per chat turn per org. |
| `SYNTHESIS_MODE_ENABLED` | true | Boolean | Synthesis mode for MULTI_PART queries per org. |
| `QUERY_CLASSIFIER_ENABLED` | true | Boolean | Query classification for adaptive retrieval per org. |
| `RETRIEVAL_CONFIG_PRESETS` | JSON presets | String (JSON) | Per-query-type weight/top_k overrides per org. |
| `WATCHER_ENABLED` | true | Boolean | File system watcher on/off per org. |
| `WATCH_POLL_INTERVAL` | 2 | Integer | Poll interval in seconds per org. |
| `WATCHER_USE_INOTIFY` | true | Boolean | inotify vs polling per org. |
| `WATCH_DIR` | `/app/uploads` | String | Watch directory per org. |

### Category D — Partial migration (mixed infrastructure / settings)

These straddle the line. The *value* could be in the UI, but the *mechanism* requires care.

| Parameter | Rationale |
|---|---|
| `OPENAI_API_BASE` | Already partially handled by OrgLLMConfig (`api_base`). The existing model can be extended to cover all org-level API overrides. However, the services must be wired to read from DB instead of `settings`. |
| `RELIK_URL` | ReLiK service URL. If ReLiK is shared across orgs, it stays in `.env`. If per-org, it moves to Category C. |

---

## Current State of Admin UI

The existing admin UI (`/dashboard/admin/`) already has:

1. **Organisations page** — CRUD for orgs, LLM config dialog (per-org: `api_base`, `model_name`, `query_model`), ingestion status badge
2. **Users page** — CRUD for users, role assignment, password reset (super admin)
3. **Data Sources page** — CRUD for DataStores, scan/recover controls (super admin only)

The LLM config dialog already allows an admin to set per-org `api_base`, `model_name`, and `query_model`. However, these stored values are **never consumed** by the services — the services read from `settings` which reads from `.env`.

---

## UI Design Proposal

### Super Admin Settings Page

A new route `/dashboard/admin/settings` accessible only to super admins. Organized into sections with a save button:

```
+------------------------------------------+
|  Super Admin — App Settings               |
+------------------------------------------+
|  [LLM & Models] [Retrieval] [Chunking]   |
|  [GraphRAG/Neo4j] [Reranker] [Features]  |
+------------------------------------------+
|                                          |
|  Section: LLM & Models                   |
|                                          |
|  Default Response Model:     [qwen/___]  |
|  Context Window Size:        [131072  ]  |
|  Query Rewriting Model:      [qwen/___]  |
|  Reasoning Model:            [_________] │ (optional)
|  Vision/OCR Model:           [qwen/___]  │ (optional)
|  Base API URL:               [http://__] │
|  Vision API URL:             [http://__] │ (optional)
|  Dense Embeddings Model:     [qwen/___]  │
|  Embedding Dimension:        [1024    ]  │
|  SPLADE Sparse Model:        [_________] │
|                                          |
|  [Save Changes] [Reset to Defaults]      |
+------------------------------------------+
```

### Admin Org Settings Page

Extend the existing **Organisations** page to include a **"Settings"** tab alongside the current **"LLM Config"** tab. Each org gets its own settings dialog:

```
Org: Acme Corp          [Save] [Reset to App Defaults]
──────────────────────────────────────────────────────

  [Retrieval] [GraphRAG] [Agentic] [File Watcher]

  + Retrieval +
  Top-K:                    [20      ]
  RRF Score Threshold:      [0.005   ]
  Dense Weight:             [0.5     ]  Sparse: [0.3]  Exact: [0.2]
  Dense Leg:  [ON]  Sparse: [ON]  Exact: [ON]
  Graph Retrieval: [ON]

  + GraphRAG +
  Graph Extraction:    [ON]
  Graph Retrieval:     [ON]
  Hops at Query Time:  [2]
  Max Chunks/Document: [0 = unlimited]
  LLM Context Budget:  [12000]
  Entity-Aware Boost:  [ON]
  Boost Factor:        [0.1]

  + Agentic +
  Query Classification:      [ON]
  Tool Calling:              [ON]
  Max Tool Iterations:       [5]
  Synthesis Mode:            [ON]
  Adaptive Retrieval:        [ON]
  Adaptive Threshold:        [55]
  Historical Memory:         [ON]
  Historical Memory Top-K:   [5]
  Answer Quality Grading:    [ON]

  + File Watcher +
  Watcher Enabled:     [ON]
  Poll Interval (s):   [2]
  Use inotify:         [ON]
```

---

## Implementation Plan

### Phase 1: Database Schema (Foundation)

Create a single settings table that supports both app-wide and org-scoped settings, avoiding N migration scripts.

```
Table: settings
  id: INTEGER PK
  key: VARCHAR(128) — unique per scope
  scope: ENUM('app', 'org')
  org_id: INTEGER FK(organisations) NULLABLE — NULL for app-wide
  value: TEXT — JSON-encoded value for complex types
  updated_at: TIMESTAMP
  UNIQUE KEY(scope, key),
  UNIQUE KEY(scope, org_id, key)
```

Migrate existing OrgLLMConfig fields (`api_base`, `model_name`, `query_model`) into this table as `scope='org'` entries for each org. OrgLLMConfig can be deprecated in a future version.

For app-wide settings, add an initial seed row for every Category B parameter with its current default value.

**New models:**

```python
class Setting(Base, TimestampMixin):
    __tablename__ = "settings"
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(128), nullable=False, index=True)
    scope = Column(String(16), nullable=False)  # "app" or "org"
    org_id = Column(Integer, ForeignKey("organisations.id"), nullable=True)
    value = Column(Text, nullable=True)  # JSON-encoded
    # UNIQUE(org_id, key) when scope=org, UNIQUE(key) when scope=app
```

**Migration script**: `alembic` revision or SQL init script that seeds the table from `config.py` defaults.

### Phase 2: Settings Access API

Create a `settings_service.py` module that:
1. Reads settings from DB (app-wide with `.env` fallback, org-scoped with app-default + `.env` fallback).
2. Provides a `get_settings(scope, org_id=None)` method that returns a flattened dict.
3. Provides `update_setting(key, value, scope, org_id=None)` with validation.
4. Provides `bulk_update(settings_dict, scope, org_id=None)`.

The key design decision: **how settings are resolved**. Three tiers of precedence:

```
1. DB org-level setting (highest priority for admin-set values)
2. DB app-level setting (super admin global defaults)
3. .env config.py default (fallback for never-changed parameters)
```

### Phase 3: Backend API Endpoints

**Super Admin settings endpoints** (`/api/admin/settings`):

```
GET    /api/admin/settings                    — list all app-wide settings
PUT    /api/admin/settings                    — bulk update app-wide settings
POST   /api/admin/settings/{key}              — update single app-wide setting
DELETE /api/admin/settings/{key}              — reset single setting to .env default
```

**Org-level settings endpoints** (`/api/admin/orgs/{org_id}/settings`):

```
GET    /api/admin/orgs/{org_id}/settings      — list all org settings
PUT    /api/admin/orgs/{org_id}/settings      — bulk update org settings
POST   /api/admin/orgs/{org_id}/settings/{key} — update single setting
DELETE /api/admin/orgs/{org_id}/settings/{key} — reset to app defaults
```

**New Pydantic schemas** in `schemas/settings.py`:

```python
class AppSettingUpdate(BaseModel):
    key: str
    value: Any

class AppSettingsBulkUpdate(BaseModel):
    settings: List[AppSettingUpdate]

class OrgSettingsBulkUpdate(BaseModel):
    settings: List[AppSettingUpdate]
```

### Phase 4: Wire Settings into Services

This is the **hardest part**. The services currently import `settings` from `config.py` at the module level. Changes require one of two approaches:

**Approach A: Context manager (preferred)**

Wrap retrieval/ingestion calls in a context that sets org-specific settings:

```python
# In services like retrieval.py, graph_service.py, document_processor.py:
from app.services.settings_service import get_settings_for_org

def search(query, org_id=None, user_id=None, ...):
    if org_id:
        org_settings = get_settings_for_org(org_id)
    else:
        org_settings = settings  # fall back to global

    # Use org_settings instead of `settings`
    top_k = org_settings.RETRIEVAL_TOP_K
    # ... etc
```

**Approach B: Function-level overrides**

Pass settings dict as optional keyword argument:

```python
def search(query, settings=None, ...):
    s = settings or global_settings
    top_k = s.RETRIEVAL_TOP_K
```

Approach A is cleaner for the retrieval pipeline since `org_id` is already available at the API layer. Approach B is simpler to implement but requires threading `settings=None` through many function signatures.

**Key services that need updating:**
- `services/retrieval.py` — uses ~30 settings for query routing, weights, thresholds
- `services/graph_service.py` — uses GRAPHRAG_*, NEO4J_LLM_CONTEXT
- `services/document_processor.py` — uses CHUNK_SIZE, OVERLAP_PERCENTAGE, GRAPHRAG_ENABLED
- `services/agentic_rag/agentic_rag.py` — uses many settings for agentic pipeline config
- `services/confidence.py` — uses thresholds for confidence scoring
- `services/reranker.py` — uses RERANKER_* settings
- `services/entity_extractor.py` — uses ENTITY_AWARE_ENABLED, ENTITY_BOOST_FACTOR
- `services/chat_service.py` — uses TOOL_CALLING_ENABLED, MAX_TOOL_ITERATIONS, etc.
- `services/historical_memory.py` — uses HISTORICAL_MEMORY_* settings

### Phase 5: Frontend UI

**Super Admin Settings Page** (`/dashboard/admin/settings/page.tsx`):
- Tabbed interface matching the section layout in the design above
- Each tab: editable form fields with type-aware validation (int, float, boolean, JSON)
- "Save" button triggers PUT to `/api/admin/settings`
- "Reset to Defaults" button deletes individual setting or resets all

**Org Settings Dialog** (extension of `/dashboard/admin/orgs/page.tsx`):
- Replace current "LLM Config" button with a 2-button action: "LLM Config" + "Settings"
- Settings opens a tabbed dialog (same sections as super admin but org-scoped)
- Fields pre-filled with org values, with a "Use App Defaults" toggle for each section
- Save triggers PUT to `/api/admin/orgs/{org_id}/settings`

**New types** in `frontend/src/lib/api-types.ts`:

```typescript
export interface AppSetting {
  key: string;
  value: string | number | boolean | null;
  category: string;
  description: string;
}

export interface OrgSetting extends AppSetting {
  org_id: number;
  overridden: boolean;  // true if differs from app default
}
```

### Phase 6: Migration & Seed Data

1. Alembic migration to create `settings` table.
2. Seed app-wide settings from `config.py` defaults.
3. Migrate existing OrgLLMConfig records into the new table.
4. Backfill org-level settings with app defaults so every org has explicit rows.

### Phase 7: Runtime Reload

Settings changes currently require a server restart because `settings` is instantiated at import time in `config.py`. Options:

1. **Lazy evaluation**: Change `config.py` to use a lazy-loaded `Settings` proxy that re-reads from DB on each call. Acceptable for org-level settings (called per request) but not ideal for app-wide settings (called millions of times).

2. **Cache with TTL**: Cache the settings dict with a short TTL (30s). New settings take effect within 30s. Simple, no restart needed.

3. **Redis pub/sub**: Broadcast settings changes to all worker processes to reload. Overkill for initial migration.

4. **Force restart**: Document that app-wide settings require a restart. Org-level settings can be applied lazily (per-request DB lookup).

**Recommendation**: Start with approach 2 for org-level settings (lazy per-request) and approach 4 for app-wide settings. Document clearly.

---

## Parameter-by-Parameter Migration Mapping

| .env Parameter | UI Location | Scope | Notes |
|---|---|---|---|
| OPENAI_MODEL | Super Admin → LLM & Models | App | Falls back if org overrides |
| OPENAI_MODEL_CONTEXT_SIZE | Super Admin → LLM & Models | App | |
| OPENAI_API_BASE | Super Admin → LLM & Models | App | Existing OrgLLMConfig.api_base can feed into this |
| VISION_MODEL | Super Admin → LLM & Models | App | |
| OPENAI_VISION_API_BASE | Super Admin → LLM & Models | App | |
| DENSE_EMBEDDINGS_MODEL | Super Admin → LLM & Models | App | |
| DENSE_EMBEDDING_DIM | Super Admin → LLM & Models | App | |
| SPLADE_MODEL | Super Admin → LLM & Models | App | |
| QUERY_MODEL | Super Admin → LLM & Models | App | Existing OrgLLMConfig.query_model |
| REASONING_MODEL | Super Admin → LLM & Models | App | |
| RETRIEVAL_TOP_K | Super Admin → Retrieval | App | |
| RETRIEVAL_MIN_RRF_SCORE | Super Admin → Retrieval | App | |
| RERANKER_ENABLED | Super Admin → Reranker | App | |
| RERANKER_MODEL | Super Admin → Reranker | App | |
| RERANKER_SCORE_THRESHOLD | Super Admin → Reranker | App | |
| HYBRID_DENSE_WEIGHT | Super Admin → Retrieval | App | |
| HYBRID_QDRANT_SPARSE_WEIGHT | Super Admin → Retrieval | App | |
| HYBRID_EXACT_WEIGHT | Super Admin → Retrieval | App | |
| RETRIEVAL_DENSE_ENABLED | Super Admin → Retrieval | App | |
| RETRIEVAL_QDRANT_SPARSE_ENABLED | Super Admin → Retrieval | App | |
| RETRIEVAL_EXACT_ENABLED | Super Admin → Retrieval | App | |
| CHUNK_SIZE | Super Admin → Chunking | App | Warning: changes don't affect existing docs |
| OVERLAP_PERCENTAGE | Super Admin → Chunking | App | Warning: changes don't affect existing docs |
| GRAPHRAG_ENABLED | Admin → GraphRAG | Org | |
| GRAPHRAG_LLM | Admin → GraphRAG | Org | |
| GRAPHRAG_RETRIEVAL_HOPS | Admin → GraphRAG | Org | |
| GRAPHRAG_MAX_CHUNKS | Admin → GraphRAG | Org | |
| NEO4J_LLM_CONTEXT | Admin → GraphRAG | Org | |
| RETRIEVAL_GRAPH_ENABLED | Admin → GraphRAG | Org | |
| ENTITY_AWARE_ENABLED | Admin → GraphRAG | Org | |
| ENTITY_BOOST_FACTOR | Admin → GraphRAG | Org | |
| TOOL_CALLING_ENABLED | Admin → Agentic | Org | |
| MAX_TOOL_ITERATIONS | Admin → Agentic | Org | |
| SYNTHESIS_MODE_ENABLED | Admin → Agentic | Org | |
| QUERY_CLASSIFIER_ENABLED | Admin → Agentic | Org | |
| RETRIEVAL_CONFIG_PRESETS | Admin → Agentic | Org | JSON editor |
| ADAPTIVE_RETRIEVAL_ENABLED | Admin → Agentic | Org | |
| ADAPTIVE_RETRIEVAL_THRESHOLD | Admin → Agentic | Org | |
| ADAPTIVE_RETRIEVAL_RERANKER_THRESHOLD | Admin → Agentic | Org | |
| HISTORICAL_MEMORY_ENABLED | Admin → Agentic | Org | |
| HISTORICAL_MEMORY_TOP_K | Admin → Agentic | Org | |
| HISTORICAL_MEMORY_SCORE_THRESHOLD | Admin → Agentic | Org | |
| ANSWER_QUALITY_GRADING_ENABLED | Admin → Agentic | Org | |
| WATCHER_ENABLED | Admin → File Watcher | Org | |
| WATCH_POLL_INTERVAL | Admin → File Watcher | Org | |
| WATCHER_USE_INOTIFY | Admin → File Watcher | Org | |
| WATCH_DIR | Admin → File Watcher | Org | |
| PROCESSING_TIMEOUT_SILENCE_S | Super Admin → Features | App | |

---

## Risk Analysis

### Runtime Changes

The biggest technical risk is that `config.py` reads `os.getenv()` at module import time. Once imported, the `settings` singleton holds those values. Changing `.env` after startup has no effect. Moving settings to DB doesn't change this — the services must be wired to read from DB instead.

**Mitigation**: Implement lazy settings access (approach 2 in Phase 7 above). Cache per-request with a 30-second TTL. Org-level settings are inherently per-request (org_id available at API layer). App-wide settings can be checked less frequently.

### Chunking Parameter Changes

`CHUNK_SIZE` and `OVERLAP_PERCENTAGE` affect ingestion-time chunk boundaries. Changing them after documents exist creates inconsistent chunk sizes. The existing `.env.example` already documents this warning.

**Mitigation**: Add a prominent warning in the UI when these parameters are changed. Log a warning when ingestion uses different chunking params than the org's current stored values.

### Retrieval Weight Changes

Changing `HYBRID_*_WEIGHT` or `RETRIEVAL_TOP_K` affects query-time behaviour immediately (no re-indexing needed). These are the safest to move to the UI.

**Mitigation**: None needed — no migration or restart required.

### GraphRAG Parameter Changes

`GRAPHRAG_MAX_CHUNKS` and `NEO4J_LLM_CONTEXT` affect ingestion. Changing them mid-stream means some documents were extracted with different budgets.

**Mitigation**: Document that changing these requires re-ingestion for consistent results. No technical enforcement needed — same pattern as CHUNK_SIZE.

### Per-Config Presets (RETRIEVAL_CONFIG_PRESETS)

This is a JSON string that defines per-query-type retrieval presets. Making it org-level means each org can have different query-type behaviours. The UI should provide a structured form (not raw JSON) with a JSON preview/export.

---

## Existing OrgLLMConfig Deprecation Path

The current OrgLLMConfig table stores 3 fields (`api_base`, `model_name`, `query_model`). These should be migrated:

1. **During migration**: Copy existing OrgLLMConfig rows into the new `settings` table:
   - `api_base` → `{org_id}:app_api_base`
   - `model_name` → `{org_id}:openai_model`
   - `query_model` → `{org_id}:query_model`

2. **After migration**: Add a deprecation notice in the DB schema. The OrgLLMConfig endpoints can continue to work (redirecting reads/writes to the new settings table) for one version, then be removed.

---

## Priority & Suggested Order

| Phase | Effort | Risk | Impact |
|---|---|---|---|
| 1. DB schema + migration | Medium | Low | Foundation |
| 3. Backend API endpoints | Medium | Low | Enables UI |
| 2. Settings service layer | High | High | Core logic |
| 5. Frontend UI | Medium | Low | User-facing |
| 4. Wire into services | Very High | Very High | Everything depends on it |
| 7. Runtime reload | Low | Medium | UX polish |
| 6. Seed data | Low | Low | Completeness |
| 7. Deprecate OrgLLMConfig | Low | Low | Cleanup |

**Recommended order**: 1 → 3 → 2 → 5 → 4 → 7 → 6.

Phase 4 (wiring) is the most labour-intensive because 10+ service files need to be updated to use org-scoped settings. Start with the retrieval pipeline (`retrieval.py`, `confidence.py`) since it's the most-read code path. Then do ingestion (`document_processor.py`, `graph_service.py`). Then chat (`chat_service.py`).

---

## What NOT to Move (and Why)

### `GRAPHRAG_LLM` — Context Sensitivity

The `GRAPHRAG_LLM` field exists in both the app-wide config (falls back to `OPENAI_MODEL`) and is partially handled by the existing OrgLLMConfig (`model_name`). The ambiguity is: does `model_name` in OrgLLMConfig map to `OPENAI_MODEL` or `GRAPHRAG_LLM`?

Looking at the codebase, OrgLLMConfig stores these three fields and the admin UI labels them generically. The consumption path is unclear — there is no org-scoped LLM client. The services always use `settings.graphrag_model` which reads from `settings.GRAPHRAG_LLM or settings.OPENAI_MODEL`.

**Recommendation**: Map OrgLLMConfig fields as follows during migration:
- `api_base` → `OPENAI_API_BASE` (app-wide default, org can override)
- `model_name` → `OPENAI_MODEL` (app-wide default, org can override)
- `query_model` → `QUERY_MODEL` (org-level override)

`GRAPHRAG_LLM` does NOT get a pre-filled value from OrgLLMConfig. It should default to `OPENAI_MODEL` (existing behavior) and admins can optionally set a separate graph extraction model in the Admin UI's GraphRAG section.

### Connection Strings (MySQL, Neo4j, Qdrant)

These should remain in `.env` because:
1. They contain credentials (passwords).
2. They rarely change — they're deployment infrastructure.
3. Moving them to DB requires encrypted storage, key rotation, audit logging — a full secrets management system.
4. The existing security model has no concept of secret storage.

If per-org infrastructure is needed in the future (e.g., different orgs use different Neo4j clusters), add a `infra_credentials` field to the settings table with encryption at rest.

### `OPENAI_API_KEY`

Never store API keys in the DB in plaintext. If per-org API keys are needed, use a secrets manager (HashiCorp Vault, AWS Secrets Manager, etc.) or the LLM provider's multi-key rotation features. The existing OrgLLMConfig approach (storing `api_base` in DB) is fine because API bases are not secrets — but keys must not be.

---

## Appendix: Current OrgLLMConfig Consumption Gap

The OrgLLMConfig model exists but is never consumed by any service. Here's the evidence:

- `services/retrieval.py:48` — imports `settings`, uses `settings.RETRIEVAL_*`, `settings.HYBRID_*`, `settings.GRAPHRAG_*`, etc. Never checks for org-scoped overrides.
- `services/graph_service.py:71` — imports `settings`, uses `settings.GRAPHRAG_ENABLED`, `settings.NEO4J_*`, etc.
- `services/document_processor.py:28` — imports `settings`, uses chunking and GRAPHRAG settings.
- `services/chat_service.py:13` — imports `settings`, uses TOOL_CALLING_ENABLED, etc.
- `api/api_v1/query.py:17` — imports `settings`, uses retrieval config.
- No code anywhere does `db.query(OrgLLMConfig).filter(...).first()` and then uses those values instead of `settings`.

The admin UI does have a "LLM Config" button that reads/writes OrgLLMConfig, but the stored values are disconnected from the actual pipeline. Moving all settings to the new unified `settings` table and wiring them into services fixes this gap.
