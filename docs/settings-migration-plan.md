# Settings Migration Plan: .env → Super Admin & Admin UIs

> **Status:** Implementation plan (not yet implemented).
> **Reconciles:** `docs/admin-ui-settings-migration.md`, `docs/env-to-admin-ui.md`, and
> `docs/admin-settings-plan-grok.md`, which disagreed on scope. This document is the
> single source of truth, incorporating corrections from all three.
> **Source of truth for current values:** `backend/app/core/config.py` and `.env.example`.

## 1. Guiding principles

Three configuration tiers, with strict precedence and strict ownership:

```
┌──────────────────────────────────────────────────────────────────────┐
│  Tier 1 — .env            Host/app environment. Immutable for the     │
│                            duration of a process run. Read once at    │
│                            import time. Owned by ops/deployment.      │
│                                                                        │
│  Tier 2 — Super Admin UI  App-level settings + app-wide DEFAULTS for  │
│  (app scope)              every org-overridable setting. Owned by     │
│                            super_admin. Stored in DB.                  │
│                                                                        │
│  Tier 3 — Admin UI        Org-level OVERRIDES on top of Tier 2        │
│  (org scope)              defaults. Owned by the admin of that org    │
│                            (scoped to own org + descendants). Stored  │
│                            in DB.                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**Resolution precedence (per org, per setting):**

```
org override (Tier 3)  →  app default (Tier 2)  →  .env value (Tier 1)  →  config.py hardcoded default
```

A setting is only org-overridable if **all** of the following hold:
1. `org_id` is available at the call site.
2. Changing it per-org does not require reloading a process-global resource (a loaded ML model, a Qdrant collection, a tokenizer, a watcher process).
3. It does not affect **ingestion of shared DataStores**. DataStores are linked to multiple orgs via the `OrganizationDataStore` junction table — ingestion settings (chunking, graph extraction, embeddings) that differ per-org would produce inconsistent indexes for the same folder. Ingestion always uses app-level effective settings.

Settings that fail any test are **app-only** (Tier 2 has the value; Tier 3 cannot override).

**Ownership rules:**

- **Super Admin** sets app-level settings and the app-wide defaults for every org-overridable setting. Super Admin endpoints use `require_super_admin` (currently defined in `core/security.py` but unused — this plan wires it in).
- **Admin** sets/overrides org-level settings only, scoped via the existing `get_admin_org_ids(db, current_user)` (own org + all descendants). An admin can never touch another org's settings or any app-level setting.
- **.env** holds only things that describe the host/app environment and do not change during a run: database/vector-store/graph/Redis endpoints, secrets, filesystem mount points, timezone, init seed, docker profiles, trusted proxies, log level.

## 2. What stays in .env (Tier 1)

These describe the deployment environment. They are read at import time, never appear in any UI, and changing them requires a process restart (and often a container rebuild/remount).

| Variable | Why it stays in .env |
|---|---|
| `MYSQL_SERVER`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE` | DB connection — required before the app starts; changing breaks live sessions. |
| `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_GRPC_PORT` | Vector store endpoint — same. |
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | Graph DB credentials — same. |
| `REDIS_URL`, `REDIS_HOST`, `REDIS_PORT`, `REDIS_INSIGHT_PORT` | Redis endpoint — same. |
| `SECRET_KEY` | JWT signing secret. Changing invalidates every session. Security-sensitive; never browseable in UI. |
| `OPENAI_API_KEY` | Secret. Never stored in cleartext DB or shown in UI. |
| `UPLOAD_DIR` | Filesystem mount point. Changing requires a remount/restart. |
| `FASTEMBED_CACHE_DIR` | Filesystem mount point for ONNX model cache. |
| `RERANKER_CACHE_DIR` | Filesystem mount point for reranker model cache. |
| `TOKENIZER_MODEL` | Local tokenizer path — must be mounted in the container. Host environment. |
| `TZ` | Container timezone. Deployment concern. |
| `LOG_LEVEL` | Root logger level. Process-level; set at startup. |
| `TRUSTED_PROXIES` | Network/proxy config. Host environment. |
| `TIMEOUT_SECONDS` | Frontend HTTP request timeout. Not a backend `Settings` field at all — read by the Next.js client only. |
| `WATCHER_USE_INOTIFY` | Host FS capability (macOS Docker must force `false`; inotify is Linux-only). Not a policy knob. |
| `WATCH_DIR` | Legacy fallback only. Real watch paths are `DataStore.folder_path` (per-DataStore, not per-org). |
| `SANDBOX_BACKEND` | Process execution capability (`restrictedpython` vs alternatives). Security/host boundary, not a policy knob. |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Session security policy. Kept out of casual UI edits to avoid accidental session lifetime changes. Stays in `.env`. |
| `ROOT_ORG`, `SUPERADMIN_USERNAME`, `SUPERADMIN_PASSWORD` | One-time init seed. Used only on first startup. |
| `COMPOSE_PROFILES` | Docker Compose directive. Not an app setting. |
| `ALGORITHM` | Hardcoded `HS256` in `config.py`. Not a tunable. |
| `PROJECT_NAME`, `VERSION`, `API_V1_STR` | Product constants in `config.py`. Not tunable. |
| `SQLALCHEMY_DATABASE_URI` | Optional DSN override. Infra. |

**Note on `SECRET_KEY` / `ACCESS_TOKEN_EXPIRE_MINUTES`:** Both stay in `.env`. `SECRET_KEY` is a secret (rotation invalidates all sessions). `ACCESS_TOKEN_EXPIRE_MINUTES` is a security policy that should not be casually editable in a UI. A future "rotate secret key" endpoint with explicit session invalidation is a separate enhancement.

**Known discrepancies between `.env.example`, docs, and code (corrected in this plan):**

- `REASONING_MODEL` is in `.env` / `.env.example` but **not declared** in `config.py` — it is a dead env var today. This plan adds it to `config.py` as part of Phase 0.
- `TIMEOUT_SECONDS` is in `.env` but is a **frontend-only** value (not a backend `Settings` field).
- Older docs use `HYBRID_QDRANT_SPARSE_WEIGHT` / `RETRIEVAL_QDRANT_SPARSE_ENABLED`; the code uses `HYBRID_SPARSE_WEIGHT` / `RETRIEVAL_SPARSE_ENABLED`. This plan uses the **code names**.

## 3. Settings registry (the single source of metadata)

Every migratable setting is described once in a Python registry. The registry drives: DB seeding, API response shape, UI rendering (tabs/fields/types), validation, and the resolution layer. This avoids N per-domain tables and N migration scripts.

**New file:** `backend/app/core/settings_registry.py`

```python
from typing import Literal, Any, Callable
from dataclasses import dataclass

Scope = Literal["app", "org"]          # app-only, or app-default + org-override
Reload = Literal["immediate", "next_request", "restart"]

@dataclass(frozen=True)
class SettingDef:
    key: str                           # canonical key, matches config.py attr name
    category: str                      # UI tab grouping
    label: str                         # human label
    type: Literal["str","int","float","bool","json","secret"]
    default: Any                       # config.py hardcoded default (Tier 1 fallback)
    scope: Scope                       # "app" = app-only; "org" = app-default + org-override
    reload: Reload                     # when a change takes effect
    min: float | None = None           # optional numeric bounds
    max: float | None = None
    choices: list[str] | None = None   # optional enum
    help: str = ""                     # shown in UI tooltip
    requires_reindex: bool = False     # show re-index warning when changed
    secret: bool = False               # mask in API responses

REGISTRY: list[SettingDef] = [ ... ]   # see §5 for the full list
```

The registry is the **only** place settings are enumerated. The DB stores only overrides/app-values keyed by `key`; the registry supplies defaults, types, and validation. This means:

- Adding a setting = one line in `REGISTRY` + one column read in the consuming service. No new table, no new migration for the schema itself (only seed data).
- The UI is generated from the registry (tabs/fields/types), so frontend and backend cannot drift.

## 4. Complete classification of every setting

Each row below is one entry in the registry. `Scope = app` means Super Admin only (Tier 2). `Scope = org` means Super Admin sets the default (Tier 2) and Admin may override per-org (Tier 3).

### 4.1 App-only settings (Super Admin; no org override)

These have no per-org meaning, or changing them per-org would require reloading a process-global resource, or they affect ingestion of shared DataStores.

| Key | Default | Type | Category/Tab | Reload | Why app-only |
|---|---|---|---|---|---|
| `OPENAI_MODEL_CONTEXT_SIZE` | 131072 | int | LLM & Models | next_request | Token budgeting. (Org can override via `context_size` on the LLM tab — see §4.2.) |
| `DENSE_EMBEDDINGS_MODEL` | `local-embedding-model` | str | LLM & Models | restart | Qdrant collections are dimension-locked; per-org would need separate collections. Shared index geometry. |
| `DENSE_EMBEDDING_DIM` | 1024 | int | LLM & Models | restart | Must match the embeddings model; tied to collection schema. |
| `SPLADE_MODEL` | `prithivida/Splade_PP_en_v1` | str | LLM & Models | restart | Loaded once by FastEmbed into a process-global ONNX session. |
| `RERANKER_MODEL` | `Xenova/ms-marco-MiniLM-L-12-v2` | str | Reranker | restart | Loaded once into a process-global cross-encoder. (Enabled/threshold ARE org-overridable — see 4.2.) |
| `MEMORY_EMBEDDING_MODEL` | None | str? | System / Memory | restart | Embedding model for Redis store; tied to global embeddings. |
| `MEMORY_ENABLED` | true | bool | System / Memory | restart | Redis long-term memory toggle; affects LangGraph store wiring (process singleton). |
| `CHUNK_SIZE` | 1500 | int | Ingestion / Chunking | ingest | **Ingestion setting.** DataStores are shared across orgs — per-org chunking would produce inconsistent indexes for the same folder. `requires_reindex=True`. |
| `OVERLAP_PERCENTAGE` | 0.20 | float | Ingestion / Chunking | ingest | **Ingestion setting.** Same shared-DataStore constraint. `requires_reindex=True`. |
| `GRAPHRAG_ENABLED` | true | bool | Ingestion / GraphRAG | ingest | **Ingestion setting.** Graph extraction runs during ingestion of shared DataStores. |
| `GRAPHRAG_MAX_CHUNKS` | 0 | int | Ingestion / GraphRAG | ingest | **Ingestion setting.** `requires_reindex=True`. |
| `NEO4J_LLM_CONTEXT` | 12000 | int | Ingestion / GraphRAG | ingest | **Ingestion setting.** Char budget for extraction batches. `requires_reindex=True`. |
| `WATCHER_ENABLED` | true | bool | System / Watcher | restart | One watcher process watches all DataStore folders. Process-level, not per-org. |
| `WATCH_POLL_INTERVAL` | 2 | int | System / Watcher | restart | PollingObserver timeout — process-level. |
| `SANDBOX_TIMEOUT_S` | 10 | int | System / Agent | next_request | Sandbox policy. App-only. |
| `TOOL_CALL_MODE` | `auto` | str | System / Agent | next_request | Agent protocol choice (`native` / `json_text` / `auto`). Protocol, not behavior. |
| `QUERY_CLASSIFIER_PROMPT` | (long prompt) | text | Query Classification | next_request | Large template; should be consistent across orgs. The **enable** toggle is org-overridable; the prompt itself is app-only. |

> **Ingestion rule:** Because DataStores can be linked to multiple orgs via `OrganizationDataStore`, ingestion always uses app-level effective settings. This applies to: `CHUNK_SIZE`, `OVERLAP_PERCENTAGE`, `GRAPHRAG_ENABLED`, `GRAPHRAG_MAX_CHUNKS`, `NEO4J_LLM_CONTEXT`, `VISION_MODEL` (for OCR during ingestion), `GRAPHRAG_LLM` (for extraction during ingestion). Future enhancement: per-DataStore overrides (out of scope for this plan).

### 4.2 App-default + org-override settings (Super Admin sets default; Admin may override)

`org_id` is available at every call site listed in §6, and changing these per-org does not reload a process-global resource.

#### LLM endpoints & model selection

| Key | Default | Type | Category/Tab | Reload | Notes |
|---|---|---|---|---|---|
| `OPENAI_API_BASE` | `http://localhost:1234/v1` | str | LLM & Models | next_request | Already partially per-org via `OrgLLMConfig.api_base`. Migrated into the unified table. |
| `OPENAI_MODEL` | `local-model` | str | LLM & Models | next_request | Already partially per-org via `OrgLLMConfig.model_name`. |
| `OPENAI_MODEL_CONTEXT_SIZE` | 131072 | int | LLM & Models | next_request | Org can have a different chat model with a different context window. Override follows `model_name`. |
| `QUERY_MODEL` | None (→ OPENAI_MODEL) | str? | LLM & Models | next_request | Already partially per-org via `OrgLLMConfig.query_model`. |
| `REASONING_MODEL` | None (→ OPENAI_MODEL) | str? | LLM & Models | next_request | **Not yet in `config.py`** — add in Phase 0. New org field. |
| `VISION_MODEL` | None | str? | LLM & Models | next_request | OCR model. Org override applies to **query-time** vision; **ingestion OCR** uses app-level setting (shared DataStores). |
| `OPENAI_VISION_API_BASE` | None (→ OPENAI_API_BASE) | str? | LLM & Models | next_request | New org field. Same ingestion caveat as `VISION_MODEL`. |
| `GRAPHRAG_LLM` | None (→ OPENAI_MODEL) | str? | GraphRAG | next_request | Org override applies to **query-time** entity extraction. **Ingestion** graph extraction uses app-level setting (shared DataStores). |

> LLM clients are built per-request in `services/agentic_rag/llm_factory.py` (`build_chat_llm`) and `services/chat/chat_service.py` (`get_effective_llm_config`), so per-org model/base overrides are already feasible and partially wired. This plan extends those resolvers to read from the unified settings table instead of `OrgLLMConfig` directly.

#### Retrieval tuning

| Key | Default | Type | Category/Tab | Reload | Notes |
|---|---|---|---|---|---|
| `RETRIEVAL_TOP_K` | 20 | int | Retrieval | next_request | |
| `RETRIEVAL_MIN_RRF_SCORE` | 0.005 | float | Retrieval | next_request | |
| `DENSE_MIN_SCORE` | 0.5 | float | Retrieval | next_request | |
| `SPARSE_MIN_SCORE` | 5.0 | float | Retrieval | next_request | |
| `EXACT_MIN_SCORE` | 0.5 | float | Retrieval | next_request | Note: MySQL FTS scale ≠ SPLADE scale. |
| `HYBRID_DENSE_WEIGHT` | 0.5 | float | Retrieval | next_request | |
| `HYBRID_SPARSE_WEIGHT` | 0.3 | float | Retrieval | next_request | |
| `HYBRID_EXACT_WEIGHT` | 0.2 | float | Retrieval | next_request | |
| `RETRIEVAL_DENSE_ENABLED` | true | bool | Retrieval | next_request | Ingestion always indexes all legs. |
| `RETRIEVAL_SPARSE_ENABLED` | true | bool | Retrieval | next_request | |
| `RETRIEVAL_EXACT_ENABLED` | true | bool | Retrieval | next_request | |
| `RETRIEVAL_GRAPH_ENABLED` | true | bool | Retrieval | next_request | Graph retrieval leg toggle (ingestion unaffected). |
| `RETRIEVAL_CONFIG_PRESETS` | (JSON) | json | Retrieval | next_request | Per-query-type presets. UI: structured form + JSON preview. |
| `ENTITY_AWARE_ENABLED` | true | bool | Retrieval | next_request | |
| `ENTITY_BOOST_FACTOR` | 0.1 | float | Retrieval | next_request | |

#### Adaptive retrieval

| Key | Default | Type | Category/Tab | Reload | Notes |
|---|---|---|---|---|---|
| `ADAPTIVE_RETRIEVAL_ENABLED` | true | bool | Adaptive Retrieval | next_request | |
| `ADAPTIVE_RETRIEVAL_THRESHOLD` | 55 | float | Adaptive Retrieval | next_request | 0–100 confidence threshold. |
| `ADAPTIVE_RETRIEVAL_RERANKER_THRESHOLD` | -5.0 | float | Adaptive Retrieval | next_request | Must be < `RERANKER_SCORE_THRESHOLD`. |
| `RETRIEVAL_RELAX_LEVEL2_RERANKER_THRESHOLD` | -8.0 | float | Adaptive Retrieval | next_request | Agentic graduated ladder deepest tier. |

#### Reranker (model is app-only; enabled/threshold are org-overridable)

| Key | Default | Type | Category/Tab | Reload | Notes |
|---|---|---|---|---|---|
| `RERANKER_ENABLED` | true | bool | Reranker | next_request | |
| `RERANKER_SCORE_THRESHOLD` | -2.0 | float | Reranker | next_request | |

#### Chunking — app-only (see §4.1)

`CHUNK_SIZE` and `OVERLAP_PERCENTAGE` are **app-only** because DataStores are shared across orgs. Per-org chunking would produce inconsistent indexes for the same folder. They appear on the Super Admin "Ingestion / Chunking" tab with a re-index warning.

> `document_processor.process_document_background` already accepts `chunk_size`/`chunk_overlap` as optional kwargs. Callers pass the **app-level** resolved values, not per-org.

#### GraphRAG (query-time only — ingestion knobs are app-only, see §4.1)

| Key | Default | Type | Category/Tab | Reload | Notes |
|---|---|---|---|---|---|
| `GRAPHRAG_RETRIEVAL_HOPS` | 1 | int | GraphRAG | next_request | 1–3 recommended. Query-time only. |
| `GRAPHRAG_RETRIEVAL_LIMIT` | 20 | int | GraphRAG | next_request | Query-time only. |
| `GRAPHRAG_ENTITY_FANOUT_CAP` | 50 | int | GraphRAG | next_request | Bounds hub-entity fan-out. Query-time only. |

#### Query classification

| Key | Default | Type | Category/Tab | Reload | Notes |
|---|---|---|---|---|---|
| `QUERY_CLASSIFIER_ENABLED` | true | bool | Query Classification | next_request | The **prompt** is app-only (§4.1); only the enable toggle is org-overridable. |

#### Agentic features

| Key | Default | Type | Category/Tab | Reload | Notes |
|---|---|---|---|---|---|
| `TOOL_CALLING_ENABLED` | true | bool | Agentic | next_request | |
| `MAX_TOOL_ITERATIONS` | 5 | int | Agentic | next_request | |
| `SYNTHESIS_MODE_ENABLED` | true | bool | Agentic | next_request | |
| `AGENT_MAX_ITERATIONS` | 8 | int | Agentic | next_request | |
| `AGENT_MAX_RETRIEVALS` | 3 | int | Agentic | next_request | |
| `AGENT_MAX_CODE_EXEC` | 3 | int | Agentic | next_request | |
| `AGENT_MAX_REFLECTIONS` | 2 | int | Agentic | next_request | |
| `AGENT_REFLECT_EVERY` | 2 | int | Agentic | next_request | |
| `AGENT_MAX_TOOL_RETRIES` | 3 | int | Agentic | next_request | |
| `AGENT_RETRY_BACKOFF_BASE` | 0.5 | float | Agentic | next_request | |
| `AGENT_MAX_CLARIFICATIONS` | 1 | int | Agentic | next_request | |
| `AGENT_HISTORY_PAIRS` | 3 | int | Agentic | next_request | |
| `AGENT_MAX_WALL_SECONDS` | 120 | float | Agentic | next_request | Wall-clock budget for the agent loop. |

#### Historical memory

| Key | Default | Type | Category/Tab | Reload | Notes |
|---|---|---|---|---|---|
| `HISTORICAL_MEMORY_ENABLED` | true | bool | Memory | next_request | |
| `HISTORICAL_MEMORY_TOP_K` | 5 | int | Memory | next_request | |
| `HISTORICAL_MEMORY_SCORE_THRESHOLD` | 2.0 | float | Memory | next_request | |

#### Context, compaction & quality (org-overridable — runtime behavior)

| Key | Default | Type | Category/Tab | Reload | Notes |
|---|---|---|---|---|---|
| `CONTEXT_RESERVED_GENERATION` | 4096 | int | Context | next_request | Token budget for generation. |
| `CONTEXT_TOOL_BUDGET` | 8192 | int | Context | next_request | Token budget for tools. |
| `HIGHLIGHTS_TOKEN_CAP` | 2000 | int | Context | next_request | |
| `CONTEXT_COMPACTION_TRIGGER_RATIO` | 0.85 | float | Context | next_request | |
| `COMPACTION_ENABLED` | true | bool | Memory | next_request | Conversation compaction. |
| `COMPACTION_KEEP_RECENT` | 10 | int | Memory | next_request | |
| `COMPACTION_SUMMARY_MAX_CHARS` | 2000 | int | Memory | next_request | |
| `ANSWER_QUALITY_GRADING_ENABLED` | true | bool | Quality | next_request | |
| `PROCESSING_TIMEOUT_SILENCE_S` | 300 | int | Quality | next_request | Chat processing silence timeout. |

#### File watcher — app-only (see §4.1)

`WATCHER_ENABLED`, `WATCH_POLL_INTERVAL` are **app-only** (one watcher process watches all DataStore folders). `WATCHER_USE_INOTIFY` and `WATCH_DIR` stay in `.env` (host capability / legacy fallback). Real watch paths are `DataStore.folder_path`, not per-org.

> **Pre-existing discrepancy:** migration `0001_add_watch_dir_to_organisations` added `watch_dir` to the `organisations` table, but `Organisation` in `models/organisation.py` does not declare the column. Since `WATCH_DIR` is a legacy fallback (real paths are on `DataStore.folder_path`), this plan does **not** add it to the model. The column can be dropped in a future cleanup migration.

## 5. The registry in full (canonical list)

The registry contains exactly the keys in §4.1 and §4.2. Each entry's `default` is copied from `config.py` so the resolution layer can fall back to it when no DB row exists. The `.env` value (Tier 1) is read at startup and stored as the initial app-level row during seeding, so the effective default after migration equals the current `.env` value.

Example entries (abbreviated; the real file lists all ~60):

```python
REGISTRY = [
    # ── App-only (scope="app") — Super Admin only, no org override ──────
    SettingDef("DENSE_EMBEDDINGS_MODEL", "LLM & Models", "Dense embeddings model",
               "str", "local-embedding-model", scope="app", reload="restart"),
    SettingDef("DENSE_EMBEDDING_DIM", "LLM & Models", "Embedding dimension",
               "int", 1024, scope="app", reload="restart", min=1),
    SettingDef("SPLADE_MODEL", "LLM & Models", "SPLADE sparse model",
               "str", "prithivida/Splade_PP_en_v1", scope="app", reload="restart"),
    SettingDef("RERANKER_MODEL", "Reranker", "Reranker model",
               "str", "Xenova/ms-marco-MiniLM-L-12-v2", scope="app", reload="restart"),
    SettingDef("CHUNK_SIZE", "Ingestion", "Chunk size (chars)",
               "int", 1500, scope="app", reload="ingest", min=100, max=1800,
               requires_reindex=True),
    SettingDef("OVERLAP_PERCENTAGE", "Ingestion", "Overlap fraction",
               "float", 0.20, scope="app", reload="ingest", min=0.0, max=0.9,
               requires_reindex=True),
    SettingDef("GRAPHRAG_ENABLED", "Ingestion", "Enable graph extraction",
               "bool", True, scope="app", reload="ingest"),
    SettingDef("GRAPHRAG_MAX_CHUNKS", "Ingestion", "Max chunks for graph extraction",
               "int", 0, scope="app", reload="ingest", requires_reindex=True),
    SettingDef("NEO4J_LLM_CONTEXT", "Ingestion", "Graph extraction LLM context budget",
               "int", 12000, scope="app", reload="ingest", requires_reindex=True),
    SettingDef("WATCHER_ENABLED", "System", "Enable file watcher",
               "bool", True, scope="app", reload="restart"),
    SettingDef("WATCH_POLL_INTERVAL", "System", "Watcher poll interval (s)",
               "int", 2, scope="app", reload="restart", min=1),
    SettingDef("QUERY_CLASSIFIER_PROMPT", "Query Classification", "Classifier prompt template",
               "text", "...", scope="app", reload="next_request"),
    SettingDef("TOOL_CALL_MODE", "System", "Tool call protocol",
               "str", "auto", scope="app", reload="next_request",
               choices=("native", "json_text", "auto")),
    SettingDef("MEMORY_ENABLED", "System", "Enable Redis long-term memory",
               "bool", True, scope="app", reload="restart"),
    # ── App-default + org-override (scope="org") ────────────────────────
    SettingDef("OPENAI_API_BASE", "LLM & Models", "Base API URL",
               "str", "http://localhost:1234/v1", scope="org", reload="next_request"),
    SettingDef("OPENAI_MODEL", "LLM & Models", "Response model",
               "str", "local-model", scope="org", reload="next_request"),
    SettingDef("OPENAI_MODEL_CONTEXT_SIZE", "LLM & Models", "Context window size",
               "int", 131072, scope="org", reload="next_request", min=1024),
    SettingDef("REASONING_MODEL", "LLM & Models", "Reasoning model",
               "str", None, scope="org", reload="next_request"),
    SettingDef("RETRIEVAL_TOP_K", "Retrieval", "Top-K",
               "int", 20, scope="org", reload="next_request", min=1, max=200),
    SettingDef("RERANKER_ENABLED", "Reranker", "Enable reranker",
               "bool", True, scope="org", reload="next_request"),
    SettingDef("GRAPHRAG_RETRIEVAL_HOPS", "GraphRAG", "Graph query hops",
               "int", 1, scope="org", reload="next_request", min=1, max=5),
    SettingDef("TOOL_CALLING_ENABLED", "Agentic", "Enable tool calling",
               "bool", True, scope="org", reload="next_request"),
    SettingDef("ANSWER_QUALITY_GRADING_ENABLED", "Quality", "Enable answer grading",
               "bool", True, scope="org", reload="next_request"),
    # ... (all remaining keys from §4)
]
```

## 6. Database schema

A single generic table holds both app-level values and org-level overrides. The registry supplies types/defaults; the table stores only what has been set.

```sql
CREATE TABLE settings (
  id          INT PRIMARY KEY AUTO_INCREMENT,
  key         VARCHAR(128)  NOT NULL,
  scope       VARCHAR(8)    NOT NULL,   -- 'app' | 'org'
  org_id      INT           NULL,       -- NULL when scope='app'
  value       TEXT          NULL,       -- JSON-encoded (str/int/float/bool/object)
  updated_by  INT           NULL,       -- user.id of last editor
  created_at  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
  updated_at  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY uq_app_key (key)                    -- only one app row per key
    -- enforced for scope='app' via app logic / partial uniqueness
  UNIQUE KEY uq_org_key (org_id, key),           -- one org row per key per org
  FOREIGN KEY (org_id) REFERENCES organisations(id) ON DELETE CASCADE,
  FOREIGN KEY (updated_by) REFERENCES users(id)  ON DELETE SET NULL,
  INDEX idx_settings_scope (scope)
);
```

> **MySQL partial uniqueness:** MySQL has no partial unique index. Two approaches:
> 1. **Application-level enforcement** (simpler): the settings service upserts by `(scope='app', key)` with `org_id IS NULL`. The service guarantees uniqueness.
> 2. **Generated column** (belt-and-suspenders): add a stored generated column that produces a unique scope-key string and put a `UNIQUE` index on it:
>    ```sql
>    scope_key VARCHAR(200) GENERATED ALWAYS AS (
>      CASE WHEN scope='app' THEN CONCAT('app::', setting_key)
>           ELSE CONCAT('org:', org_id, '::', setting_key) END
>    ) STORED,
>    UNIQUE KEY uq_scope_key (scope_key)
>    ```
> This plan recommends approach 1 for v1; approach 2 can be added if DB-level enforcement is needed.

**Optional audit table (Phase 7):**

```sql
CREATE TABLE settings_audit (
  id          BIGINT PRIMARY KEY AUTO_INCREMENT,
  scope       VARCHAR(8)   NOT NULL,
  org_id      INT          NULL,
  key         VARCHAR(128) NOT NULL,
  old_value   TEXT         NULL,
  new_value   TEXT         NULL,
  actor_user_id INT        NULL,
  created_at  TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (org_id) REFERENCES organisations(id) ON DELETE CASCADE,
  FOREIGN KEY (actor_user_id) REFERENCES users(id) ON DELETE SET NULL
);
```

Every upsert/reset inserts an audit row. This is optional for v1; add when compliance requires it.

**New model:** `backend/app/models/setting.py`

```python
from sqlalchemy import Column, Integer, String, Text, ForeignKey, Index
from .base import Base, TimestampMixin

class Setting(Base, TimestampMixin):
    __tablename__ = "settings"
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(128), nullable=False, index=True)
    scope = Column(String(8), nullable=False)        # "app" | "org"
    org_id = Column(Integer, ForeignKey("organisations.id", ondelete="CASCADE"), nullable=True)
    value = Column(Text, nullable=True)              # JSON-encoded
    updated_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    __table_args__ = (
        Index("uq_org_key", "org_id", "key", unique=True),
        Index("idx_settings_scope", "scope"),
    )
```

Register in `models/__init__.py`.

**OrgLLMConfig deprecation:** `org_llm_configs` is not dropped in this plan. During migration, its rows are copied into `settings` as `scope='org'` entries (`OPENAI_API_BASE`, `OPENAI_MODEL`, `QUERY_MODEL`). The LLM resolvers (`llm_factory.get_org_llm`, `chat_service.get_effective_llm_config`) are rewritten to read from the settings service. `OrgLLMConfig` is kept read-only for one release as a fallback, then removed in a follow-up migration.

## 7. Backend implementation

### 7.1 Settings service — `backend/app/services/settings_service.py`

Core responsibilities:

1. **Resolve** a setting for a given org: org override → app value → registry default.
2. **Read all** settings for an org as a typed dict (for service call sites).
3. **CRUD** with validation against the registry.
4. **Cache** with a short TTL.

```python
from typing import Any, Optional
from sqlalchemy.orm import Session
import json, time

from app.core.config import settings as env_settings
from app.core.settings_registry import REGISTRY, SettingDef
from app.models.setting import Setting

_REGISTRY_BY_KEY = {d.key: d for d in REGISTRY}
_CACHE_TTL = 30  # seconds

def get_setting(db: Session, key: str, org_id: Optional[int] = None) -> Any:
    """Resolve a single setting with 2-tier precedence."""
    defn = _REGISTRY_BY_KEY[key]
    # Tier 3: org override (only if scope allows)
    if defn.scope == "org" and org_id is not None:
        row = db.query(Setting).filter(
            Setting.scope == "org", Setting.org_id == org_id, Setting.key == key
        ).first()
        if row is not None and row.value is not None:
            return _decode(row.value, defn)
    # Tier 2: app value
    row = db.query(Setting).filter(
        Setting.scope == "app", Setting.org_id.is_(None), Setting.key == key
    ).first()
    if row is not None and row.value is not None:
        return _decode(row.value, defn)
    # Tier 2: registry default
    return defn.default

def get_org_settings(db: Session, org_id: Optional[int]) -> dict:
    """Resolve ALL registry keys for an org into a typed dict.
    For app-only keys, returns the app value regardless of org_id.
    For org keys, applies 2-tier precedence."""
    out = {}
    for defn in REGISTRY:
        out[defn.key] = get_setting(db, defn.key, org_id if defn.scope == "org" else None)
    return out

def upsert_app_setting(db, key, value, user_id): ...   # validate, store scope='app'
def upsert_org_setting(db, org_id, key, value, user_id): ...  # validate, store scope='org'
def reset_org_setting(db, org_id, key): ...   # delete org row → falls back to app
def reset_app_setting(db, key): ...           # delete app row → falls back to .env
```

`_decode` parses JSON and casts to the registry type. Validation (min/max/choices) runs on upsert.

**Caching:** `get_org_settings` is the hot path (called per chat request). Cache the resolved dict per `(org_id)` in process memory with a 30s TTL. Invalidate on any upsert/reset for that org or for the app scope. For v1 this is a simple module-level dict + timestamps; Redis pub/sub invalidation is a later enhancement.

### 7.2 The per-org settings object

To minimize churn at call sites that currently do `settings.RETRIEVAL_TOP_K`, introduce a lightweight accessor:

```python
# backend/app/services/settings_service.py
class OrgSettings:
    """Attribute-access wrapper over get_org_settings, with .env fallback."""
    def __init__(self, db: Session, org_id: Optional[int]):
        self._db = db
        self._org_id = org_id
        self._resolved = get_org_settings(db, org_id)
    def __getattr__(self, key):
        return self._resolved[key]
    # computed properties mirroring config.py
    @property
    def chunk_overlap(self): return int(self.CHUNK_SIZE * self.OVERLAP_PERCENTAGE)
    @property
    def effective_query_model(self): return self.QUERY_MODEL or self.OPENAI_MODEL
    @property
    def effective_vision_api_base(self): return self.OPENAI_VISION_API_BASE or self.OPENAI_API_BASE
    @property
    def graphrag_model(self): return self.GRAPHRAG_LLM or self.OPENAI_MODEL
    @property
    def retrieval_config_presets(self): return json.loads(self.RETRIEVAL_CONFIG_PRESETS)
```

Services receive an `OrgSettings` instance (or the global `settings` when no org context) instead of reading the `settings` singleton. See §7.4.

### 7.3 API endpoints

**New router:** `backend/app/api/api_v1/settings.py`

```
# ── Super Admin: app-level settings (Tier 2) ──────────────────────────
GET    /api/admin/settings                         → list all app settings (with metadata + effective value + provenance)
GET    /api/admin/settings/schema                  → registry metadata for UI form generation
PUT    /api/admin/settings                         → bulk upsert app settings
POST   /api/admin/settings/{key}                   → upsert single app setting
DELETE /api/admin/settings/{key}                   → reset app setting to .env default
GET    /api/admin/settings/effective               → full effective app config snapshot (debug)
# All guarded by require_super_admin

# ── Admin: org-level overrides (Tier 3) ───────────────────────────────
GET    /api/admin/orgs/{org_id}/settings           → list org settings (value + overridden flag + app default)
GET    /api/admin/orgs/{org_id}/settings/schema    → registry metadata for org-overridable keys only
PUT    /api/admin/orgs/{org_id}/settings           → bulk upsert org overrides (null value = clear override)
POST   /api/admin/orgs/{org_id}/settings/{key}     → upsert single org override
DELETE /api/admin/orgs/{org_id}/settings/{key}     → delete org override (revert to app default)
DELETE /api/admin/orgs/{org_id}/settings           → clear all org overrides
# All guarded by require_admin + get_admin_org_ids scope check (existing pattern)
```

**Schema endpoints:** `/settings/schema` returns the registry metadata (types, labels, categories, validation rules, scope) so the frontend can build forms dynamically without hardcoding field definitions. This prevents UI/backend drift.

**Provenance:** each setting item includes a `source` field: `"database"` (a DB row exists) or `"install_default"` (falling back to the registry default). The UI shows a badge per field so operators know where the effective value comes from.

**Scope enforcement:**
- App endpoints use `Depends(require_super_admin)`.
- Org endpoints use `Depends(require_admin)` then `get_admin_org_ids(db, current_user)` to verify `org_id` is within the admin's scope (same pattern as the existing `get_org_llm_config` endpoint).
- Org endpoints reject any key whose registry `scope != "org"` with `403` ("This setting cannot be overridden per organisation").
- App endpoints accept all keys (super admin owns both app-only defaults and the app-wide defaults for org-overridable keys).

**Pydantic schemas:** `backend/app/schemas/setting.py`

```python
class SettingItem(BaseModel):
    key: str
    value: Any
    category: str
    label: str
    type: str
    scope: Literal["app", "org"]
    overridden: bool = False        # org endpoint only: true if org row exists
    app_default: Any = None         # org endpoint only: the app-level value
    requires_reindex: bool = False
    help: str = ""

class SettingsListResponse(BaseModel):
    settings: list[SettingItem]

class SettingUpdate(BaseModel):
    key: str
    value: Any

class SettingsBulkUpdate(BaseModel):
    settings: list[SettingUpdate]
```

Wire the router in `api/api_v1/api.py`:

```python
from app.api.api_v1 import settings as settings_router
api_router.include_router(settings_router.app_router, prefix="/admin", tags=["settings"])
api_router.include_router(settings_router.org_router, prefix="/admin", tags=["settings"])
```

### 7.4 Wiring settings into services

This is the largest change. Today every service reads the `settings` singleton from `config.py`. The migration threads an `OrgSettings` (or the global `settings` fallback) into each call path. `org_id` is already available at every entry point (`chat.py`, `knowledge_base.py`).

**Call sites and required changes:**

| Service file | Current | Change |
|---|---|---|
| `services/retrieval/retrieval.py` (`hybrid_search`, `hybrid_search_with_legs`) | reads `settings.HYBRID_*`, `RETRIEVAL_TOP_K`, `ENTITY_AWARE_*`, `RERANKER_*` | Add `org_settings: OrgSettings` param. Callers in `chat.py` / `rag_retrieve.py` build it from `OrgSettings(db, org_id)`. |
| `services/graph/graph_service.py` | reads `settings.GRAPHRAG_*`, `NEO4J_LLM_CONTEXT`, `GRAPHRAG_RETRIEVAL_HOPS` | Add `org_settings` param. **Query path** uses org-scoped settings (hops, limit, fanout). **Ingestion/extract path** uses app-level settings (GRAPHRAG_ENABLED, MAX_CHUNKS, NEO4J_LLM_CONTEXT). |
| `services/ingestion/document_processor.py` | reads `settings.CHUNK_SIZE`, `OVERLAP_PERCENTAGE`, `GRAPHRAG_ENABLED` | Already accepts `chunk_size`/`chunk_overlap` kwargs. Uses **app-level** `OrgSettings` (org_id=None) — ingestion settings are app-only due to shared DataStores. |
| `services/agentic_rag/llm_factory.py` (`get_org_llm`, `build_chat_llm`) | reads `OrgLLMConfig` + `settings` | Rewrite to read LLM keys from `OrgSettings` (which reads the unified table). Drop direct `OrgLLMConfig` query. |
| `services/chat/chat_service.py` (`get_effective_llm_config`) | reads `OrgLLMConfig` + `settings` | Same: read from `OrgSettings`. |
| `services/chat/chat_service.py` (query classifier) | reads `settings.QUERY_CLASSIFIER_ENABLED`, `QUERY_CLASSIFIER_PROMPT` | Read from `org_settings`. |
| `services/chat/historical_memory.py` | reads `settings.HISTORICAL_MEMORY_*`, `RERANKER_ENABLED` | Read from `org_settings`. |
| `services/agentic_rag/tools/rag_retrieve.py` | reads `settings.ADAPTIVE_RETRIEVAL_ENABLED` | Read from `org_settings` passed via `ToolContext`. |
| `services/agentic_rag/` (agent loop config) | reads `settings.AGENT_MAX_*`, `TOOL_CALLING_*`, `SYNTHESIS_MODE_*`, `CONTEXT_*`, `HIGHLIGHTS_TOKEN_CAP`, `AGENT_MAX_WALL_SECONDS` | Read from `org_settings`. App-only keys resolve to app value automatically. |
| `services/reranker.py` | reads `settings.RERANKER_ENABLED`, `RERANKER_SCORE_THRESHOLD` | Read from `org_settings`. Model stays app-level (loaded once). |
| `services/graph/entity_extractor.py` | reads `settings.GRAPHRAG_ENABLED` | Read from `org_settings`. |
| `services/confidence.py` | reads thresholds | Read from `org_settings`. |
| Watcher service | reads `settings.WATCHER_*` | Uses **app-level** settings only (process-level watcher). `WATCHER_USE_INOTIFY` and `WATCH_DIR` stay in `.env`. Real paths from `DataStore.folder_path`. |

**Threading strategy:** Add `org_settings: OrgSettings | None = None` to function signatures; default to the global `settings` singleton when `None`. This keeps the diff small and lets unconverted callers keep working during a phased rollout. Once all call sites are converted, make the param required.

**Build location:** Construct `OrgSettings(db, current_user.org_id)` once per request in the API layer (`chat.py`, `knowledge_base.py`) and pass it down. Do not construct it deep in the service tree (avoids repeated DB/cache lookups).

**Ingestion rule (critical):** Ingestion (`document_processor`, `graph_service` extract path, `document_converter` OCR) always uses **app-level** effective settings, even when `org_id` is available. This is because DataStores are shared across orgs via `OrganizationDataStore` — per-org ingestion settings would produce inconsistent indexes for the same folder. The `OrgSettings` passed to ingestion paths is constructed with `org_id=None` (or the ingestion code reads only app-level keys). Query-time paths (retrieval, graph query, entity extraction, agent loop) use the full org-scoped `OrgSettings`.

**`OrgLLMConfig` dual-write strategy:** During the compatibility window (Phases 3–7):
1. **Keep** `org_llm_configs` table and existing `GET/PUT /api/admin/orgs/{org_id}/llm-config` endpoints.
2. **Read**: merge `OrgLLMConfig` columns with KV org keys (KV wins if present, else columns).
3. **Write**: dual-write to both the KV table and `OrgLLMConfig` columns.
4. After the frontend only uses the unified `/settings` API (Phase 6), drop `OrgLLMConfig` in Phase 7.
This is safer than a big-bang drop and keeps existing tests/clients working.

### 7.5 `REASONING_MODEL` addition to `config.py`

`REASONING_MODEL` is in `.env` / `.env.example` but **not declared** in `config.py` — it is a dead env var today. Add it in Phase 0:

```python
# in Settings class
REASONING_MODEL: Optional[str] = os.getenv("REASONING_MODEL") or None

@property
def effective_reasoning_model(self) -> str:
    return self.REASONING_MODEL or self.OPENAI_MODEL
```

This is a prerequisite for the registry entry and the org-override path.

### 7.6 Runtime reload

| Reload class | Behavior | Applies to |
|---|---|---|
| `next_request` | Resolved per request from the cache (30s TTL). Changes take effect within 30s, or immediately if the upsert invalidates the cache. | All retrieval/chunking/graphrag/agentic/memory/query/watcher/LLM-endpoint settings. |
| `restart` | Read once at startup into a process-global resource (ONNX session, Qdrant collection, tokenizer). Changing requires a restart. UI shows a "restart required" badge. | `DENSE_EMBEDDINGS_MODEL`, `DENSE_EMBEDDING_DIM`, `SPLADE_MODEL`, `RERANKER_MODEL`, `TOKENIZER_MODEL`, `MEMORY_ENABLED`, `MEMORY_EMBEDDING_MODEL`, `SANDBOX_BACKEND`. |

The cache invalidation hook lives in `settings_service.upsert_*` / `reset_*`: on any write, drop the affected `(org_id)` and the app-scope cache entries. For multi-worker setups, a Redis pub/sub invalidation channel is a later enhancement; for v1, the 30s TTL bounds staleness across workers.

### 7.7 Validation

`upsert_app_setting` / `upsert_org_setting` validate against the registry before writing:

- Type cast (str/int/float/bool/json) — reject on parse failure with `422`.
- `min`/`max` for numerics.
- `choices` for enums (`TOOL_CALL_MODE`).
- `requires_reindex=True` keys: not rejected, but the response includes a `warning` field and the endpoint logs a reindex warning. (Enforcement is a UI concern; the backend just warns.)
- `scope` check: org endpoint rejects `scope="app"` keys with `403`.

## 8. Frontend implementation

### 8.1 Super Admin Settings page

**New route:** `frontend/src/app/dashboard/admin/settings/page.tsx`

- Tabbed interface. Tabs derived from the registry `category` values: System, LLM & Models, Retrieval, Adaptive Retrieval, Reranker, Chunking, GraphRAG, Query Classification, Agentic, Memory, File Watcher.
- Each tab renders fields from `GET /api/admin/settings`, grouped by `category`.
- Field rendering by `type`: text input (str), number input (int/float, with min/max), toggle (bool), textarea (the `QUERY_CLASSIFIER_PROMPT` and any `text` type), JSON editor with preview (json), masked input (secret — not used in v1 since secrets stay in .env).
- "Restart required" badge next to `reload="restart"` fields.
- "Re-index required" warning banner on Chunking/GraphRAG tabs when the org has ingested docs (the app-settings page shows a generic warning; per-org reindex status is on the org page).
- Save: `PUT /api/admin/settings` with the changed fields only (dirty tracking).
- Reset per field: `DELETE /api/admin/settings/{key}` (reverts to .env default).
- Reset all: button that calls DELETE for every key in the tab.

**Sidebar:** add a "Settings" item to `AdminSidebar` `NAV_ITEMS` (icon: `Settings` from lucide). Visible to super_admin only — gate in the sidebar component by checking the current user's role (the page itself is guarded by `require_super_admin` on the API; the sidebar link is hidden for non-super-admins to avoid a dead link).

```
NAV_ITEMS (super_admin sees all; admin sees Orgs/Users/DataStores only):
  Orgs, Users, Data Stores, Settings (super_admin only)
```

### 8.2 Admin org settings

**Extend:** `frontend/src/app/dashboard/admin/orgs/page.tsx`

- The current "LLM Config" dialog becomes one tab inside a broader "Settings" dialog for the org.
- The dialog has the same tabs as the Super Admin page but only for `scope="org"` categories (LLM & Models, Retrieval, Adaptive Retrieval, Reranker, Chunking, GraphRAG, Query Classification, Agentic, Memory, File Watcher).
- Each field shows: current org value (if overridden), app default (greyed "inherited" placeholder), and an "override" toggle. When the toggle is off, the field is disabled and shows the app default. When on, the field is editable.
- `overridden` flag from `GET /api/admin/orgs/{org_id}/settings` drives the toggle state.
- Save: `PUT /api/admin/orgs/{org_id}/settings` with overridden fields only.
- Reset per field: `DELETE /api/admin/orgs/{org_id}/settings/{key}` (reverts to app default).
- Re-index warning on Chunking/GraphRAG tabs, shown when the org has documents (use the existing ingestion status endpoint to detect).

**Super admin viewing an org:** sees the same org-settings dialog but can also see/edit the app defaults (a link from the org dialog to the app Settings page for the relevant tab).

### 8.3 API client & types

**`frontend/src/lib/api.ts`** — add:

```typescript
export async function getAppSettings(): Promise<SettingItem[]>
export async function updateAppSettings(items: SettingUpdate[]): Promise<void>
export async function resetAppSetting(key: string): Promise<void>
export async function getOrgSettings(orgId: number): Promise<SettingItem[]>
export async function updateOrgSettings(orgId: number, items: SettingUpdate[]): Promise<void>
export async function resetOrgSetting(orgId: number, key: string): Promise<void>
```

**`frontend/src/lib/api-types.ts`** — add:

```typescript
export interface SettingItem {
  key: string;
  value: string | number | boolean | null;
  category: string;
  label: string;
  type: "str" | "int" | "float" | "bool" | "json" | "text";
  scope: "app" | "org";
  overridden?: boolean;       // org endpoint
  app_default?: any;          // org endpoint
  requires_reindex?: boolean;
  reload?: "next_request" | "restart";
  help?: string;
}
export interface SettingUpdate { key: string; value: any }
```

## 9. Migration strategy

### 9.1 Alembic migration — `backend/alembic/versions/XXXX_add_settings_table.py`

1. Create the `settings` table (§6).
2. **Seed app-level rows:** for every registry key, insert a `scope='app', org_id=NULL` row whose `value` is the current `.env` value (read from the `settings` singleton at migration time) — or the `config.py` default if the `.env` value equals the default. This makes the post-migration effective config identical to the pre-migration `.env`-driven config.
3. **Migrate `OrgLLMConfig`:** for each row, upsert `scope='org'` settings rows for `OPENAI_API_BASE` (from `api_base`), `OPENAI_MODEL` (from `model_name`), `QUERY_MODEL` (from `query_model`), skipping NULLs.
4. Do **not** drop `org_llm_configs` (kept for one release as read-only fallback).

### 9.2 Backfill / seed on startup

`backend/app/main.py` startup: the settings table is not seeded at startup. Registry defaults are used when no DB row exists. This handles fresh installs and registry additions in future releases without a new migration per key.

### 9.3 Phased rollout

| Phase | Scope | Risk |
|---|---|---|
| **0 — Align inventory** | Add `REASONING_MODEL` to `config.py`. Document canonical key names (use code names: `HYBRID_SPARSE_WEIGHT` not `HYBRID_QDRANT_SPARSE_WEIGHT`). Freeze registry list from §4. | Low — config.py addition only. |
| **1 — Schema + service** | Add `settings` table, `Setting` model, `settings_registry.py`, `settings_service.py`, `OrgSettings`. No service changes yet. Registry provides canonical defaults. | Low — additive only. |
| **2 — API** | Add `settings.py` router + schemas (including `/schema` endpoints). Super Admin and Admin endpoints live but read-only against the new table. Existing `OrgLLMConfig` endpoints still work (dual-write starts). | Low — no behavior change. |
| **3 — Wire LLM resolvers** | Rewrite `llm_factory.get_org_llm` and `chat_service.get_effective_llm_config` to read from `OrgSettings`. Migrate `OrgLLMConfig` data into KV. Dual-write begins. | Medium — LLM path is hot. Verify per-org model/base still resolves. |
| **4 — Wire retrieval + graph query + ingestion** | Thread `OrgSettings` into `retrieval.py` (query), `graph_service.py` (query path), `reranker.py`, `entity_extractor.py`, `historical_memory.py`. Ingestion paths (`document_processor`, `graph_service` extract) use **app-level** `OrgSettings` only. | High — largest diff. Do behind the `org_settings=None` default so unconverted paths keep working. |
| **5 — Wire agentic + context + memory** | Thread `OrgSettings` into the agent loop config, `rag_retrieve.py`, `token_budget.py`, compaction, context budgets. | Medium. |
| **6 — Frontend** | Super Admin Settings page (schema-driven forms) + Admin org-settings dialog. | Medium — UI only, backend already stable. |
| **7 — Hardening + deprecation** | Optional audit table. Load-test cache. Operator runbook (what needs re-ingest/reindex/restart). Drop `org_llm_configs` and its endpoints. Trim `.env.example` operational section to "install defaults & infra". | Low — cleanup. |

Each phase is independently shippable. Phases 0–2 change nothing about runtime behavior. Phases 3–5 can be done one service at a time with the `org_settings=None` fallback.

## 10. Parameter-by-parameter mapping (quick reference)

| .env variable | Tier 1 (.env) | Tier 2 (Super Admin) | Tier 3 (Admin org) | Category |
|---|---|---|---|---|
| `MYSQL_*`, `QDRANT_*`, `NEO4J_*`, `REDIS_*` | yes | — | — | (infra) |
| `SECRET_KEY` | yes | — | — | (secret) |
| `OPENAI_API_KEY` | yes | — | — | (secret) |
| `UPLOAD_DIR`, `FASTEMBED_CACHE_DIR`, `RERANKER_CACHE_DIR` | yes | — | — | (mount) |
| `TZ`, `LOG_LEVEL`, `TRUSTED_PROXIES`, `TIMEOUT_SECONDS` | yes | — | — | (host) |
| `ROOT_ORG`, `SUPERADMIN_*`, `COMPOSE_PROFILES` | yes | — | — | (init/docker) |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | yes | — | — | (security policy) |
| `TOKENIZER_MODEL`, `WATCHER_USE_INOTIFY`, `WATCH_DIR`, `SANDBOX_BACKEND` | yes | — | — | (host/mount/capability) |
| `PROCESSING_TIMEOUT_SILENCE_S` | fallback | default | override | Quality |
| `ANSWER_QUALITY_GRADING_ENABLED` | fallback | default | override | Quality |
| `DENSE_EMBEDDINGS_MODEL`, `DENSE_EMBEDDING_DIM` | — | app-only (restart) | — | LLM & Models |
| `SPLADE_MODEL` | — | app-only (restart) | — | LLM & Models |
| `RERANKER_MODEL` | — | app-only (restart) | — | Reranker |
| `SANDBOX_TIMEOUT_S` | — | app-only | — | System / Agent |
| `TOOL_CALL_MODE` | — | app-only | — | System / Agent |
| `QUERY_CLASSIFIER_PROMPT` | — | app-only | — | Query Classification |
| `COMPACTION_*`, `CONTEXT_*`, `HIGHLIGHTS_TOKEN_CAP`, `CONTEXT_COMPACTION_TRIGGER_RATIO` | fallback | default | override | Context / Memory |
| `MEMORY_ENABLED`, `MEMORY_EMBEDDING_MODEL` | — | app-only (restart) | — | System / Memory |
| `CHUNK_SIZE`, `OVERLAP_PERCENTAGE` | — | app-only (reindex) | — | Ingestion / Chunking |
| `GRAPHRAG_ENABLED`, `GRAPHRAG_MAX_CHUNKS`, `NEO4J_LLM_CONTEXT` | — | app-only (reindex) | — | Ingestion / GraphRAG |
| `WATCHER_ENABLED`, `WATCH_POLL_INTERVAL` | — | app-only (restart) | — | System / Watcher |
| `OPENAI_API_BASE` | fallback | default | override | LLM & Models |
| `OPENAI_MODEL` | fallback | default | override | LLM & Models |
| `OPENAI_MODEL_CONTEXT_SIZE` | fallback | default | override | LLM & Models |
| `QUERY_MODEL` | fallback | default | override | LLM & Models |
| `REASONING_MODEL` | fallback | default | override | LLM & Models |
| `VISION_MODEL` | fallback | default | override (query only) | LLM & Models |
| `OPENAI_VISION_API_BASE` | fallback | default | override (query only) | LLM & Models |
| `GRAPHRAG_LLM` | fallback | default | override (query only) | GraphRAG |
| `RETRIEVAL_TOP_K`, `RETRIEVAL_MIN_RRF_SCORE` | fallback | default | override | Retrieval |
| `DENSE_MIN_SCORE`, `SPARSE_MIN_SCORE`, `EXACT_MIN_SCORE` | fallback | default | override | Retrieval |
| `HYBRID_DENSE_WEIGHT`, `HYBRID_SPARSE_WEIGHT`, `HYBRID_EXACT_WEIGHT` | fallback | default | override | Retrieval |
| `RETRIEVAL_DENSE_ENABLED`, `RETRIEVAL_SPARSE_ENABLED`, `RETRIEVAL_EXACT_ENABLED` | fallback | default | override | Retrieval |
| `RETRIEVAL_GRAPH_ENABLED` | fallback | default | override | Retrieval |
| `RETRIEVAL_CONFIG_PRESETS` | fallback | default | override | Retrieval |
| `ENTITY_AWARE_ENABLED`, `ENTITY_BOOST_FACTOR` | fallback | default | override | Retrieval |
| `ADAPTIVE_RETRIEVAL_ENABLED`, `ADAPTIVE_RETRIEVAL_THRESHOLD`, `ADAPTIVE_RETRIEVAL_RERANKER_THRESHOLD` | fallback | default | override | Adaptive Retrieval |
| `RETRIEVAL_RELAX_LEVEL2_RERANKER_THRESHOLD` | fallback | default | override | Adaptive Retrieval |
| `RERANKER_ENABLED`, `RERANKER_SCORE_THRESHOLD` | fallback | default | override | Reranker |
| `GRAPHRAG_RETRIEVAL_HOPS`, `GRAPHRAG_RETRIEVAL_LIMIT`, `GRAPHRAG_ENTITY_FANOUT_CAP` | fallback | default | override | GraphRAG (query) |
| `QUERY_CLASSIFIER_ENABLED` | fallback | default | override | Query Classification |
| `TOOL_CALLING_ENABLED`, `MAX_TOOL_ITERATIONS` | fallback | default | override | Agentic |
| `SYNTHESIS_MODE_ENABLED` | fallback | default | override | Agentic |
| `AGENT_MAX_*`, `AGENT_REFLECT_EVERY`, `AGENT_RETRY_BACKOFF_BASE`, `AGENT_MAX_CLARIFICATIONS`, `AGENT_HISTORY_PAIRS`, `AGENT_MAX_WALL_SECONDS` | fallback | default | override | Agentic |
| `HISTORICAL_MEMORY_ENABLED`, `HISTORICAL_MEMORY_TOP_K`, `HISTORICAL_MEMORY_SCORE_THRESHOLD` | fallback | default | override | Memory |

## 11. Risk analysis

| Risk | Impact | Mitigation |
|---|---|---|
| **Service reads `settings` singleton after migration** | Per-org overrides silently ignored | Grep for `settings\.` in `app/services` after each phase; convert stragglers. The `org_settings=None` default prevents breakage but must be removed before declaring done. |
| **Cache staleness across workers** | Setting change takes up to 30s to propagate to other workers | Acceptable for v1. Document. Redis pub/sub invalidation as a follow-up. |
| **Chunking/graphrag changes after ingestion** | Inconsistent chunk/entity sizes across the knowledge base | UI warning + backend `requires_reindex` flag in response. No hard enforcement (same as today's `.env` warning). |
| **Embedding model/dim changed at runtime** | Qdrant collection dimension mismatch → ingestion failure | These are `scope="app", reload="restart"`. UI shows "restart required" and the setting does not take effect until restart. The startup seed re-validates dim against existing collections. |
| **Admin edits settings outside their scope** | Cross-org data leak | Existing `get_admin_org_ids` scope check on every org endpoint. App endpoints use `require_super_admin`. |
| **`OrgLLMConfig` deprecation breaks old clients** | Frontend calls removed endpoints | Phase 7 only after the frontend stops calling them. Dual-write keeps data consistent during the compatibility window. |
| **Concurrent edits (two admins same org)** | Lost update | Last-write-wins with `updated_at`. Optimistic locking (version column) is a later enhancement if needed. |
| **Config drift between .env and DB** | Confusion about the active value | `GET /api/admin/settings/effective` returns the fully resolved snapshot. UI shows provenance badge ("Database" vs "Install default") per field. |
| **Org override of ingestion setting** | Inconsistent indexes for shared DataStores | Ingestion settings are `scope="app"` — org endpoints reject them with 403. Ingestion code uses app-level `OrgSettings` only. |
| **Settings service failure** | App unusable | `get_setting()` catches DB exceptions and falls back to the registry default. No feature flag needed. |

### 11.1 Testing plan

### Backend (`docker exec rag-web-ui-backend-1 pytest`)

| Test | Assert |
|---|---|
| Precedence | org override wins; delete org restores app; delete app restores install default |
| Scope ACL | admin cannot PUT app settings (403); admin cannot edit foreign org (403) |
| Super admin | can edit app + any org |
| LLM façade | old llm-config endpoints dual-write and read consistent with unified settings |
| Retrieval wiring | monkeypatch DB override of `RETRIEVAL_TOP_K` changes value used in search |
| Ingestion isolation | org override of `CHUNK_SIZE` (rejected: app-only key) does NOT change chunk_size used in processor |
| Validation | bad types/ranges → 422; org endpoint with app-only key → 403 |
| Import seed | non-default `.env` values imported once; idempotent |

### Frontend

| Test | Assert |
|---|---|
| Role gate | non-super-admin cannot open App Settings page |
| Override UX | toggle off sends null / DELETE; field shows app default |
| Schema render | all categories mount from `/schema` endpoint |

### Manual / E2E

1. Super admin sets app `RETRIEVAL_TOP_K=10`.
2. Admin sets org A to `30`.
3. Query as user in A → observes 30; user in B → 10.
4. Change chunk size (app-only) → warning; new uploads use new size; old docs unchanged.
5. Attempt embeddings dim change with data → blocked or strong confirm.
6. Attempt org override of `CHUNK_SIZE` → 403.

## 12. File changes summary

### Backend

| File | Change |
|---|---|
| `backend/app/core/settings_registry.py` | **New** — the registry (§3, §5). |
| `backend/app/services/settings_service.py` | **New** — resolution, CRUD, cache, `OrgSettings` (§7.1, §7.2). |
| `backend/app/models/setting.py` | **New** — `Setting` model (§6). |
| `backend/app/models/__init__.py` | Export `Setting`. |
| `backend/app/models/organisation.py` | No change (watch_dir is legacy; real paths on DataStore). |
| `backend/app/schemas/setting.py` | **New** — Pydantic schemas (§7.3). |
| `backend/app/api/api_v1/settings.py` | **New** — app + org settings routers (§7.3). |
| `backend/app/api/api_v1/api.py` | Include the settings routers. |
| `backend/app/services/agentic_rag/llm_factory.py` | Read LLM config from `OrgSettings` (Phase 3). |
| `backend/app/services/chat/chat_service.py` | `get_effective_llm_config` from `OrgSettings`; query classifier from `org_settings` (Phase 3, 5). |
| `backend/app/services/retrieval/retrieval.py` | Accept `org_settings` (Phase 4). |
| `backend/app/services/graph/graph_service.py` | Accept `org_settings` (Phase 4). |
| `backend/app/services/graph/entity_extractor.py` | Accept `org_settings` (Phase 4). |
| `backend/app/services/ingestion/document_processor.py` | Resolve chunking/graphrag from `org_settings` (Phase 4). |
| `backend/app/services/reranker.py` | Accept `org_settings` (Phase 4). |
| `backend/app/services/chat/historical_memory.py` | Accept `org_settings` (Phase 4). |
| `backend/app/services/agentic_rag/tools/rag_retrieve.py` | Read adaptive retrieval from `ToolContext.org_settings` (Phase 5). |
| `backend/app/services/agentic_rag/` (agent config) | Read agent loop settings from `org_settings` (Phase 5). |
| `backend/app/services/confidence.py` | Accept `org_settings` (Phase 4). |
| `backend/app/main.py` | Startup seed: ensure app rows exist for every registry key (§9.2). |
| `backend/alembic/versions/XXXX_add_settings_table.py` | **New** migration: create table, seed app rows, migrate `OrgLLMConfig` (§9.1). |
| `backend/app/core/config.py` | Add `REASONING_MODEL` field + `effective_reasoning_model` property (Phase 0). No field removals in v1. The `settings` singleton remains the Tier 1 fallback. (Future: trim to infra-only once all reads go through `OrgSettings`.) |

### Frontend

| File | Change |
|---|---|
| `frontend/src/app/dashboard/admin/settings/page.tsx` | **New** — Super Admin settings page (§8.1). |
| `frontend/src/components/admin/admin-sidebar.tsx` | Add "Settings" nav item, super_admin-gated (§8.1). |
| `frontend/src/app/dashboard/admin/orgs/page.tsx` | Replace "LLM Config" dialog with tabbed "Settings" dialog (§8.2). |
| `frontend/src/lib/api.ts` | Add settings API functions (§8.3). |
| `frontend/src/lib/api-types.ts` | Add `SettingItem`, `SettingUpdate` (§8.3). |

### Database

| Object | Change |
|---|---|
| `settings` (new table) | Generic key/scope/org_id/value store (§6). |
| `organisations.watch_dir` | Already in DB; declare on the model (§7.5). |
| `org_llm_configs` | Data migrated to `settings`; table dropped in Phase 7. |

## 13. Success criteria

1. A fresh boot with empty settings tables behaves exactly as today's `.env`/`config.py`.
2. Super Admin can change app default `HYBRID_DENSE_WEIGHT` and all orgs without overrides pick it up within cache TTL.
3. Org Admin can override `model_name` + `RETRIEVAL_TOP_K` only for their org tree; other orgs unchanged.
4. Org Admin cannot change embedding dim, chunk size, watcher, or secrets via API (403/422).
5. Ingestion of a shared DataStore always uses app ingestion settings, regardless of org overrides.
6. Existing LLM Config UI/API continues to work through the façade until removed.
7. `.env.example` documents only deployment/infrastructure settings; operational tuning docs point at Admin UIs.
8. Pytest coverage for resolution, ACL, ingestion isolation, and at least one wired retrieval path.

## 14. Operator runbook (what needs restart/reindex/reingest)

| Change | Required action |
|---|---|
| `DENSE_EMBEDDINGS_MODEL` / `DENSE_EMBEDDING_DIM` / `SPLADE_MODEL` | Rebuild vectors / re-ingest all content + restart |
| `CHUNK_SIZE` / `OVERLAP_PERCENTAGE` | Re-ingest for consistency |
| `GRAPHRAG_ENABLED` / `GRAPHRAG_MAX_CHUNKS` / `NEO4J_LLM_CONTEXT` | Re-run graph extraction for consistency |
| `RERANKER_MODEL` | Restart backend workers to reload ONNX |
| `MEMORY_ENABLED` / `MEMORY_EMBEDDING_MODEL` | Restart backend |
| `WATCHER_ENABLED` / `WATCH_POLL_INTERVAL` | Restart backend (watcher process) |
| All other settings | Take effect within cache TTL (30s); no restart needed |

## 15. What this plan does NOT do

- Does not move secrets (`SECRET_KEY`, `OPENAI_API_KEY`) or security policy (`ACCESS_TOKEN_EXPIRE_MINUTES`) into the DB or UI. They stay in `.env`/secrets manager.
- Does not make embeddings/SPLADE/reranker-model per-org. Those are process-global resources; per-org would require per-org Qdrant collections and per-org ONNX sessions. They are app-only Super Admin settings with `reload="restart"`.
- Does not make ingestion settings (chunking, graph extraction, OCR) per-org. DataStores are shared across orgs; per-org ingestion would produce inconsistent indexes. These are app-only Super Admin settings. Per-DataStore overrides are a future enhancement.
- Does not make watcher process knobs per-org. One watcher process watches all DataStore folders. `WATCHER_USE_INOTIFY` and `WATCH_DIR` stay in `.env` (host capability / legacy fallback).
- Does not add optimistic locking for concurrent settings edits. Last-write-wins for v1.
- Does not add Redis pub/sub cache invalidation. 30s TTL for v1.
- Does not remove the `settings` singleton in `config.py`. It remains the Tier 1 fallback and the source of `.env` values for seeding. Trimming it to infra-only is a future cleanup after all service reads go through `OrgSettings`.
- Does not change the existing `/api/admin/orgs/{org_id}/llm-config` endpoints in Phase 3. They are kept as a façade over the unified settings service until Phase 7; dual-write keeps them consistent.
- Does not add per-org infrastructure (separate MySQL/Qdrant/Neo4j/Redis clusters).
