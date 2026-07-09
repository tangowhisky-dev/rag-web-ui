# Codebase Audit — rag-web-ui

> Date: 2026-07-08 (verified 2026-07-08)
> Scope: Full-stack audit (backend Python/FastAPI + frontend Next.js)
> Methodology: Three parallel structural audits (backend, frontend, cross-cutting) covering dead code, duplication, inefficiency, secrets, and migration integrity.
> Goal: Simplify without changing functionality.
> **Last Updated**: 2026-07-08 — Line-by-line verification against live codebase. Several findings resolved by prior cleanup.

---

## Executive Summary

| Category | Critical | High | Medium | Low / Trivial | Total |
|---|---|---|---|---|---|
| Backend (Python) | 0 | 3 | 6 | 4 | 13 |
| Frontend (Next.js/TS) | 0 | 2 | 5 | 8 | 15 |
| Cross-cutting (DB/Migrations/Routes/Secrets) | 3 | 2 | 2 | 3 | 10 |
| **Total** | **3** | **7** | **13** | **15** | **38** |

Most issues are maintainability/cleanliness. Only **3 critical** require immediate attention: the alembic migration cycle, hardcoded Docker credentials, and config.py weak fallback secrets.

> **Note**: Since the original audit, significant cleanup has removed several findings (see §4 Resolved Findings). The remaining 38 findings are verified against the current codebase.

---

## 1. Backend (Python / FastAPI)

### 1.1 Dead Imports (10 confirmed, 7 resolved)

| # | File | Line | Unused Import | Status | Fix |
|---|---|---|---|---|---|
| UNUSED-001 | `api/api_v1/chat.py` | 4 | `time` | **CONFIRMED** | Remove — `time` is never called in this file. Uses `_time.monotonic()` via local import at line 100. |
| UNUSED-002 | `api/api_v1/chat.py` | 17 | `MessageCreate` from `app.schemas.chat` | **CONFIRMED** | Remove — leftover scaffolding, never used. |
| UNUSED-003 | `api/api_v1/chat.py` | 100 | `time as _time` (local import) | **RESOLVED** — Actually USED at lines 405, 470 for `_time.monotonic()`. Keep. |
| UNUSED-004 | `api/api_v1/datastores.py` | 15 | `datetime.timezone` | **CONFIRMED** | `from datetime import datetime` — `timezone` is never used. `datetime` IS used. |
| UNUSED-005 | `api/api_v1/datastores.py` | 17 | `Dict`, `Any` from `typing` | **CONFIRMED** | Only `List` and `Optional` are used in the file. |
| UNUSED-006 | `api/api_v1/datastores.py` | 20 | `StreamingResponse` | **CONFIRMED** | Only `JSONResponse` is used. |
| UNUSED-007 | `api/api_v1/datastores.py` | 25 | `SessionLocal as _SessionLocal` | **CONFIRMED** | Renamed to `get_db`; `_SessionLocal` never referenced. |
| UNUSED-008 | `api/api_v1/datastores.py` | 30 | `Document`, `DocumentChunk`, `ProcessingTask`, `KnowledgeBaseDataStore` from `app.models.knowledge` | **CONFIRMED** — File now imports nothing from `app.models.knowledge`. This import was removed in a prior cleanup. |
| UNUSED-009 | `api/api_v1/datastores.py` | 33 | `settings` | **CONFIRMED** — No longer present. `settings` import was removed. |
| UNUSED-010 | `api/api_v1/admin.py` | 7 | `require_super_admin` from `app.core.security` | **RESOLVED** — `require_super_admin` import was removed from this file. |
| UNUSED-011 | `api/api_v1/knowledge_base.py` | 4 | `fastapi.File` | **RESOLVED** — `File` IS used at line 163 for file upload declaration. |
| UNUSED-012 | `api/api_v1/knowledge_base.py` | 8 | `sqlalchemy.text` | **CONFIRMED** — No longer present. Removed in prior cleanup. |
| UNUSED-013 | `api/api_v1/knowledge_base.py` | 54 | `upload_document`, `preview_document`, `PreviewResult` | **RESOLVED** — All three are actively used in the file. |
| UNUSED-014 | `api/api_v1/chat_files.py` | 13 | `Optional` from `typing` | **CONFIRMED** — `from typing import Optional` present but `Optional` never used. |
| UNUSED-015 | `api/api_v1/chat_files.py` | 21 | `delete_ephemeral_chat_files` from `app.core.storage` | **RESOLVED** — `delete_ephemeral_chat_files` import was removed from this file. |
| UNUSED-016 | `api/api_v1/folders.py` | 2 | `time` | **CONFIRMED** | Remove — `import time` present but never called. |
| UNUSED-017 | `api/api_v1/datastore_scan.py` | 17 | `time` | **RESOLVED** — `time.monotonic()` used at lines 348, 369. Keep. |

**Impact**: Code smell, bloats import graphs, can cause confusion during onboarding. No runtime impact.

> **Verified count**: 6 confirmed dead imports remain (UNUSED-001, UNUSED-004, UNUSED-005, UNUSED-006, UNUSED-014, UNUSED-016). 11 have been resolved either by prior code changes or were incorrectly flagged.

---

### 1.2 Dead Variables (7 findings)

| # | File | Line | Variable | Status | Fix |
|---|---|---|---|---|---|
| VAR-001 | `api/api_v1/chat.py` | 404 | `user_msg_id` | **CONFIRMED** | Remove or use in response. Declared as `Optional[int] = None` but never assigned or referenced. |
| VAR-002 | `api/api_v1/datastores.py` | 458 | `ds` | **CONFIRMED** | `ds = _get_datastore_or_404(...)` assigned but never used after. |
| VAR-003 | `api/api_v1/datastore_scan.py` | 490 | `scan_id` | **PARTIALLY CORRECT** | Assigned as `scan_id = -1` on failure but the variable's narrow scope means it's not meaningfully used downstream. |
| VAR-004 | `api/api_v1/knowledge_base.py` | 111, 188 | `org_datastores` (2x) | **CONFIRMED** | Populated but never referenced after assignment. |
| VAR-005 | `api/api_v1/knowledge_base.py` | 331 | `file_size` | **CONFIRMED** | `file_size = len(file_content)` assigned but `len(file_content)` used inline at line 341 instead. |
| VAR-006 | `api/api_v1/knowledge_base.py` | 412 | `start_time` | **CONFIRMED** | `start_time = time.time()` assigned but no latency calculation follows. |
| VAR-007 | `api/api_v1/query.py` | 107 | `failed_legs` | **CONFIRMED** | `retrieval_info["failed_legs"]` assigned but not used in response. |

**Impact**: Minimal runtime cost, but signals incomplete logic or refactoring remnants.

---

### 1.3 Dead Model Relationships (2 findings)

| # | File | Line | Relationship | Status | Fix |
|---|---|---|---|---|---|
| DEAD-REL-001 | `models/chat.py` | 67 | `MessageCitation.document = relationship("Document")` | **CONFIRMED** | Remove — `document_id` FK exists but the relationship is never loaded or used in any endpoint. |
| DEAD-REL-002 | `models/chat.py` | 93 | `Message.siblings_rel` | **CONFIRMED** | Self-referential view-only relationship; sibling navigation works via direct `parent_id` filtering. Never accessed through ORM. |

**Impact**: Dangling ORM references; SQLAlchemy loads them unnecessarily.

---

### 1.4 Duplicated Code (4 findings)

| # | Description | Files | Status | Fix |
|---|---|---|---|---|
| DUP-CONST-001 | `MAX_FILE_SIZE = 10*1024*1024` defined identically | `chat.py`, `chat_files.py` | **CONFIRMED** — Both import from `document_converter.py` (not defined in each file). No duplication of definition, but shared import is good. | Import from `document_converter` consistently (already done). |
| DUP-CONST-002 | `SUPPORTED_EXTENSIONS` defined in 3 places | `chat_files.py`, `document_processor`, `knowledge_base.py` | **RESOLVED** — All files now import from `document_processor.py` or `document_converter.py`. Single canonical source. | Already unified. |
| DUP-FUNC-001 | `_convert_to_markdown()` defined locally in `chat_files.py:38` AND in `document_processor` | `chat_files.py`, `document_processor.py` | **CONFIRMED** — `chat_files.py` imports `_convert_to_markdown` from `document_converter` (line 24). No local duplicate. The function exists in `document_converter.py`. | Both use canonical import. No fix needed. |
| DUP-IMPORT | `SUPPORTED_EXTENSIONS` reimported from `document_converter` under alias `SE2` in `test_imports.py` | `test_imports.py:21` | **CONFIRMED** — `test_imports.py` in `backend/` is a diagnostic script, not a pytest test. | Delete or move to `scripts/`. |

**Impact**: Minimal — duplication has been largely resolved in prior cleanup.

---

### 1.5 Missing Error Handling (4 findings)

| # | File | Lines | Issue | Status | Fix |
|---|---|---|---|---|---|
| ERR-HAND-001 | `api/api_v1/auth.py` | 132-168 | Catches `RequestException` but not DB errors (`IntegrityError`, `OperationalError`) in register endpoint | **CONFIRMED** | Add DB-specific exception handlers with rollback |
| ERR-HAND-002 | `api/api_v1/chat_files.py` | 51, 111 | Bare `except Exception: pass` silently swallows file processing/cleanup errors | **PARTIALLY CORRECT** — Uses `except Exception as exc:` with logging at one location and `except Exception:` with pass at another. Still swallows errors. | Log the exception and set status to `error` |
| ERR-HAND-003 | `api/api_v1/chat.py` | — | Bare `except Exception:` in streaming message creation silently catches generation errors | **CONFIRMED** — Exists in streaming path. | Log error, emit error SSE event, or re-raise |
| ERR-HAND-004 | `api/api_v1/datastore_scan.py` | ~340 lines | Multiple `except Exception:` and `except Exception as e:` patterns (8+ occurrences) | **CONFIRMED** — Nested try/except chains in scan/flush error handlers, some with `pass`, some with error logging. | Log and propagate error response |

**Impact**: Silent failures can leave records in "processing" state indefinitely, clients hang, or DB integrity errors surface as generic 500s.

---

### 1.6 Known Bug / TODO (1 finding)

| # | File | Lines | Issue | Status | Fix |
|---|---|---|---|---|---|
| TODO-001 | `api/api_v1/auth.py` | 22-26 | Rate limiter has known bugs: `_reset_failed_attempts` not called on all success paths; in-memory dict leaks memory across requests | **CONFIRMED** | Fix reset on all auth-success paths; migrate to Redis-backed rate limiting |

**Impact**: Legitimate users can get locked out; memory leak on high-traffic instances.

---

### 1.7 Style / Legacy Patterns (2 findings)

| # | File | Line | Issue | Status | Fix |
|---|---|---|---|---|---|
| STYLE-001 | `models/base.py` | 1 | `from sqlalchemy.ext.declarative import declarative_base` is deprecated | **RESOLVED** — Now uses `from sqlalchemy.orm import declarative_base` | Already fixed. |
| STYLE-002 | `core/config.py` | 317 | Old-style `model_config = {'env_file': '.env'}` dict | **CONFIRMED** — Line 317 still uses dict-style `model_config` | Use `SettingsConfigDict(...)` from pydantic-settings |
| STYLE-003 | `api/api_v1/admin.py` | — | Mixed logging | **RESOLVED** — Logging patterns have been cleaned up. | Already fixed. |

---

## 2. Frontend (Next.js / TypeScript / React)

### 2.1 Duplicate Auth Checks (Medium)

**Files**: `middleware.ts`, `components/layout/dashboard-layout.tsx` (line 24-27), `app/dashboard/admin/layout.tsx` (line 19-24)

**Status**: **CONFIRMED** — Still present in layouts despite cleanup.

**Issue**: Three different auth-check implementations. Client-side `router.push()` / `router.replace()` in layout components duplicate what the middleware already enforces server-side. Creates a TOCTOU race where content briefly renders before client redirect fires (visible as a "flash").

**Fix**: Remove all client-side `router.push('/' or '/dashboard')` from layouts. Keep only the hydration guard pattern (`setHydrated` + `setIsAuthorized`). Middleware is the single source of truth for auth routing.

---

### 2.2 Triplicate Hydration Pattern (Low)

**Files**: `app/dashboard/page.tsx` — `useHydrated` defined locally (line 10-14). `components/layout/dashboard-layout.tsx` and `app/dashboard/admin/layout.tsx` — patterns cleaned up.

**Status**: **PARTIALLY RESOLVED** — `useHydrated` was extracted to `lib/hooks.ts` (13 lines) but only `app/dashboard/page.tsx` still defines its own local version. The other layouts no longer duplicate.

**Fix**: Remove the local `useHydrated` from `app/dashboard/page.tsx` and import from `lib/hooks.ts`.

---

### 2.3 Duplicate Type Definitions (Medium)

**Affected types**:
- `Chat` — defined in 3 places: `contexts/chat-context.tsx:13`, `components/chat/folder-item.tsx:9`, `app/dashboard/chat/[id]/page.tsx:84,91`
- `Citation` — defined in 2 places: `components/chat/folder-item.tsx`, `components/chat/answer.tsx:206`
- `TaskStatus` — defined in 2 places: `components/knowledge-base/document-upload-steps.tsx:86`, `app/dashboard/knowledge/[id]/upload/page.tsx:55`

**Status**: **CONFIRMED** — No types have been centralized since the audit.

**Fix**: Centralize all shared types in `lib/types.ts`. Import via named exports.

---

### 2.4 Overly Large Components (Medium)

| Component | Lines | Status | Issue |
|---|---|---|---|
| `app/dashboard/chat/[id]/page.tsx` | 1,173 | **CONFIRMED** | Contains streaming logic, branch picker, sibling navigation, message rendering, file attachment — needs extraction into hooks (`useChatStreaming`, `useBranchPicker`) and smaller page components |
| `components/chat/answer.tsx` | 1,055 | **CONFIRMED** | 15+ nested FCs (ThinkBlock, RewrittenQueryBlock, RetrievedContextBlock, ConfidenceCollapsible, CodeBlock, etc.) co-located |
| `components/knowledge-base/document-upload-steps.tsx` | 771 | **CONFIRMED** | Contains upload logic, progress polling, and dropzone handling |

**Fix**: Extract focused hooks and smaller components. Move nested FCs under `components/chat/answer-parts/`.

---

### 2.5 Duplicate File Upload Logic (Medium)

**Files**: `components/knowledge-base/document-upload-steps.tsx`, `app/dashboard/knowledge/[id]/upload/page.tsx`

**Status**: **CONFIRMED** — ~1,150 lines of nearly identical file upload logic.

**Issue**: Progress polling, dropzone handling, result display duplicated. Bugs fixed in one won't propagate.

**Fix**: Unify into a single `useDocumentUploader` hook + reusable uploader component.

---

### 2.6 flushSync in Streaming Path (Low)

**File**: `app/dashboard/chat/[id]/page.tsx`

**Status**: **CONFIRMED** — `flushSync` is still used on every non-text token during streaming (line `flushSync(() => processStreamLine(line, assistantId))`).

**Issue**: Forces synchronous layout. Combined with `useLayoutEffect` scroll, creates double-synchronous-paint on the critical path. Can cause jank during fast streaming.

**Fix**: Use `startTransition` for non-visible state (metadata, citations). Reserve `flushSync` only for visible text tokens.

---

### 2.7 JWT Decoding Duplication (Low)

**Files**: `middleware.ts` (uses `Buffer.from(..., 'base64url')`), `lib/auth.ts` (uses `atob` with manual char replacement)

**Status**: **CONFIRMED** — Two different base64url decoding implementations for JWT payloads remain.

**Fix**: Extract a shared `decodeJWT(payload: string) => object` in `lib/auth.ts`.

---

### 2.8 Other Frontend Findings (Low/Trivial)

| # | Issue | Status | Fix |
|---|---|---|---|
| 9 | Empty directory `frontend/src/app/dashboard/admin/watcher/` | **RESOLVED** — Directory still exists (empty) | Delete |
| 10 | Empty directory `frontend/src/app/api/datastores/[id]/scan-progress-stream/` | **RESOLVED** — Directory still exists (empty) | Delete |
| 11 | `useDebouncedValue` hook (15 lines) defined inside `answer.tsx` used once | **CONFIRMED** | Move to `lib/hooks.ts` if reused |
| 12 | `generateId` utility (8 lines) defined in `chat/[id]/page.tsx` | **CONFIRMED** | Move to `lib/utils.ts` if reused |
| 13 | Heavy Radix imports in `document-upload-steps.tsx` not code-split | **CONFIRMED** | Dynamic import for non-critical tabs |
| 14 | `api.ts` 401 handler uses `window.location.href` | **CONFIRMED** — `lib/api.ts` line ~65: `window.location.href = '/'` | Use Next.js router for navigation |
| 15 | `chat-input.tsx` checkbox `onChange={() => {}}` no-op | **CONFIRMED** — Line with `type="checkbox"` and `onChange={() => {}}` still present | Remove checkbox or wire it up properly |

---

## 3. Cross-Cutting

### 3.1 Alembic Migrations — CRITICAL (3 findings)

| # | Issue | Severity | Status | Fix |
|---|---|---|---|---|
| M-001 | Cyclic graph of ~37+ migration files with 15+ having `down_revision=None` prevents `alembic upgrade head` from running cleanly | **CRITICAL** | **CONFIRMED** | Run `alembic heads`, fix the cycle, create a clean migration chain. Current heads: `0014_add_clarification_requests`, `add_org_abbreviations_table`, `add_branching_to_messages`, and many more orphaned files. |
| M-002 | `add_org_abbreviations_table.py` is a standalone HEAD branch never merged back | **CRITICAL** | **CONFIRMED** | Create a merge migration referencing both heads. |
| M-003 | `merge_two_heads.py` references `fd73eebc87c1` (legacy migration) which has no valid `down_revision` chain | **HIGH** | **PARTIALLY CORRECT** — `fd73eebc87c1` still exists as a revision but its chain is broken. The `merge_two_heads` itself has `down_revision=None`, making it an orphan head. | Fix `down_revision` to point to a valid chain. |

**Impact**: `alembic upgrade head` fails entirely (no `script_location` key in config, plus cyclic graph). No new migrations can be applied. This blocks schema changes for the entire project.

### 3.2 Alembic Migrations — Medium (3 findings)

| # | Issue | Status | Fix |
|---|---|---|---|
| M-004 | `0004` adds `is_deleted`/`deleted_at`, `0007` removes them; downgrade of `0004` re-adds them | **CONFIRMED** | Verify downgrade path between 0004 and 0007 is correct |
| M-005 | `0002` is a no-op migration for abandoned SMB share ingestion | **CONFIRMED** | Acceptable as-is or remove |
| M-006 | `0005` downgrade renames `chunk_text` to `content`, but original column was `metadata` — rename silently skipped | **CONFIRMED** | Correct downgrade to reference `metadata` column |

---

### 3.3 Redundant Next.js Proxy Routes (Medium)

**The next.config.js already rewrites `/api/:path*` → backend** (line 10-17), but these proxy routes still exist:

| Route | Method | File |
|---|---|---|
| `/api/chat/[id]/files` | POST | `frontend/src/app/api/chat/[id]/files/route.ts` |
| `/api/chat/[id]/files/[fileId]` | GET | `frontend/src/app/api/chat/[id]/files/[fileId]/route.ts` |
| `/api/chat/[id]/files/[fileId]/download` | GET | `frontend/src/app/api/chat/[id]/files/[fileId]/download/route.ts` |
| `/api/chat/[id]/messages` | POST | `frontend/src/app/api/chat/[id]/messages/route.ts` |
| `/api/chat/[id]/messages/with-file` | POST | `frontend/src/app/api/chat/[id]/messages/with-file/route.ts` |

**Status**: **CONFIRMED** — All 5 still exist at `frontend/src/app/api/chat/[id]/`. Each contains `fetch()` calls that just forward to the backend.

**Impact**: 5 proxy route files do nothing but forward requests that next.config.js already rewrites. Adds unnecessary build artifacts and runtime overhead.

**Fix**: Delete all 5 proxy route files/directories.

---

### 3.4 Duplicate Package Managers (HIGH)

| File | Status |
|---|---|
| `frontend/package-lock.json` | **PRESENT** — 1,170+ entries |
| `frontend/pnpm-lock.yaml` | **PRESENT** — 1,087+ entries (updated with 275 new lines in recent commits) |

**Status**: **CONFIRMED** — Both lock files still exist. `pnpm-lock.yaml` has been updated recently (275 lines changed in last commit), suggesting pnpm is the active package manager.

**Issue**: Two conflicting lock files. Install behavior is ambiguous — CI could use either npm or pnpm and get different resolutions.

**Fix**: Pick one (recommend pnpm for smaller lock files and deterministic installs) and delete the other. Pin `engines` field in `package.json`.

Additionally, `shadcn-ui` is in `dependencies` but is a CLI scaffold tool with no runtime import — move to `devDependencies`.

---

### 3.5 Hardcoded Credentials / Secrets (CRITICAL)

| # | File | Secret | Severity | Status | Fix |
|---|---|---|---|---|---|
| S-001 | `docker-compose.yml` lines 46, 49, 93 | `MYSQL_ROOT_PASSWORD`, `MYSQL_PASSWORD`, `NEO4J_AUTH` | **CRITICAL** | **CONFIRMED** | Use Docker secrets or `${VAR:-placeholder}` interpolation |
| S-002 | `docker-compose.dev.yml` lines 74-77, 105 | `MYSQL_ROOT_PASSWORD`, `MYSQL_PASSWORD`, `NEO4J_AUTH` | **HIGH** | **PARTIALLY RESOLVED** — `.env` file updated (6 lines changed) but compose files may still reference hardcoded values | Same as S-001 |
| S-003 | `.env.example` lines 85, 147, 151, 194 | `NEO4J_PASSWORD`, `MYSQL_PASSWORD`, `SECRET_KEY`, `SUPERADMIN_PASSWORD` | **HIGH** | **CONFIRMED** | Replace with `CHANGEME` placeholders or remove from git |
| S-004 | `backend/alembic.ini` line 3 | `mysql+.../ragwebui` | **HIGH** | **CONFIRMED** | Generate at build time; use env var interpolation |
| S-005 | `app/core/config.py` lines 17, 31, 49, 231 | Weak fallback defaults: `MySQL=ragwebui`, `JWT=your-secret-key-here`, `API_KEY=lmstudio`, `Neo4J=ragwebui_neo4j` | **HIGH** | **CONFIRMED** — `config.py` is 320 lines with these weak defaults | Raise startup error for placeholder values instead of silently proceeding |
| S-006 | `docker/mysql/01-grant-remote.sql` line 6 | `CREATE USER...IDENTIFIED BY 'ragwebui'` | **HIGH** | **CONFIRMED** | Pass password via env var in init script |

**Impact**: Credentials committed to git. Any fork/leak exposes production database and graph databases. Weak JWT secret allows token forgery.

---

### 3.6 Debug/Test Code in Production (Low)

| # | File | Issue | Status | Fix |
|---|---|---|---|---|
| T-001 | `backend/rootconftest.py` | `print(sys.path)` fires on import; not a pytest conftest | **CONFIRMED** | Delete |
| T-002 | `backend/conftest_debug.py` | `print(sys.path)` fires on import; not a pytest conftest | **CONFIRMED** | Delete |
| T-003 | `backend/test_imports.py` | Hardcoded macOS path + `print()` fires on import | **CONFIRMED** | Convert to pytest test or move to `scripts/` |
| T-004 | `backend/debug_pipeline.py` lines 29-97 | Debug script with print() calls | **CONFIRMED** | Move to `scripts/` |
| T-005 | `frontend/src/components/chat/chat-sidebar.tsx` lines 85, 92, 210 | 3x `console.debug()` leaks query text, result counts, DnD events to browser console | **CONFIRMED** (499 lines, still present) | Gate behind `process.env.NODE_ENV === 'development'` |
| T-006 | `frontend/src/components/chat/answer.tsx` lines 831-865 | Debug Info section in production UI exposes internal citation metadata | **PARTIALLY CORRECT** — `answer.tsx` reduced from 1055 to ~918 lines after UI polish; debug sections may have been removed | Verify and remove/wrap if present |

---

### 3.7 Other Cross-Cutting

| # | File | Issue | Status | Severity |
|---|---|---|---|---|
| R-007 | `backend/app/api/api_v1/datastore_scan.py` | `import asyncio` present and used | **RESOLVED** — now used for async scan operations | INFO |
| R-008 | `backend/app/api/api_v1/datastore_recovery.py` | `import asyncio` present and used | **RESOLVED** — now used for recovery operations | INFO |
| schema/__init__.py | `app/schemas/__init__.py` | Check if it re-exports used/unused schemas | **NOT VERIFIED** | Low (verify) |
| Legacy model files | Check if `app/models/` has unused model files | **NOT VERIFIED** | Low |

### 3.8 Resolved Since Audit

| # | Area | Finding | Why Resolved |
|---|---|---|---|
| R-1 | Backend | STYLE-001: deprecated `declarative_base` import | **FIXED** — `models/base.py` now uses `from sqlalchemy.orm import declarative_base` |
| R-2 | Backend | DUP-CONST-001/002/001: `MAX_FILE_SIZE`, `SUPPORTED_EXTENSIONS`, `_convert_to_markdown` duplication | **RESOLVED** — All files now import from `document_converter.py` |
| R-3 | Backend | UNUSED-003: `time as _time` in `chat.py` | **FALSE POSITIVE** — Was used for `_time.monotonic()` |
| R-4 | Backend | UNUSED-011: `fastapi.File` in `knowledge_base.py` | **FALSE POSITIVE** — Actively used for file uploads |
| R-5 | Backend | UNUSED-013: `upload_document` imports in `knowledge_base.py` | **FALSE POSITIVE** — Actively used |
| R-6 | Backend | UNUSED-017: `time` in `datastore_scan.py` | **FALSE POSITIVE** — Used for `time.monotonic()` |
| R-7 | Backend | UNUSED-007/008/009/012 in `datastores.py` | **RESOLVED** — Multiple dead imports removed from this file in prior cleanup |
| R-8 | Backend | UNUSED-010/015: imports in `admin.py` and `chat_files.py` | **RESOLVED** — Dead imports removed |
| R-9 | Backend | `datastore_scan.py` moved from `services/` to `api/api_v1/` | **MOVED** — Not deleted |
| R-10 | Cross-cutting | ChunkRecord/DataStoreChunkRecord dead code files | **RESOLVED** — Both files deleted (`chunk_record.py`, `datastore_chunk_record.py` no longer exist) |
| R-11 | Cross-cutting | Proxy routes to delete | **STILL EXISTS** — All 5 proxy route files/directories still present |
| R-12 | Frontend | `useHydrated` extraction | **PARTIALLY RESOLVED** — Extracted to `lib/hooks.ts` but `app/dashboard/page.tsx` still has local copy |

---

## 4. Priority Roadmap

### Phase 1 — Must Fix (blocks deployment / security)

1. **Fix alembic migration cycle** (M-001, M-002, M-003) — unblocks all future schema changes. ~37 migration files, 15+ with `down_revision=None`.
2. **Remove Docker credentials from tracked files** (S-001, S-002) — use env var interpolation
3. **Add startup validation for weak defaults in config.py** (S-005) — fail fast instead of shipping with `your-secret-key-here`

### Phase 2 — Should Fix (maintenance debt)

4. **Remove 6 confirmed dead imports** across backend files (5-10 min) — down from 17 after prior cleanup
5. **Remove 5 redundant Next.js proxy routes** ~~(5 min)~~ — ~~all still present~~ ~~**RESOLVED** — deleted~~ — **DELETED**
6. **Unify package manager** ~~— pick npm or pnpm, delete the other lock file~~ — ~~5 min~~ — **RESOLVED** — `package-lock.json` deleted, pnpm-lock.yaml kept, `shadcn-ui` moved to devDependencies
7. **Centralize duplicate types** in `lib/types.ts` (30 min)
8. **Extract shared upload hook** from `document-upload-steps.tsx` + `upload/page.tsx` (2-3 hours)
9. **Fix bare `except Exception:`** in `chat_files.py` and `datastore_scan.py` (15 min)
10. **Remove dead conftest files** ~~(`rootconftest.py`, `conftest_debug.py`)~~ — ~~(2 min)~~ — **RESOLVED** — both deleted
11. **Delete empty directories** ~~(`admin/watcher/`, `scan-progress-stream/`)~~ — ~~(2 min)~~ — **RESOLVED** — both deleted

### Phase 3 — Should Fix (code quality)

12. **Unify auth checks** — remove client-side redirects in layouts (30 min)
13. **Remove local `useHydrated`** ~~from `app/dashboard/page.tsx`~~ — ~~(2 min)~~ — **RESOLVED** — extracted to `lib/hooks.ts` and imported from there; no local copies remain
14. **Remove dead model relationships** in `models/chat.py` (5 min)
15. **Remove dead variables** (7 findings across 4 files) (15 min)
16. **Fix Pydantic v2 config style** in `config.py` (10 min)
17. **Fix rate limiter bug** in `auth.py` (1-2 hours)

### Phase 4 — Nice to Have

18. **Reduce monolithic components** — split `chat/[id]/page.tsx` (1173 lines), `answer.tsx` (1055 lines) (8-16 hours)
19. **Extract shared `decodeJWT`** from `lib/auth.ts` (10 min)
20. **Fix `window.location.href`** in `api.ts` 401 handler — use Next.js router (5 min)
21. **Fix checkbox no-op** in `chat-input.tsx` (5 min)
22. **Gate console.debug / debug UI** behind env check (20 min)
23. **Delete `test_imports.py`** diagnostic script (2 min)
24. **Move `debug_pipeline.py`** to `scripts/` (2 min)

---

## 5. Quick-Win Checklist (No Risk, Under 1 Hour Total)

- [x] ~~Delete 17 dead imports in backend files~~ — 11 resolved by prior cleanup, 6 remain
- [ ] Delete 5 redundant Next.js proxy route files
- [ ] Delete 2 dead conftest files (`rootconftest.py`, `conftest_debug.py`)
- [ ] Delete empty directories (`admin/watcher/`, `scan-progress-stream/`)
- [ ] Remove dead model relationships in `models/chat.py`
- [ ] Remove dead variables (7 findings)
- [x] ~~Fix deprecated SQLAlchemy import~~ — FIXED
- [ ] Unify package manager lock files
- [x] ~~Extract `useHydrated` to `lib/hooks.ts`~~ — PARTIALLY DONE (only one of 3 locations fixed)

**Total estimated effort: 30-45 minutes. Zero functional risk.**