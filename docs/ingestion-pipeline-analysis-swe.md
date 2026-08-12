# Ingestion Pipeline Analysis — Software Engineering Handoff

**Scope:** end-to-end study of the repository’s ingestion architecture for a self-hosted, multi-tenant RAG application.  
**Written for:** a software engineer taking ownership of the ingestion system.  
**Status:** analysis only; no code changes.  
**Date:** 2026-01-18.

---

## 1. Executive summary

The project is a self-hosted knowledge-base Q&A system with multi-tenant organisation management.  It has two document ingestion families:

1. **Direct knowledge-base (KB) upload** — user drops files into a KB, the backend stores them in a user/KB-scoped `uploads/` tree, converts, chunks, embeds, and indexes them in a KB-specific Qdrant collection (`kb_{kb_id}`).
2. **DataStore (watched folder) ingestion** — an admin creates a DataStore mapped to a folder under `/app/data`, assigns it to one or more organisations, and a filesystem watcher (or manual/startup scan) ingests files into a DataStore-specific Qdrant collection (`ds_{datastore_id}`).  Documents are shared with linked KBs via a `KnowledgeBaseDataStore` junction.

The conversion/chunk/embed pipeline is essentially shared by both paths (`process_document_background` in `backend/app/services/ingestion/document_processor.py`).  The main architectural problems are not in that pipeline but in the surrounding orchestration, multi-tenancy enforcement, duplicated lifecycle code, and state/progress model.

**Critical findings at a glance**

| Severity | Finding | Location |
|----------|---------|----------|
| Critical | `POST /{kb_id}/documents/process` allows duplicate `upload_id`s, does not lock `DocumentUpload` rows, and trusts client `upload_results` metadata such as `enable_ocr`, allowing duplicate `ProcessingTask`s. | `backend/app/api/api_v1/knowledge_base.py:405-411`, `backend/app/api/api_v1/knowledge_base.py:452-474` |
| Critical | `document_converter.py` has `MAX_FILE_SIZE = 10 * 1024 * 1024` but the upload endpoint never checks it, allowing huge files to be written to disk. | `backend/app/services/ingestion/document_converter.py:18`, `backend/app/api/api_v1/knowledge_base.py:254-330` |
| Critical | DataStore document access in `_check_document_access` only requires the user to own *any* KB linked to the DataStore, not the actual linked path. | `backend/app/api/api_v1/knowledge_base.py:703-729` |
| High | Direct-upload duplicate detection is read-check-then-insert; the only constraint is `(knowledge_base_id, file_name)`, not `(kb, filename, hash)`. | `backend/app/models/knowledge.py:69-72`, `backend/app/api/api_v1/knowledge_base.py:286-300` |
| High | No durable task queue — processing uses `asyncio.create_task`, executor threads, and fire-and-forget `BackgroundTasks`.  Restarts lose in-flight work. | `backend/app/api/api_v1/knowledge_base.py:452-474`, `backend/app/services/datastore_watcher/handler.py:854-870`, `backend/app/services/discovery/startup_recovery_service.py:380-420` |
| High | Frontend DataStore selection filters client-side by `assigned_orgs.length > 0`, not by the user’s actual `org_id`; `getTokenClaims()` returns `null` and the token `org_id` is never used in the frontend. | `frontend/src/app/dashboard/knowledge/new/page.tsx:27-37`, `frontend/src/lib/auth.ts:7-9` |
| High | Manual scan and event-driven progress counters share `last_scan_processed`, and both paths accumulate it.  Real-time and historical progress are conflated. | `backend/app/services/datastore_watcher/handler.py:1193-1211`, `backend/app/services/datastore_watcher/watcher.py:379-413` |
| Medium | Multiple overlapping orchestration implementations for the same `process_document_background` function (watcher, manual scan, startup recovery, direct upload). | See §6 and §9. |
| Medium | `_chunk_id_to_point_id` is duplicated in `document_qdrant.py` and `graph_service.py`. | `backend/app/services/ingestion/document_qdrant.py:95-97`, `backend/app/services/graph/graph_service.py:116-123` |
| Medium | Qdrant/Neo4j failures during cleanup are often logged and ignored, which can leave orphaned vectors/nodes. | `backend/app/services/datastore_watcher/handler.py:1031-1060`, `backend/app/services/cleanup/deletion_service.py:55-65` |

---

## 2. Intended architecture

The README and top-level docs describe a multi-tenant RAG system where:

- Admins create organisations and assign users and data sources to organisations.
- Users upload documents to KBs or expose local folders through DataStores.
- Documents are converted, chunked, embedded, and indexed into MySQL, Qdrant, and optionally Neo4j.
- DataStores represent watched local folders and may be shared by multiple organisations.
- Organisation and user scoping controls access to KBs and shared DataStore content.
- Filesystem changes trigger realtime ingestion, updates, and deletions.
- The frontend exposes upload, processing status, DataStore administration, scanning, and retrieval.

The data model in `backend/app/models/` and migrations is built around these key entities:

- `Organisation` (`backend/app/models/organisation.py`) — hierarchical self-referencing `parent_id` plus a materialised `path`.
- `User` (`backend/app/models/user.py`) — `org_id` (nullable), role enum (`user`/`admin`/`super_admin`).
- `KnowledgeBase` (`backend/app/models/knowledge.py`) — owned by a `user_id`, optionally scoped to an `org_id`; contains `Document`s and `DocumentChunk`s.
- `DataStore` (`backend/app/models/datastore.py`) — a watched local folder with `folder_path` unique; can be assigned to many orgs via `OrganizationDataStore`.
- `KnowledgeBaseDataStore` (`backend/app/models/knowledge.py:9-22`) — a KB may link to many DataStores and a DataStore may be linked to many KBs.
- `Document` (`backend/app/models/knowledge.py:48-73`) — has either `knowledge_base_id` (direct upload) or `data_store_id` (watched folder), with unique constraints `(knowledge_base_id, file_name)` and `(file_path, data_store_id)`.
- `DocumentChunk` and `ProcessingTask` — children of `Document` / `KnowledgeBase` / `DataStore`.
- `DataStoreFileManifest` — tracks discovered files per DataStore (`datastore_id`, `file_path`, `file_hash`, `file_size`).

Search stores are intentionally separate:

- **Qdrant** is the source of truth for chunk text and vectors; collections are named `kb_{kb_id}` or `ds_{datastore_id}`.
- **Neo4j** stores graph topology only (`Chunk`, `Entity`, `FROM_CHUNK`); `Chunk.qdrant_point_id` is the cross-reference.
- **MySQL** stores documents, chunks, uploads, tasks, settings, and metadata.

The settings architecture (`backend/app/services/settings_service.py`, `backend/app/core/settings_registry.py`) resolves in three tiers: org override → app value → registry default → `.env`/config default.  Because DataStores are shared across orgs, all ingestion-affecting settings (`CHUNK_SIZE`, `OVERLAP_PERCENTAGE`, `GRAPHRAG_ENABLED`, `GRAPHRAG_MAX_CHUNKS`, `NEO4J_LLM_CONTEXT`, `VISION_MODEL`, embedding model/dim) are `scope="app"` with `reload="ingest"`/`restart` and cannot be overridden per org.  This is explicitly documented in `docs/settings-migration-plan.md:36-40` and `backend/app/core/settings_registry.py:117-136`.

---

## 3. Entity and tenancy model

### 3.1 Organisation and user scoping

`backend/app/core/security.py:22-44` defines `get_admin_org_ids()`, which for an admin returns `[org_id] + all descendant org_ids` (BFS with a hardcoded 100-iteration limit); for `super_admin` it returns `None` (no restriction); for users without `org_id` it returns `[]`.  This helper is used in admin APIs such as `backend/app/api/api_v1/datastores.py:230-241` and `backend/app/api/api_v1/admin.py` to scope organisation listings.

KB and chat access are strictly user-scoped via `_kb_owner_filter()` in `backend/app/api/api_v1/knowledge_base.py:64-68` and `chat_owner_filter()` in `backend/app/api/api_v1/rbac.py`.  A user can only see KBs they personally created, even within the same org.  This is a deliberate design choice: ownership is `user_id`, while `org_id` is mostly used for org-level DataStore sharing and admin scoping.

### 3.2 DataStore assignment

A DataStore is created at `POST /api/admin/datastores` (`backend/app/api/api_v1/datastores.py:291-366`):

- Path must exist and be under `/app/data` (`_validate_folder_path`, lines 37-62).
- `folder_path` is globally unique.
- Non-super-admin creators are auto-assigned to their own org (`lines 352-362`).

Assignment is updated at `POST /api/admin/datastores/{id}/assign` (`backend/app/api/api_v1/datastores.py:508-574`):

- Accepts an `org_ids` list.
- Validates each org is in the admin’s scope.
- Empty `org_ids` removes all assignments within the admin’s scope.
- Duplicate `(org_id, data_store_id)` rows are prevented by a unique constraint (`backend/app/models/datastore.py:107-109`).

Listing is scoped: an admin sees only DataStores assigned to orgs in their scope; a super admin sees all (`backend/app/api/api_v1/datastores.py:224-288`).

### 3.3 KB ↔ DataStore linking

A user links a DataStore to their KB at `POST /api/knowledge-base/{kb_id}/link-datastore` (`backend/app/api/api_v1/knowledge_base.py:888-960`):

- Verifies KB ownership (`_kb_owner_filter`, line 906).
- Verifies the DataStore is assigned to the user’s org or a descendant (`lines 913-935`, BFS with a 100-iteration limit).
- Creates a `KnowledgeBaseDataStore` row.

Unlinking at `DELETE /api/knowledge-base/{kb_id}/unlink-datastore/{data_store_id}` (`backend/app/api/api_v1/knowledge_base.py:962-995`) only removes the junction row; it does not delete any documents or vectors.

### 3.4 Document access model

`Document` has two independent owners: `knowledge_base_id` and `data_store_id`.  `_check_document_access()` in `backend/app/api/api_v1/knowledge_base.py:703-729` checks:

- If `document.knowledge_base_id` is set, the user must own the KB.
- If `document.data_store_id` is set, the user must own **any** KB linked to that DataStore (or more precisely, a `KnowledgeBase` joined to `KnowledgeBaseDataStore` where `KnowledgeBase.user_id == current_user.id`).

This is the intended path for citation popups and direct document endpoints, but it is too permissive for shared DataStores: any user in any org that is linked to the same DataStore and owns at least one KB can reach all DataStore documents through the global document-by-ID endpoint.  The retrieval path uses a different filter (`get_effective_datastore_ids` in `backend/app/services/retrieval/retrieval.py:57-126`) that includes both KB-linked DataStores and org-assigned DataStores, so in practice a user in an org with a DataStore assignment already sees all of that DataStore’s documents.  The product’s sharing model is therefore “all DataStore content is shared across linked/assigned orgs,” but this is not explicitly validated or enforced by `_check_document_access()`.

---

## 4. Direct KB upload pipeline

### 4.1 Frontend flow

The primary UI is `frontend/src/components/knowledge-base/document-upload-steps.tsx`, used by `frontend/src/app/dashboard/knowledge/[id]/page.tsx`:

1. User drops files; `handleFileUpload` calls `POST /api/knowledge-base/{id}/documents/upload`.
2. The backend returns a list of `{upload_id, file_name, temp_path, status: "pending" | "exists", skip_processing}`.
3. If `status === "exists"`, the frontend treats it as completed; otherwise the file moves to the upload step.
4. User clicks “Start Processing”; `handleProcess` calls `POST /api/knowledge-base/{id}/documents/process` with `upload_results`.
5. The backend creates `ProcessingTask` rows and starts background work.
6. The frontend polls `GET /api/knowledge-base/{id}/documents/tasks?task_ids=...` every 3 seconds, backing off to 8 seconds on error, and stops after 10 consecutive errors.

A legacy `frontend/src/app/dashboard/knowledge/[id]/upload/page.tsx` exists with the same flow but a simpler 2-second polling loop.

The list page uses `useKnowledgeContext` (`frontend/src/contexts/knowledge-context.tsx:32-39`) to load the KB list without an explicit `org_id`, silently ignoring auth errors.  The backend filters by the current user, but the frontend never validates `org_id`.

### 4.2 Upload endpoint

`POST /api/knowledge-base/{kb_id}/documents/upload` (`backend/app/api/api_v1/knowledge_base.py:254-332`):

- Verifies KB ownership (`lines 264-269`).
- Checks extension against `SUPPORTED_EXTENSIONS` (`backend/app/services/ingestion/document_converter.py:23-38`).
- Computes a SHA-256 `file_hash` from the full in-memory bytes.
- Checks for an existing `Document` with the same `file_name`, `file_hash`, and `knowledge_base_id`.
- Saves a temporary file.
- Creates a `DocumentUpload` record.

**Issues observed:**
- There is no file-size check, so the 10 MB `MAX_FILE_SIZE` in `document_converter.py:18` is dead (`backend/app/services/ingestion/document_converter.py:18`).
- Duplicate detection is read-then-write; the DB only enforces uniqueness on `(knowledge_base_id, file_name)`.  Two concurrent uploads of the same filename with different content can race; two uploads of the same file (same hash) can both pass the `.first()` check and both insert, one failing the unique constraint and raising an unhandled `IntegrityError`.
- Uploads are held in memory (`file = await file.read()`), so large files can exhaust RAM.

### 4.3 Process endpoint

`POST /api/knowledge-base/{kb_id}/documents/process` (`backend/app/api/api_v1/knowledge_base.py:375-459`):

- Verifies KB ownership.
- Collects `upload_ids` from `upload_results`.
- Queries `DocumentUpload` with `id.in_(upload_ids)` and `knowledge_base_id == kb_id` (`lines 405-408`).
- If the count differs, returns `400` with “One or more upload IDs are invalid.”
- Creates `ProcessingTask` rows and returns `task_id`s.
- Uses FastAPI `BackgroundTasks` to call `add_processing_tasks_to_queue` (`lines 452-457`), which loops and calls `asyncio.create_task(process_document_background(...))` for each upload.

**Trust and concurrency gap:** the client sends a list of `upload_results` objects and `enable_ocr` per file.  The server-side query does filter `DocumentUpload` by `id.in_(upload_ids)` and `knowledge_base_id == kb_id`, and rejects unknown IDs (`lines 405-411`), so cross-KB upload IDs are not accepted.  However, the endpoint:

- Trusts the client’s per-file `enable_ocr` value without re-deriving it from the stored upload record.
- Does not de-duplicate `upload_ids`, so duplicate IDs in the same request create duplicate `ProcessingTask`s.
- Does not take a row-level lock or idempotency key on `DocumentUpload`, so concurrent `process` calls for the same upload can create multiple `ProcessingTask` rows.
- Does not check whether the upload has already been processed or is in progress, allowing duplicate work.
- Builds `task_data` from the client `upload_results` list instead of from the validated `DocumentUpload` rows, so a client can mismatch `upload_id` and `file_name`/`enable_ocr`.

### 4.4 Background processing

`process_document_background` in `backend/app/services/ingestion/document_processor.py:148-561` is the single ingestion worker.  It is called in four places:

1. `add_processing_tasks_to_queue` for direct KB uploads (`backend/app/api/api_v1/knowledge_base.py:452-474`).
2. `_run_ingestion` in the watcher handler for DataStore events (`backend/app/services/datastore_watcher/handler.py:1238-1319`).
3. `_run_ingestion` in the watcher for manual scans (`backend/app/services/datastore_watcher/watcher.py`, see §5).
4. `_run_ingestion` in `StartupRecoveryService` (`backend/app/services/discovery/startup_recovery_service.py:392-463`).

Each caller wraps the same core function differently: some pass `db=None` and open a new session, some pass a pre-existing session, some create a new `asyncio` event loop in a thread, some use the main FastAPI event loop.  This duplicated orchestration is the largest structural risk in the ingestion layer.

The core pipeline steps (`backend/app/services/ingestion/document_processor.py:148-561`):

1. Resolve `chunk_size` and `chunk_overlap` from settings (`lines 179-184`).  These are app-level because shared DataStores require consistent indexing.
2. Mark the `ProcessingTask` as `processing` (`lines 211-214`).
3. Convert with MarkItDown (`_convert_to_markdown` in `backend/app/services/ingestion/document_converter.py:132-191`):
   - `enable_ocr` is a tri-state override.
   - If `enable_ocr=True` and `VISION_MODEL` is not set, it logs a warning and falls back to text-only (`document_converter.py:159-160`).
   - Reasoning tags are stripped by `strip_reasoning_tags`.
   - On failure it falls back to raw UTF-8.
4. Clean and validate non-empty output (`lines 242-260`).
5. Split with `RecursiveCharacterTextSplitter` (`lines 269-275`), run in executor.
6. Ensure Qdrant collection (`_ensure_qdrant_collection`); collection name `kb_{kb_id}` or `ds_{datastore_id}` (`lines 284-295`).
7. For KB files, move `temp_path` to `user_{uid}/kb_{kb_id}/{file_name}` (`lines 302-309`); for DataStore files the source file stays in place.
8. Create or update the `Document` record (`lines 311-354`).
9. Delete old `DocumentChunk` and Qdrant points for this document (`lines 357-381`).  Note the conditional query at `lines 362-365` and `377-380` is a Python expression, not SQLAlchemy SQL; it happens to work when `data_store_id` is an int because `DocumentChunk.data_store_id == data_store_id` evaluates to a SQLAlchemy binary clause, but the fallback clause `DocumentChunk.kb_id == kb_id` is never evaluated as SQL.  In the `data_store_id is None` branch, the `else` value is the clause object `DocumentChunk.kb_id == kb_id`, which is truthy, so the filter becomes `DocumentChunk.data_store_id == None` — i.e. it filters for `data_store_id IS NULL`.  For direct KB uploads this is fine because their chunks have `data_store_id=None`, but it would break for a DataStore with `kb_id=None`.  This is a latent correctness issue in the delete-old-chunks branch.
10. Build chunk records in a thread (`_build_chunk_records`, `lines 386-413`).  Chunk `id` is a SHA-256 hash of the collection/file-name/chunk-text tuple.  The `qdrant_point_id` is derived deterministically from the chunk `id` via `_chunk_id_to_point_id` (`backend/app/services/ingestion/document_qdrant.py:95-97`).
11. Upsert to Qdrant in batches (`_upsert_to_qdrant`, `lines 422-430`); dense and sparse vectors are generated there.
12. Commit chunks and mark task complete (`lines 434-445`).
13. Run Neo4j graph extraction asynchronously if `GRAPHRAG_ENABLED` (`lines 449-516`).  Graph failures are non-fatal; the document is still searchable.
14. On any exception, roll back, delete the `Document` if it was committed, mark the task failed, and delete the permanent file (`lines 518-557`).

### 4.5 Progress and task state

`ProcessingTask` has `status`, `progress`, `progress_message`, and `graph_status`/`graph_error` (`backend/app/models/knowledge.py:92-114`).  The `_set_progress` callback in `process_document_background:197-209` commits the session on every progress tick.  Because it uses the same `db` session as the main transaction, this can create inconsistent state if the main transaction later rolls back (the progress commits are already persisted).  The docstring says it uses a fresh merge, but the code does not.

`ProgressTimeout` is a context manager that warns if no progress ping occurs for `PROCESSING_TIMEOUT_SILENCE_S`; it is non-fatal and does not cancel the task (`backend/app/services/ingestion/document_processor.py:219-226`).

---

## 5. DataStore ingestion pipeline

### 5.1 DataStore creation and assignment

A DataStore is a first-class watched folder (`backend/app/models/datastore.py:25-86`):

- `folder_path` is globally unique and must be under `/app/data`.
- `scan_pattern` is a comma-separated glob list.
- `auto_scan_enabled` controls whether the filesystem watcher registers the folder.
- `auto_scan_interval_minutes` is stored but the batch timer that would use it is dead code; normal event-driven processing is immediate (see §6.6).
- `last_scan_*` fields are used for both manual scans and event-driven processing, which can conflate the two progress models.

The REST surface in `backend/app/api/api_v1/datastores.py`:

- `GET /api/admin/datastores` — list, scoped to admin orgs (lines 224-288).
- `POST /api/admin/datastores` — create (lines 291-366).
- `GET /api/admin/datastores/{id}` — get (lines 369-397).
- `PATCH /api/admin/datastores/{id}` — update; if `auto_scan_enabled`/`auto_scan_interval_minutes` changed, it calls `watcher.sync_watchers_with_database()` (lines 400-480).
- `DELETE /api/admin/datastores/{id}` — delete, blocked if org assignments remain (lines 483-505).
- `POST /api/admin/datastores/{id}/assign` — bulk assign/unassign orgs (lines 508-574).

The `_datastore_in_scope()` helper at `backend/app/api/api_v1/datastores.py:159-172` checks `OrganizationDataStore` assignment.  `get_datastore` filters `assigned_orgs` to the admin’s scope (lines 391-395).

### 5.2 Discovery engine

`backend/app/services/discovery/discovery_engine.py` is the canonical scanner used by both manual scans and startup recovery:

- `hash_file()` reads the file in 8 KB chunks and computes SHA-256; it also checks size before and after reading and returns an empty hash if the size changed (`lines 37-68`).
- `_matches_pattern()` uses `fnmatch.fnmatch` against the file basename (`lines 100-111`).
- `_walk_files()` walks the folder and returns absolute paths (`lines 205-211`).
- `discover_datastore()` loads the DataStore and manifest, hashes files concurrently with `ThreadPoolExecutor` (`lines 291-429`), classifies new/modified/deleted, and calls `_upsert_manifest()` to persist new/updated manifest rows.
- `_classify_files()` compares the manifest map to collected files; deleted files are those in manifest but not on disk (`lines 214-246`).
- `_upsert_manifest()` updates existing manifest rows and adds new ones under `_FLUSH_LOCK` (`lines 249-288`).
- `discover_all()` runs discovery for every active DataStore concurrently (`lines 437-478`).

The manifest table `DataStoreFileManifest` (`backend/app/models/datastore.py:116-145`) has a unique constraint on `(datastore_id, file_path)`.  This makes discovery idempotent and supports deletion detection.

### 5.3 Startup recovery service

`StartupRecoveryService` (`backend/app/services/discovery/startup_recovery_service.py:29-606`) is started in `main.py:148-153` before the watcher:

1. Queries all active DataStores.
2. Spawns one thread per DataStore into a `ThreadPoolExecutor(max_workers=4)`.
3. Each thread runs `_discovery_pipeline_worker()`:
   - Calls `discover_datastore()`.
   - For each new/modified file, calls `process_new_file()`.
   - For each deleted file, calls `_handle_deletion_records()`.
4. `process_new_file()` checks for an existing `Document` and a non-completed `ProcessingTask`.  If a failed task exists it reuses it; if a pending/processing task exists it skips; otherwise it creates a Document and ProcessingTask and submits `_run_ingestion()`.
5. `_run_ingestion()` creates a brand-new `asyncio.new_event_loop()` in a thread, runs `process_document_background`, and then manually updates the `ProcessingTask` status to `completed` or `failed` in a fresh session (`lines 392-463`).
6. `_handle_deletion_records()` deletes Qdrant points, DB chunks/tasks/Document, Neo4j graph nodes, and the manifest entry (`lines 486-565`).

This means at startup the same files may be discovered both by recovery and by the watcher (which is started just after recovery), creating overlapping ingestion attempts.  The `(file_path, data_store_id)` unique constraint on `Document` prevents duplicates, but race conditions can produce `IntegrityError` or repeated `ProcessingTask` rows.

### 5.4 Manual scans

`POST /api/admin/datastores/{id}/scan` (`backend/app/api/api_v1/datastore_scan.py:409-578`) starts a background thread:

- Calls `watcher._init_scan(datastore_id)` to create an in-memory scan record and set `last_scan_status = "running"`.
- Runs `watcher.scan_single_datastore(datastore_id)`.

`DataStoreWatcher.scan_single_datastore()` (`backend/app/services/datastore_watcher/watcher.py:526-668`):

- Calls `discover_datastore()`.
- Updates `_active_scans[scan_id]` with discovery counts.
- Walks new/modified files and calls `_handle_file_in_scan()` for each.
- Tracks `Future`s in `_scan_futures[scan_id]`.
- Calls `_handler._handle_deletion()` for deleted files.
- Waits up to one hour per ingestion `Future`.
- Calls `_complete_scan()` to persist `last_scan_*` counters.

`_handle_file_in_scan()` is similar to the event-driven `_handle_file()` but returns a `Future` so the scan can wait.  It also has its own `_ingest_file_in_scan()` and `_update_document_in_scan()` wrappers.

`POST /api/admin/datastores/{id}/stop-scan` (`backend/app/services/datastore_watcher/watcher.py:463-498`) sets `last_scan_status = "idle"` and `_active_scans[scan_id]["status"] = "cancelled"`; the scan thread checks `_is_scan_cancelled()` between files.

`POST /api/admin/datastores/{id}/flush` (`backend/app/api/api_v1/datastore_scan.py:581-643`) calls `watcher._handler._process_pending_changes(datastore_id)` to force immediate event-driven processing.

### 5.5 Event-driven file processing

`DatastoreFileEventHandler._handle_file()` (`backend/app/services/datastore_watcher/handler.py:509-649`) is the routing function:

- Skips directories, unsupported extensions, hidden files, and non-existent files.
- Matches `scan_pattern`.
- Computes the file hash via `_compute_hash()`.
- Looks up an existing `Document` by `(file_path, data_store_id)`.
- Branches:
  - `event_type == "deleted"` → `_handle_deletion()`.
  - Document exists, hash unchanged, chunks exist → skip.
  - Document exists, hash unchanged, no chunks → `_ingest_file()` (re-ingest).
  - Document exists, hash changed → `_update_document()`.
  - New file → `_ingest_file()`.

`_ingest_file()` (`backend/app/services/datastore_watcher/handler.py:741-882`) and `_update_document()` (`backend/app/services/datastore_watcher/handler.py:884-995`):

- Validate extension, hidden status, and existence.
- Create or update `Document` and `ProcessingTask`.
- Call `_upsert_manifest()` to keep the manifest in sync.
- Create a new `asyncio` event loop and submit `_run_ingestion()` to the handler’s executor.
- Add a done callback `_on_ingestion_done()`.

`_handle_deletion()` (`backend/app/services/datastore_watcher/handler.py:1001-1093`) for DataStore files:

- Looks up `Document` by `(file_path, data_store_id)`.
- Deletes Qdrant vectors for the document’s chunks.
- Deletes `DocumentChunk`, `ProcessingTask`, and the `Document` row.
- Calls `delete_graph_for_document()`.
- **Also deletes the `DataStoreFileManifest` row** (`lines 1086-1090`).  This contradicts a prior claim that the manifest is not updated; the code does delete it.
- Commits; Qdrant/Neo4j failures are logged but do not abort the DB deletion.

### 5.6 Qdrant, Neo4j, and MySQL for DataStores

DataStore documents are indexed into Qdrant collection `ds_{datastore_id}`.  MySQL stores `Document`, `DocumentChunk`, and `ProcessingTask` rows with `data_store_id` set and `knowledge_base_id = NULL`.  Neo4j stores `Chunk` nodes with `qdrant_collection = "ds_{datastore_id}"` and `data_store_id` as attributes, plus extracted entities/relationships.

The DataStore deletion path in `backend/app/services/cleanup/deletion_service.py:218-288`:

1. Queries all documents for the DataStore.
2. Deletes Qdrant points and then the entire collection (`_delete_qdrant_for_ds`, lines 66-106).
3. Deletes Neo4j `Chunk` and orphaned entity nodes (`_delete_neo4j_for_ds`, lines 127-153).
4. Deletes `KnowledgeBaseDataStore` and `OrganizationDataStore` junction rows.
5. Deletes each `Document` (cascade deletes chunks/tasks).
6. Deletes the `DataStore`.

Files on disk are intentionally not deleted for DataStores; only DB/Qdrant/Neo4j state is removed.

### 5.7 Watcher/handler coupling

The event-driven system is split between two classes:

- `DatastoreFileEventHandler` (`backend/app/services/datastore_watcher/handler.py`) — receives watchdog events, debounces, queues, and processes file changes.
- `DataStoreWatcher` (`backend/app/services/datastore_watcher/watcher.py`) — manages the observer lifecycle, manual scans, and scan state; it owns a `DatastoreFileEventHandler` instance.

The watcher passes its own `_on_changes(datastore_id, org_id, changes)` method as the handler’s callback (`handler.py:116`).  The handler’s `_on_changes()` (lines 1341-1391) iterates changes, calls `_handle_file()`, waits for ingestion `Future`s, and updates `last_scan_processed` via `_update_scan_progress()`.  This two-class split is described in `docs/file-changes-detection-ingestion.md:303-318` and can be hard to follow because the callback chain is indirect.

---

## 6. Realtime filesystem change propagation

### 6.1 Watcher startup and path resolution

`DataStoreWatcher.start()` (`backend/app/services/datastore_watcher/watcher.py`) starts a single global `watchdog` observer rooted at `/app/data` and calls `sync_watchers_with_database()` to register active DataStores.  `sync_watchers_with_database()` (`watcher.py:208-214`) is a public wrapper that simply delegates to the private `_sync_watchers_with_database()` (`watcher.py:983-1030`); the call is `self._sync_watchers_with_database()` and is not infinite recursion.

`_sync_watchers_with_database()`:

- Builds an `org_id` lookup from `OrganizationDataStore`.
- Queries active DataStores with `auto_scan_enabled=True` (`watcher.py:1003-1010`).
- Calls `add_datastore()` for each, passing `org_id`, `folder_path`, and `auto_scan_interval_minutes`.
- Unassigned DataStores are registered with `org_id=None` (`watcher.py:1016`).

`DatastoreFileEventHandler._resolve_datastore()` (`handler.py:188-206`) sorts configured `folder_path`s by length (longest first) so `/app/data/reports/2024` wins over `/app/data/reports`.  It checks whether the event path starts with a registered folder path plus `/` or equals it exactly.

### 6.2 Debounce, write-completion delay, and pending queue

The event flow (`backend/app/services/datastore_watcher/handler.py:49-491` and `docs/file-changes-detection-ingestion.md:11-47`):

1. Watchdog fires `on_created`/`on_modified`/`on_deleted`/`on_moved`.
2. `_resolve_datastore()` finds the owning DataStore.
3. `_should_process()` (`handler.py:212-234`) checks a per-file 1-second debounce window (`_last_call`).  If the same path fires again within 1 second, it is dropped.
4. `_after_process()` resets the per-file timer only after the event is successfully queued.
5. `_dispatch()` (`handler.py:463-491`) spawns a daemon thread that sleeps 1 second, creates a `_SyntheticEvent` (a lightweight object holding the captured `src_path` and `event_type`), and calls `_queue_change()` + `_process_pending_changes()`.
6. `_queue_change()` appends to `pending_changes[datastore_id]` a dict with `path`, `event_type`, `datastore_id`, `org_id`, and `timestamp`.
7. `_process_pending_changes()` (`handler.py:285-315`) pops the queue, sets `_processing.add(datastore_id)`, and calls `self._on_changes()`.  New events for the same DataStore queue up; other DataStores process independently.

This double debounce (`_should_process` and the 1-second write-completion delay) plus the `_Debouncer.touch()` in `_dispatch()` can be hard to reason about.  The `_Debouncer` (lines 49-69) coalesces repeated events for the same path within 1 second into one dispatch.

### 6.3 Event coalescing and stability

Rapid sequences of create/modify/delete are not coalesced by final filesystem state; the handler processes each coalesced event type independently.  For example, a file that is created and immediately deleted within the 1-second window may still trigger an `_ingest_file` for a file that no longer exists, which is then skipped by the existence check.  Moves are split into a source `deleted` and destination `created` (`handler.py:419-461`).  Cross-DataStore moves are handled because each path is resolved independently.

File hashing in the handler (`_compute_hash()`, around `handler.py:680-704`) is synchronous; large files can block the event handler thread.  The discovery engine uses a `ThreadPoolExecutor` for hashing; the handler does not.

### 6.4 Manual scan and event-driven overlap

Manual scans and event-driven processing are independent:

- Manual scan does not clear `pending_changes`.
- `_processing` is a per-DataStore set used only by event-driven processing.
- Both call `process_document_background()` and update `last_scan_processed` via SQL-level accumulation (`handler.py:1193-1211`, `watcher.py:379-413`).
- Both create their own DB session per file/operation.

The two paths can process the same file concurrently.  The `Document` unique constraint `(file_path, data_store_id)` and the hash-unchanged skip branch usually prevent duplicate ingestion, but a race between manual scan and event-driven update can still re-process the same file if the hash check runs before the file is written.

### 6.5 Status and progress fields

`DataStore` fields are overloaded:

- `last_scan_status` is set to `running` by `_init_scan()` and to `completed`/`error`/`idle`/`cancelled` by `_complete_scan()` or the scan API.
- `last_scan_processed` is incremented by both manual scans and event-driven `_update_scan_progress()`.
- `pending_changes` is a live count from the handler queue, added to the API response in `datastores.py:263-286`.
- `processing` is derived from `handler._processing`.

This means the same DB fields represent two different concepts depending on the trigger, making operational debugging harder.

### 6.6 Dead batch-timer code

The handler still has `_start_batch_timer()`, `_stop_batch_timer()`, and `_flush_batch()` methods (`handler.py:240-335`).  The documentation in `docs/file-changes-detection-ingestion.md:73` and `:335-337` states they are never called in normal event-driven operation; the batch timer was removed to avoid 5-minute delays.  The `auto_scan_interval_minutes` field in the DataStore model is stored and surfaced in the UI (`frontend/src/app/dashboard/admin/data-sources/page.tsx:932-951`) but is not used for timed processing.  This is a documentation/code consistency issue and a potential source of user confusion.

---

## 7. Frontend behaviour

### 7.1 Knowledge base pages

`frontend/src/app/dashboard/knowledge/page.tsx` lists KBs.  It uses `useKnowledgeContext` (`frontend/src/contexts/knowledge-context.tsx:32-39`), which fetches `GET /api/knowledge-base` without an explicit `org_id` and silently ignores errors.  The backend filters by the current user, but the frontend never validates `org_id`.

`frontend/src/app/dashboard/knowledge/new/page.tsx`:

- Fetches `GET /api/admin/datastores` and filters client-side with `ds.assigned_orgs && ds.assigned_orgs.length > 0` (lines 29-31).
- It does **not** check whether the user’s org is in `assigned_orgs`.
- After creating the KB, it links selected DataStores one at a time, swallowing link errors (lines 63-72).  The user is not told if a link failed.

`frontend/src/app/dashboard/knowledge/[id]/page.tsx`:

- Same client-side filter for available DataStores (lines 66-77).
- Fetches the full KB after link/unlink to refresh the DataSources list (lines 100-102, 121-123), which is inefficient.

### 7.2 Upload and processing UI

`frontend/src/components/knowledge-base/document-upload-steps.tsx`:

- Defines `UploadResult.status` as only `"exists" | "pending"` (lines 52-60); the backend actually returns at least these two, and the code treats `"exists"` as completed.
- Matches upload results by `file_name` (lines 192-211), so uploading two files with the same name can collide.
- Polls tasks with a `consecutiveErrors` counter and stops after 10 errors (lines 327-397).

`frontend/src/components/knowledge-base/document-list.tsx`:

- Polls task progress and merges it with static document data.
- The `taskProgress` state (lines 55) is never pruned, so long sessions accumulate completed/failed tasks.
- Delete is disabled while `isInProgress` (line 299) but there is no cancel button.

A legacy `frontend/src/app/dashboard/knowledge/[id]/upload/page.tsx` uses a simpler 2-second fixed polling loop with no backoff or error cap.

### 7.3 DataStore admin page

`frontend/src/app/dashboard/admin/data-sources/page.tsx`:

- Fetches `GET /api/admin/datastores` and `GET /api/admin/orgs` in parallel.
- Uses three separate polling intervals (2 s when `processing`, 5 s for recovery, 2 s for recovery-complete refresh) plus a 500 ms manual-scan polling loop.
- Uses raw `fetch()` for scan and recovery endpoints (lines 194-196, 337-340, 467-471) instead of the shared `api` client, so auth error handling and 401 redirect logic are inconsistent.
- The manual-scan polling is a blocking `while` loop in an async function (lines 351-410) that continues even if the component unmounts because `scanPollRef` is only checked at the top of the loop.
- `assigned_orgs` is typed as `{ id, name }[]` (line 50), while `new/page.tsx` and `[id]/page.tsx` type it as `{ org_id, org_name }[]` (lines 14, 36).  The runtime values are `{ id, name }` from the backend, so the KB-linking UI treats `org_id` and `org_name` as `undefined`.

### 7.4 Auth and multi-tenancy in the frontend

`frontend/src/lib/auth.ts:7-9`:

```typescript
export function getTokenClaims(): TokenClaims | null {
  return null
}
```

The function is stubbed; no frontend code reads the JWT `org_id`.  All org scoping is therefore enforced by the backend.  This is a risk if the backend ever relaxes a check or if a frontend filter is used for security as in the DataStore selection case.

---

## 8. Settings and process-global ingestion constraints

### 8.1 Settings resolution

The settings service (`backend/app/services/settings_service.py:169-217`) resolves a setting in this order:

1. Org override (only if `scope == "org"` and `org_id` is set).
2. App-level value (`scope == "app"`, `org_id = NULL`).
3. Registry default.

For non-registry keys it falls back to `config.py`/env.  Values are cached in memory for a short TTL.  Secret values are encrypted with Fernet derived from `SECRET_KEY` via PBKDF2, stored with an `enc:` prefix, and masked in API responses (`backend/app/services/settings_service.py:41-74` and `backend/app/api/api_v1/settings.py`).

### 8.2 Registry classification

`backend/app/core/settings_registry.py` declares every setting with `scope` (`app`/`org`) and `reload` (`immediate`/`next_request`/`ingest`/`restart`).  All ingestion-affecting keys are `scope="app"` because DataStores are shared across orgs:

- `EMBEDDING_API_BASE`, `EMBEDDING_API_KEY`, `DENSE_EMBEDDINGS_MODEL`, `DENSE_EMBEDDING_DIM` (lines 79-91).
- `OPENAI_VISION_API_BASE`, `VISION_API_KEY`, `VISION_MODEL` (lines 93-101).
- `GRAPHRAG_API_BASE`, `GRAPHRAG_API_KEY`, `GRAPHRAG_LLM` (lines 103-111).
- `CHUNK_SIZE`, `OVERLAP_PERCENTAGE`, `GRAPHRAG_ENABLED`, `GRAPHRAG_MAX_CHUNKS`, `NEO4J_LLM_CONTEXT` (lines 117-136).
- `WATCHER_ENABLED`, `WATCH_POLL_INTERVAL` (lines 139-143).

Query-time settings are generally `scope="org"`: response model, query rewrite, retrieval tuning, reranker enable/threshold, GraphRAG query-time hops/limit/fanout, etc.  This is consistent with `docs/settings-migration-plan.md:36-40` and `:151`.

**Operational implication:** because these are app-level, changing chunk size or embedding model for a single tenant requires either a new deployment or re-indexing the whole DataStore corpus.  The registry marks them `requires_reindex=True` and `reload="ingest"`/`restart`, but there is currently no automated re-index workflow.

### 8.3 Process-global state

Several resources are process-global and not per-org:

- MarkItDown singleton with vision client (`backend/app/services/ingestion/document_converter.py:20-129`).
- FastEmbed sparse embedder and cross-encoder reranker (preloaded in `main.py:139-146`).
- Neo4j driver and LLM pipeline singletons (`backend/app/services/graph/graph_service.py:78-80`, `128-140`).
- Single `watchdog` observer rooted at `/app/data` (`backend/app/services/datastore_watcher/watcher.py`).

This means two organisations cannot use different embedding models or chunk sizes while sharing the same process/DataStore.  The current design accepts this by making those settings app-level, but it is a hard architectural constraint that should be documented clearly.

---

## 9. Persistence, cleanup, and transaction boundaries

### 9.1 MySQL transaction model

Direct KB processing uses a single `SessionLocal()` per task and commits at several points:

- Task marked `processing` (`document_processor.py:214`).
- `Document` record created/updated (`document_processor.py:353`).
- Old chunks deleted and Qdrant points deleted (`document_processor.py:381`).
- Chunks committed and task completed (`document_processor.py:440`, `445`).
- Progress commits (`document_processor.py:203`).
- Graph status updates in a separate session (`document_processor.py:461-505`).

These multiple commit points mean the operation is not atomic end-to-end.  A crash after the `Document` commit but before the Qdrant upsert leaves a `Document` with no chunks; the re-ingest/recovery branch (hash unchanged, no chunks) attempts to handle this.

### 9.2 Qdrant and Neo4j cleanup

Deletion logic is duplicated across modules:

- `backend/app/services/datastore_watcher/handler.py:1001-1093` (event-driven deletion).
- `backend/app/services/discovery/startup_recovery_service.py:486-565` (recovery deletion).
- `backend/app/services/cleanup/deletion_service.py:159-288` (KB and DataStore deletion).

All three follow the same rough order but differ in how they handle Qdrant/Neo4j failures.  The handler and recovery code log Qdrant non-404 errors as warnings and continue; `deletion_service.py` does the same.  The result is best-effort cleanup and a non-zero risk of orphaned Qdrant points or Neo4j `Chunk`/`Entity` nodes after crashes.

### 9.3 Neo4j deletion redundancy

`delete_graph_for_kb()` in `backend/app/services/graph/graph_service.py:884-979` runs four Cypher passes:

1. Delete `r {kb_id: $kb_id}` relationships.
2. `DETACH DELETE` Chunk nodes with this `kb_id` and `data_store_id IS NULL`.
3. Sweep orphaned entity nodes.
4. Defensive second sweep of Chunk nodes with this `kb_id`, then another entity sweep.

The defensive sweep exists because prior code paths sometimes skipped cleanup, but it is also a sign that the transaction boundaries are not trusted.  `delete_graph_for_document()` (`backend/app/services/graph/graph_service.py:827-881`) similarly detaches Chunk nodes and then runs a second defensive Chunk sweep.

### 9.4 Temporary/permanent file cleanup bug

`process_document_background:550-557` deletes `permanent_path` on failure for KB files.  If the `move_file()` at lines 304-309 failed, `permanent_path` is the intended destination and the file still sits at `temp_path`.  The cleanup will try to delete a file that does not exist and leave the temp file behind.  The code also does not delete `DocumentUpload` rows on failure, so failed uploads remain in the DB.

### 9.5 Deletion service ordering

`delete_kb()` in `backend/app/services/cleanup/deletion_service.py:159-215`:

1. Deletes KB files from disk.
2. Deletes the Qdrant collection for the KB.
3. Deletes Neo4j graph for the KB.
4. Deletes the KB row (the event listener in `backend/app/models/knowledge.py:141-179` then deletes direct-upload documents and sets `knowledge_base_id = NULL` on DataStore documents).

Because the event listener runs in the same SQLAlchemy `before_delete` hook, the direct-upload `Document` delete and DataStore document `kb_id = NULL` update happen within the KB deletion transaction.  However, `delete_kb()` already deletes Qdrant/Neo4j before the DB delete, so if the DB delete fails, the vector/graph data is already gone.

`delete_datastore()` (lines 218-288) blocks deletion if any `OrganizationDataStore` assignments exist, then deletes Qdrant points/collection, Neo4j nodes, junction rows, documents, and the DataStore record.  Files on disk are preserved.

---

## 10. Documentation/implementation discrepancies

| Document | Claims | Implementation | Discrepancy |
|----------|--------|----------------|-------------|
| `docs/file-changes-detection-ingestion.md:73` and `:335-337` | Batch timer removed; `_start_batch_timer()` is dead code. | `_start_batch_timer()`/`_stop_batch_timer()`/`_flush_batch()` still exist in `backend/app/services/datastore_watcher/handler.py:240-335`. | Dead code remains; `auto_scan_interval_minutes` is stored and exposed in UI but unused. |
| `docs/file-changes-detection-ingestion.md:123` | `_update_scan_progress` does direct assignment (`ds.last_scan_processed = processed`) while `_on_changes` accumulates (`ds.last_scan_processed += changes_processed`). | Both `_update_scan_progress()` in `watcher.py:379-413` and the one in `handler.py:1193-1211` use SQL `last_scan_processed = last_scan_processed + processed` (accumulation). | The doc comment about direct assignment is outdated. |
| `docs/file-changes-detection-ingestion.md` / `docs/ingestion-pipeline.md` | Event-driven processing is immediate. | Legacy 5-minute batch timer code remains, and `auto_scan_interval_minutes` is configurable. | User may expect scheduled batch scans that do not exist. |
| `docs/file-changes-detection-ingestion.md:225-228` | SSE endpoint exists but does not work through Next.js rewrites; polling is the fallback. | The frontend in `frontend/src/app/dashboard/admin/data-sources/page.tsx:345-346` confirms this and uses 500 ms polling. | SSE is effectively dead. |
| `graph_service.py:29-32` and `docs/ingestion-pipeline.md` | Extraction batches can run concurrently with `max_concurrency=4`. | `build_graph_for_document()` creates `asyncio.Semaphore(1)` at line 547, so extraction is sequential. | Comment/docs imply 4; code uses 1. |
| `docs/settings-migration-plan.md:36-40` and `:151` | Ingestion settings must be app-level because DataStores are shared. | `backend/app/core/settings_registry.py` correctly marks these `scope="app"`. | Consistent.  However there is no re-index workflow when these change despite `requires_reindex=True`. |
| `docs/file-changes-detection-ingestion.md:303-318` and `:82-84` | Two-class architecture is intentional. | `DatastoreFileEventHandler` and `DataStoreWatcher` are real, but `_on_changes` is a callback from watcher to handler and back, making the flow hard to follow. | Architecture is implemented but needs clearer ownership documentation. |

---

## 11. Ranked findings

Findings are grouped by severity and ordered by a rough effort/impact score.  Each item cites the primary code location and a recommended fix direction.

### 11.1 Critical

**C1. Duplicate `ProcessingTask` creation and client-controlled OCR**
- Where: `backend/app/api/api_v1/knowledge_base.py:405-411`, `452-474`.
- What: the `process` endpoint does filter `DocumentUpload.id` by the requested `kb_id` and rejects unknown IDs, so cross-KB upload IDs are not accepted.  However, it does not reject duplicate `upload_id` values in the same request, does not take a row-level lock or idempotency key, and does not check whether an upload is already being processed, so a single request or concurrent requests can create multiple `ProcessingTask` rows for the same upload.  It also trusts the client-supplied `enable_ocr` flag and builds `task_data` from the client’s `upload_results` list rather than from the validated `DocumentUpload` rows.
- Fix: de-duplicate `upload_ids`, take a lock or use an idempotency key, and derive `file_name`/`temp_path`/`enable_ocr` exclusively from `DocumentUpload` records; reject in-progress or already-completed uploads unless explicitly allowed.

**C2. Missing file-size enforcement**
- Where: `backend/app/api/api_v1/knowledge_base.py:254-330`, `backend/app/services/ingestion/document_converter.py:18`.
- What: `MAX_FILE_SIZE = 10 MB` is declared but never enforced; the endpoint reads the whole file into memory.
- Fix: check `len(file_content) <= MAX_FILE_SIZE` before saving; consider streaming uploads for large files.

**C3. Overly broad DataStore document access**
- Where: `backend/app/api/api_v1/knowledge_base.py:703-729`.
- What: `_check_document_access` for DataStore documents only requires the user to own any KB linked to the DataStore, not the actual linked path.  This is weaker than the org/DataStore scoping in `backend/app/services/retrieval/retrieval.py`.
- Fix: align document-by-ID access with retrieval scoping — require the DataStore to be either assigned to the user’s org or linked to a specific KB the user owns.

**C4. Direct-upload duplicate race**
- Where: `backend/app/api/api_v1/knowledge_base.py:286-300`, `backend/app/models/knowledge.py:69-72`.
- What: duplicate detection is read-then-write; the unique constraint is only on `(knowledge_base_id, file_name)`.  Two concurrent uploads of the same filename with different content can race; two uploads of the same file (same hash) can both pass the `.first()` check and insert, one failing the unique constraint.
- Fix: add a unique constraint on `(knowledge_base_id, file_name, file_hash)` and handle `IntegrityError` gracefully.

### 11.2 High

**H1. No durable task queue**
- Where: `backend/app/api/api_v1/knowledge_base.py:452-474`, `backend/app/services/datastore_watcher/handler.py:854-870`, `backend/app/services/discovery/startup_recovery_service.py:380-420`.
- What: all background work is `asyncio.create_task` or executor `Future`; a process restart can lose work or leave tasks stuck.
- Fix: introduce a durable queue (Celery, RQ, Dramatiq, or a DB-backed worker model) with retries and idempotent jobs.

**H2. Four overlapping ingestion orchestrators**
- Where: direct upload (`knowledge_base.py`), watcher (`handler.py`), manual scan (`watcher.py`), startup recovery (`startup_recovery_service.py`).
- What: each caller duplicates task creation, progress updates, loop creation, error handling, and completion logic around `process_document_background`.
- Fix: create one idempotent `IngestionJob` service that all triggers call.

**H3. Frontend DataStore filtering by client, not org**
- Where: `frontend/src/app/dashboard/knowledge/new/page.tsx:27-37`, `frontend/src/app/dashboard/knowledge/[id]/page.tsx:66-77`, `frontend/src/lib/auth.ts:7-9`.
- What: the frontend filters DataStores by `assigned_orgs.length > 0` and never checks the user’s `org_id`.  The `getTokenClaims()` helper is a stub.
- Fix: either add `org_id` to the `/api/admin/datastores` query, or have the backend filter by user org; fix `getTokenClaims()` to parse the JWT.

**H4. Silent DataStore-link failures**
- Where: `frontend/src/app/dashboard/knowledge/new/page.tsx:63-72`.
- What: link failures are caught and logged; the user is not told, so a KB can be created without the selected DataStores.
- Fix: report link failures in the UI and allow retry, or make KB creation and linking a single transaction.

**H5. `last_scan_processed` semantics overloaded**
- Where: `backend/app/services/datastore_watcher/handler.py:1193-1211`, `backend/app/services/datastore_watcher/watcher.py:379-413`, `backend/app/api/api_v1/datastores.py:89-119`.
- What: the same DB field is incremented by both manual scans and event-driven processing, and the UI cannot distinguish the two.
- Fix: split scan counters into manual/event/recovery dimensions, or keep a separate event-driven counter.

**H6. Manual scan and realtime event races**
- Where: `backend/app/services/datastore_watcher/handler.py:285-315`, `backend/app/services/datastore_watcher/watcher.py:526-668`.
- What: the two paths can process the same file concurrently.  The hash-unchanged skip is not atomic with ingestion.
- Fix: use a per-(datastore, file) advisory lock or `INSERT ... ON DUPLICATE KEY UPDATE` semantics.

**H7. Upload `process` endpoint accepts arbitrary `enable_ocr` per file without vision-model validation**
- Where: `backend/app/api/api_v1/knowledge_base.py:449-450`, `backend/app/services/ingestion/document_converter.py:149-160`.
- What: a client can request OCR when no `VISION_MODEL` is configured; the code only logs a warning and falls back to text-only.
- Fix: return `400` if `enable_ocr=True` and `VISION_MODEL` is not configured; otherwise make the fallback explicit in the UI.

### 11.3 Medium

**M1. `_chunk_id_to_point_id` duplicated**
- Where: `backend/app/services/ingestion/document_qdrant.py:95-97`, `backend/app/services/graph/graph_service.py:116-123`.
- What: the same deterministic UUIDv5 function exists in two modules.
- Fix: move to a single utility module and import everywhere.

**M2. Qdrant/Neo4j cleanup failures silently ignored**
- Where: `backend/app/services/datastore_watcher/handler.py:1031-1060`, `backend/app/services/cleanup/deletion_service.py:55-153`, `backend/app/services/discovery/startup_recovery_service.py:509-540`.
- What: non-404 Qdrant or Neo4j errors are logged and processing continues, which can leave orphaned vectors/nodes.
- Fix: distinguish expected 404s from transient/retryable errors; retry transient errors and surface persistent failures.

**M3. Defensive Neo4j sweeps suggest transaction uncertainty**
- Where: `backend/app/services/graph/graph_service.py:827-881`, `884-979`.
- What: `delete_graph_for_document()` and `delete_graph_for_kb()` run duplicate `DETACH DELETE` passes and entity sweeps.
- Fix: make the first deletion idempotent and remove the defensive second sweep once correctness is verified.

**M4. Failure cleanup targets wrong path**
- Where: `backend/app/services/ingestion/document_processor.py:550-557`.
- What: on KB failure the code deletes `permanent_path`, which may not exist if `move_file()` failed; the temp file is left behind.
- Fix: track the actual current file path and delete that; also delete `DocumentUpload` and any committed `Document` on failure.

**M5. Progress updates share the main DB session**
- Where: `backend/app/services/ingestion/document_processor.py:197-209`.
- What: `_set_progress` commits on the same session as the main transaction; if the main transaction later rolls back, progress values survive.
- Fix: use a separate DB session for progress writes, or emit progress via a side channel (Redis, in-memory task object).

**M6. Org hierarchy traversal capped at 100 levels**
- Where: `backend/app/api/api_v1/knowledge_base.py:31-32`, `917-928`; `backend/app/core/security.py:22-44`.
- What: BFS has a hardcoded 100-iteration limit.
- Fix: use a recursive CTE or configure the cap explicitly.

**M7. `DocumentChunk` delete-old filter is a Python expression, not SQL**
- Where: `backend/app/services/ingestion/document_processor.py:362-365`, `377-380`.
- What: `DocumentChunk.data_store_id == data_store_id if data_store_id else DocumentChunk.kb_id == kb_id` is evaluated as a Python ternary, not two SQL branches.
- Fix: build the SQL filter explicitly: `or_(DocumentChunk.data_store_id == data_store_id, DocumentChunk.kb_id == kb_id)` depending on which is set.

**M8. `getTokenClaims()` is a stub**
- Where: `frontend/src/lib/auth.ts:7-9`.
- What: the function always returns `null`; no frontend code can read `org_id` from the JWT.
- Fix: implement JWT parsing with signature verification if needed, or expose `/me` and org context from the backend.

**M9. Frontend mixed `fetch` and `api` client**
- Where: `frontend/src/app/dashboard/admin/data-sources/page.tsx:194-196`, `337-340`, `467-471`.
- What: raw `fetch()` bypasses the shared `api` client’s 401 handling and `ApiError` typing.
- Fix: route all calls through the `api` client.

**M10. Manual scan polling blocks after unmount**
- Where: `frontend/src/app/dashboard/admin/data-sources/page.tsx:351-410`.
- What: the `while` loop only checks `scanPollRef.current.active` at the top and after each 500 ms sleep; if the component unmounts mid-loop, it continues until the 2-minute timeout.
- Fix: use `setTimeout` recursion and a cleanup flag, and/or abort the `fetch`.

### 11.4 Low

**L1. Two upload UI implementations**
- Where: `frontend/src/components/knowledge-base/document-upload-steps.tsx`, `frontend/src/app/dashboard/knowledge/[id]/upload/page.tsx`.
- What: the legacy page has worse polling and no OCR toggles.
- Fix: remove the legacy page and route `/dashboard/knowledge/[id]/upload` to the new component.

**L2. Inconsistent `assigned_orgs` shape**
- Where: `frontend/src/app/dashboard/admin/data-sources/page.tsx:50` (`{id,name}`) vs `frontend/src/app/dashboard/knowledge/new/page.tsx:14` (`{org_id,org_name}`) vs `frontend/src/app/dashboard/knowledge/[id]/page.tsx:36` (`{org_id,org_name}`).
- What: TypeScript types are inconsistent with the backend response.
- Fix: share one `DataStore` type; use the backend shape (`{id,name}`).

**L3. Dead columns in migrations**
- Where: `backend/alembic/versions/0001_add_watch_dir_to_organisations.py`, `backend/alembic/versions/0002_add_smb_fields_to_organisations.py`.
- What: `watch_dir` and SMB fields are dead or no-ops.
- Fix: add a cleanup migration or document them as intentionally unused.

**L4. Multiple file-counting implementations**
- Where: `backend/app/api/api_v1/datastores.py:314-330`, `backend/app/services/datastore_watcher/handler.py:1213-1232`, `backend/app/services/discovery/discovery_engine.py:205-211`.
- What: the same pattern-matching file-count logic is duplicated.
- Fix: move to a shared utility.

**L5. `auto_scan_interval_minutes` exposed but unused**
- Where: `backend/app/models/datastore.py:46`, `frontend/src/app/dashboard/admin/data-sources/page.tsx:932-951`.
- What: users can configure an interval that has no effect.
- Fix: either implement timed flushing or remove the field from the UI and eventually from the model.

---

## 12. Recommended target architecture and remediation plan

### 12.1 Short-term (security and correctness)

1. **Lock and validate uploads in the process endpoint.**  Use `SELECT ... FOR UPDATE` or an idempotency key to prevent concurrent `process` calls from creating duplicate `ProcessingTask`s.  Derive `task_data` from the validated `DocumentUpload` rows, not from client `upload_results` metadata, and reject already-completed/in-progress uploads unless the user explicitly requests reprocessing.
2. **Enforce upload size limits.**  Check `len(file_content) <= MAX_FILE_SIZE` in `upload_kb_documents` and in `process_document_background` before conversion.
3. **Tighten DataStore document access.**  Rewrite `_check_document_access` so a DataStore document is accessible only if the DataStore is assigned to the user’s org or explicitly linked to a KB the user owns.
4. **Fix duplicate-upload race.**  Add a unique constraint on `(knowledge_base_id, file_name, file_hash)` and handle `IntegrityError` with an idempotent response.
5. **Validate OCR requests.**  Return `400` when `enable_ocr=True` and `VISION_MODEL` is not configured.
6. **Fix frontend DataStore org filtering.**  Server-side filter `/api/admin/datastores` by the user’s `org_id` (descendants for admins), and remove client-side filters.
7. **Surface DataStore link failures.**  If any link call fails during KB creation, show a clear warning and allow retry.

### 12.2 Medium-term (orchestration and reliability)

1. **Centralise ingestion dispatch.**  Create an `IngestionJob` domain object stored in MySQL with at-least-once semantics.  All triggers (KB upload, watcher, manual scan, recovery) should enqueue a job, not call `process_document_background` directly.
2. **Introduce a durable queue.**  Replace `asyncio.create_task` and ad-hoc executor loops with a worker queue (Celery/RQ/Dramatiq or a DB-backed worker).  Workers should be idempotent and retry transient failures.
3. **Make progress state separate from the ingest transaction.**  Use a distinct `Progress` table or Redis stream for progress; do not commit the main `db` session inside `_set_progress`.
4. **Unify cleanup logic.**  Move Qdrant/Neo4j/DB deletion into one `delete_document()` service and have the handler, recovery, and `deletion_service` call it.
5. **Centralise `_chunk_id_to_point_id`.**  Move to a single utility module and remove the duplicate in `graph_service.py`.
6. **Fix temp-file cleanup on failure.**  Track actual file location and delete from there; also delete `DocumentUpload` and any committed `Document` on failure.
7. **Align progress counters.**  Separate manual scan, event-driven, and recovery counters; do not increment `last_scan_processed` from event-driven code.
8. **Implement `getTokenClaims()`.**  Parse the JWT on the client to obtain `org_id`, role, and username; use it for defensive client-side filtering and org context.

### 12.3 Long-term (scalability and architecture)

1. **Re-index workflow for app-level ingestion settings.**  When `CHUNK_SIZE`, `OVERLAP_PERCENTAGE`, `GRAPHRAG_*`, `VISION_MODEL`, or embedding model/dim change, trigger a re-index of affected collections rather than silently producing mixed-index data.
2. **Per-DataStore or per-KB collection isolation.**  Currently DataStore vectors live in one collection per DataStore.  If per-org settings ever become necessary, the collection model must be redesigned (e.g. per-org collections or per-KB collections).
3. **Graph deletion correctness.**  Simplify `delete_graph_for_document`/`delete_graph_for_kb` to a single transactional Cypher pass and remove defensive sweeps after verifying correctness.
4. **Remove dead code.**  Delete `_start_batch_timer`, `_stop_batch_timer`, `_flush_batch`, the legacy upload page, and the dead `watch_dir`/`smb` columns.
5. **SSE or WebSocket progress.**  Fix the Next.js rewrite buffering so `scan-progress-stream` can be used, or replace polling with WebSocket/Socket.IO.
6. **Move to a recursive CTE for org hierarchy.**  Replace the 100-iteration BFS with a CTE or materialised closure table.

### 12.4 Verification checklist

- [ ] Unit test for `upload_kb_documents` rejecting oversized files.
- [ ] Unit test for `process_kb_documents` rejecting duplicate upload IDs and deriving task data from `DocumentUpload` rows.
- [ ] Unit test for duplicate upload race under concurrent requests.
- [ ] Unit test for `_check_document_access` with shared DataStores and multiple orgs.
- [ ] Integration test for watcher event → Document → Qdrant → Neo4j end-to-end.
- [ ] Integration test for manual scan + event-driven overlap.
- [ ] Frontend type-check after unifying `assigned_orgs` shape.
- [ ] Frontend tests for DataStore filtering by user org.
- [ ] Load test for concurrent file drops in a large DataStore.

---

## 13. Conclusion

The repository implements a capable multi-tenant RAG ingestion system with a clear separation between KB uploads and watched DataStore folders.  The core conversion/chunk/embed pipeline is sound, and the data model supports the intended sharing model.  The immediate risks, however, are in orchestration and access control: the four independent ingestion triggers duplicate lifecycle code, the upload and process endpoints have validation gaps that allow oversized files and duplicate tasks, the DataStore access check is looser than the retrieval model, and the frontend relies on the backend for all multi-tenancy checks while also doing unsafe client-side DataStore filtering.

The most important remediation is to centralise ingestion dispatch and validation, add a durable queue, and close the security gaps in upload ownership and DataStore document access.  Those changes will make the system robust enough to support the documented multi-tenant, self-hosted architecture.

---
