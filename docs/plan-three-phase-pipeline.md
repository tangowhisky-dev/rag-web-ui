# Three-Phase Ingestion Pipeline + Markdown Editor

## Architecture Decision Record

### Status
Approved — 2026-02-27

### Decisions
1. **Async pipeline functions** — `convert_document()` and `ingest_document()` are `async def`, matching the current `process_document_background` architecture. `run_ingestion_in_thread()` keeps its asyncio loop wrapper.
2. **Both KB and datastore documents get `converted_markdown`** — the column is populated for all documents during ingestion. The editor and re-convert work for both. KB files are moved to permanent storage (`{UPLOAD_DIR}/user_{user_id}/kb_{kb_id}/{file_name}`) on success and are available for re-convert. Files are only deleted on processing failure (cleanup).
3. **Graph build dispatch stays in `run_ingestion_in_thread`** — `process_document_full` returns `Optional[GraphBuildRequest]` like the current `process_document_background`. `run_ingestion_in_thread` handles the graph build dispatch logic (task status check, datastore-deleted check, graph-paused check).

### Context
The current ingestion pipeline converts a file to markdown, chunks it, embeds the chunks, and discards the source markdown. There is no way for an admin to review or correct the converted text. OCR artifacts (misread characters, broken table formatting, missing text) propagate through chunking and embedding with no human-in-the-loop correction point.

The user wants a two-pane markdown editor (editable source | rendered preview) that lets admins fix conversion artifacts before re-ingesting. This requires the converted markdown to be persisted.

### Decision: Three-phase pipeline with MySQL LONGTEXT storage

**Phases:**
```
Phase 1: CONVERT     →  Phase 2: INGEST     →  Phase 3: GRAPH
file → markdown         markdown → chunks      chunks → Neo4j
                        (embed + Qdrant)
```

**Storage for converted markdown: MySQL LONGTEXT column on Document.**

### Storage deliberation

Three options were considered for storing converted markdown:

#### Option A: MySQL LONGTEXT column on Document (RECOMMENDED)

Add `converted_markdown LONGTEXT` and `conversion_status VARCHAR(20)` to the `documents` table.

**Pros:**
- Zero new infrastructure. MySQL is already running, already backed up, already managed.
- Single query to load a document's markdown (no network hop, no second client library).
- Transactional consistency: markdown save + chunk deletion + status update in one DB transaction. If re-ingest fails, the markdown is still there.
- Existing patterns: `chunk_text` is already LONGTEXT, `messages.content` is LONGTEXT, `chats.history_summary` is LONGTEXT. This is a proven pattern in this codebase.
- MySQL's `max_allowed_packet` is 64MB — more than enough for any single document's markdown.
- LONGTEXT supports up to 4GB. The largest document in the current dataset is 20K chars of markdown. Even a 500-page PDF would produce ~500K chars — well within limits.

**Cons:**
- Large text in InnoDB can cause page splits and slower scans if the column is in the row buffer. Mitigation: MySQL 8+ stores LONGTEXT off-page by default when it exceeds `innodb_page_size` (16KB). The row pointer stays in the buffer pool; the text is loaded on demand. This is the same mechanism already used for `chunk_text`.
- Not full-text searchable without a FULLTEXT index. But we don't need to search the markdown — we search the chunks. The markdown is for editing, not retrieval.

#### Option B: OpenSearch / Meilisearch (REJECTED)

**Pros:**
- Full-text search on converted markdown.
- Purpose-built for text storage.

**Cons:**
- New infrastructure to deploy, monitor, back up, and upgrade. This is a significant operational burden for a feature that doesn't need search.
- Network hop for every editor load (latency, failure modes).
- No transactional consistency with chunk deletion. If OpenSearch write succeeds but MySQL chunk deletion fails, state diverges.
- The editor needs exact text retrieval, not search. Search is already handled by Qdrant + MySQL FULLTEXT on chunks.
- Adds a second source of truth for document content.

#### Option C: Disk storage mirroring datastore directory structure (REJECTED)

Store markdown at `{UPLOAD_DIR}/converted/{datastore_id}/{relative_path}.md`.

**Pros:**
- No database size impact.
- Natural backup with filesystem.
- Easy to inspect manually.

**Cons:**
- No transactional consistency with chunk/vector/graph state. File write + DB update is not atomic.
- Filesystem operations are not crash-safe without write-ahead logging. A crash between "write markdown" and "delete chunks" leaves inconsistent state.
- Path management complexity: need to mirror the exact directory structure, handle renames, handle deletions, handle datastore deletion cleanup.
- No metadata (conversion status, timestamp, hash) without a DB column anyway — so you still need a DB column, just not the text itself. The complexity savings are minimal.
- Backup must be coordinated between DB and filesystem — two backup systems instead of one.
- The existing codebase already stores large text in MySQL (chunk_text, message content). Adding filesystem storage introduces a new pattern that doesn't match existing conventions.

### Decision rationale

MySQL LONGTEXT is the right choice because:
1. The access pattern is point-lookup by document ID, not search.
2. Transactional consistency with chunk/vector/graph state is critical for the re-ingest flow.
3. The codebase already has this pattern (chunk_text, messages.content).
4. No new infrastructure to operate.
5. The largest expected markdown (~500K chars for a 500-page PDF) is well within LONGTEXT limits.

---

## Implementation Plan

### 1. Database Migration

**File:** `backend/alembic/versions/f1a2b3c4d5e6_add_converted_markdown.py`

Add columns to `documents`:

```python
def upgrade():
    op.add_column('documents', sa.Column('converted_markdown', sa.LONGTEXT(), nullable=True))
    op.add_column('documents', sa.Column('conversion_status', sa.String(20), nullable=True, index=True))
    op.add_column('documents', sa.Column('conversion_error', sa.Text(), nullable=True))
    op.add_column('documents', sa.Column('lock_version', sa.Integer(), nullable=False, server_default='0'))

def downgrade():
    op.drop_column('documents', 'lock_version')
    op.drop_column('documents', 'conversion_error')
    op.drop_column('documents', 'conversion_status')
    op.drop_column('documents', 'converted_markdown')
```

**States:**
- `conversion_status = 'pending'`: conversion queued/in progress
- `conversion_status = 'completed'`: markdown ready for edit
- `conversion_status = 'error'`: conversion failed, `conversion_error` has message
- `conversion_status = NULL`: legacy document (pre-3-phase) — recovery will queue conversion

**Backfill strategy:** Do NOT backfill `converted_markdown` for legacy documents. Set `conversion_status = NULL` for all existing rows. On startup, the recovery service queues conversion for any document with `conversion_status IS NULL` and `is_selected = True`.

### 2. Backend Model Changes

**File:** `backend/app/models/knowledge.py`

```python
class Document(Base, TimestampMixin):
    # ... existing columns ...
    converted_markdown = Column(LONGTEXT, nullable=True)
    conversion_status = Column(String(20), nullable=True, index=True)
    conversion_error = Column(Text, nullable=True)
    # Optimistic-lock version for editor save. server_default ensures
    # existing rows get 0 during migration (NOT NULL without default fails).
    lock_version = Column(Integer, default=0, server_default='0', nullable=False)
```

The `lock_version` column enables optimistic locking for concurrent edits. The `PUT /markdown` endpoint will use `WHERE id = :id AND lock_version = :expected_version` and increment the version on successful save, rejecting with 409 if the version changed.

### 3. Backend Pipeline Split

The current `process_document_background()` in `document_processor.py` does everything in one async function. Split into three independent **async** functions that can be called separately or in sequence. They run inside the existing `run_ingestion_in_thread()` asyncio event loop wrapper.

#### Phase 1: Convert

**File:** `backend/app/services/ingestion/document_processor.py`

```python
async def convert_document(
    document_id: int,
    file_path: str,
    file_name: str,
    enable_ocr: Optional[bool] = None,
    data_store_id: Optional[int] = None,
    kb_id: Optional[int] = None,
) -> str:
    """Convert a file to markdown and store it in Document.converted_markdown.

    Sets conversion_status to 'completed' on success, 'error' on failure.
    Returns the markdown text on success, raises on failure.
    Does not touch chunks, vectors, or graph.

    Async — uses loop.run_in_executor for the synchronous _convert_to_markdown call,
    matching the current process_document_background pattern.
    """
    # 1. Open DB session (or use passed session)
    # 2. Load Document, set conversion_status='processing'
    # 3. Call _convert_to_markdown(file_path, file_name, enable_ocr) via run_in_executor
    # 4. Run clean_markdown() cleanup pass
    # 5. Call extract_title(markdown, file_name, abs_path=file_path) to derive title
    # 6. Store markdown in Document.converted_markdown, update Document.title
    # 7. Set conversion_status='completed'
    # 8. Commit and return markdown
    # On any exception: rollback, set conversion_status='error', store conversion_error, DO NOT delete Document
```

**Important:** `convert_document` must never delete the `Document` record on failure. The `Document` record must survive conversion errors so the admin can see the error and retry. This is a change from the current `process_document_background` behavior, which deletes the Document if created and a later step fails.

**Signature notes:** `content_type` and `title` are NOT parameters — `content_type` is only for the Document record (already set by the caller), and `title` is derived from the markdown by `extract_title()`.

#### Phase 2: Ingest (chunk + embed + Qdrant)

```python
async def ingest_document(
    document_id: int,
    file_name: str,
    data_store_id: Optional[int] = None,
    kb_id: Optional[int] = None,
    task_id: Optional[int] = None,
    markdown_text: Optional[str] = None,
    file_path: Optional[str] = None,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> Optional[GraphBuildRequest]:
    """Chunk a document's markdown, embed, and store in Qdrant.

    If markdown_text is None, reads from Document.converted_markdown.
    Deletes existing chunks (MySQL + Qdrant) for this document before re-chunking.
    Updates ProcessingTask.status to 'completed' on success, 'failed' on failure.
    Returns a GraphBuildRequest for the caller to queue phase 3.

    Async — uses run_in_executor for chunking and await for Qdrant upsert,
    matching the current process_document_background pattern.

    progress_cb: optional callback(pct, message) for progress updates.
    The caller (run_ingestion_in_thread) passes _set_progress from the
    ProgressTimeout mechanism so ProcessingTask.progress is updated.
    """
    # 1. Open DB session; load Document and ProcessingTask by task_id
    # 2. Check is_datastore_deleted(data_store_id) — abort if deleted
    # 3. If markdown_text is None, read from Document.converted_markdown
    # 4. Delete existing DocumentChunk rows for this document
    # 5. Delete existing Qdrant points for this document
    # 6. Chunk with RecursiveCharacterTextSplitter (via run_in_executor)
    # 7. Embed + upsert to Qdrant (await _upsert_to_qdrant)
    # 8. Store DocumentChunk rows
    # 9. Update ProcessingTask.status='completed', set task.document_id if not already set
    # 10. Return GraphBuildRequest(...)
    # On failure: rollback, set ProcessingTask.status='failed', store error_message, DO NOT delete Document or markdown
```

**Key differences from current `process_document_background`:**
- `ingest_document` reads `converted_markdown` from the DB (or accepts it as an argument), rather than converting inline.
- It deletes chunks/vectors before re-chunking, but does not delete the `Document` or the `ProcessingTask` on failure.
- It accepts a `progress_cb` callback so the caller's `ProgressTimeout` mechanism still works.
- It returns a `GraphBuildRequest` so the caller can queue phase 3.

#### Phase 3: Graph (already exists, no orchestration change)

The graph build already runs separately via `run_graph_build_in_thread()`. No change — `ingest_document` returns a `GraphBuildRequest` and `run_ingestion_in_thread` handles the graph build dispatch (task status check, datastore-deleted check, graph-paused check). This is the existing behavior.

#### Orchestrator (replaces current monolithic flow)

```python
async def process_document_full(
    temp_path: str,
    file_name: str,
    kb_id: Optional[int] = None,
    data_store_id: Optional[int] = None,
    file_path: Optional[str] = None,
    document_id: Optional[int] = None,
    task_id: Optional[int] = None,
    enable_ocr: Optional[bool] = None,
    # ... other params from current process_document_background (user_id, file_hash, file_size, content_type, chunk_size, chunk_overlap, db)
) -> Optional[GraphBuildRequest]:
    """Full pipeline: convert → ingest. Used by scan/watcher and event handler.

    Returns GraphBuildRequest (or None) — the caller (run_ingestion_in_thread)
    handles graph build dispatch, same as today.
    """
    # 1. Phase 1: convert
    markdown = await convert_document(
        document_id=document_id,
        file_path=temp_path or file_path,
        file_name=file_name,
        data_store_id=data_store_id,
        kb_id=kb_id,
        enable_ocr=enable_ocr,
    )

    # 2. Phase 2: ingest
    graph_request = await ingest_document(
        document_id=document_id,
        file_name=file_name,
        data_store_id=data_store_id,
        kb_id=kb_id,
        task_id=task_id,
        markdown_text=markdown,
        file_path=file_path,
        progress_cb=_set_progress,  # from ProgressTimeout context
    )

    # 3. Return GraphBuildRequest — run_ingestion_in_thread handles phase 3
    return graph_request
```

**Note:** `process_document_full` replaces `process_document_background` as the function called by `run_ingestion_in_thread`. The `run_ingestion_in_thread` function itself is unchanged — it creates the asyncio loop, calls `process_document_full`, marks the task completed, and fires graph build. The `ProgressTimeout` context manager and `_set_progress` mechanism stay inside `process_document_full` (or are passed through to `ingest_document`).

For the scan/watcher path, the `Document` and `ProcessingTask` are created by `_ingest_file_in_scan` or `_update_document_in_scan` before `process_document_full` is called. `convert_document` updates the existing `Document` record. `ingest_document` updates the existing `ProcessingTask` record.

### 4. Re-ingest Primitive

**File:** `backend/app/services/datastore/document_management.py` (or new `backend/app/services/ingestion/reingest.py`)

Add `reset_document_for_reingest(db, document_id, data_store_id)`:

```python
def reset_document_for_reingest(
    db: Session,
    document_id: int,
    data_store_id: int,
) -> ProcessingTask:
    """Delete ingested data for a document but keep the document and manifest selected.

    Used by the markdown editor when admin saves edited text.
    Deletes chunks (MySQL + Qdrant), graph (Neo4j), and ProcessingTask,
    then creates a new pending ProcessingTask for re-ingest.
    Does NOT set is_selected=False or delete the manifest.
    """
    # 1. Delete Qdrant points for this document
    # 2. Delete DocumentChunk rows for this document
    # 3. Delete existing ProcessingTask rows for this document
    # 4. Delete Neo4j graph for this document (delete_graph_for_document)
    # 5. Create new ProcessingTask(status='pending', graph_status='pending')
    # 6. Return the new task
```

**Important:** This is different from `delete_document_data()` which is for unselecting and sets `is_selected=False`. `reset_document_for_reingest` is for re-ingesting a selected document.

**Graph build task guard:** When the old `ProcessingTask` is deleted, any in-flight `run_graph_build_in_thread` keyed by that `task_id` will find `None` when it queries the task from DB. The `run_graph_build_in_thread` function must be updated to check for `task is None` and exit cleanly with a log message, rather than crashing. This is a small guard addition to `ingestion_dispatcher.py`.

### 5. Backend API Endpoints

**File:** `backend/app/api/api_v1/datastore_documents.py` (new)

Register in `backend/app/api/api_v1/api.py`:
```python
api_router.include_router(datastore_documents.router, prefix="/admin", tags=["datastore-documents"])
```

All endpoints use `require_admin` and `_datastore_in_scope` authorization patterns from `datastores.py`.

#### GET `/api/admin/datastores/{ds_id}/documents/{doc_id}/markdown`

Returns the converted markdown for editing.

```json
{
  "document_id": 123,
  "file_name": "datasheet.pdf",
  "conversion_status": "completed",
  "markdown": "# RF Counter-Unmanned Aerial System\n\n...",
  "chunk_count": 5,
  "ingest_status": "completed",
  "graph_status": "completed",
  "lock_version": 3,
  "conversion_error": null
}
```

**Status codes:**
- 200: OK
- 404: Document or datastore not found
- 403: Admin not in scope
- 409: `conversion_status != 'completed'` — editor cannot open until conversion is ready

#### PUT `/api/admin/datastores/{ds_id}/documents/{doc_id}/markdown`

Saves edited markdown and triggers re-ingest.

Request body:
```json
{
  "markdown": "# Corrected title\n\nFixed text...",
  "lock_version": 3
}
```

**Flow:**
1. Validate markdown is non-empty.
2. Load `Document` by id. Verify `conversion_status = 'completed'`.
3. Optimistic locking: `WHERE id = :id AND lock_version = :lock_version`.
4. Update `Document.converted_markdown`, increment `lock_version`, set `conversion_status='completed'`.
5. Call `reset_document_for_reingest()` to delete chunks/vectors/graph and create a new `ProcessingTask`.
6. Submit `run_ingestion_in_thread` with the new task, which will:
   - Call `ingest_document()` (phase 2)
   - Then queue `run_graph_build_in_thread` (phase 3)
7. Return 202 Accepted with `task_id` and `lock_version` for polling.

**Status codes:**
- 202: Accepted (re-ingest queued)
- 400: Empty markdown
- 403: Not in scope
- 404: Not found
- 409: Lock version mismatch (admin should reload)

#### POST `/api/admin/datastores/{ds_id}/documents/{doc_id}/reconvert`

Re-runs phase 1 (file → markdown) from the source file. Overwrites the current markdown.

**Flow:**
1. Load `Document`. Set `conversion_status = 'pending'`, `conversion_error = None`.
2. Submit `convert_document()` in a thread pool.
3. Return 202 Accepted.
4. UI polls `GET .../markdown` until `conversion_status` is `completed` or `error`.

#### GET `/api/admin/datastores/{ds_id}/documents/{doc_id}/ingest-status`

Polls the re-ingest progress after a save or re-convert.

```json
{
  "document_id": 123,
  "conversion_status": "completed",
  "ingest_status": "processing",
  "ingest_progress": 45,
  "ingest_message": "Embedding 5 chunks...",
  "graph_status": "pending",
  "is_complete": false
}
```

### 6. Markdown Editor UI

**Route:** `/dashboard/admin/data-sources/[id]/documents/[docId]`

**File:** `frontend/src/app/dashboard/admin/data-sources/[id]/documents/[docId]/page.tsx`

#### Layout

```
┌─────────────────────────────────────────────────────────────┐
│  ← Back to files   │  datasheet.pdf   │  [Save] [Re-convert]│
├──────────────────────────┬──────────────────────────────────┤
│                          │                                  │
│   EDITOR (textarea)      │   PREVIEW (react-markdown)       │
│                          │                                  │
│   # RF Counter-UAS       │   RF Counter-UAS                 │
│                          │   ==============                 │
│   Specifications         │   Specifications                 │
│                          │                                  │
│   COVERAGE 433/868 MHz   │   COVERAGE 433/868 MHz           │
│   ...                    │   ...                            │
│                          │                                  │
├──────────────────────────┴──────────────────────────────────┤
│  Convert ●   Ingest ●   Graph ●   Chunks: 5                 │
│  ⚠ Unsaved changes — Save to re-ingest                      │
└─────────────────────────────────────────────────────────────┘
```

On viewports < 1024px, switch to a tabbed "Editor | Preview" view instead of a side-by-side split.

#### State Management

```typescript
interface EditorState {
  markdown: string;
  originalMarkdown: string;
  loading: boolean;
  saving: boolean;
  converting: boolean;
  lockVersion: number;
  conversionStatus: string | null;
  ingestStatus: string | null;
  ingestProgress: number;
  ingestMessage: string;
  graphStatus: string | null;
  showUnsavedDialog: boolean;
}

const isDirty = markdown !== originalMarkdown;
```

#### Real-time Preview

The preview pane uses a new `MarkdownPreview` component (see section 8). The editor uses a native textarea. Preview is debounced 300ms for large documents:

```typescript
const [debouncedMarkdown, setDebouncedMarkdown] = useState(markdown);
useEffect(() => {
  const timer = setTimeout(() => setDebouncedMarkdown(markdown), 300);
  return () => clearTimeout(timer);
}, [markdown]);
```

#### Save Flow

1. User clicks Save (or confirms via unsaved-changes dialog).
2. `setSaving(true)`.
3. `PUT .../markdown` with the edited markdown and current `lock_version`.
4. Backend returns 202 with `task_id` and new `lock_version`.
5. Update `lockVersion` and `originalMarkdown` to the saved markdown — clears dirty state.
6. Start polling `GET .../ingest-status` every 2s.
7. Show phase indicators in status bar.
8. When `is_complete === true`, stop polling.
9. `setSaving(false)`.

#### Re-convert Flow

1. User clicks "Re-convert" → confirmation dialog: "This will overwrite your current markdown with a fresh conversion. Unsaved edits will be lost."
2. If confirmed: `POST .../reconvert`.
3. `setConverting(true)`.
4. Poll `GET .../markdown` every 2s.
5. If `conversion_status === 'completed'`: load new markdown, `setConverting(false)`.
6. If `conversion_status === 'error'`: show error, `setConverting(false)`.
7. If the admin had unsaved edits, warn before starting the re-convert.

#### Unsaved Changes Guard

```typescript
// Block browser close/reload
useEffect(() => {
  const handler = (e: BeforeUnloadEvent) => {
    if (isDirty) {
      e.preventDefault();
      e.returnValue = "";
    }
  };
  window.addEventListener("beforeunload", handler);
  return () => window.removeEventListener("beforeunload", handler);
}, [isDirty]);

// Intercept Next.js navigation (all Links and router.push)
const router = useRouter();
const confirmLeave = (href: string) => {
  if (isDirty) {
    setShowUnsavedDialog(true);
    setPendingNavigation(href);
    return false;
  }
  return true;
};
```

Wrap all navigation in the editor (Back to files, Cancel in confirm dialog) with the guard. Do not rely solely on `beforeunload` — App Router client-side navigation does not trigger it.

### 7. Frontend: File Browser Integration

**File:** `frontend/src/app/dashboard/admin/data-sources/[id]/page.tsx`

#### Backend browse endpoint update

The `GET /api/admin/datastores/{id}/browse` endpoint already returns `document_id`, `status` (ingest), `graph_status`, `chunk_count`, `title`, and `error_message` per file (see `document_management.py:179-196`). The only missing field is `conversion_status`.

Update `get_folder_contents()` in `document_management.py` to also return `conversion_status` from the `Document` record. The `graph_status` retrieval at line 193 (`doc.processing_tasks[0].graph_status`) is fragile — it should use the same `task_statuses` query pattern (latest task by `id.desc()`) for consistency.

Update `BrowseItem` interface:
```typescript
interface BrowseItem {
  // ... existing fields ...
  document_id?: number;
  conversion_status?: "pending" | "completed" | "error" | null;  // NEW
  status?: "pending" | "processing" | "completed" | "failed";     // already exists
  graph_status?: "pending" | "completed" | "failed" | null;       // already exists
}
```

#### Row rendering

1. Add an "Actions" column to the table.
2. The file row `onClick` toggles selection. The Edit button must call `e.stopPropagation()`.
3. Edit button is only enabled when `conversion_status === 'completed'`.
4. New 3-phase badge uses three dot indicators:
   - Convert: green if completed, red if error, amber if pending, gray if not started
   - Ingest: green if completed, red if failed, amber if pending/processing, gray if not started
   - Graph: green if completed, red if failed, amber if pending, gray if not started
5. Tooltip on the badge shows full text: "Convert: completed, Ingest: completed, Graph: pending".

### 8. New MarkdownPreview Component

**File:** `frontend/src/components/markdown/markdown-preview.tsx`

A reusable ReactMarkdown wrapper for the editor. The codebase already uses `react-markdown` in `frontend/src/components/chat/answer.tsx` (line 22) and `frontend/src/app/dashboard/search/page.tsx` (line 19) — this component should match their plugin configuration for consistency.

```typescript
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";

// NOTE: rehype-raw is NOT used. Admin-edited markdown can contain arbitrary
// HTML; we do not want to render it unsanitized. GFM + math + code highlighting
// are enough.

export function MarkdownPreview({ markdown, className }: { markdown: string; className?: string }) {
  return (
    <Markdown
      className={cn("prose prose-sm dark:prose-invert max-w-none", className)}
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[rehypeHighlight, rehypeKatex]}
    >
      {markdown}
    </Markdown>
  );
}
```

**Typography plugin required:** `@tailwindcss/typography` is NOT installed. The `prose` classes in the codebase are backed by hand-written CSS in `globals.css` (lines 119-137) that only styles headings. For the editor preview to render paragraphs, lists, tables, code blocks, and blockquotes correctly, either:
- Install `@tailwindcss/typography` (`npm install @tailwindcss/typography`) and add it to `tailwind.config.ts` plugins, OR
- Write comprehensive `.prose` CSS in `globals.css` for all elements the editor will render

Installing the plugin is the simpler path.

#### Dark mode for code blocks

The existing `layout.tsx` imports `highlight.js/styles/github.css` (light only). Update `layout.tsx` to conditionally load a dark theme:

```typescript
// In layout.tsx or a theme-aware style component
import "highlight.js/styles/github.css";
import "highlight.js/styles/github-dark.css"; // for .dark
```

And in `globals.css` add:

```css
.dark .hljs {
  background: hsl(var(--muted));
  color: hsl(var(--foreground));
}
```

Also load `katex/dist/katex.min.css` globally in `layout.tsx` so math renders correctly in both editor and chat.

### 9. Sequential Handling

#### Scan pipeline (automatic)

When a scan discovers new/modified files, it processes them sequentially:
1. `_ingest_file_in_scan` or `_update_document_in_scan` creates `Document` + `ProcessingTask`.
2. Calls `process_document_full()`:
   - `convert_document()` produces and stores `converted_markdown`
   - `ingest_document()` chunks, embeds, stores in Qdrant
   - `run_graph_build_in_thread()` builds Neo4j graph

#### Manual re-ingest from editor

1. `PUT .../markdown` updates `converted_markdown`.
2. Calls `reset_document_for_reingest()` to delete old chunks/vectors/graph and create a new `ProcessingTask`.
3. Submits `run_ingestion_in_thread()` which calls `ingest_document()` then queues graph build.
4. UI polls `ingest-status`.

#### Recovery

In `startup_recovery_service.py`, add a pass over `Document` rows for active datastores:
- For any document with `conversion_status IS NULL` or `conversion_status = 'error'` and `is_selected = True`:
  - Set `conversion_status = 'pending'`
  - Queue `convert_document()`
- For any document with `ProcessingTask.status in ('pending', 'processing')`:
  - Re-queue existing task (current behavior)
- For any document with `ProcessingTask.graph_status = 'pending'` and graph not paused:
  - Re-queue graph build (current behavior)

The order matters: conversions run first, then ingests, then graph builds.

### 10. Real-time Changes

#### Editor → Preview

Native textarea input is immediate. The preview re-renders 300ms after the user stops typing.

#### Save → Re-ingest Progress

Polling every 2s via `GET .../ingest-status`. No SSE required — the single-document re-ingest is short enough for polling.

#### File Browser → Editor

Next.js client-side navigation. The editor route fetches fresh data on load.

### 11. Edge Cases and Mitigations

| Edge Case | Mitigation |
|---|---|
| Empty markdown after edit | `PUT` endpoint rejects with 400. |
| Concurrent edits | Optimistic locking via `lock_version` on `Document`. `PUT` uses `WHERE id = :id AND lock_version = :version`. Mismatch returns 409. |
| Re-ingest while scan is running | Allowed. Re-ingest targets one document; scan processes others. No shared resource except Qdrant/Neo4j, which are scoped by document. |
| Re-ingest while graph build is in progress | `reset_document_for_reingest` deletes the old `ProcessingTask`. The old in-flight graph build thread will find `task is None` when it queries the DB. A guard must be added to `run_graph_build_in_thread` to exit cleanly with a log message when the task is `None`, rather than crashing. No need to call `cancel_graph_builds_for_datastore` (which cancels ALL documents in the datastore). |
| Document deleted while editor is open | Next API call returns 404. Show error and redirect to file browser. |
| Admin leaves with unsaved changes | `beforeunload` + in-app navigation guard with `ConfirmDialog`. |
| Conversion fails | `conversion_status = 'error'`, `conversion_error` has message. Admin can see error and click "Re-convert". |
| Re-chunking a manually edited 20,000-char markdown | Same chunking logic as before. No special handling needed. |
| Legacy document with `conversion_status = NULL` | Recovery queues conversion. Admin cannot edit until conversion completes. |
| Datastore deleted/paused during re-ingest | `ingest_document` and `convert_document` check `is_datastore_deleted()` and `graph_ingestion_paused` and abort. |
| Graph ingestion paused | New `ProcessingTask.graph_status` will be `pending` and stay `pending` until the datastore is unpaused, then recovery or manual scan will resume graph builds. |

### 12. Files to Create/Modify

#### Backend

| File | Action | Description |
|---|---|---|
| `backend/alembic/versions/f1a2b3c4d5e6_add_converted_markdown.py` | Create | Migration: add `converted_markdown`, `conversion_status`, `conversion_error`, `lock_version` |
| `backend/app/models/knowledge.py` | Modify | Add 4 new columns to Document |
| `backend/app/services/ingestion/document_processor.py` | Modify | Split into `convert_document()` + `ingest_document()` + `process_document_full()` |
| `backend/app/services/ingestion/ingestion_dispatcher.py` | Modify | Ensure `run_ingestion_in_thread` calls `process_document_full`; add `task is None` guard to `run_graph_build_in_thread` |
| `backend/app/services/ingestion/reingest.py` | Create | `reset_document_for_reingest()` helper |
| `backend/app/api/api_v1/datastore_documents.py` | Create | 4 API endpoints for markdown editor |
| `backend/app/api/api_v1/api.py` | Modify | Register new router |
| `backend/app/services/datastore/document_management.py` | Modify | Optionally move `reset_document_for_reingest` here or keep in reingest.py |
| `backend/app/services/datastore_watcher/watcher.py` | Modify | `_run_ingestion` calls `process_document_full()` instead of `run_ingestion_in_thread()` directly |
| `backend/app/services/datastore_watcher/handler.py` | Modify | `_run_ingestion` calls `process_document_full()` |
| `backend/app/services/discovery/startup_recovery_service.py` | Modify | Add conversion recovery; order: convert → ingest → graph |
| `backend/app/api/api_v1/datastores.py` | Modify | Update browse endpoint response to include `conversion_status` and `graph_status` per item |

#### Frontend

| File | Action | Description |
|---|---|---|
| `frontend/src/app/dashboard/admin/data-sources/[id]/documents/[docId]/page.tsx` | Create | Two-pane markdown editor page |
| `frontend/src/components/markdown/markdown-preview.tsx` | Create | Reusable ReactMarkdown wrapper |
| `frontend/src/app/dashboard/admin/data-sources/[id]/page.tsx` | Modify | Add Edit action + 3-phase dot badge + conversion_status handling |
| `frontend/src/app/layout.tsx` | Modify | Add `katex.min.css` and dark-mode highlight.js CSS |
| `frontend/src/app/globals.css` | Modify | Override `.hljs` for dark mode; extend `.prose` styles if not installing typography plugin |
| `frontend/src/components/ui/phase-dot-badge.tsx` | Create | 3-dot phase status badge component |
| `frontend/package.json` | Modify | Add `@tailwindcss/typography` dependency |
| `frontend/tailwind.config.ts` | Modify | Add typography plugin to plugins array |

### 13. Implementation Order

1. **Migration + model** — add columns (`converted_markdown`, `conversion_status`, `conversion_error`, `lock_version` with `server_default='0'`), verify existing data is unaffected.
2. **Backend pipeline split** — create `async convert_document()` and `async ingest_document()`. Replace `process_document_background` with `process_document_full` that calls them. Wire `run_ingestion_in_thread` to call `process_document_full`. Add `task is None` guard to `run_graph_build_in_thread`. Run a full datastore scan to verify nothing broke.
3. **Re-ingest primitive** — create `reset_document_for_reingest()` in `reingest.py`.
4. **Backend API** — add the 4 endpoints. Test with curl.
5. **Frontend editor** — install `@tailwindcss/typography`, build `MarkdownPreview` component, build the two-pane page with save/re-convert/unsaved guard. Test with a real document.
6. **Frontend file browser integration** — add Edit button + 3-phase badge + `conversion_status` to browse response.
7. **Recovery update** — handle `conversion_status = NULL` and `conversion_status = 'error'`. Verify recovery on restart.
8. **End-to-end test** — scan a datastore, open editor, edit markdown, save, verify re-ingest, verify graph rebuild, verify search results changed.

### 14. What This Does NOT Change

- Qdrant storage format (same collections, same point structure)
- Neo4j graph schema (same nodes, same relationships)
- Chunking logic and parameters (same RecursiveCharacterTextSplitter)
- Embedding logic (same model, same batch size)
- Scan/discovery mechanism (same watcher, same manifest)
- Pause/resume for scans (orthogonal to the 3-phase split)
- Graph pause/resume (stays as-is, independent per datastore)
- File selection/unselection UI (the editor is additive)
- `run_ingestion_in_thread` graph build dispatch logic (task status check, datastore-deleted check, graph-paused check)
- `ProgressTimeout` mechanism (stays inside `process_document_full`, passed through to `ingest_document` via `progress_cb`)
- KB direct upload flow (KB documents get `converted_markdown` stored and can be edited/re-converted — files persist in `{UPLOAD_DIR}/user_{user_id}/kb_{kb_id}/` after successful processing; files are only deleted on failure)

---

## Cross-Verification Findings

This plan was cross-checked against the actual backend and frontend code in two passes. The following gaps were found and corrected.

### Backend gaps found and fixed

1. **No re-ingest primitive existed.** `delete_document_data()` in `document_management.py` is for unselecting — it sets `is_selected=False` and removes the manifest. A new `reset_document_for_reingest()` is needed that deletes chunks/vectors/graph but keeps the document and manifest.

2. **Per-document graph cancellation is not supported.** `cancel_graph_builds_for_datastore()` cancels ALL builds for a datastore. The fix: `reset_document_for_reingest` deletes the old `ProcessingTask`, and a `task is None` guard is added to `run_graph_build_in_thread` so it exits cleanly.

3. **The current `process_document_background` deletes the `Document` on later failure.** This would erase `converted_markdown`. The new `convert_document()` and `ingest_document()` must never delete the `Document` record; they only set `conversion_status` or `ProcessingTask.status` to `error`/`failed`.

4. **`ingest_document` needs more parameters.** The chunking/embedding code needs `data_store_id`, `kb_id`, `file_name`, `task_id`, `file_path`, and `doc_title`. The signature was expanded. Also needs `progress_cb` for the `ProgressTimeout` mechanism.

5. **Re-convert needs an error field.** `conversion_status` alone is not enough; `conversion_error` is added for failure messages.

6. **`is_datastore_deleted` and `graph_ingestion_paused` guards must be kept.** These checks are in `process_document_background` and `run_graph_build_in_thread` and must be repeated in the new phase functions.

7. **Optimistic locking requires a real column.** The `Document` model gets `lock_version` with `server_default='0'` for migration safety.

8. **Recovery must handle legacy `conversion_status = NULL` documents.** The recovery service needs to queue conversion for pre-existing documents; otherwise they can never be edited.

9. **Functions must be async, not sync.** The current pipeline is async (`process_document_background` is `async def`, uses `loop.run_in_executor`). The new functions match this pattern. `run_ingestion_in_thread` keeps its asyncio loop wrapper unchanged.

10. **Graph build dispatch stays in `run_ingestion_in_thread`.** `process_document_full` returns `Optional[GraphBuildRequest]` like the current `process_document_background`. `run_ingestion_in_thread` handles the graph build dispatch logic. No orchestration change needed.

11. **Progress tracking mechanism must be preserved.** The current code uses `ProgressTimeout` and a separate `progress_db` session. `ingest_document` accepts a `progress_cb` callback so this mechanism continues to work during re-ingest.

12. **`convert_document` signature cleaned up.** Removed `content_type` and `title` parameters — `content_type` is for the Document record (set by caller), `title` is derived by `extract_title()` inside the function.

### Frontend gaps found and fixed

1. **Browse endpoint already returns most fields.** `get_folder_contents()` already returns `document_id`, `status`, `graph_status`, `chunk_count`, `title`, `error_message`. Only `conversion_status` is missing. The `graph_status` retrieval at line 193 (`doc.processing_tasks[0].graph_status`) is fragile and should use the latest-task query pattern.

2. **File row click toggles selection, conflicting with Edit.** The Edit button must call `e.stopPropagation()` and a dedicated Actions column must be added.

3. **Re-convert polling contradicts 409 behavior.** The editor should poll `GET .../ingest-status` (or `.../markdown` and silently handle 409) while re-converting; the 409 should only stop the initial open, not the polling.

4. **Unsaved guard must cover all in-app navigation, not just back button.** A `ConfirmDialog` wrapper for all `router.push`/`Link` calls is needed.

5. **ReactMarkdown is already used in the codebase.** `answer.tsx` (line 22) and `search/page.tsx` (line 19) already use `react-markdown`. The new `MarkdownPreview` component should match their plugin configuration.

6. **`rehype-raw` is unsafe for admin-edited markdown.** Do not use `rehype-raw` in the editor. GFM + math + code highlighting are sufficient.

7. **Split pane is not responsive.** Add a tabbed view for small viewports (< 1024px).

8. **No double-dynamic route pattern exists in the app.** `.../[id]/documents/[docId]` is a new pattern. Create the directory structure: `frontend/src/app/dashboard/admin/data-sources/[id]/documents/[docId]/page.tsx`.

9. **`@tailwindcss/typography` is NOT installed.** The `prose` classes in the codebase are backed by hand-written CSS that only styles headings. The editor preview needs full prose styling — install the plugin or write comprehensive CSS.

### Next step

Implementation can begin with the Alembic migration and model changes (step 1 of section 13).
