# Three-Phase Ingestion Pipeline + Markdown Editor

## Architecture Decision Record

### Status
Proposed — awaiting approval

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

Add two columns to `documents`:

```python
def upgrade():
    op.add_column('documents', sa.Column('converted_markdown', sa.LONGTEXT(), nullable=True))
    op.add_column('documents', sa.Column('conversion_status', sa.String(20), nullable=True, index=True))
    # conversion_status: pending, completed, error
    # null = legacy document (pre-3-phase), treat as "completed" for backward compat

def downgrade():
    op.drop_column('documents', 'conversion_status')
    op.drop_column('documents', 'converted_markdown')
```

**Backfill:** No backfill needed. Legacy documents with `conversion_status = NULL` are treated as "completed" (they already have chunks). The next re-scan will populate the markdown on re-conversion.

### 2. Backend Model Changes

**File:** `backend/app/models/knowledge.py`

```python
class Document(Base, TimestampMixin):
    # ... existing columns ...
    converted_markdown = Column(LONGTEXT, nullable=True)
    conversion_status = Column(String(20), nullable=True, index=True)
    # pending, completed, error, null=legacy
```

### 3. Backend Pipeline Split

The current `process_document_background()` in `document_processor.py` does everything in one function. Split into three independent functions that can be called separately or in sequence.

#### Phase 1: Convert

**File:** `backend/app/services/ingestion/document_processor.py`

```python
def convert_document(
    document_id: int,
    file_path: str,
    file_name: str,
    enable_ocr: Optional[bool] = None,
) -> str:
    """Convert a file to markdown and store it in Document.converted_markdown.
    
    Sets conversion_status to 'completed' on success, 'error' on failure.
    Returns the markdown text.
    
    This is a synchronous function — call from a thread pool.
    """
    # 1. Call _convert_to_markdown(file_path, file_name, enable_ocr)
    # 2. Run clean_markdown() cleanup pass
    # 3. Extract title
    # 4. Store in Document.converted_markdown + conversion_status='completed'
    # 5. Return markdown
```

#### Phase 2: Ingest (chunk + embed + Qdrant)

```python
def ingest_document(
    document_id: int,
    markdown_text: Optional[str] = None,
) -> None:
    """Chunk a document's markdown, embed, and store in Qdrant.
    
    If markdown_text is None, reads from Document.converted_markdown.
    Deletes existing chunks/vectors before re-chunking.
    Sets ProcessingTask.status to 'completed' on success.
    
    This is a synchronous function — call from a thread pool.
    """
    # 1. Read markdown from Document.converted_markdown (or use provided text)
    # 2. Delete existing chunks (MySQL + Qdrant) for this document
    # 3. Chunk with RecursiveCharacterTextSplitter
    # 4. Embed + upsert to Qdrant
    # 5. Store DocumentChunk rows
    # 6. Update ProcessingTask.status = 'completed'
```

#### Phase 3: Graph (already exists)

The graph build already runs separately via `run_graph_build_in_thread()`. No change needed — it reads chunks from MySQL and builds the Neo4j graph. The only addition: after re-ingest, trigger graph rebuild.

#### Orchestrator (replaces current monolithic flow)

```python
def process_document_full(document_id, file_path, file_name, enable_ocr):
    """Full pipeline: convert → ingest → graph. Used by scan/watcher."""
    # Phase 1
    markdown = convert_document(document_id, file_path, file_name, enable_ocr)
    
    # Phase 2
    ingest_document(document_id, markdown)
    
    # Phase 3 (async, already handled by ingestion_dispatcher)
    # GraphBuildRequest is queued after ingest completes
```

**Key principle:** Each phase is independently callable. The scan pipeline calls all three in sequence. The editor calls only phase 2+3 after the admin saves edits.

### 4. Backend API Endpoints

**File:** `backend/app/api/api_v1/datastore_documents.py` (new)

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
  "is_dirty": false
}
```

**Authorization:** Admin must have scope over the datastore.
**Error:** 404 if document doesn't exist, 409 if `conversion_status` is not "completed" (markdown not ready).

#### PUT `/api/admin/datastores/{ds_id}/documents/{doc_id}/markdown`
Saves edited markdown and triggers re-ingest.

```json
{
  "markdown": "# Corrected title\n\nFixed text..."
}
```

**Flow:**
1. Validate markdown is non-empty.
2. Update `Document.converted_markdown` in DB.
3. Delete existing chunks (MySQL + Qdrant) for this document.
4. Delete existing graph (Neo4j) for this document.
5. Reset `ProcessingTask.status = "pending"`, `graph_status = "pending"`.
6. Queue re-ingest in thread pool (calls `ingest_document()` then graph build).
7. Return 202 Accepted with task ID for progress polling.

**Why not synchronous:** Re-chunking + embedding + graph build can take 10-60s. Return immediately and let the UI poll progress.

#### POST `/api/admin/datastores/{ds_id}/documents/{doc_id}/reconvert`
Re-runs phase 1 only (file → markdown). Overwrites the current markdown.

Use case: the admin wants a fresh conversion (e.g., after enabling OCR or changing the vision model), then will review and edit before re-ingesting.

**Flow:**
1. Set `conversion_status = "pending"`.
2. Call `convert_document()` in thread pool.
3. Return 202 Accepted.
4. UI polls `conversion_status` — when "completed", loads the new markdown.

#### GET `/api/admin/datastores/{ds_id}/documents/{doc_id}/ingest-status`
Polls the re-ingest progress after a save.

```json
{
  "document_id": 123,
  "ingest_status": "processing",
  "ingest_progress": 45,
  "ingest_message": "Embedding 5 chunks...",
  "graph_status": "pending",
  "is_complete": false
}
```

### 5. Frontend: Markdown Editor Route

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
│  Status: ● Converted ● Ingested ● Graph   Chunks: 5         │
│  ⚠ Unsaved changes — Save to re-ingest                      │
└─────────────────────────────────────────────────────────────┘
```

#### Component Structure

```
MarkdownEditorPage
├── Header
│   ├── Back link (to file browser)
│   ├── File name + conversion status badge
│   ├── Save button (disabled when no changes, shows "Saving..." during save)
│   └── Re-convert button (dropdown with confirmation)
├── EditorBody (flex row, 50/50 split, resizable divider)
│   ├── EditorPane
│   │   └── <textarea> with monospace font, line numbers via CSS
│   └── PreviewPane
│       └── <ReactMarkdown> with remarkGfm, remarkMath, rehypeHighlight, rehypeKatex
├── StatusBar
│   ├── Phase indicators: Convert ●  Ingest ●  Graph ●
│   ├── Chunk count
│   └── Dirty state warning
└── UnsavedChangesDialog (shown on navigation/close when dirty)
```

#### State Management

```typescript
const [markdown, setMarkdown] = useState("");
const [originalMarkdown, setOriginalMarkdown] = useState("");
const [loading, setLoading] = useState(true);
const [saving, setSaving] = useState(false);
const [converting, setConverting] = useState(false);
const [ingestStatus, setIngestStatus] = useState<IngestStatus | null>(null);
const [showUnsavedDialog, setShowUnsavedDialog] = useState(false);

const isDirty = markdown !== originalMarkdown;
```

#### Real-time Preview

The preview updates as the user types. ReactMarkdown re-renders on every keystroke. For large documents (10K+ chars), debounce the preview update by 300ms to avoid jank:

```typescript
const [debouncedMarkdown, setDebouncedMarkdown] = useState(markdown);
useEffect(() => {
  const timer = setTimeout(() => setDebouncedMarkdown(markdown), 300);
  return () => clearTimeout(timer);
}, [markdown]);
```

The editor textarea is always responsive (native input, no rendering cost). Only the preview is debounced.

#### Save Flow

1. User clicks Save (or triggers it via unsaved-changes dialog).
2. `setSaving(true)`.
3. `PUT /api/admin/datastores/{ds_id}/documents/{doc_id}/markdown` with the edited markdown.
4. Backend returns 202 with task ID.
5. `setOriginalMarkdown(markdown)` — clears dirty state.
6. Start polling `GET .../ingest-status` every 2s.
7. Show progress in status bar: "Re-ingesting: 45% — Embedding 5 chunks..."
8. When `is_complete === true`, stop polling and show final status.
9. `setSaving(false)`.

#### Unsaved Changes Guard

```typescript
// Block navigation when dirty
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

// Intercept in-app navigation (back button click)
const handleBackClick = () => {
  if (isDirty) {
    setShowUnsavedDialog(true);
  } else {
    router.push(`/dashboard/admin/data-sources/${datastoreId}`);
  }
};
```

#### Re-convert Flow

1. User clicks "Re-convert" → confirmation dialog: "This will overwrite your current markdown with a fresh conversion. Unsaved edits will be lost."
2. If confirmed: `POST .../reconvert`.
3. `setConverting(true)`.
4. Poll `GET .../markdown` until `conversion_status === "completed"`.
5. Load new markdown into editor.
6. `setConverting(false)`.

#### Split Pane

No external library. Use a simple CSS flex layout with a draggable divider:

```typescript
const [splitPercent, setSplitPercent] = useState(50);
const dividerRef = useRef<HTMLDivElement>(null);

const handleDrag = useCallback((e: MouseEvent) => {
  const container = dividerRef.current?.parentElement;
  if (!container) return;
  const rect = container.getBoundingClientRect();
  const percent = ((e.clientX - rect.left) / rect.width) * 100;
  setSplitPercent(Math.max(20, Math.min(80, percent)));
}, []);

// Mouse events on divider: mousedown → attach mousemove/mouseup to window
```

This is ~20 lines of code. No need for `react-resizable-panels`.

### 6. Frontend: File Browser Integration

**File:** `frontend/src/app/dashboard/admin/data-sources/[id]/page.tsx`

Add a "Edit" action to each file row in the table. When clicked, navigates to the editor route.

**Changes:**
- Add an "Edit" button (pencil icon) next to each file that has `conversion_status === "completed"`.
- Show a three-segment status badge: `Convert ● Ingest ● Graph` with color per phase.
  - Green: completed
  - Amber: pending/processing
  - Red: failed
  - Gray: not started
- The Edit button is disabled when `conversion_status` is not "completed" (no markdown to edit yet).

### 7. Progress Display

#### Per-Document Progress (in editor)

The status bar shows three phase indicators with real-time status:

```
Convert: ✓ Completed    Ingest: ⟳ 45% Embedding...    Graph: ⏸ Pending
```

Data comes from:
- `conversion_status` on the Document (phase 1)
- `ProcessingTask.status` + `progress` + `progress_message` (phase 2)
- `ProcessingTask.graph_status` (phase 3)

#### Datastore-Level Progress (in file browser)

The file browser already shows per-document status. Extend the status badge to show three phases:

```
[●●○] = convert done, ingest done, graph pending
[●○○] = convert done, ingest pending, graph not started
[●●●] = all complete
[●✗○] = convert done, ingest failed, graph not started
```

This is a compact 3-dot indicator. Tooltip shows full status text.

### 8. Sequential Handling

#### Scan Pipeline (automatic)

When a scan discovers new/modified files, it processes them sequentially per datastore:
1. Convert each file → store markdown
2. Ingest each file → chunks + Qdrant
3. Queue graph build for each file

The existing `ThreadPoolExecutor` and future-tracking mechanism stays. The only change is that `process_document_full()` calls `convert_document()` first, then `ingest_document()`, then queues graph build.

#### Editor Re-ingest (manual)

When an admin saves edited markdown:
1. Delete chunks + Qdrant points + Neo4j graph for this document.
2. Re-chunk the new markdown.
3. Re-embed + upsert to Qdrant.
4. Queue graph build.

This runs in a single thread pool task. The UI polls progress. No scan is triggered — this is a targeted re-ingest of one document.

#### Recovery

Recovery already handles interrupted tasks. With the 3-phase split:
- `conversion_status = "pending"` → re-convert on recovery
- `ProcessingTask.status in ("pending", "processing")` → re-ingest on recovery
- `graph_status = "pending"` → re-queue graph build on recovery

The existing recovery logic for ProcessingTask and graph_status stays. Add a new check for `conversion_status = "pending"` that re-runs `convert_document()`.

### 9. Real-time Changes

#### Editor → Preview

Debounced 300ms. The textarea is always responsive. The preview re-renders after the user pauses typing.

#### Save → Re-ingest Progress

Polling every 2s via `GET .../ingest-status`. This matches the existing polling pattern for scan progress. SSE is not needed for a single document re-ingest — polling is simpler and sufficient.

#### File Browser → Editor Navigation

Standard Next.js client-side navigation. No real-time sync needed — the editor loads fresh data on open.

### 10. Where the Markdown Editor Fits In

```
Data Sources page (list of datastores)
  └── Datastore file browser (/data-sources/[id])
        └── File row → "Edit" button
              └── Markdown editor (/data-sources/[id]/documents/[docId])
                    ├── Two-pane: editor | preview
                    ├── Save → re-ingest (phase 2+3)
                    └── Re-convert → fresh phase 1
```

The editor is a sub-route of the file browser, not a separate top-level page. This keeps the URL hierarchy intuitive and the navigation flow natural.

### 11. Edge Cases

- **Empty markdown after edit:** Reject with 400. The admin must provide non-empty content.
- **Markdown larger than max_allowed_packet (64MB):** Reject with 413. This would only happen for absurdly large documents — the current largest is 20K chars.
- **Concurrent edits:** Optimistic locking via `updated_at` timestamp. If the document was modified since the admin loaded it, return 409 and ask them to reload.
- **Re-ingest while scan is running:** Allow it. The re-ingest targets a specific document; the scan processes new/modified files. They don't conflict because re-ingest deletes and recreates chunks for one document, while the scan creates chunks for different documents.
- **Re-ingest while graph is building:** Cancel the in-flight graph build for this document before re-ingesting. Use the existing `cancel_graph_builds_for_datastore()` scoped to the document.
- **Document deleted while editor is open:** The next API call returns 404. Show "Document no longer exists" and redirect to file browser.
- **Admin navigates away with unsaved changes:** `beforeunload` event + in-app navigation guard. Show confirmation dialog.

### 12. Files to Create/Modify

#### Backend

| File | Action | Description |
|---|---|---|
| `backend/alembic/versions/f1a2b3c4d5e6_add_converted_markdown.py` | Create | Migration: add `converted_markdown` + `conversion_status` |
| `backend/app/models/knowledge.py` | Modify | Add two columns to Document |
| `backend/app/services/ingestion/document_processor.py` | Modify | Split into `convert_document()` + `ingest_document()` + `process_document_full()` |
| `backend/app/api/api_v1/datastore_documents.py` | Create | New API: get/save markdown, re-convert, ingest-status |
| `backend/app/api/api_v1/api.py` | Modify | Register new router |
| `backend/app/services/discovery/startup_recovery_service.py` | Modify | Add recovery for `conversion_status = "pending"` |

#### Frontend

| File | Action | Description |
|---|---|---|
| `frontend/src/app/dashboard/admin/data-sources/[id]/documents/[docId]/page.tsx` | Create | Two-pane markdown editor page |
| `frontend/src/app/dashboard/admin/data-sources/[id]/page.tsx` | Modify | Add "Edit" button + 3-phase status badge |
| `frontend/src/components/markdown/markdown-preview.tsx` | Create | Reusable ReactMarkdown wrapper (used in editor + chat) |

### 13. Implementation Order

1. **Migration + model** — add columns, verify existing data is unaffected.
2. **Backend pipeline split** — extract `convert_document()` and `ingest_document()` from the monolithic function. Wire `process_document_full()` into the existing scan flow. Verify scans still work.
3. **Backend API** — add the four endpoints. Test with curl.
4. **Frontend editor** — build the two-pane page. Test with a real document.
5. **Frontend file browser integration** — add Edit button + 3-phase badge.
6. **Recovery update** — handle `conversion_status = "pending"` in startup recovery.
7. **End-to-end test** — scan a datastore, edit a document's markdown, save, verify re-ingest, verify graph rebuild.

### 14. What This Does NOT Change

- Qdrant storage format (same collections, same point structure)
- Neo4j graph schema (same nodes, same relationships)
- Chunking logic (same RecursiveCharacterTextSplitter)
- Embedding logic (same model, same batch size)
- Scan/discovery mechanism (same watcher, same manifest)
- Pause/resume for scans (orthogonal to the 3-phase split)
- Graph pause/resume (stays as-is, independent)
- Knowledge base direct uploads (KB documents also get the 3-phase treatment, but the editor is datastore-only for now)
