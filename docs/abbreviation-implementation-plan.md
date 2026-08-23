# Abbreviation Expansion — Implementation Plan

## Overview

Implement the recommended suffix+glossary configuration from the analysis doc.
This plan covers: data model, CSV upload/management, ingestion expansion, query
expansion, reranker wiring, generation glossary, and admin UI.

**No LLM calls for expansion.** All expansion is deterministic (regex + CSV
lookup). The only model calls are the ones already in the pipeline.

---

## Architecture Decision: MySQL (not Redis) for Abbreviation Storage

**Decision**: Store abbreviation lists in MySQL. Cache the compiled lookup
in process memory (same pattern as `settings_service.py` 30s cache).

**Why not Redis**:
- Abbreviations change rarely (admin uploads CSV). Read frequency is high
  (every ingestion chunk, every query) but the data is static between uploads.
- The existing `settings_service.py` already uses a 30s in-process cache for
  the same access pattern. Following that pattern is simpler and proven.
- Redis would add a network hop for data that changes once per week.
- The abbreviation lookup is a `Dict[str, List[str]]` built from 2,000-3,000
  rows. Building it takes ~5ms from MySQL. Caching it in-process makes it
  ~0ms. Redis would be ~1ms — no benefit.
- Redis is already used for rate limiting and LangGraph memory. Adding
  abbreviation caching there would mix concerns and complicate the Redis
  schema.

**Why MySQL**:
- Already the source of truth for all structured data (users, orgs, settings,
  chunks).
- Existing `OrgAbbreviation` table is already in MySQL.
- Admin CRUD operations need transactional consistency (upload CSV → replace
  all rows for a list). MySQL gives this for free.
- The 30s in-process cache handles the read-heavy access pattern.

---

## Data Model

### New Table: `abbreviation_lists`

Replaces the existing `org_abbreviations` table. Supports multiple named lists
per org, each uploaded as a CSV file. Universal lists (uploaded by super_admin)
have `org_id=NULL` and are available to all orgs. Org-specific lists (uploaded
by org admin) supplement universal lists.

```sql
CREATE TABLE abbreviation_lists (
    id          INTEGER PRIMARY KEY AUTO_INCREMENT,
    name        VARCHAR(255) NOT NULL,          -- user-given name, e.g. "Military Abbreviations"
    description TEXT NULL,                      -- optional description
    org_id      INTEGER NULL,                   -- NULL = universal (super_admin), non-NULL = org-scoped
    is_enabled  BOOLEAN NOT NULL DEFAULT TRUE,  -- admin can disable without deleting
    row_count   INTEGER NOT NULL DEFAULT 0,     -- denormalized count for UI display
    created_by  INTEGER NULL,                   -- FK to users.id
    created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (org_id) REFERENCES organisations(id) ON DELETE CASCADE,
    FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL,
    INDEX idx_abbrev_lists_org (org_id),
    INDEX idx_abbrev_lists_enabled (is_enabled)
);
```

### New Table: `abbreviations`

Replaces `org_abbreviations`. Each row is one abbreviation→expansion mapping.
Multiple rows per abbreviation (multi-meaning). Belongs to a list.

```sql
CREATE TABLE abbreviations (
    id           INTEGER PRIMARY KEY AUTO_INCREMENT,
    list_id      INTEGER NOT NULL,
    abbreviation VARCHAR(64) NOT NULL,
    expanded_form VARCHAR(512) NOT NULL,
    category     VARCHAR(255) NULL,
    FOREIGN KEY (list_id) REFERENCES abbreviation_lists(id) ON DELETE CASCADE,
    INDEX idx_abbreviations_list (list_id),
    INDEX idx_abbreviations_abbr (abbreviation)
);
```

### Migration

- New Alembic migration: `0030_add_abbreviation_lists.py`
- Creates `abbreviation_lists` and `abbreviations` tables
- Drops the old `org_abbreviations` table (it was dead code, no data to migrate)
- Down revision: `0029_add_document_modified_at`

### No schema changes to `document_chunks`

The existing `chunk_text` and `chunk_metadata` columns are sufficient:
- `chunk_text`: stores suffix-expanded text (for search/embedding)
- `chunk_metadata.original_text`: stores original text (for display/citations)

`chunk_metadata` is a JSON column — adding an `original_text` key requires no
schema change.

---

## Backend: Abbreviation Service

### New file: `backend/app/services/abbreviation_service.py`

Core service for loading, caching, and applying abbreviation expansion.

```python
# Public API:

def get_active_lists(db: Session, org_id: Optional[int]) -> List[AbbreviationList]:
    """Return enabled lists for this org: universal (org_id=NULL) + org-specific."""

def build_lookup(db: Session, org_id: Optional[int]) -> AbbreviationLookup:
    """Build the compiled lookup: {abbr: [form1, form2, ...]}.
    Merges universal + org-specific lists. Cached in-process for 30s."""

def expand_suffix(text: str, lookup: AbbreviationLookup) -> str:
    """Append [Expansions: abbr=form1 form2; ...] to text."""

def expand_query_suffix(query: str, lookup: AbbreviationLookup) -> str:
    """Bidirectional query expansion: append all forms for found abbreviations."""

def build_glossary(text: str, lookup: AbbreviationLookup) -> str:
    """Build [Abbreviation Glossary] block from abbreviations found in text."""

def find_abbrs_in_text(text: str, lookup: AbbreviationLookup) -> Dict[str, List[str]]:
    """Find all abbreviations in text, return {abbr: [forms]}."""
```

### Caching

Same pattern as `settings_service.py`:
- Module-level dict: `_cache: dict[tuple[Optional[int], str], tuple[AbbreviationLookup, float]]`
- TTL: 30 seconds
- Cache key: `(org_id, "lookup")`
- Invalidated on any list upload/update/delete (call `_invalidate_cache(org_id)`)

### AbbreviationLookup (compiled form)

```python
@dataclass
class AbbreviationLookup:
    forward: Dict[str, List[str]]       # {abbr: [form1, form2, ...]}
    all_abbrs_sorted: List[str]          # sorted by length descending (for regex matching)
    compiled_patterns: Dict[str, re.Pattern]  # pre-compiled regex per abbreviation
```

Building this once and caching it avoids re-compiling 2,000 regex patterns on
every chunk/query.

---

## Backend: Admin API Endpoints

### New file: `backend/app/api/api_v1/abbreviations.py`

Replaces the existing abbreviation endpoints in `admin.py`. All endpoints
require admin or super_admin role.

### List Management

```
GET    /api/admin/abbreviation-lists
       Returns all lists visible to the caller:
       - super_admin: all lists (universal + all orgs)
       - admin: universal lists + their org's lists

GET    /api/admin/abbreviation-lists/{list_id}
       Returns a single list with its abbreviation count

POST   /api/admin/abbreviation-lists/upload
       Multipart upload: CSV file + name + description + scope (universal|org)
       - super_admin: can set scope=universal (org_id=NULL) or scope=org
       - admin: can only set scope=org (their own org_id)
       Parses CSV, creates list + abbreviation rows in a single transaction
       Upsert: if name already exists for this scope, replaces all rows

PUT    /api/admin/abbreviation-lists/{list_id}
       Update name, description, is_enabled (no CSV re-upload)

DELETE /api/admin/abbreviation-lists/{list_id}
       Delete list + all abbreviation rows (CASCADE)
       - admin can only delete their org's lists
       - super_admin can delete any list

POST   /api/admin/abbreviation-lists/{list_id}/enable
POST   /api/admin/abbreviation-lists/{list_id}/disable
       Toggle is_enabled without full PUT
```

### Abbreviation Browsing

```
GET    /api/admin/abbreviation-lists/{list_id}/abbreviations?search=CO&page=1&size=50
       Paginated abbreviation listing for a specific list
       Optional search filter on abbreviation or expanded_form
```

### CSV Upload Parsing

```python
def parse_abbreviation_csv(file: UploadFile) -> List[Dict]:
    """Parse CSV with columns: abbreviation, expanded_form, category.
    Validate: non-empty abbreviation and expanded_form.
    Skip blank rows. Return list of dicts."""
```

The CSV format matches the existing `abbreviations_enhanced.csv`:
```csv
abbreviation,expanded_form,category
CO,Commanding Officer,GENERAL
DA,Daily Allowance,GENERAL
DA,Defence Attache,GENERAL
DA,Deputy Assistant,GENERAL
```

### Authorization Rules

| Action | super_admin | admin |
|--------|-------------|-------|
| Upload universal list | Yes | No |
| Upload org list (own org) | Yes | Yes |
| Upload org list (other org) | Yes | No |
| Enable/disable universal list | Yes | No |
| Enable/disable own org list | Yes | Yes |
| Delete universal list | Yes | No |
| Delete own org list | Yes | Yes |
| View all lists | Yes | No (own org + universal only) |

### Wire into router

Add to `backend/app/api/api_v1/__init__.py`:
```python
from .abbreviations import router as abbreviations_router
api_router.include_router(abbreviations_router, prefix="/admin", tags=["abbreviations"])
```

Remove the old abbreviation endpoints from `admin.py` (lines 282-400).

---

## Backend: Ingestion Expansion

### File: `backend/app/services/ingestion/document_processor.py`

**Change**: In `_build_chunk_records()` (line 464), expand each chunk before
storing.

```python
def _build_chunk_records():
    from app.services.abbreviation_service import build_lookup, expand_suffix, find_abbrs_in_text
    
    # Build lookup once for this ingestion run
    abbr_lookup = build_lookup(db)  # org_id from context
    
    payloads = []
    db_chunks = []
    for i, chunk in enumerate(chunks):
        original_text = chunk.page_content
        expanded_text = expand_suffix(original_text, abbr_lookup) if abbr_lookup.forward else original_text
        
        # Store expanded text for search/embedding
        # Store original text in metadata for display/citations
        source_metadata = {
            k: v for k, v in chunk.metadata.items()
            if k not in ("kb_id", "document_id", "chunk_id", "file_name")
        }
        if abbr_lookup.forward and expanded_text != original_text:
            source_metadata["original_text"] = original_text
        
        chunk_id = hashlib.sha256(
            f"{scope_prefix}:{file_name}:{i}:{original_text}".encode()
        ).hexdigest()
        
        db_chunks.append(DocumentChunk(
            id=chunk_id,
            document_id=document.id,
            kb_id=kb_id if kb_id else None,
            data_store_id=data_store_id if data_store_id else None,
            file_name=file_name,
            chunk_text=expanded_text,        # <-- expanded for search
            chunk_index=i,
            chunk_metadata=source_metadata,   # <-- original_text preserved
            hash=hashlib.sha256(
                (original_text + str(chunk.metadata)).encode()
            ).hexdigest(),
        ))
        payloads.append((chunk_id, expanded_text, source_metadata, i))
    return payloads, db_chunks
```

**Key decisions**:
- `chunk_id` hash uses `original_text` (not expanded) so re-expansion with a
  new abbreviation list doesn't change chunk IDs and force re-ingestion.
- `original_text` is only stored in metadata when expansion actually changed
  the text (avoids wasting space on chunks with no abbreviations).
- The `db` session is already available in `_build_chunk_records` scope (it's
  passed through the ingestion function).

### File: `backend/app/services/ingestion/document_qdrant.py`

No changes needed. The Qdrant payload already uses `chunk_text` from the
payloads list, which now contains the expanded text. The `original_text` is
in `source_metadata` which is already stored as Qdrant payload metadata.

### Chunk size guard

In `_build_chunk_records`, after expansion, check if the expanded text exceeds
the SPLADE-safe character limit (1400 chars). If so, the chunk needs re-splitting.

```python
SPLADE_SAFE_CHAR_LIMIT = 1400

def _build_chunk_records():
    # ... existing code ...
    for i, chunk in enumerate(chunks):
        original_text = chunk.page_content
        expanded_text = expand_suffix(original_text, abbr_lookup) if abbr_lookup.forward else original_text
        
        if len(expanded_text) > SPLADE_SAFE_CHAR_LIMIT:
            # Truncate suffix, not original content
            expanded_text = _truncate_suffix(original_text, expanded_text, SPLADE_SAFE_CHAR_LIMIT)
        # ... rest of existing code ...
```

`_truncate_suffix` preserves the original text and shortens the `[Expansions: ...]`
block by dropping the rarest abbreviations first.

---

## Backend: Query Expansion

### File: `backend/app/services/agentic_rag/nodes.py`

**Change**: After `rewrite_query_node` (line 240), expand the rewritten query
for retrieval. Keep the original query for the reranker.

Add a new node: `expand_query_node`

```python
def expand_query_node(state: AgentState, db: Any = None, org_id: Any = None) -> dict:
    """Expand the rewritten query with abbreviation suffix expansion.
    
    The expanded query is used for dense/sparse/exact retrieval.
    The original query is preserved for the reranker.
    """
    from app.services.abbreviation_service import build_lookup, expand_query_suffix
    
    rewritten = state.get("rewritten_query", state.get("original_query", ""))
    org_id = org_id if org_id is not None else state.get("org_id")
    
    abbr_lookup = build_lookup(db, org_id) if db else None
    if abbr_lookup and abbr_lookup.forward:
        expanded = expand_query_suffix(rewritten, abbr_lookup)
        return {"expanded_query": expanded, "rewritten_query": rewritten}
    return {"expanded_query": rewritten, "rewritten_query": rewritten}
```

### Wire into the graph

In `agent_graph.py`, add `expand_query_node` after `rewrite_query_node` and
before the retrieval nodes:

```python
graph.add_node("rewrite_query", rewrite_query_node)
graph.add_node("expand_query", expand_query_node)      # <-- new
graph.add_node("dense_retrieval", dense_retrieval_node)
# ...

graph.add_edge("rewrite_query", "expand_query")        # <-- new
graph.add_edge("expand_query", "dense_retrieval")      # <-- changed
```

### Retrieval nodes: use expanded query

In `dense_retrieval_node` (line 443), `sparse_retrieval_node` (line 488),
`exact_retrieval_node` (line 533):

```python
# Before:
query = state.get("rewritten_query", state.get("original_query", ""))

# After:
query = state.get("expanded_query", state.get("rewritten_query", state.get("original_query", "")))
```

### Reranker: use original query

In `reranking_node` (line 306):

```python
# Before:
query = state.get("rewritten_query", state.get("original_query", ""))

# After:
query = state.get("rewritten_query", state.get("original_query", ""))
# Reranker uses the rewritten (non-expanded) query — NOT the expanded query.
# The suffix-expanded chunks provide both abbreviation and full-form tokens.
```

No change needed — the reranker already uses `rewritten_query`, which is the
non-expanded query. The expanded query is only used by the retrieval legs.

### AgentState

Add `expanded_query` to the state:

```python
class AgentState(TypedDict):
    # ... existing fields ...
    expanded_query: str  # <-- new
```

---

## Backend: Generation Glossary

### File: `backend/app/services/agentic_rag/utils.py`

**Change**: In `format_context_string()` (line 34), append a glossary block
after the chunks.

```python
def format_context_string(
    docs: list[dict],
    file_markdown: str | None = None,
    db: Any = None,
    org_id: Any = None,
) -> str:
    # ... existing chunk formatting ...
    
    # Append glossary if abbreviations are configured
    if db is not None:
        from app.services.abbreviation_service import build_lookup, build_glossary_from_docs
        abbr_lookup = build_lookup(db, org_id)
        if abbr_lookup and abbr_lookup.forward:
            # Use original_text from metadata if available, else page_content
            texts = [doc.get("metadata", {}).get("original_text", doc.get("page_content", "")) for doc in docs]
            glossary = build_glossary_from_docs(texts, abbr_lookup)
            if glossary:
                parts.append(f"[Abbreviation Glossary]\n{glossary}")
    
    # ... file_markdown append ...
    return "\n\n---\n\n".join(parts)
```

### Use original_text for generation context

In the chunk formatting loop inside `format_context_string`:

```python
# Before:
content = doc.get("page_content", "").strip()

# After:
content = doc.get("metadata", {}).get("original_text", doc.get("page_content", "")).strip()
```

This ensures the generation LLM sees clean original prose, not the
`[Expansions: ...]` suffix. The glossary block provides the abbreviation
mappings explicitly.

### Callers of format_context_string

Update all callers to pass `db` and `org_id`:

- `agent_graph.py` line 1318: `format_context_string(docs, state.get("file_markdown"), db=ctx.db, org_id=ctx.org_id)`
- `agent_graph.py` line 1377: same
- Any other callers (search for `format_context_string` and add the params)

---

## Backend: Settings

### File: `backend/app/core/settings_registry.py`

Add one new setting to control whether abbreviation expansion is enabled:

```python
SettingDef(
    key="ABBRIATION_EXPANSION_ENABLED",
    category="Retrieval",
    label="Abbreviation Expansion",
    value_type="bool",
    default=True,
    scope="org",           # org-overridable
    reload="none",
    requires_reindex=True, # changing this requires re-ingestion
    description="Enable suffix expansion of military/organisational abbreviations during ingestion and query. Requires re-ingestion of existing documents when toggled.",
),
```

This setting is checked in:
- `document_processor.py` before expansion (skip if disabled)
- `expand_query_node` before expansion (skip if disabled)
- `format_context_string` before glossary (skip if disabled)

When an admin disables expansion, existing chunks remain expanded until
re-ingested. New ingestions will store original text only. The setting
documentation notes this.

---

## Frontend: Admin UI

### New page: `/dashboard/admin/abbreviations`

Super admin sees all lists (universal + all orgs). Org admin sees universal +
their own org's lists.

**File**: `frontend/src/app/dashboard/admin/abbreviations/page.tsx`

**Layout**:
- Header: "Abbreviation Lists" with upload button
- Table of lists: Name, Scope (Universal/Org name), Status (Enabled/Disabled),
  Rows, Created, Actions (Edit, Enable/Disable, Delete)
- Upload dialog: CSV file picker, name, description, scope dropdown
  (super_admin only sees "Universal" option)

**Components used** (matching existing admin pages):
- `Table`, `TableHeader`, `TableRow`, `TableCell` (from existing admin pages)
- `Dialog`, `DialogContent`, `DialogHeader`, `DialogTitle`, `DialogFooter`
- `Button`, `Input`, `Label`, `Switch`
- `useToast` for notifications
- `api` from `@/lib/api` for HTTP calls

### New page: `/dashboard/admin/abbreviations/[listId]`

List detail page showing abbreviations in the list.

**File**: `frontend/src/app/dashboard/admin/abbreviations/[listId]/page.tsx`

**Layout**:
- Back link to lists page
- List metadata: name, description, scope, status, row count
- Search bar: filter by abbreviation or expanded_form
- Paginated table: Abbreviation, Expanded Form, Category
- Edit button: edit name/description/status
- Delete button with confirmation dialog

### Sidebar entry

**File**: `frontend/src/components/admin/admin-sidebar.tsx`

Add to `NAV_ITEMS`:
```typescript
{ label: 'Abbreviations', href: '/dashboard/admin/abbreviations', icon: BookText },
```

This shows for both admin and super_admin (both can manage abbreviation lists,
just with different scope permissions).

### Upload dialog component

**File**: `frontend/src/components/admin/abbreviation-upload-dialog.tsx`

Reusable dialog for uploading a CSV file:
- File input (accept=".csv")
- Name field (required)
- Description field (optional)
- Scope dropdown: "Universal" (super_admin only) / "My Organisation"
- Upload button with loading state
- Error display for invalid CSV format

### API client functions

**File**: `frontend/src/lib/api.ts` (or a new `frontend/src/lib/api-abbreviations.ts`)

```typescript
// List management
getAbbreviationLists(): Promise<AbbreviationList[]>
getAbbreviationList(listId: number): Promise<AbbreviationListDetail>
uploadAbbreviationList(file: File, name: string, description: string, scope: 'universal' | 'org'): Promise<AbbreviationList>
updateAbbreviationList(listId: number, data: { name?: string, description?: string, is_enabled?: boolean }): Promise<AbbreviationList>
deleteAbbreviationList(listId: number): Promise<void>
toggleAbbreviationList(listId: number, enabled: boolean): Promise<void>

// Abbreviation browsing
getAbbreviations(listId: number, params: { search?: string, page?: number, size?: number }): Promise<{ items: Abbreviation[], total: number }>
```

### TypeScript types

```typescript
interface AbbreviationList {
    id: number;
    name: string;
    description: string | null;
    org_id: number | null;     // null = universal
    org_name: string | null;   // for display
    is_enabled: boolean;
    row_count: number;
    created_at: string;
    updated_at: string;
}

interface AbbreviationListDetail extends AbbreviationList {
    abbreviations?: Abbreviation[];
}

interface Abbreviation {
    id: number;
    list_id: number;
    abbreviation: string;
    expanded_form: string;
    category: string | null;
}
```

---

## Implementation Order

### Phase 1: Backend data model + service (no pipeline changes)

1. Create Alembic migration `0030_add_abbreviation_lists.py`
2. Create models: `AbbreviationList`, `Abbreviation` in
   `backend/app/models/abbreviation.py`
3. Create `backend/app/services/abbreviation_service.py` with:
   - `build_lookup()` + 30s cache
   - `expand_suffix()`, `expand_query_suffix()`, `build_glossary()`
   - `find_abbrs_in_text()`
4. Create `backend/app/api/api_v1/abbreviations.py` with all endpoints
5. Register router in `backend/app/api/api_v1/__init__.py`
6. Remove old abbreviation endpoints from `admin.py`
7. Add `ABBRIVIATION_EXPANSION_ENABLED` to settings registry
8. Run migration
9. Test: upload the military CSV via API, verify lookup builds correctly

### Phase 2: Wire into pipeline

10. Modify `document_processor.py` `_build_chunk_records()` to expand chunks
11. Add `expand_query_node` to `nodes.py`
12. Wire `expand_query_node` into the graph in `agent_graph.py`
13. Change retrieval nodes to use `expanded_query` from state
14. Modify `format_context_string()` to use `original_text` + append glossary
15. Update callers of `format_context_string` to pass `db` and `org_id`
16. Add `expanded_query` to `AgentState`

### Phase 3: Frontend

17. Add "Abbreviations" to admin sidebar
18. Create abbreviations list page
19. Create abbreviation upload dialog component
20. Create list detail page with paginated abbreviation table
21. Add API client functions
22. Test: upload CSV via UI, enable/disable lists, browse abbreviations

### Phase 4: Verification

23. Ingest a document with abbreviations, verify chunk_text is expanded and
    original_text is in metadata
24. Query with abbreviations, verify expanded_query is used for retrieval
25. Verify reranker receives original query + expanded chunks
26. Verify generation context has original text + glossary
27. Verify citations show original text
28. Test with expansion disabled (setting off) — pipeline should work as before
29. Test with no abbreviation lists uploaded — pipeline should work as before

---

## Files Changed

### New files

| File | Purpose |
|------|---------|
| `backend/alembic/versions/0030_add_abbreviation_lists.py` | Migration |
| `backend/app/models/abbreviation.py` | `AbbreviationList` + `Abbreviation` models |
| `backend/app/services/abbreviation_service.py` | Core expansion logic + cache |
| `backend/app/api/api_v1/abbreviations.py` | Admin CRUD + CSV upload API |
| `frontend/src/app/dashboard/admin/abbreviations/page.tsx` | List management page |
| `frontend/src/app/dashboard/admin/abbreviations/[listId]/page.tsx` | List detail page |
| `frontend/src/components/admin/abbreviation-upload-dialog.tsx` | Upload dialog |

### Modified files

| File | Change |
|------|--------|
| `backend/app/models/organisation.py` | Remove `OrgAbbreviation` class + relationship |
| `backend/app/api/api_v1/admin.py` | Remove old abbreviation endpoints (lines 282-400) |
| `backend/app/api/api_v1/__init__.py` | Register abbreviations router |
| `backend/app/core/settings_registry.py` | Add `ABBRIVIATION_EXPANSION_ENABLED` |
| `backend/app/services/ingestion/document_processor.py` | Expand chunks in `_build_chunk_records` |
| `backend/app/services/agentic_rag/nodes.py` | Add `expand_query_node`, use `expanded_query` in retrieval |
| `backend/app/services/agentic_rag/agent_graph.py` | Wire `expand_query_node`, add `expanded_query` to state |
| `backend/app/services/agentic_rag/utils.py` | `format_context_string` uses `original_text` + glossary |
| `backend/app/services/retrieval/query_expander.py` | Replace with wrapper around `abbreviation_service` |
| `frontend/src/components/admin/admin-sidebar.tsx` | Add Abbreviations nav item |

### Deleted files

None. The old `query_expander.py` is repurposed as a thin wrapper.

---

## Edge Cases

1. **No abbreviation lists uploaded**: `build_lookup()` returns empty lookup.
   All expansion functions are no-ops. Pipeline works as before.

2. **Expansion disabled via setting**: All expansion functions check the
   setting first and skip if disabled. Existing expanded chunks remain
   expanded until re-ingested.

3. **Chunk with no abbreviations**: `expand_suffix()` returns original text
   unchanged. `original_text` is not stored in metadata (saves space).

4. **Chunk exceeds SPLADE limit after expansion**: `_truncate_suffix()` drops
   the rarest abbreviations from the suffix until within limit. Original text
   is never truncated.

5. **Multi-meaning abbreviation (DA)**: All forms appended in suffix. The
   retrieval models and generation LLM disambiguate based on context. No
   deterministic disambiguation attempted.

6. **Preposition collision ("to" matching "TO")**: The abbreviation list
   should not contain common English prepositions. If it does, the suffix
   approach appends the expansion without removing the original word, so
   the preposition is still readable. The generation glossary lists all
   meanings, and the LLM ignores irrelevant ones.

7. **Re-ingestion with updated abbreviation list**: Chunk IDs are based on
   original text, so re-ingestion with a new list produces the same chunk IDs
   but with different expanded text. Qdrant upsert handles this correctly.

8. **Concurrent ingestion + list upload**: The 30s cache means an ingestion
   in progress may use a slightly stale lookup. This is acceptable — the next
   ingestion will use the updated list.

9. **Large CSV (3,000+ rows)**: CSV upload parses in memory (3,000 rows is
   ~500KB — trivial). DB insert is batched in a single transaction.

10. **Org admin uploads list with same name as universal list**: Allowed.
    The org's list supplements the universal list. Both are merged in
    `build_lookup()`. If the same abbreviation appears in both, all forms
    from both lists are included.
