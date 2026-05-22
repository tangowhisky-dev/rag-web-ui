# S03: File Attachment — Research

**Date:** 2026-01-24
**Slice:** File Attachment (drag-drop, file picker, multipart endpoint, inline context injection)

## Summary

S03 adds file attachment to the chat input bar. Users drag-drop or click to attach a file; the filename appears as a chip in the input bar. On submit, the file content is read and prepended as inline context to the current message via a new `POST /api/chat/{chat_id}/messages/with-file` multipart endpoint. The existing `ChatInput` component already has a disabled file button placeholder — S03 activates it.

## Key Findings

### Frontend — ChatInput (S02 deliverable)

- `frontend/src/components/chat/chat-input.tsx` has a disabled `<Paperclip>` button at `data-testid="chat-input-file-button"`.
- `InputBar` props: `{ value, onChange, onSubmit, disabled, placeholder }`.
- Auto-resize uses hidden span measurement (1-5 lines). KB selector fetches KBs on mount via `api.get("/api/knowledge-base")`.
- `react-dropzone` and `react-file-icon` are already in `package.json` — no new deps needed.
- `api.ts` supports `FormData` (auto-sets body, skips Content-Type header for multipart).

### Frontend — Chat Page

- `frontend/src/app/dashboard/chat/[id]/page.tsx` uses raw `fetch()` with JSON for `handleSubmit`.
- Needs to support both JSON (no file) and multipart (with file) submission paths.
- SSE streaming reader logic is shared — only the request body/endpoint changes.

### Backend — Chat Endpoint

- `POST /api/chat/{chat_id}/messages` is the existing JSON streaming endpoint.
- No `PATCH /api/chat/{chat_id}` was missing — S01 already added it.
- `generate_response()` signature: `(query, messages, knowledge_base_ids, chat_id, db, use_dense, use_sparse, use_exact, use_graph_rag)`.
- No `file_context` parameter — needs to be added.

### Backend — File Reading

- `backend/app/services/document_processor.py` uses `MarkItDown` for document conversion.
- `SUPPORTED_EXTENSIONS` covers: pdf, docx, txt, md, html, csv, json, xml, epub, images, archives.
- `CONTENT_TYPE_MAP` maps extensions to MIME types.

### No Existing Patterns

- No file attachment endpoint exists — greenfield backend work.
- No drag-drop or file chip UI in the chat input — greenfield frontend work.
- No file size/type validation in the chat flow — needs to be added.

## Recommendation

### Architecture: Two parallel tracks

**Track A — Backend (Multipart Endpoint):**
1. Add `POST /api/chat/{chat_id}/messages/with-file` endpoint accepting `multipart/form-data` with fields: `file`, `message` (text), optional `messages` (chat history JSON).
2. Read file content using `MarkItDown` (reuse from document_processor).
3. Prepend file content to query as context prefix before calling `generate_response()`.
4. Return SSE stream identical to the existing endpoint.

**Track B — Frontend (File Attachment UI):**
1. Enhance `InputBar` with drag-drop zone, file picker, and file chip display.
2. Add `file` and `onFileChange` props to `InputBar`.
3. In `page.tsx`, wire file state and switch `handleSubmit` to use `FormData` + multipart endpoint when a file is attached.
4. Add inline error display below input for oversized/unsupported files.

### File Size and Type Limits

- **Max file size:** 10 MB (keeps context window reasonable for one-off queries).
- **Supported types:** pdf, docx, txt, md, html, csv, json, xml, epub, jpg, jpeg, png (the most common 13 from SUPPORTED_EXTENSIONS).
- **Validation:** Client-side (instant error on drop) + server-side (reject with 400).

### Interface Design

```typescript
// InputBar props (extended)
interface InputBarProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  disabled: boolean;
  placeholder?: string;
  file: File | null;
  onFileChange: (file: File | null) => void;
  error?: string;
}

// File chip renders inline in the input bar, above textarea
// Shows filename + file size + X button to remove
// Drag-over state adds a visual highlight to the input bar container
```

### Backend Endpoint

```python
@router.post("/{chat_id}/messages/with-file", response_model=None)
async def create_message_with_file(
    file: UploadFile = File(...),
    message: str = Form(""),
    messages: str = Form("[]"),  # chat history as JSON string
    chat_id: int = Path(...),
    ...
) -> StreamingResponse:
    # 1. Read file content via MarkItDown
    # 2. Prepend to query: f"## File Context: {file.filename}\n\n{content}\n\n{message}"
    # 3. Call generate_response() with augmented query
    # 4. Stream SSE response (identical format to existing endpoint)
```

## Natural Seams (Task Decomposition)

### T01: Backend — Multipart endpoint with file reading
- Add `POST /{chat_id}/messages/with-file` to `chat.py`.
- Read file via `MarkItDown`, validate size/type.
- Prepend file content to query, call `generate_response()`.
- Return SSE stream.
- **Files:** `backend/app/api/api_v1/chat.py`, optionally `backend/app/services/document_processor.py` (reuse `_get_markitdown()`).

### T02: Frontend — FileAttachment component + InputBar enhancement
- Create `FileAttachment` component with drag-drop, file picker, file chip display, error display.
- Extend `InputBar` with `file`, `onFileChange`, `error` props.
- Wire drag-drop zone to input bar container.
- **Files:** `frontend/src/components/chat/file-attachment.tsx`, `frontend/src/components/chat/chat-input.tsx`.

### T03: Frontend — Integration + Tests
- Update `page.tsx` with file state, switch to FormData + multipart endpoint when file attached.
- Add file validation (size, type) with inline error display.
- Update `chat-input.test.tsx` with file attachment tests.
- Create `file-attachment.test.tsx`.
- **Files:** `frontend/src/app/dashboard/chat/[id]/page.tsx`, `frontend/src/components/chat/__tests__/chat-input.test.tsx`, `frontend/src/components/chat/__tests__/file-attachment.test.tsx`.

## First Proof

T01 (backend endpoint) is the highest risk because:
- `generate_response` is an async generator — wiring file content into the streaming pipeline needs to preserve the SSE event format.
- `MarkItDown` is async and needs to handle binary file reads from `UploadFile`.
- If the endpoint doesn't stream correctly, the frontend has nothing to test against.

Build T01 first, verify with `curl` + a test file, then proceed to T02/T03.

## Verification

### Backend
```bash
# Test multipart endpoint with a PDF
curl -X POST http://localhost:8000/api/chat/1/messages/with-file \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@test.pdf" \
  -F "message=What is this document about?" \
  -F 'messages=[{"role":"user","content":"hello"}]' \
  --no-buffer

# Should return SSE stream with 1:, 2:, 0: events
```

### Frontend
```bash
# Run all chat-related tests
npx jest frontend/src/components/chat/__tests__/ --passWithNoTests

# TypeScript check
npx tsc --noEmit
```

### Manual
1. Navigate to `/dashboard/chat/{id}`.
2. Drag a PDF into the input bar — filename chip appears.
3. Type a message and submit — SSE stream returns answer with file context.
4. Drag a 15MB file — shows "File too large (max 10 MB)" error below input.
5. Drag an unsupported file type — shows "Unsupported file type" error.
6. Click X on file chip — file removed, input bar returns to normal.

## Don't Hand-Roll

- **File reading:** Use `MarkItDown` (already installed, handles all formats).
- **Drag-drop:** Use `react-dropzone` (already installed).
- **File icons:** Use `react-file-icon` (already installed).
- **File size formatting:** Use a simple utility function rather than a library — the math is trivial (`bytes / 1024 / 1024`).

## Sources

- `frontend/src/components/chat/chat-input.tsx` — ChatInput component (S02 deliverable)
- `frontend/src/app/dashboard/chat/[id]/page.tsx` — Chat page with handleSubmit
- `backend/app/api/api_v1/chat.py` — Chat endpoints
- `backend/app/services/chat_service.py` — `generate_response()` async generator
- `backend/app/services/document_processor.py` — `MarkItDown` usage and `SUPPORTED_EXTENSIONS`
- `frontend/src/lib/api.ts` — API client with FormData support
- `frontend/package.json` — `react-dropzone`, `react-file-icon` already installed