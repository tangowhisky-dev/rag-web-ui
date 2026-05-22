# S03: S03

**Goal:** Add file attachment to the chat input bar: drag-drop or file-picker attaches a file whose content is prepended as inline context via a new multipart SSE endpoint, with client-side and server-side validation for size and type.
**Demo:** Drag a PDF or text file into the input bar: filename appears in input bar chip. Submit message: SSE stream response includes the file content used as context. Oversized file shows inline error below input. Unsupported file type shows inline error. Backend logs show file_context prepended to context window.

## Must-Haves

- Drag a PDF into the input bar: filename chip appears with size and X button
- Submit message with file: SSE stream returns answer that incorporates file content as context
- Drag a 15 MB file: inline error "File too large (max 10 MB)" appears below input
- Drag an unsupported type: inline error "Unsupported file type" appears
- Click X on chip: file removed, input bar returns to normal
- Backend logs show [FILE] tag with filename and extracted content length
- All Jest tests in chat-input.test.tsx and new file-attachment.test.tsx pass
- No TypeScript errors; no ESLint errors

## Proof Level

- This slice proves: automated-tests + manual-curl-verification

## Integration Closure

POST /api/chat/{chat_id}/messages/with-file multipart endpoint is the only new API surface. It returns the same SSE event format (1:, 2:, 3:, 0:) as the existing messages endpoint so the frontend streaming reader is unchanged.

## Verification

- Backend logs structured [FILE] events: filename, content_length, extraction_duration_ms, truncated (bool). Validation rejections log at WARNING with reason. All under existing [CHAT] structured log convention.

## Tasks

- [ ] **T01: Backend — Multipart SSE endpoint with MarkItDown file extraction** `est:2h`
  **Why:** The frontend needs a streaming endpoint that accepts a file alongside the message. Without it, file attachment has no server-side counterpart.
  - Files: `backend/app/api/api_v1/chat.py`, `backend/app/services/document_processor.py`
  - Verify: cd /Users/tango16/code/rag-web-ui/.gsd/worktrees/M002 && grep -n 'with-file' backend/app/api/api_v1/chat.py && grep -n 'MarkItDown\|markitdown' backend/app/api/api_v1/chat.py && python -c "import ast; ast.parse(open('backend/app/api/api_v1/chat.py').read()); print('syntax OK')"

- [ ] **T02: Frontend — FileAttachment component + ChatInput enhancement** `est:2h`
  **Why:** The input bar has a disabled file button placeholder (S02 deliverable). S03 activates it with drag-drop, file picker, chip display, and inline error messaging.
  - Files: `frontend/src/components/chat/file-attachment.tsx`, `frontend/src/components/chat/chat-input.tsx`
  - Verify: cd /Users/tango16/code/rag-web-ui/.gsd/worktrees/M002/frontend && npx tsc --noEmit 2>&1 | head -30 && grep -n 'useDropzone' src/components/chat/file-attachment.tsx

- [ ] **T03: Frontend — Page integration, validation, and tests** `est:2h`
  **Why:** The chat page `page.tsx` owns `handleSubmit` and must switch to the multipart endpoint when a file is attached. Tests must cover both paths.
  - Files: `frontend/src/app/dashboard/chat/[id]/page.tsx`, `frontend/src/components/chat/__tests__/file-attachment.test.tsx`, `frontend/src/components/chat/__tests__/chat-input.test.tsx`
  - Verify: cd /Users/tango16/code/rag-web-ui/.gsd/worktrees/M002/frontend && npx jest src/components/chat/__tests__/ --passWithNoTests 2>&1 | tail -20 && npx tsc --noEmit 2>&1 | head -20

## Files Likely Touched

- backend/app/api/api_v1/chat.py
- backend/app/services/document_processor.py
- frontend/src/components/chat/file-attachment.tsx
- frontend/src/components/chat/chat-input.tsx
- frontend/src/app/dashboard/chat/[id]/page.tsx
- frontend/src/components/chat/__tests__/file-attachment.test.tsx
- frontend/src/components/chat/__tests__/chat-input.test.tsx
