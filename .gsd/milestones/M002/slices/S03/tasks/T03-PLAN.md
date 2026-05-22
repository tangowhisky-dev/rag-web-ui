---
estimated_steps: 3
estimated_files: 3
skills_used: []
---

# T03: Frontend — Chat page integration and tests

Why: Wire file attachment into the chat page and add comprehensive tests for the complete flow.

Do:
1. Update frontend/src/app/dashboard/chat/[id]/page.tsx:
   - Add file state (File | null) and fileError state
   - Wire onFileChange and onError to InputBar
   - In handleSubmit: when file is attached, use FormData + POST to /messages/with-file endpoint with multipart; otherwise use existing JSON endpoint
   - Read file content is handled by backend; frontend just sends the file
   - Clear file state after successful submit
2. Update frontend/src/components/chat/__tests__/chat-input.test.tsx:
   - Add tests for file attachment: file chip renders, X button removes file, error displays
   - Add tests for drag-drop interaction
3. Create frontend/src/components/chat/__tests__/file-attachment.test.tsx:
   - Test drag-drop acceptance, file chip rendering, size formatting, error states
   - Test unsupported file type detection
   - Test file removal via X button

## Inputs

- `frontend/src/app/dashboard/chat/[id]/page.tsx`
- `frontend/src/components/chat/__tests__/chat-input.test.tsx`

## Expected Output

- `frontend/src/app/dashboard/chat/[id]/page.tsx`
- `frontend/src/components/chat/__tests__/chat-input.test.tsx`
- `frontend/src/components/chat/__tests__/file-attachment.test.tsx`

## Verification

npx jest frontend/src/components/chat/__tests__/ --passWithNoTests
npx tsc --noEmit
