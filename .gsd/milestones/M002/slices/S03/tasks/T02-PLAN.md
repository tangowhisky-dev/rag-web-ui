---
estimated_steps: 3
estimated_files: 2
skills_used: []
---

# T02: Frontend — FileAttachment component and InputBar enhancement

Why: Create drag-drop file attachment component and extend InputBar with file handling, chip display, and error states.

Do:
1. Create frontend/src/components/chat/file-attachment.tsx:
   - useDropzone from react-dropzone for drag-drop + click-to-browse
   - File chip showing filename (truncated), file size (formatted bytes), and X button to remove
   - Error display below input for oversized (>10MB) or unsupported files
   - Visual drag-over highlight on the input bar container
2. Extend InputBar in chat-input.tsx:
   - Add file, onFileChange, error props to InputBarProps
   - Replace disabled file button with clickable trigger that opens dropzone
   - Render file chip above textarea when file is attached
   - Render error text below input bar when error is set
3. Export SUPPORTED_TYPES and MAX_FILE_SIZE constants for frontend validation

## Inputs

- `frontend/src/components/chat/chat-input.tsx`

## Expected Output

- `frontend/src/components/chat/file-attachment.tsx`
- `frontend/src/components/chat/chat-input.tsx`

## Verification

npx jest frontend/src/components/chat/__tests__/chat-input.test.tsx --passWithNoTests
npx tsc --noEmit
