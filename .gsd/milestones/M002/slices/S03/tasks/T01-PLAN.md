---
estimated_steps: 7
estimated_files: 1
skills_used: []
---

# T01: Backend — Multipart file attachment endpoint

Why: Add POST /{chat_id}/messages/with-file endpoint accepting multipart/form-data for file attachment in chat. File content is converted to markdown via MarkItDown and prepended as inline context.

Do:
1. Import UploadFile, Form, Path, and write a temp file from UploadFile.read()
2. Call _convert_to_markdown() on the temp path to extract text content
3. Build augmented query: f"## File Context: {filename}\n\n{content}\n\n{message}"
4. Call generate_response() with augmented query, preserving all existing params (use_dense, use_sparse, use_exact, use_graph_rag)
5. Stream SSE response with identical 1:, 2:, 0: event format
6. Add file size validation (10 MB max) and type validation against SUPPORTED_EXTENSIONS
7. Clean up temp file in a finally block

## Inputs

- `backend/app/api/api_v1/chat.py`

## Expected Output

- `backend/app/api/api_v1/chat.py`

## Verification

curl -X POST http://localhost:8000/api/chat/1/messages/with-file -F "file=@/tmp/test.txt" -F "message=what is this" -F 'messages=[{"role":"user","content":"hi"}]' --no-buffer 2>&1 | head -5
