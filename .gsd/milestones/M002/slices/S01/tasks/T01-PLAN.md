---
estimated_steps: 11
estimated_files: 1
skills_used: []
---

# T01: Add PATCH /api/chat/:id rename endpoint to backend

Why: The roadmap boundary map calls for PATCH /api/chat/:id to rename a chat; it is missing from the router. The ChatUpdate schema (title: str, knowledge_base_ids: Optional) already exists in schemas/chat.py.

Do:
1. Open backend/app/api/api_v1/chat.py.
2. After the existing DELETE /{chat_id} handler, add:
   @router.patch('/{chat_id}', response_model=ChatResponse)
   def update_chat(*, db, chat_id, chat_in: ChatUpdate, current_user) -> Any:
     - Query Chat, raise 404 if not found or not owned by user.
     - Update chat.title = chat_in.title if provided.
     - db.commit(); db.refresh(chat); return chat.
3. ChatUpdate already has Optional[knowledge_base_ids]; do NOT update KB associations in this endpoint — only title matters for S01.

Done when: grep finds router.patch in chat.py.

## Inputs

- `backend/app/api/api_v1/chat.py`
- `backend/app/schemas/chat.py`

## Expected Output

- `backend/app/api/api_v1/chat.py`

## Verification

grep -n 'router.patch' backend/app/api/api_v1/chat.py
