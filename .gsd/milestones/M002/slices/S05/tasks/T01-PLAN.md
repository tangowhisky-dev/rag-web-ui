---
estimated_steps: 9
estimated_files: 5
skills_used: []
---

# T01: Extend backend: pinned migration, PATCH flags, chat export, temperature threading

The frontend settings panel and pin feature require backend support that does not yet exist: the Chat model has no `pinned` column, `ChatUpdate` only accepts `title`, the PATCH endpoint does not accept retrieval flags or pinned, there is no full-chat export endpoint, and temperature is hardcoded to 0 in chat_service.py.

1. Create Alembic migration `backend/alembic/versions/add_pinned_to_chats.py` — adds `pinned BOOLEAN DEFAULT FALSE NOT NULL` to `chats` table.
2. Add `pinned = Column(Boolean, nullable=False, default=False, server_default='0')` to the Chat model in `backend/app/models/chat.py`.
3. Extend `ChatUpdate` in `backend/app/schemas/chat.py` with optional fields: `title: Optional[str] = None`, `pinned: Optional[bool] = None`, `use_dense: Optional[bool] = None`, `use_sparse: Optional[bool] = None`, `use_exact: Optional[bool] = None`, `use_graph_rag: Optional[bool] = None`. Also add `pinned: bool = False` and `temperature: Optional[float] = None`, `model_name: Optional[str] = None` to `ChatResponse`.
4. Update the PATCH handler in `backend/app/api/api_v1/chat.py` to apply any provided field from `ChatUpdate` (title, pinned, retrieval flags) onto the db Chat object before commit.
5. Add `GET /api/chat/{chat_id}/export` route that fetches all messages for the chat ordered by created_at, formats them as Markdown and returns `Response(content=md, media_type='text/markdown', headers={'Content-Disposition': f'attachment; filename="chat-{chat_id}.md"'})`.
6. In `backend/app/services/chat_service.py`, add `temperature: float = 0.0` and `model_name: Optional[str] = None` parameters to `generate_response()`. Use `temperature` for the final synthesis step. Pass `model_name` to the LLM init if provided.
7. In the POST messages endpoint in `chat.py`, accept optional `temperature` and `model_name` from the request body and thread them into `generate_response()`.

Done when: `python3 -m pytest backend/tests/ -q` passes, migration file exists, PATCH endpoint accepts pinned + leg flags, export endpoint returns 200 with text/markdown content-type.

## Inputs

- `backend/app/models/chat.py`
- `backend/app/schemas/chat.py`
- `backend/app/api/api_v1/chat.py`
- `backend/app/services/chat_service.py`

## Expected Output

- `backend/alembic/versions/add_pinned_to_chats.py`
- `backend/app/models/chat.py`
- `backend/app/schemas/chat.py`
- `backend/app/api/api_v1/chat.py`
- `backend/app/services/chat_service.py`

## Verification

python3 -m pytest backend/tests/ -q --tb=short
