# S05: Settings UI + Conversation Management

**Goal:** Add settings panel (model, temperature, retrieval leg toggles), chat pinning, full-chat Markdown export, and sidebar search — completing the conversation management story started in S01.
**Demo:** Click settings icon in chat header: panel slides in showing model selection, temperature slider, and retrieval leg toggles (dense/sparse/exact/graph). Toggle dense off and send query: request reflects updated config. Click pin icon on a chat in sidebar: chat moves to pinned section. Click export: downloads chat as Markdown file. Type in sidebar search box: filters chat list by title/content.

## Must-Haves

- Settings panel opens from chat header, model/temperature/retrieval toggles work and persist via PATCH, pin icon moves chats to pinned section, export downloads Markdown file, sidebar search filters by title.

## Proof Level

- This slice proves: unit tests + tsc + build

## Integration Closure

Backend migration, PATCH endpoint, and export endpoint land in T01; frontend settings panel, pin button, export button, and sidebar search land in T02 consuming those endpoints. All tests green, tsc clean.

## Verification

- None — UI-only and CRUD-level backend additions with no new async flows.

## Tasks

- [ ] **T01: Extend backend: pinned migration, PATCH flags, chat export, temperature threading** `est:1.5h`
  The frontend settings panel and pin feature require backend support that does not yet exist: the Chat model has no `pinned` column, `ChatUpdate` only accepts `title`, the PATCH endpoint does not accept retrieval flags or pinned, there is no full-chat export endpoint, and temperature is hardcoded to 0 in chat_service.py.
  - Files: `backend/alembic/versions/add_pinned_to_chats.py`, `backend/app/models/chat.py`, `backend/app/schemas/chat.py`, `backend/app/api/api_v1/chat.py`, `backend/app/services/chat_service.py`
  - Verify: python3 -m pytest backend/tests/ -q --tb=short

- [ ] **T02: Build frontend: ChatSettings panel, pin button, export button, sidebar search** `est:2h`
  The settings panel, pin, export, and search are all frontend-only additions that hang off the existing ChatSidebar and chat detail page scaffolding from S01. They consume the backend endpoints added in T01.
  - Files: `frontend/src/components/chat/chat-settings.tsx`, `frontend/src/app/dashboard/chat/[id]/page.tsx`, `frontend/src/components/chat/chat-sidebar.tsx`, `frontend/src/contexts/chat-context.tsx`, `frontend/src/__tests__/chat-settings.test.tsx`
  - Verify: cd frontend && npm test -- --passWithNoTests

## Files Likely Touched

- backend/alembic/versions/add_pinned_to_chats.py
- backend/app/models/chat.py
- backend/app/schemas/chat.py
- backend/app/api/api_v1/chat.py
- backend/app/services/chat_service.py
- frontend/src/components/chat/chat-settings.tsx
- frontend/src/app/dashboard/chat/[id]/page.tsx
- frontend/src/components/chat/chat-sidebar.tsx
- frontend/src/contexts/chat-context.tsx
- frontend/src/__tests__/chat-settings.test.tsx
