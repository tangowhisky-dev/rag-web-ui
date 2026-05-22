# S05: Settings UI + Conversation Management — Research

**Date:** 2026-05-21

## Summary

S05 adds settings panel (model selection, temperature slider, retrieval leg toggles), chat pinning, chat export as Markdown, and sidebar search. Chat model already stores `use_dense`/`use_sparse`/`use_exact`/`use_graph_rag`. PATCH endpoint exists for rename. **Missing:** PATCH for retrieval flags, full-chat export endpoint, `pinned` field, model/temperature per-chat config.

## Recommendation

Backend-first for PATCH extension + export endpoint + DB migration, then frontend settings panel + pin + export + search.

## Implementation Landscape

### Backend

**PATCH extension** (`backend/app/api/api_v1/chat.py`): Accept `use_dense`, `use_sparse`, `use_exact`, `use_graph_rag`, `pinned`.

**Schema** (`backend/app/schemas/chat.py`): Extend `ChatUpdate` with optional retrieval flags + `pinned`.

**Export** (`GET /api/chat/:id/export`): Concatenate messages to Markdown, return as download.

**DB migration**: Add `pinned BOOLEAN DEFAULT FALSE` to `chats` table.

**Model/temperature**: Accept in message payload, thread through `generate_response()`. Temperature hardcoded to 0 in 5 places — only final response needs user config.

### Frontend

**Settings panel** (`frontend/src/components/chat/chat-settings.tsx` — new): Slide-in with `<Select>` for model, range slider for temperature, `<Switch>` for retrieval legs. Uses existing `@radix-ui/react-switch`, `@radix-ui/react-select`, `@radix-ui/react-dialog`.

**Pin button**: In ChatSidebar, PATCH on toggle, filter pinned to top.

**Export button**: In chat header, fetch export endpoint, trigger download.

**Search filter**: Migrate from `chat/page.tsx` into ChatSidebar.

## Key Files

| File | Action |
|------|--------|
| `backend/app/api/api_v1/chat.py` | Modify — PATCH extension, export endpoint |
| `backend/app/schemas/chat.py` | Modify — extend ChatUpdate |
| `backend/app/models/chat.py` | Modify — add pinned |
| `backend/app/services/chat_service.py` | Modify — accept model/temperature |
| `backend/alembic/versions/xxxx_add_pinned.py` | Create |
| `frontend/src/components/chat/chat-settings.tsx` | Create |
| `frontend/src/components/chat/chat-sidebar.tsx` | Modify — pin, search |
| `frontend/src/app/dashboard/chat/[id]/page.tsx` | Modify — export, settings |

## Risks

1. DB migration for `pinned` field — verify Alembic setup
2. Temperature threading — hardcoded to 0 in 5 places
3. Model selection — per-chat requires backend to accept model name in request
