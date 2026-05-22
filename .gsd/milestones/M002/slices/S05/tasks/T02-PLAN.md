---
estimated_steps: 7
estimated_files: 5
skills_used: []
---

# T02: Build frontend: ChatSettings panel, pin button, export button, sidebar search

The settings panel, pin, export, and search are all frontend-only additions that hang off the existing ChatSidebar and chat detail page scaffolding from S01. They consume the backend endpoints added in T01.

1. Create `frontend/src/components/chat/chat-settings.tsx` — a slide-in panel with Tailwind transition. Props: `chat` (ChatResponse), `onClose: () => void`, `onUpdate: (patch: Partial<ChatPatch>) => void`. Renders: Model selector (gpt-4o, gpt-4o-mini, gpt-3.5-turbo), Temperature slider (0–1, step 0.1), four Switch toggles for use_dense/use_sparse/use_exact/use_graph_rag, Apply button that calls `PATCH /api/chat/:id`.
2. In `frontend/src/app/dashboard/chat/[id]/page.tsx`: add Settings icon button in header toggling `isSettingsOpen` state, render `<ChatSettings>`, add Export button that fetches `/api/chat/:id/export` as blob and triggers `<a download>`.
3. In `frontend/src/components/chat/chat-sidebar.tsx`: add Pin icon button per chat item calling `PATCH /api/chat/:id` with `{ pinned: !chat.pinned }`, add search input filtering chatList by title, show pinned chats first.
4. In `frontend/src/contexts/chat-context.tsx`: extend `Chat` interface with `pinned: boolean`; add `patchChat(id, patch)` helper.
5. Add Jest test `frontend/src/__tests__/chat-settings.test.tsx` verifying the temperature slider and retrieval toggles render.

Done when: `npx tsc --noEmit` exits 0, `npm test` passes including the new chat-settings test, and all listed files exist.

## Inputs

- `frontend/src/app/dashboard/chat/[id]/page.tsx`
- `frontend/src/components/chat/chat-sidebar.tsx`
- `frontend/src/contexts/chat-context.tsx`

## Expected Output

- `frontend/src/components/chat/chat-settings.tsx`
- `frontend/src/app/dashboard/chat/[id]/page.tsx`
- `frontend/src/components/chat/chat-sidebar.tsx`
- `frontend/src/contexts/chat-context.tsx`
- `frontend/src/__tests__/chat-settings.test.tsx`

## Verification

cd frontend && npm test -- --passWithNoTests
