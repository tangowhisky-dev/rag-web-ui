# S01: Sidebar + Layout Refactor

**Goal:** Add a persistent chat-list sidebar to the chat section (ChatGPT-style two-pane layout), add the missing PATCH rename endpoint, and ensure no layout regression on other dashboard pages.
**Demo:** Open http://localhost:3000/dashboard/chat in browser: sidebar lists existing chats; click New Chat → navigates to fresh chat; rename a chat inline; delete a chat with confirmation; on 375px viewport, sidebar is hidden and hamburger opens it. Navigate to /dashboard/knowledge-base — no layout regression.

## Must-Haves

- Open http://localhost:3000/dashboard/chat: sidebar lists existing chats; click New Chat → navigates to fresh chat; rename a chat inline via double-click → PATCH sent and title updates; delete a chat with confirm dialog → DELETE sent and item removed. On 375px viewport, sidebar is hidden and hamburger opens it. Navigate to /dashboard/knowledge-base — no layout regression.

## Proof Level

- This slice proves: integration

## Integration Closure

Upstream: GET/POST/DELETE /api/chat already exist. New wiring: PATCH /api/chat/:id (backend), ChatContext (frontend React context), ChatLayout component wrapping DashboardLayout. Other dashboard pages continue using DashboardLayout unchanged.

## Verification

- FastAPI logs PATCH /api/chat/:id requests; browser console logs ChatContext dispatches; sidebar renders chat list length as a data-testid attribute for assertion.

## Tasks

- [ ] **T01: Add PATCH /api/chat/:id rename endpoint to backend** `est:20m`
  Why: The roadmap boundary map calls for PATCH /api/chat/:id to rename a chat; it is missing from the router. The ChatUpdate schema (title: str, knowledge_base_ids: Optional) already exists in schemas/chat.py.
  - Files: `backend/app/api/api_v1/chat.py`
  - Verify: grep -n 'router.patch' backend/app/api/api_v1/chat.py

- [ ] **T02: Build ChatContext and ChatSidebar component** `est:1h`
  Why: The chat section needs a shared state (chat list, active chat) accessible to both the sidebar and conversation pane without prop drilling. ChatSidebar encapsulates the list, new-chat button, inline rename, delete, and mobile drawer.
  - Files: `frontend/src/contexts/chat-context.tsx`, `frontend/src/components/chat/chat-sidebar.tsx`
  - Verify: test -f frontend/src/contexts/chat-context.tsx && test -f frontend/src/components/chat/chat-sidebar.tsx

- [ ] **T03: Create ChatLayout and wire chat pages; verify build and tests** `est:1h30m`
  Why: Chat pages need to adopt the new two-pane layout (global nav from DashboardLayout + chat-list sidebar from ChatSidebar). The existing /dashboard/knowledge pages must not regress.
  - Files: `frontend/src/components/layout/chat-layout.tsx`, `frontend/src/app/dashboard/chat/page.tsx`, `frontend/src/app/dashboard/chat/[id]/page.tsx`, `frontend/src/app/dashboard/chat/new/page.tsx`
  - Verify: cd frontend && npm run test:ci

## Files Likely Touched

- backend/app/api/api_v1/chat.py
- frontend/src/contexts/chat-context.tsx
- frontend/src/components/chat/chat-sidebar.tsx
- frontend/src/components/layout/chat-layout.tsx
- frontend/src/app/dashboard/chat/page.tsx
- frontend/src/app/dashboard/chat/[id]/page.tsx
- frontend/src/app/dashboard/chat/new/page.tsx
