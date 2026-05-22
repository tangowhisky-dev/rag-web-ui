---
estimated_steps: 19
estimated_files: 4
skills_used: []
---

# T03: Create ChatLayout and wire chat pages; verify build and tests

Why: Chat pages need to adopt the new two-pane layout (global nav from DashboardLayout + chat-list sidebar from ChatSidebar). The existing /dashboard/knowledge pages must not regress.

Do:
1. Create frontend/src/components/layout/chat-layout.tsx:
   - 'use client'.
   - Props: { children: React.ReactNode; pageTitle?: string; graphRagActive?: boolean }.
   - State: isChatSidebarOpen (boolean, default false for mobile).
   - Wraps DashboardLayout; inside renders a flex container with ChatSidebar + main content.
   - Hamburger button visible only on mobile (lg:hidden) to toggle isChatSidebarOpen.
   - Wraps entire render with <ChatProvider>.

2. Update frontend/src/app/dashboard/chat/page.tsx:
   - Replace DashboardLayout with ChatLayout.
   - Render empty-state prompt ('Select a chat or start a new one') — sidebar lists all chats.

3. Update frontend/src/app/dashboard/chat/[id]/page.tsx:
   - Replace DashboardLayout wrapper with ChatLayout, passing pageTitle and graphRagActive.
   - Call setActiveChat(Number(params.id)) from useChatContext on mount.

4. Update frontend/src/app/dashboard/chat/new/page.tsx:
   - Replace DashboardLayout with ChatLayout.

5. Verify /dashboard/knowledge pages still use DashboardLayout directly (no change needed).

Done when: npm run build exits 0 and npm run test:ci exits 0.

## Inputs

- `frontend/src/contexts/chat-context.tsx`
- `frontend/src/components/chat/chat-sidebar.tsx`
- `frontend/src/components/layout/dashboard-layout.tsx`
- `frontend/src/app/dashboard/chat/page.tsx`
- `frontend/src/app/dashboard/chat/[id]/page.tsx`
- `frontend/src/app/dashboard/chat/new/page.tsx`

## Expected Output

- `frontend/src/components/layout/chat-layout.tsx`

## Verification

cd frontend && npm run test:ci
