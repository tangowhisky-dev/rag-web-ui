---
estimated_steps: 18
estimated_files: 2
skills_used: []
---

# T02: Build ChatContext and ChatSidebar component

Why: The chat section needs a shared state (chat list, active chat) accessible to both the sidebar and conversation pane without prop drilling. ChatSidebar encapsulates the list, new-chat button, inline rename, delete, and mobile drawer.

Do:
1. Create frontend/src/contexts/chat-context.tsx:
   - Export ChatContext (React.createContext) and ChatProvider component.
   - State: chatList: Chat[], activeChat: number | null.
   - Actions: setChatList, setActiveChat, renameChat(id, title) — calls PATCH /api/chat/:id then updates local list, deleteChat(id) — calls DELETE then removes from list.
   - Chat type: { id: number; title: string; created_at: string }.
   - On mount: fetch GET /api/chat and populate chatList.
   - Export useChatContext hook.

2. Create frontend/src/components/chat/chat-sidebar.tsx:
   - 'use client'; imports: useChatContext, useRouter, usePathname from next/navigation, Link.
   - Props: { isOpen: boolean; onClose: () => void }.
   - Renders a <aside> with fixed or relative positioning (handled by ChatLayout).
   - New Chat button: Link href='/dashboard/chat/new'.
   - Chat list: map chatList → clickable items. Active item highlighted (usePathname matches /dashboard/chat/[id]).
   - Inline rename: each item has edit icon (Pencil from lucide-react); on click shows input; on blur/Enter calls renameChat(id, value); Escape cancels.
   - Delete: Trash2 icon; onClick calls window.confirm then deleteChat(id).

Done when: component files exist with no TypeScript errors (verified by build in T03).

## Inputs

- `frontend/src/lib/api.ts`
- `frontend/src/components/ui/button.tsx`

## Expected Output

- `frontend/src/contexts/chat-context.tsx`
- `frontend/src/components/chat/chat-sidebar.tsx`

## Verification

test -f frontend/src/contexts/chat-context.tsx && test -f frontend/src/components/chat/chat-sidebar.tsx
