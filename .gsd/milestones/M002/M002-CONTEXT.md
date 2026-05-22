# M002: Modern Chat UI + Agentic Workflows

**Gathered:** 2025-01-28
**Status:** Ready for planning

## Project Description

A full visual redesign of the RAG Web UI chat experience. The backend pipeline (M001) is complete — M002 makes that pipeline legible and enjoyable to use. Two major structural changes: (1) a ChatGPT-style persistent left sidebar for chat history navigation, and (2) a redesigned conversation pane with an inline animated agent step timeline, polished message bubbles, and an upgraded input bar.

## Why This Milestone

M001 wired a sophisticated retrieval pipeline — query classification, hybrid 3-leg retrieval, tool calls, synthesis mode, GraphRAG — but the UI still shows a single-column page with plain collapsible blocks. Users can't easily navigate between chats, can't see the pipeline firing in a clear way, and the input bar is a basic `<input>`. M002 surfaces all that power in an interface people actually want to use. It also closes the gap between this project and the baseline experience people expect from modern chat UIs.

## User-Visible Outcome

### When this milestone is complete, the user can:

- Open the app and see a left sidebar listing all their chats; click any to jump to it; create a new chat from the sidebar; rename or delete chats inline
- Send a message and watch an inline animated step-by-step timeline appear in real time (query rewrite → retrieve N docs → tool calls → synthesize), then auto-collapse into a compact badge once the answer streams in
- Drag-and-drop (or click) a file into the chat input to attach it as one-off context for the current question (no KB ingestion)
- Use the KB selector in the input bar to choose which knowledge bases are active for the current chat
- Type in an auto-resizing textarea (Shift+Enter for newline); see an animated loading indicator on the send button while streaming

### Entry point / environment

- Entry point: `http://localhost:3000/dashboard/chat`
- Environment: local dev (Docker Compose) and browser
- Live dependencies involved: FastAPI backend (`/api/chat/*` endpoints), Qdrant, MySQL — same as M001; no new backend services

## Completion Class

- Contract complete means: Playwright/Jest component tests cover sidebar navigation, agent timeline rendering, file attachment UI, KB selector, and auto-resize textarea
- Integration complete means: full chat flow works end-to-end in Docker Compose — open sidebar, create chat, send query, watch live timeline, receive answer with citations
- Operational complete means: streaming performance is acceptable (<50ms first-token time unaffected by UI changes); no layout jank on mobile viewport

## Final Integrated Acceptance

To call this milestone complete, we must prove:

- User can create a new chat from the sidebar, send a query, watch the inline agent timeline fire step by step, receive a streamed answer with citations, then click another chat in the sidebar and see its history — all in one browser session against the live Docker stack
- File attachment UI appears on the input bar; dragging a file attaches it and the filename shows in the input; submitting the message includes the file context in the request
- On mobile viewport (375px) the sidebar collapses to a drawer accessible via a hamburger icon; the conversation pane fills the screen

## Architectural Decisions

### Sidebar state management

**Decision:** Sidebar chat list lives in a shared React context (or Zustand slice) rather than lifted to the page component.

**Rationale:** The sidebar and conversation pane are siblings in the new layout; they need to share `activeChat` and `chatList` state without prop-drilling through the layout wrapper. A context store also makes optimistic updates (rename, delete) easy without re-fetching.

**Alternatives Considered:**
- URL-only state (rely on Next.js router) — loses real-time list updates (new chat titles from the backend) without polling
- Lifted to layout page — works but makes `dashboard-layout.tsx` a god component

### Inline agent timeline replaces collapsible blocks

**Decision:** Replace the existing per-step collapsible blocks (RewrittenQueryBlock, RetrievedContextBlock, QueryClassificationBlock, ToolTraceBlock) with a single unified `AgentTimeline` component that renders steps sequentially as SSE events arrive.

**Rationale:** The current collapsibles each auto-expand then auto-collapse independently, causing visual noise. A unified timeline fires in order, animates each step as it arrives, and collapses as one unit. It reuses the same `1:`, `2:` SSE event data — no backend changes needed.

**Alternatives Considered:**
- Keep existing collapsibles, just restyle — doesn't solve the independent-collapse noise problem
- Move to a side drawer — loses the "watching the agent think" live feel

### File attachment as one-off context injection

**Decision:** File attachment adds the file content to the message payload as an inline context block — it is NOT ingested into a knowledge base.

**Rationale:** One-off file questions are a common use case ("check this contract against what's in my KB"). Ingestion would require background processing and change the KB permanently. Inline injection is immediate and scoped to the current message.

**Alternatives Considered:**
- Trigger KB ingestion — too slow, modifies persistent state
- Disable for non-text files — acceptable limitation for MVP; images deferred to OCR milestone

## Error Handling Strategy

Streaming errors (lost connection mid-stream, backend 5xx) already surface via the `3:` SSE channel and are caught in `handleSubmit`. M002 should surface these more visibly in the timeline — the failing step should show a red ✗ badge with the error message inline, rather than silently removing the assistant message.

File attachment failures (oversized file, unsupported type) should show an inline error below the input bar, not a toast, so the user can correct without losing their typed message.

Sidebar operations (rename, delete, create) should use optimistic updates with rollback on failure; a toast is acceptable for rollback notification.

## Risks and Unknowns

- **Layout CSS rework scope** — The existing `DashboardLayout` wraps every page. Introducing a resizable sidebar may require restructuring the layout tree, which touches every dashboard page (chat, knowledge-base, etc.). Estimate is fuzzy.
- **Streaming + timeline animation performance** — Adding per-step animation during high-throughput token streaming (100+ tokens/sec) could cause frame drops. FlushSync approach from M001 may need adjustment.
- **File attachment size/type limits** — Backend `/api/chat/:id/messages` doesn't currently accept file attachments. A new backend endpoint or multipart message format is required. This is a small backend addition but is a dependency for the file attachment slice.
- **Mobile sidebar** — Existing layout is not responsive. Adding a collapsible mobile drawer requires media queries and gesture handling (swipe to close). Scope could creep.

## Existing Codebase / Prior Art

- `frontend/src/app/dashboard/chat/[id]/page.tsx` — current chat page; contains all message state, streaming logic, and `processStreamLine`; will be refactored
- `frontend/src/components/chat/answer.tsx` — `Answer` component with all existing collapsible step blocks; `AgentTimeline` replaces most of these
- `frontend/src/components/layout/dashboard-layout.tsx` — current layout wrapper; will need sidebar slot added
- `frontend/src/app/dashboard/chat/page.tsx` — chat list page; content will migrate into the sidebar
- `backend/app/api/` — FastAPI routes; a new multipart endpoint needed for file attachment
- `backend/app/services/chat_service.py` — streaming pipeline; no changes needed for M002

## Relevant Requirements

- R (core-capability) — chat UI must surface the full RAG pipeline in a legible way
- M001 key files (query_classifier, retrieval, tool_registry, chat_service) — all feed data into the new `AgentTimeline` via existing `1:`, `2:` SSE events

## Scope

### In Scope

- Persistent left sidebar: chat list, new chat button, rename/delete inline, KB selector per chat
- Redesigned conversation pane layout (message bubbles, spacing, typography)
- `AgentTimeline` component replacing existing step collapsibles — live animated, auto-collapses
- Input bar: auto-resize textarea, animated send button, file attachment button, KB selector
- File attachment: UI + new backend multipart endpoint for one-off context injection
- Mobile-responsive sidebar (collapsible drawer)
- Dark mode: existing Tailwind theming is sufficient; no new theming work

### Out of Scope / Non-Goals

- New backend agentic primitives (background jobs, scheduled queries, multi-step planning)
- Slash command palette
- Message editing / re-submit
- Per-message reactions or feedback
- KB ingestion from chat file attachment
- Image/OCR file attachment (text files and PDFs only for MVP)

## Technical Constraints

- Next.js 14 App Router — keep existing routing structure; sidebar is a layout-level component, not a page
- Tailwind CSS + shadcn/ui — all new UI uses existing component library; no new CSS-in-JS
- No new npm packages beyond what's already in `package.json` unless strictly necessary
- Streaming: `flushSync` for step events must still fire before token chunks; `AgentTimeline` must not block token rendering

## Integration Points

- `GET /api/chat` — sidebar fetches chat list on mount
- `POST /api/chat` — sidebar creates new chat
- `PATCH /api/chat/:id` — rename chat
- `DELETE /api/chat/:id` — delete chat
- `POST /api/chat/:id/messages` (existing SSE endpoint) — unchanged; `AgentTimeline` reads `1:`, `2:` events from the same stream
- `POST /api/chat/:id/messages/with-file` (new) — multipart endpoint for file attachment; backend injects file content as context prefix

## Testing Requirements

- **Unit (Jest/RTL):** `AgentTimeline` renders correct steps in order; file attachment button shows filename on select; KB selector updates active KBs
- **Integration (MSW):** full streaming sequence with `1:`, `2:`, `0:` events produces correct timeline + answer
- **E2E (Playwright):** sidebar CRUD (create, rename, delete chat); send message + watch timeline animate + answer arrives; file attachment flow; mobile drawer opens/closes
- **Performance:** streaming frame rate stays ≥30fps during 100-token/sec burst (Lighthouse throttled)

## Acceptance Criteria

- Sidebar lists existing chats on load; creating a new chat adds it to the sidebar and navigates to it
- Agent timeline appears inline as SSE events fire; shows ✓ step badges for query rewrite, retrieve, tool calls, synthesis; collapses to a compact row after answer completes
- Input auto-resizes up to 5 lines; Shift+Enter inserts newline; Enter submits
- File attachment: drag-drop or click attaches file, filename shown in input, message includes file context
- KB selector in input bar persists the selection per chat (stored in chat metadata or local state)
- Mobile at 375px: sidebar hidden by default, opens via hamburger, closes on chat selection or outside tap
- All existing Jest tests in `answer.test.tsx` pass after refactor
- No TypeScript errors; no ESLint errors

## Open Questions

- **File attachment backend format** — Should file content be prepended to the user message as a system-prompt injection, or sent as a separate multipart field? Current thinking: send as `file_context` field alongside `messages`; backend prepends to the context window.
- **KB selector persistence** — Store active KBs per chat in the `chats` DB table (backend change) or in localStorage (frontend-only)? Current thinking: localStorage for MVP to avoid a schema migration.
- **Sidebar width** — Fixed (260px) or resizable? Current thinking: fixed for MVP.
