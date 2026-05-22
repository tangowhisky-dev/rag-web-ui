# M002: Modern Chat UI + Agentic Workflows — Research

**Date:** 2025-01-28

## Summary

M002 is a UI-heavy milestone layered on top of a complete M001 backend pipeline. The work splits into two distinct tracks: (a) frontend UI — layout refactor, sidebar, AgentTimeline, input bar enhancements, and rendering upgrades; (b) new backend capabilities — multi-agent orchestration (LangGraph), structured extraction, and file attachment endpoint. The frontend track is sequential and must start with the DashboardLayout/sidebar refactor, which is the single biggest risk. The backend track is largely independent and can proceed in parallel.

The existing codebase is in better shape than the context suggests. `DashboardLayout` already has a working 64px sidebar shell with mobile hamburger. `answer.tsx` already renders markdown with GFM + syntax highlighting and has `ThinkBlock`, `RewrittenQueryBlock`, `RetrievedContextBlock` — the `AgentTimeline` refactor is a consolidation, not a greenfield build. The chat model already stores per-chat retrieval flags (`use_dense`, `use_sparse`, `use_exact`, `use_graph_rag`).

**Critical gaps:** No `PATCH /api/chat/:id` endpoint for rename. No file attachment endpoint. No KaTeX/Mermaid packages. No LangGraph dependency. No `AgentTimeline` component. S04 and S05 (multi-agent, structured extraction) are net-new backend work with no existing code.

## Recommendation

Start with the layout/sidebar slice (S01) — it touches `DashboardLayout` which wraps every dashboard page, making it the highest-blast-radius change and the unblocking dependency for all other frontend slices. Prove the sidebar renders and navigates correctly before touching answer rendering or the input bar.

Run S04 (multi-agent/LangGraph) and S05 (structured extraction) in parallel with the frontend track from day one — they require no shared frontend state and their backend endpoints can be integrated into the UI later.

Order within frontend: S01 → S02+S03 (parallel) → S06+S07 (parallel). File attachment (part of S01 scope) requires a small new backend multipart endpoint — plan that as a backend task within S01 or as a thin S01 dependency.

## Implementation Landscape

### Key Files

- `frontend/src/components/layout/dashboard-layout.tsx` — Current sidebar shell. Already has `w-64` fixed sidebar + mobile toggle. **Needs:** chat list panel injected into the sidebar `<nav>` area when on `/dashboard/chat/*` routes, or as a persistent second-level nav; `activeChat` context; mobile drawer close-on-select behavior.
- `frontend/src/app/dashboard/chat/[id]/page.tsx` — 400+ line SSE streaming client. Contains `processStreamLine` with `1:`, `2:`, `3:`, `0:` event handling and `flushSync`. **Needs:** messages state lifted to shared context; input bar extracted to component; `AgentTimeline` replaces per-step collapsible blocks.
- `frontend/src/app/dashboard/chat/page.tsx` — Current chat list grid page. **Needs:** content migrated into sidebar; this route can redirect to the most recent chat or the new chat page.
- `frontend/src/components/chat/answer.tsx` — Already has `ThinkBlock`, `RewrittenQueryBlock`, `RetrievedContextBlock`, citation popovers, confidence display, tool trace. Uses `react-markdown` + `remark-gfm` + `rehype-highlight`. **Needs:** KaTeX (`remark-math` + `rehype-katex`) and Mermaid (dynamic import) added; existing step blocks replaced by `AgentTimeline`.
- `backend/app/api/api_v1/chat.py` — Has `GET/POST /api/chat`, `GET/DELETE /api/chat/:id`, `POST /api/chat/:id/messages` (SSE). **Missing:** `PATCH /api/chat/:id` (rename), `POST /api/chat/:id/messages/with-file` (multipart file attachment).
- `backend/app/models/chat.py` — `Chat` model already has `use_dense`, `use_sparse`, `use_exact`, `use_graph_rag` boolean flags. **Needs nothing** for settings UI — flags are already persisted.
- `backend/app/services/chat_service.py` — Streaming pipeline. No changes needed for M002 frontend work. S04/S05 will add alongside it.

### Missing Packages

| Package | Purpose | Slice |
|---------|---------|-------|
| `remark-math` + `rehype-katex` + `katex` | LaTeX math blocks | S02 |
| `mermaid` | Diagram rendering | S02 |
| `langgraph` (Python) | Multi-agent orchestration | S04 |

### Build Order

1. **S01 — Sidebar + Layout Refactor** (blocks all frontend work)
   - Add `PATCH /api/chat/:id` to backend (rename)
   - Add `POST /api/chat/:id/messages/with-file` to backend (file attachment)
   - Refactor `DashboardLayout` to accept a sidebar slot
   - Build `ChatSidebar` component with list, create, rename, delete
   - Extract input bar to `ChatInput` component (auto-resize, file attach, KB selector)
   - Build `AgentTimeline` replacing step collapsibles
   
2. **S02 — Markdown/KaTeX/Mermaid** (parallel with S03, after S01)
   - Install packages; add `remark-math`, `rehype-katex` to `answer.tsx` markdown renderer
   - Dynamic import `mermaid` for diagram blocks; wrap in error boundary

3. **S03 — Retrieval Transparency** (parallel with S02)
   - Expand citation popover to include scores from `metadata`
   - Add confidence breakdown panel
   - Graph traversal visualization (if data is in SSE stream)

4. **S04 — Multi-Agent (LangGraph)** ← parallel backend track from day 1
   - New `multi_agent_service.py` with researcher/synthesizer/fact-checker agents
   - New SSE events for agent handoffs visible in `AgentTimeline`

5. **S05 — Structured Extraction** ← parallel backend track from day 1
   - JSON Schema constrained output mode endpoint
   - Frontend extraction request form + JSON viewer

6. **S06 — Conversation Management** (after S01)
   - Folders, pin/unpin, full-text search (backend), export (message-level export already exists), shareable links

7. **S07 — Settings UI** (after S01)
   - Model/temperature/retrieval controls panel; chat model already stores retrieval flags

### Verification Approach

- **S01:** Playwright E2E — create chat from sidebar → appears in list; rename inline; mobile drawer opens/closes at 375px
- **S02:** RTL snapshot test — markdown with `$E=mc^2$` renders KaTeX; mermaid fenced block renders SVG
- **S04:** Integration test — multi-agent query returns 3 SSE `agent:` events; timeline shows researcher/synthesizer/fact-checker steps in order

## Don't Hand-Roll

| Problem | Existing Solution | Why Use It |
|---------|------------------|------------|
| Markdown + GFM | `react-markdown` + `remark-gfm` (already installed) | Already in `answer.tsx`; extend, don't replace |
| Syntax highlighting | `rehype-highlight` (already installed) | Works; add language detection |
| LaTeX math | `remark-math` + `rehype-katex` | Standard ecosystem; KaTeX is faster than MathJax |
| Diagram rendering | `mermaid` (dynamic import) | Client-only, must be dynamic import in Next.js |
| UI components | shadcn/ui (already installed) | Sidebar, Sheet (mobile drawer), ScrollArea all available |

## Common Pitfalls

- **`DashboardLayout` blast radius** — Every dashboard page uses it. Any CSS layout change (adding `flex`, changing `pl-64`) breaks knowledge-base pages too. Test all dashboard routes after S01, not just chat.
- **`flushSync` + AgentTimeline re-renders** — `flushSync` fires synchronously inside the SSE reader; adding animation state to `AgentTimeline` during streaming can trigger React batching warnings. Keep timeline step state in a ref during streaming, flush to state only on step completion.
- **Mermaid + SSR** — `mermaid` must be dynamically imported with `{ ssr: false }` in Next.js App Router; it accesses `window` on load.
- **`PATCH /api/chat/:id` missing** — Rename will silently fail without this endpoint. Add it before building the rename UI.
- **File content size** — Inline file injection into the context window can hit LLM token limits for large files. Need server-side truncation or user warning at, e.g., 50KB.

## Open Risks

- **Layout CSS restructure scope** — The sidebar in `DashboardLayout` uses `fixed inset-y-0 left-0` positioning with `lg:pl-64` on main content. Adding a second-level chat sidebar (inside or alongside) may require switching to a flex-row layout, which changes how all dashboard pages render. Estimate 1–2 days of layout thrash.
- **S04 LangGraph complexity** — Multi-agent with visible task decomposition is a substantial new system. LangGraph state graphs, inter-agent message passing, and streaming handoff events to the SSE channel are all net-new. Budget 3–5 days of backend work.
- **Mobile gesture handling** — Swipe-to-close on the chat sidebar drawer is not covered by shadcn/ui's `Sheet` component by default. May need a gesture library or manual pointer events.

## Requirements Assessment

**Table stakes (must ship):**
- R016 (full conversation flow end-to-end) — S01
- R011 (GFM + syntax highlighting — already partially done; KaTeX/Mermaid are the gap) — S02
- R012 (message editing / branching) — **only partially in scope**; sidebar nav is in scope but true conversation branching (tree navigation) is explicitly out of scope per the context

**Well-scoped:**
- R014 (retrieval transparency) — S03; data is already in the SSE stream
- R015 (settings UI) — S07; retrieval flags already in DB model, just need UI
- R013 (conversation management) — S06; export already partially exists

**At risk / scope creep:**
- R012 "conversation tree branching" — the context explicitly descopes message editing and re-submit. The requirement as written (branching with UI navigation) goes beyond what M002 targets. **Flag as candidate for M003 or explicit descope.**
- R008/R017 (multi-agent) — significant new backend territory; if LangGraph proves complex, it can slip to M003 without breaking the frontend milestone

**Missing / advisory:**
- No requirement capturing the `AgentTimeline` component itself — the central UI innovation of M002. Consider adding a requirement for "inline animated agent step timeline that fires step-by-step as SSE events arrive and auto-collapses."
- File attachment (one-off context injection) has no corresponding requirement. Consider adding.

## Skills Discovered

| Technology | Skill | Status |
|------------|-------|--------|
| Next.js App Router | `nextjs` | Installed |
| React patterns | `react-best-practices` | Installed |
| Tailwind / shadcn | `tailwind-design-system` | Installed |
| UI polish | `make-interfaces-feel-better` | Installed |
| FastAPI | `fastapi-templates` | Installed |

All relevant skills are already installed.
