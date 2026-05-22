# S02: AgentTimeline + Input Bar

**Goal:** Consolidate existing collapsible pipeline step blocks into a unified AgentTimeline component with animated step progression, and replace the bare input field with an auto-resize textarea + animated send button + KB selector dropdown
**Demo:** Send a query in the live app: watch AgentTimeline show Query Rewrite → Retrieve N docs → Tool Calls → Synthesize steps animate in sequence with spinners, then collapse to a compact badge row once the answer streams. Input textarea expands up to 5 lines; Shift+Enter inserts newline; KB selector dropdown changes active KBs.

## Must-Haves

- Send a query in the live app: watch AgentTimeline show Query Rewrite → Retrieve N docs → Tool Calls → Synthesize steps animate in sequence with spinners, then collapse to a compact badge row once the answer streams. Input textarea expands up to 5 lines; Shift+Enter inserts newline; KB selector dropdown changes active KBs.

## Proof Level

- This slice proves: integration

## Integration Closure

Consumes ChatContext from S01 (chat-layout.tsx). AgentTimeline replaces inline blocks in answer.tsx. ChatInput replaces bare input in [id]/page.tsx. S03 (File Attachment) depends on ChatInput's file button slot.

## Verification

- data-testid attributes on chat-input-textarea, chat-input-send-button, chat-input-kb-selector, chat-input-file-button. AgentTimeline step status transitions visible via component props.

## Tasks

- [ ] **T01: Create AgentTimeline component consolidating pipeline step blocks** `est:1h`
  The current answer.tsx has 6 separate collapsible block components (RewrittenQueryBlock, QueryClassificationBlock, ToolTraceBlock, RetrievedContextBlock, RetrievedGraphBlock, FailedLegsWarning) that render independently. Consolidate them into a single AgentTimeline component with unified animation, a compact collapsed badge row when complete, and a cleaner API for the Answer component.
  - Files: `frontend/src/components/chat/agent-timeline.tsx`, `frontend/src/components/chat/answer.tsx`
  - Verify: npx tsc --noEmit --project frontend/tsconfig.json

- [ ] **T02: Create ChatInput component with auto-resize textarea and KB selector** `est:1h`
  The current input bar in [id]/page.tsx is a bare <input> element with no auto-resize, no Shift+Enter support, and no KB selector. Build a ChatInput component with all of these features.
  - Files: `frontend/src/components/chat/chat-input.tsx`, `frontend/src/components/ui/select.tsx`
  - Verify: npx tsc --noEmit --project frontend/tsconfig.json

- [ ] **T03: Integrate AgentTimeline and ChatInput into chat page and update tests** `est:1h`
  Both new components need to be wired into the existing chat page and Answer component, and existing tests must continue passing after the refactor.
  - Files: `frontend/src/components/chat/answer.tsx`, `frontend/src/app/dashboard/chat/[id]/page.tsx`, `frontend/src/components/chat/__tests__/agent-timeline.test.tsx`, `frontend/src/components/chat/__tests__/chat-input.test.tsx`
  - Verify: cd frontend && npx jest --passWithNoTests

## Files Likely Touched

- frontend/src/components/chat/agent-timeline.tsx
- frontend/src/components/chat/answer.tsx
- frontend/src/components/chat/chat-input.tsx
- frontend/src/components/ui/select.tsx
- frontend/src/app/dashboard/chat/[id]/page.tsx
- frontend/src/components/chat/__tests__/agent-timeline.test.tsx
- frontend/src/components/chat/__tests__/chat-input.test.tsx
