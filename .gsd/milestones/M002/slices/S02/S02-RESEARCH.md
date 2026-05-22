# S02: AgentTimeline + Input Bar — Research

**Date:** 2026-05-21

## Summary

S02 consolidates existing collapsible step blocks in `answer.tsx` into a unified `AgentTimeline` component, and replaces the basic `<input>` with an auto-resize textarea + animated send button + KB selector. The existing code is in good shape: `answer.tsx` already renders ThinkBlock, RewrittenQueryBlock, RetrievedContextBlock, RetrievedGraphBlock, QueryClassificationBlock, ToolTraceBlock, FailedLegsWarning, and ConfidenceCollapsible. The chat page handles `1:`, `2:`, `0:`, `3:` SSE events with `flushSync`. This is a **consolidation and polish**, not a greenfield build.

**Critical gap:** The input bar is a bare `<input>` with no auto-resize, no Shift+Enter support, and no KB selector.

## Recommendation

Build AgentTimeline as a new component replacing individual collapsible blocks. Build InputBar as a separate component. Order: (1) AgentTimeline → (2) InputBar → (3) Integrate into chat page → (4) Update tests.

## Implementation Landscape

### AgentTimeline (`frontend/src/components/chat/agent-timeline.tsx` — new)

Consolidates: RewrittenQueryBlock, QueryClassificationBlock, ToolTraceBlock, RetrievedContextBlock, RetrievedGraphBlock, FailedLegsWarning. Each step shows spinner → ✓ badge. Collapses to compact badge row when complete. SSE mapping: `1:` → Query Rewrite, `2:` → Retrieve/Classification/Tools, `0:` → streaming complete, `3:` → error.

### InputBar (`frontend/src/components/chat/chat-input.tsx` — new)

Auto-resize textarea (1-5 lines), Shift+Enter for newline, animated send button, KB selector dropdown using `@radix-ui/react-select` (already in devDependencies).

### Integration

- `answer.tsx`: Replace individual blocks with `<AgentTimeline>`, keep ThinkBlock and ConfidenceCollapsible inline
- `[id]/page.tsx`: Replace `<input>` + `<button>` with `<InputBar>`, wire `processStreamLine` to feed steps

### Testing

New test files for AgentTimeline and InputBar. Verify existing `answer.test.tsx` still passes after extraction.

## Key Files

| File | Action |
|------|--------|
| `frontend/src/components/chat/agent-timeline.tsx` | Create |
| `frontend/src/components/chat/chat-input.tsx` | Create |
| `frontend/src/components/chat/answer.tsx` | Modify |
| `frontend/src/app/dashboard/chat/[id]/page.tsx` | Modify |
| `frontend/src/components/chat/agent-timeline.test.tsx` | Create |
| `frontend/src/components/chat/chat-input.test.tsx` | Create |

## Risks

1. flushSync + animation frames — use CSS transitions, not JS animation loops
2. Test breakage on block extraction — run tests after each extraction
3. S03 depends on InputBar —预留 file attachment button slot
