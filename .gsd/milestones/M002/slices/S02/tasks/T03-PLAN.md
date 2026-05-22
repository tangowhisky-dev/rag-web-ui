---
estimated_steps: 14
estimated_files: 3
skills_used: []
---

# T03: Integrate AgentTimeline and InputBar into chat page; verify all tests pass

Wire AgentTimeline and InputBar into the real chat page and answer component.

Steps:
1. Modify frontend/src/components/chat/answer.tsx: import AgentTimeline, replace inline block rendering with <AgentTimeline>, keep ThinkBlock and ConfidenceCollapsible inline
2. Modify frontend/src/app/dashboard/chat/[id]/page.tsx: import InputBar, replace form with <InputBar>, wire props
3. Add isStreaming prop to Answer component (derived from isLoading state)
4. Run the full test suite: npm test
5. Run TypeScript check: npx tsc --noEmit
6. Verify no regressions in answer.test.tsx

Must-Haves:
- answer.tsx renders AgentTimeline instead of individual blocks
- page.tsx uses InputBar instead of bare input
- All Jest tests pass
- TypeScript compiles cleanly
- No ESLint errors in modified files

## Inputs

- `frontend/src/components/chat/agent-timeline.tsx`
- `frontend/src/components/chat/chat-input.tsx`
- `frontend/src/components/chat/answer.tsx`
- `frontend/src/app/dashboard/chat/[id]/page.tsx`

## Expected Output

- `frontend/src/components/chat/answer.tsx`
- `frontend/src/app/dashboard/chat/[id]/page.tsx`
- `frontend/src/components/chat/__tests__/answer.test.tsx`

## Verification

npm test --no-coverage 2>&1 | tail -10
