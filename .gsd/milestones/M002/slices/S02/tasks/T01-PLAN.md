---
estimated_steps: 16
estimated_files: 2
skills_used: []
---

# T01: Create AgentTimeline component

Consolidate collapsible step blocks in answer.tsx into a unified AgentTimeline component with sequential step animation.

Steps:
1. Create frontend/src/components/chat/agent-timeline.tsx
2. Define TimelineStep interface with: id, label, icon, status ('pending' | 'active' | 'done' | 'error'), data (any payload), latencyMs
3. Build AgentTimeline FC accepting: rewrittenQuery, retrievedContext, queryClassification, toolTrace, failedLegs, isStreaming
4. Map SSE data to timeline steps: rewrittenQuery → "Query Rewrite" step; retrievedContext → "Retrieve" step with doc count; toolTrace → "Tool Calls" step; queryClassification → classification badge; failedLegs → error step
5. Each step renders: spinner (Lucide Loader2) while active, ✓ badge when done, ✗ when error
6. Use CSS transitions for expand/collapse — avoids flushSync conflicts during streaming
7. When isStreaming=false, collapse all detail panels and show compact badge row
8. Create agent-timeline.test.tsx with Jest tests: renders steps in order, transitions, collapses to badges, handles empty state

Must-Haves:
- AgentTimeline renders all 5 step types correctly from props
- CSS transitions animate step expand/collapse without JS animation loops
- Test file passes with Jest covering render, transition, and empty-state cases
- Import from lucide-react: Loader2, Check, X, Search, BookOpen, Share2, Wrench
- Pure presentational FC — no state management, all driven by props

## Inputs

- `frontend/src/components/chat/answer.tsx`
- `frontend/src/components/chat/__tests__/answer.test.tsx`
- `frontend/src/components/ui/badge.tsx`
- `frontend/src/components/ui/accordion.tsx`

## Expected Output

- `frontend/src/components/chat/agent-timeline.tsx`
- `frontend/src/components/chat/__tests__/agent-timeline.test.tsx`

## Verification

npm test -- --testPathPattern=agent-timeline.test.tsx --no-coverage 2>&1 | tail -20
