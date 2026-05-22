---
estimated_steps: 11
estimated_files: 2
skills_used: []
---

# T04: Add MermaidDiagram tests and verify all existing answer tests pass

1. Create frontend/src/components/chat/__tests__/mermaid-diagram.test.tsx:
   - Mock mermaid module and next/dynamic.
   - Test 1: renders SVG output when mermaid.render resolves.
   - Test 2: renders error block when mermaid.render throws.
   - Test 3: shows loading state before render resolves.
2. Update answer.test.tsx with mocks for new modules:
   jest.mock('remark-math', () => ({}))
   jest.mock('rehype-katex', () => ({}))
   jest.mock('katex/dist/katex.min.css', () => ({}))
   jest.mock('next/dynamic', () => (fn: any) => () => null)
3. All 3 existing answer tests must pass.

## Inputs

- `frontend/src/components/chat/__tests__/answer.test.tsx`

## Expected Output

- `frontend/src/components/chat/__tests__/mermaid-diagram.test.tsx`
- `frontend/src/components/chat/__tests__/answer.test.tsx`

## Verification

cd frontend && npm test -- --testPathPattern="answer|mermaid-diagram" --passWithNoTests
