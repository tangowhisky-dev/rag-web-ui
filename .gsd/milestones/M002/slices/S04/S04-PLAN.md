# S04: Answer Rendering + Retrieval Transparency

**Goal:** Add KaTeX math rendering, Mermaid diagram rendering, and enhanced citation popovers (retrieval score + per-leg breakdown) to the Answer component. Existing answer.test.tsx tests must continue passing.
**Demo:** Send a query whose answer contains a LaTeX equation: equation renders as typeset math. Send a query that returns a Mermaid graph: diagram renders as SVG. Hover a citation: popover shows document title, score, retrieval leg (dense/sparse/exact/graph), and confidence. Expand confidence badge: see per-leg breakdown.

## Must-Haves

- KaTeX CSS imported; remarkMath and rehypeKatex in both Markdown instances\n- Mermaid code blocks routed to MermaidDiagramDynamic\n- Citation popover shows score bar + per-leg ranks when metadata contains those fields\n- npm run build exits 0\n- tsc --noEmit exits 0\n- All existing answer tests pass

## Proof Level

- This slice proves: unit tests + tsc + build

## Integration Closure

KaTeX and Mermaid packages installed; MermaidDiagram component and answer.tsx enhancements land together; all tests green

## Verification

- None — rendering-only change

## Tasks

- [ ] **T01: Install KaTeX and Mermaid npm packages** `est:10m`
  1. cd frontend && npm install remark-math rehype-katex katex mermaid
  2. npm install --save-dev @types/katex (if available)
  3. Verify the four packages appear in package.json dependencies.
  - Files: `frontend/package.json`, `frontend/package-lock.json`
  - Verify: grep -q '"remark-math"' frontend/package.json && grep -q '"rehype-katex"' frontend/package.json && grep -q '"mermaid"' frontend/package.json

- [ ] **T02: Create MermaidDiagram component with SSR-safe dynamic import** `est:30m`
  1. Create frontend/src/components/chat/mermaid-diagram.tsx:
     - Export a default MermaidDiagram component that accepts { code: string } prop.
     - useEffect runs mermaid.initialize({ startOnLoad: false, theme: 'neutral' }) once, then calls mermaid.render('mermaid-' + uniqueId, code) and sets svg into state.
     - On error render: <div className="text-red-500 text-xs p-2 border border-red-200 rounded">Mermaid error: {err.message}</div>
     - Use useId() (React 18) for stable unique id.
     - Renders <div dangerouslySetInnerHTML={{ __html: svg }}/> when svg available, else loading spinner (animate-pulse bg-gray-100 rounded h-24).
  2. The dynamic wrapper const MermaidDiagramDynamic = dynamic(() => import('./mermaid-diagram'), { ssr: false }) is used in answer.tsx.
  - Files: `frontend/src/components/chat/mermaid-diagram.tsx`
  - Verify: test -f frontend/src/components/chat/mermaid-diagram.tsx && grep -q 'dangerouslySetInnerHTML' frontend/src/components/chat/mermaid-diagram.tsx

- [ ] **T03: Add KaTeX and Mermaid to answer.tsx; enhance citation popover with retrieval scores** `est:90m`
  KaTeX:
  1. Add imports: import remarkMath from 'remark-math', import rehypeKatex from 'rehype-katex', import 'katex/dist/katex.min.css'
  2. Add remarkMath to remarkPlugins and rehypeKatex (throwOnError: false) to rehypePlugins in both Markdown instances.
  - Files: `frontend/src/components/chat/answer.tsx`, `frontend/src/components/chat/mermaid-diagram.tsx`
  - Verify: cd frontend && npx tsc --noEmit

- [ ] **T04: Add MermaidDiagram tests and verify all existing answer tests pass** `est:40m`
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
  - Files: `frontend/src/components/chat/__tests__/mermaid-diagram.test.tsx`, `frontend/src/components/chat/__tests__/answer.test.tsx`
  - Verify: cd frontend && npm test -- --testPathPattern="answer|mermaid-diagram" --passWithNoTests

## Files Likely Touched

- frontend/package.json
- frontend/package-lock.json
- frontend/src/components/chat/mermaid-diagram.tsx
- frontend/src/components/chat/answer.tsx
- frontend/src/components/chat/__tests__/mermaid-diagram.test.tsx
- frontend/src/components/chat/__tests__/answer.test.tsx
