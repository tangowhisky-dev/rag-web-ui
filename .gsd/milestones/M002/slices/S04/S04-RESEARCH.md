# S04: Answer Rendering + Retrieval Transparency — Research

**Date:** 2026-05-21

## Summary

S04 adds KaTeX math rendering, Mermaid diagram rendering, and enhanced citation popovers with per-document retrieval scores/leg breakdown. Existing `answer.tsx` uses `react-markdown` with `remark-gfm` + `rehype-highlight`. KaTeX/Mermaid packages are **not installed**. Citation popover already fetches document info via API. The `2:` event carries confidence, score, breakdown, failed_legs, and per-document metadata with `dense_rank`/`qdrant_sparse_rank`/`exact_rank`.

## Recommendation

Install packages → KaTeX integration → Mermaid with dynamic import → Citation popover enhancement → Tests.

## Implementation Landscape

### Packages
`npm install remark-math rehype-katex katex mermaid`

### KaTeX
Add `remarkMath` and `rehypeKatex` to `react-markdown` plugins. Import KaTeX CSS.

### Mermaid (`frontend/src/components/chat/mermaid-diagram.tsx` — new)
Use `dynamic(() => import(...), { ssr: false })` to avoid SSR. Custom `pre` component detects ` ```mermaid ` blocks.

### Citation Enhancement
Add score badge and leg breakdown (dense_rank, sparse_rank, exact_rank) to CitationLink popover. Data available directly in `2:` event metadata.

### Confidence Badge
Enhance expanded view with per-leg breakdown from `breakdown` field.

## Key Files

| File | Action |
|------|--------|
| `frontend/package.json` | Modify — add packages |
| `frontend/src/components/chat/answer.tsx` | Modify — KaTeX plugins, Mermaid handler, citations |
| `frontend/src/components/chat/mermaid-diagram.tsx` | Create |
| `frontend/src/components/chat/mermaid-diagram.test.tsx` | Create |
| `frontend/src/components/chat/answer.test.tsx` | Modify |

## Risks

1. Mermaid SSR — must use `dynamic()` with `ssr: false`
2. Bundle size — KaTeX ~25KB gzipped, Mermaid ~80KB (lazy-loaded)
3. Coordinate with S02 — both touch answer.tsx but changes are orthogonal (S02 extracts blocks, S04 adds plugins)
