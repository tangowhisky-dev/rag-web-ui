---
estimated_steps: 11
estimated_files: 2
skills_used: []
---

# T03: Add KaTeX and Mermaid to answer.tsx; enhance citation popover with retrieval scores

KaTeX:
1. Add imports: import remarkMath from 'remark-math', import rehypeKatex from 'rehype-katex', import 'katex/dist/katex.min.css'
2. Add remarkMath to remarkPlugins and rehypeKatex (throwOnError: false) to rehypePlugins in both Markdown instances.

Mermaid:
1. Import dynamic from 'next/dynamic'; const MermaidDiagramDynamic = dynamic(() => import('./mermaid-diagram'), { ssr: false })
2. Add CodeBlock component: if className includes 'language-mermaid', renders MermaidDiagramDynamic; otherwise default code element.
3. Wire CodeBlock into markdownComponents as { a: CitationLink, code: CodeBlock }.

Citation popover:
1. Add optional fields to Citation interface: score?, dense_rank?, qdrant_sparse_rank?, exact_rank?, retrieval_leg?
2. After filename block in CitationLink popover, add score percentage bar and per-leg ranks grid (skip undefined legs).
3. Show retrieval_leg as colored badge if present.

## Inputs

- `frontend/src/components/chat/answer.tsx`
- `frontend/src/components/chat/mermaid-diagram.tsx`

## Expected Output

- `frontend/src/components/chat/answer.tsx`

## Verification

cd frontend && npx tsc --noEmit
