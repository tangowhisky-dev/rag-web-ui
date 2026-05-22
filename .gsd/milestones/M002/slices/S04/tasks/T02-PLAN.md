---
estimated_steps: 7
estimated_files: 1
skills_used: []
---

# T02: Create MermaidDiagram component with SSR-safe dynamic import

1. Create frontend/src/components/chat/mermaid-diagram.tsx:
   - Export a default MermaidDiagram component that accepts { code: string } prop.
   - useEffect runs mermaid.initialize({ startOnLoad: false, theme: 'neutral' }) once, then calls mermaid.render('mermaid-' + uniqueId, code) and sets svg into state.
   - On error render: <div className="text-red-500 text-xs p-2 border border-red-200 rounded">Mermaid error: {err.message}</div>
   - Use useId() (React 18) for stable unique id.
   - Renders <div dangerouslySetInnerHTML={{ __html: svg }}/> when svg available, else loading spinner (animate-pulse bg-gray-100 rounded h-24).
2. The dynamic wrapper const MermaidDiagramDynamic = dynamic(() => import('./mermaid-diagram'), { ssr: false }) is used in answer.tsx.

## Inputs

- None specified.

## Expected Output

- `frontend/src/components/chat/mermaid-diagram.tsx`

## Verification

test -f frontend/src/components/chat/mermaid-diagram.tsx && grep -q 'dangerouslySetInnerHTML' frontend/src/components/chat/mermaid-diagram.tsx
