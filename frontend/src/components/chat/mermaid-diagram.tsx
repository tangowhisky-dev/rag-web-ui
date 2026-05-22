'use client';

import { useEffect, useId, useState } from 'react';

interface MermaidDiagramProps {
  code: string;
}

export default function MermaidDiagram({ code }: MermaidDiagramProps) {
  const [svg, setSvg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const id = useId();
  // useId returns a string like ":r0:" which is not valid as a DOM id; sanitize it
  const safeId = 'mermaid-' + id.replace(/[^a-zA-Z0-9-_]/g, '');

  useEffect(() => {
    let cancelled = false;

    async function render() {
      try {
        const mermaid = (await import('mermaid')).default;
        mermaid.initialize({ startOnLoad: false, theme: 'neutral' });
        const { svg: renderedSvg } = await mermaid.render(safeId, code);
        if (!cancelled) {
          setSvg(renderedSvg);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
          setSvg(null);
        }
      }
    }

    render();

    return () => {
      cancelled = true;
    };
  }, [code, safeId]);

  if (error) {
    return (
      <div className="text-red-500 text-xs p-2 border border-red-200 rounded">
        Mermaid error: {error}
      </div>
    );
  }

  if (!svg) {
    return (
      <div className="animate-pulse bg-gray-100 rounded h-24" aria-label="Loading diagram" />
    );
  }

  return (
    <div
      className="mermaid-diagram overflow-auto"
      // eslint-disable-next-line react/no-danger
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}
