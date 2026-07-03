'use client';

import { useEffect, useId, useRef, useState } from 'react';

interface EChartsDiagramProps {
  code: string;
}

export default function EChartsDiagram({ code }: EChartsDiagramProps) {
  const [option, setOption] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const id = useId();
  const safeId = 'echarts-' + id.replace(/[^a-zA-Z0-9-_]/g, '');
  const containerRef = useRef<HTMLDivElement>(null);
  const echartsInstanceRef = useRef<unknown>(null);

  useEffect(() => {
    let cancelled = false;

    async function render() {
      try {
        const parsed = JSON.parse(code);
        if (cancelled) return;
        setOption(parsed);
        setError(null);
      } catch {
        if (!cancelled) {
          setError('Invalid echarts JSON');
          setOption(null);
        }
      }
    }

    render();
    return () => {
      cancelled = true;
    };
  }, [code]);

  useEffect(() => {
    if (!option || !containerRef.current) return;

    let cancelled = false;

    async function initChart() {
      try {
        const echarts = await import('echarts');
        const dom = containerRef.current!;

        // Dispose existing instance if any
        if (echartsInstanceRef.current) {
          const existing = echartsInstanceRef.current as { dispose: () => void };
          existing.dispose();
          echartsInstanceRef.current = null;
        }

        const instance = echarts.init(dom, undefined, { renderer: 'canvas' });
        if (cancelled) {
          instance.dispose();
          return;
        }

        instance.setOption(option! as Record<string, unknown>);
        echartsInstanceRef.current = instance;
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
        }
      }
    }

    initChart();

    return () => {
      cancelled = true;
      if (echartsInstanceRef.current) {
        const instance = echartsInstanceRef.current as { dispose: () => void };
        instance.dispose();
        echartsInstanceRef.current = null;
      }
    };
  }, [option]);

  // Resize handling
  useEffect(() => {
    if (!containerRef.current || !echartsInstanceRef.current) return;

    const observer = new ResizeObserver(() => {
      if (echartsInstanceRef.current) {
        const instance = echartsInstanceRef.current as { resize: () => void };
        instance.resize();
      }
    });

    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  if (error) {
    return (
      <div className="rounded-md border border-amber-200 bg-amber-50 dark:border-amber-800 dark:bg-amber-950/30 p-3">
        <div className="flex items-center gap-2 text-xs text-amber-700 dark:text-amber-400 mb-1">
          <span>⚠</span>
          <span className="font-medium">Chart error</span>
        </div>
        <pre className="text-[11px] text-amber-600 dark:text-amber-500 whitespace-pre-wrap break-all font-sans m-0">
          {error}
        </pre>
        <pre className="text-[11px] text-amber-600 dark:text-amber-500 whitespace-pre-wrap break-all font-sans m-0 mt-1">
          {code.slice(0, 300)}
          {code.length > 300 ? '…' : ''}
        </pre>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="echarts-diagram w-full"
      style={{ minHeight: '200px' }}
    />
  );
}
