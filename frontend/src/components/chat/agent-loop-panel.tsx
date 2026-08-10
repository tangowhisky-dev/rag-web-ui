"use client";

import dynamic from "next/dynamic";

const EChartsDiagramDynamic = dynamic(
  () => import("./echarts-diagram"),
  { ssr: false, loading: () => <div className="h-72 w-full animate-pulse bg-muted rounded" /> }
);

interface AgentLoopPanelProps {
  chartOption?: Record<string, unknown>;
  chartOptions?: Array<Record<string, unknown>>;
}

export function AgentLoopPanel({
  chartOption,
  chartOptions,
}: AgentLoopPanelProps) {
  // Older messages only ever had a single chart_option; fall back to it.
  const charts = chartOptions && chartOptions.length > 0 ? chartOptions : chartOption ? [chartOption] : [];

  if (charts.length === 0) {
    return null;
  }

  return (
    <div className="mt-3 space-y-2 text-xs text-zinc-600 dark:text-zinc-400">
      {charts.map((option, idx) => (
        <div key={idx} className="rounded border border-zinc-200 dark:border-zinc-800 p-2">
          <EChartsDiagramDynamic code={JSON.stringify(option)} />
        </div>
      ))}
    </div>
  );
}
