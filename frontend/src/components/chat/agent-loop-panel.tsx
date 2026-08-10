"use client";

import dynamic from "next/dynamic";

const EChartsDiagramDynamic = dynamic(
  () => import("./echarts-diagram"),
  { ssr: false, loading: () => <div className="h-72 w-full animate-pulse bg-muted rounded" /> }
);

interface AgentLoopPanelProps {
  plan?: Record<string, unknown>;
  chartOption?: Record<string, unknown>;
  chartOptions?: Array<Record<string, unknown>>;
}

export function AgentLoopPanel({
  plan,
  chartOption,
  chartOptions,
}: AgentLoopPanelProps) {
  const subtasks = (plan?.subtasks as Array<Record<string, unknown>>) ?? [];
  // Older messages only ever had a single chart_option; fall back to it.
  const charts = chartOptions && chartOptions.length > 0 ? chartOptions : chartOption ? [chartOption] : [];

  // Only show Plan for complex queries (more than 1 subtask).
  // Simple queries with a single subtask don't need a plan display —
  // the SubtaskList component in answer.tsx already handles this with
  // its own `taskList.length > 1` condition.
  const showPlan = subtasks.length > 1;

  if (!showPlan && charts.length === 0) {
    return null;
  }

  return (
    <div className="mt-3 space-y-2 text-xs text-zinc-600 dark:text-zinc-400">
      {showPlan && (
        <details className="rounded border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900/30 p-2">
          <summary className="font-medium cursor-pointer">
            Plan ({subtasks.length} subtasks)
          </summary>
          <ol className="mt-2 ml-4 list-decimal space-y-1">
            {subtasks.map((st, idx) => (
              <li key={idx}>
                {st.description as string} {st.tool_hint ? `(${st.tool_hint})` : ""}
              </li>
            ))}
          </ol>
        </details>
      )}

      {/* Tool calls, observations, and summary are intentionally not
          shown in the UI — they are available in backend debug logs. */}

      {charts.map((option, idx) => (
        <div key={idx} className="rounded border border-zinc-200 dark:border-zinc-800 p-2">
          <EChartsDiagramDynamic code={JSON.stringify(option)} />
        </div>
      ))}
    </div>
  );
}
