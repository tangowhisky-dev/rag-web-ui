"use client";

import dynamic from "next/dynamic";

const EChartsDiagramDynamic = dynamic(
  () => import("./echarts-diagram"),
  { ssr: false, loading: () => <div className="h-72 w-full animate-pulse bg-muted rounded" /> }
);

interface AgentLoopPanelProps {
  plan?: Record<string, unknown>;
  toolCalls?: Array<Record<string, unknown>>;
  toolObservations?: Array<Record<string, unknown>>;
  lastAnswerObject?: Record<string, unknown>;
  chartOption?: Record<string, unknown>;
  chartOptions?: Array<Record<string, unknown>>;
}

export function AgentLoopPanel({
  plan,
  toolCalls,
  toolObservations,
  lastAnswerObject,
  chartOption,
  chartOptions,
}: AgentLoopPanelProps) {
  const subtasks = (plan?.subtasks as Array<Record<string, unknown>>) ?? [];
  // Older messages only ever had a single chart_option; fall back to it.
  const charts = chartOptions && chartOptions.length > 0 ? chartOptions : chartOption ? [chartOption] : [];

  if (!plan && !toolCalls?.length && !toolObservations?.length && charts.length === 0) {
    return null;
  }

  return (
    <div className="mt-3 space-y-2 text-xs text-zinc-600 dark:text-zinc-400">
      {subtasks.length > 0 && (
        <details className="rounded border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900/30 p-2">
          <summary className="font-medium cursor-pointer">
            Plan ({subtasks.length} subtask{subtasks.length > 1 ? "s" : ""})
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

      {toolCalls && toolCalls.length > 0 && (
        <details className="rounded border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900/30 p-2">
          <summary className="font-medium cursor-pointer">
            Tool calls ({toolCalls.length})
          </summary>
          <ul className="mt-2 space-y-1">
            {toolCalls.map((tc, idx) => (
              <li key={idx}>
                <span className="font-semibold">{tc.tool as string}</span>{" "}
                <code className="text-[10px] break-all">
                  {JSON.stringify(tc.arguments ?? tc)}
                </code>
              </li>
            ))}
          </ul>
        </details>
      )}

      {toolObservations && toolObservations.length > 0 && (
        <details className="rounded border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900/30 p-2">
          <summary className="font-medium cursor-pointer">
            Observations ({toolObservations.length})
          </summary>
          <ul className="mt-2 space-y-1">
            {toolObservations.map((obs, idx) => (
              <li key={idx}>
                <span className="font-semibold">{obs.tool as string}</span>:{" "}
                {obs.error ? (
                  <span className="text-red-600">{obs.error as string}</span>
                ) : (
                  <span className="text-emerald-600">ok</span>
                )}
              </li>
            ))}
          </ul>
        </details>
      )}

      {lastAnswerObject && (
        <details className="rounded border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900/30 p-2">
          <summary className="font-medium cursor-pointer">Summary</summary>
          <p className="mt-2">{(lastAnswerObject.summary as string) ?? ""}</p>
        </details>
      )}

      {charts.map((option, idx) => (
        <div key={idx} className="rounded border border-zinc-200 dark:border-zinc-800 p-2">
          <EChartsDiagramDynamic code={JSON.stringify(option)} />
        </div>
      ))}
    </div>
  );
}
