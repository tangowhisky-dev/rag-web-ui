import { FC, useMemo, useState, useEffect, useCallback } from "react";
import { Loader2, Check, X, Search, BookOpen, Share2, Wrench } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

// ── Types ────────────────────────────────────────────────────────────────────

export interface TimelineStep {
  id: string;
  label: string;
  icon: "search" | "book" | "share" | "wrench" | "check" | "x";
  status: "pending" | "active" | "done" | "error";
  data?: Record<string, unknown>;
  latencyMs?: number;
}

interface ContextDoc {
  page_content: string;
  metadata: Record<string, any>;
}

interface QueryClassification {
  type: string;
  confidence: number;
  latency_ms: number;
  fallback: boolean;
}

interface ToolTraceEntry {
  tool_name: string;
  params?: Record<string, unknown>;
  output?: unknown;
  error?: string | null;
  latency_ms: number;
}

interface AgentTimelineProps {
  rewrittenQuery?: string;
  retrievedContext?: ContextDoc[];
  queryClassification?: QueryClassification;
  toolTrace?: ToolTraceEntry[];
  failedLegs?: string[];
  isStreaming?: boolean;
}

// ── Step status icons ────────────────────────────────────────────────────────

const StatusIcon: FC<{ status: TimelineStep["status"] }> = ({ status }) => {
  switch (status) {
    case "active":
      return <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-500" />;
    case "done":
      return <Check className="h-3.5 w-3.5 text-emerald-500" />;
    case "error":
      return <X className="h-3.5 w-3.5 text-red-500" />;
    case "pending":
    default:
      return <div className="h-3.5 w-3.5 rounded-full border-2 border-zinc-300 dark:border-zinc-600" />;
  }
};

const ICON_MAP: Record<string, FC<{ className?: string }>> = {
  search: ({ className }) => <Search className={className} />,
  book: ({ className }) => <BookOpen className={className} />,
  share: ({ className }) => <Share2 className={className} />,
  wrench: ({ className }) => <Wrench className={className} />,
  check: ({ className }) => <Check className={className} />,
  x: ({ className }) => <X className={className} />,
};

// ── Single step row ──────────────────────────────────────────────────────────

const TimelineStepRow: FC<{ step: TimelineStep; isExpanded: boolean; onToggle: () => void }> = ({
  step,
  isExpanded,
  onToggle,
}) => {
  const Icon = ICON_MAP[step.icon] || Wrench;

  const statusColor = useMemo(() => {
    switch (step.status) {
      case "active":
        return "border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-900/10";
      case "done":
        return "border-emerald-200 dark:border-emerald-800 bg-emerald-50 dark:bg-emerald-900/10";
      case "error":
        return "border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/10";
      default:
        return "border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-800/40";
    }
  }, [step.status]);

  const detailText = useMemo(() => {
    if (step.status === "pending") return null;
    if (step.id === "query_rewrite" && step.data?.query) return step.data.query as string;
    if (step.id === "retrieve" && step.data?.totalDocCount !== undefined)
      return `${step.data.totalDocCount} document${(step.data.totalDocCount as number) !== 1 ? "s" : ""}`;
    if (step.id === "classification" && step.data?.type)
      return `${step.data.type} · ${Math.round((step.data.confidence as number) * 100)}%`;
    if (step.id === "tool_calls" && step.data?.tools)
      return (step.data.tools as string[]).join(", ");
    if (step.id === "failed_legs" && step.data?.legs)
      return (step.data.legs as string[]).join(", ") + " failed";
    return null;
  }, [step.id, step.status, step.data]);

  return (
    <div className={cn("rounded-md border overflow-hidden transition-all duration-200", statusColor)}>
      <button
        onClick={onToggle}
        className="flex items-center gap-2 w-full px-3 py-1.5 text-left hover:bg-black/5 dark:hover:bg-white/5 transition-colors"
      >
        <StatusIcon status={step.status} />
        <Icon className="h-3.5 w-3.5 text-zinc-500 dark:text-zinc-400 shrink-0" />
        <span className="text-xs font-medium text-zinc-600 dark:text-zinc-300 select-none">
          {step.label}
        </span>
        {step.latencyMs !== undefined && (
          <span className="text-[10px] text-zinc-400 ml-auto select-none">
            {step.latencyMs.toFixed(0)}ms
          </span>
        )}
        <span className="text-[10px] text-zinc-400 ml-auto select-none">
          {isExpanded ? "▼" : "▶"}
        </span>
      </button>

      {/* CSS transition for expand/collapse — no JS animation loops */}
      <div
        className={cn(
          "transition-all duration-200 ease-in-out overflow-hidden",
          isExpanded ? "max-h-48 opacity-100" : "max-h-0 opacity-0"
        )}
      >
        {detailText && (
          <div className="px-3 pb-2 pt-1 border-t border-current/10">
            <p className="text-[11px] leading-[1.45] text-zinc-500 dark:text-zinc-400 whitespace-pre-wrap break-words font-sans m-0">
              {detailText}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

// ── Compact badge row (post-streaming) ───────────────────────────────────────

const BadgeRow: FC<{ steps: TimelineStep[] }> = ({ steps }) => {
  const visibleSteps = steps.filter((s) => s.status !== "pending");

  if (visibleSteps.length === 0) return null;

  return (
    <div className="flex items-center gap-1.5 flex-wrap not-prose">
      {visibleSteps.map((step) => {
        const variant =
          step.status === "error"
            ? "destructive"
            : step.status === "done"
            ? "default"
            : "secondary";

        return (
          <Badge
            key={step.id}
            variant={variant}
            className={cn(
              "text-[10px] gap-1",
              step.status === "done" && "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400 border-emerald-200 dark:border-emerald-800",
              step.status === "error" && "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400 border-red-200 dark:border-red-800"
            )}
          >
            <StatusIcon status={step.status} />
            {step.label}
            {step.latencyMs !== undefined && (
              <span className="opacity-60">{step.latencyMs.toFixed(0)}ms</span>
            )}
          </Badge>
        );
      })}
    </div>
  );
};

// ── Main component ───────────────────────────────────────────────────────────

export const AgentTimeline: FC<AgentTimelineProps> = ({
  rewrittenQuery,
  retrievedContext,
  queryClassification,
  toolTrace,
  failedLegs,
  isStreaming = false,
}) => {
  // Build timeline steps from props
  const steps: TimelineStep[] = useMemo(() => {
    const result: TimelineStep[] = [];

    // 1. Query Rewrite step
    if (rewrittenQuery) {
      result.push({
        id: "query_rewrite",
        label: "Query Rewrite",
        icon: "search",
        status: "done",
        data: { query: rewrittenQuery },
      });
    }

    // 2. Classification step
    if (queryClassification) {
      result.push({
        id: "classification",
        label: "Classification",
        icon: "check",
        status: "done",
        data: {
          type: queryClassification.type,
          confidence: queryClassification.confidence,
        },
        latencyMs: queryClassification.latency_ms,
      });
    }

    // 3. Retrieve step
    if (retrievedContext && retrievedContext.length > 0) {
      const docCount = retrievedContext.filter(
        (d) => d.metadata?.source !== "graph"
      ).length;
      const graphCount = retrievedContext.filter(
        (d) => d.metadata?.source === "graph"
      ).length;

      result.push({
        id: "retrieve",
        label: "Retrieve",
        icon: "book",
        status: "done",
        data: { totalDocCount: docCount + graphCount, docCount, graphCount },
      });
    }

    // 4. Tool Calls step
    if (toolTrace && toolTrace.length > 0) {
      const totalLatency = toolTrace.reduce((s, t) => s + t.latency_ms, 0);
      const tools = [...new Set(toolTrace.map((t) => t.tool_name))];
      const hasError = toolTrace.some((t) => t.error);

      result.push({
        id: "tool_calls",
        label: "Tool Calls",
        icon: "wrench",
        status: hasError ? "error" : "done",
        data: { tools, count: toolTrace.length },
        latencyMs: totalLatency,
      });
    }

    // 5. Failed Legs step
    if (failedLegs && failedLegs.length > 0) {
      result.push({
        id: "failed_legs",
        label: "Failed Legs",
        icon: "x",
        status: "error",
        data: { legs: failedLegs },
      });
    }

    return result;
  }, [rewrittenQuery, retrievedContext, queryClassification, toolTrace, failedLegs]);

  // Track expanded state per step
  const [expandedSteps, setExpandedSteps] = useState<Set<string>>(() => new Set());

  // Auto-collapse when streaming ends
  const prevStreaming = isStreaming;
  useEffect(() => {
    if (!isStreaming) {
      setExpandedSteps(new Set());
    }
  }, [isStreaming]);

  const toggleStep = useCallback((id: string) => {
    setExpandedSteps((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }, []);

  // No steps at all — render nothing
  if (steps.length === 0) return null;

  // When streaming is done, show compact badge row
  if (!isStreaming) {
    return <BadgeRow steps={steps} />;
  }

  // During streaming, show expandable step blocks
  return (
    <div className="flex flex-col gap-1.5 my-2">
      {steps.map((step) => (
        <TimelineStepRow
          key={step.id}
          step={step}
          isExpanded={expandedSteps.has(step.id)}
          onToggle={() => toggleStep(step.id)}
        />
      ))}
    </div>
  );
};
