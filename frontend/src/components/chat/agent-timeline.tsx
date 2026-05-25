import { FC, useMemo, useState, useEffect, useCallback } from "react";
import { Loader2, Check, X, Search, BookOpen, Share2, Wrench } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

// ── Types ────────────────────────────────────────────────────────────────────

export interface TimelineStep {
  id: string;
  label: string;
  activeLabel?: string;   // label shown while status === "active"
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

interface AgentStepEvent {
  node: string;
  latency_ms: number;
  status: string;
  [key: string]: unknown;
}

interface AgentTimelineProps {
  rewrittenQuery?: string;
  retrievedContext?: ContextDoc[];
  queryClassification?: QueryClassification;
  toolTrace?: ToolTraceEntry[];
  failedLegs?: string[];
  agentSteps?: AgentStepEvent[];
  isStreaming?: boolean;
  answerStarted?: boolean;
}

// ── Node name → display metadata ────────────────────────────────────────────

const NODE_META: Record<string, {
  label: string;
  activeLabel?: string;
  icon: TimelineStep["icon"];
}> = {
  rewrite_query:         { label: "Rewritten Query",          activeLabel: "Rewriting query…",              icon: "search"  },
  context_router:        { label: "Context Routing",          activeLabel: "Routing sources…",               icon: "share"   },
  decompose_query:       { label: "Sub-queries",              activeLabel: "Decomposing query…",             icon: "search"  },
  parallel_retrieval:    { label: "Retrieved Context",        activeLabel: "Retrieving for each sub-query…", icon: "book"    },
  graph_enrichment:      { label: "Additional Context",       activeLabel: "Fetching graph context…",        icon: "share"   },
  extract_file_sections: { label: "File Sections",            activeLabel: "Extracting file sections…",      icon: "wrench"  },
  draft_answer:          { label: "Draft Answer",             activeLabel: "Drafting answer…",               icon: "wrench"  },
  grade_coverage:        { label: "Coverage Check",           activeLabel: "Checking coverage…",             icon: "check"   },
  widened_retrieval:     { label: "Widened Search",           activeLabel: "Widening search…",               icon: "book"    },
  keyword_search_loop:   { label: "Keyword Search",           activeLabel: "Searching keywords…",            icon: "search"  },
  generate_answer:       { label: "Generating Answer",        activeLabel: "Generating answer…",             icon: "wrench"  },
  // legacy / fast-pipeline nodes (kept for backwards compat)
  kb_retrieval:          { label: "Retrieved Context",        activeLabel: "Retrieving context…",            icon: "book"    },
  grade_documents:       { label: "Grade Docs",               activeLabel: "Grading documents…",             icon: "check"   },
  merge_context:         { label: "Merge Context",            activeLabel: "Merging context…",               icon: "book"    },
};

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

const TimelineStepRow: FC<{ step: TimelineStep; isExpanded: boolean; onToggle: () => void; rewrittenQuery?: string }> = ({
  step,
  isExpanded,
  onToggle,
  rewrittenQuery,
}) => {
  const Icon = ICON_MAP[step.icon] || Wrench;
  // Show activeLabel while running so users see "Rewriting query…" etc.
  const displayLabel = step.status === "active" && step.activeLabel ? step.activeLabel : step.label;

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
    const d = step.data ?? {};

    switch (step.id) {
      // live step nodes
      case "rewrite_query":
        return rewrittenQuery
          ? rewrittenQuery
          : (d.rewritten_query ? String(d.rewritten_query) : null);
      case "context_router": {
        const sources = d.sources as string[] | undefined;
        const rationale = d.rationale as string | undefined;
        const parts: string[] = [];
        if (sources?.length) parts.push(`Sources: ${sources.join(", ")}`);
        if (rationale) parts.push(`Rationale: ${rationale}`);
        return parts.length ? parts.join("\n") : null;
      }
      case "kb_retrieval":
      case "parallel_retrieval": {
        const status = d.status as string;
        if (status === "skipped") return "Skipped (no KB sources needed)";
        const chunks = d.chunks as Array<{ preview: string; source: string; score: number; retrieval_count?: number }> | undefined;
        if (chunks?.length) {
          return chunks
            .map((c, i) => {
              const badge = c.retrieval_count && c.retrieval_count > 1 ? ` ×${c.retrieval_count}` : "";
              return `${i + 1}. ${c.preview}${c.source ? ` [${c.source}]` : ""}${badge}`;
            })
            .join("\n");
        }
        const found = d.docs_found as number | undefined;
        const searched = d.sub_queries_searched as number | undefined;
        if (found !== undefined) {
          return `${found} doc(s) retrieved${searched && searched > 1 ? ` across ${searched} sub-queries` : ""}`;
        }
        return null;
      }
      case "decompose_query": {
        const sqs = d.sub_queries as string[] | undefined;
        if (sqs?.length) {
          return sqs.map((sq, i) => `${i + 1}. ${sq}`).join("\n");
        }
        return null;
      }
      case "draft_answer": {
        const chars = d.draft_chars as number | undefined;
        return chars ? `Draft generated (${chars} chars)` : null;
      }
      case "grade_coverage": {
        const lines = d.coverage_lines as string[] | undefined;
        const attempt = d.attempt as number | undefined;
        const prefix = attempt !== undefined && attempt > 0 ? `Attempt ${attempt + 1}\n` : "";
        return lines?.length ? prefix + lines.join("\n") : null;
      }
      case "widened_retrieval": {
        const newDocs = d.new_docs_found as number | undefined;
        const total = d.total_docs as number | undefined;
        const threshold = d.threshold_relaxed_to as number | undefined;
        const parts: string[] = [];
        if (d.uncovered_sub_queries) {
          const uqs = d.uncovered_sub_queries as string[];
          parts.push(`Retrying for: ${uqs.join("; ")}`);
        }
        if (newDocs !== undefined) parts.push(`+${newDocs} new docs`);
        if (total !== undefined) parts.push(`${total} total`);
        if (threshold !== undefined) parts.push(`threshold relaxed to ${threshold}`);
        return parts.length ? parts.join("  ·  ") : null;
      }
      case "keyword_search_loop": {
        const iters = d.keyword_iterations as Array<{ sub_query: string; iteration: string; keywords: string[]; results_found: number }> | undefined;
        if (iters?.length) {
          return iters
            .map(it => `[${it.iteration}] "${it.keywords.join(", ")}" → ${it.results_found} result(s)`)
            .join("\n");
        }
        return d.new_docs_found !== undefined ? `${d.new_docs_found} doc(s) found` : null;
      }
      case "graph_enrichment": {
        const lines = d.context_lines as string[] | undefined;
        const graphDocs = d.graph_docs as number | undefined;
        const enrichedDocs = d.enriched_docs as number | undefined;
        const parts: string[] = [];
        if (graphDocs) parts.push(`${graphDocs} related chunk(s) from graph`);
        if (enrichedDocs) parts.push(`${enrichedDocs} enriched`);
        if (lines?.length) {
          return (parts.length ? parts.join("  ·  ") + "\n\n" : "") +
            lines.map((l, i) => `${i + 1}. ${l}`).join("\n");
        }
        return parts.length ? parts.join("  ·  ") : null;
      }
      case "extract_file_sections": {
        const status = d.status as string;
        if (status === "skipped") return "Skipped (no file context)";
        if (status === "passthrough") return `${d.sections ?? 0} section(s) passed through`;
        const kept = d.sections_kept as number | undefined;
        const total = d.sections_total as number | undefined;
        return kept !== undefined ? `${kept} / ${total} section(s) selected` : null;
      }
      case "grade_documents": {
        const status = d.status as string;
        if (status === "skipped") return "Skipped (no documents to grade)";
        const rel = d.relevant as number | undefined;
        const irr = d.irrelevant as number | undefined;
        const retry = d.retry as boolean | undefined;
        const parts: string[] = [];
        if (rel !== undefined) parts.push(`✓ ${rel} relevant`);
        if (irr !== undefined) parts.push(`✗ ${irr} irrelevant`);
        if (retry) parts.push("⚠ retry triggered");
        return parts.length ? parts.join("  ·  ") : null;
      }
      case "merge_context": {
        const kb = d.kb_docs as number | undefined;
        const chars = d.merged_chars as number | undefined;
        const parts: string[] = [];
        if (kb !== undefined) parts.push(`${kb} doc(s) merged`);
        if (chars !== undefined) parts.push(`${chars.toLocaleString()} chars`);
        return parts.length ? parts.join("  ·  ") : null;
      }
      case "generate_answer": {
        const usage = d.usage as { promptTokens?: number; completionTokens?: number } | undefined;
        if (usage?.promptTokens !== undefined)
          return `${usage.promptTokens} prompt + ${usage.completionTokens} completion tokens`;
        return null;
      }
      // derived steps
      case "query_rewrite":
        return d.query ? `${d.query}` : null;
      case "retrieve": {
        const total = d.totalDocCount as number | undefined;
        const graph = d.graphCount as number | undefined;
        if (total === undefined) return null;
        return graph ? `${total} total  (${d.docCount} kb + ${graph} graph)` : `${total} document(s)`;
      }
      case "classification":
        return d.type ? `${d.type} · ${Math.round((d.confidence as number) * 100)}% confidence` : null;
      case "tool_calls":
        return d.tools ? (d.tools as string[]).join(", ") : null;
      case "failed_legs":
        return d.legs ? (d.legs as string[]).join(", ") + " failed" : null;
      default:
        return null;
    }
  }, [step.id, step.status, step.data, rewrittenQuery]);

  return (
    <div className={cn("rounded-md border overflow-hidden transition-all duration-200", statusColor)}>
      <button
        onClick={onToggle}
        className="flex items-center gap-2 w-full px-3 py-1.5 text-left hover:bg-black/5 dark:hover:bg-white/5 transition-colors"
      >
        <StatusIcon status={step.status} />
        <Icon className="h-3.5 w-3.5 text-zinc-500 dark:text-zinc-400 shrink-0" />
        <span className="text-xs font-medium text-zinc-600 dark:text-zinc-300 select-none">
          {displayLabel}
        </span>
        {step.latencyMs != null && (
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
          isExpanded ? "max-h-96 opacity-100" : "max-h-0 opacity-0"
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
            {step.latencyMs != null && (
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
  agentSteps,
  isStreaming = false,
  answerStarted = false,
}) => {
  // Live steps from SSE type-4 events — rendered during and after streaming
  const liveSteps: TimelineStep[] = useMemo(() => {
    if (!agentSteps?.length) return [];
    // Deduplicate: keep the last event per node (active → done transition)
    const byNode = new Map<string, typeof agentSteps[0]>();
    for (const s of agentSteps) {
      const prev = byNode.get(s.node);
      // Prefer done/error over active; never downgrade done→active
      if (!prev || prev.status === "active" || s.status === "done" || s.status === "error") {
        byNode.set(s.node, s);
      }
    }
    return Array.from(byNode.values()).map((s) => {
      const meta = NODE_META[s.node] ?? { label: s.node, icon: "wrench" as const };
      const status: TimelineStep["status"] =
        s.status === "done" || s.status === "skipped" || s.status === "passthrough"
          ? "done"
          : s.status === "error"
          ? "error"
          : "active";
      return {
        id: s.node,
        label: meta.label,
        activeLabel: meta.activeLabel,
        icon: meta.icon,
        status,
        latencyMs: s.latency_ms,
        data: s as Record<string, unknown>,
      };
    });
  }, [agentSteps]);

  // Derived steps from post-stream metadata (2: context event)
  const derivedSteps: TimelineStep[] = useMemo(() => {
    const result: TimelineStep[] = [];

    if (rewrittenQuery) {
      result.push({
        id: "query_rewrite",
        label: "Query Rewrite",
        icon: "search",
        status: "done",
        data: { query: rewrittenQuery },
      });
    }

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

  // Prefer live steps when available; fall back to derived (post-stream)
  const steps = liveSteps.length > 0 ? liveSteps : derivedSteps;

  // Track expanded state per step
  const [expandedSteps, setExpandedSteps] = useState<Set<string>>(() => new Set());

  // Collapse all steps once the answer starts arriving
  useEffect(() => {
    if (answerStarted) {
      setExpandedSteps(new Set());
    }
  }, [answerStarted]);

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

  // Both during and after streaming: show expandable step rows
  return (
    <div className="flex flex-col gap-1.5 my-2">
      {steps.map((step) => (
        <TimelineStepRow
          key={step.id}
          step={step}
          isExpanded={expandedSteps.has(step.id)}
          onToggle={() => toggleStep(step.id)}
          rewrittenQuery={rewrittenQuery}
        />
      ))}
    </div>
  );
};
