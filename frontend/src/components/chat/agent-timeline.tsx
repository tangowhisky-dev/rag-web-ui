import { FC, useState, useEffect, useRef } from "react";
import { Loader2, Check, X, Search, BookOpen, Share2, Wrench } from "lucide-react";
import { cn } from "@/lib/utils";

// ── Types ────────────────────────────────────────────────────────────────────

export interface TimelineStep {
  id: string;
  label: string;
  activeLabel?: string;
  icon: "search" | "book" | "share" | "wrench" | "check" | "x";
  status: "pending" | "active" | "done" | "error";
  data?: Record<string, unknown>;
  latencyMs?: number;
}

interface AgentStepEvent {
  node: string;
  latency_ms: number;
  status: string;
  [key: string]: unknown;
}

// ── Icon map ─────────────────────────────────────────────────────────────────

const ICON_MAP: Record<string, FC<{ className?: string }>> = {
  search: ({ className }) => <Search className={cn("h-3 w-3 shrink-0 opacity-60", className)} />,
  book: ({ className }) => <BookOpen className={cn("h-3 w-3 shrink-0 opacity-60", className)} />,
  share: ({ className }) => <Share2 className={cn("h-3 w-3 shrink-0 opacity-60", className)} />,
  wrench: ({ className }) => <Wrench className={cn("h-3 w-3 shrink-0 opacity-60", className)} />,
  check: ({ className }) => <Check className={cn("h-3 w-3 shrink-0 opacity-60", className)} />,
  x: ({ className }) => <X className={cn("h-3 w-3 shrink-0 opacity-60", className)} />,
};

// ── Step metadata ────────────────────────────────────────────────────────────

const NODE_META: Record<string, { label: string; activeLabel?: string; icon: TimelineStep["icon"] }> = {
  // Core pipeline nodes (new agentic agent)
  load_context:          { label: "Loading context…",      activeLabel: "Loading context…",      icon: "book" },
  plan:                  { label: "Planning…",             activeLabel: "Planning…",             icon: "wrench" },
  think:                 { label: "Thinking…",             activeLabel: "Thinking…",             icon: "wrench" },
  tool:                  { label: "Using tools…",          activeLabel: "Using tools…",          icon: "wrench" },
  reflect:               { label: "Reflecting…",           activeLabel: "Reflecting…",           icon: "check" },
  reflect_final:         { label: "Final reflection…",     activeLabel: "Final reflection…",     icon: "check" },
  finalize:              { label: "Finalizing answer…",    activeLabel: "Finalizing answer…",    icon: "wrench" },
  answer_scoring:        { label: "Scoring answer…",       activeLabel: "Scoring answer…",       icon: "check" },
  save_memory:           { label: "Saving memory…",        activeLabel: "Saving memory…",        icon: "book" },
  clarify_interrupt:     { label: "Clarifying…",           activeLabel: "Clarifying…",           icon: "share" },
  rewrite_query:         { label: "Rewriting query…",        activeLabel: "Rewriting query…", icon: "search" },
  classify_query:        { label: "Classifying query…",      activeLabel: "Classifying query…", icon: "search" },
  rewrite_subtask_query: { label: "Rewriting subqueries…",   activeLabel: "Rewriting subqueries…", icon: "search" },
  exact_retrieval:       { label: "Full-text retrieval…",    activeLabel: "Full-text retrieval…", icon: "search" },
  sparse_retrieval:      { label: "Sparse keyword retrieval…", activeLabel: "Sparse keyword retrieval…", icon: "search" },
  dense_retrieval:       { label: "Vector retrieval…",       activeLabel: "Vector retrieval…", icon: "search" },
  merge:                 { label: "Merging results…",        activeLabel: "Merging results…", icon: "book" },
  neo4j_expansion:       { label: "Graph expansion…",        activeLabel: "Graph expansion…", icon: "share" },
  reranking:             { label: "Reranking…",              activeLabel: "Reranking…", icon: "check" },
  filter:                { label: "Filtering…",              activeLabel: "Filtering…", icon: "check" },
  sufficiency_check:     { label: "Checking sufficiency…",   activeLabel: "Checking sufficiency…", icon: "check" },
  collect_context:       { label: "Collecting context…",     activeLabel: "Collecting context…", icon: "book" },
  prepare_final_context: { label: "Preparing context…",      activeLabel: "Preparing context…", icon: "book" },
  generating:            { label: "Generating answer…",      activeLabel: "Generating answer…", icon: "wrench" },
  answer_evaluation:     { label: "Evaluating answer…",      activeLabel: "Evaluating answer…", icon: "check" },
  finalize_answer:       { label: "Finalizing answer…",      activeLabel: "Finalizing answer…", icon: "wrench" },
  request_clarification: { label: "Requesting clarification…", activeLabel: "Requesting clarification…", icon: "share" },
  context_router:        { label: "Routing sources…",        activeLabel: "Routing sources…", icon: "share" },
  complex_query:         { label: "Analyzing complexity…",   activeLabel: "Analyzing complexity…", icon: "wrench" },
  decompose_query:       { label: "Decomposing query…",      activeLabel: "Decomposing query…", icon: "search" },
  parallel_retrieval:    { label: "Parallel retrieval…",     activeLabel: "Retrieving…", icon: "book" },
  graph_enrichment:      { label: "Fetching graph context…", activeLabel: "Fetching graph context…", icon: "share" },
  extract_file_sections: { label: "Extracting file sections…", activeLabel: "Extracting…", icon: "wrench" },
  draft_answer:          { label: "Drafting answer…",        activeLabel: "Drafting answer…", icon: "wrench" },
  grade_coverage:        { label: "Checking coverage…",      activeLabel: "Checking coverage…", icon: "check" },
  widened_retrieval:     { label: "Widening search…",        activeLabel: "Widening search…", icon: "book" },
  keyword_search_loop:   { label: "Searching keywords…",     activeLabel: "Searching keywords…", icon: "search" },
  generate_answer:       { label: "Generating answer…",      activeLabel: "Generating answer…", icon: "wrench" },
  kb_retrieval:          { label: "Retrieving context…",     activeLabel: "Retrieving…", icon: "book" },
  grade_documents:       { label: "Grading documents…",      activeLabel: "Grading documents…", icon: "check" },
  merge_context:         { label: "Merging context…",        activeLabel: "Merging context…", icon: "book" },
  synthesize:            { label: "Synthesizing final answer…", activeLabel: "Synthesizing…", icon: "wrench" },
};

// ── Props ────────────────────────────────────────────────────────────────────

export interface AgentTimelineProps {
  agentSteps?: AgentStepEvent[];
  isStreaming: boolean;
}

// ── Transient progress badge ────────────────────────────────────────────────

export const AgentTimeline: FC<AgentTimelineProps> = ({ agentSteps, isStreaming }) => {
  const [steps, setSteps] = useState<
    Array<{ id: string; displayLabel: string; icon: TimelineStep["icon"]; status: TimelineStep["status"]; latencyMs?: number }>
  >([]);
  const [streaming, setStreaming] = useState(true);

  // Track steps from SSE events — only keep latest per node
  useEffect(() => {
    if (!agentSteps?.length) {
      setSteps([]);
      return;
    }
    const byNode = new Map<string, AgentStepEvent>();
    for (const s of agentSteps) {
      const prev = byNode.get(s.node);
      if (!prev || prev.status === "active" || s.status === "done" || s.status === "error") {
        byNode.set(s.node, s);
      }
    }
    const mapped: typeof steps = [];
    for (const [node, s] of byNode) {
      const meta = NODE_META[node] ?? { label: s.node, activeLabel: s.node, icon: "wrench" as const };
      const status: TimelineStep["status"] =
        s.status === "done" || s.status === "skipped" || s.status === "passthrough"
          ? "done"
          : s.status === "error"
          ? "error"
          : "active";
      // Show activeLabel when running, regular label when done/error
      const displayLabel = status === "active" && meta.activeLabel ? meta.activeLabel : meta.label;
      mapped.push({ id: node, displayLabel, icon: meta.icon, status, latencyMs: s.latency_ms });
    }
    setSteps(mapped);
  }, [agentSteps]);

  // Track streaming state and auto-dismiss after streaming ends
  const dismissRef = useRef<ReturnType<typeof setTimeout>>();

  useEffect(() => {
    setStreaming(isStreaming);
    if (isStreaming) {
      if (dismissRef.current) {
        clearTimeout(dismissRef.current);
        dismissRef.current = undefined;
      }
    } else if (steps.length > 0) {
      // Auto-dismiss after 1.5s of streaming completion
      dismissRef.current = setTimeout(() => setSteps([]), 1500);
    }
  }, [isStreaming, steps.length]);

  useEffect(() => {
    return () => {
      if (dismissRef.current) clearTimeout(dismissRef.current);
    };
  }, []);

  if (steps.length === 0) return null;

  return (
    <div className="mb-3 space-y-1.5">
      {steps.map((step) => {
        const IconComp = ICON_MAP[step.icon] ?? Wrench;
        return (
          <div
            key={step.id}
            className={cn(
              "relative flex items-center gap-2 px-3 py-1.5 rounded-md text-xs select-none overflow-hidden",
              "transition-all duration-500",
              streaming
                ? step.status === "active"
                  ? "bg-blue-50/80 dark:bg-blue-900/15 border border-blue-200/80 dark:border-blue-800/60 translate-y-0"
                  : "bg-emerald-50/80 dark:bg-emerald-900/15 border border-emerald-200/80 dark:border-emerald-800/60 translate-y-0"
                : "opacity-0 -translate-y-2"
            )}
          >
            {/* Status indicator */}
            {step.status === "active" && (
              <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-500 shrink-0" />
            )}
            {step.status === "done" && (
              <Check className="h-3.5 w-3.5 text-emerald-500 shrink-0" />
            )}
            {step.status === "error" && (
              <X className="h-3.5 w-3.5 text-red-500 shrink-0" />
            )}

            {/* Step icon */}
            <IconComp className="h-3 w-3 shrink-0 opacity-60" />

            {/* Label */}
            <span className={cn(
              "font-medium leading-tight",
              step.status === "active"
                ? "text-blue-700 dark:text-blue-300"
                : step.status === "done"
                ? "text-emerald-700 dark:text-emerald-300"
                : "text-red-700 dark:text-red-300"
            )}>
              {step.displayLabel}
            </span>

            {/* Latency */}
            {step.latencyMs != null && (
              <span className="text-[10px] text-zinc-400 ml-auto shrink-0 tabular-nums">
                {step.latencyMs.toFixed(0)}ms
              </span>
            )}

            {/* Shimmer overlay for active */}
            {step.status === "active" && (
              <div className="absolute inset-0 status-shimmer rounded-md pointer-events-none" />
            )}
          </div>
        );
      })}
    </div>
  );
};
