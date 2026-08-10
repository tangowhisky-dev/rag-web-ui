import { useEffect, useState, useRef } from "react";

// ── Node → Phase mapping ─────────────────────────────────────────────────────

const NODE_PHASE: Record<string, string> = {
  // Phase 1: Analyzing query
  rewrite_query: "Analyzing query …",
  classify_query: "Analyzing query …",
  load_context: "Analyzing query …",
  plan: "Analyzing query …",
  clarify_interrupt: "Analyzing query …",

  // Phase 2: Gathering sources
  exact_retrieval: "Gathering sources …",
  sparse_retrieval: "Gathering sources …",
  dense_retrieval: "Gathering sources …",
  neo4j_expansion: "Gathering sources …",

  // Phase 3: Removing clutter & synthesizing
  merge: "Removing clutter & synthesizing …",
  reranking: "Removing clutter & synthesizing …",
  filter: "Removing clutter & synthesizing …",
  collect_context: "Removing clutter & synthesizing …",
  prepare_final_context: "Removing clutter & synthesizing …",

  // Phase 4: Thinking & tools
  think: "Thinking …",
  // Note: the "tool" node is intentionally NOT mapped here. Tool call
  // labels (e.g. "Retrieving from knowledge base") are shown directly
  // from the toolCalls prop when tools are running — see the render
  // body below. This avoids a generic "Using tools …" flash before the
  // specific label arrives.

  // Phase 5: Reflecting
  reflect: "Reflecting …",
  reflect_final: "Verifying …",

  // Phase 6: Generating answer
  generating: "Generating answer …",
  generate_answer: "Generating answer …",

  // Phase 7: Finalizing answer
  finalize: "Finalizing answer …",
  answer_scoring: "Finalizing answer …",
  finalize_answer: "Finalizing answer …",

  // Phase 8: Calculating confidence
  answer_evaluation: "Calculating confidence …",
};

// ── Types ────────────────────────────────────────────────────────────────────

export interface AgentStepEvent {
  node: string;
  latency_ms: number;
  status: string;
  [key: string]: unknown;
}

export interface AgenticProgressProps {
  agentSteps?: AgentStepEvent[];
  isStreaming: boolean;
  toolCalls?: Array<Record<string, unknown>>;
  toolObservations?: Array<Record<string, unknown>>;
}

// ── Component ────────────────────────────────────────────────────────────────

export const AgenticProgress = ({ agentSteps, isStreaming, toolCalls, toolObservations }: AgenticProgressProps) => {
  // Track unique phases in order of appearance
  const [phases, setPhases] = useState<string[]>([]);
  const dismissRef = useRef<ReturnType<typeof setTimeout>>();

  useEffect(() => {
    if (!agentSteps?.length) {
      setPhases([]);
      return;
    }

    // Deduplicate phases while preserving order.
    // Phases 1 (Analyzing query), 4 (Generating answer), and 5 (Calculating confidence) appear once.
    // Phases 2 (Gathering sources) and 3 (Removing clutter) appear per-task
    // in complex queries, so we allow duplicates for those.
    const dedupPhases = new Set(["Analyzing query …", "Thinking …", "Reflecting …", "Finalizing answer …", "Generating answer …", "Calculating confidence …"]);
    const seen = new Set<string>();
    const unique: string[] = [];
    for (const step of agentSteps) {
      const phase = NODE_PHASE[step.node];
      if (!phase) continue;
      if (dedupPhases.has(phase)) {
        if (!seen.has(phase)) {
          seen.add(phase);
          unique.push(phase);
        }
      } else {
        // Allow duplicates for gathering/sources phases
        unique.push(phase);
      }
    }
    setPhases(unique);
  }, [agentSteps]);

  // Auto-dismiss after streaming ends
  useEffect(() => {
    if (isStreaming) {
      if (dismissRef.current) {
        clearTimeout(dismissRef.current);
        dismissRef.current = undefined;
      }
    } else if (phases.length > 0) {
      dismissRef.current = setTimeout(() => setPhases([]), 1500);
    }
  }, [isStreaming, phases.length]);

  useEffect(() => {
    return () => {
      if (dismissRef.current) clearTimeout(dismissRef.current);
    };
  }, []);

  // Determine the tool label if a tool call is still in flight
  // (hasn't received an observation yet).  This is shown in place of
  // the generic phase text and takes priority over everything else
  // while streaming.
  let toolLabel: string | null = null;
  if (isStreaming && toolCalls && toolCalls.length > 0) {
    const obsCount = toolObservations?.length ?? 0;
    if (toolCalls.length > obsCount) {
      const latest = toolCalls[toolCalls.length - 1];
      const label = latest?.label as string | undefined;
      if (label) {
        toolLabel = `${label} …`;
      }
    }
  }

  // If we have a tool label, show it — even if phases is empty (the
  // tool node is not in NODE_PHASE, so phases may be empty during
  // tool execution).
  if (toolLabel) {
    return (
      <div>
        <span className="text-[12px] text-zinc-500 dark:text-zinc-400 leading-tight">
          {toolLabel}
        </span>
      </div>
    );
  }

  if (phases.length === 0) return null;

  const currentPhase = phases[phases.length - 1];

  return (
    <div>
      <span className="text-[12px] text-zinc-500 dark:text-zinc-400 leading-tight">
        {currentPhase}
      </span>
    </div>
  );
};
