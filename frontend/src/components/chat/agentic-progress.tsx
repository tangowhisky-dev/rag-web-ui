import { useEffect, useState, useRef } from "react";

// ── Node → Phase mapping ─────────────────────────────────────────────────────

const NODE_PHASE: Record<string, string> = {
  // Phase 1: Retrieving memories & re-writing query
  load_historical_memory: "Retrieving memories & re-writing query …",
  summarize_history: "Retrieving memories & re-writing query …",
  rewrite_query: "Retrieving memories & re-writing query …",
  classify_query: "Retrieving memories & re-writing query …",

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

  // Phase 4: Generating answer
  generating: "Generating answer …",
  generate_answer: "Generating answer …",

  // Phase 5: Calculating confidence
  answer_evaluation: "Calculating confidence …",
  finalize_answer: "Calculating confidence …",
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
}

// ── Component ────────────────────────────────────────────────────────────────

export const AgenticProgress = ({ agentSteps, isStreaming }: AgenticProgressProps) => {
  // Track unique phases in order of appearance
  const [phases, setPhases] = useState<string[]>([]);
  const dismissRef = useRef<ReturnType<typeof setTimeout>>();

  useEffect(() => {
    if (!agentSteps?.length) {
      setPhases([]);
      return;
    }

    // Deduplicate phases while preserving order.
    // Phases 2 (Gathering sources) and 3 (Removing clutter) appear per-task
    // in complex queries, so we allow duplicates for those.
    // Phases 1, 4, 5 appear once and are deduplicated.
    const dedupPhases = new Set(["Retrieving memories & re-writing query …", "Generating answer …", "Calculating confidence …"]);
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

  if (phases.length === 0) return null;

  return (
    <div className="space-y-0">
      {phases.map((phase, i) => {
        // Last phase is current (active), previous ones are fading
        const isActive = i === phases.length - 1;
        return (
          <div
            key={`${phase}-${i}`}
            className={`transition-all duration-500 ${
              isActive
                ? "opacity-100"
                : "opacity-0 -translate-y-1"
            }`}
          >
            <span className="text-[12px] text-zinc-500 dark:text-zinc-400 leading-tight">
              {phase}
            </span>
          </div>
        );
      })}
    </div>
  );
};
