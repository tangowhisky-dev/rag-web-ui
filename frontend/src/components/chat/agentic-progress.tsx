import { useEffect, useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";

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

// Major phases that should override a sticky tool label.
// When one of these becomes the current phase, the tool label is cleared.
const MAJOR_PHASES = new Set([
  "Analyzing query …",
  "Thinking …",
  "Finalizing answer …",
  "Generating answer …",
  "Calculating confidence …",
]);

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
  const [phases, setPhases] = useState<string[]>([]);
  const dismissRef = useRef<ReturnType<typeof setTimeout>>();
  // Sticky tool label: persists after the tool observation arrives,
  // until a major phase (Thinking, Finalizing, etc.) takes over.
  const stickyToolLabelRef = useRef<string | null>(null);

  useEffect(() => {
    if (!agentSteps?.length) {
      setPhases([]);
      stickyToolLabelRef.current = null;
      setHistory([]);
      return;
    }

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
      dismissRef.current = setTimeout(() => {
        setPhases([]);
        setHistory([]);
      }, 1500);
    }
  }, [isStreaming, phases.length]);

  useEffect(() => {
    return () => {
      if (dismissRef.current) clearTimeout(dismissRef.current);
    };
  }, []);

  // ── Compute display text ──────────────────────────────────────────────────

  // 1. If a tool call is in flight, show its label (fresh or updated).
  let activeToolLabel: string | null = null;
  if (isStreaming && toolCalls && toolCalls.length > 0) {
    const obsCount = toolObservations?.length ?? 0;
    if (toolCalls.length > obsCount) {
      const latest = toolCalls[toolCalls.length - 1];
      const label = latest?.label as string | undefined;
      if (label) {
        activeToolLabel = `${label} …`;
      }
    }
  }

  // 2. Update the sticky label.
  //    - When a tool is in flight, set the sticky label.
  //    - When no tool is in flight, keep the sticky label UNLESS the
  //      current phase is a major phase (Thinking, Finalizing, etc.).
  //      Intermediate phases (Gathering sources, Reflecting, Removing
  //      clutter) do NOT clear the sticky label — this prevents the
  //      rapid flash of intermediate text after a tool completes.
  const currentPhase = phases.length > 0 ? phases[phases.length - 1] : null;

  if (activeToolLabel) {
    stickyToolLabelRef.current = activeToolLabel;
  } else if (currentPhase && MAJOR_PHASES.has(currentPhase)) {
    stickyToolLabelRef.current = null;
  }

  // 3. Decide what to show: sticky tool label > current phase.
  const displayText = stickyToolLabelRef.current ?? currentPhase;

  // 4. Track history of display texts (last 3) for the 3-line display.
  //    Adjusting state during render — React's recommended pattern for
  //    derived state. Avoids an extra render cycle vs useEffect.
  const [history, setHistory] = useState<string[]>([]);
  if (displayText && history[history.length - 1] !== displayText) {
    setHistory(prev => [...prev, displayText].slice(-3));
  }

  if (history.length === 0 && !displayText) return null;

  // Show up to 3 lines: oldest two in thin grey, current in bold grey.
  const lines = displayText
    ? [...history.slice(0, -1), displayText].slice(-3)
    : history;

  if (lines.length === 0) return null;

  return (
    <div className="flex flex-col justify-end gap-0 overflow-hidden" style={{ minHeight: `${lines.length * 16}px` }}>
      <AnimatePresence initial={false} mode="popLayout">
        {lines.map((line, i) => {
          const isCurrent = i === lines.length - 1;
          return (
            <motion.span
              key={line}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className={
                isCurrent
                  ? "text-[12px] font-semibold text-zinc-600 dark:text-zinc-300 leading-tight"
                  : "text-[11px] font-normal text-zinc-400 dark:text-zinc-500 leading-tight"
              }
            >
              {line}
            </motion.span>
          );
        })}
      </AnimatePresence>
    </div>
  );
};
