import { useEffect, useState, useRef, useMemo } from "react";
import {
  ChainOfThought,
  ChainOfThoughtContent,
  ChainOfThoughtHeader,
  ChainOfThoughtStep,
} from "@/components/ai-elements/chain-of-thought";
import { Shimmer } from "@/components/ai-elements/shimmer";
import {
  Tool,
  ToolContent,
  ToolHeader,
  ToolInput,
  ToolOutput,
  type ToolState,
} from "@/components/ai-elements/tool";
import {
  SearchIcon,
  BrainIcon,
  FileTextIcon,
  CheckCircleIcon,
  CodeIcon,
  BarChartIcon,
  SparklesIcon,
  WrenchIcon,
  BookOpenIcon,
  ScanTextIcon,
  ScanSearchIcon,
  ZoomInIcon,
  DatabaseIcon,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

// ── Node → Phase mapping ─────────────────────────────────────────────────────

const NODE_PHASE: Record<string, string> = {
  // Phase 1: Analyzing query
  rewrite_query: "Analyzing query",
  classify_query: "Analyzing query",
  load_context: "Analyzing query",
  plan: "Analyzing query",
  clarify_interrupt: "Analyzing query",

  // Phase 2: Gathering sources
  exact_retrieval: "Gathering sources",
  sparse_retrieval: "Gathering sources",
  dense_retrieval: "Gathering sources",
  neo4j_expansion: "Gathering sources",

  // Phase 3: Removing clutter & synthesizing
  merge: "Synthesizing",
  reranking: "Synthesizing",
  filter: "Synthesizing",
  collect_context: "Synthesizing",
  prepare_final_context: "Synthesizing",

  // Phase 4: Thinking & tools
  think: "Thinking",
  // "tool" node is not mapped — tool calls are shown as Tool cards.

  // Phase 5: Reflecting
  reflect: "Reflecting",
  reflect_final: "Verifying",

  // Phase 6: Generating answer
  generating: "Generating answer",
  generate_answer: "Generating answer",

  // Phase 7: Finalizing answer
  finalize: "Finalizing answer",
  answer_scoring: "Finalizing answer",
  finalize_answer: "Finalizing answer",

  // Phase 8: Calculating confidence
  answer_evaluation: "Calculating confidence",
};

// Map phase labels to icons
const PHASE_ICONS: Record<string, LucideIcon> = {
  "Analyzing query": SearchIcon,
  "Gathering sources": ZoomInIcon,
  Synthesizing: FileTextIcon,
  Thinking: BrainIcon,
  Reflecting: BrainIcon,
  Verifying: CheckCircleIcon,
  "Generating answer": FileTextIcon,
  "Finalizing answer": CheckCircleIcon,
  "Calculating confidence": BarChartIcon,
};

// Map tool names to icons
const TOOL_ICONS: Record<string, LucideIcon> = {
  rag_retrieve: ScanSearchIcon,
  kb_metadata: DatabaseIcon,
  kb_grep: ScanTextIcon,
  kb_outline: BookOpenIcon,
  kb_read: FileTextIcon,
  file_read: FileTextIcon,
  file_summarize: FileTextIcon,
  file_extract_table: FileTextIcon,
  code_execute: CodeIcon,
  chart_generate: BarChartIcon,
  summarize_answer: SparklesIcon,
  extract_data: WrenchIcon,
};

// ── Types ────────────────────────────────────────────────────────────────────

export interface AgentStepEvent {
  node: string;
  latency_ms: number;
  status: string;
  [key: string]: unknown;
}

export interface ProgressMessage {
  phase: string;
  message: string;
  details?: Record<string, unknown>;
  rewritten_query?: string;
  original_query?: string;
}

export interface AgenticProgressProps {
  agentSteps?: AgentStepEvent[];
  isStreaming: boolean;
  toolCalls?: Array<Record<string, unknown>>;
  toolObservations?: Array<Record<string, unknown>>;
  progressMessages?: ProgressMessage[];
}

// ── Helpers ──────────────────────────────────────────────────────────────────

interface ToolCallPair {
  call: Record<string, unknown>;
  observation?: Record<string, unknown>;
}

function pairToolCallsAndObservations(
  toolCalls: Array<Record<string, unknown>>,
  toolObservations: Array<Record<string, unknown>>
): ToolCallPair[] {
  return toolCalls.map((call, i) => ({
    call,
    observation: toolObservations[i],
  }));
}

function getToolState(pair: ToolCallPair): ToolState {
  if (pair.observation) {
    if (pair.observation.error) return "output-error";
    return "output-available";
  }
  return "input-available";
}

function getToolLabel(call: Record<string, unknown>): string {
  return (call.label as string) || (call.tool as string) || "Tool";
}

function getToolName(call: Record<string, unknown>): string {
  return (call.tool as string) || "tool";
}

// ── Component ────────────────────────────────────────────────────────────────

export const AgenticProgress = ({
  agentSteps,
  isStreaming,
  toolCalls,
  toolObservations,
}: AgenticProgressProps) => {
  const [isOpen, setIsOpen] = useState(true);
  const dismissRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  // Build deduplicated phase list — every phase appears at most once.
  // The agent loop may revisit retrieval/synthesis/verification nodes
  // across iterations; showing each repeat would produce a confusing
  // timeline (e.g. "Gathering sources" x6).
  const phases = useMemo(() => {
    if (!agentSteps?.length) return [] as string[];
    const seen = new Set<string>();
    const unique: string[] = [];
    for (const step of agentSteps) {
      const phase = NODE_PHASE[step.node];
      if (!phase) continue;
      if (!seen.has(phase)) {
        seen.add(phase);
        unique.push(phase);
      }
    }
    return unique;
  }, [agentSteps]);

  // Pair tool calls with their observations
  const toolPairs = useMemo(() => {
    if (!toolCalls?.length) return [] as ToolCallPair[];
    return pairToolCallsAndObservations(
      toolCalls,
      toolObservations ?? []
    );
  }, [toolCalls, toolObservations]);

  // Unified timeline: phases with tool cards inserted right after
  // "Gathering sources" (where retrieval tools belong chronologically).
  // If there's no "Gathering sources" phase, tools fall through to the end.
  const timeline = useMemo(() => {
    type Entry =
      | { kind: "phase"; phase: string; phaseIdx: number }
      | { kind: "tool"; pair: ToolCallPair; toolIdx: number };
    const entries: Entry[] = [];
    const gatheringIdx = phases.indexOf("Gathering sources");
    phases.forEach((phase, i) => {
      entries.push({ kind: "phase", phase, phaseIdx: i });
      if (i === gatheringIdx) {
        toolPairs.forEach((pair, ti) => {
          entries.push({ kind: "tool", pair, toolIdx: ti });
        });
      }
    });
    if (gatheringIdx === -1) {
      toolPairs.forEach((pair, ti) => {
        entries.push({ kind: "tool", pair, toolIdx: ti });
      });
    }
    return entries;
  }, [phases, toolPairs]);

  // Auto-collapse after streaming ends
  useEffect(() => {
    if (isStreaming) {
      if (dismissRef.current) {
        clearTimeout(dismissRef.current);
        dismissRef.current = undefined;
      }
      Promise.resolve().then(() => setIsOpen(true));
    } else if (phases.length > 0 || toolPairs.length > 0) {
      dismissRef.current = setTimeout(() => {
        setIsOpen(false);
      }, 2000);
    }
  }, [isStreaming, phases.length, toolPairs.length]);

  useEffect(() => {
    return () => {
      if (dismissRef.current) clearTimeout(dismissRef.current);
    };
  }, []);

  if (phases.length === 0 && toolPairs.length === 0) return null;

  // Determine which phase is currently active (last phase while streaming)
  const currentPhaseIdx = isStreaming ? phases.length - 1 : -1;

  return (
    <div className="not-prose mb-2">
      <ChainOfThought open={isOpen} onOpenChange={setIsOpen}>
        <ChainOfThoughtHeader>
          {isStreaming ? <Shimmer duration={1.5}>Agent working…</Shimmer> : "Agent timeline"}
        </ChainOfThoughtHeader>
        <ChainOfThoughtContent>
          {timeline.map((entry) => {
            if (entry.kind === "phase") {
              const phase = entry.phase;
              const i = entry.phaseIdx;
              const isActive = i === currentPhaseIdx;
              const isComplete = i < currentPhaseIdx || !isStreaming;
              const Icon = PHASE_ICONS[phase] ?? BrainIcon;
              return (
                <ChainOfThoughtStep
                  key={`phase-${phase}-${i}`}
                  icon={Icon}
                  label={
                    isActive ? (
                      <Shimmer duration={1.5}>{`${phase}…`}</Shimmer>
                    ) : (
                      phase
                    )
                  }
                  status={isComplete ? "complete" : isActive ? "active" : "pending"}
                />
              );
            }

            // Tool card entry
            const pair = entry.pair;
            const i = entry.toolIdx;
            const toolName = getToolName(pair.call);
            const label = getToolLabel(pair.call);
            const state = getToolState(pair);
            const isRunning = state === "input-available" && isStreaming;
            const ToolIcon = TOOL_ICONS[toolName] ?? WrenchIcon;

            // Extract a short summary from the observation result
            const obsResult = pair.observation?.result as Record<string, unknown> | undefined;
            const obsError = pair.observation?.error as string | undefined;
            const resultSummary = obsResult
              ? typeof obsResult === "object" && "matches" in obsResult
                ? `${(obsResult.matches as unknown[]).length} matches`
                : typeof obsResult === "object" && "docs" in obsResult
                  ? `${(obsResult.docs as unknown[]).length} docs retrieved`
                  : typeof obsResult === "object" && "content" in obsResult
                    ? `Read ${(obsResult as Record<string, unknown>).total_tokens ?? "?"} tokens`
                    : typeof obsResult === "object" && "headings" in obsResult
                      ? `${(obsResult.headings as unknown[]).length} headings`
                      : undefined
              : undefined;

            return (
              <ChainOfThoughtStep
                key={`tool-${i}`}
                icon={ToolIcon}
                label={
                  isRunning ? (
                    <Shimmer duration={1.5}>{`${label}…`}</Shimmer>
                  ) : (
                    label
                  )
                }
                description={resultSummary}
                status={isRunning ? "active" : "complete"}
              >
                <Tool defaultOpen={false}>
                  <ToolHeader
                    title={label}
                    state={state}
                  />
                  <ToolContent>
                    <ToolInput input={pair.call.arguments ?? {}} />
                    <ToolOutput
                      output={obsResult ? JSON.stringify(obsResult, null, 2) : undefined}
                      errorText={obsError}
                    />
                  </ToolContent>
                </Tool>
              </ChainOfThoughtStep>
            );
          })}
        </ChainOfThoughtContent>
      </ChainOfThought>
    </div>
  );
};
