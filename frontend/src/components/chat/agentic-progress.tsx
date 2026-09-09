import { useEffect, useState, useRef, useMemo } from "react";
import {
  ChainOfThought,
  ChainOfThoughtContent,
  ChainOfThoughtHeader,
  ChainOfThoughtStep,
} from "@/components/ai-elements/chain-of-thought";
import { Shimmer } from "@/components/ai-elements/shimmer";
import {
  type ToolState,
} from "@/components/ai-elements/tool";
import {
  Task,
  TaskTrigger,
  TaskContent,
  TaskItem,
} from "@/components/ai-elements/task";
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
  Loader2Icon,
  FileCheckIcon,
  XCircleIcon,
  ChevronDownIcon,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

// ── Node → Phase mapping ─────────────────────────────────────────────────────

const NODE_PHASE: Record<string, string> = {
  // Phase 1: Analyzing query
  load_context: "Analyzing query",
  plan: "Analyzing query",
  clarify_interrupt: "Analyzing query",

  // Phase 2: Gathering sources (atomic search tools run inside the tool node)
  // "tool" node is not mapped — tool calls are shown as Tool cards.

  // Phase 3: Thinking & sufficiency
  think: "Thinking",
  sufficiency_check: "Verifying",

  // Phase 4: Generating answer
  generating: "Generating answer",
  generate_answer: "Generating answer",

  // Phase 5: Finalizing answer
  finalize: "Finalizing answer",
  answer_scoring: "Finalizing answer",
  finalize_answer: "Finalizing answer",

  // Phase 6: Calculating confidence
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
  keyword_search: ScanSearchIcon,
  semantic_search: ScanSearchIcon,
  rerank_results: ScanSearchIcon,
  graph_expand: ScanSearchIcon,
  kb_metadata: DatabaseIcon,
  kb_grep: ScanTextIcon,
  kb_outline: BookOpenIcon,
  file_read: FileTextIcon,
  file_extract_table: FileTextIcon,
  code_execute: CodeIcon,
  chart_generate: BarChartIcon,
  summarize: SparklesIcon,
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

export interface SubagentProgressEvent {
  subagent_id: string;
  sub_query: string;
  status: "started" | "tool_call" | "tool_done" | "done";
  tool?: string;
  label?: string;
  hit_count?: number;
  error?: string | boolean | null;
  evidence_count?: number;
  summary?: string;
  subagent_type?: "retrieval" | "office";
  iteration?: number;
}

export interface AgenticProgressProps {
  agentSteps?: AgentStepEvent[];
  isStreaming: boolean;
  toolCalls?: Array<Record<string, unknown>>;
  toolObservations?: Array<Record<string, unknown>>;
  progressMessages?: ProgressMessage[];
  subagentProgress?: SubagentProgressEvent[];
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
  subagentProgress,
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

  // Unified timeline: phases with tool cards inserted after "Thinking"
  // (in the atomic-tools pipeline, tools run after think and before
  // sufficiency_check/verifying).
  const timeline = useMemo(() => {
    type Entry =
      | { kind: "phase"; phase: string; phaseIdx: number }
      | { kind: "tool"; pair: ToolCallPair; toolIdx: number };
    const entries: Entry[] = [];
    const toolAnchorIdx = phases.indexOf("Thinking");
    phases.forEach((phase, i) => {
      entries.push({ kind: "phase", phase, phaseIdx: i });
      if (i === toolAnchorIdx) {
        toolPairs.forEach((pair, ti) => {
          entries.push({ kind: "tool", pair, toolIdx: ti });
        });
      }
    });
    if (toolAnchorIdx === -1) {
      toolPairs.forEach((pair, ti) => {
        entries.push({ kind: "tool", pair, toolIdx: ti });
      });
    }
    return entries;
  }, [phases, toolPairs]);

  // Group subagent progress events by subagent_id, preserving arrival order.
  // Each subagent becomes one Task card with its internal tool calls as items.
  const subagentGroups = useMemo(() => {
    if (!subagentProgress?.length) return [] as Array<{
      id: string;
      subQuery: string;
      isOffice: boolean;
      isDone: boolean;
      succeeded: boolean;
      items: Array<{ text: string; status: "active" | "complete" | "error" }>;
    }>;
    const groups: Record<string, {
      id: string;
      subQuery: string;
      isOffice: boolean;
      isDone: boolean;
      succeeded: boolean;
      items: Array<{ text: string; status: "active" | "complete" | "error" }>;
    }> = {};
    const order: string[] = [];
    for (const ev of subagentProgress) {
      const sid = ev.subagent_id;
      if (!groups[sid]) {
        groups[sid] = {
          id: sid,
          subQuery: ev.sub_query,
          isOffice: ev.subagent_type === "office",
          isDone: false,
          succeeded: false,
          items: [],
        };
        order.push(sid);
      }
      const g = groups[sid];
      if (ev.status === "started") {
        g.subQuery = ev.sub_query;
        g.isOffice = ev.subagent_type === "office";
      } else if (ev.status === "tool_call") {
        g.items.push({
          text: ev.label || ev.tool || "tool call",
          status: "active",
        });
      } else if (ev.status === "tool_done") {
        // Replace the last active item or append
        const lastActive = [...g.items].reverse().findIndex((i) => i.status === "active");
        if (lastActive >= 0) {
          const idx = g.items.length - 1 - lastActive;
          g.items[idx] = {
            text: ev.error
              ? `${ev.label || ev.tool || "tool"}: failed`
              : ev.hit_count !== undefined
                ? `${ev.label || ev.tool || "tool"}: ${ev.hit_count} results`
                : ev.label || ev.tool || "tool",
            status: ev.error ? "error" : "complete",
          };
        }
      } else if (ev.status === "done") {
        g.isDone = true;
        g.succeeded = (ev.evidence_count ?? 0) > 0;
        g.items.push({
          text: g.isOffice
            ? (g.succeeded ? `Created document` : "Failed to create document")
            : `Returned ${ev.evidence_count ?? 0} results`,
          status: "complete",
        });
      }
    }
    return order.map((id) => groups[id]);
  }, [subagentProgress]);

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

  if (phases.length === 0 && toolPairs.length === 0 && subagentGroups.length === 0) return null;

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

            // Summary comes from the backend (to: event) or falls back to error
            const summary = (pair.observation?.summary as string | undefined) ?? undefined;
            const obsError = pair.observation?.error as string | undefined;

            // retrieve_parallel: render sub-agent Task cards BEFORE the
            // synthesis step, so the timeline shows the correct sequence:
            // sub-agents search → results merged.
            if (toolName === "retrieve_parallel") {
              const details = pair.observation?.details as
                | {
                    sub_queries?: string[];
                    count?: number;
                  }
                | undefined;
              return (
                <div key={`tool-${i}`} className="space-y-2">
                  {subagentGroups.map((sg) => {
                    const isInProgress = !sg.isDone;
                    const subagentFailed = sg.isDone && !sg.succeeded;
                    const Icon = sg.isOffice
                      ? (sg.isDone
                          ? (subagentFailed ? XCircleIcon : FileCheckIcon)
                          : Loader2Icon)
                      : (sg.isDone ? CheckCircleIcon : SearchIcon);
                    const titlePrefix = sg.isOffice ? "Office subagent" : "Subagent";
                    const titleAction = sg.isDone
                      ? (sg.isOffice
                          ? (subagentFailed ? "Failed to create" : "Created")
                          : "Searched for")
                      : "Searching for";
                    return (
                      <Task key={sg.id} defaultOpen={isInProgress}>
                        <TaskTrigger
                          title={`${titlePrefix}: ${titleAction} ${sg.subQuery}`}
                        >
                          <div className="flex w-full cursor-pointer items-center gap-2 text-muted-foreground text-xs transition-colors hover:text-foreground">
                            {isInProgress ? (
                              <Loader2Icon className="size-4 animate-spin text-muted-foreground" />
                            ) : (
                              <Icon className={`size-4 ${subagentFailed ? "text-red-600" : "text-emerald-600"}`} />
                            )}
                            <span className="text-xs">
                              {titlePrefix}: {titleAction} <span className="text-foreground font-medium">{sg.subQuery}</span>
                            </span>
                            <ChevronDownIcon className="ml-auto size-4 transition-transform group-data-[state=open]:rotate-180" />
                          </div>
                        </TaskTrigger>
                        <TaskContent>
                          {sg.items.map((item, idx) => (
                            <TaskItem key={idx}>
                              {item.status === "error" ? (
                                <span className="flex items-center gap-1.5 text-[11px] text-red-600">
                                  <XCircleIcon className="size-3" />
                                  {item.text}
                                </span>
                              ) : (
                                <span className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
                                  <span>—</span>
                                  {item.text}
                                </span>
                              )}
                            </TaskItem>
                          ))}
                        </TaskContent>
                      </Task>
                    );
                  })}
                  <ChainOfThoughtStep
                    icon={SparklesIcon}
                    label={
                      isRunning ? (
                        <Shimmer duration={1.5}>Synthesizing subagent results…</Shimmer>
                      ) : (
                        `Synthesized ${details?.count ?? 0} merged hits`
                      )
                    }
                    status={isRunning ? "active" : "complete"}
                  />
                </div>
              );
            }

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
                description={obsError ? <span className="text-red-600">{obsError}</span> : summary}
                status={isRunning ? "active" : "complete"}
              />
            );
          })}
        </ChainOfThoughtContent>
      </ChainOfThought>
    </div>
  );
};
