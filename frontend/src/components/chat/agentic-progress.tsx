import { useEffect, useState, useRef, useMemo } from "react";
import {
  ChainOfThought,
  ChainOfThoughtContent,
  ChainOfThoughtHeader,
  ChainOfThoughtStep,
} from "@/components/ai-elements/chain-of-thought";
import { Shimmer } from "@/components/ai-elements/shimmer";
import {
  Task,
  TaskTrigger,
  TaskContent,
  TaskItem,
} from "@/components/ai-elements/task";
import {
  Collapsible,
  CollapsibleTrigger,
  CollapsibleContent,
} from "@/components/ui/collapsible";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
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

// ── Unified timeline event type ──────────────────────────────────────────────
// Emitted by the backend via `tl:` SSE events. See helpers.py _emit_timeline().

export interface TimelineEvent {
  id: string;
  type: "phase" | "thinking" | "tool_call" | "tool_result" | "subagent_start" | "subagent_step" | "subagent_done";
  ts?: number;
  label?: string;
  status?: string;
  content?: string;
  elapsed?: number;
  tool?: string;
  summary?: string;
  details?: Record<string, unknown> | null;
  error?: string | null;
  hit_count?: number;
  subagent_id?: string;
  subagent_type?: "retrieval" | "office";
  step_type?: "thinking" | "tool";
  succeeded?: boolean;
  evidence_count?: number;
}

export interface AgenticProgressProps {
  timelineEvents?: TimelineEvent[];
  isStreaming: boolean;
}

// ── Icon maps ────────────────────────────────────────────────────────────────

const PHASE_ICONS: Record<string, LucideIcon> = {
  "Analyzing query": SearchIcon,
  "Gathering sources": ZoomInIcon,
  "Synthesizing": FileTextIcon,
  "Thinking": BrainIcon,
  "Reflecting": BrainIcon,
  "Verifying": CheckCircleIcon,
  "Generating answer": FileTextIcon,
  "Finalizing answer": CheckCircleIcon,
  "Confidence calculation": BarChartIcon,
  "Saving memory": DatabaseIcon,
};

const TOOL_ICONS: Record<string, LucideIcon> = {
  keyword_search: ScanSearchIcon,
  semantic_search: ScanSearchIcon,
  rerank_results: ScanSearchIcon,
  graph_expand: ScanSearchIcon,
  title_search: ScanSearchIcon,
  kb_metadata: DatabaseIcon,
  kb_grep: ScanTextIcon,
  kb_outline: BookOpenIcon,
  file_read: FileTextIcon,
  file_extract_table: FileTextIcon,
  code_execute: CodeIcon,
  chart_generate: BarChartIcon,
  summarize: SparklesIcon,
  extract_data: WrenchIcon,
  retrieve_parallel: SearchIcon,
  create_office_document: FileCheckIcon,
};

const SUBAGENT_ICONS: Record<string, { active: LucideIcon; done: LucideIcon; failed: LucideIcon }> = {
  retrieval: { active: SearchIcon, done: CheckCircleIcon, failed: XCircleIcon },
  office: { active: Loader2Icon, done: FileCheckIcon, failed: XCircleIcon },
};

// ── Render entry types ───────────────────────────────────────────────────────
// The flat timeline events are processed into render entries. Subagent events
// are grouped into a single entry with their internal steps.

interface SubagentItem {
  id: string;
  stepType: "thinking" | "tool";
  label: string;
  status: string;
  hitCount?: number;
  error?: boolean;
  summary?: string;
}

type RenderEntry =
  | { kind: "phase"; event: TimelineEvent }
  | { kind: "thinking"; event: TimelineEvent }
  | { kind: "tool"; event: TimelineEvent }
  | { kind: "subagent"; startEvent: TimelineEvent; steps: SubagentItem[]; doneEvent?: TimelineEvent };

// Tools that delegate to a subagent — their tool_call/tool_result events
// are suppressed in the timeline because the subagent's own events
// (subagent_start/subagent_step/subagent_done) handle the display.
const SUBAGENT_TOOL_NAMES = new Set(["create_office_document"]);

function buildRenderEntries(events: TimelineEvent[]): RenderEntry[] {
  const entries: RenderEntry[] = [];
  let currentSubagent: { startEvent: TimelineEvent; steps: SubagentItem[] } | null = null;

  for (const ev of events) {
    switch (ev.type) {
      case "phase":
        entries.push({ kind: "phase", event: ev });
        break;
      case "thinking":
        entries.push({ kind: "thinking", event: ev });
        break;
      case "tool_call":
      case "tool_result":
        // Skip tool events for subagent-delegating tools — the subagent
        // events (subagent_start/subagent_done) handle the display.
        if (ev.tool && SUBAGENT_TOOL_NAMES.has(ev.tool)) {
          break;
        }
        entries.push({ kind: "tool", event: ev });
        break;
      case "subagent_start":
        currentSubagent = { startEvent: ev, steps: [] };
        break;
      case "subagent_step": {
        if (currentSubagent) {
          // Merge by id — if the step already exists (active → complete update),
          // update it in place. Otherwise append.
          const existing = currentSubagent.steps.find((s) => s.id === ev.id);
          if (existing) {
            existing.status = ev.status ?? existing.status;
            existing.hitCount = ev.hit_count ?? existing.hitCount;
            existing.error = ev.error ? true : existing.error;
            existing.summary = ev.summary ?? existing.summary;
          } else {
            currentSubagent.steps.push({
              id: ev.id,
              stepType: ev.step_type ?? "tool",
              label: ev.label ?? ev.tool ?? "step",
              status: ev.status ?? "active",
              hitCount: ev.hit_count,
              error: ev.error ? true : false,
              summary: ev.summary,
            });
          }
        }
        break;
      }
      case "subagent_done":
        if (currentSubagent) {
          entries.push({
            kind: "subagent",
            startEvent: currentSubagent.startEvent,
            steps: currentSubagent.steps,
            doneEvent: ev,
          });
          currentSubagent = null;
        }
        break;
    }
  }
  // If a subagent is still in progress (no done event yet), add it
  if (currentSubagent) {
    entries.push({
      kind: "subagent",
      startEvent: currentSubagent.startEvent,
      steps: currentSubagent.steps,
    });
  }
  return entries;
}

// ── ThinkingStep: "Thought (Ns)" as the clickable trigger ───────────────────
// The label itself expands/collapses the reasoning content — no nested
// "Show reasoning" sub-component.

const MAX_REASONING_HEIGHT = 100; // px — fixed height prevents wobbling from line wrapping
const MAX_SUBAGENT_STEPS = 5;

interface ThinkingStepProps {
  content: string;
  elapsed?: number;
  isActive: boolean;
  isComplete: boolean;
}

const ThinkingStep = ({
  content,
  elapsed,
  isActive,
  isComplete,
}: ThinkingStepProps) => {
  const [isOpen, setIsOpen] = useState(isActive);
  const [expanded, setExpanded] = useState(false);

  // Auto-open when streaming starts, auto-close 1s after streaming ends.
  useEffect(() => {
    if (isActive) {
      setIsOpen(true);
    } else if (isComplete) {
      const timer = setTimeout(() => setIsOpen(false), 1000);
      return () => clearTimeout(timer);
    }
  }, [isActive, isComplete]);

  const label = isActive
    ? <Shimmer duration={1.5}>Thinking…</Shimmer>
    : `Thought${elapsed ? ` (${Math.round(elapsed)}s)` : ""}`;

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <ChainOfThoughtStep
        icon={BrainIcon}
        label={
          <CollapsibleTrigger asChild>
            <button className="flex w-full items-center gap-1 text-left cursor-pointer hover:text-foreground transition-colors">
              <span>{label}</span>
              <ChevronDownIcon
                className="size-3 ml-auto transition-transform text-muted-foreground/60"
                style={{ transform: isOpen ? "rotate(180deg)" : "rotate(0deg)" }}
              />
            </button>
          </CollapsibleTrigger>
        }
        status={isComplete ? "complete" : "active"}
      >
        <CollapsibleContent className="text-xs text-muted-foreground overflow-hidden">
          <div className="mt-2 relative">
            <div
              className="reasoning-content data-[state=closed]:fade-out-0 data-[state=open]:slide-in-from-top-2 data-[state=closed]:animate-out data-[state=open]:animate-in outline-none overflow-y-auto"
              style={{ maxHeight: expanded ? undefined : MAX_REASONING_HEIGHT }}
            >
              {/* Use plain text during streaming for performance; markdown when complete. */}
              {isActive ? (
                <pre className="whitespace-pre-wrap break-words font-sans text-xs leading-relaxed">
                  {content}
                </pre>
              ) : (
                <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
                  {content}
                </ReactMarkdown>
              )}
            </div>
            {/* Gradient fade: transparent at top → solid background at bottom.
                Spans full width, only visible when collapsed. */}
            {!expanded && (
              <div
                className="pointer-events-none absolute bottom-0 left-0 right-0 h-16"
                style={{
                  background: "linear-gradient(to bottom, transparent, hsl(var(--background)))",
                }}
              />
            )}
          </div>
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            className="w-full flex items-center justify-center gap-1 text-muted-foreground/70 hover:text-foreground transition-colors text-[11px] py-1"
          >
            <ChevronDownIcon
              className="size-3 transition-transform"
              style={{ transform: expanded ? "rotate(180deg)" : "rotate(0deg)" }}
            />
            {expanded ? "Show less" : "Show all"}
          </button>
        </CollapsibleContent>
      </ChainOfThoughtStep>
    </Collapsible>
  );
};

// ── Component ────────────────────────────────────────────────────────────────

export const AgenticProgress = ({
  timelineEvents,
  isStreaming,
}: AgenticProgressProps) => {
  const [isOpen, setIsOpen] = useState(true);
  const [showFullTimeline, setShowFullTimeline] = useState(false);
  const dismissRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const entries = useMemo(() => buildRenderEntries(timelineEvents ?? []), [timelineEvents]);

  // Auto-collapse after streaming ends
  useEffect(() => {
    if (isStreaming) {
      if (dismissRef.current) {
        clearTimeout(dismissRef.current);
        dismissRef.current = undefined;
      }
      Promise.resolve().then(() => setIsOpen(true));
    } else if (entries.length > 0) {
      dismissRef.current = setTimeout(() => {
        setIsOpen(false);
      }, 2000);
    }
  }, [isStreaming, entries.length]);

  useEffect(() => {
    return () => {
      if (dismissRef.current) clearTimeout(dismissRef.current);
    };
  }, []);

  if (entries.length === 0) return null;

  // Limit visible timeline to last N entries to prevent unbounded growth.
  const MAX_VISIBLE = 12;
  const hasMore = entries.length > MAX_VISIBLE;
  const visibleEntries = showFullTimeline ? entries : entries.slice(-MAX_VISIBLE);
  const hiddenCount = entries.length - MAX_VISIBLE;

  return (
    <div className="not-prose mb-2">
      <ChainOfThought open={isOpen} onOpenChange={setIsOpen}>
        <ChainOfThoughtHeader>
          {isStreaming ? <Shimmer duration={1.5}>Agent working…</Shimmer> : "Agent timeline"}
        </ChainOfThoughtHeader>
        <ChainOfThoughtContent>
          {hasMore && !showFullTimeline && (
            <button
              type="button"
              onClick={() => setShowFullTimeline(true)}
              className="flex items-center gap-1 text-muted-foreground/60 hover:text-foreground transition-colors text-[11px] mb-1"
            >
              <ChevronDownIcon className="size-3 rotate-180" />
              {hiddenCount} earlier steps hidden
            </button>
          )}
          {visibleEntries.map((entry) => {
            // Use event ID for stable keys to prevent unmount/remount flickering.
            const key = "startEvent" in entry
              ? `subagent-${entry.startEvent.id}`
              : `${entry.kind}-${entry.event.id}`;

            // ── Phase entry ───────────────────────────────────────────────
            if (entry.kind === "phase") {
              const ev = entry.event;
              const isActive = ev.status === "active" && isStreaming;
              const isComplete = ev.status === "complete" || !isStreaming;
              const Icon = PHASE_ICONS[ev.label ?? ""] ?? BrainIcon;
              return (
                <ChainOfThoughtStep
                  key={key}
                  icon={Icon}
                  label={
                    isActive ? (
                      <Shimmer duration={1.5}>{`${ev.label}…`}</Shimmer>
                    ) : (
                      ev.label ?? "Step"
                    )
                  }
                  status={isComplete ? "complete" : isActive ? "active" : "pending"}
                />
              );
            }

            // ── Thinking entry (inline reasoning) ─────────────────────────
            if (entry.kind === "thinking") {
              const ev = entry.event;
              const content = ev.content;
              const isActive = ev.status === "active" && isStreaming;
              const isComplete = ev.status === "complete" || !isStreaming;

              if (content && content.trim().length > 0) {
                // "Thought (Ns)" is the clickable trigger — clicking it
                // expands/collapses the reasoning text directly.
                return (
                  <ThinkingStep
                    key={key}
                    content={content}
                    elapsed={ev.elapsed}
                    isActive={isActive}
                    isComplete={isComplete}
                  />
                );
              }

              // No reasoning content — simple "Thinking" step
              return (
                <ChainOfThoughtStep
                  key={key}
                  icon={BrainIcon}
                  label={
                    isActive ? (
                      <Shimmer duration={1.5}>Thinking…</Shimmer>
                    ) : (
                      `Thought${ev.elapsed ? ` (${Math.round(ev.elapsed)}s)` : ""}`
                    )
                  }
                  status={isComplete ? "complete" : "active"}
                />
              );
            }

            // ── Tool entry ────────────────────────────────────────────────
            if (entry.kind === "tool") {
              const ev = entry.event;
              const isActive = ev.status === "active" && isStreaming;
              const isComplete = ev.status === "complete" || !isStreaming;
              const toolName = ev.tool ?? "tool";
              const label = ev.label ?? toolName;
              const ToolIcon = TOOL_ICONS[toolName] ?? WrenchIcon;
              const summary = ev.summary;
              const obsError = ev.error;

              return (
                <ChainOfThoughtStep
                  key={key}
                  icon={ToolIcon}
                  label={
                    isActive ? (
                      <Shimmer duration={1.5}>{`${label}…`}</Shimmer>
                    ) : (
                      <span>
                        {label}
                        {summary ? <span className="text-muted-foreground"> · {summary}</span> : null}
                      </span>
                    )
                  }
                  description={obsError ? <span className="text-red-600">{obsError}</span> : undefined}
                  status={isComplete ? "complete" : "active"}
                />
              );
            }

            // ── Subagent entry ────────────────────────────────────────────
            if (entry.kind === "subagent") {
              const startEv = entry.startEvent;
              const doneEv = entry.doneEvent;
              const subagentType = startEv.subagent_type ?? "retrieval";
              const subagentLabel = startEv.label ?? "subagent";
              const isInProgress = !doneEv;
              const succeeded = doneEv?.succeeded ?? false;
              const icons = SUBAGENT_ICONS[subagentType] ?? SUBAGENT_ICONS.retrieval;
              // const titlePrefix = subagentType === "office" ? "Subagent" : "Subagent";
              const titlePrefix = "Subagent";
              const titleAction = doneEv
                ? (subagentType === "office"
                    ? (succeeded ? "Created" : "Failed to create")
                    : "Searched for")
                : (subagentType === "office" ? "Creating" : "Searching for");

              return (
                <Task key={key} defaultOpen={isInProgress}>
                  <TaskTrigger title={`${titlePrefix}: ${titleAction} ${subagentLabel}`}>
                    <div className="flex w-full cursor-pointer items-center gap-2 text-muted-foreground text-xs transition-colors hover:text-foreground">
                      {isInProgress ? (
                        <Loader2Icon className="size-4 animate-spin text-muted-foreground" />
                      ) : (
                        (() => {
                          const Icon = succeeded ? icons.done : icons.failed;
                          return <Icon className={`size-4 ${succeeded ? "text-emerald-600" : "text-red-600"}`} />;
                        })()
                      )}
                      <span className="text-xs">
                        {titlePrefix}: {titleAction} <span className="text-muted-foreground font-medium">{subagentLabel}</span>
                      </span>
                      <ChevronDownIcon className="ml-auto size-4 transition-transform group-data-[state=open]:rotate-180" />
                    </div>
                  </TaskTrigger>
                  <TaskContent>
                    {entry.steps.length > MAX_SUBAGENT_STEPS && (
                      <div className="text-muted-foreground/60 text-[11px] mb-1">
                        … {entry.steps.length - MAX_SUBAGENT_STEPS} earlier steps hidden
                      </div>
                    )}
                    {(entry.steps.length > MAX_SUBAGENT_STEPS
                      ? entry.steps.slice(-MAX_SUBAGENT_STEPS)
                      : entry.steps
                    ).map((item, i) => {
                      if (item.error) {
                        return (
                          <TaskItem key={item.id ?? i}>
                            <span className="flex items-center gap-1.5 text-[11px] text-red-600">
                              <XCircleIcon className="size-3" />
                              {item.label}: failed
                            </span>
                          </TaskItem>
                        );
                      }
                      // For office subagents, suppress result suffix — the label
                      // alone is sufficient ("Generating/Updating Office document").
                      // For retrieval subagents, show summary or hit count.
                      const isOffice = subagentType === "office";
                      const resultSuffix = isOffice ? "" : (
                        item.summary
                          ? ` · ${item.summary}`
                          : (item.hitCount !== undefined ? ` · ${item.hitCount} results` : "")
                      );
                      return (
                        <TaskItem key={item.id ?? i}>
                          <span className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
                            <span>—</span>
                            {item.label}{resultSuffix}
                          </span>
                        </TaskItem>
                      );
                    })}
                  </TaskContent>
                </Task>
              );
            }

            return null;
          })}
          {hasMore && showFullTimeline && (
            <button
              type="button"
              onClick={() => setShowFullTimeline(false)}
              className="flex items-center gap-1 text-muted-foreground/60 hover:text-foreground transition-colors text-[11px] mt-1"
            >
              <ChevronDownIcon className="size-3" />
              Collapse to recent steps
            </button>
          )}
        </ChainOfThoughtContent>
      </ChainOfThought>
    </div>
  );
};
