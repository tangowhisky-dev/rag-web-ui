import React, {
  FC,
  useMemo,
  useEffect,
  useState,
  useRef,
  useCallback,
  ClassAttributes,
} from "react";
import { AnchorHTMLAttributes } from "react";
import { ChevronDown, ChevronRight, Brain, Search, BookOpen, Share2, Copy, Trash2, FileText, FileImage, FileType } from "lucide-react";
import { AgentTimeline } from "./agent-timeline";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Skeleton } from "@/components/ui/skeleton";
import { Divider } from "@/components/ui/divider";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import dynamic from "next/dynamic";

const MermaidDiagramDynamic = dynamic(
  () => import("./mermaid-diagram"),
  { ssr: false }
);
import { api } from "@/lib/api";
import { cleanChunkText } from "@/lib/utils";
import { FileIcon } from "react-file-icon";

// Debounce hook to prevent rapid state updates during streaming
const useDebouncedValue = <T,>(value: T, delay: number): T => {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
};

const ThinkBlock: FC<{ content: string; isComplete: boolean }> = ({
  content,
  isComplete,
}) => {
  const [isExpanded, setIsExpanded] = useState(!isComplete);
  const [elapsedMs, setElapsedMs] = useState(0);
  const startTimeRef = useRef<number>(Date.now());
  const finalMsRef = useRef<number | null>(null);
  const contentRef = useRef<HTMLDivElement>(null);

  // Single effect: run interval while thinking, freeze + collapse when done.
  // We never call setElapsedMs synchronously inside the completion branch to
  // avoid triggering the "Maximum update depth exceeded" cascade.
  useEffect(() => {
    if (isComplete) {
      // Record final elapsed time into a ref (no setState = no re-render loop)
      if (finalMsRef.current === null) {
        finalMsRef.current = Date.now() - startTimeRef.current;
      }
      const timer = setTimeout(() => setIsExpanded(false), 1500);
      return () => clearTimeout(timer);
    }
    // Tick every 100 ms while the model is still thinking
    const interval = setInterval(() => {
      setElapsedMs(Date.now() - startTimeRef.current);
    }, 100);
    return () => clearInterval(interval);
  }, [isComplete]);

  // Auto-scroll to bottom as content streams in
  useEffect(() => {
    if (!isComplete && isExpanded && contentRef.current) {
      contentRef.current.scrollTop = contentRef.current.scrollHeight;
    }
  }, [content, isComplete, isExpanded]);

  const displayMs = finalMsRef.current ?? elapsedMs;
  const seconds = displayMs / 1000;
  const label = isComplete
    ? seconds < 1
      ? "Thought for less than a second"
      : `Thought for ${seconds.toFixed(1)} seconds`
    : `Thinking... (${seconds.toFixed(1)}s)`;

  return (
    <div className="my-2 rounded-md border border-gray-100 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/40 w-full">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-1.5 w-full px-3 py-1.5 text-left rounded-t-md hover:bg-gray-100 dark:hover:bg-gray-700/40 transition-colors group"
      >
        {isExpanded ? (
          <ChevronDown className="h-3 w-3 text-gray-400 shrink-0" />
        ) : (
          <ChevronRight className="h-3 w-3 text-gray-400 shrink-0" />
        )}
        <Brain className={`h-3 w-3 shrink-0 ${isComplete ? "text-gray-400" : "text-blue-400 animate-pulse"}`} />
        <span className="text-xs text-gray-400 font-medium select-none">
          {label}
        </span>
      </button>
      {isExpanded && (
        <div
          ref={contentRef}
          className="px-3 pb-2 pt-1 max-h-48 overflow-y-auto overflow-x-hidden border-t border-gray-100 dark:border-gray-700"
        >
          <pre className="text-[11px] leading-[1.45] text-gray-400 dark:text-gray-500 whitespace-pre-wrap break-words font-sans m-0">
            {content}
          </pre>
        </div>
      )}
    </div>
  );
};

interface ContextDoc {
  page_content: string;
  metadata: Record<string, any>;
}

const RewrittenQueryBlock: FC<{ query: string }> = ({ query }) => {
  const [isExpanded, setIsExpanded] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => setIsExpanded(false), 1500);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="my-2 rounded-md border border-gray-100 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/40 w-full">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-1.5 w-full px-3 py-1.5 text-left rounded-t-md hover:bg-gray-100 dark:hover:bg-gray-700/40 transition-colors"
      >
        {isExpanded ? (
          <ChevronDown className="h-3 w-3 text-gray-400 shrink-0" />
        ) : (
          <ChevronRight className="h-3 w-3 text-gray-400 shrink-0" />
        )}
        <Search className="h-3 w-3 shrink-0 text-gray-400" />
        <span className="text-xs text-gray-400 font-medium select-none">
          Rewritten Query
        </span>
      </button>
      {isExpanded && (
        <div className="px-3 pb-2 pt-1 border-t border-gray-100 dark:border-gray-700">
          <p className="text-[11px] leading-[1.45] text-gray-400 dark:text-gray-500 whitespace-pre-wrap break-words font-sans m-0">
            {query}
          </p>
        </div>
      )}
    </div>
  );
};

const RetrievedContextBlock: FC<{ docs: ContextDoc[] }> = ({ docs }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="my-2 rounded-md border border-gray-100 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/40 w-full">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-1.5 w-full px-3 py-1.5 text-left rounded-t-md hover:bg-gray-100 dark:hover:bg-gray-700/40 transition-colors"
      >
        {isExpanded ? (
          <ChevronDown className="h-3 w-3 text-gray-400 shrink-0" />
        ) : (
          <ChevronRight className="h-3 w-3 text-gray-400 shrink-0" />
        )}
        <BookOpen className="h-3 w-3 shrink-0 text-gray-400" />
        <span className="text-xs text-gray-400 font-medium select-none">
          Retrieved {docs.length} context{docs.length !== 1 ? "s" : ""}
        </span>
      </button>
      {isExpanded && (
        <div className="px-3 pb-2 pt-1 max-h-64 overflow-y-auto border-t border-gray-100 dark:border-gray-700 space-y-2">
          {docs.map((doc, i) => (
            <div key={i} className="text-[11px] leading-[1.45] text-gray-400 dark:text-gray-500 font-sans">
              <span className="font-semibold text-gray-500 dark:text-gray-400">[{i + 1}] </span>
              {doc.page_content.length > 300
                ? doc.page_content.slice(0, 300) + "..."
                : doc.page_content}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

interface Citation {
  id: number;
  text: string;
  metadata: Record<string, any>;
  score?: number;
  dense_rank?: number;
  qdrant_sparse_rank?: number;
  exact_rank?: number;
  retrieval_leg?: string;
}

interface KnowledgeBaseInfo {
  name: string;
}

interface DocumentInfo {
  file_name: string;
  knowledge_base: KnowledgeBaseInfo;
}

const RetrievedGraphBlock: FC<{ docs: ContextDoc[] }> = ({ docs }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="my-2 rounded-md border border-purple-100 dark:border-purple-900/40 bg-purple-50/50 dark:bg-purple-900/10 w-full">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-1.5 w-full px-3 py-1.5 text-left rounded-t-md hover:bg-purple-100/60 dark:hover:bg-purple-900/20 transition-colors"
      >
        {isExpanded ? (
          <ChevronDown className="h-3 w-3 text-purple-400 shrink-0" />
        ) : (
          <ChevronRight className="h-3 w-3 text-purple-400 shrink-0" />
        )}
        <Share2 className="h-3 w-3 shrink-0 text-purple-400" />
        <span className="text-xs text-purple-500 dark:text-purple-400 font-medium select-none">
          Retrieved Graph Knowledge
        </span>
        <span className="ml-auto text-[10px] text-purple-400 dark:text-purple-500 font-normal select-none">
          {docs.length} node{docs.length !== 1 ? "s" : ""}
        </span>
      </button>
      {isExpanded && (
        <div className="px-3 pb-2 pt-1 max-h-64 overflow-y-auto border-t border-purple-100 dark:border-purple-900/40 space-y-2">
          {docs.map((doc, i) => (
            <div key={i} className="text-[11px] leading-[1.45] text-purple-700 dark:text-purple-300 font-sans">
              <span className="font-semibold text-purple-500 dark:text-purple-400">[G{i + 1}] </span>
              <span className="whitespace-pre-wrap">
                {cleanChunkText(doc.page_content).length > 400
                  ? cleanChunkText(doc.page_content).slice(0, 400) + "…"
                  : cleanChunkText(doc.page_content)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

interface CitationInfo {
  knowledge_base: KnowledgeBaseInfo;
  document: DocumentInfo;
}

// ── Query Classification badge ─────────────────────────────────────────────

interface QueryClassification {
  type: string;
  confidence: number;
  latency_ms: number;
  fallback: boolean;
}

const QUERY_TYPE_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  FACTUAL:        { bg: "bg-blue-50 dark:bg-blue-900/20",   text: "text-blue-700 dark:text-blue-300",   border: "border-blue-200 dark:border-blue-800"   },
  ENTITY_CENTRIC: { bg: "bg-purple-50 dark:bg-purple-900/20", text: "text-purple-700 dark:text-purple-300", border: "border-purple-200 dark:border-purple-800" },
  MULTI_PART:     { bg: "bg-orange-50 dark:bg-orange-900/20", text: "text-orange-700 dark:text-orange-300", border: "border-orange-200 dark:border-orange-800" },
  AMBIGUOUS:      { bg: "bg-zinc-50 dark:bg-zinc-800/40",   text: "text-zinc-600 dark:text-zinc-300",    border: "border-zinc-200 dark:border-zinc-700"    },
};

const QueryClassificationBlock: FC<{ classification: QueryClassification; synthesisMode?: boolean }> = ({
  classification,
  synthesisMode,
}) => {
  const colors = QUERY_TYPE_COLORS[classification.type] ?? QUERY_TYPE_COLORS["AMBIGUOUS"];
  const pct = Math.round(classification.confidence * 100);

  return (
    <div className={`flex items-center gap-2 rounded-md border ${colors.border} ${colors.bg} px-2.5 py-1 text-[11px] not-prose`}>
      <span className={`font-semibold tracking-wide ${colors.text}`}>
        {classification.type.replace("_", " ")}
      </span>
      <span className={`opacity-70 ${colors.text}`}>·</span>
      <span className={`${colors.text} opacity-80`}>{pct}% confidence</span>
      <span className={`opacity-70 ${colors.text}`}>·</span>
      <span className={`${colors.text} opacity-70`}>{Math.round(classification.latency_ms)}ms</span>
      {classification.fallback && (
        <>
          <span className={`opacity-70 ${colors.text}`}>·</span>
          <span className="text-amber-600 dark:text-amber-400 opacity-90">fallback</span>
        </>
      )}
      {synthesisMode && (
        <>
          <span className={`opacity-70 ${colors.text}`}>·</span>
          <span className="text-emerald-600 dark:text-emerald-400 font-medium">⊕ Synthesis</span>
        </>
      )}
    </div>
  );
};

// ── Tool trace timeline ────────────────────────────────────────────────────

interface ToolTraceEntry {
  tool_name: string;
  params?: Record<string, unknown>;
  output?: unknown;
  error?: string | null;
  latency_ms: number;
}

const TOOL_ICONS: Record<string, string> = {
  search_documents:    "🔍",
  extract_entities:    "🏷",
  summarize_chunks:    "📝",
  synthesize_documents:"⊕",
};

const ToolTraceBlock: FC<{ trace: ToolTraceEntry[] }> = ({ trace }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  if (trace.length === 0) return null;

  return (
    <div className="my-2 rounded-md border border-gray-100 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/40 w-full not-prose">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-1.5 w-full px-3 py-1.5 text-left rounded-t-md hover:bg-gray-100 dark:hover:bg-gray-700/40 transition-colors"
      >
        {isExpanded ? (
          <ChevronDown className="h-3 w-3 text-gray-400 shrink-0" />
        ) : (
          <ChevronRight className="h-3 w-3 text-gray-400 shrink-0" />
        )}
        <span className="text-xs text-gray-400 font-medium select-none">
          Tool calls ({trace.length})
        </span>
        <span className="ml-auto text-[10px] text-gray-400 select-none">
          {trace.reduce((s, t) => s + t.latency_ms, 0).toFixed(0)}ms total
        </span>
      </button>
      {isExpanded && (
        <div className="px-3 pb-2 pt-1 border-t border-gray-100 dark:border-gray-700 space-y-2 max-h-72 overflow-y-auto">
          {trace.map((entry, i) => (
            <div key={i} className="text-[11px] font-sans">
              <div className="flex items-center gap-2">
                <span>{TOOL_ICONS[entry.tool_name] ?? "🔧"}</span>
                <span className="font-semibold text-gray-600 dark:text-gray-300">{entry.tool_name}</span>
                <span className="text-gray-400">{entry.latency_ms.toFixed(0)}ms</span>
                {entry.error ? (
                  <span className="text-red-500 text-[10px]">✗ error</span>
                ) : (
                  <span className="text-emerald-500 text-[10px]">✓</span>
                )}
              </div>
              {entry.error && (
                <p className="mt-0.5 ml-5 text-red-500 dark:text-red-400">{entry.error}</p>
              )}
              {!entry.error && entry.output !== undefined && (
                <pre className="mt-0.5 ml-5 text-gray-400 dark:text-gray-500 whitespace-pre-wrap break-all overflow-hidden max-h-20">
                  {JSON.stringify(entry.output, null, 2).slice(0, 400)}
                  {JSON.stringify(entry.output).length > 400 ? "…" : ""}
                </pre>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// ── Failed legs warning ───────────────────────────────────────────────────────

const FailedLegsWarning: FC<{ legs: string[] }> = ({ legs }) => {
  if (legs.length === 0) return null;
  const names: Record<string, string> = {
    dense: "vector",
    qdrant_sparse: "sparse",
    exact: "keyword",
    graph: "graph",
  };
  return (
    <div className="flex items-center gap-1.5 rounded border border-amber-200 dark:border-amber-800 bg-amber-50 dark:bg-amber-900/20 px-2.5 py-1 text-[11px] text-amber-700 dark:text-amber-400 not-prose">
      <span>⚠</span>
      <span>Retrieval leg{legs.length > 1 ? "s" : ""} failed: {legs.map(l => names[l] ?? l).join(", ")}</span>
    </div>
  );
};


type ConfidenceLevel = "very_high" | "high" | "medium" | "low" | "none";

const CONFIDENCE_CONFIG: Record<ConfidenceLevel, {
  steps: number;   // how many of 4 steps are filled
  label: string;
  stepColor: string;
  textColor: string;
  bgColor: string;
  borderColor: string;
}> = {
  very_high: { steps: 4, label: "Very High",  stepColor: "bg-emerald-500", textColor: "text-emerald-700", bgColor: "bg-emerald-50",  borderColor: "border-emerald-200" },
  high:      { steps: 3, label: "High",       stepColor: "bg-green-500",   textColor: "text-green-700",   bgColor: "bg-green-50",    borderColor: "border-green-200"   },
  medium:    { steps: 2, label: "Medium",     stepColor: "bg-yellow-500",  textColor: "text-yellow-700",  bgColor: "bg-yellow-50",   borderColor: "border-yellow-200"  },
  low:       { steps: 1, label: "Low",        stepColor: "bg-orange-500",  textColor: "text-orange-700",  bgColor: "bg-orange-50",   borderColor: "border-orange-200"  },
  none:      { steps: 0, label: "None",       stepColor: "bg-red-400",     textColor: "text-red-700",     bgColor: "bg-red-50",      borderColor: "border-red-200"     },
};

// ── Confidence collapsible (bottom-right of each answer) ────────────────────

const CONFIDENCE_COLORS: Record<ConfidenceLevel, { bar: string; text: string; bg: string; border: string }> = {
  very_high: { bar: "bg-emerald-500", text: "text-emerald-700 dark:text-emerald-400", bg: "bg-emerald-50 dark:bg-emerald-900/20", border: "border-emerald-200 dark:border-emerald-800" },
  high:      { bar: "bg-green-500",   text: "text-green-700 dark:text-green-400",     bg: "bg-green-50 dark:bg-green-900/20",     border: "border-green-200 dark:border-green-800"   },
  medium:    { bar: "bg-yellow-500",  text: "text-yellow-700 dark:text-yellow-400",   bg: "bg-yellow-50 dark:bg-yellow-900/20",   border: "border-yellow-200 dark:border-yellow-800" },
  low:       { bar: "bg-orange-500",  text: "text-orange-700 dark:text-orange-400",   bg: "bg-orange-50 dark:bg-orange-900/20",   border: "border-orange-200 dark:border-orange-800" },
  none:      { bar: "bg-red-400",     text: "text-red-700 dark:text-red-400",         bg: "bg-red-50 dark:bg-red-900/20",         border: "border-red-200 dark:border-red-800"       },
};

const ConfidenceCollapsible: FC<{
  level: ConfidenceLevel;
  score?: number;
  suggestion?: string | null;
  breakdown?: Record<string, unknown>;
}> = ({ level, score, suggestion, breakdown }) => {
  const [open, setOpen] = useState(false);
  const cfg = CONFIDENCE_COLORS[level];
  const label = CONFIDENCE_CONFIG[level].label;
  const pct = score !== undefined ? score : 0;

  return (
    <div className={`rounded-md border ${cfg.border} ${cfg.bg} text-xs not-prose`}>
      {/* collapsed header — always visible */}
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 w-full px-3 py-1.5 text-left"
      >
        {open ? (
          <ChevronDown className={`h-3 w-3 shrink-0 ${cfg.text}`} />
        ) : (
          <ChevronRight className={`h-3 w-3 shrink-0 ${cfg.text}`} />
        )}
        <span className={`font-medium shrink-0 ${cfg.text}`}>
          Confidence: {label}{score !== undefined ? ` · ${score}/100` : ""}
        </span>
        {/* inline progress bar */}
        <div className="flex-1 h-1.5 rounded-full bg-zinc-200 dark:bg-zinc-700 overflow-hidden">
          <div
            className={`h-full rounded-full transition-all ${cfg.bar}`}
            style={{ width: `${pct}%` }}
          />
        </div>
      </button>

      {/* expanded body */}
      {open && (
        <div className={`px-3 pb-2 pt-1 border-t ${cfg.border} space-y-1`}>
          {suggestion && (
            <p className={`${cfg.text} opacity-80`}>{suggestion}</p>
          )}
          {breakdown && Object.keys(breakdown).length > 0 && (
            <div className="space-y-0.5">
              {Object.entries(breakdown)
                .filter(([k]) => !["mode", "total", "failed_legs", "enabled_legs", "producing_legs"].includes(k))
                .map(([k, v]) => (
                  <div key={k} className="flex justify-between gap-4">
                    <span className="text-zinc-500 dark:text-zinc-400">{k.replace(/_/g, " ")}</span>
                    <span className={`font-medium ${cfg.text}`}>{String(v)}</span>
                  </div>
                ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// ── CodeBlock: renders mermaid fences as diagrams, others as <code> ─────────

const CodeBlock: FC<React.HTMLAttributes<HTMLElement> & { inline?: boolean }> = ({
  className,
  children,
  inline,
  ...rest
}) => {
  if (!inline && className?.includes("language-mermaid")) {
    return (
      <MermaidDiagramDynamic
        code={String(children).replace(/\n$/, "")}
      />
    );
  }
  return (
    <code className={className} {...rest}>
      {children}
    </code>
  );
};

export const Answer: FC<{
  messageId?: string;
  chatId?: string;
  markdown: string;
  citations?: Citation[];
  rewrittenQuery?: string;
  retrievedContext?: ContextDoc[];
  confidence?: "very_high" | "high" | "medium" | "low" | "none";
  confidenceScore?: number;
  confidenceBreakdown?: Record<string, unknown>;
  suggestion?: string | null;
  failedLegs?: string[];
  queryClassification?: QueryClassification;
  toolTrace?: ToolTraceEntry[];
  synthesisMode?: boolean;
  isStreaming?: boolean;
  onDelete?: (id: string) => void;
}> = ({ messageId, chatId, markdown, citations = [], rewrittenQuery, retrievedContext, confidence, confidenceScore, confidenceBreakdown, suggestion, failedLegs, queryClassification, toolTrace, synthesisMode, isStreaming = false, onDelete }) => {
  const [citationInfoMap, setCitationInfoMap] = useState<
    Record<string, CitationInfo>
  >({});

  // Debounce citations to prevent rapid API calls during streaming
  const debouncedCitations = useDebouncedValue(citations, 300);

  // Keep refs so CitationLink can read the latest data without changing its
  // identity (avoiding react-markdown remounting all <a> elements every render)
  const citationsRef = useRef(debouncedCitations);
  const citationInfoMapRef = useRef(citationInfoMap);
  citationsRef.current = debouncedCitations;
  citationInfoMapRef.current = citationInfoMap;

  const parsedContent = useMemo(() => {
    // Non-anchored: handles models that emit text before <think> (preamble)
    const completeMatch = markdown.match(/([\s\S]*?)<think>([\s\S]*?)<\/think>([\s\S]*)$/);
    if (completeMatch) {
      const preamble = completeMatch[1];
      const thinkContent = completeMatch[2].trim();
      const afterThink = completeMatch[3].trim();
      return {
        thinkContent,
        isThinkingComplete: true,
        // Preserve any preamble text before the <think> block
        answerText: preamble ? `${preamble.trim()}\n\n${afterThink}`.trim() : afterThink,
      };
    }
    // <think> opened but not yet closed — still streaming
    const openMatch = markdown.match(/([\s\S]*?)<think>([\s\S]*)$/);
    if (openMatch) {
      const preamble = openMatch[1];
      return {
        thinkContent: openMatch[2],
        isThinkingComplete: false,
        answerText: preamble.trim(),
      };
    }
    return { thinkContent: null, isThinkingComplete: false, answerText: markdown };
  }, [markdown]);

  useEffect(() => {
    const fetchCitationInfo = async () => {
      const infoMap: Record<string, CitationInfo> = {};

      for (const citation of debouncedCitations) {
        const { kb_id, document_id } = citation.metadata;
        if (!kb_id || !document_id) continue;

        const key = `${kb_id}-${document_id}`;
        if (infoMap[key]) continue;

        try {
          const [kb, doc] = await Promise.all([
            api.get(`/api/knowledge-base/${kb_id}`),
            api.get(`/api/knowledge-base/${kb_id}/documents/${document_id}`),
          ]);

          infoMap[key] = {
            knowledge_base: {
              name: kb.name,
            },
            document: {
              file_name: doc.file_name,
              knowledge_base: {
                name: kb.name,
              },
            },
          };
        } catch (error) {
          console.error("Failed to fetch citation info:", error);
        }
      }

      setCitationInfoMap(infoMap);
    };

    if (debouncedCitations.length > 0) {
      fetchCitationInfo();
    }
  }, [debouncedCitations]);

  // Stable component reference — never recreated, reads current data from refs.
  // This prevents react-markdown from unmounting/remounting all <a> elements
  // whenever citationInfoMap or debouncedCitations change, which was causing
  // Radix Popover state cascades and "Maximum update depth exceeded".
  const CitationLink = useCallback(
    (
      props: ClassAttributes<HTMLAnchorElement> &
        AnchorHTMLAttributes<HTMLAnchorElement>
    ) => {
      const citationId = props.href?.match(/^(\d+)$/)?.[1];
      const citation = citationId
        ? citationsRef.current[parseInt(citationId) - 1]
        : null;

      if (!citation) {
        return <a>[{props.href}]</a>;
      }

      const citationInfo =
        citationInfoMapRef.current[
          `${citation.metadata.kb_id}-${citation.metadata.document_id}`
        ];

      return (
        <Popover>
          <PopoverTrigger asChild>
            <a
              {...props}
              href="#"
              role="button"
              className="inline-flex items-center gap-1 px-1.5 py-0.5 text-xs font-medium text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-950/50 rounded hover:bg-blue-100 dark:hover:bg-blue-900/60 transition-colors relative"
            >
              <span className="absolute -top-3 -right-1">[{props.href}]</span>
            </a>
          </PopoverTrigger>
          <PopoverContent
            side="top"
            align="start"
            className="max-w-2xl w-[calc(100vw-100px)] p-4 rounded-lg shadow-lg"
          >
            <div className="text-sm space-y-3">
              {citationInfo && (
                <div className="flex items-center gap-2 text-xs font-medium text-foreground bg-muted p-2 rounded">
                  <div className="w-5 h-5 flex items-center justify-center">
                    <FileIcon
                      extension={
                        citationInfo.document.file_name.split(".").pop() || ""
                      }
                      color="#E2E8F0"
                      labelColor="#94A3B8"
                    />
                  </div>
                  <span className="truncate">
                    {citationInfo.knowledge_base.name} /{" "}
                    {citationInfo.document.file_name}
                  </span>
                </div>
              )}
              {/* Score + retrieval leg */}
              {(citation.score !== undefined || citation.retrieval_leg) && (
                <div className="flex items-center gap-2 flex-wrap">
                  {citation.score !== undefined && (
                    <div className="flex items-center gap-1.5 flex-1 min-w-[120px]">
                      <span className="text-xs text-muted-foreground shrink-0">Score:</span>
                      <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
                        <div
                          className="h-full rounded-full bg-blue-500 transition-all"
                          style={{ width: `${Math.round(citation.score * 100)}%` }}
                        />
                      </div>
                      <span className="text-xs text-foreground shrink-0 font-medium">
                        {Math.round(citation.score * 100)}%
                      </span>
                    </div>
                  )}
                  {citation.retrieval_leg && (
                    <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded uppercase tracking-wide ${
                      citation.retrieval_leg === "dense"
                        ? "bg-blue-100 dark:bg-blue-950/60 text-blue-700 dark:text-blue-300"
                        : citation.retrieval_leg === "sparse" || citation.retrieval_leg === "qdrant_sparse"
                        ? "bg-purple-100 dark:bg-purple-950/60 text-purple-700 dark:text-purple-300"
                        : citation.retrieval_leg === "exact"
                        ? "bg-emerald-100 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-300"
                        : citation.retrieval_leg === "graph"
                        ? "bg-orange-100 dark:bg-orange-950/60 text-orange-700 dark:text-orange-300"
                        : "bg-muted text-muted-foreground"
                    }`}>
                      {citation.retrieval_leg.replace("qdrant_", "")}
                    </span>
                  )}
                </div>
              )}
              {/* Per-leg rank breakdown */}
              {(citation.dense_rank !== undefined ||
                citation.qdrant_sparse_rank !== undefined ||
                citation.exact_rank !== undefined) && (
                <div className="grid grid-cols-3 gap-1 text-[10px]">
                  {citation.dense_rank !== undefined && (
                    <div className="flex flex-col items-center rounded bg-blue-50 dark:bg-blue-950/40 px-1.5 py-1">
                      <span className="text-blue-600 dark:text-blue-400 font-medium">Dense</span>
                      <span className="text-blue-800 dark:text-blue-200 font-semibold">#{citation.dense_rank}</span>
                    </div>
                  )}
                  {citation.qdrant_sparse_rank !== undefined && (
                    <div className="flex flex-col items-center rounded bg-purple-50 dark:bg-purple-950/40 px-1.5 py-1">
                      <span className="text-purple-600 dark:text-purple-400 font-medium">Sparse</span>
                      <span className="text-purple-800 dark:text-purple-200 font-semibold">#{citation.qdrant_sparse_rank}</span>
                    </div>
                  )}
                  {citation.exact_rank !== undefined && (
                    <div className="flex flex-col items-center rounded bg-emerald-50 dark:bg-emerald-950/40 px-1.5 py-1">
                      <span className="text-emerald-600 dark:text-emerald-400 font-medium">Exact</span>
                      <span className="text-emerald-800 dark:text-emerald-200 font-semibold">#{citation.exact_rank}</span>
                    </div>
                  )}
                </div>
              )}
              <Divider />
              <div className="text-foreground leading-relaxed prose prose-sm dark:prose-invert max-w-none">
                <Markdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeHighlight, [rehypeKatex, { throwOnError: false }]]}>
                  {cleanChunkText(citation.text)}
                </Markdown>
              </div>
              <Divider />
              {Object.keys(citation.metadata).length > 0 && (
                <div className="text-xs text-muted-foreground bg-muted p-2 rounded">
                  <div className="font-medium mb-2">Debug Info:</div>
                  <div className="space-y-1">
                    {Object.entries(citation.metadata).map(([key, value]) => (
                      <div key={key} className="flex">
                        <span className="font-medium min-w-[100px]">
                          {key}:
                        </span>
                        <span className="text-foreground/80">{String(value)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </PopoverContent>
        </Popover>
      );
    },
    [] // stable — reads from refs
  );

  // Memoize the components object so react-markdown never sees a new reference
  const markdownComponents = useMemo(() => ({ a: CitationLink, code: CodeBlock }), [CitationLink]);

  // Key changes only when citation info is first fetched; this forces a single
  // controlled remount of <Markdown> (so popover content updates after the
  // async fetch), instead of continuous uncontrolled remounts during streaming.
  const citationInfoKey = Object.keys(citationInfoMap).sort().join(",");

  // ── Action handlers ────────────────────────────────────────────────────────
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(() => {
    const plain = parsedContent.answerText.replace(/\[citation:\d+\]/g, "").trim();
    navigator.clipboard.writeText(plain).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }, [parsedContent.answerText]);

  const handleDelete = useCallback(async () => {
    if (!messageId || !chatId) return;
    const confirmed = window.confirm("Delete this message? This cannot be undone.");
    if (!confirmed) return;
    try {
      await api.delete(`/api/chat/${chatId}/messages/${messageId}`);
      onDelete?.(messageId);
    } catch (e) {
      console.error("Failed to delete message:", e);
    }
  }, [messageId, chatId, onDelete]);

  const handleExport = useCallback(async (format: "pdf" | "word" | "image") => {
    if (!messageId || !chatId) return;
    const ext = format === "word" ? "docx" : format === "image" ? "png" : "pdf";
    const url = `/api/chat/${chatId}/messages/${messageId}/export?format=${format}`;
    const token = typeof window !== "undefined" ? window.localStorage.getItem("token") || "" : "";
    try {
      const res = await fetch(url, { headers: { Authorization: `Bearer ${token}` } });
      if (!res.ok) throw new Error(`Export failed: ${res.status}`);
      const blob = await res.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `answer.${ext}`;
      a.click();
      URL.revokeObjectURL(a.href);
    } catch (e) {
      console.error("Export failed:", e);
    }
  }, [messageId, chatId]);

  if (!markdown && !rewrittenQuery && (!retrievedContext || retrievedContext.length === 0)) {
    return (
      <div className="flex flex-col gap-2">
        <Skeleton className="max-w-sm h-4 bg-zinc-200" />
        <Skeleton className="max-w-lg h-4 bg-zinc-200" />
        <Skeleton className="max-w-2xl h-4 bg-zinc-200" />
        <Skeleton className="max-w-lg h-4 bg-zinc-200" />
        <Skeleton className="max-w-xl h-4 bg-zinc-200" />
      </div>
    );
  }

  return (
    <div className="prose prose-sm max-w-full">
      {/* AgentTimeline consolidates: rewrittenQuery, queryClassification, toolTrace, failedLegs, retrievedContext */}
      <AgentTimeline
        rewrittenQuery={rewrittenQuery}
        retrievedContext={retrievedContext}
        queryClassification={queryClassification}
        toolTrace={toolTrace}
        failedLegs={failedLegs}
        isStreaming={isStreaming}
      />
      {confidence === "none" && suggestion && (
        <div className="flex items-start gap-2 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800 mb-2">
          <span className="mt-0.5 shrink-0">⚠</span>
          <span>{suggestion}</span>
        </div>
      )}
      {!markdown && (
        <div className="flex flex-col gap-2 mt-2">
          <Skeleton className="max-w-sm h-4 bg-zinc-200" />
          <Skeleton className="max-w-lg h-4 bg-zinc-200" />
          <Skeleton className="max-w-2xl h-4 bg-zinc-200" />
        </div>
      )}
      {parsedContent.thinkContent !== null && (
        <ThinkBlock
          content={parsedContent.thinkContent}
          isComplete={parsedContent.isThinkingComplete}
        />
      )}
      {parsedContent.answerText && (
        <Markdown
          key={citationInfoKey}
          remarkPlugins={[remarkGfm, remarkMath]}
          rehypePlugins={[rehypeHighlight, [rehypeKatex, { throwOnError: false }]]}
          components={markdownComponents}
        >
          {parsedContent.answerText}
        </Markdown>
      )}

      {/* ── Bottom bar: actions left, confidence right ─────────────────────── */}
      {markdown && (
        <div className="flex items-start justify-between gap-4 mt-3 not-prose">
          {/* Left: action buttons */}
          <div className="flex items-center gap-1 flex-shrink-0">
            <button
              onClick={handleCopy}
              title={copied ? "Copied!" : "Copy text"}
              className="flex items-center gap-1 px-2 py-1 rounded text-xs text-zinc-500 hover:text-zinc-800 hover:bg-zinc-100 dark:hover:text-zinc-200 dark:hover:bg-zinc-800 transition-colors"
            >
              <Copy className="h-3.5 w-3.5" />
              <span>{copied ? "Copied" : "Copy"}</span>
            </button>
            <button
              onClick={handleDelete}
              title="Delete message"
              className="flex items-center gap-1 px-2 py-1 rounded text-xs text-zinc-500 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors"
            >
              <Trash2 className="h-3.5 w-3.5" />
              <span>Delete</span>
            </button>
            <button
              onClick={() => handleExport("word")}
              title="Export as Word"
              className="flex items-center gap-1 px-2 py-1 rounded text-xs text-zinc-500 hover:text-blue-600 hover:bg-blue-50 dark:hover:bg-blue-900/20 transition-colors"
            >
              <FileText className="h-3.5 w-3.5" />
              <span>Word</span>
            </button>
            <button
              onClick={() => handleExport("pdf")}
              title="Export as PDF"
              className="flex items-center gap-1 px-2 py-1 rounded text-xs text-zinc-500 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors"
            >
              <FileType className="h-3.5 w-3.5" />
              <span>PDF</span>
            </button>
            <button
              onClick={() => handleExport("image")}
              title="Export as image"
              className="flex items-center gap-1 px-2 py-1 rounded text-xs text-zinc-500 hover:text-purple-600 hover:bg-purple-50 dark:hover:bg-purple-900/20 transition-colors"
            >
              <FileImage className="h-3.5 w-3.5" />
              <span>Image</span>
            </button>
          </div>

          {/* Right: confidence collapsible */}
          {confidence && confidence !== "none" && (
            <div className="flex-1 min-w-0 max-w-xs">
              <ConfidenceCollapsible
                level={confidence}
                score={confidenceScore}
                suggestion={suggestion}
                breakdown={confidenceBreakdown}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
};
