import React, {
  FC,
  useMemo,
  useEffect,
  useState,
  useRef,
  useCallback,
  useContext,
  createContext,
  ClassAttributes,
} from "react";
import { AnchorHTMLAttributes } from "react";
import { Copy, Trash2, FileText, FileImage, FileType } from "lucide-react";
import { AgenticProgress, AgentStepEvent } from "./agentic-progress";
import { AgentLoopPanel } from "./agent-loop-panel";
import { SelectionActions } from "./selection-actions";
import {
  Reasoning,
  ReasoningTrigger,
  ReasoningContent,
} from "@/components/ai-elements/reasoning";
import {
  Task as TaskCollapsible,
  TaskTrigger,
  TaskContent,
  TaskItem,
} from "@/components/ai-elements/task";
import { Shimmer } from "@/components/ai-elements/shimmer";
import { Suggestions, Suggestion } from "@/components/ai-elements/suggestion";
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
  { ssr: false, loading: () => <div className="h-48 w-full animate-pulse bg-muted rounded" /> }
);
const EChartsDiagramDynamic = dynamic(
  () => import("./echarts-diagram"),
  { ssr: false, loading: () => <div className="h-72 w-full animate-pulse bg-muted rounded" /> }
);
import { api, handleAuthRedirect } from "@/lib/api";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
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

interface CitationMetadata {
  kb_id?: number;
  document_id?: number;
  source?: string;
  [key: string]: unknown;
}

interface Citation {
  id: number;
  text: string;
  metadata: CitationMetadata;
  kb_id?: number;
  data_store_id?: number;
  document_id?: number;
  file_name?: string;
  chunk_index?: number;
  score?: number;
  dense_rank?: number;
  sparse_rank?: number;
  exact_rank?: number;
  retrieval_leg?: string;
  _legs?: string[];
  _reranker_score?: number;
  qdrant_point_id?: string;
}

interface KnowledgeBaseInfo {
  name: string;
}

interface DocumentInfo {
  file_name: string;
  title?: string | null;
  knowledge_base: KnowledgeBaseInfo;
}

interface CitationInfo {
  knowledge_base: KnowledgeBaseInfo;
  document: DocumentInfo;
}

// Minimal shape returned by the generic /api/knowledge-base/documents/{id} endpoint.
// Used for data store documents that don't have a kb_id.
interface GenericDocInfo {
  file_name: string;
  title?: string | null;
  parent_name: string | null;
}

function parseThinkContent(markdown: string): {
  thinkContent: string | null;
  isThinkingComplete: boolean;
  answerText: string;
} {
  let thinkContent: string | null = null;
  let isThinkingComplete = false;
  let answerText = markdown;

  for (const tag of ["think", "reasoning"]) {
    const completeMatch = markdown.match(
      new RegExp(
        `([\\s\\S]*?)<${tag}>([\\s\\S]*?)<\\s*/\\s*${tag}\\s*>([\\s\\S]*)$`,
      ),
    );
    if (completeMatch) {
      const preamble = completeMatch[1];
      thinkContent = completeMatch[2].trim();
      const afterThink = completeMatch[3].trim();
      answerText = preamble ? `${preamble.trim()}\n\n${afterThink}`.trim() : afterThink;
      isThinkingComplete = true;
      break;
    }
    const openMatch = markdown.match(
      new RegExp(`([\\s\\S]*?)<${tag}>([\\s\\S]*)$`),
    );
    if (openMatch) {
      thinkContent = openMatch[2];
      answerText = openMatch[1].trim();
      isThinkingComplete = false;
      break;
    }
  }

  if (thinkContent === null) {
    const channelMatch = markdown.match(
      new RegExp(
        `([\\s\\S]*?)<\\s*\\|channel\\s*>thought([\\s\\S]*?)<\\s*channel\\s*\\|>([\\s\\S]*)$`,
      ),
    );
    if (channelMatch) {
      const preamble = channelMatch[1];
      thinkContent = channelMatch[2].trim();
      const afterThink = channelMatch[3].trim();
      answerText = preamble ? `${preamble.trim()}\n\n${afterThink}`.trim() : afterThink;
      isThinkingComplete = true;
    }
    if (thinkContent === null) {
      const channelOpenMatch = markdown.match(
        new RegExp(
          `([\\s\\S]*?)<\\s*\\|channel\\s*>thought([\\s\\S]*)$`,
        ),
      );
      if (channelOpenMatch) {
        const preamble = channelOpenMatch[1];
        thinkContent = channelOpenMatch[2];
        answerText = preamble.trim();
        isThinkingComplete = false;
      }
    }
  }

  return { thinkContent, isThinkingComplete, answerText };
}

function buildFetchPairs(citations: Citation[]) {
  const seenKb = new Set<string>();
  const kbPairs: Array<{ key: string; kbId: number; docId: number }> = [];
  const seenGeneric = new Set<string>();
  const genericDocIds: Array<{ key: string; docId: number }> = [];

  for (const citation of citations) {
    const meta = citation.metadata || {};
    const effectiveKbId = citation.kb_id ?? meta.kb_id;
    const effectiveDocId = citation.document_id ?? meta.document_id;
    if (!effectiveDocId) continue;

    if (effectiveKbId) {
      const key = `${effectiveKbId}-${effectiveDocId}`;
      if (seenKb.has(key)) continue;
      seenKb.add(key);
      kbPairs.push({ key, kbId: effectiveKbId, docId: effectiveDocId });
    } else {
      const key = `doc-${effectiveDocId}`;
      if (seenGeneric.has(key)) continue;
      seenGeneric.add(key);
      genericDocIds.push({ key, docId: effectiveDocId });
    }
  }

  return { kbPairs, genericDocIds };
}

async function fetchKbBatch(
  kbPairs: Array<{ key: string; kbId: number; docId: number }>,
): Promise<Array<{ key: string; info: CitationInfo } | null>> {
  return Promise.all(
    kbPairs.map(async ({ key, kbId, docId }) => {
      try {
        const [kb, doc] = await Promise.all([
          api.get(`/api/knowledge-base/${kbId}`),
          api.get(`/api/knowledge-base/${kbId}/documents/${docId}`),
        ]);
        return {
          key,
          info: {
            knowledge_base: { name: kb.name },
            document: {
              file_name: doc.file_name,
              title: doc.title,
              knowledge_base: { name: kb.name },
            },
          } as CitationInfo,
        };
      } catch (error) {
        console.error("Failed to fetch citation info:", error);
        return null;
      }
    }),
  );
}

async function fetchGenericBatch(
  genericDocIds: Array<{ key: string; docId: number }>,
): Promise<Array<{ key: string; info: GenericDocInfo } | null>> {
  return Promise.all(
    genericDocIds.map(async ({ key, docId }) => {
      try {
        const doc = await api.get(`/api/knowledge-base/documents/${docId}`);
        return {
          key,
          info: {
            file_name: doc.file_name,
            title: doc.title,
            parent_name: doc.parent_name,
          } as GenericDocInfo,
        };
      } catch (error) {
        console.error("Failed to fetch generic doc info:", error);
        return null;
      }
    }),
  );
}

// ── CodeBlock: renders mermaid/echarts fences as diagrams, others as <code> ───

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
  if (!inline && className?.includes("language-echarts")) {
    return (
      <EChartsDiagramDynamic
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

const DEBUG_KEYS: string[] = ['kb_id', 'data_store_id', 'document_id', 'chunk_index', '_legs', '_reranker_score', 'qdrant_point_id'];

const DebugDetails: FC<{ citation: Citation }> = ({ citation }) => {
  const hasDebug = DEBUG_KEYS.some(
    (k) => k in citation && (citation as any)[k] !== undefined && (citation as any)[k] !== null,
  );
  if (!hasDebug) return null;
  return (
    <details className="text-xs group">
      <summary className="cursor-pointer select-none list-none flex items-center gap-1 text-muted-foreground hover:text-foreground transition-colors py-0.5">
        <svg className="h-3 w-3 transition-transform group-open:rotate-90 shrink-0" viewBox="0 0 12 12" fill="currentColor">
          <path d="M4.5 2 L9 6 L4.5 10" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
        Debug Info
      </summary>
      <div className="mt-1.5 bg-muted text-muted-foreground p-2 rounded space-y-1">
        {DEBUG_KEYS.filter(
          (k) => k in citation && (citation as any)[k] !== undefined && (citation as any)[k] !== null,
        ).map((key) => {
          const legNames: Record<string, string> = {
            dense: "vector",
            exact: "keyword",
            graph: "graph",
          };
          const raw = (citation as any)[key];
          if (key === '_legs' && Array.isArray(raw)) {
            return (
              <div key={key} className="flex">
                <span className="font-medium min-w-[100px] shrink-0">{key}:</span>
                <span className="text-foreground/80 break-all">{raw.map((l: string) => legNames[l] ?? l).join(", ")}</span>
              </div>
            );
          }
          return (
            <div key={key} className="flex">
              <span className="font-medium min-w-[100px] shrink-0">{key}:</span>
              <span className="text-foreground/80 break-all">{String(raw)}</span>
            </div>
          );
        })}
      </div>
    </details>
  );
};

const RETRIEVAL_LEG_COLORS: Record<string, string> = {
  dense: "bg-blue-100 dark:bg-blue-950/60 text-blue-700 dark:text-blue-300",
  sparse: "bg-purple-100 dark:bg-purple-950/60 text-purple-700 dark:text-purple-300",
  exact: "bg-emerald-100 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-300",
  graph: "bg-orange-100 dark:bg-orange-950/60 text-orange-700 dark:text-orange-300",
};

const RetrievalLegBadge: FC<{ leg: string }> = ({ leg }) => {
  const colorClass = RETRIEVAL_LEG_COLORS[leg] ?? "bg-muted text-muted-foreground";
  return (
    <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded uppercase tracking-wide ${colorClass}`}>
      {leg.replace("qdrant_", "")}
    </span>
  );
};

const CitationScoreBar: FC<{ score: number }> = ({ score }) => (
  <div className="flex items-center gap-1.5 flex-1 min-w-[120px]">
    <span className="text-xs text-muted-foreground shrink-0">Score:</span>
    <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
      <div
        className="h-full rounded-full bg-blue-500 transition-all"
        style={{ width: `${Math.round(score * 100)}%` }}
      />
    </div>
    <span className="text-xs text-foreground shrink-0 font-medium">
      {Math.round(score * 100)}%
    </span>
  </div>
);

const CitationScoreAndLeg: FC<{ citation: Citation }> = ({ citation }) => {
  if (citation.score === undefined && !citation.retrieval_leg) return null;
  return (
    <div className="flex items-center gap-2 flex-wrap">
      {citation.score !== undefined && <CitationScoreBar score={citation.score} />}
      {citation.retrieval_leg && <RetrievalLegBadge leg={citation.retrieval_leg} />}
    </div>
  );
};

const CitationRankBreakdown: FC<{ citation: Citation }> = ({ citation }) => {
  if (citation.dense_rank === undefined && citation.sparse_rank === undefined && citation.exact_rank === undefined) return null;
  return (
    <div className="grid grid-cols-3 gap-1 text-[10px]">
      {citation.dense_rank !== undefined && (
        <div className="flex flex-col items-center rounded bg-blue-50 dark:bg-blue-950/40 px-1.5 py-1">
          <span className="text-blue-600 dark:text-blue-400 font-medium">Dense</span>
          <span className="text-blue-800 dark:text-blue-200 font-semibold">#{citation.dense_rank}</span>
        </div>
      )}
      {citation.sparse_rank !== undefined && (
        <div className="flex flex-col items-center rounded bg-purple-50 dark:bg-purple-950/40 px-1.5 py-1">
          <span className="text-purple-600 dark:text-purple-400 font-medium">Sparse</span>
          <span className="text-purple-800 dark:text-purple-200 font-semibold">#{citation.sparse_rank}</span>
        </div>
      )}
      {citation.exact_rank !== undefined && (
        <div className="flex flex-col items-center rounded bg-emerald-50 dark:bg-emerald-950/40 px-1.5 py-1">
          <span className="text-emerald-600 dark:text-emerald-400 font-medium">Exact</span>
          <span className="text-emerald-800 dark:text-emerald-200 font-semibold">#{citation.exact_rank}</span>
        </div>
      )}
    </div>
  );
};

function buildDownloadUrl(effectiveKbId: number | undefined, effectiveDocId: number | undefined): string | null {
  if (effectiveKbId && effectiveDocId) {
    return `/api/knowledge-base/${effectiveKbId}/documents/${effectiveDocId}/download`;
  }
  if (effectiveDocId) {
    return `/api/knowledge-base/documents/${effectiveDocId}/download`;
  }
  return null;
}

function shouldShowFilename(displayTitle: string | null, displayFileName?: string): boolean {
  if (!displayTitle || !displayFileName) return false;
  return displayTitle !== displayFileName.replace(/\.[^.]+$/, "").replace(/[_-]/g, " ").replace(/\s+/g, " ").trim();
}

function firstNonNull<T>(...values: (T | null | undefined)[]): T | null {
  for (const v of values) {
    if (v !== null && v !== undefined) return v;
  }
  return null;
}

function resolveDisplayFields(
  citationInfo: CitationInfo | undefined,
  genericInfo: GenericDocInfo | null | undefined,
) {
  const displayTitle = firstNonNull(citationInfo?.document.title, genericInfo?.title);
  const displayFileName = firstNonNull(citationInfo?.document.file_name, genericInfo?.file_name) ?? undefined;
  const displayParentName = firstNonNull(citationInfo?.knowledge_base.name, genericInfo?.parent_name) ?? "Unknown";
  return { displayTitle, displayFileName, displayParentName };
}

const CitationFileHeader: FC<{
  citationInfo: CitationInfo | undefined;
  genericInfo: GenericDocInfo | null | undefined;
  effectiveKbId: number | undefined;
  effectiveDocId: number | undefined;
}> = ({ citationInfo, genericInfo, effectiveKbId, effectiveDocId }) => {
  const { displayTitle, displayFileName, displayParentName } = resolveDisplayFields(citationInfo, genericInfo);
  const showFilenameInCitation = shouldShowFilename(displayTitle, displayFileName);
  const downloadUrl = buildDownloadUrl(effectiveKbId, effectiveDocId);
  if (!displayTitle && !displayFileName) return null;
  return (
    <div className="flex items-center gap-2 text-xs font-medium text-foreground bg-muted p-2 rounded">
      <div className="w-5 h-5 flex items-center justify-center shrink-0">
        <FileIcon
          extension={displayFileName?.split(".").pop() || ""}
          color="#E2E8F0"
          labelColor="#94A3B8"
        />
      </div>
      {downloadUrl ? (
        <a
          href={downloadUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="truncate hover:underline text-primary"
          title={`Open ${displayFileName}`}
        >
          <span className="font-medium">{displayParentName}</span>
          {" / "}
          <span className="font-medium">{displayTitle || displayFileName}</span>
          {showFilenameInCitation && (
            <span className="text-muted-foreground/70"> ({displayFileName})</span>
          )}
        </a>
      ) : (
        <span className="truncate">
          <span className="font-medium">{displayParentName}</span>
          {" / "}
          <span className="font-medium">{displayTitle || displayFileName}</span>
          {showFilenameInCitation && (
            <span className="text-muted-foreground/70"> ({displayFileName})</span>
          )}
        </span>
      )}
    </div>
  );
};

const CitationLinkContext = createContext<{
  citations: Citation[];
  citationInfoMap: Record<string, CitationInfo>;
  genericDocMap: Record<string, GenericDocInfo>;
}>(null as any);

type CitationLinkProps = ClassAttributes<HTMLAnchorElement> &
  AnchorHTMLAttributes<HTMLAnchorElement>;

const CitationLink: FC<CitationLinkProps> = (props) => {
  const { citations, citationInfoMap, genericDocMap } = useContext(CitationLinkContext);

  const citationId = props.href?.match(/^(\d+)$/)?.[1];
  const citation = citationId
    ? citations.find((c: any) => c.id === parseInt(citationId)) ??
      citations[parseInt(citationId) - 1]
    : null;

  if (!citation) {
    return <a>[{props.href}]</a>;
  }

  const top = citation as Record<string, any>;
  const meta = (citation.metadata as Record<string, any>) || {};
  const effectiveKbId = top.kb_id ?? meta.kb_id;
  const effectiveDocId = top.document_id ?? meta.document_id;
  const citationInfo = citationInfoMap[`${effectiveKbId}-${effectiveDocId}`];
  const genericInfo = effectiveDocId ? genericDocMap[`doc-${effectiveDocId}`] : null;

  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          className="inline-flex items-center px-1 py-0.5 text-xs font-medium text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-950/50 rounded hover:bg-blue-100 dark:hover:bg-blue-900/60 transition-colors"
        >
          [{props.href}]
        </button>
      </PopoverTrigger>
      <PopoverContent
        side="top"
        align="start"
        sideOffset={6}
        collisionPadding={12}
        className="max-w-2xl w-[calc(100vw-100px)] p-0 rounded-lg shadow-lg overflow-hidden"
      >
        <div className="text-sm space-y-3 max-h-[min(70vh,520px)] overflow-y-auto p-4" style={{ scrollbarGutter: "stable" }}>
          <CitationFileHeader
            citationInfo={citationInfo}
            genericInfo={genericInfo}
            effectiveKbId={effectiveKbId}
            effectiveDocId={effectiveDocId}
          />
          <CitationScoreAndLeg citation={citation} />
          <CitationRankBreakdown citation={citation} />
          <Divider />
          <div className="text-foreground leading-relaxed prose prose-sm dark:prose-invert max-w-none">
            <Markdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeHighlight, [rehypeKatex, { throwOnError: false }]]}>
              {cleanChunkText(citation.text)}
            </Markdown>
          </div>
          <Divider />
          <DebugDetails citation={citation} />
        </div>
      </PopoverContent>
    </Popover>
  );
};

const TaskStatusIcon: FC<{ status: string }> = ({ status }) => {
  if (status === "done") {
    return (
      <svg className="text-emerald-500" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg" width="12" height="12">
        <circle cx="6" cy="6" r="5.5" fill="currentColor" fillOpacity="0.15" stroke="currentColor" strokeWidth="1"/>
        <path d="M3.5 6l1.8 1.8L8.5 4.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
    );
  }
  if (status === "active") {
    return (
      <svg className="text-primary animate-spin" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg" width="12" height="12">
        <circle cx="6" cy="6" r="4.5" stroke="currentColor" strokeOpacity="0.2" strokeWidth="1.5"/>
        <path d="M6 1.5A4.5 4.5 0 0 1 10.5 6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
      </svg>
    );
  }
  return (
    <svg className="text-muted-foreground/40" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg" width="12" height="12">
      <circle cx="6" cy="6" r="5.5" stroke="currentColor" strokeWidth="1"/>
    </svg>
  );
};

// ── Answer Component ─────────────────────────────────────────────────────────

export const Answer: FC<{
  messageId?: string;
  chatId?: string;
  markdown: string;
  citations?: Citation[];
  rewrittenQuery?: string;
  confidence?: "very_high" | "high" | "medium" | "low" | "none";
  confidenceScore?: number;
  confidenceBreakdown?: Record<string, unknown>;
  suggestion?: string | null;
  failedLegs?: string[];
  toolTrace?: Array<{
    tool_name: string;
    params?: Record<string, unknown>;
    output?: unknown;
    error?: string | null;
    latency_ms: number;
  }>;
  agentSteps?: AgentStepEvent[];
  taskList?: Array<{ id: number; text: string; status: string }>;
  progressMessages?: Array<{ phase: string; message: string; details?: Record<string, unknown>; rewritten_query?: string; original_query?: string }>;
  synthesisMode?: boolean;
  isStreaming?: boolean;
  onDelete?: (id: string) => void;
  // Final evaluation metrics (from answer_evaluation_node)
  finalConfidence?: number;
  finalConfidenceLevel?: "very_high" | "high" | "medium" | "low" | "none";
  faithfulness?: number;
  completeness?: number;
  retrievalScore?: number;
  // Enterprise agent loop state
  plan?: Record<string, unknown>;
  toolCalls?: Array<Record<string, unknown>>;
  toolObservations?: Array<Record<string, unknown>>;
  lastAnswerObject?: {
    followups?: string[];
    retry_strategy?: string;
    suggestion?: string;
    [key: string]: unknown;
  };
  chartOption?: Record<string, unknown>;
  chartOptions?: Array<Record<string, unknown>>;
  onFollowUp?: (query: string) => void;
}> = React.memo(({ messageId, chatId, markdown, citations = [], rewrittenQuery, confidence, confidenceScore, suggestion, failedLegs, agentSteps, taskList, progressMessages, isStreaming = false, onDelete, finalConfidence, finalConfidenceLevel, faithfulness, completeness, retrievalScore, toolCalls, toolObservations, chartOption, chartOptions, lastAnswerObject, onFollowUp }) => {
  const [citationInfoMap, setCitationInfoMap] = useState<
    Record<string, CitationInfo>
  >({});
  // Map for data store documents (no kb_id) — keyed by doc_id
  const [genericDocMap, setGenericDocMap] = useState<
    Record<string, GenericDocInfo>
  >({});

  // Debounce citations to prevent rapid API calls during streaming
  const debouncedCitations = useDebouncedValue(citations, 300);

  // renderKey forces <Markdown> to remount when citations become ready.
  // Only bump on the citations-empty -> citations-present transition itself
  // (already captured by the "with-citations"/"no-citations" key segment
  // below) — do NOT also bump on the isStreaming edge, since answer_rewrite
  // now delivers citations well before streaming ends, and remounting again
  // at stream-end caused a redundant visible "refresh" with no content change.
  const [renderKey, setRenderKey] = useState(0);
  const hadCitationsRef = useRef(citations.length > 0);
  useEffect(() => {
    if (citations.length > 0 && !hadCitationsRef.current) {
      hadCitationsRef.current = true;
      setRenderKey((k) => k + 1);
    }
  }, [citations.length]);

  // Extract generate_answer latency from agentSteps
  const generateAnswerLatencyMs = useMemo(() => {
    if (!agentSteps?.length) return null;
    const doneStep = agentSteps.find(
      (s) => s.node === "generate_answer" && s.status === "done",
    );
    return doneStep?.latency_ms ?? null;
  }, [agentSteps]);

  // Filter out generate_answer from agentSteps — we display its latency inline
  const filteredAgentSteps = useMemo(() => {
    if (!agentSteps?.length) return undefined;
    return agentSteps.filter((s) => s.node !== "generate_answer");
  }, [agentSteps]);

  const parsedContent = useMemo(() => parseThinkContent(markdown), [markdown]);

  useEffect(() => {
    const controller = new AbortController();
    let cancelled = false;

    const fetchCitationInfo = async () => {
      const { kbPairs, genericDocIds } = buildFetchPairs(debouncedCitations);

      const kbResults = await fetchKbBatch(kbPairs);
      const genericResults = await fetchGenericBatch(genericDocIds);

      if (cancelled) return;

      const infoMap: Record<string, CitationInfo> = {};
      for (const r of kbResults) {
        if (r) infoMap[r.key] = r.info;
      }
      setCitationInfoMap(infoMap);

      const gMap: Record<string, GenericDocInfo> = {};
      for (const r of genericResults) {
        if (r) gMap[r.key] = r.info;
      }
      setGenericDocMap(gMap);
    };

    if (debouncedCitations.length > 0) {
      fetchCitationInfo();
    }

    return () => { cancelled = true; controller.abort(); };
  }, [debouncedCitations]);

  const citationCtxValue = useMemo(() => ({
    citations,
    citationInfoMap,
    genericDocMap,
  }), [citations, citationInfoMap, genericDocMap]);

  const markdownComponents = useMemo(() => ({ a: CitationLink, code: CodeBlock }), []);

  // ── Action handlers ────────────────────────────────────────────────────────
  const [copied, setCopied] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);

  const handleCopy = useCallback(() => {
    const text = parsedContent.answerText
      .replace(/```echarts\s*\n[\s\S]*?\n```/g, "")
      .trim();
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }, [parsedContent.answerText]);

  const handleDelete = useCallback(async () => {
    if (!messageId || !chatId) return;
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

    // Collect rendered chart PNGs from this message's content only.
    // ECharts instances are stored on the DOM element; getDataURL
    // produces a base64 PNG data URL.
    const charts: string[] = [];
    if (contentRef.current) {
      const containers = contentRef.current.querySelectorAll(".echarts-diagram");
      containers.forEach((el) => {
        const instance = (el as any).__echarts_instance__;
        if (instance && typeof instance.getDataURL === "function") {
          try {
            charts.push(instance.getDataURL({ type: "png", pixelRatio: 2 }));
          } catch {
            // skip if chart not ready
          }
        }
      });
    }

    const url = `/api/chat/${chatId}/messages/${messageId}/export`;
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ format, charts }),
      });
      if (await handleAuthRedirect(res)) return;
      if (!res.ok) throw new Error(`Export failed: ${res.status}`);
      const blob = await res.blob();
      const blobUrl = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = blobUrl;
      a.download = `answer.${ext}`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
    } catch (e) {
      console.error("Export failed:", e);
    }
  }, [messageId, chatId]);

  const contentRef = useRef<HTMLDivElement>(null);

  if (!markdown && !rewrittenQuery) {
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
    <div className="prose prose-sm max-w-full" ref={contentRef}>
      {/* Subtask checklist — shown during streaming for complex multi-part queries */}
      {taskList && taskList.length > 1 && (
        <div className="not-prose mb-2">
          <TaskCollapsible defaultOpen={true}>
            <TaskTrigger title={
              taskList.some(t => t.status === "active" || t.status === "pending")
                ? `Subtasks (${taskList.filter(t => t.status === "done").length}/${taskList.length})`
                : `Subtasks (${taskList.length} completed)`
            } />
            <TaskContent>
              {taskList.map((task) => (
                <TaskItem key={task.id}>
                  <div className="flex items-start gap-2">
                    <span className="mt-0.5 shrink-0 w-3.5 h-3.5 flex items-center justify-center">
                      <TaskStatusIcon status={task.status} />
                    </span>
                    <span className={`text-[12px] leading-snug ${
                      task.status === "done"
                        ? "text-muted-foreground line-through decoration-muted-foreground/40"
                        : task.status === "active"
                        ? "text-foreground font-medium"
                        : "text-muted-foreground"
                    }`}>
                      {task.status === "active" ? <Shimmer duration={2}>{task.text}</Shimmer> : task.text}
                    </span>
                  </div>
                </TaskItem>
              ))}
            </TaskContent>
          </TaskCollapsible>
        </div>
      )}
      {/* Agentic progress — transient, grey, fades between phases.
          Single source of truth for status text; raw per-leg progress
          events (dense/sparse/exact/neo4j) are folded into "Gathering
          sources …" here instead of also being shown verbatim. */}
      <AgenticProgress agentSteps={filteredAgentSteps} isStreaming={isStreaming} toolCalls={toolCalls} toolObservations={toolObservations} progressMessages={progressMessages} />

      {/* Confidence warning (no confidence) */}
      {confidence === "none" && suggestion && (
        <div className="flex items-start gap-2 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800 mb-2">
          <span className="mt-0.5 shrink-0">⚠</span>
          <span>{suggestion}</span>
        </div>
      )}
      {parsedContent.thinkContent !== null && (
        <Reasoning
          isStreaming={!parsedContent.isThinkingComplete}
          defaultOpen={!parsedContent.isThinkingComplete}
        >
          <ReasoningTrigger />
          <ReasoningContent>
            {parsedContent.thinkContent}
          </ReasoningContent>
        </Reasoning>
      )}
      
      {parsedContent.answerText && (
        <CitationLinkContext.Provider value={citationCtxValue}>
          <div className="prose prose-sm dark:prose-invert max-w-none">
            <Markdown
              key={`${citations.length > 0 ? "with-citations" : "no-citations"}-${renderKey}`}
              remarkPlugins={[remarkGfm, remarkMath]}
              rehypePlugins={[rehypeHighlight, [rehypeKatex, { throwOnError: false }]]}
              components={markdownComponents}
            >
              {parsedContent.answerText}
            </Markdown>
          </div>
        </CitationLinkContext.Provider>
      )}
      
      {isStreaming && (
        <span className="inline-block w-2 h-4 ml-0.5 align-middle bg-foreground/80 animate-pulse" aria-hidden="true" />
      )}

      {/* Selection actions — floating toolbar on text selection */}
      {onFollowUp && !isStreaming && parsedContent.answerText && (
        <SelectionActions
          containerRef={contentRef}
          onAction={onFollowUp}
          disabled={isStreaming}
        />
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
            {/* Obsolete: "Refresh Citations" button — citation metadata is
                fetched automatically in the useEffect above. Removed to
                reduce UI clutter. Re-enable if manual re-fetch is needed.
            {citations.length > 0 && !isStreaming && (
              <button
                onClick={() => setCitationRefreshTick((t) => t + 1)}
                title="Refresh citations"
                className="flex items-center gap-1 px-2 py-1 rounded text-xs text-zinc-500 hover:text-zinc-800 hover:bg-zinc-100 dark:hover:text-zinc-200 dark:hover:bg-zinc-800 transition-colors"
              >
                <RefreshCw className="h-3.5 w-3.5" />
                <span>Citations</span>
              </button>
            )} */}
            <button
              onClick={() => handleExport("word")}
              title="Export as Word"
              className="flex items-center gap-1 px-2 py-1 rounded text-xs text-zinc-500 hover:text-blue-600 hover:bg-zinc-100 dark:hover:text-zinc-200 dark:hover:bg-zinc-800 transition-colors"
            >
              <FileText className="h-3.5 w-3.5" />
              <span>Word</span>
            </button>
            <button
              onClick={() => handleExport("pdf")}
              title="Export as PDF"
              className="flex items-center gap-1 px-2 py-1 rounded text-xs text-zinc-500 hover:text-red-600 hover:bg-zinc-100 dark:hover:text-zinc-200 dark:hover:bg-zinc-800 transition-colors"
            >
              <FileType className="h-3.5 w-3.5" />
              <span>PDF</span>
            </button>
            <button
              onClick={() => handleExport("image")}
              title="Export as image"
              className="flex items-center gap-1 px-2 py-1 rounded text-xs text-zinc-500 hover:text-purple-600 hover:bg-zinc-100 dark:hover:text-zinc-200 dark:hover:bg-zinc-800 transition-colors"
            >
              <FileImage className="h-3.5 w-3.5" />
              <span>Image</span>
            </button>
            <button
              onClick={() => setConfirmDelete(true)}
              title="Delete message"
              className="flex items-center gap-1 px-2 py-1 rounded text-xs text-zinc-500 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors"
            >
              <Trash2 className="h-3.5 w-3.5" />
              <span>Delete</span>
            </button>
          </div>

          {/* Right: confidence + speed */}
          <div className="flex-1 min-w-0 max-w-[14rem] flex flex-col gap-1.5">
            {finalConfidence !== undefined && (
              <ConfidenceCollapsible
                level={finalConfidenceLevel}
                score={confidenceScore}
                suggestion={suggestion}
                finalConfidence={finalConfidence}
                finalConfidenceLevel={finalConfidenceLevel}
                faithfulness={faithfulness}
                completeness={completeness}
                retrievalScore={retrievalScore}
                failedLegs={failedLegs}
              />
            )}
            {generateAnswerLatencyMs !== null && (
              <span className="text-[10px] text-zinc-400 dark:text-zinc-500 select-none">
                Generated in {generateAnswerLatencyMs < 1000 ? `${generateAnswerLatencyMs}ms` : `${(generateAnswerLatencyMs / 1000).toFixed(1)}s`}
              </span>
            )}
          </div>
        </div>
      )}

      {/* ── Follow-up suggestions ─────────────────────────────────────────── */}
      {!isStreaming && lastAnswerObject?.followups && Array.isArray(lastAnswerObject.followups) && lastAnswerObject.followups.length > 0 && onFollowUp && (
        <div className="mt-3 not-prose">
          {typeof lastAnswerObject.retry_strategy === "string" && lastAnswerObject.retry_strategy && (
            <span className="text-xs text-muted-foreground mr-2">
              {lastAnswerObject.retry_strategy === "widen" ? "Try a broader search:" :
               lastAnswerObject.retry_strategy === "narrow" ? "Try a narrower search:" :
               lastAnswerObject.retry_strategy === "pinpoint" ? "Look up this exact ID:" : ""}
            </span>
          )}
          <Suggestions>
            {(lastAnswerObject.followups as string[]).map((s: string, i: number) => (
              <Suggestion key={i} suggestion={s} onClick={onFollowUp} />
            ))}
          </Suggestions>
        </div>
      )}

      <AgentLoopPanel
        // New messages already have the chart(s) inlined as ```echarts fences
        // by finalize_node's marker substitution — only fall back to the
        // panel's own render for older messages that never got one.
        chartOption={markdown.includes("```echarts") ? undefined : chartOption}
        chartOptions={markdown.includes("```echarts") ? undefined : chartOptions}
      />

      <ConfirmDialog
        open={confirmDelete}
        title="Delete message"
        description="Delete this message? This cannot be undone."
        confirmText="Delete"
        destructive
        onConfirm={() => {
          setConfirmDelete(false);
          handleDelete();
        }}
        onCancel={() => setConfirmDelete(false)}
      />
    </div>
  );
});

Answer.displayName = "Answer";

// ── ConfidenceCollapsible: final evaluation metrics ──────────────────────────

type ConfidenceLevel = "very_high" | "high" | "medium" | "low" | "none";

const CONFIDENCE_COLORS: Record<ConfidenceLevel, { bar: string; text: string; bg: string; border: string }> = {
  very_high: { bar: "bg-[hsl(var(--confidence-very-high))]", text: "text-[hsl(var(--confidence-very-high))]", bg: "bg-[hsl(var(--confidence-very-high)/10%)]", border: "border-[hsl(var(--confidence-very-high)/30%)]" },
  high:      { bar: "bg-[hsl(var(--confidence-high))]",      text: "text-[hsl(var(--confidence-high))]",      bg: "bg-[hsl(var(--confidence-high)/10%)]",      border: "border-[hsl(var(--confidence-high)/30%)]"     },
  medium:    { bar: "bg-[hsl(var(--confidence-medium))]",    text: "text-[hsl(var(--confidence-medium))]",    bg: "bg-[hsl(var(--confidence-medium)/10%)]",    border: "border-[hsl(var(--confidence-medium)/30%)]"   },
  low:       { bar: "bg-[hsl(var(--confidence-low))]",       text: "text-[hsl(var(--confidence-low))]",       bg: "bg-[hsl(var(--confidence-low)/10%)]",       border: "border-[hsl(var(--confidence-low)/30%)]"      },
  none:      { bar: "bg-[hsl(var(--confidence-none))]",      text: "text-[hsl(var(--confidence-none))]",      bg: "bg-[hsl(var(--confidence-none)/10%)]",       border: "border-[hsl(var(--confidence-none)/30%)]"      },
};

// Obsolete: retry threshold — used by the commented-out retry button below.
// const RETRY_THRESHOLD = 0.4;

// Score quartiles (≤25/≤50/≤75/>75) drive both the bar color and the
// Very Low/Low/High/Very High label, independent of the backend's
// confidence_level (which uses different "medium" semantics).
function getScoreBucket(pct: number): { key: ConfidenceLevel; label: string } {
  if (pct <= 25) return { key: "low", label: "Very Low" };
  if (pct <= 50) return { key: "medium", label: "Low" };
  if (pct <= 75) return { key: "high", label: "High" };
  return { key: "very_high", label: "Very High" };
}

const ConfidenceCollapsible: FC<{
  level?: ConfidenceLevel;
  score?: number;
  suggestion?: string | null;
  finalConfidence?: number;
  finalConfidenceLevel?: ConfidenceLevel;
  faithfulness?: number;
  completeness?: number;
  retrievalScore?: number;
  failedLegs?: string[];
}> = ({
  score,
  suggestion,
  finalConfidence,
  faithfulness,
  completeness,
  retrievalScore,
  failedLegs,
}) => {
  const [open, setOpen] = useState(false);

  const displayConfidence = finalConfidence !== undefined ? finalConfidence : (score !== undefined ? score / 100 : 0);
  const displayPct = Math.min(100, Math.max(0, Math.round(displayConfidence * 100)));
  const { key: displayLevel, label } = getScoreBucket(displayPct);
  const cfg = CONFIDENCE_COLORS[displayLevel];
  // Obsolete: showRetry — used by the commented-out retry button below.
  // const showRetry = finalConfidence !== undefined && finalConfidence < RETRY_THRESHOLD;

  return (
    <div className={`rounded-md border ${cfg.border} ${cfg.bg} text-xs not-prose`}>
      {/* collapsed header */}
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 w-full px-3 py-1.5 text-left"
      >
        {open ? (
          <svg className="h-3 w-3 shrink-0" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M3 4.5L6 7.5L9 4.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        ) : (
          <svg className="h-3 w-3 shrink-0" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M4.5 3L7.5 6L4.5 9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        )}
        <span className={`font-medium shrink-0 ${cfg.text}`}>
          Confidence: {label}{displayPct > 0 ? ` · ${displayPct}/100` : ""}
        </span>
        {/* Obsolete: retry button — no backend retry API exists; the
            onClick only collapsed the panel, which the user can already
            do by clicking the header. Re-enable if a real retry endpoint
            is added.
        {showRetry && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              setOpen(false);
            }}
            className="shrink-0 text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-300 transition-colors"
            title="Retry with relaxed parameters"
          >
            <RefreshCw className="h-3 w-3" />
          </button>
        )} */}
      </button>

      {/* progress bar */}
      <div className="px-3">
        <div className="h-1.5 rounded-full bg-zinc-200 dark:bg-zinc-700 overflow-hidden">
          <div
            className={`h-full rounded-full transition-all ${cfg.bar}`}
            style={{ width: `${displayPct}%` }}
          />
        </div>
      </div>

      {/* expanded body */}
      {open && (
        <div className={`px-3 pb-2 pt-1 border-t ${cfg.border} space-y-2`}>
          {(finalConfidence !== undefined || faithfulness !== undefined || completeness !== undefined) && (
            <div className="space-y-1">
              {retrievalScore !== undefined && (
                <div className="flex justify-between gap-4">
                  <span className="text-zinc-500 dark:text-zinc-400">Retrieval quality</span>
                  <span className={`font-medium ${cfg.text}`}>{retrievalScore}/100</span>
                </div>
              )}
              {faithfulness !== undefined && (
                <div className="flex justify-between gap-4">
                  <span className="text-zinc-500 dark:text-zinc-400">Faithfulness</span>
                  <span className={`font-medium ${cfg.text}`}>{faithfulness}/100</span>
                </div>
              )}
              {completeness !== undefined && (
                <div className="flex justify-between gap-4">
                  <span className="text-zinc-500 dark:text-zinc-400">Completeness</span>
                  <span className={`font-medium ${cfg.text}`}>{completeness}/100</span>
                </div>
              )}
            </div>
          )}
          {failedLegs && failedLegs.length > 0 && (
            <div className="flex items-center gap-1.5 rounded border border-amber-200 dark:border-amber-800 bg-amber-50 dark:bg-amber-900/20 px-2 py-1 text-[10px] text-amber-700 dark:text-amber-400">
              <span>⚠</span>
              <span>Retrieval leg{failedLegs.length > 1 ? "s" : ""} failed: {failedLegs.map(l => l).join(", ")}</span>
            </div>
          )}
          {suggestion && (
            <p className={`${cfg.text} opacity-80`}>{suggestion}</p>
          )}
        </div>
      )}
    </div>
  );
};
