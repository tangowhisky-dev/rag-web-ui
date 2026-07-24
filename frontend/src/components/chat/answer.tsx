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
import { Copy, Trash2, FileText, FileImage, FileType, RefreshCw, Brain } from "lucide-react";
import { AgenticProgress, AgentStepEvent } from "./agentic-progress";
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
const EChartsDiagramDynamic = dynamic(
  () => import("./echarts-diagram"),
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

interface ContextDoc {
  page_content: string;
  metadata: Record<string, any>;
}

interface Citation {
  id: number;
  text: string;
  metadata: Record<string, any>;
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
  knowledge_base: KnowledgeBaseInfo;
}

interface CitationInfo {
  knowledge_base: KnowledgeBaseInfo;
  document: DocumentInfo;
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

// ── Answer Component ─────────────────────────────────────────────────────────

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
  queryClassification?: {
    type: string;
    confidence: number;
    latency_ms: number;
    fallback: boolean;
  };
  toolTrace?: Array<{
    tool_name: string;
    params?: Record<string, unknown>;
    output?: unknown;
    error?: string | null;
    latency_ms: number;
  }>;
  agentSteps?: AgentStepEvent[];
  taskList?: Array<{ id: number; text: string; status: string }>;
  progressMessages?: Array<{ phase: string; message: string; details?: Record<string, unknown> }>;
  synthesisMode?: boolean;
  isStreaming?: boolean;
  onDelete?: (id: string) => void;
  // Final evaluation metrics (from answer_evaluation_node)
  finalConfidence?: number;
  finalConfidenceLevel?: "very_high" | "high" | "medium" | "low" | "none";
  faithfulness?: number;
  completeness?: number;
}> = ({ messageId, chatId, markdown, citations = [], rewrittenQuery, retrievedContext, confidence, confidenceScore, confidenceBreakdown, suggestion, failedLegs, queryClassification, toolTrace, agentSteps, taskList, progressMessages, synthesisMode, isStreaming = false, onDelete, finalConfidence, finalConfidenceLevel, faithfulness, completeness }) => {
  const [citationInfoMap, setCitationInfoMap] = useState<
    Record<string, CitationInfo>
  >({});

  // Debounce citations to prevent rapid API calls during streaming
  const debouncedCitations = useDebouncedValue(citations, 300);

  // Keep refs so CitationLink can read the latest data without changing its
  // identity (avoiding react-markdown remounting all <a> elements every render).
  const citationsRef = useRef(citations);
  const citationInfoMapRef = useRef(citationInfoMap);
  citationsRef.current = citations;
  citationInfoMapRef.current = citationInfoMap;

  // renderKey forces <Markdown> to remount when citations become ready.
  const [renderKey, setRenderKey] = useState(0);
  const wasStreamingRef = useRef(false);
  useEffect(() => {
    if (isStreaming) {
      wasStreamingRef.current = true;
    } else if (wasStreamingRef.current) {
      wasStreamingRef.current = false;
      setRenderKey((k) => k + 1);
    }
  }, [isStreaming]);

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

  const parsedContent = useMemo(() => {
      let thinkContent: string | null = null;
      let isThinkingComplete = false;
      let answerText = markdown;

      // 1. OpenAI/DeepSeek/Qwen HTML-style (full block)
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
        // Open/unclosed
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

      // 2. Gemma channel-style
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
    }, [markdown]);

  useEffect(() => {
    const fetchCitationInfo = async () => {
      const infoMap: Record<string, CitationInfo> = {};

      for (const citation of debouncedCitations) {
        // During streaming, kb_id/document_id are nested in citation.metadata.
        // After reload via API, they are flattened to the top level.
        const top = citation as Record<string, any>;
        const meta = (citation.metadata as Record<string, any>) || {};
        const effectiveKbId = top.kb_id ?? top.kb_id ?? meta.kb_id;
        const effectiveDocId = top.document_id ?? top.document_id ?? meta.document_id;
        if (!effectiveKbId || !effectiveDocId) continue;

        const key = `${effectiveKbId}-${effectiveDocId}`;
        if (infoMap[key]) continue;

        try {
          const [kb, doc] = await Promise.all([
            api.get(`/api/knowledge-base/${effectiveKbId}`),
            api.get(`/api/knowledge-base/${effectiveKbId}/documents/${effectiveDocId}`),
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

      // During streaming, kb_id/document_id are nested in metadata.
      // After reload via API, they are flattened to top level.
      const top = citation as Record<string, any>;
      const meta = (citation.metadata as Record<string, any>) || {};
      const effectiveKbId = top.kb_id ?? meta.kb_id;
      const effectiveDocId = top.document_id ?? meta.document_id;
      const citationInfo =
        citationInfoMapRef.current[
          `${effectiveKbId}-${effectiveDocId}`
        ];

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
                        : citation.retrieval_leg === "sparse"
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
                citation.sparse_rank !== undefined ||
                citation.exact_rank !== undefined) && (
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
              )}
              <Divider />
              <div className="text-foreground leading-relaxed prose prose-sm dark:prose-invert max-w-none">
                <Markdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeHighlight, [rehypeKatex, { throwOnError: false }]]}>
                  {cleanChunkText(citation.text)}
                </Markdown>
              </div>
              <Divider />
              {['kb_id', 'data_store_id', 'document_id', 'chunk_index', '_legs', '_reranker_score', 'qdrant_point_id']
                .some(k => k in citation && (citation as any)[k] !== undefined && (citation as any)[k] !== null) && (
                <details className="text-xs group">
                  <summary className="cursor-pointer select-none list-none flex items-center gap-1 text-muted-foreground hover:text-foreground transition-colors py-0.5">
                    <svg className="h-3 w-3 transition-transform group-open:rotate-90 shrink-0" viewBox="0 0 12 12" fill="currentColor">
                      <path d="M4.5 2 L9 6 L4.5 10" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                    Debug Info
                  </summary>
                  <div className="mt-1.5 bg-muted text-muted-foreground p-2 rounded space-y-1">
                    {['kb_id', 'data_store_id', 'document_id', 'chunk_index', '_legs', '_reranker_score', 'qdrant_point_id']
                      .filter(k => k in citation && (citation as any)[k] !== undefined && (citation as any)[k] !== null)
                      .map(key => {
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
                              <span className="text-foreground/80 break-all">{raw.map(l => legNames[l] ?? l).join(", ")}</span>
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

  // Normalize model citation output: the model sometimes outputs
  // `[citation](N)` or `[citation](N)(N)` instead of `[N](N)`.
  // This ensures react-markdown renders a proper link with the number as text,
  // and removes the duplicate `(N)` plain-text suffix.
  const normalizedMarkdown = useMemo(() => {
    let text = parsedContent.answerText;
    // Case 1: [citation](N)(N) — replace the whole pattern with [N](N)
    text = text.replace(/\[citation\]\((\d+)\)\((\d+)\)/g, "[$1]($1)");
    // Case 2: [citation](N) — replace with [N](N)
    text = text.replace(/\[citation\]\((\d+)\)/g, "[$1]($1)");
    return text;
  }, [parsedContent.answerText]);

  return (
    <div className="prose prose-sm max-w-full">
      {/* Subtask checklist — shown during streaming for complex multi-part queries */}
      {taskList && taskList.length > 1 && (
        <SubtaskList tasks={taskList as SubtaskItem[]} />
      )}
      {/* Agentic progress — transient, grey, fades between phases */}
      <AgenticProgress agentSteps={filteredAgentSteps} isStreaming={isStreaming} />

      {/* Confidence warning (no confidence) */}
      {confidence === "none" && suggestion && (
        <div className="flex items-start gap-2 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800 mb-2">
          <span className="mt-0.5 shrink-0">⚠</span>
          <span>{suggestion}</span>
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
          key={`${citations.length > 0 ? "with-citations" : "no-citations"}-${renderKey}`}
          remarkPlugins={[remarkGfm, remarkMath]}
          rehypePlugins={[rehypeHighlight, [rehypeKatex, { throwOnError: false }]]}
          components={markdownComponents}
        >
          {normalizedMarkdown}
        </Markdown>
      )}
      
      {isStreaming && (
        <span className="inline-block w-2 h-4 ml-0.5 align-middle bg-foreground/80 animate-pulse" aria-hidden="true" />
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
            {citations.length > 0 && !isStreaming && (
              <button
                onClick={() => setRenderKey((k) => k + 1)}
                title="Refresh citations"
                className="flex items-center gap-1 px-2 py-1 rounded text-xs text-zinc-500 hover:text-zinc-800 hover:bg-zinc-100 dark:hover:text-zinc-200 dark:hover:bg-zinc-800 transition-colors"
              >
                <RefreshCw className="h-3.5 w-3.5" />
                <span>Citations</span>
              </button>
            )}
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
              onClick={handleDelete}
              title="Delete message"
              className="flex items-center gap-1 px-2 py-1 rounded text-xs text-zinc-500 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors"
            >
              <Trash2 className="h-3.5 w-3.5" />
              <span>Delete</span>
            </button>
          </div>

          {/* Right: confidence + speed */}
          <div className="flex-1 min-w-0 max-w-xs flex flex-col gap-1.5">
            {finalConfidence !== undefined && (
              <ConfidenceCollapsible
                level={finalConfidenceLevel}
                score={confidenceScore}
                suggestion={suggestion}
                finalConfidence={finalConfidence}
                finalConfidenceLevel={finalConfidenceLevel}
                faithfulness={faithfulness}
                completeness={completeness}
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
    </div>
  );
};

// ── SubtaskList: live TODO checklist for complex multi-subtask queries ────────

interface SubtaskItem {
  id: number;
  text: string;
  status: "pending" | "active" | "done";
}

const SubtaskList: FC<{ tasks: SubtaskItem[] }> = ({ tasks }) => {
  if (!tasks.length) return null;

  return (
    <div className="not-prose mb-0 px-0 space-y-0">
      <p className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground mb-0">
        Subtasks
      </p>
      {tasks.map((task) => (
        <div key={task.id} className="flex items-start gap-2">
          <span className="mt-0.5 shrink-0 w-3.5 h-3.5 flex items-center justify-center">
            {task.status === "done" ? (
              <svg className="text-emerald-500" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg" width="12" height="12">
                <circle cx="6" cy="6" r="5.5" fill="currentColor" fillOpacity="0.15" stroke="currentColor" strokeWidth="1"/>
                <path d="M3.5 6l1.8 1.8L8.5 4.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            ) : task.status === "active" ? (
              <svg className="text-primary animate-spin" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg" width="12" height="12">
                <circle cx="6" cy="6" r="4.5" stroke="currentColor" strokeOpacity="0.2" strokeWidth="1.5"/>
                <path d="M6 1.5A4.5 4.5 0 0 1 10.5 6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
              </svg>
            ) : (
              <svg className="text-muted-foreground/40" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg" width="12" height="12">
                <circle cx="6" cy="6" r="5.5" stroke="currentColor" strokeWidth="1"/>
              </svg>
            )}
          </span>
          <span className={`text-[12px] leading-snug ${
            task.status === "done"
              ? "text-muted-foreground line-through decoration-muted-foreground/40"
              : task.status === "active"
              ? "text-foreground font-medium"
              : "text-muted-foreground"
          }`}>
            {task.text}
          </span>
        </div>
      ))}
    </div>
  );
};

// ── ThinkBlock: reasoning model chain-of-thought ─────────────────────────────

const ThinkBlock: FC<{ content: string; isComplete: boolean }> = ({
  content,
  isComplete,
}) => {
  const [isExpanded, setIsExpanded] = useState(!isComplete);
  const [elapsedMs, setElapsedMs] = useState(0);
  const startTimeRef = useRef<number>(Date.now());
  const finalMsRef = useRef<number | null>(null);
  const contentRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isComplete) {
      if (finalMsRef.current === null) {
        finalMsRef.current = Date.now() - startTimeRef.current;
      }
      const timer = setTimeout(() => setIsExpanded(false), 1500);
      return () => clearTimeout(timer);
    }
    const interval = setInterval(() => {
      setElapsedMs(Date.now() - startTimeRef.current);
    }, 100);
    return () => clearInterval(interval);
  }, [isComplete]);

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
          <svg className="h-3 w-3 text-gray-400 shrink-0" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M3 4.5L6 7.5L9 4.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        ) : (
          <svg className="h-3 w-3 text-gray-400 shrink-0" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M4.5 3L7.5 6L4.5 9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
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

// ── ConfidenceCollapsible: final evaluation metrics ──────────────────────────

type ConfidenceLevel = "very_high" | "high" | "medium" | "low" | "none";

const CONFIDENCE_CONFIG: Record<ConfidenceLevel, {
  steps: number;
  label: string;
  stepColor: string;
  textColor: string;
  bgColor: string;
  borderColor: string;
}> = {
  very_high: { steps: 4, label: "Very High",  stepColor: "bg-[hsl(var(--confidence-very-high))]", textColor: "text-[hsl(var(--confidence-very-high))]", bgColor: "bg-[hsl(var(--confidence-very-high)/10%)]",  borderColor: "border-[hsl(var(--confidence-very-high)/30%)]" },
  high:      { steps: 3, label: "High",       stepColor: "bg-[hsl(var(--confidence-high))]",      textColor: "text-[hsl(var(--confidence-high))]",      bgColor: "bg-[hsl(var(--confidence-high)/10%)]",      borderColor: "border-[hsl(var(--confidence-high)/30%)]"     },
  medium:    { steps: 2, label: "Medium",     stepColor: "bg-[hsl(var(--confidence-medium))]",    textColor: "text-[hsl(var(--confidence-medium))]",    bgColor: "bg-[hsl(var(--confidence-medium)/10%)]",    borderColor: "border-[hsl(var(--confidence-medium)/30%)]"   },
  low:       { steps: 1, label: "Low",        stepColor: "bg-[hsl(var(--confidence-low))]",       textColor: "text-[hsl(var(--confidence-low))]",       bgColor: "bg-[hsl(var(--confidence-low)/10%)]",       borderColor: "border-[hsl(var(--confidence-low)/30%)]"      },
  none:      { steps: 0, label: "None",       stepColor: "bg-[hsl(var(--confidence-none))]",      textColor: "text-[hsl(var(--confidence-none))]",      bgColor: "bg-[hsl(var(--confidence-none)/10%)]",      borderColor: "border-[hsl(var(--confidence-none)/30%)]"      },
};

const CONFIDENCE_COLORS: Record<ConfidenceLevel, { bar: string; text: string; bg: string; border: string }> = {
  very_high: { bar: "bg-[hsl(var(--confidence-very-high))]", text: "text-[hsl(var(--confidence-very-high))]", bg: "bg-[hsl(var(--confidence-very-high)/10%)]", border: "border-[hsl(var(--confidence-very-high)/30%)]" },
  high:      { bar: "bg-[hsl(var(--confidence-high))]",      text: "text-[hsl(var(--confidence-high))]",      bg: "bg-[hsl(var(--confidence-high)/10%)]",      border: "border-[hsl(var(--confidence-high)/30%)]"     },
  medium:    { bar: "bg-[hsl(var(--confidence-medium))]",    text: "text-[hsl(var(--confidence-medium))]",    bg: "bg-[hsl(var(--confidence-medium)/10%)]",    border: "border-[hsl(var(--confidence-medium)/30%)]"   },
  low:       { bar: "bg-[hsl(var(--confidence-low))]",       text: "text-[hsl(var(--confidence-low))]",       bg: "bg-[hsl(var(--confidence-low)/10%)]",       border: "border-[hsl(var(--confidence-low)/30%)]"      },
  none:      { bar: "bg-[hsl(var(--confidence-none))]",      text: "text-[hsl(var(--confidence-none))]",      bg: "bg-[hsl(var(--confidence-none)/10%)]",       border: "border-[hsl(var(--confidence-none)/30%)]"      },
};

const RETRY_THRESHOLD = 0.4;

const ConfidenceCollapsible: FC<{
  level?: ConfidenceLevel;
  score?: number;
  suggestion?: string | null;
  finalConfidence?: number;
  finalConfidenceLevel?: ConfidenceLevel;
  faithfulness?: number;
  completeness?: number;
  failedLegs?: string[];
}> = ({
  level,
  score,
  suggestion,
  finalConfidence,
  finalConfidenceLevel,
  faithfulness,
  completeness,
  failedLegs,
}) => {
  const [open, setOpen] = useState(false);

  const displayConfidence = finalConfidence !== undefined ? finalConfidence : (score !== undefined ? score / 100 : 0);
  const displayLevel = finalConfidenceLevel ?? level ?? "medium";
  const displayPct = Math.min(100, Math.max(0, Math.round(displayConfidence * 100)));
  const cfg = CONFIDENCE_COLORS[displayLevel];
  const label = CONFIDENCE_CONFIG[displayLevel].label;
  const showRetry = finalConfidence !== undefined && finalConfidence < RETRY_THRESHOLD;

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
        )}
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
              {score !== undefined && (
                <div className="flex justify-between gap-4">
                  <span className="text-zinc-500 dark:text-zinc-400">Retrieval confidence</span>
                  <span className={`font-medium ${cfg.text}`}>{score}/100</span>
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
