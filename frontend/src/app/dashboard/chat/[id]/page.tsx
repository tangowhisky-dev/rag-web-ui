"use client";

// crypto.randomUUID() is only available in secure contexts (HTTPS).
// Fall back to a manual UUID v4 implementation for HTTP (local dev).
function generateId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
  });
}

import { useEffect, useLayoutEffect, useRef, useState, useMemo, useCallback, use } from "react";

import { useRouter } from "next/navigation";
import Image from "next/image";
import { Copy, Check, Trash2, ChevronDown } from "lucide-react";
import { useChatContext } from "@/contexts/chat-context";
import { api, ApiError, handleAuthRedirect } from "@/lib/api";
import { APP_LOGO_SRC } from "@/lib/app-config";
import { cancelStream } from "@/lib/cancel-stream";
import { copyToClipboard } from "@/lib/utils";
import { useToast } from "@/components/ui/use-toast";
import { Answer } from "@/components/chat/answer";
import { InputBar } from "@/components/chat/chat-input";
import { MessageFileChip, type UploadedFile } from "@/components/chat/file-attachment";
import { BranchPicker } from "@/components/chat/branch-picker";
import ClarificationDialog from "@/components/chat/clarification-dialog";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { LoadingDots } from "@/components/ui/loading-dots";

interface AgentStep {
  node: string;
  latency_ms: number;
  status: string;
  // optional per-node detail fields emitted by the backend
  [key: string]: unknown;
}

interface Message {
  id: string;
  // Stable identity for React keys — unlike `id`, this never changes even
  // when `id` is swapped from a client UUID to the persisted DB id on the
  // "done" event (that swap previously caused the whole answer, including
  // any embedded chart, to remount once confidence scoring finished).
  clientId: string;
  role: "assistant" | "user" | "system" | "data";
  content: string;
  citations?: Citation[];
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
  synthesisMode?: boolean;
  agentSteps?: AgentStep[];
  file_name?: string;  // filename of attached chat file, if any
  file_id?: number;    // chat_files.id — needed for download URL
  // Final evaluation metrics (from answer_evaluation_node)
  finalConfidence?: number;
  finalConfidenceLevel?: "very_high" | "high" | "medium" | "low" | "none";
  retrievalScore?: number;
  faithfulness?: number;
  completeness?: number;
  // Enterprise agent loop per-turn state
  plan?: Record<string, unknown>;
  toolCalls?: Array<Record<string, unknown>>;
  toolObservations?: Array<Record<string, unknown>>;
  lastAnswerObject?: Record<string, unknown>;
  chartOptions?: Array<Record<string, unknown>>;
}

interface ChatMessage {
  id: number;
  content: string;
  role: "assistant" | "user";
  created_at: string;
  confidence_level?: string;
  confidence_score?: number;
  confidence_breakdown?: string;
  final_confidence?: number;
  final_confidence_level?: string;
  faithfulness?: number;
  completeness?: number;
  retrieval_score?: number;
  file_name?: string;
  file_id?: number;
  citations?: Citation[];
}

interface ChatMeta {
  id: number;
  knowledge_bases?: Array<{ id: number; name: string }>;
  [key: string]: unknown;
}

interface Citation {
  id: number;
  text: string;
  metadata: Record<string, unknown>;
}

function ChatPageInner({ params }: { params: { id: string } }) {
  const router = useRouter();
  const abortControllerRef = useRef<AbortController | null>(null);
  const { toast } = useToast();
  const { setActiveChat, setGraphRagActive, bumpChatToTop } = useChatContext();
  const [isInitialLoad, setIsInitialLoad] = useState(true);
  const [allKbs, setAllKbs] = useState<Array<{ id: number; name: string }>>([]);
  const [associatedKbIds, setAssociatedKbIds] = useState<number[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<UploadedFile | null>(null);
  const [fileError, setFileError] = useState<string>("");
  const pollRefs = useRef<Set<ReturnType<typeof setInterval>>>(new Set());

  // ── Delete message confirmation ───────────────────────────────────────────
  const [confirmDeleteMsgId, setConfirmDeleteMsgId] = useState<string | null>(null);

  // ── Copy feedback for user message bubble ─────────────────────────────────
  const [copiedMsgId, setCopiedMsgId] = useState<string | null>(null);
  const handleCopyUserMessage = useCallback((id: string, content: string) => {
    copyToClipboard(content).then(() => {
      setCopiedMsgId(id);
      setTimeout(() => setCopiedMsgId(null), 1500);
    });
  }, []);

  // ── Clarification state ───────────────────────────────────────────────────
  const [clarificationState, setClarificationState] = useState<{
    question: string;
    options: string[];
    rationale?: string;
    assistantId: string;
    clarificationId: number;
    attempt: number;
    maxAttempts: number;
  } | null>(null);

  // ── Progress & task list state (new agentic agent) ────────────────────────
  const [progressMessages, setProgressMessages] = useState<Array<{
    phase: string;
    message: string;
    details?: Record<string, unknown>;
  }>>([]);

  const [taskList, setTaskList] = useState<Array<{
    id: number;
    text: string;
    status: string;
    progress?: unknown;
  }>>([]);

  const [thinkingContent, setThinkingContent] = useState<{
    content: string;
    done: boolean;
  } | null>(null);

  // ── Pagination state ────────────────────────────────────────────────────────
  const [hasMoreMessages, setHasMoreMessages] = useState(false);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const topSentinelRef = useRef<HTMLDivElement>(null);
  // Stores scrollHeight before prepending older messages so we can restore position
  const pendingScrollAdjustRef = useRef<number | null>(null);
  // Track whether user is actively scrolled to the bottom
  const [isAtBottom, setIsAtBottom] = useState(true);
  // Whether auto-scroll is locked (user scrolled up during generation)
  const [autoScrollLocked, setAutoScrollLocked] = useState(false);

  // Poll file status until ready or error
  const startPolling = (fileId: number) => {
    const interval = setInterval(async () => {
      try {
        const data = await api.get(`/api/chat/${params.id}/files/${fileId}`);
        setUploadedFile((prev) => prev ? { ...prev, ...data } : null);
        if (data.status === "ready" || data.status === "error") {
          clearInterval(interval);
          pollRefs.current.delete(interval);
          if (data.status === "error") {
            setFileError(data.error_message || "File processing failed.");
          }
        }
      } catch { /* ignore */ }
    }, 1200);
    pollRefs.current.add(interval);
  };

  // Upload file immediately on attach
  const handleFileAccepted = async (file: File) => {
    setFileError("");
    const formData = new FormData();
    formData.append("file", file);
    try {
      const data: UploadedFile = await api.post(`/api/chat/${params.id}/files`, formData);
      setUploadedFile(data);
      if (data.status === "processing") startPolling(data.id);
    } catch (err) {
      if (err instanceof ApiError) {
        setFileError(err.message || "Upload failed.");
      } else {
        setFileError("Upload failed. Please try again.");
      }
    }
  };

  const handleFileRemove = async () => {
    if (uploadedFile) {
      api.delete(`/api/chat/${params.id}/files/${uploadedFile.id}`).catch(() => {});
    }
    setUploadedFile(null);
    setFileError("");
    pollRefs.current.forEach((id) => clearInterval(id));
    pollRefs.current.clear();
  };

  useEffect(() => {
    const refs = pollRefs.current;
    return () => {
      refs.forEach((id) => clearInterval(id));
      refs.clear();
    };
  }, []);

  useEffect(() => {
    setActiveChat(Number(params.id));
    // Only reset graphRagActive on unmount — don't clear activeChat.
    // Clearing activeChat causes a brief null state that triggers an
    // extra context re-render of all consumers (including the sidebar)
    // during chat-to-chat navigation. The new page sets activeChat
    // immediately in its effect, so the old value persists harmlessly
    // until then.
    return () => { setGraphRagActive(false); };
  }, [params.id, setActiveChat, setGraphRagActive]);

  // ── Message formatter (shared by initial load and paginated load) ───────────
  const formatMessage = useCallback((msg: ChatMessage): Message => {
    if (msg.role !== "assistant" || !msg.content)
      return {
        id: msg.id.toString(),
        clientId: msg.id.toString(),
        role: msg.role,
        content: msg.content,
        file_name: msg.file_name ?? undefined,
        file_id: msg.file_id ?? undefined,
        citations: msg.citations ?? [],
      };

    // Assistant message — citations come from the API, content is the raw answer text
    return {
      id: msg.id.toString(),
      clientId: msg.id.toString(),
      role: msg.role,
      content: msg.content,
      citations: msg.citations ?? [],
      confidence: msg.confidence_level as Message["confidence"] | undefined,
      confidenceScore: msg.confidence_score ?? undefined,
      confidenceBreakdown: msg.confidence_breakdown
        ? (() => { try { return JSON.parse(msg.confidence_breakdown); } catch { return undefined; } })()
        : undefined,
      finalConfidence: msg.final_confidence ?? undefined,
      finalConfidenceLevel: msg.final_confidence_level as Message["finalConfidenceLevel"] | undefined,
      faithfulness: msg.faithfulness ?? undefined,
      completeness: msg.completeness ?? undefined,
      retrievalScore: msg.retrieval_score ?? undefined,
      file_name: msg.file_name ?? undefined,
      file_id: msg.file_id ?? undefined,
    };
  }, []);

  const fetchChat = useCallback(async () => {
    try {
      // Fetch metadata (no messages), first page, and all accessible KBs in parallel
      const [meta, page, kbList] = await Promise.all([
        api.get(`/api/chat/${params.id}?include_messages=false`),
        api.get(`/api/chat/${params.id}/messages/paginated?limit=20`),
        api.get("/api/knowledge-base"),
      ]);
      const kbs = (meta as ChatMeta).knowledge_bases ?? [];
      setAssociatedKbIds(kbs.map((kb) => kb.id));
      const allKbList = Array.isArray(kbList) ? kbList : (kbList as any).items ?? [];
      setAllKbs(allKbList.map((kb: any) => ({ id: kb.id, name: kb.name })));
      setMessages(page.messages.map(formatMessage));
      setHasMoreMessages(page.has_more);
    } catch (error) {
      console.error("Failed to fetch chat:", error);
      if (error instanceof ApiError) {
        toast({ title: "Error", description: error.message, variant: "destructive" });
      }
      router.push("/dashboard/chat");
    } finally {
      setIsInitialLoad(false);
    }
  }, [params.id, formatMessage, toast, router, setMessages]);

  useEffect(() => {
    if (!isInitialLoad) return;
    let cancelled = false;
    (async () => {
      if (!cancelled) await fetchChat();
    })();
    return () => { cancelled = true; };
  }, [isInitialLoad, fetchChat]);

  // Restore scroll position after prepending older messages (must be synchronous pre-paint)
  useLayoutEffect(() => {
    if (pendingScrollAdjustRef.current !== null && scrollContainerRef.current) {
      const delta = scrollContainerRef.current.scrollHeight - pendingScrollAdjustRef.current;
      scrollContainerRef.current.scrollTop += delta;
      pendingScrollAdjustRef.current = null;
    }
  }, [messages]);

  // Scroll to bottom only on initial load (once messages first arrive)
  useEffect(() => {
    if (!isInitialLoad && messages.length > 0 && scrollContainerRef.current) {
      scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isInitialLoad]);

  // Ref mirror of autoScrollLocked so the scroll listener can read the
  // current value without being re-created on every change.
  const autoScrollLockedRef = useRef(false);
  useEffect(() => { autoScrollLockedRef.current = autoScrollLocked; }, [autoScrollLocked]);

  // Track scroll position and auto-scroll during streaming.
  // Listener is attached once (empty deps) — no churn on every scroll event.
  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container) return;

    const nearBottom = () => {
      const threshold = 80;
      return container.scrollHeight - container.scrollTop - container.clientHeight <= threshold;
    };

    const onScroll = () => {
      const atBottom = nearBottom();
      setIsAtBottom(atBottom);
      if (!atBottom) {
        setAutoScrollLocked(true);
      } else if (atBottom && autoScrollLockedRef.current) {
        setAutoScrollLocked(false);
      }
    };

    container.addEventListener("scroll", onScroll, { passive: true });
    return () => container.removeEventListener("scroll", onScroll);
  }, []);

  // Auto-scroll during streaming while not locked
  useEffect(() => {
    if (!isLoading || autoScrollLocked) return;
    const container = scrollContainerRef.current;
    if (!container) return;
    container.scrollTop = container.scrollHeight;
  }, [messages, isLoading, autoScrollLocked, progressMessages, taskList, thinkingContent]);

  const loadMoreMessages = useCallback(async () => {
    if (isLoadingMore || !hasMoreMessages || !messages.length) return;
    setIsLoadingMore(true);
    // Save scrollHeight before DOM changes so useLayoutEffect can restore position
    pendingScrollAdjustRef.current = scrollContainerRef.current?.scrollHeight ?? null;
    try {
      const oldestId = messages[0].id;
      const page = await api.get(
        `/api/chat/${params.id}/messages/paginated?limit=20&before_id=${oldestId}`
      );
      const older = page.messages.map(formatMessage);
      setMessages((prev) => [...older, ...prev]);
      setHasMoreMessages(page.has_more);
    } catch (e) {
      console.error("Failed to load older messages:", e);
    } finally {
      setIsLoadingMore(false);
    }
  }, [isLoadingMore, hasMoreMessages, messages, params.id, formatMessage, setMessages, setHasMoreMessages, setIsLoadingMore]);

  // IntersectionObserver: trigger load-more when top sentinel enters view
  useEffect(() => {
    const sentinel = topSentinelRef.current;
    const container = scrollContainerRef.current;
    if (!sentinel || !container) return;
    const observer = new IntersectionObserver(
      (entries) => { if (entries[0].isIntersecting) loadMoreMessages(); },
      { root: container, threshold: 0 }
    );
    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [loadMoreMessages]);

  const scrollToBottom = useCallback(() => {
    if (scrollContainerRef.current) {
      scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
    }
  }, []);

  // Stable callback for Answer's onDelete — prevents re-rendering every
  // Answer when the parent re-renders during streaming.
  const handleDeleteMessage = useCallback((id: string) => {
    setMessages((prev) => prev.filter((m) => m.id !== id));
  }, [setMessages]);

  const flushToBrowser = async () => {
    await new Promise<void>((resolve) => {
      requestAnimationFrame(() => {
        // Auto-scroll during streaming only if user is already near the bottom.
        // If they've scrolled up to read older messages, don't hijack their position.
        const container = scrollContainerRef.current;
        if (container) {
          const { scrollTop, scrollHeight, clientHeight } = container;
          if (scrollHeight - scrollTop - clientHeight < 120) {
            container.scrollTop = scrollHeight;
          }
        }
        resolve();
      });
    });
  };

  const appendAssistantChunk = (
    assistantId: string,
    updater: (message: Message) => Message
  ) => {
    setMessages((prev) =>
      prev.map((message) =>
        message.id === assistantId ? updater(message) : message
      )
    );
  };

  const processStreamLine = (line: string, assistantId: string, _userMessageId?: string) => {
    const trimmedLine = line.trim();

    if (!trimmedLine || trimmedLine === "d:[DONE]") {
      return;
    }

    // 2: retrieved context — emitted right after step 2, before LLM starts
    if (trimmedLine.startsWith("2:")) {
      try {
        const payload = JSON.parse(trimmedLine.slice(2)) as {
          docs?: Array<{ page_content: string; metadata: Record<string, any> }>;
          context?: Array<{ page_content: string; metadata: Record<string, any> }>;
          confidence?: string;
          score?: number;
          suggestion?: string | null;
          failed_legs?: string[];
          breakdown?: Record<string, unknown>;
          tool_trace?: Array<{
            tool_name: string;
            params?: Record<string, unknown>;
            output?: unknown;
            error?: string | null;
            latency_ms: number;
          }>;
          synthesis_mode?: boolean;
        };
        // 2: now carries confidence metadata only; citations arrive via r:.
        // Normalize confidence level: backend sends uppercase (HIGH/MEDIUM/LOW),
        // frontend type uses lowercase. Map backend values → frontend enum.
        const rawConfidence = payload.confidence?.toLowerCase() as Message["confidence"] | undefined;
        appendAssistantChunk(assistantId, (message) => ({
          ...message,
          confidence: rawConfidence,
          confidenceScore: payload.score,
          confidenceBreakdown: payload.breakdown,
          suggestion: payload.suggestion,
          failedLegs: payload.failed_legs,
          toolTrace: payload.tool_trace,
          synthesisMode: payload.synthesis_mode,
        }));
      } catch (e) {
        console.error("Failed to parse context event:", e);
      }
      return;
    }

    if (trimmedLine.startsWith("0:")) {
      try {
        const payload = JSON.parse(trimmedLine.slice(2)) as string;

        if (payload.includes("__LLM_RESPONSE__")) {
          const [, initialResponseText] = payload.split("__LLM_RESPONSE__");
          appendAssistantChunk(assistantId, (message) => ({
            ...message,
            content: initialResponseText || message.content,
          }));
          return;
        }

        appendAssistantChunk(assistantId, (message) => ({
          ...message,
          content: `${message.content}${payload}`,
        }));
      } catch (error) {
        console.error("Failed to parse stream line:", trimmedLine, error);
      }
      return;
    }

    // 4: agent_step — LangGraph node start/finish events for AgenticProgress
    if (trimmedLine.startsWith("4:")) {
      try {
        const step = JSON.parse(trimmedLine.slice(2)) as AgentStep;
        appendAssistantChunk(assistantId, (message) => ({
          ...message,
          agentSteps: [...(message.agentSteps ?? []), step],
        }));
      } catch (e) {
        console.error("Failed to parse agent_step event:", e);
      }
      return;
    }

    // r: answer_rewrite — citation-normalised full answer + cited docs replaces streamed tokens
    if (trimmedLine.startsWith("r:")) {
      try {
        const payload = JSON.parse(trimmedLine.slice(2)) as {
          content: string;
          citations?: Array<{ page_content: string; metadata: Record<string, any> }>;
        };
        const citations: Citation[] = (payload.citations ?? []).map((doc, index) => {
          // Flatten metadata fields to top-level so CitationLink can read
          // them directly (works during streaming AND on reload via API).
          const c: Record<string, any> = {
            id: index + 1,
            text: doc.page_content,
            metadata: doc.metadata,
          };
          if (doc.metadata) {
            Object.keys(doc.metadata).forEach((k) => {
              c[k] = doc.metadata[k];
            });
          }
          return c as Citation;
        });
        appendAssistantChunk(assistantId, (message) => ({
          ...message,
          content: payload.content,
          citations,
        }));
      } catch (e) {
        console.error("Failed to parse answer_rewrite event:", e);
      }
      return;
    }

    // p: progress — transient status messages (new agentic agent)
    if (trimmedLine.startsWith("p:")) {
      try {
        const payload = JSON.parse(trimmedLine.slice(2)) as {
          phase: string;
          message: string;
          details?: Record<string, unknown>;
        };
        setProgressMessages((prev) => [...prev, payload]);
      } catch (e) {
        console.error("Failed to parse progress event:", e);
      }
      return;
    }

    // t: task_list — subtask list with status (new agentic agent)
    if (trimmedLine.startsWith("t:")) {
      try {
        const payload = JSON.parse(trimmedLine.slice(2)) as {
          tasks: Array<{
            id: number;
            text: string;
            status: string;
            progress?: unknown;
          }>;
        };
        setTaskList(payload.tasks);
      } catch (e) {
        console.error("Failed to parse task_list event:", e);
      }
      return;
    }

    // th: thinking — chain-of-thought from reasoning models (new agentic agent)
    if (trimmedLine.startsWith("th:")) {
      try {
        const payload = JSON.parse(trimmedLine.slice(3)) as {
          content: string;
          done: boolean;
        };
        setThinkingContent(payload);
      } catch (e) {
        console.error("Failed to parse thinking event:", e);
      }
      return;
    }

    // pl: plan — enterprise agent subtask plan
    if (trimmedLine.startsWith("pl:")) {
      try {
        const payload = JSON.parse(trimmedLine.slice(3)) as { plan?: Record<string, unknown> };
        appendAssistantChunk(assistantId, (message) => ({
          ...message,
          plan: payload.plan,
        }));
      } catch (e) {
        console.error("Failed to parse plan event:", e);
      }
      return;
    }

    // tc: tool_call — enterprise agent tool invocation
    if (trimmedLine.startsWith("tc:")) {
      try {
        const payload = JSON.parse(trimmedLine.slice(3)) as Record<string, unknown>;
        appendAssistantChunk(assistantId, (message) => ({
          ...message,
          toolCalls: [...(message.toolCalls ?? []), payload],
        }));
      } catch (e) {
        console.error("Failed to parse tool_call event:", e);
      }
      return;
    }

    // to: tool_observation — enterprise agent tool result
    if (trimmedLine.startsWith("to:")) {
      try {
        const payload = JSON.parse(trimmedLine.slice(3)) as Record<string, unknown>;
        appendAssistantChunk(assistantId, (message) => ({
          ...message,
          toolObservations: [...(message.toolObservations ?? []), payload],
        }));
      } catch (e) {
        console.error("Failed to parse tool_observation event:", e);
      }
      return;
    }

    // tr: tool_retry — tool call failed, being retried
    if (trimmedLine.startsWith("tr:")) {
      try {
        const payload = JSON.parse(trimmedLine.slice(3)) as {
          tool: string;
          attempt: number;
          max_retries: number;
          success: boolean;
          error?: string;
        };
        setProgressMessages((prev) => [
          ...prev,
          {
            phase: "tool_retry",
            message: `Retrying ${payload.tool} (attempt ${payload.attempt}/${payload.max_retries})`,
            details: payload,
          },
        ]);
      } catch (e) {
        console.error("Failed to parse tool_retry event:", e);
      }
      return;
    }

    // la: last_answer — enterprise agent structured summary + chart options
    if (trimmedLine.startsWith("la:")) {
      try {
        const payload = JSON.parse(trimmedLine.slice(3)) as { last_answer_object?: Record<string, unknown> };
        appendAssistantChunk(assistantId, (message) => ({
          ...message,
          lastAnswerObject: payload.last_answer_object,
          chartOptions: payload.last_answer_object?.chart_options as Array<Record<string, unknown>> | undefined,
        }));
      } catch (e) {
        console.error("Failed to parse last_answer event:", e);
      }
      return;
    }

    if (trimmedLine.startsWith("3:")) {
      const errorMessage = trimmedLine.slice(2);
      throw new Error(errorMessage || "Streaming request failed");
    }

    // d: done — replace client-side UUID with server-side integer messageId
    if (trimmedLine.startsWith("d:")) {
      try {
        const payload = JSON.parse(trimmedLine.slice(2)) as {
          messageId?: number;
          userMessageId?: number;
          usage?: {
            final_confidence?: number;
            confidence_level?: string;
            faithfulness?: number;
            completeness?: number;
            retrieval_score?: number;
          };
        };
        const usage = payload.usage;
        setMessages((prev) => {
          // Find the user message preceding this assistant message
          const assistantIdx = prev.findIndex((m) => m.id === assistantId);
          const userId = assistantIdx > 0 ? prev[assistantIdx - 1].id : null;

          return prev.map((message) => {
            // Update assistant message ID
            if (message.id === assistantId) {
              return {
                ...message,
                id: payload.messageId?.toString() ?? message.id,
                finalConfidence: usage?.final_confidence,
                finalConfidenceLevel: usage?.confidence_level as Message["finalConfidenceLevel"],
                faithfulness: usage?.faithfulness,
                completeness: usage?.completeness,
                retrievalScore: usage?.retrieval_score,
              };
            }
            // Update preceding user message ID
            if (userId && message.id === userId && payload.userMessageId) {
              return { ...message, id: payload.userMessageId.toString() };
            }
            return message;
          });
        });
      } catch (e) {
        console.error("Failed to parse done event:", e);
      }
      return;
    }

    // c: clarification — agent needs user to clarify their query
    if (trimmedLine.startsWith("c:")) {
      try {
        const payload = JSON.parse(trimmedLine.slice(2)) as {
          question: string;
          options: string[];
          rationale?: string;
          attempt?: number;
          max_attempts?: number;
          clarification_id?: number;
        };
        setClarificationState({
          question: payload.question,
          options: payload.options || [],
          rationale: payload.rationale,
          assistantId,
          clarificationId: payload.clarification_id ?? 0,
          attempt: payload.attempt || 1,
          maxAttempts: payload.max_attempts || 2,
        });
        // Pause loading state while waiting for clarification
        setIsLoading(false);
      } catch (e) {
        console.error("Failed to parse clarification event:", e);
      }
      return;
    }

    // C: clarification_ready — user responded, agent resuming
    if (trimmedLine.startsWith("C:")) {
      try {
        const payload = JSON.parse(trimmedLine.slice(2)) as { resumed: boolean };
        if (payload.resumed) {
          setClarificationState(null);
          setIsLoading(true);
        }
      } catch (e) {
        console.error("Failed to parse clarification_ready event:", e);
      }
      return;
    }
  };

  /** Core SSE streaming: POST to /messages and pipe events into the given assistantId slot. */
  const streamFromMessages = async (
    requestMessages: Array<{ role: string; content: string }>,
    assistantId: string,
    fileId?: number,
    parentMessageId?: string,
    userMessageId?: string,
  ) => {
    // Reset transient state for new query
    setProgressMessages([]);
    setTaskList([]);
    setThinkingContent(null);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    const response = await fetch(`/api/chat/${params.id}/messages`, {
      method: "POST",
      credentials: "include",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        messages: requestMessages,
        ...(fileId ? { file_id: fileId } : {}),
        ...(parentMessageId ? { parent_message_id: parentMessageId } : {}),
      }),
      signal: controller.signal,
    });

    if (await handleAuthRedirect(response)) {
      return;
    }

    if (!response.ok || !response.body) {
      throw new Error(`Request failed with status ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        const t = line.trim();
        if (t) {
          processStreamLine(line, assistantId, userMessageId);
        }
      }
      // Always flush so UI updates progressively between agent steps
      await flushToBrowser();
    }

    if (buffer.trim()) {
      processStreamLine(buffer, assistantId, userMessageId);
      await flushToBrowser();
    }
  };

  const handleSubmit = async (overrideInput?: string) => {

    const trimmedInput = (typeof overrideInput === "string" ? overrideInput : input).trim();
    if (!trimmedInput || isLoading) {
      return;
    }

    const assistantId = generateId();
    const userId = generateId();
    const userMessage: Message = {
      id: userId,
      clientId: userId,
      role: "user",
      content: trimmedInput,
      ...(uploadedFile ? { file_name: uploadedFile.file_name, file_id: uploadedFile.id } : {}),
    };
    const assistantMessage: Message = {
      id: assistantId,
      clientId: assistantId,
      role: "assistant",
      content: "",
      citations: [],
    };

    // Bump this chat to the top of the sidebar (unpinned section)
    bumpChatToTop(Number(params.id));

    const requestMessages = messages
      .filter((message) => message.role === "user" || message.role === "assistant")
      .map((message) => ({
        role: message.role,
        content: message.content,
      }))
      .concat({
        role: "user" as const,
        content: trimmedInput,
      });

    setInput("");
    const sentFile = uploadedFile;
    setUploadedFile(null);
    setFileError("");
    setIsLoading(true);
    setAutoScrollLocked(false);
    setIsAtBottom(true);
    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setTimeout(scrollToBottom, 0);

    try {
      await streamFromMessages(requestMessages, assistantId, sentFile?.id, undefined, userId);
    } catch (error) {
      // AbortError is intentional (user clicked stop) — drop the placeholder, no toast
      if (error instanceof Error && error.name === "AbortError") {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, content: m.content || "*(generation stopped)*" }
              : m
          )
        );
      } else {
        console.error("Failed to stream chat:", error);
        setMessages((prev) => prev.filter((message) => message.id !== assistantId));
        toast({
          title: "Error",
          description:
            error instanceof Error ? error.message : "Failed to send message",
          variant: "destructive",
        });
      }
    } finally {
      abortControllerRef.current = null;
      setIsLoading(false);
    }
  };

  /** Abort the in-flight stream request and notify the server to cancel. */
  const handleStop = () => {
    // Best-effort: notify server to cancel, then always abort the client-side stream.
    cancelStream(params.id).catch(() => {});
    abortControllerRef.current?.abort();
  };

  /** Called by BranchPicker when the user saves an edit and a new branch message is created. */
  const handleBranch = async (originalId: string, newMessageId: string, newContent: string) => {
    if (isLoading) return;

    // Build request messages up to and including the newly-branched user message
    // (read from current state before mutating it)
    const idx = messages.findIndex((m) => m.id === originalId);
    const requestMessages = messages
      .slice(0, idx + 1)
      .filter((m) => m.role === "user" || m.role === "assistant")
      .map((m) => ({
        role: m.role,
        content: m.id === originalId ? newContent : m.content,
      }));

    const newAssistantId = generateId();

    // Swap user message + remove old assistant reply + insert new assistant placeholder
    setMessages((prev) => {
      const updated = prev.map((m) =>
        m.id === originalId ? { ...m, id: newMessageId, content: newContent } : m
      );
      const userIdx = updated.findIndex((m) => m.id === newMessageId);
      const result = [...updated];
      // Remove following assistant message if present
      if (userIdx >= 0 && userIdx + 1 < result.length && result[userIdx + 1].role === "assistant") {
        result.splice(userIdx + 1, 1);
      }
      // Insert new assistant placeholder right after user message
      result.splice(userIdx + 1, 0, {
        id: newAssistantId,
        clientId: newAssistantId,
        role: "assistant" as const,
        content: "",
        citations: [],
      });
      return result;
    });

    setIsLoading(true);

    // Persist the active branch so reload picks the new branch
    api.put(`/api/chat/${params.id}/active-branch`, {
      parent_message_id: parseInt(originalId),
      selected_message_id: parseInt(newMessageId),
    }).catch(() => {});

    try {
      await streamFromMessages(requestMessages, newAssistantId, undefined, newMessageId);
    } catch (error) {
      console.error("Failed to stream after branch:", error);
      setMessages((prev) => prev.filter((m) => m.id !== newAssistantId));
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to regenerate response",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  /** Called by BranchPicker when the user navigates to a sibling branch.
   *  Swaps both the user message and its paired assistant reply. */
  const handleNavigate = (
    targetUserMsg: { id: string; content: string },
    targetAssistantMsg: { id: string; content: string; [key: string]: unknown } | null,
    currentUserMsgId: string,
    parentMessageId: string,
  ) => {
    setMessages((prev) => {
      const userIdx = prev.findIndex((m) => m.id === currentUserMsgId);
      if (userIdx < 0) return prev;

      const updated = prev.map((m) =>
        m.id === currentUserMsgId
          ? { ...m, id: targetUserMsg.id, content: targetUserMsg.content }
          : m
      );

      // Find the assistant message right after the user message and swap it
      const result = [...updated];
      const newUserIdx = result.findIndex((m) => m.id === targetUserMsg.id);
      if (newUserIdx >= 0 && newUserIdx + 1 < result.length && result[newUserIdx + 1].role === "assistant") {
        if (targetAssistantMsg) {
          result[newUserIdx + 1] = {
            ...result[newUserIdx + 1],
            id: targetAssistantMsg.id,
            clientId: targetAssistantMsg.id,
            content: targetAssistantMsg.content,
            citations: (targetAssistantMsg.citations as Message["citations"]) ?? [],
            confidence: targetAssistantMsg.confidence_level as Message["confidence"] | undefined,
            confidenceScore: targetAssistantMsg.confidence_score as number | undefined,
            confidenceBreakdown: targetAssistantMsg.confidence_breakdown
              ? typeof targetAssistantMsg.confidence_breakdown === "string"
                ? JSON.parse(targetAssistantMsg.confidence_breakdown as string)
                : targetAssistantMsg.confidence_breakdown
              : undefined,
            finalConfidence: targetAssistantMsg.final_confidence as number | undefined,
            finalConfidenceLevel: targetAssistantMsg.final_confidence_level as Message["finalConfidenceLevel"] | undefined,
            faithfulness: targetAssistantMsg.faithfulness as number | undefined,
            completeness: targetAssistantMsg.completeness as number | undefined,
            retrievalScore: targetAssistantMsg.retrieval_score as number | undefined,
            // Clear streaming-only fields from the previous branch
            agentSteps: undefined,
            toolCalls: undefined,
            toolObservations: undefined,
            toolTrace: undefined,
            plan: undefined,
            synthesisMode: undefined,
            suggestion: undefined,
            failedLegs: undefined,
            lastAnswerObject: undefined,
            chartOptions: undefined,
          };
        } else {
          // No assistant reply for this branch — show placeholder
          result[newUserIdx + 1] = {
            ...result[newUserIdx + 1],
            content: "",
            citations: [],
          };
        }
      }
      return result;
    });

    // Persist the active branch selection so reload picks the right branch
    api.put(`/api/chat/${params.id}/active-branch`, {
      parent_message_id: parseInt(parentMessageId),
      selected_message_id: parseInt(targetUserMsg.id),
    }).catch((e) => console.error("[NAVIGATE] PUT failed", e));
  };

  /** Handle user's clarification response */
  const handleClarificationResponse = async (response: string) => {
    if (!clarificationState) return;

    const { assistantId, clarificationId } = clarificationState;

    // Add user's clarification as a message
    const clarificationId2 = generateId();
    const clarificationMessage: Message = {
      id: clarificationId2,
      clientId: clarificationId2,
      role: "user",
      content: response,
    };

    setMessages((prev) => [...prev, clarificationMessage]);
    setClarificationState(null);

    // Reset transient state — prevents stale progress/task/thinking messages
    // from the initial stream from persisting into the resumed stream.
    setProgressMessages([]);
    setTaskList([]);
    setThinkingContent(null);
    setIsLoading(true);

    // Send clarification to backend and pipe the resumed SSE stream.
    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    try {
      const res = await fetch(`/api/chat/clarification`, {
        method: "POST",
        credentials: "include",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          chat_id: Number(params.id),
          clarification_id: clarificationId,
          response: response,
        }),
        signal: abortController.signal,
      });

      if (await handleAuthRedirect(res)) {
        return;
      }

      if (!res.ok || !res.body) {
        throw new Error(`Clarification submission failed: ${res.status}`);
      }

      // Pipe the resumed SSE stream through processStreamLine
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.trim()) {
            processStreamLine(line, assistantId);
          }
        }
        await flushToBrowser();
      }

      if (buffer.trim()) {
        processStreamLine(buffer, assistantId);
        await flushToBrowser();
      }
    } catch (error) {
      console.error("Failed to send clarification:", error);
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to send clarification",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  /** Handle user skipping clarification */
  const handleClarificationSkip = () => {
    if (!clarificationState) return;
    // Cancel the in-flight clarification stream if one is running.
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    setClarificationState(null);
    setIsLoading(false);
  };

  const lastAssistantId = useMemo(() => {
    let last: string | undefined;
    for (const m of messages) {
      if (m.role === "assistant") last = m.id;
    }
    return last;
  }, [messages]);

  const handleFollowUp = (query: string) => {
    handleSubmit(query);
  };

  const [kbToggling, setKbToggling] = useState(false);
  const kbPatchInFlight = useRef<Promise<unknown> | null>(null);

  const handleKbToggle = useCallback(async (kbId: number) => {
    if (kbPatchInFlight.current) return;  // block while in-flight
    const prev = associatedKbIds;
    const isAssociated = prev.includes(kbId);
    if (isAssociated && prev.length <= 1) return;  // don't remove the last KB
    const next = isAssociated
      ? prev.filter((id) => id !== kbId)
      : [...prev, kbId];
    setAssociatedKbIds(next);
    setKbToggling(true);
    const patch = api.patch(`/api/chat/${params.id}`, { knowledge_base_ids: next })
      .then(() => { kbPatchInFlight.current = null; })
      .catch((err) => {
        console.error("Failed to update chat KBs:", err);
        setAssociatedKbIds(prev);  // revert
        kbPatchInFlight.current = null;
      })
      .finally(() => setKbToggling(false));
    kbPatchInFlight.current = patch;
  }, [params.id, associatedKbIds]);

  return (
    <>
      <div className="flex flex-col h-full relative">

        {/* Scroll area */}
        <div ref={scrollContainerRef} className="flex-1 overflow-y-auto min-h-0 pt-14 pb-28">
          {/* Top sentinel — IntersectionObserver triggers loadMoreMessages when visible */}
          <div ref={topSentinelRef} className="h-px" />
          {isLoadingMore && (
            <div className="flex justify-center py-3">
              <LoadingDots size="sm" />
            </div>
          )}
          {messages.length === 0 && !isLoading && !isInitialLoad ? (
            /* Welcome / empty state */
            <div className="flex flex-col items-center justify-center h-full gap-4 px-4 text-center">
              <Image src={APP_LOGO_SRC} alt="logo" width={64} height={64} className="w-16 h-16 rounded-2xl" />
              <h2 className="text-2xl font-semibold">How can I help you today?</h2>
              <p className="text-sm text-muted-foreground">Ask anything about your knowledge base</p>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto px-4 py-6 space-y-6 pb-8">
              {messages.map((message) =>
                message.role === "assistant" ? (
                  <div key={message.clientId} className="flex items-start gap-3">
                    {/* Avatar */}
                    <Image
                      src={APP_LOGO_SRC}
                      width={28}
                      height={28}
                      className="h-7 w-7 rounded-full shrink-0 mt-0.5"
                      alt="assistant"
                    />
                    {/* Content */}
                    <div className="flex-1 min-w-0 text-sm">
                      {isLoading && !message.content ? (
                        <div className="flex items-center justify-center py-2" aria-label="Generating response…">
                          <div className="relative w-5 h-5">
                            <div className="absolute inset-0 rounded-full bg-primary/40 animate-pulse" />
                            <div className="absolute inset-0 rounded-full bg-primary/60"
                                 style={{ animation: 'skeleton-size 1.5s ease-in-out infinite' }} />
                          </div>
                        </div>
                      ) : (
                        <Answer
                          key={message.clientId}
                          messageId={message.id}
                          chatId={params.id}
                          markdown={message.content}
                          citations={message.citations}
                          confidence={message.confidence}
                          confidenceScore={message.confidenceScore}
                          confidenceBreakdown={message.confidenceBreakdown}
                          suggestion={message.suggestion}
                          failedLegs={message.failedLegs}
                          toolTrace={message.id === lastAssistantId ? message.toolTrace : undefined}
                          agentSteps={message.id === lastAssistantId ? message.agentSteps : undefined}
                          taskList={message.id === lastAssistantId ? taskList : undefined}
                          progressMessages={message.id === lastAssistantId && isLoading ? progressMessages : undefined}
                          synthesisMode={message.synthesisMode}
                          isStreaming={isLoading && message.id === lastAssistantId}
                          onDelete={handleDeleteMessage}
                          finalConfidence={message.finalConfidence}
                          finalConfidenceLevel={message.finalConfidenceLevel}
                          faithfulness={message.faithfulness}
                          completeness={message.completeness}
                          retrievalScore={message.retrievalScore}
                          plan={message.plan}
                          toolCalls={message.toolCalls}
                          toolObservations={message.toolObservations}
                          lastAnswerObject={message.lastAnswerObject}
                          chartOptions={message.chartOptions}
                          onFollowUp={handleFollowUp}
                        />
                      )}
                    </div>
                  </div>
                ) : (
                  <div key={message.clientId} className="flex justify-end items-start gap-2 group">
                    <div className="flex flex-col items-end gap-1 max-w-[70%]">
                      <div className="flex flex-row items-center gap-2">
                        {message.file_name && message.file_id && (
                          <MessageFileChip
                            fileName={message.file_name}
                            fileId={message.file_id}
                            chatId={params.id}
                          />
                        )}
                        <div className="bg-primary text-primary-foreground rounded-2xl rounded-tr-sm px-4 py-2.5 text-sm">
                          {message.content}
                        </div>
                      </div>{/* end flex-row bubble+chip */}
                      {/* Hover actions */}
                      <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                        <BranchPicker
                          messageId={message.id}
                          chatId={params.id}
                          content={message.content}
                          onBranch={(newMsgId, newContent) =>
                            handleBranch(message.id, newMsgId, newContent)
                          }
                          onNavigate={(targetUser, targetAssistant, currentMsgId, parentId) =>
                            handleNavigate(targetUser, targetAssistant, currentMsgId, parentId)
                          }
                          disabled={isLoading}
                        />
                        <button
                          onClick={() => handleCopyUserMessage(message.id, message.content)}
                          title={copiedMsgId === message.id ? "Copied!" : "Copy"}
                          className="flex items-center gap-1 px-2 py-0.5 rounded text-xs text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
                        >
                          {copiedMsgId === message.id ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
                          {copiedMsgId === message.id ? "Copied" : "Copy"}
                        </button>
                        <button
                          onClick={() => setConfirmDeleteMsgId(message.id)}
                          title="Delete"
                          className="flex items-center gap-1 px-2 py-0.5 rounded text-xs text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-colors"
                        >
                          <Trash2 className="h-3 w-3" />
                          Delete
                        </button>
                      </div>
                    </div>
                  </div>
                )
              )}

              {/* Clarification dialog — shown when agent needs user input */}
              {clarificationState && (
                <div className="max-w-3xl mx-auto px-4">
                  <ClarificationDialog
                    question={clarificationState.question}
                    options={clarificationState.options}
                    rationale={clarificationState.rationale}
                    attempt={clarificationState.attempt}
                    maxAttempts={clarificationState.maxAttempts}
                    onRespond={handleClarificationResponse}
                    onSkip={handleClarificationSkip}
                  />
                </div>
              )}

            </div>
          )}
        </div>

        {/* Floating input bar */}
        <div className="absolute bottom-0 left-0 right-0 z-10 pb-4 pt-2 px-4">
          <div className="max-w-3xl mx-auto relative">
            {/* Scroll-to-bottom button */}
            {!isAtBottom && (
              <button
                onClick={scrollToBottom}
                aria-label="Scroll to bottom"
                className="absolute -top-12 left-1/2 -translate-x-1/2 h-9 w-9 rounded-full border border-border bg-background/80 backdrop-blur-sm shadow-md flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
              >
                <ChevronDown className="h-5 w-5" />
              </button>
            )}
            <InputBar
              value={input}
              onChange={setInput}
              onSubmit={handleSubmit}
              onStop={handleStop}
              disabled={isLoading}
              placeholder="Type your message..."
              uploadedFile={uploadedFile}
              onFileAccepted={handleFileAccepted}
              onFileRemove={handleFileRemove}
              fileError={fileError}
              onFileError={setFileError}
              knowledgeBases={allKbs}
              selectedKbIds={associatedKbIds}
              onKbToggle={handleKbToggle}
              kbToggling={kbToggling}
            />
          </div>
        </div>
      </div>

      <ConfirmDialog
        open={confirmDeleteMsgId !== null}
        title="Delete message"
        description="Delete this message? This cannot be undone."
        confirmText="Delete"
        destructive
        onConfirm={async () => {
          const id = confirmDeleteMsgId;
          setConfirmDeleteMsgId(null);
          if (!id) return;
          try {
            await api.delete(`/api/chat/${params.id}/messages/${id}`);
            setMessages((prev) => prev.filter((m) => m.id !== id));
          } catch (e) {
            console.error("Failed to delete message:", e);
          }
        }}
        onCancel={() => setConfirmDeleteMsgId(null)}
      />
    </>
  );
}

export default function ChatPage({ params }: { params: Promise<{ id: string }> }) {
  const resolvedParams = use(params);
  return <ChatPageInner params={resolvedParams} />;
}
