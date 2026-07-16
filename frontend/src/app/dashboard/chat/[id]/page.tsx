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

import { useEffect, useLayoutEffect, useRef, useState, useMemo, useCallback } from "react";

import { useRouter } from "next/navigation";
import { Copy, Trash2, Lightbulb } from "lucide-react";
import { useChatContext } from "@/contexts/chat-context";
import ChatSettings from "@/components/chat/chat-settings";
import type { ChatPatch } from "@/components/chat/chat-settings";
import { api, ApiError } from "@/lib/api";
import { APP_LOGO_SRC } from "@/lib/app-config";
import { cancelStream } from "@/lib/cancel-stream";
import { useToast } from "@/components/ui/use-toast";
import { Answer } from "@/components/chat/answer";
import { InputBar } from "@/components/chat/chat-input";
import { MessageFileChip, type UploadedFile } from "@/components/chat/file-attachment";
import { BranchPicker } from "@/components/chat/branch-picker";
import ClarificationDialog from "@/components/chat/clarification-dialog";

interface AgentStep {
  node: string;
  latency_ms: number;
  status: string;
  // optional per-node detail fields emitted by the backend
  [key: string]: unknown;
}

interface Message {
  id: string;
  role: "assistant" | "user" | "system" | "data";
  content: string;
  citations?: Citation[];
  rewrittenQuery?: string;
  retrievedContext?: Array<{ page_content: string; metadata: Record<string, any> }>;
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
  synthesisMode?: boolean;
  agentSteps?: AgentStep[];
  file_name?: string;  // filename of attached chat file, if any
  file_id?: number;    // chat_files.id — needed for download URL
  // Final evaluation metrics (from answer_evaluation_node)
  finalConfidence?: number;
  finalConfidenceLevel?: "very_high" | "high" | "medium" | "low" | "none";
  retrievalConfidence?: number;
  faithfulness?: number;
  completeness?: number;
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
  file_name?: string;
  file_id?: number;
  citations?: Citation[];
}

interface Chat {
  id: number;
  title: string;
  messages: ChatMessage[];
}

interface Citation {
  id: number;
  text: string;
  metadata: Record<string, any>;
}

function ChatPageInner({ params }: { params: { id: string } }) {
  const router = useRouter();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const { toast } = useToast();
  const { setActiveChat, setGraphRagActive } = useChatContext();
  const [isInitialLoad, setIsInitialLoad] = useState(true);
  const [chatTitle, setChatTitle] = useState<string | undefined>();
  const [temperature, setTemperature] = useState(0.7);
  const [modelName, setModelName] = useState("gpt-4o");
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<UploadedFile | null>(null);
  const [fileError, setFileError] = useState<string>("");
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Clarification state ───────────────────────────────────────────────────
  const [clarificationState, setClarificationState] = useState<{
    question: string;
    options: string[];
    rationale?: string;
    assistantId: string;
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

  // Poll file status until ready or error
  const startPolling = (fileId: number) => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const token = localStorage.getItem("token");
        const res = await fetch(`/api/chat/${params.id}/files/${fileId}`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!res.ok) return;
        const data = await res.json();
        setUploadedFile((prev) => prev ? { ...prev, ...data } : null);
        if (data.status === "ready" || data.status === "error") {
          clearInterval(pollRef.current!);
          pollRef.current = null;
          if (data.status === "error") {
            setFileError(data.error_message || "File processing failed.");
          }
        }
      } catch { /* ignore */ }
    }, 1200);
  };

  // Upload file immediately on attach
  const handleFileAccepted = async (file: File) => {
    setFileError("");
    const token = localStorage.getItem("token");
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await fetch(`/api/chat/${params.id}/files`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        setFileError(err.detail || "Upload failed.");
        return;
      }
      const data: UploadedFile = await res.json();
      setUploadedFile(data);
      if (data.status === "processing") startPolling(data.id);
    } catch (err) {
      setFileError("Upload failed. Please try again.");
    }
  };

  const handleFileRemove = async () => {
    if (uploadedFile) {
      const token = localStorage.getItem("token");
      fetch(`/api/chat/${params.id}/files/${uploadedFile.id}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` },
      }).catch(() => {});
    }
    setUploadedFile(null);
    setFileError("");
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
  };

  useEffect(() => { return () => { if (pollRef.current) clearInterval(pollRef.current); }; }, []);

  useEffect(() => {
    setActiveChat(Number(params.id));
    return () => { setActiveChat(null); setGraphRagActive(false); };
  }, [params.id, setActiveChat, setGraphRagActive]);

  useEffect(() => {
    if (isInitialLoad) {
      fetchChat();
      // NOTE: do NOT setIsInitialLoad(false) here — fetchChat does it in its finally block
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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

  // ── Message formatter (shared by initial load and paginated load) ───────────
  const formatMessage = useCallback((msg: ChatMessage): Message => {
    if (msg.role !== "assistant" || !msg.content)
      return {
        id: msg.id.toString(),
        role: msg.role,
        content: msg.content,
        file_name: msg.file_name ?? undefined,
        file_id: msg.file_id ?? undefined,
        citations: msg.citations ?? [],
      };

    // Assistant message — citations come from the API, content is the raw answer text
    return {
      id: msg.id.toString(),
      role: msg.role,
      content: msg.content,
      citations: msg.citations ?? [],
      confidence: msg.confidence_level as Message["confidence"] | undefined,
      confidenceScore: msg.confidence_score ?? undefined,
      confidenceBreakdown: msg.confidence_breakdown
        ? JSON.parse(msg.confidence_breakdown)
        : undefined,
      finalConfidence: msg.final_confidence ?? undefined,
      finalConfidenceLevel: msg.final_confidence_level as Message["finalConfidenceLevel"] | undefined,
      faithfulness: msg.faithfulness ?? undefined,
      completeness: msg.completeness ?? undefined,
      file_name: msg.file_name ?? undefined,
      file_id: msg.file_id ?? undefined,
    };
  }, []);

  const fetchChat = async () => {
    try {
      // Fetch metadata (no messages) and first page in parallel
      const [meta, page] = await Promise.all([
        api.get(`/api/chat/${params.id}?include_messages=false`),
        api.get(`/api/chat/${params.id}/messages/paginated?limit=20`),
      ]);
      setChatTitle(meta.title);
      setTemperature((meta as any).temperature ?? 0.7);
      setModelName((meta as any).model_name ?? "gpt-4o");
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
  };

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
  }, [isLoadingMore, hasMoreMessages, messages, params.id, formatMessage]);

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

  const scrollToBottom = () => {
    if (scrollContainerRef.current) {
      scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
    }
  };

  const markdownParse = (text: string) => {
    return text
      .replace(/\[\[([cC])itation/g, "[citation")
      .replace(/[cC]itation:(\d+)]]/g, "citation:$1]")
      .replace(/\[\[([cC]itation:\d+)]](?!])/g, `[$1]`)
      .replace(/\[[cC]itation:(\d+)]/g, "[citation]($1)")
      // Agentic pipeline emits [KB-N] labels instead of [N](N).
      .replace(/\[KB-(\d+)\]/g, "[citation]($1)")
      // Fallback: plain [N] that the model emits instead of [citation:N].
      // Only match standalone bracketed numbers (not part of markdown list
      // syntax "1." or already-converted "[citation](N)").
      .replace(/(?<!\()\[(\d+)\](?!\()/g, "[citation]($1)");
  };

  const parseContextCitations = (base64Part: string): Citation[] => {
    if (!base64Part) {
      return [];
    }

    const contextData = JSON.parse(atob(base64Part.trim())) as {
      context: Array<{
        page_content: string;
        metadata: Record<string, any>;
      }>;
    };

    return (
      contextData.context.map((citation, index) => ({
        id: index + 1,
        text: citation.page_content,
        metadata: citation.metadata,
      })) || []
    );
  };

  const parseContextData = (base64Part: string): {
    citations: Citation[];
    rewrittenQuery: string | undefined;
    retrievedContext: Array<{ page_content: string; metadata: Record<string, any> }>;
  } => {
    if (!base64Part) return { citations: [], rewrittenQuery: undefined, retrievedContext: [] };

    const contextData = JSON.parse(atob(base64Part.trim())) as {
      context: Array<{ page_content: string; metadata: Record<string, any> }>;
      rewritten_query?: string;
    };

    const citations = contextData.context.map((doc, index) => ({
      id: index + 1,
      text: doc.page_content,
      metadata: doc.metadata,
    }));

    return {
      citations,
      rewrittenQuery: contextData.rewritten_query,
      retrievedContext: contextData.context,
    };
  };

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

  const processStreamLine = (line: string, assistantId: string) => {
    const trimmedLine = line.trim();

    if (!trimmedLine || trimmedLine === "d:[DONE]") {
      return;
    }

    // 1: rewritten query — emitted right after step 1, before retrieval
    if (trimmedLine.startsWith("1:")) {
      try {
        const payload = JSON.parse(trimmedLine.slice(2)) as { rewritten_query: string };
        appendAssistantChunk(assistantId, (message) => ({
          ...message,
          rewrittenQuery: payload.rewritten_query,
        }));
      } catch (e) {
        console.error("Failed to parse rewritten_query event:", e);
      }
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
          query_classification?: {
            type: string;
            confidence: number;
            latency_ms: number;
            fallback: boolean;
          };
          tool_trace?: Array<{
            tool_name: string;
            params?: Record<string, unknown>;
            output?: unknown;
            error?: string | null;
            latency_ms: number;
          }>;
          synthesis_mode?: boolean;
        };
        // Backend fast_pipeline sends "docs"; rag_graph sends "context"
        const rawDocs = payload.docs ?? payload.context ?? [];
        const citations: Citation[] = rawDocs.map((doc, index) => {
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
        // Normalize confidence level: backend sends uppercase (HIGH/MEDIUM/LOW),
        // frontend type uses lowercase. Map backend values → frontend enum.
        const rawConfidence = payload.confidence?.toLowerCase() as Message["confidence"] | undefined;
        appendAssistantChunk(assistantId, (message) => ({
          ...message,
          citations,
          retrievedContext: rawDocs,
          confidence: rawConfidence,
          confidenceScore: payload.score,
          confidenceBreakdown: payload.breakdown,
          suggestion: payload.suggestion,
          failedLegs: payload.failed_legs,
          queryClassification: payload.query_classification,
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
          // citations/rewrittenQuery/retrievedContext already set by 1:/2: events;
          // only apply the initial response text chunk here
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

    // 4: agent_step — LangGraph node start/finish events for AgentTimeline
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

    // r: answer_rewrite — citation-normalised full answer replaces streamed tokens
    if (trimmedLine.startsWith("r:")) {
      try {
        const payload = JSON.parse(trimmedLine.slice(2)) as { content: string };
        appendAssistantChunk(assistantId, (message) => ({
          ...message,
          content: payload.content,
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
        const payload = JSON.parse(trimmedLine.slice(2)) as {
          content: string;
          done: boolean;
        };
        setThinkingContent(payload);
      } catch (e) {
        console.error("Failed to parse thinking event:", e);
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
          usage?: {
            final_confidence?: number;
            confidence_level?: string;
            faithfulness?: number;
            completeness?: number;
          };
        };
        const usage = payload.usage;
        setMessages((prev) =>
          prev.map((message) =>
            message.id === assistantId
              ? {
                  ...message,
                  id: payload.messageId?.toString() ?? message.id,
                  finalConfidence: usage?.final_confidence,
                  finalConfidenceLevel: usage?.confidence_level as Message["finalConfidenceLevel"],
                  faithfulness: usage?.faithfulness,
                  completeness: usage?.completeness,
                }
              : message
          )
        );
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
        };
        setClarificationState({
          question: payload.question,
          options: payload.options || [],
          rationale: payload.rationale,
          assistantId,
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
    fileId?: number
  ) => {
    // Reset transient state for new query
    setProgressMessages([]);
    setTaskList([]);
    setThinkingContent(null);

    const token =
      typeof window !== "undefined"
        ? window.localStorage.getItem("token") || ""
        : "";

    const controller = new AbortController();
    abortControllerRef.current = controller;

    const response = await fetch(`/api/chat/${params.id}/messages`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        messages: requestMessages,
        ...(fileId ? { file_id: fileId } : {}),
      }),
      signal: controller.signal,
    });

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

      let hasTokenLines = false;
      for (const line of lines) {
        const t = line.trim();
        if (t.startsWith("1:") || t.startsWith("2:")) {
          processStreamLine(line, assistantId);
        } else if (t) {
          processStreamLine(line, assistantId);
          hasTokenLines = true;
        }
      }
      // Always flush so UI updates progressively between agent steps
      await flushToBrowser();
    }

    if (buffer.trim()) {
      processStreamLine(buffer, assistantId);
      await flushToBrowser();
    }
  };

  const handleSubmit = async () => {

    const trimmedInput = input.trim();
    if (!trimmedInput || isLoading) {
      return;
    }

    const assistantId = generateId();
    const userMessage: Message = {
      id: generateId(),
      role: "user",
      content: trimmedInput,
      ...(uploadedFile ? { file_name: uploadedFile.file_name, file_id: uploadedFile.id } : {}),
    };
    const assistantMessage: Message = {
      id: assistantId,
      role: "assistant",
      content: "",
      citations: [],
    };

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
    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setTimeout(scrollToBottom, 0);

    try {
      await streamFromMessages(requestMessages, assistantId, sentFile?.id);
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
        role: "assistant" as const,
        content: "",
        citations: [],
      });
      return result;
    });

    setIsLoading(true);
    try {
      await streamFromMessages(requestMessages, newAssistantId);
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

  /** Called by BranchPicker when the user navigates to a sibling branch. */
  const handleNavigate = (targetMessageId: string, targetContent: string, currentMessageId: string) => {
    setMessages((prev) =>
      prev.map((m) =>
        m.id === currentMessageId
          ? { ...m, id: targetMessageId, content: targetContent }
          : m
      )
    );
  };

  /** Handle user's clarification response */
  const handleClarificationResponse = async (response: string) => {
    if (!clarificationState) return;

    const { assistantId } = clarificationState;

    // Add user's clarification as a message
    const clarificationMessage: Message = {
      id: generateId(),
      role: "user",
      content: response,
    };

    setMessages((prev) => [...prev, clarificationMessage]);
    setClarificationState(null);
    setIsLoading(true);

    // Send clarification to backend
    try {
      const token = localStorage.getItem("token") || "";
      await fetch(`/api/chat/clarification`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          chat_id: Number(params.id),
          message_id: Number(assistantId),
          response: response,
        }),
      });

      // Resume streaming with clarified query
      const requestMessages = messages
        .filter((m) => m.role === "user" || m.role === "assistant")
        .map((m) => ({
          role: m.role,
          content: m.content,
        }))
        .concat({ role: "user", content: response });

      await streamFromMessages(requestMessages, assistantId);
    } catch (error) {
      console.error("Failed to send clarification:", error);
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to send clarification",
        variant: "destructive",
      });
      setIsLoading(false);
    }
  };

  /** Handle user skipping clarification */
  const handleClarificationSkip = () => {
    if (!clarificationState) return;
    setClarificationState(null);
    setIsLoading(true);
  };

  const processedMessages = useMemo(() => {
    return messages.map((message) => {
      if (message.role !== "assistant" || !message.content) return message;

      return {
        ...message,
        content: markdownParse(message.content),
      };
    });
  }, [messages]);

  const lastAssistantId = useMemo(() => {
    const assistants = processedMessages.filter((m) => m.role === "assistant");
    return assistants[assistants.length - 1]?.id;
  }, [processedMessages]);

  const handleExport = async () => {
    try {
      const res = await fetch(`/api/chat/${params.id}/export`, {
        headers: { Authorization: `Bearer ${localStorage.getItem("token") ?? ""}` },
      });
      if (!res.ok) throw new Error("Export failed");
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `chat-${params.id}.md`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (e) {
      console.error("Export failed", e);
    }
  };

  const handleSettingsUpdate = (patch: Partial<ChatPatch>) => {
    if (patch.temperature !== undefined) setTemperature(patch.temperature);
    if (patch.model_name !== undefined) setModelName(patch.model_name);
  };

  return (
    <>
      <div className="flex flex-col h-full relative">


        {isSettingsOpen && (
          <ChatSettings
            chat={{
              id: Number(params.id),
              title: chatTitle ?? "",
              temperature,
              model_name: modelName,
            }}
            onClose={() => setIsSettingsOpen(false)}
            onUpdate={handleSettingsUpdate}
          />
        )}

        {/* Scroll area */}
        <div ref={scrollContainerRef} className="flex-1 overflow-y-auto min-h-0 pt-14 pb-28">
          {/* Top sentinel — IntersectionObserver triggers loadMoreMessages when visible */}
          <div ref={topSentinelRef} className="h-px" />
          {isLoadingMore && (
            <div className="flex justify-center py-3">
              <div className="h-4 w-4 rounded-full border-2 border-primary border-t-transparent animate-spin" />
            </div>
          )}
          {processedMessages.length === 0 && !isLoading && !isInitialLoad ? (
            /* Welcome / empty state */
            <div className="flex flex-col items-center justify-center h-full gap-4 px-4 text-center">
              <img src={APP_LOGO_SRC} alt="logo" className="w-16 h-16 rounded-2xl" />
              <h2 className="text-2xl font-semibold">How can I help you today?</h2>
              <p className="text-sm text-muted-foreground">Ask anything about your knowledge base</p>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto px-4 py-6 space-y-6 pb-8">
              {processedMessages.map((message) =>
                message.role === "assistant" ? (
                  <div key={message.id} className="flex items-start gap-3">
                    {/* Avatar */}
                    <img
                      src={APP_LOGO_SRC}
                      className="h-7 w-7 rounded-full shrink-0 mt-0.5"
                      alt="assistant"
                    />
                    {/* Content */}
                    <div className="flex-1 min-w-0 text-sm">
                      {isLoading && !message.content && !message.rewrittenQuery ? (
                        <div className="flex items-center justify-center py-2" aria-label="Generating response…">
                          <div className="relative w-5 h-5">
                            <div className="absolute inset-0 rounded-full bg-primary/40 animate-pulse" />
                            <div className="absolute inset-0 rounded-full bg-primary/60"
                                 style={{ animation: 'skeleton-size 1.5s ease-in-out infinite' }} />
                          </div>
                        </div>
                      ) : (
                        <Answer
                          key={message.id}
                          messageId={message.id}
                          chatId={params.id}
                          markdown={message.content}
                          citations={message.citations}
                          rewrittenQuery={message.id === lastAssistantId ? message.rewrittenQuery : undefined}
                          retrievedContext={message.id === lastAssistantId ? message.retrievedContext : undefined}
                          confidence={message.confidence}
                          confidenceScore={message.confidenceScore}
                          confidenceBreakdown={message.confidenceBreakdown}
                          suggestion={message.suggestion}
                          failedLegs={message.failedLegs}
                          queryClassification={message.id === lastAssistantId ? message.queryClassification : undefined}
                          toolTrace={message.id === lastAssistantId ? message.toolTrace : undefined}
                          agentSteps={message.id === lastAssistantId ? message.agentSteps : undefined}
                          taskList={message.id === lastAssistantId ? taskList : undefined}
                          progressMessages={message.id === lastAssistantId && isLoading ? progressMessages : undefined}
                          synthesisMode={message.synthesisMode}
                          isStreaming={isLoading && message.id === lastAssistantId}
                          onDelete={(id) => setMessages((prev) => prev.filter((m) => m.id !== id))}
                          finalConfidence={message.finalConfidence}
                          finalConfidenceLevel={message.finalConfidenceLevel}
                          faithfulness={message.faithfulness}
                          completeness={message.completeness}
                        />
                      )}
                    </div>
                  </div>
                ) : (
                  <div key={message.id} className="flex justify-end items-start gap-2 group">
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
                          onNavigate={(siblingId, siblingContent) =>
                            handleNavigate(siblingId, siblingContent, message.id)
                          }
                          disabled={isLoading}
                        />
                        <button
                          onClick={() => navigator.clipboard.writeText(message.content)}
                          title="Copy"
                          className="flex items-center gap-1 px-2 py-0.5 rounded text-xs text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
                        >
                          <Copy className="h-3 w-3" />
                          Copy
                        </button>
                        <button
                          onClick={async () => {
                            if (!window.confirm("Delete this message? This cannot be undone.")) return;
                            try {
                              await api.delete(`/api/chat/${params.id}/messages/${message.id}`);
                              setMessages((prev) => prev.filter((m) => m.id !== message.id));
                            } catch (e) {
                              console.error("Failed to delete message:", e);
                            }
                          }}
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
          <div className="max-w-3xl mx-auto">
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
            />
          </div>
        </div>
      </div>
    </>
  );
}

export default function ChatPage({ params }: { params: { id: string } }) {
  return <ChatPageInner params={params} />;
}
