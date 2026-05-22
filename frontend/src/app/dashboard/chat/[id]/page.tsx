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

import { useEffect, useRef, useState, useMemo } from "react";
import { flushSync } from "react-dom";
import { useRouter } from "next/navigation";
import { Copy, Trash2 } from "lucide-react";
import ChatLayout from "@/components/layout/chat-layout";
import { useChatContext, ChatProvider } from "@/contexts/chat-context";
import ChatSettings from "@/components/chat/chat-settings";
import type { ChatPatch } from "@/components/chat/chat-settings";
import { api, ApiError } from "@/lib/api";
import { APP_LOGO_SRC } from "@/lib/app-config";
import { useToast } from "@/components/ui/use-toast";
import { Answer } from "@/components/chat/answer";
import { InputBar } from "@/components/chat/chat-input";

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
}

interface ChatMessage {
  id: number;
  content: string;
  role: "assistant" | "user";
  created_at: string;
  confidence_level?: string;
  confidence_score?: number;
  confidence_breakdown?: string;
}

interface Chat {
  id: number;
  title: string;
  use_graph_rag: boolean;
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
  const { toast } = useToast();
  const { setActiveChat } = useChatContext();
  const [isInitialLoad, setIsInitialLoad] = useState(true);
  const [chatTitle, setChatTitle] = useState<string | undefined>();
  const [useGraphRag, setUseGraphRag] = useState(false);
  const [useDense, setUseDense] = useState(true);
  const [useSparse, setUseSparse] = useState(true);
  const [useExact, setUseExact] = useState(false);
  const [temperature, setTemperature] = useState(0.7);
  const [modelName, setModelName] = useState("gpt-4o");
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [attachedFile, setAttachedFile] = useState<File | null>(null);
  const [fileError, setFileError] = useState<string>("");

  useEffect(() => {
    setActiveChat(Number(params.id));
    return () => setActiveChat(null);
  }, [params.id, setActiveChat]);

  useEffect(() => {
    if (isInitialLoad) {
      fetchChat();
      setIsInitialLoad(false);
    }
  }, [isInitialLoad]);

  useEffect(() => {
    if (!isInitialLoad) {
      scrollToBottom();
    }
  }, [messages, isInitialLoad]);

  const fetchChat = async () => {
    try {
      const data: Chat = await api.get(`/api/chat/${params.id}`);
      setChatTitle(data.title);
      setUseGraphRag(data.use_graph_rag ?? false);
      setUseDense((data as any).use_dense ?? true);
      setUseSparse((data as any).use_sparse ?? true);
      setUseExact((data as any).use_exact ?? false);
      setTemperature((data as any).temperature ?? 0.7);
      setModelName((data as any).model_name ?? "gpt-4o");
      const formattedMessages = data.messages.map((msg) => {
        if (msg.role !== "assistant" || !msg.content)
          return {
            id: msg.id.toString(),
            role: msg.role,
            content: msg.content,
          };

        try {
          if (!msg.content.includes("__LLM_RESPONSE__")) {
            return {
              id: msg.id.toString(),
              role: msg.role,
              content: msg.content,
            };
          }

          const [base64Part, responseText] =
            msg.content.split("__LLM_RESPONSE__");

          const contextData = base64Part
            ? (JSON.parse(atob(base64Part.trim())) as {
                context: Array<{
                  page_content: string;
                  metadata: Record<string, any>;
                }>;
              })
            : null;

          const citations: Citation[] =
            contextData?.context.map((citation, index) => ({
              id: index + 1,
              text: citation.page_content,
              metadata: citation.metadata,
            })) || [];

          return {
            id: msg.id.toString(),
            role: msg.role,
            content: responseText || "",
            citations,
            confidence: msg.confidence_level as Message["confidence"] | undefined,
            confidenceScore: msg.confidence_score ?? undefined,
            confidenceBreakdown: msg.confidence_breakdown
              ? JSON.parse(msg.confidence_breakdown)
              : undefined,
          };
        } catch (e) {
          console.error("Failed to process message:", e);
          return {
            id: msg.id.toString(),
            role: msg.role,
            content: msg.content,
          };
        }
      });
      setMessages(formattedMessages);
    } catch (error) {
      console.error("Failed to fetch chat:", error);
      if (error instanceof ApiError) {
        toast({
          title: "Error",
          description: error.message,
          variant: "destructive",
        });
      }
      router.push("/dashboard/chat");
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const markdownParse = (text: string) => {
    return text
      .replace(/\[\[([cC])itation/g, "[citation")
      .replace(/[cC]itation:(\d+)]]/g, "citation:$1]")
      .replace(/\[\[([cC]itation:\d+)]](?!])/g, `[$1]`)
      .replace(/\[[cC]itation:(\d+)]/g, "[citation]($1)")
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
      requestAnimationFrame(() => resolve());
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
          context: Array<{ page_content: string; metadata: Record<string, any> }>;
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
        const citations: Citation[] = payload.context.map((doc, index) => ({
          id: index + 1,
          text: doc.page_content,
          metadata: doc.metadata,
        }));
        // Normalize confidence level: backend sends uppercase (HIGH/MEDIUM/LOW),
        // frontend type uses lowercase. Map backend values → frontend enum.
        const rawConfidence = payload.confidence?.toLowerCase() as Message["confidence"] | undefined;
        appendAssistantChunk(assistantId, (message) => ({
          ...message,
          citations,
          retrievedContext: payload.context,
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

    if (trimmedLine.startsWith("3:")) {
      const errorMessage = trimmedLine.slice(2);
      throw new Error(errorMessage || "Streaming request failed");
    }
  };

  const handleSubmit = async () => {

    const trimmedInput = input.trim();
    if (!trimmedInput || isLoading) {
      return;
    }

    const token =
      typeof window !== "undefined"
        ? window.localStorage.getItem("token") || ""
        : "";

    const assistantId = generateId();
    const userMessage: Message = {
      id: generateId(),
      role: "user",
      content: trimmedInput,
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
    setAttachedFile(null);
    setFileError("");
    setIsLoading(true);
    setMessages((prev) => [...prev, userMessage, assistantMessage]);

    try {
      let response: Response;
      if (attachedFile) {
        const formData = new FormData();
        formData.append("file", attachedFile);
        formData.append("messages", JSON.stringify(requestMessages));
        response = await fetch(`/api/chat/${params.id}/messages/with-file`, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${token}`,
          },
          body: formData,
        });
      } else {
        response = await fetch(`/api/chat/${params.id}/messages`, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ messages: requestMessages }),
        });
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

        // Process step events (1:, 2:) with immediate flushSync so collapsibles
        // appear before LLM tokens. Batch all token lines (0:) into a single
        // state update per reader chunk — avoids one rAF per token.
        let hasTokenLines = false;
        for (const line of lines) {
          const t = line.trim();
          if (t.startsWith("1:") || t.startsWith("2:")) {
            flushSync(() => processStreamLine(line, assistantId));
          } else if (t) {
            processStreamLine(line, assistantId);
            hasTokenLines = true;
          }
        }
        // One yield per chunk keeps UI responsive without per-token rAF overhead
        if (hasTokenLines) await flushToBrowser();
      }

      if (buffer.trim()) {
        processStreamLine(buffer, assistantId);
        await flushToBrowser();
      }
    } catch (error) {
      console.error("Failed to stream chat:", error);
      setMessages((prev) => prev.filter((message) => message.id !== assistantId));

      toast({
        title: "Error",
        description:
          error instanceof Error ? error.message : "Failed to send message",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
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
    if (patch.use_graph_rag !== undefined) setUseGraphRag(patch.use_graph_rag);
    if (patch.use_dense !== undefined) setUseDense(patch.use_dense);
    if (patch.use_sparse !== undefined) setUseSparse(patch.use_sparse);
    if (patch.use_exact !== undefined) setUseExact(patch.use_exact);
    if (patch.temperature !== undefined) setTemperature(patch.temperature);
    if (patch.model_name !== undefined) setModelName(patch.model_name);
  };

  return (
    <ChatLayout pageTitle={chatTitle ?? undefined} graphRagActive={useGraphRag}>
      <div className="flex flex-col h-full relative">


        {isSettingsOpen && (
          <ChatSettings
            chat={{
              id: Number(params.id),
              title: chatTitle ?? "",
              temperature,
              model_name: modelName,
              use_dense: useDense,
              use_sparse: useSparse,
              use_exact: useExact,
              use_graph_rag: useGraphRag,
            }}
            onClose={() => setIsSettingsOpen(false)}
            onUpdate={handleSettingsUpdate}
          />
        )}

        {/* Scroll area */}
        <div className="flex-1 overflow-y-auto min-h-0 pb-28">
          {processedMessages.length === 0 && !isLoading ? (
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
                        <div className="flex items-center gap-1 py-2">
                          <div className="w-2 h-2 rounded-full bg-primary animate-bounce" />
                          <div className="w-2 h-2 rounded-full bg-primary animate-bounce [animation-delay:0.2s]" />
                          <div className="w-2 h-2 rounded-full bg-primary animate-bounce [animation-delay:0.4s]" />
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
                          synthesisMode={message.synthesisMode}
                          isStreaming={isLoading && message.id === lastAssistantId}
                          onDelete={(id) => setMessages((prev) => prev.filter((m) => m.id !== id))}
                        />
                      )}
                    </div>
                  </div>
                ) : (
                  <div key={message.id} className="flex justify-end items-start gap-2 group">
                    <div className="flex flex-col items-end gap-1 max-w-[70%]">
                      <div className="bg-primary text-primary-foreground rounded-2xl rounded-tr-sm px-4 py-2.5 text-sm">
                        {message.content}
                      </div>
                      {/* Hover actions */}
                      <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
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
              <div ref={messagesEndRef} />
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
              disabled={isLoading}
              placeholder="Type your message..."
              file={attachedFile}
              onFileChange={setAttachedFile}
              fileError={fileError}
              onFileError={setFileError}
            />
          </div>
        </div>
      </div>
    </ChatLayout>
  );
}

export default function ChatPage({ params }: { params: { id: string } }) {
  return (
    <ChatProvider>
      <ChatPageInner params={params} />
    </ChatProvider>
  );
}
