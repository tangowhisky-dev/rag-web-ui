"use client";

import { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { api, ApiError } from "@/lib/api";
import { useChatContext } from "@/contexts/chat-context";
import { useToast } from "@/components/ui/use-toast";
import { LoadingDots } from "@/components/ui/loading-dots";
import { Plus } from "lucide-react";

interface KnowledgeBase {
  id: number;
  name: string;
  description: string;
}

export default function NewChatPage() {
  const router = useRouter();
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([]);
  const [selectedKBs, setSelectedKBs] = useState<number[]>([]);
  const [title, setTitle] = useState("");
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const { toast } = useToast();
  const { addChat } = useChatContext();

  const fetchKnowledgeBases = useCallback(async () => {
    try {
      const data = await api.get("/api/knowledge-base");
      setKnowledgeBases(data);
      setIsLoading(false);
    } catch (error) {
      console.error("Failed to fetch knowledge bases:", error);
      if (error instanceof ApiError) {
        toast({ title: "Error", description: error.message, variant: "destructive" });
      }
    }
  }, [toast]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      await fetchKnowledgeBases();
      if (cancelled) return;
    })();
    return () => { cancelled = true; };
  }, [fetchKnowledgeBases]);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (selectedKBs.length === 0) { setError("Please select at least one knowledge base"); return; }
    setError("");
    setIsSubmitting(true);
    try {
      const data = await api.post("/api/chat", {
        title,
        knowledge_base_ids: selectedKBs,
      });
      addChat({
        id: data.id,
        title: data.title,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        pinned: false,
        folder_id: null,
      });
      router.push(`/dashboard/chat/${data.id}`);
    } catch (error) {
      console.error("Failed to create chat:", error);
      if (error instanceof ApiError) {
        setError(error.message);
        toast({ title: "Error", description: error.message, variant: "destructive" });
      } else {
        setError("Failed to create chat");
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  if (!isLoading && knowledgeBases.length === 0) {
    return (
      <div className="max-w-2xl mx-auto text-center py-16">
        <h2 className="text-3xl font-bold tracking-tight mb-4">No Knowledge Bases Found</h2>
        <p className="text-muted-foreground mb-8">
          You need to create at least one knowledge base before starting a chat.
        </p>
        <Link
          href="/dashboard/knowledge"
          className="inline-flex items-center justify-center rounded-md text-sm font-medium bg-primary text-primary-foreground hover:bg-primary/90 h-10 px-4 py-2"
        >
          <Plus className="mr-2 h-4 w-4" />
          Create Knowledge Base
        </Link>
      </div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto space-y-4 pt-16">
      <div>
        <h2 className="text-3xl font-bold tracking-tight">Start New Chat</h2>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="space-y-2">
          <label htmlFor="title" className="text-sm font-medium leading-none">
            Chat Title
          </label>
          <input
            id="title"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            type="text"
            required
            className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            placeholder="Enter chat title"
          />
        </div>

        <div className="space-y-1">
          <label className="text-sm font-medium leading-none">
            Select knowledge bases to chat with
          </label>
          <div className="grid gap-4 md:grid-cols-2">
            {isLoading ? (
              <div className="col-span-2 flex justify-center py-8">
                <LoadingDots label="Loading knowledge bases" />
              </div>
            ) : (
              knowledgeBases.map((kb) => (
                <label
                  key={kb.id}
                  className={`group flex items-center space-x-3 rounded-lg border p-4 cursor-pointer transition-all duration-200 hover:shadow-md ${
                    selectedKBs.includes(kb.id) ? "border-primary bg-primary/5 shadow-sm" : "hover:border-primary/50"
                  }`}
                >
                  <input
                    type="checkbox"
                    className="peer h-4 w-4 shrink-0 rounded border border-primary"
                    checked={selectedKBs.includes(kb.id)}
                    onChange={() => {
                      setSelectedKBs((prev) =>
                        prev.includes(kb.id)
                          ? prev.filter((id) => id !== kb.id)
                          : [...prev, kb.id]
                      );
                    }}
                  />
                  <div className="flex-1 space-y-1">
                    <p className="font-medium group-hover:text-primary transition-colors">{kb.name}</p>
                    <p className="text-sm text-muted-foreground line-clamp-2">
                      {kb.description || "No description provided"}
                    </p>
                  </div>
                </label>
              ))
            )}
          </div>
        </div>

        {error && <div className="text-sm text-red-500">{error}</div>}

        <div className="flex justify-end space-x-4">
          <button
            type="button"
            onClick={() => router.back()}
            className="inline-flex items-center justify-center rounded-md text-sm font-medium border border-input bg-background hover:bg-accent h-10 px-4 py-2"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={isSubmitting || selectedKBs.length === 0}
            className="inline-flex items-center justify-center rounded-md text-sm font-medium bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50 h-10 px-4 py-2"
          >
            {isSubmitting ? "Creating..." : "Start Chat"}
          </button>
        </div>
      </form>
    </div>
  );
}
