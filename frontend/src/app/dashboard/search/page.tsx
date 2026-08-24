"use client";

import { useEffect, useState, useCallback } from "react";
import { Search as SearchIcon, FileText, ExternalLink, Info, Loader2 } from "lucide-react";
import DashboardLayout from "@/components/layout/dashboard-layout";
import { api, ApiError } from "@/lib/api";
import { useHydrated } from "@/lib/hooks";
import { cn } from "@/lib/utils";
import { useToast } from "@/components/ui/use-toast";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";

interface KbItem {
  id: number;
  name: string;
}

interface SearchResult {
  chunk_text: string;
  file_name: string;
  document_id: number;
  kb_id: number | null;
  data_store_id: number | null;
  chunk_index: number | null;
  reranker_score: number;
}

interface SearchResponse {
  query: string;
  expanded_query: string;
  results: SearchResult[];
  total: number;
  latency_ms: number;
}

function snippet(text: string, maxChars = 350): string {
  const clean = text.replace(/\s+/g, " ").trim();
  if (clean.length <= maxChars) return clean;
  return clean.slice(0, maxChars).trimEnd() + "…";
}

export default function SearchPage() {
  const hydrated = useHydrated();
  const { toast } = useToast();

  const [kbs, setKbs] = useState<KbItem[]>([]);
  const [selectedKbIds, setSelectedKbIds] = useState<number[]>([]);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [expandedQuery, setExpandedQuery] = useState("");
  const [hasSearched, setHasSearched] = useState(false);
  const [loading, setLoading] = useState(false);
  const [latencyMs, setLatencyMs] = useState(0);

  // Fetch user's KBs on mount
  useEffect(() => {
    api.get("/api/knowledge-base").then((data: any) => {
      const list = Array.isArray(data) ? data : data.items ?? [];
      const items = list.map((kb: any) => ({ id: kb.id, name: kb.name }));
      setKbs(items);
      setSelectedKbIds(items.map((kb: KbItem) => kb.id));
    }).catch((err) => {
      console.error("Failed to fetch KBs:", err);
    });
  }, []);

  const toggleKb = useCallback((kbId: number) => {
    setSelectedKbIds((prev) =>
      prev.includes(kbId)
        ? prev.filter((id) => id !== kbId)
        : [...prev, kbId]
    );
  }, []);

  const handleSearch = useCallback(async (e?: React.FormEvent) => {
    e?.preventDefault();
    const trimmed = query.trim();
    if (!trimmed || loading) return;
    if (selectedKbIds.length === 0) {
      toast({ title: "Select a knowledge base", description: "Choose at least one KB to search.", variant: "destructive" });
      return;
    }

    setLoading(true);
    setHasSearched(true);
    try {
      const res = await api.post("/api/search", {
        query: trimmed,
        kb_ids: selectedKbIds,
      }) as SearchResponse;
      setResults(res.results);
      setExpandedQuery(res.expanded_query);
      setLatencyMs(res.latency_ms);
    } catch (err) {
      if (err instanceof ApiError) {
        toast({ title: "Search failed", description: err.message, variant: "destructive" });
      } else {
        toast({ title: "Search failed", description: "Unexpected error", variant: "destructive" });
      }
      setResults([]);
    } finally {
      setLoading(false);
    }
  }, [query, loading, selectedKbIds, toast]);

  if (!hydrated) {
    return (
      <DashboardLayout pageTitle="Search">
        <div className="flex items-center justify-center py-20 text-muted-foreground">Loading…</div>
      </DashboardLayout>
    );
  }

  const showExpandedTooltip = expandedQuery && expandedQuery !== query.trim();

  return (
    <DashboardLayout pageTitle="Search">
      <div className="max-w-3xl mx-auto">
        {/* Search bar */}
        <form onSubmit={handleSearch} className="relative mb-4">
          <div className="flex items-center gap-2">
            <div className="relative flex-1">
              <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search your knowledge bases…"
                className="w-full rounded-lg border bg-background pl-10 pr-4 py-2.5 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                autoFocus
              />
            </div>
            {showExpandedTooltip && (
              <Popover>
                <PopoverTrigger asChild>
                  <button
                    type="button"
                    title="Abbreviation glossary"
                    className="shrink-0 rounded-full p-1.5 text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
                  >
                    <Info className="h-4 w-4" />
                  </button>
                </PopoverTrigger>
                <PopoverContent side="bottom" align="center" className="max-w-md w-80 text-xs">
                  <div className="space-y-1.5">
                    <p className="font-medium text-foreground">Expanded query</p>
                    <p className="text-muted-foreground whitespace-pre-wrap break-words">{expandedQuery}</p>
                  </div>
                </PopoverContent>
              </Popover>
            )}
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="shrink-0 rounded-lg bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : "Search"}
            </button>
          </div>
        </form>

        {/* KB pills */}
        <div className="flex flex-wrap gap-2 mb-6">
          {kbs.map((kb) => {
            const selected = selectedKbIds.includes(kb.id);
            return (
              <button
                key={kb.id}
                type="button"
                onClick={() => toggleKb(kb.id)}
                className={cn(
                  "h-7 rounded-full border px-2.5 text-xs shrink-0 transition-colors",
                  selected
                    ? "border-primary bg-primary/10 text-primary"
                    : "border-border bg-muted/40 text-muted-foreground/60 hover:bg-muted hover:text-muted-foreground"
                )}
              >
                {kb.name}
              </button>
            );
          })}
          {kbs.length === 0 && (
            <p className="text-xs text-muted-foreground">No knowledge bases available.</p>
          )}
        </div>

        {/* Results */}
        {loading && (
          <div className="flex items-center justify-center py-20 text-muted-foreground">
            <Loader2 className="h-5 w-5 animate-spin mr-2" />
            Searching…
          </div>
        )}

        {!loading && hasSearched && results.length === 0 && (
          <div className="text-center py-20 text-muted-foreground">
            <p className="text-sm">No results found for &ldquo;{query}&rdquo;.</p>
          </div>
        )}

        {!loading && !hasSearched && (
          <div className="text-center py-20 text-muted-foreground">
            <SearchIcon className="h-8 w-8 mx-auto mb-3 opacity-40" />
            <p className="text-sm">Search across your selected knowledge bases.</p>
          </div>
        )}

        {!loading && results.length > 0 && (
          <div className="space-y-4">
            <p className="text-xs text-muted-foreground">
              {results.length} result{results.length !== 1 ? "s" : ""} · {latencyMs}ms
            </p>
            {results.map((result, i) => (
              <a
                key={`${result.document_id}-${result.chunk_index}-${i}`}
                href={`/api/knowledge-base/documents/${result.document_id}/download`}
                target="_blank"
                rel="noopener noreferrer"
                className="block rounded-lg border bg-card p-4 hover:border-primary/30 transition-colors group"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-center gap-2 min-w-0">
                    <FileText className="h-4 w-4 text-muted-foreground shrink-0" />
                    <span className="text-sm font-medium text-foreground truncate group-hover:text-primary transition-colors">
                      {result.file_name}
                    </span>
                  </div>
                  <ExternalLink className="h-3.5 w-3.5 text-muted-foreground/50 shrink-0 group-hover:text-primary transition-colors" />
                </div>
                <p className="mt-2 text-sm text-muted-foreground leading-relaxed line-clamp-4">
                  {snippet(result.chunk_text)}
                </p>
                <div className="mt-2 flex items-center gap-3 text-xs text-muted-foreground/70">
                  {result.kb_id && (
                    <span>
                      KB: {kbs.find((kb) => kb.id === result.kb_id)?.name ?? `#${result.kb_id}`}
                    </span>
                  )}
                  <span className="tabular-nums">Score: {result.reranker_score.toFixed(2)}</span>
                </div>
              </a>
            ))}
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}
