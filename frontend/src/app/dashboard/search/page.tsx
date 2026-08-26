"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import {
  Search as SearchIcon,
  FileText,
  FileCode,
  FileType,
  FileSpreadsheet,
  File,
  Info,
  Loader2,
  ChevronDown,
  Clock,
  Sparkles,
} from "lucide-react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
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
  original_text: string | null;
  title: string | null;
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

interface HistoryItem {
  id: number;
  query: string;
  result_count: number;
  created_at: string;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function snippet(text: string, maxChars = 500): string {
  const clean = text.trim();
  if (clean.length <= maxChars) return clean;
  return clean.slice(0, maxChars).trimEnd() + "…";
}

function fileIcon(name: string) {
  const ext = name.split(".").pop()?.toLowerCase() ?? "";
  if (["py", "js", "ts", "tsx", "jsx", "go", "rs", "java", "c", "cpp", "rb", "sh", "yaml", "yml", "json", "xml"].includes(ext))
    return FileCode;
  if (["pdf"].includes(ext))
    return FileType;
  if (["xls", "xlsx", "csv"].includes(ext))
    return FileSpreadsheet;
  if (["md", "txt", "rtf"].includes(ext))
    return FileText;
  return File;
}

function scoreTier(score: number): { label: string; className: string } {
  if (score >= 0.8) return { label: "high", className: "bg-success/15 text-success border-success/20" };
  if (score >= 0.5) return { label: "med", className: "bg-warning/15 text-warning border-warning/20" };
  return { label: "low", className: "bg-muted text-muted-foreground border-border" };
}

// Clean a filename into a title-like string for comparison, so we don't
// show a redundant filename line when the title is just the cleaned name.
function cleanFilename(name: string): string {
  const stem = name.replace(/\.[^.]+$/, "").replace(/[_-]/g, " ").replace(/\./g, " ");
  return stem.replace(/\s+/g, " ").trim();
}

// Bold query terms in markdown source so they render emphasized within
// the Markdown component. Wraps whole-word matches in ** for bold.
function highlightInMarkdown(text: string, query: string): string {
  const terms = query
    .split(/\s+/)
    .map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
    .filter((t) => t.length >= 2);
  if (terms.length === 0) return text;

  const regex = new RegExp(`\\b(${terms.join("|")})\\b`, "gi");
  return text.replace(regex, "**$1**");
}

// ── Skeleton ─────────────────────────────────────────────────────────────────

function SkeletonCard() {
  return (
    <div className="rounded-lg border bg-card p-4 animate-pulse">
      <div className="flex items-center gap-2">
        <div className="h-4 w-4 rounded bg-muted" />
        <div className="h-4 w-40 rounded bg-muted" />
      </div>
      <div className="mt-3 space-y-2">
        <div className="h-3 w-full rounded bg-muted/60" />
        <div className="h-3 w-4/5 rounded bg-muted/60" />
        <div className="h-3 w-3/5 rounded bg-muted/60" />
      </div>
      <div className="mt-3 flex gap-3">
        <div className="h-3 w-20 rounded bg-muted/40" />
        <div className="h-3 w-16 rounded bg-muted/40" />
      </div>
    </div>
  );
}

// ── Page ─────────────────────────────────────────────────────────────────────

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
  const [kbPickerOpen, setKbPickerOpen] = useState(false);

  // Empty-state data
  const [recentSearches, setRecentSearches] = useState<HistoryItem[]>([]);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [suggestionsLoading, setSuggestionsLoading] = useState(false);

  const inputRef = useRef<HTMLInputElement>(null);

  // Fetch KBs on mount
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

  // Fetch recent searches + LLM suggestions for empty state
  useEffect(() => {
    if (hasSearched) return;
    api.get("/api/search/history?limit=3").then((data: any) => {
      const items = Array.isArray(data) ? data : [];
      setRecentSearches(items);
    }).catch(() => {});

    setSuggestionsLoading(true);
    api.post("/api/search/suggestions").then((data: any) => {
      setSuggestions(data.suggestions ?? []);
    }).catch(() => {
      setSuggestions([]);
    }).finally(() => {
      setSuggestionsLoading(false);
    });
  }, [hasSearched]);

  const toggleKb = useCallback((kbId: number) => {
    setSelectedKbIds((prev) =>
      prev.includes(kbId)
        ? prev.filter((id) => id !== kbId)
        : [...prev, kbId]
    );
  }, []);

  const handleSearch = useCallback(async (q?: string) => {
    const searchQuery = (q ?? query).trim();
    if (!searchQuery || loading) return;
    if (selectedKbIds.length === 0) {
      toast({ title: "Select a knowledge base", description: "Choose at least one KB to search.", variant: "destructive" });
      return;
    }

    if (q) setQuery(q);
    setLoading(true);
    setHasSearched(true);
    setKbPickerOpen(false);
    try {
      const res = await api.post("/api/search", {
        query: searchQuery,
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
  const selectedCount = selectedKbIds.length;

  return (
    <DashboardLayout pageTitle="Search">
      <div className="max-w-5xl mx-auto">
        {/* Search bar */}
        <form
          onSubmit={(e) => { e.preventDefault(); handleSearch(); }}
          className="relative mb-3"
        >
          <div className="flex items-center gap-2">
            <div className="relative flex-1">
              <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <input
                ref={inputRef}
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
              className="shrink-0 rounded-lg bg-primary p-2.5 text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              title="Search"
            >
              {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <SearchIcon className="h-4 w-4" />}
            </button>
          </div>
        </form>

        {/* KB selection — collapsed after search, expanded before */}
        {hasSearched ? (
          <div className="mb-4">
            <button
              type="button"
              onClick={() => setKbPickerOpen((v) => !v)}
              className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              <span className="tabular-nums">{selectedCount} KB{selectedCount !== 1 ? "s" : ""} selected</span>
              <ChevronDown className={cn("h-3.5 w-3.5 transition-transform", kbPickerOpen && "rotate-180")} />
            </button>
            {kbPickerOpen && (
              <div className="flex flex-wrap gap-2 mt-2">
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
              </div>
            )}
          </div>
        ) : (
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
        )}

        {/* Loading skeleton */}
        {loading && (
          <div className="space-y-3">
            <div className="h-4 w-32 rounded bg-muted/40 animate-pulse" />
            <SkeletonCard />
            <SkeletonCard />
            <SkeletonCard />
            <SkeletonCard />
          </div>
        )}

        {/* No results */}
        {!loading && hasSearched && results.length === 0 && (
          <div className="text-center py-20 text-muted-foreground">
            <p className="text-sm">No results found for &ldquo;{query}&rdquo;.</p>
          </div>
        )}

        {/* Empty state — recent searches + LLM suggestions */}
        {!loading && !hasSearched && (
          <div className="py-12">
            <div className="text-center mb-8">
              <SearchIcon className="h-8 w-8 mx-auto mb-3 opacity-40" />
              <p className="text-sm text-muted-foreground">Search across your selected knowledge bases.</p>
            </div>

            {/* LLM suggestions */}
            {suggestionsLoading && (
              <div className="max-w-md mx-auto space-y-2">
                {[0, 1, 2].map((i) => (
                  <div key={i} className="h-9 rounded-lg bg-muted/40 animate-pulse" />
                ))}
              </div>
            )}
            {!suggestionsLoading && suggestions.length > 0 && (
              <div className="max-w-md mx-auto mb-8 animate-in fade-in duration-500">
                <div className="flex items-center gap-1.5 mb-2 text-xs text-muted-foreground">
                  <Sparkles className="h-3.5 w-3.5" />
                  <span>Suggested searches</span>
                </div>
                <div className="space-y-2">
                  {suggestions.map((s, i) => (
                    <button
                      key={i}
                      type="button"
                      onClick={() => handleSearch(s)}
                      className="block w-full text-left rounded-lg border bg-card px-3 py-2 text-sm text-foreground hover:border-primary/30 hover:bg-accent/40 transition-colors"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Recent searches */}
            {recentSearches.length > 0 && (
              <div className="max-w-md mx-auto">
                <div className="flex items-center gap-1.5 mb-2 text-xs text-muted-foreground">
                  <Clock className="h-3.5 w-3.5" />
                  <span>Recent searches</span>
                </div>
                <div className="space-y-1">
                  {recentSearches.map((item) => (
                    <button
                      key={item.id}
                      type="button"
                      onClick={() => handleSearch(item.query)}
                      className="flex items-center justify-between w-full text-left rounded-md px-3 py-1.5 text-sm text-muted-foreground hover:text-foreground hover:bg-muted/40 transition-colors group"
                    >
                      <span className="truncate">{item.query}</span>
                      <span className="text-xs text-muted-foreground/50 tabular-nums shrink-0 ml-2">
                        {item.result_count} result{item.result_count !== 1 ? "s" : ""}
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Results */}
        {!loading && results.length > 0 && (
          <div className="space-y-3">
            <p className="text-xs text-muted-foreground tabular-nums">
              {results.length} result{results.length !== 1 ? "s" : ""} · {latencyMs}ms
            </p>
            {results.map((result, i) => {
              const Icon = fileIcon(result.file_name);
              const tier = scoreTier(result.reranker_score);
              const kbName = result.kb_id
                ? kbs.find((kb) => kb.id === result.kb_id)?.name ?? `#${result.kb_id}`
                : result.data_store_id
                  ? `DS #${result.data_store_id}`
                  : null;
              const displayTitle = result.title || result.file_name;
              const showFilename = result.title && result.title !== cleanFilename(result.file_name);
              return (
                <a
                  key={`${result.document_id}-${result.chunk_index}-${i}`}
                  href={`/api/knowledge-base/documents/${result.document_id}/download`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block rounded-lg border bg-card p-4 hover:border-primary/30 transition-colors group"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex items-center gap-2 min-w-0">
                      <Icon className="h-4 w-4 text-muted-foreground shrink-0" />
                      <div className="min-w-0">
                        <span className="text-sm font-medium text-foreground truncate block group-hover:text-primary transition-colors">
                          {displayTitle}
                        </span>
                        {showFilename && (
                          <span className="text-xs text-muted-foreground/70 truncate block">
                            {result.file_name}
                          </span>
                        )}
                      </div>
                    </div>
                    <span className={cn(
                      "shrink-0 rounded-full border px-2 py-0.5 text-xs font-medium tabular-nums",
                      tier.className,
                    )}>
                      {result.reranker_score.toFixed(2)}
                    </span>
                  </div>
                  <div className="mt-2 text-sm text-muted-foreground leading-relaxed prose prose-sm dark:prose-invert max-w-none pointer-events-none line-clamp-4">
                    <Markdown remarkPlugins={[remarkGfm]}>
                      {highlightInMarkdown(snippet(result.original_text || result.chunk_text), query)}
                    </Markdown>
                  </div>
                  {kbName && (
                    <div className="mt-2 text-xs text-muted-foreground/70">
                      {kbName}
                    </div>
                  )}
                </a>
              );
            })}
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}
