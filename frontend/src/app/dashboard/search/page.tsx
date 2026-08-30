"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import {
  Search as SearchIcon,
  Info,
  Loader2,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Clock,
  Sparkles,
} from "lucide-react";
import DashboardLayout from "@/components/layout/dashboard-layout";
import { api, ApiError } from "@/lib/api";
import { useHydrated } from "@/lib/hooks";
import { cn } from "@/lib/utils";
import { useToast } from "@/components/ui/use-toast";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import GroupedResultCard, { groupResultsByDocument, type SearchResult } from "@/components/search/GroupedResultCard";

// ── Composite search + AI icon ───────────────────────────────────────────────
// A magnifying glass with a sparkle inside the lens, conveying AI-powered search.

function SearchAiIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 120 120"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      {/* Magnifying glass lens */}
      <circle
        cx="48"
        cy="48"
        r="34"
        stroke="currentColor"
        strokeWidth="6"
        strokeLinecap="round"
        opacity="0.35"
      />
      {/* Magnifying glass handle */}
      <line
        x1="74"
        y1="74"
        x2="104"
        y2="104"
        stroke="currentColor"
        strokeWidth="7"
        strokeLinecap="round"
        opacity="0.35"
      />
      {/* Four-point sparkle inside the lens (AI) */}
      <path
        d="M48 28 C50 38, 58 46, 68 48 C58 50, 50 58, 48 68 C46 58, 38 50, 28 48 C38 46, 46 38, 48 28 Z"
        fill="currentColor"
      />
      {/* Small accent sparkle */}
      <path
        d="M62 60 C63 64, 66 67, 70 68 C66 69, 63 72, 62 76 C61 72, 58 69, 54 68 C58 67, 61 64, 62 60 Z"
        fill="currentColor"
        opacity="0.5"
      />
    </svg>
  );
}

interface KbItem {
  id: number;
  name: string;
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
  const [groupedResults, setGroupedResults] = useState<ReturnType<typeof groupResultsByDocument>>([]);
  const [expandedQuery, setExpandedQuery] = useState("");
  const [hasSearched, setHasSearched] = useState(false);
  const [loading, setLoading] = useState(false);
  const [latencyMs, setLatencyMs] = useState(0);
  const [kbPickerOpen, setKbPickerOpen] = useState(false);
  const [page, setPage] = useState(0);
  const PAGE_SIZE = 10;

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
    setPage(0);
    try {
      const res = await api.post("/api/search", {
        query: searchQuery,
        kb_ids: selectedKbIds,
      }) as SearchResponse;
      setResults(res.results);
      setGroupedResults(groupResultsByDocument(res.results));
      setExpandedQuery(res.expanded_query);
      setLatencyMs(res.latency_ms);
    } catch (err) {
      if (err instanceof ApiError) {
        toast({ title: "Search failed", description: err.message, variant: "destructive" });
      } else {
        toast({ title: "Search failed", description: "Unexpected error", variant: "destructive" });
      }
      setResults([]);
      setGroupedResults([]);
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
      <div
        className={cn(
          "flex flex-col transition-all duration-500 ease-in-out",
          hasSearched
            ? "items-start pt-6"
            : "items-center justify-center min-h-[70vh]",
        )}
      >
        <div className="w-full max-w-3xl mx-auto">
          {/* Hero icon — only before search */}
          {!hasSearched && (
            <div className="flex flex-col items-center mb-8 animate-in fade-in zoom-in-95 duration-700">
              <SearchAiIcon className="h-24 w-24 text-primary" />
              <h1 className="mt-4 text-2xl font-semibold tracking-tight text-foreground">
                AI-powered search across your knowledge bases
              </h1>
              {/* <p className="mt-1 text-sm text-muted-foreground">
                AI-powered retrieval across your knowledge bases
              </p> */}
            </div>
          )}

          {/* Search bar — single input, Enter to submit */}
          <form
            onSubmit={(e) => { e.preventDefault(); handleSearch(); }}
            className={cn(
              "relative transition-all duration-500",
              hasSearched ? "mb-4" : "mb-6",
            )}
          >
          <div className="relative">
            <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Type + Enter to search…"
              className="w-full rounded-lg border bg-background pl-10 pr-10 py-2.5 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
              autoFocus
            />
            {/* Loading spinner or expanded-query indicator inside the input */}
            <div className="absolute right-2.5 top-1/2 -translate-y-1/2 flex items-center gap-1">
              {loading && <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />}
              {!loading && showExpandedTooltip && (
                <Popover>
                  <PopoverTrigger asChild>
                    <button
                      type="button"
                      title="Abbreviation glossary"
                      className="rounded-full p-0.5 text-muted-foreground hover:text-foreground transition-colors"
                    >
                      <Info className="h-4 w-4" />
                    </button>
                  </PopoverTrigger>
                  <PopoverContent side="bottom" align="end" className="max-w-md w-80 text-xs">
                    <div className="space-y-1.5">
                      <p className="font-medium text-foreground">Expanded query</p>
                      <p className="text-muted-foreground whitespace-pre-wrap break-words">{expandedQuery}</p>
                    </div>
                  </PopoverContent>
                </Popover>
              )}
            </div>
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
        {!loading && hasSearched && groupedResults.length === 0 && (
          <div className="text-center py-20 text-muted-foreground">
            <p className="text-sm">No results found for &ldquo;{query}&rdquo;.</p>
          </div>
        )}

        {/* Empty state — recent searches + LLM suggestions */}
        {!loading && !hasSearched && (
          <div className="py-10">
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
        {!loading && groupedResults.length > 0 && (() => {
          const totalPages = Math.ceil(groupedResults.length / PAGE_SIZE);
          const pageResults = groupedResults.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
          const startIdx = page * PAGE_SIZE + 1;
          const endIdx = Math.min((page + 1) * PAGE_SIZE, groupedResults.length);
          const totalChunks = results.length;
          return (
            <div className="space-y-3">
              <p className="text-xs text-muted-foreground tabular-nums">
                {groupedResults.length} document{groupedResults.length !== 1 ? "s" : ""} ({totalChunks} chunk{totalChunks !== 1 ? "s" : ""}) · {latencyMs}ms
              </p>
              {pageResults.map((group, i) => {
                const kbName = group.kbId
                  ? kbs.find((kb) => kb.id === group.kbId)?.name ?? `#${group.kbId}`
                  : group.dataStoreId
                    ? `DS #${group.dataStoreId}`
                    : null;
                return (
                  <a
                    key={`${group.documentId}-${i}`}
                    href={`/api/knowledge-base/documents/${group.documentId}/download`}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <GroupedResultCard group={group} query={query} kbName={kbName ?? undefined} />
                  </a>
                );
              })}
              {totalPages > 1 && (
                <div className="flex items-center justify-between pt-2">
                  <span className="text-xs text-muted-foreground tabular-nums">
                    {startIdx}–{endIdx} of {groupedResults.length} documents
                  </span>
                  <div className="flex items-center gap-1">
                    <button
                      type="button"
                      onClick={() => { setPage((p) => Math.max(0, p - 1)); window.scrollTo({ top: 0, behavior: "smooth" }); }}
                      disabled={page === 0}
                      className="rounded-md border p-1.5 text-muted-foreground hover:text-foreground hover:bg-muted transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                      aria-label="Previous page"
                    >
                      <ChevronLeft className="h-4 w-4" />
                    </button>
                    <span className="text-xs text-muted-foreground tabular-nums px-2">
                      {page + 1} / {totalPages}
                    </span>
                    <button
                      type="button"
                      onClick={() => { setPage((p) => Math.min(totalPages - 1, p + 1)); window.scrollTo({ top: 0, behavior: "smooth" }); }}
                      disabled={page >= totalPages - 1}
                      className="rounded-md border p-1.5 text-muted-foreground hover:text-foreground hover:bg-muted transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                      aria-label="Next page"
                    >
                      <ChevronRight className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              )}
            </div>
          );
        })()}
        </div>
      </div>
    </DashboardLayout>
  );
}
