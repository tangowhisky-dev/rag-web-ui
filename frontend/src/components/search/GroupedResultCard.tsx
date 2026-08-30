"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import {
  ChevronLeft,
  ChevronRight,
  FileText,
  FileCode,
  FileType,
  FileSpreadsheet,
  File,
} from "lucide-react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";

export interface SearchResult {
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

export interface GroupedSearchResult {
  documentId: number;
  fileName: string;
  title: string | null;
  kbId: number | null;
  dataStoreId: number | null;
  chunks: SearchResult[];
  bestScore: number;
  totalChunks: number;
}

interface GroupedResultCardProps {
  group: GroupedSearchResult;
  query: string;
  kbName?: string;
}

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

function cleanFilename(name: string): string {
  const stem = name.replace(/\.[^.]+$/, "").replace(/[_-]/g, " ").replace(/\./g, " ");
  return stem.replace(/\s+/g, " ").trim();
}

function highlightInMarkdown(text: string, query: string): string {
  const terms = query
    .split(/\s+/)
    .map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
    .filter((t) => t.length >= 2);
  if (terms.length === 0) return text;

  const regex = new RegExp(`\\b(${terms.join("|")})\\b`, "gi");
  return text.replace(regex, "**$1**");
}

export function groupResultsByDocument(results: SearchResult[]): GroupedSearchResult[] {
  const grouped = new Map<number, SearchResult[]>();

  for (const result of results) {
    const docId = result.document_id;
    if (!grouped.has(docId)) {
      grouped.set(docId, []);
    }
    grouped.get(docId)!.push(result);
  }

  const groupedResults: GroupedSearchResult[] = [];
  for (const [documentId, chunks] of grouped.entries()) {
    // Sort chunks by reranker_score (descending) within each group
    chunks.sort((a, b) => b.reranker_score - a.reranker_score);

    groupedResults.push({
      documentId,
      fileName: chunks[0].file_name,
      title: chunks[0].title,
      kbId: chunks[0].kb_id,
      dataStoreId: chunks[0].data_store_id,
      chunks,
      bestScore: chunks[0].reranker_score,
      totalChunks: chunks.length,
    });
  }

  // Sort groups by best score (descending)
  groupedResults.sort((a, b) => b.bestScore - a.bestScore);

  return groupedResults;
}

export default function GroupedResultCard({ group, query, kbName }: GroupedResultCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [visibleChunks, setVisibleChunks] = useState(3);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  // Responsive: show 2 chunks on mobile, 3 on tablet, 4 on desktop
  useEffect(() => {
    const updateVisible = () => {
      if (window.innerWidth < 768) setVisibleChunks(2);
      else if (window.innerWidth < 1024) setVisibleChunks(3);
      else setVisibleChunks(4);
    };
    updateVisible();
    window.addEventListener("resize", updateVisible);
    return () => window.removeEventListener("resize", updateVisible);
  }, []);

  const Icon = fileIcon(group.fileName);
  const tier = scoreTier(group.bestScore);
  const displayTitle = group.title || group.fileName;
  const showFilename = group.title && group.title !== cleanFilename(group.fileName);

  const canScrollLeft = currentIndex > 0;
  const canScrollRight = currentIndex < group.chunks.length - visibleChunks;

  const scrollLeft = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    if (currentIndex > 0) {
      const newIndex = currentIndex - 1;
      setCurrentIndex(newIndex);
      scrollContainerRef.current?.scrollTo({
        left: newIndex * (scrollContainerRef.current.clientWidth / visibleChunks),
        behavior: "smooth",
      });
    }
  }, [currentIndex, visibleChunks]);

  const scrollRight = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    if (currentIndex < group.chunks.length - visibleChunks) {
      const newIndex = currentIndex + 1;
      setCurrentIndex(newIndex);
      scrollContainerRef.current?.scrollTo({
        left: newIndex * (scrollContainerRef.current.clientWidth / visibleChunks),
        behavior: "smooth",
      });
    }
  }, [currentIndex, visibleChunks, group.chunks.length]);

  const handleMouseEnter = () => setIsExpanded(true);
  const handleMouseLeave = () => setIsExpanded(false);
  const handleFocus = () => setIsExpanded(true);
  const handleBlur = (e: React.FocusEvent) => {
    if (!e.currentTarget.contains(e.relatedTarget)) setIsExpanded(false);
  };

  return (
    <div
      className="rounded-lg border bg-card transition-all duration-300 ease-in-out hover:shadow-lg hover:scale-[1.01] group"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onFocus={handleFocus}
      onBlur={handleBlur}
      tabIndex={0}
    >
      {/* Header - always visible */}
      <div className="p-4">
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-center gap-2 min-w-0">
            <Icon className="h-4 w-4 text-muted-foreground shrink-0" />
            <div className="min-w-0">
              <span className="text-sm font-medium text-foreground truncate block group-hover:text-primary transition-colors">
                {displayTitle}
              </span>
              {showFilename && (
                <span className="text-xs text-muted-foreground/70 truncate block">
                  {group.fileName}
                </span>
              )}
              <span className="text-xs text-muted-foreground/50">
                {group.totalChunks} chunk{group.totalChunks !== 1 ? "s" : ""}
              </span>
            </div>
          </div>
          <span className={cn(
            "shrink-0 rounded-full border px-2 py-0.5 text-xs font-medium tabular-nums",
            tier.className,
          )}>
            {group.bestScore.toFixed(2)}
          </span>
        </div>
      </div>

      {/* Collapsed: show first chunk preview */}
      <div
        className={cn(
          "px-4 overflow-hidden transition-all duration-300 ease-in-out",
          isExpanded ? "max-h-0 opacity-0 pb-0" : "max-h-40 opacity-100 pb-4",
        )}
      >
        <div className="text-sm text-muted-foreground leading-relaxed prose prose-sm dark:prose-invert max-w-none pointer-events-none line-clamp-3">
          <Markdown remarkPlugins={[remarkGfm]}>
            {highlightInMarkdown(snippet(group.chunks[0].original_text || group.chunks[0].chunk_text), query)}
          </Markdown>
        </div>
      </div>

      {/* Expanded: horizontal carousel of all chunks */}
      <div
        className={cn(
          "overflow-hidden transition-all duration-300 ease-in-out border-t border-border/50",
          isExpanded ? "max-h-[400px] opacity-100" : "max-h-0 opacity-0 border-t-0",
        )}
      >
        <div className="relative">
          {/* Navigation buttons */}
          {group.chunks.length > visibleChunks && (
            <>
              <button
                onClick={scrollLeft}
                disabled={!canScrollLeft}
                className="absolute left-2 top-1/2 -translate-y-1/2 z-10 rounded-full bg-background/80 backdrop-blur-sm border p-1.5 text-muted-foreground hover:text-foreground hover:bg-background transition-all disabled:opacity-30 disabled:cursor-not-allowed shadow-sm"
                aria-label="Previous chunk"
              >
                <ChevronLeft className="h-4 w-4" />
              </button>
              <button
                onClick={scrollRight}
                disabled={!canScrollRight}
                className="absolute right-2 top-1/2 -translate-y-1/2 z-10 rounded-full bg-background/80 backdrop-blur-sm border p-1.5 text-muted-foreground hover:text-foreground hover:bg-background transition-all disabled:opacity-30 disabled:cursor-not-allowed shadow-sm"
                aria-label="Next chunk"
              >
                <ChevronRight className="h-4 w-4" />
              </button>
            </>
          )}

          {/* Carousel container */}
          <div
            ref={scrollContainerRef}
            className="flex overflow-x-auto snap-x snap-mandatory scrollbar-hide"
            style={{
              scrollbarWidth: "none",
              msOverflowStyle: "none",
            }}
          >
            {group.chunks.map((chunk, index) => {
              const chunkTier = scoreTier(chunk.reranker_score);
              return (
                <div
                  key={`${chunk.document_id}-${chunk.chunk_index}-${index}`}
                  className="flex-shrink-0 w-full snap-start"
                  style={{
                    width: `calc(100% / ${visibleChunks})`,
                  }}
                >
                  <div className="p-4 border-l border-r border-border/30 first:border-l-0 last:border-r-0 h-full">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs text-muted-foreground/70">
                        Chunk {index + 1} of {group.totalChunks}
                      </span>
                      <span className={cn(
                        "shrink-0 rounded-full border px-1.5 py-0.5 text-[10px] font-medium tabular-nums",
                        chunkTier.className,
                      )}>
                        {chunk.reranker_score.toFixed(2)}
                      </span>
                    </div>
                    <div className="text-sm text-muted-foreground leading-relaxed prose prose-sm dark:prose-invert max-w-none pointer-events-none line-clamp-4">
                      <Markdown remarkPlugins={[remarkGfm]}>
                        {highlightInMarkdown(snippet(chunk.original_text || chunk.chunk_text), query)}
                      </Markdown>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Footer - KB name */}
      {kbName && (
        <div className="px-4 pb-3 pt-0">
          <div className="text-xs text-muted-foreground/70">
            {kbName}
          </div>
        </div>
      )}
    </div>
  );
}
