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
import rehypeRaw from "rehype-raw";
import { cn } from "@/lib/utils";

// The result card is wrapped in an <a download> link. Markdown content may
// contain raw URLs / mailto: links that render as nested <a> tags, causing
// hydration errors. Override <a> to render as a plain <span>.
const markdownComponents = {
  a: ({ children }: React.AnchorHTMLAttributes<HTMLAnchorElement>) => (
    <span>{children}</span>
  ),
};

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

const FILE_ICON_MAP: Record<string, typeof File> = {
  py: FileCode, js: FileCode, ts: FileCode, tsx: FileCode, jsx: FileCode,
  go: FileCode, rs: FileCode, java: FileCode, c: FileCode, cpp: FileCode,
  rb: FileCode, sh: FileCode, yaml: FileCode, yml: FileCode, json: FileCode, xml: FileCode,
  pdf: FileType,
  xls: FileSpreadsheet, xlsx: FileSpreadsheet, csv: FileSpreadsheet,
  md: FileText, txt: FileText, rtf: FileText,
};

function FileIcon({ name, ...props }: { name: string } & React.SVGProps<SVGSVGElement>) {
  const ext = name.split(".").pop()?.toLowerCase() ?? "";
  const Icon = FILE_ICON_MAP[ext] ?? File;
  return <Icon {...props} />;
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

const STOP_WORDS = new Set([
  "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
  "been", "being", "have", "has", "had", "do", "does", "did", "will",
  "would", "could", "should", "may", "might", "must", "can", "of", "in",
  "on", "at", "to", "for", "with", "by", "from", "as", "that", "this",
  "these", "those", "it", "its", "if", "then", "than", "so", "no", "not",
]);

function buildHighlightPatterns(query: string): string[] {
  const raw = query.split(/\s+/).filter(Boolean);
  if (raw.length === 0) return [];

  // Strip leading and trailing stop words, keep stop words between keywords.
  let start = 0;
  let end = raw.length;
  while (start < end && STOP_WORDS.has(raw[start].toLowerCase())) start++;
  while (end > start && STOP_WORDS.has(raw[end - 1].toLowerCase())) end--;
  const trimmed = raw.slice(start, end);
  if (trimmed.length === 0) return [];

  const esc = (t: string) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

  const patterns: string[] = [];
  // Full phrase (stop words kept between keywords).
  patterns.push(esc(trimmed.join(" ")));
  // Individual non-stop terms (length >= 2).
  for (const t of trimmed) {
    if (t.length >= 2 && !STOP_WORDS.has(t.toLowerCase())) {
      patterns.push(esc(t));
    }
  }
  return patterns;
}

function highlightInMarkdown(text: string, query: string): string {
  // Strip markdown links — keep only the link text so the result card's
  // outer <a> wrapper doesn't get nested <a> tags from rendered markdown.
  text = text.replace(/\[([^\]]+)\]\([^)]+\)/g, "$1");

  const patterns = buildHighlightPatterns(query);
  if (patterns.length === 0) return text;

  // Longer patterns first so the regex engine matches the full phrase
  // before falling back to individual terms at the same position.
  patterns.sort((a, b) => b.length - a.length);
  const regex = new RegExp(`\\b(${patterns.join("|")})\\b`, "gi");
  return text.replace(regex, '<mark class="search-hit">$1</mark>');
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

const VISIBLE_CHUNKS_BREAKPOINTS = [2, 3, 4] as const;

function getVisibleChunks(width: number): number {
  return VISIBLE_CHUNKS_BREAKPOINTS[Number(width >= 768) + Number(width >= 1024)];
}

function useChunkScroll(
  totalChunks: number,
  visibleChunks: number,
  scrollContainerRef: React.RefObject<HTMLDivElement | null>,
) {
  const [currentIndex, setCurrentIndex] = useState(0);

  const canScrollLeft = currentIndex > 0;
  const canScrollRight = currentIndex < totalChunks - visibleChunks;

  const scrollLeft = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setCurrentIndex((prev) => {
      if (prev <= 0) return prev;
      const newIndex = prev - 1;
      scrollContainerRef.current?.scrollTo({
        left: newIndex * (scrollContainerRef.current.clientWidth / visibleChunks),
        behavior: "smooth",
      });
      return newIndex;
    });
  }, [visibleChunks, scrollContainerRef, setCurrentIndex]);

  const scrollRight = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setCurrentIndex((prev) => {
      if (prev >= totalChunks - visibleChunks) return prev;
      const newIndex = prev + 1;
      scrollContainerRef.current?.scrollTo({
        left: newIndex * (scrollContainerRef.current.clientWidth / visibleChunks),
        behavior: "smooth",
      });
      return newIndex;
    });
  }, [totalChunks, visibleChunks, scrollContainerRef, setCurrentIndex]);

  return { canScrollLeft, canScrollRight, scrollLeft, scrollRight };
}

interface ChunkPreviewProps {
  chunk: SearchResult;
  index: number;
  totalChunks: number;
  visibleChunks: number;
  query: string;
}

function ChunkPreview({ chunk, index, totalChunks, visibleChunks, query }: ChunkPreviewProps) {
  const chunkTier = scoreTier(chunk.reranker_score);
  return (
    <div
      className="flex-shrink-0 w-full snap-start"
      style={{
        width: `calc(100% / ${visibleChunks})`,
      }}
    >
      <div className="p-4 border-l border-r border-border/30 first:border-l-0 last:border-r-0 h-full">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs text-muted-foreground/70">
            Chunk {index + 1} of {totalChunks}
          </span>
          <span className={cn(
            "shrink-0 rounded-full border px-1.5 py-0.5 text-[10px] font-medium tabular-nums",
            chunkTier.className,
          )}>
            {chunk.reranker_score.toFixed(2)}
          </span>
        </div>
        <div className="text-sm text-muted-foreground leading-relaxed prose prose-sm dark:prose-invert max-w-none pointer-events-none line-clamp-4">
          <Markdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]} components={markdownComponents}>
            {highlightInMarkdown(snippet(chunk.original_text || chunk.chunk_text), query)}
          </Markdown>
        </div>
      </div>
    </div>
  );
}

export default function GroupedResultCard({ group, query, kbName }: GroupedResultCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [visibleChunks, setVisibleChunks] = useState(3);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  // Responsive: show 2 chunks on mobile, 3 on tablet, 4 on desktop
  useEffect(() => {
    const updateVisible = () => setVisibleChunks(getVisibleChunks(window.innerWidth));
    updateVisible();
    window.addEventListener("resize", updateVisible);
    return () => window.removeEventListener("resize", updateVisible);
  }, []);

  const tier = scoreTier(group.bestScore);
  const displayTitle = group.title || group.fileName;
  const showFilename = group.title && group.title !== cleanFilename(group.fileName);

  const { canScrollLeft, canScrollRight, scrollLeft, scrollRight } =
    useChunkScroll(group.chunks.length, visibleChunks, scrollContainerRef);

  const handleMouseEnter = () => setIsExpanded(true);
  const handleMouseLeave = () => setIsExpanded(false);

  return (
    <div
      className="rounded-lg border border-border bg-card transition-all duration-300 ease-in-out hover:shadow-lg hover:scale-[1.01] hover:border-blue-400/50 group"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {/* Header - always visible */}
      <div className="p-4">
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-center gap-2 min-w-0">
            <FileIcon name={group.fileName} className="h-4 w-4 text-muted-foreground shrink-0" />
            <div className="min-w-0">
              <span className="text-sm font-medium text-foreground truncate block group-hover:text-primary transition-colors">
                {displayTitle}
              </span>
              {showFilename && (
                <span className="text-xs text-muted-foreground/70 truncate block">
                  {group.fileName}
                </span>
              )}
            </div>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <span className="text-xs text-muted-foreground/50 tabular-nums">
              {group.totalChunks} chunk{["", "s"][Number(group.totalChunks !== 1)]}
            </span>
            <span className={cn(
              "shrink-0 rounded-full border px-2 py-0.5 text-xs font-medium tabular-nums",
              tier.className,
            )}>
              {group.bestScore.toFixed(2)}
            </span>
          </div>
        </div>
      </div>

      {/* Collapsed: show first chunk preview */}
      <div
        className={cn(
          "px-4 overflow-hidden transition-all duration-300 ease-in-out",
          ["max-h-40 opacity-100 pb-4", "max-h-0 opacity-0 pb-0"][Number(isExpanded)],
        )}
      >
        <div className="text-sm text-muted-foreground leading-relaxed prose prose-sm dark:prose-invert max-w-none pointer-events-none line-clamp-3">
          <Markdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]} components={markdownComponents}>
            {highlightInMarkdown(snippet(group.chunks[0].original_text || group.chunks[0].chunk_text), query)}
          </Markdown>
        </div>
      </div>

      {/* Expanded: horizontal carousel of all chunks */}
      <div
        className={cn(
          "overflow-hidden transition-all duration-300 ease-in-out border-t border-border/50",
          ["max-h-0 opacity-0 border-t-0", "max-h-[400px] opacity-100"][Number(isExpanded)],
        )}
      >
        <div className="relative">
          {/* Navigation buttons */}
          {group.chunks.length > visibleChunks && (
            <>
              <button
                onClick={scrollLeft}
                onPointerDown={(e) => { e.preventDefault(); e.stopPropagation(); }}
                className={cn(
                  "absolute left-2 top-1/2 -translate-y-1/2 z-10 rounded-full bg-background/80 backdrop-blur-sm border p-1.5 text-muted-foreground hover:text-foreground hover:bg-background transition-all shadow-sm",
                  !canScrollLeft && "opacity-30 cursor-not-allowed hover:text-muted-foreground hover:bg-background/80",
                )}
                aria-label="Previous chunk"
                aria-disabled={!canScrollLeft}
              >
                <ChevronLeft className="h-4 w-4" />
              </button>
              <button
                onClick={scrollRight}
                onPointerDown={(e) => { e.preventDefault(); e.stopPropagation(); }}
                className={cn(
                  "absolute right-2 top-1/2 -translate-y-1/2 z-10 rounded-full bg-background/80 backdrop-blur-sm border p-1.5 text-muted-foreground hover:text-foreground hover:bg-background transition-all shadow-sm",
                  !canScrollRight && "opacity-30 cursor-not-allowed hover:text-muted-foreground hover:bg-background/80",
                )}
                aria-label="Next chunk"
                aria-disabled={!canScrollRight}
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
            {group.chunks.map((chunk, index) => (
              <ChunkPreview
                key={`${chunk.document_id}-${chunk.chunk_index}-${index}`}
                chunk={chunk}
                index={index}
                totalChunks={group.totalChunks}
                visibleChunks={visibleChunks}
                query={query}
              />
            ))}
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
