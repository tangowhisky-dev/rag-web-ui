/**
 * Shared text-highlighting utilities.
 *
 * Used by:
 * - Search page result cards (highlight query terms in source snippets)
 * - Chat citation popovers (highlight cited keywords in source text)
 *
 * The approach is keyword-level: extract distinctive terms from a query
 * (or from cited answer sentences), strip stop words, then wrap matches
 * in <mark class="search-hit"> tags. The CSS class is defined in globals.css.
 */

// Common English stop words — filtered from highlight patterns to avoid
// over-highlighting. Stop words *between* meaningful terms are preserved
// in the full-phrase pattern so multi-word phrases still match intact.
export const STOP_WORDS = new Set([
  "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
  "been", "being", "have", "has", "had", "do", "does", "did", "will",
  "would", "could", "should", "may", "might", "must", "can", "of", "in",
  "on", "at", "to", "for", "with", "by", "from", "as", "that", "this",
  "these", "those", "it", "its", "if", "then", "than", "so", "no", "not",
]);

/**
 * Build regex patterns from a query string for highlighting.
 *
 * Returns an array of escaped regex strings, longest first:
 * 1. The full phrase (stop words kept between keywords)
 * 2. Individual non-stop terms (length >= 2)
 */
export function buildHighlightPatterns(query: string): string[] {
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

/**
 * Strip leading/trailing punctuation from a word. Keeps internal hyphens
 * (e.g. "Ciphertext-Only") and apostrophes (e.g. "don't").
 */
function stripPunctuation(word: string): string {
  return word.replace(/^[^\w]+|[^\w]+$/g, "");
}

/**
 * Build highlight patterns from a cited answer sentence.
 *
 * Unlike buildHighlightPatterns (designed for short search queries), this
 * handles long cited sentences by:
 * 1. Stripping punctuation from each word (so "COMMAND:" matches "COMMAND")
 * 2. Building all n-gram phrases (2-5 words) where the n-gram starts and
 *    ends with a non-stop word, keeping stop words in the middle
 * 3. Also including individual non-stop terms
 *
 * This ensures multi-word terms like "degrees of freedom" or
 * "CIPHERING MODE COMMAND" are matched as phrases even when embedded
 * in a longer cited sentence.
 */
export function buildCitationPatterns(citedText: string): string[] {
  if (!citedText) return [];

  // Split into words and strip punctuation from each.
  const rawWords = citedText.split(/\s+/).filter(Boolean);
  const words = rawWords.map(stripPunctuation).filter((w) => w.length > 0);
  if (words.length === 0) return [];

  const esc = (t: string) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const seen = new Set<string>();
  const patterns: string[] = [];

  const add = (p: string) => {
    if (!seen.has(p)) {
      seen.add(p);
      patterns.push(p);
    }
  };

  // Build n-gram phrases of size 2-5 where the phrase starts and ends
  // with a non-stop word. Stop words are kept in the middle so phrases
  // like "degrees of freedom" match intact.
  const MAX_NGRAM = 5;
  for (let n = 2; n <= MAX_NGRAM; n++) {
    for (let i = 0; i <= words.length - n; i++) {
      const ngram = words.slice(i, i + n);
      // Skip if starts or ends with a stop word.
      if (STOP_WORDS.has(ngram[0].toLowerCase())) continue;
      if (STOP_WORDS.has(ngram[ngram.length - 1].toLowerCase())) continue;
      // Must contain at least 2 non-stop words to be meaningful.
      const nonStopCount = ngram.filter((w) => !STOP_WORDS.has(w.toLowerCase())).length;
      if (nonStopCount < 2) continue;
      add(esc(ngram.join(" ")));
    }
  }

  // Individual non-stop terms (length >= 2).
  for (const w of words) {
    if (w.length >= 2 && !STOP_WORDS.has(w.toLowerCase())) {
      add(esc(w));
    }
  }

  return patterns;
}

/**
 * Highlight query terms in a text string by wrapping matches in
 * <mark class="search-hit"> tags. Strips markdown links first to
 * avoid nested <a> tags when the result is rendered inside an outer
 * anchor wrapper (search result cards).
 *
 * For citation popovers, pass `stripMarkdownLinks=false` to preserve
 * markdown link syntax so react-markdown can render links normally.
 *
 * Pass `patterns` to use pre-built patterns (e.g. from buildCitationPatterns)
 * instead of deriving them from `query` via buildHighlightPatterns.
 */
export function highlightInMarkdown(
  text: string,
  query: string,
  opts?: { stripMarkdownLinks?: boolean; patterns?: string[] },
): string {
  if (opts?.stripMarkdownLinks !== false) {
    // Strip markdown links — keep only the link text so the result card's
    // outer <a> wrapper doesn't get nested <a> tags from rendered markdown.
    text = text.replace(/\[([^\]]+)\]\([^)]+\)/g, "$1");
  }

  const patterns = opts?.patterns ?? buildHighlightPatterns(query);
  if (patterns.length === 0) return text;

  // Longer patterns first so the regex engine matches the full phrase
  // before falling back to individual terms at the same position.
  patterns.sort((a, b) => b.length - a.length);
  const regex = new RegExp(`\\b(${patterns.join("|")})\\b`, "gi");
  return text.replace(regex, '<mark class="search-hit">$1</mark>');
}

/**
 * Extract the preceding text context for each citation in the answer.
 *
 * Works by splitting the answer text into segments at citation marker
 * boundaries. Each citation marker gets the text segment preceding it
 * (from the previous citation marker, or start of text). This is the
 * text the LLM wrote that cites that source — the keywords in it are
 * what should be highlighted in the source chunk.
 *
 * Handles all citation formats:
 * - [N]        — bare numeric (streaming, before backend normalization)
 * - [EN]       — evidence label (streaming)
 * - [N](N)     — markdown link (after backend normalization, on reload)
 * - [EN](N)    — evidence label link (after backend normalization)
 *
 * Handles adjacent citation markers with various separators:
 * - [1][2]     — no space
 * - [1] [2]    — space
 * - [1], [2]   — comma + space
 * - [1],[2]    — comma no space
 *
 * For adjacent markers where the segment between them is empty or just
 * punctuation/whitespace, the citation inherits the previous non-empty
 * segment (so [1][2] at end of a sentence both get the same context).
 *
 * Returns a map: citationId -> concatenated context text (capped at 500 chars).
 */
export function extractCitedContext(answerText: string): Record<number, string> {
  if (!answerText) return {};

  // Remove code blocks and inline code to avoid parsing citations inside them.
  const codeSegments: string[] = [];
  let processed = answerText.replace(/```[\s\S]*?```/g, (m) => {
    codeSegments.push(m);
    return `\x00CODE${codeSegments.length - 1}\x00`;
  });
  processed = processed.replace(/`[^`]*`/g, (m) => {
    codeSegments.push(m);
    return `\x00CODE${codeSegments.length - 1}\x00`;
  });

  // Find all citation markers with their positions.
  // Matches [N], [EN], [N](N), [EN](N) — captures the numeric ID.
  const markerRegex = /\[E?(\d+)\](?:\(\d+\))?/gi;
  const markers: Array<{ id: number; start: number; end: number }> = [];
  let match: RegExpExecArray | null;
  while ((match = markerRegex.exec(processed)) !== null) {
    markers.push({
      id: parseInt(match[1]),
      start: match.index,
      end: match.index + match[0].length,
    });
  }

  if (markers.length === 0) return {};

  // For each marker, extract the preceding text segment.
  // The segment is the text from the end of the previous marker (or start
  // of text) to the start of this marker. If the segment is empty or just
  // punctuation/whitespace (adjacent markers), inherit the previous
  // non-empty segment so both citations share the same context.
  const result: Record<number, string> = {};
  let lastNonEmptySegment = "";

  for (let i = 0; i < markers.length; i++) {
    const marker = markers[i];
    const segmentStart = i > 0 ? markers[i - 1].end : 0;
    const segmentEnd = marker.start;
    let segment = processed.slice(segmentStart, segmentEnd);

    // Strip any remaining citation marker fragments and clean up.
    segment = segment
      .replace(/\[E?\d+\](?:\(\d+\))?/gi, "")
      .replace(/\(\d+\)/g, "") // stray (N) artifacts from partial matching
      .trim();

    // If segment is empty or just punctuation/whitespace (adjacent markers
    // separated by comma, space, etc.), inherit the previous non-empty segment.
    if (!segment || /^[,\s;:.\-]+$/.test(segment)) {
      segment = lastNonEmptySegment;
    } else {
      lastNonEmptySegment = segment;
    }

    if (!segment) continue;

    if (!result[marker.id]) {
      result[marker.id] = segment;
    } else {
      result[marker.id] += " " + segment;
    }
  }

  return result;
}
