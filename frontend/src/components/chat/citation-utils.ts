/** Citation text utilities — used by answer rendering and tests. */

// Convert citation markers to [N](N) markdown links so react-markdown
// renders them as clickable <a> elements handled by CitationLink.
// Handles two forms the LLM emits:
//   [E1]  — evidence-label format (used during streaming before backend normalization)
//   [N]   — bare numeric format (after backend normalization, or LLM fallback)
// Skips citations inside code blocks and already-linked [N](...) forms.
export function preprocessCitations(text: string): string {
  if (!text) return text;
  // Extract code blocks and inline code first to avoid replacing inside them.
  const codeSegments: string[] = [];
  let processed = text.replace(/```[\s\S]*?```/g, (m) => {
    codeSegments.push(m);
    return `\x00CODE${codeSegments.length - 1}\x00`;
  });
  processed = processed.replace(/`[^`]*`/g, (m) => {
    codeSegments.push(m);
    return `\x00CODE${codeSegments.length - 1}\x00`;
  });
  // Replace [EN] not followed by ( with [N](N)
  processed = processed.replace(/\[E(\d+)\](?!\()/gi, "[$1]($1)");
  // Replace [N] not followed by ( with [N](N)
  processed = processed.replace(/\[(\d+)\](?!\()/g, "[$1]($1)");
  // Restore code blocks
  processed = processed.replace(/\x00CODE(\d+)\x00/g, (_, i) => codeSegments[parseInt(i)]);
  return processed;
}
