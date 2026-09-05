/** Citation text utilities — used by answer rendering and tests. */

// Convert bare [N] citation markers to [N](N) markdown links so react-markdown
// renders them as clickable <a> elements handled by CitationLink.
// Skips [N] inside code blocks, headings, and already-linked [N](...) forms.
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
  // Replace [N] not followed by ( with [N](N)
  processed = processed.replace(/\[(\d+)\](?!\()/g, "[$1]($1)");
  // Restore code blocks
  processed = processed.replace(/\x00CODE(\d+)\x00/g, (_, i) => codeSegments[parseInt(i)]);
  return processed;
}
