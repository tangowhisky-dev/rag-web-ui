import { preprocessCitations } from "../citation-utils";
import type { CitationRef } from "@/types/chat";

describe("preprocessCitations", () => {
  it("converts bare [N] to [N](N) markdown links", () => {
    const result = preprocessCitations("Hello [1] world [2]");
    expect(result).toBe("Hello [1](1) world [2](2)");
  });

  it("does not touch already-linked [N](N) citations", () => {
    const result = preprocessCitations("See [1](1) for details");
    expect(result).toBe("See [1](1) for details");
  });

  it("does not touch [N] inside inline code", () => {
    const result = preprocessCitations("Code: `arr[1]` and [1]");
    expect(result).toContain("`arr[1]`");
    expect(result).toContain("[1](1)");
  });

  it("does not touch [N] inside fenced code blocks", () => {
    const result = preprocessCitations("```\narr[1]\n```\nText [1]");
    expect(result).toContain("arr[1]");
    expect(result).toContain("Text [1](1)");
  });

  it("handles empty string", () => {
    expect(preprocessCitations("")).toBe("");
  });

  it("handles multiple same-number citations", () => {
    const result = preprocessCitations("Foo [1]. Bar [1].");
    expect(result).toBe("Foo [1](1). Bar [1](1).");
  });
});

describe("CitationRef kind metadata", () => {
  it("supports all citation_kind values", () => {
    const kinds: CitationRef["citation_kind"][] = [
      "chunk", "file", "section", "range", "grep", "table", "outline",
    ];
    for (const kind of kinds) {
      const ref: CitationRef = { citation_kind: kind, document_id: 1, source_tool: "search_dense" };
      expect(ref.citation_kind).toBe(kind);
    }
  });

  it("carries line-range fields for range citations", () => {
    const ref: CitationRef = {
      citation_kind: "range",
      document_id: 1,
      start_line: 10,
      end_line: 20,
      source_tool: "kb_read",
    };
    expect(ref.start_line).toBe(10);
    expect(ref.end_line).toBe(20);
  });

  it("carries match_line for grep citations", () => {
    const ref: CitationRef = {
      citation_kind: "grep",
      document_id: 1,
      match_line: 42,
      source_tool: "kb_grep",
    };
    expect(ref.match_line).toBe(42);
  });

  it("carries quoted_text for snippet display", () => {
    const ref: CitationRef = {
      citation_kind: "chunk",
      document_id: 1,
      chunk_index: 3,
      quoted_text: "A mutex is a synchronization primitive...",
      source_tool: "search_dense",
    };
    expect(ref.quoted_text).toContain("mutex");
  });
});
