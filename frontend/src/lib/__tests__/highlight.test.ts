import {
  buildHighlightPatterns,
  highlightInMarkdown,
  extractCitedContext,
  buildCitationPatterns,
} from "@/lib/highlight";

describe("buildHighlightPatterns", () => {
  it("returns empty for empty query", () => {
    expect(buildHighlightPatterns("")).toEqual([]);
  });

  it("returns empty for stop-words-only query", () => {
    expect(buildHighlightPatterns("the is a")).toEqual([]);
  });

  it("strips leading and trailing stop words", () => {
    const patterns = buildHighlightPatterns("the vulnerabilities of GMR-2");
    // Full phrase should keep "of" between keywords
    expect(patterns).toContain("vulnerabilities of GMR-2");
    expect(patterns).toContain("vulnerabilities");
    expect(patterns).toContain("GMR-2");
  });

  it("escapes regex special characters", () => {
    const patterns = buildHighlightPatterns("arr[0]");
    // The brackets should be escaped
    expect(patterns.some((p) => p.includes("\\["))).toBe(true);
  });
});

describe("highlightInMarkdown", () => {
  it("wraps matching terms in mark tags", () => {
    const result = highlightInMarkdown("The solar power is great", "solar power");
    expect(result).toContain('<mark class="search-hit">solar power</mark>');
  });

  it("highlights individual terms when full phrase absent", () => {
    const result = highlightInMarkdown("Solar energy and wind power", "solar power");
    expect(result).toContain('<mark class="search-hit">Solar</mark>');
    expect(result).toContain('<mark class="search-hit">power</mark>');
  });

  it("returns text unchanged when no patterns", () => {
    const result = highlightInMarkdown("Hello world", "");
    expect(result).toBe("Hello world");
  });

  it("strips markdown links by default", () => {
    const result = highlightInMarkdown("[click here](http://example.com) for info", "info");
    expect(result).not.toContain("http://example.com");
    expect(result).toContain("click here");
  });

  it("preserves markdown links when stripMarkdownLinks=false", () => {
    const result = highlightInMarkdown("[click here](http://example.com)", "click", {
      stripMarkdownLinks: false,
    });
    expect(result).toContain("](http://example.com)");
    expect(result).toContain('<mark class="search-hit">click</mark>');
  });
});

describe("extractCitedContext", () => {
  it("returns empty for empty text", () => {
    expect(extractCitedContext("")).toEqual({});
  });

  it("extracts text preceding a citation marker", () => {
    const result = extractCitedContext("Solar power converts sunlight into electricity [1].");
    expect(result[1]).toContain("Solar power converts sunlight into electricity");
    expect(result[1]).not.toContain("[1]");
  });

  it("handles [E1] evidence-label format", () => {
    const result = extractCitedContext("The system uses photovoltaic cells [E1].");
    expect(result[1]).toContain("photovoltaic cells");
  });

  it("handles [N](N) markdown link format (after backend normalization, on reload)", () => {
    const result = extractCitedContext(
      "Solar power converts sunlight into electricity [1](1)."
    );
    expect(result[1]).toContain("Solar power converts sunlight into electricity");
    expect(result[1]).not.toContain("[1]");
    expect(result[1]).not.toContain("(1)");
  });

  it("handles [EN](N) evidence-label link format", () => {
    const result = extractCitedContext(
      "The system uses photovoltaic cells [E1](1)."
    );
    expect(result[1]).toContain("photovoltaic cells");
  });

  it("handles adjacent citations with no space [1][2]", () => {
    const result = extractCitedContext("Hybrid systems combine both technologies [1][2].");
    expect(result[1]).toContain("Hybrid systems combine both technologies");
    // [2] is adjacent to [1] so it inherits the same segment
    expect(result[2]).toContain("Hybrid systems combine both technologies");
  });

  it("handles adjacent citations with space [1] [2]", () => {
    const result = extractCitedContext("Hybrid systems combine both technologies [1] [2].");
    expect(result[1]).toContain("Hybrid systems combine both technologies");
    expect(result[2]).toContain("Hybrid systems combine both technologies");
  });

  it("handles adjacent citations with comma and space [1], [2]", () => {
    const result = extractCitedContext("Hybrid systems combine both technologies [1], [2].");
    expect(result[1]).toContain("Hybrid systems combine both technologies");
    expect(result[2]).toContain("Hybrid systems combine both technologies");
  });

  it("handles adjacent citations with comma no space [1],[2]", () => {
    const result = extractCitedContext("Hybrid systems combine both technologies [1],[2].");
    expect(result[1]).toContain("Hybrid systems combine both technologies");
    expect(result[2]).toContain("Hybrid systems combine both technologies");
  });

  it("handles multiple segments citing the same source", () => {
    const result = extractCitedContext(
      "Solar is abundant [1]. Wind is variable [1]. Hydro is stable [2]."
    );
    expect(result[1]).toContain("Solar is abundant");
    expect(result[1]).toContain("Wind is variable");
    expect(result[2]).toContain("Hydro is stable");
  });

  it("ignores citations inside code blocks", () => {
    const result = extractCitedContext("```\narr[1]\n```\nText here [1].");
    expect(result[1]).toContain("Text here");
    expect(result[1]).not.toContain("arr[1]");
  });

  it("does not cap segment length", () => {
    const longSentence = "This is a very long sentence about solar power. ".repeat(20);
    const result = extractCitedContext(`${longSentence}[1]. More text [1].`);
    expect(result[1].length).toBeGreaterThan(500);
  });

  it("works with mixed [N] and [N](N) formats in same text", () => {
    const result = extractCitedContext(
      "First claim [1](1). Second claim [2]. Third claim [3](3)."
    );
    expect(result[1]).toContain("First claim");
    expect(result[2]).toContain("Second claim");
    expect(result[3]).toContain("Third claim");
  });
});

describe("buildCitationPatterns", () => {
  it("returns empty for empty text", () => {
    expect(buildCitationPatterns("")).toEqual([]);
  });

  it("strips punctuation from words so they match clean source text", () => {
    const patterns = buildCitationPatterns("Attack on CIPHERING MODE COMMAND: vulnerability");
    // The colon after COMMAND should be stripped, so the phrase pattern
    // should be "CIPHERING MODE COMMAND" without the colon.
    expect(patterns).toContain("CIPHERING MODE COMMAND");
  });

  it("builds n-gram sub-phrases, not just full-sentence + individual words", () => {
    const patterns = buildCitationPatterns(
      "Attack on CIPHERING MODE COMMAND: targeting limited degrees of freedom"
    );
    // "degrees of freedom" is a 3-gram with stop word "of" in the middle.
    expect(patterns).toContain("degrees of freedom");
  });

  it("strips quotes from quoted terms", () => {
    const patterns = buildCitationPatterns('exploits "CIPHERING MODE COMMAND" vulnerability');
    expect(patterns).toContain("CIPHERING MODE COMMAND");
  });

  it("deduplicates patterns", () => {
    const patterns = buildCitationPatterns("solar power, solar power");
    const phraseCount = patterns.filter((p) => p === "solar power").length;
    expect(phraseCount).toBe(1);
  });

  it("preserves internal hyphens in terms", () => {
    const patterns = buildCitationPatterns("Ciphertext-Only attack");
    expect(patterns).toContain("Ciphertext-Only");
  });

  it("handles multi-citation sentences with clause boundaries", () => {
    const patterns = buildCitationPatterns(
      "Ciphertext-Only Attack on CIPHERING MODE COMMAND: This attack exploits a vulnerability in the CIPHERING MODE COMMAND message type, specifically targeting its limited degrees of freedom"
    );
    expect(patterns).toContain("CIPHERING MODE COMMAND");
    expect(patterns).toContain("degrees of freedom");
    expect(patterns).toContain("Ciphertext-Only Attack");
  });

  it("does not build n-grams starting or ending with stop words", () => {
    const patterns = buildCitationPatterns("the solar power is great");
    // "the solar" should not be a pattern (starts with stop word)
    expect(patterns).not.toContain("the solar");
    // "power is" should not be a pattern (ends with stop word)
    expect(patterns).not.toContain("power is");
    // "solar power" should be a pattern
    expect(patterns).toContain("solar power");
  });
});
