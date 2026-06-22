# Answering Modes

RAG Web UI supports three answering modes, selectable via the pill bar in the chat input.

---

## ⚡ Fast

**Model:** `OPENAI_MODEL`
**Typical latency:** 3–8 s
**LLM calls per turn:** 2 (rewrite + answer)

### Pipeline

```
rewrite_query
  → hybrid_search_with_legs (all legs in parallel: dense + sparse + exact)
  → stream LLM answer
```

### When to use

Quick factual lookups, single-answer questions, conversational follow-ups.

### Agent timeline steps shown

| Step | Active label | Done detail |
|---|---|---|
| Rewrite Query | "Rewriting query…" | Rewritten query text |
| Retrieved Context | "Retrieving context…" | N docs, chunk previews with source |
| Additional Context | "Fetching graph context…" | Graph doc counts + context lines (only if Neo4j returned data) |
| Generating Answer | "Generating answer…" | Token usage |

---

## 🧠 Thinking

**Model:** `REASONING_MODEL` (falls back to `OPENAI_MODEL` when unset)
**Typical latency:** 5–15 s (depends on reasoning depth)
**LLM calls per turn:** 2 (rewrite + answer)

Identical pipeline to Fast — only the model changes. Use `REASONING_MODEL` to point at a reasoning/chain-of-thought model (e.g. `o3-mini`, `qwen3-thinking`). The model's internal reasoning is not streamed to the UI.

---

## 🤖 Agentic

**Model:** `OPENAI_MODEL` (with `QUERY_MODEL` for rewrite/decompose/grade)
**Typical latency:** 15–40 s (varies with sub-query count and retry depth)
**LLM calls per turn:** 5–12 (rewrite + route + decompose + parallel retrieval + draft + grade × up to 3 + generate)

### Pipeline

```
rewrite_query
  → context_router          (smart source routing: kb / file / both)
  → decompose_query         (2–5 atomic sub-queries)
  → parallel_retrieval      (hybrid search per sub-query, reinforced dedup)
  → extract_file_sections   (LLM selects relevant file sections — agentic only)
  → draft_answer            (per-sub-query draft for grading)
  → grade_coverage          (✓/~/✗ per sub-query)
  → [if uncovered, attempt 0] widened_retrieval (reranker −5.0)
  → [if still uncovered, attempt 1] keyword_search_loop (broad → narrow FULLTEXT)
  → generate_answer (with partial-answer transparency if sub-queries remain uncovered)
```

### What makes Agentic different

| Capability | Fast | Thinking | Agentic |
|---|---|---|---|
| Source routing (file vs KB vs both) | ❌ always both | ❌ always both | ✅ LLM-adaptive |
| Sub-query decomposition | ❌ single query | ❌ single query | ✅ 2–5 sub-queries |
| Reinforced scoring | ❌ | ❌ | ✅ multi-query dedup |
| File section extraction | ❌ full content | ❌ full content | ✅ LLM selects 3–6 sections |
| Draft-grade loop | ❌ | ❌ | ✅ graded before final answer |
| Retry on bad coverage | ❌ | ❌ | ✅ widened → keyword fallback |
| Partial-answer transparency | ❌ | ❌ | ✅ states what couldn't be found |

### Retry escalation

1. **Attempt 0 — parallel retrieval**: standard hybrid search per sub-query.
2. **Attempt 1 — widened retrieval**: re-retrieves *only for uncovered sub-queries*; reranker threshold relaxed from −2.0 to −5.0, accepting weaker matches.
3. **Attempt 2 — keyword search loop**: LLM extracts broad (3–4 terms) and narrow (1–2 compound phrases) keywords per uncovered sub-query; runs MySQL FULLTEXT directly; broad first, narrow if broad finds nothing; max 3 sub-queries × 2 iterations = 6 searches ceiling.
4. **Final answer**: generated from all accumulated context; uncovered sub-queries explicitly noted if retries exhausted.

### Agent timeline steps shown

| Step | Active label | Done detail |
|---|---|---|
| Rewrite Query | "Rewriting query…" | Rewritten query text |
| Context Routing | "Routing sources…" | Route decision + rationale |
| Sub-queries | "Decomposing query…" | Numbered list of sub-queries |
| Retrieved Context | "Retrieving for each sub-query…" | N docs, chunk previews with reinforcement count |
| File Sections | "Extracting file sections…" | Sections kept / total |
| Draft Answer | "Drafting answer…" | Draft character count |
| Coverage Check | "Checking coverage…" | ✓/~/✗ per sub-query |
| Widened Search | "Widening search…" | New docs found, threshold used, uncovered sub-queries |
| Keyword Search | "Searching keywords…" | Per-iteration: keywords used + results found |
| Generating Answer | "Generating answer…" | Token usage, partial flag |

### When to use

- Multi-part questions: "Compare X and Y, then explain why Z"
- Research queries spanning multiple documents
- Ambiguous questions where keyword matching alone would miss relevant content
- Queries where you need to know what _wasn't_ found

---

## Switching modes mid-conversation

Mode is stateless — each message can use a different mode. The mode selector persists in React state but is not stored server-side. Switch freely between turns.

## Stop button

During generation, the Send button becomes a red Stop button (■). Clicking Stop calls `AbortController.abort()` on the in-flight fetch. The partial streamed message is preserved with `*(generation stopped)*` appended. No error toast is shown for user-initiated stops.

## Environment variables

| Variable | Description |
|---|---|
| `OPENAI_MODEL` | Model for Fast mode and Agentic generation |
| `QUERY_MODEL` | Model for rewrite, decompose, draft, grade (falls back to `OPENAI_MODEL`) |
| `REASONING_MODEL` | Model for Thinking mode (falls back to `OPENAI_MODEL`) |
