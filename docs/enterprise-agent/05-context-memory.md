# 05 — Context & Memory

How the agent manages the LLM context window across long multi-turn conversations. Replaces the current count-based compaction with token-accurate budgeting, adds the `last_answer_object` as a first-class referent, and wires up the currently-unused long-term recall.

---

## 1. The problem

Current behavior (`agentic_rag/nodes.py:87-158`):
- Compaction triggers when `len(messages) > COMPACTION_HISTORY_THRESHOLD` — a **message count**, not a token count.
- A turn that attaches a 50k-token file counts as one message. The window overflows long before the count fires.
- The previous assistant answer is reconstructed from the message list and truncated at `COMPACTION_ASSISTANT_MAX_CHARS` (`nodes.py:718-732`). "Summarise it in 10 points" operates on a possibly-truncated referent.
- `RedisMemory.search_memory()` exists (`redis_memory.py:90-174`) but `load_subtask_memory_node` is a no-op (`nodes.py:1080-1087`). Cross-thread recall is wired but unused.
- Token estimates are character heuristics (`utils.py`), not tokenizer-accurate.

Target: token-accurate budgeting, a structured previous-answer object, sliding window with importance, and proactive long-term recall.

---

## 2. `last_answer_object` — the structured previous answer

After every `finalize_answer_node`, an LLM call (query model, cheap) extracts:

```python
class LastAnswerObject(BaseModel):
    summary: str                       # 2-3 sentences
    key_points: list[str]              # bullet points
    data: list[DataPoint] | None       # numbers/stats with labels
    citations: list[CitationRef]       # chunk refs (doc_id, chunk_index)
    chart_option: dict | None          # ECharts JSON if a chart was produced
    followups: list[str]               # suggested follow-ups

class DataPoint(BaseModel):
    label: str
    value: float | str
    unit: str | None = None
    context: str | None = None         # sentence it appeared in
```

**Storage**: `messages.last_answer_object` (JSON column, new migration).

**Use in next turn**: `load_context_node` loads the most recent `last_answer_object` into `AgentState.last_answer_object`. The planner sees it. The tools `summarize_answer` and `extract_data` (source=`last_answer`) operate on it directly — no truncation, no reconstruction from raw text.

**Why this fixes "summarise it" / "give me the stats" / "make it a chart"**: the referent is a compact structured object, not a truncated blob. `extract_data` pulls `data` (already structured) or, if `data` is empty, re-extracts from the answer text. `chart_generate` consumes the extracted data deterministically. The chain is: previous answer → `last_answer_object.data` → `chart_generate` → ECharts JSON. No LLM guessing at chart JSON from prose.

**Extraction robustness** (critical — local models produce malformed JSON frequently):
1. LLM extraction call with `LAST_ANSWER_EXTRACT_PROMPT` (includes "return valid JSON" instruction).
2. Pydantic validation of the response. If valid → store.
3. If invalid → one retry with "you returned invalid JSON: <error>. Fix it and return valid JSON only."
4. If still invalid → **rule-based fallback** populates a partial object:
   - `summary`: first 2 sentences of the answer.
   - `key_points`: sentence split (every sentence ending in `.`/`!`/`?` becomes a point, capped at 10).
   - `data`: regex sweep for `<number> <unit?>` patterns near label words (e.g., "revenue: $4.2M", "grew 12%").
   - `citations`: empty (the message row's existing `MessageCitation` rows are the source of truth anyway).
   - `chart_option`: None.
   - `followups`: empty.
5. A partial object is better than a missing one — "summarise it" still works on `key_points`, "give me the stats" still works on `data`. The fallback is logged so operators can see how often the LLM fails extraction.

**Cost**: one extra query-model call per turn for extraction (plus occasional retry). Cheap relative to the generation call. Can be skipped if the answer is short (heuristic: under 500 tokens → skip, the raw text fits anyway; `last_answer_object` is set to a minimal `{summary: <first 2 sentences>, key_points: [], data: [], ...}`).

---

## 3. Token-accurate budgeting

### 3.1 Tokenizer

`token_budget.py` provides `get_tokenizer(model_name)`:
- OpenAI-family models: `tiktoken` (offline-safe, bundled encodings).
- Local/HF models: `transformers.AutoTokenizer.from_pretrained(model_name)` — requires the tokenizer files locally (already present for any model the gateway serves; cache in `FASTEMBED_CACHE_DIR` or a new `TOKENIZER_CACHE_DIR`).
- Fallback: `tiktoken` with `cl100k_base` if the model-specific tokenizer is unavailable. Logged as a warning.

`TOKENIZER_MODEL` env var (defaults to `OPENAI_MODEL`) selects which tokenizer to use for budgeting. If the gateway serves a model whose tokenizer isn't available, the operator sets `TOKENIZER_MODEL` to a close proxy.

### 3.2 Context budget

```
CONTEXT_WINDOW = OPENAI_MODEL_CONTEXT_SIZE      (e.g., 131072)
RESERVED_GENERATION = CONTEXT_RESERVED_GENERATION (default 4096)
TOOL_BUDGET = CONTEXT_TOOL_BUDGET                (default 8192)
AVAILABLE = CONTEXT_WINDOW - RESERVED_GENERATION - TOOL_BUDGET
```

`load_context_node` assembles the context payload and tracks `used_tokens`. If `used_tokens > AVAILABLE * 0.85`, compaction runs *before* planning, not after.

### 3.3 What goes in the context payload (ordered by priority)

1. System prompt (fixed).
2. Tool registry summary (fixed, small).
3. Compaction summary (if any) — the structured summary of older turns.
4. `last_answer_object` (compact, high-value referent).
5. Recalled long-term memory (top 3 turns, truncated to ~500 tokens each).
6. Recent turns sliding window (see §4).
7. Current user query.
8. Attached file metadata (names/types/sizes — *not* content; content is fetched on demand by `file_read`).

Total tracked by `count_tokens` per section. If over budget, compact from the bottom of the priority list upward (drop recalled memory before recent turns; drop recent turns before `last_answer_object`).

---

## 4. Sliding window with importance

### 4.1 Window

Keep the last `W` turns verbatim (default `W=6` turns = 3 user + 3 assistant). Older turns are represented by the compaction summary.

### 4.2 Importance scoring

Not all turns are equal. Before compaction, score each older turn 0–1:
- **Recency**: exponential decay, half-life 10 turns.
- **Has data**: +0.3 if the turn's `last_answer_object.data` is non-empty (numbers are likely referenced again).
- **Has chart**: +0.2 if `chart_option` is non-empty.
- **User-marked**: +0.5 if the user pinned/bookmarked the message (new optional feature; without it, this signal is 0).
- **Cited**: +0.2 if the turn has citations (grounded answers are more reusable).

Turns above an importance threshold (default 0.5) are kept verbatim in a "highlights" section even after the window slides past them, up to a token cap (`HIGHLIGHTS_TOKEN_CAP`, default 2000). Turns below are folded into the compaction summary.

### 4.3 Compaction prompt (modified)

The existing `COMPACTION_SYSTEM_PROMPT` (`prompts.py:13-68`) produces a structured summary (Goal, Topics, Decisions, Retrieved Docs, Progress, Critical Context, Next Steps). Keep it. Add: a "Key Data" section that lists important numbers from compacted turns (pulled from their `last_answer_object.data`), so numbers survive compaction.

### 4.4 Tool observation compaction

Tool outputs (retrieved docs, code results) are large and lower-density than conversation. Separate compaction:
- `rag_retrieve` observations: keep only the top 5 chunks by score; drop the rest after the agent has used them (i.e., after the next think step references them).
- `code_execute` observations: keep `result` + last 20 lines of stdout; drop full stdout.
- `file_read` observations: keep only if the agent is still working with that file; otherwise summarize to "read file X, sections A/B/C".

This prevents tool output from crowding out conversation in long sessions.

---

## 5. Long-term recall (wiring up the unused store)

### 5.1 Proactive recall

`load_context_node` calls `RedisMemory.search_memory(rewritten_query, user_id, chat_id, limit=3)`. Results are injected as `<recalled_memory>` context:

```
<recalled_memory>
From an earlier conversation (chat_id, date):
  User: ...
  Assistant: [summary]
</recalled_memory>
```

This is the fix for the no-op `load_subtask_memory_node`. The store already exists and is already populated by `save_memory_node`; it just was never read.

### 5.2 On-demand recall (deferred)

An explicit `memory_recall` tool was considered and pruned — proactive recall in `load_context_node` (§5.1) is sufficient for multi-turn, and an on-demand tool added LLM selection ambiguity without changing outcomes. If deeper past-search becomes needed, add it later.

### 5.3 Recall budget

Top 3 results, each truncated to ~500 tokens. Total recall budget ~1500 tokens. Falls within the context payload priority order (§3.3, item 5).

---

## 6. Per-section token accounting

`load_context_node` returns a `ContextPayload` with per-section token counts:

```python
class ContextPayload(BaseModel):
    sections: list[{"name": str, "tokens": int, "content": str}]
    total_tokens: int
    budget: int
    compacted: bool
```

This is logged (debug) and surfaced in the `d:` done event as `usage.context_tokens` for observability. The frontend can show a "context: 12k / 128k tokens" indicator (optional).

---

## 7. Edge cases

| Case | Handling |
|---|---|
| First turn (no history, no last_answer_object) | `load_context_node` returns just system prompt + query + tool registry. No compaction. |
| Very long single file attached | `file_markdown` is *not* loaded into context by default. Only metadata. Agent calls `file_read` / `file_summarize` on demand. Fixes the silent-truncation gap. |
| Many turns, no compaction yet | Sliding window + importance highlights keep recent + high-value turns; compaction triggers on token budget. |
| Compaction summary itself grows too large | Cap compaction summary at `COMPACTION_SUMMARY_MAX_CHARS` (existing). If exceeded, recursively summarize the summary (rare). |
| Tokenizer unavailable | Fallback to `tiktoken cl100k_base`, log warning. Budgets drift slightly but safety margins (0.85 trigger, reserved generation) absorb it. |
| Local model with tiny context (8k) | `CONTEXT_TOOL_BUDGET` and reserved generation dominate. Agent loop still works but fewer turns fit. Operator tunes `W` (window) down. Documented in troubleshooting. |

---

## 8. What this replaces in the current code

| Current | Replacement |
|---|---|
| `compaction_node` count trigger (`nodes.py:87-158`) | Token trigger via `token_budget.py` |
| `_build_generation_messages` truncation (`nodes.py:649-789`) | `load_context_node` priority-ordered payload |
| `load_subtask_memory_node` no-op (`nodes.py:1080-1087`) | `load_context_node` proactive recall |
| Character-heuristic token estimation (`utils.py`) | `token_budget.count_tokens` |
| `COMPACTION_ASSISTANT_MAX_CHARS` truncation of prior answer | `last_answer_object` structured referent |
| Implicit "previous answer in context window" | Explicit `last_answer_object` + `summarize_answer`/`extract_data` tools |
