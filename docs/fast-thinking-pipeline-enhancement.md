# Fast & Thinking Pipeline Enhancement Plan

## Current State Analysis

### Fast/Thinking Pipeline Flow (unchanged)
```
fast_stream() / thinking_stream() (same function, different model)
  │
  ├─ 1. _rewrite_query()          — 1 LLM call (QUERY_MODEL)
  │     Uses: recent_lc_history (3-turn sliding window)
  │     Output: standalone rewritten query
  │
  ├─ 2. hybrid_search_with_legs() — 0 LLM calls (embeddings + reranker = local)
  │     Retrieval legs: dense(Qdrant) + sparse(SPLADE) + exact(MySQL FTS)
  │     RRF merge with weights from preset (FACTUAL/ENTITY/etc.)
  │     Reranker: ms-marco-MiniLM-L-12-v2, threshold = RERANKER_SCORE_THRESHOLD (-2.0)
  │     Output: docs with _reranker_score in metadata
  │
  ├─ 3. score_retrieval()         — 0 LLM calls
  │     Signals: top_score(60%) + evidence_count(10%) + mean_score(30%)
  │     Score: 0-100, level: none/low/medium/high/very_high
  │     Thresholds: none(0), low(30), medium(55), high(80), very_high(80+)
  │
  ├─ 4. Build context string      — 0 LLM calls
  │     Context includes: numbered KB chunks [KB-1], [KB-2]... + file_markdown
  │     System prompt: _ANSWER_SYSTEM_PROMPT (formatting + citation rules)
  │
  ├─ 5. Build messages            — 0 LLM calls
  │     System: _ANSWER_SYSTEM_PROMPT
  │     System (optional): "[Conversation summary so far]\n{existing_summary}"
  │     User: "Context:\n{merged}\n\nQuestion: {rewritten}"
  │
  └─ 6. Stream answer             — 1 LLM call (OPENAI_MODEL / REASONING_MODEL)
       Token streaming with AgentTimeline events
```

**Total LLM calls: 2** (rewrite + answer) — fixed budget, cannot exceed without changing the pipeline identity.

### How Chat History Is Currently Handled

| Phase | What's Injected | Source | How |
|-------|-----------------|--------|-----|
| **Rewrite** | `recent_lc_history` (3 turns) | DB query (last 6 messages) | LangChain HumanMessage/AIMessage objects passed to QUERY_MODEL |
| **Answer** | `existing_summary` | DB column `Chat.history_summary` (built post-turn in background) | System prompt: `"[Conversation summary so far]\n{summary}"` |
| **Answer** | Sliding window | **NOT injected** | Intentionally excluded — "raw prior answers pollute the context and cause the LLM to treat its own previous responses as user statements" |

**Key insight:** The conversation summary IS already injected into context. It covers all messages beyond the 3-turn window. But it's a **compressed LLM summary** — it may lose specific details, exact numbers, or nuanced context that users might need.

**Historical memory retrieval** would ADD to this — fetching specific relevant past messages (not the summary) and including them in context. This gives the LLM both the broad context (summary) and specific details (retrieved past messages).

---

## Enhancement 1: Adaptive Retrieval Threshold

### Problem
Current pipeline retrieves once with fixed thresholds. If the initial retrieval is poor (low confidence), the LLM gets weak context and generates a poor answer — with no recovery.

### Solution: Two-pass retrieval with dynamic threshold adjustment

```
Pass 1 (standard):
  rewrite → hybrid_search (preset weights, threshold=-2.0, top_k=10) → score_retrieval
     │
     └─ confidence ≥ 55 ("medium") → proceed to answer
     │
     └─ confidence < 55 → trigger Pass 2

Pass 2 (adaptive — only when needed):
  hybrid_search (relaxed weights, threshold=-5.0, top_k=15) → score_retrieval → merge with Pass 1 results → answer
```

### Why these specific values?

| Parameter | Pass 1 (Standard) | Pass 2 (Adaptive) | Rationale |
|-----------|-------------------|-------------------|-----------|
| **Confidence threshold** | ≥ 55 | < 55 | 55 = "medium" level. Above this, the reranker found genuinely relevant docs. Below, we need broader recall. |
| **Reranker threshold** | -2.0 | -5.0 | Empirically: relevant chunks score 1–10, irrelevant score -5 to -11. -2.0 is a good balance. -5.0 widens to capture marginal-but-relevant docs. |
| **top_k** | 10 | 15 | More candidates for reranking gives it more to choose from. The reranker still filters — relaxed threshold means more docs, not noisier docs. |
| **Retrieval legs** | Per preset | Same legs, wider scope | No need to enable/disable legs — just accept lower-quality matches from existing legs. |

### Implementation details

**Where to change:** `backend/app/services/fast_pipeline.py` — `fast_stream()` function

**Changes:**

1. **After Pass 1 retrieval, check confidence before building context:**
```python
# Existing: confidence check was only for emitting the "context" event.
# New: check if we need Pass 2.
if conf_result.score < 55 and settings.ADAPTIVE_RETRIEVAL_ENABLED:
    logger.info("[FAST] confidence=%.0f < 55, triggering adaptive retrieval", conf_result.score)
    raw_docs = _adaptive_retrieval(rewritten, kb_ids, db, use_graph_rag, datastore_ids)
    # Re-score merged results
    conf_result = score_retrieval(raw_docs, new_retrieval_info)
```

2. **New `_adaptive_retrieval()` function:**
```python
def _adaptive_retrieval(
    query: str,
    kb_ids: List[int],
    db: Session,
    use_graph_rag: bool,
    datastore_ids: List[int],
) -> List[LangchainDocument]:
    """
    Second-pass retrieval with relaxed thresholds.
    
    Strategy:
    1. Run hybrid_search_with_legs with relaxed parameters
    2. Merge results with Pass 1 docs (dedup by content hash)
    3. Re-run reranker on merged pool (the cross-encoder will re-score everything fresh)
    4. Return top-N that clear the relaxed threshold
    """
    from app.services.retrieval import hybrid_search_with_legs
    from app.services.confidence import score_retrieval
    
    # Run with relaxed parameters
    retrieval_result = await hybrid_search_with_legs(
        query=query,
        kb_ids=kb_ids,
        db=db,
        use_dense=True, use_sparse=True, use_exact=True,  # always all legs
        use_graph_rag=use_graph_rag,
        datastore_ids=datastore_ids,
    )
    
    # Merge with Pass 1 results (dedup)
    pass1_docs_set = {_content_hash(d.page_content) for d in raw_docs}
    new_docs = [d for d in retrieval_result["docs"] 
                if _content_hash(d.page_content) not in pass1_docs_set]
    
    # Combine Pass 1 + new docs
    merged = raw_docs + new_docs
    
    # Re-run reranker on the full merged pool with relaxed threshold
    from app.services.reranker import rerank
    if settings.RERANKER_ENABLED:
        merged = rerank(query=query, docs=merged, score_threshold=-5.0)
    
    logger.info("[FAST] adaptive retrieval: pass1=%d, new=%d, merged=%d",
                len(raw_docs), len(new_docs), len(merged))
    return merged
```

3. **New config variable:**
```python
# In backend/app/core/config.py
ADAPTIVE_RETRIEVAL_ENABLED: bool = os.getenv("ADAPTIVE_RETRIEVAL_ENABLED", "true").lower() == "true"
```

### Cost analysis
- **LLM calls:** Still 2 (rewrite + answer). No new LLM calls.
- **Local computation:** One additional set of embeddings + one additional reranking pass (~50ms on CPU).
- **Impact:** Users who need it get it (confidence < 55). Users who don't need it pay nothing.
- **Expected trigger rate:** ~20-30% of queries (based on typical reranker score distributions).

### What changes in the event stream
- Same events emitted as before, BUT when adaptive retrieval fires:
  - The "context" event shows `adaptive: true` in breakdown
  - AgentTimeline shows "Widening search…" step (same as agentic widened_retrieval)

---

## Enhancement 2: Answer Quality Grading

### Problem
The current pipeline has no quality check after generation. The LLM might generate a well-formed but hallucinated or incomplete answer — especially when confidence is low.

### Solution: Optional post-generation quality check using QUERY_MODEL

**Key constraint:** We already use 2 LLM calls (rewrite + answer). Adding a quality check makes it 3. This is acceptable IF:
- It only triggers when confidence is low (< 55, the same threshold as adaptive retrieval)
- It uses QUERY_MODEL (already available, smaller/faster)
- It's clearly optional and feature-flagged

### How it works

After the answer is generated (before returning to client), run a quick quality check:

```python
async def _grade_answer_quality(
    query: str,
    rewritten_query: str,
    context: str,
    answer: str,
    api_base: Optional[str] = None,
    query_model: Optional[str] = None,
) -> dict:
    """
    Grade the answer on faithfulness, completeness, and coherence.
    
    Returns:
        {
            "faithfulness": 0.85,   # 0-1: are all claims backed by context?
            "completeness": 0.90,   # 0-1: does it answer the user's question?
            "coherence": 0.95,      # 0-1: is it well-structured?
            "verdict": "pass",       # "pass" | "needs_improvement" | "unsatisfactory"
            "suggestions": [...]     # actionable feedback (empty if pass)
        }
    """
```

**LLM prompt (uses QUERY_MODEL, ~50 tokens max output):**
```
You are a RAG answer quality grader. Evaluate this answer:

USER QUERY: {query}
CONTEXT: {context_summary}
ANSWER: {answer}

Grade:
1. faithfulness (0.0-1.0): Are all factual claims in the answer supported by the context?
   Count any claim without a corresponding [KB-N] citation.
2. completeness (0.0-1.0): Does the answer address the user's question?
   If the user asked multiple questions, is each one answered?
3. coherence (0.0-1.0): Is the answer well-structured and readable?
   Well-formatted, no repetition, logical flow.

Rules:
- Be strict: hallucinated claims should be caught.
- Output ONLY JSON: {"faithfulness": 0.XX, "completeness": 0.XX, "coherence": 0.XX}
- No text, no explanation, no preamble.
```

### How it differs from existing methods

| Aspect | Current Approach | Proposed Approach |
|--------|-----------------|-------------------|
| **Faithfulness** | Rely on citation rules in system prompt | Explicit grading: LLM checks each claim against context |
| **Completeness** | Rely on LLM being "helpful" | Explicit grading: LLM checks if all parts of the question are answered |
| **Coherence** | Rely on formatting rules in system prompt | Explicit grading: LLM checks structure and flow |
| **Trigger** | Always (no check) | Only when confidence < 55 (adaptive retrieval already triggered) |
| **Cost** | 0 extra calls | 1 extra QUERY_MODEL call (small/fast, ~200ms) |

### Verdict action matrix

| Verdict | Action |
|---------|--------|
| **pass** (all scores ≥ 0.7) | Answer as-is (no changes) |
| **needs_improvement** (any score < 0.7, ≥ 0.5) | Regenerate with feedback: append `Suggestions: {suggestions}` to the answer prompt, regenerate (this is the 3rd LLM call's value-add) |
| **unsatisfactory** (any score < 0.5) | Return the original answer with a disclaimer: "I may not have all the information to answer this question accurately." |

### Integration into fast_stream()

```python
# ── 5. Stream answer ──────────────────────────────────────────────────────
# (existing streaming code...)

# ── 6. Quality grading (conditional) ──────────────────────────────────────
if conf_result.score < 55 and settings.ANSWER_QUALITY_GRADING_ENABLED:
    quality = await _grade_answer_quality(
        query=query,
        rewritten_query=rewritten,
        context=merged,
        answer=raw_answer,
        api_base=api_base,
        query_model=query_model,
    )
    
    if quality["verdict"] == "needs_improvement":
        logger.info("[FAST] quality needs improvement: %s", quality)
        # Regenerate with feedback (this is the 3rd LLM call)
        # ... regenerate and stream the improved answer
    elif quality["verdict"] == "unsatisfactory":
        logger.info("[FAST] answer unsatisfactory, adding disclaimer")
        raw_answer += "\n\n[Note: I may not have complete information to answer this question accurately.]"
```

### Cost analysis
- **Default case (confidence ≥ 55):** Still 2 LLM calls, zero overhead.
- **Low confidence (confidence < 55):** 3 LLM calls total (rewrite + answer + quality check + optional regeneration).
- **Extra latency:** ~300-500ms for the quality check. Regeneration adds another ~1-2s.
- **Expected trigger rate:** ~20-30% of queries (same as adaptive retrieval).

---

## Enhancement 3: Historical Memory Retrieval

### Problem
Currently, only the 3-turn sliding window is used (for rewrite). All older history is compressed into a summary. The summary provides broad context but loses:
- Specific details (exact numbers, names, dates)
- Nuanced answers to specific sub-questions
- The user's own previous questions (which inform context)

### Solution: Retrieve relevant past messages from MySQL using reranker

**Key insight:** The reranker (ms-marco-MiniLM-L-12-v2) is a local ONNX model. It's cheap (~4ms per chunk) and doesn't count as an LLM call. We can use it to score past messages without breaking the LLM call budget.

### How it works

1. **After rewrite, before retrieval:** Query MySQL for past assistant messages (excluding the 3-turn sliding window that's already used for rewrite)
2. **Rerank them** against the rewritten query
3. **Select top-K** that clear a threshold (recommend K=5, threshold=2.0)
4. **Inject them** into the context string as `[Historical Memory]` blocks
5. **The summary (`existing_summary`)** continues to provide broad context — now complemented by specific retrieved details

### Implementation

**New file:** `backend/app/services/historical_memory.py`

```python
"""
Historical memory retrieval for fast/thinking pipelines.

Scores past assistant messages against the current query using the reranker.
The reranker is a local ONNX model (no LLM call), so this fits within the
2-LLM-call budget of fast/thinking pipelines.
"""
import logging
from typing import List, Optional
from langchain_core.documents import Document as LangchainDocument
from sqlalchemy import text
from app.core.config import settings

logger = logging.getLogger(__name__)


async def retrieve_historical_memory(
    chat_id: int,
    query: str,
    db: Any,
    window_messages: int = 6,  # exclude last N messages (3 pairs)
    top_k: int = 5,
    score_threshold: float = 2.0,
) -> List[dict]:
    """
    Retrieve relevant past assistant messages from MySQL for this chat.
    
    Strategy:
    1. Query all assistant messages beyond the sliding window
    2. Score each against the query using the reranker
    3. Return top-K that clear threshold
    
    Returns list of serialised documents with _source_type="historical_memory"
    so the context builder can label them appropriately.
    """
    from app.services.reranker import rerank
    
    # Query past assistant messages (excluding recent window)
    # The window_messages param excludes the last N messages (default 6 = 3 turns)
    result = db.execute(
        text("""
            SELECT content, role
            FROM messages
            WHERE chat_id = :chat_id
              AND role = 'assistant'
            ORDER BY id ASC
            LIMIT :limit OFFSET :offset
        """),
        {
            "chat_id": chat_id,
            "limit": 50,  # cap at 50 for performance
            "offset": window_messages,
        },
    ).fetchall()
    
    if not result:
        logger.info("[HIST_MEM] chat_id=%d | no past messages beyond window", chat_id)
        return []
    
    # Build LangchainDocument for each past message
    docs = [
        LangchainDocument(
            page_content=row.content.strip()[:2000],  # cap to ~500 tokens for reranker
            metadata={
                "_source_type": "historical_memory",
                "message_id": None,  # not stored in this query
                "source": "conversation_history",
            },
        )
        for row in result if row.content.strip()
    ]
    
    if not docs:
        return []
    
    # Score with reranker
    ranked = rerank(query=query, docs=docs, score_threshold=score_threshold)
    
    if not ranked:
        logger.info("[HIST_MEM] chat_id=%d | no past messages above threshold=%.2f",
                    chat_id, score_threshold)
        return []
    
    # Take top-K and serialise
    top_docs = ranked[:top_k]
    logger.info("[HIST_MEM] chat_id=%d | scored=%d | above_threshold=%d | top_k=%d",
                chat_id, len(docs), len(ranked), len(top_docs))
    
    return [
        {
            "page_content": doc.page_content,
            "metadata": {
                **doc.metadata,
                "_reranker_score": round(float(doc.metadata.get("_reranker_score", 0)), 4),
            },
        }
        for doc in top_docs
    ]
```

### How it integrates into `fast_stream()`

```python
# ── Between rewrite and retrieval ─────────────────────────────────────

# Historical memory retrieval (reranker-based, no LLM call)
historical_docs = []
if settings.HISTORICAL_MEMORY_ENABLED:
    historical_docs = await retrieve_historical_memory(
        chat_id=chat_id,
        query=rewritten,
        db=db,
        window_messages=6,  # exclude 3-turn sliding window
        top_k=5,
        score_threshold=2.0,
    )

# ── Later, when building context string ──

context_parts: list[str] = []

# Historical memory first (they're about past conversations, not KB)
for i, doc in enumerate(historical_docs, 1):
    content = doc.get("page_content", "").strip()
    header = f"[Historical Memory {i}]"
    context_parts.append(f"{header}\n{content}")

# KB chunks
for i, doc in enumerate(serialised_docs, 1):
    content = doc.get("page_content", "").strip()
    source = doc.get("metadata", {}).get("source", "")
    header = f"[KB-{i}]" + (f" ({source})" if source else "")
    context_parts.append(f"{header}\n{content}")

# File content
if file_markdown:
    context_parts.append(f"[File Content]\n{file_markdown}")
```

### Cost analysis
- **LLM calls:** Still 2 (rewrite + answer). Historical retrieval uses reranker only.
- **Local computation:** One MySQL query + one reranking pass (~50ms on CPU).
- **Context size:** Adds up to 5 past messages × ~2000 chars = ~10,000 chars (but reranker scores will select only the most relevant).
- **Impact:** Users with long conversations benefit from specific past details being available.

### New config variables
```python
# In backend/app/core/config.py
HISTORICAL_MEMORY_ENABLED: bool = os.getenv("HISTORICAL_MEMORY_ENABLED", "true").lower() == "true"
HISTORICAL_MEMORY_TOP_K: int = int(os.getenv("HISTORICAL_MEMORY_TOP_K", "5"))
HISTORICAL_MEMORY_SCORE_THRESHOLD: float = float(os.getenv("HISTORICAL_MEMORY_SCORE_THRESHOLD", "2.0"))
```

---

## Complete Enhanced Pipeline Flow

```
fast_stream()
  │
  ├─ 1. _rewrite_query()          — 1 LLM call (QUERY_MODEL)
  │     Uses: recent_lc_history (3-turn sliding window)
  │     Output: standalone rewritten query
  │
  ├─ 2. Historical memory retrieval (NEW) — 0 LLM calls (reranker only)
  │     Query: past assistant messages beyond 3-turn window
  │     Score: reranker (ms-marco-MiniLM-L-12-v2, threshold=2.0)
  │     Output: top-5 most relevant past messages
  │
  ├─ 3. hybrid_search_with_legs() — 0 LLM calls
  │     Pass 1: standard retrieval (preset weights, threshold=-2.0, top_k=10)
  │     Score confidence
  │     │
  │     └─ confidence < 55 → Pass 2 (NEW, adaptive retrieval)
  │           Relaxed: threshold=-5.0, top_k=15
  │           Merge with Pass 1 results (dedup)
  │           Re-rerank merged pool
  │
  ├─ 4. Build context string      — 0 LLM calls
  │     Order: [Historical Memory] → [KB-N] → [File Content]
  │     System prompt: _ANSWER_SYSTEM_PROMPT
  │
  ├─ 5. Build messages            — 0 LLM calls
  │     System: _ANSWER_SYSTEM_PROMPT
  │     System: "[Conversation summary so far]\n{existing_summary}"
  │     User: "Context:\n{merged}\n\nQuestion: {rewritten}"
  │
  └─ 6. Stream answer             — 1 LLM call (OPENAI_MODEL / REASONING_MODEL)
       Token streaming
       │
       └─ 7. Quality grading (NEW, conditional) — 1 LLM call (QUERY_MODEL)
             Only when confidence < 55 AND quality grading enabled
             Grade: faithfulness + completeness + coherence
             Verdict: pass | needs_improvement | unsatisfactory
             If needs_improvement → regenerate with feedback
             If unsatisfactory → add disclaimer to answer
```

**Total LLM calls:**
- **Default case (confidence ≥ 55):** 2 (same as before)
- **Low confidence (confidence < 55, quality passes):** 2 (+ 0.3s local for adaptive retrieval + reranking)
- **Low confidence + quality needs improvement:** 3 (rewrite + answer + quality check + regeneration)

---

## Configuration Summary

| Variable | Default | Description |
|----------|---------|-------------|
| `ADAPTIVE_RETRIEVAL_ENABLED` | `true` | Enable two-pass retrieval when confidence < 55 |
| `ADAPTIVE_RETRIEVAL_THRESHOLD` | `55` | Confidence score below which adaptive retrieval triggers |
| `ADAPTIVE_RETRIEVAL_RERANKER_THRESHOLD` | `-5.0` | Relaxed reranker threshold for Pass 2 |
| `ADAPTIVE_RETRIEVAL_TOP_K` | `15` | top_k for Pass 2 |
| `ANSWER_QUALITY_GRADING_ENABLED` | `true` | Enable post-generation quality check |
| `HISTORICAL_MEMORY_ENABLED` | `true` | Enable retrieval of past messages from MySQL |
| `HISTORICAL_MEMORY_TOP_K` | `5` | Number of past messages to retrieve |
| `HISTORICAL_MEMORY_SCORE_THRESHOLD` | `2.0` | Reranker threshold for historical memory |
| `HISTORICAL_MEMORY_WINDOW` | `6` | Exclude last N messages (3 pairs) from historical retrieval |

---

## Open Questions / Ambiguities

1. **Quality grading prompt design:** The current draft prompt is minimal. Should we expand it to also check for:
   - Tone appropriateness (professional vs casual)?
   - Over-confidence on uncertain answers?
   - Citation accuracy (does every [KB-N] actually exist in context)?
   → Recommendation: start minimal, iterate based on real-world feedback.

2. **What happens when quality grading says "needs_improvement":** The regeneration step uses the same model but with feedback appended. Should this be:
   - The same model (OPENAI_MODEL) — cheapest, consistent with current answer
   - The reasoning model (if Thinking mode) — might produce better answers but adds cost
   → Recommendation: use the same model for consistency.

3. **Historical memory vs summary:** Both are injected into context. The summary covers everything; historical memory adds specifics. Should we:
   - Always inject both (current plan)
   - Skip the summary if historical memory retrieves enough detail
   → Recommendation: always inject both — summary provides context, historical memory provides detail. They're complementary, not redundant.

4. **Performance under load:** The adaptive retrieval + historical memory add local computation (embeddings + reranking). Under high concurrency, this could increase CPU usage. Should we:
   - Add a concurrency limit (e.g., only allow N adaptive retrievals per second)
   - Keep it simple and monitor CPU usage first
   → Recommendation: keep it simple for now, add limits if CPU becomes a bottleneck.
