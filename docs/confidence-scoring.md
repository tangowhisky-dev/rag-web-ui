# Confidence Scoring

Every query response includes a confidence score (0.0–1.0) that reflects the
quality of the retrieved evidence and the generated answer. The score is
computed in `answer_evaluation_node` and mapped to one of five levels displayed
in the UI as a stepped progress bar.

---

## Levels

| Level     | Score range | Bar steps | Meaning |
|-----------|-------------|-----------|---------|
| Very High | >0.8        | ████      | Strong retrieved evidence with high faithfulness and completeness |
| High      | 0.6–0.8     | ███░      | Good evidence, answer well-supported |
| Medium    | 0.3–0.6     | ██░░      | Partial evidence or "no information found" with irrelevant context |
| Low       | 0–0.3       | █░░░      | Weak evidence, answer mostly from general knowledge |
| None      | 0           | ░░░░      | No answer generated or evaluation failed |

---

## Scoring model

The final confidence is a weighted sum of three signals, each scored 0–100:

```
final_confidence = (0.4 · retrieval_score + 0.3 · faithfulness + 0.3 · completeness) / 100
```

Clamped to [0, 1].

### Retrieval score (40%)

Reflects the quality of retrieved evidence. Computed from the best search/rerank
score in `state.retrieved_docs`:

- **Cross-encoder reranker score** (`_reranker_score`): sigmoid-normalized to 0–1
  (cross-encoder scores can be negative; sigmoid maps them to 0–1)
- **Dense cosine similarity** (`score` from `search_dense`): already 0–1, used directly
- **SPLADE sparse score** (`score` from `search_sparse`): clamped to 0–1 (can be 0–10+)
- **MySQL FTS score** (`score` from `search_exact`): clamped to 0–1 (can be 0–10+)
- **Document-level match** (`kb_search_documents`, `kb_read`): set to 0.9 (high confidence by definition)

The best score across all observations is stored in `state.best_retrieval_confidence`
and read by `answer_evaluation_node`.

**Special case:** If the answer says "no information found" (contains phrases like
"no mention", "do not contain", "no relevant"), `retrieval_score` is set to 0
regardless of search scores — the retrieved evidence was not used to answer the query.

### Faithfulness (30%)

What percentage of the answer is actually supported by the retrieved context?
- 100 = everything cited or clearly supported by context
- 0 = answer is mostly or entirely external knowledge
- If the retrieved context is empty or irrelevant, faithfulness = 0
- If the answer accurately reports "no information found" and the context IS
  irrelevant, faithfulness = 100 (the answer accurately reports the lack of evidence)

Evaluated by the LLM using `EVALUATION_SYSTEM_PROMPT`.

### Completeness (30%)

How thoroughly does the answer address the query?
- 100 = all aspects of the query are fully addressed
- 0 = answer misses key parts of the query
- If the answer says "no information found" and the query asks for specific facts,
  completeness = 0 unless the information genuinely does not exist in the KB

Evaluated by the LLM using `EVALUATION_SYSTEM_PROMPT`.

---

## Implementation

| File | Role |
|------|------|
| `backend/app/services/agentic_rag/nodes.py` | `answer_evaluation_node` — computes final confidence |
| `backend/app/services/agentic_rag/agent_graph/tooling.py` | `_merge_observation_docs` — computes `best_confidence` from search/rerank scores |
| `backend/app/services/agentic_rag/evaluator.py` | LLM-based faithfulness/completeness evaluation |
| `backend/app/services/agentic_rag/graph_state.py` | `best_retrieval_confidence` state field |
| `backend/app/services/retrieval/retrieval.py` | Stores `score` in document metadata for confidence propagation |
| `backend/app/services/chat/chat_service.py` | Emits confidence in `d:` stream event |
| `frontend/src/components/chat/answer.tsx` | `ConfidenceBar` component |

### Stream event payload (`d:` done event)

```json
{
  "confidence":          "high",
  "confidenceScore":     67,
  "faithfulness":        85,
  "completeness":        80,
  "retrievalScore":      0.85,
  "finalConfidence":     80,
  "finalConfidenceLevel":"high",
  "followups":           ["What are the safety requirements?", "How does it compare to slow charging?"],
  "messageId":           12345,
  "userMessageId":       12344
}
```

The confidence scores are emitted in the `d:` (done) SSE event after the answer stream completes. The `la:` (last_answer) event carries the LastAnswerObject with citations and chart_options.

---

## UI

`ConfidenceBar` renders when the `d:` event arrives — after the answer stream completes. It shows:

- Label: "Retrieval confidence · Very High · 92/100"
- Four rectangular step segments, filled left-to-right based on level
- Colour-coded per level (emerald / green / yellow / orange / red)

The bar is hidden for `confidence = "none"` — instead a distinct amber warning
banner is shown, since there are no results to qualify.

---

## Confidence in the Agentic Pipeline

In the atomic-tools pipeline, the confidence score is computed by `answer_scoring` (formerly `answer_evaluation_node`) using the 3-signal formula:

```
final_confidence = 0.4 × retrieval_score + 0.3 × faithfulness + 0.3 × completeness
```

- **retrieval_score**: best search/rerank score from state.retrieved_docs (0–1)
- **faithfulness**: LLM-graded — what percentage of the answer is supported by cited evidence (0–100)
- **completeness**: LLM-graded — does the answer address the full query (0–100)

This reflects whether the draft-grade loop succeeded in answering each part of the question, not just whether documents were retrieved. The `breakdown` field also includes `sub_queries` (list) and `retrieval_attempts` (1–3) for debugging.

The `grade_coverage` node emits per-sub-query coverage lines (✓/~/✗) in its `agent_step` event — these are shown in the collapsible timeline detail, not in the confidence bar.

---

## Tuning

The weights are in `confidence.py` as plain arithmetic — change them there.
The level thresholds (80 / 55 / 30) are also in the same file.

Signal B (cross-leg agreement) is intentionally the highest weight. If your
deployment uses only one retrieval leg (e.g. dense-only), B will always be 0 and
scores will be structurally capped at 65. In that case consider raising the
weight of A and C to compensate, or enabling additional legs.
