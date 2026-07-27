# 06 — Roadmap

Phased delivery. Per AGENTS.md §4: every phase defines success as something runnable, not "plausibly done".

---

## Phase 0 — Foundation (no behavior change)

**Goal**: land the infrastructure the loop depends on, without touching the active pipeline.

**Work**
1. `token_budget.py` + tokenizer wiring (`05` §3).
2. `llm_factory.py` + wire existing `OrgLLMConfig` columns (`api_base`/`model_name`/`query_model`) — no new columns (`04` A6).
3. `LastAnswerObject` schema + `messages.last_answer_object`/`plan`/`tool_calls` columns + `tool_call_audit` table + Alembic migration (`04` A9).
4. SSE v4 event emitters in `streaming.py` (`04` A7).
5. New `requirements.txt` deps installed in the Docker image (`04` A11).
6. `AGENT_SYSTEM_PROMPT` (unified guardrail) drafted (`04` A4) — not yet wired, just reviewed.

**Verification**
- `docker exec rag-web-ui-backend-1 pytest` — all existing tests pass (no behavior change).
- `docker exec rag-web-ui-backend-1 python -c "from app.services.agentic_rag.token_budget import count_tokens; print(count_tokens('hello world', 'qwen/qwen3.5-9b'))"` returns an int.
- Per-org LLM config: set `OrgLLMConfig.model_name` for a test org, call `GET /api/query` with that org's user, confirm the model in the `d:` usage event matches.
- `tool_call_audit` table exists and accepts a fixture row.

**Exit criterion**: existing pipeline unchanged, new infra available, all existing tests green.

---

## Phase 1 — Tools (still no loop)

**Goal**: build and test every tool in isolation. Tools are callable directly (not yet wired into a graph). Each tool includes RBAC re-check + audit-row write.

**Work**
1. `tool_context.py` — `ToolContext` dataclass + `enforce_rbac` + `write_audit` helpers (`04` A1).
2. `rag_retrieve` — lift retrieval node functions (incl. `neo4j_expansion` via `graph_expand`) into the tool (`03` §1).
3. `file_read`, `file_summarize`, `file_extract_table` (`03` §2-4).
4. `code_execute` + `sandbox.py` (RestrictedPython backend) (`03` §5).
5. `chart_generate` + `echarts_builder.py` (`03` §6).
6. `summarize_answer`, `extract_data` (generalized: last_answer | retrieved_docs | file | specified) (`03` §7-8).
7. `clarify` (`03` §9).
8. Per-tool tests including RBAC denial + audit-row assertions (`04` A12).

**Pruned from earlier draft:** `graph_retrieve` (merged into `rag_retrieve`), `memory_recall` (proactive recall covers it), `table_generate` (deferred).

**Verification**
- Each tool has a test that calls it with a fixture input and asserts the output schema.
- `code_execute`: test denylist (socket, subprocess refused), timeout (10s sleep killed), data injection (pandas DataFrame in → result out), plot capture (matplotlib figure → PNG ref).
- `chart_generate`: test pie/bar/line from a fixture table; test invalid data (empty series) returns `valid=False`.
- `file_summarize`: test a 20k-token fixture file produces a summary under 1k tokens with `chunks_processed > 1`.
- `extract_data`: test extraction from each source (last_answer, retrieved_docs, file); test malformed-JSON retry + regex fallback.
- `rag_retrieve`: run against the eval harness KB; compare confidence + doc count to the current pipeline's retrieval for the same query (parity expected — same functions). Test RBAC: a kb_id the user doesn't own is dropped and logged.
- Every tool test asserts an audit row was written with the correct status.

**Exit criterion**: all tools pass tests; `rag_retrieve` retrieval parity with current pipeline confirmed; RBAC + audit verified on every tool.

---

## Phase 2 — Agent loop

**Goal**: the loop runs end-to-end. Includes the offline tool-calling fallback, unified guardrail, parallel subtask dispatch, and reflect replanning.

**Work**
1. `tool_call_parser.py` — native + JSON-text + final-answer-default tiers (`04` A2b).
2. `agent_graph.py` with `load_context`, `rewrite`, `compaction` (token-based), `plan`, `think`, `tool` (parallel dispatch), `reflect`, `finalize`, `answer_scoring`, `save_memory` nodes (`04` A2).
3. `Plan` schema + `AGENT_SYSTEM_PROMPT` (guardrail) + `PLAN_SYSTEM_PROMPT`, `THINK_SYSTEM_PROMPT` (with JSON-text fallback instructions), `REFLECT_SYSTEM_PROMPT` (with concrete replanning rules), `LAST_ANSWER_EXTRACT_PROMPT` (`04` A4).
4. `AgentState` extensions (`04` A2).
5. `run_agent_loop` entry point + flag dispatch in `chat_service.generate_response` (`04` C1).
6. `last_answer_object` extraction in `finalize_answer_node` (with Pydantic validation + retry + rule-based fallback) + persistence in `save_memory_node` (`05` §2).
7. Proactive long-term recall in `load_context_node` (`05` §5).
8. Tests: `test_agent_loop.py` (incl. parallel dispatch), `test_agent_loop_budget.py`, `test_agent_loop_rbac.py` (prompt-injection denial), `test_agent_loop_audit.py`, `test_token_budget.py`, `test_last_answer_object.py` (incl. fallback), `test_tool_call_parser.py` (`04` A12).

**Verification**
- `docker exec rag-web-ui-backend-1 pytest tests/test_agent_loop.py tests/test_agent_loop_rbac.py tests/test_agent_loop_audit.py tests/test_tool_call_parser.py` pass.
- `TOOL_CALL_MODE=json_text` manual test: point the loop at a gateway/model that does not support native function-calling; confirm the JSON-text fallback parses tool calls and the loop completes.
- `TOOL_CALL_MODE=native` manual test: point at a gateway that does support it; confirm native path works and is logged.
- Manual: enable flag for a test chat, run the sample conversation from the user's request:
  1. "Give me key findings of previous year work." → `rag_retrieve` → answer + `last_answer_object`.
  2. "Summarise it in 10 points." → planner sees `last_answer_object`, calls `summarize_answer(max_points=10)` → 10 bullets.
  3. "Give me key statistics in these findings." → `extract_data(source=retrieved_docs)` → inline stats.
  4. "Make it a pie chart." → `chart_generate` from extracted data → ECharts renders in the final answer.
  5. Upload a Word file, "Summarise this file." → `file_summarize` (map-reduce) → summary.
  6. Multi-part: "Give me findings on X, Y, and Z." → planner emits 3 subtasks → 3 `rag_retrieve` calls dispatched in parallel → synthesized answer. Confirm latency ≈ 1× not 3×.
- Confirm each step shows `tc:`/`to:` events in the SSE stream (curl the endpoint, inspect raw events).
- Confirm every tool call in the sample conversation wrote an audit row (query `tool_call_audit`).

**Exit criterion**: sample conversation works end-to-end under the flag (both tool-call modes); RBAC denial + audit verified; parallel dispatch verified; all loop tests pass; existing pipeline still works with flag off.

---

## Phase 3 — Frontend (minimal)

**Goal**: render the loop's new events in the UI as plain text/chips. No fancy panels in v1.

**Work**
1. `useReducer` for agent-turn state in `page.tsx` (`04` B1).
2. SSE handlers for `pl:`/`tc:`/`to:`/`la:` in `page.tsx` (`04` B2).
3. `agent-events.tsx` — minimal rendering: plan one-liner + collapsible subtask list, tool-call chips, observation chips (`04` B3).
4. `echarts-diagram.tsx` — kept as-is; charts appear in the final answer markdown (`04` B4).
5. Context-token indicator (optional, from `d:` usage) (`05` §6).

**Deferred to a later polish phase:** `plan-panel.tsx` (rich plan panel with dependency arrows), `tool-call-card.tsx` (per-tool-type expandable cards), `zustand` store, inline chart rendering in tool cards, file-action quick-action chips, "edit chart" affordance. The behavior does not depend on any of these.

**Verification**
- `npm run test:ci` from `frontend/` — existing tests pass.
- Visual: run the sample conversation with the flag on; confirm plan one-liner, tool-call chips, observation chips, and final-answer chart all render. Screenshot before/after per AGENTS.md §5.
- Lint: `next lint` clean.
- Typecheck: `tsc --noEmit` clean.

**Exit criterion**: frontend renders the new events minimally; existing chat/KB/admin UIs unchanged.

---

## Phase 4 — Eval parity and rollout

**Goal**: prove the loop is at least as good as the rigid pipeline, then enable it.

**Work**
1. Run the eval harness (`eval/`) against both paths (flag off vs on) on the SQuAD KB.
2. Compare: Precision@K, Recall@K, MRR, answer faithfulness/completeness scores.
3. Add eval cases for the new capabilities (multi-turn reference, file summarization, chart generation) — these have no SQuAD equivalent; write a small fixture-based eval set.
4. Enable per-org for a test org.
5. Enable globally.
6. Cleanup PR: delete `graph.py` rigid path, `classify_query_node`, subgraphs, `CLASSIFY_SYSTEM_PROMPT`, `QueryAnalysis` (`04` C2 step 4).

**Verification**
- Eval report: loop ≥ rigid pipeline on retrieval metrics; loop strictly better on multi-turn + file + chart cases (rigid pipeline cannot do them).
- No regression in existing `pytest` suite after cleanup.
- Loop is the sole production path; no feature flag required.

**Exit criterion**: loop is the production path; rigid pipeline deleted; eval parity documented.

---

## Phase 5 — Frontend polish (optional, post-rollout)

**Goal**: the rich agent UX deferred from Phase 3.

**Work**
1. `plan-panel.tsx` — intent badge, subtask list with `tool_hint` icons, dependency arrows, live status.
2. `tool-call-card.tsx` — per-tool-type expandable cards (retrieval docs, code + stdout, inline chart, file sections).
3. `zustand` store — extract agent-turn state from `page.tsx` reducer if re-renders are a measured problem.
4. Inline chart rendering in tool cards (chart appears as soon as `chart_generate` returns, before final answer).
5. `file-attachment.tsx` quick-action chips ("Summarize", "Extract tables").
6. "Edit chart" affordance — sends "change this chart to X" follow-up; agent re-calls `chart_generate` with same data + different `chart_type`.
7. Admin "agent activity" view — query `tool_call_audit` per chat/turn.

**Verification**: visual; `npm run test:ci`; `next lint`; `tsc --noEmit`.

**Exit criterion**: polish ships without behavior change.

---

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Local model is weak at tool selection | Medium | `tool_hint` from planner guides selection; `applicable_tools` filters out irrelevant tools; fall back to single-tool turns if multi-tool fails. Per-org model override so a stronger model can be used for agent turns. |
| Local gateway doesn't support native function-calling | High | JSON-text fallback in `tool_call_parser.py` (`02` §3.2). `TOOL_CALL_MODE` lets the operator force the working mode. Final-answer default prevents the loop from getting stuck. |
| `code_execute` sandbox escape | Low (internal users) / High (hostile) | RestrictedPython for v1 (AST-level restrictions). nsjail for hardened prod (Phase 5+). No network imports regardless. Timeout on every execution. |
| Loop runs forever / burns tokens | Medium | Hard caps: `AGENT_MAX_ITERATIONS`, per-tool caps. Forced finalize on budget exhaustion. Observe token spend per turn (audit table); alert if consistently near cap. |
| Chart builder doesn't cover a chart type the user wants | Medium | Builder supports pie/bar/line/scatter/area. For unsupported types, fall back to LLM-generated JSON with validation (current behavior). Document the supported set. |
| Tokenizer mismatch with local model | Medium | `TOKENIZER_MODEL` env var + safety margins (0.85 trigger, reserved generation). Fallback to `cl100k_base`. |
| Long-term recall noise (irrelevant turns injected) | Low | Top-3 only, truncated, below `last_answer_object` in priority (dropped first if over budget). |
| Migration breaks existing chats | Low | New columns nullable; `last_answer_object` populated going forward; old messages render normally. |
| Per-org LLM config wiring changes model mid-conversation | Low | Config is read per-turn; a mid-conversation change applies on the next turn. Acceptable and expected. |
| Prompt injection makes the planner pass another org's kb_id | Medium | Per-tool RBAC re-check is the enforcement boundary (`02` §5.1). The planner is not trusted; the tool denies and writes an audit row with `status=denied`. |
| `last_answer_object` extraction fails (malformed JSON) | Medium | Pydantic validation + one retry + rule-based regex fallback (`05` §2). Partial object still usable for "summarise it" / "give me the stats". |
| Reflect node doesn't recover from tool failures | Medium | Concrete replanning rules in `REFLECT_SYSTEM_PROMPT` (`02` §7), not left to LLM discretion alone. Empty retrieval → rewrite → re-retrieve (capped); chart invalid → re-extract + re-build; code error → retry with fix. |

---

## Rollback

The agent loop is the sole path — there is no feature flag or fallback to the rigid pipeline.

- **Before the rigid pipeline is deleted:** rollback requires reverting the cutover PR. Keep the cleanup PR small and isolated so revert is clean.
- **After the rigid pipeline is deleted:** rollback requires a full revert of the implementation PR. The database migration (Phase 0) is additive — new columns are nullable and the `tool_call_audit` table can be dropped. No data loss occurs on revert.

---

## Out of scope (explicit)

- Web search / web fetch (offline constraint).
- Image generation (no local model assumed for it; can be added later behind a tool if a local image model is deployed).
- Email/calendar/external-API actions (offline enterprise; not part of this plan).
- Voice input/output.
- Multi-agent orchestration (multiple agent loops coordinating). The single loop with subtasks covers the requirements; multi-agent adds complexity without clear benefit for this use case.
- Replacing Qdrant/Neo4j/Redis/MySQL. The data layer is correct for the workload.
- **Pruned from earlier draft of this plan:** `graph_retrieve` as a separate tool (merged into `rag_retrieve`), `memory_recall` as an on-demand tool (proactive recall covers it), `table_generate` (LLM markdown tables cover the example; add back only if unreliable in practice), `OrgLLMConfig` column expansion (`reasoning_model`/`vision_model`/etc. — env vars remain the source), `zustand`/`plan-panel`/`tool-call-card` fancy UI in v1 (deferred to Phase 5), `nsjail` sandbox in v1 (RestrictedPython first; nsjail is Phase 5+ hardening), "edit chart" `from_option`/`mutation` parameters (re-call `chart_generate` with same data + different type instead).
