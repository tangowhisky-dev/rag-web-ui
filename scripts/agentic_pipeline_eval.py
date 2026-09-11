#!/usr/bin/env python3
"""
Agentic retrieval pipeline — end-to-end stage evaluation script.

Drives the live RAG app over HTTP only (same style as presenton_test.py):
register/login → create KB → upload authority-tagged documents → set
document_status / effective dates → run chat queries → parse the SSE stream
and evaluate what the agent saw at every stage.

Corpus design (one fact per doc, deliberately colliding):

    travel-policy-v1.txt   superseded  2023-01-01 → 2024-06-30   cap = $45/day
    travel-policy-v2.txt   active      2024-07-01 → (ongoing)    cap = $60/day
    remote-work-draft.txt  draft       2026-01-01 → (ongoing)    3 days/week
    expense-policy-old.txt superseded  2022-01-01 → 2023-12-31   receipts > $25
    benefits-2030.txt      active      2030-01-01 → (ongoing)    gym $100/mo

Scenarios exercised:

    S1 current-state    "What is the current daily meal reimbursement cap?"
                        → agent should prefer active/in-window evidence ($60);
                          superseded $45 must not be presented as current.
    S2 historical       "What was the daily meal cap in 2023?"
                        → superseded doc must still be retrievable + labelled.
    S3 comparison       "How did the meal cap change?"
                        → both versions; answer should contrast 45 → 60.
    S4 draft discovery  "Is there a remote work policy being prepared?"
                        → draft doc tagged status=draft.
    S5 introspection    "List the documents and their lifecycle status."
                        → kb_metadata list_documents exposes status fields.

Per stage the script records:
    - tool calls      (tl: tool_call / subagent_step — name + compacted args)
    - context         (2: events — retrieved_docs metadata incl. authority tags)
    - answer          (0: tokens / r: rewrite — text + citations)
    - confidence      (2: confidence/score, d: usage)

Usage:
    python scripts/agentic_pipeline_eval.py \
        --base-url http://localhost:8000/api \
        --username super_admin --password super_admin123

All flags have env-var defaults (BASE_URL / USERNAME / PASSWORD).
"""

import argparse
import json
import os
import re
import sys
import time

import requests

# ── Test corpus ───────────────────────────────────────────────────────────────

DOCS = [
    {
        "file_name": "travel-policy-v1.txt",
        "content": (
            "Corporate Travel Policy (Version 1)\n\n"
            "Effective 2023.\n\n"
            "The daily meal reimbursement cap is USD 45 per day.\n"
            "Flights must be booked in economy class for all employees.\n"
            "Hotel stays are capped at USD 180 per night.\n"
        ),
        "document_status": "superseded",
        "effective_from": "2023-01-01",
        "effective_to": "2024-06-30",
        "version": "1.0",
    },
    {
        "file_name": "travel-policy-v2.txt",
        "content": (
            "Corporate Travel Policy (Version 2)\n\n"
            "Effective from July 2024 onward.\n\n"
            "The daily meal reimbursement cap is USD 60 per day.\n"
            "Premium economy class is permitted on flights longer than 6 hours.\n"
            "Hotel stays are capped at USD 220 per night.\n"
        ),
        "document_status": "active",
        "effective_from": "2024-07-01",
        "effective_to": None,
        "version": "2.0",
    },
    {
        "file_name": "remote-work-draft.txt",
        "content": (
            "Remote Work Policy (DRAFT — not yet approved)\n\n"
            "Under this draft policy, employees may work remotely up to "
            "3 days per week with manager approval.\n"
        ),
        "document_status": "draft",
        "effective_from": "2026-01-01",
        "effective_to": None,
        "version": "0.9",
    },
    {
        "file_name": "expense-policy-old.txt",
        "content": (
            "Expense Reporting Policy (Archived 2022 Edition)\n\n"
            "Receipts are required for all expenses above USD 25.\n"
            "Reports must be submitted within 60 days of the expense date.\n"
        ),
        "document_status": "superseded",
        "effective_from": "2022-01-01",
        "effective_to": "2023-12-31",
        "version": "1.0",
    },
    {
        "file_name": "benefits-2030.txt",
        "content": (
            "Employee Benefits Policy (Future Version, takes effect 2030)\n\n"
            "From 2030, every employee receives a gym stipend of USD 100 "
            "per month and a home-office allowance of USD 500 per year.\n"
        ),
        "document_status": "active",
        "effective_from": "2030-01-01",
        "effective_to": None,
        "version": "3.0",
    },
    {
        # Multi-chunk doc (~4 chunks at CHUNK_SIZE=1500, 20% overlap) — the
        # incident-response fact sits mid-document so a hit on it should
        # pull in neighbor chunks via inject_neighbor_context.
        "file_name": "security-handbook.txt",
        "content": (
            "Corporate Security Handbook\n\n"
            "Section 1 — Access Control\n"
            "All physical access requires a badge. Tailgating is prohibited. "
            "Visitors must be signed in at reception and escorted at all times. "
            "Badge photographs are refreshed every two years. Lost badges must be "
            "reported to security within one working day. Access rights follow "
            "least-privilege and are reviewed quarterly by team leads. Contractors "
            "receive time-boxed badges that expire automatically at contract end. "
            + "Filler text to pad this section out for chunking purposes. " * 18
            + "\nSection 2 — Network Security\n"
            "All remote administrative access requires the corporate VPN. "
            "Split tunnelling is disabled on managed devices. Guest Wi-Fi is "
            "isolated on a separate VLAN and may not reach internal subnets. "
            "Firewall rule changes require change-board approval and a rollback "
            "plan. DNS egress is restricted to the corporate resolvers. "
            + "More filler text so this section crosses a chunk boundary. " * 18
            + "\nSection 3 — Incident Response\n"
            "Security incidents must be reported to the SOC within 4 hours of "
            "discovery. Severity-1 incidents convene the response bridge within "
            "30 minutes. Evidence preservation takes priority over remediation. "
            "Post-incident reviews are mandatory within 10 business days. "
            + "Padding sentences to keep this middle section well inside the doc. " * 18
            + "\nSection 4 — Data Classification\n"
            "Data is classified public, internal, confidential, or restricted. "
            "Restricted data requires encryption at rest and in transit. "
            "Classification labels appear in document footers and metadata. "
            + "Trailing filler so the document ends with a final section chunk. " * 18
            + "\n"
        ),
        "document_status": "active",
        "effective_from": "2025-01-01",
        "effective_to": None,
        "version": "1.0",
    },
]

SCENARIOS = [
    {
        "id": "S1_current",
        "question": "What is the current daily meal reimbursement cap for corporate travel?",
        "expect_answer_contains": ["60"],
        "expect_answer_absent_current": ["45"],  # may appear only if qualified as old
        "expect_status_in_context": {"active"},
        "note": "current-state query — $60 must win; $45 is superseded",
    },
    {
        "id": "S2_historical",
        "question": "What was the daily meal reimbursement cap under the travel policy in 2023?",
        "expect_answer_contains": ["45"],
        "expect_status_in_context": {"superseded"},
        "note": "historical query — superseded evidence must surface, labelled",
    },
    {
        "id": "S3_comparison",
        "question": "How did the daily meal reimbursement cap change between the old and current travel policy?",
        "expect_answer_contains": ["45", "60"],
        "expect_status_in_context": set(),
        "note": "comparison — both versions must be citable",
    },
    {
        "id": "S4_draft",
        "question": "Is there a remote work policy being prepared? What does it propose?",
        "expect_answer_contains": ["3"],
        "expect_status_in_context": {"draft"},
        "note": "draft discovery — status=draft must be visible to the LLM",
    },
    {
        "id": "S5_introspect",
        "question": "List the documents in this knowledge base with their lifecycle status.",
        "expect_answer_contains": [],
        "expect_status_in_context": set(),
        "note": "kb_metadata list_documents must return status/effective fields",
    },
    {
        "id": "S6_neighbors",
        "question": "What does the security handbook require for security incident response reporting?",
        "expect_answer_contains": ["4 hours"],
        "expect_status_in_context": set(),
        "expect_neighbors": True,
        "note": "chunk hit on Section 3 → prev/next neighbor chunks injected, contiguous, deduplicated",
    },
]

AUTHORITY_KEYS = ("document_status", "effective_from", "effective_to", "version")

# ── HTTP client ───────────────────────────────────────────────────────────────


class Client:
    def __init__(self, base: str, timeout: int = 30):
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.s = requests.Session()
        self.token = None

    def _h(self, json_body=True):
        h = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        if json_body:
            h["Content-Type"] = "application/json"
        return h

    def login(self, username, password):
        r = self.s.post(f"{self.base}/auth/token",
                        data={"username": username, "password": password},
                        headers={"Content-Type": "application/x-www-form-urlencoded"},
                        timeout=self.timeout)
        r.raise_for_status()
        self.token = r.json()["access_token"]

    def create_kb(self, name):
        r = self.s.post(f"{self.base}/knowledge-base",
                        json={"name": name, "description": "agentic eval"},
                        headers=self._h(), timeout=self.timeout)
        r.raise_for_status()
        return r.json()["id"]

    def upload(self, kb_id, filename, content):
        r = self.s.post(f"{self.base}/knowledge-base/{kb_id}/documents/upload",
                        files=[("files", (filename, content.encode(), "text/plain"))],
                        headers=self._h(json_body=False), timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def process(self, kb_id, upload_results):
        r = self.s.post(f"{self.base}/knowledge-base/{kb_id}/documents/process",
                        json=upload_results, headers=self._h(), timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def ingest_status(self, kb_id):
        r = self.s.get(f"{self.base}/query/kb/{kb_id}/ingest-status",
                       headers=self._h(), timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def get_kb(self, kb_id):
        r = self.s.get(f"{self.base}/knowledge-base/{kb_id}",
                       headers=self._h(), timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def get_markdown(self, kb_id, doc_id):
        r = self.s.get(f"{self.base}/knowledge-base/{kb_id}/documents/{doc_id}/markdown",
                       headers=self._h(), timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def put_markdown(self, kb_id, doc_id, body):
        r = self.s.put(f"{self.base}/knowledge-base/{kb_id}/documents/{doc_id}/markdown",
                       json=body, headers=self._h(), timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def create_chat(self, title, kb_ids):
        r = self.s.post(f"{self.base}/chat",
                        json={"title": title, "knowledge_base_ids": kb_ids},
                        headers=self._h(), timeout=self.timeout)
        r.raise_for_status()
        return r.json()["id"]

    def chat_message_stream(self, chat_id, question, debug=True):
        """POST a user message; yields parsed SSE data-channel lines.

        debug=true turns on `type="debug"` timeline events: tool_observation
        (tool input/output), think_input (the think-node prompt), and
        finalize_context (the exact evidence block the answer LLM cites).
        """
        r = self.s.post(f"{self.base}/chat/{chat_id}/messages",
                        json={"messages": [{"role": "user", "content": question}],
                              "debug": debug},
                        headers=self._h(), timeout=None, stream=True)
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue
            m = re.match(r"^([a-z0-9]+):(.*)$", raw)
            if not m:
                continue  # ':' flush comments and blank lines
            channel, payload = m.group(1), m.group(2)
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                data = payload
            yield channel, data


# ── SSE scenario runner ───────────────────────────────────────────────────────


def run_scenario(client, chat_id, scenario, max_wait=900):
    """Run one query, collecting per-stage events from the SSE stream."""
    timeline = []          # tl: events
    context_events = []    # 2: events (retrieved docs + confidence)
    thinking = []          # th: events
    tokens = []            # 0: answer text chunks
    rewrite = None         # r: final answer + citations
    done = None            # d: usage
    plan = None            # pl:
    interrupt = None       # c: clarification
    debug_events = []      # tl: type="debug" stage internals

    t0 = time.time()
    for channel, data in client.chat_message_stream(chat_id, scenario["question"]):
        if time.time() - t0 > max_wait:
            return {"error": f"timeout after {max_wait}s"}
        if channel == "tl":
            timeline.append(data)
            if data.get("type") == "debug":
                debug_events.append(data)
        elif channel == "2":
            context_events.append(data)
        elif channel == "th":
            thinking.append(data)
        elif channel == "0":
            tokens.append(data if isinstance(data, str) else json.dumps(data))
        elif channel == "r":
            rewrite = data
        elif channel == "d":
            done = data
        elif channel == "pl":
            plan = data
        elif channel == "c":
            interrupt = data

    answer = rewrite.get("content") if rewrite else "".join(tokens)

    # Split debug events by stage for per-stage assertions.
    by_stage = {}
    for e in debug_events:
        by_stage.setdefault(e.get("stage", "?"), []).append(e.get("data", {}))

    return {
        "timeline": timeline,
        "context_events": context_events,
        "thinking": thinking,
        "answer": answer or "",
        "citations": (rewrite or {}).get("citations", []),
        "done": done,
        "plan": plan,
        "interrupt": interrupt,
        "elapsed_s": round(time.time() - t0, 1),
        "debug": by_stage,
    }


# ── Stage evaluators ──────────────────────────────────────────────────────────


def _doc_meta(doc):
    return doc.get("metadata", {}) if isinstance(doc, dict) else {}


def _collect_hit_statuses(tool_obs):
    """Statuses seen on tool outputs (search hits, title_search docs, file_read)."""
    statuses = set()
    for obs in tool_obs:
        res = obs.get("result") or {}
        if not isinstance(res, dict):
            continue
        if res.get("document_status"):
            statuses.add(res["document_status"])
        for key in ("hits", "docs", "documents"):
            for item in res.get(key) or []:
                if isinstance(item, dict):
                    s = item.get("document_status") or \
                        (item.get("metadata") or {}).get("document_status")
                    if s:
                        statuses.add(s)
    return statuses


def evaluate(scenario, result, doc_map):
    """Score one scenario's stage inputs/outputs. Returns (checks, notes).

    Each check: (name, passed: bool|None, detail).  None = informational /
    not assertable (LLM-dependent behaviour we surface but don't hard-fail).
    """
    checks, notes = [], []
    dbg = result.get("debug", {})

    # ── Stage 0: did the turn complete? ────────────────────────────────────
    if result.get("error"):
        checks.append(("stream_completed", False, result["error"]))
        return checks, notes
    checks.append(("stream_completed", result["done"] is not None or bool(result["answer"]),
                   f"{result['elapsed_s']}s"))
    if result.get("interrupt"):
        notes.append("agent asked for clarification (interrupt event)")

    # ── Stage 1: tool INPUTS — what the agent invoked, with what args ──────
    tool_calls = [e for e in result["timeline"]
                  if e.get("type") in ("tool_call", "subagent_step") and e.get("tool")]
    tool_names = [e.get("tool") for e in tool_calls]
    retrieval_tools = {"keyword_search", "semantic_search", "title_search",
                       "retrieve_parallel", "graph_expand", "file_read",
                       "kb_metadata", "kb_grep", "kb_outline"}
    used_retrieval = [t for t in tool_names if t in retrieval_tools]
    checks.append(("retrieval_tool_called", bool(used_retrieval),
                   f"tools={tool_names or 'none'}"))

    authority_filter_used = False
    for e in tool_calls:
        args = e.get("arguments") or {}
        f = args.get("filters") or {}
        if any(k in f for k in ("document_status", "effective_as_of",
                                "effective_window_start")) or \
           any(k in args for k in ("document_status", "effective_as_of")):
            authority_filter_used = True
    # Filters are a capability, not a requirement — report usage, don't fail.
    checks.append(("authority_filter_used", True if authority_filter_used else None,
                   "informational — LLM may answer from tagged hits without a hard filter"))

    # ── Stage 2: tool OUTPUTS — every executed call emits its I/O ──────────
    tool_obs = dbg.get("tool_observation", [])
    n_tool_results = len([e for e in result["timeline"]
                          if e.get("type") in ("tool_result", "subagent_step")
                          and e.get("status") == "complete" and e.get("tool")])
    checks.append(("tool_io_coverage", len(tool_obs) >= n_tool_results or not tool_calls,
                   f"{len(tool_obs)} tool_observation events vs {n_tool_results} completions"))
    obs_statuses = _collect_hit_statuses(tool_obs)
    checks.append(("tool_outputs_tagged", bool(obs_statuses) or not all_ctx_docs_or_hits(tool_obs),
                   f"statuses on tool outputs: {sorted(obs_statuses) or 'none'}"))  # noqa: E501

    # ── Stage 3: think INPUT — the prompt the ReAct loop consumed ──────────
    think_prompts = [d.get("prompt", "") for d in dbg.get("think_input", [])]
    sub_prompts = [d.get("prompt", "") for d in dbg.get("subagent_think_input", [])]
    all_think = "\n".join(think_prompts + sub_prompts)
    checks.append(("think_input_seen", bool(think_prompts or sub_prompts),
                   f"{len(think_prompts)} think + {len(sub_prompts)} subagent prompts"))
    # Markers appear as `status=X` (evidence/doc lines), `status_counts={...}`
    # (hit summaries) or `"document_status"` (kb_metadata JSON dumps).
    think_sees_status = ("status=" in all_think or "status_counts=" in all_think
                         or "document_status" in all_think)
    if obs_statuses - {"active"}:
        checks.append(("status_visible_in_think", think_sees_status,
                       f"observed {sorted(obs_statuses)}, markers={'yes' if think_sees_status else 'NO'}"))

    # ── Stage 4: context (2:) events — merged docs reaching state ──────────
    # Only doc-content retrieval merges into retrieved_docs; metadata-only
    # tools (kb_metadata list_documents) legitimately emit no context event.
    evidence_retrieved = _evidence_retrieved(tool_obs)
    all_ctx_docs = []
    for ev in result["context_events"]:
        all_ctx_docs.extend(ev.get("docs", []))
    ctx_statuses = {_doc_meta(d).get("document_status") for d in all_ctx_docs}
    ctx_statuses.discard(None)
    per_event_counts = [len(ev.get("docs", [])) for ev in result["context_events"]]
    checks.append(("context_docs_seen",
                   True if (all_ctx_docs or not evidence_retrieved) else False,
                   f"{len(all_ctx_docs)} ctx docs across {len(per_event_counts)} events "
                   f"(per-event={per_event_counts}), statuses={sorted(ctx_statuses) or 'none'}"))

    # ── Stage 5: evidence block the answer was written from ────────────────
    # v2: when the think node writes the answer inline, the citable evidence
    # lives in the LAST think prompt's "Retrieved evidence" section. The
    # separate finalize_context event only fires on the fallback path.
    fin = dbg.get("finalize_context", [])
    fin_text = "\n".join(d.get("context_text", "") for d in fin)
    fin_statuses = set(re.findall(r"status=(\w+)", fin_text))
    checks.append(("finalize_context_seen", True if fin_text else None,
                   f"fallback path only — {len(fin)} events"))

    last_think = think_prompts[-1] if think_prompts else ""
    ev_block = last_think.split("Retrieved evidence")[-1] if "Retrieved evidence" in last_think else ""
    ev_statuses = set(re.findall(r"status=(\w+)", ev_block))
    if ctx_statuses - {"active"}:
        checks.append(("status_markers_in_evidence", (ctx_statuses - {"active"}) <= ev_statuses,
                       f"ctx={sorted(ctx_statuses)} evidence-markers={sorted(ev_statuses)}"))

    expected = scenario.get("expect_status_in_context") or set()
    if expected:
        seen_anywhere = ctx_statuses | obs_statuses | fin_statuses
        checks.append(("expected_statuses_visible", expected <= seen_anywhere,
                       f"want {sorted(expected)}; seen {sorted(seen_anywhere)}"))

    # ── Stage 6: answer OUTPUT ─────────────────────────────────────────────
    ans = result["answer"]
    checks.append(("answer_nonempty", bool(ans.strip()), f"{len(ans)} chars"))
    for needle in scenario.get("expect_answer_contains", []):
        checks.append((f"answer_contains '{needle}'", needle in ans, ""))

    cited_ids = set()
    for c in result["citations"]:
        ref = c.get("citation_ref") or _doc_meta(c).get("citation_ref") or {}
        did = ref.get("document_id") or _doc_meta(c).get("document_id")
        if did is not None:
            cited_ids.add(did)
    known_ids = set(doc_map.values())
    if cited_ids:
        checks.append(("citations_in_kb", cited_ids <= known_ids,
                       f"cited={sorted(cited_ids)} uploaded={sorted(known_ids)}"))

    # ── Dedup: the LAST context event is the final merged retrieved_docs —
    # check it for chunk-level duplicates (earlier events are cumulative
    # snapshots of the same list, so cross-event repeats are expected).
    final_docs = (result["context_events"][-1].get("docs", [])
                  if result["context_events"] else [])
    seen_keys = set()
    dupes = []
    for d in final_docs:
        m = _doc_meta(d)
        key = m.get("content_hash") or \
            (m.get("document_id"), m.get("chunk_index"), len(d.get("page_content") or ""))
        if key in seen_keys:
            dupes.append(key)
        seen_keys.add(key)
    checks.append(("no_duplicate_chunks", not dupes,
                   f"dupes={dupes[:3]}" if dupes else f"{len(seen_keys)} unique ctx docs"))

    # ── Neighbor-chunk injection (multi-chunk scenario) ─────────────────────
    # Conditional: only assertable when a chunk-level search actually ran —
    # the model may legitimately use file_read instead (whole-doc evidence).
    if scenario.get("expect_neighbors"):
        n_hits = 0
        n_neighbor = 0
        for o in tool_obs:
            res = o.get("result") or {}
            for h in (res.get("hits") or []):
                if isinstance(h, dict):
                    n_hits += 1
                    if h.get("_is_neighbor"):
                        n_neighbor += 1
        by_doc: dict = {}
        for d in final_docs:
            m = _doc_meta(d)
            if m.get("document_id") is not None and m.get("chunk_index") is not None:
                by_doc.setdefault(m["document_id"], set()).add(m["chunk_index"])
        contiguous = next(
            ((did, sorted(ci)) for did, ci in by_doc.items() if len(ci) > 1), None)
        if n_hits == 0:
            checks.append(("neighbors_injected", None,
                           "no chunk search ran this turn — injection path not exercised"))
        else:
            checks.append(("neighbors_injected", n_neighbor > 0 or contiguous is not None,
                           f"{n_hits} hits, {n_neighbor} _is_neighbor; contiguous run={contiguous}"))

    return checks, notes


def all_ctx_docs_or_hits(tool_obs):
    """True if any tool output carried doc-shaped items (hits/docs/documents)
    where status tagging is checkable."""
    for obs in tool_obs:
        res = obs.get("result") or {}
        if isinstance(res, dict) and (res.get("hits") or res.get("docs") or res.get("documents")):
            return True
    return False


def _evidence_retrieved(tool_obs):
    """True if content-bearing evidence flowed (search hits, docs with
    content, file_read) — i.e. a context event is expected."""
    for obs in tool_obs:
        res = obs.get("result") or {}
        if not isinstance(res, dict):
            continue
        if res.get("content"):
            return True
        for key in ("hits", "docs"):
            for item in res.get(key) or []:
                if isinstance(item, dict) and (item.get("content_len") or item.get("page_content")):
                    return True
    return False


def print_scenario_report(scenario, result, checks, notes, verbose=False):
    print(f"\n{'─' * 70}")
    print(f"{scenario['id']}: {scenario['question']}")
    print(f"  note: {scenario['note']}")
    for name, passed, detail in checks:
        mark = "PASS" if passed is True else ("FAIL" if passed is False else "info")
        print(f"  [{mark:4}] {name}  {detail}")
    for n in notes:
        print(f"  note: {n}")
    if verbose:
        print(f"\n  ANSWER: {result['answer'][:600]}")
        for e in result["timeline"]:
            if e.get("type") in ("tool_call", "subagent_step") and e.get("tool"):
                args = json.dumps(e.get("arguments"), default=str)
                print(f"  tool> {e['tool']} args={args[:200]}")
            elif e.get("type") == "tool_result":
                print(f"  res>  {e.get('tool')}: {e.get('summary', '')[:100]}")
        # Per-stage debug dumps — the literal inputs/outputs the agent saw.
        for stage, events in (result.get("debug") or {}).items():
            print(f"\n  ── stage: {stage} ({len(events)} events) ──")
            for d in events:
                body = json.dumps(d, default=str)
                print(f"     {body[:800]}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="Agentic pipeline E2E stage evaluation")
    p.add_argument("--base-url", default=os.getenv("BASE_URL", "http://localhost:8000/api"))
    p.add_argument("--username", default=os.getenv("USERNAME", "super_admin"))
    p.add_argument("--password", default=os.getenv("PASSWORD", "super_admin123"))
    p.add_argument("--verbose", "-v", action="store_true", help="print answers + tool args")
    p.add_argument("--keep-kb", action="store_true", help="don't delete the KB afterwards")
    p.add_argument("--kb-id", type=int, default=None,
                   help="reuse an existing KB that already has the eval docs tagged")
    p.add_argument("--only", default=None, help="run a single scenario id, e.g. S6_neighbors")
    args = p.parse_args()

    client = Client(args.base_url)
    print(f"=== Agentic pipeline E2E eval ===")
    print(f"Target: {args.base_url}\n")

    print("1. Logging in…")
    client.login(args.username, args.password)
    print("   ok")

    doc_map = {}  # file_name -> document_id
    kb_id = args.kb_id

    if kb_id is None:
        print("\n2. Creating KB + uploading documents…")
        kb_id = client.create_kb("agentic-eval-" + str(int(time.time())))
        uploads = []
        for d in DOCS:
            uploads.extend(client.upload(kb_id, d["file_name"], d["content"]))
        client.process(kb_id, uploads)
        print(f"   kb_id={kb_id}, {len(uploads)} uploads queued")

        print("\n3. Waiting for ingest…")
        deadline = time.time() + 600
        while time.time() < deadline:
            st = client.ingest_status(kb_id)
            print(f"   {st.get('completed')}/{st.get('total')} done, {st.get('failed')} failed", end="\r")
            if st.get("ready"):
                print("\n   ingest complete")
                break
            if st.get("failed", 0) > 0 and st.get("completed", 0) + st.get("failed", 0) >= st.get("total", 0):
                print("\n   ingest finished with failures")
                break
            time.sleep(4)
        else:
            print("\n   TIMEOUT waiting for ingest")
            sys.exit(1)

        print("\n4. Tagging documents with lifecycle metadata…")
        kb = client.get_kb(kb_id)
        docs_by_name = {d["file_name"]: d for d in kb.get("documents", [])}
        for spec in DOCS:
            doc = docs_by_name.get(spec["file_name"])
            if not doc:
                print(f"   !! {spec['file_name']} not found in KB — skipped")
                continue
            doc_map[spec["file_name"]] = doc["id"]
            md = client.get_markdown(kb_id, doc["id"])
            client.put_markdown(kb_id, doc["id"], {
                "markdown": md["markdown"],
                "lock_version": md["lock_version"],
                "document_status": spec["document_status"],
                "effective_from": spec["effective_from"],
                "effective_to": spec["effective_to"],
                "version": spec["version"],
            })
            print(f"   {spec['file_name']} (doc {doc['id']}): "
                  f"{spec['document_status']} {spec['effective_from']}→{spec['effective_to'] or '…'} v{spec['version']}")
    else:
        print(f"\n2. Reusing KB {kb_id} — skipping ingest + tagging")
        kb = client.get_kb(kb_id)
        doc_map = {d["file_name"]: d["id"] for d in kb.get("documents", [])}

    if len(doc_map) < len(DOCS) and args.kb_id is None:
        print("   !! not all eval docs were tagged — continuing anyway")

    print("\n5. Running scenarios…")
    scenarios = [s for s in SCENARIOS if args.only is None or s["id"] == args.only]
    if not scenarios:
        print(f"   !! --only {args.only} matched nothing")
        sys.exit(1)
    all_results = []
    for scenario in scenarios:
        chat_id = client.create_chat(f"eval-{scenario['id']}", [kb_id])
        print(f"\n   {scenario['id']} → chat {chat_id}: {scenario['question'][:70]}…")
        result = run_scenario(client, chat_id, scenario)
        checks, notes = evaluate(scenario, result, doc_map)
        print_scenario_report(scenario, result, checks, notes, verbose=args.verbose)
        all_results.append((scenario["id"], checks, result))

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    hard_fail = 0
    for sid, checks, _ in all_results:
        fails = [n for n, p, _ in checks if p is False]
        infos = [n for n, p, _ in checks if p is None]
        status = "FAIL" if fails else "PASS"
        if fails:
            hard_fail += 1
        print(f"  {sid:<16} {status}  failed={fails or 'none'}  info={infos or 'none'}")

    print(f"\n  hard failures: {hard_fail}")
    if not args.keep_kb and args.kb_id is None:
        client.s.delete(f"{client.base}/knowledge-base/{kb_id}", headers=client._h())
        print(f"  cleaned up KB {kb_id}")
    else:
        print(f"  KB {kb_id} kept (rerun with --kb-id {kb_id} to skip ingest)")
    sys.exit(1 if hard_fail else 0)


if __name__ == "__main__":
    main()
