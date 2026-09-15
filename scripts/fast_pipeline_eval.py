#!/usr/bin/env python3
"""Fast-pipeline E2E monitor — per-stage I/O dump for the single-round pipeline.

Drives the live app over HTTP+SSE against an EXISTING knowledge base (no
ingestion). All turns share one chat so mode switching mid-conversation is
exercised (fast → fast follow-up → agentic → fast …).

Monitors per turn:
  pl:   plan event — fast_plan's intent / resolved_query / emitted tool_calls
  tl:   timeline — tool_call inputs (arguments), tool_observation outputs,
        phase steps, debug stage internals (fast_plan prompt + raw output,
        tool payloads, finalize context)
  2:    context — docs the answer LLM cites; checked for chunk duplication,
        neighbor injection, and unnecessary trimming vs. tool hits
  r:/0: final answer + citations

Usage:
    python3 fast_pipeline_eval.py --kb 3                 # lifecycle corpus
    python3 fast_pipeline_eval.py --kb 2 --chat-id 42    # reuse a chat
    python3 fast_pipeline_eval.py --kb 3 --turns fast,fast,agentic

Run inside the backend container (requests + API on :8000):
    docker cp scripts/fast_pipeline_eval.py rag-web-ui-backend-1:/tmp/fp.py
    docker exec rag-web-ui-backend-1 python /tmp/fp.py --kb 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time

import requests

# ── Per-KB scripted conversations ────────────────────────────────────────────
# Each turn: mode, question, expectations. `checks` are advisory expectations
# evaluated as pass/fail/info after the stream finishes.

KB3_TURNS = [
    {
        "id": "T1_current_fast",
        "mode": "fast",
        "question": "What is the current daily meal reimbursement cap for corporate travel?",
        "expect_answer_contains": ["60"],
        "expect_intent": "retrieve",
        "expect_tools_any": {"keyword_search", "semantic_search"},
        "expect_status_in_context": {"active"},
    },
    {
        "id": "T2_followup_fast",
        "mode": "fast",
        "question": "Summarize the answer you just gave.",
        "expect_answer_contains": ["60"],
        "expect_intent": "answer_from_history",
        "expect_tools_any": set(),          # no retrieval expected
        "expect_status_in_context": set(),
    },
    {
        "id": "T3_compare_agentic",
        "mode": "agentic",
        "question": "What was the meal cap in 2023, and how does it compare to the current one?",
        "expect_answer_contains": ["45", "60"],
        "expect_status_in_context": {"superseded", "active"},
    },
    {
        "id": "T4_draft_fast",
        "mode": "fast",
        "question": "How many days per week is the remote work draft proposing?",
        "expect_answer_contains": ["3"],
        "expect_intent": "retrieve",
        "expect_tools_any": set(),
        "expect_status_in_context": {"draft"},
    },
    {
        "id": "T5_fast_again",
        "mode": "fast",
        "question": "What gym stipend does the future benefits policy grant, and when does it take effect?",
        "expect_answer_contains": ["100", "2030"],
        "expect_intent": "retrieve",
        "expect_status_in_context": {"active"},
    },
    {
        "id": "T6_direct_fast",
        "mode": "fast",
        "question": "What is 12 percent of 340? Answer with just the number.",
        "expect_answer_contains": ["40.8"],
        "expect_intent": "direct",
        "expect_tools_any": set(),          # no retrieval — pure knowledge
        "expect_status_in_context": set(),
    },
    {
        "id": "T7_chat_summary_fast",
        "mode": "fast",
        "question": "Summarize this chat so far in two bullet points.",
        "expect_answer_contains": [],
        "expect_intent": None,            # answer_from_history or a summarize step both OK
        "expect_tools_any": set(),
        "expect_status_in_context": set(),
    },
    {
        "id": "T8_chain_fast",
        "mode": "fast",
        "question": (
            "Extract the daily meal caps from the 2023 and current travel policies, "
            "compute the percentage increase, and chart the two values."
        ),
        "expect_answer_contains": ["45", "60"],
        "expect_intent": "retrieve",
        # INFO-level: which processing steps the planner actually emitted
        "watch_step_tools": {"extract_data", "code_execute", "chart_generate"},
    },
]

KB2_TURNS = [
    {
        "id": "T1_arxiv_fast",
        "mode": "fast",
        "question": "What is this paper about? Give the main contribution.",
        "expect_answer_contains": [],
        "expect_intent": "retrieve",
        "expect_tools_any": {"keyword_search", "semantic_search"},
        "expect_status_in_context": set(),
    },
    {
        "id": "T2_followup_fast",
        "mode": "fast",
        "question": "Summarize that in two sentences.",
        "expect_answer_contains": [],
        "expect_intent": "answer_from_history",
        "expect_tools_any": set(),
        "expect_status_in_context": set(),
    },
    {
        "id": "T3_agentic",
        "mode": "agentic",
        "question": "Which specific methods or models does the paper evaluate?",
        "expect_answer_contains": [],
        "expect_status_in_context": set(),
    },
]

KB_SCRIPTS = {3: KB3_TURNS, 2: KB2_TURNS}


class Client:
    def __init__(self, base, timeout=120):
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

    def create_chat(self, title, kb_ids):
        r = self.s.post(f"{self.base}/chat",
                        json={"title": title, "knowledge_base_ids": kb_ids},
                        headers=self._h(), timeout=self.timeout)
        r.raise_for_status()
        return r.json()["id"]

    def stream(self, chat_id, question, mode=None, debug=True):
        body = {"messages": [{"role": "user", "content": question}], "debug": debug}
        if mode:
            body["mode"] = mode
        r = self.s.post(f"{self.base}/chat/{chat_id}/messages",
                        json=body, headers=self._h(), timeout=None, stream=True)
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue
            m = re.match(r"^([a-z0-9]+):(.*)$", raw)
            if not m:
                continue
            try:
                data = json.loads(m.group(2))
            except json.JSONDecodeError:
                data = m.group(2)
            yield m.group(1), data


def run_turn(client, chat_id, turn, max_wait=600):
    """Stream one turn; collect all channels."""
    out = {"timeline": [], "context": [], "thinking": [], "tokens": [],
           "rewrite": None, "done": None, "plan": None, "debug": {}}
    t0 = time.time()
    for ch, data in client.stream(chat_id, turn["question"], mode=turn.get("mode")):
        if time.time() - t0 > max_wait:
            out["error"] = f"timeout {max_wait}s"
            break
        if ch == "tl":
            out["timeline"].append(data)
            if data.get("type") == "debug":
                out["debug"].setdefault(data.get("stage", "?"), []).append(data.get("data", {}))
        elif ch == "2":
            out["context"].append(data)
        elif ch == "th":
            out["thinking"].append(data)
        elif ch == "0":
            out["tokens"].append(data if isinstance(data, str) else json.dumps(data))
        elif ch == "r":
            out["rewrite"] = data
        elif ch == "d":
            out["done"] = data
        elif ch == "pl":
            out["plan"] = data
    out["answer"] = (out["rewrite"] or {}).get("content") or "".join(out["tokens"])
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def _content_key(doc):
    """Dedup key: content_hash if present else a hash of the text."""
    meta = doc.get("metadata") or {}
    h = meta.get("content_hash") or doc.get("content_hash")
    if h:
        return h
    text = doc.get("page_content") or doc.get("content") or ""
    return "sha:" + hashlib.sha256(text.encode()).hexdigest()[:16]


def check_turn(turn, out):
    """Return list of (name, pass|fail|info, detail)."""
    checks = []

    def add(name, result, detail=""):
        checks.append((name, result, detail))

    add("stream_completed", out["done"] is not None or out["answer"] != "")
    add("answer_nonempty", bool((out["answer"] or "").strip()),
        f"{len(out['answer'])} chars")
    for frag in turn.get("expect_answer_contains", []):
        add(f"answer_contains[{frag!r}]", frag.lower() in out["answer"].lower())

    # ── plan stage ──
    plan = out.get("plan") or {}
    plan_obj = plan.get("plan") if isinstance(plan.get("plan"), dict) else plan
    tool_calls = [e for e in out["timeline"] if e.get("type") == "tool_call"]
    tools_used = [e.get("tool") for e in tool_calls]
    if turn.get("mode") == "fast":
        add("plan_event", bool(plan), f"intent={plan_obj.get('intent')!r} resolved={plan_obj.get('resolved_query')!r}")
        exp_intent = turn.get("expect_intent")
        if exp_intent:
            add(f"intent={exp_intent}", plan_obj.get("intent") == exp_intent,
                f"got {plan_obj.get('intent')!r}")
        exp_tools = turn.get("expect_tools_any")
        if exp_tools:
            add("expected_tools_fired", bool(set(tools_used) & exp_tools), f"tools={tools_used}")
        if exp_intent in ("answer_from_history", "direct"):
            add("no_retrieval_calls", not tools_used, f"tools={tools_used}")
        # fast_plan debug stage must exist (prompt + raw model output)
        fp = out["debug"].get("fast_plan") or []
        add("fast_plan_debug", bool(fp),
            f"{len(fp)} event(s)" if fp else "missing — no stage internals")
    else:
        add("agentic_loop", "info", f"tools={tools_used}")

    # ── tool I/O ──
    tool_obs = out["debug"].get("tool_observation") or []
    n_hits_total = 0
    statuses = set()
    dup_tool_calls = 0
    seen_sig = set()
    for e in tool_calls:
        sig = (e.get("tool"), json.dumps(e.get("arguments") or {}, sort_keys=True))
        if sig in seen_sig:
            dup_tool_calls += 1
        seen_sig.add(sig)
    add("no_duplicate_tool_calls", dup_tool_calls == 0,
        f"{dup_tool_calls} identical tool+args pairs" if dup_tool_calls else f"{len(tool_calls)} unique calls")

    for obs in tool_obs:
        res = (obs.get("data") or {}).get("result") or obs.get("result") or {}
        for key in ("hits", "docs", "documents"):
            items = res.get(key) or []
            n_hits_total += len(items)
            for it in items:
                if isinstance(it, dict):
                    s = it.get("document_status") or (it.get("metadata") or {}).get("document_status")
                    if s:
                        statuses.add(s)
        if res.get("document_status"):
            statuses.add(res["document_status"])
    if tools_used:
        add("tool_observations_seen", bool(tool_obs), f"{len(tool_obs)} obs, {n_hits_total} hits")

    # ── context docs: dedup + neighbors + no unnecessary trimming ──
    # retrieved_docs is a _last_value channel: each tool node emits the FULL
    # merged list, so `2:` events are cumulative snapshots. Judge dedup on the
    # FINAL event (the merged state), not the flattened stream.
    ctx_docs = []
    for ce in out["context"]:
        ctx_docs.extend(ce.get("docs") or ce.get("documents") or [])
    last_ctx_docs = []
    for ce in out["context"]:
        d = ce.get("docs") or ce.get("documents") or []
        if d:
            last_ctx_docs = d
    keys = [_content_key(d) for d in last_ctx_docs]
    intra_event_dups = len(keys) - len(set(keys))
    add("no_duplicate_context_docs", intra_event_dups == 0,
        f"final context event: {len(last_ctx_docs)} docs, {len(set(keys))} unique"
        + (f" (stream snapshots carried {len(ctx_docs)} doc entries)" if len(ctx_docs) != len(last_ctx_docs) else ""))
    n_neighbors = sum(1 for d in last_ctx_docs
                      if (d.get("metadata") or {}).get("_is_neighbor") or d.get("_is_neighbor"))
    if n_neighbors:
        add("neighbors_injected", "info", f"{n_neighbors} neighbor chunk(s)")
    watch = turn.get("watch_step_tools")
    if watch:
        used_step_tools = {t for t in tools_used if t in watch}
        add("step_tools_used", "info",
            f"processing tools fired: {sorted(used_step_tools) or 'none'}")
    # Context should carry all merged evidence — count vs. tool hits is
    # informational (rerank/elbow legitimately trims weak hits).
    if n_hits_total and ctx_docs:
        add("evidence_survives", "info",
            f"{n_hits_total} raw hits → {len(set(keys))} unique context docs")
    for s in turn.get("expect_status_in_context", set()):
        add(f"status_{s}_visible", s in statuses or any(
            (d.get("metadata") or {}).get("document_status") == s for d in last_ctx_docs),
            f"statuses={sorted(statuses)}")

    # ── finalize input ──
    fin = out["debug"].get("finalize_context") or out["debug"].get("finalize") or []
    if fin:
        add("finalize_context_debug", "info", f"{len(fin)} event(s)")

    cites = (out.get("rewrite") or {}).get("citations") or []
    if cites:
        add("citations_present", "info", f"{len(cites)} citation(s)")
    return checks


def report(turn, out, checks, verbose=False):
    mode = turn.get("mode", "?")
    print(f"\n  ── {turn['id']} [{mode}] {out['elapsed_s']}s ──")
    print(f"     Q: {turn['question'][:80]}")
    plan = out.get("plan") or {}
    plan_obj = plan.get("plan") if isinstance(plan.get("plan"), dict) else plan
    if plan_obj:
        print(f"     plan: intent={plan_obj.get('intent')!r} resolved={plan_obj.get('resolved_query')!r}")
        for c in plan_obj.get("tool_calls") or []:
            print(f"       → {c['tool']}({json.dumps(c.get('arguments') or {})[:100]})")
    for e in out["timeline"]:
        if e.get("type") == "tool_call":
            print(f"     call: {e.get('tool')} {json.dumps(e.get('arguments') or {})[:100]}")
    fp = (out["debug"].get("fast_plan") or [{}])[0]
    if fp.get("raw_output"):
        print(f"     fast_plan raw: {fp['raw_output'][:200]}")
    ctx_docs = []
    for ce in out["context"]:
        d = ce.get("docs") or ce.get("documents") or []
        if d:
            ctx_docs = d
    if ctx_docs:
        print(f"     context: {len(ctx_docs)} docs")
        for d in ctx_docs[:8]:
            meta = d.get("metadata") or {}
            print(f"       - doc={meta.get('document_id') or d.get('document_id')} "
                  f"chunk={meta.get('chunk_index')} status={meta.get('document_status')} "
                  f"ver={meta.get('version')} neighbor={bool(meta.get('_is_neighbor'))}")
        if len(ctx_docs) > 8:
            print(f"       … +{len(ctx_docs)-8} more")
    print(f"     answer ({len(out['answer'])} chars): {out['answer'][:300]}")
    for name, res, detail in checks:
        mark = "PASS" if res is True else ("INFO" if res == "info" else "FAIL" if res is False else "----")
        line = f"     [{mark}] {name}"
        if detail and (verbose or res is False or res == "info"):
            line += f"  — {detail}"
        print(line)


def inspect(out):
    """Deep stage-payload audit: sizes, duplication, trimming."""
    import re
    from collections import Counter
    print("\n  ══ stage-payload inspection ══")
    for d in out["debug"].get("fast_plan") or []:
        print(f"  fast_plan prompt: {len(d.get('prompt') or '')} chars")
        raw = d.get("raw_output") or ""
        print(f"  fast_plan raw output: {len(raw)} chars"
              + ("  [EVENT-TRUNCATED]" if len(raw) >= 3900 else ""))
        parsed = d.get("parsed") or {}
        print(f"  plan steps: {[s.get('tool') for s in (parsed.get('steps') or [])]}")
    for i, d in enumerate(out["debug"].get("fast_arg_compile") or [], 1):
        p = d.get("prompt") or ""
        print(f"  arg_compile[{i}] tool={d.get('tool')}: prompt {len(p)} chars"
              f", data_block={'Extracted data' in p}"
              + ("  [EVENT-TRUNCATED]" if len(p) >= 5900 else ""))
        print(f"    compiled args: {(d.get('raw_output') or '')[:220]}")
    # repeated tools: show full args to spot near-dupes
    from collections import Counter as _C
    tcalls = [e for e in out["timeline"] if e.get("type") == "tool_call"]
    repeated = {t for t, n in _C(e.get("tool") for e in tcalls).items() if n > 1}
    for e in tcalls:
        if e.get("tool") in repeated:
            print(f"  repeated call {e.get('tool')}: "
                  f"{json.dumps(e.get('arguments') or {}, default=str)[:400]}")
    for d in out["debug"].get("tool_observation") or []:
        res = json.dumps(d.get("result") or {}, default=str)
        print(f"  obs {d.get('tool')}: result={len(res)} chars err={d.get('error')}")
    for i, ce in enumerate(out["context"], 1):
        docs = ce.get("docs") or ce.get("documents") or []
        tot = sum(len(str(d.get("page_content", ""))) for d in docs if isinstance(d, dict))
        print(f"  context event {i}: {len(docs)} docs, ~{tot} content chars")
    for d in out["debug"].get("finalize_context") or []:
        ct = d.get("context_text") or ""
        headers = re.findall(r'document="([^"]+)"', ct)
        dupes = {k: v for k, v in Counter(headers).items() if v > 1}
        print(f"  finalize_context: {len(ct)} chars, doc_count={d.get('doc_count')}, "
              f"{len(headers)} evidence items"
              + (f", repeated titles: {dupes}" if dupes else ", no repeats"))
    print(f"  answer: {len(out['answer'])} chars")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000/api")
    ap.add_argument("--user", default="super_admin")
    ap.add_argument("--password", default="super_admin123")
    ap.add_argument("--kb", type=int, default=3, help="existing KB id (no ingestion)")
    ap.add_argument("--chat-id", type=int, default=None, help="reuse an existing chat")
    ap.add_argument("--turns", default=None,
                    help="comma-separated modes to override scripted modes, e.g. fast,fast,fast")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--inspect", type=str, default=None,
                    help="turn id to deep-inspect, e.g. T8_chain_fast")
    args = ap.parse_args()

    turns = KB_SCRIPTS.get(args.kb)
    if turns is None:
        print(f"no scripted turns for KB {args.kb}; known: {sorted(KB_SCRIPTS)}")
        sys.exit(2)
    if args.turns:
        modes = [m.strip() for m in args.turns.split(",")]
        for t, m in zip(turns, modes):
            t["mode"] = m

    client = Client(args.base)
    print(f"=== fast-pipeline monitor ===\ntarget: {args.base} | kb: {args.kb}")
    client.login(args.user, args.password)
    print("login ok")

    chat_id = args.chat_id or client.create_chat(f"fastpipe-eval-kb{args.kb}", [args.kb])
    print(f"chat {chat_id}\n")

    hard_fail = 0
    for turn in turns:
        out = run_turn(client, chat_id, turn)
        checks = check_turn(turn, out)
        report(turn, out, checks, verbose=args.verbose)
        if args.inspect == turn["id"]:
            inspect(out)
        if any(res is False for _, res, _ in checks):
            hard_fail += 1

    print(f"\n{'=' * 70}\nturns with failures: {hard_fail}\n{'=' * 70}")
    sys.exit(1 if hard_fail else 0)


if __name__ == "__main__":
    main()
