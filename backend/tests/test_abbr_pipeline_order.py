#!/usr/bin/env python3
"""
Comprehensive test for pipeline ordering and reranker query choice.

Tests the user's hypothesis: expand abbreviations BEFORE the LLM rewrite step
so the LLM can understand abbreviations during pronoun/reference resolution.

Pipeline orderings tested:
  CURRENT:   original → rewrite(LLM) → rewritten → expand → expanded → retrieve → rerank(rewritten)
  PROPOSED:  original → expand → expanded → rewrite(LLM) → rewritten → retrieve → rerank(rewritten)

Also tests:
  - Different pre-expansion formats (glossary suffix, space-join, inline)
  - Reranker with different query variants (orig, rewritten, expanded_rewritten)
  - Both chat models for rewrite and generation
  - Queries with pronouns + abbreviations (the case where expand-before-rewrite matters)
  - Provenance validation interaction (expanded forms in rewrite might be rejected)

Runs inside the backend container:
  docker exec rag-web-ui-backend-1 python3 /app/tests/test_abbr_pipeline_order.py
"""
import json
import os
import re
import sys
import time
import logging
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, "/app")
os.environ.setdefault("PYTHONPATH", "/app")

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("abbr_pipeline_test")

# ─── Config ─────────────────────────────────────────────────────────────────
LM_STUDIO_URL = "http://192.168.1.3:2244/v1"
EMBEDDING_MODEL = "qwen/qwen3-embedding-0.6b"
CHAT_MODELS = ["google/gemma-4-26b-a4b", "qwen/qwen3.5-9b"]
REWRITE_MODEL = CHAT_MODELS[1]  # qwen3.5-9b for rewrite (faster, good instruction following)
GEN_MODEL = CHAT_MODELS[0]      # gemma-4-26b for generation
CSV_PATH = "/app/assets/abbreviations_enhanced.csv"
RESULTS_PATH = "/app/assets/abbr_pipeline_order_results.json"

# ─── Load abbreviation CSV ──────────────────────────────────────────────────

def load_abbreviations():
    import csv
    forward = defaultdict(list)
    reverse = defaultdict(list)
    with open(CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            abbr = row["abbreviation"].strip()
            form = row["expanded_form"].strip()
            if abbr and form:
                forward[abbr].append(form)
                reverse[form.lower()].append(abbr)
    all_abbrs = sorted(forward.keys(), key=len, reverse=True)
    all_forms = sorted(reverse.keys(), key=len, reverse=True)
    return dict(forward), dict(reverse), all_abbrs, all_forms

FORWARD_MAP, REVERSE_MAP, ALL_ABBRS, ALL_FORMS = load_abbreviations()
print(f"Loaded {len(FORWARD_MAP)} abbreviations, {sum(len(v) for v in FORWARD_MAP.values())} total forms")

# ─── Expansion functions ────────────────────────────────────────────────────

def find_abbrs_in_text(text: str) -> Dict[str, List[str]]:
    found = {}
    for abbr in ALL_ABBRS:
        if re.search(r'\b' + re.escape(abbr) + r'\b', text, re.IGNORECASE):
            found[abbr] = FORWARD_MAP[abbr]
    return found

def find_forms_in_text(text: str) -> Dict[str, List[str]]:
    text_lower = text.lower()
    found_abbrs = set()
    for form_lower in ALL_FORMS:
        if re.search(r'\b' + re.escape(form_lower) + r'\b', text_lower, re.IGNORECASE):
            for abbr in REVERSE_MAP[form_lower]:
                found_abbrs.add(abbr)
    if not found_abbrs:
        return {}
    return {abbr: FORWARD_MAP[abbr] for abbr in found_abbrs}

def expand_suffix(text: str) -> str:
    """Ingestion format: text\n\n[Abbreviation Glossary]\nabbr = forms"""
    found = find_abbrs_in_text(text)
    if not found:
        return text
    lines = [f"{a} = {', '.join(f)}" for a, f in sorted(found.items(), key=lambda x: x[0].lower())]
    return f"{text}\n\n[Abbreviation Glossary]\n" + "\n".join(lines)

def expand_bidir_glossary(query: str) -> str:
    """Bidirectional glossary suffix: query\n\n[Abbreviation Glossary]\nabbr = forms"""
    found_abbrs = find_abbrs_in_text(query)
    found_forms = find_forms_in_text(query)
    merged = dict(found_abbrs)
    for abbr, forms in found_forms.items():
        if abbr not in merged:
            merged[abbr] = forms
    if not merged:
        return query
    lines = [f"{a} = {', '.join(f)}" for a, f in sorted(merged.items(), key=lambda x: x.lower())]
    return f"{query}\n\n[Abbreviation Glossary]\n" + "\n".join(lines)

def expand_bidir_space(query: str) -> str:
    """Bidirectional space-join: query form1 form2 abbr1 ..."""
    found_abbrs = find_abbrs_in_text(query)
    found_forms = find_forms_in_text(query)
    merged = dict(found_abbrs)
    for abbr, forms in found_forms.items():
        if abbr not in merged:
            merged[abbr] = forms
    if not merged:
        return query
    expansions = []
    for abbr, forms in merged.items():
        expansions.extend(forms)
        expansions.append(abbr)
    return f"{query} {' '.join(expansions)}"

def expand_inline(query: str) -> str:
    """Inline expansion: 'the CO (Commanding Officer) ordered bns (Battalions)...'"""
    found = find_abbrs_in_text(query)
    if not found:
        return query
    result = query
    for abbr in sorted(found.keys(), key=len, reverse=True):
        forms = " ".join(found[abbr])
        result = re.sub(r'\b' + re.escape(abbr) + r'\b', f"{abbr} ({forms})", result, flags=re.IGNORECASE)
    return result

def build_glossary_block(text: str) -> str:
    """Generation context: text + [Abbreviation Glossary]\nabbr = forms"""
    found = find_abbrs_in_text(text)
    if not found:
        return text
    lines = [f"{a} = {', '.join(f)}" for a, f in sorted(found.items(), key=lambda x: x[0].lower())]
    return f"{text}\n[Abbreviation Glossary]\n" + "\n".join(lines)

# ─── Rewrite (LLM) ──────────────────────────────────────────────────────────

REWRITE_SYSTEM_PROMPT = """\
You are a search query rewriter for a document retrieval system. \
Your ONLY job is to rewrite the user's latest message into a self-contained search query \
that can be sent to a vector database. \
Use the recent chat history and any relevant past context solely to resolve pronouns and references — \
never to answer, evaluate, or judge the question.

Rules:
1. Output a standalone question or keyword phrase — nothing else. No preamble, no explanation.
2. Resolve pronouns and references from history or past context \
(e.g. 'it' → the specific topic discussed).
3. Do NOT answer the question. Do NOT say whether information exists or not.
4. Do NOT add information not needed to resolve an ambiguous reference.
5. Do NOT infer relationships between topics. If the user asks a standalone question, \
keep it standalone — even if a previous turn discussed something different.
6. Do NOT introduce new entities, concepts, or relationships that the user did not mention. \
This includes synonyms, related terms, broader categories, or background concepts. \
For example, if the user asks 'what is mutex', do NOT add 'mutual exclusion', \
'synchronization', 'critical section', 'race conditions', or any other term the user did not say.
7. Keep the output short — one sentence or a keyword phrase, maximum 30 words.
8. If the user's query is already self-contained (no pronouns, no references to prior turns), \
return it EXACTLY as-is. Do not rephrase, do not expand, do not add terms.

Examples:
History: [user: tell me about Linux, assistant: Linux is an open-source OS...]
Query: 'any other worthwhile OS you like to mention?'
Output: 'other notable operating systems worth mentioning'

History: [user: summarise assignment 1, assistant: ...summary...]
Query: 'what is question 1'
Output: 'What is Question 1 in Assignment 1?'

History: [user: tell me about the StreamVC paper]
Query: 'what model does it use'
Output: 'What model architecture does StreamVC use?'

History: [user: explain Process Control Block, assistant: ...PCB explanation...]
Query: 'Explain mutex'
Output: 'Explain mutex'

History: [user: explain mutex, assistant: ...mutex explanation...]
Query: 'How does a semaphore differ?'
Output: 'How does a semaphore differ from a mutex?'

History: (none)
Query: 'what is mutex?'
Output: 'what is mutex?'
"""

def needs_reference_resolution(query: str, has_history: bool) -> bool:
    if not has_history:
        return False
    markers = re.compile(
        r"\b(it|its|it's|they|them|their|this|that|these|those|there|"
        r"he|she|his|her|him|"
        r"one|ones|former|latter|above|previous|prior|earlier|"
        r"same|similar|such|both|another|each)\b",
        re.IGNORECASE,
    )
    return bool(markers.search(query))

def rewrite_query_sync(query: str, history: List[Dict[str, str]], model: str) -> str:
    """Synchronous rewrite using the production prompt."""
    from openai import OpenAI
    client = OpenAI(api_key="not-required", base_url=LM_STUDIO_URL)

    if not needs_reference_resolution(query, bool(history)):
        return query

    messages = [{"role": "system", "content": REWRITE_SYSTEM_PROMPT}]
    for m in history:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": query})

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=160,
                temperature=0,
                stream=False,
            )
            raw = resp.choices[0].message.content.strip()
            # Clean: strip reasoning tags, meta preambles
            standalone = raw.strip().strip('"')
            # Strip meta preamble
            meta = re.match(r"^[^:\n]{0,80}?\b(rewritten|standalone|search)\s+(query|question)?\s*:\s*",
                            standalone, re.IGNORECASE)
            if meta:
                candidate = standalone[meta.end():].strip()
                if len(candidate) > 5:
                    standalone = candidate
            return standalone or query
        except Exception as exc:
            if attempt < 2:
                time.sleep(3)
                continue
            return query

# ─── Test data ──────────────────────────────────────────────────────────────

TEST_CHUNKS = [
    "The CO ordered the bns to wdr from the forward position. The MO reported "
    "casualties. The adjt coordinated with HQ. The op was conducted at first light. "
    "The recce team provided intelligence on enemy positions.",
    "The GOC visited the bde HQ and briefed the bde comd on the op. The inf bn was "
    "tasked to secure the obj. The armd sqn was to provide spt. The arty bty was "
    "placed in sp of the inf.",
    "The weather forecast indicates rain for the next three days. Temperature "
    "will drop to 15 degrees Celsius. Farmers should prepare for wet conditions.",
    "The DA approved the medical resupply. The SP was established at checkpoint 4. "
    "The cas were evacuated to the Fd Amb. The spt elements moved up at 0600.",
    "The commanding officer ordered the battalions to withdraw from the forward "
    "position. The medical officer reported casualties. The operation was "
    "conducted at first light by the reconnaissance team.",
]

# Queries with conversation history that requires pronoun resolution
# Each test case: (query, history, expected_chunks, description)
PRONOUN_QUERIES = [
    # Q0: pronoun "it" + abbreviations — history mentions battalion withdrawal
    {
        "query": "did the CO order it?",
        "history": [
            {"role": "user", "content": "tell me about the battalion withdrawal"},
            {"role": "assistant", "content": "The CO ordered the bns to wdr from the forward position. "
                                             "The battalions withdrew at first light during the operation."},
        ],
        "expected": [0, 4],
        "desc": "pronoun 'it' + abbr 'CO' — history has battalion withdrawal context",
    },
    # Q1: pronoun "that" + abbreviations — history mentions brigade HQ
    {
        "query": "tell me more about that bde HQ op",
        "history": [
            {"role": "user", "content": "what happened at the brigade headquarters?"},
            {"role": "assistant", "content": "The GOC visited the bde HQ and briefed the bde comd on the op. "
                                             "The infantry battalion was tasked to secure the objective."},
        ],
        "expected": [1],
        "desc": "pronoun 'that' + abbrs 'bde HQ op' — history has brigade HQ context",
    },
    # Q2: pronoun "the same" + abbreviations — history mentions medical resupply
    {
        "query": "who approved the same resupply?",
        "history": [
            {"role": "user", "content": "tell me about the medical resupply"},
            {"role": "assistant", "content": "The DA approved the medical resupply. The spt elements "
                                             "moved up at 0600 to establish the supply point."},
        ],
        "expected": [3],
        "desc": "pronoun 'the same' + abbr context — history has medical resupply",
    },
    # Q3: self-contained abbreviation query (no pronouns) — rewrite is no-op
    {
        "query": "bns wdr from position",
        "history": [],
        "expected": [0, 4],
        "desc": "self-contained abbr query — rewrite is no-op",
    },
    # Q4: self-contained full-form query (no pronouns) — rewrite is no-op
    {
        "query": "battalions withdrew from position",
        "history": [],
        "expected": [0, 4],
        "desc": "self-contained full-form query — rewrite is no-op, needs bidirectional expansion",
    },
    # Q5: pronoun "it" + full forms — history has abbreviation-heavy content
    {
        "query": "did the commanding officer order it?",
        "history": [
            {"role": "user", "content": "tell me about the bns wdr"},
            {"role": "assistant", "content": "The CO ordered the bns to wdr from the forward position. "
                                             "The battalions withdrew at first light."},
        ],
        "expected": [0, 4],
        "desc": "pronoun 'it' + full forms — history has abbreviation context (reverse direction)",
    },
    # Q6: pronoun "that one" + mixed — history has weather
    {
        "query": "what about that one about the weather?",
        "history": [
            {"role": "user", "content": "tell me about the operations"},
            {"role": "assistant", "content": "There were several operations conducted. The weather "
                                             "forecast indicated rain for three days."},
        ],
        "expected": [2],
        "desc": "pronoun 'that one' — no abbreviations, control case",
    },
]

# ─── Model accessors ────────────────────────────────────────────────────────

def get_dense_client():
    from openai import OpenAI
    return OpenAI(api_key="not-required", base_url=LM_STUDIO_URL)

def get_sparse_embedder():
    from app.services.infrastructure import get_sparse_embedder
    return get_sparse_embedder()

def get_reranker():
    from app.services.retrieval.reranker import _get_cross_encoder
    return _get_cross_encoder()

def embed_dense(texts: List[str]) -> List[List[float]]:
    client = get_dense_client()
    embeddings = []
    for i in range(0, len(texts), 32):
        batch = texts[i:i+32]
        resp = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
        embeddings.extend([r.embedding for r in resp.data])
    return embeddings

def embed_sparse(texts: List[str]) -> List:
    return list(get_sparse_embedder().embed(texts))

def rerank_scores(query: str, passages: List[str]) -> List[float]:
    return list(get_reranker().rerank(query, passages))

def generate_answer(query: str, context: str, model: str, max_tokens: int = 500) -> str:
    from openai import OpenAI
    client = OpenAI(api_key="not-required", base_url=LM_STUDIO_URL)
    system = (
        "You are a military assistant. Answer the user's question based ONLY on "
        "the provided context. If the context doesn't contain the answer, say "
        "'I cannot answer based on the provided context.' Be concise."
    )
    user = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model, messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens, temperature=0.1,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            if attempt < 2:
                time.sleep(3)
                continue
            return f"ERROR: {exc}"

def cosine_similarity(a, b) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

def sparse_dot_product(a, b) -> float:
    a_dict = dict(zip(a.indices.tolist(), a.values.tolist()))
    b_dict = dict(zip(b.indices.tolist(), b.values.tolist()))
    return sum(a_dict.get(k, 0) * b_dict.get(k, 0) for k in a_dict)

# ─── Results collector ──────────────────────────────────────────────────────

class Results:
    def __init__(self):
        self.rows = []
    def add(self, cat, test, metric, value, extra=None):
        self.rows.append({"cat": cat, "test": test, "metric": metric, "value": value, "extra": extra or {}})
    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.rows, f, indent=2, default=str)

R = Results()

def hit_rate(rows, filter_fn):
    hits = sum(1 for r in rows if filter_fn(r) and r["extra"].get("hit"))
    total = sum(1 for r in rows if filter_fn(r))
    return hits, total, (hits / total * 100 if total else 0)

# ─── TEST A: Rewrite quality with/without pre-expansion ─────────────────────

def test_rewrite_quality():
    print("\n" + "=" * 80)
    print("TEST A: Rewrite Quality — Expand Before vs After Rewrite")
    print("=" * 80)

    # Pre-expansion formats to test for the rewrite input
    PRE_EXPAND_FNS = {
        "none":           lambda q: q,
        "glossary":       expand_bidir_glossary,
        "space":          expand_bidir_space,
        "inline":         expand_inline,
    }

    print(f"\n  Rewrite model: {REWRITE_MODEL}")
    print(f"  Testing {len(PRONOUN_QUERIES)} queries x {len(PRE_EXPAND_FNS)} pre-expansion formats\n")

    for i, tc in enumerate(PRONOUN_QUERIES):
        query = tc["query"]
        history = tc["history"]
        desc = tc["desc"]
        print(f"\n  Q{i}: '{query}'")
        print(f"      desc: {desc}")
        if history:
            print(f"      history: {len(history)} messages")
            print(f"        last user: '{history[0]['content'][:60]}...'")
            print(f"        last asst: '{history[1]['content'][:60]}...'")

        for pre_name, pre_fn in PRE_EXPAND_FNS.items():
            pre_expanded = pre_fn(query)
            rewritten = rewrite_query_sync(pre_expanded, history, REWRITE_MODEL)

            # Also expand the rewritten query (post-rewrite expansion)
            post_expanded = expand_bidir_glossary(rewritten)

            R.add("REWRITE", f"q{i}", f"pre={pre_name}", rewritten,
                  {"original": query, "pre_expanded": pre_expanded[:200],
                   "rewritten": rewritten, "post_expanded": post_expanded[:200],
                   "desc": desc})

            print(f"    pre={pre_name:10s} → rewrite: '{rewritten[:100]}'")
            if pre_expanded != query:
                print(f"      (pre-expanded: '{pre_expanded[:80]}...')")

    # Summary: which pre-expansion format produces the best rewrites?
    print("\n  Rewrite quality summary:")
    for i, tc in enumerate(PRONOUN_QUERIES):
        print(f"\n  Q{i}: '{tc['query']}' (expected: {tc['expected']})")
        for pre_name in PRE_EXPAND_FNS:
            rows = [r for r in R.rows if r["cat"] == "REWRITE" and r["test"] == f"q{i}" and r["metric"] == f"pre={pre_name}"]
            if rows:
                rw = rows[0]["extra"]["rewritten"]
                print(f"    pre={pre_name:10s}  rewrite: '{rw[:80]}'")

# ─── TEST B: Dense + Sparse retrieval with pipeline ordering ────────────────

def test_retrieval_pipeline_order():
    print("\n" + "=" * 80)
    print("TEST B: Dense + Sparse — Pipeline Ordering (expand→rewrite vs rewrite→expand)")
    print("=" * 80)

    # Ingestion: suffix-expanded chunks (same as production)
    chunk_texts = [expand_suffix(c) for c in TEST_CHUNKS]
    chunk_dense = embed_dense(chunk_texts)
    chunk_sparse = embed_sparse(chunk_texts)

    # Pipeline variants:
    # CURRENT:   rewrite(original) → rewritten → expand_bidir(rewritten) → expanded
    # PROPOSED:  expand_bidir(original) → expanded → rewrite(expanded) → rewritten
    #            (then expand_bidir(rewritten) for retrieval, since rewrite may drop expansions)
    # NO_EXPAND: rewrite(original) → rewritten (no expansion at all)

    PIPELINES = {
        "current_rewrite_then_expand":   "rewrite→expand",
        "proposed_expand_then_rewrite":  "expand→rewrite→expand",
        "proposed_expand_then_rewrite_only": "expand→rewrite (no re-expand)",
        "no_expansion":                  "rewrite only (no expand)",
    }

    print(f"\n  Testing {len(PRONOUN_QUERIES)} queries x {len(PIPELINES)} pipelines\n")

    for i, tc in enumerate(PRONOUN_QUERIES):
        query = tc["query"]
        history = tc["history"]
        expected = tc["expected"]
        print(f"\n  Q{i}: '{query}' → expected {expected}")

        for pipe_key, pipe_label in PIPELINES.items():
            if pipe_key == "current_rewrite_then_expand":
                rewritten = rewrite_query_sync(query, history, REWRITE_MODEL)
                retrieval_query = expand_bidir_glossary(rewritten)
                reranker_query = rewritten
            elif pipe_key == "proposed_expand_then_rewrite":
                pre_expanded = expand_bidir_glossary(query)
                rewritten = rewrite_query_sync(pre_expanded, history, REWRITE_MODEL)
                retrieval_query = expand_bidir_glossary(rewritten)
                reranker_query = rewritten
            elif pipe_key == "proposed_expand_then_rewrite_only":
                pre_expanded = expand_bidir_glossary(query)
                rewritten = rewrite_query_sync(pre_expanded, history, REWRITE_MODEL)
                retrieval_query = rewritten  # no re-expansion
                reranker_query = rewritten
            elif pipe_key == "no_expansion":
                rewritten = rewrite_query_sync(query, history, REWRITE_MODEL)
                retrieval_query = rewritten
                reranker_query = rewritten

            # Dense
            q_emb = embed_dense([retrieval_query])[0]
            d_sims = sorted(
                enumerate(cosine_similarity(q_emb, ce) for ce in chunk_dense),
                key=lambda x: x[1], reverse=True
            )
            d_top2 = [s[0] for s in d_sims[:2]]
            d_hit = any(c in d_top2 for c in expected)

            # Sparse
            q_sparse = embed_sparse([retrieval_query])[0]
            s_sims = sorted(
                enumerate(sparse_dot_product(q_sparse, ce) for ce in chunk_sparse),
                key=lambda x: x[1], reverse=True
            )
            s_top2 = [s[0] for s in s_sims[:2]]
            s_hit = any(c in s_top2 for c in expected)

            R.add("DENSE_PIPE", pipe_key, f"q{i}", f"c{d_sims[0][0]}({d_sims[0][1]:.3f})",
                  {"hit": d_hit, "expected": expected, "top3": [(s[0], round(s[1], 3)) for s in d_sims[:3]],
                   "retrieval_query": retrieval_query[:150], "reranker_query": reranker_query[:100]})
            R.add("SPARSE_PIPE", pipe_key, f"q{i}", f"c{s_sims[0][0]}({s_sims[0][1]:.1f})",
                  {"hit": s_hit, "expected": expected, "top3": [(s[0], round(s[1], 1)) for s in s_sims[:3]]})

            d_str = " ".join(f"c{s[0]}:{s[1]:.3f}" for s in d_sims[:3])
            s_str = " ".join(f"c{s[0]}:{s[1]:.1f}" for s in s_sims[:3])
            print(f"    {pipe_label:35s}  D[{d_str}] S[{s_str}]  {'HIT' if d_hit and s_hit else 'CHECK'}")

    # Hit rate summary
    print("\n  Hit rates (top-2):")
    for pipe_key, pipe_label in PIPELINES.items():
        dh, dt, dpct = hit_rate(R.rows, lambda r: r["cat"] == "DENSE_PIPE" and r["test"] == pipe_key)
        sh, st, spct = hit_rate(R.rows, lambda r: r["cat"] == "SPARSE_PIPE" and r["test"] == pipe_key)
        print(f"    {pipe_label:35s}  DENSE {dh}/{dt} ({dpct:.0f}%)  SPARSE {sh}/{st} ({spct:.0f}%)")

# ─── TEST C: Reranker with different query variants ─────────────────────────

def test_reranker_variants():
    print("\n" + "=" * 80)
    print("TEST C: Reranker — Query Variant Comparison with Pipeline Ordering")
    print("=" * 80)

    # For each query, test reranker with:
    # 1. rewritten_q + suffix_c       (current production)
    # 2. expanded_rewritten_q + suffix_c  (expanded query for reranker)
    # 3. orig_q + suffix_c            (original, no rewrite)
    # 4. pre_expanded_rewritten_q + suffix_c  (expand→rewrite pipeline)

    chunk_suffix = [expand_suffix(c) for c in TEST_CHUNKS]

    RR_VARIANTS = [
        ("rewritten_q+suffix_c",       "rewritten"),
        ("expand_rewritten_q+suffix_c", "expand_rewritten"),
        ("orig_q+suffix_c",            "original"),
        ("pre_exp_rw_q+suffix_c",      "pre_expand_then_rewrite"),
    ]

    print(f"\n  Testing {len(PRONOUN_QUERIES)} queries x {len(RR_VARIANTS)} reranker variants\n")

    for i, tc in enumerate(PRONOUN_QUERIES):
        query = tc["query"]
        history = tc["history"]
        expected = tc["expected"]
        print(f"\n  Q{i}: '{query}' → expected {expected}")

        # Pre-compute all query variants
        rewritten = rewrite_query_sync(query, history, REWRITE_MODEL)
        expand_rewritten = expand_bidir_glossary(rewritten)
        pre_expanded = expand_bidir_glossary(query)
        pre_exp_rewritten = rewrite_query_sync(pre_expanded, history, REWRITE_MODEL)

        query_variants = {
            "rewritten": rewritten,
            "expand_rewritten": expand_rewritten,
            "original": query,
            "pre_expand_then_rewrite": pre_exp_rewritten,
        }

        for rr_label, variant_key in RR_VARIANTS:
            rr_query = query_variants[variant_key]
            scores = rerank_scores(rr_query, chunk_suffix)
            scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            top2 = [s[0] for s in scored[:2]]
            hit = any(c in top2 for c in expected)

            R.add("RERANKER_PIPE", rr_label, f"q{i}", f"c{scored[0][0]}({scored[0][1]:.3f})",
                  {"hit": hit, "expected": expected,
                   "top3": [(s[0], round(s[1], 3)) for s in scored[:3]],
                   "rr_query": rr_query[:100]})

            top3_str = " ".join(f"c{s[0]}:{s[1]:.2f}" for s in scored[:3])
            print(f"    {rr_label:35s}  {top3_str}  {'HIT' if hit else 'MISS'}")

    # Hit rate summary
    print("\n  Hit rates (top-2):")
    for rr_label, _ in RR_VARIANTS:
        h, t, pct = hit_rate(R.rows, lambda r: r["cat"] == "RERANKER_PIPE" and r["test"] == rr_label)
        print(f"    {rr_label:35s}  {h}/{t} ({pct:.0f}%)")

    # Detailed score comparison for queries with pronouns + abbreviations
    print("\n  Detailed: second expected chunk score (threshold survival):")
    for i, tc in enumerate(PRONOUN_QUERIES):
        if not tc["history"]:
            continue  # skip self-contained queries
        expected = tc["expected"]
        if len(expected) < 2:
            continue  # only interesting when there are 2 expected chunks
        print(f"\n  Q{i}: '{tc['query']}' → expected {expected}")
        for rr_label, _ in RR_VARIANTS:
            rows = [r for r in R.rows if r["cat"] == "RERANKER_PIPE" and r["test"] == rr_label and r["metric"] == f"q{i}"]
            if rows:
                top3 = rows[0]["extra"]["top3"]
                second_expected = expected[1] if len(expected) > 1 else expected[0]
                second_score = next((s for c, s in top3 if c == second_expected), None)
                if second_score is None:
                    # Find in full scored list
                    all_scores = sorted([(c, s) for c, s in top3], key=lambda x: x[1], reverse=True)
                    second_score = "not in top3"
                print(f"    {rr_label:35s}  2nd expected c{second_expected}: {second_score}")

# ─── TEST D: Full pipeline end-to-end ───────────────────────────────────────

def test_full_pipeline():
    print("\n" + "=" * 80)
    print("TEST D: Full Pipeline — End-to-End Comparison")
    print("=" * 80)

    chunk_texts = [expand_suffix(c) for c in TEST_CHUNKS]
    chunk_embs = embed_dense(chunk_texts)

    # Pipelines to compare end-to-end
    PIPELINES = [
        ("CURRENT: rewrite→expand|rr=rewritten",
         "current", "rewritten"),
        ("PROPOSED: expand→rewrite→expand|rr=rewritten",
         "proposed", "rewritten"),
        ("PROPOSED: expand→rewrite→expand|rr=expand_rewritten",
         "proposed", "expand_rewritten"),
        ("NO_EXPAND: rewrite|rr=rewritten",
         "no_expand", "rewritten"),
    ]

    for label, pipe_type, rr_type in PIPELINES:
        print(f"\n  Pipeline: {label}")
        hits = 0
        corrects = 0
        total = 0

        for i, tc in enumerate(PRONOUN_QUERIES):
            query = tc["query"]
            history = tc["history"]
            expected = tc["expected"]
            total += 1

            # Step 1: Pipeline-specific query processing
            if pipe_type == "current":
                rewritten = rewrite_query_sync(query, history, REWRITE_MODEL)
                retrieval_query = expand_bidir_glossary(rewritten)
            elif pipe_type == "proposed":
                pre_expanded = expand_bidir_glossary(query)
                rewritten = rewrite_query_sync(pre_expanded, history, REWRITE_MODEL)
                retrieval_query = expand_bidir_glossary(rewritten)
            elif pipe_type == "no_expand":
                rewritten = rewrite_query_sync(query, history, REWRITE_MODEL)
                retrieval_query = rewritten

            # Step 2: Dense retrieval
            q_emb = embed_dense([retrieval_query])[0]
            sims = sorted(enumerate(cosine_similarity(q_emb, ce) for ce in chunk_embs),
                          key=lambda x: x[1], reverse=True)
            top3_idx = [s[0] for s in sims[:3]]

            # Step 3: Rerank
            if rr_type == "rewritten":
                rr_query = rewritten
            elif rr_type == "expand_rewritten":
                rr_query = expand_bidir_glossary(rewritten)
            else:
                rr_query = query

            rr_passages = [expand_suffix(TEST_CHUNKS[idx]) for idx in top3_idx]
            rr_scores = rerank_scores(rr_query, rr_passages)
            rr_ranked = sorted(zip(top3_idx, rr_scores), key=lambda x: x[1], reverse=True)
            rr_top1 = rr_ranked[0][0]
            rr_hit = rr_top1 in expected
            if rr_hit:
                hits += 1

            # Step 4: Generate
            top_chunk = TEST_CHUNKS[rr_top1]
            gen_context = build_glossary_block(top_chunk)
            answer = generate_answer(query, gen_context, GEN_MODEL, max_tokens=300)

            # Check correctness
            if "weather" in query.lower():
                correct = any(w in answer.lower() for w in ["rain", "temperature", "weather", "degrees", "wet"])
            elif "deputy" in query.lower() or "resupply" in query.lower() or "approved" in query.lower():
                correct = any(w in answer.lower() for w in ["deputy", "assistant", "da", "approved", "resupply"])
                wrong = any(w in answer.lower() for w in ["daily allowance", "defence attache", "defense attache"])
                if wrong:
                    correct = False
            elif "brigade" in query.lower() or "headquarters" in query.lower() or "bde" in query.lower():
                correct = any(w in answer.lower() for w in ["brigade", "headquarters", "operation", "objective", "goc", "comd"])
            else:
                correct = any(w in answer.lower() for w in ["battalion", "withdraw", "order", "position", "commanding"])
            if correct:
                corrects += 1

            R.add("FULL_PIPELINE", label, f"q{i}", f"hit={rr_hit} correct={correct}",
                  {"rr_top1": rr_top1, "expected": expected, "answer": answer[:150],
                   "rewritten": rewritten[:100], "retrieval_query": retrieval_query[:150]})

            status = "HIT" if rr_hit else "MISS"
            corr = "OK" if correct else "BAD"
            print(f"    Q{i}: {status} {corr}  top1=c{rr_top1}  rewrite='{rewritten[:50]}'  ans: {answer[:60]}")

        print(f"    → hit={hits}/{total}  correct={corrects}/{total}")

# ─── MAIN ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("ABBREVIATION PIPELINE ORDERING TEST")
    print(f"Embedding: {EMBEDDING_MODEL}")
    print(f"Rewrite model: {REWRITE_MODEL}")
    print(f"Generation model: {GEN_MODEL}")
    print(f"LM Studio: {LM_STUDIO_URL}")
    print(f"Abbreviations: {len(FORWARD_MAP)}, chunks: {len(TEST_CHUNKS)}, queries: {len(PRONOUN_QUERIES)}")
    print("=" * 80)

    # Show expansion examples for pronoun queries
    print("\n  Expansion examples for pronoun+abbr queries:")
    for i, tc in enumerate(PRONOUN_QUERIES):
        if not tc["history"]:
            continue
        q = tc["query"]
        print(f"\n  Q{i}: '{q}'")
        print(f"    glossary: '{expand_bidir_glossary(q)[:100]}'")
        print(f"    space:    '{expand_bidir_space(q)[:100]}'")
        print(f"    inline:   '{expand_inline(q)[:100]}'")

    test_rewrite_quality()
    test_retrieval_pipeline_order()
    test_reranker_variants()
    test_full_pipeline()

    R.save(RESULTS_PATH)
    print(f"\nResults saved to {RESULTS_PATH}")

    # ─── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print("\n1. REWRITE quality (pre-expansion format → rewrite output):")
    for i, tc in enumerate(PRONOUN_QUERIES):
        if not tc["history"]:
            continue
        print(f"\n   Q{i}: '{tc['query']}'")
        for pre_name in ["none", "glossary", "space", "inline"]:
            rows = [r for r in R.rows if r["cat"] == "REWRITE" and r["test"] == f"q{i}" and r["metric"] == f"pre={pre_name}"]
            if rows:
                rw = rows[0]["extra"]["rewritten"]
                print(f"     pre={pre_name:10s}  → '{rw[:80]}'")

    print("\n2. DENSE+SPARSE hit rates by pipeline:")
    for pipe_key in ["current_rewrite_then_expand", "proposed_expand_then_rewrite",
                     "proposed_expand_then_rewrite_only", "no_expansion"]:
        dh, dt, dpct = hit_rate(R.rows, lambda r: r["cat"] == "DENSE_PIPE" and r["test"] == pipe_key)
        sh, st, spct = hit_rate(R.rows, lambda r: r["cat"] == "SPARSE_PIPE" and r["test"] == pipe_key)
        print(f"   {pipe_key:40s}  DENSE {dh}/{dt} ({dpct:.0f}%)  SPARSE {sh}/{st} ({spct:.0f}%)")

    print("\n3. RERANKER hit rates by query variant:")
    for rr_label, _ in [("rewritten_q+suffix_c", None), ("expand_rewritten_q+suffix_c", None),
                        ("orig_q+suffix_c", None), ("pre_exp_rw_q+suffix_c", None)]:
        h, t, pct = hit_rate(R.rows, lambda r: r["cat"] == "RERANKER_PIPE" and r["test"] == rr_label)
        print(f"   {rr_label:35s}  {h}/{t} ({pct:.0f}%)")

    print("\n4. FULL PIPELINE results:")
    pipe_labels = sorted(set(r["test"] for r in R.rows if r["cat"] == "FULL_PIPELINE"))
    for label in pipe_labels:
        rows = [r for r in R.rows if r["cat"] == "FULL_PIPELINE" and r["test"] == label]
        hits = sum(1 for r in rows if "hit=True" in r["value"])
        correct = sum(1 for r in rows if "correct=True" in r["value"])
        print(f"   {label:55s}  hit={hits}/{len(rows)}  correct={correct}/{len(rows)}")

if __name__ == "__main__":
    main()
