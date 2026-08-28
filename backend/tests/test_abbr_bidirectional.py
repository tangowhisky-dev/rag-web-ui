#!/usr/bin/env python3
"""
Test proposed abbreviation expansion changes BEFORE modifying the codebase.

Proposed changes:
  1. QUERY EXPANSION: Use glossary suffix format (like ingestion) instead of
     space-joined forms. Make it bidirectional (abbr→forms AND form→abbr).
     "bns wdr from position" → "bns wdr from position\n\n[Abbreviation Glossary]\nbns = Battalions\nwdr = Withdraw..."
     "battalions withdrew"    → "battalions withdrew\n\n[Abbreviation Glossary]\nbns = Battalions\nwdr = Withdraw..."

  2. RERANKER: Use original_query (not rewritten_query) + suffix-expanded chunks.

Tests:
  A. Dense retrieval: current vs proposed query formats
  B. Sparse retrieval: current vs proposed query formats
  C. Reranker: orig_q+suffix_c vs rewritten_q+suffix_c vs orig_q+orig_c
  D. Generation: both chat models (gemma-4-26b-a4b, qwen3.5-9b)
  E. Full pipeline: proposed combination end-to-end

Runs inside the backend container:
  docker exec rag-web-ui-backend-1 python3 /app/tests/test_abbr_bidirectional.py
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
logger = logging.getLogger("abbr_bidir_test")

# ─── Config ─────────────────────────────────────────────────────────────────
LM_STUDIO_URL = "http://192.168.1.3:2244/v1"
EMBEDDING_MODEL = "qwen/qwen3-embedding-0.6b"
CHAT_MODELS = ["google/gemma-4-26b-a4b", "qwen/qwen3.5-9b"]
CSV_PATH = "/app/assets/abbreviations_enhanced.csv"
RESULTS_PATH = "/app/assets/abbr_bidirectional_results.json"

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
    # Sort reverse forms by length (longest first) to avoid partial matches
    all_forms = sorted(reverse.keys(), key=len, reverse=True)
    return dict(forward), dict(reverse), all_abbrs, all_forms

FORWARD_MAP, REVERSE_MAP, ALL_ABBRS, ALL_FORMS = load_abbreviations()
print(f"Loaded {len(FORWARD_MAP)} abbreviations, {sum(len(v) for v in FORWARD_MAP.values())} total forms")
print(f"Reverse map: {len(REVERSE_MAP)} unique forms → abbreviations")

# ─── Expansion functions ────────────────────────────────────────────────────

def find_abbrs_in_text(text: str) -> Dict[str, List[str]]:
    """Find all abbreviations in text. Returns {abbr: [forms]}."""
    found = {}
    for abbr in ALL_ABBRS:
        pattern = re.compile(r'\b' + re.escape(abbr) + r'\b', re.IGNORECASE)
        if pattern.search(text):
            found[abbr] = FORWARD_MAP[abbr]
    return found

def find_forms_in_text(text: str) -> Dict[str, List[str]]:
    """Find all expanded forms in text. Returns {abbr: [forms]} (reverse lookup).

    If text contains "battalions", returns {"bns": ["Battalions", ...]}.
    """
    text_lower = text.lower()
    found_abbrs = set()
    for form_lower in ALL_FORMS:
        pattern = re.compile(r'\b' + re.escape(form_lower) + r'\b', re.IGNORECASE)
        if pattern.search(text_lower):
            for abbr in REVERSE_MAP[form_lower]:
                found_abbrs.add(abbr)
    if not found_abbrs:
        return {}
    result = {}
    for abbr in found_abbrs:
        result[abbr] = FORWARD_MAP[abbr]
    return result

# --- CURRENT implementation (what's in production) ---

def expand_query_current(query: str) -> str:
    """Current production: space-join forms, forward-only (abbr→forms)."""
    found = find_abbrs_in_text(query)
    if not found:
        return query
    expansions = []
    for abbr, forms in found.items():
        expansions.extend(forms)
    return f"{query} {' '.join(expansions)}"

# --- PROPOSED implementation ---

def expand_query_glossary_suffix(query: str) -> str:
    """Glossary suffix format, forward-only (abbr→forms).

    "bns wdr" → "bns wdr\n\n[Abbreviation Glossary]\nbns = Battalions\nwdr = Withdraw..."
    """
    found = find_abbrs_in_text(query)
    if not found:
        return query
    lines = [f"{a} = {', '.join(f)}" for a, f in sorted(found.items(), key=lambda x: x.lower())]
    return f"{query}\n\n[Abbreviation Glossary]\n" + "\n".join(lines)

def expand_query_bidirectional_glossary(query: str) -> str:
    """Glossary suffix format, bidirectional (abbr→forms AND form→abbr).

    "bns wdr"           → "bns wdr\n\n[Abbreviation Glossary]\nbns = Battalions\nwdr = Withdraw..."
    "battalions withdrew" → "battalions withdrew\n\n[Abbreviation Glossary]\nbns = Battalions\nwdr = Withdraw..."
    """
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

def expand_query_bidirectional_space(query: str) -> str:
    """Bidirectional but space-joined (for comparison with glossary format)."""
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
        expansions.append(abbr)  # also append the abbreviation itself
    return f"{query} {' '.join(expansions)}"

# --- Ingestion expansion (suffix format, same as production) ---

def expand_suffix(text: str) -> str:
    """Ingestion suffix expansion: text\n\n[Abbreviation Glossary]\nabbr = forms"""
    found = find_abbrs_in_text(text)
    if not found:
        return text
    lines = [f"{a} = {', '.join(f)}" for a, f in sorted(found.items(), key=lambda x: x[0].lower())]
    return f"{text}\n\n[Abbreviation Glossary]\n" + "\n".join(lines)

def build_glossary_block(text: str) -> str:
    """Build glossary block for generation: text + [Abbreviation Glossary]\nabbr = forms"""
    found = find_abbrs_in_text(text)
    if not found:
        return text
    lines = [f"{a} = {', '.join(f)}" for a, f in sorted(found.items(), key=lambda x: x[0].lower())]
    return f"{text}\n[Abbreviation Glossary]\n" + "\n".join(lines)

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

TEST_QUERIES = [
    "battalions withdrew from position",       # Q0: full-form query
    "bns wdr from position",                    # Q1: abbreviation query
    "CO ordered bns to wdr",                    # Q2: mixed abbreviation query
    "brigade headquarters operation objective", # Q3: full-form query (needs reverse)
    "deputy assistant approved resupply",       # Q4: full-form query (multi-meaning DA)
    "weather forecast rain temperature",        # Q5: no abbreviations (control)
]

EXPECTED_MATCHES = {
    "battalions withdrew from position": [0, 4],
    "bns wdr from position": [0, 4],
    "CO ordered bns to wdr": [0, 4],
    "brigade headquarters operation objective": [1],
    "deputy assistant approved resupply": [3],
    "weather forecast rain temperature": [2],
}

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

def get_chat_client():
    from openai import OpenAI
    return OpenAI(api_key="not-required", base_url=LM_STUDIO_URL)

# ─── Embedding/scoring functions ────────────────────────────────────────────

def embed_dense(texts: List[str]) -> List[List[float]]:
    client = get_dense_client()
    embeddings = []
    for i in range(0, len(texts), 32):
        batch = texts[i:i+32]
        resp = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
        embeddings.extend([r.embedding for r in resp.data])
    return embeddings

def embed_sparse(texts: List[str]) -> List:
    embedder = get_sparse_embedder()
    return list(embedder.embed(texts))

def rerank_scores(query: str, passages: List[str]) -> List[float]:
    encoder = get_reranker()
    return list(encoder.rerank(query, passages))

def generate_answer(query: str, context: str, model: str, max_tokens: int = 500) -> str:
    client = get_chat_client()
    system = (
        "You are a military assistant. Answer the user's question based ONLY on "
        "the provided context. If the context doesn't contain the answer, say "
        "'I cannot answer based on the provided context.' Be concise."
    )
    user = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=0.1,
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

# ─── Query expansion variants ───────────────────────────────────────────────

QUERY_VARIANTS = {
    "original":      lambda q: q,
    "current":       expand_query_current,                    # space-join, forward-only
    "glossary":      expand_query_glossary_suffix,             # glossary suffix, forward-only
    "bidir_glossary": expand_query_bidirectional_glossary,     # glossary suffix, bidirectional (PROPOSED)
    "bidir_space":   expand_query_bidirectional_space,         # space-join, bidirectional (for comparison)
}

# ─── TEST A+B: Dense + Sparse ───────────────────────────────────────────────

def test_embeddings():
    print("\n" + "=" * 80)
    print("TEST A+B: Dense + Sparse — Current vs Proposed Query Expansion")
    print("=" * 80)

    # Show what each variant produces
    print("\n  Query expansion examples:")
    for q in TEST_QUERIES[:4]:
        print(f"\n  Query: '{q}'")
        for vname, vfn in QUERY_VARIANTS.items():
            expanded = vfn(q)
            print(f"    {vname:16s}: {expanded[:120]}{'...' if len(expanded) > 120 else ''}")

    # Ingestion: use suffix expansion (same as production)
    ing_fn = expand_suffix
    chunk_texts = [ing_fn(c) for c in TEST_CHUNKS]

    print("\n  Embedding chunks (suffix-expanded)...")
    chunk_dense = embed_dense(chunk_texts)
    chunk_sparse = embed_sparse(chunk_texts)

    print("  Embedding query variants...")
    query_dense = {}
    query_sparse = {}
    for vname, vfn in QUERY_VARIANTS.items():
        texts = [vfn(q) for q in TEST_QUERIES]
        query_dense[vname] = embed_dense(texts)
        query_sparse[vname] = embed_sparse(texts)
        print(f"    {vname}: done")

    # Compute similarities
    print("\n  Results (top-3 per query):")
    for q_idx, query in enumerate(TEST_QUERIES):
        expected = EXPECTED_MATCHES[query]
        print(f"\n  Q{q_idx}: '{query}' → expected {expected}")
        for vname in QUERY_VARIANTS:
            # Dense
            d_sims = sorted(
                enumerate(cosine_similarity(query_dense[vname][q_idx], ce) for ce in chunk_dense),
                key=lambda x: x[1], reverse=True
            )
            d_top2 = [s[0] for s in d_sims[:2]]
            d_hit = any(c in d_top2 for c in expected)
            R.add("DENSE", f"q={vname}", f"q{q_idx}", f"c{d_sims[0][0]}({d_sims[0][1]:.3f})",
                  {"hit": d_hit, "expected": expected, "top3": [(s[0], round(s[1], 3)) for s in d_sims[:3]]})

            # Sparse
            s_sims = sorted(
                enumerate(sparse_dot_product(query_sparse[vname][q_idx], ce) for ce in chunk_sparse),
                key=lambda x: x[1], reverse=True
            )
            s_top2 = [s[0] for s in s_sims[:2]]
            s_hit = any(c in s_top2 for c in expected)
            R.add("SPARSE", f"q={vname}", f"q{q_idx}", f"c{s_sims[0][0]}({s_sims[0][1]:.1f})",
                  {"hit": s_hit, "expected": expected, "top3": [(s[0], round(s[1], 1)) for s in s_sims[:3]]})

            d_str = " ".join(f"c{s[0]}:{s[1]:.3f}" for s in d_sims[:3])
            s_str = " ".join(f"c{s[0]}:{s[1]:.1f}" for s in s_sims[:3])
            print(f"    {vname:16s}  D[{d_str}] S[{s_str}]")

    # Hit rate summary
    print("\n  Hit rates (top-2):")
    for label, cat in [("DENSE", "DENSE"), ("SPARSE", "SPARSE")]:
        print(f"\n  {label}:")
        for vname in QUERY_VARIANTS:
            h, t, pct = hit_rate(R.rows, lambda r: r["cat"] == cat and f"q={vname}" in r["test"])
            print(f"    {vname:16s}  {h}/{t} ({pct:.0f}%)")

# ─── TEST C: Reranker ───────────────────────────────────────────────────────

def test_reranker():
    print("\n" + "=" * 80)
    print("TEST C: Reranker — orig_q vs current_q vs proposed_q + suffix_c")
    print("=" * 80)

    # Reranker combinations
    # Key question: does using original_query (not rewritten) + suffix chunks work?
    # Also: how does the proposed bidirectional glossary query compare?
    RR_COMBOS = [
        # (label, query_fn, chunk_fn)
        ("orig_q+orig_c",        lambda q: q,                              lambda t: t),
        ("orig_q+suffix_c",      lambda q: q,                              expand_suffix),
        ("current_q+suffix_c",   expand_query_current,                     expand_suffix),
        ("glossary_q+suffix_c",  expand_query_glossary_suffix,             expand_suffix),
        ("bidir_glos_q+suffix_c", expand_query_bidirectional_glossary,     expand_suffix),
        ("bidir_space_q+suffix_c", expand_query_bidirectional_space,       expand_suffix),
        ("orig_q+orig_c",        lambda q: q,                              lambda t: t),
    ]

    print(f"\n  Scoring {len(RR_COMBOS)} reranker combos x {len(TEST_QUERIES)} queries...")
    for q_label, q_fn, c_fn in RR_COMBOS:
        for q_idx, query in enumerate(TEST_QUERIES):
            q_text = q_fn(query)
            passages = [c_fn(c) for c in TEST_CHUNKS]
            scores = rerank_scores(q_text, passages)
            scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            expected = EXPECTED_MATCHES[query]
            top2 = [s[0] for s in scored[:2]]
            hit = any(c in top2 for c in expected)
            R.add("RERANKER", q_label, f"q{q_idx}", f"c{scored[0][0]}({scored[0][1]:.3f})",
                  {"hit": hit, "expected": expected, "top3": [(s[0], round(s[1], 3)) for s in scored[:3]]})

    # Hit rate summary
    print("\n  Hit rates (top-2):")
    seen = set()
    for q_label, _, _ in RR_COMBOS:
        if q_label in seen:
            continue
        seen.add(q_label)
        h, t, pct = hit_rate(R.rows, lambda r: r["cat"] == "RERANKER" and r["test"] == q_label)
        print(f"    {q_label:30s}  {h}/{t} ({pct:.0f}%)")

    # Detailed per-query for key combos
    print("\n  Detailed (top-3 per query) for key combos:")
    key_combos = ["orig_q+orig_c", "orig_q+suffix_c", "current_q+suffix_c", "bidir_glos_q+suffix_c"]
    for q_idx, query in enumerate(TEST_QUERIES):
        print(f"\n  Q{q_idx}: '{query}' → {EXPECTED_MATCHES[query]}")
        for combo in key_combos:
            rows = [r for r in R.rows if r["cat"] == "RERANKER" and r["test"] == combo and r["metric"] == f"q{q_idx}"]
            if rows:
                top3 = rows[0]["extra"]["top3"]
                hit = rows[0]["extra"]["hit"]
                print(f"    {combo:30s}  {' '.join(f'c{c}:{s}' for c,s in top3)} {'HIT' if hit else 'MISS'}")

# ─── TEST D: Generation ─────────────────────────────────────────────────────

def test_generation():
    print("\n" + "=" * 80)
    print("TEST D: Generation — Both Chat Models")
    print("=" * 80)

    # Test 1: Abbreviation-heavy chunk, full-form query
    chunk = TEST_CHUNKS[0]
    query = "battalions withdrew from position"

    ctx_opts = {
        "original": chunk,
        "glossary": build_glossary_block(chunk),
        "suffix": expand_suffix(chunk),
    }

    for model in CHAT_MODELS:
        print(f"\n  Model: {model}")
        print(f"  Query: '{query}'")
        print(f"  Chunk: '{chunk[:60]}...'")

        for ctx_name, ctx_text in ctx_opts.items():
            answer = generate_answer(query, ctx_text, model)
            correct = any(w in answer.lower() for w in ["battalion", "withdraw", "order", "position"])
            R.add("GENERATION", f"model={model} ctx={ctx_name}", "test1",
                  correct, {"answer": answer[:200], "query": query, "chunk": 0, "model": model})
            print(f"\n    ctx={ctx_name} ({len(ctx_text)} chars): {'CORRECT' if correct else 'WRONG'}")
            print(f"      {answer[:200]}")

    # Test 2: Multi-meaning abbreviation (DA)
    chunk2 = TEST_CHUNKS[3]
    query2 = "who approved the medical resupply?"

    ctx_opts2 = {
        "original": chunk2,
        "glossary": build_glossary_block(chunk2),
    }

    for model in CHAT_MODELS:
        print(f"\n  Model: {model}")
        print(f"  Query: '{query2}'")
        print(f"  Chunk: '{chunk2[:60]}...'")

        for ctx_name, ctx_text in ctx_opts2.items():
            answer = generate_answer(query2, ctx_text, model)
            # Correct if it mentions DA/deputy assistant/approved, wrong if "daily allowance" or "defence attache"
            correct = any(w in answer.lower() for w in ["deputy", "assistant", "da", "approved"])
            wrong = any(w in answer.lower() for w in ["daily allowance", "defence attache", "defense attache"])
            if wrong:
                correct = False
            R.add("GENERATION", f"model={model} multi_ctx={ctx_name}", "test2",
                  correct, {"answer": answer[:200], "query": query2, "chunk": 3, "model": model})
            print(f"\n    ctx={ctx_name} ({len(ctx_text)} chars): {'CORRECT' if correct else 'WRONG'}")
            print(f"      {answer[:200]}")

    # Test 3: Abbreviation query, full-form chunk (reverse direction)
    chunk3 = TEST_CHUNKS[4]
    query3 = "bns wdr from position"

    ctx_opts3 = {
        "original": chunk3,
        "glossary": build_glossary_block(chunk3),
    }

    for model in CHAT_MODELS:
        print(f"\n  Model: {model}")
        print(f"  Query: '{query3}'")
        print(f"  Chunk: '{chunk3[:60]}...'")

        for ctx_name, ctx_text in ctx_opts3.items():
            answer = generate_answer(query3, ctx_text, model)
            correct = any(w in answer.lower() for w in ["battalion", "withdraw", "commanding", "position"])
            R.add("GENERATION", f"model={model} abbr_q_ctx={ctx_name}", "test3",
                  correct, {"answer": answer[:200], "query": query3, "chunk": 4, "model": model})
            print(f"\n    ctx={ctx_name} ({len(ctx_text)} chars): {'CORRECT' if correct else 'WRONG'}")
            print(f"      {answer[:200]}")

# ─── TEST E: Full Pipeline ──────────────────────────────────────────────────

def test_full_pipeline():
    print("\n" + "=" * 80)
    print("TEST E: Full Pipeline (Dense → Reranker → Generation)")
    print("=" * 80)

    # Proposed pipeline: ingestion=suffix, query=bidir_glossary, reranker=orig_q+suffix_c, gen=glossary
    # Compare against: current pipeline: ingestion=suffix, query=current, reranker=orig_q+suffix_c, gen=glossary
    pipelines = [
        ("CURRENT:  suffix|current_q|orig_q+suffix_c|glossary",
         "current", "orig_q+suffix_c", "glossary"),
        ("PROPOSED: suffix|bidir_glos_q|orig_q+suffix_c|glossary",
         "bidir_glossary", "orig_q+suffix_c", "glossary"),
        ("PROPOSED: suffix|glossary_q|orig_q+suffix_c|glossary",
         "glossary", "orig_q+suffix_c", "glossary"),
        ("BASELINE: suffix|original_q|orig_q+suffix_c|glossary",
         "original", "orig_q+suffix_c", "glossary"),
    ]

    # Use first chat model for generation
    gen_model = CHAT_MODELS[0]

    chunk_texts = [expand_suffix(c) for c in TEST_CHUNKS]
    chunk_embs = embed_dense(chunk_texts)

    for label, q_variant, rr_combo, gen_ctx in pipelines:
        print(f"\n  Pipeline: {label}")
        q_fn = QUERY_VARIANTS[q_variant]

        for q_idx, query in enumerate(TEST_QUERIES):
            expected = EXPECTED_MATCHES[query]

            # Dense retrieval
            q_text = q_fn(query)
            q_emb = embed_dense([q_text])[0]
            sims = sorted(enumerate(cosine_similarity(q_emb, ce) for ce in chunk_embs),
                          key=lambda x: x[1], reverse=True)
            top3_idx = [s[0] for s in sims[:3]]

            # Rerank — use ORIGINAL query (proposed) + suffix chunks
            rr_query = query  # always original query for reranker
            rr_passages = [expand_suffix(TEST_CHUNKS[i]) for i in top3_idx]
            rr_scores = rerank_scores(rr_query, rr_passages)
            rr_ranked = sorted(zip(top3_idx, rr_scores), key=lambda x: x[1], reverse=True)
            rr_top1 = rr_ranked[0][0]
            rr_hit = rr_top1 in expected

            # Generate
            top_chunk = TEST_CHUNKS[rr_top1]
            gen_context = build_glossary_block(top_chunk)
            answer = generate_answer(query, gen_context, gen_model, max_tokens=300)

            # Check correctness
            if "deputy" in query.lower() or "approved" in query.lower():
                correct = any(w in answer.lower() for w in ["deputy", "assistant", "da", "approved"])
                wrong = any(w in answer.lower() for w in ["daily allowance", "defence attache", "defense attache"])
                if wrong:
                    correct = False
            elif "weather" in query.lower():
                correct = any(w in answer.lower() for w in ["rain", "temperature", "weather", "degrees", "wet"])
            else:
                correct = any(w in answer.lower() for w in ["battalion", "withdraw", "order", "position",
                                                              "brigade", "headquarters", "operation", "objective"])

            R.add("PIPELINE", label, f"q{q_idx}", f"hit={rr_hit} correct={correct}",
                  {"rr_top1": rr_top1, "expected": expected, "answer": answer[:150],
                   "query": query, "q_variant": q_variant})

            status = "HIT" if rr_hit else "MISS"
            corr = "OK" if correct else "BAD"
            print(f"    Q{q_idx}: {status} {corr}  top1=c{rr_top1}  answer: {answer[:80]}")

    # Summary
    print("\n  Pipeline summary:")
    for label, _, _, _ in pipelines:
        rows = [r for r in R.rows if r["cat"] == "PIPELINE" and r["test"] == label]
        hits = sum(1 for r in rows if "hit=True" in r["value"])
        correct = sum(1 for r in rows if "correct=True" in r["value"])
        print(f"    {label:65s}  hit={hits}/{len(rows)}  correct={correct}/{len(rows)}")

# ─── MAIN ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("ABBREVIATION BIDIRECTIONAL EXPANSION TEST")
    print(f"Embedding: {EMBEDDING_MODEL}")
    print(f"Chat models: {', '.join(CHAT_MODELS)}")
    print(f"LM Studio: {LM_STUDIO_URL}")
    print(f"Abbreviations: {len(FORWARD_MAP)}, chunks: {len(TEST_CHUNKS)}, queries: {len(TEST_QUERIES)}")
    print("=" * 80)

    # Show bidirectional expansion examples
    print("\n  Bidirectional expansion examples:")
    test_cases = [
        "bns wdr from position",
        "battalions withdrew from position",
        "CO ordered bns to wdr",
        "brigade headquarters operation objective",
        "deputy assistant approved resupply",
        "weather forecast rain temperature",
    ]
    for q in test_cases:
        current = expand_query_current(q)
        proposed = expand_query_bidirectional_glossary(q)
        print(f"\n  Query:    '{q}'")
        print(f"  Current:  '{current[:100]}{'...' if len(current) > 100 else ''}'")
        print(f"  Proposed: '{proposed[:100]}{'...' if len(proposed) > 100 else ''}'")

    test_embeddings()
    test_reranker()
    test_generation()
    test_full_pipeline()

    R.save(RESULTS_PATH)
    print(f"\nResults saved to {RESULTS_PATH}")

    # ─── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print("\n1. DENSE hit rates (top-2) by query variant:")
    for vname in QUERY_VARIANTS:
        h, t, pct = hit_rate(R.rows, lambda r: r["cat"] == "DENSE" and f"q={vname}" in r["test"])
        print(f"   {vname:16s}  {h}/{t} ({pct:.0f}%)")

    print("\n2. SPARSE hit rates (top-2) by query variant:")
    for vname in QUERY_VARIANTS:
        h, t, pct = hit_rate(R.rows, lambda r: r["cat"] == "SPARSE" and f"q={vname}" in r["test"])
        print(f"   {vname:16s}  {h}/{t} ({pct:.0f}%)")

    print("\n3. RERANKER hit rates (top-2):")
    rr_combos = sorted(set(r["test"] for r in R.rows if r["cat"] == "RERANKER"))
    for combo in rr_combos:
        h, t, pct = hit_rate(R.rows, lambda r: r["cat"] == "RERANKER" and r["test"] == combo)
        print(f"   {combo:30s}  {h}/{t} ({pct:.0f}%)")

    print("\n4. GENERATION correctness by model+context:")
    gen_tests = sorted(set(r["test"] for r in R.rows if r["cat"] == "GENERATION"))
    for gt in gen_tests:
        rows = [r for r in R.rows if r["cat"] == "GENERATION" and r["test"] == gt]
        correct = sum(1 for r in rows if r["value"])
        print(f"   {gt:55s}  {correct}/{len(rows)} correct")

    print("\n5. PIPELINE results:")
    for label, _, _, _ in [("CURRENT:  suffix|current_q|orig_q+suffix_c|glossary", None, None, None),
                           ("PROPOSED: suffix|bidir_glos_q|orig_q+suffix_c|glossary", None, None, None),
                           ("PROPOSED: suffix|glossary_q|orig_q+suffix_c|glossary", None, None, None),
                           ("BASELINE: suffix|original_q|orig_q+suffix_c|glossary", None, None, None)]:
        rows = [r for r in R.rows if r["cat"] == "PIPELINE" and r["test"] == label]
        if rows:
            hits = sum(1 for r in rows if "hit=True" in r["value"])
            correct = sum(1 for r in rows if "correct=True" in r["value"])
            print(f"   {label:65s}  hit={hits}/{len(rows)}  correct={correct}/{len(rows)}")

if __name__ == "__main__":
    main()
