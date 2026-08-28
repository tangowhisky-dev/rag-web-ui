#!/usr/bin/env python3
"""
Comprehensive abbreviation expansion test suite v2.

Tests ALL combinations of expansion options for each pipeline stage using
real models:
  - Dense embeddings: qwen3-embedding-0.6b (LM Studio, dim=1024)
  - Sparse embeddings: SPLADE PP en v1 (FastEmbed, local ONNX)
  - Reranker: ms-marco-MiniLM-L-12-v2 (FastEmbed, local ONNX)
  - Generation: gemma-4-12b (LM Studio, no thinking)

Expansion strategies tested:
  INGESTION:  none | suffix | replace | glossary_suffix | replace+glossary
  QUERY:      original | suffix | replace | llm_replace | llm_replace+glossary
  RERANKER:   orig_q+orig_c | orig_q+glossary_c | orig_q+suffix_c | orig_q+replace_c
              | exp_q+orig_c | exp_q+glossary_c | exp_q+suffix_c
              | llm_q+orig_c | llm_q+glossary_c
  GENERATION: orig_only | glossary | suffix | replace | replace+glossary | llm_glossary

Runs inside the backend container. No app code changes.
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
logger = logging.getLogger("abbr_test")

# ─── Load abbreviation CSV ─────────────────────────────────────────────────
CSV_PATH = "/app/assets/abbreviations_enhanced.csv"

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
    return dict(forward), dict(reverse), all_abbrs

FORWARD_MAP, REVERSE_MAP, ALL_ABBRS = load_abbreviations()
print(f"Loaded {len(FORWARD_MAP)} abbreviations, {sum(len(v) for v in FORWARD_MAP.values())} total forms")

# ─── Expansion functions ───────────────────────────────────────────────────

def find_abbrs_in_text(text: str) -> Dict[str, List[str]]:
    """Find all abbreviations in text, return {abbr: [forms]}."""
    found = {}
    for abbr in ALL_ABBRS:
        pattern = re.compile(r'\b' + re.escape(abbr) + r'\b', re.IGNORECASE)
        if pattern.search(text):
            found[abbr] = FORWARD_MAP[abbr]
    return found

def expand_suffix(text: str) -> str:
    found = find_abbrs_in_text(text)
    if not found:
        return text
    lines = [f"{a} = {', '.join(f)}" for a, f in sorted(found.items(), key=lambda x: x[0].lower())]
    return f"{text}\n\n[Abbreviation Glossary]\n" + "\n".join(lines)

def expand_replace(text: str) -> str:
    result = text
    for abbr in ALL_ABBRS:
        forms = FORWARD_MAP[abbr]
        pattern = re.compile(r'\b' + re.escape(abbr) + r'\b', re.IGNORECASE)
        result = pattern.sub(" ".join(forms), result)
    return result

def build_glossary(text: str) -> str:
    found = find_abbrs_in_text(text)
    if not found:
        return ""
    return "\n".join(f"{a} = {', '.join(f)}" for a, f in sorted(found.items(), key=lambda x: x[0].lower()))

def expand_glossary_suffix(text: str) -> str:
    glossary = build_glossary(text)
    if not glossary:
        return text
    return f"{text}\n[Abbreviation Glossary]\n{glossary}"

def expand_replace_plus_glossary(text: str) -> str:
    """Replace abbreviations with first form, append glossary with all forms."""
    found = find_abbrs_in_text(text)
    if not found:
        return text
    # Replace each abbreviation with its first (primary) form
    result = text
    for abbr in sorted(found.keys(), key=len, reverse=True):
        primary = found[abbr][0]
        pattern = re.compile(r'\b' + re.escape(abbr) + r'\b', re.IGNORECASE)
        result = pattern.sub(primary, result)
    # Append glossary with all forms
    glossary = "\n".join(f"{a} = {', '.join(f)}" for a, f in sorted(found.items(), key=lambda x: x[0].lower()))
    return f"{result}\n[Abbreviation Glossary]\n{glossary}"

def expand_query_suffix(query: str) -> str:
    result = query
    found_abbrs = set()
    for abbr in ALL_ABBRS:
        pattern = re.compile(r'\b' + re.escape(abbr) + r'\b', re.IGNORECASE)
        if pattern.search(query):
            found_abbrs.add(abbr)
    for abbr in found_abbrs:
        result += " " + " ".join(FORWARD_MAP[abbr])
    query_lower = query.lower()
    for form_lower, abbrs in REVERSE_MAP.items():
        pattern = re.compile(r'\b' + re.escape(form_lower) + r'\b', re.IGNORECASE)
        if pattern.search(query_lower):
            for abbr in abbrs:
                if abbr not in found_abbrs:
                    result += " " + abbr
    return result

def expand_query_replace(query: str) -> str:
    return expand_replace(query)

# ─── Test data ─────────────────────────────────────────────────────────────

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
    "battalions withdrew from position",
    "bns wdr from position",
    "CO ordered bns to wdr",
    "brigade headquarters operation objective",
    "deputy assistant approved resupply",
    "weather forecast rain temperature",
]

EXPECTED_MATCHES = {
    "battalions withdrew from position": [0, 4],
    "bns wdr from position": [0, 4],
    "CO ordered bns to wdr": [0, 4],
    "brigade headquarters operation objective": [1],
    "deputy assistant approved resupply": [3],
    "weather forecast rain temperature": [2],
}

# ─── Model accessors ───────────────────────────────────────────────────────

# LM Studio configuration — hardcoded fallbacks for when DB settings are unavailable
# (e.g. test DB doesn't have app_settings table populated).
_LM_STUDIO_BASE = os.environ.get("LM_STUDIO_BASE_URL", "http://192.168.1.3:2244/v1")
_LM_STUDIO_KEY = os.environ.get("LM_STUDIO_API_KEY", "dummy")
_DENSE_MODEL = os.environ.get("DENSE_EMBEDDINGS_MODEL", "qwen/qwen3-embedding-0.6b")
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "google/gemma-4-26b-a4b")

def get_dense_embedder():
    from openai import OpenAI
    from app.db.session import SessionLocal
    from app.services.settings_service import get_setting
    db = SessionLocal()
    try:
        api_key = get_setting(db, "EMBEDDING_API_KEY", None) or _LM_STUDIO_KEY
        api_base = get_setting(db, "EMBEDDING_API_BASE", None) or _LM_STUDIO_BASE
        model = get_setting(db, "DENSE_EMBEDDINGS_MODEL", None) or _DENSE_MODEL
    finally:
        db.close()
    return OpenAI(api_key=api_key, base_url=api_base), model

def get_sparse_embedder():
    from app.services.infrastructure import get_sparse_embedder
    return get_sparse_embedder()

def get_reranker():
    from app.services.retrieval.reranker import _get_cross_encoder
    return _get_cross_encoder()

def get_generation_client():
    from openai import OpenAI
    from app.db.session import SessionLocal
    from app.services.settings_service import get_setting
    db = SessionLocal()
    try:
        api_key = get_setting(db, "OPENAI_API_KEY", None) or _LM_STUDIO_KEY
        api_base = get_setting(db, "OPENAI_API_BASE", None) or _LM_STUDIO_BASE
    finally:
        db.close()
    return OpenAI(api_key=api_key, base_url=api_base), GENERATION_MODEL

# ─── Embedding/scoring functions ───────────────────────────────────────────

def embed_dense(texts: List[str]) -> List[List[float]]:
    client, model = get_dense_embedder()
    embeddings = []
    for i in range(0, len(texts), 32):
        batch = texts[i:i+32]
        resp = client.embeddings.create(input=batch, model=model)
        embeddings.extend([r.embedding for r in resp.data])
    return embeddings

def embed_sparse(texts: List[str]) -> List:
    embedder = get_sparse_embedder()
    return list(embedder.embed(texts))

def rerank_scores(query: str, passages: List[str]) -> List[float]:
    encoder = get_reranker()
    return list(encoder.rerank(query, passages))

def llm_replace_abbrs(query: str) -> str:
    """Use LLM to replace abbreviations with correct full forms in context."""
    client, model = get_generation_client()
    system = (
        "You are a military abbreviation expander. Replace each military "
        "abbreviation in the user's query with its correct full form. "
        "Output ONLY the rewritten query, nothing else. "
        "If no abbreviations are present, output the query unchanged."
    )
    user = f"Query: {query}\nRewritten:"
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=200,
                temperature=0.0,
            )
            content = resp.choices[0].message.content.strip()
            # Take first non-empty line
            lines = [l.strip() for l in content.split("\n") if l.strip()]
            return lines[0] if lines else query
        except Exception as exc:
            if attempt < 2:
                time.sleep(3)
                continue
            return query

def llm_replace_plus_glossary(query: str) -> str:
    """LLM replacement + append deterministic glossary."""
    replaced = llm_replace_abbrs(query)
    glossary = build_glossary(replaced)
    if glossary:
        return f"{replaced}\n[Abbreviation Glossary]\n{glossary}"
    return replaced

def generate_answer(query: str, context: str, max_tokens: int = 500) -> str:
    client, model = get_generation_client()
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

# ─── Results collector ─────────────────────────────────────────────────────

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

# ─── Ingestion expansion options ───────────────────────────────────────────

INGESTION_OPTS = {
    "none": lambda t: t,
    "suffix": expand_suffix,
    "replace": expand_replace,
    "glossary_suffix": expand_glossary_suffix,
    "replace+glossary": expand_replace_plus_glossary,
}

QUERY_OPTS = {
    "original": lambda q: q,
    "suffix": expand_query_suffix,
    "replace": expand_query_replace,
}

# ─── TEST 1+2: Dense + Sparse ──────────────────────────────────────────────

def test_embeddings():
    print("\n" + "=" * 80)
    print("TEST 1+2: Dense + Sparse Embedding Similarity")
    print("=" * 80)

    # Embed all chunk variants
    print("\nEmbedding chunk variants (dense + sparse)...")
    chunk_dense = {}
    chunk_sparse = {}
    for ing_name, ing_fn in INGESTION_OPTS.items():
        texts = [ing_fn(c) for c in TEST_CHUNKS]
        chunk_dense[ing_name] = embed_dense(texts)
        chunk_sparse[ing_name] = embed_sparse(texts)
        print(f"  {ing_name}: done")

    # Embed all query variants (deterministic only — LLM variants tested separately)
    print("Embedding query variants (dense + sparse)...")
    query_dense = {}
    query_sparse = {}
    for q_name, q_fn in QUERY_OPTS.items():
        texts = [q_fn(q) for q in TEST_QUERIES]
        query_dense[q_name] = embed_dense(texts)
        query_sparse[q_name] = embed_sparse(texts)
        print(f"  {q_name}: done")

    # Compute similarities
    print("\nResults (top-3 per query, dense cosine / sparse dot):")
    for q_idx, query in enumerate(TEST_QUERIES):
        expected = EXPECTED_MATCHES[query]
        print(f"\n  Q{q_idx}: '{query}' → expected {expected}")
        for ing_name in INGESTION_OPTS:
            for q_name in QUERY_OPTS:
                # Dense
                d_sims = sorted(
                    enumerate(cosine_similarity(query_dense[q_name][q_idx], ce) for ce in chunk_dense[ing_name]),
                    key=lambda x: x[1], reverse=True
                )
                d_top2 = [s[0] for s in d_sims[:2]]
                d_hit = any(c in d_top2 for c in expected)
                R.add("DENSE", f"ing={ing_name} q={q_name}", f"q{q_idx}", f"c{d_sims[0][0]}({d_sims[0][1]:.3f})", {"hit": d_hit, "expected": expected, "top3": [(s[0], round(s[1], 3)) for s in d_sims[:3]]})

                # Sparse
                s_sims = sorted(
                    enumerate(sparse_dot_product(query_sparse[q_name][q_idx], ce) for ce in chunk_sparse[ing_name]),
                    key=lambda x: x[1], reverse=True
                )
                s_top2 = [s[0] for s in s_sims[:2]]
                s_hit = any(c in s_top2 for c in expected)
                R.add("SPARSE", f"ing={ing_name} q={q_name}", f"q{q_idx}", f"c{s_sims[0][0]}({s_sims[0][1]:.1f})", {"hit": s_hit, "expected": expected, "top3": [(s[0], round(s[1], 1)) for s in s_sims[:3]]})

                d_str = " ".join(f"c{s[0]}:{s[1]:.3f}" for s in d_sims[:3])
                s_str = " ".join(f"c{s[0]}:{s[1]:.1f}" for s in s_sims[:3])
                print(f"    ing={ing_name:16s} q={q_name:10s} D[{d_str}] S[{s_str}]")

    # Print hit rate summary
    print("\n  Hit rates (top-2):")
    for label, cat in [("DENSE", "DENSE"), ("SPARSE", "SPARSE")]:
        print(f"\n  {label}:")
        for ing_name in INGESTION_OPTS:
            for q_name in QUERY_OPTS:
                h, t, pct = hit_rate(R.rows, lambda r: r["cat"] == cat and f"ing={ing_name}" in r["test"] and f"q={q_name}" in r["test"])
                print(f"    ing={ing_name:16s} q={q_name:10s}  {h}/{t} ({pct:.0f}%)")

# ─── TEST 3: Reranker ──────────────────────────────────────────────────────

def test_reranker():
    print("\n" + "=" * 80)
    print("TEST 3: Reranker (ms-marco-MiniLM-L-12-v2)")
    print("=" * 80)

    # Reranker combinations: (query_variant, chunk_variant)
    RR_COMBOS = [
        ("orig_q", "orig_c",       lambda q: q,                  lambda t: t),
        ("orig_q", "suffix_c",     lambda q: q,                  expand_suffix),
        ("orig_q", "replace_c",    lambda q: q,                  expand_replace),
        ("orig_q", "glossary_c",   lambda q: q,                  expand_glossary_suffix),
        ("orig_q", "repglos_c",    lambda q: q,                  expand_replace_plus_glossary),
        ("exp_q",  "orig_c",       expand_query_suffix,          lambda t: t),
        ("exp_q",  "suffix_c",     expand_query_suffix,          expand_suffix),
        ("exp_q",  "glossary_c",   expand_query_suffix,          expand_glossary_suffix),
        ("exp_q",  "repglos_c",    expand_query_suffix,          expand_replace_plus_glossary),
        ("rep_q",  "orig_c",       expand_query_replace,         lambda t: t),
        ("rep_q",  "glossary_c",   expand_query_replace,         expand_glossary_suffix),
        ("rep_q",  "repglos_c",    expand_query_replace,         expand_replace_plus_glossary),
    ]

    print(f"\nScoring {len(RR_COMBOS)} reranker combos x {len(TEST_QUERIES)} queries...")
    for q_label, c_label, q_fn, c_fn in RR_COMBOS:
        for q_idx, query in enumerate(TEST_QUERIES):
            q_text = q_fn(query)
            passages = [c_fn(c) for c in TEST_CHUNKS]
            scores = rerank_scores(q_text, passages)
            scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            expected = EXPECTED_MATCHES[query]
            top2 = [s[0] for s in scored[:2]]
            hit = any(c in top2 for c in expected)
            R.add("RERANKER", f"q={q_label} c={c_label}", f"q{q_idx}", f"c{scored[0][0]}({scored[0][1]:.3f})", {"hit": hit, "expected": expected, "top3": [(s[0], round(s[1], 3)) for s in scored[:3]]})

    # Print results
    print("\n  Hit rates (top-2):")
    for q_label, c_label, _, _ in RR_COMBOS:
        h, t, pct = hit_rate(R.rows, lambda r: r["cat"] == "RERANKER" and f"q={q_label}" in r["test"] and f"c={c_label}" in r["test"])
        print(f"    q={q_label:8s} c={c_label:12s}  {h}/{t} ({pct:.0f}%)")

    # Detailed per-query for key combos
    print("\n  Detailed (top-3 per query) for key combos:")
    key_combos = [("orig_q", "orig_c"), ("orig_q", "glossary_c"), ("orig_q", "repglos_c"), ("exp_q", "orig_c"), ("exp_q", "glossary_c")]
    for q_idx, query in enumerate(TEST_QUERIES):
        print(f"\n  Q{q_idx}: '{query}' → {EXPECTED_MATCHES[query]}")
        for q_label, c_label in key_combos:
            rows = [r for r in R.rows if r["cat"] == "RERANKER" and f"q={q_label}" in r["test"] and f"c={c_label}" in r["test"] and r["metric"] == f"q{q_idx}"]
            if rows:
                top3 = rows[0]["extra"]["top3"]
                hit = rows[0]["extra"]["hit"]
                print(f"    q={q_label:8s} c={c_label:12s}  {' '.join(f'c{c}:{s}' for c,s in top3)} {'✓' if hit else '✗'}")

# ─── TEST 4: LLM Query Expansion ───────────────────────────────────────────

def test_llm_query_expansion():
    print("\n" + "=" * 80)
    print("TEST 4: LLM Query Expansion (gemma-4-12b)")
    print("=" * 80)

    abbr_queries = [q for q in TEST_QUERIES if any(
        re.search(r'\b' + re.escape(a) + r'\b', q, re.IGNORECASE) for a in ALL_ABBRS if len(a) >= 2
    )]

    print(f"\nLLM-expanding {len(abbr_queries)} abbreviation queries...")
    llm_expanded = {}
    for query in abbr_queries:
        expanded = llm_replace_abbrs(query)
        llm_expanded[query] = expanded
        print(f"  '{query}' → '{expanded}'")
        R.add("LLM_EXPAND", "query_replace", query[:30], expanded, {"original": query, "expanded": expanded})

    # Compare in dense retrieval (against original chunks)
    if llm_expanded:
        print("\nComparing LLM vs deterministic vs original in dense retrieval:")
        chunk_embs = embed_dense(TEST_CHUNKS)

        for query, llm_exp in llm_expanded.items():
            llm_emb = embed_dense([llm_exp])[0]
            det_exp = expand_query_suffix(query)
            det_emb = embed_dense([det_exp])[0]
            orig_emb = embed_dense([query])[0]

            expected = EXPECTED_MATCHES.get(query, [])
            llm_sims = sorted(enumerate(cosine_similarity(llm_emb, ce) for ce in chunk_embs), key=lambda x: x[1], reverse=True)
            det_sims = sorted(enumerate(cosine_similarity(det_emb, ce) for ce in chunk_embs), key=lambda x: x[1], reverse=True)
            orig_sims = sorted(enumerate(cosine_similarity(orig_emb, ce) for ce in chunk_embs), key=lambda x: x[1], reverse=True)

            for label, sims in [("original", orig_sims), ("llm_replace", llm_sims), ("det_suffix", det_sims)]:
                top2 = [s[0] for s in sims[:2]]
                hit = any(c in top2 for c in expected)
                R.add("LLM_VS_DET", f"q_{label}", f"hit_{query[:20]}", hit, {"top1": sims[0][0], "top1_sim": round(sims[0][1], 3), "expected": expected})

            print(f"\n  '{query}' → expected {expected}")
            print(f"    original:     {' '.join(f'c{s[0]}:{s[1]:.3f}' for s in orig_sims[:3])}")
            print(f"    llm_replace:  {' '.join(f'c{s[0]}:{s[1]:.3f}' for s in llm_sims[:3])}")
            print(f"    det_suffix:   {' '.join(f'c{s[0]}:{s[1]:.3f}' for s in det_sims[:3])}")

# ─── TEST 5: Generation Quality ────────────────────────────────────────────

def test_generation():
    print("\n" + "=" * 80)
    print("TEST 5: Generation Quality (gemma-4-12b)")
    print("=" * 80)

    # Test 1: Abbreviation-heavy chunk, full-form query
    chunk = TEST_CHUNKS[0]
    query = "battalions withdrew from position"

    ctx_opts = {
        "original": chunk,
        "glossary": expand_glossary_suffix(chunk),
        "suffix": expand_suffix(chunk),
        "replace": expand_replace(chunk),
        "replace+glossary": expand_replace_plus_glossary(chunk),
    }

    print(f"\n  Query: '{query}'")
    print(f"  Chunk: '{chunk[:60]}...'")

    for ctx_name, ctx_text in ctx_opts.items():
        answer = generate_answer(query, ctx_text)
        correct = any(w in answer.lower() for w in ["battalion", "withdraw", "order", "position"])
        R.add("GENERATION", f"ctx={ctx_name}", "correct", correct, {"answer": answer[:150], "query": query, "chunk": 0})
        print(f"\n  ctx={ctx_name} ({len(ctx_text)} chars):")
        print(f"    {answer[:200]}")

    # Test 2: Multi-meaning abbreviation (DA)
    chunk2 = TEST_CHUNKS[3]
    query2 = "who approved the medical resupply?"

    ctx_opts2 = {
        "original": chunk2,
        "glossary": expand_glossary_suffix(chunk2),
        "suffix": expand_suffix(chunk2),
        "replace+glossary": expand_replace_plus_glossary(chunk2),
    }

    print(f"\n  Query: '{query2}'")
    print(f"  Chunk: '{chunk2[:60]}...'")

    for ctx_name, ctx_text in ctx_opts2.items():
        answer = generate_answer(query2, ctx_text)
        correct = any(w in answer.lower() for w in ["deputy", "assistant", "attache", "da", "approved"])
        R.add("GENERATION", f"multi_ctx={ctx_name}", "correct", correct, {"answer": answer[:150], "query": query2, "chunk": 3})
        print(f"\n  ctx={ctx_name} ({len(ctx_text)} chars):")
        print(f"    {answer[:200]}")

    # Test 3: Abbreviation query, full-form chunk (reverse direction)
    chunk3 = TEST_CHUNKS[4]
    query3 = "bns wdr from position"

    ctx_opts3 = {
        "original": chunk3,
        "glossary": expand_glossary_suffix(chunk3),
    }

    print(f"\n  Query: '{query3}'")
    print(f"  Chunk: '{chunk3[:60]}...'")

    for ctx_name, ctx_text in ctx_opts3.items():
        answer = generate_answer(query3, ctx_text)
        correct = any(w in answer.lower() for w in ["battalion", "withdraw", "commanding", "position"])
        R.add("GENERATION", f"abbr_q_ctx={ctx_name}", "correct", correct, {"answer": answer[:150], "query": query3, "chunk": 4})
        print(f"\n  ctx={ctx_name} ({len(ctx_text)} chars):")
        print(f"    {answer[:200]}")

# ─── TEST 6: Full Pipeline (Dense → Reranker → Generation) ─────────────────

def test_full_pipeline():
    print("\n" + "=" * 80)
    print("TEST 6: Full Pipeline (Dense → Reranker → Generation)")
    print("=" * 80)

    # Test the most promising combinations
    pipeline_combos = [
        # (ingestion, query, reranker_q, reranker_c, gen_context)
        ("none",            "original", "orig_q", "orig_c",    "glossary"),
        ("none",            "suffix",   "orig_q", "glossary_c","glossary"),
        ("suffix",          "suffix",   "orig_q", "orig_c",    "glossary"),
        ("suffix",          "suffix",   "exp_q",  "glossary_c","glossary"),
        ("replace",         "original", "orig_q", "orig_c",    "glossary"),
        ("replace+glossary","original", "orig_q", "orig_c",    "replace+glossary"),
        ("replace+glossary","suffix",   "orig_q", "repglos_c", "replace+glossary"),
        ("glossary_suffix", "suffix",   "orig_q", "glossary_c","glossary"),
    ]

    test_query = "battalions withdrew from position"
    expected = EXPECTED_MATCHES[test_query]

    print(f"\n  Query: '{test_query}' → expected {expected}")

    ing_fns = {
        "none": lambda t: t,
        "suffix": expand_suffix,
        "replace": expand_replace,
        "glossary_suffix": expand_glossary_suffix,
        "replace+glossary": expand_replace_plus_glossary,
    }
    rr_q_fns = {
        "orig_q": lambda q: q,
        "exp_q": expand_query_suffix,
    }
    rr_c_fns = {
        "orig_c": lambda t: t,
        "glossary_c": expand_glossary_suffix,
        "repglos_c": expand_replace_plus_glossary,
    }
    gen_fns = {
        "glossary": lambda t: expand_glossary_suffix(t),
        "replace+glossary": lambda t: expand_replace_plus_glossary(t),
        "original": lambda t: t,
    }
    q_fns = {
        "original": lambda q: q,
        "suffix": expand_query_suffix,
    }

    for ing_name, q_name, rr_q, rr_c, gen_ctx in pipeline_combos:
        # Dense retrieval
        chunk_texts = [ing_fns[ing_name](c) for c in TEST_CHUNKS]
        chunk_embs = embed_dense(chunk_texts)
        q_text = q_fns[q_name](test_query)
        q_emb = embed_dense([q_text])[0]
        sims = sorted(enumerate(cosine_similarity(q_emb, ce) for ce in chunk_embs), key=lambda x: x[1], reverse=True)
        top3_idx = [s[0] for s in sims[:3]]

        # Rerank
        rr_query = rr_q_fns[rr_q](test_query)
        rr_passages = [rr_c_fns[rr_c](TEST_CHUNKS[i]) for i in top3_idx]
        rr_scores = rerank_scores(rr_query, rr_passages)
        rr_ranked = sorted(zip(top3_idx, rr_scores), key=lambda x: x[1], reverse=True)
        rr_top1 = rr_ranked[0][0]
        rr_hit = rr_top1 in expected

        # Generate
        top_chunk = TEST_CHUNKS[rr_top1]
        gen_context = gen_fns[gen_ctx](top_chunk)
        answer = generate_answer(test_query, gen_context, max_tokens=300)
        answer_correct = any(w in answer.lower() for w in ["battalion", "withdraw", "order", "position"])

        R.add("PIPELINE", f"ing={ing_name} q={q_name} rr={rr_q}+{rr_c} gen={gen_ctx}",
              "result", f"hit={rr_hit} correct={answer_correct}",
              {"rr_top1": rr_top1, "expected": expected, "answer": answer[:100]})

        print(f"  ing={ing_name:16s} q={q_name:10s} rr={rr_q}+{rr_c:12s} gen={gen_ctx:16s} → top1=c{rr_top1} hit={rr_hit} correct={answer_correct}")
        print(f"    answer: {answer[:120]}")

# ─── MAIN ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print(f"ABBREVIATION EXPANSION TEST SUITE v2")
    print(f"Models: dense=qwen3-embedding-0.6b, sparse=SPLADE, reranker=MiniLM, gen={GENERATION_MODEL}")
    print(f"Abbreviations: {len(FORWARD_MAP)}, chunks: {len(TEST_CHUNKS)}, queries: {len(TEST_QUERIES)}")
    print("=" * 80)

    test_embeddings()
    test_reranker()
    test_llm_query_expansion()
    test_generation()
    test_full_pipeline()

    R.save("/app/assets/abbr_test_results_v2.json")
    print(f"\nResults saved to /app/assets/abbr_test_results_v2.json")

    # ─── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print("\n1. DENSE hit rates (top-2):")
    for ing_name in INGESTION_OPTS:
        for q_name in QUERY_OPTS:
            h, t, pct = hit_rate(R.rows, lambda r: r["cat"] == "DENSE" and f"ing={ing_name}" in r["test"] and f"q={q_name}" in r["test"])
            if t: print(f"   ing={ing_name:16s} q={q_name:10s}  {h}/{t} ({pct:.0f}%)")

    print("\n2. SPARSE hit rates (top-2):")
    for ing_name in INGESTION_OPTS:
        for q_name in QUERY_OPTS:
            h, t, pct = hit_rate(R.rows, lambda r: r["cat"] == "SPARSE" and f"ing={ing_name}" in r["test"] and f"q={q_name}" in r["test"])
            if t: print(f"   ing={ing_name:16s} q={q_name:10s}  {h}/{t} ({pct:.0f}%)")

    print("\n3. RERANKER hit rates (top-2):")
    rr_combos = sorted(set(r["test"] for r in R.rows if r["cat"] == "RERANKER"))
    for combo in rr_combos:
        h, t, pct = hit_rate(R.rows, lambda r: r["cat"] == "RERANKER" and r["test"] == combo)
        if t: print(f"   {combo:35s}  {h}/{t} ({pct:.0f}%)")

    print("\n4. GENERATION correctness:")
    gen_tests = sorted(set(r["test"] for r in R.rows if r["cat"] == "GENERATION"))
    for gt in gen_tests:
        rows = [r for r in R.rows if r["cat"] == "GENERATION" and r["test"] == gt]
        correct = sum(1 for r in rows if r["value"])
        print(f"   {gt:35s}  {correct}/{len(rows)} correct")

    print("\n5. PIPELINE results:")
    for r in R.rows:
        if r["cat"] == "PIPELINE":
            print(f"   {r['test']:60s}  {r['value']}")

if __name__ == "__main__":
    main()
