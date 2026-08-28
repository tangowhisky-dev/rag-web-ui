#!/usr/bin/env python3
"""
Test LLM-based abbreviation replacement with glossary context.

Unlike deterministic replacement (which picks the first form from the CSV),
this gives the LLM all possible meanings and asks it to choose the correct
one based on surrounding context.

Tests:
  1. LLM replacement quality on chunks (does it pick the right meaning?)
  2. LLM replacement quality on queries
  3. Dense retrieval with LLM-replaced chunks vs suffix chunks
  4. Reranker with LLM-replaced chunks
  5. Generation with LLM-replaced context
  6. Full pipeline comparison

Model: gemma-4-12b (no thinking)
"""
import json
import os
import re
import sys
import time
from collections import defaultdict
from typing import List, Dict

sys.path.insert(0, "/app")
os.environ.setdefault("PYTHONPATH", "/app")

import pytest


@pytest.fixture(scope="module")
def llm_replaced_chunks():
    """Module-scoped fixture: run LLM replacement on all test chunks once.

    test_llm_replacement_quality() was originally designed as a script
    function that returns a list.  pytest treats its return value as a
    test failure (PytestReturnNotNoneWarning), and downstream tests
    (test_dense_retrieval, test_reranker, test_full_pipeline) expect
    the result as a parameter.  This fixture bridges the gap.
    """
    chunks = []
    for chunk in TEST_CHUNKS:
        llm_replaced = llm_replace_with_glossary(chunk)
        chunks.append(llm_replaced)
    return chunks


@pytest.fixture(scope="module")
def llm_queries():
    """Module-scoped fixture: run LLM query replacement once."""
    queries = {}
    for query in TEST_QUERIES:
        found = any(
            re.search(r'\b' + re.escape(a) + r'\b', query, re.IGNORECASE)
            for a in ALL_ABBRS if len(a) >= 2
        )
        if found:
            queries[query] = llm_replace_query_with_glossary(query)
    return queries

# ─── Load abbreviation CSV ─────────────────────────────────────────────────
CSV_PATH = "/app/assets/abbreviations_enhanced.csv"

def load_abbreviations():
    import csv
    forward = defaultdict(list)
    with open(CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            abbr = row["abbreviation"].strip()
            form = row["expanded_form"].strip()
            if abbr and form:
                forward[abbr].append(form)
    all_abbrs = sorted(forward.keys(), key=len, reverse=True)
    return dict(forward), all_abbrs

FORWARD_MAP, ALL_ABBRS = load_abbreviations()
print(f"Loaded {len(FORWARD_MAP)} abbreviations")

# ─── Expansion functions ───────────────────────────────────────────────────

def find_abbrs_in_text(text: str) -> Dict[str, List[str]]:
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

def expand_replace_det(text: str) -> str:
    """Deterministic replacement - picks first form (may be wrong)."""
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

def expand_query_suffix(query: str) -> str:
    result = query
    found_abbrs = set()
    for abbr in ALL_ABBRS:
        pattern = re.compile(r'\b' + re.escape(abbr) + r'\b', re.IGNORECASE)
        if pattern.search(query):
            found_abbrs.add(abbr)
    for abbr in found_abbrs:
        result += " " + " ".join(FORWARD_MAP[abbr])
    return result

# ─── LLM replacement with glossary ─────────────────────────────────────────

# LM Studio configuration — hardcoded fallbacks for when DB settings are unavailable.
_LM_STUDIO_BASE = os.environ.get("LM_STUDIO_BASE_URL", "http://192.168.1.3:2244/v1")
_LM_STUDIO_KEY = os.environ.get("LM_STUDIO_API_KEY", "dummy")
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "google/gemma-4-26b-a4b")

def get_llm_client():
    from openai import OpenAI
    from app.db.session import SessionLocal
    from app.services.settings_service import get_setting
    db = SessionLocal()
    try:
        api_key = get_setting(db, "OPENAI_API_KEY", None) or _LM_STUDIO_KEY
        api_base = get_setting(db, "OPENAI_API_BASE", None) or _LM_STUDIO_BASE
    finally:
        db.close()
    return OpenAI(api_key=api_key, base_url=api_base)

def llm_replace_with_glossary(text: str) -> str:
    """Use LLM to replace abbreviations with correct forms based on context.

    Gives the LLM a glossary of all possible meanings and asks it to pick
    the right one based on surrounding text.
    """
    found = find_abbrs_in_text(text)
    if not found:
        return text

    glossary = "\n".join(f"- {a}: {', '.join(f)}" for a, f in sorted(found.items()))

    system = (
        "You are a military abbreviation expander. You will receive a text "
        "containing military abbreviations and a glossary of all possible "
        "meanings for each abbreviation. Replace each abbreviation with the "
        "CORRECT expanded form based on the surrounding context. "
        "If an abbreviation is actually a common word (like 'to', 'in', 'on') "
        "used as a preposition, do NOT replace it. "
        "Output ONLY the rewritten text, nothing else."
    )
    user = f"Glossary:\n{glossary}\n\nText:\n{text}\n\nRewritten text:"

    client = get_llm_client()
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=1000,
                temperature=0.0,
            )
            content = resp.choices[0].message.content.strip()
            # Take the text, clean up any markdown
            content = re.sub(r'^```.*?\n', '', content, flags=re.MULTILINE)
            content = re.sub(r'\n```$', '', content)
            return content.strip()
        except Exception as exc:
            if attempt < 2:
                time.sleep(3)
                continue
            print(f"  LLM replace failed: {exc}")
            return text

def llm_replace_query_with_glossary(query: str) -> str:
    """LLM replacement for queries - same approach but optimized for short text."""
    found = find_abbrs_in_text(query)
    if not found:
        return query

    glossary = "\n".join(f"- {a}: {', '.join(f)}" for a, f in sorted(found.items()))

    system = (
        "You are a military abbreviation expander. Replace each abbreviation "
        "in the query with its correct expanded form based on context. "
        "Use the glossary to determine the correct meaning. "
        "If a word is a common preposition (to, in, on, by), do NOT replace it. "
        "Output ONLY the rewritten query, nothing else."
    )
    user = f"Glossary:\n{glossary}\n\nQuery: {query}\n\nRewritten query:"

    client = get_llm_client()
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=300,
                temperature=0.0,
            )
            content = resp.choices[0].message.content.strip()
            lines = [l.strip() for l in content.split("\n") if l.strip()]
            return lines[0] if lines else query
        except Exception as exc:
            if attempt < 2:
                time.sleep(3)
                continue
            return query

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

def embed_dense(texts: List[str]) -> List[List[float]]:
    from openai import OpenAI
    from app.db.session import SessionLocal
    from app.services.settings_service import get_setting
    db = SessionLocal()
    try:
        api_key = get_setting(db, "EMBEDDING_API_KEY", None) or "not-required"
        api_base = get_setting(db, "EMBEDDING_API_BASE", None)
        model = get_setting(db, "DENSE_EMBEDDINGS_MODEL", None)
    finally:
        db.close()
    client = OpenAI(api_key=api_key, base_url=api_base)
    embeddings = []
    for i in range(0, len(texts), 32):
        batch = texts[i:i+32]
        resp = client.embeddings.create(input=batch, model=model)
        embeddings.extend([r.embedding for r in resp.data])
    return embeddings

def rerank_scores(query: str, passages: List[str]) -> List[float]:
    from app.services.retrieval.reranker import _get_cross_encoder
    encoder = _get_cross_encoder()
    return list(encoder.rerank(query, passages))

def generate_answer(query: str, context: str, max_tokens: int = 300) -> str:
    client = get_llm_client()
    system = (
        "You are a military assistant. Answer the user's question based ONLY on "
        "the provided context. Be concise."
    )
    user = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=GENERATION_MODEL,
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

# ─── TEST 1: LLM Replacement Quality ───────────────────────────────────────

def test_llm_replacement_quality():
    print("\n" + "=" * 80)
    print("TEST 1: LLM Replacement Quality (gemma-4-12b + glossary)")
    print("=" * 80)

    print("\nReplacing abbreviations in chunks with LLM + glossary context...\n")

    llm_replaced_chunks = []
    for i, chunk in enumerate(TEST_CHUNKS):
        found = find_abbrs_in_text(chunk)
        det_replaced = expand_replace_det(chunk)
        llm_replaced = llm_replace_with_glossary(chunk)
        llm_replaced_chunks.append(llm_replaced)

        print(f"Chunk {i}: '{chunk[:70]}...'")
        print(f"  Abbreviations found: {list(found.keys())}")
        print(f"  Det replace:  {det_replaced[:120]}...")
        print(f"  LLM replace:  {llm_replaced[:120]}...")
        print()

    return llm_replaced_chunks

# ─── TEST 2: LLM Query Replacement Quality ─────────────────────────────────

def test_llm_query_replacement():
    print("=" * 80)
    print("TEST 2: LLM Query Replacement Quality")
    print("=" * 80)

    abbr_queries = [q for q in TEST_QUERIES if any(
        re.search(r'\b' + re.escape(a) + r'\b', q, re.IGNORECASE) for a in ALL_ABBRS if len(a) >= 2
    )]

    print(f"\nLLM-replacing {len(abbr_queries)} abbreviation queries...\n")

    llm_queries = {}
    for query in abbr_queries:
        llm_q = llm_replace_query_with_glossary(query)
        det_suffix = expand_query_suffix(query)
        llm_queries[query] = llm_q
        print(f"  Original:    '{query}'")
        print(f"  LLM replace: '{llm_q}'")
        print(f"  Det suffix:  '{det_suffix[:80]}...'")
        print()

    return llm_queries

# ─── TEST 3: Dense Retrieval Comparison ────────────────────────────────────

def test_dense_retrieval(llm_replaced_chunks, llm_queries):
    print("=" * 80)
    print("TEST 3: Dense Retrieval — LLM Replace vs Suffix vs Det Replace")
    print("=" * 80)

    # Embed all chunk variants
    chunk_variants = {
        "original": TEST_CHUNKS,
        "suffix": [expand_suffix(c) for c in TEST_CHUNKS],
        "det_replace": [expand_replace_det(c) for c in TEST_CHUNKS],
        "llm_replace": llm_replaced_chunks,
    }

    print("\nEmbedding chunk variants...")
    chunk_embs = {}
    for name, texts in chunk_variants.items():
        chunk_embs[name] = embed_dense(texts)
        print(f"  {name}: done")

    # Test with different query variants
    print("\nResults (top-3 per query):")
    for query in TEST_QUERIES:
        expected = EXPECTED_MATCHES[query]
        print(f"\n  Q: '{query}' → expected {expected}")

        # Original query
        q_emb_orig = embed_dense([query])[0]
        # Suffix query
        q_suffix = expand_query_suffix(query)
        q_emb_suffix = embed_dense([q_suffix])[0]
        # LLM query (if available)
        q_llm = llm_queries.get(query)
        q_emb_llm = embed_dense([q_llm])[0] if q_llm else None

        query_variants = [("q=orig", q_emb_orig), ("q=suffix", q_emb_suffix)]
        if q_emb_llm:
            query_variants.append(("q=llm_rep", q_emb_llm))

        for q_name, qe in query_variants:
            for ing_name in chunk_variants:
                sims = sorted(
                    enumerate(cosine_similarity(qe, ce) for ce in chunk_embs[ing_name]),
                    key=lambda x: x[1], reverse=True
                )
                top2 = [s[0] for s in sims[:2]]
                hit = any(c in top2 for c in expected)
                top3_str = " ".join(f"c{s[0]}:{s[1]:.3f}" for s in sims[:3])
                marker = "✓" if hit else "✗"
                print(f"    {q_name:12s} ing={ing_name:12s}  {top3_str}  {marker}")

# ─── TEST 4: Reranker Comparison ───────────────────────────────────────────

def test_reranker(llm_replaced_chunks, llm_queries):
    print("\n" + "=" * 80)
    print("TEST 4: Reranker — LLM Replace vs Suffix vs Original")
    print("=" * 80)

    chunk_variants = {
        "orig_c": TEST_CHUNKS,
        "suffix_c": [expand_suffix(c) for c in TEST_CHUNKS],
        "det_rep_c": [expand_replace_det(c) for c in TEST_CHUNKS],
        "llm_rep_c": llm_replaced_chunks,
    }

    print(f"\nScoring reranker combos for {len(TEST_QUERIES)} queries...")

    combos = [
        ("orig_q", "orig_c",     lambda q: q,                  "orig_c"),
        ("orig_q", "suffix_c",   lambda q: q,                  "suffix_c"),
        ("orig_q", "det_rep_c",  lambda q: q,                  "det_rep_c"),
        ("orig_q", "llm_rep_c",  lambda q: q,                  "llm_rep_c"),
        ("suffix_q", "orig_c",   expand_query_suffix,          "orig_c"),
        ("suffix_q", "llm_rep_c", expand_query_suffix,         "llm_rep_c"),
    ]
    # Add LLM query combos
    for query in TEST_QUERIES:
        if query in llm_queries:
            combos.append(("llm_q", "orig_c", lambda q, _q=query: llm_queries[_q], "orig_c"))
            combos.append(("llm_q", "llm_rep_c", lambda q, _q=query: llm_queries[_q], "llm_rep_c"))
            break  # Just need the function reference

    # Actually, let's do it properly - test all combos
    results = {}
    for q_idx, query in enumerate(TEST_QUERIES):
        expected = EXPECTED_MATCHES[query]
        q_variants = {
            "orig_q": query,
            "suffix_q": expand_query_suffix(query),
        }
        if query in llm_queries:
            q_variants["llm_q"] = llm_queries[query]

        for q_name, q_text in q_variants.items():
            for c_name, c_texts in chunk_variants.items():
                passages = c_texts
                scores = rerank_scores(q_text, passages)
                scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
                top2 = [s[0] for s in scored[:2]]
                hit = any(c in top2 for c in expected)
                key = f"{q_name}+{c_name}"
                if key not in results:
                    results[key] = {"hits": 0, "total": 0, "details": []}
                results[key]["hits"] += 1 if hit else 0
                results[key]["total"] += 1
                results[key]["details"].append((q_idx, scored[0][0], scored[0][1], hit, expected))

    print("\n  Hit rates (top-2):")
    for key in sorted(results.keys()):
        r = results[key]
        pct = r["hits"] / r["total"] * 100 if r["total"] else 0
        print(f"    {key:30s}  {r['hits']}/{r['total']} ({pct:.0f}%)")

    print("\n  Detailed (top-1 per query) for key combos:")
    key_combos = ["orig_q+orig_c", "orig_q+suffix_c", "orig_q+llm_rep_c",
                  "suffix_q+orig_c", "suffix_q+llm_rep_c", "llm_q+orig_c", "llm_q+llm_rep_c"]
    for q_idx, query in enumerate(TEST_QUERIES):
        print(f"\n  Q{q_idx}: '{query}' → {EXPECTED_MATCHES[query]}")
        for key in key_combos:
            if key in results:
                detail = [d for d in results[key]["details"] if d[0] == q_idx]
                if detail:
                    _, top1, score, hit, expected = detail[0]
                    marker = "✓" if hit else "✗"
                    print(f"    {key:30s}  c{top1}({score:.3f}) {marker}")

# ─── TEST 5: Generation Quality ────────────────────────────────────────────

def test_generation(llm_replaced_chunks):
    print("\n" + "=" * 80)
    print("TEST 5: Generation Quality — LLM Replace vs Suffix vs Glossary")
    print("=" * 80)

    test_cases = [
        (0, "battalions withdrew from position",
         "Does the context describe battalions withdrawing?"),
        (3, "who approved the medical resupply?",
         "Who approved the resupply? (expect: Deputy Assistant / DA)"),
        (0, "bns wdr from position",
         "Does the context describe battalions withdrawing? (abbr query)"),
    ]

    for chunk_idx, query, description in test_cases:
        chunk = TEST_CHUNKS[chunk_idx]
        llm_chunk = llm_replaced_chunks[chunk_idx]
        suffix_chunk = expand_suffix(chunk)
        glossary = build_glossary(chunk)

        print(f"\n  Query: '{query}'")
        print(f"  Chunk {chunk_idx}: '{chunk[:60]}...'")
        print(f"  Expected: {description}")

        context_opts = {
            "original": chunk,
            "llm_replace": llm_chunk,
            "suffix": suffix_chunk,
            "glossary": f"{chunk}\n[Abbreviation Glossary]\n{glossary}" if glossary else chunk,
            "llm_replace+glossary": f"{llm_chunk}\n[Abbreviation Glossary]\n{glossary}" if glossary else llm_chunk,
        }

        for ctx_name, ctx_text in context_opts.items():
            answer = generate_answer(query, ctx_text)
            print(f"\n  ctx={ctx_name} ({len(ctx_text)} chars):")
            print(f"    {answer[:200]}")

# ─── TEST 6: Full Pipeline ─────────────────────────────────────────────────

def test_full_pipeline(llm_replaced_chunks, llm_queries):
    print("\n" + "=" * 80)
    print("TEST 6: Full Pipeline — LLM Replace vs Suffix")
    print("=" * 80)

    test_query = "battalions withdrew from position"
    expected = EXPECTED_MATCHES[test_query]

    pipelines = [
        ("suffix: ing=suffix q=suffix rr=orig+suffix gen=glossary",
         "suffix", expand_query_suffix(test_query), test_query,
         [expand_suffix(c) for c in TEST_CHUNKS], "glossary"),
        ("llm: ing=llm_rep q=orig rr=orig+llm_rep gen=llm_rep",
         "llm_replace", test_query, test_query,
         llm_replaced_chunks, "llm_replace"),
        ("llm: ing=llm_rep q=llm_q rr=llm_q+llm_rep gen=llm_rep",
         "llm_replace", llm_queries.get(test_query, test_query), llm_queries.get(test_query, test_query),
         llm_replaced_chunks, "llm_replace"),
        ("llm: ing=llm_rep q=suffix rr=orig+llm_rep gen=llm_rep+glossary",
         "llm_replace", expand_query_suffix(test_query), test_query,
         llm_replaced_chunks, "llm_replace+glossary"),
        ("hybrid: ing=suffix q=suffix rr=orig+llm_rep gen=glossary",
         "suffix", expand_query_suffix(test_query), test_query,
         llm_replaced_chunks, "glossary"),
    ]

    print(f"\n  Query: '{test_query}' → expected {expected}")

    for label, ing_name, dense_q, rr_q, chunk_texts, gen_ctx_name in pipelines:
        # Dense retrieval
        chunk_embs = embed_dense(chunk_texts)
        q_emb = embed_dense([dense_q])[0]
        sims = sorted(enumerate(cosine_similarity(q_emb, ce) for ce in chunk_embs), key=lambda x: x[1], reverse=True)
        top3_idx = [s[0] for s in sims[:3]]

        # Rerank
        rr_passages = [chunk_texts[i] for i in top3_idx]
        rr_scores = rerank_scores(rr_q, rr_passages)
        rr_ranked = sorted(zip(top3_idx, rr_scores), key=lambda x: x[1], reverse=True)
        rr_top1 = rr_ranked[0][0]
        rr_hit = rr_top1 in expected

        # Generate
        top_chunk = TEST_CHUNKS[rr_top1]
        if gen_ctx_name == "glossary":
            glossary = build_glossary(top_chunk)
            gen_context = f"{top_chunk}\n[Abbreviation Glossary]\n{glossary}" if glossary else top_chunk
        elif gen_ctx_name == "llm_replace":
            gen_context = llm_replaced_chunks[rr_top1]
        elif gen_ctx_name == "llm_replace+glossary":
            glossary = build_glossary(top_chunk)
            gen_context = f"{llm_replaced_chunks[rr_top1]}\n[Abbreviation Glossary]\n{glossary}" if glossary else llm_replaced_chunks[rr_top1]
        else:
            gen_context = top_chunk

        answer = generate_answer(test_query, gen_context)
        answer_correct = any(w in answer.lower() for w in ["battalion", "withdraw", "order", "position"])

        print(f"\n  {label}")
        print(f"    dense_top3={top3_idx} rerank_top1=c{rr_top1} hit={rr_hit} correct={answer_correct}")
        print(f"    answer: {answer[:150]}")

# ─── MAIN ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print(f"LLM REPLACEMENT WITH GLOSSARY TEST SUITE")
    print(f"Model: {GENERATION_MODEL}")
    print(f"Abbreviations: {len(FORWARD_MAP)}, chunks: {len(TEST_CHUNKS)}, queries: {len(TEST_QUERIES)}")
    print("=" * 80)

    llm_chunks = test_llm_replacement_quality()
    llm_queries = test_llm_query_replacement()
    test_dense_retrieval(llm_chunks, llm_queries)
    test_reranker(llm_chunks, llm_queries)
    test_generation(llm_chunks)
    test_full_pipeline(llm_chunks, llm_queries)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)

if __name__ == "__main__":
    main()
