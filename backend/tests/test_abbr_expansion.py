#!/usr/bin/env python3
"""
Comprehensive abbreviation expansion test suite.

Tests all expansion options for each pipeline stage using REAL models:
  - Dense embeddings: qwen/qwen3-embedding-0.6b (LM Studio, dim=1024)
  - Sparse embeddings: SPLADE PP en v1 (FastEmbed, local ONNX)
  - Reranker: ms-marco-MiniLM-L-12-v2 (FastEmbed, local ONNX)
  - Generation: qwen/qwen3.5-9b (LM Studio)

Test matrix:
  INGESTION:  none | suffix | replace | glossary_suffix
  RETRIEVAL:  original_query | expanded_query_suffix | expanded_query_replace | llm_expanded
  RERANKER:   orig_q+orig_chunk | exp_q+orig_chunk | orig_q+glossary_chunk | orig_q+suffix_chunk | exp_q+suffix_chunk
  GENERATION: orig_context | glossary_context | suffix_context | llm_glossary_context

Runs inside the backend container. No app code changes — uses app's model
singletons and DB session directly.
"""
import json
import os
import re
import sys
import time
import hashlib
import logging
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# ─── Setup paths ───────────────────────────────────────────────────────────
sys.path.insert(0, "/app")
os.environ.setdefault("PYTHONPATH", "/app")

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("abbr_test")

# ─── Load abbreviation CSV ─────────────────────────────────────────────────
CSV_PATH = "/app/assets/abbreviations_enhanced.csv"

def load_abbreviations():
    """Load the abbreviation CSV into forward and reverse maps."""
    import csv
    forward = defaultdict(list)  # abbr -> [expanded forms]
    reverse = defaultdict(list)  # expanded_form_lower -> [abbr]
    with open(CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            abbr = row["abbreviation"].strip()
            form = row["expanded_form"].strip()
            if abbr and form:
                forward[abbr].append(form)
                reverse[form.lower()].append(abbr)
    # Sort abbreviations by length (longest first) for matching
    all_abbrs = sorted(forward.keys(), key=len, reverse=True)
    return dict(forward), dict(reverse), all_abbrs

FORWARD_MAP, REVERSE_MAP, ALL_ABBRS = load_abbreviations()
print(f"Loaded {len(FORWARD_MAP)} abbreviations, {sum(len(v) for v in FORWARD_MAP.values())} total forms")

# ─── Expansion functions ───────────────────────────────────────────────────

def expand_suffix(text: str) -> str:
    """Suffix mode: append [Expansions: abbr=form1 form2; ...] at end."""
    found = {}
    for abbr in ALL_ABBRS:
        pattern = re.compile(r'\b' + re.escape(abbr) + r'\b', re.IGNORECASE)
        if pattern.search(text):
            forms = FORWARD_MAP[abbr]
            found[abbr] = " ".join(forms)
    if not found:
        return text
    parts = [f"{a}={f}" for a, f in found.items()]
    return f"{text} [Expansions: {'; '.join(parts)}]"

def expand_replace(text: str) -> str:
    """Replace mode: replace abbreviation with all expanded forms."""
    result = text
    for abbr in ALL_ABBRS:
        forms = FORWARD_MAP[abbr]
        pattern = re.compile(r'\b' + re.escape(abbr) + r'\b', re.IGNORECASE)
        result = pattern.sub(" ".join(forms), result)
    return result

def build_glossary(text: str) -> str:
    """Build a compact glossary for abbreviations found in text."""
    found = {}
    for abbr in ALL_ABBRS:
        pattern = re.compile(r'\b' + re.escape(abbr) + r'\b', re.IGNORECASE)
        if pattern.search(text):
            forms = FORWARD_MAP[abbr]
            found[abbr] = ", ".join(forms)
    if not found:
        return ""
    return "\n".join(f"{a} = {f}" for a, f in sorted(found.items(), key=lambda x: x[0].lower()))

def expand_glossary_suffix(text: str) -> str:
    """Glossary suffix: append [Abbreviation Glossary]\n... at end."""
    glossary = build_glossary(text)
    if not glossary:
        return text
    return f"{text}\n[Abbreviation Glossary]\n{glossary}"

def expand_query_suffix(query: str) -> str:
    """Bidirectional query expansion in suffix mode."""
    result = query
    # Forward: find abbreviations in query, append all forms
    found_abbrs = set()
    for abbr in ALL_ABBRS:
        pattern = re.compile(r'\b' + re.escape(abbr) + r'\b', re.IGNORECASE)
        if pattern.search(query):
            found_abbrs.add(abbr)
    for abbr in found_abbrs:
        forms = FORWARD_MAP[abbr]
        result += " " + " ".join(forms)
    # Reverse: find full forms in query, append abbreviations
    query_lower = query.lower()
    for form_lower, abbrs in REVERSE_MAP.items():
        pattern = re.compile(r'\b' + re.escape(form_lower) + r'\b', re.IGNORECASE)
        if pattern.search(query_lower):
            for abbr in abbrs:
                if abbr not in found_abbrs:
                    result += " " + abbr
    return result

def expand_query_replace(query: str) -> str:
    """Replace abbreviations in query with their expanded forms."""
    return expand_replace(query)

# ─── Test documents ────────────────────────────────────────────────────────

# Military text with abbreviations that are common and obscure
TEST_CHUNKS = [
    # Chunk 1: Common abbreviations (CO, MO, HQ) + obscure (wdr, bns, adjt)
    "The CO ordered the bns to wdr from the forward position. The MO reported "
    "casualties. The adjt coordinated with HQ. The op was conducted at first light. "
    "The recce team provided intelligence on enemy positions.",

    # Chunk 2: More military abbreviations
    "The GOC visited the bde HQ and briefed the bde comd on the op. The inf bn was "
    "tasked to secure the obj. The armd sqn was to provide spt. The arty bty was "
    "placed in sp of the inf.",

    # Chunk 3: Non-military chunk (should not match military abbreviations)
    "The weather forecast indicates rain for the next three days. Temperature "
    "will drop to 15 degrees Celsius. Farmers should prepare for wet conditions.",

    # Chunk 4: Abbreviation-heavy chunk with multiple meanings
    "The DA approved the medical resupply. The SP was established at checkpoint 4. "
    "The cas were evacuated to the Fd Amb. The spt elements moved up at 0600.",

    # Chunk 5: Mixed - some abbreviations, some full forms
    "The commanding officer ordered the battalions to withdraw from the forward "
    "position. The medical officer reported casualties. The operation was "
    "conducted at first light by the reconnaissance team.",
]

# Test queries — mix of abbreviation queries and full-form queries
TEST_QUERIES = [
    # Query 1: Full form query (should match abbreviation chunks)
    "battalions withdrew from position",
    # Query 2: Abbreviation query (should match full-form chunks)
    "bns wdr from position",
    # Query 3: Mixed query
    "CO ordered bns to wdr",
    # Query 4: Full form that should match chunk 2
    "brigade headquarters operation objective",
    # Query 5: Should match chunk 4 (DA has multiple meanings)
    "deputy assistant approved resupply",
    # Query 6: Irrelevant query (should not match military chunks)
    "weather forecast rain temperature",
]

# Expected chunk matches for each query (0-indexed)
EXPECTED_MATCHES = {
    "battalions withdrew from position": [0, 4],      # chunks with wdr/bns or full forms
    "bns wdr from position": [0, 4],                  # same chunks, reverse direction
    "CO ordered bns to wdr": [0, 4],                  # chunk 0 has abbreviations, chunk 4 has full forms
    "brigade headquarters operation objective": [1],  # chunk 1
    "deputy assistant approved resupply": [3],        # chunk 4 (DA = Deputy Assistant)
    "weather forecast rain temperature": [2],         # chunk 2 (non-military)
}

# ─── Model accessors ───────────────────────────────────────────────────────

def get_dense_embedder():
    """Get the OpenAI-compatible dense embedding client."""
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
    return OpenAI(api_key=api_key, base_url=api_base), model

def get_sparse_embedder():
    """Get the SPLADE sparse embedder."""
    from app.services.infrastructure import get_sparse_embedder
    return get_sparse_embedder()

def get_reranker():
    """Get the cross-encoder reranker."""
    from app.services.retrieval.reranker import _get_cross_encoder
    return _get_cross_encoder()

def get_generation_client():
    """Get the OpenAI-compatible generation client."""
    from openai import OpenAI
    from app.db.session import SessionLocal
    from app.services.settings_service import get_setting
    db = SessionLocal()
    try:
        api_key = get_setting(db, "OPENAI_API_KEY", None) or "not-required"
        api_base = get_setting(db, "OPENAI_API_BASE", None)
        model = get_setting(db, "OPENAI_MODEL", None)
    finally:
        db.close()
    return OpenAI(api_key=api_key, base_url=api_base), model

# ─── Embedding functions ───────────────────────────────────────────────────

def embed_dense(texts: List[str]) -> List[List[float]]:
    """Embed texts with the dense model."""
    client, model = get_dense_embedder()
    embeddings = []
    for i in range(0, len(texts), 32):
        batch = texts[i:i+32]
        resp = client.embeddings.create(input=batch, model=model)
        embeddings.extend([r.embedding for r in resp.data])
    return embeddings

def embed_sparse(texts: List[str]) -> List:
    """Embed texts with SPLADE."""
    embedder = get_sparse_embedder()
    return list(embedder.embed(texts))

def rerank_scores(query: str, passages: List[str]) -> List[float]:
    """Score query-passage pairs with the cross-encoder."""
    encoder = get_reranker()
    return list(encoder.rerank(query, passages))

def strip_thinking(text: str) -> str:
    """Strip Qwen3 thinking tags and numbered-list thinking from model output."""
    # Strip thinking blocks
    text = re.sub(r"\u2728.*?\u2728", "", text, flags=re.DOTALL).strip()
    # If the model outputs thinking without tags (numbered list format),
    # find the last non-numbered, non-bullet line as the answer
    lines = text.split("\n")
    answer_lines = []
    in_thinking = True
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if not in_thinking:
                answer_lines.append(line)
            continue
        if in_thinking:
            if stripped.startswith("*") or stripped.startswith("#"):
                continue
            if re.match(r"^\d+\.", stripped):
                continue
            if stripped.startswith("**") and stripped.endswith("**"):
                continue
            in_thinking = False
        answer_lines.append(line)
    result = "\n".join(answer_lines).strip()
    return result if result else text.strip()


def generate_answer(query: str, context: str, max_tokens: int = 500) -> str:
    """Generate an answer using the LLM."""
    client, model = get_generation_client()
    system = (
        "You are a military assistant. Answer the user's question based ONLY on "
        "the provided context. If the context doesn't contain the answer, say "
        "'I cannot answer based on the provided context.' Be concise. "
        "Do not show your thinking process."
    )
    user = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=0.1,
    )
    content = resp.choices[0].message.content
    return strip_thinking(content)


# ─── Similarity functions ──────────────────────────────────────────────────

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def sparse_dot_product(a, b) -> float:
    """Compute dot product between two sparse vectors."""
    # SPLADE returns SparseEmbedding with .indices and .values
    a_dict = dict(zip(a.indices.tolist(), a.values.tolist()))
    b_dict = dict(zip(b.indices.tolist(), b.values.tolist()))
    return sum(a_dict.get(k, 0) * b_dict.get(k, 0) for k in a_dict)

# ─── Test runner ───────────────────────────────────────────────────────────

class TestResults:
    def __init__(self):
        self.results = []

    def add(self, category: str, test_name: str, metric: str, value, extra: dict = None):
        self.results.append({
            "category": category,
            "test": test_name,
            "metric": metric,
            "value": value,
            "extra": extra or {},
        })

    def print_summary(self):
        print("\n" + "=" * 80)
        print("COMPREHENSIVE TEST RESULTS SUMMARY")
        print("=" * 80)
        current_cat = ""
        for r in self.results:
            if r["category"] != current_cat:
                current_cat = r["category"]
                print(f"\n── {current_cat} ──")
            val = r["value"]
            if isinstance(val, float):
                val = f"{val:.4f}"
            extra_str = ""
            if r["extra"]:
                extra_str = " | " + " ".join(f"{k}={v}" for k, v in r["extra"].items() if not isinstance(v, (list, dict)))
            print(f"  {r['test']:40s} {r['metric']:25s} = {val}{extra_str}")

# Skip tests that already completed successfully in prior runs
SKIP_DENSE = os.environ.get('SKIP_DENSE', '0') == '1'
SKIP_SPARSE = os.environ.get('SKIP_SPARSE', '0') == '1'
SKIP_RERANKER = os.environ.get('SKIP_RERANKER', '0') == '1'

results = TestResults()

# ─── TEST 1: Dense Embedding Similarity ────────────────────────────────────
def test_dense_embeddings():
    """Test how different ingestion/query expansion options affect dense cosine similarity."""
    if SKIP_DENSE:
        print("\nSKIP: Dense embeddings (SKIP_DENSE=1)")
        return
    print("\n" + "─" * 80)
    print("TEST 1: Dense Embedding Similarity (qwen3-embedding-0.6b)")
    print("─" * 80)

    # Ingestion expansion options
    ingestion_options = {
        "none": lambda t: t,
        "suffix": expand_suffix,
        "replace": expand_replace,
        "glossary_suffix": expand_glossary_suffix,
    }

    # Query expansion options
    query_options = {
        "original": lambda q: q,
        "suffix": expand_query_suffix,
        "replace": expand_query_replace,
    }

    # Embed all chunk variants
    print("\nEmbedding chunk variants...")
    chunk_variants = {}  # option_name -> [embedded chunks]
    for ing_name, ing_fn in ingestion_options.items():
        chunk_texts = [ing_fn(c) for c in TEST_CHUNKS]
        chunk_embs = embed_dense(chunk_texts)
        chunk_variants[ing_name] = list(zip(chunk_texts, chunk_embs))
        print(f"  {ing_name}: {len(chunk_texts)} chunks embedded")

    # Embed all query variants
    print("Embedding query variants...")
    query_variants = {}  # option_name -> [(query_text, query_emb), ...]
    for q_name, q_fn in query_options.items():
        query_texts = [q_fn(q) for q in TEST_QUERIES]
        query_embs = embed_dense(query_texts)
        query_variants[q_name] = list(zip(query_texts, query_embs))
        print(f"  {q_name}: {len(query_texts)} queries embedded")

    # Compute cosine similarity for all combinations
    print("\nComputing cosine similarities...")
    for ing_name in ingestion_options:
        for q_name in query_options:
            for q_idx, (q_text, q_emb) in enumerate(query_variants[q_name]):
                sims = []
                for c_idx, (c_text, c_emb) in enumerate(chunk_variants[ing_name]):
                    sim = cosine_similarity(q_emb, c_emb)
                    sims.append((c_idx, sim))

                # Sort by similarity (descending)
                sims.sort(key=lambda x: x[1], reverse=True)
                top_match = sims[0]
                expected = EXPECTED_MATCHES.get(TEST_QUERIES[q_idx], [])

                # Check if any expected chunk is in top-2
                top2_chunks = [s[0] for s in sims[:2]]
                hit = any(c in top2_chunks for c in expected)

                results.add(
                    "DENSE",
                    f"ing={ing_name} q={q_name}",
                    f"query_{q_idx}_top1",
                    f"chunk_{top_match[0]}({top_match[1]:.3f})",
                    {"expected": expected, "hit": hit, "query": TEST_QUERIES[q_idx][:30]},
                )

    # Print a detailed table for the most interesting combinations
    print("\nDetailed results (top-3 matches per query):")
    for q_idx, query in enumerate(TEST_QUERIES):
        print(f"\n  Query {q_idx}: '{query}'")
        print(f"  Expected chunks: {EXPECTED_MATCHES.get(query, [])}")
        for ing_name in ingestion_options:
            for q_name in query_options:
                q_text, q_emb = query_variants[q_name][q_idx]
                sims = []
                for c_idx, (c_text, c_emb) in enumerate(chunk_variants[ing_name]):
                    sim = cosine_similarity(q_emb, c_emb)
                    sims.append((c_idx, sim))
                sims.sort(key=lambda x: x[1], reverse=True)
                top3 = " ".join(f"c{s[0]}:{s[1]:.3f}" for s in sims[:3])
                print(f"    ing={ing_name:15s} q={q_name:10s} → {top3}")

# ─── TEST 2: Sparse (SPLADE) Similarity ────────────────────────────────────
def test_sparse_embeddings():
    """Test how different expansion options affect SPLADE sparse similarity."""
    if SKIP_SPARSE:
        print("\nSKIP: Sparse embeddings (SKIP_SPARSE=1)")
        return
    print("\n" + "─" * 80)
    print("TEST 2: Sparse (SPLADE) Similarity (Splade_PP_en_v1)")
    print("─" * 80)

    ingestion_options = {
        "none": lambda t: t,
        "suffix": expand_suffix,
        "replace": expand_replace,
        "glossary_suffix": expand_glossary_suffix,
    }

    query_options = {
        "original": lambda q: q,
        "suffix": expand_query_suffix,
        "replace": expand_query_replace,
    }

    # Embed all chunk variants
    print("\nEmbedding chunk variants with SPLADE...")
    chunk_variants = {}
    for ing_name, ing_fn in ingestion_options.items():
        chunk_texts = [ing_fn(c) for c in TEST_CHUNKS]
        chunk_embs = embed_sparse(chunk_texts)
        chunk_variants[ing_name] = list(zip(chunk_texts, chunk_embs))
        print(f"  {ing_name}: done")

    # Embed all query variants
    print("Embedding query variants with SPLADE...")
    query_variants = {}
    for q_name, q_fn in query_options.items():
        query_texts = [q_fn(q) for q in TEST_QUERIES]
        query_embs = embed_sparse(query_texts)
        query_variants[q_name] = list(zip(query_texts, query_embs))
        print(f"  {q_name}: done")

    # Compute sparse dot product for all combinations
    print("\nComputing sparse similarities...")
    for ing_name in ingestion_options:
        for q_name in query_options:
            for q_idx, (q_text, q_emb) in enumerate(query_variants[q_name]):
                sims = []
                for c_idx, (c_text, c_emb) in enumerate(chunk_variants[ing_name]):
                    sim = sparse_dot_product(q_emb, c_emb)
                    sims.append((c_idx, sim))

                sims.sort(key=lambda x: x[1], reverse=True)
                top_match = sims[0]
                expected = EXPECTED_MATCHES.get(TEST_QUERIES[q_idx], [])
                top2_chunks = [s[0] for s in sims[:2]]
                hit = any(c in top2_chunks for c in expected)

                results.add(
                    "SPARSE",
                    f"ing={ing_name} q={q_name}",
                    f"query_{q_idx}_top1",
                    f"chunk_{top_match[0]}({top_match[1]:.1f})",
                    {"expected": expected, "hit": hit, "query": TEST_QUERIES[q_idx][:30]},
                )

    # Print detailed table
    print("\nDetailed results (top-3 matches per query):")
    for q_idx, query in enumerate(TEST_QUERIES):
        print(f"\n  Query {q_idx}: '{query}'")
        print(f"  Expected chunks: {EXPECTED_MATCHES.get(query, [])}")
        for ing_name in ingestion_options:
            for q_name in query_options:
                q_text, q_emb = query_variants[q_name][q_idx]
                sims = []
                for c_idx, (c_text, c_emb) in enumerate(chunk_variants[ing_name]):
                    sim = sparse_dot_product(q_emb, c_emb)
                    sims.append((c_idx, sim))
                sims.sort(key=lambda x: x[1], reverse=True)
                top3 = " ".join(f"c{s[0]}:{s[1]:.1f}" for s in sims[:3])
                print(f"    ing={ing_name:15s} q={q_name:10s} → {top3}")

# ─── TEST 3: Reranker (Cross-Encoder) ──────────────────────────────────────
def test_reranker():
    """Test how different expansion options affect cross-encoder reranker scores."""
    if SKIP_RERANKER:
        print("\nSKIP: Reranker (SKIP_RERANKER=1)")
        return
    print("\n" + "─" * 80)
    print("TEST 3: Reranker (ms-marco-MiniLM-L-12-v2)")
    print("─" * 80)

    # Reranker options: (query_variant, chunk_variant)
    # We test all meaningful combinations
    reranker_combos = [
        ("orig_q", "orig_chunk",     lambda q: q,                 lambda t: t),
        ("orig_q", "suffix_chunk",   lambda q: q,                 expand_suffix),
        ("orig_q", "replace_chunk",  lambda q: q,                 expand_replace),
        ("orig_q", "glossary_chunk", lambda q: q,                 expand_glossary_suffix),
        ("exp_q",  "orig_chunk",     expand_query_suffix,         lambda t: t),
        ("exp_q",  "suffix_chunk",   expand_query_suffix,         expand_suffix),
        ("exp_q",  "glossary_chunk", expand_query_suffix,         expand_glossary_suffix),
        ("rep_q",  "orig_chunk",     expand_query_replace,        lambda t: t),
        ("rep_q",  "glossary_chunk", expand_query_replace,        expand_glossary_suffix),
    ]

    print("\nScoring all query-chunk combinations with cross-encoder...")
    for q_label, c_label, q_fn, c_fn in reranker_combos:
        for q_idx, query in enumerate(TEST_QUERIES):
            q_text = q_fn(query)
            passages = [c_fn(c) for c in TEST_CHUNKS]
            scores = rerank_scores(q_text, passages)

            scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            top_match = scored[0]
            expected = EXPECTED_MATCHES.get(query, [])
            top2_chunks = [s[0] for s in scored[:2]]
            hit = any(c in top2_chunks for c in expected)

            results.add(
                "RERANKER",
                f"q={q_label} c={c_label}",
                f"query_{q_idx}_top1",
                f"chunk_{top_match[0]}({top_match[1]:.3f})",
                {"expected": expected, "hit": hit, "query": query[:30]},
            )

    # Print detailed table
    print("\nDetailed results (top-3 matches per query):")
    for q_idx, query in enumerate(TEST_QUERIES):
        print(f"\n  Query {q_idx}: '{query}'")
        print(f"  Expected chunks: {EXPECTED_MATCHES.get(query, [])}")
        for q_label, c_label, q_fn, c_fn in reranker_combos:
            q_text = q_fn(query)
            passages = [c_fn(c) for c in TEST_CHUNKS]
            scores = rerank_scores(q_text, passages)
            scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            top3 = " ".join(f"c{s[0]}:{s[1]:.3f}" for s in scored[:3])
            print(f"    q={q_label:8s} c={c_label:15s} → {top3}")

# ─── TEST 4: Generation Quality ────────────────────────────────────────────
def test_generation():
    """Test how different context formats affect LLM answer quality."""
    print("\n" + "─" * 80)
    print("TEST 4: Generation Quality (qwen3.5-9b)")
    print("─" * 80)

    # We test generation with a query that requires abbreviation understanding
    # Use chunk 0 (has abbreviations) and query "battalions withdrew from position"
    test_chunk = TEST_CHUNKS[0]
    test_query = "battalions withdrew from position"

    context_options = {
        "original_only": test_chunk,
        "with_glossary": expand_glossary_suffix(test_chunk),
        "with_suffix": expand_suffix(test_chunk),
        "with_replace": expand_replace(test_chunk),
    }

    print(f"\nQuery: '{test_query}'")
    print(f"Chunk: '{test_chunk[:80]}...'")

    for ctx_name, ctx_text in context_options.items():
        print(f"\n  Context option: {ctx_name}")
        print(f"  Context length: {len(ctx_text)} chars")
        answer = generate_answer(test_query, ctx_text, max_tokens=300)
        print(f"  Answer: {answer[:200]}")

        # Check if the answer correctly identifies that battalions withdrew
        correct = any(word in answer.lower() for word in ["battalion", "withdraw", "order", "position"])
        results.add(
            "GENERATION",
            f"ctx={ctx_name}",
            "answer_correct",
            correct,
            {"answer_preview": answer[:100].replace("\n", " ")},
        )

    # Also test with a multi-meaning abbreviation (DA)
    test_chunk2 = TEST_CHUNKS[3]
    test_query2 = "who approved the medical resupply?"

    context_options2 = {
        "original_only": test_chunk2,
        "with_glossary": expand_glossary_suffix(test_chunk2),
        "with_suffix": expand_suffix(test_chunk2),
    }

    print(f"\n\nQuery: '{test_query2}'")
    print(f"Chunk: '{test_chunk2[:80]}...'")

    for ctx_name, ctx_text in context_options2.items():
        print(f"\n  Context option: {ctx_name}")
        answer = generate_answer(test_query2, ctx_text, max_tokens=300)
        print(f"  Answer: {answer[:200]}")

        # Check if the answer correctly identifies DA as Deputy Assistant or Defence Attache
        correct = any(word in answer.lower() for word in ["deputy", "assistant", "attache", "da"])
        results.add(
            "GENERATION",
            f"multi_meaning_ctx={ctx_name}",
            "answer_correct",
            correct,
            {"answer_preview": answer[:100].replace("\n", " ")},
        )

# ─── TEST 5: LLM-Based Query Expansion ─────────────────────────────────────
def test_llm_query_expansion():
    """Test using the LLM to expand abbreviations in queries."""
    print("\n" + "─" * 80)
    print("TEST 5: LLM-Based Query Expansion (qwen3.5-9b)")
    print("─" * 80)

    client, model = get_generation_client()

    def llm_expand(query: str) -> str:
        """Use the LLM to expand abbreviations in a query."""
        system = (
            "You are a military abbreviation expander. Given a query that may contain "
            "military abbreviations, expand each abbreviation to its full form. "
            "Keep the original words. Append expanded forms at the end. "
            "Output ONLY the expanded query, nothing else.\n\n"
            "Example:\n"
            "Input: bns wdr from position\n"
            "Output: bns wdr from position battalions withdraw from position"
        )
        user = f"Input: {query}\nOutput:"
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=500,
            temperature=0.0,
        )
        content = resp.choices[0].message.content
        content = strip_thinking(content)
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        # Take the last line (after any thinking)
        lines = [l.strip() for l in content.split("\n") if l.strip() and not l.strip().startswith("*") and not l.strip().startswith("#")]
        if lines:
            return lines[-1]
        return content

    # Test LLM expansion on abbreviation queries
    abbr_queries = [q for q in TEST_QUERIES if any(
        re.search(r'\b' + re.escape(a) + r'\b', q, re.IGNORECASE) for a in ALL_ABBRS if len(a) >= 2
    )]

    print(f"\nTesting LLM expansion on {len(abbr_queries)} queries...")
    llm_expanded = {}
    for query in abbr_queries:
        expanded = llm_expand(query)
        llm_expanded[query] = expanded
        print(f"  '{query}' → '{expanded}'")
        results.add(
            "LLM_EXPANSION",
            f"query_expand",
            query[:30],
            expanded[:80],
            {"original": query, "expanded": expanded},
        )

    # Compare LLM expansion vs deterministic expansion in dense retrieval
    if llm_expanded:
        print("\nComparing LLM vs deterministic expansion in dense retrieval...")
        # Use original chunks (no ingestion expansion)
        chunk_embs = embed_dense(TEST_CHUNKS)

        for query, llm_exp in llm_expanded.items():
            # LLM expanded query
            llm_emb = embed_dense([llm_exp])[0]
            llm_sims = sorted(
                enumerate(cosine_similarity(llm_emb, ce) for ce in chunk_embs),
                key=lambda x: x[1], reverse=True
            )

            # Deterministic suffix expansion
            det_exp = expand_query_suffix(query)
            det_emb = embed_dense([det_exp])[0]
            det_sims = sorted(
                enumerate(cosine_similarity(det_emb, ce) for ce in chunk_embs),
                key=lambda x: x[1], reverse=True
            )

            # Original query
            orig_emb = embed_dense([query])[0]
            orig_sims = sorted(
                enumerate(cosine_similarity(orig_emb, ce) for ce in chunk_embs),
                key=lambda x: x[1], reverse=True
            )

            expected = EXPECTED_MATCHES.get(query, [])
            print(f"\n  Query: '{query}'")
            print(f"  LLM expanded: '{llm_exp}'")
            print(f"  Det expanded: '{det_exp[:80]}...'")
            print(f"  Expected chunks: {expected}")
            print(f"  Original:  {' '.join(f'c{s[0]}:{s[1]:.3f}' for s in orig_sims[:3])}")
            print(f"  LLM exp:   {' '.join(f'c{s[0]}:{s[1]:.3f}' for s in llm_sims[:3])}")
            print(f"  Det exp:   {' '.join(f'c{s[0]}:{s[1]:.3f}' for s in det_sims[:3])}")

            for label, sims in [("original", orig_sims), ("llm", llm_sims), ("det_suffix", det_sims)]:
                top2 = [s[0] for s in sims[:2]]
                hit = any(c in top2 for c in expected)
                results.add(
                    "LLM_VS_DET",
                    f"q_expand_{label}",
                    f"query_hit",
                    hit,
                    {"query": query[:30], "top1": f"chunk_{sims[0][0]}", "top1_sim": f"{sims[0][1]:.3f}"},
                )

# ─── TEST 6: Combined Pipeline (Dense + Reranker) ──────────────────────────
def test_combined_pipeline():
    """Test the full retrieval pipeline: dense retrieval → reranker → generation."""
    print("\n" + "─" * 80)
    print("TEST 6: Combined Pipeline (Dense → Reranker → Generation)")
    print("─" * 80)

    # Test the best combinations from previous tests
    # We'll use a realistic scenario: 5 chunks, query, retrieve top-3, rerank, generate

    ingestion_options = {
        "none": lambda t: t,
        "suffix": expand_suffix,
        "glossary_suffix": expand_glossary_suffix,
    }

    query_options = {
        "original": lambda q: q,
        "suffix": expand_query_suffix,
    }

    reranker_options = {
        "orig_q_orig_c": (lambda q: q, lambda t: t),
        "orig_q_glossary_c": (lambda q: q, expand_glossary_suffix),
        "exp_q_orig_c": (expand_query_suffix, lambda t: t),
        "exp_q_glossary_c": (expand_query_suffix, expand_glossary_suffix),
    }

    test_query = "battalions withdrew from position"
    expected = EXPECTED_MATCHES[test_query]

    print(f"\nQuery: '{test_query}'")
    print(f"Expected chunks: {expected}")

    for ing_name, ing_fn in ingestion_options.items():
        # Embed chunks
        chunk_texts = [ing_fn(c) for c in TEST_CHUNKS]
        chunk_embs = embed_dense(chunk_texts)

        for q_name, q_fn in query_options.items():
            q_text = q_fn(test_query)
            q_emb = embed_dense([q_text])[0]

            # Dense retrieval: get top-3
            sims = sorted(
                enumerate(cosine_similarity(q_emb, ce) for ce in chunk_embs),
                key=lambda x: x[1], reverse=True
            )
            top3_idx = [s[0] for s in sims[:3]]
            top3_scores = [s[1] for s in sims[:3]]

            # Rerank top-3
            for rr_name, (rr_q_fn, rr_c_fn) in reranker_options.items():
                rr_query = rr_q_fn(test_query)
                rr_passages = [rr_c_fn(TEST_CHUNKS[i]) for i in top3_idx]
                rr_scores = rerank_scores(rr_query, rr_passages)

                # Sort by reranker score
                rr_ranked = sorted(zip(top3_idx, rr_scores), key=lambda x: x[1], reverse=True)
                rr_top1 = rr_ranked[0][0]
                rr_hit = rr_top1 in expected

                results.add(
                    "PIPELINE",
                    f"ing={ing_name} q={q_name} rr={rr_name}",
                    "reranked_top1",
                    f"chunk_{rr_top1}",
                    {"hit": rr_hit, "expected": expected, "dense_top3": top3_idx},
                )

                # Generate answer with top-1 chunk
                top_chunk = TEST_CHUNKS[rr_top1]  # original text
                # Build context with glossary
                glossary = build_glossary(top_chunk)
                if glossary:
                    context = f"{top_chunk}\n[Abbreviation Glossary]\n{glossary}"
                else:
                    context = top_chunk

                answer = generate_answer(test_query, context, max_tokens=200)
                answer_correct = any(w in answer.lower() for w in ["battalion", "withdraw", "order", "position"])

                results.add(
                    "PIPELINE",
                    f"ing={ing_name} q={q_name} rr={rr_name}",
                    "answer_correct",
                    answer_correct,
                    {"answer_preview": answer[:80].replace("\n", " ")},
                )

                print(f"  ing={ing_name:15s} q={q_name:10s} rr={rr_name:25s} "
                      f"→ dense_top3={top3_idx} rerank_top1=chunk_{rr_top1} "
                      f"hit={rr_hit} answer_correct={answer_correct}")

# ─── MAIN ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("ABBREVIATION EXPANSION COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print(f"Abbreviations loaded: {len(FORWARD_MAP)}")
    print(f"Test chunks: {len(TEST_CHUNKS)}")
    print(f"Test queries: {len(TEST_QUERIES)}")

    # Run all tests
    test_dense_embeddings()
    test_sparse_embeddings()
    test_reranker()
    test_llm_query_expansion()
    test_combined_pipeline()
    test_generation()

    # Print summary
    results.print_summary()

    # Save results to JSON
    output_path = "/app/assets/abbr_test_results.json"
    with open(output_path, "w") as f:
        json.dump(results.results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Print final conclusions
    print("\n" + "=" * 80)
    print("FINAL CONCLUSIONS")
    print("=" * 80)

    # Analyze hit rates
    categories = defaultdict(lambda: {"hits": 0, "total": 0})
    for r in results.results:
        if "hit" in r["extra"]:
            cat = r["category"]
            categories[cat]["total"] += 1
            if r["extra"]["hit"]:
                categories[cat]["hits"] += 1

    print("\nHit rates by category:")
    for cat, stats in sorted(categories.items()):
        rate = stats["hits"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {cat:20s}: {stats['hits']}/{stats['total']} ({rate:.1f}%)")

    # Analyze by ingestion option
    print("\nHit rates by ingestion option (dense + sparse):")
    for ing_name in ["none", "suffix", "replace", "glossary_suffix"]:
        hits = 0
        total = 0
        for r in results.results:
            if r["category"] in ("DENSE", "SPARSE") and f"ing={ing_name}" in r["test"]:
                total += 1
                if r["extra"].get("hit"):
                    hits += 1
        rate = hits / total * 100 if total > 0 else 0
        print(f"  ing={ing_name:15s}: {hits}/{total} ({rate:.1f}%)")

    # Analyze by query option
    print("\nHit rates by query option (dense + sparse):")
    for q_name in ["original", "suffix", "replace"]:
        hits = 0
        total = 0
        for r in results.results:
            if r["category"] in ("DENSE", "SPARSE") and f"q={q_name}" in r["test"]:
                total += 1
                if r["extra"].get("hit"):
                    hits += 1
        rate = hits / total * 100 if total > 0 else 0
        print(f"  q={q_name:10s}: {hits}/{total} ({rate:.1f}%)")

    # Analyze reranker combinations
    print("\nHit rates by reranker combination:")
    for q_label in ["orig_q", "exp_q", "rep_q"]:
        for c_label in ["orig_chunk", "suffix_chunk", "replace_chunk", "glossary_chunk"]:
            hits = 0
            total = 0
            for r in results.results:
                if r["category"] == "RERANKER" and f"q={q_label}" in r["test"] and f"c={c_label}" in r["test"]:
                    total += 1
                    if r["extra"].get("hit"):
                        hits += 1
            if total > 0:
                rate = hits / total * 100
                print(f"  q={q_label:8s} c={c_label:15s}: {hits}/{total} ({rate:.1f}%)")

if __name__ == "__main__":
    main()
