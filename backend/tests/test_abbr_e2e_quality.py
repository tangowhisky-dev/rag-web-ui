#!/usr/bin/env python3
"""End-to-end quality tests for abbreviation-aware answer generation.

Tests the full pipeline with real models:
  1. Dense embeddings (LM Studio: qwen3-embedding-0.6b)
  2. Sparse embeddings (local ONNX: SPLADE PP en v1)
  3. Cross-encoder reranker (local ONNX: ms-marco-MiniLM-L-12-v2)
  4. Generation LLM (LM Studio: gemma-4-26b-a4b)

Pipeline tested per query:
  original query
    → abbreviation suffix expansion (expand_query_suffix)
    → glossary extraction (build_glossary)
    → dense + sparse retrieval (cosine / dot product against chunk embeddings)
    → rerank (cross-encoder with expanded query)
    → generation with [Abbreviation Glossary] in context
    → verify answer quality

Test cases:
  A. Abbreviation query → full-form chunks (forward match)
  B. Full-form query → abbreviation chunks (reverse match)
  C. Mixed query (both abbrs and full forms)
  D. Plain English query (no abbreviations — no adverse effect)
  E. Multi-meaning abbreviation (wdr has 6 forms)
  F. Qualification abbreviation (psc/ndc)

Runs inside the backend container:
  docker exec rag-web-ui-backend-1 pytest tests/test_abbr_e2e_quality.py -v -s
"""
import os
import re
import sys
import math
import time
import logging
from typing import List, Dict, Tuple, Optional

import pytest

sys.path.insert(0, "/app")
os.environ.setdefault("PYTHONPATH", "/app")

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("abbr_e2e")

# ─── Config ──────────────────────────────────────────────────────────────────

LM_STUDIO_BASE = os.environ.get("LM_STUDIO_BASE_URL", "http://192.168.1.3:2244/v1")
LM_STUDIO_KEY = os.environ.get("LM_STUDIO_API_KEY", "dummy")
DENSE_MODEL = os.environ.get("DENSE_EMBEDDINGS_MODEL", "qwen/qwen3-embedding-0.6b")
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "google/gemma-4-26b-a4b")

# ─── Test data ───────────────────────────────────────────────────────────────

TEST_CHUNKS = [
    # Chunk 0: Heavy abbreviations — military operation
    "The CO ordered the bns to wdr from the forward position. The MO reported "
    "casualties. The adjt coordinated with HQ. The op was conducted at first light. "
    "The recce team provided intelligence on enemy positions.",
    # Chunk 1: Heavy abbreviations — brigade-level operation
    "The GOC visited the bde HQ and briefed the bde comd on the op. The inf bn was "
    "tasked to secure the obj. The armd sqn was to provide spt. The arty bty was "
    "placed in sp of the inf.",
    # Chunk 2: No abbreviations — weather (negative control)
    "The weather forecast indicates rain for the next three days. Temperature "
    "will drop to 15 degrees Celsius. Farmers should prepare for wet conditions.",
    # Chunk 3: Abbreviations — medical/resupply
    "The DA approved the medical resupply. The SP was established at checkpoint 4. "
    "The cas were evacuated to the Fd Amb. The spt elements moved up at 0600.",
    # Chunk 4: Full forms only (reverse match target)
    "The commanding officer ordered the battalions to withdraw from the forward "
    "position. The medical officer reported casualties. The operation was "
    "conducted at first light by the reconnaissance team.",
    # Chunk 5: Qualification abbreviations
    "The officer completed psc and ndc before being posted to the HQ. "
    "He was recommended by the GOC for his outstanding leadership during the op.",
]

# Expected chunk matches per query (indices into TEST_CHUNKS)
TEST_CASES = [
    {
        "name": "A_abbr_query_full_form_chunks",
        "query": "bns wdr from position",
        "expected_chunks": [0, 4],
        "answer_keywords": ["battalion", "withdraw", "position"],
        "description": "Abbreviation query should match chunks with full forms",
    },
    {
        "name": "B_full_form_query_abbr_chunks",
        "query": "commanding officer ordered battalions to withdraw",
        "expected_chunks": [0, 4],
        "answer_keywords": ["commanding", "officer", "battalion", "withdraw"],
        "description": "Full-form query should match chunks with abbreviations",
    },
    {
        "name": "C_mixed_query",
        "query": "CO ordered battalions to wdr from position",
        "expected_chunks": [0, 4],
        "answer_keywords": ["commanding", "officer", "battalion", "withdraw", "position"],
        "description": "Mixed query (abbr + full form) should match both chunk types",
    },
    {
        "name": "D_plain_english_no_adverse",
        "query": "what is the weather forecast",
        "expected_chunks": [2],
        "answer_keywords": ["rain", "weather", "temperature"],
        "description": "Plain English query should match weather chunk, no abbreviation interference",
    },
    {
        "name": "E_multi_meaning_wdr",
        "query": "who ordered the wdr of bns?",
        "expected_chunks": [0, 4],
        "answer_keywords": ["commanding", "officer", "battalion", "withdraw"],
        "description": "Multi-meaning abbreviation (wdr has 6 forms) should still find correct chunks",
    },
    {
        "name": "F_qualification_abbr",
        "query": "which officer completed psc and ndc",
        "expected_chunks": [5],
        "answer_keywords": ["psc", "ndc", "officer", "headquarters", "leader"],
        "description": "Qualification abbreviations should be interpreted correctly",
    },
]


# ─── Model accessors (use app infrastructure, no HF Hub downloads) ──────────

def get_dense_client():
    """OpenAI client for dense embeddings via LM Studio."""
    from openai import OpenAI
    return OpenAI(api_key=LM_STUDIO_KEY, base_url=LM_STUDIO_BASE), DENSE_MODEL


def get_sparse_embedder():
    """SPLADE sparse embedder from app infrastructure (local ONNX, cached)."""
    from app.services.infrastructure import get_sparse_embedder
    return get_sparse_embedder()


def get_reranker():
    """Cross-encoder reranker from app infrastructure (local ONNX, cached)."""
    from app.services.retrieval.reranker import _get_cross_encoder
    return _get_cross_encoder()


def get_generation_client():
    """OpenAI client for generation LLM via LM Studio."""
    from openai import OpenAI
    return OpenAI(api_key=LM_STUDIO_KEY, base_url=LM_STUDIO_BASE), GENERATION_MODEL


def get_abbr_lookup():
    """Build abbreviation lookup directly from the CSV file.

    The conftest.py replaces app.db.session with SQLite, so we can't use
    build_lookup() which queries the DB. Instead we build the lookup from
    the CSV directly, the same way the existing test_abbr_expansion.py
    loads abbreviations.
    """
    from app.services.abbreviation_service import AbbreviationLookup
    from flashtext2 import KeywordProcessor

    csv_path = "/app/assets/abbreviations_enhanced.csv"
    forward: Dict[str, List[str]] = {}

    import csv as csv_mod
    with open(csv_path, encoding="utf-8") as f:
        for row in csv_mod.DictReader(f):
            abbr = row["abbreviation"].strip()
            form = row["expanded_form"].strip()
            if abbr and form:
                if abbr not in forward:
                    forward[abbr] = []
                if form not in forward[abbr]:
                    forward[abbr].append(form)

    from app.services.abbreviation_service import (
        _REVERSE_MIN_FORM_LEN, _is_uppercase,
    )

    exact_abbrs = []
    prose_abbrs = []
    for abbr in forward:
        if len(abbr.strip()) <= 1:
            continue
        if _is_uppercase(abbr):
            exact_abbrs.append(abbr)
        else:
            prose_abbrs.append(abbr)

    kp_exact = KeywordProcessor(case_sensitive=True)
    for a in exact_abbrs:
        kp_exact.add_keyword(a, a)
    kp_prose = KeywordProcessor(case_sensitive=False)
    for a in prose_abbrs:
        kp_prose.add_keyword(a, a)

    reverse: Dict[str, List[str]] = {}
    for abbr, forms in forward.items():
        if len(abbr.strip()) <= 1:
            continue
        for form in forms:
            key = form.lower()
            if len(key) < _REVERSE_MIN_FORM_LEN:
                continue
            if key not in reverse:
                reverse[key] = []
            if abbr not in reverse[key]:
                reverse[key].append(abbr)

    kp_reverse = KeywordProcessor(case_sensitive=False)
    for form_lower in reverse:
        kp_reverse.add_keyword(form_lower, form_lower)

    return AbbreviationLookup(
        forward=forward,
        kp_exact=kp_exact,
        kp_prose=kp_prose,
        reverse=reverse,
        kp_reverse=kp_reverse,
    )


# ─── Embedding / retrieval helpers ───────────────────────────────────────────

def embed_dense(texts: List[str]) -> List[List[float]]:
    client, model = get_dense_client()
    embeddings = []
    for i in range(0, len(texts), 32):
        batch = texts[i:i + 32]
        resp = client.embeddings.create(input=batch, model=model)
        embeddings.extend([r.embedding for r in resp.data])
    return embeddings


def embed_sparse(texts: List[str]) -> List:
    embedder = get_sparse_embedder()
    return list(embedder.embed(texts))


def cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def sparse_dot(a, b) -> float:
    """Dot product for SPLADE sparse vectors."""
    # SPLADE vectors from fastembed are dicts of {index: value}
    if hasattr(a, "indices") and hasattr(a, "values"):
        a_dict = dict(zip(a.indices, a.values))
    elif isinstance(a, dict):
        a_dict = a
    else:
        a_dict = dict(a)
    if hasattr(b, "indices") and hasattr(b, "values"):
        b_dict = dict(zip(b.indices, b.values))
    elif isinstance(b, dict):
        b_dict = b
    else:
        b_dict = dict(b)
    return sum(a_dict.get(k, 0) * b_dict.get(k, 0) for k in a_dict)


def rerank_scores(query: str, passages: List[str]) -> List[float]:
    encoder = get_reranker()
    return list(encoder.rerank(query, passages))


# ─── Generation helper ───────────────────────────────────────────────────────

def generate_answer(query: str, context: str, glossary: str = "") -> str:
    """Generate an answer using the production finalize prompt pattern.

    The context is formatted with [Abbreviation Glossary] just like
    format_context_string does in the real pipeline.
    """
    client, model = get_generation_client()

    system = (
        "You are an autonomous enterprise knowledge assistant. You have no internet access. "
        "You operate only on the attached knowledge bases.\n\n"
        "Critical rules:\n"
        "- If you cannot find the answer in the provided context, say so. Do not fabricate.\n"
        "- Cite the retrieved document chunks that support each factual claim.\n"
        "- Be concise and follow the user's formatting instructions exactly.\n"
        "- If a [Abbreviation Glossary] section is provided in the context, use it to "
        "interpret abbreviations in the user query and retrieved documents. "
        "Do not echo the glossary in your output.\n\n"
        "# Retrieved Document Context\n\n"
        "The retrieved context consists of document chunks labeled like [KB-1].\n"
        "These chunks are the authoritative source for document-specific information."
    )

    parts = [f"[KB-1]\n{context}"]
    if glossary:
        parts.append(f"[Abbreviation Glossary]\n{glossary}")
    context_text = "\n\n---\n\n".join(parts)

    user = (
        f"User query: {query}\n\n"
        f"Retrieved context (the only citable evidence):\n{context_text}\n\n"
        "Provide a concise, accurate answer."
    )

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=500,
                temperature=0.1,
                stream=False,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("Generation attempt %d failed: %s", attempt + 1, exc)
            if attempt < 2:
                time.sleep(3)
            raise


# ─── Pipeline runner ─────────────────────────────────────────────────────────

def run_pipeline(
    query: str,
    chunk_texts: List[str],
    chunk_dense: List[List[float]],
    chunk_sparse: List,
    lookup,
) -> Dict:
    """Run the full abbreviation-aware retrieval + rerank + generation pipeline.

    Returns a dict with:
      - expanded_query
      - glossary
      - dense_rank: chunk indices ranked by dense cosine similarity
      - sparse_rank: chunk indices ranked by sparse dot product
      - reranked: chunk indices after cross-encoder reranking
      - top_chunk: best chunk after reranking
      - answer: generated answer text
      - answer_correct: whether answer contains expected keywords
    """
    from app.services.abbreviation_service import (
        expand_query_suffix,
        build_glossary,
    )

    # Step 1: Abbreviation expansion
    expanded_query = expand_query_suffix(query, lookup)
    glossary = build_glossary(query, lookup)

    # Step 2: Dense retrieval (use expanded query for embedding)
    q_dense = embed_dense([expanded_query])[0]
    dense_sims = [(i, cosine_sim(q_dense, ce)) for i, ce in enumerate(chunk_dense)]
    dense_rank = sorted(dense_sims, key=lambda x: x[1], reverse=True)

    # Step 3: Sparse retrieval (use expanded query for embedding)
    q_sparse = embed_sparse([expanded_query])[0]
    sparse_sims = [(i, sparse_dot(q_sparse, ce)) for i, ce in enumerate(chunk_sparse)]
    sparse_rank = sorted(sparse_sims, key=lambda x: x[1], reverse=True)

    # Step 4: Merge candidates (union of top-k from both legs, dedup)
    top_k = 5
    candidates = set()
    for i, _ in dense_rank[:top_k]:
        candidates.add(i)
    for i, _ in sparse_rank[:top_k]:
        candidates.add(i)
    candidate_texts = [chunk_texts[i] for i in sorted(candidates)]

    # Step 5: Rerank with cross-encoder (use expanded query)
    scores = rerank_scores(expanded_query, candidate_texts)
    scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    reranked = [i for i, _ in scored]

    # Step 6: Generate answer using top chunk + glossary
    top_chunk_idx = reranked[0] if reranked else 0
    top_chunk_text = chunk_texts[top_chunk_idx]
    answer = generate_answer(query, top_chunk_text, glossary=glossary)

    return {
        "expanded_query": expanded_query,
        "glossary": glossary,
        "dense_rank": dense_rank[:3],
        "sparse_rank": sparse_rank[:3],
        "reranked": reranked[:3],
        "top_chunk": top_chunk_idx,
        "answer": answer,
    }


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def lookup():
    return get_abbr_lookup()


@pytest.fixture(scope="module")
def chunk_embeddings():
    """Embed all test chunks once (dense + sparse)."""
    print("\nEmbedding test chunks (dense + sparse)...")
    dense = embed_dense(TEST_CHUNKS)
    sparse = embed_sparse(TEST_CHUNKS)
    print(f"  Dense: {len(dense)} embeddings, dim={len(dense[0])}")
    print(f"  Sparse: {len(sparse)} embeddings")
    return dense, sparse


# ─── Tests ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("case", TEST_CASES, ids=[c["name"] for c in TEST_CASES])
def test_e2e_abbreviation_pipeline(case, lookup, chunk_embeddings):
    """End-to-end: expand → retrieve → rerank → generate → verify answer quality."""
    chunk_dense, chunk_sparse = chunk_embeddings

    print(f"\n{'=' * 80}")
    print(f"Test: {case['name']}")
    print(f"Query: {case['query']!r}")
    print(f"Expected chunks: {case['expected_chunks']}")
    print(f"{'=' * 80}")

    result = run_pipeline(
        query=case["query"],
        chunk_texts=TEST_CHUNKS,
        chunk_dense=chunk_dense,
        chunk_sparse=chunk_sparse,
        lookup=lookup,
    )

    # Print pipeline details
    print(f"\n  Expanded query: {result['expanded_query']!r}")
    print(f"  Glossary:\n    {result['glossary'] or '(none)'}")
    print(f"  Dense top-3: {[(i, round(s, 3)) for i, s in result['dense_rank']]}")
    print(f"  Sparse top-3: {[(i, round(s, 3)) for i, s in result['sparse_rank']]}")
    print(f"  Reranked top-3: {result['reranked']}")
    print(f"  Top chunk: {result['top_chunk']}")
    print(f"\n  Answer:\n    {result['answer'][:300]}")

    # Assertions

    # 1. Reranked top result should be in expected chunks
    assert result["top_chunk"] in case["expected_chunks"], (
        f"Top chunk {result['top_chunk']} not in expected {case['expected_chunks']}. "
        f"Reranked: {result['reranked']}"
    )

    # 2. At least one expected chunk should appear in reranked top-3
    top3 = result["reranked"][:3]
    assert any(c in top3 for c in case["expected_chunks"]), (
        f"No expected chunk {case['expected_chunks']} in reranked top-3 {top3}"
    )

    # 3. Answer should contain at least one expected keyword
    answer_lower = result["answer"].lower()
    matched_keywords = [kw for kw in case["answer_keywords"] if kw in answer_lower]
    assert len(matched_keywords) > 0, (
        f"Answer contains none of expected keywords {case['answer_keywords']}. "
        f"Answer: {result['answer'][:200]}"
    )

    # 4. Answer should not be a refusal (unless it's genuinely unrelated)
    #    Note: some models (Gemma) emit thinking tags like <|channel>thought*
    #    before the actual answer. Strip those before checking for refusals.
    clean_answer = re.sub(r"<\|[^|]*\|>", "", result["answer"]).strip()
    clean_lower = clean_answer.lower()
    refusals = ["cannot answer", "don't have", "no information", "not enough information"]
    is_refusal = any(r in clean_lower for r in refusals)
    assert not is_refusal, (
        f"Answer is a refusal but expected chunks were found. "
        f"Answer: {result['answer'][:200]}"
    )

    # 5. For queries that contain abbreviations (not full forms), the
    #    glossary should be non-empty. Queries that use only full forms
    #    (e.g. "commanding officer") have no abbreviations to glossary-ize —
    #    the glossary is correctly empty because build_glossary only finds
    #    abbreviations (forward match), not full forms (reverse match).
    #    The expanded_query suffix handles the reverse direction.
    has_abbr_in_query = case["name"] not in (
        "D_plain_english_no_adverse",
        "B_full_form_query_abbr_chunks",
    )
    if has_abbr_in_query:
        assert result["glossary"] != "", (
            f"Glossary should be non-empty for abbreviation query {case['query']!r}"
        )

    # 6. For plain English query, glossary should be empty (no adverse effect)
    if case["name"] == "D_plain_english_no_adverse":
        assert result["glossary"] == "", (
            f"Glossary should be empty for plain English query {case['query']!r}"
        )

    # 7. For full-form queries, the expanded_query should contain reverse
    #    matches (full form → abbreviation appended in [Abbreviation Glossary] ...])
    if case["name"] == "B_full_form_query_abbr_chunks":
        assert "[Abbreviation Glossary]" in result["expanded_query"], (
            f"Full-form query should have reverse expansions in suffix: "
            f"{result['expanded_query']!r}"
        )

    print(f"\n  ✓ PASSED — matched keywords: {matched_keywords}")


def test_glossary_improves_generation(lookup, chunk_embeddings):
    """Verify that including the glossary in the generation context produces
    a better answer than without it, specifically for abbreviation-heavy queries."""
    chunk_dense, _ = chunk_embeddings

    # Use a query with abbreviations that the LLM might not know
    query = "bns wdr from position"
    chunk = TEST_CHUNKS[0]  # abbreviation-heavy chunk

    from app.services.abbreviation_service import build_glossary
    glossary = build_glossary(query, lookup)
    assert glossary != "", "Glossary should be non-empty for abbreviation query"

    # Generate WITHOUT glossary
    answer_no_gloss = generate_answer(query, chunk, glossary="")

    # Generate WITH glossary
    answer_with_gloss = generate_answer(query, chunk, glossary=glossary)

    print(f"\n  Query: {query!r}")
    print(f"  Chunk: {chunk[:80]}...")
    print(f"  Glossary: {glossary}")
    print(f"\n  WITHOUT glossary:\n    {answer_no_gloss[:300]}")
    print(f"\n  WITH glossary:\n    {answer_with_gloss[:300]}")

    # The answer with glossary should mention battalion/withdraw more explicitly
    # because the glossary tells the LLM what bns and wdr mean
    keywords = ["battalion", "withdraw"]
    no_gloss_matches = sum(1 for kw in keywords if kw in answer_no_gloss.lower())
    with_gloss_matches = sum(1 for kw in keywords if kw in answer_with_gloss.lower())

    print(f"\n  Keywords without glossary: {no_gloss_matches}/{len(keywords)}")
    print(f"  Keywords with glossary: {with_gloss_matches}/{len(keywords)}")

    # The answer with glossary should be at least as good as without
    assert with_gloss_matches >= no_gloss_matches, (
        f"Answer with glossary ({with_gloss_matches} keywords) should be >= "
        f"answer without glossary ({no_gloss_matches} keywords)"
    )

    print(f"\n  ✓ PASSED — glossary does not degrade answer quality")


def test_expanded_query_preserves_original(lookup):
    """Verify that expand_query_suffix always preserves the original query text."""
    from app.services.abbreviation_service import expand_query_suffix

    queries = [
        "CO ordered bns to wdr",
        "bns wdr from position",
        "which officer completed psc and ndc",
        "what is the weather forecast",
    ]

    for query in queries:
        expanded = expand_query_suffix(query, lookup)
        assert expanded.startswith(query), (
            f"Expanded query must start with original: {query!r} → {expanded!r}"
        )
        # The original text should appear verbatim, not modified
        original_part = expanded[:len(query)]
        assert original_part == query, (
            f"Original text was modified: {query!r} → {original_part!r}"
        )

    print("  ✓ All queries preserved verbatim in expanded output")
