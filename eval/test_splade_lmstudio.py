#!/usr/bin/env python3
"""
test_splade_lmstudio.py — check whether the SPLADE model served by LM Studio
produces real sparse output or just a dense proxy.

Usage:
    python test_splade_lmstudio.py
    python test_splade_lmstudio.py --base-url http://192.168.1.22:1234/v1 --model text-embedding-splade-v3
"""

import argparse
import math
import os
import sys

from openai import OpenAI


def embed(client: OpenAI, model: str, texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.getenv("OPENAI_API_BASE", "http://192.168.1.22:1234/v1"))
    parser.add_argument("--model",    default="splade-v3-distilbert")
    parser.add_argument("--api-key",  default=os.getenv("OPENAI_API_KEY", "lmstudio"))
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    print(f"base_url : {args.base_url}")
    print(f"model    : {args.model}\n")

    # ── 1. Format / dimensionality / sparsity ─────────────────────────────────
    probe = "The quick brown fox jumps over the lazy dog"
    [emb] = embed(client, args.model, [probe])

    dim = len(emb)
    nnz = sum(1 for v in emb if abs(v) > 1e-6)
    neg = sum(1 for v in emb if v < 0)

    print(f"raw (first 20 values) : {[round(v, 6) for v in emb[:20]]}")
    print(f"raw (last  20 values) : {[round(v, 6) for v in emb[-20:]]}")
    print()
    print(f"dim      : {dim}   (real SPLADE expects ~30522)")
    print(f"nnz      : {nnz}/{dim}   (real SPLADE expects ~200-600 non-zero)")
    print(f"negatives: {neg}   (real SPLADE: 0 — ReLU forces all weights >= 0)")

    # ── 2. Semantic discriminability ──────────────────────────────────────────
    similar_pairs = [
        ("The capital of France is Paris.",
         "Paris is the largest city and capital of France."),
        ("Machine learning models require large amounts of training data.",
         "Deep learning algorithms need huge datasets to learn effectively."),
    ]
    dissimilar_pairs = [
        ("The capital of France is Paris.",
         "Quantum mechanics describes the behaviour of subatomic particles."),
        ("Recipe: mix flour, eggs, and butter to make a cake.",
         "The stock market closed higher on strong technology earnings."),
    ]

    print()
    sim_scores, dis_scores = [], []

    for a, b in similar_pairs:
        ea, eb = embed(client, args.model, [a, b])
        s = cosine(ea, eb)
        sim_scores.append(s)
        print(f"  similar    {s:+.4f}  {a[:55]!r}")

    for a, b in dissimilar_pairs:
        ea, eb = embed(client, args.model, [a, b])
        s = cosine(ea, eb)
        dis_scores.append(s)
        print(f"  dissimilar {s:+.4f}  {a[:55]!r}")

    sep = sum(sim_scores) / len(sim_scores) - sum(dis_scores) / len(dis_scores)
    print(f"\n  separation : {sep:+.4f}   (higher = better discrimination)")

    # ── Verdict ────────────────────────────────────────────────────────────────
    print()
    is_splade = (dim >= 30000 and nnz < 1000 and neg == 0)
    if is_splade:
        print("PASS — output looks like real SPLADE. Safe for the sparse Qdrant leg.")
    else:
        reasons = []
        if dim < 30000:
            reasons.append(f"dim={dim} (expected ~30522)")
        if nnz >= 1000:
            reasons.append(f"nnz={nnz} (expected 200-600)")
        if neg > 0:
            reasons.append(f"{neg} negative weights (SPLADE ReLU forbids negatives)")
        print("FAIL — not valid SPLADE output: " + ", ".join(reasons))
        print("Keep SPLADE_MODEL=prithivida/Splade_PP_en_v1 with FastEmbed (CPU).")
        sys.exit(1)


if __name__ == "__main__":
    main()
