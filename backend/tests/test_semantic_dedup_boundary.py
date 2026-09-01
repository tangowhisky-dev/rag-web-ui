#!/usr/bin/env python3
"""Map the cosine similarity boundary of qwen3-embedding-0.6b.

Tests text pairs at varying semantic distances to find where 0.95
similarity falls — what counts as "redundant" vs "different meaning"
under the current DEDUP_SEMANTIC_THRESHOLD.

Run inside the backend container:
    python tests/test_semantic_dedup_boundary.py
"""
import os
import sys
import itertools
import numpy as np

sys.path.insert(0, "/app")
os.environ.setdefault("PYTHONPATH", "/app")

from openai import OpenAI
from app.db.session import SessionLocal
from app.services.settings_service import get_setting

LM_STUDIO_BASE = os.environ.get("LM_STUDIO_BASE_URL", "http://192.168.1.3:2244/v1")
LM_STUDIO_KEY = os.environ.get("LM_STUDIO_API_KEY", "dummy")


def get_embedder():
    db = SessionLocal()
    try:
        api_key = get_setting(db, "EMBEDDING_API_KEY", None) or LM_STUDIO_KEY
        api_base = get_setting(db, "EMBEDDING_API_BASE", None) or LM_STUDIO_BASE
        model = get_setting(db, "DENSE_EMBEDDINGS_MODEL", None) or "qwen/qwen3-embedding-0.6b"
    finally:
        db.close()
    return OpenAI(api_key=api_key, base_url=api_base), model


def embed(client, model, texts):
    resp = client.embeddings.create(input=texts, model=model)
    return [np.array(d.embedding, dtype=np.float32) for d in resp.data]


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ─── Text pairs grouped by expected semantic distance ──────────────────────

PAIRS = [
    # Group 1: Exact duplicates / trivial reformatting
    ("EXACT DUPLICATE",
     "The commanding officer ordered the battalion to withdraw from the forward position.",
     "The commanding officer ordered the battalion to withdraw from the forward position."),

    ("PUNCTUATION CHANGE",
     "The commanding officer ordered the battalion to withdraw from the forward position.",
     "The commanding officer ordered the battalion to withdraw from the forward position!"),

    ("WORD ORDER (passive)",
     "The commanding officer ordered the battalion to withdraw from the forward position.",
     "The battalion was ordered by the commanding officer to withdraw from the forward position."),

    ("SYNONYM SUBSTITUTION",
     "The commanding officer ordered the battalion to withdraw from the forward position.",
     "The commanding officer instructed the battalion to retreat from the advanced position."),

    # Group 2: Paraphrase — same meaning, different words
    ("PARAPHRASE (military)",
     "The CO ordered the bns to wdr from the forward position.",
     "Battalions were instructed to pull back from the front line by their commander."),

    ("PARAPHRASE (general)",
     "Revenue increased by 15% in Q3 2024 driven by strong sales in the European market.",
     "Third quarter 2024 saw a 15% revenue growth, primarily from European sales."),

    ("PARAPHRASE (technical)",
     "The server crashed due to a memory leak in the authentication module.",
     "A memory leak in the auth service caused the server to go down."),

    # Group 3: Same topic, different facts/details
    ("SAME TOPIC, DIFFERENT DETAIL",
     "Revenue increased by 15% in Q3 2024 driven by strong sales in the European market.",
     "Revenue increased by 8% in Q3 2024 driven by strong sales in the Asian market."),

    ("SAME TOPIC, DIFFERENT ENTITY",
     "The commanding officer ordered the battalion to withdraw from the forward position.",
     "The commanding officer ordered the squadron to advance toward the enemy position."),

    ("SAME TOPIC, DIFFERENT EVENT",
     "The server crashed due to a memory leak in the authentication module.",
     "The server crashed due to a disk space exhaustion in the logging module."),

    # Group 4: Same domain, different subtopic
    ("SAME DOMAIN, DIFFERENT SUBTOPIC",
     "The battalion conducted a flanking maneuver at dawn to secure the eastern ridge.",
     "The brigade headquarters coordinated artillery strikes on the western valley."),

    ("SAME DOC, DIFFERENT SECTION",
     "Chapter 1: Introduction to linear algebra and vector spaces.",
     "Chapter 5: Eigenvalue decomposition and its applications in quantum mechanics."),

    # Group 5: Related but clearly different
    ("RELATED BUT DIFFERENT",
     "The battalion conducted a flanking maneuver at dawn to secure the eastern ridge.",
     "The logistics team resupplied the forward operating base with ammunition and rations."),

    ("RELATED BUT DIFFERENT (tech)",
     "The server crashed due to a memory leak in the authentication module.",
     "The database migration failed because of a foreign key constraint violation."),

    # Group 6: Completely unrelated
    ("UNRELATED",
     "The commanding officer ordered the battalion to withdraw from the forward position.",
     "The weather forecast predicts heavy rainfall and thunderstorms for the weekend."),

    ("UNRELATED (extreme)",
     "Revenue increased by 15% in Q3 2024 driven by strong sales in the European market.",
     "The cat sat on the mat while the children played in the garden."),
]

# ─── Also test chunk-level pairs (longer text, more realistic) ──────────────

CHUNK_PAIRS = [
    ("CHUNK: near-identical sections",
     """## Forward Position Status
The 3rd Battalion reported heavy enemy contact at grid reference 485-672.
Casualties were light: 2 wounded, 0 KIA. The battalion commander requested
immediate artillery support and air assets to suppress enemy positions
on the eastern ridge. Air support was unavailable due to weather conditions.
The battalion maintained defensive positions throughout the night.""",
     """## Forward Position Status
The 3rd Battalion reported heavy enemy contact at grid reference 485-672.
Casualties were light: 2 wounded, 0 KIA. The battalion commander requested
immediate artillery support and air assets to suppress enemy positions
on the eastern ridge. Air support was unavailable due to weather conditions.
The battalion maintained defensive positions throughout the night."""),

    ("CHUNK: same event, different report wording",
     """## Forward Position Status
The 3rd Battalion reported heavy enemy contact at grid reference 485-672.
Casualties were light: 2 wounded, 0 KIA. The battalion commander requested
immediate artillery support and air assets to suppress enemy positions
on the eastern ridge. Air support was unavailable due to weather conditions.
The battalion maintained defensive positions throughout the night.""",
     """## 3rd Battalion Engagement Report
At grid 485-672, the 3rd Bn came under significant enemy fire. Two soldiers
were wounded, none killed. The CO called for arty and air support to pin
down enemy forces on the east ridge, but weather scrubbed the air mission.
The unit held its defensive line overnight."""),

    ("CHUNK: same operation, different phase",
     """## Forward Position Status
The 3rd Battalion reported heavy enemy contact at grid reference 485-672.
Casualties were light: 2 wounded, 0 KIA. The battalion commander requested
immediate artillery support and air assets to suppress enemy positions
on the eastern ridge. Air support was unavailable due to weather conditions.
The battalion maintained defensive positions throughout the night.""",
     """## After-Action Review
Following the engagement at grid 485-672, the 3rd Battalion conducted a
withdrawal to Phase Line Echo. The after-action review identified three
lessons: (1) air support timing needs improvement, (2) casualty evacuation
routes were blocked, (3) night vision equipment was insufficient for the
terrain. Recommendations were forwarded to brigade headquarters."""),

    ("CHUNK: same document, different chapter",
     """## Chapter 3: Authentication Architecture
The system uses JWT tokens with RS256 signing. Access tokens expire after
15 minutes; refresh tokens expire after 7 days. The token validation
middleware checks signature, expiry, and issuer claims. Failed validations
return 401 with a WWW-Authenticate header.""",
     """## Chapter 7: Database Connection Pooling
The application uses PgBouncer as a connection pooler with a maximum of
50 connections. Idle timeout is set to 300 seconds. Connection lifecycle
is managed by the SQLAlchemy engine with pool_pre_ping enabled to detect
stale connections early."""),
]


def main():
    print("=" * 90)
    print("SEMANTIC DEDUP BOUNDARY ANALYSIS")
    print("Model: qwen/qwen3-embedding-0.6b | Threshold: 0.95")
    print("=" * 90)

    client, model = get_embedder()

    all_texts = []
    for _, a, b in PAIRS + CHUNK_PAIRS:
        all_texts.extend([a, b])

    print(f"\nEmbedding {len(all_texts)} texts...")
    vectors = embed(client, model, all_texts)
    print("Done.\n")

    # ─── Pair results ──────────────────────────────────────────────────────
    print("-" * 90)
    print(f"{'GROUP':<42} {'SIMILARITY':>10}  {'VERDICT':>12}")
    print("-" * 90)

    idx = 0
    results = []
    for label, _, _ in PAIRS:
        sim = cosine(vectors[idx], vectors[idx + 1])
        verdict = "DEDUPED" if sim >= 0.95 else "KEPT"
        results.append((label, sim, verdict))
        print(f"{label:<42} {sim:>10.4f}  {verdict:>12}")
        idx += 2

    print("-" * 90)
    print(f"\n{'CHUNK-LEVEL PAIRS':<42} {'SIMILARITY':>10}  {'VERDICT':>12}")
    print("-" * 90)

    for label, _, _ in CHUNK_PAIRS:
        sim = cosine(vectors[idx], vectors[idx + 1])
        verdict = "DEDUPED" if sim >= 0.95 else "KEPT"
        results.append((label, sim, verdict))
        print(f"{label:<42} {sim:>10.4f}  {verdict:>12}")
        idx += 2

    print("-" * 90)

    # ─── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("ANALYSIS")
    print("=" * 90)

    deduped = [(l, s) for l, s, v in results if v == "DEDUPED"]
    kept = [(l, s) for l, s, v in results if v == "KEPT"]

    print(f"\nPairs DEDUPED (sim >= 0.95): {len(deduped)}")
    for l, s in deduped:
        print(f"  {s:.4f}  {l}")

    print(f"\nPairs KEPT (sim < 0.95): {len(kept)}")
    for l, s in kept:
        print(f"  {s:.4f}  {l}")

    # Find the boundary
    all_sims = sorted([s for _, s, _ in results])
    print(f"\nSimilarity range: {all_sims[0]:.4f} — {all_sims[-1]:.4f}")
    print(f"Sorted: {[f'{s:.3f}' for s in all_sims]}")

    # Find the gap around 0.95
    below = [s for s in all_sims if s < 0.95]
    above = [s for s in all_sims if s >= 0.95]
    if below and above:
        print(f"\nBoundary gap: {max(below):.4f} (kept) → {min(above):.4f} (deduped)")
        print(f"Gap width: {min(above) - max(below):.4f}")

    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
