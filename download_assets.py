#!/usr/bin/env python3
"""
download_assets.py — pre-download all model assets required by the backend.

Currently downloads:
  - FastEmbed SPLADE sparse-embedding model (used for hybrid retrieval)

Usage:
    python download_assets.py [--cache-dir PATH] [--model MODEL_NAME]

Defaults are read from environment variables (or .env) matching the backend config:
    FASTEMBED_CACHE_DIR  (default: ./assets/fastembed)
    SPLADE_MODEL         (default: prithivida/Splade_PP_en_v1)
"""

import argparse
import os
import sys
import time


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader — no external dependency required."""
    if not os.path.isfile(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def download_splade(model_name: str, cache_dir: str) -> None:
    print(f"  model      : {model_name}")
    print(f"  cache_dir  : {os.path.abspath(cache_dir)}")

    try:
        from fastembed import SparseTextEmbedding
    except ImportError:
        print("\n[ERROR] fastembed is not installed.")
        print("        Run:  pip install fastembed")
        sys.exit(1)

    os.makedirs(cache_dir, exist_ok=True)

    print("\nDownloading / verifying model files …")
    t0 = time.time()
    SparseTextEmbedding(model_name=model_name, cache_dir=cache_dir)
    elapsed = time.time() - t0

    print(f"Done in {elapsed:.1f}s.\n")


def main() -> None:
    _load_dotenv()

    default_cache = os.getenv("FASTEMBED_CACHE_DIR", "./assets/fastembed")
    default_model = os.getenv("SPLADE_MODEL", "prithivida/Splade_PP_en_v1")

    parser = argparse.ArgumentParser(description="Pre-download RAG-Web-UI model assets.")
    parser.add_argument(
        "--cache-dir",
        default=default_cache,
        help=f"FastEmbed cache directory (default: {default_cache})",
    )
    parser.add_argument(
        "--model",
        default=default_model,
        help=f"SPLADE model name (default: {default_model})",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("RAG-Web-UI asset downloader")
    print("=" * 60)
    print("\n[1/1] SPLADE sparse-embedding model (FastEmbed)")
    download_splade(model_name=args.model, cache_dir=args.cache_dir)

    print("All assets downloaded successfully.")
    print("You can now start the backend — no network access needed for model loading.")


if __name__ == "__main__":
    main()
