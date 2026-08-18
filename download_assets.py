#!/usr/bin/env python3
"""
download_assets.py — pre-download all model assets required by the backend and frontend.

This script is intended to be run on the HOST machine from the project root
directory (rag-web-ui/). It downloads model files into ./assets/ subdirectories.
In Docker, ./assets is volume-mounted to /app/assets, so the backend and frontend
can read them at /app/assets/... without network access.

Downloads:
  1. FastEmbed SPLADE sparse-embedding model (hybrid retrieval, sparse leg)
  2. FastEmbed ONNX cross-encoder reranker model (post-RRF reranking)
  3. Whisper tiny.en ONNX model (browser-based voice-to-text)
  4. HuggingFace tokenizer files (accurate token counting in the backend)

Usage:
    python download_assets.py [options]

Options:
    --cache-dir PATH          FastEmbed cache directory
    --model MODEL             SPLADE model name
    --reranker-cache-dir PATH Reranker cache directory
    --reranker-model MODEL    Cross-encoder model name
    --whisper-dir PATH        Whisper model directory
    --tokenizer-dir PATH      Tokenizer base directory
    --tokenizer-repo REPO     HuggingFace repo for tokenizer (e.g. google/gemma-2-2b)
    --skip-splade             Skip SPLADE download
    --skip-reranker           Skip reranker download
    --skip-whisper            Skip Whisper model download
    --skip-tokenizer          Skip tokenizer download
"""

import argparse
import os
import sys
import time
import urllib.request


# ─── Project root detection ──────────────────────────────────────────────

# Files/dirs that must exist in the project root for this script to work.
_PROJECT_MARKERS = [
    "backend",
    "frontend",
    "docker-compose.yml",
    "download_assets.py",
]


def _resolve_project_root() -> str:
    """Return the project root directory.

    The script must be run from the project root (rag-web-ui/) so that
    relative paths like ./assets/ resolve correctly. We verify this by
    checking for known project markers.
    """
    cwd = os.getcwd()
    missing = [m for m in _PROJECT_MARKERS if not os.path.exists(os.path.join(cwd, m))]
    if missing:
        print(f"[ERROR] This script must be run from the project root directory")
        print(f"        (rag-web-ui/), but the following markers are missing from")
        print(f"        the current directory ({cwd}):")
        for m in missing:
            print(f"          - {m}")
        sys.exit(1)
    return cwd


def _ensure_assets_dir(project_root: str) -> str:
    """Ensure ./assets/ exists and return its path."""
    assets_dir = os.path.join(project_root, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    return assets_dir


# ─── .env loader ─────────────────────────────────────────────────────────

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


def _env_or_default(env_key: str, default: str) -> str:
    """Read an env var, stripping Docker container paths if necessary.

    .env files in this project use container paths (e.g. /app/assets/...).
    When running on the host, we translate /app/assets/ → ./assets/.
    """
    val = os.getenv(env_key)
    if val:
        # Translate Docker container paths to host-relative paths.
        if val.startswith("/app/assets/"):
            return val.replace("/app/assets/", "./assets/", 1)
        return val
    return default


# ─── Download functions ──────────────────────────────────────────────────

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


def download_reranker(model_name: str, cache_dir: str) -> None:
    print(f"  model      : {model_name}")
    print(f"  cache_dir  : {os.path.abspath(cache_dir)}")

    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
    except ImportError:
        print("\n[ERROR] fastembed is not installed.")
        print("        Run:  pip install fastembed")
        sys.exit(1)

    os.makedirs(cache_dir, exist_ok=True)

    print("\nDownloading / verifying model files …")
    t0 = time.time()
    TextCrossEncoder(model_name=model_name, cache_dir=cache_dir)
    elapsed = time.time() - t0

    print(f"Done in {elapsed:.1f}s.\n")


def download_whisper(dest_dir: str) -> None:
    """Download Whisper tiny.en ONNX model files for browser-based STT."""
    print(f"  dest_dir   : {os.path.abspath(dest_dir)}")

    base_url = "https://huggingface.co/Xenova/whisper-tiny.en/resolve/main"
    config_files = [
        "config.json", "tokenizer.json", "tokenizer_config.json",
        "normalizer.json", "preprocessor_config.json",
        "special_tokens_map.json", "vocab.json", "merges.txt",
    ]
    onnx_files = [
        "encoder_model_quantized.onnx",
        "decoder_model_merged_quantized.onnx",
        "decoder_with_past_model_quantized.onnx",
    ]

    os.makedirs(os.path.join(dest_dir, "onnx"), exist_ok=True)

    t0 = time.time()
    for fname in config_files:
        fpath = os.path.join(dest_dir, fname)
        if os.path.exists(fpath):
            continue
        print(f"  Downloading {fname}…")
        urllib.request.urlretrieve(f"{base_url}/{fname}", fpath)

    for fname in onnx_files:
        fpath = os.path.join(dest_dir, "onnx", fname)
        if os.path.exists(fpath):
            continue
        print(f"  Downloading onnx/{fname}…")
        urllib.request.urlretrieve(f"{base_url}/onnx/{fname}", fpath)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s.\n")


def download_tokenizer(dest_dir: str, repo_id: str) -> None:
    """Download tokenizer files from a HuggingFace model repo.

    Only downloads the files needed for tokenization — not model weights.
    """
    print(f"  repo       : {repo_id}")
    print(f"  dest_dir   : {os.path.abspath(dest_dir)}")

    base_url = f"https://huggingface.co/{repo_id}/resolve/main"
    tokenizer_files = [
        "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "vocab.json", "merges.txt",
        "config.json",
    ]

    os.makedirs(dest_dir, exist_ok=True)

    t0 = time.time()
    downloaded = 0
    for fname in tokenizer_files:
        fpath = os.path.join(dest_dir, fname)
        if os.path.exists(fpath):
            continue
        url = f"{base_url}/{fname}"
        try:
            urllib.request.urlretrieve(url, fpath)
            downloaded += 1
            print(f"  Downloaded {fname}")
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                # File requires auth, is forbidden, or doesn't exist in this repo.
                # Not all repos have all files (e.g. merges.txt is BPE-only,
                # some repos are gated). Skip silently.
                continue
            raise

    if downloaded == 0:
        print("  (all files already present or skipped)")
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s.\n")


# ─── Main ────────────────────────────────────────────────────────────────

def main() -> None:
    project_root = _resolve_project_root()
    assets_dir = _ensure_assets_dir(project_root)
    _load_dotenv()

    # Resolve defaults relative to the project root (host paths, not container paths).
    default_splade_cache   = _env_or_default("FASTEMBED_CACHE_DIR", os.path.join(assets_dir, "fastembed"))
    default_splade_model   = _env_or_default("SPLADE_MODEL", "prithivida/Splade_PP_en_v1")
    default_reranker_cache = _env_or_default("RERANKER_CACHE_DIR", os.path.join(assets_dir, "reranker"))
    default_reranker_model = _env_or_default("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-12-v2")
    default_whisper_dir    = os.path.join(assets_dir, "whisper")
    default_tokenizer_dir  = os.path.join(assets_dir, "tokenizers", "Google", "Gemma4-12B")
    default_tokenizer_repo = "google/gemma-2-2b"

    parser = argparse.ArgumentParser(description="Pre-download RAG-Web-UI model assets.")
    parser.add_argument("--cache-dir", default=default_splade_cache,
                        help=f"FastEmbed cache directory (default: {default_splade_cache})")
    parser.add_argument("--model", default=default_splade_model,
                        help=f"SPLADE model name (default: {default_splade_model})")
    parser.add_argument("--reranker-cache-dir", default=default_reranker_cache,
                        help=f"Reranker cache directory (default: {default_reranker_cache})")
    parser.add_argument("--reranker-model", default=default_reranker_model,
                        help=f"Cross-encoder model name (default: {default_reranker_model})")
    parser.add_argument("--whisper-dir", default=default_whisper_dir,
                        help=f"Whisper model directory (default: {default_whisper_dir})")
    parser.add_argument("--tokenizer-dir", default=default_tokenizer_dir,
                        help=f"Tokenizer destination directory (default: {default_tokenizer_dir})")
    parser.add_argument("--tokenizer-repo", default=default_tokenizer_repo,
                        help=f"HuggingFace repo for tokenizer (default: {default_tokenizer_repo})")
    parser.add_argument("--skip-splade", action="store_true",
                        help="Skip SPLADE model download")
    parser.add_argument("--skip-reranker", action="store_true",
                        help="Skip reranker model download")
    parser.add_argument("--skip-whisper", action="store_true",
                        help="Skip Whisper model download")
    parser.add_argument("--skip-tokenizer", action="store_true",
                        help="Skip tokenizer download")
    args = parser.parse_args()

    print("=" * 60)
    print("RAG-Web-UI asset downloader")
    print(f"  project root : {project_root}")
    print(f"  assets dir   : {assets_dir}")
    print("=" * 60)

    steps = []
    if not args.skip_splade:
        steps.append(("SPLADE sparse-embedding model (FastEmbed)",
                      lambda: download_splade(model_name=args.model, cache_dir=args.cache_dir)))
    if not args.skip_reranker:
        steps.append(("Cross-encoder reranker model (FastEmbed ONNX)",
                      lambda: download_reranker(model_name=args.reranker_model, cache_dir=args.reranker_cache_dir)))
    if not args.skip_whisper:
        steps.append(("Whisper tiny.en model (browser STT, ONNX quantized)",
                      lambda: download_whisper(dest_dir=args.whisper_dir)))
    if not args.skip_tokenizer:
        steps.append(("HuggingFace tokenizer (backend token counting)",
                      lambda: download_tokenizer(dest_dir=args.tokenizer_dir, repo_id=args.tokenizer_repo)))

    for i, (label, fn) in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] {label}")
        fn()

    print("All assets downloaded successfully.")
    print("You can now start the services — no network access needed for model loading.")


if __name__ == "__main__":
    main()
