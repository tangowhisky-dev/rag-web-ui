#!/usr/bin/env python3
"""Benchmark 3 vision/OCR models on PDFs and images from data/tech/comm.

Tests:
  1. Speed: wall-clock time per OCR call
  2. Accuracy: text extraction quality (char count, keyword presence, structure)
  3. API compatibility: standard OpenAI chat completions with image_url content
  4. Interchangeability: can they be swapped without code changes?

Models tested:
  - qwen/qwen3.5-9b  (general VLM with OCR capability)
  - zai/glm-ocr      (dedicated OCR model)
  - deepseek/ocr-2   (dedicated OCR model)

Test files from data/tech/comm:
  - PDFs (text-based + scanned/image-based)
  - Images (JPEG, PNG, screenshots)

Run inside the backend container:
  docker exec rag-web-ui-backend-1 python3 /app/tests/test_vision_ocr_benchmark.py
"""
import base64
import io
import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pymupdf
from openai import OpenAI
from PIL import Image

sys.path.insert(0, "/app")
os.environ.setdefault("PYTHONPATH", "/app")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("ocr_benchmark")

# ─── Config ──────────────────────────────────────────────────────────────────

VISION_BASE_URL = os.environ.get("VISION_BENCHMARK_BASE_URL", "http://192.168.1.21:4321/v1")
VISION_API_KEY = os.environ.get("VISION_BENCHMARK_API_KEY", "not-required")

MODELS = [
    "qwen/qwen3.5-9b",
    "zai/glm-ocr",
    "deepseek/ocr-2",
]

# The exact OCR prompt used by the production ingestion pipeline
# (from app/services/ingestion/document_converter.py)
OCR_PROMPT = (
    "Extract all text from this image into clean, naturally flowing paragraphs, "
    "while preserving document structure and any table or sub-element layout.\n\n"
    "Rules:\n"
    "- Remove unnatural line breaks within sentences\n"
    "- Join split words and sentences caused by column layout or line wrapping\n"
    "- Keep proper paragraph breaks where the topic clearly changes\n"
    "- Preserve tables using Markdown table syntax\n"
    "- Preserve all original meaning and technical terms exactly\n"
    "- Output only the extracted text, no explanations or commentary"
)

# ─── Test files ──────────────────────────────────────────────────────────────

DATA_DIR = "/app/data/tech/comm"

def collect_test_files() -> Dict[str, List[str]]:
    """Collect test files grouped by type."""
    files = {"images": [], "pdfs": []}
    for root, dirs, filenames in os.walk(DATA_DIR):
        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            path = os.path.join(root, f)
            if ext in (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"):
                files["images"].append(path)
            elif ext == ".pdf":
                files["pdfs"].append(path)
    return files


def render_pdf_page_to_image(pdf_path: str, page_num: int = 1, dpi: int = 150) -> bytes:
    """Render a PDF page to PNG bytes (same as production _ocr_pdf_page)."""
    with pymupdf.open(pdf_path) as doc:
        page = doc[page_num - 1]
        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png")


def get_pdf_page_count(pdf_path: str) -> int:
    with pymupdf.open(pdf_path) as doc:
        return doc.page_count


def get_pdf_text_length(pdf_path: str) -> int:
    """Get the text length extractable without OCR (for comparison)."""
    with pymupdf.open(pdf_path) as doc:
        total = 0
        for page in doc:
            total += len(page.get_text())
        return total


# ─── OCR call (same as production _ocr_image_bytes) ─────────────────────────

def ocr_image(client: OpenAI, model: str, image_bytes: bytes, mime_type: str = "image/png") -> Tuple[str, float]:
    """Send image to vision model, return (text, elapsed_seconds).
    
    Uses the exact same API call pattern as the production pipeline:
    client.chat.completions.create with image_url content type.
    """
    # Normalize image to PNG (same as production)
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")

    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": OCR_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
        max_tokens=2000,
        temperature=0.0,
    )
    elapsed = time.perf_counter() - t0
    text = (resp.choices[0].message.content or "").strip()
    return text, elapsed


# ─── Accuracy scoring ────────────────────────────────────────────────────────

def score_text(text: str, expected_keywords: List[str] = None) -> Dict:
    """Score the OCR output on basic quality metrics."""
    if not text:
        return {
            "char_count": 0,
            "word_count": 0,
            "line_count": 0,
            "has_structure": False,
            "keyword_hits": 0,
            "keyword_total": 0,
            "looks_garbled": True,
        }
    
    words = text.split()
    lines = text.split("\n")
    
    # Check for markdown structure (headers, tables, lists)
    has_structure = any(
        line.startswith("#") or "|" in line or line.startswith("- ") or line.startswith("* ")
        for line in lines
    )
    
    # Check for garbled output (excessive repeated chars, non-printable)
    garbled_chars = sum(1 for c in text if ord(c) < 32 and c not in "\n\r\t")
    looks_garbled = garbled_chars > len(text) * 0.05
    
    # Keyword matching
    keyword_hits = 0
    keyword_total = 0
    if expected_keywords:
        text_lower = text.lower()
        keyword_total = len(expected_keywords)
        keyword_hits = sum(1 for kw in expected_keywords if kw.lower() in text_lower)
    
    return {
        "char_count": len(text),
        "word_count": len(words),
        "line_count": len(lines),
        "has_structure": has_structure,
        "keyword_hits": keyword_hits,
        "keyword_total": keyword_total,
        "looks_garbled": looks_garbled,
    }


# ─── Benchmark runner ────────────────────────────────────────────────────────

def run_benchmark():
    files = collect_test_files()
    
    print(f"\n{'=' * 80}")
    print("Vision/OCR Model Benchmark")
    print(f"{'=' * 80}")
    print(f"\nEndpoint: {VISION_BASE_URL}")
    print(f"Models: {MODELS}")
    print(f"\nTest files:")
    print(f"  Images: {len(files['images'])}")
    for f in files["images"]:
        print(f"    - {os.path.basename(f)} ({os.path.getsize(f) // 1024}KB)")
    print(f"  PDFs: {len(files['pdfs'])}")
    for f in files["pdfs"]:
        pages = get_pdf_page_count(f)
        text_len = get_pdf_text_length(f)
        print(f"    - {os.path.basename(f)} ({pages} pages, {os.path.getsize(f) // 1024}KB, native_text={text_len} chars)")
    
    # Prepare test images: all images + first page of each PDF
    test_images = []
    for img_path in files["images"]:
        with open(img_path, "rb") as f:
            test_images.append({
                "name": os.path.basename(img_path),
                "source": img_path,
                "type": "image",
                "bytes": f.read(),
            })
    
    for pdf_path in files["pdfs"]:
        pages = get_pdf_page_count(pdf_path)
        text_len = get_pdf_text_length(pdf_path)
        # Test first page and (if multi-page) a middle page
        pages_to_test = [1]
        if pages > 2:
            pages_to_test.append(pages // 2 + 1)
        for pn in pages_to_test:
            try:
                img_bytes = render_pdf_page_to_image(pdf_path, pn)
                test_images.append({
                    "name": f"{os.path.basename(pdf_path)} [page {pn}]",
                    "source": pdf_path,
                    "type": "pdf_page",
                    "page": pn,
                    "bytes": img_bytes,
                    "pdf_has_native_text": text_len > 100,
                })
            except Exception as e:
                print(f"  WARNING: could not render page {pn} of {pdf_path}: {e}")
    
    print(f"\nTotal test images: {len(test_images)}")
    
    # Run each model on each image
    results = []
    
    for model in MODELS:
        print(f"\n{'─' * 80}")
        print(f"Model: {model}")
        print(f"{'─' * 80}")
        
        client = OpenAI(api_key=VISION_API_KEY, base_url=VISION_BASE_URL)
        
        model_results = []
        total_time = 0
        success_count = 0
        error_count = 0
        
        for img in test_images:
            name = img["name"]
            img_bytes = img["bytes"]
            img_size_kb = len(img_bytes) // 1024
            
            print(f"\n  OCR: {name} ({img_size_kb}KB)")
            
            try:
                text, elapsed = ocr_image(client, model, img_bytes)
                total_time += elapsed
                success_count += 1
                
                scores = score_text(text)
                
                # Print preview
                preview = text[:200].replace("\n", " ↵ ")
                print(f"    Time: {elapsed:.2f}s")
                print(f"    Chars: {scores['char_count']}, Words: {scores['word_count']}, Lines: {scores['line_count']}")
                print(f"    Structure: {scores['has_structure']}, Garbled: {scores['looks_garbled']}")
                print(f"    Preview: {preview}...")
                
                model_results.append({
                    "model": model,
                    "image": name,
                    "type": img["type"],
                    "size_kb": img_size_kb,
                    "elapsed_s": round(elapsed, 2),
                    "char_count": scores["char_count"],
                    "word_count": scores["word_count"],
                    "line_count": scores["line_count"],
                    "has_structure": scores["has_structure"],
                    "looks_garbled": scores["looks_garbled"],
                    "text_preview": text[:500],
                    "text_full": text,
                    "success": True,
                })
            except Exception as e:
                error_count += 1
                print(f"    ERROR: {e}")
                model_results.append({
                    "model": model,
                    "image": name,
                    "type": img["type"],
                    "size_kb": img_size_kb,
                    "elapsed_s": 0,
                    "char_count": 0,
                    "word_count": 0,
                    "line_count": 0,
                    "has_structure": False,
                    "looks_garbled": True,
                    "text_preview": "",
                    "text_full": "",
                    "success": False,
                    "error": str(e),
                })
        
        avg_time = total_time / success_count if success_count else 0
        print(f"\n  Summary for {model}:")
        print(f"    Success: {success_count}/{len(test_images)}")
        print(f"    Errors: {error_count}")
        print(f"    Total time: {total_time:.2f}s")
        print(f"    Avg time per image: {avg_time:.2f}s")
        
        results.extend(model_results)
    
    # ─── Comparison table ───────────────────────────────────────────────────
    print(f"\n\n{'=' * 80}")
    print("COMPARISON TABLE")
    print(f"{'=' * 80}\n")
    
    # Per-model summary
    print(f"{'Model':<25} {'Success':>8} {'Avg Time':>10} {'Avg Chars':>10} {'Avg Words':>10} {'Structure':>10}")
    print("-" * 75)
    for model in MODELS:
        model_results = [r for r in results if r["model"] == model]
        success = sum(1 for r in model_results if r["success"])
        times = [r["elapsed_s"] for r in model_results if r["success"]]
        chars = [r["char_count"] for r in model_results if r["success"]]
        words = [r["word_count"] for r in model_results if r["success"]]
        structure = sum(1 for r in model_results if r["has_structure"])
        avg_time = sum(times) / len(times) if times else 0
        avg_chars = sum(chars) / len(chars) if chars else 0
        avg_words = sum(words) / len(words) if words else 0
        print(f"{model:<25} {success:>8} {avg_time:>9.2f}s {avg_chars:>10.0f} {avg_words:>10.0f} {structure:>10}")
    
    # Per-image comparison
    print(f"\n\nPer-image comparison:")
    print(f"{'Image':<45} {'Model':<25} {'Time':>7} {'Chars':>7} {'Words':>7} {'Struct':>7}")
    print("-" * 100)
    for img in test_images:
        for model in MODELS:
            matching = [r for r in results if r["model"] == model and r["image"] == img["name"]]
            if matching:
                r = matching[0]
                status = "✓" if r["success"] else "✗"
                print(f"{img['name'][:44]:<45} {model:<25} {r['elapsed_s']:>6.2f}s {r['char_count']:>7} {r['word_count']:>7} {str(r['has_structure']):>7} {status}")
        print()
    
    # ─── API compatibility check ────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("API COMPATIBILITY CHECK")
    print(f"{'=' * 80}\n")
    
    for model in MODELS:
        model_results = [r for r in results if r["model"] == model]
        success = sum(1 for r in model_results if r["success"])
        total = len(model_results)
        api_compatible = success == total
        print(f"  {model}:")
        print(f"    OpenAI chat completions API: {'✓ Compatible' if api_compatible else '✗ Issues'}")
        print(f"    image_url content type: {'✓ Supported' if api_compatible else '✗ Not supported'}")
        print(f"    base64 data URI: {'✓ Supported' if api_compatible else '✗ Not supported'}")
        print(f"    Success rate: {success}/{total}")
        print(f"    Interchangeable: {'✓ Yes' if api_compatible else '✗ No — requires code changes'}")
        print()
    
    # Save results
    output_path = "/app/assets/ocr_benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "endpoint": VISION_BASE_URL,
            "models": MODELS,
            "test_images": len(test_images),
            "results": results,
        }, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run_benchmark()
