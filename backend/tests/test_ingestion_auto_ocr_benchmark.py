"""Automatic OCR benchmark: MarkItDown vs AnyDoc on data/test PDFs.

For each file, both engines decide internally which pages/images need OCR and
use the same qwen/qwen3.5-9b-nothink vision endpoint.  The test reports:

* wall-clock time
* number of vision/OCR calls and time spent in them
* detected pages needing OCR (pdf-inspector for anydoc, markitdown-ocr itself)
* detected images above a size threshold and OCR'd by anydoc
* output length and sample

Run inside the backend container:

    docker exec rag-web-ui-backend-1 pytest tests/test_ingestion_auto_ocr_benchmark.py -s -v
"""

import base64
import io
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple

import pymupdf
import pytest
from openai import OpenAI
from PIL import Image

import anydoc
import pdf_inspector
from markitdown import MarkItDown

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

VISION_BASE_URL = os.environ.get("OPENAI_VISION_API_BASE", "http://192.168.1.3:2244/v1")
VISION_MODEL = os.environ.get("VISION_MODEL", "qwen/qwen3.5-9b")
VISION_API_KEY = os.environ.get("VISION_API_KEY", "not-required")

# Only OCR embedded raster images larger than this (pixel count).  This filters
# out the 1x1 mask objects that pdfplumber sometimes reports and keeps the
# benchmark focused on real pictures/diagrams.
MIN_IMAGE_PIXELS = 10_000

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

TEST_FILES = [
    "/app/data/test/A Crash Course in Linux Networking.pdf",
    "/app/data/test/Eavesdropping on Satellite Telecommunication Systems.pdf",
]

OUTPUT_DIR = Path("/app/data/benchmark_outputs")


class _ChatCompletions:
    """Thin wrapper that counts calls and aggregate wall time."""

    def __init__(self, real):
        self._real = real
        self.calls = 0
        self.total_seconds = 0.0

    def create(self, *args, **kwargs):
        self.calls += 1
        t0 = time.perf_counter()
        try:
            return self._real.create(*args, **kwargs)
        finally:
            self.total_seconds += time.perf_counter() - t0


class _Chat:
    def __init__(self, client):
        self.completions = _ChatCompletions(client.chat.completions)


class CountingOpenAIClient:
    """OpenAI-compatible client that records vision calls for the benchmark."""

    def __init__(self, base_url: str, api_key: str):
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.chat = _Chat(self._client)


def _vision_ocr(client: CountingOpenAIClient, image_bytes: bytes, ext: str = "png") -> str:
    """Call the vision model on an image and return extracted text."""
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
    )
    return (response.choices[0].message.content or "").strip()


def _detect_big_images(path: str) -> List[Tuple[int, int, bytes, str, int, int]]:
    """Return (page_1_indexed, image_index, bytes, ext, width, height) for real images."""
    big_images = []
    with pymupdf.open(path) as doc:
        for page_index in range(doc.page_count):
            page = doc[page_index]
            for img_index, img in enumerate(page.get_images(full=True), start=1):
                xref, *_ = img
                width, height = img[2], img[3]
                if width * height < MIN_IMAGE_PIXELS:
                    continue
                try:
                    pix = doc.extract_image(xref)
                except Exception as e:
                    logger.warning(f"Could not extract image xref {xref} on page {page_index + 1}: {e}")
                    continue
                image_bytes = pix.get("image")
                ext = pix.get("ext", "png")
                if not image_bytes:
                    continue
                big_images.append((page_index + 1, img_index, image_bytes, ext, width, height))
    return big_images


def _convert_markitdown_auto(path: str, client: CountingOpenAIClient) -> dict:
    """MarkItDown with its OCR plugin enabled (existing pipeline)."""
    t0 = time.perf_counter()
    converter = MarkItDown(
        enable_plugins=True,
        llm_client=client,
        llm_model=VISION_MODEL,
        llm_prompt=OCR_PROMPT,
    )
    result = converter.convert(path)
    elapsed = time.perf_counter() - t0
    return {
        "markdown": result.text_content or "",
        "elapsed_seconds": elapsed,
        "vision_calls": client.chat.completions.calls,
        "vision_seconds": client.chat.completions.total_seconds,
    }


def _convert_anydoc_auto(path: str, client: CountingOpenAIClient) -> dict:
    """AnyDoc + pdf-inspector + fitz image extraction."""
    t0 = time.perf_counter()

    # 1. Native anydoc conversion (fast, text-based PDFs)
    try:
        base_markdown = anydoc.to_markdown(path)
    except anydoc.UnsupportedError:
        base_markdown = ""

    base_elapsed = time.perf_counter() - t0

    # 2. Detect which pages the pdf-inspector thinks need OCR
    inspector_result = pdf_inspector.process_pdf(path)
    pages_needing_ocr = inspector_result.pages_needing_ocr or []

    # 3. Detect real embedded images
    big_images = _detect_big_images(path)
    image_ocr_parts: List[str] = []

    # 4. OCR images that are large enough to be real pictures/diagrams
    vision_seconds = 0.0
    vision_calls = 0

    def _ocr_one_image(item):
        page_index, img_index, image_bytes, ext, width, height = item
        t_ocr = time.perf_counter()
        try:
            text = _vision_ocr(client, image_bytes, ext)
        except Exception as e:
            logger.warning(f"AnyDoc image OCR failed for page {page_index} image {img_index}: {e}")
            text = ""
        return page_index, img_index, text, time.perf_counter() - t_ocr

    # Process images in parallel (up to 2 concurrent calls to the local endpoint).
    if big_images:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for page, img_idx, text, call_time in pool.map(_ocr_one_image, big_images):
                if text:
                    image_ocr_parts.append(
                        f"\n\n## OCR for image {img_idx} on page {page}\n\n{text}"
                    )

    # 5. OCR any pages that pdf-inspector says are scanned / image-only
    for page_num in pages_needing_ocr:
        with pymupdf.open(path) as doc:
            page = doc[page_num - 1]
            pix = page.get_pixmap(dpi=150)
            image_bytes = pix.tobytes("png")
        t_ocr = time.perf_counter()
        try:
            text = _vision_ocr(client, image_bytes)
        except Exception as e:
            logger.warning(f"AnyDoc page OCR failed for page {page_num}: {e}")
            text = ""
        vision_seconds += time.perf_counter() - t_ocr
        if text:
            image_ocr_parts.append(f"\n\n## OCR for page {page_num}\n\n{text}")

    elapsed = time.perf_counter() - t0
    vision_calls = client.chat.completions.calls
    vision_seconds = client.chat.completions.total_seconds

    return {
        "markdown": base_markdown + "".join(image_ocr_parts),
        "elapsed_seconds": elapsed,
        "base_elapsed_seconds": base_elapsed,
        "vision_calls": vision_calls,
        "vision_seconds": vision_seconds,
        "pages_needing_ocr": pages_needing_ocr,
        "images_detected": [
            {"page": p, "index": i, "width": w, "height": h, "pixels": w * h}
            for p, i, _, _, w, h in big_images
        ],
    }


def _metrics(text: str) -> dict:
    return {
        "chars": len(text),
        "words": len(text.split()),
        "lines": text.count("\n"),
        "headings": sum(1 for line in text.splitlines() if line.lstrip().startswith("#")),
    }


@pytest.mark.parametrize("pdf_path", TEST_FILES)
def test_ingestion_engine_benchmark(pdf_path: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    name = Path(pdf_path).stem
    logger.info(f"\n=== Benchmarking {name} ===")

    # Re-create a fresh counting client for each engine so call counts are isolated.
    markitdown_client = CountingOpenAIClient(VISION_BASE_URL, VISION_API_KEY)
    anydoc_client = CountingOpenAIClient(VISION_BASE_URL, VISION_API_KEY)

    md_result = _convert_markitdown_auto(pdf_path, markitdown_client)
    any_result = _convert_anydoc_auto(pdf_path, anydoc_client)

    # Save markdown outputs for manual inspection.
    md_out = OUTPUT_DIR / f"{name}.markitdown.md"
    ad_out = OUTPUT_DIR / f"{name}.anydoc.md"
    md_out.write_text(md_result["markdown"], encoding="utf-8")
    ad_out.write_text(any_result["markdown"], encoding="utf-8")

    md_metrics = _metrics(md_result["markdown"])
    any_metrics = _metrics(any_result["markdown"])

    report = {
        "file": pdf_path,
        "name": name,
        "markitdown": {
            **md_metrics,
            "elapsed_seconds": round(md_result["elapsed_seconds"], 3),
            "vision_calls": md_result["vision_calls"],
            "vision_seconds": round(md_result["vision_seconds"], 3),
            "output_path": str(md_out),
        },
        "anydoc": {
            **any_metrics,
            "elapsed_seconds": round(any_result["elapsed_seconds"], 3),
            "base_elapsed_seconds": round(any_result["base_elapsed_seconds"], 3),
            "vision_calls": any_result["vision_calls"],
            "vision_seconds": round(any_result["vision_seconds"], 3),
            "pages_needing_ocr": any_result["pages_needing_ocr"],
            "images_detected": any_result["images_detected"],
            "output_path": str(ad_out),
        },
    }

    report_path = OUTPUT_DIR / f"{name}.report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info(f"  MarkItDown : {md_result['elapsed_seconds']:.3f}s, "
                f"{md_result['vision_calls']} vision calls, "
                f"{md_metrics['chars']} chars")
    logger.info(f"  AnyDoc     : {any_result['elapsed_seconds']:.3f}s, "
                f"{any_result['vision_calls']} vision calls, "
                f"{any_metrics['chars']} chars")
    logger.info(f"  Report     : {report_path}")

    # Expose the report as pytest output for easy reading.
    print(json.dumps(report, indent=2))
