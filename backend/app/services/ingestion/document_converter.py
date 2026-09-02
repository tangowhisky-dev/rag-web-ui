"""Document conversion — swappable markdown engines with automatic OCR.

Public API (unchanged from the previous markitdown-only module):

    _convert_to_markdown(abs_path, file_name, enable_ocr=None) -> str
    SUPPORTED_EXTENSIONS, CONTENT_TYPE_MAP, MAX_FILE_SIZE

Engine selection is driven by the ``MARKDOWN_ENGINE`` setting
(default ``"anydoc"``).  To swap engines in the future, add a new
``MarkdownEngine`` subclass and register it in ``_ENGINES``.

enable_ocr semantics (tri-state, same as before):
    None  → use global VISION_MODEL setting
    True  → force OCR on  (requires VISION_MODEL)
    False → force OCR off for this document
"""

import base64
import io
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Protocol

from openai import OpenAI as SyncOpenAI

from app.services.infrastructure import strip_reasoning_tags

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# ── Extension / content-type maps ──────────────────────────────────────────

SUPPORTED_EXTENSIONS = {
    # Office documents (anydoc)
    ".pdf", ".docx", ".doc", ".docm",
    ".pptx", ".ppt", ".pps", ".pot", ".pptm", ".ppsx", ".ppsm",
    ".xlsx", ".xls", ".xlsm", ".xlsb",
    ".odt", ".ods", ".odp",
    ".rtf", ".epub",
    # Text / markup (direct read)
    ".txt", ".md", ".html", ".htm",
    ".csv", ".json", ".xml",
    # Email (stdlib email module — handles both .eml and .mhtml)
    ".eml", ".mhtml",
    # Images (vision OCR)
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp",
}

CONTENT_TYPE_MAP = {
    ".pdf":   "application/pdf",
    ".docx":  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc":   "application/msword",
    ".docm":  "application/vnd.ms-word.document.macroEnabled.12",
    ".pptx":  "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".ppt":   "application/vnd.ms-powerpoint",
    ".pps":   "application/vnd.ms-powerpoint.slideshow",
    ".pot":   "application/vnd.ms-powerpoint.template",
    ".pptm":  "application/vnd.ms-powerpoint.presentation.macroEnabled.12",
    ".ppsx":  "application/vnd.openxmlformats-officedocument.presentationml.slideshow",
    ".ppsm":  "application/vnd.ms-powerpoint.slideshow.macroEnabled.12",
    ".xlsx":  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls":   "application/vnd.ms-excel",
    ".xlsm":  "application/vnd.ms-excel.sheet.macroEnabled.12",
    ".xlsb":  "application/vnd.ms-excel.sheet.binary.macroEnabled.12",
    ".odt":   "application/vnd.oasis.opendocument.text",
    ".ods":   "application/vnd.oasis.opendocument.spreadsheet",
    ".odp":   "application/vnd.oasis.opendocument.presentation",
    ".rtf":   "application/rtf",
    ".epub":  "application/epub+zip",
    ".txt":   "text/plain",
    ".md":    "text/markdown",
    ".html":  "text/html",
    ".htm":   "text/html",
    ".mhtml": "application/x-mimearchive",
    ".csv":   "text/csv",
    ".json":  "application/json",
    ".xml":   "application/xml",
    ".eml":   "message/rfc822",
    ".jpg":   "image/jpeg",
    ".jpeg":  "image/jpeg",
    ".png":   "image/png",
    ".gif":   "image/gif",
    ".bmp":   "image/bmp",
    ".tiff":  "image/tiff",
    ".webp":  "image/webp",
}

# Extensions handled by anydoc natively
_ANYDOC_EXTENSIONS = {
    ".doc", ".docx", ".docm",
    ".ppt", ".pps", ".pot", ".pptx", ".pptm", ".ppsx", ".ppsm",
    ".xls", ".xlsx", ".xlsm", ".xlsb",
    ".odt", ".ods", ".odp",
    ".rtf", ".epub", ".csv",
}

# Extensions read directly as UTF-8 text
_TEXT_EXTENSIONS = {".txt", ".md", ".json", ".xml"}

# Extensions that need HTML-to-text conversion
_HTML_EXTENSIONS = {".html", ".htm"}

# Email / MHTML (stdlib email module)
_EMAIL_EXTENSIONS = {".eml", ".mhtml"}

# Image extensions (OCR via vision model)
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}

# Minimum pixel count for an embedded PDF image to be considered worth OCR'ing.
# Filters out 1×1 mask objects and decorative dots.
_MIN_IMAGE_PIXELS = 10_000

# OCR prompt reused across engines
# Previous prompt (7-rule English instruction) — designed for general vision LLMs
# (GPT-4V, Qwen-VL). Caused hallucinations on GLM-OCR because the model is a
# specialized 0.9B OCR model trained on fixed prompts, not a general LLM that
# follows free-form instructions. Kept for reference and potential model-aware
# fallback in the future.
#
# _OCR_PROMPT_LEGACY = (
#     "Extract all text from this image into clean, naturally flowing paragraphs, "
#     "while preserving document structure and any table or sub-element layout.\n\n"
#     "Rules:\n"
#     "- Remove unnatural line breaks within sentences\n"
#     "- Join split words and sentences caused by column layout or line wrapping\n"
#     "- Keep proper paragraph breaks where the topic clearly changes\n"
#     "- Preserve tables using Markdown table syntax\n"
#     "- Preserve all original meaning and technical terms exactly\n"
#     "- Output only the extracted text, no explanations or commentary"
# )
#
# GLM-OCR official prompt — the model was trained on exactly this prompt
# for text recognition. See https://huggingface.co/zai-org/GLM-OCR
# Using any other prompt is out-of-distribution and can cause hallucinations.
_OCR_PROMPT = "Text Recognition:"


# ── Vision config (shared across engines) ──────────────────────────────────

@dataclass
class VisionConfig:
    model: str
    client: SyncOpenAI
    max_tokens: Optional[int] = None


_vision_config: Optional[VisionConfig] = None
_vision_lock = threading.Lock()


def _get_vision_config() -> Optional[VisionConfig]:
    """Lazy singleton for vision/OCR config, read from DB settings.

    Returns None when VISION_MODEL is not set (OCR disabled globally).
    """
    global _vision_config
    if _vision_config is not None:
        return _vision_config
    with _vision_lock:
        if _vision_config is not None:
            return _vision_config
        from app.services.settings_service import get_setting
        from app.db.session import SessionLocal
        db = SessionLocal()
        try:
            vision_model = get_setting(db, "VISION_MODEL", None)
            if not vision_model:
                logger.debug("[converter] OCR disabled — VISION_MODEL not set")
                return None
            api_key = (
                get_setting(db, "VISION_API_KEY", None)
                or get_setting(db, "OPENAI_API_KEY", None)
                or "not-required"
            )
            api_base = (
                get_setting(db, "OPENAI_VISION_API_BASE", None)
                or get_setting(db, "OPENAI_API_BASE", None)
            )
            if not api_base:
                logger.warning("[converter] VISION_MODEL set but no API base — OCR disabled")
                return None
            max_tokens = int(get_setting(db, "VISION_MAX_TOKENS", None) or 4096)
            _vision_config = VisionConfig(
                model=vision_model,
                client=SyncOpenAI(api_key=api_key, base_url=api_base),
                max_tokens=max_tokens,
            )
            logger.debug("[converter] OCR enabled — vision_model=%s base=%s max_tokens=%d", vision_model, api_base, max_tokens)
            return _vision_config
        finally:
            db.close()


def _resolve_ocr(enable_ocr: Optional[bool]) -> Optional[VisionConfig]:
    """Resolve the tri-state enable_ocr flag to a VisionConfig or None."""
    if enable_ocr is False:
        return None
    config = _get_vision_config()
    if enable_ocr is True and not config:
        logger.warning("[converter] OCR requested but VISION_MODEL not set — falling back to text-only")
    return config


# ── OCR primitives ─────────────────────────────────────────────────────────

def _ocr_image_bytes(image_bytes: bytes, config: VisionConfig) -> str:
    """Send an image to the vision model and return extracted text."""
    from PIL import Image
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    resp = config.client.chat.completions.create(
        model=config.model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": _OCR_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
        max_tokens=config.max_tokens or 4096,
    )
    return (resp.choices[0].message.content or "").strip()


def _ocr_pdf_page(pdf_path: str, page_num_1indexed: int, config: VisionConfig) -> str:
    """Render a PDF page to an image and OCR it."""
    import pymupdf
    with pymupdf.open(pdf_path) as doc:
        page = doc[page_num_1indexed - 1]
        pix = page.get_pixmap(dpi=150)
        image_bytes = pix.tobytes("png")
    return _ocr_image_bytes(image_bytes, config)


def _detect_pdf_images(pdf_path: str) -> list:
    """Return [(page_1indexed, img_index, image_bytes, ext, width, height)]
    for embedded raster images above _MIN_IMAGE_PIXELS."""
    import pymupdf
    big_images = []
    with pymupdf.open(pdf_path) as doc:
        for page_idx in range(doc.page_count):
            page = doc[page_idx]
            for img_idx, img in enumerate(page.get_images(full=True), start=1):
                width, height = img[2], img[3]
                if width * height < _MIN_IMAGE_PIXELS:
                    continue
                try:
                    pix = doc.extract_image(img[0])
                except Exception as e:
                    logger.warning("[converter] could not extract image xref %s on page %d: %s", img[0], page_idx + 1, e)
                    continue
                image_bytes = pix.get("image")
                if not image_bytes:
                    continue
                big_images.append((page_idx + 1, img_idx, image_bytes, pix.get("ext", "png"), width, height))
    return big_images


# ── Engine protocol ────────────────────────────────────────────────────────

class MarkdownEngine(Protocol):
    """Swappable markdown conversion engine."""

    def convert(self, abs_path: str, file_name: str, enable_ocr: Optional[bool],
                progress_cb: Optional[callable] = None) -> str:
        ...


# ── AnyDoc engine (default) ────────────────────────────────────────────────

class AnydocEngine:
    """anydoc + pdf-inspector + fitz image extraction + vision OCR.

    PDF:  pdf-inspector for text extraction + per-page OCR routing.
          Embedded raster images above a size threshold are OCR'd too.
    Office docs (docx, pptx, xlsx, odt, rtf, epub, csv, …): anydoc.
    Images (jpg, png, webp, …): vision model OCR.
    Text (txt, md, json, xml): direct UTF-8 read.
    HTML: tag-stripped to text via BeautifulSoup.
    EML / MHTML: stdlib email module → extract text/plain and text/html parts.
    """

    name = "anydoc"

    def convert(self, abs_path: str, file_name: str, enable_ocr: Optional[bool],
                progress_cb: Optional[callable] = None) -> str:
        ext = os.path.splitext(abs_path)[1].lower()

        if ext == ".pdf":
            markdown = self._convert_pdf(abs_path, enable_ocr, progress_cb=progress_cb)
        elif ext in _IMAGE_EXTENSIONS:
            markdown = self._convert_image(abs_path, enable_ocr)
        elif ext in _ANYDOC_EXTENSIONS:
            markdown = self._convert_anydoc(abs_path)
        elif ext in _TEXT_EXTENSIONS:
            markdown = self._read_text(abs_path)
        elif ext in _HTML_EXTENSIONS:
            markdown = self._convert_html(abs_path)
        elif ext in _EMAIL_EXTENSIONS:
            markdown = self._convert_email(abs_path)
        else:
            # Unknown extension — raw text fallback
            markdown = self._read_text(abs_path)

        cleaned = strip_reasoning_tags(markdown)
        if len(cleaned) < len(markdown):
            logger.debug("[anydoc] stripped %d chars of reasoning tags from %s",
                        len(markdown) - len(cleaned), file_name)
        logger.debug("[anydoc] converted %s → %d chars of markdown (ocr=%s)",
                    file_name, len(cleaned), enable_ocr is not False)
        return cleaned

    # ── PDF ────────────────────────────────────────────────────────────────

    def _convert_pdf(self, abs_path: str, enable_ocr: Optional[bool],
                     progress_cb: Optional[callable] = None) -> str:
        import pdf_inspector
        result = pdf_inspector.process_pdf(abs_path)
        markdown = result.markdown or ""

        config = _resolve_ocr(enable_ocr)
        if not config:
            return markdown

        ocr_parts = []

        # 1. OCR pages flagged by pdf-inspector as scanned/image-based
        pages_needing_ocr = result.pages_needing_ocr or []
        if pages_needing_ocr:
            logger.debug("[anydoc] OCR'ing %d scanned pages in %s",
                        len(pages_needing_ocr), os.path.basename(abs_path))
            for page_num in pages_needing_ocr:
                try:
                    text = _ocr_pdf_page(abs_path, page_num, config)
                    if text:
                        ocr_parts.append(f"\n\n## OCR for page {page_num}\n\n{text}")
                except Exception as e:
                    logger.warning("[anydoc] page %d OCR failed: %s", page_num, e)
                # Ping progress between OCR pages so the silence timeout
                # doesn't kill long-running multi-page PDFs.
                if progress_cb:
                    progress_cb(5, f"OCR page {page_num}/{len(pages_needing_ocr)}…")

        # 2. OCR embedded raster images (diagrams, figures)
        big_images = _detect_pdf_images(abs_path)
        if big_images:
            logger.debug("[anydoc] OCR'ing %d embedded images in %s",
                        len(big_images), os.path.basename(abs_path))

            def _ocr_img(item):
                page, idx, img_bytes, ext, w, h = item
                try:
                    return page, idx, _ocr_image_bytes(img_bytes, config)
                except Exception as e:
                    logger.warning("[anydoc] image OCR failed page %d img %d: %s", page, idx, e)
                    return page, idx, ""

            with ThreadPoolExecutor(max_workers=2) as pool:
                for page, idx, text in pool.map(_ocr_img, big_images):
                    if text:
                        ocr_parts.append(f"\n\n## OCR for image {idx} on page {page}\n\n{text}")

        return markdown + "".join(ocr_parts)

    # ── Office docs ────────────────────────────────────────────────────────

    def _convert_anydoc(self, abs_path: str) -> str:
        import anydoc
        try:
            return anydoc.to_markdown(abs_path)
        except anydoc.UnsupportedError as e:
            logger.warning("[anydoc] unsupported format for %s: %s — trying raw text", abs_path, e)
            return self._read_text(abs_path)
        except anydoc.ConvertError as e:
            logger.warning("[anydoc] conversion failed for %s: %s — trying raw text", abs_path, e)
            return self._read_text(abs_path)

    # ── Images ─────────────────────────────────────────────────────────────

    def _convert_image(self, abs_path: str, enable_ocr: Optional[bool]) -> str:
        config = _resolve_ocr(enable_ocr)
        if not config:
            return ""
        try:
            with open(abs_path, "rb") as f:
                image_bytes = f.read()
            return _ocr_image_bytes(image_bytes, config)
        except Exception as e:
            logger.warning("[anydoc] image OCR failed for %s: %s", abs_path, e)
            return ""

    # ── Text / HTML / Email ────────────────────────────────────────────────

    def _read_text(self, abs_path: str) -> str:
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception:
            return ""

    def _convert_html(self, abs_path: str) -> str:
        try:
            from bs4 import BeautifulSoup
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
            # Remove script/style blocks
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
            # Collapse excessive blank lines
            lines = [line.strip() for line in text.splitlines()]
            return "\n".join(line for line in lines if line)
        except Exception:
            return self._read_text(abs_path)

    def _convert_email(self, abs_path: str) -> str:
        """Parse .eml and .mhtml files via the stdlib email module.

        Extracts text/plain parts directly, and converts text/html parts
        to plain text via BeautifulSoup. Skips binary attachments (images,
        application/* parts).
        """
        import email
        try:
            with open(abs_path, "rb") as f:
                msg = email.message_from_binary_file(f)
            parts = []
            subject = msg.get("Subject", "")
            if subject:
                parts.append(f"# {subject}")
            for header in ("From", "To", "Date"):
                val = msg.get(header, "")
                if val:
                    parts.append(f"**{header}:** {val}")
            parts.append("")

            def _extract_part(part) -> None:
                ctype = part.get_content_type()
                payload = part.get_payload(decode=True)
                if not payload:
                    return
                text = payload.decode("utf-8", errors="replace")
                if ctype == "text/plain":
                    parts.append(text)
                elif ctype == "text/html":
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(text, "html.parser")
                    for tag in soup(["script", "style"]):
                        tag.decompose()
                    html_text = soup.get_text(separator="\n")
                    lines = [line.strip() for line in html_text.splitlines()]
                    cleaned = "\n".join(line for line in lines if line)
                    if cleaned:
                        parts.append(cleaned)

            if msg.is_multipart():
                for part in msg.walk():
                    if part.is_multipart():
                        continue
                    _extract_part(part)
            else:
                _extract_part(msg)

            return "\n".join(parts)
        except Exception:
            return self._read_text(abs_path)


# ── MarkItDown engine (legacy, kept for swap) ──────────────────────────────

class MarkitdownEngine:
    """Legacy markitdown engine with markitdown-ocr plugin.

    Kept so the engine can be swapped back via the MARKDOWN_ENGINE setting
    without code changes.
    """

    name = "markitdown"

    def __init__(self) -> None:
        from markitdown import MarkItDown
        self._MarkItDown = MarkItDown
        self._singleton = None
        self._lock = threading.Lock()

    def _get_singleton(self) -> "object":
        if self._singleton is not None:
            return self._singleton
        with self._lock:
            if self._singleton is not None:
                return self._singleton
            from app.services.settings_service import get_setting
            from app.db.session import SessionLocal
            db = SessionLocal()
            try:
                vision_model = get_setting(db, "VISION_MODEL", None)
                if vision_model:
                    api_key = (
                        get_setting(db, "VISION_API_KEY", None)
                        or get_setting(db, "OPENAI_API_KEY", None)
                        or "not-required"
                    )
                    api_base = (
                        get_setting(db, "OPENAI_VISION_API_BASE", None)
                        or get_setting(db, "OPENAI_API_BASE", None)
                    )
                    vision_client = SyncOpenAI(api_key=api_key, base_url=api_base)
                    self._singleton = self._MarkItDown(
                        enable_plugins=True,
                        llm_client=vision_client,
                        llm_model=vision_model,
                        llm_prompt=_OCR_PROMPT,
                    )
                    logger.debug("[markitdown] OCR enabled — vision_model=%s base=%s", vision_model, api_base)
                else:
                    self._singleton = self._MarkItDown()
                    logger.debug("[markitdown] OCR disabled — VISION_MODEL not set")
            finally:
                db.close()
            return self._singleton

    def convert(self, abs_path: str, file_name: str, enable_ocr: Optional[bool],
                progress_cb: Optional[callable] = None) -> str:
        if enable_ocr is False:
            md_instance = self._MarkItDown()
            logger.debug("[markitdown] OCR disabled for this document (per-document override)")
        elif enable_ocr is True:
            md_instance = self._get_singleton()
            config = _get_vision_config()
            if not config:
                logger.warning("[markitdown] OCR requested but VISION_MODEL not set — falling back to text-only")
        else:
            md_instance = self._get_singleton()

        result = md_instance.convert(abs_path)
        markdown_text = result.text_content or ""
        cleaned = strip_reasoning_tags(markdown_text)
        if len(cleaned) < len(markdown_text):
            logger.debug("[markitdown] stripped %d chars of reasoning tags from %s",
                        len(markdown_text) - len(cleaned), file_name)
        logger.debug("[markitdown] converted %s → %d chars of markdown (ocr=%s)",
                    file_name, len(cleaned), enable_ocr is not False)
        return cleaned


# ── Engine registry + public API ───────────────────────────────────────────

_ENGINES = {
    "anydoc": AnydocEngine,
    "markitdown": MarkitdownEngine,
}

_engine_instance: Optional[MarkdownEngine] = None
_engine_lock = threading.Lock()


def _get_engine() -> MarkdownEngine:
    """Return the active engine, selected by the MARKDOWN_ENGINE setting."""
    global _engine_instance
    if _engine_instance is not None:
        return _engine_instance
    with _engine_lock:
        if _engine_instance is not None:
            return _engine_instance
        from app.services.settings_service import get_setting
        from app.db.session import SessionLocal
        db = SessionLocal()
        try:
            engine_name = get_setting(db, "MARKDOWN_ENGINE", None) or "anydoc"
        finally:
            db.close()
        engine_cls = _ENGINES.get(engine_name, AnydocEngine)
        _engine_instance = engine_cls()
        logger.debug("[converter] active engine: %s", _engine_instance.name)
        return _engine_instance


# ── Title extraction ────────────────────────────────────────────────────────
#
# Multi-signal cascade inspired by Google's title-link generation, SciPlore
# Xtract (font-size heuristic), and Docear's PDF Inspector (largest text on
# first page). See:
#   - https://developers.google.com/search/docs/appearance/title-link
#   - https://docear.org/papers/SciPlore%20Xtract%20--%20Extracting%20Titles%20from%20Scientific%20PDF%20Documents%20by%20Analyzing%20Style%20Information%20(Font%20Size)-preprint.pdf
#   - https://dl.acm.org/doi/10.1145/2467696.2467789

import re as _re

_H1_RE = _re.compile(r"^#\s+(.+?)\s*$", _re.MULTILINE)
_H2_RE = _re.compile(r"^##\s+(.+?)\s*$", _re.MULTILINE)
_PDF_TITLE_EXTS = {".pdf"}
_DOCX_TITLE_EXTS = {".docx", ".docm"}
_HTML_TITLE_EXTS = {".html", ".htm", ".xhtml"}

# Boilerplate patterns that are not document titles.
# Matches lines like "Page 1 of 10", "Confidential", "DRAFT v2", dates, etc.
_BOILERPLATE_RE = _re.compile(
    r"^("
    r"page\s+\d+\s*(of\s+\d+)?"
    r"|confidential"
    r"|draft\b"
    r"|version\s+[\d.]+"
    r"|v\s*[\d.]+"
    r"|\d{1,2}[/\-\.]\d{1,2}([/\-\.]\d{2,4})?"  # dates
    r"|\d{4}[/\-\.]\d{1,2}([/\-\.]\d{1,2})?"     # ISO dates
    r"|^\s*\d+\s*$"                               # just a number
    r"|untitle[d]?"
    r"|^\s*$"
    # Web page navigation / UI elements (common in web-to-PDF and saved pages)
    r"|sign\s*in"
    r"|log\s*in"
    r"|log\s*out"
    r"|sign\s*up"
    r"|sign\s*out"
    r"|register"
    r"|subscribe"
    r"|unsubscribe"
    r"|search"
    r"|menu"
    r"|write\b"
    r"|read\b"
    r"|share"
    r"|save\b"
    r"|print"
    r"|download"
    r"|skip\s+to\s+(main\s+)?content"
    r"|skip\s+navigation"
    r"|home\b"
    r"|about\s+us"
    r"|contact\s+us"
    r"|all\s+rights\s+reserved"
    r"|privacy\s+policy"
    r"|terms\s+of\s+(service|use)"
    r"|cookie\s+(policy|preferences)"
    # Document converter watermarks
    r"|translated\s+from\b"
    r"|converted\s+from\b"
    r"|created\s+with\b"
    r"|generated\s+by\b"
    r")",
    _re.IGNORECASE,
)


def _clean_filename_to_title(file_name: str) -> str:
    """Last-resort title: strip extension, replace separators with spaces."""
    stem = os.path.splitext(file_name)[0]
    stem = stem.replace("_", " ").replace("-", " ").replace(".", " ")
    stem = _re.sub(r"\s+", " ", stem).strip()
    return stem[:512] if stem else file_name


def _is_boilerplate(line: str) -> bool:
    """Check if a line is boilerplate (page numbers, dates, 'Confidential', etc.)."""
    return bool(_BOILERPLATE_RE.match(line.strip()))


def _clean_title(title: str) -> str:
    """Strip markdown formatting, trailing page numbers, and whitespace."""
    # Strip markdown emphasis and code markers
    title = _re.sub(r"\*+|_+|`+", "", title).strip()
    # Strip trailing page numbers like "Page 3" or "- 3"
    title = _re.sub(r"\s*[-–—]\s*\d+\s*$", "", title)
    title = _re.sub(r"\s*[|]\s*\d+\s*$", "", title)
    # Strip common title prefixes
    title = _re.sub(r"^(title|document|report)\s*[:\-–]\s*", "", title, flags=_re.IGNORECASE)
    return title.strip()[:512]


def _extract_title_from_markdown_headings(markdown_text: str) -> str | None:
    """Extract title from the first H1 or H2 heading in markdown.

    This is a strong signal — explicit headings are reliable title indicators.
    Used for PDFs where the font heuristic failed (OCR'd text is noisy and
    the first-line fallback would pick up random words).
    """
    # 1. First H1 heading (skip boilerplate H1s, try next)
    for m in _H1_RE.finditer(markdown_text):
        title = _clean_title(m.group(1))
        if title and len(title) <= 512 and not _is_boilerplate(title):
            return title

    # 2. First H2 heading (some converters use H2 for the document title)
    for m in _H2_RE.finditer(markdown_text):
        title = _clean_title(m.group(1))
        if title and len(title) <= 512 and not _is_boilerplate(title):
            return title
    return None


def _extract_title_from_markdown(markdown_text: str) -> str | None:
    """Extract title from markdown using a heading-first, then first-line cascade.

    Priority:
    1. First H1 heading (skip boilerplate H1s, try next)
    2. First H2 heading (many converters emit H2 for document titles)
    3. First non-empty line that looks like a title
       (short, no terminal punctuation, not a list/table element, not boilerplate)
    """
    # Try headings first
    heading_title = _extract_title_from_markdown_headings(markdown_text)
    if heading_title:
        return heading_title

    # 3. First non-empty line that looks like a title
    for line in markdown_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Strip markdown formatting BEFORE skip checks so that **bold** titles
        # and _italic_ titles are not skipped just because they start with * or _
        stripped = _re.sub(r"^[*`>]+|[*`_]+$", "", line).strip()
        if not stripped:
            continue
        # Skip list items, table rows, code fences, headings
        if line.startswith(("- ", "* ", "+ ", "|", "```", "#")):
            continue
        # Skip lines that are only punctuation/numbers
        if _re.match(r"^[\d\W]+$", stripped):
            continue
        # Skip boilerplate (page numbers, dates, "Confidential", etc.)
        if _is_boilerplate(stripped):
            continue
        # Must be reasonably short and not end with sentence punctuation
        if len(stripped) > 200:
            continue
        if stripped.endswith((".", ";", ",", ":", "!", "?")):
            continue
        # Strip markdown formatting
        clean = _clean_title(stripped)
        if clean and not _is_boilerplate(clean):
            return clean
    return None


def _extract_pdf_metadata_title(abs_path: str) -> str | None:
    """Extract title from PDF metadata (Title field in document info).

    According to Microsoft Research, ~33.5% of title fields are bogus.
    Common bogus patterns: "Microsoft Word - filename.doc", "Microsoft Excel -",
    raw filenames, "Untitled". We filter these out.
    """
    try:
        import pymupdf
        with pymupdf.open(abs_path) as doc:
            metadata = doc.metadata or {}
            title = (metadata.get("title") or "").strip()
            if not title or title.lower() == "untitled":
                return None
            # Filter common bogus metadata patterns
            bogus_prefixes = (
                "microsoft word - ",
                "microsoft excel - ",
                "microsoft powerpoint - ",
                "microsoft office - ",
            )
            if title.lower().startswith(bogus_prefixes):
                return None
            # Filter if it looks like a filename (has extension)
            if _re.match(r"^[\w,\s]+\.docx?$", title, _re.IGNORECASE):
                return None
            if _re.match(r"^[\w,\s]+\.xlsx?m?$", title, _re.IGNORECASE):
                return None
            if _re.match(r"^[\w,\s]+\.pdf$", title, _re.IGNORECASE):
                return None
            if not _is_boilerplate(title):
                return _clean_title(title)
    except Exception:
        pass
    return None


def _collect_pdf_font_spans(blocks: list) -> list[tuple[float, str]]:
    """Collect text spans with font sizes from PDF page blocks.

    Group consecutive spans on the same line into text fragments,
    using the max font size of the line.
    """
    spans = []
    for block in blocks:
        if block.get("type", 0) != 0:  # skip image blocks
            continue
        for line in block.get("lines", []):
            line_spans = []
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text:
                    continue
                line_spans.append((span.get("size", 0), text))
            if line_spans:
                # Merge spans on the same line, use max font size
                max_size = max(s[0] for s in line_spans)
                line_text = " ".join(s[1] for s in line_spans).strip()
                if line_text:
                    spans.append((max_size, line_text))
    return spans


def _filter_title_candidates(spans: list[tuple[float, str]], max_font_size: float) -> list[tuple[float, str]]:
    """Collect all lines with the largest font size (within 0.5pt tolerance).

    Sort by vertical position (first occurrence = topmost).
    """
    return [
        (size, text) for size, text in spans
        if abs(size - max_font_size) < 0.5
    ]


def _select_non_repeated_candidates(title_candidates: list[tuple[float, str]]) -> list[tuple[float, str]]:
    """Filter out repeated text (navigation headers / running elements).

    If the largest-font line appears 2+ times, it's likely a navigation
    header or running element, not a title. Return None so the cascade
    falls through to TOC/metadata, which are more reliable for web-to-PDF
    conversions with repeated nav.
    """
    from collections import Counter
    candidate_texts = [t for _, t in title_candidates]
    text_counts = Counter(candidate_texts)
    return [
        (s, t) for s, t in title_candidates
        if text_counts[t] == 1
    ]


def _extract_pdf_font_title(abs_path: str) -> str | None:
    """Extract title from PDF by finding the largest text on the first page.

    This is the SciPlore Xtract / Docear PDF Inspector heuristic: the largest
    font size on the first page is almost always the document title. Achieves
    ~70-78% accuracy with zero ML, just font-size analysis.
    """
    try:
        import pymupdf
        with pymupdf.open(abs_path) as doc:
            if doc.page_count == 0:
                return None
            page = doc[0]
            blocks = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT).get("blocks", [])
            if not blocks:
                return None

            spans = _collect_pdf_font_spans(blocks)
            if not spans:
                return None

            max_font_size = max(s[0] for s in spans)
            title_candidates = _filter_title_candidates(spans, max_font_size)
            if not title_candidates:
                return None

            non_repeated = _select_non_repeated_candidates(title_candidates)
            if not non_repeated:
                # All largest-font lines are repeated (navigation headers).
                # The font heuristic is unreliable for this page layout.
                return None

            # Use the first non-repeated candidate
            best = non_repeated[0][1]

            # If the first candidate is very short (< 10 chars) and there's a second
            # one right after, join them (subtitle pattern).
            if len(best) < 10 and len(non_repeated) > 1:
                best = f"{best} {non_repeated[1][1]}"

            if best and not _is_boilerplate(best):
                return _clean_title(best)
    except Exception:
        pass
    return None


def _pdf_page1_has_text(abs_path: str) -> bool:
    """Check if PDF page 1 has any extractable text (not image-only)."""
    try:
        import pymupdf
        with pymupdf.open(abs_path) as doc:
            if doc.page_count == 0:
                return False
            page = doc[0]
            blocks = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT).get("blocks", [])
            for block in blocks:
                if block.get("type", 0) != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("text", "").strip():
                            return True
            return False
    except Exception:
        return False


def _extract_pdf_toc_title(abs_path: str) -> str | None:
    """Extract title from PDF table of contents (outline).

    If the PDF has an embedded outline, the first top-level entry (level 1)
    is often the title or the first major section heading — a strong title
    signal that's more reliable than metadata.
    """
    try:
        import pymupdf
        with pymupdf.open(abs_path) as doc:
            toc = doc.get_toc()
            if not toc:
                return None
            # toc is a list of [level, title, page_number]
            # Look for the first level-1 entry
            for entry in toc:
                if entry[0] == 1:
                    title = entry[1].strip()
                    if title and not _is_boilerplate(title):
                        return _clean_title(title)
    except Exception:
        pass
    return None


def _extract_docx_title(abs_path: str) -> str | None:
    """Extract title from DOCX core properties (document.xml/core.xml)."""
    try:
        from docx import Document as DocxDocument
        doc = DocxDocument(abs_path)
        # Core properties (title, author, subject, etc.)
        cp = doc.core_properties
        title = (cp.title or "").strip() if cp else ""
        if title and title.lower() != "untitled" and not _is_boilerplate(title):
            return _clean_title(title)
    except Exception:
        pass
    return None


def _extract_html_title(abs_path: str) -> str | None:
    """Extract title from HTML <title> tag."""
    try:
        from bs4 import BeautifulSoup
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        # <title> tag
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text().strip()
            if title and not _is_boilerplate(title):
                return _clean_title(title)
        # og:title meta tag
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title = og_title["content"].strip()
            if title and not _is_boilerplate(title):
                return _clean_title(title)
    except Exception:
        pass
    return None


def _extract_pdf_title_signals(abs_path: str) -> str | None:
    """Try PDF-specific title signals: font heuristic, TOC, metadata."""
    font_title = _extract_pdf_font_title(abs_path)
    if font_title:
        return font_title

    toc_title = _extract_pdf_toc_title(abs_path)
    if toc_title:
        return toc_title

    return _extract_pdf_metadata_title(abs_path)


def _extract_markdown_title_for_cascade(markdown_text: str, ext: str, abs_path: str | None) -> str | None:
    """Try markdown-based title extraction (steps 6-7 of the cascade).

    For PDFs: only try headings, and only if page 1 has extractable text.
    For non-PDFs: try the full heading + first-line cascade.
    """
    if ext in _PDF_TITLE_EXTS:
        if _pdf_page1_has_text(abs_path):
            return _extract_title_from_markdown_headings(markdown_text)
        return None
    return _extract_title_from_markdown(markdown_text)


def extract_title(markdown_text: str, file_name: str, abs_path: str | None = None) -> str:
    """Extract a document title using a multi-signal priority cascade.

    Priority order (inspired by Google's title-link generation):
    1. PDF largest font on first page (SciPlore/Docear heuristic, ~70-78%
       accuracy — checked before metadata because ~33.5% of PDF metadata
       titles are bogus per Microsoft Research)
    2. PDF table of contents first level-1 entry
    3. PDF metadata Title field (filtered for bogus patterns)
    4. DOCX core properties title
    5. HTML <title> or og:title tag
    6. First H1 or H2 heading in the converted markdown
    7. First non-empty line that looks like a title (not boilerplate)
       — skipped for PDFs where font/TOC/metadata all failed (noisy OCR text)
    8. Cleaned filename as last resort

    Always returns a non-empty string. All signals are validated against
    boilerplate patterns (page numbers, dates, "Confidential", "DRAFT",
    web navigation, converter watermarks, etc.) and cleaned of markdown
    formatting and trailing page numbers.
    """
    ext = os.path.splitext(abs_path or file_name)[1].lower() if abs_path else ""

    # 1-3. PDF-specific signals (font heuristic, TOC, metadata)
    if ext in _PDF_TITLE_EXTS:
        pdf_title = _extract_pdf_title_signals(abs_path)
        if pdf_title:
            return pdf_title

    # 4. DOCX core properties
    if ext in _DOCX_TITLE_EXTS:
        docx_title = _extract_docx_title(abs_path)
        if docx_title:
            return docx_title

    # 5. HTML <title> or og:title tag
    if ext in _HTML_TITLE_EXTS:
        html_title = _extract_html_title(abs_path)
        if html_title:
            return html_title

    # 6 + 7. Markdown content (H1/H2 heading or first title-like line)
    md_title = _extract_markdown_title_for_cascade(markdown_text, ext, abs_path)
    if md_title:
        return md_title

    # 8. Cleaned filename
    return _clean_filename_to_title(file_name)


def _convert_to_markdown(
    abs_path: str, file_name: str, enable_ocr: Optional[bool] = None,
    progress_cb: Optional[callable] = None,
) -> str:
    """Convert any supported file to clean Markdown text.

    enable_ocr: tri-state override.
      None  → use global VISION_MODEL setting (default behaviour)
      True  → force OCR on (requires VISION_MODEL to be configured)
      False → force OCR off for this document regardless of global setting

    progress_cb: optional callable(pct, msg) called between OCR pages
    so the ProgressTimeout silence watchdog doesn't kill long PDFs.

    Delegates to the active MarkdownEngine.  Falls back to raw UTF-8 if
    the engine raises.
    """
    try:
        return _get_engine().convert(abs_path, file_name, enable_ocr, progress_cb=progress_cb)
    except Exception as e:
        logger.warning("[converter] %s engine failed for %s (%s) — falling back to raw text",
                       _get_engine().name, file_name, e)
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception:
            return ""
