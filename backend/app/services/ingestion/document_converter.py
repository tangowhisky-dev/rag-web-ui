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
    ".txt", ".md", ".html", ".htm", ".mhtml",
    ".csv", ".json", ".xml",
    # Email (stdlib email module / fallback)
    ".msg", ".eml",
    # Images (vision OCR)
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff",
    # Archives
    ".zip",
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
    ".mhtml": "message/rfc822",
    ".csv":   "text/csv",
    ".json":  "application/json",
    ".xml":   "application/xml",
    ".msg":   "application/vnd.ms-outlook",
    ".eml":   "message/rfc822",
    ".jpg":   "image/jpeg",
    ".jpeg":  "image/jpeg",
    ".png":   "image/png",
    ".gif":   "image/gif",
    ".bmp":   "image/bmp",
    ".tiff":  "image/tiff",
    ".zip":   "application/zip",
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
_HTML_EXTENSIONS = {".html", ".htm", ".mhtml"}

# Image extensions (OCR via vision model)
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}

# Minimum pixel count for an embedded PDF image to be considered worth OCR'ing.
# Filters out 1×1 mask objects and decorative dots.
_MIN_IMAGE_PIXELS = 10_000

# OCR prompt reused across engines
_OCR_PROMPT = (
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


# ── Vision config (shared across engines) ──────────────────────────────────

@dataclass
class VisionConfig:
    model: str
    client: SyncOpenAI


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
                logger.info("[converter] OCR disabled — VISION_MODEL not set")
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
            _vision_config = VisionConfig(
                model=vision_model,
                client=SyncOpenAI(api_key=api_key, base_url=api_base),
            )
            logger.info("[converter] OCR enabled — vision_model=%s base=%s", vision_model, api_base)
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
    )
    return (resp.choices[0].message.content or "").strip()


def _ocr_pdf_page(pdf_path: str, page_num_1indexed: int, config: VisionConfig) -> str:
    """Render a PDF page to an image and OCR it."""
    import fitz
    with fitz.open(pdf_path) as doc:
        page = doc[page_num_1indexed - 1]
        pix = page.get_pixmap(dpi=150)
        image_bytes = pix.tobytes("png")
    return _ocr_image_bytes(image_bytes, config)


def _detect_pdf_images(pdf_path: str) -> list:
    """Return [(page_1indexed, img_index, image_bytes, ext, width, height)]
    for embedded raster images above _MIN_IMAGE_PIXELS."""
    import fitz
    big_images = []
    with fitz.open(pdf_path) as doc:
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

    def convert(self, abs_path: str, file_name: str, enable_ocr: Optional[bool]) -> str:
        ...


# ── AnyDoc engine (default) ────────────────────────────────────────────────

class AnydocEngine:
    """anydoc + pdf-inspector + fitz image extraction + vision OCR.

    PDF:  pdf-inspector for text extraction + per-page OCR routing.
          Embedded raster images above a size threshold are OCR'd too.
    Office docs (docx, pptx, xlsx, odt, rtf, epub, csv, …): anydoc.
    Images (jpg, png, …): vision model OCR.
    Text (txt, md, json, xml): direct UTF-8 read.
    HTML: tag-stripped to text via BeautifulSoup.
    EML: stdlib email module → plain text.
    Other (msg, zip, …): raw UTF-8 fallback.
    """

    name = "anydoc"

    def convert(self, abs_path: str, file_name: str, enable_ocr: Optional[bool]) -> str:
        ext = os.path.splitext(abs_path)[1].lower()

        if ext == ".pdf":
            markdown = self._convert_pdf(abs_path, enable_ocr)
        elif ext in _IMAGE_EXTENSIONS:
            markdown = self._convert_image(abs_path, enable_ocr)
        elif ext in _ANYDOC_EXTENSIONS:
            markdown = self._convert_anydoc(abs_path)
        elif ext in _TEXT_EXTENSIONS:
            markdown = self._read_text(abs_path)
        elif ext in _HTML_EXTENSIONS:
            markdown = self._convert_html(abs_path)
        elif ext == ".eml":
            markdown = self._convert_eml(abs_path)
        else:
            # .msg, .zip, or anything else — raw text fallback
            markdown = self._read_text(abs_path)

        cleaned = strip_reasoning_tags(markdown)
        if len(cleaned) < len(markdown):
            logger.info("[anydoc] stripped %d chars of reasoning tags from %s",
                        len(markdown) - len(cleaned), file_name)
        logger.info("[anydoc] converted %s → %d chars of markdown (ocr=%s)",
                    file_name, len(cleaned), enable_ocr is not False)
        return cleaned

    # ── PDF ────────────────────────────────────────────────────────────────

    def _convert_pdf(self, abs_path: str, enable_ocr: Optional[bool]) -> str:
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
            logger.info("[anydoc] OCR'ing %d scanned pages in %s",
                        len(pages_needing_ocr), os.path.basename(abs_path))
            for page_num in pages_needing_ocr:
                try:
                    text = _ocr_pdf_page(abs_path, page_num, config)
                    if text:
                        ocr_parts.append(f"\n\n## OCR for page {page_num}\n\n{text}")
                except Exception as e:
                    logger.warning("[anydoc] page %d OCR failed: %s", page_num, e)

        # 2. OCR embedded raster images (diagrams, figures)
        big_images = _detect_pdf_images(abs_path)
        if big_images:
            logger.info("[anydoc] OCR'ing %d embedded images in %s",
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

    def _convert_eml(self, abs_path: str) -> str:
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
            if msg.is_multipart():
                for part in msg.walk():
                    ctype = part.get_content_type()
                    if ctype == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            parts.append(payload.decode("utf-8", errors="replace"))
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    parts.append(payload.decode("utf-8", errors="replace"))
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
                    logger.info("[markitdown] OCR enabled — vision_model=%s base=%s", vision_model, api_base)
                else:
                    self._singleton = self._MarkItDown()
                    logger.info("[markitdown] OCR disabled — VISION_MODEL not set")
            finally:
                db.close()
            return self._singleton

    def convert(self, abs_path: str, file_name: str, enable_ocr: Optional[bool]) -> str:
        if enable_ocr is False:
            md_instance = self._MarkItDown()
            logger.info("[markitdown] OCR disabled for this document (per-document override)")
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
            logger.info("[markitdown] stripped %d chars of reasoning tags from %s",
                        len(markdown_text) - len(cleaned), file_name)
        logger.info("[markitdown] converted %s → %d chars of markdown (ocr=%s)",
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
        logger.info("[converter] active engine: %s", _engine_instance.name)
        return _engine_instance


def _convert_to_markdown(abs_path: str, file_name: str, enable_ocr: Optional[bool] = None) -> str:
    """Convert any supported file to clean Markdown text.

    enable_ocr: tri-state override.
      None  → use global VISION_MODEL setting (default behaviour)
      True  → force OCR on (requires VISION_MODEL to be configured)
      False → force OCR off for this document regardless of global setting

    Delegates to the active MarkdownEngine.  Falls back to raw UTF-8 if
    the engine raises.
    """
    try:
        return _get_engine().convert(abs_path, file_name, enable_ocr)
    except Exception as e:
        logger.warning("[converter] %s engine failed for %s (%s) — falling back to raw text",
                       _get_engine().name, file_name, e)
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception:
            return ""
