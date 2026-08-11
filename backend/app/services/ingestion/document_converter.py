"""Document conversion helpers — markitdown, OCR, extension maps.

Split from document_processor.py for maintainability.
"""

import logging
import os
from typing import Optional

from markitdown import MarkItDown
from openai import OpenAI as SyncOpenAI

from app.core.config import settings
from app.services.infrastructure import strip_reasoning_tags

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

_markitdown: Optional[MarkItDown] = None

# Supported file extensions (markitdown handles all of these)
SUPPORTED_EXTENSIONS = {
    # Documents
    ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
    # Text / Markup
    ".txt", ".md", ".html", ".htm", ".mhtml",
    # Data formats
    ".csv", ".json", ".xml",
    # Email
    ".msg", ".eml",
    # Books
    ".epub",
    # Images (OCR via markitdown-ocr)
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff",
    # Archives (recursively processes contents)
    ".zip",
}

CONTENT_TYPE_MAP = {
    ".pdf":   "application/pdf",
    ".docx":  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc":   "application/msword",
    ".pptx":  "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".ppt":   "application/vnd.ms-powerpoint",
    ".xlsx":  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls":   "application/vnd.ms-excel",
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
    ".epub":  "application/epub+zip",
    ".jpg":   "image/jpeg",
    ".jpeg":  "image/jpeg",
    ".png":   "image/png",
    ".gif":   "image/gif",
    ".bmp":   "image/bmp",
    ".tiff":  "image/tiff",
    ".zip":   "application/zip",
}


def _get_markitdown() -> MarkItDown:
    """
    Lazy singleton for MarkItDown converter.

    When VISION_MODEL is configured, the markitdown-ocr plugin is
    activated with an llm_client so it can OCR embedded images in PDFs,
    DOCX, PPTX, and XLSX files automatically.

    When VISION_MODEL is unset, MarkItDown is initialised without a
    client — markitdown-ocr still loads (if installed) but silently skips
    OCR, which is identical to the previous behaviour.

    Vision model and API credentials are resolved from app-level settings
    (super_admin only) since ingestion is a shared process.
    """
    global _markitdown
    if _markitdown is None:
        from app.services.settings_service import get_setting
        from app.db.session import SessionLocal
        _db = SessionLocal()
        try:
            vision_model = get_setting(_db, "VISION_MODEL", None)
            if vision_model:
                # Vision API key: VISION_API_KEY → OPENAI_API_KEY → placeholder
                api_key = get_setting(_db, "VISION_API_KEY", None) or get_setting(_db, "OPENAI_API_KEY", None)
                # Vision base URL: OPENAI_VISION_API_BASE → OPENAI_API_BASE → .env
                api_base = get_setting(_db, "OPENAI_VISION_API_BASE", None) or get_setting(_db, "OPENAI_API_BASE", None)
                if not api_key:
                    api_key = "not-required"
                vision_client = SyncOpenAI(
                    api_key=api_key,
                    base_url=api_base,
                )
                _markitdown = MarkItDown(
                    enable_plugins=True,
                    llm_client=vision_client,
                    llm_model=vision_model,
                    llm_prompt=(
                        "Extract all text from this image into clean, naturally flowing paragraphs, "
                        "while preserving document structure and any table or sub-element layout.\n\n"
                        "Rules:\n"
                        "- Remove unnatural line breaks within sentences\n"
                        "- Join split words and sentences caused by column layout or line wrapping\n"
                        "- Keep proper paragraph breaks where the topic clearly changes\n"
                        "- Preserve tables using Markdown table syntax\n"
                        "- Preserve all original meaning and technical terms exactly\n"
                        "- Output only the extracted text, no explanations or commentary"
                    ),
                )
                logger.info(
                    "[markitdown] OCR enabled — vision_model=%s base=%s",
                    vision_model, api_base,
                )
            else:
                _markitdown = MarkItDown()
                logger.info(
                    "[markitdown] OCR disabled — VISION_MODEL not set"
                )
        finally:
            _db.close()
    return _markitdown


def _convert_to_markdown(abs_path: str, file_name: str, enable_ocr: Optional[bool] = None) -> str:
    """
    Convert any supported file to clean Markdown text using markitdown.

    enable_ocr: tri-state override.
      None  → use global VISION_MODEL setting (default behaviour)
      True  → force OCR on (requires VISION_MODEL to be configured)
      False → force OCR off for this document regardless of global setting

    Think traces (reasoning tags configured in REASONING_TAGS) are stripped
    before returning. Falls back gracefully to raw UTF-8 if conversion fails.
    """
    try:
        if enable_ocr is False:
            # Per-document OCR override: use a plain MarkItDown with no vision client.
            md_instance = MarkItDown()
            logger.info("[markitdown] OCR disabled for this document (per-document override)")
        elif enable_ocr is True:
            # Explicit on: use the configured singleton (which has OCR if set up).
            md_instance = _get_markitdown()
            from app.services.settings_service import get_setting
            from app.db.session import SessionLocal
            _db = SessionLocal()
            try:
                vision_model = get_setting(_db, "VISION_MODEL", None)
            finally:
                _db.close()
            if not vision_model:
                logger.warning("[markitdown] OCR requested but VISION_MODEL not set — falling back to text-only")
        else:
            # Default: respect global setting.
            md_instance = _get_markitdown()
        result = md_instance.convert(abs_path)
        markdown_text = result.text_content or ""

        # Strip reasoning tag blocks configured in settings.REASONING_TAGS.
        cleaned = strip_reasoning_tags(markdown_text)

        if len(cleaned) < len(markdown_text):
            stripped_chars = len(markdown_text) - len(cleaned)
            logger.info(
                "[markitdown] stripped %d chars of reasoning tags from %s",
                stripped_chars, file_name,
            )

        logger.info(
            "[markitdown] converted %s → %d chars of markdown (ocr=%s)",
            file_name, len(cleaned), bool(md_instance),
        )
        return cleaned
    except Exception as e:
        logger.warning(
            "[markitdown] conversion failed for %s (%s) — falling back to raw text",
            file_name, e
        )
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception:
            return ""
