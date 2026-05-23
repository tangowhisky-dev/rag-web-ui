"""Fast regex/heuristic markdown cleanup for document ingestion.

Strips common noise introduced by PDF extraction and OCR:
- Stray page numbers
- Repeated short header/footer lines
- Broken table separator rows
- OCR garbage lines (mostly non-alphanumeric characters)
- Excessive consecutive blank lines

No LLM calls, no external dependencies beyond stdlib re.
"""
import logging
import re
from typing import List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Stray page numbers: "Page 3", "3 of 47", "- 12 -", or a bare integer alone on a line
_PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:Page\s+\d+|\d+\s+of\s+\d+|-\s*\d+\s*-|\d+)\s*$",
    re.IGNORECASE,
)

# Broken table separator: line consists only of |, -, :, +, space — no real word chars
_TABLE_SEP_RE = re.compile(r"^\s*[|\-:+\s]+\s*$")

# Minimum line length to bother checking for OCR garbage (avoids false positives on short lines)
_OCR_MIN_LEN = 4
# Fraction of non-alphanumeric chars that triggers removal
_OCR_GARBAGE_THRESHOLD = 0.60

# Maximum chars for a line to be considered a short repeated header/footer
_HEADER_MAX_LEN = 60
# How many times a short line must appear before we deduplicate it
_HEADER_REPEAT_THRESHOLD = 3


def _is_ocr_garbage(line: str) -> bool:
    """Return True if the line looks like OCR noise (mostly non-alphanumeric)."""
    stripped = line.strip()
    if len(stripped) < _OCR_MIN_LEN:
        return False
    non_alnum = sum(1 for c in stripped if not c.isalnum() and not c.isspace())
    return (non_alnum / len(stripped)) > _OCR_GARBAGE_THRESHOLD


def clean_markdown(text: str) -> str:
    """Clean noise from markdown text produced by document conversion.

    Rules applied in order:
    1. Strip stray page numbers
    2. Strip broken table separator rows
    3. Strip OCR garbage lines
    4. Remove repeated short header/footer lines (keep first occurrence)
    5. Collapse 3+ consecutive blank lines to 2

    Returns the cleaned text. The input is never mutated.
    """
    if not text:
        return text

    lines: List[str] = text.splitlines()
    total_in = len(lines)

    # --- Pass 1: count line frequencies for header deduplication ---
    line_freq: dict[str, int] = {}
    for line in lines:
        stripped = line.strip()
        if 0 < len(stripped) <= _HEADER_MAX_LEN:
            line_freq[stripped] = line_freq.get(stripped, 0) + 1

    repeated_headers = {s for s, count in line_freq.items() if count >= _HEADER_REPEAT_THRESHOLD}
    seen_headers: set[str] = set()

    # --- Pass 2: filter lines ---
    filtered: List[str] = []
    removed_page = 0
    removed_table_sep = 0
    removed_ocr = 0
    removed_headers = 0

    for line in lines:
        stripped = line.strip()

        # Rule 1: stray page numbers
        if _PAGE_NUMBER_RE.match(line):
            removed_page += 1
            continue

        # Rule 2: broken table separator
        if stripped and _TABLE_SEP_RE.match(line) and "|" in line:
            removed_table_sep += 1
            continue

        # Rule 3: OCR garbage
        if _is_ocr_garbage(line):
            removed_ocr += 1
            continue

        # Rule 4: repeated short headers — keep first, drop rest
        if stripped in repeated_headers:
            if stripped in seen_headers:
                removed_headers += 1
                continue
            seen_headers.add(stripped)

        filtered.append(line)

    # --- Pass 3: collapse excessive blank lines ---
    collapsed: List[str] = []
    blank_run = 0
    removed_blanks = 0

    for line in filtered:
        if line.strip() == "":
            blank_run += 1
            if blank_run <= 2:
                collapsed.append(line)
            else:
                removed_blanks += 1
        else:
            blank_run = 0
            collapsed.append(line)

    total_removed = removed_page + removed_table_sep + removed_ocr + removed_headers + removed_blanks
    logger.debug(
        "[CLEANUP] lines_in=%d removed=%d (page=%d table_sep=%d ocr=%d headers=%d blanks=%d)",
        total_in,
        total_removed,
        removed_page,
        removed_table_sep,
        removed_ocr,
        removed_headers,
        removed_blanks,
    )

    return "\n".join(collapsed)
