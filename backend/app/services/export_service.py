"""
Message export service.

Supports three formats:
  pdf   — reportlab PDF
  word  — python-docx .docx
  image — Pillow PNG screenshot of the answer text

All formats strip markdown syntax and render plain text.
"""
from __future__ import annotations

import io
import re
import textwrap
from typing import Literal

ExportFormat = Literal["pdf", "word", "image"]


from app.services.reasoning_tags import strip_reasoning_tags

def _strip_markdown(text: str) -> str:
    """Very lightweight markdown stripper — enough for clean export."""
    text = strip_reasoning_tags(text)
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"`{1,3}(.*?)`{1,3}", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[citation:\d+\]", "", text)
    text = re.sub(r"\[citation\]\(\d+\)", "", text)
    return text.strip()


def export_to_pdf(answer_text: str) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.units import cm

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
    )
    styles = getSampleStyleSheet()
    body_style = styles["BodyText"]
    body_style.fontSize = 11
    body_style.leading = 16

    clean = _strip_markdown(answer_text)
    elements = []
    for para in clean.split("\n\n"):
        para = para.strip()
        if para:
            elements.append(Paragraph(para.replace("\n", "<br/>"), body_style))
            elements.append(Spacer(1, 0.3 * cm))

    doc.build(elements)
    return buf.getvalue()


def export_to_word(answer_text: str) -> bytes:
    from docx import Document
    from docx.shared import Pt, Cm

    doc = Document()
    # Margins
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    clean = _strip_markdown(answer_text)
    for para in clean.split("\n\n"):
        para = para.strip()
        if para:
            p = doc.add_paragraph(para)
            p.style.font.size = Pt(11)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def export_to_image(answer_text: str) -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    clean = _strip_markdown(answer_text)
    width = 900
    padding = 40
    font_size = 16
    line_height = font_size + 6

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    # Word-wrap
    wrap_width = (width - 2 * padding) // (font_size // 2 + 2)
    lines: list[str] = []
    for para in clean.split("\n"):
        if para.strip():
            lines.extend(textwrap.wrap(para, width=wrap_width) or [""])
        else:
            lines.append("")

    height = padding * 2 + len(lines) * line_height
    img = Image.new("RGB", (width, max(height, 200)), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    y = padding
    for line in lines:
        draw.text((padding, y), line, fill=(30, 30, 30), font=font)
        y += line_height

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def export_message(answer_text: str, fmt: ExportFormat) -> tuple[bytes, str, str]:
    """
    Returns (content_bytes, media_type, filename).
    """
    if fmt == "pdf":
        data = export_to_pdf(answer_text)
        return data, "application/pdf", "answer.pdf"
    elif fmt == "word":
        data = export_to_word(answer_text)
        return data, "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "answer.docx"
    elif fmt == "image":
        data = export_to_image(answer_text)
        return data, "image/png", "answer.png"
    else:
        raise ValueError(f"Unknown export format: {fmt}")


# ── Synthesis report generation ────────────────────────────────────────────────

def generate_synthesis_report(
    answer: str,
    tool_trace: list,
    query: str,
    kb_ids: list,
) -> str:
    """
    Generate a structured Markdown synthesis report from LLM answer + tool trace.

    Extracts source documents from search/synthesize tool calls in the trace,
    deduplicates them, and appends a ## Sources section.

    Args:
        answer:     Final LLM-generated answer text (may already contain ## sections).
        tool_trace: List of ToolResult dicts from the tool calling loop.
        query:      Original user query.
        kb_ids:     Knowledge base IDs queried.

    Returns:
        Complete Markdown string with header, answer, and sources.
    """
    import datetime

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Extract source file names from tool trace
    sources: dict = {}  # filename -> chunk count
    for entry in tool_trace:
        output = entry.get("output") or {}
        # synthesize_documents returns {"chunks": [...]}
        # search_documents returns a list directly
        chunks = []
        if isinstance(output, dict):
            chunks = output.get("chunks", [])
        elif isinstance(output, list):
            chunks = output

        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            src = chunk.get("source") or "unknown"
            sources[src] = sources.get(src, 0) + 1

    # Build header
    lines = [
        f"# Synthesis Report",
        f"",
        f"**Query:** {query}",
        f"**Knowledge Bases:** {', '.join(str(k) for k in kb_ids)}",
        f"**Generated:** {timestamp}",
        f"",
        f"---",
        f"",
        answer.strip(),
    ]

    # Append sources section if any were found
    if sources:
        lines += [
            "",
            "---",
            "",
            "## Sources",
            "",
        ]
        for src, count in sorted(sources.items(), key=lambda x: -x[1]):
            lines.append(f"- **{src}** ({count} chunk{'s' if count != 1 else ''} cited)")

    return "\n".join(lines)
