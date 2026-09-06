"""office_generate tool — create Office documents via OfficeCLI.

Creates a new .pptx, .docx, or .xlsx from state.accumulated_data and/or
state.retrieved_docs. The LLM provides structure (format, slide/section
titles, chart types, layout). The tool reads data from state and translates
it into OfficeCLI batch commands deterministically — the LLM never passes
raw data values.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import time
from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field, model_validator

from app.core.storage import save_ephemeral_file
from app.models.chat import ChatFile
from app.services.agentic_rag.tool_context import ToolContext, write_audit
from app.services.agentic_rag.tools.base import BaseAgentTool
from app.services.settings_service import get_setting

logger = logging.getLogger(__name__)

# ── Themes (from the officecli-pptx skill) ──────────────────────────────────
THEMES: dict[str, dict] = {
    "midnight": {"bg": "0D1B2A", "text": "FFFFFF", "accent": "CADCFC",
                 "heading_font": "Georgia", "body_font": "Calibri"},
    "coral": {"bg": "FFFFFF", "text": "333333", "accent": "F96167",
              "heading_font": "Trebuchet MS", "body_font": "Calibri"},
    "ocean": {"bg": "1A1A2E", "text": "FFFFFF", "accent": "16213E",
              "heading_font": "Georgia", "body_font": "Calibri"},
    "forest": {"bg": "F5F5F0", "text": "1B1B1B", "accent": "2D5016",
               "heading_font": "Palatino", "body_font": "Calibri"},
    "slate": {"bg": "1E2761", "text": "FFFFFF", "accent": "CADCFC",
              "heading_font": "Georgia", "body_font": "Calibri"},
}

_CONTENT_TYPES = {
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


# ── Input schemas ────────────────────────────────────────────────────────────

class OfficeSlideSpec(BaseModel):
    title: str = Field(description="Slide title")
    layout: str = Field(default="blank", description="blank, title, section, content")
    bullets: Optional[List[str]] = None
    content: Optional[str] = Field(default=None, description="Slide body text (alternative to bullets — will be split into bullets by newline)")
    chart_type: Optional[str] = Field(default=None, description="bar, line, pie, column, scatter, area, doughnut")
    chart_title: Optional[str] = None
    background: Optional[str] = Field(default=None, description="Hex color override, e.g. 1A1A2E")

    @model_validator(mode="before")
    @classmethod
    def _normalize_bullets(cls, data: Any) -> Any:
        """Normalize bullets and accept field aliases from skill files."""
        if isinstance(data, dict):
            # Accept aliases from skill files: title_text → title, subtitle_text → subtitle.
            # Drop speaker_notes and slide_number (not used — keeps tool calls small).
            if "title_text" in data and "title" not in data:
                data["title"] = data.pop("title_text")
            if "subtitle_text" in data and "subtitle" not in data:
                data["subtitle"] = data.pop("subtitle_text")
            data.pop("speaker_notes", None)
            data.pop("slide_number", None)
            # If 'content' is provided but 'bullets' is not, split content into bullets.
            if data.get("content") and not data.get("bullets"):
                data["bullets"] = [b.strip() for b in str(data["content"]).split("\n") if b.strip()]
            data.pop("content", None)
            # If bullets is a string, split into list (LLM often passes comma-separated string).
            bullets = data.get("bullets")
            if isinstance(bullets, str):
                # Split by newline first, then by comma if still one item.
                parts = [b.strip() for b in bullets.split("\n") if b.strip()]
                if len(parts) <= 1:
                    parts = [b.strip() for b in bullets.split(",") if b.strip()]
                data["bullets"] = parts
        return data


class OfficeSectionSpec(BaseModel):
    heading: str = Field(description="Section heading text")
    level: int = Field(default=1, description="1=H1, 2=H2, 3=H3")
    paragraphs: Optional[List[str]] = None
    table: Optional[dict] = Field(default=None, description="Table spec: {headers: [...], rows: [[...]]}")
    chart: Optional[dict] = Field(default=None, description="Chart spec: {type, title}")


class OfficeSheetSpec(BaseModel):
    name: str = Field(description="Sheet name")
    headers: Optional[List[str]] = None
    rows: Optional[List[List[Any]]] = None
    data: Optional[List[dict]] = Field(default=None, description="Data as list of dicts (e.g. [{label: 'X', value: 10}]). Automatically converted to headers + rows.")
    chart: Optional[dict] = Field(default=None, description="Chart spec: {type, title, data_range}")

    @model_validator(mode="before")
    @classmethod
    def _data_to_rows(cls, data_val: Any) -> Any:
        """Normalize data into headers + rows.

        Handles three LLM patterns:
        1. 'data' as list of dicts → convert to headers + rows
        2. 'rows' as list of dicts → convert to list of lists + extract headers
        3. 'data' as dict with wrapper key (e.g. {"revenue_streams": [...]}) → unwrap
        """
        if isinstance(data_val, dict):
            # Handle 'data' field
            raw_data = data_val.get("data")
            if raw_data and isinstance(raw_data, dict):
                # Unwrap dict with single wrapper key
                if len(raw_data) == 1:
                    inner = list(raw_data.values())[0]
                    if isinstance(inner, list):
                        raw_data = inner
                        data_val["data"] = inner
            if raw_data and isinstance(raw_data, list) and not data_val.get("rows"):
                if raw_data and isinstance(raw_data[0], dict):
                    # Flatten nested {"label": N} structures from json_repair
                    flat = []
                    for d in raw_data:
                        flat.append({k: (v.get("label") if isinstance(v, dict) and "label" in v else v) for k, v in d.items()})
                    headers = list(flat[0].keys())
                    rows = [[d.get(h, "") for h in headers] for d in flat]
                    data_val["headers"] = data_val.get("headers") or headers
                    data_val["rows"] = rows
            # Handle 'rows' as list of dicts
            rows = data_val.get("rows")
            if rows and isinstance(rows, list) and rows and isinstance(rows[0], dict):
                headers = list(rows[0].keys())
                data_val["headers"] = data_val.get("headers") or headers
                data_val["rows"] = [[d.get(h, "") for h in headers] for d in rows]
        return data_val


class OfficeGenerateInput(BaseModel):
    format: str = Field(description="Document format: pptx, docx, or xlsx")
    title: Optional[str] = Field(default=None, description="Document title")
    subtitle: Optional[str] = Field(default=None, description="Document subtitle (pptx cover, docx title page)")
    slides: Optional[List[OfficeSlideSpec]] = None
    sections: Optional[List[OfficeSectionSpec]] = None
    sheets: Optional[List[OfficeSheetSpec]] = None
    theme: Optional[str] = Field(default="midnight", description="Color theme: midnight, coral, ocean, forest, slate")
    font_heading: Optional[str] = Field(default=None, description="Override heading font")
    font_body: Optional[str] = Field(default=None, description="Override body font")
    append: bool = Field(default=False, description="If True, append to the last generated file of the same format instead of creating a new one. Use this for multi-slide decks: call office_generate with 1-2 slides at a time, append=True for all calls after the first.")


# ── Data translation helpers ─────────────────────────────────────────────────

def _data_to_chart_format(data: list[dict]) -> tuple[str, str]:
    """Convert accumulated_data [{label, value, unit, context}] to OfficeCLI chart format.

    Returns (categories, data_string) where:
    - categories = "Q1,Q2,Q3,Q4"
    - data_string = "Series1:1,2,3,4;Series2:5,6,7,8" (multi-series by context)
    """
    if not data:
        return "", ""

    # Group by context for multi-series support
    contexts: dict[str, list] = {}
    for point in data:
        ctx_key = point.get("context") or "default"
        contexts.setdefault(ctx_key, []).append(point)

    all_labels: list[str] = []
    for p in data:
        label = str(p.get("label", ""))
        if label not in all_labels:
            all_labels.append(label)

    if len(contexts) == 1:
        points = list(contexts.values())[0]
        categories = ",".join(str(p.get("label", "")) for p in points)
        values = ",".join(_fmt_val(p.get("value")) for p in points)
        series = f"Values:{values}"
    else:
        categories = ",".join(all_labels)
        series_parts = []
        for ctx_name, points in contexts.items():
            value_map = {str(p.get("label", "")): p.get("value") for p in points}
            values = [_fmt_val(value_map.get(l, 0)) for l in all_labels]
            # Sanitize context name for series label
            safe_name = re.sub(r"[,;:]", " ", ctx_name)[:30].strip() or "Series"
            series_parts.append(f"{safe_name}:{','.join(values)}")
        series = ";".join(series_parts)

    return categories, series


def _fmt_val(v) -> str:
    """Format a value for OfficeCLI chart data string."""
    if isinstance(v, (int, float)):
        if isinstance(v, float) and v == int(v):
            return str(int(v))
        return str(v)
    # Try to extract numeric from string
    try:
        fv = float(v)
        if fv == int(fv):
            return str(int(fv))
        return str(fv)
    except (ValueError, TypeError):
        return "0"


def _generate_filename(spec: OfficeGenerateInput) -> str:
    safe_title = re.sub(r"[^a-zA-Z0-9_-]", "_", spec.title or "document")[:50]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe_title}_{timestamp}.{spec.format}"


# ── Batch builders ───────────────────────────────────────────────────────────

def _build_pptx_batch(spec: OfficeGenerateInput, data: list[dict]) -> list[dict]:
    """Build OfficeCLI batch items for a PowerPoint deck."""
    palette = THEMES.get(spec.theme or "midnight", THEMES["midnight"])
    h_font = spec.font_heading or palette["heading_font"]
    b_font = spec.font_body or palette["body_font"]
    items: list[dict] = []

    # Cover slide — only create if the deck has a title/subtitle at the
    # deck level. If the LLM only provides individual slides (no deck title),
    # skip the cover and use the LLM's slides directly starting at slide[1].
    slide_offset = 1
    if spec.title or spec.subtitle:
        items.append({"command": "add", "parent": "/", "type": "slide", "props": {"layout": "blank"}})
        bg = palette["bg"]
        items.append({"command": "set", "path": "/slide[1]", "props": {"background": bg}})
        if spec.title:
            items.append({"command": "add", "parent": "/slide[1]", "type": "shape",
                           "props": {"text": spec.title, "font": h_font, "size": "36pt", "bold": "true",
                                     "color": palette["text"], "x": "1.27cm", "y": "4cm",
                                     "width": "31.33cm", "height": "3cm"}})
        if spec.subtitle:
            items.append({"command": "add", "parent": "/slide[1]", "type": "shape",
                           "props": {"text": spec.subtitle, "font": b_font, "size": "20pt",
                                     "color": palette["accent"], "x": "1.27cm", "y": "7cm",
                                     "width": "31.33cm", "height": "2cm"}})
        slide_offset = 2

    # Content slides
    for i, slide in enumerate(spec.slides or [], slide_offset):
        items.append({"command": "add", "parent": "/", "type": "slide", "props": {"layout": slide.layout}})
        slide_bg = slide.background or palette["bg"]
        items.append({"command": "set", "path": f"/slide[{i}]", "props": {"background": slide_bg}})

        # Title (>=36pt bold per skill guidelines)
        items.append({"command": "add", "parent": f"/slide[{i}]", "type": "shape",
                       "props": {"text": slide.title, "font": h_font, "size": "36pt", "bold": "true",
                                 "color": palette["text"], "x": "1.27cm", "y": "1.27cm",
                                 "width": "31.33cm", "height": "2.5cm"}})

        # Bullets (>=18pt per skill guidelines)
        if slide.bullets:
            bullet_text = "\n".join(f"• {b}" for b in slide.bullets)
            has_chart = slide.chart_type is not None
            body_y = "4cm" if has_chart else "4cm"
            body_height = "6cm" if has_chart else "12cm"
            items.append({"command": "add", "parent": f"/slide[{i}]", "type": "shape",
                           "props": {"text": bullet_text, "font": b_font, "size": "18pt",
                                     "color": palette["text"], "x": "1.27cm", "y": body_y,
                                     "width": "31.33cm", "height": body_height}})

        # Chart
        if slide.chart_type:
            categories, series_data = _data_to_chart_format(data)
            if categories:
                chart_y = "11cm" if slide.bullets else "5cm"
                items.append({"command": "add", "parent": f"/slide[{i}]", "type": "chart",
                               "props": {"chartType": slide.chart_type,
                                         "title": slide.chart_title or slide.title,
                                         "categories": categories,
                                         "data": series_data,
                                         "x": "2cm", "y": chart_y,
                                         "width": "29cm", "height": "7cm",
                                         "legend": "bottom"}})

    return items


def _build_docx_batch(spec: OfficeGenerateInput, data: list[dict]) -> list[dict]:
    """Build OfficeCLI batch items for a Word document."""
    palette = THEMES.get(spec.theme or "midnight", THEMES["midnight"])
    h_font = spec.font_heading or "Georgia"
    b_font = spec.font_body or "Calibri"
    items: list[dict] = []

    # Title
    if spec.title:
        items.append({"command": "add", "parent": "/body", "type": "paragraph",
                       "props": {"text": spec.title, "style": "Title", "size": "24pt",
                                 "bold": "true", "color": palette["accent"],
                                 "spaceAfter": "12pt"}})
    if spec.subtitle:
        items.append({"command": "add", "parent": "/body", "type": "paragraph",
                       "props": {"text": spec.subtitle, "size": "14pt",
                                 "color": "666666", "spaceAfter": "18pt"}})

    for section in spec.sections or []:
        # Heading
        heading_size = {1: "18pt", 2: "14pt", 3: "12pt"}.get(section.level, "12pt")
        heading_style = {1: "Heading1", 2: "Heading2", 3: "Heading3"}.get(section.level, "Heading3")
        items.append({"command": "add", "parent": "/body", "type": "paragraph",
                       "props": {"text": section.heading, "style": heading_style,
                                 "size": heading_size, "bold": "true",
                                 "color": palette["accent"], "spaceBefore": "12pt",
                                 "spaceAfter": "6pt"}})

        # Paragraphs
        for para in section.paragraphs or []:
            items.append({"command": "add", "parent": "/body", "type": "paragraph",
                           "props": {"text": para, "font": b_font, "size": "11pt",
                                     "color": "333333", "spaceAfter": "6pt",
                                     "lineSpacing": "1.15"}})

        # Table
        if section.table:
            headers = section.table.get("headers", [])
            rows = section.table.get("rows", [])
            if headers and rows:
                items.append({"command": "add", "parent": "/body", "type": "table",
                               "props": {"rows": len(rows) + 1, "cols": len(headers),
                                         "width": "100%"}})
                # Header row
                for j, h in enumerate(headers):
                    items.append({"command": "set",
                                   "path": f"/body/tbl[last()]/tr[1]",
                                   "props": {f"c{j+1}": h, "header": "true"}})
                # Data rows
                for ri, row in enumerate(rows, 2):
                    for j, val in enumerate(row):
                        items.append({"command": "set",
                                       "path": f"/body/tbl[last()]/tr[{ri}]",
                                       "props": {f"c{j+1}": str(val)}})

        # Chart (as an image-like element — docx charts via OfficeCLI)
        if section.chart:
            categories, series_data = _data_to_chart_format(data)
            if categories:
                items.append({"command": "add", "parent": "/body", "type": "chart",
                               "props": {"chartType": section.chart.get("type", "bar"),
                                         "title": section.chart.get("title", section.heading),
                                         "categories": categories,
                                         "data": series_data,
                                         "width": "15cm", "height": "8cm"}})

    # Sources section (if we have retrieved docs)
    # The tool receives retrieved_docs via state, but we only add sources
    # when the LLM explicitly requests them via a section heading "Sources".

    return items


def _build_xlsx_batch(spec: OfficeGenerateInput, data: list[dict]) -> list[dict]:
    """Build OfficeCLI batch items for an Excel workbook."""
    palette = THEMES.get(spec.theme or "midnight", THEMES["midnight"])
    items: list[dict] = []

    for si, sheet in enumerate(spec.sheets or [], 1):
        sheet_name = sheet.name or f"Sheet{si}"
        if si == 1:
            # First sheet is created by default with create
            items.append({"command": "set", "path": f"/Sheet1",
                           "props": {"name": sheet_name[:31]}})
            sheet_path = f"/{sheet_name[:31]}"
        else:
            items.append({"command": "add", "parent": "/", "type": "sheet",
                           "props": {"name": sheet_name[:31]}})
            sheet_path = f"/{sheet_name[:31]}"

        # Headers
        headers = sheet.headers or []
        rows = sheet.rows or []

        # If no explicit rows but we have accumulated_data, use it
        if not rows and data and headers:
            rows = [[d.get(h.lower().replace(" ", "_"), d.get("label", "")) for h in headers] for d in data]

        if headers:
            for j, h in enumerate(headers):
                col = chr(ord("A") + j) if j < 26 else f"A{chr(ord('A') + j - 26)}"
                items.append({"command": "set", "path": f"{sheet_path}/{col}1",
                               "props": {"value": h, "bold": "true",
                                         "font.color": palette["accent"]}})
            # Data rows
            for ri, row in enumerate(rows, 2):
                for j, val in enumerate(row):
                    col = chr(ord("A") + j) if j < 26 else f"A{chr(ord('A') + j - 26)}"
                    items.append({"command": "set", "path": f"{sheet_path}/{col}{ri}",
                                   "props": {"value": val}})

            # Chart
            if sheet.chart:
                chart_type = sheet.chart.get("type", "bar")
                chart_title = sheet.chart.get("title", sheet_name)
                n_rows = len(rows)
                n_cols = len(headers)
                if n_rows > 0 and n_cols > 0:
                    end_col = chr(ord("A") + n_cols - 1) if n_cols <= 26 else "Z"
                    data_range = f"{sheet_name[:31]}!A1:{end_col}{n_rows + 1}"
                    items.append({"command": "add", "parent": sheet_path, "type": "chart",
                                   "props": {"chartType": chart_type,
                                             "title": chart_title,
                                             "dataRange": data_range}})

    return items


# ── Tool ─────────────────────────────────────────────────────────────────────

class OfficeGenerateTool(BaseAgentTool):
    name: str = "office_generate"
    ui_label: str = "Generating Office document"
    description: str = (
        "Create or append to an Office document. Only three formats supported: pptx, docx, xlsx. "
        "Any other format will be rejected. "
        "Data is read automatically from state.accumulated_data — do NOT pass data values. "
        "Provide only structure: format, title, slides/sections/sheets, chart types, theme. "
        "For multi-slide decks: call office_generate with 1-2 slides at a time. "
        "First call creates the file (append=false). Subsequent calls use append=true "
        "to add slides to the same file. This avoids JSON corruption from large tool calls. "
        "Call office_load_skill first to get design guidelines. "
        "Returns file_id for download."
    )
    args_schema: type[BaseModel] = OfficeGenerateInput

    @staticmethod
    def _normalize_keys(obj: Any) -> Any:
        """Strip extra quotes from dict keys (LLM tool call parsing artifact).

        Some LLM providers return nested JSON with keys like '"title"' instead
        of 'title'. This recursively normalizes all dict keys.
        """
        if isinstance(obj, dict):
            return {
                (k.strip('"\'') if isinstance(k, str) else k): OfficeGenerateTool._normalize_keys(v)
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [OfficeGenerateTool._normalize_keys(item) for item in obj]
        return obj

    def prepare_arguments(self, args: dict) -> dict:
        """Normalize nested dict keys before Pydantic validation."""
        return self._normalize_keys(args)

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use arun() for agent tools.")

    async def _execute(self, input_obj: OfficeGenerateInput) -> dict:
        t0 = time.monotonic()
        ctx: ToolContext = self.ctx

        # 1. Check OfficeCLI binary
        binary = get_setting(ctx.db, "OFFICECLI_BINARY_PATH", ctx.org_id)
        if not shutil.which(binary):
            return {"ok": False, "result": {}, "error": f"OfficeCLI binary not found at '{binary}'. Install it or set OFFICECLI_BINARY_PATH.", "tokens": 0}

        # 2. Read data from state (optional — text-only docs don't need it)
        data = ctx.state.get("accumulated_data", []) if ctx.state else []

        if not (input_obj.slides or input_obj.sections or input_obj.sheets):
            return {"ok": False, "result": {}, "error": "No document structure provided. Provide slides (for pptx), sections (for docx), or sheets (for xlsx) with titles and content.", "tokens": 0}

        # 3. Prepare work directory
        work_dir = get_setting(ctx.db, "OFFICECLI_WORK_DIR", ctx.org_id)
        chat_work_dir = os.path.join(work_dir, str(ctx.chat_id or "default"))
        os.makedirs(chat_work_dir, exist_ok=True)

        fmt = input_obj.format.lower()

        # 4. Build batch items
        if fmt == "pptx":
            items = _build_pptx_batch(input_obj, data)
        elif fmt == "docx":
            items = _build_docx_batch(input_obj, data)
        elif fmt == "xlsx":
            items = _build_xlsx_batch(input_obj, data)
        else:
            return {"ok": False, "result": {}, "error": f"Unsupported format: {fmt}. Use pptx, docx, or xlsx.", "tokens": 0}

        if not items:
            return {"ok": False, "result": {}, "error": "No content generated. Provide slides, sections, or sheets.", "tokens": 0}

        # 5. Determine file path — append to existing or create new
        existing_file_id = None
        existing_chat_file = None

        if input_obj.append and ctx.state:
            # Find the last generated file of the same format
            gen_files = ctx.state.get("generated_files", []) or []
            last_match = None
            for f in reversed(gen_files):
                if f.get("format") == fmt:
                    last_match = f
                    break
            if last_match:
                existing_file_id = last_match["file_id"]
                existing_chat_file = ctx.db.query(ChatFile).filter(ChatFile.id == existing_file_id).first()
                if existing_chat_file and os.path.exists(existing_chat_file.stored_path):
                    # Copy existing file to work dir for modification
                    file_name = existing_chat_file.file_name
                    file_path = os.path.join(chat_work_dir, file_name)
                    shutil.copy2(existing_chat_file.stored_path, file_path)
                else:
                    existing_chat_file = None
                    input_obj.append = False

        if not input_obj.append:
            file_name = _generate_filename(input_obj)
            file_path = os.path.join(chat_work_dir, file_name)

        # 6. Execute via OfficeCLI SDK
        try:
            import officecli
            if input_obj.append and existing_chat_file:
                # Open existing file and add items
                with officecli.open(file_path, binary=binary, auto_install=False) as doc:
                    for chunk_start in range(0, len(items), 50):
                        chunk = items[chunk_start:chunk_start + 50]
                        doc.batch(chunk)
                    doc.send({"command": "save"})
            else:
                # Create new file
                with officecli.create(file_path, "--force", binary=binary, auto_install=False) as doc:
                    for chunk_start in range(0, len(items), 50):
                        chunk = items[chunk_start:chunk_start + 50]
                        doc.batch(chunk)
                    doc.send({"command": "save"})
        except ImportError:
            return {"ok": False, "result": {}, "error": "officecli SDK not installed. Run: pip install officecli-sdk", "tokens": 0}
        except Exception as exc:
            logger.warning("[office_generate] OfficeCLI failed: %s", exc)
            return {"ok": False, "result": {}, "error": f"OfficeCLI error: {exc}", "tokens": 0}

        # 7. Read the generated file
        if not os.path.exists(file_path):
            return {"ok": False, "result": {}, "error": "OfficeCLI did not produce a file.", "tokens": 0}

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        if not file_bytes:
            return {"ok": False, "result": {}, "error": "Generated file is empty.", "tokens": 0}

        # 8. Save to ephemeral storage (overwrites prior stored_path on append)
        stored_path = save_ephemeral_file(ctx.chat_id, file_name, file_bytes)

        if existing_chat_file:
            # Update existing ChatFile record with new size and path
            existing_chat_file.stored_path = stored_path
            existing_chat_file.file_size = len(file_bytes)
            ctx.db.commit()
            ctx.db.refresh(existing_chat_file)
            chat_file = existing_chat_file
        else:
            # Create new ChatFile record
            chat_file = ChatFile(
                chat_id=ctx.chat_id,
                message_id=ctx.message_id,
                file_name=file_name,
                stored_path=stored_path,
                file_size=len(file_bytes),
                content_type=_CONTENT_TYPES.get(fmt, "application/octet-stream"),
                status="ready",
                is_generated=True,
            )
            ctx.db.add(chat_file)
            ctx.db.commit()
            ctx.db.refresh(chat_file)

        # 9. Update state — replace existing file_ref or add new one
        file_ref = {
            "file_id": chat_file.id,
            "file_name": file_name,
            "format": fmt,
            "path": stored_path,
            "title": input_obj.title,
        }
        if ctx.state is not None:
            existing = ctx.state.get("generated_files", []) or []
            if existing_file_id:
                ctx.state["generated_files"] = [
                    file_ref if f.get("file_id") == existing_file_id else f
                    for f in existing
                ]
            else:
                ctx.state["generated_files"] = existing + [file_ref]

        # 10. Clean up temp file
        try:
            os.remove(file_path)
        except OSError:
            pass

        # 11. Emit file event for real-time frontend display
        from app.services.agentic_rag.agent_graph.helpers import _writer
        writer = _writer()
        writer({
            "event": "file",
            "file_id": chat_file.id,
            "file_name": file_name,
            "format": fmt,
            "title": input_obj.title,
        })

        # 12. Count charts
        chart_count = 0
        for s in input_obj.slides or []:
            if s.chart_type:
                chart_count += 1
        for s in input_obj.sections or []:
            if s.chart:
                chart_count += 1
        for s in input_obj.sheets or []:
            if s.chart:
                chart_count += 1

        latency_ms = round((time.monotonic() - t0) * 1000)
        write_audit(ctx, "office_generate", {"format": fmt, "title": input_obj.title, "append": input_obj.append},
                    {"file_id": chat_file.id, "file_size": len(file_bytes), "chart_count": chart_count},
                    latency_ms=latency_ms, status="ok")

        return {
            "ok": True,
            "result": {
                "file_id": chat_file.id,
                "file_name": file_name,
                "format": fmt,
                "title": input_obj.title,
                "path": stored_path,
                "file_size": len(file_bytes),
                "appended": input_obj.append,
                "slide_count": len(input_obj.slides) if input_obj.slides else None,
                "section_count": len(input_obj.sections) if input_obj.sections else None,
                "sheet_count": len(input_obj.sheets) if input_obj.sheets else None,
                "chart_count": chart_count,
            },
            "error": None,
            "tokens": 50,
        }
