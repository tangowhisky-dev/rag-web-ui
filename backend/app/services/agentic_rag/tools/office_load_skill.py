"""office_load_skill tool — load OfficeCLI design guidelines on demand.

Returns a condensed summary of the relevant OfficeCLI skill file so the LLM
knows design guidelines, font hierarchies, quality standards, and command
syntax for the target format. The LLM calls this BEFORE office_generate.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

from pydantic import BaseModel, Field

from app.services.agentic_rag.tool_context import ToolContext, write_audit
from app.services.agentic_rag.tools.base import BaseAgentTool

logger = logging.getLogger(__name__)

# Skill files are vendored at backend/skills/ (copied from the OfficeCLI repo).
_SKILLS_BASE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "skills")

SKILL_MAP: dict[str, dict[str, str]] = {
    "pptx": {
        "base": "officecli-pptx/SKILL.md",
        "pitch-deck": "officecli-pitch-deck/SKILL.md",
    },
    "docx": {
        "base": "officecli-docx/SKILL.md",
        "academic-paper": "officecli-academic-paper/SKILL.md",
    },
    "xlsx": {
        "base": "officecli-xlsx/SKILL.md",
        "data-dashboard": "officecli-data-dashboard/SKILL.md",
        "financial-model": "officecli-financial-model/SKILL.md",
    },
}

# Condensed design rules per format — extracted from the full skill files.
# These are the key rules the LLM needs; the full files are too large (40-67KB)
# to inject into the think prompt without flooding the context window.
_CONDENSED_RULES: dict[str, str] = {
    "pptx": """## PowerPoint Design Rules (Condensed)

### Typography
- Title: >=36pt bold, Georgia or Trebuchet MS
- Body: >=18pt, Calibri
- Maximum 2 fonts per deck
- One idea per slide

### Color Palette (pick one theme)
- midnight: bg=#0D1B2A, text=#FFFFFF, accent=#CADCFC
- coral: bg=#FFFFFF, text=#333333, accent=#F96167
- ocean: bg=#1A1A2E, text=#FFFFFF, accent=#16213E
- forest: bg=#F5F5F0, text=#1B1B1B, accent=#2D5016
- slate: bg=#1E2761, text=#FFFFFF, accent=#CADCFC

### Layout (widescreen 33.87cm x 19.05cm)
- Margins: 1.27cm all sides
- Title: x=1.27cm, y=1.27cm, width=31.33cm, height=2.5cm
- Body: x=1.27cm, y=4cm, width=31.33cm, height=12cm
- Chart: x=2cm, y=5cm (or 11cm if bullets above), width=29cm, height=7cm

### Charts
- Chart types: bar, line, pie, column, scatter, area, doughnut
- Data format: categories="Q1,Q2,Q3,Q4", data="Series1:1,2,3,4;Series2:5,6,7,8"
- Multi-series: separate series with semicolons
- Always include legend: "bottom"

### Quality Checks
- Every slide should have a non-text visual (chart or image)
- Speaker notes on content slides
- No text overflow — keep bullets to 5 per slide, 1 line each
- Validate with office_inspect(mode=issues) after generation
""",
    "docx": """## Word Document Design Rules (Condensed)

### Typography
- Title: 24pt bold, Georgia, accent color
- H1: 18pt bold, Georgia, accent color
- H2: 14pt bold, Georgia, accent color
- H3: 12pt bold, Georgia, accent color
- Body: 11pt, Calibri, #333333, line spacing 1.15

### Color Palette
- accent: #2D5016 (forest) or #1E2761 (slate) or #F96167 (coral)
- body text: #333333
- muted: #666666

### Structure
- Title page: Title (24pt) + subtitle (14pt, muted)
- Body: H1 → H2 → H3 → paragraphs
- Tables: header row bold with accent color, 100% width
- Charts: 15cm wide, 8cm tall

### Charts
- Same data format as PPTX: categories="...", data="Series:..."
- Chart types: bar, line, pie, column, scatter, area

### Quality Checks
- Clear heading hierarchy (H1 > H2 > H3)
- No empty sections
- Validate with office_inspect(mode=issues) after generation
""",
    "xlsx": """## Excel Workbook Design Rules (Condensed)

### Structure
- Sheet names: max 31 chars, descriptive
- First row: bold headers with accent color
- Data starts row 2
- One sheet per topic

### Formatting
- Headers: bold, accent color (e.g. #2D5016 or #1E2761)
- Number columns: right-aligned
- Date columns: ISO format (YYYY-MM-DD)

### Charts
- Chart types: bar, line, pie, column, scatter, area
- Data range: "SheetName!A1:D10" (includes header row)
- Title: descriptive, matches sheet purpose

### Formulas
- Use formulas for computed columns (e.g. =SUM(B2:B10), =AVERAGE(C2:C10))
- Reference cells by address (A1, B2, etc.)

### Quality Checks
- No empty sheets
- Headers present on all sheets
- Validate with office_inspect(mode=issues) after generation
""",
}


class OfficeLoadSkillInput(BaseModel):
    format: str = Field(description="Target document format: pptx, docx, or xlsx")
    skill: str = Field(
        default="base",
        description="Skill profile: 'base' for the format-specific guidelines, or a specialized profile like 'pitch-deck', 'data-dashboard', 'financial-model', 'academic-paper'",
    )


class OfficeLoadSkillTool(BaseAgentTool):
    name: str = "office_load_skill"
    ui_label: str = "Loading OfficeCLI design guidelines"
    description: str = (
        "Load OfficeCLI design guidelines for the target document format. "
        "Call this BEFORE office_generate to get font sizes, color palettes, "
        "layout rules, and quality check criteria. Returns a condensed summary "
        "of key design rules. Only call once per turn."
    )
    args_schema: type[BaseModel] = OfficeLoadSkillInput

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use arun() for agent tools.")

    async def _execute(self, input_obj: OfficeLoadSkillInput) -> dict:
        t0 = time.monotonic()
        ctx: ToolContext = self.ctx

        fmt = input_obj.format.lower()
        skill_name = input_obj.skill or "base"

        format_map = SKILL_MAP.get(fmt)
        if not format_map:
            return {"ok": False, "result": {}, "error": f"Unsupported format: {fmt}. Use pptx, docx, or xlsx.", "tokens": 0}

        skill_rel = format_map.get(skill_name)
        if not skill_rel:
            available = ", ".join(format_map.keys())
            return {"ok": False, "result": {}, "error": f"Skill '{skill_name}' not found for format {fmt}. Available: {available}", "tokens": 0}

        # Return condensed rules instead of the full skill file (which is 40-67KB).
        # The condensed rules contain the key design parameters the LLM needs.
        content = _CONDENSED_RULES.get(fmt, "")
        if not content:
            # Fallback: read the full file but truncate to first 2000 chars
            skill_path = os.path.join(_SKILLS_BASE, skill_rel)
            if os.path.exists(skill_path):
                with open(skill_path, encoding="utf-8") as f:
                    content = f.read()[:2000] + "\n... (truncated)"
            else:
                return {"ok": False, "result": {}, "error": f"Skill file not found: {skill_rel}", "tokens": 0}

        latency_ms = round((time.monotonic() - t0) * 1000)
        write_audit(ctx, "office_load_skill", input_obj.model_dump(), {"format": fmt, "skill": skill_name, "content_len": len(content)}, latency_ms=latency_ms, status="ok")

        return {
            "ok": True,
            "result": {
                "skill_content": content,
                "format": fmt,
                "skill": skill_name,
            },
            "error": None,
            "tokens": len(content) // 4,
        }
