"""office_load_skill tool — load OfficeCLI design guidelines on demand.

Reads the vendored OfficeCLI skill files from backend/skills/ and extracts
the sections the LLM needs: design principles, creating/editing (charts,
animations, connectors, groups), QA workflow, and pitfalls. Setup, help,
and shell-quoting sections are stripped (irrelevant — office_generate uses
the Python SDK, not the CLI). Specialized profiles (pitch-deck, data-dashboard,
financial-model, academic-paper) are returned in full minus the same
irrelevant sections.
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
# __file__ is at backend/app/services/agentic_rag/tools/office_load_skill.py
# Four dirs up = backend/ (the project root inside the container is /app).
_SKILLS_BASE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "skills")

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

# Sections to always strip — irrelevant because office_generate uses the
# Python SDK, not the CLI shell. These are the first 2-3 sections in every
# skill file.
_STRIP_SECTIONS: set[str] = {
    "## Setup",
    "## ⚠️ Help-First Rule",
    "## Help-First Rule",
    "## Shell & Execution Discipline",
    "## Mental Model & Inheritance",
    "## Mental Model",
}

# For base skills only: extract only these sections (skip Common Workflow,
# Quick Start, Reading & Analysis, CSV/bulk import, Raw-set XML appendix).
# These sections are CLI-usage tutorials not needed by the SDK-based generator.
_KEEP_SECTIONS_BASE: set[str] = {
    "## Requirements for Outputs",
    "## Design Principles",
    "## Creating & Editing",
    "## Chart Axis-by-Role",
    "## QA (Required)",
    "## Common Pitfalls",
    "## Known Issues & Pitfalls",
}


def _strip_frontmatter(content: str) -> str:
    """Remove YAML frontmatter (--- ... ---) if present."""
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            return content[end + 3:].lstrip()
    return content


def _extract_sections(content: str, keep: set[str] | None = None) -> str:
    """Split by ## headers and return selected sections.

    If keep is None, returns all sections except those in _STRIP_SECTIONS
    (used for specialized profiles).
    If keep is provided, returns only those sections that are in both
    keep and not in _STRIP_SECTIONS (used for base skills).
    """
    lines = content.split("\n")
    sections: list[tuple[str, list[str]]] = []
    current_header: str | None = None
    current_body: list[str] = []

    for line in lines:
        if line.startswith("## "):
            if current_header is not None:
                sections.append((current_header, current_body))
            current_header = line.strip()
            current_body = []
        else:
            current_body.append(line)
    if current_header is not None:
        sections.append((current_header, current_body))

    result_parts: list[str] = []
    for header, body in sections:
        if header in _STRIP_SECTIONS:
            continue
        if keep is not None and header not in keep:
            continue
        result_parts.append(header + "\n" + "\n".join(body).rstrip())

    return "\n\n".join(result_parts).strip()


def _load_skill_content(skill_rel: str, is_base: bool) -> str:
    """Read and extract relevant sections from a vendored skill file.

    Base skills: extract only design principles, creating/editing, QA, pitfalls.
    Specialized profiles: return full file minus setup/help/shell sections.
    """
    skill_path = os.path.join(_SKILLS_BASE, skill_rel)
    if not os.path.exists(skill_path):
        raise FileNotFoundError(f"Skill file not found: {skill_rel}")

    with open(skill_path, encoding="utf-8") as f:
        raw = f.read()

    content = _strip_frontmatter(raw)
    if is_base:
        return _extract_sections(content, keep=_KEEP_SECTIONS_BASE)
    else:
        return _extract_sections(content, keep=None)


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
        "layout rules, chart formats, QA workflow, and quality check criteria. "
        "Only call once per turn."
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

        is_base = skill_name == "base"
        try:
            content = _load_skill_content(skill_rel, is_base)
        except FileNotFoundError as exc:
            return {"ok": False, "result": {}, "error": str(exc), "tokens": 0}

        latency_ms = round((time.monotonic() - t0) * 1000)
        write_audit(ctx, "office_load_skill", input_obj.model_dump(),
                    {"format": fmt, "skill": skill_name, "content_len": len(content)},
                    latency_ms=latency_ms, status="ok")

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
