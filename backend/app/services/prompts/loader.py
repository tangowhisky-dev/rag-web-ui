"""Prompt file loader.

Loads markdown prompt files from app/prompts/ and provides
a function to append chart generation instructions to system prompts.
"""

from __future__ import annotations

import os
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_chart_instructions() -> str:
    """Load chart generation instructions from charts-documentation.md.

    Returns an empty string if the file doesn't exist (graceful degradation).
    """
    path = _PROMPTS_DIR / "charts-documentation.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def append_chart_instructions(system_prompt: str) -> str:
    """Append chart generation instructions to a system prompt.

    Only appends if chart instructions exist and the prompt doesn't
    already contain chart-related content (idempotent guard).
    """
    chart_instructions = load_chart_instructions()
    if not chart_instructions:
        return system_prompt
    if "echarts" in system_prompt.lower():
        return system_prompt
    return f"{system_prompt}\n\n## Chart Generation\n{chart_instructions}"
