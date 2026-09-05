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


def append_chart_placeholder_instructions(system_prompt: str, chart_count: int) -> str:
    """Append instructions telling the model to place chart markers, not JSON.

    Used when chart_generate already produced valid chart_option(s) this turn:
    the model never needs to write ECharts JSON itself, it only marks where
    each chart belongs. finalize_node substitutes the markers deterministically.
    """
    if chart_count <= 0:
        return system_prompt
    if chart_count == 1:
        markers = "the literal marker [[CHART_1]]"
    else:
        markers = "the literal markers " + ", ".join(f"[[CHART_{i}]]" for i in range(1, chart_count + 1))
    instructions = (
        f"\n\n## Chart Placement\n"
        f"{chart_count} chart(s) have already been generated for this answer. "
        f"Do NOT write any chart JSON or ```echarts code yourself. "
        f"Instead, insert {markers} at the point(s) in your answer where each "
        f"chart is most relevant, each marker exactly once. The markers will be "
        f"replaced with the actual charts automatically."
    )
    return f"{system_prompt}{instructions}"


def append_office_placeholder_instructions(system_prompt: str, office_files: list[dict]) -> str:
    """Append instructions telling the model to place document download markers.

    Used when office_generate already produced document(s) this turn:
    the model inserts [[DOC_N]] markers where download links should appear.
    finalize_node substitutes the markers with actual download links.
    """
    if not office_files:
        return system_prompt
    count = len(office_files)
    if count == 1:
        markers = "the literal marker [[DOC_1]]"
    else:
        markers = "the literal markers " + ", ".join(f"[[DOC_{i}]]" for i in range(1, count + 1))
    file_desc = "; ".join(
        f"{f.get('format', '').upper()}: {f.get('title') or f.get('file_name', 'document')}"
        for f in office_files
    )
    instructions = (
        f"\n\n## Office Document Placement\n"
        f"{count} Office document(s) have been generated: {file_desc}. "
        f"Insert {markers} at the point(s) in your answer where each "
        f"download link should appear, each marker exactly once. "
        f"The markers will be replaced with download links automatically. "
        f"After each marker, include a one-line summary of what the document contains. "
        f"Do NOT describe the document's content in detail — the document itself "
        f"is the detailed output. Summarize what was created and highlight key findings."
    )
    return f"{system_prompt}{instructions}"
