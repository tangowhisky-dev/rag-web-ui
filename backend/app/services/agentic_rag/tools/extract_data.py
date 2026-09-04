"""extract_data tool — pull numbers/stats from previous answer, retrieved docs, or files.

Supports a map-reduce pattern for aggregate queries:
- source="retrieved_docs" with document_ids=[1,2,3] processes each document
  separately and accumulates results into state["accumulated_data"].
- source="accumulated" reads previously accumulated data (for chart_generate).
- Multiple extract_data calls across batches append to accumulated_data.
- chart_generate reads from accumulated_data via the "accumulated" source.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from app.models.chat import ChatFile, Message
from app.services.agentic_rag.llm_factory import build_chat_llm
from app.services.agentic_rag.prompts import EXTRACT_DATA_PROMPT
from app.services.agentic_rag.schemas import DataPoint, LastAnswerObject
from app.services.agentic_rag.tool_context import ToolContext, enforce_rbac, write_audit
from app.services.agentic_rag.tools.base import BaseAgentTool

logger = logging.getLogger(__name__)

_MAX_LLM_RETRIES = 3
# Max chars per document when extracting — prevents single-doc overflow.
_MAX_CHARS_PER_DOC = 8000


class ExtractDataInput(BaseModel):
    source: str = Field(
        default="last_answer",
        description="Source of text to extract from: last_answer, retrieved_docs, "
        "accumulated, file, or specified. Use 'retrieved_docs' with document_ids "
        "for batch extraction from specific documents. Use 'accumulated' to return "
        "previously accumulated data (e.g. before chart_generate).",
    )
    source_id: Optional[int] = Field(
        default=None,
        description="For source='file': the ChatFile id. For source='specified': the Message id.",
    )
    document_ids: Optional[List[int]] = Field(
        default=None,
        description="For source='retrieved_docs': extract from only these document_ids "
        "(from kb_search_documents metadata). If null, extracts from all retrieved docs "
        "(first 10). Use this for batch processing: call extract_data with document_ids "
        "for 5-10 docs at a time, then chart_generate with source='accumulated'.",
    )
    focus: Optional[str] = Field(
        default=None,
        description="What kind of data to extract, e.g. 'monthly counts', 'topics covered', 'revenue'.",
    )


_NUMBER_RE = re.compile(r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*([a-zA-Z%$€£]+)?")


def _rule_based_extract(text: str, focus: Optional[str] = None) -> list[dict]:
    """Fallback regex extractor for numbers near label words."""
    points = []
    for m in _NUMBER_RE.finditer(text):
        value = m.group(1).replace(",", "")
        unit = (m.group(2) or "").strip()
        start = max(0, m.start() - 40)
        context = text[start : m.end() + 40].replace("\n", " ")
        points.append({
            "label": focus or "statistic",
            "value": float(value) if "." in value else int(value),
            "unit": unit or None,
            "context": context,
        })
    return points[:20]


def _extract_from_chart_options(chart_options: list[dict]) -> list[dict]:
    """Extract {label, value} pairs deterministically from ECharts option dicts."""
    points: list[dict] = []
    for opt in chart_options:
        labels = opt.get("xAxis", {}).get("data", []) if isinstance(opt.get("xAxis"), dict) else []
        series_list = opt.get("series", [])
        if not series_list:
            continue
        values = series_list[0].get("data", [])
        for j in range(min(len(labels), len(values))):
            try:
                val = float(values[j]) if isinstance(values[j], (int, float)) else values[j]
            except (TypeError, ValueError):
                continue
            points.append({
                "label": str(labels[j]),
                "value": val,
                "unit": None,
                "context": f"Chart data point {j + 1}",
            })
    return points


class _ExtractResult(BaseModel):
    """Structured output schema for LLM-based extraction."""
    data: List[DataPoint] = Field(default_factory=list, description="Extracted data points.")


def _extract_from_last_answer(
    input_obj: ExtractDataInput, ctx: ToolContext, tool: ExtractDataTool, t0: float,
) -> tuple[str, Optional[dict]]:
    lao = ctx.state.get("last_answer_object") if ctx.state else None
    if lao and isinstance(lao, LastAnswerObject) and lao.data:
        points = [
            {"label": dp.label, "value": dp.value, "unit": dp.unit, "context": dp.context}
            for dp in lao.data
        ]
        return "", tool._finish(input_obj, points, 0, t0)
    if lao and isinstance(lao, LastAnswerObject) and lao.chart_options:
        points = _extract_from_chart_options(lao.chart_options)
        if points:
            return "", tool._finish(input_obj, points, 0, t0)
    text = ""
    if lao and hasattr(lao, "summary"):
        text = lao.summary + "\n" + "\n".join(lao.key_points or [])
    if not text:
        text = "No previous answer available."
    return text, None


def _select_docs_by_ids(docs: list[dict], document_ids: list[int]) -> list[dict]:
    """Filter retrieved_docs to only those matching document_ids."""
    selected = []
    for d in docs:
        meta = d.get("metadata", {}) if isinstance(d, dict) else {}
        doc_id = meta.get("document_id")
        if doc_id in document_ids:
            selected.append(d)
    return selected


def _extract_from_retrieved_docs(
    input_obj: ExtractDataInput, ctx: ToolContext, tool: ExtractDataTool, t0: float,
) -> tuple[str, Optional[dict]]:
    """Return text from retrieved docs, optionally filtered by document_ids.

    When document_ids is specified, returns each document's content separated
    by a header so the LLM can attribute extracted data to specific documents.
    When document_ids is null, returns the first 10 docs concatenated (legacy behavior).
    """
    docs = ctx.state.get("retrieved_docs", []) if ctx.state else []

    if input_obj.document_ids:
        docs = _select_docs_by_ids(docs, input_obj.document_ids)
        if not docs:
            return "", {"ok": False, "result": {}, "error": "No retrieved docs match the specified document_ids.", "tokens": 0}
        # Build text with document headers so the LLM can attribute data.
        parts = []
        for d in docs:
            meta = d.get("metadata", {})
            title = meta.get("title", f"doc_{meta.get('document_id', '?')}")
            content = d.get("page_content", "")[:_MAX_CHARS_PER_DOC]
            parts.append(f"--- Document: {title} (id={meta.get('document_id')}) ---\n{content}")
        return "\n\n".join(parts), None
    else:
        # Legacy: first 10 docs, concatenated.
        parts = []
        for d in docs[:10]:
            parts.append(d.get("page_content", ""))
        return "\n\n".join(parts), None


def _extract_from_file(
    input_obj: ExtractDataInput, ctx: ToolContext, tool: ExtractDataTool, t0: float,
) -> tuple[str, Optional[dict]]:
    if not input_obj.source_id:
        return "Unsupported source.", None
    rbac = enforce_rbac(ctx, file_id=input_obj.source_id)
    if rbac.get("file_id") is None:
        return "", {"ok": False, "result": {}, "error": "Access denied to file.", "tokens": 0}
    cf = ctx.db.query(ChatFile).filter(ChatFile.id == rbac["file_id"]).first()
    text = cf.markdown_content or "" if cf else ""
    return text, None


def _extract_from_specified(
    input_obj: ExtractDataInput, ctx: ToolContext, tool: ExtractDataTool, t0: float,
) -> tuple[str, Optional[dict]]:
    if not input_obj.source_id:
        return "Unsupported source.", None
    if not ctx.chat_id:
        return "", {"ok": False, "result": {}, "error": "Access denied: no chat context.", "tokens": 0}
    msg = ctx.db.query(Message).filter(
        Message.id == input_obj.source_id,
        Message.chat_id == ctx.chat_id,
    ).first()
    text = msg.content if msg else ""
    if not text:
        text = "Message not found."
    return text, None


def _extract_from_accumulated(
    input_obj: ExtractDataInput, ctx: ToolContext, tool: ExtractDataTool, t0: float,
) -> tuple[str, Optional[dict]]:
    """Return previously accumulated data points directly — no LLM call needed."""
    accumulated = ctx.state.get("accumulated_data", []) if ctx.state else []
    if not accumulated:
        return "", {"ok": False, "result": {}, "error": "No accumulated data. Call extract_data with source='retrieved_docs' first.", "tokens": 0}
    return "", tool._finish(input_obj, accumulated, 0, t0)


SOURCE_EXTRACTORS = {
    "last_answer": _extract_from_last_answer,
    "retrieved_docs": _extract_from_retrieved_docs,
    "accumulated": _extract_from_accumulated,
    "file": _extract_from_file,
    "specified": _extract_from_specified,
}


async def _extract_with_llm(text: str, ctx: ToolContext, focus: Optional[str]) -> list[dict]:
    prompt = EXTRACT_DATA_PROMPT.format(
        focus=focus or 'any statistics',
        text=text,
    )
    points: list[dict] = []
    try:
        llm = build_chat_llm(ctx.org_id, ctx.db, role="query", temperature=0.0)

        for attempt in range(_MAX_LLM_RETRIES):
            try:
                structured = llm.with_structured_output(_ExtractResult, method="json_schema")
                result = await structured.ainvoke([{"role": "user", "content": prompt}])
                if result and result.data:
                    points = [
                        {"label": dp.label, "value": dp.value, "unit": dp.unit, "context": dp.context}
                        for dp in result.data
                    ]
                    break
            except Exception as exc:
                logger.debug("[extract_data] structured output attempt %d failed: %s", attempt + 1, exc)

        if not points:
            from app.services.agentic_rag.agent_graph import _extract_json_block
            response = await llm.ainvoke([{"role": "user", "content": prompt}])
            raw = str(response.content)
            block = _extract_json_block(raw)
            if block:
                try:
                    points = json.loads(block)
                except json.JSONDecodeError:
                    from json_repair import repair_json
                    points = json.loads(repair_json(block))
                if not isinstance(points, list):
                    points = []
    except Exception as exc:
        logger.warning("[extract_data] LLM extraction failed: %s", exc)

    if not points:
        points = _rule_based_extract(text, focus)
    return points


def _validate_points(points: list[dict]) -> list[dict]:
    validated = []
    for p in points:
        try:
            dp = DataPoint(**p)
            validated.append({"label": dp.label, "value": dp.value, "unit": dp.unit, "context": dp.context})
        except Exception:
            validated.append(p)
    return validated


class ExtractDataTool(BaseAgentTool):
    name: str = "extract_data"
    ui_label: str = "Extracting data"
    description: str = (
        "Extract structured data from the previous answer, retrieved documents, "
        "an attached file, a specified message, or previously accumulated data. "
        "Use source='retrieved_docs' with document_ids to extract from specific "
        "documents in batches. Results accumulate in state — call with "
        "source='accumulated' to retrieve all accumulated data before chart_generate. "
        "Sources: last_answer, retrieved_docs, accumulated, file, specified."
    )
    args_schema: type[BaseModel] = ExtractDataInput

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use arun() for agent tools.")

    async def _execute(self, input_obj: ExtractDataInput) -> dict:
        t0 = time.monotonic()
        ctx: ToolContext = self.ctx

        extractor = SOURCE_EXTRACTORS.get(input_obj.source)
        if extractor:
            text, early_return = extractor(input_obj, ctx, self, t0)
        else:
            text, early_return = "Unsupported source.", None

        if early_return is not None:
            return early_return

        # Truncate to prevent LLM overflow, but allow more room when
        # processing specific documents (they're already individually capped).
        max_chars = 24000 if input_obj.document_ids else 6000
        text = text[:max_chars]
        points = await _extract_with_llm(text, ctx, input_obj.focus)
        validated = _validate_points(points)

        # Accumulate into state for chart_generate to read later.
        if input_obj.source == "retrieved_docs" and ctx.state is not None:
            existing = ctx.state.get("accumulated_data", []) or []
            existing = existing + validated
            # Write back so tool_node can pick it up via state_update.
            # tool_node exposes ctx.state as the live state; the return
            # dict is merged by the graph.
            ctx.state["accumulated_data"] = existing

        latency_ms = round((time.monotonic() - t0) * 1000)
        return self._finish(input_obj, validated, latency_ms, t0)

    def _finish(self, input_obj: ExtractDataInput, points: list[dict], latency_ms: int, t0: float) -> dict:
        accumulated_count = 0
        if self.ctx and self.ctx.state:
            accumulated_count = len(self.ctx.state.get("accumulated_data", []) or [])
        write_audit(
            self.ctx,
            "extract_data",
            input_obj.model_dump(),
            {"point_count": len(points), "accumulated_total": accumulated_count},
            latency_ms=latency_ms,
            status="ok",
        )
        return {
            "ok": True,
            "result": {
                "data": points,
                "count": len(points),
                "accumulated_total": accumulated_count,
            },
            "error": None,
            "tokens": len(str(points)) // 4,
        }
