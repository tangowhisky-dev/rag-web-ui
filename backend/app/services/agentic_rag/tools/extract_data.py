"""extract_data tool — pull numbers/stats from previous answer, retrieved docs, or files."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.models.chat import ChatFile, Message
from app.services.agentic_rag.llm_factory import build_chat_llm
from app.services.agentic_rag.schemas import DataPoint, LastAnswerObject
from app.services.agentic_rag.tool_context import ToolContext, enforce_rbac, write_audit
from app.services.agentic_rag.tools.base import BaseAgentTool

logger = logging.getLogger(__name__)


class ExtractDataInput(BaseModel):
    source: str = Field(
        default="last_answer",
        description="Source of text to extract from: last_answer, retrieved_docs, file, or specified.",
    )
    source_id: Optional[int] = Field(
        default=None,
        description="For source='file': the ChatFile id. For source='specified': the Message id.",
    )
    focus: Optional[str] = Field(default=None, description="What kind of numbers to extract, e.g. 'sales'.")


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


class ExtractDataTool(BaseAgentTool):
    name: str = "extract_data"
    description: str = (
        "Extract structured numbers and statistics from the previous answer, "
        "retrieved documents, or an attached file. Use before chart_generate."
    )
    args_schema: type[BaseModel] = ExtractDataInput

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use arun() for agent tools.")

    async def _execute(self, input_obj: ExtractDataInput) -> dict:
        t0 = time.monotonic()
        ctx: ToolContext = self.ctx

        text = ""
        if input_obj.source == "last_answer":
            lao = getattr(ctx.state, "last_answer_object", None) if ctx.state else None
            if lao and isinstance(lao, LastAnswerObject) and lao.data:
                # Already structured data; return it directly.
                points = [
                    {"label": dp.label, "value": dp.value, "unit": dp.unit, "context": dp.context}
                    for dp in lao.data
                ]
                return self._finish(input_obj, points, 0, t0)
            if lao and hasattr(lao, "summary"):
                text = lao.summary + "\n" + "\n".join(lao.key_points or [])
            if not text:
                text = "No previous answer available."
        elif input_obj.source == "retrieved_docs":
            docs = getattr(ctx.state, "retrieved_docs", []) if ctx.state else []
            parts = []
            for d in docs[:10]:
                parts.append(d.get("page_content", ""))
            text = "\n\n".join(parts)
        elif input_obj.source == "file_id" and input_obj.source_id:
            rbac = enforce_rbac(ctx, file_id=input_obj.source_id)
            if rbac.get("file_id") is None:
                return {"ok": False, "result": {}, "error": "Access denied to file.", "tokens": 0}
            cf = ctx.db.query(ChatFile).filter(ChatFile.id == rbac["file_id"]).first()
            text = cf.markdown_content or "" if cf else ""
        else:
            text = "Unsupported source."

        text = text[:6000]

        prompt = (
            "Extract all explicit numerical statistics from the text below. "
            "Return a JSON list of objects with keys: label, value, unit, context. "
            f"Focus: {input_obj.focus or 'any statistics'}.\n\n{text}"
        )

        points: list[dict] = []
        try:
            llm = build_chat_llm(ctx.org_id, ctx.db, role="query", temperature=0.0)
            response = await llm.ainvoke([{"role": "user", "content": prompt}])
            raw = str(response.content)
            # Try to find a JSON list in the response.
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                points = json.loads(match.group(0))
                if not isinstance(points, list):
                    points = []
        except Exception as exc:
            logger.warning("[extract_data] LLM extraction failed: %s", exc)

        if not points:
            points = _rule_based_extract(text, input_obj.focus)

        # Validate against DataPoint schema where possible.
        validated = []
        for p in points:
            try:
                dp = DataPoint(**p)
                validated.append({"label": dp.label, "value": dp.value, "unit": dp.unit, "context": dp.context})
            except Exception:
                validated.append(p)

        latency_ms = round((time.monotonic() - t0) * 1000)
        return self._finish(input_obj, validated, latency_ms, t0)

    def _finish(self, input_obj: ExtractDataInput, points: list[dict], latency_ms: int, t0: float) -> dict:
        write_audit(
            self.ctx,
            "extract_data",
            input_obj.model_dump(),
            {"point_count": len(points)},
            latency_ms=latency_ms,
            status="ok",
        )
        return {
            "ok": True,
            "result": {
                "data": points,
                "count": len(points),
            },
            "error": None,
            "tokens": len(str(points)) // 4,
        }
