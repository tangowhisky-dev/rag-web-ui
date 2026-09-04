"""chart_generate tool — build a deterministic ECharts option from data."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.services.agentic_rag.tool_context import ToolContext, write_audit
from app.services.agentic_rag.tools.base import BaseAgentTool

logger = logging.getLogger(__name__)


def _normalize_data(data: list[dict]) -> list[dict]:
    normalized = []
    for d in data:
        name = d.get("label") or d.get("name") or d.get("category") or str(d)
        value = d.get("value")
        if value is None:
            # try numeric columns
            for k, v in d.items():
                if isinstance(v, (int, float)):
                    value = v
                    if name == str(d):
                        name = str(d.get(k.replace("_", " ").title()))
                    break
        try:
            value = float(value) if not isinstance(value, (int, float)) else value
        except Exception:
            continue
        normalized.append({"name": name, "value": value})
    return normalized


def _build_pie(title: str, normalized: list[dict], **_: Any) -> dict:
    return {
        "title": {"text": title, "left": "center"},
        "tooltip": {"trigger": "item"},
        "series": [{
            "type": "pie",
            "radius": "60%",
            "data": normalized,
            "emphasis": {"itemStyle": {"shadowBlur": 10, "shadowOffsetX": 0, "shadowColor": "rgba(0,0,0,0.5)"}},
        }],
    }


def _build_bar(title: str, names: list, values: list, input_obj: ChartGenerateInput, **_: Any) -> dict:
    return {
        "title": {"text": title},
        "tooltip": {"trigger": "axis"},
        "xAxis": {"type": "category", "data": names, "name": input_obj.x_label or ""},
        "yAxis": {"type": "value", "name": input_obj.y_label or ""},
        "series": [{"type": "bar", "data": values}],
    }


def _build_line(title: str, names: list, values: list, input_obj: ChartGenerateInput, chart_type: str, **_: Any) -> dict:
    series = {"type": "line", "data": values}
    if chart_type == "area":
        series["areaStyle"] = {}
    return {
        "title": {"text": title},
        "tooltip": {"trigger": "axis"},
        "xAxis": {"type": "category", "data": names, "name": input_obj.x_label or ""},
        "yAxis": {"type": "value", "name": input_obj.y_label or ""},
        "series": [series],
    }


def _build_scatter(title: str, values: list, chart_type: str, **_: Any) -> dict:
    series_type = "effectScatter" if chart_type == "effectscatter" else "scatter"
    scatter = [[i, v] for i, v in enumerate(values)]
    return {
        "title": {"text": title},
        "xAxis": {"type": "value"},
        "yAxis": {"type": "value"},
        "series": [{"type": series_type, "data": scatter}],
    }


def _build_radar(title: str, names: list, values: list, **_: Any) -> dict:
    max_value = max(values) * 1.2 if values else 100
    return {
        "title": {"text": title},
        "tooltip": {"trigger": "item"},
        "radar": {"indicator": [{"name": n, "max": max_value} for n in names]},
        "series": [{"type": "radar", "data": [{"value": values, "name": title}]}],
    }


def _build_gauge(title: str, names: list, values: list, **_: Any) -> dict:
    # Gauge shows a single value; use the first data point.
    return {
        "title": {"text": title},
        "series": [{
            "type": "gauge",
            "data": [{"name": names[0], "value": values[0]}],
            "detail": {"formatter": "{value}"},
        }],
    }


def _build_funnel(title: str, normalized: list[dict], **_: Any) -> dict:
    return {
        "title": {"text": title, "left": "center"},
        "tooltip": {"trigger": "item"},
        "series": [{"type": "funnel", "data": normalized, "sort_": "descending"}],
    }


_CHART_BUILDERS = {
    "pie": _build_pie,
    "bar": _build_bar,
    "line": _build_line,
    "area": _build_line,
    "scatter": _build_scatter,
    "effectscatter": _build_scatter,
    "radar": _build_radar,
    "gauge": _build_gauge,
    "funnel": _build_funnel,
}


class ChartGenerateInput(BaseModel):
    chart_type: str = Field(default="bar", description="pie, bar, line, scatter, area, effectScatter, radar, gauge, funnel")
    data: list[dict] = Field(
        default_factory=list,
        description="List of {label, value} or {name, value}. If empty, reads from "
        "accumulated_data in state (populated by prior extract_data calls).",
    )
    title: Optional[str] = Field(default=None)
    x_label: Optional[str] = Field(default=None)
    y_label: Optional[str] = Field(default=None)


class ChartGenerateTool(BaseAgentTool):
    name: str = "chart_generate"
    ui_label: str = "Generating chart"
    description: str = (
        "Generate an ECharts option JSON from structured data. "
        "Use after extract_data to create pie/bar/line/scatter/radar/gauge/funnel charts. "
        "If data is empty, automatically reads from accumulated_data (populated by "
        "prior extract_data calls with source='retrieved_docs')."
    )
    args_schema: type[BaseModel] = ChartGenerateInput

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use arun() for agent tools.")

    async def _execute(self, input_obj: ChartGenerateInput) -> dict:
        t0 = time.monotonic()
        ctx: ToolContext = self.ctx

        data = input_obj.data
        if not data and ctx.state:
            # Fall back to accumulated_data from prior extract_data calls.
            data = ctx.state.get("accumulated_data", []) or []

        if not data:
            return {"ok": False, "result": {}, "error": "No data provided and no accumulated_data in state. Call extract_data first.", "tokens": 0}

        normalized = _normalize_data(data)

        if not normalized:
            return {"ok": False, "result": {}, "error": "No numeric values found.", "tokens": 0}

        chart_type = input_obj.chart_type.lower()
        title = input_obj.title or "Chart"
        names = [d["name"] for d in normalized]
        values = [d["value"] for d in normalized]

        builder = _CHART_BUILDERS.get(chart_type)
        if builder is None:
            return {"ok": False, "result": {}, "error": f"Unsupported chart type: {chart_type}", "tokens": 0}

        option = builder(title=title, names=names, values=values, normalized=normalized, input_obj=input_obj, chart_type=chart_type)

        latency_ms = round((time.monotonic() - t0) * 1000)
        write_audit(ctx, "chart_generate", input_obj.model_dump(), {"chart_type": chart_type, "series_count": len(normalized)}, latency_ms=latency_ms, status="ok")

        return {
            "ok": True,
            "result": {
                "chart_option": option,
                "valid": True,
                "chart_type": chart_type,
            },
            "error": None,
            "tokens": len(str(option)) // 4,
        }
