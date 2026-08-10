"""chart_generate tool — build a deterministic ECharts option from data."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.services.agentic_rag.tool_context import ToolContext, write_audit
from app.services.agentic_rag.tools.base import BaseAgentTool

logger = logging.getLogger(__name__)


class ChartGenerateInput(BaseModel):
    chart_type: str = Field(default="bar", description="pie, bar, line, scatter, area, effectScatter, radar, gauge, funnel")
    data: list[dict] = Field(default_factory=list, description="List of {label, value} or {name, value}.")
    title: Optional[str] = Field(default=None)
    x_label: Optional[str] = Field(default=None)
    y_label: Optional[str] = Field(default=None)


class ChartGenerateTool(BaseAgentTool):
    name: str = "chart_generate"
    ui_label: str = "Generating chart"
    description: str = (
        "Generate an ECharts option JSON from structured data. "
        "Use after extract_data to create pie/bar/line/scatter/radar/gauge/funnel charts."
    )
    args_schema: type[BaseModel] = ChartGenerateInput

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use arun() for agent tools.")

    async def _execute(self, input_obj: ChartGenerateInput) -> dict:
        t0 = time.monotonic()
        ctx: ToolContext = self.ctx

        if not input_obj.data:
            return {"ok": False, "result": {}, "error": "No data provided.", "tokens": 0}

        # Normalize data to {name, value}
        normalized = []
        for d in input_obj.data:
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

        if not normalized:
            return {"ok": False, "result": {}, "error": "No numeric values found.", "tokens": 0}

        chart_type = input_obj.chart_type.lower()
        title = input_obj.title or "Chart"
        names = [d["name"] for d in normalized]
        values = [d["value"] for d in normalized]

        option: dict[str, Any]
        if chart_type == "pie":
            option = {
                "title": {"text": title, "left": "center"},
                "tooltip": {"trigger": "item"},
                "series": [{
                    "type": "pie",
                    "radius": "60%",
                    "data": normalized,
                    "emphasis": {"itemStyle": {"shadowBlur": 10, "shadowOffsetX": 0, "shadowColor": "rgba(0,0,0,0.5)"}},
                }],
            }
        elif chart_type == "bar":
            option = {
                "title": {"text": title},
                "tooltip": {"trigger": "axis"},
                "xAxis": {"type": "category", "data": names, "name": input_obj.x_label or ""},
                "yAxis": {"type": "value", "name": input_obj.y_label or ""},
                "series": [{"type": "bar", "data": values}],
            }
        elif chart_type in ("line", "area"):
            series = {"type": "line", "data": values}
            if chart_type == "area":
                series["areaStyle"] = {}
            option = {
                "title": {"text": title},
                "tooltip": {"trigger": "axis"},
                "xAxis": {"type": "category", "data": names, "name": input_obj.x_label or ""},
                "yAxis": {"type": "value", "name": input_obj.y_label or ""},
                "series": [series],
            }
        elif chart_type in ("scatter", "effectscatter"):
            series_type = "effectScatter" if chart_type == "effectscatter" else "scatter"
            scatter = [[i, v] for i, v in enumerate(values)]
            option = {
                "title": {"text": title},
                "xAxis": {"type": "value"},
                "yAxis": {"type": "value"},
                "series": [{"type": series_type, "data": scatter}],
            }
        elif chart_type == "radar":
            max_value = max(values) * 1.2 if values else 100
            option = {
                "title": {"text": title},
                "tooltip": {"trigger": "item"},
                "radar": {"indicator": [{"name": n, "max": max_value} for n in names]},
                "series": [{"type": "radar", "data": [{"value": values, "name": title}]}],
            }
        elif chart_type == "gauge":
            # Gauge shows a single value; use the first data point.
            option = {
                "title": {"text": title},
                "series": [{
                    "type": "gauge",
                    "data": [{"name": names[0], "value": values[0]}],
                    "detail": {"formatter": "{value}"},
                }],
            }
        elif chart_type == "funnel":
            option = {
                "title": {"text": title, "left": "center"},
                "tooltip": {"trigger": "item"},
                "series": [{"type": "funnel", "data": normalized, "sort_": "descending"}],
            }
        else:
            return {"ok": False, "result": {}, "error": f"Unsupported chart type: {chart_type}", "tokens": 0}

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
