"""ECharts option builders used by chart_generate."""

from __future__ import annotations

from typing import Any


def _axis_data(data: list[dict]) -> tuple[list[str], list[Any]]:
    labels = [str(d.get("label", d.get("x", ""))) for d in data]
    values = [d.get("value", d.get("y", 0)) for d in data]
    return labels, values


def build_bar_option(title: str, data: list[dict], x_label: str | None, y_label: str | None) -> dict:
    labels, values = _axis_data(data)
    return {
        "title": {"text": title, "left": "center"},
        "tooltip": {"trigger": "axis"},
        "xAxis": {"type": "category", "data": labels, "name": x_label or "", "nameLocation": "middle", "nameGap": 30},
        "yAxis": {"type": "value", "name": y_label or ""},
        "series": [{"type": "bar", "data": values}],
        "grid": {"left": "10%", "right": "10%", "bottom": "15%", "containLabel": True},
    }


def build_line_option(title: str, data: list[dict], x_label: str | None, y_label: str | None) -> dict:
    labels, values = _axis_data(data)
    return {
        "title": {"text": title, "left": "center"},
        "tooltip": {"trigger": "axis"},
        "xAxis": {"type": "category", "data": labels, "name": x_label or "", "nameLocation": "middle", "nameGap": 30},
        "yAxis": {"type": "value", "name": y_label or ""},
        "series": [{"type": "line", "data": values, "smooth": True}],
        "grid": {"left": "10%", "right": "10%", "bottom": "15%", "containLabel": True},
    }


def build_pie_option(title: str, data: list[dict]) -> dict:
    values = [{"name": str(d.get("label", d.get("name", ""))), "value": d.get("value", 0)} for d in data]
    return {
        "title": {"text": title, "left": "center"},
        "tooltip": {"trigger": "item"},
        "series": [
            {
                "type": "pie",
                "radius": ["40%", "70%"],
                "data": values,
                "emphasis": {
                    "itemStyle": {"shadowBlur": 10, "shadowOffsetX": 0, "shadowColor": "rgba(0,0,0,0.5)"}
                },
            }
        ],
        "grid": {"left": "10%", "right": "10%", "bottom": "15%", "containLabel": True},
    }


def build_scatter_option(title: str, data: list[dict], x_label: str | None, y_label: str | None) -> dict:
    values = [[d.get("x", d.get("label", 0)), d.get("y", d.get("value", 0))] for d in data]
    return {
        "title": {"text": title, "left": "center"},
        "tooltip": {"trigger": "item"},
        "xAxis": {"type": "value", "name": x_label or ""},
        "yAxis": {"type": "value", "name": y_label or ""},
        "series": [{"type": "scatter", "data": values}],
        "grid": {"left": "10%", "right": "10%", "bottom": "15%", "containLabel": True},
    }


BUILDERS = {
    "bar": build_bar_option,
    "line": build_line_option,
    "pie": build_pie_option,
    "scatter": build_scatter_option,
}
