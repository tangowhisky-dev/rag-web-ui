"""
Message export service.

Supports three formats:
  pdf   — reportlab PDF
  word  — python-docx .docx
  image — Pillow PNG screenshot of the answer text

All formats strip markdown syntax and render plain text.

ECharts charts are detected in answer text, rendered to PNG via pyecharts,
and embedded in the output (Word gets image + raw JSON, PDF/image get image only).
"""
from __future__ import annotations

import io
import json
import os
import re
import tempfile
import textwrap
from typing import Literal, Tuple

ExportFormat = Literal["pdf", "word", "image"]

# Regex to match echarts code blocks: ```echarts ... ```
_ECHARTS_BLOCK_RE = re.compile(r'```echarts\s*\n(.*?)\n```', re.DOTALL)


from app.services.infrastructure.reasoning_tags import strip_reasoning_tags

def _strip_markdown(text: str) -> str:
    """Very lightweight markdown stripper — enough for clean export."""
    text = strip_reasoning_tags(text)
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"`{1,3}(.*?)`{1,3}", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[citation:\d+\]", "", text)
    text = re.sub(r"\[citation\]\(\d+\)", "", text)
    return text.strip()


def _render_echarts_to_png(json_str: str) -> bytes | None:
    """Render an ECharts JSON config to PNG bytes using pyecharts.

    Returns None if rendering fails (malformed JSON, unsupported type, etc.).
    """
    try:
        config = json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return None

    try:
        from pyecharts import options as opts
        from pyecharts.charts import (
            Bar, Line, Pie, Scatter, Radar, Gauge, Funnel,
            HeatMap, WordCloud, Boxplot, Candlestick, Parallel,
            Graph, Tree, Sankey, Chord, Liquid, Calendar,
            Polar, PictorialBar, ThemeRiver,
            Bar3D, Line3D, Scatter3D, Surface3D, Lines3D,
            Grid,
        )

        # Determine chart type from first series
        series_list = config.get("series", [])
        if not isinstance(series_list, list) or not series_list:
            return None

        chart_type = series_list[0].get("type", "bar")

        # Build a pyecharts chart instance based on type
        chart_map = {
            "bar": Bar,
            "line": Line,
            "pie": Pie,
            "scatter": Scatter,
            "radar": Radar,
            "gauge": Gauge,
            "funnel": Funnel,
            "heatmap": HeatMap,
            "wordcloud": WordCloud,
            "boxplot": Boxplot,
            "candlestick": Candlestick,
            "parallel": Parallel,
            "effectScatter": Scatter,
            "graph": Graph,
            "tree": Tree,
            "sankey": Sankey,
            "chord": Chord,
            "liquidFill": Liquid,
            "calendar": HeatMap,  # Calendar uses HeatMap internally
            "polar": Bar,  # Polar uses bar/line with polar coordinate system
            "pictorialBar": PictorialBar,
            "themeRiver": ThemeRiver,
            "bar3D": Bar3D,
            "line3D": Line3D,
            "scatter3D": Scatter3D,
            "surface": Surface3D,
            "lines3D": Lines3D,
            "grid": Grid,
            "treemap": Tree,  # Treemap uses Tree internally
            "sunburst": Tree,  # Sunburst uses Tree internally
            "custom": Bar,  # Custom chart fallback to bar
        }

        ChartClass = chart_map.get(chart_type)
        if ChartClass is None:
            return None

        chart = ChartClass()

        # Extract data based on chart type
        _apply_config_to_chart(chart, config)

        # Render to PNG bytes
        import base64
        png_bytes = chart.render_embed()
        # render_embed returns HTML with embedded SVG/PNG — extract the image
        # For most chart types, pyecharts renders as SVG in render_embed.
        # We need actual PNG. Use render to a temp file instead.
        return _chart_to_png(chart)

    except Exception:
        return None


def _chart_to_png(chart) -> bytes | None:
    """Render a pyecharts chart instance to PNG bytes."""
    try:
        import tempfile
        import subprocess

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            chart.render(f.name)
            html_path = f.name

        png_path = html_path.rstrip(".html") + ".png"
        try:
            # Try using pyecharts built-in screenshot if available
            try:
                from pyecharts.render.snapshot import make_snapshot
                make_snapshot(chart=chart, file_name=png_path)
            except ImportError:
                # Older pyecharts version — try alternate import
                from pyecharts.render import snapshot
                snapshot.make_snapshot(chart=chart, file_name=png_path)
        except Exception:
            # Fallback: render to HTML, then we'll need an external tool
            # For now, return None and let the caller handle gracefully
            os.unlink(html_path)
            return None

        os.unlink(html_path)

        if os.path.exists(png_path):
            with open(png_path, "rb") as f:
                data = f.read()
            os.unlink(png_path)
            return data
        else:
            if os.path.exists(png_path):
                os.unlink(png_path)
            return None
    except Exception:
        return None


def _apply_config_to_chart(chart, config: dict):
    """Apply ECharts config dict to a pyecharts chart instance."""
    from pyecharts import options as opts

    # Apply title
    title_cfg = config.get("title", {})
    if title_cfg:
        text = title_cfg.get("text", "")
        if text:
            chart.set_global_opts(title_opts=opts.TitleOpts(title=text))

    # Apply tooltip
    tooltip_cfg = config.get("tooltip", {})
    if tooltip_cfg:
        trigger = tooltip_cfg.get("trigger", "axis")
        if trigger:
            # Get existing global opts and update tooltip
            current_opts = getattr(chart, "_options", {})
            # pyecharts handles tooltip via set_global_opts
            pass  # tooltip is set via set_global_opts below

    # Extract series data
    series_list = config.get("series", [])
    if not isinstance(series_list, list):
        return

    # Get xAxis data for categorical charts
    xaxis_cfg = config.get("xAxis", {})
    x_data = []
    if isinstance(xaxis_cfg, dict):
        x_data = xaxis_cfg.get("data", [])

    # Get y-axis config
    yaxis_cfg = config.get("yAxis", {})
    y_type = "value" if isinstance(yaxis_cfg, dict) else "value"

    # Process each series
    for i, series in enumerate(series_list):
        if not isinstance(series, dict):
            continue

        name = series.get("name", f"Series {i+1}")
        s_type = series.get("type", "bar")
        data = series.get("data", [])

        if s_type in ("bar", "line", "effectScatter"):
            if x_data and data and isinstance(data[0], (int, float)):
                # Simple categorical x-axis with numeric y values
                chart.add(
                    series_name=name,
                    x_axis=list(x_data),
                    y_axis=list(data),
                    is_symbol_show=False,
                )
            elif x_data and data and isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                # Data with name/value pairs
                chart.add(
                    series_name=name,
                    x_axis=list(x_data),
                    y_axis=[d.get("value", d.get("y", 0)) for d in data],
                    is_symbol_show=False,
                )
            else:
                chart.add(
                    series_name=name,
                    x_axis=list(x_data),
                    y_axis=list(data),
                    is_symbol_show=False,
                )
        elif s_type == "pie":
            if data and isinstance(data[0], dict):
                pie_data = [opts.DataItem(name=d["name"], value=d["value"]) for d in data]
            elif data:
                pie_data = [opts.DataItem(name=str(d), value=d) for d in data]
            else:
                pie_data = []
            chart.add(
                series_name=name,
                data_label_opts=opts.LabelOpts(show=True),
                radius=["40%", "75%"],
                data=pie_data,
            )
        elif s_type == "scatter":
            if data and isinstance(data[0], list):
                scatter_data = [tuple(d) for d in data]
            else:
                scatter_data = list(data)
            chart.add(
                series_name=name,
                xaxis_data=list(x_data) if x_data else None,
                series_data=scatter_data,
            )
        elif s_type == "radar":
            # Radar needs indicator range from config
            indicator = config.get("radar", {}).get("indicator", [])
            radar_indicators = [
                opts.RadarIndicatorItem(name=item.get("name", ""), max_=item.get("max", 100))
                for item in indicator
            ]
            if radar_indicators:
                chart.add_schema(
                    schema=radar_indicators,
                    start_angle=90,
                )
            chart.add(
                series_name=name,
                data=list(data) if data else [],
            )
        elif s_type == "gauge":
            if data:
                gauge_data = [opts.GaugeDataItem(name=d.get("name", ""), value=d.get("value", 0)) for d in data]
            else:
                gauge_data = []
            chart.add(
                series_name="",
                data=gauge_data,
                detail_opts=opts.LabelOpts(formatter="{value}%"),
            )
        elif s_type == "funnel":
            if data and isinstance(data[0], dict):
                funnel_data = [opts.DataItem(name=d["name"], value=d["value"]) for d in data]
            elif data:
                funnel_data = [opts.DataItem(name=str(d[0]) if isinstance(d, (list, tuple)) else str(d), value=d[1] if isinstance(d, (list, tuple)) and len(d) > 1 else d) for d in data]
            else:
                funnel_data = []
            chart.add(
                series_name=name,
                data=funnel_data,
                sort_="descending",
            )
        elif s_type == "heatmap":
            chart.add(
                series_name=name,
                data=list(data) if data else [],
                xaxis_data=list(x_data) if x_data else None,
            )
        elif s_type == "wordcloud":
            if data and isinstance(data[0], dict):
                word_data = [(d["name"], d["value"]) for d in data]
            elif data:
                word_data = [(str(d), 1) for d in data]
            else:
                word_data = []
            chart.add(
                series_name=name,
                data_range=[0, max(v for _, v in word_data) if word_data else 1],
                word_size_range=["12", "60"],
                data=word_data,
            )
        elif s_type == "boxplot":
            chart.add(
                series_name=name,
                x_axis=list(x_data) if x_data else [],
                y_axis=list(data) if data else [],
            )
        elif s_type == "candlestick":
            chart.add(
                series_name=name,
                x_axis=list(x_data) if x_data else [],
                y_axis=list(data) if data else [],
            )
        elif s_type == "parallel":
            chart.add(
                series_name=name,
                data=list(data) if data else [],
            )
        elif s_type == "graph":
            # Graph has nodes and links in series
            nodes = series.get("data", [])
            links = config.get("links", [])
            if nodes:
                node_data = [opts.GraphNode(name=n.get("name", f"Node {i}")) for i, n in enumerate(nodes)]
            else:
                node_data = []
            if links:
                link_data = [opts.GraphLink(source=l.get("source", ""), target=l.get("target", "")) for l in links]
            else:
                link_data = []
            chart.add(
                series_name=name,
                nodes=node_data,
                links=link_data,
                layout=series.get("layout", "force"),
                is_roam=True,
                is_label_show=True,
            )
        elif s_type == "tree":
            # Tree has hierarchical data
            tree_data = series.get("data", [])
            if tree_data:
                chart.add(
                    series_name=name,
                    data=tree_data,
                    layout=series.get("layout", "orthogonal"),
                    orient=series.get("orient", "vertical"),
                    symbol_size="auto",
                    leaves={},
                )
            else:
                chart.add(series_name=name, data=[], layout="orthogonal")
        elif s_type == "sankey":
            # Sankey has nodes and links
            nodes = series.get("data", [])
            links = series.get("links", [])
            if nodes:
                node_data = [opts.SankeyNode(name=n.get("name", f"Node {i}")) for i, n in enumerate(nodes)]
            else:
                node_data = []
            if links:
                link_data = [
                    opts.SankeyLink(source=l.get("source", ""), target=l.get("target", ""), value=l.get("value", 0))
                    for l in links
                ]
            else:
                link_data = []
            chart.add(
                series_name=name,
                nodes=node_data,
                links=link_data,
                layout_iterations=16,
                orient="horizontal",
                level_distance="auto",
            )
        elif s_type == "chord":
            # Chord has data and links
            data = series.get("data", [])
            links = series.get("links", [])
            if data:
                chord_data = [opts.ChordData(name=n.get("name", f"Node {i}")) for i, n in enumerate(data)]
            else:
                chord_data = []
            if links:
                link_data = [
                    opts.ChordLink(source=l.get("source", ""), target=l.get("target", ""), value=l.get("value", 0))
                    for l in links
                ]
            else:
                link_data = []
            chart.add(
                series_name=name,
                data=chord_data,
                links=link_data,
                tooltip_opts=opts.TooltipOpts(trigger="item"),
            )
        elif s_type == "liquidFill":
            # Liquid fill has data array with values
            if data:
                chart.add(
                    series_name=name,
                    data=data[:3] if len(data) > 3 else data,  # Max 3 waves
                    is_liquid_outline_show=True,
                    radius="70%",
                )
        elif s_type == "calendar":
            # Calendar uses heatmap with calendar coordinate system
            if data:
                chart.add(
                    series_name=name,
                    type_="heatmap",
                    calendar_index=0,
                    data=data,
                )
                chart.set_global_opts(
                    visualmap_opts=opts.VisualMapOpts(
                        min=data[0][1] if data and len(data[0]) > 1 else 0,
                        max=data[0][1] if data and len(data[0]) > 1 else 10,
                        orient="horizontal",
                        pos_bottom="10",
                        pos_left="center",
                    )
                )
        elif s_type == "polar":
            # Polar uses bar/line with polar coordinate system
            if x_data and data and isinstance(data[0], (int, float)):
                chart.add(
                    series_name=name,
                    x_axis=list(x_data),
                    y_axis=list(data),
                    coordinate_system="polar",
                    label_opts=opts.LabelOpts(is_show=True),
                )
        elif s_type == "pictorialBar":
            # Pictorial bar uses bar with symbol
            if x_data and data and isinstance(data[0], (int, float)):
                chart.add(
                    series_name=name,
                    x_axis=list(x_data),
                    y_axis=list(data),
                    symbol_size=["auto", "auto"],
                    symbol_margin="25%",
                    label_opts=opts.LabelOpts(is_show=True),
                )
        elif s_type == "themeRiver":
            # Theme river has data with [date, value, name]
            if data:
                theme_data = []
                for item in data:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        theme_data.append((item[0], item[1], item[2] if len(item) > 2 else ""))
                    elif isinstance(item, dict):
                        theme_data.append((item.get("date", item.get("time", "")), item.get("value", 0), item.get("name", "")))
                chart.add(
                    series_name=name,
                    data=theme_data,
                    singleaxis_opts=opts.SingleAxisOpts(type_="time", pos_top="50", pos_bottom="50"),
                )
        elif s_type in ("bar3D", "line3D", "scatter3D"):
            # 3D charts use 3D coordinate system
            if data:
                chart.add(
                    series_name=name,
                    data=data,
                    shading="lambert",
                    label_opts=opts.LabelOpts(is_show=False),
                )
                chart.set_global_opts(
                    visualmap_opts=opts.VisualMapOpts(
                        max=data[0][2] if data and len(data[0]) > 2 else 100,
                        orient="horizontal",
                        pos_bottom="0",
                        pos_left="center",
                    )
                )
        elif s_type == "surface":
            # Surface3D
            if data:
                chart.add(
                    series_name=name,
                    data=data,
                    shading="color",
                    label_opts=opts.LabelOpts(is_show=False),
                )
        elif s_type == "lines3D":
            # Lines3D
            if data:
                chart.add(
                    series_name=name,
                    data=data,
                    lines3D_opts=opts.Lines3DOpts(
                        effect_opts=opts.EffectOpts(
                            period=4,
                            symbol_size=6,
                        )
                    ),
                    label_opts=opts.LabelOpts(is_show=False),
                )
        elif s_type == "grid":
            # Grid combines multiple charts
            for j, grid_series in enumerate(series_list):
                if not isinstance(grid_series, dict):
                    continue
                grid_name = grid_series.get("name", f"Series {j+1}")
                grid_type = grid_series.get("type", "bar")
                grid_data = grid_series.get("data", [])
                grid_xaxis_idx = grid_series.get("xAxisIndex", j % 2)
                grid_yaxis_idx = grid_series.get("yAxisIndex", j % 2)
                if grid_type in ("bar", "line"):
                    grid_x_data = []
                    if grid_xaxis_idx < len(series_list):
                        other_series = series_list[grid_xaxis_idx]
                        if isinstance(other_series, dict):
                            other_xaxis = config.get("xAxis", {})
                            if isinstance(other_xaxis, dict):
                                grid_x_data = other_xaxis.get("data", [])
                    if grid_x_data and grid_data and isinstance(grid_data[0], (int, float)):
                        if grid_type == "bar":
                            chart.add(
                                series_name=grid_name,
                                x_axis=grid_x_data,
                                y_axis=grid_data,
                                xaxis_index=grid_xaxis_idx,
                                yaxis_index=grid_yaxis_idx,
                            )
                        else:
                            chart.add(
                                series_name=grid_name,
                                x_axis=grid_x_data,
                                y_axis=grid_data,
                                xaxis_index=grid_xaxis_idx,
                                yaxis_index=grid_yaxis_idx,
                                is_smooth=True,
                            )
        elif s_type in ("treemap", "sunburst"):
            # Treemap and sunburst use Tree with hierarchical data
            if data:
                chart.add(
                    series_name=name,
                    data=data,
                    leaf_depth=2,
                    levels=[
                        {},
                        {"itemstyle": {"color": "#5470c6", "color0": "#91cc75"}},
                        {"itemstyle": {"color": "#fac858", "color0": "#ee6666"}},
                    ],
                )
        elif s_type == "custom":
            # Custom chart fallback to bar
            if x_data and data and isinstance(data[0], (int, float)):
                chart.add(
                    series_name=name,
                    x_axis=list(x_data),
                    y_axis=list(data),
                )


def process_echarts_blocks(text: str) -> Tuple[str, list[bytes], list[str]]:
    """Find echarts code blocks in text and render them to PNG.

    Returns:
        (processed_text, png_bytes_list, raw_json_list)
        - processed_text: original text with echarts blocks replaced by placeholders
        - png_bytes_list: list of PNG bytes for each chart
        - raw_json_list: list of raw JSON strings from each block
    """
    png_list: list[bytes] = []
    raw_json_list: list[str] = []
    placeholders: list[str] = []

    def _replace_block(match: re.Match) -> str:
        json_str = match.group(1).strip()
        raw_json_list.append(json_str)
        placeholder = f"\n[ECHARTS_CHART_{len(png_list)}]\n"
        placeholders.append(placeholder)

        png = _render_echarts_to_png(json_str)
        if png:
            png_list.append(png)
        return placeholder

    processed = _ECHARTS_BLOCK_RE.sub(_replace_block, text)
    return processed, png_list, raw_json_list


def _insert_png_in_docx(doc, png_bytes: bytes, index: int):
    """Insert a PNG image into a docx document at the current position."""
    from docx.oxml.ns import qn
    from docx.shared import Inches

    buf = io.BytesIO(png_bytes)
    doc.add_picture(buf, width=Inches(5.5))
    # Add a blank paragraph after the image for spacing
    doc.add_paragraph()


def export_to_pdf(answer_text: str) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
    from reportlab.lib.units import cm

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
    )
    styles = getSampleStyleSheet()
    body_style = styles["BodyText"]
    body_style.fontSize = 11
    body_style.leading = 16

    clean = _strip_markdown(answer_text)
    clean, png_list, _ = process_echarts_blocks(clean)

    elements = []
    for para in clean.split("\n\n"):
        para = para.strip()
        if not para:
            continue

        # Check if this paragraph is a chart placeholder
        chart_match = re.match(r'\[ECHARTS_CHART_(\d+)\]', para)
        if chart_match:
            idx = int(chart_match.group(1))
            if idx < len(png_list):
                img_buf = io.BytesIO(png_list[idx])
                from reportlab.lib.utils import ImageReader
                img = ImageReader(img_buf)
                # Read image dimensions and scale to fit page width
                avail_width = 612 - doc.leftMargin - doc.rightMargin  # A4 width in points
                orig_w, orig_h = img.getSize()
                scale = min(avail_width / orig_w, 400 / orig_h) if orig_w > 0 else 0.8
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                elements.append(Image(img, width=new_w, height=new_h))
                elements.append(Spacer(1, 0.3 * cm))
            continue

        elements.append(Paragraph(para.replace("\n", "<br/>"), body_style))
        elements.append(Spacer(1, 0.3 * cm))

    doc.build(elements)
    return buf.getvalue()


def export_to_word(answer_text: str) -> bytes:
    from docx import Document
    from docx.shared import Pt, Cm, Inches

    doc = Document()
    # Margins
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    clean = _strip_markdown(answer_text)
    clean, png_list, raw_json_list = process_echarts_blocks(clean)

    for para in clean.split("\n\n"):
        para = para.strip()
        if not para:
            continue

        # Check if this paragraph is a chart placeholder
        chart_match = re.match(r'\[ECHARTS_CHART_(\d+)\]', para)
        if chart_match:
            idx = int(chart_match.group(1))
            # Insert the chart image
            if idx < len(png_list):
                _insert_png_in_docx(doc, png_list[idx], idx)
            # Also include the raw JSON for reference
            if idx < len(raw_json_list):
                json_para = doc.add_paragraph()
                run = json_para.add_run(raw_json_list[idx])
                run.font.size = Pt(8)
                run.font.name = 'Courier New'
            continue

        p = doc.add_paragraph(para)
        p.style.font.size = Pt(11)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def export_to_image(answer_text: str) -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    clean = _strip_markdown(answer_text)
    clean, png_list, _ = process_echarts_blocks(clean)

    width = 900
    padding = 40
    font_size = 16
    line_height = font_size + 6

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    # Build lines, inserting chart images where placeholders appear
    all_lines: list[str | tuple['Image.Image', int]] = []  # str or (PIL Image, height)
    wrap_width = (width - 2 * padding) // (font_size // 2 + 2)

    for para in clean.split("\n"):
        para = para.strip()
        if not para:
            all_lines.append("")
            continue

        # Check if this is a chart placeholder
        chart_match = re.match(r'\[ECHARTS_CHART_(\d+)\]', para)
        if chart_match:
            idx = int(chart_match.group(1))
            if idx < len(png_list):
                chart_img = Image.open(io.BytesIO(png_list[idx]))
                # Scale to fit width
                max_h = 400
                ratio = min(width / chart_img.width, max_h / chart_img.height)
                new_w = int(chart_img.width * ratio)
                new_h = int(chart_img.height * ratio)
                chart_img = chart_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                all_lines.append(("chart", chart_img, new_h))
            continue

        # Regular text — word wrap
        wrapped = textwrap.wrap(para, width=wrap_width) if para else [""]
        all_lines.extend(wrapped or [""])

    # Calculate total height
    chart_heights = [item[2] for item in all_lines if isinstance(item, tuple)]
    text_count = sum(1 for item in all_lines if isinstance(item, str))
    height = padding * 2 + text_count * line_height + sum(chart_heights) + len(chart_heights) * 20

    img = Image.new("RGB", (width, max(height, 200)), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    y = padding
    for item in all_lines:
        if isinstance(item, str):
            if item:
                draw.text((padding, y), item, fill=(30, 30, 30), font=font)
            y += line_height
        elif isinstance(item, tuple) and item[0] == "chart":
            _, chart_img, chart_h = item
            img.paste(chart_img, (padding, y))
            y += chart_h + 20

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def export_message(answer_text: str, fmt: ExportFormat) -> tuple[bytes, str, str]:
    """
    Returns (content_bytes, media_type, filename).
    """
    if fmt == "pdf":
        data = export_to_pdf(answer_text)
        return data, "application/pdf", "answer.pdf"
    elif fmt == "word":
        data = export_to_word(answer_text)
        return data, "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "answer.docx"
    elif fmt == "image":
        data = export_to_image(answer_text)
        return data, "image/png", "answer.png"
    else:
        raise ValueError(f"Unknown export format: {fmt}")


# ── Synthesis report generation ────────────────────────────────────────────────

def generate_synthesis_report(
    answer: str,
    tool_trace: list,
    query: str,
    kb_ids: list,
) -> str:
    """
    Generate a structured Markdown synthesis report from LLM answer + tool trace.

    Extracts source documents from search/synthesize tool calls in the trace,
    deduplicates them, and appends a ## Sources section.

    Args:
        answer:     Final LLM-generated answer text (may already contain ## sections).
        tool_trace: List of ToolResult dicts from the tool calling loop.
        query:      Original user query.
        kb_ids:     Knowledge base IDs queried.

    Returns:
        Complete Markdown string with header, answer, and sources.
    """
    import datetime

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Extract source file names from tool trace
    sources: dict = {}  # filename -> chunk count
    for entry in tool_trace:
        output = entry.get("output") or {}
        # synthesize_documents returns {"chunks": [...]}
        # search_documents returns a list directly
        chunks = []
        if isinstance(output, dict):
            chunks = output.get("chunks", [])
        elif isinstance(output, list):
            chunks = output

        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            src = chunk.get("source") or "unknown"
            sources[src] = sources.get(src, 0) + 1

    # Build header
    lines = [
        f"# Synthesis Report",
        f"",
        f"**Query:** {query}",
        f"**Knowledge Bases:** {', '.join(str(k) for k in kb_ids)}",
        f"**Generated:** {timestamp}",
        f"",
        f"---",
        f"",
        answer.strip(),
    ]

    # Append sources section if any were found
    if sources:
        lines += [
            "",
            "---",
            "",
            "## Sources",
            "",
        ]
        for src, count in sorted(sources.items(), key=lambda x: -x[1]):
            lines.append(f"- **{src}** ({count} chunk{'s' if count != 1 else ''} cited)")

    return "\n".join(lines)
