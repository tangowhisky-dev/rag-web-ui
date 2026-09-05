"""file_extract_table tool — extract tables from CSV/Excel/HTML files."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.models.chat import ChatFile
from app.services.agentic_rag.tool_context import ToolContext, enforce_rbac, write_audit
from app.services.agentic_rag.tools.base import BaseAgentTool

logger = logging.getLogger(__name__)


class FileExtractTableInput(BaseModel):
    file_id: Optional[int] = Field(default=None)
    table_index: int = Field(default=0)
    filter: Optional[str] = Field(default=None)
    accumulate: bool = Field(default=False, description="When true and the table has exactly 2 columns (label, value), append rows to state.accumulated_data for chart_generate or office_generate to consume.")


def _resolve_file(ctx: ToolContext, file_id: Optional[int]) -> tuple:
    if file_id is None and ctx.chat_id:
        cf = (
            ctx.db.query(ChatFile)
            .filter(ChatFile.chat_id == ctx.chat_id)
            .order_by(ChatFile.id.desc())
            .first()
        )
        file_id = cf.id if cf else None

    if not file_id:
        return None, {"ok": False, "result": {}, "error": "No attached file found.", "tokens": 0}

    rbac = enforce_rbac(ctx, file_id=file_id)
    if rbac.get("file_id") is None:
        return None, {"ok": False, "result": {}, "error": "Access denied to file.", "tokens": 0}
    file_id = rbac["file_id"]

    cf = ctx.db.query(ChatFile).filter(ChatFile.id == file_id).first()
    if not cf:
        return None, {"ok": False, "result": {}, "error": "File not found.", "tokens": 0}

    return cf, None


def _parse_table(cf: Any, table_index: int) -> tuple:
    try:
        import pandas as pd
    except Exception as exc:
        return None, {"ok": False, "result": {}, "error": f"pandas not available: {exc}", "tokens": 0}

    df = None
    try:
        if cf.content_type.endswith("csv") or cf.file_name.lower().endswith(".csv"):
            df = pd.read_csv(cf.stored_path)
        elif cf.content_type.endswith(("xlsx", "xls")) or cf.file_name.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(cf.stored_path)
        elif "html" in cf.content_type or cf.file_name.lower().endswith(".html"):
            tables = pd.read_html(cf.markdown_content or "")
            df = tables[table_index] if 0 <= table_index < len(tables) else None
        else:
            tables = pd.read_html(cf.markdown_content or "")
            df = tables[table_index] if 0 <= table_index < len(tables) else None
    except Exception as exc:
        logger.warning("[file_extract_table] parse failed: %s", exc)
        return None, {"ok": False, "result": {}, "error": f"Could not extract table: {exc}", "tokens": 0}

    if df is None:
        return None, {"ok": False, "result": {}, "error": "No table found.", "tokens": 0}

    return df, None


def _apply_filter(df: Any, filter_expr: Optional[str]) -> Any:
    if filter_expr:
        try:
            df = df.query(filter_expr)
        except Exception as exc:
            logger.warning("[file_extract_table] filter failed: %s", exc)
    return df


class FileExtractTableTool(BaseAgentTool):
    name: str = "file_extract_table"
    ui_label: str = "Extracting table from file"
    description: str = (
        "Extract a structured table from a CSV, Excel, or HTML table in an "
        "attached file. Returns JSON columns and rows for chart_generate or code_execute."
    )
    args_schema: type[BaseModel] = FileExtractTableInput

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use arun() for agent tools.")

    async def _execute(self, input_obj: FileExtractTableInput) -> dict:
        t0 = time.monotonic()
        ctx: ToolContext = self.ctx

        cf, err = _resolve_file(ctx, input_obj.file_id)
        if err:
            return err

        df, err = _parse_table(cf, input_obj.table_index)
        if err:
            return err

        df = _apply_filter(df, input_obj.filter)

        rows = df.head(1000).values.tolist()
        columns = df.columns.tolist()

        latency_ms = round((time.monotonic() - t0) * 1000)
        write_audit(ctx, "file_extract_table", input_obj.model_dump(), {"row_count": len(rows), "columns": columns}, latency_ms=latency_ms, status="ok")

        # When accumulate=true and the table is label/value shaped (2 columns),
        # convert rows to DataPoint format and append to state.accumulated_data
        # so chart_generate or office_generate can consume them without the
        # LLM having to re-transcribe the values.
        accumulated_count = 0
        if input_obj.accumulate and ctx.state is not None and len(columns) == 2:
            from app.services.agentic_rag.schemas import DataPoint
            points = []
            for row in rows:
                try:
                    points.append(DataPoint(
                        label=str(row[0]),
                        value=row[1],
                        unit=None,
                        context=f"from {cf.file_name}",
                    ))
                except Exception:
                    continue
            if points:
                existing = ctx.state.get("accumulated_data", []) or []
                ctx.state["accumulated_data"] = existing + [p.model_dump() for p in points]
                accumulated_count = len(points)

        return {
            "ok": True,
            "result": {
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
                "file_name": cf.file_name,
                "accumulated": accumulated_count,
            },
            "error": None,
            "tokens": len(str(rows)) // 4,
        }
