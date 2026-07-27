"""file_extract_table tool — extract tables from CSV/Excel/HTML files."""

from __future__ import annotations

import logging
import time
from io import StringIO
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


class FileExtractTableTool(BaseAgentTool):
    name: str = "file_extract_table"
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

        file_id = input_obj.file_id
        if file_id is None and ctx.chat_id:
            cf = (
                ctx.db.query(ChatFile)
                .filter(ChatFile.chat_id == ctx.chat_id)
                .order_by(ChatFile.id.desc())
                .first()
            )
            file_id = cf.id if cf else None

        if not file_id:
            return {"ok": False, "result": {}, "error": "No attached file found.", "tokens": 0}

        rbac = enforce_rbac(ctx, file_id=file_id)
        if rbac.get("file_id") is None:
            return {"ok": False, "result": {}, "error": "Access denied to file.", "tokens": 0}
        file_id = rbac["file_id"]

        cf = ctx.db.query(ChatFile).filter(ChatFile.id == file_id).first()
        if not cf:
            return {"ok": False, "result": {}, "error": "File not found.", "tokens": 0}

        try:
            import pandas as pd
        except Exception as exc:
            return {"ok": False, "result": {}, "error": f"pandas not available: {exc}", "tokens": 0}

        df = None
        try:
            if cf.content_type.endswith("csv") or cf.file_name.lower().endswith(".csv"):
                df = pd.read_csv(StringIO(cf.markdown_content or ""))
            elif cf.content_type.endswith(("xlsx", "xls")) or cf.file_name.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(cf.stored_path)
            elif "html" in cf.content_type or cf.file_name.lower().endswith(".html"):
                tables = pd.read_html(cf.markdown_content or "")
                df = tables[input_obj.table_index] if 0 <= input_obj.table_index < len(tables) else None
            else:
                # Try markdown/html table extraction
                tables = pd.read_html(cf.markdown_content or "")
                df = tables[input_obj.table_index] if 0 <= input_obj.table_index < len(tables) else None
        except Exception as exc:
            logger.warning("[file_extract_table] parse failed: %s", exc)
            return {"ok": False, "result": {}, "error": f"Could not extract table: {exc}", "tokens": 0}

        if df is None:
            return {"ok": False, "result": {}, "error": "No table found.", "tokens": 0}

        if input_obj.filter:
            try:
                df = df.query(input_obj.filter)
            except Exception as exc:
                logger.warning("[file_extract_table] filter failed: %s", exc)

        rows = df.head(1000).values.tolist()
        columns = df.columns.tolist()

        latency_ms = round((time.monotonic() - t0) * 1000)
        write_audit(ctx, "file_extract_table", input_obj.model_dump(), {"row_count": len(rows), "columns": columns}, latency_ms=latency_ms, status="ok")

        return {
            "ok": True,
            "result": {
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
                "file_name": cf.file_name,
            },
            "error": None,
            "tokens": len(str(rows)) // 4,
        }
