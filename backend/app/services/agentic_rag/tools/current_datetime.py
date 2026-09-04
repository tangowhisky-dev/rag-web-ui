"""current_datetime tool — gives the LLM awareness of the current date/time.

Lets the agent reason temporally: "latest", "most recent", "this week", "last month"
all depend on knowing what "now" is. Without this, the LLM has no reliable way
to distinguish a document titled "Weekly Update 1-7 Aug 26" from "Weekly Update
21-28 Aug 26" — both look like dates, but only one is closest to now.

Call this before deciding which document is "latest" when titles or content
contain dates.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel

from app.services.agentic_rag.tool_context import ToolContext, write_audit
from app.services.agentic_rag.tools.base import BaseAgentTool


class CurrentDatetimeInput(BaseModel):
    pass


class CurrentDatetimeTool(BaseAgentTool):
    name: str = "current_datetime"
    ui_label: str = "Checking current date and time"
    description: str = (
        "Returns the current UTC date and time. Call this before deciding which "
        "document is 'latest', 'most recent', or 'newest' — you need to know what "
        "'now' is to compare dates in document titles and content. "
        "No arguments needed."
    )
    args_schema: type[BaseModel] = CurrentDatetimeInput
    ctx: ToolContext | None = None  # type: ignore[assignment]

    async def _execute(self, input_obj: CurrentDatetimeInput) -> dict:
        now = datetime.now(timezone.utc)
        result = {
            "current_datetime_utc": now.isoformat(),
            "current_date": now.strftime("%Y-%m-%d"),
            "current_time": now.strftime("%H:%M:%S UTC"),
            "day_of_week": now.strftime("%A"),
            "week_of_year": now.isocalendar().week,
            "year": now.year,
            "month": now.month,
        }
        if self.ctx is not None:
            write_audit(self.ctx, "current_datetime", {}, result, tokens_in=0, tokens_out=0, status="ok", latency_ms=0)
        return {"ok": True, "result": result, "error": None, "tokens": 0}
