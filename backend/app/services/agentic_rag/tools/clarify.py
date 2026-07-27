"""clarify tool — no-op marker; clarification is handled by the graph interrupt."""

from __future__ import annotations

from pydantic import BaseModel

from app.services.agentic_rag.tools.base import BaseAgentTool


class ClarifyInput(BaseModel):
    question: str = ""


class ClarifyTool(BaseAgentTool):
    name: str = "clarify"
    description: str = "Ask the user a clarification question. Handled by the agent graph interrupt."
    args_schema: type[BaseModel] = ClarifyInput

    async def _execute(self, input_obj: ClarifyInput) -> dict:
        return {
            "ok": False,
            "result": {},
            "error": "clarify is handled by the agent graph, not a tool call.",
            "tokens": 0,
        }
