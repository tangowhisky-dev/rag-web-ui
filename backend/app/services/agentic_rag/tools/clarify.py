"""clarify tool — human-in-the-loop clarification via LangGraph interrupt().

When the LLM calls this tool, the graph pauses and the user is asked a
question. The user's response is returned as a tool observation, and the
think node continues with the clarified information.

Uses LangGraph's interrupt()/resume mechanism:
  1. interrupt() raises GraphInterrupt → LangGraph checkpoints and pauses
  2. Runner detects __interrupt__ in stream → emits interrupt SSE event
  3. User responds via /clarification API → Command(resume=response)
  4. interrupt() returns the user's response → tool returns it as observation
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from langgraph.types import interrupt
from pydantic import BaseModel, Field

from app.services.agentic_rag.tool_context import ToolContext, write_audit
from app.services.agentic_rag.tools.base import BaseAgentTool

logger = logging.getLogger(__name__)


class ClarifyInput(BaseModel):
    question: str = Field(description=(
        "The question to ask the user. Be specific and concise. "
        "Example: 'Which document do you mean — the Q3 report or the Q4 report?'"
    ))
    options: Optional[list[str]] = Field(
        default=None,
        description="Optional list of suggested answers for the user to pick from.",
    )


class ClarifyTool(BaseAgentTool):
    name: str = "clarify"
    ui_label: str = "Asking for clarification"
    description: str = "Ask the user a question to resolve ambiguity in their query."
    prompt_snippet: str = "Ask the user to resolve query ambiguity"
    prompt_guidelines: list[str] = [
        "clarify: Use only when ambiguity materially changes the retrieval target or answer. If a reasonable interpretation can be searched or answered, do not clarify.",
        "clarify: Ask concise questions. Max 2 calls per turn. The tool pauses the pipeline and resumes on user response.",
    ]
    args_schema: type[BaseModel] = ClarifyInput

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use arun() for agent tools.")

    async def _execute(self, input_obj: ClarifyInput) -> dict:
        t0 = time.monotonic()
        ctx: ToolContext = self.ctx

        question = input_obj.question.strip()
        options = input_obj.options or []

        # interrupt() raises GraphInterrupt. Do NOT catch it — LangGraph
        # catches it at the graph level, checkpoints, and pauses. When
        # resumed via Command(resume=value), interrupt() returns the value.
        # The runner detects __interrupt__ in the stream and emits the
        # interrupt SSE event to the frontend.
        interrupt_payload = {"question": question}
        if options:
            interrupt_payload["options"] = options

        user_response = interrupt(interrupt_payload)

        response_text = str(user_response) if user_response else ""

        write_audit(ctx, "clarify", {"question": question, "response": response_text})

        return {
            "ok": True,
            "result": {
                "question": question,
                "user_response": response_text,
                "message": f"User responded: {response_text}",
            },
            "error": None,
            "tokens": 0,
            "terminate": False,
        }
