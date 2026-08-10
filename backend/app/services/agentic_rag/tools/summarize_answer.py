"""summarize_answer tool — summarize previous answers, messages, or files."""

from __future__ import annotations

import logging
import time
from typing import Optional

from pydantic import BaseModel, Field

from app.services.agentic_rag.tools.base import BaseAgentTool
from app.services.agentic_rag.tool_context import ToolContext, enforce_rbac, write_audit
from app.services.agentic_rag.llm_factory import build_chat_llm
from app.services.agentic_rag.schemas import LastAnswerObject
from app.models.chat import Message, ChatFile

logger = logging.getLogger(__name__)


class SummarizeAnswerInput(BaseModel):
    source: str = Field(default="last_answer", description="last_answer, message_id, or file_id.")
    source_id: Optional[int] = Field(default=None)
    max_points: int = Field(default=10, ge=1, le=50)
    format: str = Field(default="bullet", description="bullet or paragraph")


class SummarizeAnswerTool(BaseAgentTool):
    name: str = "summarize_answer"
    ui_label: str = "Summarizing answer"
    description: str = (
        "Summarize the previous assistant answer or a file. "
        "Use for 'summarize it in 10 points' or 'tl;dr' requests."
    )
    args_schema: type[BaseModel] = SummarizeAnswerInput

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use arun() for agent tools.")

    async def _execute(self, input_obj: SummarizeAnswerInput) -> dict:
        t0 = time.monotonic()
        ctx: ToolContext = self.ctx

        text = ""
        if input_obj.source == "last_answer":
            # ctx.state is a dict (AgentState/MessagesState) at runtime.
            lao = ctx.state.get("last_answer_object") if ctx.state else None
            if lao:
                text = getattr(lao, "summary", "") + "\n" + "\n".join(getattr(lao, "key_points", []))
            if not text:
                text = "No previous answer available."
        elif input_obj.source == "message_id" and input_obj.source_id:
            if not ctx.chat_id:
                text = "Access denied."
            else:
                msg = ctx.db.query(Message).filter(
                    Message.id == input_obj.source_id,
                    Message.chat_id == ctx.chat_id,
                ).first()
                text = msg.content if msg else ""
            if not text:
                text = "Message not found."
        elif input_obj.source == "file_id" and input_obj.source_id:
            rbac = enforce_rbac(ctx, file_id=input_obj.source_id)
            if rbac.get("file_id") is None:
                text = "Access denied."
            else:
                cf = ctx.db.query(ChatFile).filter(ChatFile.id == rbac["file_id"]).first()
                text = cf.markdown_content or "" if cf else ""
        else:
            text = "Unsupported source."

        max_chars = 4000
        text = text[:max_chars]

        prompt = (
            f"Summarize the following text into at most {input_obj.max_points} "
            f"{input_obj.format}s. Be concise and preserve key facts.\n\n{text}"
        )

        try:
            llm = build_chat_llm(ctx.org_id, ctx.db, role="query", temperature=0.0)
            response = await llm.ainvoke([{"role": "user", "content": prompt}])
            summary = str(response.content).strip()
        except Exception as exc:
            logger.warning("[summarize_answer] LLM failed: %s", exc)
            summary = "[Summary unavailable due to model error.]"

        bullets = [s.strip("- ").strip() for s in summary.splitlines() if s.strip()]
        latency_ms = round((time.monotonic() - t0) * 1000)
        write_audit(ctx, "summarize_answer", input_obj.model_dump(), {"summary": summary[:200]}, latency_ms=latency_ms, status="ok")

        return {
            "ok": True,
            "result": {
                "summary": summary,
                "key_points": bullets[: input_obj.max_points],
                "max_points": input_obj.max_points,
            },
            "error": None,
            "tokens": len(text) // 4,
        }
