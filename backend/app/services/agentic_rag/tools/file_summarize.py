"""file_summarize tool — map-reduce summarization of attached files."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from pydantic import BaseModel, Field

from app.models.chat import ChatFile
from app.services.agentic_rag.llm_factory import build_chat_llm
from app.services.agentic_rag.tool_context import ToolContext, enforce_rbac, write_audit
from app.services.agentic_rag.token_budget import count_tokens
from app.services.agentic_rag.tools.base import BaseAgentTool

logger = logging.getLogger(__name__)


class FileSummarizeInput(BaseModel):
    file_id: Optional[int] = Field(default=None)
    focus: Optional[str] = Field(default=None)
    max_points: Optional[int] = Field(default=10)
    chunk_size: int = Field(default=4000, ge=1000, le=8000)


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
    if not cf or not cf.markdown_content:
        return None, {"ok": False, "result": {}, "error": "File not found or not processed.", "tokens": 0}

    return cf, None


async def _summarize_chunks(llm, chunks: list[str], focus: Optional[str]) -> list[str]:
    async def _summarize_chunk(chunk: str, idx: int) -> str:
        prompt = (
            f"Summarize the following part of a document. "
            f"Focus: {focus or 'key points'}. Keep it concise.\n\n{chunk}"
        )
        try:
            resp = await llm.ainvoke([{"role": "user", "content": prompt}])
            return str(resp.content).strip()
        except Exception as exc:
            logger.warning("[file_summarize] chunk %d failed: %s", idx, exc)
            return ""

    chunk_summaries: list[str] = []
    for i in range(0, len(chunks), 3):
        batch = chunks[i:i + 3]
        results = await asyncio.gather(*[_summarize_chunk(c, i + j) for j, c in enumerate(batch)])
        chunk_summaries.extend(results)
    return chunk_summaries


class FileSummarizeTool(BaseAgentTool):
    name: str = "file_summarize"
    ui_label: str = "Summarizing file"
    description: str = (
        "Summarize a large attached file. Use when the user says 'summarise this file' "
        "or the file is too big to fit in the context window."
    )
    args_schema: type[BaseModel] = FileSummarizeInput

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use arun() for agent tools.")

    async def _execute(self, input_obj: FileSummarizeInput) -> dict:
        t0 = time.monotonic()
        ctx: ToolContext = self.ctx

        cf, err = _resolve_file(ctx, input_obj.file_id)
        if err:
            return err

        text = cf.markdown_content
        chunk_chars = input_obj.chunk_size * 4  # rough
        chunks = [text[i:i + chunk_chars] for i in range(0, len(text), chunk_chars)]

        llm = build_chat_llm(ctx.org_id, ctx.db, role="query", temperature=0.0)
        chunk_summaries = await _summarize_chunks(llm, chunks, input_obj.focus)

        combined = "\n\n".join(s for s in chunk_summaries if s)
        reduce_prompt = (
            f"Combine the following section summaries into a final summary with "
            f"{input_obj.max_points or 10} key points. "
            f"Focus: {input_obj.focus or 'key points'}.\n\n{combined}"
        )

        try:
            final_resp = await llm.ainvoke([{"role": "user", "content": reduce_prompt}])
            final_summary = str(final_resp.content).strip()
        except Exception as exc:
            logger.warning("[file_summarize] reduce failed: %s", exc)
            final_summary = combined[:2000]

        bullets = [s.strip("- ").strip() for s in final_summary.splitlines() if s.strip()]

        latency_ms = round((time.monotonic() - t0) * 1000)
        write_audit(ctx, "file_summarize", input_obj.model_dump(), {"file_name": cf.file_name, "chunks": len(chunks)}, latency_ms=latency_ms, status="ok")

        return {
            "ok": True,
            "result": {
                "summary": final_summary,
                "key_points": bullets[: input_obj.max_points],
                "file_name": cf.file_name,
                "chunks_processed": len(chunks),
            },
            "error": None,
            "tokens": count_tokens(final_summary),
        }
