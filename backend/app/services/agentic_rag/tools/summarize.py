"""summarize tool — summarize arbitrary text via single-call or map-reduce.

Consolidates answer and file summarization under a single `summarize` tool. Takes raw
text as input (the model retrieves it via file_read or uses conversation
context). Auto-selects single-call for small text or map-reduce for large
text. Errors if input exceeds the token cap instead of silently truncating.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.services.agentic_rag.llm_factory import build_chat_llm
from app.services.agentic_rag.prompts import (
    FILE_SUMMARIZE_MAP_PROMPT,
    FILE_SUMMARIZE_REDUCE_PROMPT,
    SUMMARIZE_ANSWER_PROMPT,
)
from app.services.agentic_rag.tool_context import ToolContext, write_audit
from app.services.agentic_rag.token_budget import count_tokens
from app.services.agentic_rag.tools.base import BaseAgentTool

logger = logging.getLogger(__name__)

# Below this char count, use a single LLM call. Above, use map-reduce.
SINGLE_CALL_THRESHOLD = 4000  # chars (~1000 tokens)
# Hard cap on input size. Errors instead of silently truncating.
MAX_INPUT_TOKENS = 32000


class SummarizeInput(BaseModel):
    text: str = Field(description="Text to summarize.")
    focus: Optional[str] = Field(
        default=None,
        description="What to focus the summary on, e.g. 'financial results', 'action items'.",
    )
    max_points: int = Field(default=10, ge=1, le=50)
    format: str = Field(default="bullet", description="'bullet' or 'paragraph'.")


async def _single_call(
    llm, text: str, focus: Optional[str], max_points: int, fmt: str,
) -> str:
    prompt = SUMMARIZE_ANSWER_PROMPT.format(
        max_points=max_points, format=fmt, text=text,
    )
    resp = await llm.ainvoke([{"role": "user", "content": prompt}])
    return str(resp.content).strip()


async def _map_reduce(
    llm, text: str, focus: Optional[str], max_points: int,
) -> str:
    chunk_chars = 4000
    chunks = [text[i:i + chunk_chars] for i in range(0, len(text), chunk_chars)]

    async def _summarize_chunk(chunk: str, idx: int) -> str:
        prompt = FILE_SUMMARIZE_MAP_PROMPT.format(
            focus=focus or "key points", chunk=chunk,
        )
        try:
            resp = await llm.ainvoke([{"role": "user", "content": prompt}])
            return str(resp.content).strip()
        except Exception as exc:
            logger.warning("[summarize] chunk %d failed: %s", idx, exc)
            return ""

    chunk_summaries: list[str] = []
    for i in range(0, len(chunks), 3):
        batch = chunks[i:i + 3]
        results = await asyncio.gather(
            *[_summarize_chunk(c, i + j) for j, c in enumerate(batch)]
        )
        chunk_summaries.extend(results)

    combined = "\n\n".join(s for s in chunk_summaries if s)
    reduce_prompt = FILE_SUMMARIZE_REDUCE_PROMPT.format(
        max_points=max_points,
        focus=focus or "key points",
        combined=combined,
    )
    try:
        resp = await llm.ainvoke([{"role": "user", "content": reduce_prompt}])
        return str(resp.content).strip()
    except Exception as exc:
        logger.warning("[summarize] reduce failed: %s", exc)
        return combined[:2000]


class SummarizeTool(BaseAgentTool):
    name: str = "summarize"
    ui_label: str = "Summarizing"
    description: str = (
        "Summarize text. Auto-selects single-call for small text or map-reduce for large text. "
        "Pass the text directly — retrieve it first via file_read if needed."
    )
    prompt_snippet: str = "Summarize text (single-call or map-reduce)"
    prompt_guidelines: list[str] = [
        "summarize: Best for TL;DR, executive summaries, reformatting, or shortening text. Pass the text to summarize directly.",
        "summarize: For large files, call file_read with offset/limit first to get the relevant portion, then pass that text to summarize.",
        "summarize: Input is capped at 32K tokens. If exceeded, the tool returns an error — read a smaller portion with file_read.",
    ]
    args_schema: type[BaseModel] = SummarizeInput

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use arun() for agent tools.")

    async def _execute(self, input_obj: SummarizeInput) -> dict:
        t0 = time.monotonic()
        ctx: ToolContext = self.ctx

        text = input_obj.text
        token_count = count_tokens(text)
        if token_count > MAX_INPUT_TOKENS:
            return {
                "ok": False,
                "result": {},
                "error": (
                    f"Input is {token_count} tokens, max is {MAX_INPUT_TOKENS}. "
                    "Use file_read with offset/limit to read a smaller portion."
                ),
                "tokens": 0,
            }

        from app.services.settings_service import get_setting
        tool_temp = get_setting(ctx.db, "TOOL_CALL_TEMPERATURE", ctx.org_id)
        llm = build_chat_llm(ctx.org_id, ctx.db, role="utility", temperature=tool_temp)

        if len(text) <= SINGLE_CALL_THRESHOLD:
            mechanism = "single"
            summary = await _single_call(
                llm, text, input_obj.focus, input_obj.max_points, input_obj.format,
            )
        else:
            mechanism = "map_reduce"
            summary = await _map_reduce(
                llm, text, input_obj.focus, input_obj.max_points,
            )

        bullets = [s.strip("- ").strip() for s in summary.splitlines() if s.strip()]

        latency_ms = round((time.monotonic() - t0) * 1000)
        write_audit(
            ctx, "summarize", input_obj.model_dump(),
            {"mechanism": mechanism, "summary_length": len(summary)},
            latency_ms=latency_ms, status="ok",
        )

        return {
            "ok": True,
            "result": {
                "summary": summary,
                "key_points": bullets[:input_obj.max_points],
                "mechanism": mechanism,
            },
            "error": None,
            "tokens": count_tokens(summary),
        }
