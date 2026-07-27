"""LangGraph-based agentic RAG pipeline.

Routes to the new enterprise agent loop when AGENT_LOOP_ENABLED is true,
otherwise falls back to the legacy agentic RAG runner.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, List, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


async def run_agentic_rag(
    query: str,
    chat_id: int,
    knowledge_base_ids: List[int],
    db: Any,
    file_markdown: Optional[str] = None,
    temperature: float = 0.0,
    model_name: Optional[str] = None,
    display_query: Optional[str] = None,
    api_base: Optional[str] = None,
    query_model: Optional[str] = None,
    org_id: Optional[int] = None,
    user_id: Optional[int] = None,
    generate_answer: bool = True,
    message_id: Optional[int] = None,
) -> AsyncGenerator[dict, None]:
    """Route between the legacy pipeline and the new enterprise agent loop."""
    if settings.AGENT_LOOP_ENABLED:
        from .agent_runner import run_agent_loop
        async for event in run_agent_loop(
            query=query,
            kb_ids=knowledge_base_ids,
            db=db,
            file_markdown=file_markdown,
            temperature=temperature,
            model_name=model_name,
            api_base=api_base,
            org_id=org_id,
            chat_id=chat_id,
            user_id=user_id,
            message_id=message_id,
            display_query=display_query,
            query_model=query_model,
        ):
            yield event
        return

    from .graph_runner import run_agentic_rag as _run
    async for event in _run(
        query=query,
        kb_ids=knowledge_base_ids,
        db=db,
        file_markdown=file_markdown,
        temperature=temperature,
        model_name=model_name,
        api_base=api_base,
        org_id=org_id,
        chat_id=chat_id,
        user_id=user_id,
        generate_answer=generate_answer,
    ):
        yield event
