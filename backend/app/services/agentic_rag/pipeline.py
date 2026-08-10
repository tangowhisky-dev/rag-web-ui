"""Enterprise agent pipeline — always uses the autonomous agent loop."""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, List, Optional

logger = logging.getLogger(__name__)


async def run_agentic_rag(
    query: str,
    chat_id: int,
    knowledge_base_ids: List[int],
    db: Any,
    file_markdown: Optional[str] = None,
    display_query: Optional[str] = None,
    org_id: Optional[int] = None,
    user_id: Optional[int] = None,
    message_id: Optional[int] = None,
) -> AsyncGenerator[dict, None]:
    """Run the enterprise agent loop and stream SSE events."""
    from .agent_runner import run_agent_loop
    async for event in run_agent_loop(
        query=query,
        kb_ids=knowledge_base_ids,
        db=db,
        file_markdown=file_markdown,
        org_id=org_id,
        chat_id=chat_id,
        user_id=user_id,
        message_id=message_id,
        display_query=display_query,
    ):
        yield event
