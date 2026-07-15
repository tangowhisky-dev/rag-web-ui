"""LangGraph-based agentic RAG pipeline.

The agent operates via a LangGraph StateGraph with nested subgraph architecture:
1. Rewrite query using chat history
2. Classify query (LLM-based structured classification)
3. For simple queries: direct retrieval → stream answer
4. For complex queries: decompose → iterate subtasks via agent subgraph → synthesize
5. All tokens, progress, and thinking traces stream in real-time

Node flow:
  START → rewrite → classify → [direct_retrieval | agent_loop] → synthesize → END

SSE Event Protocol:
  p:  progress       - transient status messages
  t:  task_list      - subtask list with status
  th: thinking       - reasoning model chain-of-thought
  0:  token          - streaming answer text
  1:  rewritten_query - standalone query
  2:  context        - retrieved documents
  3:  error          - exception message
  d:  done           - finish reason + usage
"""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, List, Optional

logger = logging.getLogger(__name__)


async def run_agentic_rag(
    query: str,
    chat_id: int,
    knowledge_base_ids: List[int],
    db: Any,
    recent_lc_history: list,
    existing_summary: Optional[str] = None,
    file_markdown: Optional[str] = None,
    temperature: float = 0.0,
    model_name: Optional[str] = None,
    display_query: Optional[str] = None,
    api_base: Optional[str] = None,
    query_model: Optional[str] = None,
    org_id: Optional[int] = None,
    generate_answer: bool = True,
) -> AsyncGenerator[dict, None]:
    """Single autonomous agentic agent via LangGraph StateGraph.

    Streams everything in real-time: tokens, progress, thinking traces.
    """
    from .graph_runner import run_agentic_rag as _run
    async for event in _run(
        query=query,
        kb_ids=knowledge_base_ids,
        db=db,
        recent_lc_history=recent_lc_history,
        existing_summary=existing_summary,
        file_markdown=file_markdown,
        temperature=temperature,
        model_name=model_name,
        api_base=api_base,
        org_id=org_id,
        chat_id=chat_id,
        generate_answer=generate_answer,
    ):
        yield event
