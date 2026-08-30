"""Branch selection and clarification state-machine endpoints.

Provides the active-branch setter used by the branch-switcher UI, plus the
clarification polling and submission endpoints that drive the agentic RAG
pipeline's interrupt/resume flow: when the agent needs more information it
creates a pending ClarificationRequest, the frontend polls for it, and the
user's response resumes the paused LangGraph execution via SSE streaming.
"""
import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel
from fastapi import Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.api_v1.chat import router
from app.db.session import get_db
from app.models.user import User
from app.models.chat import Chat, Message
from app.api.api_v1.rbac import chat_owner_filter as _chat_owner_filter
from app.core.security import get_current_user
from app.models.clarification import ClarificationRequest as ClarificationRequestModel

logger = logging.getLogger(__name__)


class SetActiveBranchRequest(BaseModel):
    parent_message_id: int
    selected_message_id: int


@router.put("/{chat_id}/active-branch")
def set_active_branch(
    *,
    db: Session = Depends(get_db),
    chat_id: int,
    body: SetActiveBranchRequest,
    current_user: User = Depends(get_current_user),
) -> Any:
    """Record which branch the user is currently viewing for a branching point.

    Stored in chat.active_branches as {parent_message_id: selected_message_id}.
    Used on reload to pick the right branch.
    """
    chat = (
        db.query(Chat)
        .filter(Chat.id == chat_id, _chat_owner_filter(current_user))
        .first()
    )
    if not chat:
        logger.warning("[ACTIVE_BRANCH] chat %s not found", chat_id)
        raise HTTPException(status_code=404, detail="Chat not found")

    branches = dict(chat.active_branches or {})
    branches[str(body.parent_message_id)] = body.selected_message_id
    chat.active_branches = branches
    db.commit()
    return {"ok": True}


# -- Clarification State Machine -----------------------------------------------


class ClarificationSubmitRequest(BaseModel):
    """Request body for user clarification response."""
    chat_id: int
    clarification_id: int  # The ClarificationRequest.id from the pending endpoint
    response: str          # User's clarification answer


class ClarificationPendingResponse(BaseModel):
    """Response from the pending clarification check."""
    pending: bool
    question: str = ""
    options: list[str] = []
    rationale: str = ""
    clarification_id: int = 0
    attempt: int = 1
    max_attempts: int = 2


@router.get("/clarification/pending", response_model=ClarificationPendingResponse)
async def get_pending_clarification(
    *,
    db: Session = Depends(get_db),
    chat_id: int = Query(...),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Check if there is a pending clarification request for this chat.

    The frontend polls this endpoint every 2 seconds while waiting for
    the agent to respond. If the agent needs clarification, it creates
    a ClarificationRequest row and yields a 'c:' SSE event. The frontend
    polls this endpoint to get the question details.
    """
    # Verify chat ownership
    chat = (
        db.query(Chat)
        .filter(
            Chat.id == chat_id,
            _chat_owner_filter(current_user),
        )
        .first()
    )
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Find the most recent pending clarification for this chat
    pending = (
        db.query(ClarificationRequestModel)
        .filter(
            ClarificationRequestModel.chat_id == chat_id,
            ClarificationRequestModel.status == "pending",
        )
        .order_by(ClarificationRequestModel.created_at.desc())
        .first()
    )

    if not pending:
        return ClarificationPendingResponse(pending=False)

    return ClarificationPendingResponse(
        pending=True,
        question=pending.question or "",
        options=pending.options or [],
        rationale=pending.rationale or "",
        clarification_id=pending.id,
        attempt=pending.attempt,
        max_attempts=2,
    )


def _extract_final_answer(final_state: dict) -> str:
    """Extract the final answer string from the LangGraph final state."""
    fa_raw = final_state.get("final_answer") or final_state.get("answer", "")
    if isinstance(fa_raw, str):
        return fa_raw
    if isinstance(fa_raw, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) and p.get("type") == "text" else str(p)
            for p in fa_raw
        )
    return str(fa_raw)


def _extract_token_usage(transformer, final_state: dict) -> tuple[int, int]:
    """Extract (input_tokens, completion_tokens) from transformer or final state."""
    input_tokens = getattr(transformer, "_input_tokens", 0)
    completion_tokens = getattr(transformer, "_output_tokens", 0)
    if input_tokens == 0 and completion_tokens == 0:
        answer_usage = final_state.get("answer_usage")
        if isinstance(answer_usage, dict):
            input_tokens = answer_usage.get("input_tokens", 0) or 0
            completion_tokens = answer_usage.get("output_tokens", 0) or 0
    return input_tokens, completion_tokens


@router.post("/clarification")
async def submit_clarification(
    *,
    db: Session = Depends(get_db),
    body: ClarificationSubmitRequest,
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """
    Submit user's clarification response to an in-progress agent query.

    Updates the ClarificationRequest row, stores the response as a user
    message, then resumes the paused LangGraph execution and streams
    the resumed events back as SSE so the frontend can render them.
    """
    # Verify chat ownership
    chat = (
        db.query(Chat)
        .filter(
            Chat.id == body.chat_id,
            _chat_owner_filter(current_user),
        )
        .first()
    )
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Find and verify the clarification request
    clarification = (
        db.query(ClarificationRequestModel)
        .filter(
            ClarificationRequestModel.id == body.clarification_id,
            ClarificationRequestModel.chat_id == body.chat_id,
            ClarificationRequestModel.status == "pending",
        )
        .first()
    )
    if not clarification:
        raise HTTPException(status_code=404, detail="Clarification request not found or already answered")

    # Update the clarification request with user response
    clarification.user_response = body.response
    clarification.status = "answered"
    clarification.answered_at = datetime.now(timezone.utc)

    # Store clarification response as a user message so the LLM sees it
    # in the normal conversation history (same as any other user query).
    clarification_msg = Message(
        content=body.response,
        role="user",
        chat_id=body.chat_id,
    )

    db.add(clarification_msg)
    db.commit()
    db.refresh(clarification)

    # Resume the paused LangGraph execution and stream it as SSE.
    from langgraph.types import Command
    from app.services.agentic_rag.agent_graph import build_agent_graph
    from app.services.agentic_rag.llm_factory import get_org_llm
    from app.services.agentic_rag.redis_memory import get_redis_memory
    from app.services.agentic_rag.streaming import AgenticRAGTransformer
    from app.services.agentic_rag.tool_context import ToolContext

    memory = await get_redis_memory()
    thread_id = f"chat-{body.chat_id}"
    config = {"configurable": {"thread_id": thread_id}}
    org_cfg = get_org_llm(current_user.org_id, db, role="chat")
    ctx = ToolContext(
        db=db,
        user_id=current_user.id,
        org_id=current_user.org_id,
        chat_id=body.chat_id,
        message_id=clarification.assistant_message_id,
        qdrant_client=None,
        redis_memory=memory,
        org_llm_config=org_cfg,
        state=None,
    )
    graph = build_agent_graph(ctx)

    logger.info(
        "[CLARIFICATION] chat_id=%d clarification_id=%d resuming | user=%d",
        body.chat_id, clarification.id, current_user.id,
    )

    async def response_stream():
        resumed_stream = await graph.astream_events(
            Command(resume=body.response),
            config=config,
            version="v3",
            transformers=[AgenticRAGTransformer],
        )

        transformer = resumed_stream.extensions["events"]

        async def _drain_raw() -> None:
            async for _ in resumed_stream:
                pass

        raw_task = asyncio.create_task(_drain_raw())

        def _prefix(event: dict) -> str:
            """Map transformer event names to the SSE prefixes processStreamLine expects."""
            name = event.get("event", "")
            mapping = {
                "token": "0",
                "rewritten_query": "1",
                "expanded_query": "eq",
                "context": "2",
                "error": "3",
                "agent_step": "4",
                "progress": "p",
                "task_list": "t",
                "thinking": "th",
                "answer_rewrite": "r",
                "plan": "pl",
                "tool_call": "tc",
                "tool_observation": "to",
                "last_answer": "la",
                "interrupt": "c",
            }
            return mapping.get(name) or name

        try:
            async for event in transformer:
                if event:
                    yield f"{_prefix(event)}:{json.dumps(event)}\n"
        finally:
            if not raw_task.done():
                raw_task.cancel()
                try:
                    await raw_task
                except asyncio.CancelledError:
                    pass

        # Check if interrupted again (re-interrupt during clarification round-trip)
        # NOTE: `interrupted` and `interrupts` are async methods on
        # AsyncGraphRunStream (not properties) — they must be awaited.
        if await resumed_stream.interrupted():
            pending_interrupts = await resumed_stream.interrupts()
            interrupt_value = (
                str(pending_interrupts[0].value) if pending_interrupts else ""
            )

            # Create a new pending request for this re-interrupt.
            # Reuse the assistant_message_id from the original clarification
            # (already fetched above, no need for a second DB query).
            clar_req = ClarificationRequestModel(
                chat_id=body.chat_id,
                assistant_message_id=clarification.assistant_message_id,
                question=interrupt_value,
                rationale="Re-interrupt during clarification round-trip",
                status="pending",
                attempt=clarification.attempt + 1,
            )
            db.add(clar_req)
            db.commit()
            db.refresh(clar_req)

            interrupt_payload = {
                'question': interrupt_value,
                'clarification_id': clar_req.id,
                'attempt': clar_req.attempt,
                'max_attempts': 2,
            }
            yield f"c:{json.dumps(interrupt_payload)}\n"
            return

        # Emit done event with final state
        final_output = await resumed_stream.output()
        final_state = final_output if isinstance(final_output, dict) else getattr(final_output, "values", {}) or {}

        final_answer = _extract_final_answer(final_state)

        # Collect usage from transformer
        input_tokens, completion_tokens = _extract_token_usage(transformer, final_state)

        done_payload = {
            "finishReason": "stop",
            "usage": {
                "promptTokens": input_tokens,
                "completionTokens": completion_tokens,
                "final_confidence": final_state.get("final_confidence", 0.0),
                "confidence_level": final_state.get("confidence_level", "none"),
                "faithfulness": final_state.get("faithfulness", 0),
                "completeness": final_state.get("completeness", 0),
            },
            "full_response": final_answer,
        }
        yield f"done:{json.dumps(done_payload)}\n"

    return StreamingResponse(
        response_stream(),
        media_type="text/event-stream",
        headers={
            "x-vercel-ai-data-stream": "v1",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
