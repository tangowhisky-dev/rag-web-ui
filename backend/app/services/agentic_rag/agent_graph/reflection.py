"""Clarification and answer-scoring nodes.

clarify_interrupt_node: pauses execution to ask the user for clarification;
  resumes on response via LangGraph interrupt/resume.
answer_scoring_node: delegates to the LLM-based answer evaluation.

The execution-summary builder and verifier now live in execution_check.py.
reflect_node and reflect_final_node have been removed — the sufficiency_check
node replaces reflect_final, and reflect_node's deterministic recovery rules
are no longer needed in the atomic-tools topology.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from app.services.agentic_rag.nodes import _agent_step, answer_evaluation_node
from app.services.agentic_rag.schemas import Plan

logger = logging.getLogger(__name__)


async def clarify_interrupt_node(state) -> dict:
    """Pause execution and ask the user for clarification; resumes on response."""
    with _agent_step("clarify_interrupt"):
        plan = state.get("plan") or Plan()
        question = ""
        if isinstance(plan, Plan):
            question = plan.clarification_question or ""
        if not question:
            question = "Could you clarify what you need?"

        # No try/except and no pre-emitted custom event here.
        # `interrupt()` raises GraphInterrupt, which subclasses Exception —
        # catching it swallowed the pause and let the graph run on with an
        # empty answer. Emitting a custom "interrupt" event *before* the call
        # also let the consumer close the stream before LangGraph could
        # persist the interrupt checkpoint, so the resume had nothing to
        # resume. The interrupt is surfaced from the graph's own
        # `__interrupt__` update in agent_runner instead.
        user_response = interrupt({"question": question})

        response_text = str(user_response) if user_response else ""
        return {
            # add_messages appends: return only the new message.
            "messages": [HumanMessage(content=response_text)],
            "clarification_response": response_text,
            "clarification_count": state.get("clarification_count", 0) + 1,
            "needs_clarification": False,
        }


async def answer_scoring_node(state, ctx: "ToolContext") -> dict:
    """Evaluate the final answer quality."""
    with _agent_step("answer_scoring"):
        return await answer_evaluation_node(state, ctx=ctx)
