"""Agent loop graph for the enterprise agent.

Atomic tools topology:
  load_context → plan → think → [tool → sufficiency_check → think ...] → finalize → save_memory

This package splits the original monolithic agent_graph.py into focused
sub-modules. All public names are re-exported here so existing imports
(`from app.services.agentic_rag.agent_graph import build_agent_graph`)
continue to work.
"""

from __future__ import annotations

# Build
from .build import build_agent_graph

# Helpers
from .helpers import (
    _coerce_observation,
    _extract_balanced,
    _extract_json_block,
    _is_transient_error,
    _substitute_chart_markers,
    _tool_call_budget,
    _total_tool_budget,
    _wall_clock_exceeded,
    _writer,
)

# Observations
from .observations import (
    _compact_observations,
    _format_retrieval_obs_compact,
    _format_retrieval_obs_full,
    _non_retrieval_observations_text,
    _observations_metadata_text,
    _observations_text,
    _prune_contiguous_overlaps,
    _strip_overlap,
    _tool_descriptions_text,
    _tried_search_queries,
)

# Compaction
from .compaction import (
    _build_compaction_llm,
    _compact_if_needed,
    _compact_messages_llm,
    _compact_stage1_observations,
    _compact_stage2_docs,
    _compact_stage3_messages,
    _trim_docs_to_budget,
)

# Load context
from .load_context import load_context_node

# Planning
from .planning import (
    _build_plan_user_prompt,
    _check_clarification_budget,
    _invoke_plan_llm,
    plan_node,
    route_plan,
)

# Thinking
from .thinking import (
    _build_think_prompt,
    _invoke_think_llm,
    _parse_tool_calls,
    _rebuild_think_after_compaction,
    _think_early_exit,
    route_think,
    think_node,
)

# Tooling
from .tooling import (
    _dispatch_tool_calls,
    _merge_observation_docs,
    _merge_retrieved_docs,
    _retry_failed_calls,
    _run_tool,
    _seed_existing_docs,
    tool_node,
)

# Execution check (moved from reflection.py)
from .execution_check import (
    _build_execution_summary,
    _build_subtask_status,
    _collect_tool_failures,
    _count_successful_by_tool,
    _retrieval_hit_count,
    _verify_execution,
)

# Sufficiency check (replaces reflect_final)
from .sufficiency import (
    route_sufficiency,
    sufficiency_check_node,
)

# Finalization
from .finalization import (
    _build_finalize_prompt,
    _build_last_answer_object,
    _stream_final_answer,
    finalize_node,
    save_memory_node,
)

# Reflection (active nodes only; reflect_node/reflect_final_node removed)
from .reflection import (
    answer_scoring_node,
    clarify_interrupt_node,
)

# Re-export external names that tests and app code patch on this module.
# Each sub-module imports these directly; patching must target the
# specific sub-module (e.g. agent_graph.thinking.get_setting).
from app.core.config import settings
from app.core.settings_registry import get_def
from app.services.agentic_rag.llm_factory import build_chat_llm
from app.services.agentic_rag.nodes import (
    _agent_step,
    answer_evaluation_node,
    history_to_text,
    select_recent_history,
)
from app.services.agentic_rag.prompts import (
    AGENT_SYSTEM_PROMPT,
    FINALIZE_ANSWER_PROMPT,
    FINALIZE_GUARDRAIL_PROMPT,
    LAST_ANSWER_EXTRACT_PROMPT,
    PLAN_SYSTEM_PROMPT,
    THINK_SYSTEM_PROMPT,
)
from app.services.agentic_rag.schemas import LastAnswerObject, Observation, Plan, Subtask
from app.services.agentic_rag.token_budget import count_tokens
from app.services.agentic_rag.tool_call_parser import parse_think_response
from app.services.agentic_rag.tool_context import ToolContext, write_audit
from app.services.agentic_rag.tools import applicable_tools
from app.services.agentic_rag.utils import format_context_string, group_docs_by_document, normalize_citations
from app.services.settings_service import get_setting

# LangGraph / LangChain re-exports
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

# Stdlib re-exports
import asyncio
import json
import re
import time
from functools import partial
from typing import Any, Optional

# Module-level logger (some sub-modules reference agent_graph.logger)
import logging

logger = logging.getLogger(__name__)

# Graph state
from app.services.agentic_rag.graph_state import AgentState

__all__ = [
    # Build
    "build_agent_graph",
    # Helpers
    "_coerce_observation",
    "_extract_balanced",
    "_extract_json_block",
    "_is_transient_error",
    "_substitute_chart_markers",
    "_tool_call_budget",
    "_total_tool_budget",
    "_wall_clock_exceeded",
    "_writer",
    # Observations
    "_compact_observations",
    "_format_retrieval_obs_compact",
    "_format_retrieval_obs_full",
    "_non_retrieval_observations_text",
    "_observations_metadata_text",
    "_observations_text",
    "_prune_contiguous_overlaps",
    "_strip_overlap",
    "_tool_descriptions_text",
    "_tried_search_queries",
    # Compaction
    "_build_compaction_llm",
    "_compact_if_needed",
    "_compact_messages_llm",
    "_compact_stage1_observations",
    "_compact_stage2_docs",
    "_compact_stage3_messages",
    "_trim_docs_to_budget",
    # Load context
    "load_context_node",
    # Planning
    "_build_plan_user_prompt",
    "_check_clarification_budget",
    "_invoke_plan_llm",
    "plan_node",
    "route_plan",
    # Thinking
    "_build_think_prompt",
    "_invoke_think_llm",
    "_parse_tool_calls",
    "_rebuild_think_after_compaction",
    "_think_early_exit",
    "route_think",
    "think_node",
    # Tooling
    "_dispatch_tool_calls",
    "_merge_observation_docs",
    "_merge_retrieved_docs",
    "_retry_failed_calls",
    "_run_tool",
    "_seed_existing_docs",
    "tool_node",
    # Execution check
    "_build_execution_summary",
    "_build_subtask_status",
    "_collect_tool_failures",
    "_count_successful_by_tool",
    "_retrieval_hit_count",
    "_verify_execution",
    # Sufficiency check
    "route_sufficiency",
    "sufficiency_check_node",
    # Finalization
    "_build_finalize_prompt",
    "_build_last_answer_object",
    "_stream_final_answer",
    "finalize_node",
    "save_memory_node",
    # Reflection (active)
    "answer_scoring_node",
    "clarify_interrupt_node",
    # External names
    "settings",
    "get_def",
    "build_chat_llm",
    "get_setting",
    "applicable_tools",
    "count_tokens",
    "parse_think_response",
    "ToolContext",
    "write_audit",
    "format_context_string",
    "group_docs_by_document",
    "normalize_citations",
    "AgentState",
    "LastAnswerObject",
    "Observation",
    "Plan",
    "Subtask",
    "AIMessage",
    "HumanMessage",
    "END",
    "StateGraph",
    "interrupt",
    "answer_evaluation_node",
    "history_to_text",
    "select_recent_history",
    "_agent_step",
    "AGENT_SYSTEM_PROMPT",
    "FINALIZE_ANSWER_PROMPT",
    "FINALIZE_GUARDRAIL_PROMPT",
    "LAST_ANSWER_EXTRACT_PROMPT",
    "PLAN_SYSTEM_PROMPT",
    "THINK_SYSTEM_PROMPT",
    # Stdlib
    "asyncio",
    "json",
    "re",
    "time",
    "partial",
    "Any",
    "Optional",
    "logger",
]
