"""Agent graph package — shared modules for the v2 pipeline.

Shared modules used by v2:
  - helpers.py: budgets, writer, wall-clock, chart/office marker substitution,
    unified timeline event emitter
  - tooling.py: _run_tool, _merge_retrieved_docs, _summarize_result
  - observations.py: observation formatting, tool descriptions, search history
  - compaction.py: context compaction for prompt budget management
  - finalization.py: _build_finalize_prompt, _stream_final_answer
  - load_context.py: load_context_node
"""

from __future__ import annotations

# Shared helpers (used by v2)
from .helpers import (
    _coerce_observation,
    _emit_timeline,
    _extract_balanced,
    _extract_json_block,
    _is_transient_error,
    _substitute_chart_markers,
    _tool_call_budget,
    _total_tool_budget,
    _wall_clock_exceeded,
    _writer,
)

# Shared observations (used by v2)
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

# Shared compaction (used by v2)
from .compaction import (
    _build_compaction_llm,
    _compact_if_needed,
    _compact_messages_llm,
    _compact_stage1_observations,
    _compact_stage2_docs,
    _compact_stage3_messages,
    _trim_docs_to_budget,
)

# Shared load context (used by v2)
from .load_context import load_context_node

# Shared tooling (used by v2)
from .tooling import (
    _merge_retrieved_docs,
    _run_tool,
    _summarize_result,
)

# Shared finalization (used by v2)
from .finalization import (
    _build_finalize_prompt,
    _build_last_answer_object_deterministic,
    _stream_final_answer,
    finalize_node,
    save_memory_node,
)

# Re-exports used by shared modules and external code
from app.core.config import settings
from app.core.settings_registry import get_def
from app.services.agentic_rag.llm_factory import build_chat_llm
from app.services.agentic_rag.nodes import (
    _agent_step,
    answer_evaluation_node,
    history_to_text,
    select_recent_history,
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

# Module-level logger
import logging

logger = logging.getLogger(__name__)

# Graph state
from app.services.agentic_rag.graph_state import AgentState

__all__ = [
    # Helpers
    "_coerce_observation",
    "_emit_timeline",
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
    # Tooling
    "_merge_retrieved_docs",
    "_run_tool",
    "_summarize_result",
    # Finalization
    "_build_finalize_prompt",
    "_build_last_answer_object_deterministic",
    "_stream_final_answer",
    "finalize_node",
    "save_memory_node",
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
