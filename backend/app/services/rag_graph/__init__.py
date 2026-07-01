"""LangGraph-based multi-agent RAG orchestration — Agentic Pipeline v2.

Pipeline flow:
  rewrite_query
    → context_router          (smart source routing: kb / file / both)
    → decompose_query         (split into 2-5 atomic sub-queries)
    → parallel_retrieval      (hybrid search per sub-query, reinforced dedup)
    → extract_file_sections   (select relevant file sections per sub-query)
    → draft_answer            (draft answer for grading — not final output)
    → grade_coverage          (LLM grades which sub-queries are covered)
    → [conditional_router]
        ├─ all covered          → generate_answer (final)
        ├─ uncovered, attempt=0 → widened_retrieval  → draft_answer → grade_coverage
        ├─ uncovered, attempt=1 → keyword_search_loop → draft_answer → grade_coverage
        └─ attempt >= 2         → generate_answer (partial / unable)
"""

from __future__ import annotations

from app.services.rag_graph.schemas import (
    RAGGraphState,
    EVENT_AGENT_STEP,
    EVENT_REWRITTEN,
    EVENT_CONTEXT,
    EVENT_TOKEN,
    EVENT_DONE,
    _RouterOutput,
    _SectionOutput,
    _SubQueriesOutput,
    _CoverageItem,
    _CoverageOutput,
    _KeywordsOutput,
)
from app.services.rag_graph.helpers import (
    _get_llm,
    _serialise_doc,
    _dedup_and_reinforce,
    _build_context_string,
)
from app.services.rag_graph.nodes import (
    rewrite_query_node,
    context_router_node,
    chat_history_retrieval_node,
    decompose_query_node,
    parallel_retrieval_node,
    extract_file_sections_node,
    draft_answer_node,
    grade_coverage_node,
    _route_after_grade,
    widened_retrieval_node,
    keyword_search_loop_node,
    generate_answer_node,
    _build_rag_graph,
    run_stream,
)

__all__ = [
    # Schemas
    "RAGGraphState",
    "EVENT_AGENT_STEP",
    "EVENT_REWRITTEN",
    "EVENT_CONTEXT",
    "EVENT_TOKEN",
    "EVENT_DONE",
    "_RouterOutput",
    "_SectionOutput",
    "_SubQueriesOutput",
    "_CoverageItem",
    "_CoverageOutput",
    "_KeywordsOutput",
    # Helpers
    "_get_llm",
    "_serialise_doc",
    "_dedup_and_reinforce",
    "_build_context_string",
    # Nodes
    "rewrite_query_node",
    "context_router_node",
    "chat_history_retrieval_node",
    "decompose_query_node",
    "parallel_retrieval_node",
    "extract_file_sections_node",
    "draft_answer_node",
    "grade_coverage_node",
    "_route_after_grade",
    "widened_retrieval_node",
    "keyword_search_loop_node",
    "generate_answer_node",
    "_build_rag_graph",
    "run_stream",
]
