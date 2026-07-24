"""Pydantic models for LangGraph structured output."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class SubtaskRouting(BaseModel):
    """Per-subtask routing flags used to decide which context sources to use."""

    needs_retrieval: bool = Field(
        default=True,
        description="True if this subtask needs document retrieval (vector, sparse, exact search, Neo4j graph expansion). False for chat-only follow-ups like 'what did I say', 'explain what you mentioned', 'summarize the conversation'.",
    )
    needs_file_content: bool = Field(
        default=False,
        description="True if this subtask needs the content of an attached file.",
    )
    needs_file_metadata: bool = Field(
        default=False,
        description="True if this subtask only needs file names/descriptions (not content).",
    )


class QueryAnalysis(BaseModel):
    """Structured output for query classification."""

    is_clear: bool = Field(
        default=True,
        description="Whether the user's question is clear and answerable from the knowledge base.",
    )
    questions: List[str] = Field(
        default_factory=list,
        description="List of rewritten, self-contained questions extracted from the query.",
    )
    clarification_needed: str = Field(
        default="",
        description="Explanation of what additional information is needed, or empty string if none.",
    )
    # Per-subtask routing flags (one per question in `questions`)
    subtask_routing: List[SubtaskRouting] = Field(
        default_factory=list,
        description="Per-subtask routing decision. Each entry corresponds to the question at the same index. Controls which context sources the subtask uses: retrieval, file content, or chat history only.",
    )
    # Subtask dependencies (v2)
    subtask_dependencies: List[List[int]] = Field(
        default_factory=list,
        description="Each entry is a list of subtask indices that subtask i depends on. For independent subtasks, the list is empty. For dependent subtasks, e.g. [0] means subtask 1 depends on subtask 0. Length must match questions list.",
    )


class AnswerEvaluation(BaseModel):
    """Structured output for answer quality evaluation."""

    score: int = Field(
        ge=0,
        le=100,
        description="Overall quality score from 0 to 100.",
    )
    faithfulness: int = Field(
        ge=0,
        le=100,
        description="How well the answer is grounded in the retrieved context (0-100).",
    )
    completeness: int = Field(
        ge=0,
        le=100,
        description="How completely the answer addresses all parts of the query (0-100).",
    )
    citation_quality: int = Field(
        ge=0,
        le=100,
        description="How accurate and properly formatted the citations are (0-100).",
    )
    issues: List[str] = Field(
        default_factory=list,
        description="List of specific quality issues found, or empty if none.",
    )
