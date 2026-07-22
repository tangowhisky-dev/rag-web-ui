"""Pydantic models for LangGraph structured output."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class QueryAnalysis(BaseModel):
    """Structured output for query classification."""

    is_clear: bool = Field(
        description="Whether the user's question is clear and answerable from the knowledge base."
    )
    questions: List[str] = Field(
        description="List of rewritten, self-contained questions extracted from the query."
    )
    clarification_needed: str = Field(
        description="Explanation of what additional information is needed, or empty string if none."
    )


class QueryDecomposition(BaseModel):
    """Structured output for subtask decomposition."""

    subtasks: List[str] = Field(
        description="List of 2-5 focused subtasks extracted from the query. Each should be a standalone question."
    )


class SubtaskIndependence(BaseModel):
    """Structured output for subtask independence analysis."""

    dependencies: List[bool] = Field(
        description="List of booleans, one per subtask. True = fully independent (can be executed in parallel), False = dependent (requires sequential processing)."
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
