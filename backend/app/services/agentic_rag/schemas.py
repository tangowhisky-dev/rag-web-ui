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
