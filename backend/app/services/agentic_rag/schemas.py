"""Pydantic models for LangGraph structured output."""

from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator


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
    issues: List[str] = Field(
        default_factory=list,
        description="List of specific quality issues found, or empty if none.",
    )


class DataPoint(BaseModel):
    """A single labelled data value extracted from an answer or document."""

    label: str = Field(description="Human-readable label for the value.")
    value: Union[float, str] = Field(description="Numeric or string value.")
    unit: Optional[str] = Field(default=None, description="Unit of measurement, if any.")
    context: Optional[str] = Field(default=None, description="Sentence or phrase the value appeared in.")


class CitationRef(BaseModel):
    """Reference to a document chunk."""

    document_id: int
    chunk_index: int

    @field_validator("document_id", "chunk_index", mode="before")
    @classmethod
    def _coerce_kb_label(cls, v):
        # The extraction LLM often copies the answer's own citation label
        # verbatim, e.g. "KB-2" instead of the numeric id it refers to.
        if isinstance(v, str):
            digits = "".join(ch for ch in v if ch.isdigit())
            if digits:
                return int(digits)
        return v


class LastAnswerObject(BaseModel):
    """Structured representation of the assistant's last answer."""

    summary: str = Field(description="2-3 sentence summary of the answer.")
    key_points: List[str] = Field(default_factory=list, description="Bullet points.")
    data: Optional[List[DataPoint]] = Field(default=None, description="Numbers and statistics mentioned.")
    citations: List[CitationRef] = Field(default_factory=list, description="Chunk refs used.")
    chart_option: Optional[dict] = Field(default=None, description="Deprecated: first entry of chart_options, kept for backward compatibility with older stored messages.")
    chart_options: List[dict] = Field(default_factory=list, description="ECharts option JSON for each chart_generate call this turn, if any.")
    followups: List[str] = Field(default_factory=list, description="Suggested follow-up questions.")
    suggestion: str = Field(default="", description="One-line assessment of answer completeness, or empty string.")
    retry_strategy: str = Field(default="", description="Suggestion label: widen|narrow|pinpoint|")


class Subtask(BaseModel):
    """One step in the agent's plan."""

    id: str = Field(description="Unique subtask id, e.g. 'a', 'b'.")
    description: str = Field(description="What the subtask should accomplish.")
    tool_hint: str = Field(default="any", description="Preferred tool name or 'any'.")
    depends_on: List[str] = Field(default_factory=list, description="Subtask ids that must complete first.")
    expected_output: str = Field(default="", description="What the agent expects to observe.")
    # Per-subtask retrieval parameters. When the subtask uses rag_retrieve
    # or kb_search_documents, these let the planner express a specific
    # retrieval strategy per sub-query (e.g. subtask A searches for the
    # latest weekly update, subtask B searches for the Q3 report).
    # If null, the acting LLM decides based on the query and query_intent.
    suggested_filters: Optional[dict] = Field(
        default=None,
        description="Metadata filters for this subtask's retrieval: {title_contains, content_type, created_after, created_before, document_ids}.",
    )
    suggested_sort: Optional[dict] = Field(
        default=None,
        description="Sort spec for this subtask: {field, direction}. Use {field: 'created_at', direction: 'desc'} for 'latest'/'most recent'.",
    )
    suggested_legs: Optional[List[str]] = Field(
        default=None,
        description="Retrieval legs for this subtask: subset of ['dense', 'sparse', 'exact']. Use ['exact','sparse'] for literal title lookups.",
    )


class Plan(BaseModel):
    """Plan produced by the planner for one user turn."""

    intent: str = Field(
        default="rag",
        description="One of: rag, file_action, previous_answer_action, computation, chart, conversation, mixed.",
    )
    subtasks: List[Subtask] = Field(default_factory=list, description="Subtasks to execute.")
    needs_clarification: bool = Field(default=False, description="True if the user must clarify.")
    clarification_question: Optional[str] = Field(default=None, description="Question to ask the user.")


class QueryIntent(BaseModel):
    """Suggested filters/sort extracted from the query by rewrite_query_node.

    Folded into the existing rewrite LLM call — no separate node or extra latency.
    Only populated when a KB profile is available and the LLM produces valid JSON.
    """
    suggested_filters: Optional[dict] = Field(
        default=None,
        description="Metadata filters for rag_retrieve (title_contains, file_name_contains, content_type, created_after, created_before, document_ids).",
    )
    suggested_sort: Optional[dict] = Field(
        default=None,
        description="Sort spec for rag_retrieve: {field, direction}.",
    )
    suggested_legs: Optional[List[str]] = Field(
        default=None,
        description="Retrieval legs to run: subset of ['dense', 'sparse', 'exact']. null = let the agent decide.",
    )
    semantic_ratio: Optional[float] = Field(
        default=None,
        description="0.0 = keyword-only (exact+sparse), 1.0 = vector-only (dense), 0.5 = hybrid. "
                    "null = let the agent decide. Used by rag_retrieve to select legs adaptively.",
    )
    reasoning: str = Field(default="", description="Why these filters/sort/legs were suggested.")


class Observation(BaseModel):
    """Result of one tool call appended to the agent state."""

    tool: str
    observation_id: str = Field(default="", description="Id assigned by the dispatcher.")
    arguments: dict = Field(default_factory=dict)
    result: dict = Field(default_factory=dict)
    error: Optional[str] = Field(default=None)
    tokens: int = Field(default=0)


class Artifact(BaseModel):
    """Generated artifact, such as a plot, chart option, or exported file."""

    artifact_type: str = Field(default="file", description="plot, file, chart_option, or table.")
    name: str = Field(default="")
    mime_type: Optional[str] = Field(default=None)
    content_ref: str = Field(default="", description="Path, URL, or opaque reference.")
