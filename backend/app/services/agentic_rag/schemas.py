"""Pydantic models for LangGraph structured output."""

from __future__ import annotations

from typing import List, Optional, Union

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
    clarification_questions: List[str] = Field(
        default_factory=list,
        description="List of 2-4 specific clarification questions for the user to answer. Only populated when is_clear=False. Questions should be concrete and answerable (e.g., 'Which domain: computer science, biology, or physics?').",
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


class LastAnswerObject(BaseModel):
    """Structured representation of the assistant's last answer."""

    summary: str = Field(description="2-3 sentence summary of the answer.")
    key_points: List[str] = Field(default_factory=list, description="Bullet points.")
    data: Optional[List[DataPoint]] = Field(default=None, description="Numbers and statistics mentioned.")
    citations: List[CitationRef] = Field(default_factory=list, description="Chunk refs used.")
    chart_option: Optional[dict] = Field(default=None, description="ECharts option JSON, if any.")
    followups: List[str] = Field(default_factory=list, description="Suggested follow-up questions.")


class Subtask(BaseModel):
    """One step in the agent's plan."""

    id: str = Field(description="Unique subtask id, e.g. 'a', 'b'.")
    description: str = Field(description="What the subtask should accomplish.")
    tool_hint: str = Field(default="any", description="Preferred tool name or 'any'.")
    depends_on: List[str] = Field(default_factory=list, description="Subtask ids that must complete first.")
    expected_output: str = Field(default="", description="What the agent expects to observe.")


class Plan(BaseModel):
    """Plan produced by the planner for one user turn."""

    intent: str = Field(
        default="rag",
        description="One of: rag, file_action, previous_answer_action, computation, chart, conversation, mixed.",
    )
    subtasks: List[Subtask] = Field(default_factory=list, description="Subtasks to execute.")
    needs_clarification: bool = Field(default=False, description="True if the user must clarify.")
    clarification_question: Optional[str] = Field(default=None, description="Question to ask the user.")


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
