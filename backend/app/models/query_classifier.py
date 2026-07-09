from enum import Enum
from pydantic import BaseModel


class QueryType(str, Enum):
    """4-way query classification for adaptive retrieval routing."""
    FACTUAL = "FACTUAL"
    ENTITY_CENTRIC = "ENTITY_CENTRIC"
    MULTI_PART = "MULTI_PART"
    AMBIGUOUS = "AMBIGUOUS"


class QueryClassification(BaseModel):
    """Result of LLM-based query classification."""
    type: QueryType
    confidence: float
    latency_ms: float
    fallback: bool = False
