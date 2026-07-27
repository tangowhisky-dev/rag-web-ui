"""Token-accurate context budgeting for the agent loop."""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

from app.core.config import settings

logger = logging.getLogger(__name__)

try:
    import tiktoken
except Exception:  # pragma: no cover - optional dependency guard
    tiktoken = None  # type: ignore

try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    AutoTokenizer = None


def _tiktoken_encoder(model_name: str):
    """Return a tiktoken encoder, falling back to cl100k_base."""
    if tiktoken is None:
        return None
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        try:
            return tiktoken.get_encoding(model_name)
        except Exception:
            logger.warning("Tokenizer '%s' not found; using cl100k_base fallback", model_name)
            try:
                return tiktoken.get_encoding("cl100k_base")
            except Exception:
                return None


def _hf_tokenizer(model_name: str):
    """Return a HuggingFace tokenizer if files are available locally."""
    if AutoTokenizer is None:
        return None
    try:
        return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        return None


def get_tokenizer(model_name: Optional[str] = None):
    """Load a tokenizer for the given model name.

    Order:
    1. tiktoken (OpenAI-compatible models)
    2. transformers AutoTokenizer (local HuggingFace models)
    3. tiktoken cl100k_base fallback
    """
    model_name = model_name or settings.TOKENIZER_MODEL or settings.OPENAI_MODEL
    if not model_name:
        return None

    enc = _tiktoken_encoder(model_name)
    if enc:
        return enc

    enc = _hf_tokenizer(model_name)
    if enc:
        return enc

    logger.warning("No tokenizer available for '%s'; count_tokens will use character heuristic", model_name)
    return None


def count_tokens(text: Union[str, list, dict], model_name: Optional[str] = None) -> int:
    """Count tokens in ``text`` using the best available tokenizer."""
    if not isinstance(text, str):
        text = str(text)
    tokenizer = get_tokenizer(model_name)
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception:
            pass
    # Final fallback: rough character heuristic (1 token ≈ 4 characters).
    return len(text) // 4


class ContextBudget:
    """Track token spend against the LLM context window."""

    def __init__(
        self,
        context_size: Optional[int] = None,
        reserved_generation: Optional[int] = None,
        tool_budget: Optional[int] = None,
    ):
        self.context_size = context_size or settings.OPENAI_MODEL_CONTEXT_SIZE
        self.reserved_generation = reserved_generation or settings.CONTEXT_RESERVED_GENERATION
        self.tool_budget = tool_budget or settings.CONTEXT_TOOL_BUDGET
        self.available = max(self.context_size - self.reserved_generation - self.tool_budget, 0)
        self.used = 0
        self.trigger_ratio = settings.CONTEXT_COMPACTION_TRIGGER_RATIO
        self.compaction_threshold = int(self.available * self.trigger_ratio)

    def add(self, tokens: int) -> None:
        self.used += max(tokens, 0)

    @property
    def remaining(self) -> int:
        return max(self.available - self.used, 0)

    def needs_compaction(self) -> bool:
        return self.used >= self.compaction_threshold
