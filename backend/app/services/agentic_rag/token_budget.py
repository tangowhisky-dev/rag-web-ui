"""Token-accurate context budgeting for the agent loop."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Union
from sqlalchemy.orm import Session

from app.core.config import settings

logger = logging.getLogger(__name__)

try:
    import tiktoken
except Exception:  # pragma: no cover - optional dependency guard
    tiktoken = None  # type: ignore

try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    AutoTokenizer = None  # type: ignore


# Module-level singleton: loaded once, reused across all count_tokens calls.
_tokenizer: Any = None
_tokenizer_loaded: bool = False


def _is_local_path(name: str) -> bool:
    """True if ``name`` points to a local directory (HF repo id or mounted path)."""
    return os.path.isdir(name)


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
            return None


def _hf_tokenizer(model_name: str):
    """Return a HuggingFace tokenizer.

    Uses ``local_files_only=True`` when ``model_name`` is a local directory
    (offline deployment) so AutoTokenizer never tries to hit the network.
    """
    if AutoTokenizer is None:
        return None
    try:
        local_only = _is_local_path(model_name)
        return AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
    except Exception as exc:
        logger.warning("HF tokenizer load failed for '%s': %s", model_name, exc)
        return None


def get_tokenizer(model_name: Optional[str] = None):
    """Load and cache a tokenizer for the given model name.

    Order:
    1. If ``TOKENIZER_MODEL`` is set (explicit override), try HF first.
       This is the offline-deployment path: a local directory mounted in
       the container.
    2. tiktoken (OpenAI-compatible models).
    3. HuggingFace AutoTokenizer (HF Hub repo id or local path).
    4. tiktoken cl100k_base fallback.

    The result is cached as a module-level singleton — the first call pays
    the load cost, subsequent calls reuse it.
    """
    global _tokenizer, _tokenizer_loaded
    if _tokenizer_loaded:
        return _tokenizer

    resolved = model_name or settings.TOKENIZER_MODEL or settings.OPENAI_MODEL
    if not resolved:
        _tokenizer_loaded = True
        return None

    # Explicit TOKENIZER_MODEL override → HF first (offline local path).
    if settings.TOKENIZER_MODEL and not model_name:
        enc = _hf_tokenizer(resolved)
        if enc is not None:
            _tokenizer = enc
            _tokenizer_loaded = True
            return _tokenizer
        # Fall through to tiktoken / cl100k_base below.

    # tiktoken for OpenAI-compatible model names.
    enc = _tiktoken_encoder(resolved)
    if enc is not None:
        _tokenizer = enc
        _tokenizer_loaded = True
        return _tokenizer

    # HF as a secondary fallback (HF Hub repo id that transformers can resolve).
    enc = _hf_tokenizer(resolved)
    if enc is not None:
        _tokenizer = enc
        _tokenizer_loaded = True
        return _tokenizer

    # Last resort: cl100k_base (approximate for non-OpenAI models).
    if tiktoken is not None:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            logger.warning(
                "No exact tokenizer for '%s'; using cl100k_base fallback", resolved
            )
            _tokenizer = enc
            _tokenizer_loaded = True
            return _tokenizer
        except Exception:
            pass

    logger.warning(
        "No tokenizer available for '%s'; count_tokens will use character heuristic",
        resolved,
    )
    _tokenizer_loaded = True
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
        db: Optional[Session] = None,
        org_id: Optional[int] = None,
    ):
        if db is not None:
            from app.services.settings_service import get_setting
            self.context_size = context_size or get_setting(db, "OPENAI_MODEL_CONTEXT_SIZE", org_id)
            self.reserved_generation = reserved_generation or get_setting(db, "CONTEXT_RESERVED_GENERATION", org_id)
            self.tool_budget = tool_budget or get_setting(db, "CONTEXT_TOOL_BUDGET", org_id)
            self.trigger_ratio = get_setting(db, "CONTEXT_COMPACTION_TRIGGER_RATIO", org_id)
        else:
            self.context_size = context_size or settings.OPENAI_MODEL_CONTEXT_SIZE
            self.reserved_generation = reserved_generation or settings.CONTEXT_RESERVED_GENERATION
            self.tool_budget = tool_budget or settings.CONTEXT_TOOL_BUDGET
            self.trigger_ratio = settings.CONTEXT_COMPACTION_TRIGGER_RATIO
        self.available = max(self.context_size - self.reserved_generation - self.tool_budget, 0)
        self.used = 0
        self.compaction_threshold = int(self.available * self.trigger_ratio)

    def add(self, tokens: int) -> None:
        self.used += max(tokens, 0)

    @property
    def remaining(self) -> int:
        return max(self.available - self.used, 0)

    def needs_compaction(self) -> bool:
        return self.used >= self.compaction_threshold
