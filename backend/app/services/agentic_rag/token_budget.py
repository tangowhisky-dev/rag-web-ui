"""Token estimation for the agent loop.

Uses a character-based heuristic (4 chars ≈ 1 token) for pre-request
estimates, calibrated by provider-reported usage data when available.
No model-specific tokenizer files are required — the heuristic adapts
automatically regardless of which LLM an admin selects.

The calibration ratio is updated whenever the provider returns exact
prompt_tokens for a known text. Subsequent estimates for similar text
use the calibrated ratio instead of the default 4 chars/token.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional, Union
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# ─── Heuristic ───────────────────────────────────────────────────────────

CHARS_PER_TOKEN = 4  # industry-standard rough estimate, same as Pi


def estimate_tokens(text: str) -> int:
    """Estimate token count from character length."""
    return max(len(text), 1) // CHARS_PER_TOKEN


# ─── Calibration ─────────────────────────────────────────────────────────

_calibration_lock = threading.Lock()
_calibration_ratio: float = CHARS_PER_TOKEN  # chars per token, adjustable
_calibrated: bool = False


def calibrate(chars: int, actual_tokens: int) -> None:
    """Update the calibration ratio from a provider-reported usage pair.

    Called after an LLM response includes ``prompt_tokens`` for a prompt
    whose character length is known. The ratio is smoothed to avoid
    over-fitting to a single request.
    """
    global _calibration_ratio, _calibrated
    if chars <= 0 or actual_tokens <= 0:
        return
    observed = chars / actual_tokens
    with _calibration_lock:
        if not _calibrated:
            _calibration_ratio = observed
            _calibrated = True
        else:
            # Exponential moving average — smooth out per-request variance.
            _calibration_ratio = 0.7 * _calibration_ratio + 0.3 * observed
        logger.debug("[token_budget] calibration ratio: %.2f chars/token", _calibration_ratio)


def _current_ratio() -> float:
    with _calibration_lock:
        return _calibration_ratio


# ─── Public API ──────────────────────────────────────────────────────────

def count_tokens(text: Union[str, list, dict], model_name: Optional[str] = None) -> int:
    """Estimate token count for ``text``.

    The ``model_name`` argument is accepted for backward compatibility but
    is no longer used — the heuristic + calibration approach is model-agnostic.
    """
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return 0
    return max(len(text) // int(_current_ratio()), 1)


def record_usage(prompt_text: str, prompt_tokens: int) -> None:
    """Calibrate the estimator using provider-reported usage.

    Call this when the LLM response includes ``prompt_tokens`` (or
    ``input_tokens``) and the corresponding prompt text is available.
    """
    if prompt_tokens > 0 and prompt_text:
        calibrate(len(prompt_text), prompt_tokens)


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
            from app.services.settings_service import get_setting
            from app.db.session import SessionLocal
            _db = SessionLocal()
            try:
                self.context_size = context_size or get_setting(_db, "OPENAI_MODEL_CONTEXT_SIZE", None)
                self.reserved_generation = reserved_generation or get_setting(_db, "CONTEXT_RESERVED_GENERATION", None)
                self.tool_budget = tool_budget or get_setting(_db, "CONTEXT_TOOL_BUDGET", None)
                self.trigger_ratio = get_setting(_db, "CONTEXT_COMPACTION_TRIGGER_RATIO", None)
            finally:
                _db.close()
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
