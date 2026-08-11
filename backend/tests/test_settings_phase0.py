"""
test_settings_phase0.py — Phase 0: REASONING_MODEL + RUNTIME_SETTINGS_ENABLED.

Verifies:
  1. REASONING_MODEL is declared on Settings and reads from env.
  2. effective_reasoning_model falls back to OPENAI_MODEL when REASONING_MODEL is unset.
  3. effective_reasoning_model returns REASONING_MODEL when set.
  4. RUNTIME_SETTINGS_ENABLED feature flag exists and defaults to True.
"""
import os

from app.core.config import Settings


def test_reasoning_model_declared():
    """REASONING_MODEL must be a declared field on Settings."""
    s = Settings()
    assert hasattr(s, "REASONING_MODEL")


def test_reasoning_model_falls_back_to_openai_model():
    """When REASONING_MODEL is unset, effective_reasoning_model returns OPENAI_MODEL."""
    s = Settings(REASONING_MODEL=None, OPENAI_MODEL="gpt-4o")
    assert s.effective_reasoning_model == "gpt-4o"


def test_reasoning_model_uses_explicit_value():
    """When REASONING_MODEL is set, effective_reasoning_model returns it."""
    s = Settings(REASONING_MODEL="o3-mini", OPENAI_MODEL="gpt-4o")
    assert s.effective_reasoning_model == "o3-mini"


def test_runtime_settings_enabled_default_true():
    """RUNTIME_SETTINGS_ENABLED must default to True."""
    s = Settings()
    assert s.RUNTIME_SETTINGS_ENABLED is True


def test_runtime_settings_enabled_can_be_disabled():
    """RUNTIME_SETTINGS_ENABLED can be set to False."""
    s = Settings(RUNTIME_SETTINGS_ENABLED=False)
    assert s.RUNTIME_SETTINGS_ENABLED is False
