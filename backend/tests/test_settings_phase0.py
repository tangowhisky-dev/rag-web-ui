"""
test_settings_phase0.py — Phase 0: RUNTIME_SETTINGS_ENABLED + registry settings.

Verifies:
  1. REASONING_MODEL is registered in the settings registry (no longer on Settings).
  2. RUNTIME_SETTINGS_ENABLED feature flag exists and defaults to True.
"""
import os

from app.core.config import Settings


def test_reasoning_model_declared():
    """REASONING_MODEL must be registered in the settings registry."""
    from app.core.settings_registry import get_def
    assert get_def("REASONING_MODEL") is not None


def test_runtime_settings_enabled_default_true():
    """RUNTIME_SETTINGS_ENABLED must default to True."""
    s = Settings()
    assert s.RUNTIME_SETTINGS_ENABLED is True


def test_runtime_settings_enabled_can_be_disabled():
    """RUNTIME_SETTINGS_ENABLED can be set to False."""
    s = Settings(RUNTIME_SETTINGS_ENABLED=False)
    assert s.RUNTIME_SETTINGS_ENABLED is False
