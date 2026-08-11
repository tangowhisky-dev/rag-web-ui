"""
test_settings_phase0.py — Phase 0: config.py / registry separation.

Verifies:
  1. REASONING_MODEL is registered in the settings registry (not on Settings).
  2. Registry-managed keys are NOT on the Settings class.
  3. Infrastructure settings (DB, Redis, Qdrant, etc.) ARE on the Settings class.
"""
from app.core.config import Settings


def test_reasoning_model_declared():
    """REASONING_MODEL must be registered in the settings registry."""
    from app.core.settings_registry import get_def
    assert get_def("REASONING_MODEL") is not None


def test_registry_keys_not_on_settings():
    """Registry-managed keys must not be on the Settings class."""
    from app.core.settings_registry import REGISTRY
    config_fields = set(Settings.model_fields.keys())
    for defn in REGISTRY:
        assert defn.key not in config_fields, \
            f"{defn.key} is in both the registry and Settings — should be one or the other"


def test_infrastructure_settings_on_settings():
    """Core infrastructure settings must remain on the Settings class."""
    required = {
        "MYSQL_SERVER", "MYSQL_PORT", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DATABASE",
        "SECRET_KEY", "ALGORITHM", "ACCESS_TOKEN_EXPIRE_MINUTES",
        "QDRANT_HOST", "QDRANT_PORT",
        "REDIS_URL", "REDIS_HOST", "REDIS_PORT",
        "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD",
        "UPLOAD_DIR", "WATCH_DIR",
        "SPLADE_MODEL", "FASTEMBED_CACHE_DIR",
        "RERANKER_MODEL", "RERANKER_CACHE_DIR",
        "TOKENIZER_MODEL",
        "SANDBOX_BACKEND",
        "LOG_LEVEL",
    }
    config_fields = set(Settings.model_fields.keys())
    missing = required - config_fields
    assert not missing, f"Missing infrastructure settings on Settings: {missing}"
