"""
test_settings_redesign.py — Tests for the per-role credentials redesign.

Verifies:
  1. Dead settings are removed from registry and config.
  2. Role-specific API keys and base URLs are registered.
  3. Embedding settings are app-only (not org-overridable).
  4. Encryption/decryption works for secret settings.
  5. Masked values are returned in API metadata.
  6. Masked values sent on write are ignored (no-op).
  7. Role-aware LLM resolution falls back correctly.
  8. OrgLLMConfig model is removed.
  9. Computed properties are removed from config.
"""
import pytest
from unittest.mock import patch, MagicMock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from app.models.base import Base


@pytest.fixture()
def db_session():
    """Create an in-memory SQLite session for testing."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


# ── 1. Dead settings removed ─────────────────────────────────────────────────

def test_retrieval_min_rrf_score_not_in_registry():
    """RETRIEVAL_MIN_RRF_SCORE was removed from the registry."""
    from app.core.settings_registry import get_def
    assert get_def("RETRIEVAL_MIN_RRF_SCORE") is None


def test_max_tool_iterations_not_in_registry():
    """MAX_TOOL_ITERATIONS was removed from the registry."""
    from app.core.settings_registry import get_def
    assert get_def("MAX_TOOL_ITERATIONS") is None


def test_retrieval_min_rrf_score_not_in_config():
    """RETRIEVAL_MIN_RRF_SCORE is not a Settings field."""
    from app.core.config import Settings
    s = Settings()
    assert not hasattr(s, "RETRIEVAL_MIN_RRF_SCORE")


def test_max_tool_iterations_not_in_config():
    """MAX_TOOL_ITERATIONS is not a Settings field."""
    from app.core.config import Settings
    s = Settings()
    assert not hasattr(s, "MAX_TOOL_ITERATIONS")


# ── 2. Role-specific settings registered ─────────────────────────────────────

@pytest.mark.parametrize("key", [
    "OPENAI_API_KEY",
    "QUERY_API_KEY",
    "REASONING_API_KEY",
    "VISION_API_KEY",
    "GRAPHRAG_API_KEY",
    "EMBEDDING_API_KEY",
])
def test_api_key_settings_registered(key):
    """Each role-specific API key is registered."""
    from app.core.settings_registry import get_def
    defn = get_def(key)
    assert defn is not None, f"{key} not registered"
    assert defn.secret is True, f"{key} should be secret"


@pytest.mark.parametrize("key", [
    "OPENAI_API_BASE",
    "QUERY_API_BASE",
    "REASONING_API_BASE",
    "OPENAI_VISION_API_BASE",
    "GRAPHRAG_API_BASE",
    "EMBEDDING_API_BASE",
])
def test_base_url_settings_registered(key):
    """Each role-specific base URL is registered."""
    from app.core.settings_registry import get_def
    defn = get_def(key)
    assert defn is not None, f"{key} not registered"


# ── 3. Embedding settings are app-only ───────────────────────────────────────

@pytest.mark.parametrize("key", [
    "DENSE_EMBEDDINGS_MODEL",
    "DENSE_EMBEDDING_DIM",
    "EMBEDDING_API_KEY",
    "EMBEDDING_API_BASE",
    "MEMORY_EMBEDDING_MODEL",
])
def test_embedding_settings_not_org_overridable(key):
    """Embedding settings must not be org-overridable."""
    from app.core.settings_registry import is_org_overridable
    assert not is_org_overridable(key), f"{key} should not be org-overridable"


def test_openai_api_key_is_org_overridable():
    """OPENAI_API_KEY is org-overridable (encrypted in DB)."""
    from app.core.settings_registry import is_org_overridable
    assert is_org_overridable("OPENAI_API_KEY")


# ── 4. Encryption/decryption ─────────────────────────────────────────────────

def test_encrypt_decrypt_roundtrip():
    """Encrypting then decrypting returns the original value."""
    from app.services.settings_service import _encrypt, _decrypt
    original = "sk-test-key-12345"
    encrypted = _encrypt(original)
    assert encrypted.startswith("enc:")
    assert _decrypt(encrypted) == original


def test_decrypt_plaintext_fallback():
    """Decrypting a plaintext value (no enc: prefix) returns it as-is."""
    from app.services.settings_service import _decrypt
    assert _decrypt("plaintext-value") == "plaintext-value"


def test_encode_encrypts_secret():
    """_encode encrypts secret settings."""
    from app.services.settings_service import _encode
    from app.core.settings_registry import get_def
    defn = get_def("OPENAI_API_KEY")
    encoded = _encode("sk-secret", defn)
    assert encoded.startswith("enc:")


def test_decode_decrypts_secret():
    """_decode decrypts secret settings."""
    from app.services.settings_service import _encode, _decode
    from app.core.settings_registry import get_def
    defn = get_def("OPENAI_API_KEY")
    encoded = _encode("sk-secret", defn)
    decoded = _decode(encoded, defn)
    assert decoded == "sk-secret"


# ── 5. Masked values in API responses ────────────────────────────────────────

def test_mask_secret_short():
    """Short secrets are fully masked."""
    from app.services.settings_service import _mask_secret
    assert _mask_secret("abc") == "••••"


def test_mask_secret_long():
    """Long secrets show last 4 chars."""
    from app.services.settings_service import _mask_secret
    assert _mask_secret("sk-1234567890") == "••••7890"


def test_mask_secret_empty():
    """Empty values are masked."""
    from app.services.settings_service import _mask_secret
    assert _mask_secret("") == "••••"
    assert _mask_secret(None) == "••••"


# ── 6. Masked values on write are ignored ────────────────────────────────────

def test_upsert_app_setting_ignores_masked_secret(db_session):
    """Sending a masked secret value is a no-op."""
    from app.services.settings_service import upsert_app_setting, get_setting, clear_cache
    clear_cache()
    # First set a real value
    upsert_app_setting(db_session, "OPENAI_API_KEY", "sk-real-key-12345")
    assert get_setting(db_session, "OPENAI_API_KEY", None) == "sk-real-key-12345"

    # Now send a masked value — should be ignored
    upsert_app_setting(db_session, "OPENAI_API_KEY", "••••1234")
    assert get_setting(db_session, "OPENAI_API_KEY", None) == "sk-real-key-12345"


# ── 7. Role-aware LLM resolution ─────────────────────────────────────────────

def test_get_org_llm_chat_role(db_session):
    """get_org_llm with chat role returns OPENAI_MODEL and OPENAI_API_BASE."""
    from app.services.agentic_rag.llm_factory import get_org_llm
    cfg = get_org_llm(None, db_session, role="chat")
    assert "api_base" in cfg
    assert "model_name" in cfg
    assert "api_key" in cfg


def test_get_org_llm_query_role_falls_back(db_session):
    """get_org_llm with query role falls back to OPENAI_MODEL when QUERY_MODEL unset."""
    from app.services.agentic_rag.llm_factory import get_org_llm
    from app.services.settings_service import get_setting, clear_cache
    clear_cache()
    cfg = get_org_llm(None, db_session, role="query")
    # query_model falls back to OPENAI_MODEL (registry default)
    expected = get_setting(db_session, "OPENAI_MODEL", None)
    assert cfg["model_name"] == expected


def test_get_org_llm_role_specific_key_falls_back(db_session):
    """Role-specific key falls back to OPENAI_API_KEY."""
    from app.services.agentic_rag.llm_factory import get_org_llm
    from app.services.settings_service import get_setting, clear_cache
    clear_cache()
    cfg = get_org_llm(None, db_session, role="vision")
    # VISION_API_KEY falls back to OPENAI_API_KEY (registry default)
    expected = get_setting(db_session, "OPENAI_API_KEY", None)
    assert cfg["api_key"] == expected


def test_get_org_llm_graph_role_uses_graphrag_model(db_session):
    """get_org_llm with graph role uses GRAPHRAG_LLM."""
    from app.services.agentic_rag.llm_factory import get_org_llm
    from app.services.settings_service import get_setting, clear_cache
    clear_cache()
    cfg = get_org_llm(None, db_session, role="graph")
    # GRAPHRAG_LLM is app-only; falls back to OPENAI_MODEL when unset
    expected = get_setting(db_session, "GRAPHRAG_LLM", None) or get_setting(db_session, "OPENAI_MODEL", None)
    assert cfg["model_name"] == expected


# ── 8. OrgLLMConfig removed ──────────────────────────────────────────────────

def test_org_llm_config_model_file_deleted():
    """The org_llm_config model file no longer exists."""
    import os
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "app", "models", "org_llm_config.py"
    )
    assert not os.path.exists(model_path)


def test_org_llm_config_not_importable():
    """OrgLLMConfig cannot be imported."""
    try:
        from app.models.org_llm_config import OrgLLMConfig  # noqa
        assert False, "Should not be importable"
    except ImportError:
        pass


def test_org_llm_config_not_in_models_init():
    """OrgLLMConfig is not exported from models.__init__."""
    from app import models
    assert not hasattr(models, "OrgLLMConfig")


def test_legacy_llm_config_endpoints_removed():
    """The legacy /orgs/{id}/llm-config endpoints no longer exist."""
    from app.api.api_v1.admin import org_router
    paths = [route.path for route in org_router.routes]
    assert not any("llm-config" in p for p in paths)


# ── 9. Computed properties removed ───────────────────────────────────────────

@pytest.mark.parametrize("prop", [
    "effective_query_model",
    "effective_reasoning_model",
    "effective_vision_api_base",
    "graphrag_model",
])
def test_computed_property_removed(prop):
    """Computed properties are no longer on Settings."""
    from app.core.config import Settings
    s = Settings()
    assert not hasattr(s, prop), f"Settings still has {prop}"


# ── 10. Setting schema includes secret flag ──────────────────────────────────

def test_setting_item_schema_has_secret():
    """SettingItem schema includes the secret field."""
    from app.schemas.setting import SettingItem
    fields = SettingItem.model_fields
    assert "secret" in fields
    assert "is_set" in fields


def test_setting_schema_item_has_secret():
    """SettingSchemaItem schema includes the secret field."""
    from app.schemas.setting import SettingSchemaItem
    fields = SettingSchemaItem.model_fields
    assert "secret" in fields
