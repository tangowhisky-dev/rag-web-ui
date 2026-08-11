"""
test_settings_phase3.py — Phase 3: LLM resolvers wired to settings service.

Tests:
  1. get_org_llm falls back to env defaults when no DB rows.
  2. get_org_llm uses app-level settings when set.
  3. get_org_llm uses org-level overrides when set.
  4. get_org_llm role="query" uses query_model.
  5. get_org_llm role="reasoning" uses reasoning_model.
  6. get_effective_llm_config falls back to env defaults.
  7. get_effective_llm_config uses org overrides.
  8. API key always comes from .env (never from DB).
"""
import pytest
from sqlalchemy.orm import sessionmaker

from app.core.config import settings as env_settings
from app.models.base import Base
from app.models.organisation import Organisation
import app.models.user  # noqa
import app.models.knowledge  # noqa
import app.models.chat  # noqa
import app.models.datastore  # noqa
import app.models.setting  # noqa
import app.models.org_llm_config  # noqa

from app.services.settings_service import (
    upsert_app_setting, upsert_org_setting, clear_cache,
)
from app.services.agentic_rag.llm_factory import get_org_llm, build_chat_llm
from app.services.chat.chat_service import get_effective_llm_config


@pytest.fixture()
def db_session():
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool

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
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_cache()
    yield
    clear_cache()


def _create_org(db, name="TestOrg"):
    org = Organisation(name=name, parent_id=None, path="/1")
    db.add(org)
    db.commit()
    db.refresh(org)
    return org


# ---------------------------------------------------------------------------
# get_org_llm
# ---------------------------------------------------------------------------

def test_get_org_llm_falls_back_to_env(db_session):
    """With no DB rows, get_org_llm returns .env/config.py defaults."""
    cfg = get_org_llm(None, db_session, role="chat")
    assert cfg["api_base"] == env_settings.OPENAI_API_BASE
    assert cfg["model_name"] == env_settings.OPENAI_MODEL
    assert cfg["api_key"] == env_settings.OPENAI_API_KEY


def test_get_org_llm_uses_app_settings(db_session):
    """App-level DB settings override .env defaults."""
    upsert_app_setting(db_session, "OPENAI_API_BASE", "https://app-level.example.com")
    upsert_app_setting(db_session, "OPENAI_MODEL", "app-model")
    clear_cache()

    cfg = get_org_llm(None, db_session, role="chat")
    assert cfg["api_base"] == "https://app-level.example.com"
    assert cfg["model_name"] == "app-model"


def test_get_org_llm_uses_org_override(db_session):
    """Org-level overrides win over app-level and .env."""
    org = _create_org(db_session)
    upsert_app_setting(db_session, "OPENAI_API_BASE", "https://app-level.example.com")
    upsert_app_setting(db_session, "OPENAI_MODEL", "app-model")
    upsert_org_setting(db_session, org.id, "OPENAI_API_BASE", "https://org-level.example.com")
    upsert_org_setting(db_session, org.id, "OPENAI_MODEL", "org-model")
    clear_cache()

    cfg = get_org_llm(org.id, db_session, role="chat")
    assert cfg["api_base"] == "https://org-level.example.com"
    assert cfg["model_name"] == "org-model"


def test_get_org_llm_query_role(db_session):
    """role='query' uses QUERY_MODEL."""
    org = _create_org(db_session)
    upsert_org_setting(db_session, org.id, "OPENAI_MODEL", "chat-model")
    upsert_org_setting(db_session, org.id, "QUERY_MODEL", "query-model")
    clear_cache()

    cfg = get_org_llm(org.id, db_session, role="query")
    assert cfg["model_name"] == "query-model"


def test_get_org_llm_reasoning_role(db_session):
    """role='reasoning' uses REASONING_MODEL, falling back to OPENAI_MODEL."""
    org = _create_org(db_session)
    upsert_org_setting(db_session, org.id, "OPENAI_MODEL", "chat-model")
    upsert_org_setting(db_session, org.id, "REASONING_MODEL", "reasoning-model")
    clear_cache()

    cfg = get_org_llm(org.id, db_session, role="reasoning")
    assert cfg["model_name"] == "reasoning-model"


def test_get_org_llm_reasoning_falls_back_to_chat(db_session):
    """When REASONING_MODEL is unset/None, reasoning role falls back to OPENAI_MODEL."""
    org = _create_org(db_session)
    # Set REASONING_MODEL to None at app level to override any .env value
    upsert_app_setting(db_session, "REASONING_MODEL", None)
    upsert_org_setting(db_session, org.id, "OPENAI_MODEL", "chat-model")
    clear_cache()

    cfg = get_org_llm(org.id, db_session, role="reasoning")
    assert cfg["model_name"] == "chat-model"


def test_get_org_llm_api_key_always_from_env(db_session):
    """API key must always come from .env, never from DB."""
    org = _create_org(db_session)
    upsert_org_setting(db_session, org.id, "OPENAI_MODEL", "org-model")
    clear_cache()

    cfg = get_org_llm(org.id, db_session, role="chat")
    assert cfg["api_key"] == env_settings.OPENAI_API_KEY


# ---------------------------------------------------------------------------
# get_effective_llm_config
# ---------------------------------------------------------------------------

def test_get_effective_llm_config_fallback(db_session):
    """When org_id is None, all values fall back to settings defaults."""
    cfg = get_effective_llm_config(None, db_session)
    assert cfg["api_base"] == env_settings.OPENAI_API_BASE
    assert cfg["model_name"] == env_settings.OPENAI_MODEL
    # query_model falls back to model_name when unset
    assert cfg["query_model"] == (env_settings.QUERY_MODEL or env_settings.OPENAI_MODEL)


def test_get_effective_llm_config_org_override(db_session):
    """Org overrides are reflected in the config."""
    org = _create_org(db_session)
    upsert_org_setting(db_session, org.id, "OPENAI_API_BASE", "https://org.example.com")
    upsert_org_setting(db_session, org.id, "OPENAI_MODEL", "org-model")
    upsert_org_setting(db_session, org.id, "QUERY_MODEL", "org-query-model")
    clear_cache()

    cfg = get_effective_llm_config(org.id, db_session)
    assert cfg["api_base"] == "https://org.example.com"
    assert cfg["model_name"] == "org-model"
    assert cfg["query_model"] == "org-query-model"


def test_get_effective_llm_config_query_model_falls_back(db_session):
    """When QUERY_MODEL is unset/None, query_model falls back to model_name."""
    org = _create_org(db_session)
    # Set QUERY_MODEL to None at app level to override any .env value
    upsert_app_setting(db_session, "QUERY_MODEL", None)
    upsert_org_setting(db_session, org.id, "OPENAI_MODEL", "org-model")
    clear_cache()

    cfg = get_effective_llm_config(org.id, db_session)
    assert cfg["query_model"] == "org-model"
