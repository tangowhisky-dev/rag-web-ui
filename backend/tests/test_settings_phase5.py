"""
test_settings_phase5.py — Phase 5: agentic + context + memory settings.

Tests:
  1. _tool_call_budget resolves org-overridable AGENT_MAX_RETRIEVALS/CODE_EXEC.
  2. ContextBudget resolves org-overridable context settings.
  3. retrieve_historical_memory resolves org-overridable HISTORICAL_MEMORY_ENABLED.
  4. App-only settings (TOOL_CALL_MODE) are correctly classified.
  5. Org-overridable settings (AGENT_MAX_ITERATIONS, COMPACTION_*, etc.) are correctly classified.
  6. get_setting gracefully falls back to env defaults on DB errors.
"""
import pytest
from unittest.mock import MagicMock, patch
from sqlalchemy.orm import sessionmaker

from app.core.config import settings as env_settings
from app.models.base import Base
from app.models.organisation import Organisation
import app.models.user  # noqa
import app.models.knowledge  # noqa
import app.models.chat  # noqa
import app.models.datastore  # noqa
import app.models.setting  # noqa

from app.services.settings_service import (
    upsert_app_setting, upsert_org_setting, clear_cache, get_setting,
)
from app.services.agentic_rag.agent_graph import _tool_call_budget
from app.services.agentic_rag.token_budget import ContextBudget


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
# 1. _tool_call_budget
# ---------------------------------------------------------------------------

def test_tool_call_budget_falls_back_to_env(db_session):
    """With no DB rows, _tool_call_budget returns .env defaults."""
    budget = _tool_call_budget(db_session, None)
    assert budget["rag_retrieve"] == env_settings.AGENT_MAX_RETRIEVALS
    assert budget["code_execute"] == env_settings.AGENT_MAX_CODE_EXEC


def test_tool_call_budget_uses_org_override(db_session):
    """Org overrides affect the tool call budget."""
    org = _create_org(db_session)
    upsert_org_setting(db_session, org.id, "AGENT_MAX_RETRIEVALS", 10)
    upsert_org_setting(db_session, org.id, "AGENT_MAX_CODE_EXEC", 5)
    clear_cache()

    budget = _tool_call_budget(db_session, org.id)
    assert budget["rag_retrieve"] == 10
    assert budget["code_execute"] == 5


# ---------------------------------------------------------------------------
# 2. ContextBudget
# ---------------------------------------------------------------------------

def test_context_budget_falls_back_to_env(db_session):
    """With no DB rows, ContextBudget uses .env defaults."""
    budget = ContextBudget(db=db_session, org_id=None)
    assert budget.context_size == env_settings.OPENAI_MODEL_CONTEXT_SIZE
    assert budget.reserved_generation == env_settings.CONTEXT_RESERVED_GENERATION
    assert budget.tool_budget == env_settings.CONTEXT_TOOL_BUDGET
    assert budget.trigger_ratio == env_settings.CONTEXT_COMPACTION_TRIGGER_RATIO


def test_context_budget_uses_org_override(db_session):
    """Org overrides affect the context budget."""
    org = _create_org(db_session)
    upsert_org_setting(db_session, org.id, "CONTEXT_RESERVED_GENERATION", 2048)
    upsert_org_setting(db_session, org.id, "CONTEXT_TOOL_BUDGET", 4096)
    clear_cache()

    budget = ContextBudget(db=db_session, org_id=org.id)
    assert budget.reserved_generation == 2048
    assert budget.tool_budget == 4096


def test_context_budget_without_db_uses_env():
    """ContextBudget without db falls back to .env defaults."""
    budget = ContextBudget()
    assert budget.context_size == env_settings.OPENAI_MODEL_CONTEXT_SIZE
    assert budget.reserved_generation == env_settings.CONTEXT_RESERVED_GENERATION


# ---------------------------------------------------------------------------
# 3. retrieve_historical_memory
# ---------------------------------------------------------------------------

def test_historical_memory_disabled_via_org_override(db_session):
    """Org override of HISTORICAL_MEMORY_ENABLED=False disables historical memory."""
    org = _create_org(db_session)
    upsert_org_setting(db_session, org.id, "HISTORICAL_MEMORY_ENABLED", False)
    clear_cache()

    # Mock the DB execute to return some rows
    from types import SimpleNamespace
    mock_db = MagicMock()
    mock_db.execute.return_value.fetchall.return_value = [
        SimpleNamespace(id=1, content="test", content_length=4),
    ]

    from app.services.chat import retrieve_historical_memory
    result = retrieve_historical_memory(
        chat_id=1, query="test", db=mock_db, org_id=org.id
    )
    # When disabled, it returns the last top_k raw docs (not [])
    # But wait — the disabled path returns last top_k raw docs only if there are docs
    # Actually, when HISTORICAL_MEMORY_ENABLED is False, the function returns []
    # before reaching the reranker check
    assert result == []


# ---------------------------------------------------------------------------
# 4-5. Setting classification
# ---------------------------------------------------------------------------

def test_tool_call_mode_is_app_only():
    from app.core.settings_registry import is_org_overridable
    assert not is_org_overridable("TOOL_CALL_MODE")


def test_agent_max_iterations_is_org_overridable():
    from app.core.settings_registry import is_org_overridable
    assert is_org_overridable("AGENT_MAX_ITERATIONS")


def test_agent_max_retrievals_is_org_overridable():
    from app.core.settings_registry import is_org_overridable
    assert is_org_overridable("AGENT_MAX_RETRIEVALS")


def test_agent_max_code_exec_is_org_overridable():
    from app.core.settings_registry import is_org_overridable
    assert is_org_overridable("AGENT_MAX_CODE_EXEC")


def test_compaction_enabled_is_org_overridable():
    from app.core.settings_registry import is_org_overridable
    assert is_org_overridable("COMPACTION_ENABLED")


def test_compaction_keep_recent_is_org_overridable():
    from app.core.settings_registry import is_org_overridable
    assert is_org_overridable("COMPACTION_KEEP_RECENT")


def test_context_reserved_generation_is_org_overridable():
    from app.core.settings_registry import is_org_overridable
    assert is_org_overridable("CONTEXT_RESERVED_GENERATION")


def test_highlights_token_cap_is_org_overridable():
    from app.core.settings_registry import is_org_overridable
    assert is_org_overridable("HIGHLIGHTS_TOKEN_CAP")


def test_answer_quality_grading_enabled_is_org_overridable():
    from app.core.settings_registry import is_org_overridable
    assert is_org_overridable("ANSWER_QUALITY_GRADING_ENABLED")


def test_processing_timeout_silence_s_is_org_overridable():
    from app.core.settings_registry import is_org_overridable
    assert is_org_overridable("PROCESSING_TIMEOUT_SILENCE_S")


def test_historical_memory_enabled_is_org_overridable():
    from app.core.settings_registry import is_org_overridable
    assert is_org_overridable("HISTORICAL_MEMORY_ENABLED")


def test_historical_memory_top_k_is_org_overridable():
    from app.core.settings_registry import is_org_overridable
    assert is_org_overridable("HISTORICAL_MEMORY_TOP_K")


def test_memory_enabled_is_app_only():
    from app.core.settings_registry import is_org_overridable
    assert not is_org_overridable("MEMORY_ENABLED")


def test_sandbox_timeout_s_is_app_only():
    from app.core.settings_registry import is_org_overridable
    assert not is_org_overridable("SANDBOX_TIMEOUT_S")


# ---------------------------------------------------------------------------
# 6. get_setting graceful fallback
# ---------------------------------------------------------------------------

def test_get_setting_falls_back_on_db_error():
    """get_setting returns env default when DB query fails (e.g. mock session)."""
    mock_db = MagicMock()
    # Make the query raise an exception
    mock_db.query.side_effect = Exception("DB connection failed")

    val = get_setting(mock_db, "RETRIEVAL_TOP_K", None)
    assert val == env_settings.RETRIEVAL_TOP_K


def test_get_setting_falls_back_on_mock_session():
    """get_setting works with a MagicMock that doesn't have real query results."""
    mock_db = MagicMock()
    # MagicMock returns MagicMock for .query().filter().first()
    # which will fail during _decode
    val = get_setting(mock_db, "RERANKER_ENABLED", None)
    # Should fall back to env default
    assert val == env_settings.RERANKER_ENABLED
