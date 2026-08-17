"""
test_settings_phase5.py — Phase 5: agentic + context + memory settings.

Tests:
  1. _tool_call_budget resolves org-overridable AGENT_MAX_RETRIEVALS/CODE_EXEC.
  2. ContextBudget resolves org-overridable context settings.
  3. App-only settings (TOOL_CALL_MODE) are correctly classified.
  4. Org-overridable settings (AGENT_MAX_ITERATIONS, COMPACTION_*, etc.) are correctly classified.
  5. get_setting gracefully falls back to env defaults on DB errors.
"""
import pytest
from unittest.mock import MagicMock, patch
from sqlalchemy.orm import sessionmaker

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

def test_tool_call_budget_falls_back_to_registry(db_session):
    """With no DB rows, _tool_call_budget returns registry defaults."""
    from app.core.settings_registry import get_def
    budget = _tool_call_budget(db_session, None)
    assert budget["rag_retrieve"] == get_def("AGENT_MAX_RETRIEVALS").default
    assert budget["code_execute"] == get_def("AGENT_MAX_CODE_EXEC").default


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
    """With no DB rows, ContextBudget uses registry defaults."""
    from app.services.settings_service import get_setting, clear_cache
    clear_cache()
    budget = ContextBudget(db=db_session, org_id=None)
    assert budget.context_size == get_setting(db_session, "OPENAI_MODEL_CONTEXT_SIZE", None)
    assert budget.reserved_generation == get_setting(db_session, "CONTEXT_RESERVED_GENERATION", None)
    assert budget.tool_budget == get_setting(db_session, "CONTEXT_TOOL_BUDGET", None)
    assert budget.trigger_ratio == get_setting(db_session, "CONTEXT_COMPACTION_TRIGGER_RATIO", None)


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
    """ContextBudget without db falls back to registry defaults via a temporary session."""
    from app.services.settings_service import get_setting, clear_cache
    clear_cache()
    budget = ContextBudget()
    # Resolve expected values the same way ContextBudget does without db
    from app.db.session import SessionLocal
    _db = SessionLocal()
    try:
        expected_ctx = get_setting(_db, "OPENAI_MODEL_CONTEXT_SIZE", None)
        expected_reserved = get_setting(_db, "CONTEXT_RESERVED_GENERATION", None)
    finally:
        _db.close()
    assert budget.context_size == expected_ctx
    assert budget.reserved_generation == expected_reserved


# ---------------------------------------------------------------------------
# 3-4. Setting classification
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


def test_answer_quality_grading_enabled_is_org_overridable():
    from app.core.settings_registry import is_org_overridable
    assert is_org_overridable("ANSWER_QUALITY_GRADING_ENABLED")


def test_processing_timeout_silence_s_is_org_overridable():
    from app.core.settings_registry import is_org_overridable
    assert is_org_overridable("PROCESSING_TIMEOUT_SILENCE_S")


def test_memory_enabled_is_app_only():
    from app.core.settings_registry import is_org_overridable
    assert not is_org_overridable("MEMORY_ENABLED")


# ---------------------------------------------------------------------------
# 6. get_setting graceful fallback
# ---------------------------------------------------------------------------

def test_get_setting_falls_back_on_db_error():
    """get_setting returns env default when DB query fails (e.g. mock session)."""
    mock_db = MagicMock()
    # Make the query raise an exception
    mock_db.query.side_effect = Exception("DB connection failed")

    val = get_setting(mock_db, "RETRIEVAL_TOP_K", None)
    from app.core.settings_registry import get_def
    assert val == get_def("RETRIEVAL_TOP_K").default


def test_get_setting_falls_back_on_mock_session():
    """get_setting works with a MagicMock that doesn't have real query results."""
    mock_db = MagicMock()
    # MagicMock returns MagicMock for .query().filter().first()
    # which will fail during _decode
    val = get_setting(mock_db, "RERANKER_ENABLED", None)
    # Should fall back to registry default
    from app.core.settings_registry import get_def
    assert val == get_def("RERANKER_ENABLED").default
