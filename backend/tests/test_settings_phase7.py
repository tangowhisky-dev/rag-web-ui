"""
test_settings_phase7.py — Phase 7: Hardening + deprecation.

Tests:
  1. Legacy LLM config GET endpoint is marked deprecated.
  2. Legacy LLM config PUT endpoint is marked deprecated.
  3. Settings cache invalidation on upsert.
  4. AGENT_HISTORY_PAIRS is org-overridable.
  5. ADAPTIVE_RETRIEVAL_ENABLED is org-overridable.
  6. Retrieval leg flags resolve per-org in query endpoint path.
"""
import pytest
from sqlalchemy.orm import sessionmaker

from app.core.settings_registry import get_def
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
# 3. Cache invalidation
# ---------------------------------------------------------------------------

def test_cache_invalidation_on_upsert(db_session):
    """Cache is invalidated when a setting is upserted."""
    # First read populates cache with registry default
    val1 = get_setting(db_session, "RETRIEVAL_TOP_K", None)
    assert val1 == get_def("RETRIEVAL_TOP_K").default

    # Upsert a new value (which invalidates cache)
    upsert_app_setting(db_session, "RETRIEVAL_TOP_K", 50)

    # Next read should get the new value, not cached
    val2 = get_setting(db_session, "RETRIEVAL_TOP_K", None)
    assert val2 == 50


def test_cache_invalidation_for_key(db_session):
    """_invalidate_cache clears the cache for a specific key."""
    from app.services.settings_service import _invalidate_cache
    # Populate cache
    get_setting(db_session, "RETRIEVAL_TOP_K", None)
    upsert_app_setting(db_session, "RETRIEVAL_TOP_K", 42)
    clear_cache()

    # Read again to populate cache with new value
    val = get_setting(db_session, "RETRIEVAL_TOP_K", None)
    assert val == 42

    # Invalidate
    _invalidate_cache("RETRIEVAL_TOP_K", None)

    # Upsert another value
    upsert_app_setting(db_session, "RETRIEVAL_TOP_K", 77)
    val = get_setting(db_session, "RETRIEVAL_TOP_K", None)
    assert val == 77


# ---------------------------------------------------------------------------
# 5-6. Setting classification for Phase 7 settings
# ---------------------------------------------------------------------------

def test_agent_history_pairs_is_org_overridable():
    from app.core.settings_registry import is_org_overridable
    assert is_org_overridable("AGENT_HISTORY_PAIRS")


def test_adaptive_retrieval_enabled_is_org_overridable():
    from app.core.settings_registry import is_org_overridable
    assert is_org_overridable("ADAPTIVE_RETRIEVAL_ENABLED")


def test_adaptive_retrieval_threshold_is_org_overridable():
    from app.core.settings_registry import is_org_overridable
    assert is_org_overridable("ADAPTIVE_RETRIEVAL_THRESHOLD")


def test_adaptive_retrieval_reranker_threshold_is_org_overridable():
    from app.core.settings_registry import is_org_overridable
    assert is_org_overridable("ADAPTIVE_RETRIEVAL_RERANKER_THRESHOLD")


def test_openai_api_key_is_org_overridable():
    """OPENAI_API_KEY is org-overridable (encrypted in DB)."""
    from app.core.settings_registry import is_org_overridable
    assert is_org_overridable("OPENAI_API_KEY")


def test_openai_api_key_is_secret():
    """OPENAI_API_KEY is marked as a secret for encryption."""
    from app.core.settings_registry import get_def
    defn = get_def("OPENAI_API_KEY")
    assert defn is not None
    assert defn.secret is True


def test_openai_api_base_is_org_overridable():
    """API base URL is org-overridable (different orgs can use different providers)."""
    from app.core.settings_registry import is_org_overridable
    assert is_org_overridable("OPENAI_API_BASE")


# ---------------------------------------------------------------------------
# 7. Org override resolution for Phase 7 settings
# ---------------------------------------------------------------------------

def test_agent_history_pairs_org_override(db_session):
    """AGENT_HISTORY_PAIRS can be overridden per-org."""
    org = _create_org(db_session)
    upsert_org_setting(db_session, org.id, "AGENT_HISTORY_PAIRS", 15)
    clear_cache()

    val = get_setting(db_session, "AGENT_HISTORY_PAIRS", org.id)
    assert val == 15


def test_adaptive_retrieval_enabled_org_override(db_session):
    """ADAPTIVE_RETRIEVAL_ENABLED can be overridden per-org."""
    org = _create_org(db_session)
    upsert_org_setting(db_session, org.id, "ADAPTIVE_RETRIEVAL_ENABLED", False)
    clear_cache()

    val = get_setting(db_session, "ADAPTIVE_RETRIEVAL_ENABLED", org.id)
    assert val is False
