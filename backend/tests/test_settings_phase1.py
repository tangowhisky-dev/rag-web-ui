"""
test_settings_phase1.py — Phase 1: registry, model, service, resolution, CRUD.

Tests:
  1. Registry completeness: all keys present, no duplicates.
  2. 2-tier precedence: org override → app value → registry default.
  3. Reset semantics: delete org → app value; delete app → registry default.
  4. Validation: type coercion, min/max, unknown keys.
  5. Scope enforcement: org CRUD rejects app-only keys.
  6. OrgSettings accessor: attribute access + computed properties.
  7. Cache invalidation on upsert/reset.
"""
import json
import pytest
from sqlalchemy.orm import sessionmaker

from app.core.settings_registry import (
    REGISTRY, REGISTRY_BY_KEY, ORG_OVERRIDABLE_KEYS, APP_ONLY_KEYS,
    get_def, is_org_overridable, all_keys,
)
from app.models.setting import Setting
from app.models.base import Base
from app.models.organisation import Organisation
import app.models.user  # noqa
import app.models.knowledge  # noqa
import app.models.chat  # noqa
import app.models.datastore  # noqa
import app.models.setting  # noqa

from app.services import settings_service
from app.services.settings_service import (
    get_setting, get_org_settings, OrgSettings,
    upsert_app_setting, upsert_org_setting,
    reset_app_setting, reset_org_setting, reset_all_org_settings,
    validate_value, clear_cache,
    get_all_app_settings_with_meta, get_all_org_settings_with_meta,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_session():
    """Create an in-memory SQLite session for testing."""
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
    """Clear the settings cache before each test."""
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
# 1. Registry completeness
# ---------------------------------------------------------------------------

def test_registry_no_duplicates():
    """Every key in the registry must be unique."""
    keys = [d.key for d in REGISTRY]
    assert len(keys) == len(set(keys)), f"Duplicate keys: {[k for k in keys if keys.count(k) > 1]}"


def test_registry_lookup():
    """REGISTRY_BY_KEY must contain all keys."""
    for d in REGISTRY:
        assert get_def(d.key) is not None
        assert get_def(d.key).key == d.key


def test_registry_unknown_key():
    """get_def returns None for unknown keys."""
    assert get_def("NONEXISTENT_KEY") is None


def test_org_overridable_keys_disjoint_from_app_only():
    """App-only and org-overridable keys must be disjoint."""
    assert APP_ONLY_KEYS.isdisjoint(ORG_OVERRIDABLE_KEYS)


def test_all_keys_covers_registry():
    """all_keys() returns every key in the registry."""
    assert set(all_keys()) == set(d.key for d in REGISTRY)


# ---------------------------------------------------------------------------
# 2. 3-tier precedence
# ---------------------------------------------------------------------------

def test_precedence_falls_back_to_registry_default(db_session):
    """With no DB rows, get_setting returns the registry default."""
    val = get_setting(db_session, "RETRIEVAL_TOP_K", org_id=None)
    assert val == get_def("RETRIEVAL_TOP_K").default


def test_precedence_app_value_overrides_env(db_session):
    """App-level DB row overrides .env default."""
    upsert_app_setting(db_session, "RETRIEVAL_TOP_K", 50)
    val = get_setting(db_session, "RETRIEVAL_TOP_K", org_id=None)
    assert val == 50


def test_precedence_org_overrides_app(db_session):
    """Org-level override wins over app-level value."""
    org = _create_org(db_session)
    upsert_app_setting(db_session, "RETRIEVAL_TOP_K", 50)
    upsert_org_setting(db_session, org.id, "RETRIEVAL_TOP_K", 100)
    val = get_setting(db_session, "RETRIEVAL_TOP_K", org_id=org.id)
    assert val == 100


def test_precedence_org_without_override_uses_app(db_session):
    """Org without override inherits app value."""
    org = _create_org(db_session)
    upsert_app_setting(db_session, "RETRIEVAL_TOP_K", 50)
    val = get_setting(db_session, "RETRIEVAL_TOP_K", org_id=org.id)
    assert val == 50


def test_precedence_app_only_key_ignores_org_id(db_session):
    """App-only keys return app value even when org_id is provided."""
    org = _create_org(db_session)
    upsert_app_setting(db_session, "CHUNK_SIZE", 2000)
    val = get_setting(db_session, "CHUNK_SIZE", org_id=org.id)
    assert val == 2000


# ---------------------------------------------------------------------------
# 3. Reset semantics
# ---------------------------------------------------------------------------

def test_reset_org_reverts_to_app(db_session):
    """Deleting org override reverts to app value."""
    org = _create_org(db_session)
    upsert_app_setting(db_session, "RETRIEVAL_TOP_K", 50)
    upsert_org_setting(db_session, org.id, "RETRIEVAL_TOP_K", 100)
    assert get_setting(db_session, "RETRIEVAL_TOP_K", org_id=org.id) == 100

    reset_org_setting(db_session, org.id, "RETRIEVAL_TOP_K")
    assert get_setting(db_session, "RETRIEVAL_TOP_K", org_id=org.id) == 50


def test_reset_app_reverts_to_env(db_session):
    """Deleting app value reverts to registry default."""
    upsert_app_setting(db_session, "RETRIEVAL_TOP_K", 50)
    assert get_setting(db_session, "RETRIEVAL_TOP_K", org_id=None) == 50

    reset_app_setting(db_session, "RETRIEVAL_TOP_K")
    assert get_setting(db_session, "RETRIEVAL_TOP_K", org_id=None) == get_def("RETRIEVAL_TOP_K").default


def test_reset_all_org_settings(db_session):
    """Resetting all org overrides clears every org row."""
    org = _create_org(db_session)
    upsert_org_setting(db_session, org.id, "RETRIEVAL_TOP_K", 100)
    upsert_org_setting(db_session, org.id, "RERANKER_ENABLED", False)

    reset_all_org_settings(db_session, org.id)

    rows = db_session.query(Setting).filter(Setting.org_id == org.id).all()
    assert len(rows) == 0


# ---------------------------------------------------------------------------
# 4. Validation
# ---------------------------------------------------------------------------

def test_validate_int_coercion():
    assert validate_value("RETRIEVAL_TOP_K", "42") == 42


def test_validate_int_min_max():
    with pytest.raises(ValueError, match="must be >="):
        validate_value("RETRIEVAL_TOP_K", 0)
    with pytest.raises(ValueError, match="must be <="):
        validate_value("RETRIEVAL_TOP_K", 500)


def test_validate_float():
    assert validate_value("HYBRID_DENSE_WEIGHT", "0.7") == 0.7


def test_validate_bool_from_string():
    assert validate_value("RERANKER_ENABLED", "true") is True
    assert validate_value("RERANKER_ENABLED", "false") is False


def test_validate_bool_from_bool():
    assert validate_value("RERANKER_ENABLED", True) is True


def test_validate_unknown_key():
    with pytest.raises(ValueError, match="Unknown setting key"):
        validate_value("NONEXISTENT", 42)


def test_validate_choices():
    assert validate_value("TOOL_CALL_MODE", "auto") == "auto"
    with pytest.raises(ValueError, match="must be one of"):
        validate_value("TOOL_CALL_MODE", "invalid")


def test_validate_int_rejects_string():
    with pytest.raises(ValueError, match="must be an integer"):
        validate_value("RETRIEVAL_TOP_K", "not_a_number")


# ---------------------------------------------------------------------------
# 5. Scope enforcement
# ---------------------------------------------------------------------------

def test_org_crud_rejects_app_only_key(db_session):
    """upsert_org_setting must reject app-only keys."""
    org = _create_org(db_session)
    with pytest.raises(ValueError, match="cannot be overridden per organisation"):
        upsert_org_setting(db_session, org.id, "CHUNK_SIZE", 2000)


def test_is_org_overridable():
    assert is_org_overridable("RETRIEVAL_TOP_K") is True
    assert is_org_overridable("CHUNK_SIZE") is False
    assert is_org_overridable("DENSE_EMBEDDINGS_MODEL") is False


# ---------------------------------------------------------------------------
# 6. OrgSettings accessor
# ---------------------------------------------------------------------------

def test_org_settings_attribute_access(db_session):
    """OrgSettings exposes settings as attributes."""
    org = _create_org(db_session)
    upsert_app_setting(db_session, "RETRIEVAL_TOP_K", 50)
    os = OrgSettings(db_session, org.id)
    assert os.RETRIEVAL_TOP_K == 50


def test_org_settings_computed_chunk_overlap(db_session):
    """chunk_overlap = CHUNK_SIZE * OVERLAP_PERCENTAGE."""
    os = OrgSettings(db_session, None)
    assert os.chunk_overlap == int(os.CHUNK_SIZE * os.OVERLAP_PERCENTAGE)


def test_org_settings_unknown_attr_raises(db_session):
    os = OrgSettings(db_session, None)
    with pytest.raises(AttributeError):
        _ = os.NONEXISTENT_SETTING


# ---------------------------------------------------------------------------
# 7. Cache invalidation
# ---------------------------------------------------------------------------

def test_cache_invalidation_on_app_upsert(db_session):
    """Upserting an app setting invalidates the cache."""
    # First read populates cache
    v1 = get_setting(db_session, "RETRIEVAL_TOP_K", None)
    # Upsert changes the value
    upsert_app_setting(db_session, "RETRIEVAL_TOP_K", 99)
    # Second read should see the new value, not the cached one
    v2 = get_setting(db_session, "RETRIEVAL_TOP_K", None)
    assert v2 == 99


def test_cache_invalidation_on_org_upsert(db_session):
    """Upserting an org setting invalidates the cache for that org."""
    org = _create_org(db_session)
    upsert_app_setting(db_session, "RETRIEVAL_TOP_K", 50)
    v1 = get_setting(db_session, "RETRIEVAL_TOP_K", org.id)
    assert v1 == 50

    upsert_org_setting(db_session, org.id, "RETRIEVAL_TOP_K", 77)
    v2 = get_setting(db_session, "RETRIEVAL_TOP_K", org.id)
    assert v2 == 77


# ---------------------------------------------------------------------------
# 8. Metadata API helpers
# ---------------------------------------------------------------------------

def test_get_all_app_settings_with_meta(db_session):
    """get_all_app_settings_with_meta returns all registry keys with metadata."""
    items = get_all_app_settings_with_meta(db_session)
    assert len(items) == len(REGISTRY)
    for item in items:
        assert "key" in item
        assert "value" in item
        assert "source" in item
        assert item["source"] in ("database", "install_default")


def test_get_all_org_settings_with_meta(db_session):
    """get_all_org_settings_with_meta returns only org-overridable keys."""
    org = _create_org(db_session)
    items = get_all_org_settings_with_meta(db_session, org.id)
    assert len(items) == len([d for d in REGISTRY if d.scope == "org"])
    for item in items:
        assert item["scope"] == "org"
        assert "overridden" in item
        assert "app_default" in item
