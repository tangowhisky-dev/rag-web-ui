"""
test_settings_preflight.py — Tests for post-login settings validation.

Verifies:
  1. Preflight returns ok=True when all critical settings are set.
  2. Preflight returns ok=False with error when OPENAI_API_BASE is unset.
  3. Preflight does NOT require API keys (local servers don't need them).
  4. Preflight reports who_can_fix correctly (org_admin vs super_admin).
  5. Preflight checks embedding settings (app-only, super_admin fixable).
  6. Preflight includes optional ingestion settings for super_admin only.
  7. Preflight respects org overrides (org-level key satisfies the check).
  8. API endpoint /api/auth/preflight returns the right structure.
"""
import pytest
from unittest.mock import patch

from app.services.settings_preflight import check_required_settings
from app.services.settings_service import upsert_app_setting, upsert_org_setting, clear_cache


@pytest.fixture()
def db_session():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from app.models.base import Base
    import app.models.user  # noqa
    import app.models.organisation  # noqa
    import app.models.setting  # noqa

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


def _create_org(db, name="TestOrg"):
    from app.models.organisation import Organisation
    org = Organisation(name=name, parent_id=None)
    db.add(org)
    db.commit()
    db.refresh(org)
    return org


# ── 1. All settings set → ok=True ────────────────────────────────────────────

def test_preflight_ok_when_all_settings_set(db_session):
    """When all critical settings have values, preflight returns ok=True."""
    upsert_app_setting(db_session, "OPENAI_API_BASE", "https://api.openai.com/v1")
    upsert_app_setting(db_session, "OPENAI_MODEL", "gpt-4o")
    upsert_app_setting(db_session, "EMBEDDING_API_BASE", "https://api.openai.com/v1")
    upsert_app_setting(db_session, "DENSE_EMBEDDINGS_MODEL", "text-embedding-3-small")
    clear_cache()

    result = check_required_settings(db_session, "user", org_id=None)
    errors = [i for i in result.issues if i.severity == "error"]
    assert len(errors) == 0
    assert result.ok is True


# ── 2. EMBEDDING_API_BASE unset with no fallback → error ─────────────────────

def test_preflight_error_when_embedding_base_unset(db_session):
    """When EMBEDDING_API_BASE is unset and OPENAI_API_BASE fallback also
    resolves to None, preflight returns ok=False with error.

    We patch get_setting to return None for both keys, simulating a deployment
    where neither is configured and registry defaults are overridden to None.
    """
    clear_cache()
    from app.services import settings_service as _ss
    original_get = _ss.get_setting

    def _mock_get(db, key, org_id=None):
        if key in ("EMBEDDING_API_BASE", "OPENAI_API_BASE"):
            return None
        return original_get(db, key, org_id)

    with patch("app.services.settings_preflight.get_setting", _mock_get):
        result = check_required_settings(db_session, "user", org_id=None)

    errors = [i for i in result.issues if i.severity == "error"]
    assert len(errors) > 0
    assert result.ok is False
    embed_issue = next(i for i in errors if i.key == "EMBEDDING_API_BASE")
    assert embed_issue is not None


# ── 3. API keys are NOT required ──────────────────────────────────────────────

def test_preflight_does_not_require_api_keys(db_session):
    """API keys should not be required — local servers don't need them."""
    upsert_app_setting(db_session, "OPENAI_API_BASE", "http://localhost:1234/v1")
    upsert_app_setting(db_session, "OPENAI_MODEL", "local-model")
    upsert_app_setting(db_session, "EMBEDDING_API_BASE", "http://localhost:1234/v1")
    upsert_app_setting(db_session, "DENSE_EMBEDDINGS_MODEL", "local-embedding-model")
    clear_cache()

    # No API keys set anywhere — should still be ok
    with patch("app.services.settings_service.os.getenv", return_value=None):
        result = check_required_settings(db_session, "user", org_id=None)

    key_issues = [i for i in result.issues if "API_KEY" in i.key]
    assert len(key_issues) == 0
    assert result.ok is True


# ── 4. who_can_fix is correct ────────────────────────────────────────────────

def test_preflight_who_can_fix_org_setting(db_session):
    """Org-scoped setting reports who_can_fix='org_admin'."""
    clear_cache()
    from app.services import settings_service as _ss
    original_get = _ss.get_setting

    def _mock_get(db, key, org_id=None):
        if key in ("OPENAI_API_BASE", "OPENAI_MODEL", "EMBEDDING_API_BASE"):
            return None
        return original_get(db, key, org_id)

    with patch("app.services.settings_preflight.get_setting", _mock_get):
        result = check_required_settings(db_session, "user", org_id=1)

    base_issue = next(i for i in result.issues if i.key == "OPENAI_API_BASE")
    assert base_issue.who_can_fix == "org_admin"
    assert base_issue.scope == "org"


def test_preflight_who_can_fix_app_setting(db_session):
    """App-scoped setting reports who_can_fix='super_admin'."""
    clear_cache()
    from app.services import settings_service as _ss
    original_get = _ss.get_setting

    def _mock_get(db, key, org_id=None):
        if key in ("EMBEDDING_API_BASE", "OPENAI_API_BASE"):
            return None
        return original_get(db, key, org_id)

    with patch("app.services.settings_preflight.get_setting", _mock_get):
        result = check_required_settings(db_session, "user", org_id=None)

    embed_issue = next(i for i in result.issues if i.key == "EMBEDDING_API_BASE")
    assert embed_issue.who_can_fix == "super_admin"
    assert embed_issue.scope == "app"


# ── 5. Embedding settings checked ────────────────────────────────────────────

def test_preflight_checks_embedding_base(db_session):
    """EMBEDDING_API_BASE falls back to OPENAI_API_BASE. If both resolve to
    None, error is reported."""
    clear_cache()
    from app.services import settings_service as _ss
    original_get = _ss.get_setting

    def _mock_get(db, key, org_id=None):
        if key in ("EMBEDDING_API_BASE", "OPENAI_API_BASE"):
            return None
        return original_get(db, key, org_id)

    with patch("app.services.settings_preflight.get_setting", _mock_get):
        result = check_required_settings(db_session, "user", org_id=None)

    embed_issues = [i for i in result.issues if i.key == "EMBEDDING_API_BASE"]
    assert len(embed_issues) == 1
    assert embed_issues[0].severity == "error"


def test_preflight_embedding_base_satisfied_by_fallback(db_session):
    """EMBEDDING_API_BASE is satisfied when OPENAI_API_BASE is set at app level."""
    upsert_app_setting(db_session, "OPENAI_API_BASE", "http://localhost:1234/v1")
    upsert_app_setting(db_session, "OPENAI_MODEL", "local-model")
    upsert_app_setting(db_session, "DENSE_EMBEDDINGS_MODEL", "local-embedding-model")
    clear_cache()

    result = check_required_settings(db_session, "user", org_id=None)
    embed_issues = [i for i in result.issues if i.key == "EMBEDDING_API_BASE"]
    assert len(embed_issues) == 0


# ── 6. Optional ingestion settings for super_admin only ──────────────────────

def test_preflight_optional_settings_for_super_admin(db_session):
    """Super_admin sees optional ingestion settings as warnings."""
    upsert_app_setting(db_session, "OPENAI_API_BASE", "http://localhost:1234/v1")
    upsert_app_setting(db_session, "OPENAI_MODEL", "local-model")
    upsert_app_setting(db_session, "EMBEDDING_API_BASE", "http://localhost:1234/v1")
    upsert_app_setting(db_session, "DENSE_EMBEDDINGS_MODEL", "local-embedding-model")
    clear_cache()

    with patch("app.services.settings_service.os.getenv", return_value=None):
        result = check_required_settings(db_session, "super_admin", org_id=None)

    warnings = [i for i in result.issues if i.severity == "warning"]
    keys = {i.key for i in warnings}
    # VISION_MODEL has no fallback (default=None) → reported as warning
    assert "VISION_MODEL" in keys
    # GRAPHRAG_LLM falls back to OPENAI_MODEL (has default) → not reported
    assert "GRAPHRAG_LLM" not in keys


def test_preflight_no_optional_settings_for_normal_user(db_session):
    """Normal users don't see optional ingestion settings."""
    upsert_app_setting(db_session, "OPENAI_API_BASE", "http://localhost:1234/v1")
    upsert_app_setting(db_session, "OPENAI_MODEL", "local-model")
    upsert_app_setting(db_session, "EMBEDDING_API_BASE", "http://localhost:1234/v1")
    upsert_app_setting(db_session, "DENSE_EMBEDDINGS_MODEL", "local-embedding-model")
    clear_cache()

    with patch("app.services.settings_service.os.getenv", return_value=None):
        result = check_required_settings(db_session, "user", org_id=None)

    warnings = [i for i in result.issues if i.severity == "warning"]
    assert len(warnings) == 0


# ── 7. Org override satisfies check ──────────────────────────────────────────

def test_preflight_org_override_satisfies_check(db_session):
    """Org-level OPENAI_API_BASE satisfies the check for that org's users."""
    org = _create_org(db_session)
    upsert_org_setting(db_session, org.id, "OPENAI_API_BASE", "http://org-llm:1234/v1")
    upsert_org_setting(db_session, org.id, "OPENAI_MODEL", "org-model")
    upsert_app_setting(db_session, "EMBEDDING_API_BASE", "http://localhost:1234/v1")
    upsert_app_setting(db_session, "DENSE_EMBEDDINGS_MODEL", "local-embedding-model")
    clear_cache()

    with patch("app.services.settings_service.os.getenv", return_value=None):
        result = check_required_settings(db_session, "user", org_id=org.id)

    base_issues = [i for i in result.issues if i.key == "OPENAI_API_BASE"]
    assert len(base_issues) == 0
    assert result.ok is True


def test_preflight_different_org_not_satisfied(db_session):
    """Org-level base for org A doesn't satisfy check for org B.

    We mock get_setting so that OPENAI_API_BASE returns a value only for
    org_a and None for org_b, simulating org isolation without relying on
    registry defaults.
    """
    org_a = _create_org(db_session, "OrgA")
    org_b = _create_org(db_session, "OrgB")
    upsert_app_setting(db_session, "EMBEDDING_API_BASE", "http://localhost:1234/v1")
    upsert_app_setting(db_session, "DENSE_EMBEDDINGS_MODEL", "local-embedding-model")
    clear_cache()

    from app.services import settings_service as _ss
    original_get = _ss.get_setting

    def _mock_get(db, key, org_id=None):
        if key == "OPENAI_API_BASE" and org_id == org_b.id:
            return None
        if key == "OPENAI_MODEL" and org_id == org_b.id:
            return None
        return original_get(db, key, org_id)

    with patch("app.services.settings_preflight.get_setting", _mock_get):
        result = check_required_settings(db_session, "user", org_id=org_b.id)

    base_issues = [i for i in result.issues if i.key == "OPENAI_API_BASE"]
    assert len(base_issues) == 1
    assert result.ok is False


# ── 8. API endpoint structure ────────────────────────────────────────────────

def test_preflight_endpoint_returns_structure(db_session):
    """The preflight function returns a dict with the expected keys."""
    upsert_app_setting(db_session, "OPENAI_API_BASE", "http://localhost:1234/v1")
    upsert_app_setting(db_session, "OPENAI_MODEL", "local-model")
    upsert_app_setting(db_session, "EMBEDDING_API_BASE", "http://localhost:1234/v1")
    upsert_app_setting(db_session, "DENSE_EMBEDDINGS_MODEL", "local-embedding-model")
    clear_cache()

    result = check_required_settings(db_session, "user", org_id=None)
    d = result.to_dict()

    assert "role" in d
    assert "org_id" in d
    assert "ok" in d
    assert "issues" in d
    assert isinstance(d["issues"], list)
    for issue in d["issues"]:
        assert "key" in issue
        assert "label" in issue
        assert "severity" in issue
        assert "who_can_fix" in issue
        assert "scope" in issue
        assert "is_set" in issue
