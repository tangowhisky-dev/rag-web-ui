"""
test_settings_preflight.py — Tests for post-login settings validation.

Verifies:
  1. Preflight returns ok=True when all critical settings are set.
  2. Preflight returns ok=False with error when OPENAI_API_KEY is unset.
  3. Preflight reports who_can_fix correctly (org_admin vs super_admin).
  4. Preflight checks embedding settings (app-only, super_admin fixable).
  5. Preflight includes optional ingestion settings for super_admin only.
  6. Preflight respects org overrides (org-level key satisfies the check).
  7. API endpoint /api/auth/preflight returns the right structure.
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
    upsert_app_setting(db_session, "OPENAI_API_KEY", "sk-test-key")
    upsert_app_setting(db_session, "OPENAI_API_BASE", "https://api.openai.com/v1")
    upsert_app_setting(db_session, "OPENAI_MODEL", "gpt-4o")
    upsert_app_setting(db_session, "EMBEDDING_API_KEY", "sk-embed-key")
    upsert_app_setting(db_session, "EMBEDDING_API_BASE", "https://api.openai.com/v1")
    upsert_app_setting(db_session, "DENSE_EMBEDDINGS_MODEL", "text-embedding-3-small")
    clear_cache()

    result = check_required_settings(db_session, "user", org_id=None)
    errors = [i for i in result.issues if i.severity == "error"]
    assert len(errors) == 0
    assert result.ok is True


# ── 2. OPENAI_API_KEY unset → error ──────────────────────────────────────────

def test_preflight_error_when_api_key_unset(db_session):
    """When OPENAI_API_KEY is unset, preflight returns ok=False with error."""
    clear_cache()
    # Patch os.getenv to return None for OPENAI_API_KEY
    with patch("app.services.settings_service.os.getenv", return_value=None):
        result = check_required_settings(db_session, "user", org_id=None)

    errors = [i for i in result.issues if i.severity == "error"]
    assert len(errors) > 0
    assert result.ok is False
    key_issue = next(i for i in errors if i.key == "OPENAI_API_KEY")
    assert "Chat will not work" in key_issue.message


# ── 3. who_can_fix is correct ────────────────────────────────────────────────

def test_preflight_who_can_fix_org_setting(db_session):
    """Org-scoped setting reports who_can_fix='org_admin'."""
    clear_cache()
    with patch("app.services.settings_service.os.getenv", return_value=None):
        result = check_required_settings(db_session, "user", org_id=1)

    key_issue = next(i for i in result.issues if i.key == "OPENAI_API_KEY")
    assert key_issue.who_can_fix == "org_admin"
    assert key_issue.scope == "org"


def test_preflight_who_can_fix_app_setting(db_session):
    """App-scoped setting reports who_can_fix='super_admin'."""
    clear_cache()
    with patch("app.services.settings_service.os.getenv", return_value=None):
        result = check_required_settings(db_session, "user", org_id=None)

    embed_issue = next(i for i in result.issues if i.key == "EMBEDDING_API_KEY")
    assert embed_issue.who_can_fix == "super_admin"
    assert embed_issue.scope == "app"


# ── 4. Embedding settings checked ────────────────────────────────────────────

def test_preflight_checks_embedding_key_with_fallback(db_session):
    """EMBEDDING_API_KEY falls back to OPENAI_API_KEY. If both unset, error."""
    clear_cache()
    with patch("app.services.settings_service.os.getenv", return_value=None):
        result = check_required_settings(db_session, "user", org_id=None)

    embed_issue = next(
        i for i in result.issues if i.key == "EMBEDDING_API_KEY"
    )
    assert "OPENAI_API_KEY" in embed_issue.message
    assert embed_issue.severity == "error"


def test_preflight_embedding_key_satisfied_by_fallback(db_session):
    """EMBEDDING_API_KEY is satisfied when OPENAI_API_KEY is set at app level."""
    upsert_app_setting(db_session, "OPENAI_API_KEY", "sk-test-key")
    clear_cache()

    result = check_required_settings(db_session, "user", org_id=None)
    embed_issues = [i for i in result.issues if i.key == "EMBEDDING_API_KEY"]
    assert len(embed_issues) == 0


# ── 5. Optional ingestion settings for super_admin only ──────────────────────

def test_preflight_optional_settings_for_super_admin(db_session):
    """Super_admin sees optional ingestion settings as warnings."""
    upsert_app_setting(db_session, "OPENAI_API_KEY", "sk-test-key")
    upsert_app_setting(db_session, "EMBEDDING_API_KEY", "sk-embed-key")
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
    upsert_app_setting(db_session, "OPENAI_API_KEY", "sk-test-key")
    upsert_app_setting(db_session, "EMBEDDING_API_KEY", "sk-embed-key")
    clear_cache()

    with patch("app.services.settings_service.os.getenv", return_value=None):
        result = check_required_settings(db_session, "user", org_id=None)

    warnings = [i for i in result.issues if i.severity == "warning"]
    assert len(warnings) == 0


# ── 6. Org override satisfies check ──────────────────────────────────────────

def test_preflight_org_override_satisfies_check(db_session):
    """Org-level OPENAI_API_KEY satisfies the check for that org's users."""
    org = _create_org(db_session)
    upsert_org_setting(db_session, org.id, "OPENAI_API_KEY", "sk-org-key")
    upsert_app_setting(db_session, "EMBEDDING_API_KEY", "sk-embed-key")
    clear_cache()

    with patch("app.services.settings_service.os.getenv", return_value=None):
        result = check_required_settings(db_session, "user", org_id=org.id)

    key_issues = [i for i in result.issues if i.key == "OPENAI_API_KEY"]
    assert len(key_issues) == 0
    assert result.ok is True


def test_preflight_different_org_not_satisfied(db_session):
    """Org-level key for org A doesn't satisfy check for org B."""
    org_a = _create_org(db_session, "OrgA")
    org_b = _create_org(db_session, "OrgB")
    upsert_org_setting(db_session, org_a.id, "OPENAI_API_KEY", "sk-org-a-key")
    upsert_app_setting(db_session, "EMBEDDING_API_KEY", "sk-embed-key")
    clear_cache()

    with patch("app.services.settings_service.os.getenv", return_value=None):
        result = check_required_settings(db_session, "user", org_id=org_b.id)

    key_issues = [i for i in result.issues if i.key == "OPENAI_API_KEY"]
    assert len(key_issues) == 1
    assert result.ok is False


# ── 7. API endpoint structure ────────────────────────────────────────────────

def test_preflight_endpoint_returns_structure(db_session):
    """The preflight function returns a dict with the expected keys."""
    upsert_app_setting(db_session, "OPENAI_API_KEY", "sk-test-key")
    upsert_app_setting(db_session, "EMBEDDING_API_KEY", "sk-embed-key")
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
