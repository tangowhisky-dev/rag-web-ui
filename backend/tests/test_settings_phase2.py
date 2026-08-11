"""
test_settings_phase2.py — Phase 2: API endpoints for Super Admin and Admin settings.

Tests:
  1. Super admin can list/get/update/reset app settings.
  2. Admin cannot access app settings endpoints (403).
  3. Regular user cannot access any settings endpoints (403).
  4. Admin can list/update/reset org settings within scope.
  5. Admin cannot edit org settings outside their scope (403).
  6. Org endpoint rejects app-only keys (403).
  7. Validation errors return 422.
  8. Schema endpoints return registry metadata.
  9. Bulk update with mixed valid/invalid keys.
 10. Delete all org overrides.
"""
import pytest
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from app.main import app as fastapi_app  # noqa: conftest must run first

import app.db.session as _session_mod
from app.db.session import get_db
from app.models.user import User, UserRole
from app.models.base import Base  # noqa
import app.models.user  # noqa
import app.models.knowledge  # noqa
import app.models.chat  # noqa
import app.models.organisation  # noqa
import app.models.datastore  # noqa
import app.models.setting  # noqa
import app.models.org_llm_config  # noqa
from app.models.organisation import Organisation
from app.services.settings_service import clear_cache

engine = _session_mod.engine
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def reset_db():
    Base.metadata.create_all(bind=engine)
    clear_cache()
    yield
    Base.metadata.drop_all(bind=engine)
    clear_cache()


@pytest.fixture()
def db():
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def client():
    fastapi_app.dependency_overrides[get_db] = override_get_db
    with TestClient(fastapi_app) as c:
        yield c
    fastapi_app.dependency_overrides.clear()


# ── Helpers ───────────────────────────────────────────────────────────────

def create_user(db, username, password, role, org_id=None):
    from app.core.security import get_password_hash
    if org_id is None:
        root = db.query(Organisation).filter(Organisation.parent_id.is_(None)).first()
        org_id = root.id if root else None
    user = User(
        username=username,
        email=f"{username}@example.com",
        hashed_password=get_password_hash(password),
        is_active=True,
        role=role,
        org_id=org_id,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def create_org(db, name, parent_id=None):
    if parent_id is None:
        root = db.query(Organisation).filter(Organisation.parent_id.is_(None)).first()
        parent_id = root.id if root else None
    org = Organisation(name=name, parent_id=parent_id)
    db.add(org)
    db.flush()
    parent = db.query(Organisation).filter(Organisation.id == parent_id).first()
    org.path = f"{parent.path}/{org.id}" if parent else f"/{org.id}"
    db.commit()
    db.refresh(org)
    return org


def get_token(client, username, password):
    resp = client.post("/api/auth/token", data={"username": username, "password": password})
    assert resp.status_code == 200, f"login failed: {resp.text}"
    return resp.json()["access_token"]


def setup_root_org(db):
    """Ensure a root org exists."""
    root = db.query(Organisation).filter(Organisation.parent_id.is_(None)).first()
    if not root:
        root = Organisation(name="Root", parent_id=None, path="/1")
        db.add(root)
        db.commit()
        db.refresh(root)
    return root


# ---------------------------------------------------------------------------
# 1. Super admin app settings
# ---------------------------------------------------------------------------

def test_super_admin_can_list_app_settings(client, db):
    root = setup_root_org(db)
    create_user(db, "superadmin", "pass123", UserRole.super_admin, root.id)
    token = get_token(client, "superadmin", "pass123")

    resp = client.get("/api/admin/settings", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    data = resp.json()
    assert "settings" in data
    assert len(data["settings"]) > 0
    # Every item should have required fields
    for item in data["settings"]:
        assert "key" in item
        assert "value" in item
        assert "scope" in item
        assert "source" in item


def test_super_admin_can_update_app_setting(client, db):
    root = setup_root_org(db)
    create_user(db, "superadmin2", "pass123", UserRole.super_admin, root.id)
    token = get_token(client, "superadmin2", "pass123")

    resp = client.post(
        "/api/admin/settings/RETRIEVAL_TOP_K",
        json={"key": "RETRIEVAL_TOP_K", "value": 42},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    # Verify it was saved
    resp2 = client.get("/api/admin/settings", headers={"Authorization": f"Bearer {token}"})
    items = {s["key"]: s for s in resp2.json()["settings"]}
    assert items["RETRIEVAL_TOP_K"]["value"] == 42
    assert items["RETRIEVAL_TOP_K"]["source"] == "database"


def test_super_admin_can_reset_app_setting(client, db):
    root = setup_root_org(db)
    create_user(db, "superadmin3", "pass123", UserRole.super_admin, root.id)
    token = get_token(client, "superadmin3", "pass123")

    # Set a value
    client.post(
        "/api/admin/settings/RETRIEVAL_TOP_K",
        json={"key": "RETRIEVAL_TOP_K", "value": 42},
        headers={"Authorization": f"Bearer {token}"},
    )
    # Reset it
    resp = client.delete(
        "/api/admin/settings/RETRIEVAL_TOP_K",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "reset"

    # Verify it reverted
    resp2 = client.get("/api/admin/settings", headers={"Authorization": f"Bearer {token}"})
    items = {s["key"]: s for s in resp2.json()["settings"]}
    assert items["RETRIEVAL_TOP_K"]["source"] == "install_default"


def test_super_admin_bulk_update(client, db):
    root = setup_root_org(db)
    create_user(db, "superadmin4", "pass123", UserRole.super_admin, root.id)
    token = get_token(client, "superadmin4", "pass123")

    resp = client.put(
        "/api/admin/settings",
        json={"settings": [
            {"key": "RETRIEVAL_TOP_K", "value": 15},
            {"key": "RERANKER_ENABLED", "value": False},
        ]},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    results = {r["key"]: r for r in resp.json()["results"]}
    assert results["RETRIEVAL_TOP_K"]["status"] == "ok"
    assert results["RERANKER_ENABLED"]["status"] == "ok"


def test_super_admin_effective_snapshot(client, db):
    root = setup_root_org(db)
    create_user(db, "superadmin5", "pass123", UserRole.super_admin, root.id)
    token = get_token(client, "superadmin5", "pass123")

    resp = client.get("/api/admin/settings/effective", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    data = resp.json()
    assert "RETRIEVAL_TOP_K" in data


# ---------------------------------------------------------------------------
# 2. Admin cannot access app settings
# ---------------------------------------------------------------------------

def test_admin_cannot_access_app_settings(client, db):
    root = setup_root_org(db)
    create_user(db, "adminuser", "pass123", UserRole.admin, root.id)
    token = get_token(client, "adminuser", "pass123")

    resp = client.get("/api/admin/settings", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403


def test_admin_cannot_update_app_settings(client, db):
    root = setup_root_org(db)
    create_user(db, "adminuser2", "pass123", UserRole.admin, root.id)
    token = get_token(client, "adminuser2", "pass123")

    resp = client.post(
        "/api/admin/settings/RETRIEVAL_TOP_K",
        json={"key": "RETRIEVAL_TOP_K", "value": 42},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# 3. Regular user cannot access any settings
# ---------------------------------------------------------------------------

def test_regular_user_rejected_app_settings(client, db):
    root = setup_root_org(db)
    create_user(db, "regularuser", "pass123", UserRole.user, root.id)
    token = get_token(client, "regularuser", "pass123")

    resp = client.get("/api/admin/settings", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403


def test_regular_user_rejected_org_settings(client, db):
    root = setup_root_org(db)
    org = create_org(db, "TestOrg")
    create_user(db, "regularuser2", "pass123", UserRole.user, root.id)
    token = get_token(client, "regularuser2", "pass123")

    resp = client.get(
        f"/api/admin/orgs/{org.id}/settings",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# 4. Admin can manage org settings within scope
# ---------------------------------------------------------------------------

def test_admin_can_list_org_settings(client, db):
    root = setup_root_org(db)
    create_user(db, "adminuser3", "pass123", UserRole.admin, root.id)
    token = get_token(client, "adminuser3", "pass123")

    resp = client.get(
        f"/api/admin/orgs/{root.id}/settings",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "settings" in data
    # Only org-overridable keys
    for item in data["settings"]:
        assert item["scope"] == "org"


def test_admin_can_update_org_setting(client, db):
    root = setup_root_org(db)
    create_user(db, "adminuser4", "pass123", UserRole.admin, root.id)
    token = get_token(client, "adminuser4", "pass123")

    resp = client.post(
        f"/api/admin/orgs/{root.id}/settings/RETRIEVAL_TOP_K",
        json={"key": "RETRIEVAL_TOP_K", "value": 30},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200

    # Verify
    resp2 = client.get(
        f"/api/admin/orgs/{root.id}/settings",
        headers={"Authorization": f"Bearer {token}"},
    )
    items = {s["key"]: s for s in resp2.json()["settings"]}
    assert items["RETRIEVAL_TOP_K"]["overridden"] is True
    assert items["RETRIEVAL_TOP_K"]["effective"] == 30


def test_admin_can_reset_org_setting(client, db):
    root = setup_root_org(db)
    create_user(db, "adminuser5", "pass123", UserRole.admin, root.id)
    token = get_token(client, "adminuser5", "pass123")

    # Set an override
    client.post(
        f"/api/admin/orgs/{root.id}/settings/RETRIEVAL_TOP_K",
        json={"key": "RETRIEVAL_TOP_K", "value": 30},
        headers={"Authorization": f"Bearer {token}"},
    )
    # Reset it
    resp = client.delete(
        f"/api/admin/orgs/{root.id}/settings/RETRIEVAL_TOP_K",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "reset"

    # Verify override is gone
    resp2 = client.get(
        f"/api/admin/orgs/{root.id}/settings",
        headers={"Authorization": f"Bearer {token}"},
    )
    items = {s["key"]: s for s in resp2.json()["settings"]}
    assert items["RETRIEVAL_TOP_K"]["overridden"] is False


def test_admin_bulk_update_org_settings(client, db):
    root = setup_root_org(db)
    create_user(db, "adminuser6", "pass123", UserRole.admin, root.id)
    token = get_token(client, "adminuser6", "pass123")

    resp = client.put(
        f"/api/admin/orgs/{root.id}/settings",
        json={"settings": [
            {"key": "RETRIEVAL_TOP_K", "value": 25},
            {"key": "RERANKER_ENABLED", "value": False},
        ]},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    results = {r["key"]: r for r in resp.json()["results"]}
    assert results["RETRIEVAL_TOP_K"]["status"] == "ok"
    assert results["RERANKER_ENABLED"]["status"] == "ok"


def test_admin_delete_all_org_settings(client, db):
    root = setup_root_org(db)
    create_user(db, "adminuser7", "pass123", UserRole.admin, root.id)
    token = get_token(client, "adminuser7", "pass123")

    # Set some overrides
    client.put(
        f"/api/admin/orgs/{root.id}/settings",
        json={"settings": [
            {"key": "RETRIEVAL_TOP_K", "value": 25},
            {"key": "RERANKER_ENABLED", "value": False},
        ]},
        headers={"Authorization": f"Bearer {token}"},
    )
    # Delete all
    resp = client.delete(
        f"/api/admin/orgs/{root.id}/settings",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "all_cleared"

    # Verify all overrides are gone
    resp2 = client.get(
        f"/api/admin/orgs/{root.id}/settings",
        headers={"Authorization": f"Bearer {token}"},
    )
    for item in resp2.json()["settings"]:
        assert item["overridden"] is False


# ---------------------------------------------------------------------------
# 5. Admin cannot edit outside scope
# ---------------------------------------------------------------------------

def test_admin_cannot_edit_outside_scope(client, db):
    root = setup_root_org(db)
    # Create a separate root-level org (not a child of root) so it's out of scope
    other_root = Organisation(name="OtherRoot", parent_id=None, path="/999")
    db.add(other_root)
    db.commit()
    db.refresh(other_root)
    create_user(db, "adminuser8", "pass123", UserRole.admin, root.id)
    token = get_token(client, "adminuser8", "pass123")

    resp = client.get(
        f"/api/admin/orgs/{other_root.id}/settings",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403


def test_super_admin_can_edit_any_org(client, db):
    root = setup_root_org(db)
    other_org = create_org(db, "OtherOrg2")
    create_user(db, "superadmin6", "pass123", UserRole.super_admin, root.id)
    token = get_token(client, "superadmin6", "pass123")

    resp = client.get(
        f"/api/admin/orgs/{other_org.id}/settings",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 6. Org endpoint rejects app-only keys
# ---------------------------------------------------------------------------

def test_org_endpoint_rejects_app_only_key(client, db):
    root = setup_root_org(db)
    create_user(db, "adminuser9", "pass123", UserRole.admin, root.id)
    token = get_token(client, "adminuser9", "pass123")

    resp = client.post(
        f"/api/admin/orgs/{root.id}/settings/CHUNK_SIZE",
        json={"key": "CHUNK_SIZE", "value": 2000},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403
    assert "cannot be overridden" in resp.json()["detail"].lower()


def test_org_bulk_update_rejects_app_only_key(client, db):
    root = setup_root_org(db)
    create_user(db, "adminuser10", "pass123", UserRole.admin, root.id)
    token = get_token(client, "adminuser10", "pass123")

    resp = client.put(
        f"/api/admin/orgs/{root.id}/settings",
        json={"settings": [
            {"key": "CHUNK_SIZE", "value": 2000},
            {"key": "RETRIEVAL_TOP_K", "value": 25},
        ]},
        headers={"Authorization": f"Bearer {token}"},
    )
    results = {r["key"]: r for r in resp.json()["results"]}
    assert results["CHUNK_SIZE"]["status"] == "error"
    assert results["RETRIEVAL_TOP_K"]["status"] == "ok"


# ---------------------------------------------------------------------------
# 7. Validation errors return 422
# ---------------------------------------------------------------------------

def test_validation_error_returns_422(client, db):
    root = setup_root_org(db)
    create_user(db, "superadmin7", "pass123", UserRole.super_admin, root.id)
    token = get_token(client, "superadmin7", "pass123")

    resp = client.post(
        "/api/admin/settings/RETRIEVAL_TOP_K",
        json={"key": "RETRIEVAL_TOP_K", "value": "not_a_number"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422


def test_validation_min_max_returns_422(client, db):
    root = setup_root_org(db)
    create_user(db, "superadmin8", "pass123", UserRole.super_admin, root.id)
    token = get_token(client, "superadmin8", "pass123")

    resp = client.post(
        "/api/admin/settings/RETRIEVAL_TOP_K",
        json={"key": "RETRIEVAL_TOP_K", "value": 0},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422


def test_unknown_key_returns_422(client, db):
    root = setup_root_org(db)
    create_user(db, "superadmin9", "pass123", UserRole.super_admin, root.id)
    token = get_token(client, "superadmin9", "pass123")

    resp = client.post(
        "/api/admin/settings/NONEXISTENT_KEY",
        json={"key": "NONEXISTENT_KEY", "value": 42},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 8. Schema endpoints
# ---------------------------------------------------------------------------

def test_app_settings_schema(client, db):
    root = setup_root_org(db)
    create_user(db, "superadmin10", "pass123", UserRole.super_admin, root.id)
    token = get_token(client, "superadmin10", "pass123")

    resp = client.get("/api/admin/settings/schema", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    data = resp.json()
    assert "settings" in data
    assert len(data["settings"]) > 0
    for item in data["settings"]:
        assert "key" in item
        assert "value_type" in item
        assert "category" in item
        assert "scope" in item


def test_org_settings_schema(client, db):
    root = setup_root_org(db)
    create_user(db, "adminuser11", "pass123", UserRole.admin, root.id)
    token = get_token(client, "adminuser11", "pass123")

    resp = client.get(
        f"/api/admin/orgs/{root.id}/settings/schema",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    # Only org-overridable keys
    for item in data["settings"]:
        assert item["scope"] == "org"


# ---------------------------------------------------------------------------
# 9. 404 for unknown org
# ---------------------------------------------------------------------------

def test_org_settings_unknown_org_returns_404(client, db):
    root = setup_root_org(db)
    create_user(db, "adminuser12", "pass123", UserRole.admin, root.id)
    token = get_token(client, "adminuser12", "pass123")

    resp = client.get(
        "/api/admin/orgs/99999/settings",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 404
