"""
test_admin.py — Tests for the require_admin() guard and /api/v1/auth/admin-only endpoint.

Covers:
  1. Regular user (role=user) is rejected with 403
  2. Admin user (role=admin) is accepted with 200
  3. Super-admin user (role=super_admin) is accepted with 200
  4. Unauthenticated request is rejected with 401 or 403
"""
import pytest
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

# conftest.py has already patched MySQL dialect types and app.db.session.
from app.main import app as fastapi_app  # noqa: conftest must run first

import app.db.session as _session_mod
from app.db.session import get_db
from app.models.user import User, UserRole
from app.models.base import Base  # noqa
import app.models.user  # noqa
import app.models.knowledge  # noqa
import app.models.chat  # noqa
import app.models.organisation  # noqa
from app.models.organisation import Organisation

engine = _session_mod.engine
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_user(db, username: str, password: str, role: UserRole, org_id=None) -> User:
    """Create a User with the given role directly in the DB."""
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


def get_token(client, username: str, password: str) -> str:
    """POST to /api/v1/auth/token and return the access_token."""
    resp = client.post("/api/auth/token", data={"username": username, "password": password})
    assert resp.status_code == 200, f"login failed: {resp.text}"
    return resp.json()["access_token"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_admin_only_rejects_regular_user(client, db):
    """/api/auth/admin-only must return 403 for a user with role=user."""
    create_user(db, "regularuser", "pass123", UserRole.user)
    token = get_token(client, "regularuser", "pass123")

    resp = client.get("/api/auth/admin-only",
                      headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403, f"Expected 403, got {resp.status_code}: {resp.text}"


def test_admin_only_accepts_admin(client, db):
    """/api/auth/admin-only must return 200 for a user with role=admin."""
    create_user(db, "adminuser", "pass123", UserRole.admin)
    token = get_token(client, "adminuser", "pass123")

    resp = client.get("/api/auth/admin-only",
                      headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    assert resp.json()["role"] == "admin"


def test_admin_only_accepts_super_admin(client, db):
    """/api/auth/admin-only must return 200 for a user with role=super_admin."""
    create_user(db, "superadmin", "pass123", UserRole.super_admin)
    token = get_token(client, "superadmin", "pass123")

    resp = client.get("/api/auth/admin-only",
                      headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    assert resp.json()["role"] == "super_admin"


def test_admin_only_rejects_unauthenticated(client):
    """/api/auth/admin-only must reject requests with no token (401 or 403)."""
    resp = client.get("/api/auth/admin-only")
    assert resp.status_code in (401, 403), \
        f"Expected 401 or 403, got {resp.status_code}: {resp.text}"


# ---------------------------------------------------------------------------
# Org helpers
# ---------------------------------------------------------------------------

def create_org(db, name: str, parent_id=None) -> Organisation:
    """Create an Organisation directly in the DB."""
    if parent_id is None:
        root = db.query(Organisation).filter(Organisation.parent_id.is_(None)).first()
        parent_id = root.id if root else None
    org = Organisation(name=name, parent_id=parent_id)
    db.add(org)
    db.flush()
    if parent_id is not None:
        parent = db.query(Organisation).filter(Organisation.id == parent_id).first()
        org.path = f"{parent.path}/{org.id}"
    else:
        org.path = f"/{org.id}"
    db.commit()
    db.refresh(org)
    return org


def get_admin_token(client, db) -> str:
    # Create or get root org if it doesn't exist
    root_org = db.query(Organisation).filter(Organisation.parent_id.is_(None)).first()
    if not root_org:
        root_org = Organisation(name="Root", parent_id=None, path="/1")
        db.add(root_org)
        db.commit()

    # Create or get admin helper user with org_id
    from app.core.security import get_password_hash
    admin = db.query(User).filter(User.username == "adminhelper").first()
    if not admin:
        admin = User(
            username="adminhelper",
            email="adminhelper@example.com",
            hashed_password=get_password_hash("pass123"),
            is_active=True,
            role=UserRole.admin,
            org_id=root_org.id,
        )
        db.add(admin)
        db.commit()
    return get_token(client, "adminhelper", "pass123")


# ---------------------------------------------------------------------------
# Org tests
# ---------------------------------------------------------------------------

def test_admin_list_orgs_empty(client, db):
    """List orgs should return at least the seed org if no superadmin exists."""
    # The seed function creates a "Root Organization" on startup.
    # If no superadmin exists, the seed creates one too (get_admin_token below).
    # So we can't expect an empty list — just verify the response is valid.
    token = get_admin_token(client, db)
    resp = client.get("/api/admin/orgs", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_admin_create_org(client, db):
    token = get_admin_token(client, db)
    # Use root org as parent
    root_org = db.query(Organisation).filter(Organisation.parent_id.is_(None)).first()
    resp = client.post(
        "/api/admin/orgs",
        json={"name": "Acme", "parent_id": root_org.id},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "Acme"
    assert data["parent_id"] == root_org.id
    assert root_org.path in data["path"]


def test_admin_create_child_org(client, db):
    token = get_admin_token(client, db)
    root_org = db.query(Organisation).filter(Organisation.parent_id.is_(None)).first()

    # Create parent via helper (root org)
    parent_org = create_org(db, "Parent")
    parent_id = parent_org.id
    parent_path = parent_org.path

    # Create child via API
    child_resp = client.post(
        "/api/admin/orgs",
        json={"name": "Child", "parent_id": parent_id},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert child_resp.status_code == 201
    child = child_resp.json()
    assert child["parent_id"] == parent_id
    # Path must include parent id
    assert str(parent_id) in child["path"]
    assert child["path"] == f"{parent_path}/{child['id']}"


def test_admin_update_org(client, db):
    token = get_admin_token(client, db)
    org = create_org(db, "Original")

    resp = client.patch(
        f"/api/admin/orgs/{org.id}",
        json={"name": "Updated"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["name"] == "Updated"


def test_admin_delete_org(client, db):
    token = get_admin_token(client, db)
    org = create_org(db, "ToDelete")

    resp = client.delete(
        f"/api/admin/orgs/{org.id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 204

    # Confirm gone
    list_resp = client.get("/api/admin/orgs", headers={"Authorization": f"Bearer {token}"})
    ids = [o["id"] for o in list_resp.json()]
    assert org.id not in ids


def test_admin_delete_org_with_children_returns_409(client, db):
    token = get_admin_token(client, db)
    parent = create_org(db, "ParentOrg")
    create_org(db, "ChildOrg", parent_id=parent.id)

    resp = client.delete(
        f"/api/admin/orgs/{parent.id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 409


def test_admin_orgs_rejects_regular_user(client, db):
    create_user(db, "regularuser2", "pass123", UserRole.user)
    token = get_token(client, "regularuser2", "pass123")

    resp = client.get("/api/admin/orgs", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403


def test_admin_create_org_duplicate_name_returns_400(client, db):
    """Creating two orgs with the same name must return 400, not 500."""
    token = get_admin_token(client, db)
    root_org = db.query(Organisation).filter(Organisation.parent_id.is_(None)).first()
    client.post("/api/admin/orgs",
                json={"name": "DuplicateOrg", "parent_id": root_org.id},
                headers={"Authorization": f"Bearer {token}"})

    resp = client.post("/api/admin/orgs",
                       json={"name": "DuplicateOrg", "parent_id": root_org.id},
                       headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 400, resp.text
    assert "already exists" in resp.json()["detail"].lower()


def test_admin_update_org_duplicate_name_returns_400(client, db):
    """Renaming an org to a name already taken by another org must return 400."""
    token = get_admin_token(client, db)
    root_org = db.query(Organisation).filter(Organisation.parent_id.is_(None)).first()
    client.post("/api/admin/orgs",
                json={"name": "Taken", "parent_id": root_org.id},
                headers={"Authorization": f"Bearer {token}"})
    second = client.post("/api/admin/orgs",
                         json={"name": "Second", "parent_id": root_org.id},
                         headers={"Authorization": f"Bearer {token}"}).json()

    resp = client.patch(
        f"/api/admin/orgs/{second['id']}",
        json={"name": "Taken"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 400, resp.text
    assert "already exists" in resp.json()["detail"].lower()


def test_admin_update_org_same_name_is_ok(client, db):
    """PATCHing an org with its current name must succeed (no false duplicate)."""
    token = get_admin_token(client, db)
    org = create_org(db, "StableName")

    resp = client.patch(
        f"/api/admin/orgs/{org.id}",
        json={"name": "StableName"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200, resp.text


# ---------------------------------------------------------------------------
# User tests
# ---------------------------------------------------------------------------

def test_admin_list_users(client, db):
    token = get_admin_token(client, db)
    resp = client.get("/api/admin/users", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    # At minimum the admin helper user exists
    assert isinstance(resp.json(), list)


def test_admin_create_user_with_role_and_org(client, db):
    token = get_admin_token(client, db)
    org = create_org(db, "TestOrg")

    resp = client.post(
        "/api/admin/users",
        json={"username": "newuser", "email": "newuser@example.com", "password": "secret123",
              "role": "user", "org_id": org.id},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["username"] == "newuser"
    assert data["role"] == "user"
    assert data["org_id"] == org.id


def test_admin_cannot_promote_user_to_admin(client, db):
    """An admin user should NOT be able to promote another user to admin role."""
    token = get_admin_token(client, db)
    user = create_user(db, "roletest", "pass123", UserRole.user)

    resp = client.patch(
        f"/api/admin/users/{user.id}",
        json={"role": "admin"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403, resp.text


def test_admin_cannot_promote_user_to_super_admin(client, db):
    """An admin user should NOT be able to promote another user to super_admin role."""
    token = get_admin_token(client, db)
    user = create_user(db, "roletest2", "pass123", UserRole.user)

    resp = client.patch(
        f"/api/admin/users/{user.id}",
        json={"role": "super_admin"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403, resp.text


def test_super_admin_can_promote_user_to_admin(client, db):
    """A super_admin user SHOULD be able to promote another user to admin role."""
    create_user(db, "superadmin", "pass123", UserRole.super_admin)
    token = get_token(client, "superadmin", "pass123")
    user = create_user(db, "roletest3", "pass123", UserRole.user)

    resp = client.patch(
        f"/api/admin/users/{user.id}",
        json={"role": "admin"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["role"] == "admin"


def test_admin_deactivate_user(client, db):
    token = get_admin_token(client, db)
    # Create user with the same org as the admin helper
    root_org = db.query(Organisation).filter(Organisation.parent_id.is_(None)).first()
    user = create_user(db, "deactivatetest", "pass123", UserRole.user, org_id=root_org.id)

    # Use PATCH to deactivate
    resp = client.patch(
        f"/api/admin/users/{user.id}",
        json={"is_active": False},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["is_active"] is False

    # User should still exist in list
    list_resp = client.get("/api/admin/users", headers={"Authorization": f"Bearer {token}"})
    users = {u["id"]: u for u in list_resp.json()}
    assert users[user.id]["is_active"] is False


def test_admin_users_rejects_regular_user(client, db):
    create_user(db, "plainuser", "pass123", UserRole.user)
    token = get_token(client, "plainuser", "pass123")

    resp = client.get("/api/admin/users", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403


def test_admin_create_user_duplicate_email_returns_400(client, db):
    """Creating a user with an already-registered email must return 400."""
    token = get_admin_token(client, db)
    root_org = db.query(Organisation).filter(Organisation.parent_id.is_(None)).first()
    client.post("/api/admin/users",
                json={"username": "user_a", "email": "dup@example.com",
                      "password": "pass1234", "role": "user", "org_id": root_org.id},
                headers={"Authorization": f"Bearer {token}"})

    resp = client.post("/api/admin/users",
                       json={"username": "user_b", "email": "dup@example.com",
                             "password": "pass1234", "role": "user", "org_id": root_org.id},
                       headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 400, resp.text
    assert "email" in resp.json()["detail"].lower()


def test_admin_create_user_duplicate_username_returns_400(client, db):
    """Creating a user with an already-taken username must return 400."""
    token = get_admin_token(client, db)
    root_org = db.query(Organisation).filter(Organisation.parent_id.is_(None)).first()
    client.post("/api/admin/users",
                json={"username": "dupuser", "email": "a@example.com",
                      "password": "pass1234", "role": "user", "org_id": root_org.id},
                headers={"Authorization": f"Bearer {token}"})

    resp = client.post("/api/admin/users",
                       json={"username": "dupuser", "email": "b@example.com",
                             "password": "pass1234", "role": "user", "org_id": root_org.id},
                       headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 400, resp.text
    assert "username" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# User deletion and deactivation tests
# ---------------------------------------------------------------------------

def test_admin_cannot_delete_user(client, db):
    """An admin user should NOT be able to permanently delete a user (only super admin)."""
    token = get_admin_token(client, db)
    user = create_user(db, "deleteblocktest", "pass1234", UserRole.user)

    resp = client.delete(
        f"/api/admin/users/{user.id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403, resp.text


def test_super_admin_can_delete_user(client, db):
    """A super_admin user SHOULD be able to permanently delete a user."""
    create_user(db, "superadmin", "pass123", UserRole.super_admin)
    token = get_token(client, "superadmin", "pass123")
    user = create_user(db, "deletetest2", "pass123", UserRole.user)

    resp = client.delete(
        f"/api/admin/users/{user.id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["username"] == "deletetest2"

    # User should no longer exist in list
    list_resp = client.get("/api/admin/users", headers={"Authorization": f"Bearer {token}"})
    users = {u["id"]: u for u in list_resp.json()}
    assert user.id not in users


def test_reactivate_user(client, db):
    """Reactivating a deactivated user should set is_active=True."""
    token = get_admin_token(client, db)
    user = create_user(db, "reactivatetest", "pass123", UserRole.user)

    # Deactivate first
    client.patch(
        f"/api/admin/users/{user.id}",
        json={"is_active": False},
        headers={"Authorization": f"Bearer {token}"},
    )

    # Reactivate
    resp = client.patch(
        f"/api/admin/users/{user.id}",
        json={"is_active": True},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["is_active"] is True


def test_deactivated_user_cannot_login(client, db):
    """A deactivated (is_active=False) user should not be able to login."""
    token = get_admin_token(client, db)
    user = create_user(db, "deactivatedtest", "pass123", UserRole.user)

    # Deactivate the user
    client.patch(
        f"/api/admin/users/{user.id}",
        json={"is_active": False},
        headers={"Authorization": f"Bearer {token}"},
    )

    # Login should fail
    resp = client.post("/api/auth/token",
                       data={"username": "deactivatedtest", "password": "pass123"})
    assert resp.status_code == 401, resp.text
    assert "Inactive user" in resp.json()["detail"]


def test_deleted_user_cannot_login(client, db):
    """A permanently deleted user should not be able to login (user no longer exists)."""
    create_user(db, "superadmin4", "pass123", UserRole.super_admin)
    token = get_token(client, "superadmin4", "pass123")
    user = create_user(db, "deletedlogin", "pass123", UserRole.user)

    # Permanently delete the user
    client.delete(
        f"/api/admin/users/{user.id}",
        headers={"Authorization": f"Bearer {token}"},
    )

    # Try to login — should fail (user doesn't exist)
    resp = client.post("/api/auth/token",
                       data={"username": "deletedlogin", "password": "pass123"})
    assert resp.status_code == 401, resp.text


def test_deactivated_user_in_list_shows_is_active_false(client, db):
    """A deactivated user should appear in the list with is_active=False."""
    create_user(db, "superadmin5", "pass123", UserRole.super_admin)
    token = get_token(client, "superadmin5", "pass123")
    user = create_user(db, "deactivatedlist", "pass123", UserRole.user)

    # Deactivate
    client.patch(
        f"/api/admin/users/{user.id}",
        json={"is_active": False},
        headers={"Authorization": f"Bearer {token}"},
    )

    # List should still include the deactivated user
    list_resp = client.get("/api/admin/users", headers={"Authorization": f"Bearer {token}"})
    assert list_resp.status_code == 200
    users = {u["id"]: u for u in list_resp.json()}
    assert users[user.id]["is_active"] is False


# ---------------------------------------------------------------------------
# LLM config is now managed via the unified settings API.
# The legacy /orgs/{id}/llm-config endpoints have been removed.
# OrgLLMConfig table has been dropped; data migrated to settings table.
# See test_settings_phase*.py for settings API tests.
# ---------------------------------------------------------------------------


# ── Unit tests: get_effective_llm_config ──────────────────────────────────────

def test_effective_llm_config_fallback(db):
    """When org_id is None, all values fall back to settings defaults."""
    from app.services.chat import get_effective_llm_config
    from app.core.config import settings

    cfg = get_effective_llm_config(None, db)
    assert cfg["api_base"] == settings.OPENAI_API_BASE
    assert cfg["model_name"] == settings.OPENAI_MODEL
    assert cfg["query_model"] == settings.QUERY_MODEL or settings.OPENAI_MODEL


def test_effective_llm_config_org_override(db):
    """When an org override exists in the settings table, those values are returned."""
    from app.services.chat import get_effective_llm_config
    from app.services.settings_service import upsert_org_setting, clear_cache

    clear_cache()
    org = create_org(db, "LLMConfigOrg5")
    upsert_org_setting(db, org.id, "OPENAI_API_BASE", "https://custom.example.com")
    upsert_org_setting(db, org.id, "OPENAI_MODEL", "custom-model")
    upsert_org_setting(db, org.id, "QUERY_MODEL", "custom-query-model")

    cfg = get_effective_llm_config(org.id, db)
    assert cfg["api_base"] == "https://custom.example.com"
    assert cfg["model_name"] == "custom-model"
    assert cfg["query_model"] == "custom-query-model"
