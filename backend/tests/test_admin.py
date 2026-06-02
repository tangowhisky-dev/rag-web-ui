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

def create_user(db, username: str, password: str, role: UserRole) -> User:
    """Create a User with the given role directly in the DB."""
    from app.core.security import get_password_hash
    user = User(
        username=username,
        email=f"{username}@example.com",
        hashed_password=get_password_hash(password),
        is_active=True,
        role=role,
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
    create_user(db, "adminhelper", "pass123", UserRole.admin)
    return get_token(client, "adminhelper", "pass123")


# ---------------------------------------------------------------------------
# Org tests
# ---------------------------------------------------------------------------

def test_admin_list_orgs_empty(client, db):
    token = get_admin_token(client, db)
    resp = client.get("/api/admin/orgs", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json() == []


def test_admin_create_org(client, db):
    token = get_admin_token(client, db)
    resp = client.post(
        "/api/admin/orgs",
        json={"name": "Acme"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "Acme"
    assert data["parent_id"] is None
    assert data["path"] == f"/{data['id']}"


def test_admin_create_child_org(client, db):
    token = get_admin_token(client, db)

    # Create parent first
    parent_resp = client.post(
        "/api/admin/orgs",
        json={"name": "Parent"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert parent_resp.status_code == 201
    parent_id = parent_resp.json()["id"]
    parent_path = parent_resp.json()["path"]

    # Create child
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
    client.post("/api/admin/orgs", json={"name": "DuplicateOrg"},
                headers={"Authorization": f"Bearer {token}"})

    resp = client.post("/api/admin/orgs", json={"name": "DuplicateOrg"},
                       headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 400, resp.text
    assert "already exists" in resp.json()["detail"].lower()


def test_admin_update_org_duplicate_name_returns_400(client, db):
    """Renaming an org to a name already taken by another org must return 400."""
    token = get_admin_token(client, db)
    client.post("/api/admin/orgs", json={"name": "Taken"},
                headers={"Authorization": f"Bearer {token}"})
    second = client.post("/api/admin/orgs", json={"name": "Second"},
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


def test_admin_update_user_role(client, db):
    token = get_admin_token(client, db)
    user = create_user(db, "roletest", "pass123", UserRole.user)

    resp = client.patch(
        f"/api/admin/users/{user.id}",
        json={"role": "admin"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["role"] == "admin"


def test_admin_deactivate_user(client, db):
    token = get_admin_token(client, db)
    user = create_user(db, "deactivatetest", "pass123", UserRole.user)

    resp = client.delete(
        f"/api/admin/users/{user.id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 204, resp.text

    # User should still exist but is_active=False
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
    client.post("/api/admin/users",
                json={"username": "user_a", "email": "dup@example.com",
                      "password": "pass123", "role": "user"},
                headers={"Authorization": f"Bearer {token}"})

    resp = client.post("/api/admin/users",
                       json={"username": "user_b", "email": "dup@example.com",
                             "password": "pass123", "role": "user"},
                       headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 400, resp.text
    assert "email" in resp.json()["detail"].lower()


def test_admin_create_user_duplicate_username_returns_400(client, db):
    """Creating a user with an already-taken username must return 400."""
    token = get_admin_token(client, db)
    client.post("/api/admin/users",
                json={"username": "dupuser", "email": "a@example.com",
                      "password": "pass123", "role": "user"},
                headers={"Authorization": f"Bearer {token}"})

    resp = client.post("/api/admin/users",
                       json={"username": "dupuser", "email": "b@example.com",
                             "password": "pass123", "role": "user"},
                       headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 400, resp.text
    assert "username" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# LLM config admin tests
# ---------------------------------------------------------------------------

def test_admin_get_llm_config_empty(client, db):
    """GET /orgs/{id}/llm-config returns 404 when no config row exists."""
    token = get_admin_token(client, db)
    org = create_org(db, "LLMOrg1")

    resp = client.get(
        f"/api/admin/orgs/{org.id}/llm-config",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 404, resp.text


def test_admin_put_llm_config_creates(client, db):
    """PUT creates a config row and GET returns saved values."""
    token = get_admin_token(client, db)
    org = create_org(db, "LLMOrg2")

    resp = client.put(
        f"/api/admin/orgs/{org.id}/llm-config",
        json={"api_base": "https://llm.example.com", "model_name": "gpt-4o", "query_model": "gpt-4o-mini"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["api_base"] == "https://llm.example.com"
    assert data["model_name"] == "gpt-4o"
    assert data["query_model"] == "gpt-4o-mini"
    assert data["org_id"] == org.id

    get_resp = client.get(
        f"/api/admin/orgs/{org.id}/llm-config",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert get_resp.status_code == 200, get_resp.text
    assert get_resp.json()["api_base"] == "https://llm.example.com"


def test_admin_put_llm_config_updates(client, db):
    """Second PUT with different values overwrites the existing config."""
    token = get_admin_token(client, db)
    org = create_org(db, "LLMOrg3")

    client.put(
        f"/api/admin/orgs/{org.id}/llm-config",
        json={"api_base": "https://old.example.com", "model_name": "old-model"},
        headers={"Authorization": f"Bearer {token}"},
    )

    resp = client.put(
        f"/api/admin/orgs/{org.id}/llm-config",
        json={"api_base": "https://new.example.com", "model_name": "new-model"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["api_base"] == "https://new.example.com"
    assert data["model_name"] == "new-model"


def test_admin_llm_config_rejects_regular_user(client, db):
    """PUT /orgs/{id}/llm-config returns 403 for non-admin users."""
    create_user(db, "plainuser2", "pass123", UserRole.user)
    token = get_token(client, "plainuser2", "pass123")
    org = create_org(db, "LLMOrg4")

    resp = client.put(
        f"/api/admin/orgs/{org.id}/llm-config",
        json={"api_base": "https://llm.example.com"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403, resp.text


# ── Unit tests: get_effective_llm_config ──────────────────────────────────────

def test_effective_llm_config_fallback(db):
    """When org_id is None, all values fall back to settings defaults."""
    from app.services.chat_service import get_effective_llm_config
    from app.core.config import settings

    cfg = get_effective_llm_config(None, db)
    assert cfg["api_base"] == settings.OPENAI_API_BASE
    assert cfg["model_name"] == settings.OPENAI_MODEL
    assert cfg["query_model"] == settings.effective_query_model


def test_effective_llm_config_org_override(db):
    """When an OrgLLMConfig row exists for the org, those values are returned."""
    from app.services.chat_service import get_effective_llm_config
    from app.models.org_llm_config import OrgLLMConfig
    import app.models.org_llm_config  # noqa: ensure table is registered

    org = create_org(db, "LLMConfigOrg5")
    row = OrgLLMConfig(
        org_id=org.id,
        api_base="https://custom.example.com",
        model_name="custom-model",
        query_model="custom-query-model",
    )
    db.add(row)
    db.commit()

    cfg = get_effective_llm_config(org.id, db)
    assert cfg["api_base"] == "https://custom.example.com"
    assert cfg["model_name"] == "custom-model"
    assert cfg["query_model"] == "custom-query-model"
