"""
test_jwt_auth.py — Tests for JWT auth layer with role and org_id claims.

Covers:
  1. Login token carries `role` and `org_id` claims
  2. GET /api/auth/test-token returns `role` field in response
  3. /api/auth/admin-only rejects role=user with 403
  4. /api/auth/admin-only accepts role=admin with 200
"""
import pytest
from jose import jwt
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
from app.core import security
from app.core.config import settings

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


def _register_and_login(client, username="testuser", password="testpass123",
                        email=None, role=UserRole.user, db=None):
    """Register a user (optionally with a specific role) then login, return token."""
    if email is None:
        email = f"{username}@example.com"

    # Register via API (creates user with default role=user)
    resp = client.post("/api/auth/register", json={
        "email": email,
        "username": username,
        "password": password,
        "is_active": True,
        "is_superuser": False,
    })
    assert resp.status_code == 200, f"register failed: {resp.text}"

    # If a non-default role is requested, update directly in DB
    if role != UserRole.user and db is not None:
        user = db.query(User).filter(User.username == username).first()
        user.role = role
        db.commit()

    # Login
    token_resp = client.post("/api/auth/token", data={
        "username": username,
        "password": password,
    })
    assert token_resp.status_code == 200, f"login failed: {token_resp.text}"
    return token_resp.json()["access_token"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_login_token_has_role_and_org_id(client):
    """Token returned by /api/auth/token must carry `role` and `org_id` claims."""
    token = _register_and_login(client)

    payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])

    assert "role" in payload, "JWT payload missing `role` claim"
    assert payload["role"] == "user"
    assert "org_id" in payload, "JWT payload missing `org_id` claim"


def test_get_current_user_returns_role(client):
    """GET /api/auth/test-token with valid Bearer token must return JSON with `role`."""
    token = _register_and_login(client)

    resp = client.post("/api/auth/test-token",
                       headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200, resp.text

    data = resp.json()
    assert "role" in data, f"Response missing `role` field: {data}"
    assert data["role"] == "user"


def test_admin_only_rejects_user_role(client):
    """/api/auth/admin-only must return 403 for a user with role=user."""
    token = _register_and_login(client)

    resp = client.get("/api/auth/admin-only",
                      headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403, f"Expected 403, got {resp.status_code}: {resp.text}"


def test_admin_only_accepts_admin_role(client, db):
    """/api/auth/admin-only must return 200 for a user with role=admin."""
    token = _register_and_login(client, username="adminuser",
                                email="admin@example.com",
                                role=UserRole.admin, db=db)

    resp = client.get("/api/auth/admin-only",
                      headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

    data = resp.json()
    assert data["role"] == "admin"
