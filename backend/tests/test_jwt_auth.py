"""
test_jwt_auth.py — Tests for JWT auth layer with role and org_id claims.

Covers:
  1. Login token carries `role` and `org_id` claims
  2. GET /api/auth/test-token returns `role` field in response

Admin-only access-control assertions live in test_admin.py.
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


def _create_and_login(client, db, username="testuser", password="testpass123",
                      email=None, role=UserRole.user):
    """Create a user directly in the DB then login, return token."""
    if email is None:
        email = f"{username}@example.com"

    user = User(
        username=username,
        email=email,
        hashed_password=security.get_password_hash(password),
        is_active=True,
        role=role,
    )
    db.add(user)
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

def test_login_token_has_role_and_org_id(client, db):
    """Token returned by /api/auth/token must carry `role` and `org_id` claims."""
    token = _create_and_login(client, db)

    payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])

    assert "role" in payload, "JWT payload missing `role` claim"
    assert payload["role"] == "user"
    assert "org_id" in payload, "JWT payload missing `org_id` claim"


def test_get_current_user_returns_role(client, db):
    """GET /api/auth/test-token with valid Bearer token must return JSON with `role`."""
    token = _create_and_login(client, db)

    resp = client.get("/api/auth/test-token",
                      headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200, resp.text

    data = resp.json()
    assert "role" in data, f"Response missing `role` field: {data}"
    assert data["role"] == "user"



