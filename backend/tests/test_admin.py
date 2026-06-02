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
