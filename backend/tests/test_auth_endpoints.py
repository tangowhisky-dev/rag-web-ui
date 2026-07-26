"""Tests for /api/auth endpoints not already covered by test_jwt_auth.py and test_admin.py."""
import pytest
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from app.main import app as fastapi_app  # noqa: conftest must run first

import app.db.session as _session_mod
from app.db.session import get_db
from app.models.base import Base  # noqa
from app.models.user import User, UserRole
from app.core import security

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


def _create_user(db, username, password, role=UserRole.user):
    user = User(
        username=username,
        email=f"{username}@example.com",
        hashed_password=security.get_password_hash(password),
        is_active=True,
        role=role,
    )
    db.add(user)
    db.commit()
    return user


def _get_token(client, username, password):
    resp = client.post("/api/auth/token", data={"username": username, "password": password})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def test_register_is_disabled(client):
    resp = client.post("/api/auth/register", json={
        "username": "newuser",
        "email": "newuser@example.com",
        "password": "Pass12345",
    })
    assert resp.status_code == 403


def test_change_password_success(client, db):
    _create_user(db, "changer", "OldPass123", role=UserRole.user)
    token = _get_token(client, "changer", "OldPass123")

    resp = client.post(
        "/api/auth/change-password",
        json={"current_password": "OldPass123", "new_password": "NewPass456"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200

    # Old token should no longer work because token_version was bumped.
    resp2 = client.post(
        "/api/auth/test-token",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp2.status_code == 401

    # New password should allow login.
    resp3 = client.post("/api/auth/token", data={"username": "changer", "password": "NewPass456"})
    assert resp3.status_code == 200


def test_change_password_wrong_current(client, db):
    _create_user(db, "changer2", "OldPass123")
    token = _get_token(client, "changer2", "OldPass123")

    resp = client.post(
        "/api/auth/change-password",
        json={"current_password": "WrongPass", "new_password": "NewPass456"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 400


def test_change_password_unauthenticated(client):
    resp = client.post(
        "/api/auth/change-password",
        json={"current_password": "x", "new_password": "y"},
    )
    assert resp.status_code in (401, 403)


def test_logout_clears_cookie(client):
    resp = client.post("/api/auth/logout")
    assert resp.status_code == 200
    cookie_header = resp.headers.get("set-cookie", "")
    assert "token=" in cookie_header.lower() or "max-age=0" in cookie_header.lower() or resp.cookies.get("token") is not None
