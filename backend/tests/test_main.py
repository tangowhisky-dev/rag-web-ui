"""Tests for top-level FastAPI application endpoints."""
import pytest
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

# conftest.py patches the DB before any app import.
from app.main import app as fastapi_app  # noqa
import app.main as main_module
from app.models.base import Base  # noqa
from app.models.organisation import Organisation
from app.models.user import User, UserRole
from app.core import security
from app.db.session import get_db
import app.db.session as _session_mod

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
def client(monkeypatch):
    monkeypatch.setattr(main_module, "_seed_root_org_and_superadmin", lambda: None)
    fastapi_app.dependency_overrides[get_db] = override_get_db
    with TestClient(fastapi_app) as c:
        yield c
    fastapi_app.dependency_overrides.clear()


def _create_admin_and_get_token(client, db):
    org = Organisation(name="Root", parent_id=None, path="/1")
    db.add(org)
    db.commit()
    user = User(
        username="admin",
        email="admin@example.com",
        hashed_password=security.get_password_hash("pass123"),
        is_active=True,
        role=UserRole.super_admin,
        org_id=org.id,
    )
    db.add(user)
    db.commit()
    resp = client.post("/api/auth/token", data={"username": "admin", "password": "pass123"})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def test_root_endpoint():
    with TestClient(fastapi_app) as client:
        resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["message"].startswith("Welcome")


def test_health_check_endpoint():
    with TestClient(fastapi_app) as client:
        resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_client_config_endpoint(client, db):
    token = _create_admin_and_get_token(client, db)
    resp = client.get("/api/config", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    data = resp.json()
    assert "chunk_size" in data
    assert "chunk_overlap" in data
