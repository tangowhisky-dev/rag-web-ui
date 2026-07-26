"""Tests for top-level FastAPI application endpoints."""
import pytest
from fastapi.testclient import TestClient

# conftest.py patches the DB before any app import.
from app.main import app as fastapi_app  # noqa
from app.models.base import Base  # noqa
import app.db.session as _session_mod

engine = _session_mod.engine


@pytest.fixture(autouse=True)
def reset_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


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


def test_client_config_endpoint():
    with TestClient(fastapi_app) as client:
        resp = client.get("/api/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "chunk_size" in data
    assert "chunk_overlap" in data
