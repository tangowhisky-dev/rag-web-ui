"""
test_ingestion_features.py — Tests for QueryExpander and OrgAbbreviation admin endpoints.

Groups:
  1. QueryExpander unit tests (no DB)
  2. Admin abbreviation CRUD tests (TestClient + SQLite via conftest)
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
import app.models.org_llm_config  # noqa
from app.models.organisation import Organisation
from app.core.security import get_password_hash

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


def create_org(db, name: str) -> Organisation:
    org = Organisation(name=name)
    db.add(org)
    db.flush()
    org.path = f"/{org.id}"
    db.commit()
    db.refresh(org)
    return org


def get_token(client, username: str, password: str) -> str:
    resp = client.post("/api/auth/token", data={"username": username, "password": password})
    assert resp.status_code == 200, f"login failed: {resp.text}"
    return resp.json()["access_token"]


def get_admin_token(client, db) -> str:
    create_user(db, "adminuser", "pass123", UserRole.admin)
    return get_token(client, "adminuser", "pass123")


# ---------------------------------------------------------------------------
# 1. QueryExpander unit tests (no DB)
# ---------------------------------------------------------------------------

from app.services.query_expander import expand


def test_query_expander_basic():
    result = expand("what is KB?", {"KB": "Knowledge Base"})
    assert result == "what is Knowledge Base?"


def test_query_expander_case_insensitive():
    result = expand("kb docs", {"KB": "Knowledge Base"})
    assert result == "Knowledge Base docs"


def test_query_expander_word_boundary():
    """MKBS must NOT be replaced; only standalone KB should expand."""
    result = expand("MKBS and KB", {"KB": "Knowledge Base"})
    assert result == "MKBS and Knowledge Base"


def test_query_expander_empty_dict():
    result = expand("hello KB", {})
    assert result == "hello KB"


def test_query_expander_multiple():
    result = expand("KB and DR plan", {"KB": "Knowledge Base", "DR": "Disaster Recovery"})
    assert result == "Knowledge Base and Disaster Recovery plan"


# ---------------------------------------------------------------------------
# 2. Admin abbreviation CRUD tests
# ---------------------------------------------------------------------------

def test_create_abbreviation(client, db):
    token = get_admin_token(client, db)
    org = create_org(db, "TestOrg")

    resp = client.post(
        f"/api/admin/orgs/{org.id}/abbreviations",
        json={"short": "KB", "expansion": "Knowledge Base"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["org_id"] == org.id
    assert data["short"] == "KB"
    assert data["expansion"] == "Knowledge Base"
    assert "id" in data


def test_list_abbreviations(client, db):
    token = get_admin_token(client, db)
    org = create_org(db, "TestOrg2")

    client.post(
        f"/api/admin/orgs/{org.id}/abbreviations",
        json={"short": "KB", "expansion": "Knowledge Base"},
        headers={"Authorization": f"Bearer {token}"},
    )

    resp = client.get(
        f"/api/admin/orgs/{org.id}/abbreviations",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200, resp.text
    items = resp.json()
    assert len(items) == 1
    assert items[0]["short"] == "KB"


def test_delete_abbreviation(client, db):
    token = get_admin_token(client, db)
    org = create_org(db, "TestOrg3")

    create_resp = client.post(
        f"/api/admin/orgs/{org.id}/abbreviations",
        json={"short": "KB", "expansion": "Knowledge Base"},
        headers={"Authorization": f"Bearer {token}"},
    )
    abbrev_id = create_resp.json()["id"]

    del_resp = client.delete(
        f"/api/admin/orgs/{org.id}/abbreviations/{abbrev_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert del_resp.status_code == 204, del_resp.text

    list_resp = client.get(
        f"/api/admin/orgs/{org.id}/abbreviations",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert list_resp.json() == []


def test_create_abbreviation_duplicate_returns_409(client, db):
    token = get_admin_token(client, db)
    org = create_org(db, "TestOrg4")

    client.post(
        f"/api/admin/orgs/{org.id}/abbreviations",
        json={"short": "KB", "expansion": "Knowledge Base"},
        headers={"Authorization": f"Bearer {token}"},
    )
    resp = client.post(
        f"/api/admin/orgs/{org.id}/abbreviations",
        json={"short": "KB", "expansion": "Knowledge Bases"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 409, resp.text


def test_admin_abbreviation_rejects_regular_user(client, db):
    create_user(db, "regularuser", "pass123", UserRole.user)
    token = get_token(client, "regularuser", "pass123")
    org = create_org(db, "TestOrg5")

    resp = client.post(
        f"/api/admin/orgs/{org.id}/abbreviations",
        json={"short": "KB", "expansion": "Knowledge Base"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403, resp.text
