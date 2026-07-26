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
from app.models.knowledge import KnowledgeBase, ProcessingTask
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

def create_user(db, username: str, password: str, role: UserRole, org_id=None) -> User:
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


def create_org(db, name: str, parent_id=None) -> Organisation:
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

from app.services.retrieval import expand


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


# ---------------------------------------------------------------------------
# 3. Org ingestion status tests
# ---------------------------------------------------------------------------

def create_kb(db, name: str, org_id: int, user_id: int) -> KnowledgeBase:
    kb = KnowledgeBase(name=name, org_id=org_id, user_id=user_id)
    db.add(kb)
    db.commit()
    db.refresh(kb)
    return kb


def create_task(db, kb_id: int, status: str) -> ProcessingTask:
    task = ProcessingTask(knowledge_base_id=kb_id, status=status)
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def test_org_ingestion_status_idle_no_tasks(client, db):
    """Org with a KB but no processing tasks returns idle with zero counts."""
    token = get_admin_token(client, db)
    admin = db.query(User).filter(User.username == "adminuser").first()
    org = create_org(db, "OrgIdle")
    create_kb(db, "KB1", org.id, admin.id)

    resp = client.get(
        f"/api/admin/orgs/{org.id}/ingestion-status",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["org_id"] == org.id
    assert data["status"] == "idle"
    assert data["total_docs"] == 0
    assert data["pending_docs"] == 0
    assert data["completed_docs"] == 0
    assert data["failed_docs"] == 0
    assert data["last_run_at"] is None


def test_org_ingestion_status_running(client, db):
    """At least one processing task → status=running."""
    token = get_admin_token(client, db)
    admin = db.query(User).filter(User.username == "adminuser").first()
    org = create_org(db, "OrgRunning")
    kb = create_kb(db, "KB1", org.id, admin.id)
    create_task(db, kb.id, "completed")
    create_task(db, kb.id, "processing")

    resp = client.get(
        f"/api/admin/orgs/{org.id}/ingestion-status",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "running"
    assert data["total_docs"] == 2
    assert data["processing_docs"] == 1
    assert data["completed_docs"] == 1


def test_org_ingestion_status_failed(client, db):
    """No processing tasks but at least one failed → status=failed."""
    token = get_admin_token(client, db)
    admin = db.query(User).filter(User.username == "adminuser").first()
    org = create_org(db, "OrgFailed")
    kb = create_kb(db, "KB1", org.id, admin.id)
    create_task(db, kb.id, "completed")
    create_task(db, kb.id, "failed")

    resp = client.get(
        f"/api/admin/orgs/{org.id}/ingestion-status",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "failed"
    assert data["failed_docs"] == 1
    assert data["completed_docs"] == 1


def test_org_ingestion_status_completed(client, db):
    """All tasks completed → status=completed, last_run_at is set."""
    token = get_admin_token(client, db)
    admin = db.query(User).filter(User.username == "adminuser").first()
    org = create_org(db, "OrgCompleted")
    kb = create_kb(db, "KB1", org.id, admin.id)
    create_task(db, kb.id, "completed")
    create_task(db, kb.id, "completed")

    resp = client.get(
        f"/api/admin/orgs/{org.id}/ingestion-status",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "completed"
    assert data["total_docs"] == 2
    assert data["completed_docs"] == 2
    assert data["last_run_at"] is not None


def test_org_ingestion_status_multiple_kbs(client, db):
    """Tasks aggregated across multiple KBs for the same org."""
    token = get_admin_token(client, db)
    admin = db.query(User).filter(User.username == "adminuser").first()
    org = create_org(db, "OrgMultiKB")
    kb1 = create_kb(db, "KB1", org.id, admin.id)
    kb2 = create_kb(db, "KB2", org.id, admin.id)
    create_task(db, kb1.id, "completed")
    create_task(db, kb2.id, "pending")

    resp = client.get(
        f"/api/admin/orgs/{org.id}/ingestion-status",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["total_docs"] == 2
    assert data["completed_docs"] == 1
    assert data["pending_docs"] == 1
    # not all completed and no processing/failed → idle
    assert data["status"] == "idle"


def test_org_ingestion_status_not_found(client, db):
    """Unknown org_id returns 404."""
    token = get_admin_token(client, db)

    resp = client.get(
        "/api/admin/orgs/99999/ingestion-status",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 404, resp.text


def test_org_ingestion_status_rejects_non_admin(client, db):
    """Regular user receives 403."""
    create_user(db, "regularuser2", "pass123", UserRole.user)
    token = get_token(client, "regularuser2", "pass123")
    org = create_org(db, "OrgSecure")

    resp = client.get(
        f"/api/admin/orgs/{org.id}/ingestion-status",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403, resp.text


# ── Group 4: ProgressTimeout ──────────────────────────────────────────────
import asyncio as _asyncio
from app.services.infrastructure.progress_timeout import ProgressTimeout


@pytest.mark.asyncio
async def test_progress_timeout_fires():
    """Timeout callback fires when no ping arrives within the silence window."""
    fired = []
    async with ProgressTimeout(silence_seconds=1, on_timeout=lambda: fired.append(True)) as pt:
        await _asyncio.sleep(2)  # exceed 1-second silence — do NOT call pt.ping()
    assert fired, "on_timeout should have been called"


@pytest.mark.asyncio
async def test_progress_timeout_no_fire():
    """Timeout callback does NOT fire when pings arrive before the silence window."""
    fired = []
    async with ProgressTimeout(silence_seconds=2, on_timeout=lambda: fired.append(True)) as pt:
        for _ in range(5):
            pt.ping()
            await _asyncio.sleep(0.3)  # 5 x 0.3s = 1.5s total, silence never exceeds 2s
    assert not fired, "on_timeout must not fire when pings arrive in time"
