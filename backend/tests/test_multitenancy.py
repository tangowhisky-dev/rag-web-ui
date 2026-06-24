"""Multi-tenancy isolation tests.

Verifies that org-scoped KB and Chat endpoints enforce org boundaries:
- Cross-org access returns 404.
- Same-org access returns 200.
- No-org (org_id=None) users see only their own resources.
- List endpoints exclude cross-org resources.

API prefixes (from app/api/api_v1/api.py + main.py):
  - /api/auth/...
  - /api/knowledge-base/...
  - /api/chat/...
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

# conftest.py has already patched MySQL dialect types and app.db.session.
from app.main import app as fastapi_app  # noqa: conftest must run first

import app.db.session as _session_mod
from app.db.session import get_db
from app.models.base import Base
from app.models.user import User, UserRole
from app.models.organisation import Organisation
import app.models.knowledge  # noqa: ensure tables are registered
import app.models.chat  # noqa: ensure tables are registered
from app.core.security import get_password_hash

engine = _session_mod.engine
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_org(db, name):
    org = Organisation(name=name)
    db.add(org)
    db.commit()
    db.refresh(org)
    return org


def _make_user(db, username, org_id=None, role=UserRole.user):
    user = User(
        username=username,
        email=f"{username}@test.com",
        hashed_password=get_password_hash("pass"),
        is_active=True,
        role=role,
        org_id=org_id,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _login(client, username, password="pass"):
    r = client.post("/api/auth/token", data={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _create_kb(client, token, name="kb"):
    r = client.post(
        "/api/knowledge-base",
        json={"name": name, "description": ""},
        headers=_auth(token),
    )
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _create_chat(client, token, kb_id, title="chat"):
    r = client.post(
        "/api/chat",
        json={"title": title, "knowledge_base_ids": [kb_id]},
        headers=_auth(token),
    )
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_db():
    """Drop and recreate all tables before each test."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client():
    fastapi_app.dependency_overrides[get_db] = override_get_db
    with TestClient(fastapi_app) as c:
        yield c
    fastapi_app.dependency_overrides.clear()


@pytest.fixture
def db():
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


# ── tests ─────────────────────────────────────────────────────────────────────

def test_cross_org_kb_get_returns_404(client, db):
    """User B in org2 cannot GET a KB created by User A in org1."""
    org1 = _make_org(db, "org1")
    org2 = _make_org(db, "org2")
    _make_user(db, "alice", org_id=org1.id)
    _make_user(db, "bob", org_id=org2.id)
    token_a = _login(client, "alice")
    token_b = _login(client, "bob")
    kb_id = _create_kb(client, token_a, "alice-kb")
    r = client.get(f"/api/knowledge-base/{kb_id}", headers=_auth(token_b))
    assert r.status_code == 404


def test_same_org_kb_get_returns_404(client, db):
    """User C in org1 cannot GET a KB created by User A in org1.

    KB access is user-scoped (user_id filter), not org-scoped.
    Each user can only access their own KBs regardless of org membership.
    """
    org1 = _make_org(db, "org1")
    _make_user(db, "alice", org_id=org1.id)
    _make_user(db, "carol", org_id=org1.id)
    token_a = _login(client, "alice")
    token_c = _login(client, "carol")
    kb_id = _create_kb(client, token_a, "alice-kb")
    r = client.get(f"/api/knowledge-base/{kb_id}", headers=_auth(token_c))
    assert r.status_code == 404


def test_cross_org_kb_list_excludes_other_org(client, db):
    """GET list returns only own-org KBs."""
    org1 = _make_org(db, "org1")
    org2 = _make_org(db, "org2")
    _make_user(db, "alice", org_id=org1.id)
    _make_user(db, "bob", org_id=org2.id)
    token_a = _login(client, "alice")
    token_b = _login(client, "bob")
    kb_id = _create_kb(client, token_a, "alice-kb")
    r = client.get("/api/knowledge-base", headers=_auth(token_b))
    assert r.status_code == 200
    ids = [kb["id"] for kb in r.json()]
    assert kb_id not in ids


def test_no_org_user_sees_own_kb_only(client, db):
    """User with org_id=None sees only their own KBs (user_id filter fallback)."""
    _make_user(db, "solo", org_id=None)
    _make_user(db, "other", org_id=None)
    token_s = _login(client, "solo")
    token_o = _login(client, "other")
    kb_id = _create_kb(client, token_s, "solo-kb")
    r = client.get(f"/api/knowledge-base/{kb_id}", headers=_auth(token_o))
    assert r.status_code == 404


def test_cross_org_chat_get_returns_404(client, db):
    """User B in org2 cannot GET a Chat created by User A in org1."""
    org1 = _make_org(db, "org1")
    org2 = _make_org(db, "org2")
    _make_user(db, "alice", org_id=org1.id)
    _make_user(db, "bob", org_id=org2.id)
    token_a = _login(client, "alice")
    token_b = _login(client, "bob")
    kb_id = _create_kb(client, token_a, "alice-kb")
    chat_id = _create_chat(client, token_a, kb_id)
    r = client.get(f"/api/chat/{chat_id}", headers=_auth(token_b))
    assert r.status_code == 404


def test_same_org_chat_get_returns_404(client, db):
    """User C in org1 cannot GET a Chat created by User A in org1.

    Chat access is user-scoped (user_id filter), not org-scoped.
    Each user can only access their own Chats regardless of org membership.
    """
    org1 = _make_org(db, "org1")
    _make_user(db, "alice", org_id=org1.id)
    _make_user(db, "carol", org_id=org1.id)
    token_a = _login(client, "alice")
    token_c = _login(client, "carol")
    kb_id = _create_kb(client, token_a, "alice-kb")
    chat_id = _create_chat(client, token_a, kb_id)
    r = client.get(f"/api/chat/{chat_id}", headers=_auth(token_c))
    assert r.status_code == 404
