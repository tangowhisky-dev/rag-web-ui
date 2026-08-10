"""Tests for the stateless /api/query endpoint."""
import pytest
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from app.main import app as fastapi_app  # noqa: conftest must run first
import app.main as main_module

import app.db.session as _session_mod
from app.db.session import get_db
from app.models.base import Base  # noqa
from app.models.knowledge import KnowledgeBase
from app.models.organisation import Organisation
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
def client(monkeypatch):
    monkeypatch.setattr(main_module, "_seed_root_org_and_superadmin", lambda: None)
    fastapi_app.dependency_overrides[get_db] = override_get_db
    with TestClient(fastapi_app) as c:
        yield c
    fastapi_app.dependency_overrides.clear()


def _create_user(db, username, org=None):
    user = User(
        username=username,
        email=f"{username}@example.com",
        hashed_password=security.get_password_hash("pass123"),
        is_active=True,
        role=UserRole.user,
        org_id=org.id if org else None,
    )
    db.add(user)
    db.commit()
    return user


def _get_token(client, username):
    resp = client.post("/api/auth/token", data={"username": username, "password": "pass123"})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def test_query_returns_answer_and_contexts(client, db, monkeypatch):
    root = Organisation(name="Root", parent_id=None, path="/1")
    db.add(root)
    db.commit()
    user = _create_user(db, "query_user", org=root)
    kb = KnowledgeBase(name="KB", user_id=user.id, org_id=root.id)
    db.add(kb)
    db.commit()
    db.refresh(kb)
    token = _get_token(client, "query_user")

    import app.api.api_v1.query as query_module

    async def _mock_run(*, query, knowledge_base_ids, db, chat_id):
        yield {
            "event": "context",
            "docs": [
                {
                    "page_content": "Relevant content",
                    "metadata": {"_reranker_score": 10.0, "source": "test"},
                }
            ],
            "score": 80,
        }
        yield {"event": "done", "full_response": "This is the answer."}

    monkeypatch.setattr(query_module, "run_agentic_rag", _mock_run)

    resp = client.post(
        "/api/query",
        json={"question": "What is RAG?", "kb_ids": [kb.id], "generate_answer": True},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["question"] == "What is RAG?"
    assert data["answer"] == "This is the answer."
    assert len(data["contexts"]) == 1
    assert data["contexts"][0]["content"] == "Relevant content"
    assert "retrieval_info" in data


def test_query_rejects_unknown_kb(client, db):
    root = Organisation(name="Root", parent_id=None, path="/1")
    db.add(root)
    db.commit()
    user = _create_user(db, "query_user2", org=root)
    token = _get_token(client, "query_user2")

    resp = client.post(
        "/api/query",
        json={"question": "Q", "kb_ids": [9999]},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 404
