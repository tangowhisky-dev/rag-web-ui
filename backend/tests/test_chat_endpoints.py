"""Tests for /api/chat CRUD endpoints (not the streaming message endpoint)."""
from unittest.mock import patch

import pytest
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from app.main import app as fastapi_app  # noqa: conftest must run first
import app.main as main_module

import app.db.session as _session_mod
from app.db.session import get_db
from app.models.base import Base  # noqa
from app.models.chat import Chat, Message
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


def test_create_and_list_chats(client, db):
    root = Organisation(name="Root", parent_id=None, path="/1")
    db.add(root)
    db.commit()
    user = _create_user(db, "chat_user", org=root)
    kb = KnowledgeBase(name="KB", user_id=user.id, org_id=root.id)
    db.add(kb)
    db.commit()
    db.refresh(kb)
    token = _get_token(client, "chat_user")

    resp = client.post(
        "/api/chat",
        json={"title": "Chat 1", "knowledge_base_ids": [kb.id]},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["title"] == "Chat 1"
    chat_id = data["id"]

    resp2 = client.get("/api/chat", headers={"Authorization": f"Bearer {token}"})
    assert resp2.status_code == 200
    assert len(resp2.json()) == 1
    assert resp2.json()[0]["id"] == chat_id


def test_get_and_update_chat(client, db):
    root = Organisation(name="Root", parent_id=None, path="/1")
    db.add(root)
    db.commit()
    user = _create_user(db, "chat_user2", org=root)
    kb = KnowledgeBase(name="KB2", user_id=user.id, org_id=root.id)
    db.add(kb)
    db.commit()
    db.refresh(kb)
    chat = Chat(title="Chat", user_id=user.id, org_id=root.id)
    chat.knowledge_bases = [kb]
    db.add(chat)
    db.commit()
    token = _get_token(client, "chat_user2")

    resp = client.get(f"/api/chat/{chat.id}", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["title"] == "Chat"

    resp2 = client.patch(
        f"/api/chat/{chat.id}",
        json={"title": "Renamed", "pinned": True},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp2.status_code == 200
    assert resp2.json()["title"] == "Renamed"
    assert resp2.json()["pinned"] is True


def test_delete_chat(client, db, monkeypatch):
    root = Organisation(name="Root", parent_id=None, path="/1")
    db.add(root)
    db.commit()
    user = _create_user(db, "chat_user3", org=root)
    chat = Chat(title="ToDelete", user_id=user.id, org_id=root.id)
    db.add(chat)
    db.commit()
    token = _get_token(client, "chat_user3")

    import app.services.agentic_rag.redis_memory as redis_mem
    monkeypatch.setattr(redis_mem, "delete_chat_redis_sync", lambda _chat_id, user_id=None: None)

    resp = client.delete(f"/api/chat/{chat.id}", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert db.query(Chat).filter(Chat.id == chat.id).first() is None


def test_cancel_chat_stream(client, db):
    root = Organisation(name="Root", parent_id=None, path="/1")
    db.add(root)
    db.commit()
    user = _create_user(db, "chat_user4", org=root)
    chat = Chat(title="Chat", user_id=user.id, org_id=root.id)
    db.add(chat)
    db.commit()
    token = _get_token(client, "chat_user4")

    resp = client.post(f"/api/chat/{chat.id}/cancel", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"


def test_get_messages_paginated(client, db):
    root = Organisation(name="Root", parent_id=None, path="/1")
    db.add(root)
    db.commit()
    user = _create_user(db, "chat_user5", org=root)
    chat = Chat(title="Chat", user_id=user.id, org_id=root.id)
    db.add(chat)
    db.commit()
    db.refresh(chat)
    for i in range(3):
        db.add(Message(chat_id=chat.id, role="user" if i % 2 == 0 else "assistant", content=f"msg {i}"))
    db.commit()
    token = _get_token(client, "chat_user5")

    resp = client.get(f"/api/chat/{chat.id}/messages/paginated?limit=2", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["messages"]) == 2
    assert "has_more" in data


def test_search_chats(client, db):
    root = Organisation(name="Root", parent_id=None, path="/1")
    db.add(root)
    db.commit()
    user = _create_user(db, "chat_user6", org=root)
    chat = Chat(title="Chat", user_id=user.id, org_id=root.id)
    db.add(chat)
    db.commit()
    db.refresh(chat)
    db.add(Message(chat_id=chat.id, role="user", content="unique search term xyz123"))
    db.commit()
    token = _get_token(client, "chat_user6")

    resp = client.get("/api/chat/search?q=xyz123", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    results = resp.json()
    assert any("xyz123" in r["snippet"] for r in results)
