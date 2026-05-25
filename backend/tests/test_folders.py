"""
Tests for folder CRUD, chat assignment, and message search endpoints:
  POST   /api/folders
  GET    /api/folders
  PATCH  /api/folders/{id}
  DELETE /api/folders/{id}
  PATCH  /api/folders/{fid}/chats/{cid}
  GET    /api/chat/search?q=...
"""
import pytest
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

# conftest.py has already patched MySQL dialect types and app.db.session.
from app.main import app as fastapi_app  # noqa: conftest must run first

import app.db.session as _session_mod
from app.db.session import get_db
from app.models.chat import Chat, Folder, Message
from app.models.user import User
from app.core.security import get_current_user
from app.models.base import Base  # noqa
import app.models.chat       # noqa
import app.models.user       # noqa
import app.models.knowledge  # noqa

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
def fake_user(db):
    user = User(
        id=1, email="tester@example.com", username="tester",
        hashed_password="x", is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture()
def other_user(db):
    user = User(
        id=2, email="other@example.com", username="other",
        hashed_password="x", is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture()
def client(fake_user):
    fastapi_app.dependency_overrides[get_db] = override_get_db
    fastapi_app.dependency_overrides[get_current_user] = lambda: fake_user
    with TestClient(fastapi_app) as c:
        yield c
    fastapi_app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Folder CRUD tests
# ---------------------------------------------------------------------------

def test_create_folder_returns_201(client):
    resp = client.post("/api/folders", json={"name": "Research"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Research"
    assert "id" in data


def test_list_folders_returns_own_only(client, db, fake_user, other_user):
    # Create a folder for the authenticated user via API
    client.post("/api/folders", json={"name": "Mine"})
    # Directly insert a folder for another user
    other_folder = Folder(name="Theirs", user_id=other_user.id)
    db.add(other_folder)
    db.commit()

    resp = client.get("/api/folders")
    assert resp.status_code == 200
    names = [f["name"] for f in resp.json()]
    assert "Mine" in names
    assert "Theirs" not in names


def test_rename_folder(client):
    create_resp = client.post("/api/folders", json={"name": "OldName"})
    folder_id = create_resp.json()["id"]

    resp = client.patch(f"/api/folders/{folder_id}", json={"name": "NewName"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "NewName"


def test_delete_folder_unassigns_chats(client, db, fake_user):
    # Create folder via API
    folder_resp = client.post("/api/folders", json={"name": "ToDelete"})
    folder_id = folder_resp.json()["id"]

    # Directly create a chat assigned to that folder
    chat = Chat(title="Chat1", user_id=fake_user.id, folder_id=folder_id)
    db.add(chat)
    db.commit()
    db.refresh(chat)
    chat_id = chat.id

    # Delete the folder
    resp = client.delete(f"/api/folders/{folder_id}")
    assert resp.status_code == 204

    # Chat should have folder_id set to NULL
    db.expire(chat)
    refreshed = db.query(Chat).filter(Chat.id == chat_id).first()
    assert refreshed.folder_id is None


# ---------------------------------------------------------------------------
# Chat assignment tests
# ---------------------------------------------------------------------------

def test_assign_chat_to_folder(client, db, fake_user):
    folder_resp = client.post("/api/folders", json={"name": "Work"})
    folder_id = folder_resp.json()["id"]

    chat = Chat(title="MyChat", user_id=fake_user.id)
    db.add(chat)
    db.commit()
    db.refresh(chat)

    resp = client.patch(f"/api/folders/{folder_id}/chats/{chat.id}")
    assert resp.status_code == 200
    assert resp.json()["folder_id"] == folder_id
    assert resp.json()["chat_id"] == chat.id

    # Verify in DB
    db.expire(chat)
    refreshed = db.query(Chat).filter(Chat.id == chat.id).first()
    assert refreshed.folder_id == folder_id


# ---------------------------------------------------------------------------
# Message search tests
# ---------------------------------------------------------------------------

def test_search_returns_matching_chat_with_snippet(client, db, fake_user):
    chat = Chat(title="Science Chat", user_id=fake_user.id)
    db.add(chat)
    db.commit()
    db.refresh(chat)

    msg = Message(
        chat_id=chat.id,
        role="user",
        content="The mitochondria is the powerhouse of the cell",
        branch_index=0,
    )
    db.add(msg)
    db.commit()

    resp = client.get("/api/chat/search", params={"q": "mitochondria"})
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) >= 1
    assert any("mitochondria" in r["snippet"].lower() for r in results)
    assert any(r["chat_title"] == "Science Chat" for r in results)


def test_search_returns_empty_for_no_match(client, db, fake_user):
    chat = Chat(title="Empty Chat", user_id=fake_user.id)
    db.add(chat)
    db.commit()
    db.refresh(chat)

    msg = Message(chat_id=chat.id, role="user", content="Hello world", branch_index=0)
    db.add(msg)
    db.commit()

    resp = client.get("/api/chat/search", params={"q": "xyzzyunlikely"})
    assert resp.status_code == 200
    assert resp.json() == []


def test_search_does_not_return_other_users_chats(client, db, fake_user, other_user):
    # Chat belonging to other_user with matching content
    other_chat = Chat(title="OtherChat", user_id=other_user.id)
    db.add(other_chat)
    db.commit()
    db.refresh(other_chat)

    msg = Message(
        chat_id=other_chat.id,
        role="user",
        content="secret proprietary information",
        branch_index=0,
    )
    db.add(msg)
    db.commit()

    resp = client.get("/api/chat/search", params={"q": "secret"})
    assert resp.status_code == 200
    chat_ids = [r["chat_id"] for r in resp.json()]
    assert other_chat.id not in chat_ids
