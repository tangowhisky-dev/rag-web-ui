"""
Tests for the message-branching endpoints:
  PATCH /api/chat/{chat_id}/messages/{message_id}
  GET   /api/chat/{chat_id}/messages/{message_id}/siblings
"""
import pytest
from sqlalchemy.orm import sessionmaker

from fastapi.testclient import TestClient

# conftest.py has already:
#   - patched MySQL dialect types (LONGTEXT → Text)
#   - replaced app.db.session with a SQLite stub
# So importing app.main is now safe.
from app.main import app as fastapi_app  # noqa: conftest must run first

# Re-import the stub's components so tests share the same engine/session.
import app.db.session as _session_mod  # the stub set by conftest
from app.db.session import get_db
from app.models.chat import Chat, Message
from app.models.user import User
from app.core.security import get_current_user
from app.models.base import Base  # noqa – ensures all models are registered
import app.models.chat  # noqa – registers Chat/Message tables
import app.models.user  # noqa – registers User table
import app.models.knowledge  # noqa – registers ProcessingTask and related tables

# Use the same engine the stub exposes so startup events and our fixtures
# share the same in-memory database.
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
    """Re-create all tables before each test so tests are isolated."""
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
    user = User(id=1, email="tester@example.com", username="tester", hashed_password="x", is_active=True)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture()
def client(fake_user):
    """TestClient with DB and auth overrides applied."""
    from app.main import app as fastapi_app  # local to avoid module-level shadowing
    fastapi_app.dependency_overrides[get_db] = override_get_db
    fastapi_app.dependency_overrides[get_current_user] = lambda: fake_user
    with TestClient(fastapi_app) as c:
        yield c
    fastapi_app.dependency_overrides.clear()


def _seed_chat_and_message(db, user_id: int, content: str = "Hello") -> tuple[Chat, Message]:
    chat = Chat(user_id=user_id, title="test chat")
    db.add(chat)
    db.flush()
    msg = Message(
        content=content,
        role="user",
        chat_id=chat.id,
        branch_index=0,
    )
    db.add(msg)
    db.commit()
    db.refresh(chat)
    db.refresh(msg)
    return chat, msg


# ---------------------------------------------------------------------------
# PATCH /api/chat/{chat_id}/messages/{message_id}
# ---------------------------------------------------------------------------

class TestEditMessage:
    def test_creates_new_branch_message(self, client, db, fake_user):
        chat, msg = _seed_chat_and_message(db, fake_user.id, content="Original")
        resp = client.patch(
            f"/api/chat/{chat.id}/messages/{msg.id}",
            json={"content": "Edited"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["content"] == "Edited"
        assert data["role"] == "user"
        assert data["branch_index"] == 1
        assert data["parent_message_id"] == msg.id

    def test_original_message_preserved(self, client, db, fake_user):
        chat, msg = _seed_chat_and_message(db, fake_user.id, content="Original")
        client.patch(f"/api/chat/{chat.id}/messages/{msg.id}", json={"content": "Edited"})
        original = db.query(Message).filter(Message.id == msg.id).first()
        assert original is not None
        assert original.content == "Original"
        assert original.branch_index == 0

    def test_second_edit_increments_branch_index(self, client, db, fake_user):
        chat, msg = _seed_chat_and_message(db, fake_user.id, content="Original")
        r1 = client.patch(f"/api/chat/{chat.id}/messages/{msg.id}", json={"content": "Edit 1"})
        r2 = client.patch(f"/api/chat/{chat.id}/messages/{msg.id}", json={"content": "Edit 2"})
        assert r1.json()["branch_index"] == 1
        assert r2.json()["branch_index"] == 2

    def test_edit_on_already_branched_message_shares_parent(self, client, db, fake_user):
        """Editing a branch message (branch_index=1) should share the same parent."""
        chat, original = _seed_chat_and_message(db, fake_user.id, content="Original")
        branch1 = client.patch(
            f"/api/chat/{chat.id}/messages/{original.id}", json={"content": "Edit 1"}
        ).json()
        branch2 = client.patch(
            f"/api/chat/{chat.id}/messages/{branch1['id']}", json={"content": "Edit 2"}
        ).json()
        # Both branches should share the same parent (the original message id)
        assert branch1["parent_message_id"] == original.id
        assert branch2["parent_message_id"] == original.id
        assert branch2["branch_index"] == 2

    def test_edit_returns_404_for_unknown_message(self, client):
        resp = client.patch("/api/chat/1/messages/999999", json={"content": "x"})
        assert resp.status_code == 404

    def test_edit_rejects_missing_content(self, client, db, fake_user):
        chat, msg = _seed_chat_and_message(db, fake_user.id)
        resp = client.patch(f"/api/chat/{chat.id}/messages/{msg.id}", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/chat/{chat_id}/messages/{message_id}/siblings
# ---------------------------------------------------------------------------

class TestGetSiblings:
    def test_siblings_of_root_message_returns_itself_only(self, client, db, fake_user):
        chat, msg = _seed_chat_and_message(db, fake_user.id)
        resp = client.get(f"/api/chat/{chat.id}/messages/{msg.id}/siblings")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert len(data) == 1
        assert data[0]["id"] == msg.id
        assert data[0]["branch_index"] == 0

    def test_siblings_after_one_edit(self, client, db, fake_user):
        chat, msg = _seed_chat_and_message(db, fake_user.id, content="Original")
        branch = client.patch(
            f"/api/chat/{chat.id}/messages/{msg.id}", json={"content": "Edit 1"}
        ).json()
        # Query siblings from the original
        resp = client.get(f"/api/chat/{chat.id}/messages/{msg.id}/siblings")
        assert resp.status_code == 200
        ids = {s["id"] for s in resp.json()}
        assert msg.id in ids
        assert branch["id"] in ids
        assert len(ids) == 2

    def test_siblings_from_branch_same_as_from_root(self, client, db, fake_user):
        chat, msg = _seed_chat_and_message(db, fake_user.id, content="Original")
        branch = client.patch(
            f"/api/chat/{chat.id}/messages/{msg.id}", json={"content": "Edit 1"}
        ).json()
        resp_from_root = client.get(f"/api/chat/{chat.id}/messages/{msg.id}/siblings").json()
        resp_from_branch = client.get(f"/api/chat/{chat.id}/messages/{branch['id']}/siblings").json()
        assert {s["id"] for s in resp_from_root} == {s["id"] for s in resp_from_branch}

    def test_siblings_ordered_by_branch_index(self, client, db, fake_user):
        chat, msg = _seed_chat_and_message(db, fake_user.id, content="Original")
        client.patch(f"/api/chat/{chat.id}/messages/{msg.id}", json={"content": "Edit 1"})
        client.patch(f"/api/chat/{chat.id}/messages/{msg.id}", json={"content": "Edit 2"})
        resp = client.get(f"/api/chat/{chat.id}/messages/{msg.id}/siblings").json()
        indices = [s["branch_index"] for s in resp]
        assert indices == sorted(indices)

    def test_siblings_returns_404_for_unknown_message(self, client):
        resp = client.get("/api/chat/1/messages/999999/siblings")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Structured-log smoke test (slice-level verification requirement)
# ---------------------------------------------------------------------------

class TestBranchCreationLog:
    def test_branch_created_log_emitted(self, client, db, fake_user, caplog):
        import logging
        chat, msg = _seed_chat_and_message(db, fake_user.id, content="Original")
        with caplog.at_level(logging.INFO, logger="app.api.api_v1.chat"):
            client.patch(f"/api/chat/{chat.id}/messages/{msg.id}", json={"content": "Edited"})
        log_text = " ".join(caplog.messages)
        assert "message.branch_created" in log_text
        assert f"original_id={msg.id}" in log_text
