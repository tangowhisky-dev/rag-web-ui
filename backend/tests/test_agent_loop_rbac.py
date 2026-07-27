"""RBAC tests for the enterprise agent tools."""

import asyncio
from uuid import uuid4

import pytest
from sqlalchemy.orm import sessionmaker

import app.db.session as _session_mod
from app.core.security import get_password_hash
from app.models.base import Base
from app.models.chat import Chat, ChatFile, Folder
from app.models.knowledge import KnowledgeBase
from app.models.organisation import Organisation
from app.models.user import User, UserRole
from app.services.agentic_rag.tool_context import ToolContext
from app.services.agentic_rag.tools.file_read import FileReadTool

engine = _session_mod.engine
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


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


def _seed(db):
    """Create a minimal org/user/chat/file graph for one user."""
    org = Organisation(name=f"test-org-{uuid4().hex[:8]}")
    db.add(org)
    db.commit()
    db.refresh(org)

    user = User(
        username=f"user-{uuid4().hex[:8]}",
        email=f"{uuid4().hex[:8]}@example.com",
        hashed_password=get_password_hash("pass"),
        is_active=True,
        role=UserRole.user,
        org_id=org.id,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    kb = KnowledgeBase(name="kb", user_id=user.id, org_id=org.id)
    db.add(kb)
    db.commit()
    db.refresh(kb)

    chat = Chat(title="test", user_id=user.id, org_id=org.id)
    chat.knowledge_bases = [kb]
    db.add(chat)
    db.commit()
    db.refresh(chat)

    cf = ChatFile(
        chat_id=chat.id,
        file_name="doc.md",
        stored_path="/tmp/doc.md",
        file_size=100,
        content_type="text/plain",
        markdown_content="secret content",
        status="ready",
    )
    db.add(cf)
    db.commit()
    db.refresh(cf)

    return user, chat, kb, cf


class TestFileToolRbac:
    def test_allowed_file_read(self, db):
        user, chat, _kb, cf = _seed(db)
        ctx = ToolContext(db=db, user_id=user.id, org_id=user.org_id, chat_id=chat.id)
        tool = FileReadTool()
        tool.ctx = ctx
        result = asyncio.run(tool.arun({"file_id": cf.id}))
        assert result["ok"] is True

    def test_denied_file_read_for_other_chat(self, db):
        user, chat, _kb, cf = _seed(db)
        other_chat = Chat(title="other", user_id=user.id, org_id=user.org_id)
        db.add(other_chat)
        db.commit()

        ctx = ToolContext(db=db, user_id=user.id, org_id=user.org_id, chat_id=other_chat.id)
        tool = FileReadTool()
        tool.ctx = ctx
        result = asyncio.run(tool.arun({"file_id": cf.id}))
        assert result["ok"] is False
        assert "denied" in result["error"].lower()

    def test_denied_file_read_for_other_user(self, db):
        _user, _chat, _kb, cf = _seed(db)
        org = Organisation(name=f"other-org-{uuid4().hex[:8]}")
        db.add(org)
        db.commit()
        other = User(
            username=f"other-{uuid4().hex[:8]}",
            email=f"{uuid4().hex[:8]}@example.com",
            hashed_password=get_password_hash("pass"),
            is_active=True,
            role=UserRole.user,
            org_id=org.id,
        )
        db.add(other)
        db.commit()

        ctx = ToolContext(db=db, user_id=other.id, org_id=other.org_id, chat_id=None)
        tool = FileReadTool()
        tool.ctx = ctx
        result = asyncio.run(tool.arun({"file_id": cf.id}))
        assert result["ok"] is False
