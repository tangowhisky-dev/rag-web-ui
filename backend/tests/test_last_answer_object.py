"""Tests for LastAnswerObject persistence and structured-output resilience."""

import asyncio
from datetime import timezone, datetime
from uuid import uuid4

import pytest
from sqlalchemy.orm import sessionmaker

import app.db.session as _session_mod
from app.core.security import get_password_hash
from app.models.base import Base
from app.models.chat import Chat, Message, ToolCallAudit
from app.models.organisation import Organisation
from app.models.user import User, UserRole
from app.services.agentic_rag.agent_graph import save_memory_node
from app.services.agentic_rag.schemas import LastAnswerObject, Observation
from app.services.agentic_rag.tool_context import ToolContext

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
    chat = Chat(title="test", user_id=user.id, org_id=org.id)
    db.add(chat)
    db.commit()
    db.refresh(chat)
    msg = Message(content="", role="assistant", chat_id=chat.id)
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return user, chat, msg


def test_last_answer_object_model_validates_minimal():
    lao = LastAnswerObject(summary="hello", key_points=["a", "b"])
    assert lao.summary == "hello"
    assert lao.data is None


def test_last_answer_object_tolerates_extra_keys():
    payload = {
        "summary": "s",
        "key_points": [],
        "unknown_key": 123,
    }
    lao = LastAnswerObject.model_validate(payload)
    assert lao.summary == "s"


def test_save_memory_persists_final_answer_and_object(db):
    _user, _chat, msg = _seed(db)
    lao = LastAnswerObject(
        summary="answer summary",
        key_points=["point 1"],
        data=[{"label": "x", "value": 1}],
    )
    obs = Observation(tool="extract_data", arguments={"source": "retrieved_docs"}, result={"n": 1}, error=None, latency_ms=0, tokens=0)
    ctx = ToolContext(db=db, user_id=0, org_id=None, chat_id=None)
    result = asyncio.run(save_memory_node(
        {
            "message_id": msg.id,
            "final_answer": "final text",
            "last_answer_object": lao,
            "observations": [obs],
        },
        ctx,
    ))
    assert result == {}
    db.refresh(msg)
    assert msg.content == "final text"
    assert msg.last_answer_object == lao.model_dump()
    assert msg.tool_calls == [obs.model_dump()]


def test_save_memory_skips_when_no_message_id(db):
    ctx = ToolContext(db=db, user_id=0, org_id=None, chat_id=None)
    result = asyncio.run(save_memory_node({"final_answer": "x"}, ctx))
    assert result == {}
