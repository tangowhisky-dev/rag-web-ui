"""Tests for /api/knowledge-base CRUD and data-source linking."""
from unittest.mock import patch
from datetime import datetime, timezone

import pytest
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from app.main import app as fastapi_app  # noqa: conftest must run first
import app.main as main_module

import app.db.session as _session_mod
from app.db.session import get_db
from app.models.base import Base  # noqa
from app.models.datastore import DataStore, OrganizationDataStore
from app.models.knowledge import KnowledgeBase, ProcessingTask
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


def _create_user(db, username, role=UserRole.user, org=None):
    user = User(
        username=username,
        email=f"{username}@example.com",
        hashed_password=security.get_password_hash("pass123"),
        is_active=True,
        role=role,
        org_id=org.id if org else None,
    )
    db.add(user)
    db.commit()
    return user


def _get_token(client, username):
    resp = client.post("/api/auth/token", data={"username": username, "password": "pass123"})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def test_create_and_list_knowledge_base(client, db):
    root = Organisation(name="Root", parent_id=None, path="/1")
    db.add(root)
    db.commit()
    user = _create_user(db, "user1", org=root)
    token = _get_token(client, "user1")

    resp = client.post(
        "/api/knowledge-base",
        json={"name": "KB1", "description": "Test KB"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["name"] == "KB1"
    kb_id = data["id"]

    resp2 = client.get("/api/knowledge-base", headers={"Authorization": f"Bearer {token}"})
    assert resp2.status_code == 200
    assert len(resp2.json()) == 1
    assert resp2.json()[0]["id"] == kb_id


def test_get_knowledge_base(client, db):
    root = Organisation(name="Root", parent_id=None, path="/1")
    db.add(root)
    db.commit()
    user = _create_user(db, "user2", org=root)
    token = _get_token(client, "user2")

    create = client.post(
        "/api/knowledge-base",
        json={"name": "KB2"},
        headers={"Authorization": f"Bearer {token}"},
    )
    kb_id = create.json()["id"]

    resp = client.get(f"/api/knowledge-base/{kb_id}", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "KB2"


def test_update_knowledge_base(client, db):
    root = Organisation(name="Root", parent_id=None, path="/1")
    db.add(root)
    db.commit()
    user = _create_user(db, "user3", org=root)
    token = _get_token(client, "user3")

    create = client.post(
        "/api/knowledge-base",
        json={"name": "KB3"},
        headers={"Authorization": f"Bearer {token}"},
    )
    kb_id = create.json()["id"]

    resp = client.put(
        f"/api/knowledge-base/{kb_id}",
        json={"name": "KB3-updated", "description": "updated"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["name"] == "KB3-updated"


def test_delete_knowledge_base(client, db, monkeypatch):
    root = Organisation(name="Root", parent_id=None, path="/1")
    db.add(root)
    db.commit()
    user = _create_user(db, "user4", org=root)
    token = _get_token(client, "user4")

    create = client.post(
        "/api/knowledge-base",
        json={"name": "KB4"},
        headers={"Authorization": f"Bearer {token}"},
    )
    kb_id = create.json()["id"]

    # Avoid real cross-service cleanup in unit test.
    import app.services.cleanup as cleanup_module
    monkeypatch.setattr(cleanup_module, "delete_kb", lambda _db, _kb_id, _user_id: ({"message": "deleted"}, 200))

    resp = client.delete(f"/api/knowledge-base/{kb_id}", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["message"] == "deleted"


def test_link_and_unlink_datastore(client, db):
    root = Organisation(name="Root", parent_id=None, path="/1")
    db.add(root)
    db.commit()
    user = _create_user(db, "user5", org=root)
    token = _get_token(client, "user5")

    kb = KnowledgeBase(name="KB5", user_id=user.id, org_id=root.id)
    db.add(kb)
    db.commit()
    db.refresh(kb)

    ds = DataStore(name="DS", folder_path="/app/data/ds", is_active=True)
    db.add(ds)
    db.commit()
    db.refresh(ds)
    db.add(OrganizationDataStore(org_id=root.id, data_store_id=ds.id, is_active=True))
    db.commit()

    resp = client.post(
        f"/api/knowledge-base/{kb.id}/link-datastore",
        json={"data_store_id": ds.id},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert "linked" in resp.json()["message"].lower()

    resp2 = client.delete(
        f"/api/knowledge-base/{kb.id}/unlink-datastore/{ds.id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp2.status_code == 200


def test_ingest_status_endpoint(client, db):
    root = Organisation(name="Root", parent_id=None, path="/1")
    db.add(root)
    db.commit()
    user = _create_user(db, "user6", org=root)
    kb = KnowledgeBase(name="KB6", user_id=user.id, org_id=root.id)
    db.add(kb)
    db.commit()
    db.refresh(kb)

    db.add(ProcessingTask(knowledge_base_id=kb.id, status="completed"))
    db.add(ProcessingTask(knowledge_base_id=kb.id, status="pending"))
    db.commit()

    token = _get_token(client, "user6")
    resp = client.get(f"/api/query/kb/{kb.id}/ingest-status", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert data["completed"] == 1
    assert data["pending"] == 1
    assert data["ready"] is False


def test_ocr_availability_endpoint(client, db):
    """GET /ocr-availability returns ocr_available based on VISION_MODEL setting."""
    root = Organisation(name="Root", parent_id=None, path="/1")
    db.add(root)
    db.commit()
    user = _create_user(db, "user_ocr", org=root)
    token = _get_token(client, "user_ocr")

    with patch("app.services.settings_service.get_setting", return_value=None):
        resp = client.get(
            "/api/knowledge-base/ocr-availability",
            headers={"Authorization": f"Bearer {token}"},
        )
    assert resp.status_code == 200
    assert resp.json()["ocr_available"] is False

    with patch("app.services.settings_service.get_setting", return_value="gpt-4o"):
        resp = client.get(
            "/api/knowledge-base/ocr-availability",
            headers={"Authorization": f"Bearer {token}"},
        )
    assert resp.status_code == 200
    assert resp.json()["ocr_available"] is True


def test_upload_rejects_oversized_file(client, db):
    """POST /{kb_id}/documents/upload returns 413 for files > MAX_FILE_SIZE."""
    root = Organisation(name="Root", parent_id=None, path="/1")
    db.add(root)
    db.commit()
    user = _create_user(db, "user_big", org=root)
    token = _get_token(client, "user_big")

    kb = KnowledgeBase(name="KB-big", user_id=user.id, org_id=root.id)
    db.add(kb)
    db.commit()
    db.refresh(kb)

    # Create a file just over 10MB
    from app.services.ingestion.document_converter import MAX_FILE_SIZE
    big_content = b"x" * (MAX_FILE_SIZE + 1)

    resp = client.post(
        f"/api/knowledge-base/{kb.id}/documents/upload",
        files={"files": ("big.txt", big_content, "text/plain")},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 413
    assert "exceeds maximum size" in resp.json()["detail"]


def test_process_endpoint_deduplicates_upload_ids(client, db):
    """POST /{kb_id}/documents/process deduplicates upload_ids and skips
    uploads that already have a pending/processing task."""
    from app.models.knowledge import DocumentUpload

    root = Organisation(name="Root", parent_id=None, path="/1")
    db.add(root)
    db.commit()
    user = _create_user(db, "user_dedup", org=root)
    token = _get_token(client, "user_dedup")

    kb = KnowledgeBase(name="KB-dedup", user_id=user.id, org_id=root.id)
    db.add(kb)
    db.commit()
    db.refresh(kb)

    upload = DocumentUpload(
        knowledge_base_id=kb.id,
        file_name="test.txt",
        file_hash="abc123",
        file_size=100,
        content_type="text/plain",
        temp_path="temp/test.txt",
        created_at=datetime.now(timezone.utc),
    )
    db.add(upload)
    db.commit()
    db.refresh(upload)

    # Send the same upload_id twice in one request
    resp = client.post(
        f"/api/knowledge-base/{kb.id}/documents/process",
        json=[
            {"upload_id": upload.id, "enable_ocr": False},
            {"upload_id": upload.id, "enable_ocr": True},
        ],
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    tasks = resp.json()["tasks"]
    # Should only create one task despite duplicate upload_id
    assert len(tasks) == 1
