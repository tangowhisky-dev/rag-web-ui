"""Tests for /api/admin/datastores CRUD endpoints."""
import uuid
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from app.main import app as fastapi_app  # noqa: conftest must run first
import app.main as main_module

import app.db.session as _session_mod
from app.db.session import get_db
from app.models.base import Base  # noqa
from app.models.datastore import DataStore, OrganizationDataStore
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


@pytest.fixture(autouse=True)
def patch_watcher_and_validation(monkeypatch):
    """Avoid real filesystem / watcher dependencies for datastore endpoints."""
    import app.api.api_v1.datastores as ds_module
    import app.services.cleanup as cleanup_module

    monkeypatch.setattr(ds_module, "_validate_folder_path", lambda p: f"/app/data/{uuid.uuid4().hex}")
    mock_watcher = MagicMock()
    mock_watcher.is_running = True
    mock_watcher.get_status.return_value = {"datastores": [], "active_scans": []}
    mock_watcher.sync_watchers_with_database.return_value = None
    monkeypatch.setattr(ds_module, "_get_watcher", lambda: mock_watcher)
    monkeypatch.setattr(cleanup_module, "delete_datastore", lambda _db, _id: ({}, 204))


def _create_admin(db, username, role=UserRole.super_admin, org=None):
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


def test_list_datastores_empty(client, db):
    root = Organisation(name="Root", parent_id=None, path="/1")
    db.add(root)
    db.commit()
    _create_admin(db, "admin", org=root)
    token = _get_token(client, "admin")

    resp = client.get("/api/admin/datastores", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["items"] == []
    assert body["total"] == 0


def test_create_and_get_datastore(client, db):
    root = Organisation(name="Root", parent_id=None, path="/1")
    db.add(root)
    db.commit()
    _create_admin(db, "admin", org=root)
    token = _get_token(client, "admin")

    resp = client.post(
        "/api/admin/datastores",
        json={"name": "DS1", "folder_path": "/app/data/ds1", "scan_pattern": "*.txt"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["name"] == "DS1"
    ds_id = data["id"]

    resp2 = client.get(f"/api/admin/datastores/{ds_id}", headers={"Authorization": f"Bearer {token}"})
    assert resp2.status_code == 200
    assert resp2.json()["id"] == ds_id


def test_update_datastore(client, db):
    root = Organisation(name="Root", parent_id=None, path="/1")
    db.add(root)
    db.commit()
    _create_admin(db, "admin", org=root)
    token = _get_token(client, "admin")

    create = client.post(
        "/api/admin/datastores",
        json={"name": "DS1", "folder_path": "/app/data/ds1"},
        headers={"Authorization": f"Bearer {token}"},
    )
    ds_id = create.json()["id"]

    resp = client.patch(
        f"/api/admin/datastores/{ds_id}",
        json={"name": "DS1-renamed"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["name"] == "DS1-renamed"


def test_delete_datastore(client, db):
    root = Organisation(name="Root", parent_id=None, path="/1")
    db.add(root)
    db.commit()
    _create_admin(db, "admin", org=root)
    token = _get_token(client, "admin")

    create = client.post(
        "/api/admin/datastores",
        json={"name": "DS1", "folder_path": "/app/data/ds1"},
        headers={"Authorization": f"Bearer {token}"},
    )
    ds_id = create.json()["id"]

    resp = client.delete(f"/api/admin/datastores/{ds_id}", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 204


def test_assign_and_unassign_datastore(client, db):
    root = Organisation(name="Root", parent_id=None, path="/1")
    db.add(root)
    db.commit()
    child = Organisation(name="Child", parent_id=root.id, path=f"/{root.id}/2")
    db.add(child)
    db.commit()
    _create_admin(db, "admin", org=root)
    token = _get_token(client, "admin")

    create = client.post(
        "/api/admin/datastores",
        json={"name": "DS1", "folder_path": "/app/data/ds1"},
        headers={"Authorization": f"Bearer {token}"},
    )
    ds_id = create.json()["id"]

    resp = client.post(
        f"/api/admin/datastores/{ds_id}/assign",
        json={"org_ids": [child.id]},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert db.query(OrganizationDataStore).filter_by(data_store_id=ds_id, org_id=child.id).first() is not None

    resp2 = client.request(
        "DELETE",
        f"/api/admin/datastores/{ds_id}/assign",
        json={"org_ids": [child.id]},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp2.status_code == 200
    assert db.query(OrganizationDataStore).filter_by(data_store_id=ds_id, org_id=child.id).first() is None


def test_get_datastore_status(client, db):
    root = Organisation(name="Root", parent_id=None, path="/1")
    db.add(root)
    db.commit()
    _create_admin(db, "admin", org=root)
    token = _get_token(client, "admin")

    create = client.post(
        "/api/admin/datastores",
        json={"name": "DS1", "folder_path": "/app/data/ds1"},
        headers={"Authorization": f"Bearer {token}"},
    )
    ds_id = create.json()["id"]

    resp = client.get(f"/api/admin/datastores/{ds_id}/status", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["datastore_id"] == ds_id
    assert data["pending_changes"] == 0
