"""Tests for the /api/admin/counts dashboard endpoint."""
from unittest.mock import patch

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
    # Disable the seed so counts reflect only test data.
    monkeypatch.setattr(main_module, "_seed_root_org_and_superadmin", lambda: None)
    fastapi_app.dependency_overrides[get_db] = override_get_db
    with TestClient(fastapi_app) as c:
        yield c
    fastapi_app.dependency_overrides.clear()


def _create_admin_user(db, username="adminuser", org=None):
    user = User(
        username=username,
        email=f"{username}@example.com",
        hashed_password=security.get_password_hash("pass123"),
        is_active=True,
        role=UserRole.admin,
        org_id=org.id if org else None,
    )
    db.add(user)
    db.commit()
    return user


def _get_admin_token(client, username, password="pass123"):
    resp = client.post("/api/auth/token", data={"username": username, "password": password})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def test_admin_counts_requires_admin(client):
    resp = client.get("/api/admin/counts")
    assert resp.status_code in (401, 403)


def test_admin_counts_returns_totals_for_superadmin(client, db):
    root = Organisation(name="Root", parent_id=None, path="/1")
    db.add(root)
    db.commit()
    child = Organisation(name="Child", parent_id=root.id, path=f"/{root.id}/2")
    db.add(child)
    db.commit()

    admin = User(
        username="super",
        email="super@example.com",
        hashed_password=security.get_password_hash("pass123"),
        is_active=True,
        role=UserRole.super_admin,
        org_id=root.id,
    )
    db.add(admin)
    user = User(
        username="u1", email="u1@example.com",
        hashed_password=security.get_password_hash("pass"),
        is_active=True, role=UserRole.user, org_id=child.id,
    )
    db.add(user)
    ds = DataStore(name="DS1", folder_path="/tmp/ds1", is_active=True)
    db.add(ds)
    db.commit()
    db.refresh(ds)
    db.add(OrganizationDataStore(org_id=child.id, data_store_id=ds.id, is_active=True))
    db.commit()

    token = _get_admin_token(client, "super")
    resp = client.get("/api/admin/counts", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["organizations"] == 2
    assert data["users"] == 2  # super + u1
    assert data["data_sources"] == 1


def test_admin_counts_scopes_to_org_tree_for_admin(client, db):
    root = Organisation(name="Root", parent_id=None, path="/1")
    db.add(root)
    db.commit()
    child = Organisation(name="Child", parent_id=root.id, path=f"/{root.id}/2")
    other = Organisation(name="Other", parent_id=root.id, path=f"/{root.id}/3")
    db.add(child)
    db.add(other)
    db.commit()

    _create_admin_user(db, "scopedadmin", org=child)
    user_in_scope = User(
        username="inscope", email="in@example.com",
        hashed_password=security.get_password_hash("pass"),
        is_active=True, role=UserRole.user, org_id=child.id,
    )
    user_out_scope = User(
        username="outscope", email="out@example.com",
        hashed_password=security.get_password_hash("pass"),
        is_active=True, role=UserRole.user, org_id=other.id,
    )
    db.add(user_in_scope)
    db.add(user_out_scope)
    db.commit()

    token = _get_admin_token(client, "scopedadmin")
    resp = client.get("/api/admin/counts", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    data = resp.json()
    # Scoped admin only sees child (itself) and descendants; neither include root or other.
    assert data["organizations"] == 1
    assert data["users"] == 2  # scopedadmin + inscope
