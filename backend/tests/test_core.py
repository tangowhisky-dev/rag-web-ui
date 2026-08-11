"""
Unit tests for core utilities: security, storage, and settings.

Uses conftest's in-memory SQLite stub for DB-dependent security helpers.
"""
import os
import shutil
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy.orm import sessionmaker

# conftest.py has already patched MySQL dialect types and app.db.session.
from app.main import app as fastapi_app  # noqa: conftest must run first

import app.db.session as _session_mod
from app.db.session import get_db
from app.models.base import Base  # noqa
from app.models.organisation import Organisation
from app.models.user import User, UserRole

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


# ── Security ─────────────────────────────────────────────────────────────────


def test_password_hash_verifies():
    from app.core.security import get_password_hash, verify_password

    hashed = get_password_hash("Secret123!")
    assert isinstance(hashed, str)
    assert verify_password("Secret123!", hashed)
    assert not verify_password("WrongPass", hashed)


def test_validate_password_strength():
    from app.core.security import validate_password_strength

    assert validate_password_strength("short1") == "Password must be at least 8 characters"
    assert validate_password_strength("onlyletters") == "Password must contain at least one number"
    assert validate_password_strength("12345678") == "Password must contain at least one letter"
    assert validate_password_strength("ValidPass1") == ""


def test_get_admin_org_ids_returns_none_for_superadmin(db):
    from app.core.security import get_admin_org_ids

    user = User(username="sa", email="sa@example.com", role=UserRole.super_admin)
    assert get_admin_org_ids(db, user) is None


def test_get_admin_org_ids_returns_empty_without_org(db):
    from app.core.security import get_admin_org_ids

    user = User(username="u", email="u@example.com", role=UserRole.admin, org_id=None)
    assert get_admin_org_ids(db, user) == []


def test_get_admin_org_ids_collects_descendants(db):
    from app.core.security import get_admin_org_ids

    root = Organisation(name="Root", parent_id=None, path="/1")
    db.add(root)
    db.commit()
    child = Organisation(name="Child", parent_id=root.id, path=f"/{root.id}/2")
    grandchild = Organisation(name="Grandchild", parent_id=2, path=f"/{root.id}/2/3")
    db.add(child)
    db.add(grandchild)
    db.commit()

    user = User(username="u", email="u@example.com", role=UserRole.admin, org_id=root.id)
    assert sorted(get_admin_org_ids(db, user)) == [root.id, child.id, grandchild.id]


# ── Storage ──────────────────────────────────────────────────────────────────


def _unique_test_dir():
    """Return a fresh temporary directory under UPLOAD_DIR."""
    from app.core import storage

    base = storage._base()
    suffix = f"test_core_{uuid.uuid4().hex}"
    return base / suffix


@pytest.fixture()
def temp_upload_dir():
    from app.core import storage

    # Point UPLOAD_DIR at a fresh temp directory for this test.
    original_base = storage._base
    test_dir = _unique_test_dir()
    test_dir.mkdir(parents=True, exist_ok=True)

    def _override():
        return test_dir

    storage._base = _override
    yield test_dir
    storage._base = original_base
    if test_dir.exists():
        shutil.rmtree(test_dir)


def test_init_storage_creates_directory(temp_upload_dir):
    from app.core.storage import init_storage

    init_storage()
    assert temp_upload_dir.exists()


def test_save_and_get_abs_path(temp_upload_dir):
    from app.core.storage import get_abs_path, save_file

    save_file("nested/file.txt", b"hello")
    abs_path = get_abs_path("nested/file.txt")
    assert Path(abs_path).read_bytes() == b"hello"


def test_move_file(temp_upload_dir):
    from app.core.storage import get_abs_path, move_file, save_file

    save_file("src/a.txt", b"content")
    move_file("src/a.txt", "dst/b.txt")
    assert not Path(get_abs_path("src/a.txt")).exists()
    assert Path(get_abs_path("dst/b.txt")).read_bytes() == b"content"


def test_delete_file_idempotent(temp_upload_dir):
    from app.core.storage import delete_file, save_file

    save_file("to_delete.txt", b"x")
    delete_file("to_delete.txt")
    assert not Path(temp_upload_dir / "to_delete.txt").exists()
    delete_file("missing.txt")  # should not raise


def test_kb_path_and_delete_kb_files(temp_upload_dir):
    from app.core.storage import delete_kb_files, kb_path, save_file

    save_file(f"{kb_path(1, 2)}/doc.txt", b"data")
    delete_kb_files(1, 2)
    assert not (temp_upload_dir / "user_1" / "kb_2").exists()


def test_list_files(temp_upload_dir):
    from app.core.storage import list_files, save_file

    save_file("prefix/a.txt", b"1")
    save_file("prefix/sub/b.txt", b"2")
    save_file("other/c.txt", b"3")
    paths = list_files("prefix")
    assert sorted(paths) == ["prefix/a.txt", "prefix/sub/b.txt"]


def test_ephemeral_chat_file_lifecycle(temp_upload_dir):
    from app.core.storage import (
        delete_ephemeral_chat_files,
        ephemeral_chat_dir,
        save_ephemeral_file,
    )

    path = save_ephemeral_file(42, "file.txt", b"ephemeral")
    assert Path(path).exists()
    assert Path(path).read_bytes() == b"ephemeral"
    assert ephemeral_chat_dir(42).exists()

    save_ephemeral_file(42, "file.txt", b"second")
    files = sorted(temp_upload_dir.glob("ephemeral/42/*"))
    assert len(files) == 2

    delete_ephemeral_chat_files(42)
    assert not (temp_upload_dir / "ephemeral" / "42").exists()


# ── Config ───────────────────────────────────────────────────────────────────


def test_settings_get_database_url_uses_override():
    from app.core.config import Settings

    settings = Settings(SQLALCHEMY_DATABASE_URI="sqlite:///:memory:")
    assert settings.get_database_url == "sqlite:///:memory:"


def test_settings_get_database_url_builds_mysql():
    from app.core.config import Settings

    settings = Settings(
        SQLALCHEMY_DATABASE_URI=None,
        MYSQL_USER="u",
        MYSQL_PASSWORD="p",
        MYSQL_SERVER="db",
        MYSQL_PORT="3306",
        MYSQL_DATABASE="rag",
    )
    assert settings.get_database_url == "mysql+mysqlconnector://u:p@db:3306/rag"


def test_settings_chunk_overlap():
    from app.core.config import Settings

    settings = Settings(CHUNK_SIZE=1000, OVERLAP_PERCENTAGE=0.25)
    assert settings.chunk_overlap == 250


def test_settings_retrieval_config_presets():
    from app.core.config import Settings

    settings = Settings()
    presets = settings.retrieval_config_presets
    assert "FACTUAL" in presets
    assert "AMBIGUOUS" in presets
    assert all(
        key in presets["FACTUAL"]
        for key in ("dense_weight", "sparse_weight", "exact_weight", "top_k")
    )
