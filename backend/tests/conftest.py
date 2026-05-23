"""
conftest.py — test database bootstrap

Patches app.db.session with an in-memory SQLite engine *before* any app
module that reads session.py is imported.  This allows all unit tests to
run without a running MySQL server.

Import order matters:
  1. Set env var
  2. Patch MySQL-only dialect types (LONGTEXT, JSON) → Text
  3. Replace app.db.session in sys.modules with a SQLite stub
"""
import os
import sys
from types import ModuleType

# 1. Set env vars first.
os.environ.setdefault("SQLALCHEMY_DATABASE_URI", "sqlite://")
os.environ.setdefault("UPLOAD_DIR", "/tmp/rag_test_uploads")

# 2. Patch MySQL-only types so SQLite can create the tables.
from sqlalchemy import Text as _Text
from sqlalchemy.dialects import mysql as _mysql
_mysql.LONGTEXT = _Text
_mysql.JSON = _Text  # MySQL JSON → generic Text for SQLite

# 3. Stub out app.db.session *before* any app.* import.
from sqlalchemy import create_engine as _create_engine
from sqlalchemy.orm import sessionmaker as _sessionmaker
from sqlalchemy.pool import StaticPool as _StaticPool

# Use StaticPool so all connections share the same in-memory SQLite database.
_sqlite_engine = _create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=_StaticPool,
)
_SqliteSession = _sessionmaker(autocommit=False, autoflush=False, bind=_sqlite_engine)


def _get_db():
    db = _SqliteSession()
    try:
        yield db
    finally:
        db.close()


_session_stub = ModuleType("app.db.session")
_session_stub.engine = _sqlite_engine
_session_stub.SessionLocal = _SqliteSession
_session_stub.get_db = _get_db
# Pre-populate parent package too so `import app.db.session` resolves cleanly.
_db_pkg = ModuleType("app.db")
_db_pkg.session = _session_stub
sys.modules["app.db"] = _db_pkg
sys.modules["app.db.session"] = _session_stub
