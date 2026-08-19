"""
conftest.py — test database bootstrap

Patches app.db.session with a file-based SQLite engine *before* any app
module that reads session.py is imported.  This allows all unit tests to
run without a running MySQL server.

For integration tests that need the real MySQL database, set the
DATABASE_USE_MYSQL environment variable to true.

Import order matters:
  1. Set env vars first (before any app.* import loads settings)
  2. Patch MySQL-only dialect types (LONGTEXT, JSON) → Text
  3. Replace app.db.session in sys.modules with a SQLite stub

Usage:
  # Run tests against SQLite (default):
  pytest

  # Run tests against the Docker MySQL container:
  DATABASE_USE_MYSQL=true pytest
"""
import os
import sys
import tempfile
from types import ModuleType

# Determine which database to use for tests.
# If DATABASE_USE_MYSQL is set to "true", use the Docker MySQL container.
# Otherwise, use file-based SQLite for fast unit tests.
_DATABASE_USE_MYSQL = os.environ.get("DATABASE_USE_MYSQL", "").lower() == "true"

if _DATABASE_USE_MYSQL:
    # Use the Docker MySQL container for integration tests.
    # Credentials from .env:
    _mysql_host = os.getenv("MYSQL_SERVER", "127.0.0.1")
    _mysql_port = os.getenv("MYSQL_PORT", "3306")
    _mysql_user = os.getenv("MYSQL_USER", "ragwebui")
    _mysql_password = os.getenv("MYSQL_PASSWORD", "ragwebui")
    _mysql_database = os.getenv("MYSQL_DATABASE", "ragwebui_test")
    _sqlalchemy_url = (
        f"mysql+mysqlconnector://{_mysql_user}:{_mysql_password}"
        f"@{_mysql_host}:{_mysql_port}/{_mysql_database}"
    )
    os.environ["SQLALCHEMY_DATABASE_URI"] = _sqlalchemy_url
    _USE_SQLITE = False
else:
    # Use a file-based SQLite database for unit tests.
    # File-based (not in-memory) is required so that NullPool can give
    # each session its own connection without losing data.  In-memory
    # SQLite with StaticPool shares one connection across all sessions,
    # which causes stale identity map entries between tests.
    _sqlite_dir = tempfile.mkdtemp(prefix="rag_test_")
    _sqlite_path = os.path.join(_sqlite_dir, "test.db")
    _sqlalchemy_url = f"sqlite:///{_sqlite_path}"
    os.environ["SQLALCHEMY_DATABASE_URI"] = _sqlalchemy_url
    # Override UPLOAD_DIR so init_storage() doesn't try to create /app/uploads
    # (which is read-only on macOS). Must use direct assignment, not setdefault,
    # because the .env file may have already set it.
    os.environ["UPLOAD_DIR"] = "/tmp/rag_test_uploads"
    _USE_SQLITE = True

# Patch MySQL-only types so SQLite can create the tables.
from sqlalchemy import Text as _Text
from sqlalchemy.types import JSON as _JSON
from sqlalchemy.dialects import mysql as _mysql
_mysql.LONGTEXT = _Text
_mysql.JSON = _JSON  # MySQL JSON → generic JSON for SQLite

if _USE_SQLITE:
    # Stub out app.db.session *before* any app.* import.
    from sqlalchemy import create_engine as _create_engine
    from sqlalchemy.orm import sessionmaker as _sessionmaker
    from sqlalchemy.pool import NullPool as _NullPool

    # NullPool gives each session its own connection.  When a session closes,
    # the connection is closed too — no stale state leaks between tests.
    # This fixes the ObjectDeletedError / StaleDataError that occurred with
    # StaticPool when reset_db dropped tables while other sessions still
    # held objects in their identity maps.
    _sqlite_engine = _create_engine(
        _sqlalchemy_url,
        connect_args={"check_same_thread": False},
        poolclass=_NullPool,
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
