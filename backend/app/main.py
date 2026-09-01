import logging
import os

from app.api.api_v1.api import api_router
from app.core.config import settings
from app.core.security import get_password_hash
from app.core.storage import init_storage
from app.db.session import SessionLocal
from app.models.knowledge import ProcessingTask, Document
from app.models.organisation import Organisation
from app.models.user import User, UserRole
from app.services.datastore_watcher import DataStoreWatcher
from app.services.discovery import StartupRecoveryService
from app.services.retrieval.reranker import preload_cross_encoder
from app.services.infrastructure.utils import preload_sparse_embedder
from app.services.settings_service import clear_cache
from fastapi import FastAPI

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def _seed_root_org_and_superadmin() -> None:
    """Create ROOT_ORG and a superadmin user if none exists.

    Reads from env:
        ROOT_ORG            – org name (required)
        SUPERADMIN_USERNAME  – admin username (required)
        SUPERADMIN_PASSWORD  – admin password (required)

    Raises ``RuntimeError`` if any required variable is missing, preventing
    the application from starting with default credentials.
    """
    org_name = os.environ.get("ROOT_ORG", "").strip()
    admin_username = os.environ.get("SUPERADMIN_USERNAME", "").strip()
    admin_password = os.environ.get("SUPERADMIN_PASSWORD", "").strip()

    if not org_name:
        raise RuntimeError(
            "Environment variable ROOT_ORG must be set and non-empty. "
            "No superadmin can be created without a root organisation."
        )
    if not admin_username:
        raise RuntimeError(
            "Environment variable SUPERADMIN_USERNAME must be set and non-empty."
        )
    if not admin_password:
        raise RuntimeError(
            "Environment variable SUPERADMIN_PASSWORD must be set and non-empty."
        )

    db = SessionLocal()
    try:
        # 1. Check if a superadmin already exists — if yes, skip entirely
        superadmin = (
            db.query(User)
            .filter(User.role == UserRole.super_admin)
            .first()
        )
        if superadmin:
            logging.getLogger(__name__).info(
                "[SEED] Superadmin '%s' already exists (id=%d), skipping.",
                superadmin.username,
                superadmin.id,
            )
            return

        # 2. Ensure ROOT_ORG exists
        org = db.query(Organisation).filter(Organisation.name == org_name).first()
        if not org:
            org = Organisation(name=org_name, path="/1")
            db.add(org)
            db.flush()
            logging.getLogger(__name__).info(
                "[SEED] Created organisation id=%d name=%s", org.id, org_name
            )

        # 3. Create superadmin user
        # Check for duplicate username or email
        existing = (
            db.query(User)
            .filter(
                (User.username == admin_username)
                | (User.email == f"{admin_username}@example.com")
            )
            .first()
        )
        if existing:
            logging.getLogger(__name__).warning(
                "[SEED] Username '%s' or email already exists, skipping superadmin creation.",
                admin_username,
            )
            return

        user = User(
            username=admin_username,
            email=f"{admin_username}@example.com",
            hashed_password=get_password_hash(admin_password),
            role=UserRole.super_admin,
            is_active=True,
            org_id=org.id,
        )
        db.add(user)
        db.commit()
        logging.getLogger(__name__).info(
            "[SEED] Created superadmin user username=%s id=%d org_id=%d",
            admin_username,
            user.id,
            org.id,
        )
    except Exception as e:
        db.rollback()
        logging.getLogger(__name__).error(
            "[SEED] Failed to seed root org/superadmin: %s", e
        )
    finally:
        db.close()

# Suppress noisy INFO notifications from the Neo4j driver and neo4j-graphrag
# internals. These fire on every schema op ("index already exists") and every
# relationship merge ("cartesian product") — none are actionable at INFO level.
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)
logging.getLogger("neo4j_graphrag").setLevel(logging.WARNING)


async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle for the application."""
    # Validate SECRET_KEY — reject the insecure default placeholder.
    if settings.SECRET_KEY == "your-secret-key-here":
        raise RuntimeError(
            "SECRET_KEY is set to the insecure default 'your-secret-key-here'. "
            "Set the SECRET_KEY environment variable to a strong random value."
        )

    # Seed root organisation and superadmin (runs before other startup tasks)
    _seed_root_org_and_superadmin()

    # Initialize local file storage
    init_storage()

    # Pre-load ML models so the first request doesn't pay the cold-start penalty.
    # Each preload is resilient to failure — the lazy path will retry on first use.
    try:
        preload_cross_encoder()
    except Exception as exc:
        logging.getLogger(__name__).warning("Cross-encoder preload failed: %s", exc)

    try:
        preload_sparse_embedder()
    except Exception as exc:
        logging.getLogger(__name__).warning("Sparse embedder preload failed: %s", exc)

    # Start the startup recovery service FIRST
    try:
        _services["recovery"] = StartupRecoveryService()
        _services["recovery"].start()
    except Exception as e:
        logging.getLogger(__name__).error("Failed to start recovery service: %s", e)

    # Cross-store reconciliation: remove orphaned Qdrant vectors, Neo4j
    # graph nodes, and MySQL chunk/task rows left by failed transactions
    # or crashed workers.  Runs in a background thread so the app becomes
    # available immediately.  The watcher is started inside the same
    # thread after reconciliation completes, preserving the ordering
    # invariant (reconcile before new ingestion begins).
    def _reconcile_then_start_watcher():
        try:
            from app.services.cleanup import run_reconciliation
            run_reconciliation()
        except Exception as e:
            logging.getLogger(__name__).warning("Startup reconciliation failed: %s", e)
        # Start the DataStore watcher after reconciliation completes
        global watcher_service
        from app.services.settings_service import get_setting
        _db = SessionLocal()
        try:
            _watcher_enabled = get_setting(_db, "WATCHER_ENABLED", None)
        finally:
            _db.close()
        if _watcher_enabled:
            try:
                _services["watcher"] = DataStoreWatcher()
                _services["watcher"].start()
                watcher_service = _services["watcher"]
            except Exception as e:
                logging.getLogger(__name__).error("Failed to start DataStoreWatcher: %s", e)

    import threading as _threading
    _threading.Thread(
        target=_reconcile_then_start_watcher,
        name="startup-reconcile",
        daemon=True,
    ).start()

    # Reset any tasks left in "processing" state from a previous worker crash.
    # KB tasks with a valid document are reset to "pending" so the user can
    # retry via the manual retry endpoint. Tasks without a document are reset
    # to "failed" since there's nothing to retry.
    db = SessionLocal()
    try:
        stuck = db.query(ProcessingTask).filter(ProcessingTask.status == "processing").all()
        if stuck:
            logger = logging.getLogger(__name__)
            pending_count = 0
            failed_count = 0
            for t in stuck:
                doc = db.query(Document).filter(Document.id == t.document_id).first() if t.document_id else None
                if doc:
                    t.status = "pending"
                    t.error_message = "Worker restarted while task was in progress — ready for retry"
                    pending_count += 1
                else:
                    t.status = "failed"
                    t.error_message = "Worker restarted while task was in progress"
                    failed_count += 1
            db.commit()
            logger.warning(
                f"Startup: reset {len(stuck)} stuck task(s) — {pending_count} to pending, {failed_count} to failed"
            )
    finally:
        db.close()

    yield

    # Shutdown
    if _services["watcher"] is not None:
        _services["watcher"].stop()
    if _services["recovery"] is not None:
        _services["recovery"].stop()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    redirect_slashes=False,
    lifespan=lifespan,
)

# DataStoreWatcher — started on demand during startup
_services = {"watcher": None, "recovery": None}
watcher_service = None

# Include routers
app.include_router(api_router, prefix=settings.API_V1_STR)

# Serve shared assets directory (e.g. Whisper model files for browser STT).
# Mounted at /assets so the frontend can proxy /assets/* here.
import os as _os
from fastapi.staticfiles import StaticFiles as _StaticFiles
# In Docker: backend at /app/app/main.py, assets at /app/assets.
# On host:  backend at backend/app/main.py, assets at ./assets.
_app_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_assets_dir = _os.path.join(_app_root, "assets")
if not _os.path.isdir(_assets_dir):
    _assets_dir = "/app/assets"  # Docker absolute path
if _os.path.isdir(_assets_dir):
    app.mount("/assets", _StaticFiles(directory=_assets_dir), name="assets")


@app.get("/")
def root():
    return {"message": "Welcome to InsightCore API"}


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "version": settings.VERSION,
    }
