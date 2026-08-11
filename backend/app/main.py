import logging
import os

from app.api.api_v1.api import api_router
from app.core.config import settings
from app.core.security import get_password_hash
from app.core.storage import init_storage
from app.db.session import SessionLocal
from app.models.knowledge import ProcessingTask
from app.models.organisation import Organisation
from app.models.user import User, UserRole
from app.services.datastore_watcher import DataStoreWatcher
from app.services.discovery import StartupRecoveryService
from app.services.retrieval.reranker import preload_cross_encoder
from app.services.infrastructure.utils import preload_sparse_embedder
from app.services.settings_service import seed_app_settings
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
    # Seed root organisation and superadmin (runs before other startup tasks)
    _seed_root_org_and_superadmin()

    # Seed app settings from .env (only values that differ from config.py defaults)
    try:
        db = SessionLocal()
        try:
            seed_app_settings(db)
        finally:
            db.close()
    except Exception as e:
        logging.getLogger(__name__).warning("Failed to seed app settings: %s", e)

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

    # Start the DataStore watcher service after recovery
    global watcher_service
    if settings.WATCHER_ENABLED:
        try:
            _services["watcher"] = DataStoreWatcher()
            _services["watcher"].start()
            watcher_service = _services["watcher"]
        except Exception as e:
            logging.getLogger(__name__).error("Failed to start DataStoreWatcher: %s", e)

    # Reset any tasks left in "processing" state from a previous worker crash.
    db = SessionLocal()
    try:
        stuck = db.query(ProcessingTask).filter(ProcessingTask.status == "processing").all()
        if stuck:
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Startup: resetting {len(stuck)} stuck 'processing' task(s) to 'failed'"
            )
            for t in stuck:
                t.status = "failed"
                t.error_message = "Worker restarted while task was in progress"
            db.commit()
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


@app.get("/")
def root():
    return {"message": "Welcome to InsightCore API"}


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "version": settings.VERSION,
    }
