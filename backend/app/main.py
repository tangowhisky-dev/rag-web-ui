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
from app.services.startup_recovery_service import StartupRecoveryService
from fastapi import FastAPI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def _seed_root_org_and_superadmin() -> None:
    """Create ROOT_ORG and a superadmin user if none exists.

    Reads from env:
        ROOT_ORG       – org name (default: "Root Organization")
        SUPERADMIN_USERNAME – admin username (default: "admin")
        SUPERADMIN_PASSWORD – admin password (default: "admin123")
    Only creates the org/user if they don't already exist.
    """
    org_name = os.environ.get("ROOT_ORG", "Root Organization").strip()
    admin_username = os.environ.get("SUPERADMIN_USERNAME", "admin").strip()
    admin_password = os.environ.get("SUPERADMIN_PASSWORD", "admin123").strip()

    if not org_name or not admin_username or not admin_password:
        logging.getLogger(__name__).warning(
            "[SEED] Skipping root org/superadmin seed: ROOT_ORG, SUPERADMIN_USERNAME "
            "and SUPERADMIN_PASSWORD must all be set and non-empty."
        )
        return

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

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    redirect_slashes=False,
)

# DataStoreWatcher — started on demand during startup
watcher_service: DataStoreWatcher | None = None

# StartupRecoveryService — background ingestion on app start
startup_recovery: StartupRecoveryService | None = None

# Include routers
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.on_event("startup")
async def startup_event():
    global watcher_service

    # Seed root organisation and superadmin (runs before other startup tasks)
    _seed_root_org_and_superadmin()

    # Initialize local file storage
    init_storage()

    # Load best tuning config from disk (precedence: .env > best_config > defaults)
    try:
        from app.services.auto_tune import load_best_config
        load_best_config()
    except Exception as e:
        logging.getLogger(__name__).warning("Failed to load best tuning config: %s", e)

    # Start the DataStore watcher service if enabled
    if settings.WATCHER_ENABLED:
        try:
            watcher_service = DataStoreWatcher()
            watcher_service.start()
        except Exception as e:
            logging.getLogger(__name__).error("Failed to start DataStoreWatcher: %s", e)

        # Start the startup recovery service (background ingestion)
        try:
            startup_recovery = StartupRecoveryService()
            startup_recovery.start()
        except Exception as e:
            logging.getLogger(__name__).error("Failed to start recovery service: %s", e)

    # Reset any tasks left in "processing" state from a previous worker crash.
    # With --reload, a file-write event kills the worker mid-flight leaving tasks
    # permanently stuck. On restart we mark them failed so clients don't hang.
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


@app.get("/")
def root():
    return {"message": "Welcome to InsightCore API"}


@app.on_event("shutdown")
async def shutdown_event():
    global watcher_service, startup_recovery
    if watcher_service is not None:
        watcher_service.stop()
    if startup_recovery is not None:
        startup_recovery.stop()


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "version": settings.VERSION,
    }
