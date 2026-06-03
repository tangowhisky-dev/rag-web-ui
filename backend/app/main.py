import logging

from app.api.api_v1.api import api_router
from app.core.config import settings
from app.core.storage import init_storage
from app.db.session import SessionLocal
from app.models.knowledge import ProcessingTask
from app.services.watcher_service import WatcherService
from fastapi import FastAPI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

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

# WatcherService — started on demand during startup
watcher_service: WatcherService | None = None

# Include routers
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.on_event("startup")
async def startup_event():
    global watcher_service

    # Initialize local file storage
    init_storage()

    # Load best tuning config from disk (precedence: .env > best_config > defaults)
    try:
        from app.services.auto_tune import load_best_config
        load_best_config()
    except Exception as e:
        logging.getLogger(__name__).warning("Failed to load best tuning config: %s", e)

    # Start the file watcher service if enabled
    if settings.WATCHER_ENABLED:
        try:
            watcher_service = WatcherService()
            watcher_service.start()
        except Exception as e:
            logging.getLogger(__name__).error("Failed to start WatcherService: %s", e)

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
    global watcher_service
    if watcher_service is not None:
        watcher_service.stop()


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "version": settings.VERSION,
    }
