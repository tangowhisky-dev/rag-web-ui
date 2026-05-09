import logging

from app.api.api_v1.api import api_router
from app.core.config import settings
from app.core.storage import init_storage
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

# Include routers
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.on_event("startup")
async def startup_event():
    # Initialize local file storage
    init_storage()


@app.get("/")
def root():
    return {"message": "Welcome to InsightCore API"}


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "version": settings.VERSION,
    }
