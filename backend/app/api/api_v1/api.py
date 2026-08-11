from fastapi import APIRouter
from app.api.api_v1 import auth, knowledge_base, chat, query, chat_files, folders, admin
from app.api.api_v1 import datastores, datastore_scan, datastore_recovery
from app.api.api_v1 import settings as settings_router
from app.core.config import settings

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(knowledge_base.router, prefix="/knowledge-base", tags=["knowledge-base"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(chat_files.router, prefix="/chat", tags=["chat-files"])
api_router.include_router(query.router, prefix="/query", tags=["query"])
api_router.include_router(folders.router, prefix="/folders", tags=["folders"])
api_router.include_router(admin.org_router, prefix="/admin", tags=["admin"])
api_router.include_router(admin.user_router, prefix="/admin", tags=["admin"])
api_router.include_router(settings_router.app_router, prefix="/admin", tags=["settings"])
api_router.include_router(settings_router.org_router, prefix="/admin", tags=["settings"])
api_router.include_router(datastores.router, prefix="/admin", tags=["datastores"])
api_router.include_router(datastore_scan.router, prefix="/admin", tags=["datastores"])
api_router.include_router(datastore_recovery.router, prefix="/admin", tags=["datastores"])


@api_router.get("/config", tags=["config"])
def get_client_config():
    """Expose non-sensitive runtime configuration to the frontend."""
    from app.db.session import SessionLocal
    from app.services.settings_service import get_setting
    db = SessionLocal()
    try:
        chunk_size = get_setting(db, "CHUNK_SIZE", None)
        overlap_pct = get_setting(db, "OVERLAP_PERCENTAGE", None)
        return {
            "chunk_size": chunk_size,
            "chunk_overlap": int(chunk_size * overlap_pct),
        }
    finally:
        db.close()
