from fastapi import APIRouter
from app.api.api_v1 import auth, knowledge_base, chat, query, chat_files, folders, admin, watcher
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
api_router.include_router(watcher.router, prefix="/admin", tags=["watcher"])


@api_router.get("/config", tags=["config"])
def get_client_config():
    """Expose non-sensitive runtime configuration to the frontend."""
    return {
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.chunk_overlap,
    }