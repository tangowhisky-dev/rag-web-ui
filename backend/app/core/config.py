import os
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Deployment & infrastructure configuration only.

    All runtime-configurable settings (LLM, retrieval, agent, memory, ingestion)
    are managed via the settings table (Super Admin UI / Admin UI) with defaults
    from the settings registry. Nothing in this class is runtime-settable.
    """

    PROJECT_NAME: str = "InsightCore"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"

    # Root logger level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # MySQL
    MYSQL_SERVER: str = os.getenv("MYSQL_SERVER", "localhost")
    MYSQL_PORT: int = int(os.getenv("MYSQL_PORT", "3306"))
    MYSQL_USER: str = os.getenv("MYSQL_USER", "ragwebui")
    MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD", "ragwebui")
    MYSQL_DATABASE: str = os.getenv("MYSQL_DATABASE", "ragwebui")
    SQLALCHEMY_DATABASE_URI: Optional[str] = None

    @property
    def get_database_url(self) -> str:
        if self.SQLALCHEMY_DATABASE_URI:
            return self.SQLALCHEMY_DATABASE_URI
        return (
            f"mysql+mysqlconnector://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}"
            f"@{self.MYSQL_SERVER}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
        )

    # JWT
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "360"))

    # Comma-separated list of trusted proxy IP addresses. The wildcard '*' trusts
    # all proxies. Only used for extracting the real client IP from X-Forwarded-*
    # headers; if the direct peer is not trusted the backend's peer IP is used.
    TRUSTED_PROXIES: str = os.getenv("TRUSTED_PROXIES", "")

    # File storage
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "/app/uploads")

    # Watcher — legacy fallback directory (per-org watch config is in Organisation model)
    WATCH_DIR: str = os.getenv("WATCH_DIR", "/app/uploads")
    # Use inotify (Linux native) instead of polling observer.
    # inotify provides near-instant event delivery on ext4/xfs with Docker bind-mounts.
    # Falls back to PollingObserver on macOS, Windows, or when inotify is unavailable.
    WATCHER_USE_INOTIFY: bool = os.getenv("WATCHER_USE_INOTIFY", "true").lower() == "true"

    # Embedded models (not user-changeable at runtime — loaded once into process singletons)
    SPLADE_MODEL: str = os.getenv("SPLADE_MODEL", "prithivida/Splade_PP_en_v1")
    FASTEMBED_CACHE_DIR: str = os.getenv("FASTEMBED_CACHE_DIR", "/app/assets/fastembed")
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-12-v2")
    RERANKER_CACHE_DIR: str = os.getenv("RERANKER_CACHE_DIR", "/app/assets/reranker")

    # Tokenizer for accurate token counting (local HuggingFace tokenizer directory)
    TOKENIZER_MODEL: Optional[str] = os.getenv("TOKENIZER_MODEL") or None

    # Qdrant vector store
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "qdrant")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_GRPC_PORT: int = int(os.getenv("QDRANT_GRPC_PORT", "6334"))

    # Redis Stack — LangGraph short-term (checkpointer) + long-term (store) memory
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    REDIS_HOST: str = os.getenv("REDIS_HOST", "redis")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_INSIGHT_PORT: int = int(os.getenv("REDIS_INSIGHT_PORT", "8001"))

    # Neo4j (graph store)
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "ragwebui_neo4j")

    # Sandbox backend for code_execute tool
    SANDBOX_BACKEND: str = os.getenv("SANDBOX_BACKEND", "restrictedpython")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
