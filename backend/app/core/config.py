import os
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "RAG Web UI"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api"

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
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "10080"))

    # File storage
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "/app/uploads")

    # Retrieval
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "6"))

    # Hybrid search weights (BM25 + dense vector) and result count.
    HYBRID_DENSE_WEIGHT: float = float(os.getenv("HYBRID_DENSE_WEIGHT", "0.7"))
    HYBRID_SPARSE_WEIGHT: float = float(os.getenv("HYBRID_SPARSE_WEIGHT", "0.3"))
    # Cosine distance cutoff for the dense leg (Chroma: 0=identical, 2=opposite).
    # Excludes documents with no real semantic signal from either leg.
    # Default 2.0 keeps all dense results; lower (e.g. 1.2) to be more selective.
    HYBRID_MAX_DENSE_DISTANCE: float = float(os.getenv("HYBRID_MAX_DENSE_DISTANCE", "2.0"))

    # LLM + Embeddings (OpenAI-compatible)
    OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "http://localhost:1234/v1")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "lmstudio")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "local-model")
    OPENAI_EMBEDDINGS_MODEL: str = os.getenv("OPENAI_EMBEDDINGS_MODEL", "local-embedding-model")

    # Vector Store
    VECTOR_STORE_TYPE: str = os.getenv("VECTOR_STORE_TYPE", "chroma")
    CHROMA_DB_HOST: str = os.getenv("CHROMA_DB_HOST", "chromadb")
    CHROMA_DB_PORT: int = int(os.getenv("CHROMA_DB_PORT", "8000"))

    class Config:
        env_file = ".env"


settings = Settings()
