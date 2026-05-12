from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

engine = create_engine(
    settings.get_database_url,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,    # re-validate connections before use; prevents stale-connection errors
    pool_recycle=3600,     # recycle connections every hour so MySQL doesn't close them first
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 