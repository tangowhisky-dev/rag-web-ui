from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.dialects.mysql import LONGTEXT, JSON
from app.models.base import Base, TimestampMixin


class SearchHistory(Base, TimestampMixin):
    """Log of user search queries for auditing purposes.

    Stores the original query, abbreviation-expanded query, selected KB IDs,
    result count, and latency. Does NOT store the search results themselves.
    """
    __tablename__ = "search_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    query = Column(String(1024), nullable=False)
    expanded_query = Column(LONGTEXT, nullable=True)
    kb_ids = Column(JSON, nullable=True)
    result_count = Column(Integer, default=0, nullable=False)
    latency_ms = Column(Integer, default=0, nullable=False)
