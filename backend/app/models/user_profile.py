"""User profile model — persistent storage for the autonomous agent's user preferences.

Scoped by (user_id, org_id) to enforce multi-tenant isolation.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, UniqueConstraint, func
from datetime import datetime, timezone

from app.models.base import Base


class UserProfile(Base):
    """Persistent user profile for the autonomous agent."""

    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False, index=True)
    org_id = Column(Integer, nullable=True, index=True)

    # JSON: {category: {key: {category, key, value, source, confidence}}}
    preferences_json = Column(Text, nullable=True, default="{}")

    # JSON: list of {description, query_type, preferred_response_format, recorded_at}
    query_patterns_json = Column(Text, nullable=True, default="[]")

    # JSON: list of domain focus keywords
    domain_focus_json = Column(Text, nullable=True, default="[]")

    communication_style = Column(String(32), nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    __table_args__ = (
        UniqueConstraint("user_id", "org_id", name="uq_user_profiles_user_org"),
    )
