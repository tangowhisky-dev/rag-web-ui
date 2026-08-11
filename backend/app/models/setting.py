from sqlalchemy import Column, Integer, String, Text, ForeignKey, Index
from .base import Base, TimestampMixin


class Setting(Base, TimestampMixin):
    """Generic key-value settings table with 3-tier precedence.

    scope='app', org_id=NULL  → app-level default (set by super admin)
    scope='org', org_id=<id>  → org-level override (set by admin)

    Resolution: org override → app value → .env/config.py default (via registry).
    """
    __tablename__ = "settings"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(128), nullable=False, index=True)
    scope = Column(String(8), nullable=False)  # "app" | "org"
    org_id = Column(Integer, ForeignKey("organisations.id", ondelete="CASCADE"), nullable=True)
    value = Column(Text, nullable=True)  # JSON-encoded
    updated_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    __table_args__ = (
        Index("uq_org_key", "org_id", "key", unique=True),
        Index("idx_settings_scope", "scope"),
    )
