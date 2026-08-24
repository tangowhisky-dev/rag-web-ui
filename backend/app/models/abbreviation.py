"""SQLAlchemy models for abbreviation lists and abbreviations."""
from sqlalchemy import Column, Integer, String, Text, Boolean, ForeignKey, Index
from sqlalchemy.orm import relationship

from .base import Base, TimestampMixin


class AbbreviationList(Base, TimestampMixin):
    """A named list of abbreviations uploaded as a CSV file.

    org_id=NULL  → universal list (uploaded by super_admin, available to all orgs)
    org_id=<id>  → org-specific list (uploaded by admin, supplements universal lists)
    """
    __tablename__ = "abbreviation_lists"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    org_id = Column(Integer, ForeignKey("organisations.id", ondelete="CASCADE"), nullable=True, index=True)
    is_enabled = Column(Boolean, nullable=False, default=True, index=True)
    row_count = Column(Integer, nullable=False, default=0)
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    abbreviations = relationship("Abbreviation", back_populates="list", cascade="all, delete-orphan")


class Abbreviation(Base):
    """A single abbreviation→expansion mapping. Multiple rows per abbreviation (multi-meaning)."""
    __tablename__ = "abbreviations"

    id = Column(Integer, primary_key=True, index=True)
    list_id = Column(Integer, ForeignKey("abbreviation_lists.id", ondelete="CASCADE"), nullable=False, index=True)
    abbreviation = Column(String(128), nullable=False, index=True)
    expanded_form = Column(String(1024), nullable=False)
    category = Column(String(255), nullable=True)

    list = relationship("AbbreviationList", back_populates="abbreviations")
