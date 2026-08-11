from sqlalchemy import Column, Integer, String, ForeignKey, UniqueConstraint, Text
from sqlalchemy.orm import relationship

from .base import Base, TimestampMixin


class Organisation(Base, TimestampMixin):
    __tablename__ = "organisations"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    parent_id = Column(Integer, ForeignKey("organisations.id"), nullable=True, index=True)
    path = Column(String(1024), nullable=True)  # materialized path e.g. "/1/3/7"

    # Adjacency-list relationships
    parent = relationship("Organisation", remote_side="Organisation.id", back_populates="children")
    children = relationship("Organisation", back_populates="parent")

    # Reverse relationships (populated by FK on the other models)
    users = relationship("User", back_populates="organisation")
    knowledge_bases = relationship("KnowledgeBase", back_populates="organisation")
    chats = relationship("Chat", back_populates="organisation")
    abbreviations = relationship("OrgAbbreviation", back_populates="organisation", cascade="all, delete-orphan")
    data_store_links = relationship(
        "OrganizationDataStore",
        back_populates="organisation",
        cascade="all, delete-orphan",
    )


class OrgAbbreviation(Base):
    __tablename__ = "org_abbreviations"

    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(Integer, ForeignKey("organisations.id"), nullable=False, index=True)
    short = Column(String(64), nullable=False)
    expansion = Column(String(512), nullable=False)

    __table_args__ = (UniqueConstraint("org_id", "short", name="uq_org_abbreviations_org_short"),)

    organisation = relationship("Organisation", back_populates="abbreviations")
