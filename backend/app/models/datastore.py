"""DataStore model — first-class document source (local folder).

A DataStore represents a folder on the local filesystem that the watcher
monitors for new/modified files.  Multiple organisations can share the same
DataStore via the ``OrganizationDataStore`` junction table.
"""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Boolean,
    ForeignKey,
    UniqueConstraint,
    DateTime,
)
from sqlalchemy.orm import relationship
from datetime import datetime, timezone

from .base import Base


class DataStore(Base):
    """A local folder that can be watched for document ingestion.

    Can be assigned to multiple organisations via the junction table.
    """

    __tablename__ = "data_stores"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # Folder configuration
    folder_path = Column(String(512), nullable=False, unique=True)
    scan_pattern = Column(String(100), default="*")  # e.g. "*.pdf,*.docx"

    # Status
    is_active = Column(Boolean, default=True)

    # Auto-scan settings
    auto_scan_enabled = Column(Boolean, default=False)
    auto_scan_interval_minutes = Column(Integer, default=60)

    # Ingestion tracking
    last_scan_at = Column(DateTime, nullable=True)
    last_scan_status = Column(String(50), default="never")
    last_scan_error = Column(Text, nullable=True)
    last_scan_total_files = Column(Integer, default=0)
    last_scan_processed = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Relationships
    organization_links = relationship(
        "OrganizationDataStore",
        back_populates="data_store",
        cascade="all, delete-orphan",
    )
    documents = relationship(
        "Document", back_populates="data_store", cascade="all, delete-orphan"
    )
    chunks = relationship(
        "DocumentChunk", back_populates="data_store", cascade="all, delete-orphan"
    )
    processing_tasks = relationship(
        "ProcessingTask", back_populates="data_store", cascade="all, delete-orphan",
        primaryjoin="DataStore.id == ProcessingTask.data_store_id",
    )


class OrganizationDataStore(Base):
    """Junction table: many-to-many between Organisation and DataStore.

    Allows multiple organisations to share the same DataStore folder.
    """

    __tablename__ = "organization_data_stores"

    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(
        Integer, ForeignKey("organisations.id", ondelete="CASCADE"), nullable=False
    )
    data_store_id = Column(
        Integer, ForeignKey("data_stores.id", ondelete="CASCADE"), nullable=False
    )
    is_active = Column(Boolean, default=True)
    assigned_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    __table_args__ = (
        UniqueConstraint("org_id", "data_store_id", name="uq_org_datastore"),
    )

    # Relationships
    organisation = relationship("Organisation", back_populates="data_store_links")
    data_store = relationship("DataStore", back_populates="organization_links")
