from sqlalchemy import Column, Integer, String, ForeignKey, Text, DateTime, JSON, BigInteger
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import relationship
from app.models.base import Base, TimestampMixin
from app.models.datastore import DataStore
from datetime import datetime, timezone
import sqlalchemy as sa

class KnowledgeBaseDataStore(Base, TimestampMixin):
    __tablename__ = "knowledge_base_datastores"

    id = Column(Integer, primary_key=True, index=True)
    knowledge_base_id = Column(Integer, ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True)
    data_store_id = Column(Integer, ForeignKey("data_stores.id", ondelete="CASCADE"), nullable=False, index=True)

    __table_args__ = (
        sa.UniqueConstraint('knowledge_base_id', 'data_store_id'),
    )

    # Relationships
    knowledge_base = relationship("KnowledgeBase", back_populates="data_sources")
    data_store = relationship("DataStore", primaryjoin="KnowledgeBaseDataStore.data_store_id == DataStore.id", backref="kb_assignments")


class KnowledgeBase(Base, TimestampMixin):
    __tablename__ = "knowledge_bases"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(LONGTEXT)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    org_id = Column(Integer, ForeignKey('organisations.id'), nullable=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    # Note: We use event listener for conditional cascade:
    # - Documents with data_store_id=NULL (direct uploads) → delete with KB
    # - Documents with data_store_id!=NULL (from DataStore) → set kb_id=NULL, keep in DataStore
    documents = relationship("Document", back_populates="knowledge_base")
    user = relationship("User", back_populates="knowledge_bases")
    organisation = relationship('Organisation', back_populates='knowledge_bases')
    processing_tasks = relationship("ProcessingTask", back_populates="knowledge_base", cascade="all, delete-orphan")
    chunks = relationship("DocumentChunk", back_populates="knowledge_base", cascade="all, delete-orphan")
    document_uploads = relationship("DocumentUpload", back_populates="knowledge_base", cascade="all, delete-orphan")
    data_sources = relationship("KnowledgeBaseDataStore", back_populates="knowledge_base", cascade="all, delete-orphan")

class Document(Base, TimestampMixin):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    file_path = Column(String(1024), nullable=False)  # Path in local storage
    file_name = Column(String(255), nullable=False)  # Actual file name
    file_size = Column(BigInteger, nullable=False)  # File size in bytes
    content_type = Column(String(100), nullable=False)  # MIME type
    file_hash = Column(String(64), index=True)  # SHA-256 hash of file content
    knowledge_base_id = Column(Integer, ForeignKey("knowledge_bases.id", ondelete="SET NULL"), nullable=True, index=True)
    data_store_id = Column(Integer, ForeignKey("data_stores.id", ondelete="CASCADE"), nullable=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    knowledge_base = relationship("KnowledgeBase", back_populates="documents") 
    data_store = relationship("DataStore", back_populates="documents")
    processing_tasks = relationship("ProcessingTask", back_populates="document", cascade="all, delete-orphan")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

    __table_args__ = (
        # Ensure file_name is unique within each knowledge base
        sa.UniqueConstraint('knowledge_base_id', 'file_name', name='uq_kb_file_name'),
        # Ensure a file is only represented once per data store
        sa.UniqueConstraint('file_path', 'data_store_id', name='uq_document_file_path_datastore'),
    )

class DocumentUpload(Base):
    __tablename__ = "document_uploads"
    
    id = Column(Integer, primary_key=True, index=True)
    knowledge_base_id = Column(Integer, ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False)
    file_name = Column(String, nullable=False)
    file_hash = Column(String, nullable=False)
    file_size = Column(BigInteger, nullable=False)
    content_type = Column(String, nullable=False)
    temp_path = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    status = Column(String, nullable=False, server_default="pending")
    error_message = Column(Text)
    
    # Relationships
    knowledge_base = relationship("KnowledgeBase", back_populates="document_uploads")

class ProcessingTask(Base):
    __tablename__ = "processing_tasks"

    id = Column(Integer, primary_key=True, index=True)
    knowledge_base_id = Column(Integer, ForeignKey("knowledge_bases.id", ondelete="CASCADE"))
    data_store_id = Column(Integer, ForeignKey("data_stores.id", ondelete="CASCADE"), nullable=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=True, index=True)
    document_upload_id = Column(Integer, ForeignKey("document_uploads.id", ondelete="CASCADE"), nullable=True)
    status = Column(String(50), default="pending", index=True)  # pending, processing, completed, failed
    error_message = Column(Text, nullable=True)
    progress = Column(Integer, default=0, nullable=True)          # 0-100
    progress_message = Column(String(255), nullable=True)         # human-readable stage label
    # Graph extraction status: null = not attempted, pending, completed, failed
    graph_status = Column(String(50), nullable=True)
    graph_error = Column(Text, nullable=True)                     # last error if graph_status=failed
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    knowledge_base = relationship("KnowledgeBase", back_populates="processing_tasks")
    data_store = relationship("DataStore", back_populates="processing_tasks",
                               primaryjoin="ProcessingTask.data_store_id == DataStore.id")
    document = relationship("Document", back_populates="processing_tasks")
    document_upload = relationship("DocumentUpload", backref="processing_tasks")

class DocumentChunk(Base, TimestampMixin):
    __tablename__ = "document_chunks"

    id = Column(String(64), primary_key=True)  # SHA-256 hash as ID
    kb_id = Column(Integer, ForeignKey("knowledge_bases.id"), nullable=True)
    data_store_id = Column(Integer, ForeignKey("data_stores.id", ondelete="CASCADE"), nullable=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    file_name = Column(String(255), nullable=False)
    chunk_text = Column(LONGTEXT, nullable=False)   # the actual chunk text — FULLTEXT indexed
    chunk_index = Column(Integer, nullable=True)    # position within the document (0-based)
    chunk_metadata = Column(JSON, nullable=True)    # variable source metadata (page, source path, etc.)
    hash = Column(String(64), nullable=False, index=True)  # content hash for change detection

    # Relationships
    knowledge_base = relationship("KnowledgeBase", back_populates="chunks")
    data_store = relationship("DataStore", back_populates="chunks",
                              primaryjoin="DocumentChunk.data_store_id == DataStore.id")
    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        sa.Index('idx_kb_file_name', 'kb_id', 'file_name'),
    )
# Event listener for conditional document deletion on KB deletion
from sqlalchemy import event

@event.listens_for(KnowledgeBase, "before_delete")
def receive_before_delete(mapper, connection, target):
    """
    When a KnowledgeBase is deleted:
    - Delete documents that were directly uploaded (data_store_id IS NULL)
    - For documents from DataStore (data_store_id IS NOT NULL), only set knowledge_base_id to NULL
    """
    # Delete directly uploaded documents (no DataStore link)
    connection.execute(
        Document.__table__.delete().where(
            Document.knowledge_base_id == target.id,
            Document.data_store_id.is_(None)
        )
    )
    
    # For DataStore documents, just set kb_id to NULL (they persist in DataStore)
    connection.execute(
        Document.__table__.update().where(
            Document.knowledge_base_id == target.id,
            Document.data_store_id.isnot(None)
        ).values(knowledge_base_id=None)
    )
    
    # Also handle DocumentChunk - delete chunks for directly uploaded docs
    # Chunks for DataStore docs should remain (they're tied to the document, not KB)
    # First get IDs of directly uploaded docs being deleted
    from sqlalchemy import select
    stmt = select(Document.id).where(
        Document.knowledge_base_id == target.id,
        Document.data_store_id.is_(None)
    )
    doc_ids = [row[0] for row in connection.execute(stmt)]
    
    if doc_ids:
        connection.execute(
            DocumentChunk.__table__.delete().where(
                DocumentChunk.document_id.in_(doc_ids)
            )
        )
