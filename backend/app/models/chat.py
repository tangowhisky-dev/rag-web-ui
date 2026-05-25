from sqlalchemy import Column, Integer, String, ForeignKey, Boolean, Table, BigInteger, Text, DateTime
from sqlalchemy.dialects.mysql import LONGTEXT, JSON
from sqlalchemy.orm import relationship
from app.models.base import Base, TimestampMixin
from datetime import datetime


class Folder(Base, TimestampMixin):
    """Chat folder — a named group owned by a single user."""
    __tablename__ = "folders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    # Relationships
    chats = relationship("Chat", back_populates="folder")


# Association table for many-to-many relationship between Chat and KnowledgeBase
chat_knowledge_bases = Table(
    "chat_knowledge_bases",
    Base.metadata,
    Column("chat_id", Integer, ForeignKey("chats.id"), primary_key=True),
    Column("knowledge_base_id", Integer, ForeignKey("knowledge_bases.id"), primary_key=True),
)

class Chat(Base, TimestampMixin):
    __tablename__ = "chats"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    history_summary = Column(LONGTEXT, nullable=True)  # rolling summary of messages beyond the sliding window
    use_graph_rag = Column(Boolean, nullable=False, default=False, server_default="0")
    use_dense     = Column(Boolean, nullable=False, default=True,  server_default="1")
    use_sparse    = Column(Boolean, nullable=False, default=True,  server_default="1")
    use_exact     = Column(Boolean, nullable=False, default=True,  server_default="1")
    pinned        = Column(Boolean, nullable=False, default=False, server_default="0")
    folder_id     = Column(Integer, ForeignKey("folders.id", ondelete="SET NULL"), nullable=True, index=True)

    # Relationships
    messages = relationship("Message", back_populates="chat", cascade="all, delete-orphan")
    chat_files = relationship("ChatFile", back_populates="chat", cascade="all, delete-orphan")
    user = relationship("User", back_populates="chats")
    folder = relationship("Folder", back_populates="chats")
    knowledge_bases = relationship(
        "KnowledgeBase",
        secondary=chat_knowledge_bases,
        backref="chats"
    )

class Message(Base, TimestampMixin):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(LONGTEXT, nullable=False)
    role = Column(String(50), nullable=False)
    chat_id = Column(Integer, ForeignKey("chats.id"), nullable=False)
    # Branching fields
    parent_message_id = Column(Integer, ForeignKey("messages.id"), nullable=True, index=True)
    branch_index = Column(Integer, nullable=False, default=0, server_default="0")
    # Confidence fields — populated for assistant messages only
    confidence_level = Column(String(20), nullable=True)
    confidence_score = Column(Integer, nullable=True)
    confidence_breakdown = Column(LONGTEXT, nullable=True)  # JSON string

    # Relationships
    chat = relationship("Chat", back_populates="messages")
    siblings_rel = relationship(
        "Message",
        primaryjoin="Message.parent_message_id == foreign(Message.id)",
        uselist=True,
        viewonly=True,
    )

class ChatFile(Base):
    """File uploaded within a chat session — scoped to a single chat, deleted with it."""
    __tablename__ = "chat_files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    chat_id = Column(Integer, ForeignKey("chats.id", ondelete="CASCADE"), nullable=False, index=True)
    message_id = Column(Integer, ForeignKey("messages.id", ondelete="SET NULL"), nullable=True, index=True)
    file_name = Column(String(255), nullable=False)
    stored_path = Column(String(512), nullable=True)  # absolute path on disk
    file_size = Column(BigInteger, nullable=False)
    content_type = Column(String(100), nullable=False)
    markdown_content = Column(Text, nullable=True)   # extracted markdown; set after processing
    token_count = Column(Integer, nullable=True)      # estimated tokens in markdown_content
    # status: processing | ready | error
    status = Column(String(20), nullable=False, default="processing")
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    chat = relationship("Chat", back_populates="chat_files")
    message = relationship("Message", backref="chat_file", foreign_keys=[message_id])
