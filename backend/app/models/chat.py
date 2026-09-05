from uuid import uuid4
from sqlalchemy import Column, Integer, String, ForeignKey, Boolean, Table, BigInteger, Text, DateTime, Float, CheckConstraint
from sqlalchemy.dialects.mysql import LONGTEXT, JSON
from sqlalchemy.orm import relationship, backref
from app.models.base import Base, TimestampMixin
from datetime import datetime, timezone


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
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    history_summary = Column(LONGTEXT, nullable=True)  # rolling summary of messages beyond the sliding window
    pinned        = Column(Boolean, nullable=False, default=False, server_default="0")
    folder_id     = Column(Integer, ForeignKey("folders.id", ondelete="SET NULL"), nullable=True, index=True)
    org_id        = Column(Integer, ForeignKey('organisations.id'), nullable=True, index=True)
    # JSON map of {parent_message_id: selected_child_message_id} tracking which
    # branch the user is currently viewing for each branching point.
    # On reload, the message loader uses this to pick the right branch.
    # Defaults to the latest branch (highest branch_index) when not set.
    active_branches = Column(JSON, nullable=True)

    # Relationships
    messages = relationship("Message", back_populates="chat", cascade="all, delete-orphan")
    chat_files = relationship("ChatFile", back_populates="chat", cascade="all, delete-orphan")
    user = relationship("User", back_populates="chats")
    folder = relationship("Folder", back_populates="chats")
    organisation = relationship('Organisation', back_populates='chats')
    knowledge_bases = relationship(
        "KnowledgeBase",
        secondary=chat_knowledge_bases,
        backref="chats"
    )

class MessageCitation(Base):
    """Links a retrieved document citation to a chat message."""
    __tablename__ = "message_citations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(Integer, ForeignKey("messages.id", ondelete="CASCADE"), nullable=False, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False, default=0)
    citation_index = Column(Integer, nullable=False)
    citation_metadata = Column(JSON, nullable=True)  # transient: score, dense_rank, sparse_rank, exact_rank, retrieval_leg
    # New CitationRef fields (nullable for backward compat with old rows)
    citation_kind = Column(String(20), nullable=True)
    section = Column(String(255), nullable=True)
    start_char = Column(Integer, nullable=True)
    end_char = Column(Integer, nullable=True)
    start_line = Column(Integer, nullable=True)
    end_line = Column(Integer, nullable=True)
    page = Column(Integer, nullable=True)
    match_line = Column(Integer, nullable=True)
    source_tool = Column(String(50), nullable=True)

    # Relationships
    document = relationship("Document")
    message = relationship("Message", back_populates="citations")

    class Config:
        from_attributes = True


class Message(Base, TimestampMixin):
    __tablename__ = "messages"
    # The self-referential parent_message_id FK has ondelete="CASCADE" at the
    # DB level. When SQLAlchemy deletes a parent message via the ORM cascade,
    # the DB also removes its children — so SQLAlchemy's row count check
    # (expected vs actual deleted) mismatches. This is benign: the rows are
    # gone either way. Suppress the warning so it doesn't clutter logs.
    __mapper_args__ = {"confirm_deleted_rows": False}

    id = Column(Integer, primary_key=True, index=True)
    content = Column(LONGTEXT, nullable=False)
    role = Column(String(50), nullable=False)
    chat_id = Column(Integer, ForeignKey("chats.id"), nullable=False)
    # Branching fields
    parent_message_id = Column(Integer, ForeignKey("messages.id", ondelete="CASCADE"), nullable=True, index=True)
    branch_index = Column(Integer, nullable=False, default=0, server_default="0")
    # Confidence fields — populated for assistant messages only
    confidence_level = Column(String(20), nullable=True)
    confidence_score = Column(Integer, nullable=True)
    confidence_breakdown = Column(LONGTEXT, nullable=True)  # JSON string
    # Final answer evaluation (from answer_evaluation_node)
    final_confidence = Column(Float, nullable=True)
    final_confidence_level = Column(String(20), nullable=True)
    faithfulness = Column(Integer, nullable=True)
    completeness = Column(Integer, nullable=True)
    retrieval_score = Column(Integer, nullable=True)   # 0-100, from retrieval_confidence
    rewritten_query = Column(LONGTEXT, nullable=True)  # standalone retrieval query after rewrite
    expanded_query = Column(LONGTEXT, nullable=True)   # abbreviation-expanded user query (original + glossary suffix)
    last_answer_object = Column(JSON, nullable=True)    # structured summary of the assistant answer
    plan = Column(JSON, nullable=True)                  # plan object for this turn (debugging/replay)
    tool_calls = Column(JSON, nullable=True)            # array of tool-call/observation records

    # Relationships
    chat = relationship("Chat", back_populates="messages")
    citations = relationship("MessageCitation", back_populates="message", cascade="all, delete-orphan")
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
    stored_path = Column(String(1024), nullable=True)  # absolute path on disk
    file_size = Column(BigInteger, nullable=False)
    content_type = Column(String(100), nullable=False)
    markdown_content = Column(Text, nullable=True)   # extracted markdown; set after processing
    token_count = Column(Integer, nullable=True)      # estimated tokens in markdown_content
    # status: processing | ready | error
    status = Column(String(20), nullable=False, default="processing")
    error_message = Column(Text, nullable=True)
    # True for agent-generated Office documents (OfficeCLI), False for user uploads.
    is_generated = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    chat = relationship("Chat", back_populates="chat_files")
    message = relationship("Message", backref="chat_file", foreign_keys=[message_id])


class ToolCallAudit(Base):
    """Audit trail for every tool call made by the agent loop."""
    __tablename__ = "tool_call_audit"

    __table_args__ = (
        CheckConstraint(
            "status IN ('ok','error','denied','timeout','budget_exceeded')",
            name="tool_call_audit_status_check",
        ),
    )

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    chat_id = Column(Integer, ForeignKey("chats.id", ondelete="CASCADE"), nullable=False, index=True)
    message_id = Column(Integer, ForeignKey("messages.id", ondelete="SET NULL"), nullable=True, index=True)
    iteration = Column(Integer, nullable=False, default=0)
    tool_name = Column(String(50), nullable=False)
    arguments = Column(JSON, nullable=True)
    result_summary = Column(JSON, nullable=True)
    tokens_in = Column(Integer, nullable=False, default=0)
    tokens_out = Column(Integer, nullable=False, default=0)
    latency_ms = Column(Integer, nullable=False, default=0)
    status = Column(String(20), nullable=False)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    chat = relationship("Chat", backref=backref("tool_call_audits", cascade="all, delete-orphan"))
    message = relationship("Message", backref="tool_call_audits")
