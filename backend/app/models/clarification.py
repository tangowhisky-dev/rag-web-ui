"""ClarificationRequest model — tracks in-flight clarification requests during agent streaming.

When the agent needs user clarification, it creates a ClarificationRequest row.
The frontend polls /clarification/pending to check for outstanding requests.
The user responds via /clarification/submit, which sets the response and triggers
agent resume.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime, timezone

from app.models.base import Base


class ClarificationRequest(Base):
    """Tracks a clarification request issued by the agent during streaming."""
    __tablename__ = "clarification_requests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    chat_id = Column(Integer, ForeignKey("chats.id", ondelete="CASCADE"), nullable=False, index=True)
    assistant_message_id = Column(Integer, ForeignKey("messages.id"), nullable=False, index=True)

    # The agent's question to the user
    question = Column(Text, nullable=False)
    # Optional suggested answers (sensible defaults)
    options = Column(JSON, nullable=True)  # ["Option 1", "Option 2"]
    # Why clarification is needed
    rationale = Column(Text, nullable=True)

    # User's response (set after user answers)
    user_response = Column(Text, nullable=True)

    # Status: pending | answered | expired
    status = Column(String(20), nullable=False, default="pending", server_default="pending")

    # Attempt tracking (max 2)
    attempt = Column(Integer, nullable=False, default=1, server_default="1")

    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    answered_at = Column(DateTime, nullable=True)

    # Relationships
    chat = relationship("Chat", backref="clarification_requests")
    assistant_message = relationship("Message", foreign_keys=[assistant_message_id])
