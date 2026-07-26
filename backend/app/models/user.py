import enum
from sqlalchemy import Boolean, Column, Integer, String, ForeignKey, DateTime, func
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import relationship
from app.models.base import Base, TimestampMixin


class UserRole(str, enum.Enum):
    user = 'user'
    admin = 'admin'
    super_admin = 'super_admin'


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    role = Column(SAEnum(UserRole), nullable=False, default=UserRole.user, server_default='user')
    org_id = Column(Integer, ForeignKey('organisations.id'), nullable=True, index=True)
    token_version = Column(Integer, default=0, server_default='0', nullable=False)

    # Relationships
    # cascade='all, delete-orphan' ensures KBs, chats, and their children are deleted when the user is deleted
    knowledge_bases = relationship("KnowledgeBase", back_populates="user", cascade="all, delete-orphan")
    chats = relationship("Chat", back_populates="user", cascade="all, delete-orphan")
    organisation = relationship('Organisation', back_populates='users')