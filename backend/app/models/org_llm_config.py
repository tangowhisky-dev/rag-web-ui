from sqlalchemy import Column, Integer, String, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship

from .base import Base, TimestampMixin


class OrgLLMConfig(Base, TimestampMixin):
    __tablename__ = "org_llm_configs"

    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(Integer, ForeignKey("organisations.id"), unique=True, index=True, nullable=False)
    api_base = Column(String(512), nullable=True)
    model_name = Column(String(255), nullable=True)
    query_model = Column(String(255), nullable=True)

    organisation = relationship("Organisation", back_populates="llm_config")
