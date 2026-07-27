"""Per-organisation LLM factory for the agent loop."""

from __future__ import annotations

from typing import Optional

from langchain_openai import ChatOpenAI
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.org_llm_config import OrgLLMConfig


def get_org_llm(org_id: Optional[int], db: Session, role: str = "chat") -> dict:
    """Resolve OpenAI-compatible LLM config for ``org_id`` and ``role``.

    Roles:
    - "chat"    -> main response model
    - "query"   -> rewrite / summarisation / extraction model

    Falls back to ``settings`` when no per-org config exists.
    """
    api_base = settings.OPENAI_API_BASE
    model_name = settings.OPENAI_MODEL
    query_model = settings.effective_query_model
    api_key = settings.OPENAI_API_KEY

    if org_id is not None and db is not None:
        row = db.query(OrgLLMConfig).filter(OrgLLMConfig.org_id == org_id).first()
        if row:
            if row.api_base:
                api_base = row.api_base
            if row.model_name:
                model_name = row.model_name
            if row.query_model:
                query_model = row.query_model

    if role == "query":
        model_name = query_model or model_name

    return {
        "api_base": api_base,
        "model_name": model_name,
        "api_key": api_key,
    }


def build_chat_llm(
    org_id: Optional[int],
    db: Session,
    role: str = "chat",
    temperature: float = 0.7,
    streaming: bool = False,
    **kwargs,
) -> ChatOpenAI:
    """Return a configured ``ChatOpenAI`` instance for the given org and role."""
    cfg = get_org_llm(org_id, db, role=role)
    return ChatOpenAI(
        openai_api_base=cfg["api_base"],
        openai_api_key=cfg["api_key"],
        model=cfg["model_name"],
        temperature=temperature,
        streaming=streaming,
        **kwargs,
    )
