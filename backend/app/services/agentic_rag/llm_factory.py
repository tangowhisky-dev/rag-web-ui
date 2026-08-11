"""Per-organisation LLM factory for the agent loop."""

from __future__ import annotations

from typing import Optional

from langchain_openai import ChatOpenAI
from sqlalchemy.orm import Session

from app.core.config import settings
from app.services.settings_service import get_setting


def get_org_llm(org_id: Optional[int], db: Session, role: str = "chat") -> dict:
    """Resolve OpenAI-compatible LLM config for ``org_id`` and ``role``.

    Roles:
    - "chat"     -> main response model
    - "query"    -> rewrite / summarisation / extraction model
    - "reasoning" -> reasoning / thinking model

    Reads from the unified settings service (3-tier precedence:
    org override → app value → .env/config.py default).
    """
    api_base = get_setting(db, "OPENAI_API_BASE", org_id)
    model_name = get_setting(db, "OPENAI_MODEL", org_id)
    query_model = get_setting(db, "QUERY_MODEL", org_id) or model_name
    reasoning_model = get_setting(db, "REASONING_MODEL", org_id) or model_name
    api_key = settings.OPENAI_API_KEY  # always from .env (secret)

    if role == "query":
        model_name = query_model
    elif role == "reasoning":
        model_name = reasoning_model

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
