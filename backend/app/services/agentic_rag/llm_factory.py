"""Per-organisation LLM factory for the agent loop.

Role-aware resolution of API key, base URL, and model name. Each role has
its own fallback chain:

  Role-specific setting → OPENAI_* setting → .env default

All reads go through the settings service (3-tier precedence:
org override → app value → .env/config.py default).
"""

from __future__ import annotations

from typing import Optional

from langchain_openai import ChatOpenAI
from sqlalchemy.orm import Session

from app.services.settings_service import get_setting


# Role → (role-specific key setting, role-specific base URL setting)
_ROLE_KEY_MAP = {
    "chat":      ("OPENAI_API_KEY",    "OPENAI_API_BASE"),
    "query":     ("QUERY_API_KEY",     "QUERY_API_BASE"),
    "reasoning": ("REASONING_API_KEY", "REASONING_API_BASE"),
    "vision":    ("VISION_API_KEY",    "OPENAI_VISION_API_BASE"),
    "graph":     ("GRAPHRAG_API_KEY",  "GRAPHRAG_API_BASE"),
}


def get_org_llm(org_id: Optional[int], db: Session, role: str = "chat") -> dict:
    """Resolve OpenAI-compatible LLM config for ``org_id`` and ``role``.

    Roles:
    - "chat"      -> main response model
    - "query"     -> rewrite / summarisation / extraction model
    - "reasoning" -> reasoning / thinking model
    - "vision"    -> vision / OCR model
    - "graph"     -> graph extraction model

    Key and base URL resolve with per-role fallback to the main OPENAI_* settings.
    Model resolution: role-specific model → OPENAI_MODEL.
    """
    role_key, role_base = _ROLE_KEY_MAP.get(role, ("OPENAI_API_KEY", "OPENAI_API_BASE"))

    # Key: role-specific → OPENAI_API_KEY (same tier) → placeholder.
    # Local servers (LM Studio, Ollama) don't require a key, but the OpenAI
    # client library rejects None/empty — supply a placeholder when unset.
    api_key = get_setting(db, role_key, org_id) or get_setting(db, "OPENAI_API_KEY", org_id)
    if not api_key:
        api_key = "not-required"

    # Base URL: role-specific → OPENAI_API_BASE (same tier) → .env fallback
    api_base = get_setting(db, role_base, org_id) or get_setting(db, "OPENAI_API_BASE", org_id)

    # Model: role-specific model → OPENAI_MODEL
    if role == "query":
        model_name = get_setting(db, "QUERY_MODEL", org_id) or get_setting(db, "OPENAI_MODEL", org_id)
    elif role == "reasoning":
        model_name = get_setting(db, "REASONING_MODEL", org_id) or get_setting(db, "OPENAI_MODEL", org_id)
    elif role == "vision":
        model_name = get_setting(db, "VISION_MODEL", org_id) or get_setting(db, "OPENAI_MODEL", org_id)
    elif role == "graph":
        model_name = get_setting(db, "GRAPHRAG_LLM", org_id) or get_setting(db, "OPENAI_MODEL", org_id)
    else:
        model_name = get_setting(db, "OPENAI_MODEL", org_id)

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
