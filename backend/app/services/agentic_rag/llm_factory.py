"""Per-organisation LLM factory for the agent loop.

Role-aware resolution of API key, base URL, and model name. Each role has
its own fallback chain:

  Role-specific setting → OPENAI_* setting → .env default

All reads go through the settings service (3-tier precedence:
org override → app value → .env/config.py default).

Two roles for the retrieval/agent pipeline:
- "chat"     → primary model (think, plan, answer, sub-agents)
- "utility"  → utility model (synonyms, evaluation, compaction, extraction)

Ingestion roles (not part of the agent loop):
- "vision"   → vision / OCR model
- "graph"    → graph extraction model
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_openai import ChatOpenAI
from sqlalchemy.orm import Session

from app.services.settings_service import get_setting

logger = logging.getLogger(__name__)


# ── Monkey-patch: preserve reasoning_content from thinking models ─────────
# langchain-openai 1.6.0 drops `reasoning_content` from both streaming deltas
# and non-streaming messages. We patch the two converter functions to extract
# it into `additional_kwargs` so downstream code can access it.
import langchain_openai.chat_models.base as _lc_base
from typing import Any, Mapping, cast
from langchain_core.messages import (
    AIMessage, AIMessageChunk, BaseMessage, BaseMessageChunk,
    FunctionMessage, FunctionMessageChunk, HumanMessage, HumanMessageChunk,
    SystemMessage, SystemMessageChunk, ToolMessage, ToolMessageChunk,
)

_orig_convert_delta = _lc_base._convert_delta_to_message_chunk

def _patched_convert_delta(_dict: Mapping[str, Any], default_class: type[BaseMessageChunk]) -> BaseMessageChunk:
    chunk = _orig_convert_delta(_dict, default_class)
    rc = _dict.get("reasoning_content")
    if rc and isinstance(chunk, AIMessageChunk):
        chunk.additional_kwargs["reasoning_content"] = rc
    return chunk

_orig_convert_dict = _lc_base._convert_dict_to_message

def _patched_convert_dict(_dict: Mapping[str, Any]) -> BaseMessage:
    msg = _orig_convert_dict(_dict)
    rc = _dict.get("reasoning_content")
    if rc and isinstance(msg, AIMessage):
        msg.additional_kwargs["reasoning_content"] = rc
    return msg

_lc_base._convert_delta_to_message_chunk = _patched_convert_delta
_lc_base._convert_dict_to_message = _patched_convert_dict
logger.debug("[llm_factory] patched langchain-openai to preserve reasoning_content")


# Role → (role-specific key setting, role-specific base URL setting)
_ROLE_KEY_MAP = {
    "chat":      ("OPENAI_API_KEY",    "OPENAI_API_BASE"),
    "utility":   ("UTILITY_API_KEY",   "UTILITY_API_BASE"),
    "vision":    ("VISION_API_KEY",    "OPENAI_VISION_API_BASE"),
    "graph":     ("GRAPHRAG_API_KEY",  "GRAPHRAG_API_BASE"),
}


def get_org_llm(org_id: Optional[int], db: Session, role: str = "chat") -> dict:
    """Resolve OpenAI-compatible LLM config for ``org_id`` and ``role``.

    Roles:
    - "chat"      -> primary model (think, plan, answer, sub-agents)
    - "utility"   -> utility model (synonyms, evaluation, compaction, extraction)
    - "vision"    -> vision / OCR model (ingestion)
    - "graph"     -> graph extraction model (ingestion)

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
    if role == "utility":
        model_name = get_setting(db, "UTILITY_MODEL", org_id) or get_setting(db, "OPENAI_MODEL", org_id)
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
