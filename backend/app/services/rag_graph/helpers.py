"""Helper functions for the RAG graph — LLM access, doc serialization, dedup, context building.

Split from rag_graph.py for maintainability.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.services.utils import content_hash

logger = logging.getLogger(__name__)


def _get_llm(model_name: Optional[str] = None, temperature: float = 0.0, api_base: Optional[str] = None):
    return ChatOpenAI(
        model=model_name or settings.OPENAI_MODEL,
        temperature=temperature,
        openai_api_base=api_base or settings.OPENAI_API_BASE,
        openai_api_key=settings.OPENAI_API_KEY,
        streaming=True,
    )


async def _invoke_structured(llm, messages: list, schema: type) -> str:
    """
    Invoke LLM with json_schema constrained output.
    Falls back to unconstrained invocation if the server rejects the format.
    strict=False intentional: local models often refuse strict mode.
    """
    json_schema_format = {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "strict": False,
            "schema": schema.model_json_schema(),
        },
    }
    try:
        resp = await llm.ainvoke(messages, response_format=json_schema_format)
    except Exception as exc:
        err = str(exc)
        if "response_format" in err or "json_schema" in err or "400" in err:
            resp = await llm.ainvoke(messages)
        else:
            raise
    return resp.content


def _serialise_doc(doc: Any) -> dict:
    if isinstance(doc, dict):
        return doc
    if hasattr(doc, "page_content"):
        return {"page_content": doc.page_content, "metadata": dict(doc.metadata)}
    return {"page_content": str(doc), "metadata": {}}


def _dedup_and_reinforce(doc_lists: List[List[dict]]) -> List[dict]:
    """
    Merge multiple lists of serialised docs.
    A chunk found in N sub-query results has its score multiplied (reinforced).
    Returns sorted by reinforced score descending.
    """
    seen: Dict[str, dict] = {}
    for docs in doc_lists:
        for doc in docs:
            text = doc.get("page_content", "")
            h = content_hash(text)
            meta = doc.get("metadata", {})
            score = float(meta.get("_rrf_score", meta.get("score", 0.001)))
            if h in seen:
                prev_meta = seen[h].get("metadata", {})
                prev_score = float(prev_meta.get("_reinforced_score", score))
                new_meta = dict(prev_meta)
                new_meta["_reinforced_score"] = prev_score + score
                new_meta["_retrieval_count"] = prev_meta.get("_retrieval_count", 1) + 1
                seen[h] = {"page_content": text, "metadata": new_meta}
            else:
                new_meta = dict(meta)
                new_meta["_reinforced_score"] = score
                new_meta["_retrieval_count"] = 1
                seen[h] = {"page_content": text, "metadata": new_meta}

    result = list(seen.values())
    result.sort(key=lambda d: d.get("metadata", {}).get("_reinforced_score", 0), reverse=True)
    return result


def _build_context_string(docs: List[dict], file_markdown: Optional[str] = None) -> str:
    """
    Build context string for the LLM.

    Chat history docs (metadata._source_type == 'chat_history') are placed first
    under a plain [Prior Answer] label with no number — the LLM is instructed
    not to cite them. KB chunks follow with sequential [KB-N] numbering so
    citations remain consistent.
    """
    parts: List[str] = []

    # ── Chat history docs first — no citation number ───────────────────────────
    for doc in docs:
        if doc.get("metadata", {}).get("_source_type") == "chat_history":
            content = doc.get("page_content", "").strip()
            parts.append(f"[Prior Answer]\n{content}")

    # ── KB chunks with sequential numbers for citations ─────────────────────
    kb_counter = 0
    for doc in docs:
        if doc.get("metadata", {}).get("_source_type") != "chat_history":
            kb_counter += 1
            content = doc.get("page_content", "").strip()
            meta = doc.get("metadata", {})
            source = meta.get("source") or meta.get("file_name", "")
            header = f"[KB-{kb_counter}]" + (f" ({source})" if source else "")
            parts.append(f"{header}\n{content}")

    if file_markdown and file_markdown.strip():
        parts.append(f"[FILE CONTENT]\n{file_markdown.strip()}")

    return "\n\n---\n\n".join(parts)
