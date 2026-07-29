"""Helper functions for the LangGraph agent graph."""

from __future__ import annotations

import json
import re
from typing import Any, List

from app.services.agentic_rag.prompts import REWRITE_SYSTEM_PROMPT


def estimate_context_tokens(text: str) -> int:
    """Rough token estimation from character count.
    
    Uses ~1 token per 4 chars for English text.
    """
    return int(len(text) * 0.25)


def estimate_messages_tokens(messages: list) -> int:
    """Estimate total tokens for a list of messages."""
    total = 0
    for msg in messages:
        content = getattr(msg, "content", str(msg)) if hasattr(msg, "content") else msg.get("content", "")
        total += estimate_context_tokens(str(content))
    return total


def strip_reasoning_tags(text: str) -> str:
    """Strip <think>...</think> tags from text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def format_context_string(docs: list[dict], file_markdown: str | None = None) -> str:
    """Format a list of serialized documents into a context string for the LLM.

    Each doc becomes ``[KB-N] (source)\\ncontent``.  If *file_markdown* is
    provided it is appended after a ``[File Content]`` header.

    Contiguous chunks from the same document have their overlap pruned
    so the LLM doesn't see duplicated text (300 chars per adjacent pair
    at 20% overlap). Citation indices are unaffected — pruning only
    shortens the content, not the chunk's position in the list.
    """
    from app.services.agentic_rag.agent_graph import _prune_contiguous_overlaps

    pruned_docs = _prune_contiguous_overlaps(docs) if docs else docs
    parts: list[str] = []
    for i, doc in enumerate(pruned_docs, 1):
        content = doc.get("page_content", "").strip()
        source = doc.get("metadata", {}).get("source", "")
        header = f"[KB-{i}]" + (f" ({source})" if source else "")
        parts.append(f"{header}\n{content}")
    if file_markdown:
        parts.append(f"[File Content]\n{file_markdown}")
    return "\n\n---\n\n".join(parts)


def normalize_citations(answer: str, docs: list) -> tuple[str, list[int]]:
    """Validate, deduplicate, and renumber inline citations in an LLM answer.

    - Parses [N](N) markdown citation links.
    - Normalizes [citation](N) and [citation](N)(N) variants to [N](N).
    - Removes any citation whose index is outside the provided docs range.
    - Renumbers remaining citations 1..M by first appearance in the answer.
    - Returns the rewritten answer and the list of original 1-based doc indices
      in display order.
    """
    if not answer:
        return answer or "", []

    # Strip any existing citation markers when no docs are available.
    if not docs:
        cleaned = re.sub(r"\[citation\]\(\d+\)\(\d+\)", "", answer)
        cleaned = re.sub(r"\[citation\]\(\d+\)", "", cleaned)
        cleaned = re.sub(r"\[\d+\]\(\d+\)", "", cleaned)
        return cleaned.strip(), []

    # Normalize common malformed variants emitted by some models to [N](N).
    answer = re.sub(r"\[citation\]\((\d+)\)\((\d+)\)", r"[\1](\1)", answer)
    answer = re.sub(r"\[citation\]\((\d+)\)", r"[\1](\1)", answer)

    max_index = len(docs)
    valid_cited: list[int] = []
    seen: set[int] = set()

    # Collect unique valid original indices in first-appearance order.
    for match in re.finditer(r"\[(\d+)\]\((\d+)\)", answer):
        n = int(match.group(1))
        # Guard against mismatched brackets like [1](2) — require both numbers equal.
        if n != int(match.group(2)):
            continue
        if 1 <= n <= max_index and n not in seen:
            valid_cited.append(n)
            seen.add(n)

    index_map = {orig: new for new, orig in enumerate(valid_cited, start=1)}

    def _replace_marker(match: re.Match) -> str:
        n = int(match.group(1))
        m = int(match.group(2))
        # Only rewrite well-formed [N](N); strip malformed or out-of-range markers.
        if n == m and n in index_map:
            new_idx = index_map[n]
            return f"[{new_idx}]({new_idx})"
        return ""

    normalized = re.sub(r"\[(\d+)\]\((\d+)\)", _replace_marker, answer)
    return normalized, valid_cited


def rewrite_query(
    query: str,
    recent_history: list,
    memory_context: str = "",
    api_base: str | None = None,
    query_model: str | None = None,
    openai_api_key: str = "",
    openai_api_base: str = "",
) -> str:
    """Rewrite *query* into a self-contained search query using chat history.

    Uses the recent conversation turns plus any relevant long-term memory
    context (from the Redis store) to resolve pronouns and references.
    Returns the original query on failure or when the model echoes an answer
    instead of rewriting.
    """
    if not recent_history and not memory_context:
        return query

    memory_section = ""
    if memory_context:
        memory_section = (
            "\n\nRelevant past context (older conversation turns):\n"
            f"{memory_context}\n\n"
            "Use this past context to resolve references that go beyond the recent messages."
        )

    system_msg = REWRITE_SYSTEM_PROMPT.format(memory_section=memory_section)

    messages: list[dict] = [{"role": "system", "content": system_msg}]
    from langchain_core.messages import HumanMessage, AIMessage

    for m in recent_history:
        if isinstance(m, HumanMessage):
            messages.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            messages.append({"role": "assistant", "content": m.content[:400]})
    messages.append({"role": "user", "content": query})

    from openai import OpenAI as _OAI
    client = _OAI(api_key=openai_api_key, base_url=api_base or openai_api_base)
    resp = client.chat.completions.create(
        model=query_model or "default",
        messages=messages,
        max_tokens=60,
        temperature=0,
        stream=False,
        extra_body={"thinking": {"type": "disabled"}},
    )
    raw_rewrite = (resp.choices[0].message.content or "").strip()
    standalone = strip_reasoning_tags(raw_rewrite) or query

    # Strip meta-commentary preamble that some models emit
    if re.search(
        r"\buser\b.*\rewrite\b|\brewritten\b|\bstandalone\b",
        standalone, re.IGNORECASE,
    ):
        if ":" in standalone:
            candidate = standalone.rsplit(":", 1)[-1].strip()
        else:
            sentences = re.split(r"(?<=[.?!])\s+", standalone)
            candidate = sentences[-1].strip()
        if len(candidate) > 5:
            standalone = candidate

    # Guard: if the rewriter echoed the previous assistant response
    answer_patterns = [
        r"\bthere\s+is\s+no\s+information\b",
        r"\bthe\s+context\s+does?\s+not\s+contain\b",
        r"\bi\s+cannot\s+answer\b",
        r"\bi\s+don't\s+have\s+enough\b",
        r"\bno\s+information\s+found\b",
    ]
    if any(re.search(p, standalone, re.IGNORECASE) for p in answer_patterns):
        standalone = query

    return standalone
