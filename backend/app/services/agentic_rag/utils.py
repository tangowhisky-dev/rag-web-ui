"""Helper functions for the LangGraph agent graph."""

from __future__ import annotations

import json
import re
from typing import Any, List


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
    """
    parts: list[str] = []
    for i, doc in enumerate(docs, 1):
        content = doc.get("page_content", "").strip()
        source = doc.get("metadata", {}).get("source", "")
        header = f"[KB-{i}]" + (f" ({source})" if source else "")
        parts.append(f"{header}\n{content}")
    if file_markdown:
        parts.append(f"[File Content]\n{file_markdown}")
    return "\n\n---\n\n".join(parts)


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

    system_msg = (
        "You are a search query rewriter for a document retrieval system. "
        "Your ONLY job is to rewrite the user's latest message into a self-contained search query "
        "that can be sent to a vector database. "
        "Use the recent chat history and any relevant past context solely to resolve pronouns and references — "
        "never to answer, evaluate, or judge the question.\n\n"
        "Rules:\n"
        "1. Output a standalone question or keyword phrase — nothing else.\n"
        "2. Resolve pronouns and references from history or past context "
        "(e.g. 'it' → the specific topic discussed).\n"
        "3. Do NOT answer the question. Do NOT say whether information exists or not.\n"
        "4. Do NOT add information not needed to resolve an ambiguous reference.\n"
        "5. DO NOT infer relationships between topics. If the user asks a standalone question, "
        "keep it standalone — even if a previous turn discussed something different.\n"
        "6. Do NOT introduce new entities, concepts, or relationships that the user did not mention.\n"
        "7. Keep the output short — one sentence or a keyword phrase, maximum 30 words.\n"
        f"{memory_section}\n\n"
        "CRITICAL: If the user's query is already self-contained, return it almost unchanged. "
        "Only modify when there is an ambiguous pronoun or reference that requires resolution.\n\n"
        "Examples:\n"
        "History: [user: tell me about Linux, assistant: Linux is an open-source OS...]\n"
        "Query: 'any other worthwhile OS you like to mention?'\n"
        "Output: 'other notable operating systems worth mentioning'\n\n"
        "History: [user: summarise assignment 1, assistant: ...summary...]\n"
        "Query: 'what is question 1'\n"
        "Output: 'What is Question 1 in Assignment 1?'\n\n"
        "History: [user: tell me about the StreamVC paper]\n"
        "Query: 'what model does it use'\n"
        "Output: 'What model architecture does StreamVC use?'\n\n"
        "History: [user: explain Process Control Block, assistant: ...PCB explanation...]\n"
        "Query: 'Explain mutex'\n"
        "Output: 'What is a mutex?'\n\n"
        "History: [user: explain mutex, assistant: ...mutex explanation...]\n"
        "Query: 'How does a semaphore differ?'\n"
        "Output: 'How does a semaphore differ from a mutex?'"
    )

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
