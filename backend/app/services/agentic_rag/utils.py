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


def extract_chat_id(state: dict, chat_id: int) -> int:
    """Extract chat_id from state or fall back to the parameter."""
    if isinstance(state, dict):
        return state.get("chat_id", chat_id)
    return chat_id


def strip_reasoning_tags(text: str) -> str:
    """Strip <think>...</think> tags from text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_thinking_content(text: str) -> tuple[str, str]:
    """Extract thinking content and response from text with reasoning tags.
    
    Returns (thinking_content, response_text).
    """
    match = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "", text.strip()


def truncate_to_words(text: str, max_words: int) -> str:
    """Truncate text to at most max_words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def safe_json_parse(text: str) -> dict | list | None:
    """Safely parse JSON from text, extracting from markdown code blocks if present."""
    try:
        # Try direct parse first
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in markdown code blocks
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None

    # Try to extract first JSON object/array
    match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    return None


def format_context_string(docs: list[dict]) -> str:
    """Format a list of serialized documents into a context string for the LLM."""
    parts = []
    for i, doc in enumerate(docs, 1):
        content = doc.get("page_content", "").strip()
        source = doc.get("metadata", {}).get("source", "")
        header = f"[KB-{i}]" + (f" ({source})" if source else "")
        parts.append(f"{header}\n{content}")
    return "\n\n---\n\n".join(parts)


def build_task_list_events(
    subtasks: list[str],
    completed_idx: int,
    total: int,
) -> list[dict]:
    """Build task_list events for the current subtask index.
    
    Args:
        subtasks: List of subtask strings.
        completed_idx: Index of the current subtask being executed.
        total: Total number of subtasks.
        
    Returns:
        List of task_list event dicts.
    """
    return [{
        "event": "task_list",
        "tasks": [
            {
                "id": i,
                "text": s,
                "status": "done" if i < completed_idx else ("running" if i == completed_idx else "pending"),
                "progress": None,
            }
            for i, s in enumerate(subtasks)
        ],
    }]
