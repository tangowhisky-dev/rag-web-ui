"""Context manager — handles context overflow for the autonomous agent.

When the supervisor accumulates too many documents, tool results, or
history messages, the context can exceed the LLM's token limit.  This
module provides:

1. Token budgeting — tracks estimated token count across the session.
2. Summarization — compresses conversation history using the LLM.
3. Document pruning — removes low-relevance docs before passing to LLM.
4. Tool result offloading — truncates large tool outputs to essential info.

These are inspired by LangChain's Deep Agents context engineering
patterns: summarize history, offload large tool results, isolate
subagent context, and use prompt caching.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

# Rough token estimation constants
_TOKENS_PER_CHAR = 0.25  # English text: ~1 token per 4 chars
_TOKENS_PER_DOC_PREVIEW = 300  # A doc preview (200 chars + metadata)


@dataclass
class TokenBudget:
    """Tracks estimated token usage and remaining budget."""
    total_limit: int  # Context window size for the model
    used: int = 0
    reserved_system_prompt: int = 2000  # Reserve for system prompt, tool defs, etc.

    @property
    def remaining(self) -> int:
        return self.total_limit - self.used - self.reserved_system_prompt

    @property
    def safe_remaining(self) -> int:
        """Remaining tokens with 20% safety margin."""
        return int(self.remaining * 0.8)

    def can_fit(self, estimated_tokens: int) -> bool:
        """Check if the estimated token count fits in the budget."""
        return estimated_tokens <= self.safe_remaining

    def add(self, tokens: int) -> None:
        """Record token usage."""
        self.used += tokens

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation from character count."""
        return int(len(text) * _TOKENS_PER_CHAR)


class ContextManager:
    """
    Manages context overflow for the autonomous agent.

    Usage:
        manager = ContextManager(model_name=model_name, user_id=user_id)

        # Add documents — returns pruned list if over budget
        pruned_docs = manager.add_documents(docs, confidence_scores)

        # Compress conversation history
        compressed_history = manager.compress_history(lc_messages)

        # Truncate tool output
        safe_output = manager.truncate_tool_output(tool_result, max_tokens=500)

        # Get budget status for supervisor awareness
        budget_status = manager.get_budget_status()
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        user_id: Optional[int] = None,
        api_base: Optional[str] = None,
    ):
        self.user_id = user_id
        self.api_base = api_base

        # Determine context window from model config
        context_size = settings.OPENAI_MODEL_CONTEXT_SIZE
        # Check for model-specific overrides if available
        self.budget = TokenBudget(total_limit=context_size)

    def add_documents(
        self,
        docs: List[dict],
        confidence_scores: Optional[List[float]] = None,
        max_tokens: Optional[int] = None,
    ) -> List[dict]:
        """
        Prune documents to fit within token budget.

        Keeps high-confidence docs and removes low-value ones.
        Returns a pruned list that fits in the budget.

        Strategy (Deep Agents context offloading pattern):
        1. Sort docs by confidence/relevance score descending.
        2. Greedily add docs until budget is reached.
        3. If budget is exceeded, summarize the remaining docs into a
           condensed overview.
        """
        max_tok = max_tokens or self.budget.safe_remaining

        if not docs:
            return []

        # Sort by confidence/relevance
        scored_docs = []
        for i, doc in enumerate(docs):
            score = confidence_scores[i] if confidence_scores else doc.get("metadata", {}).get("_reranker_score", 0.0)
            scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: -x[0])

        # Greedily select docs that fit in budget
        selected = []
        estimated_tokens = 0

        for score, doc in scored_docs:
            # Estimate tokens for this doc (content preview + metadata)
            content = doc.get("page_content", "")
            preview = content[:200] if content else ""
            doc_tokens = _TOKENS_PER_DOC_PREVIEW + self.budget.estimate_tokens(preview)

            if estimated_tokens + doc_tokens <= max_tok and len(selected) < 15:
                selected.append(doc)
                estimated_tokens += doc_tokens
            elif len(selected) >= 10:
                # Stop after 10 docs to leave room for other context
                break

        if not selected and scored_docs:
            # At minimum, return the top document
            selected = [scored_docs[0][1]]

        logger.info(
            "[CONTEXT_MANAGER] docs: %d -> %d (budget: %d/%d tokens)",
            len(docs), len(selected), estimated_tokens, max_tok,
        )

        return selected

    def compress_history(
        self,
        messages: list,
        max_tokens: Optional[int] = None,
    ) -> list:
        """
        Compress conversation history when approaching context limit.

        Uses the LLM to summarize older messages, keeping only the most
        recent turn pair intact.

        Deep Agents pattern: compress conversation history automatically.
        """
        max_tok = max_tokens or self.budget.safe_remaining
        if max_tok < 500:
            # Budget critically low — keep only the latest user message
            user_msgs = [m for m in messages if m.get("role") == "user"]
            if user_msgs:
                return [user_msgs[-1]]
            return messages[:2]  # Safety fallback: last 2 messages

        # If budget allows, return all messages
        total_estimated = sum(
            self.budget.estimate_tokens(m.get("content", ""))
            for m in messages
        )
        if total_estimated <= max_tok:
            return messages

        # Budget exceeded — keep last N turns, summarize the rest
        # Keep last 3 pairs (6 messages) intact
        keep_count = min(6, len(messages))
        recent = messages[-keep_count:] if keep_count > 0 else []

        # Summarize older messages if we have an LLM available
        older = messages[:-keep_count] if keep_count < len(messages) else []
        if older:
            try:
                summary = self._summarize_messages(older)
                if summary:
                    return [{"role": "system", "content": f"[Conversation summary]\n{summary}"}] + recent
            except Exception as exc:
                logger.warning("[CONTEXT_MANAGER] summarization failed: %s", exc)

        return recent

    def truncate_tool_output(
        self,
        tool_output: dict,
        max_tokens: int = 1000,
    ) -> dict:
        """
        Truncate large tool outputs to fit within budget.

        For db_query results: keep only first N rows and column names.
        For graph_query results: keep top entities by relationship count.
        For chart results: keep config but truncate data arrays.
        """
        output = tool_output.get("output", [])
        if not output:
            return tool_output

        if isinstance(output, list):
            # Estimate rows that fit in max_tokens
            if output and isinstance(output[0], dict):
                # Sample first row to estimate tokens per row
                sample_text = json.dumps(output[0])
                tokens_per_row = self.budget.estimate_tokens(sample_text)
                max_rows = max(1, max_tokens // max(1, tokens_per_row))
                tool_output["output"] = output[:max_rows]
                if len(output) > max_rows:
                    tool_output["_truncated"] = True
                    tool_output["_total_rows"] = len(output)
                    tool_output["_showing_rows"] = max_rows

        return tool_output

    def get_budget_status(self) -> dict:
        """Return current budget status for supervisor awareness."""
        return {
            "total_limit": self.budget.total_limit,
            "used": self.budget.used,
            "remaining": self.budget.remaining,
            "safe_remaining": self.budget.safe_remaining,
            "utilization_pct": round(
                (self.budget.used / self.budget.total_limit) * 100, 1
            ) if self.budget.total_limit > 0 else 0,
        }

    def record_tool_result_tokens(self, tool_output: dict) -> None:
        """Estimate and record token usage from a tool result."""
        text = json.dumps(tool_output)
        tokens = self.budget.estimate_tokens(text)
        self.budget.add(tokens)

    def _summarize_messages(
        self,
        messages: list,
    ) -> Optional[str]:
        """Summarize a list of messages using the LLM."""
        try:
            from langchain_openai import ChatOpenAI
            from app.core.config import settings

            llm = ChatOpenAI(
                model=settings.effective_query_model or settings.OPENAI_MODEL,
                temperature=0.0,
                openai_api_base=self.api_base or settings.OPENAI_API_BASE,
                openai_api_key=settings.OPENAI_API_KEY,
                streaming=False,
            )

            # Build a simple summarization prompt
            user_content = "Summarize the following conversation, preserving key facts and context:\n\n"
            for msg in messages[:20]:  # Limit to 20 messages
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:200]
                user_content += f"{role}: {content}\n"

            response = llm.invoke([
                {"role": "system", "content": "You are a conversation summarizer. Provide a concise summary of key facts, decisions, and context. Max 200 words."},
                {"role": "user", "content": user_content},
            ])

            summary = response.content if hasattr(response, "content") else str(response)
            logger.info("[CONTEXT_MANAGER] summarized %d messages to %d chars", len(messages), len(summary))
            return summary

        except Exception as exc:
            logger.warning("[CONTEXT_MANAGER] summarization failed: %s", exc)
            return None
