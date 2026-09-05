"""Answer quality evaluation and structured extraction for the agentic pipeline.

Evaluates generated answers against the retrieved context and original query.
Also extracts summary, key points, numerical data, and follow-up questions
in a single LLM call.
"""

from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

from app.services.agentic_rag.prompts import EVALUATION_AND_FOLLOWUP_PROMPT
from app.services.agentic_rag.schemas import DataPoint

logger = logging.getLogger(__name__)


@dataclass
class AnswerEvaluation:
    """Result of the combined evaluation + extraction LLM call."""
    faithfulness: int  # 0-100
    completeness: int  # 0-100
    flags: List[str]
    summary: str = ""
    key_points: List[str] = field(default_factory=list)
    data: List[dict] = field(default_factory=list)  # List of {label, value, unit, context}
    followups: List[str] = field(default_factory=list)
    retry_strategy: str = ""
    raw_response: str = ""


def _parse_evaluation_response(raw: str) -> AnswerEvaluation:
    """Parse the LLM evaluation response into an AnswerEvaluation object.

    Handles common LLM output issues: markdown fences, trailing text,
    truncation at max_tokens, and malformed JSON.
    """
    try:
        text = raw.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.splitlines()
            lines = lines[1:] if lines else lines
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # Extract JSON object from response
        start = text.find("{")
        if start < 0:
            raise ValueError("No JSON object found")

        # Try the full {…} block first, then progressively shorter suffixes
        end = text.rfind("}")
        if end > start:
            text = text[start : end + 1]
        else:
            raise ValueError("No closing brace found")

        data = _try_json_loads(text)
        return AnswerEvaluation(
            faithfulness=int(data.get("faithfulness", 50)),
            completeness=int(data.get("completeness", 50)),
            flags=data.get("flags", []) or [],
            summary=data.get("summary", ""),
            key_points=data.get("key_points", []) or [],
            data=data.get("data", []) or [],
            followups=data.get("followups", []) or [],
            retry_strategy=data.get("retry_strategy", ""),
            raw_response=raw,
        )
    except Exception as exc:
        logger.warning("[EVAL] parse failed: %s | response=%r", exc, raw[:500])
        return _default_evaluation(f"Parse error: {exc}")


def _try_json_loads(text: str) -> dict:
    """Attempt to parse JSON, with progressive repair for truncated/malformed output.

    LLM responses often fail initial parsing due to:
    - Truncation at max_tokens boundary
    - Trailing text after the JSON object
    - Unescaped special characters

    Strategy: try the full text first, then progressively strip content
    from the end until parsing succeeds.
    """
    # Fast path: try direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 1: Walk from the end, trimming trailing content.
    # Use coarser steps first, then finer steps near the closing brace.
    max_len = len(text)
    for cut in range(max_len - 1, 0, -2):
        candidate = text[:cut].rstrip()
        if not candidate.endswith("}"):
            continue
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 2: Find the innermost {…} block (heavily truncated case)
    inner_start = text.rfind("{")
    inner_end = text.rfind("}")
    if inner_end > inner_start:
        candidate = text[inner_start : inner_end + 1]
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: Strip to the outermost {…} block
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            pass

    # Give up — let the caller handle the exception
    raise ValueError(f"Could not parse JSON from {len(text)}-char response")


def _default_evaluation(error: str = "") -> AnswerEvaluation:
    """Return a default evaluation when evaluation fails."""
    return AnswerEvaluation(
        faithfulness=50,
        completeness=50,
        flags=[f"Evaluation unavailable: {error}"] if error else ["Evaluation skipped"],
        raw_response="",
    )


async def evaluate_answer(
    query: str,
    answer: str,
    context_preview: str,
    confidence_level: str = "medium",
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    query_model: Optional[str] = None,
) -> AnswerEvaluation:
    """Evaluate answer quality and extract structured data in one LLM call.

    Combines faithfulness/completeness scoring with summary, key_points,
    data extraction, followup generation, and retry_strategy — all in a
    single LLM call to save latency.

    Args:
        query: The original user query.
        answer: The generated answer text (full, not truncated).
        context_preview: Retrieved cited context (for faithfulness check).
        confidence_level: Retrieval confidence level (very_high/high/medium/low/none).
        api_base: Optional OpenAI-compatible base URL override.
        api_key: Optional API key override.
        query_model: Optional model name override.

    Returns:
        AnswerEvaluation with quality metrics and extracted fields.
    """
    user_prompt = f"""Query: {query}

Retrieved Context:
{context_preview}

Generated Answer:
{answer}

Evaluate the quality of this answer and extract structured data.
"""

    try:
        from openai import AsyncOpenAI as _OAI
        # Fall back to app-level settings when no explicit overrides provided
        if api_key is None or api_base is None or query_model is None:
            from app.services.settings_service import get_setting
            from app.db.session import SessionLocal
            _db = SessionLocal()
            try:
                if api_key is None:
                    api_key = get_setting(_db, "QUERY_API_KEY", None) or get_setting(_db, "OPENAI_API_KEY", None)
                if api_base is None:
                    api_base = get_setting(_db, "QUERY_API_BASE", None) or get_setting(_db, "OPENAI_API_BASE", None)
                if query_model is None:
                    query_model = get_setting(_db, "QUERY_MODEL", None) or get_setting(_db, "OPENAI_MODEL", None)
            finally:
                _db.close()
        client = _OAI(
            api_key=api_key,
            base_url=api_base,
        )

        resp = await client.chat.completions.create(
            model=query_model,
            messages=[
                {"role": "system", "content": EVALUATION_AND_FOLLOWUP_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=2000,
            temperature=0,
            stream=False,
            extra_body={"thinking": {"type": "disabled"}},
        )

        raw = (resp.choices[0].message.content or "").strip()
        return _parse_evaluation_response(raw)

    except Exception as exc:
        logger.warning("[EVAL] evaluation failed: %s", exc)
        return _default_evaluation(str(exc))


def summarize_evaluation(evaluation: AnswerEvaluation) -> str:
    """Return a human-readable summary of evaluation results."""
    parts = []
    parts.append(f"Faithfulness: {evaluation.faithfulness}/100")
    parts.append(f"Completeness: {evaluation.completeness}/100")

    if evaluation.flags:
        parts.append("Issues: " + "; ".join(evaluation.flags))

    return " | ".join(parts)
