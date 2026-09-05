"""Answer quality evaluation and structured extraction for the agentic pipeline.

Two separate LLM calls:
1. evaluate_answer — faithfulness/completeness scoring + followup generation
   (user-facing, runs in the graph before the done event)
2. extract_structured — summary/key_points/data extraction
   (housekeeping for next-turn tool use, runs as a background task after
   the done event so the user is not blocked)
"""

from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

from app.services.agentic_rag.prompts import EVALUATION_PROMPT, EXTRACTION_PROMPT
from app.services.agentic_rag.schemas import DataPoint

logger = logging.getLogger(__name__)


@dataclass
class AnswerEvaluation:
    """Result of the evaluation LLM call (grading + followups)."""
    faithfulness: int  # 0-100
    completeness: int  # 0-100
    followups: List[str] = field(default_factory=list)
    raw_response: str = ""


@dataclass
class StructuredExtraction:
    """Result of the background extraction LLM call (summary/key_points/data)."""
    summary: str = ""
    key_points: List[str] = field(default_factory=list)
    data: List[dict] = field(default_factory=list)
    raw_response: str = ""


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

    raise ValueError(f"Could not parse JSON from {len(text)}-char response")


def _strip_code_fences(raw: str) -> str:
    """Strip markdown code fences and extract the JSON object from raw text."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:] if lines else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found")
    end = text.rfind("}")
    if end > start:
        text = text[start : end + 1]
    else:
        raise ValueError("No closing brace found")
    return text


def _parse_evaluation_response(raw: str) -> AnswerEvaluation:
    """Parse the evaluation LLM response into an AnswerEvaluation object."""
    try:
        text = _strip_code_fences(raw)
        data = _try_json_loads(text)
        return AnswerEvaluation(
            faithfulness=int(data.get("faithfulness", 50)),
            completeness=int(data.get("completeness", 50)),
            followups=data.get("followups", []) or [],
            raw_response=raw,
        )
    except Exception as exc:
        logger.warning("[EVAL] parse failed: %s | response=%r", exc, raw[:500])
        return _default_evaluation(f"Parse error: {exc}")


def _parse_extraction_response(raw: str) -> StructuredExtraction:
    """Parse the extraction LLM response into a StructuredExtraction object."""
    try:
        text = _strip_code_fences(raw)
        data = _try_json_loads(text)
        return StructuredExtraction(
            summary=data.get("summary", ""),
            key_points=data.get("key_points", []) or [],
            data=data.get("data", []) or [],
            raw_response=raw,
        )
    except Exception as exc:
        logger.warning("[EXTRACT] parse failed: %s | response=%r", exc, raw[:500])
        return StructuredExtraction()


def _default_evaluation(error: str = "") -> AnswerEvaluation:
    """Return a default evaluation when evaluation fails."""
    return AnswerEvaluation(
        faithfulness=50,
        completeness=50,
        raw_response="",
    )


def _resolve_llm_kwargs(api_base: Optional[str], api_key: Optional[str],
                        query_model: Optional[str]) -> tuple[str, str, str]:
    """Resolve LLM kwargs, falling back to app-level settings."""
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
    return api_base, api_key, query_model


async def evaluate_answer(
    query: str,
    answer: str,
    context_preview: str,
    confidence_level: str = "medium",
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    query_model: Optional[str] = None,
) -> AnswerEvaluation:
    """Evaluate answer quality and generate follow-ups in one LLM call.

    Args:
        query: The original user query.
        answer: The generated answer text (full, not truncated).
        context_preview: Retrieved cited context (for faithfulness check).
        confidence_level: Retrieval confidence level (very_high/high/medium/low/none).
        api_base: Optional OpenAI-compatible base URL override.
        api_key: Optional API key override.
        query_model: Optional model name override.

    Returns:
        AnswerEvaluation with faithfulness, completeness, and followups.
    """
    user_prompt = f"""Query: {query}

Retrieved Context:
{context_preview}

Generated Answer:
{answer}

Evaluate the quality of this answer and generate follow-up questions.
"""

    try:
        from openai import AsyncOpenAI as _OAI
        api_base, api_key, query_model = _resolve_llm_kwargs(api_base, api_key, query_model)
        client = _OAI(api_key=api_key, base_url=api_base)

        resp = await client.chat.completions.create(
            model=query_model,
            messages=[
                {"role": "system", "content": EVALUATION_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1000,
            temperature=0,
            stream=False,
            extra_body={"thinking": {"type": "disabled"}},
        )

        raw = (resp.choices[0].message.content or "").strip()
        return _parse_evaluation_response(raw)

    except Exception as exc:
        logger.warning("[EVAL] evaluation failed: %s", exc)
        return _default_evaluation(str(exc))


async def extract_structured(
    answer: str,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    query_model: Optional[str] = None,
) -> StructuredExtraction:
    """Extract summary, key_points, and data from the answer text.

    This is a background housekeeping call — it runs after the done event
    so the user is not blocked. Failures are silent (return empty defaults).

    Args:
        answer: The generated answer text (full, not truncated).
        api_base: Optional OpenAI-compatible base URL override.
        api_key: Optional API key override.
        query_model: Optional model name override.

    Returns:
        StructuredExtraction with summary, key_points, and data.
    """
    try:
        from openai import AsyncOpenAI as _OAI
        api_base, api_key, query_model = _resolve_llm_kwargs(api_base, api_key, query_model)
        client = _OAI(api_key=api_key, base_url=api_base)

        resp = await client.chat.completions.create(
            model=query_model,
            messages=[
                {"role": "user", "content": EXTRACTION_PROMPT.format(answer=answer)},
            ],
            max_tokens=2000,
            temperature=0,
            stream=False,
            extra_body={"thinking": {"type": "disabled"}},
        )

        raw = (resp.choices[0].message.content or "").strip()
        return _parse_extraction_response(raw)

    except Exception as exc:
        logger.warning("[EXTRACT] extraction failed: %s", exc)
        return StructuredExtraction()


def summarize_evaluation(evaluation: AnswerEvaluation) -> str:
    """Return a human-readable summary of evaluation results."""
    parts = []
    parts.append(f"Faithfulness: {evaluation.faithfulness}/100")
    parts.append(f"Completeness: {evaluation.completeness}/100")

    return " | ".join(parts)
