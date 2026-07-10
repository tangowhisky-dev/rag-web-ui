"""Answer quality evaluation for the agentic pipeline.

Evaluates generated answers against the retrieved context and original query.
Returns structured quality metrics that can be displayed in the UI.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, List, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

EVALUATION_SYSTEM_PROMPT = """\
You are an answer quality evaluator. Given a query, retrieved context, and generated answer,
assess the quality of the answer.

Rules:
- faithfulness (0-100): What percentage of the answer is actually supported by the retrieved context?
  - 100 = everything cited or clearly supported by context
  - 0 = answer is mostly or entirely external knowledge
- completeness (0-100): How thoroughly does the answer address the query?
  - 100 = all aspects of the query are fully addressed
  - 0 = answer misses key parts of the query
- citation_quality (0-100): Are citations properly used and relevant?
  - 100 = all citations are accurate and relevant
  - 0 = no citations or fabricated citations
- confidence_match (boolean): Does the confidence level match the answer quality?
  - true = high quality answer with high confidence, or low quality with low confidence
  - false = mismatch between answer quality and confidence

Output ONLY a JSON object with these keys:
{
  "faithfulness": <0-100>,
  "completeness": <0-100>,
  "citation_quality": <0-100>,
  "confidence_match": true/false,
  "flags": [<list of issue descriptions, empty if no issues>]
}
"""


@dataclass
class AnswerEvaluation:
    """Result of answer quality evaluation."""
    faithfulness: int  # 0-100
    completeness: int  # 0-100
    citation_quality: int  # 0-100
    confidence_match: bool
    flags: List[str]
    raw_response: str = ""


def _parse_evaluation_response(raw: str) -> AnswerEvaluation:
    """Parse the LLM evaluation response into an AnswerEvaluation object."""
    try:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if not m:
            return _default_evaluation(f"Failed to parse evaluation: {raw[:100]}")

        data = json.loads(m.group(0))
        return AnswerEvaluation(
            faithfulness=int(data.get("faithfulness", 50)),
            completeness=int(data.get("completeness", 50)),
            citation_quality=int(data.get("citation_quality", 50)),
            confidence_match=bool(data.get("confidence_match", True)),
            flags=data.get("flags", []) or [],
            raw_response=raw,
        )
    except Exception as exc:
        logger.warning("[EVAL] parse failed: %s", exc)
        return _default_evaluation(f"Parse error: {exc}")


def _default_evaluation(error: str = "") -> AnswerEvaluation:
    """Return a default evaluation when evaluation fails."""
    return AnswerEvaluation(
        faithfulness=50,
        completeness=50,
        citation_quality=50,
        confidence_match=True,
        flags=[f"Evaluation unavailable: {error}"] if error else ["Evaluation skipped"],
        raw_response="",
    )


async def evaluate_answer(
    query: str,
    answer: str,
    context_preview: str,
    confidence_level: str = "medium",
) -> AnswerEvaluation:
    """Evaluate answer quality using an LLM call.

    Args:
        query: The original user query.
        answer: The generated answer text.
        context_preview: First 2000 chars of retrieved context (for faithfulness check).
        confidence_level: Retrieval confidence level (very_high/high/medium/low/none).

    Returns:
        AnswerEvaluation with quality metrics.
    """
    # Truncate context to keep the evaluation prompt manageable
    truncated_context = context_preview[:2000]

    user_prompt = f"""Query: {query}

Retrieved Context (excerpt):
{truncated_context}

Generated Answer:
{answer}

Evaluate the quality of this answer based on the retrieved context.
"""

    try:
        from openai import AsyncOpenAI as _OAI
        client = _OAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE,
        )

        resp = await client.chat.completions.create(
            model=settings.effective_query_model or settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=200,
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
    parts.append(f"Citation quality: {evaluation.citation_quality}/100")

    if evaluation.confidence_match:
        parts.append("Confidence matches quality: Yes")
    else:
        parts.append("Confidence matches quality: No (mismatch detected)")

    if evaluation.flags:
        parts.append("Issues: " + "; ".join(evaluation.flags))

    return " | ".join(parts)
