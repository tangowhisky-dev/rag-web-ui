"""LangGraph node implementations for the agentic RAG pipeline."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, AsyncGenerator, List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.services.infrastructure import strip_reasoning_tags
from app.services.infrastructure.utils import _serialise_doc
from app.services.retrieval import score_retrieval
from app.services.retrieval import hybrid_search_with_legs, get_effective_datastore_ids
from app.services.prompts.loader import append_chart_instructions

from .graph_state import AgentState
from .schemas import QueryAnalysis
from .utils import estimate_messages_tokens

logger = logging.getLogger(__name__)

_ANSWER_SYSTEM_PROMPT = append_chart_instructions("""\
You are a helpful assistant. Answer the user's question using ONLY the provided context.
If the context is insufficient, say so clearly.

FORMATTING RULES:
- Use ### headers to divide multi-part answers (e.g., "### 1. Definition", "### 2. How It Works").
- Use numbered lists for sequential steps or algorithms.
- Use bullet points with **bold terms** for features, attributes, or comparisons.
- Use inline code for variable names, identifiers, and technical terms (e.g., `wait()`, `Available[j]`).
- For simple single-concept questions, plain prose is fine - do not force structure.

CITATION RULES:
When you use information from a chunk, cite it as a markdown link with ONLY the number as both text and href:
  Example: process scheduling [1](1) involves saving the CPU state [2](2).
The number must match the [KB-N] label of the chunk you are citing.
Do NOT invent citations. Only cite chunks you actually used.

IMPORTANT: Do NOT repeat the user's question in your answer. Just provide the answer directly.
""")

_REWRITE_SYSTEM = """\
You are a search query rewriter for a document retrieval system.
Your ONLY job is to rewrite the user's latest message into a self-contained search query
that can be sent to a vector database.
Use the chat history solely to resolve pronouns and references -
never to answer, evaluate, or judge the question.

Rules:
1. Output a standalone question or keyword phrase - nothing else.
2. Resolve pronouns and references from history.
3. Do NOT answer the question.
4. Keep the output short - one sentence or a keyword phrase, maximum 30 words.

Examples:
History: [user: tell me about Linux, assistant: Linux is an open-source OS...]
Query: 'any other worthwhile OS you like to mention?'
Output: 'other notable operating systems worth mentioning'

History: [user: tell me about the StreamVC paper]
Query: 'what model does it use'
Output: 'What model architecture does StreamVC use?'"""

_THINKING_KEYWORDS = [
    "compare", "contrast", "analyze", "evaluate", "design",
    "reason", "deduce", "infer", "explain why", "explain how",
    "tradeoff", "pros and cons", "architect", "implement",
    "discuss", "argue", "assess", "critique", "weigh",
    "implications", "limitations", "strengths", "weaknesses",
]


# ---------------------------------------------------------------------------
# Utility: model selection
# ---------------------------------------------------------------------------

def _select_model(subtask_text: str, is_complex: bool) -> str:
    """Auto-select model based on query nature."""
    lower = subtask_text.lower()
    is_thinking = any(kw in lower for kw in _THINKING_KEYWORDS)
    if is_thinking or is_complex:
        return settings.REASONING_MODEL or settings.OPENAI_MODEL
    return settings.OPENAI_MODEL


def _get_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    api_base: Optional[str] = None,
    streaming: bool = False,
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name or settings.OPENAI_MODEL,
        temperature=temperature,
        openai_api_base=api_base or settings.OPENAI_API_BASE,
        openai_api_key=settings.OPENAI_API_KEY,
        streaming=streaming,
    )


# ---------------------------------------------------------------------------
# Node: rewrite_query
# ---------------------------------------------------------------------------

async def rewrite_query_node(
    state: AgentState,
    api_base: Optional[str] = None,
) -> dict:
    """Rewrite query using chat history."""
    messages = state.get("messages", [])
    recent_history = []
    for m in messages:
        if isinstance(m, HumanMessage):
            recent_history.append(m)
        elif isinstance(m, AIMessage):
            recent_history.append(m)

    query = state.get("original_query", "")
    if not recent_history:
        return {"rewritten_query": query}

    rewrite_messages = [{"role": "system", "content": _REWRITE_SYSTEM}]
    for m in recent_history:
        if isinstance(m, HumanMessage):
            rewrite_messages.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            rewrite_messages.append({"role": "assistant", "content": m.content[:400]})
    rewrite_messages.append({"role": "user", "content": query})

    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=api_base or settings.OPENAI_API_BASE,
    )
    resp = await client.chat.completions.create(
        model=settings.effective_query_model,
        messages=rewrite_messages,
        max_tokens=60,
        temperature=0,
        stream=False,
        extra_body={"thinking": {"type": "disabled"}},
    )
    raw = (resp.choices[0].message.content or "").strip()
    rewritten = strip_reasoning_tags(raw) or query

    answer_patterns = [
        r"\bthere\s+is\s+no\s+information\b",
        r"\bthe\s+context\s+does?\s+not\s+contain\b",
        r"\bi\s+cannot\s+answer\b",
        r"\bi\s+don't\s+have\s+enough\b",
        r"\bno\s+information\s+found\b",
    ]
    if any(re.search(p, rewritten, re.IGNORECASE) for p in answer_patterns):
        rewritten = query

    return {"rewritten_query": rewritten}


# ---------------------------------------------------------------------------
# Node: classify_query (LLM-based classification)
# ---------------------------------------------------------------------------

async def classify_query_node(
    state: AgentState,
) -> dict:
    """Classify query using structured LLM output."""
    rewritten = state.get("rewritten_query", "")
    query = state.get("original_query", "")

    try:
        llm = _get_llm(streaming=False)
        llm_structured = llm.with_structured_output(QueryAnalysis)

        response = llm_structured.invoke([
            {"role": "system", "content": (
                "You are a query classifier. Analyze the user's question and respond with structured data.\n\n"
                "Rules:\n"
                "- is_clear: true if the question is clear and answerable from documents.\n"
                "- questions: list of self-contained questions extracted from the query (1 if simple, 2-5 if complex).\n"
                "- clarification_needed: explanation of missing info, or empty string if clear.\n"
                "Output ONLY a JSON object with keys: is_clear, questions, clarification_needed."
            )},
            {"role": "user", "content": rewritten},
        ])
        is_clear = getattr(response, "is_clear", True)
        questions = getattr(response, "questions", [rewritten]) or [rewritten]
    except Exception as exc:
        logger.warning("[CLASSIFY] structured classification failed: %s - using heuristic", exc)
        is_clear = True
        questions = _heuristic_classify(query, rewritten)

    subtasks = questions if len(questions) > 1 else [rewritten]

    return {
        "question_is_clear": is_clear,
        "subtasks": subtasks,
        "is_complex": len(subtasks) > 1,
    }


def _heuristic_classify(query: str, rewritten: str) -> List[str]:
    """Fallback heuristic classification for when structured output fails."""
    combined = (query + " " + rewritten).lower()
    # Multi-part indicators
    multi = re.search(
        r"\b(and|or|but|yet|also|plus|along with|as well as|in addition)\b.*"
        r"\b(and|or|but|yet|also|plus|along with|as well as|in addition)\b",
        combined,
    )
    if multi:
        return [rewritten]

    questions = re.findall(
        r"\b(what|how|why|when|where|which|compare|list|explain)\b", combined
    )
    if len(questions) >= 3:
        return [rewritten]

    if len(rewritten.split()) > 30:
        return [rewritten]

    return [rewritten]


# ---------------------------------------------------------------------------
# Node: request_clarification
# ---------------------------------------------------------------------------

def request_clarification_node(state: AgentState) -> dict:
    """Ask the user for clarification when the query is unclear."""
    pending = state.get("pending_query", "")
    clarifications = state.get("clarification_questions", [])

    if clarifications:
        context = "\n".join(f"{i+1}. {c}" for i, c in enumerate(clarifications))
        clarification_msg = (
            f"I need more information to answer your question about: '{pending}'\n\n"
            f"Please clarify:\n{context}"
        )
    else:
        clarification_msg = f"I need more information to understand your question: '{pending}'"

    return {
        "messages": [AIMessage(content=clarification_msg, name="clarification")],
    }


# ---------------------------------------------------------------------------
# Node: direct retrieval (simple path)
# ---------------------------------------------------------------------------

async def direct_retrieval_node(
    state: AgentState,
    db: Any,
    kb_ids: List[int] | None = None,
    org_id: int | None = None,
    file_markdown: str | None = None,
    use_dense: bool = True,
    use_sparse: bool = True,
    use_exact: bool = True,
    use_graph_rag: bool = False,
) -> dict:
    """Simple path: search + rerank. Returns state update with retrieved docs."""
    kb_ids = kb_ids or state.get("kb_ids", [])
    org_id = org_id if org_id is not None else state.get("org_id")
    file_markdown = file_markdown or state.get("file_markdown")

    rewritten = state.get("rewritten_query", state.get("original_query", ""))

    datastore_ids = get_effective_datastore_ids(kb_ids, org_id, db)

    retrieval_result = await hybrid_search_with_legs(
        query=rewritten,
        kb_ids=kb_ids,
        db=db,
        use_dense=use_dense,
        use_sparse=use_sparse,
        use_exact=use_exact,
        use_graph_rag=use_graph_rag,
        datastore_ids=datastore_ids,
        return_full_pool=True,
    )

    docs = retrieval_result.get("docs", [])
    retrieval_info = retrieval_result.get("retrieval_info", {})
    failed_legs = retrieval_info.get("failed_legs", [])

    conf_result = score_retrieval(docs, retrieval_info) if docs else None
    conf_score = conf_result.score if conf_result else 0

    serialised = [_serialise_doc(d) for d in docs]
    context_text = ""
    if serialised:
        parts = []
        for i, doc in enumerate(serialised, 1):
            content = doc.get("page_content", "").strip()
            source = doc.get("metadata", {}).get("source", "")
            header = f"[KB-{i}]" + (f" ({source})" if source else "")
            parts.append(f"{header}\n{content}")
        if file_markdown:
            parts.append(f"[File Content]\n{file_markdown}")
        context_text = "\n\n---\n\n".join(parts)

    return {
        "retrieved_docs": serialised,
        "retrieved_contexts": [context_text],
        "retrieval_confidence": conf_score / 100.0 if conf_score else 0.0,
        "retrieval_iterations": 1,
    }


# ---------------------------------------------------------------------------
# Node: generate (stream answer)
# ---------------------------------------------------------------------------

async def generate_node(
    state: AgentState,
    llm: ChatOpenAI | None = None,
    api_base: Optional[str] = None,
) -> dict:
    """Generate an answer from context. Returns the full answer text."""
    from app.services.infrastructure import strip_reasoning_tags

    query = state.get("rewritten_query", state.get("original_query", ""))
    context_text = state.get("retrieved_contexts", [""])[0] if state.get("retrieved_contexts") else ""
    existing_summary = state.get("existing_summary", "")
    model_name = state.get("model_used", settings.OPENAI_MODEL)

    messages: list = [{"role": "system", "content": _ANSWER_SYSTEM_PROMPT}]

    if existing_summary:
        messages.append({
            "role": "system",
            "content": f"[Conversation summary so far]\n{existing_summary}",
        })

    context_section = f"\nContext:\n{context_text}\n\nQuestion: {query}" if context_text else query
    messages.append({"role": "user", "content": context_section})

    model = llm or _get_llm(model_name, 0.0, api_base=api_base)
    streamed_parts: list[str] = []
    thinking_parts: list[str] = []
    usage: dict = {"promptTokens": 0, "completionTokens": 0}

    try:
        async for chunk in model.astream(messages):
            token: str = chunk.content or ""

            if settings.REASONING_MODEL and model_name == settings.REASONING_MODEL:
                is_thinking = True
                stripped = strip_reasoning_tags(token)
                if stripped != token:
                    full_match = re.search(r'</think>(.*?)</think>', token, re.DOTALL)
                    if full_match:
                        thinking_parts.append(full_match.group(1))
                        token = token[full_match.end():]
                    else:
                        open_match = re.search(r'</think>', token)
                        if open_match:
                            token = token[open_match.end():]

                if stripped and token != stripped:
                    thinking_parts.append(stripped)

            if token:
                streamed_parts.append(token)

            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                usage = {
                    "promptTokens": chunk.usage_metadata.get("input_tokens", 0),
                    "completionTokens": chunk.usage_metadata.get("output_tokens", 0),
                }
    except Exception as exc:
        logger.error("[NODE] generation failed: %s", exc)
        streamed_parts.append("I encountered an error generating the response. Please try again.")

    answer = "".join(streamed_parts)
    normalised = re.sub(
        r'\[(\d+)\](?!\()',
        lambda m: f'[{m.group(1)}]({m.group(1)})',
        answer,
    )

    # Validate chart JSON
    chart_pattern = re.compile(r'\[chart\](.*?)\[/chart\]', re.DOTALL)
    matches = list(chart_pattern.finditer(normalised))
    if matches:
        valid_count = 0
        for i, match in enumerate(matches):
            try:
                json.loads(match.group(1))
                valid_count += 1
            except (json.JSONDecodeError, TypeError):
                pass
        logger.info("[CHART] validation: %d valid of %d", valid_count, len(matches))

    return {
        "answer": normalised or answer,
        "thinking_chunks": thinking_parts,
    }


# ---------------------------------------------------------------------------
# Node: orchestrator (subgraph entry point)
# ---------------------------------------------------------------------------

async def orchestrator_node(
    state: AgentState,
    llm: ChatOpenAI,
    api_base: Optional[str] = None,
) -> dict:
    """Agent subgraph orchestrator."""
    messages = state.get("messages", [])

    tool_call_count = sum(
        1 for m in messages if hasattr(m, "tool_calls") and m.tool_calls
    )
    iteration_count = state.get("retrieval_iterations", 0)

    if iteration_count >= 8 or tool_call_count >= 20:
        return {"_orchestrator_result": "fallback"}

    return {"_orchestrator_result": "generate", "current_subtask_index": 0}


# ---------------------------------------------------------------------------
# Node: collect_answer
# ---------------------------------------------------------------------------

def collect_answer_node(state: AgentState) -> dict:
    """Collect the answer from the agent subgraph."""
    answer = state.get("answer", "")
    return {
        "subtask_answers": [{"answer": answer}],
    }


# ---------------------------------------------------------------------------
# Node: synthesize
# ---------------------------------------------------------------------------

async def synthesize_node(
    state: AgentState,
    llm: ChatOpenAI | None = None,
    api_base: Optional[str] = None,
) -> dict:
    """Synthesize final answer from subtask answers or direct answer."""
    subtask_answers = state.get("subtask_answers", [])
    final_answer = ""

    if len(subtask_answers) > 1:
        synthesis_parts = []
        for i, sa in enumerate(subtask_answers):
            answer_text = sa.get("answer", "") if isinstance(sa, dict) else str(sa)
            synthesis_parts.append(f"### Answer {i+1}\n\n{answer_text}")
        combined = "\n\n---\n\n".join(synthesis_parts)
        final_answer = combined
    elif subtask_answers:
        first = subtask_answers[0]
        final_answer = first.get("answer", first) if isinstance(first, dict) else first
    else:
        final_answer = state.get("answer", "")

    return {
        "final_answer": final_answer,
    }


# ---------------------------------------------------------------------------
# Node: fallback response
# ---------------------------------------------------------------------------

def fallback_response_node(state: AgentState) -> dict:
    """Fallback when budget exceeded or retrieval fails completely."""
    question = state.get("rewritten_query", state.get("original_query", ""))
    return {
        "messages": [AIMessage(content=(
            f"I wasn't able to find sufficient information in the documents "
            f"to fully answer your question about '{question}'. "
            f"You might want to try rephrasing or providing more context."
        ), name="fallback")],
    }


# ---------------------------------------------------------------------------
# Node: summarize history
# ---------------------------------------------------------------------------

async def summarize_history_node(
    state: AgentState,
    llm: ChatOpenAI,
) -> dict:
    """Reduce conversation history using rolling summaries."""
    messages = state.get("messages", [])

    plain_msgs = [m for m in messages if not getattr(m, "tool_calls", None)]
    keep_count = 4
    older = plain_msgs[:-keep_count] if len(plain_msgs) > keep_count else []
    if older:
        existing_summary = state.get("existing_summary", "").strip()
        conversation = f"Existing summary:\n{existing_summary or '(none)'}\n\n"
        conversation += "New messages:\n" + "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content[:200]}"
            for m in older
        )

        response = await llm.ainvoke([
            {"role": "system", "content": (
                "You are a conversation summarizer. Provide a concise summary of key facts, "
                "decisions, and context. Max 200 words."
            )},
            {"role": "user", "content": conversation},
        ])

        return {"existing_summary": response.content.strip() if hasattr(response, "content") else str(response)}

    return {}


# ---------------------------------------------------------------------------
# Node: compress context (between retrieval iterations)
# ---------------------------------------------------------------------------

def compress_context_node(state: AgentState) -> dict:
    """Compress accumulated retrieval context to free token budget."""
    retrieval_keys = set(state.get("retrieval_keys", set()) or set())

    for doc in state.get("retrieved_docs", []):
        source = doc.get("metadata", {}).get("source", "")
        if source:
            retrieval_keys.add(f"source:{source}")

    return {
        "retrieval_keys": list(retrieval_keys),
    }


# ---------------------------------------------------------------------------
# Node: should_compress_context (routing decision)
# ---------------------------------------------------------------------------

def should_compress_context(state: AgentState) -> str:
    """Decide whether to compress context based on token budget."""
    from .utils import estimate_messages_tokens

    messages = state.get("messages", [])
    current_tokens = estimate_messages_tokens(messages)
    max_allowed = settings.OPENAI_MODEL_CONTEXT_SIZE * 0.8

    if current_tokens > max_allowed:
        return "compress"
    return "next"
