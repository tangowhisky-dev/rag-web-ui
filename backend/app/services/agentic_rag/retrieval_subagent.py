"""Retrieval sub-agent for parallel evidence gathering.

A mini think→tool loop dedicated to retrieval. Has only search/read tools
and a focused prompt: "find the best evidence for this sub-query and return
concise results with citation info."

The main agent calls `retrieve_parallel` (a wrapper tool) which spawns one
or more of these sub-agents concurrently. Each sub-agent:
1. Searches the KB (exact, dense, sparse, document-level)
2. Optionally reranks and reads full documents
3. Returns top evidence chunks with citation metadata

The main agent synthesizes results from all sub-agents into a cohesive answer.

Activation policy: the main agent decides when to parallelize. For simple
queries it calls search tools directly. For complex multi-part queries it
calls retrieve_parallel with independent sub-queries.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from app.services.agentic_rag.llm_factory import build_chat_llm
from app.services.agentic_rag.tool_call_parser import parse_think_response
from app.services.agentic_rag.schemas import Observation

logger = logging.getLogger(__name__)


RETRIEVAL_SUBAGENT_PROMPT = """\
You are a retrieval specialist. Your job: find the best evidence for a single\
 sub-query, diagnose retrieval failures, and return a structured result the\
 parent agent can use. Do not write prose answers.

# Available Tools

- keyword_search: Lexical keyword match. Best for identifiers, code, error\
 messages, jargon, exact terms. Args: {{"query": "...", "top_k": 5}}
- semantic_search: Dense vector search. Best for conceptual or paraphrased\
 questions. Args: {{"query": "...", "top_k": 5}}
- title_search: Document-level metadata search by title, status, date.\
 Args: {{"title_contains": "...", "document_status": "active", "metadata_only": true}}
- graph_expand: Find related entities/chunks through Neo4j graph relationships.\
 Args: {{"seed_entity_names": [...], "rel_type": "...", "hops": 1}}
- file_read: Read a specific document or file by ID.\
 Args: {{"document_id": N, "offset": 1, "limit": 200}}
- kb_grep: Regex or literal search within one document.\
 Args: {{"pattern": "...", "document_id": N}}
- kb_outline: Get document outline/structure. Args: {{"document_id": N}}
- rerank_results: Rerank a mixed result set. Args: {{"top_k": 5}}

# Strategy

1. Pick the tool that matches the sub-query type: named/ID → keyword_search or\
 title_search; conceptual → semantic_search; multi-hop/relationship → graph_expand;\
 in-document lookup → kb_grep.
2. When a search fails, do NOT repeat it with a reworded query. Change exactly\
 one dimension:
   - lexical ↔ semantic
   - broader ↔ narrower
   - document-level ↔ section-level (file_read/kb_grep)
   - current ↔ historical (title_search with date filters)
   - direct ↔ relationship (graph_expand)
   - content ↔ metadata (title_search)
3. If you find the right document but need more context: call file_read or\
 kb_grep. If results are mixed, call rerank_results.

# Failure Modes and Recovery

After a weak or failed search, classify the problem and use the matching recovery:

- NO_HITS: nothing returned. Switch modality (lexical ↔ semantic) or try\
 title_search / graph_expand.
- LOW_RELEVANCE: hits are off-topic. Make the query broader or narrower; switch\
 to keyword_search if semantic is too fuzzy, or semantic if keyword is too strict.
- LOW_SPECIFICITY: results are too vague. Add a distinctive term or switch to\
 kb_grep / file_read for a specific document.
- MISSING_ENTITY: the entity is not found by direct search. Use graph_expand from\
 a known, related seed entity.
- MISSING_RELATIONSHIP: a connection between two entities is needed. Use graph_expand.
- MISSING_VERSION: you need the active/current version. Use title_search with\
 document_status="active" and effective_as_of.
- MISSING_DATE: a date is needed. Call current_datetime, then use title_search with\
 modified_after / modified_before / effective_as_of.
- CONFLICTING_SOURCES: different sources disagree. Use title_search for the latest/\
 authoritative version, or file_read the specific documents.
- INDEX_FAILURE: keyword/semantic did not find a known phrase. Use kb_grep with a\
 precise pattern.

# Output Format

When you have enough evidence, or have exhausted the budget, return a single JSON\
 object (no markdown, no tool call):

{{
  "query": "the original sub-query",
  "evidence": [
    {{
      "citation_ref": {{
        "document_id": 42,
        "citation_kind": "chunk",
        "chunk_index": 3,
        "page": 7,
        "quoted_text": "...",
        "source_tool": "semantic_search"
      }},
      "document_id": 42,
      "score": 0.91
    }}
  ],
  "gaps": ["list missing facts needed to fully answer the sub-query"],
  "conflicts": ["list any contradictions found in the evidence"],
  "complete": true_or_false,
  "failure_mode": "NO_HITS | LOW_RELEVANCE | ... or null if complete",
  "strategy": "the recovery or next-step strategy, or null if complete"
}}

- `evidence` should cite the top 5-10 most useful chunks or documents you found.\
 Do not include full text — only citation refs, document_id, and score.
- `gaps` and `conflicts` are arrays of strings. Use [] if none.
- `complete` is true only if the sub-query is fully answered by the evidence.
- `failure_mode` and `strategy` are required when `complete` is false.

# Rules

- You have a limited tool-call budget. The prompt shows how many calls remain.
- Do not answer the question in prose; only return the JSON above.
- Do not repeat the same tool with a reworded query.
- Keep `evidence` concise — the parent will read the actual chunks.
"""


def _build_retrieval_user_prompt(
    sub_query: str,
    tools_text: str,
    iteration: int,
    tool_budget: int,
    calls_used: int,
    observations: list[Observation],
) -> str:
    """Build the user prompt for the retrieval sub-agent."""
    parts: list[str] = []
    parts.append(f"Sub-query: {sub_query}\n\n")
    parts.append(f"Available tools:\n{tools_text}\n\n")

    if observations:
        parts.append("Prior tool calls:\n")
        for i, obs in enumerate(observations, 1):
            args_str = json.dumps(obs.arguments, default=str)[:150]
            parts.append(f"  {i}. {obs.tool}({args_str})")
            if obs.error:
                parts.append(f"     → ERROR: {obs.error[:200]}\n")
            else:
                # Show hit count and top result title
                result = obs.result or {}
                if "hits" in result:
                    hits = result["hits"]
                    top_title = hits[0].get("title", "") if hits else ""
                    parts.append(f"     → {len(hits)} hits, top: {top_title[:60]}\n")
                elif "docs" in result:
                    docs = result["docs"]
                    top_title = docs[0].get("title", "") if docs else ""
                    parts.append(f"     → {len(docs)} docs, top: {top_title[:60]}\n")
                else:
                    parts.append(f"     → {json.dumps(result, default=str)[:100]}\n")
        parts.append("\n")

    remaining = tool_budget - calls_used
    parts.append(f"Tool calls remaining: {remaining}/{tool_budget}\n")
    if remaining <= 0:
        parts.append("\nYou have exhausted your tool-call budget. Write your JSON summary now.")
    else:
        parts.append("\nCall the next tool, or write your final JSON in the required output format if you have enough evidence.")
    return "".join(parts)


def _extract_evidence_from_observations(observations: list[Observation]) -> list[dict]:
    """Extract evidence chunks from search/read observations."""
    evidence: list[dict] = []
    seen_hashes: set[str] = set()

    for obs in observations:
        if obs.error:
            continue
        result = obs.result or {}

        # search_* tools return "hits"
        hits = result.get("hits", [])
        for hit in hits:
            content_hash = hit.get("content_hash", "")
            if content_hash and content_hash in seen_hashes:
                continue
            if content_hash:
                seen_hashes.add(content_hash)
            evidence.append({
                "document_id": hit.get("document_id"),
                "chunk_index": hit.get("chunk_index"),
                "page": hit.get("page"),
                "title": hit.get("title", ""),
                "file_name": hit.get("file_name", ""),
                "content": hit.get("content", ""),
                "score": hit.get("score", 0.0),
                "citation_ref": hit.get("citation_ref", {}),
                "source_tool": obs.tool,
            })

        # title_search returns "docs" with structure:
        # {"page_content": "...", "metadata": {"document_id": N, "title": "...", ...}}
        docs = result.get("docs", [])
        for doc in docs:
            meta = doc.get("metadata", {}) if isinstance(doc, dict) else {}
            doc_id = doc.get("id") or doc.get("document_id") or meta.get("document_id")
            content = doc.get("page_content") or doc.get("content") or ""
            title = meta.get("title") or meta.get("file_name") or doc.get("title") or doc.get("file_name") or ""
            file_name = meta.get("file_name") or doc.get("file_name", "")
            if content:
                evidence.append({
                    "document_id": doc_id,
                    "chunk_index": meta.get("chunk_index"),
                    "page": meta.get("page"),
                    "title": title,
                    "file_name": file_name,
                    "content": content[:2000],
                    "score": meta.get("_reranker_score", meta.get("score", 0.0)),
                    "citation_ref": {
                        "document_id": doc_id,
                        "citation_kind": "document",
                        "quoted_text": content[:200],
                        "source_tool": obs.tool,
                        "citation_id": "",
                    },
                    "source_tool": obs.tool,
                })

        # file_read returns "content"
        if obs.tool == "file_read":
            content = result.get("content", "")
            if content:
                doc_id = obs.arguments.get("document_id")
                evidence.append({
                    "document_id": doc_id,
                    "chunk_index": None,
                    "page": None,
                    "title": result.get("title", ""),
                    "file_name": result.get("file_name", ""),
                    "content": content[:2000],
                    "score": 1.0,
                    "citation_ref": {
                        "document_id": doc_id,
                        "citation_kind": "document",
                        "quoted_text": content[:200],
                        "source_tool": "file_read",
                        "citation_id": "",
                    },
                    "source_tool": "file_read",
                })

    return evidence


async def run_retrieval_subagent(
    ctx,
    sub_query: str,
    tool_budget: int = 25,
) -> dict:
    """Run a single retrieval sub-agent loop.

    Args:
        ctx: ToolContext (shared with main agent).
        sub_query: A single sub-query to retrieve evidence for.
        tool_budget: Total tool-call budget (inherited from AGENT_TOTAL_TOOL_BUDGET).

    Returns:
        dict with keys: ok, evidence (list of dicts), summary, query
    """
    # Lazy imports
    from app.services.agentic_rag.agent_graph.helpers import _writer as _get_writer
    from app.services.agentic_rag.agent_graph.tooling import _run_tool
    from app.services.agentic_rag.agent_graph.observations import _tool_descriptions_text
    from app.services.agentic_rag.tools import build_tools
    from app.services.settings_service import get_setting

    writer = _get_writer()

    # Build search/read tools only
    all_tools = build_tools(ctx)
    retrieval_tool_names = {
        "keyword_search", "semantic_search",
        "title_search", "file_read", "kb_outline", "kb_grep",
        "rerank_results", "graph_expand",
    }
    tools = {t.name: t for t in all_tools if t.name in retrieval_tool_names}
    tools_list = list(tools.values())
    tools_text = _tool_descriptions_text(tools_list)

    system = RETRIEVAL_SUBAGENT_PROMPT
    observations: list[Observation] = []
    counts: dict[str, int] = {}
    final_state = {}
    total_budget = get_setting(ctx.db, "AGENT_TOTAL_TOOL_BUDGET", ctx.org_id)

    iteration = 0
    while True:
        iteration += 1
        calls_used = sum(counts.values())
        if calls_used >= tool_budget:
            break

        user = _build_retrieval_user_prompt(
            sub_query, tools_text, iteration, tool_budget, calls_used, observations,
        )

        try:
            tool_temp = get_setting(ctx.db, "TOOL_CALL_TEMPERATURE", ctx.org_id)
            llm = build_chat_llm(ctx.org_id, ctx.db, role="chat", temperature=tool_temp)
            resp = await llm.bind_tools(tools_list).ainvoke([
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ])
        except Exception as exc:
            logger.warning("[retrieval_subagent] LLM call failed: %s", exc)
            break

        parsed = parse_think_response(resp, mode="auto")
        tool_calls = parsed.tool_calls

        if not tool_calls:
            # Sub-agent wrote the final JSON output.
            if isinstance(parsed.final_answer, str):
                try:
                    final_state = json.loads(parsed.final_answer.strip())
                except json.JSONDecodeError:
                    logger.warning(
                        "[retrieval_subagent] final answer is not valid JSON: %s",
                        parsed.final_answer[:200],
                    )
                    final_state = {"complete": False, "gaps": ["Sub-agent did not return valid JSON"]}
            break

        # Execute tool calls
        for tc in tool_calls:
            name = tc.get("tool")
            args = tc.get("arguments", {})
            tool = tools.get(name)

            # Total tool-call budget (shared with main agent)
            if sum(counts.values()) >= total_budget:
                observations.append(Observation(
                    tool=name, arguments=args, result={},
                    error=f"Total tool-call budget ({total_budget}) reached. Write your final JSON now.",
                    tokens=0,
                ))
                break

            if tool is None:
                observations.append(Observation(
                    tool=name, arguments=args, result={},
                    error=f"Tool '{name}' not available", tokens=0,
                ))
                continue

            if calls_used >= tool_budget:
                observations.append(Observation(
                    tool=name, arguments=args, result={},
                    error=f"Tool-call budget ({tool_budget}) exhausted. Write your final JSON.",
                    tokens=0,
                ))
                continue

            result = await _run_tool(tool, name, args)
            obs = Observation(
                tool=result["tool"], arguments=result["arguments"],
                result=result.get("result", {}), error=result.get("error"),
                tokens=result.get("tokens", 0),
            )
            observations.append(obs)
            counts[name] = counts.get(name, 0) + 1
            calls_used += 1

    # Extract evidence from all observations and merge with any citations
    # the sub-agent explicitly included in its final JSON.
    extracted = _extract_evidence_from_observations(observations)
    explicit_citations = {c.get("document_id") for c in final_state.get("evidence", []) if c.get("document_id")}
    if explicit_citations and final_state.get("evidence"):
        # Re-order extracted evidence so explicitly cited documents appear first,
        # while still keeping the full content for the parent agent.
        ordered = sorted(
            extracted,
            key=lambda e: (e.get("document_id") not in explicit_citations),
        )
    else:
        ordered = extracted

    evidence = ordered[:15]  # Cap at 15 chunks to keep main agent context clean
    if final_state:
        complete = bool(final_state.get("complete", False))
    else:
        complete = len(evidence) > 0
    gaps = list(final_state.get("gaps", []))
    conflicts = list(final_state.get("conflicts", []))
    failure_mode = final_state.get("failure_mode") or None
    strategy = final_state.get("strategy") or None
    summary = "; ".join(gaps + conflicts) if (gaps or conflicts) else ("complete" if complete else "no evidence")

    return {
        "ok": len(evidence) > 0,
        "evidence": evidence,
        "query": sub_query,
        "complete": complete,
        "gaps": gaps,
        "conflicts": conflicts,
        "failure_mode": failure_mode,
        "strategy": strategy,
        "summary": summary,
    }


async def run_retrieval_subagents_parallel(
    ctx,
    sub_queries: list[str],
    tool_budget: int = 25,
) -> list[dict]:
    """Run multiple retrieval sub-agents in parallel.

    Args:
        ctx: ToolContext (shared — each sub-agent gets its own copy of tools).
        sub_queries: List of independent sub-queries.
        tool_budget: Total tool-call budget per sub-agent (inherited from AGENT_TOTAL_TOOL_BUDGET).

    Returns:
        list of dicts (one per sub-query): ok, evidence, summary, query
    """
    writer = _get_writer() if False else None  # writer not needed here

    tasks = [
        run_retrieval_subagent(ctx, q, tool_budget=tool_budget)
        for q in sub_queries
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    normalized: list[dict] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning("[retrieval_parallel] sub-agent %d failed: %s", i, result)
            normalized.append({
                "ok": False,
                "evidence": [],
                "query": sub_queries[i],
                "complete": False,
                "gaps": [f"Sub-agent failed: {result}"],
                "conflicts": [],
                "failure_mode": "INDEX_FAILURE",
                "strategy": "Retry with kb_grep or direct search",
                "summary": f"Sub-agent failed: {result}",
            })
        else:
            normalized.append(result)
    return normalized
