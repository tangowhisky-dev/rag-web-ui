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
 sub-query and return concise results with citation info.

# Available Tools

- search_exact: Exact phrase match. Best for named documents, specific terms.\
 Args: {{"query": "...", "top_k": 5}}
- search_dense: Semantic search. Best for conceptual questions.\
 Args: {{"query": "...", "top_k": 5}}
- search_sparse: Keyword/BM25 search. Best for broad term matching.\
 Args: {{"query": "...", "top_k": 5}}
- kb_search_documents: Find documents by title or metadata.\
 Args: {{"title_contains": "...", "metadata_only": false}}
- kb_read: Read a specific document by ID.\
 Args: {{"document_id": N, "max_chars": 2000}}
- kb_outline: Get document outline/structure.\
 Args: {{"document_id": N}}
- kb_grep: Regex search within documents.\
 Args: {{"pattern": "...", "document_id": N}}
- rerank_results: Rerank already-retrieved results by relevance.\
 Call after a search if results seem mixed.

# Strategy

1. For NAMED documents or specific terms: start with search_exact or\
 kb_search_documents.
2. For CONCEPTUAL questions: start with search_dense.
3. For BROAD keyword matching: start with search_sparse.
4. If first search returns irrelevant results: try a different search type\
 or rerank_results.
5. If you find the right document but need more context: call kb_read.
6. Do NOT repeat the same search with the same query.

# Rules

- You have at most {max_iterations} rounds.
- Return evidence, NOT an answer. Do not write prose explanations.
- When you have enough evidence, write a JSON summary (no tool calls):
  {{"evidence_found": true, "summary": "brief description of what was found",\
 "query": "the sub-query"}}
- If no relevant evidence found:
  {{"evidence_found": false, "summary": "no relevant results", "query": "..."}}
- Keep your output concise — the orchestrator will synthesize.
"""


def _build_retrieval_user_prompt(
    sub_query: str,
    tools_text: str,
    iteration: int,
    max_iter: int,
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

    parts.append(f"Round: {iteration}/{max_iter}\n")
    if iteration >= max_iter:
        parts.append("\nYou have reached the limit. Write your JSON summary now.")
    else:
        parts.append("\nCall the next tool, or write your JSON summary if you have enough evidence.")
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

        # kb_search_documents returns "docs" with structure:
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

        # kb_read returns "content" or "sections"
        if obs.tool == "kb_read":
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
                        "source_tool": "kb_read",
                        "citation_id": "",
                    },
                    "source_tool": "kb_read",
                })

    return evidence


async def run_retrieval_subagent(
    ctx,
    sub_query: str,
    max_iterations: int = 4,
) -> dict:
    """Run a single retrieval sub-agent loop.

    Args:
        ctx: ToolContext (shared with main agent).
        sub_query: A single sub-query to retrieve evidence for.
        max_iterations: Max think→tool rounds.

    Returns:
        dict with keys: ok, evidence (list of dicts), summary, query
    """
    # Lazy imports
    from app.services.agentic_rag.agent_graph.helpers import _writer as _get_writer
    from app.services.agentic_rag.agent_graph.tooling import _run_tool
    from app.services.agentic_rag.agent_graph.observations import _tool_descriptions_text
    from app.services.agentic_rag.tools import build_tools

    writer = _get_writer()

    # Build search/read tools only
    all_tools = build_tools(ctx)
    retrieval_tool_names = {
        "search_exact", "search_dense", "search_sparse",
        "kb_search_documents", "kb_read", "kb_outline", "kb_grep",
        "rerank_results",
    }
    tools = {t.name: t for t in all_tools if t.name in retrieval_tool_names}
    tools_list = list(tools.values())
    tools_text = _tool_descriptions_text(tools_list)

    system = RETRIEVAL_SUBAGENT_PROMPT.format(max_iterations=max_iterations)
    observations: list[Observation] = []
    counts: dict[str, int] = {}
    summary = ""

    for iteration in range(1, max_iterations + 1):
        user = _build_retrieval_user_prompt(
            sub_query, tools_text, iteration, max_iterations, observations,
        )

        try:
            llm = build_chat_llm(ctx.org_id, ctx.db, role="chat", temperature=0.0)
            resp = await llm.bind_tools(tools_list).ainvoke([
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ])
        except Exception as exc:
            logger.warning("[retrieval_subagent] LLM call failed: %s", exc)
            break

        parsed = parse_think_response(resp, mode="auto")
        tool_calls = parsed.tool_calls

        if iteration >= max_iterations:
            tool_calls = []

        if not tool_calls:
            # Sub-agent wrote a summary
            if isinstance(parsed.final_answer, str):
                summary = parsed.final_answer.strip()
            break

        # Execute tool calls
        for tc in tool_calls:
            name = tc.get("tool")
            args = tc.get("arguments", {})
            tool = tools.get(name)
            if tool is None:
                observations.append(Observation(
                    tool=name, arguments=args, result={},
                    error=f"Tool '{name}' not available", tokens=0,
                ))
                continue

            # Per-tool cap
            cap = _retrieval_tool_cap(name)
            if counts.get(name, 0) >= cap:
                observations.append(Observation(
                    tool=name, arguments=args, result={},
                    error=f"Tool '{name}' cap ({cap}) reached. Use a different tool or write summary.",
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

    # Extract evidence from all observations
    evidence = _extract_evidence_from_observations(observations)

    return {
        "ok": len(evidence) > 0,
        "evidence": evidence[:15],  # Cap at 15 chunks to keep main agent context clean
        "summary": summary,
        "query": sub_query,
    }


def _retrieval_tool_cap(tool_name: str) -> int:
    """Per-tool cap for retrieval sub-agent."""
    caps = {
        "search_exact": 2,
        "search_dense": 2,
        "search_sparse": 2,
        "kb_search_documents": 3,
        "kb_read": 3,
        "kb_outline": 2,
        "kb_grep": 2,
        "rerank_results": 1,
    }
    return caps.get(tool_name, 2)


async def run_retrieval_subagents_parallel(
    ctx,
    sub_queries: list[str],
    max_iterations: int = 4,
) -> list[dict]:
    """Run multiple retrieval sub-agents in parallel.

    Args:
        ctx: ToolContext (shared — each sub-agent gets its own copy of tools).
        sub_queries: List of independent sub-queries.
        max_iterations: Max think→tool rounds per sub-agent.

    Returns:
        list of dicts (one per sub-query): ok, evidence, summary, query
    """
    writer = _get_writer() if False else None  # writer not needed here

    tasks = [
        run_retrieval_subagent(ctx, q, max_iterations=max_iterations)
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
                "summary": f"Sub-agent failed: {result}",
                "query": sub_queries[i],
            })
        else:
            normalized.append(result)
    return normalized
