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
 messages, jargon, exact terms. Args: {"query": "...", "top_k": 5}
- semantic_search: Dense vector search. Best for conceptual or paraphrased\
 questions. Args: {"query": "...", "top_k": 5}
- title_search: Document-level metadata search by title, status, date.\
 Args: {"title_contains": "...", "document_status": "active", "metadata_only": true}
- graph_expand: Find related entities/chunks through Neo4j graph relationships.\
 Args: {"seed_entity_names": [...], "rel_type": "...", "hops": 1}
- file_read: Read a specific document or file by ID.\
 Use the document_id from prior search results (shown as doc_id=N).\
 Args: {"document_id": N, "offset": 1, "limit": 200}
- kb_grep: Regex or literal search within one document.\
 Use the document_id from prior search results.\
 Args: {"pattern": "...", "document_id": N}
- kb_outline: Get document outline/structure.\
 Use the document_id from prior search results.\
 Args: {"document_id": N}

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
 kb_grep. Search results are already reranked — no separate rerank call needed.

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

{
  "query": "the original sub-query",
  "evidence": [
    {
      "citation_ref": {
        "document_id": 42,
        "citation_kind": "chunk",
        "chunk_index": 3,
        "page": 7,
        "quoted_text": "...",
        "source_tool": "semantic_search"
      },
      "document_id": 42,
      "score": 0.91
    }
  ],
  "gaps": ["list missing facts needed to fully answer the sub-query"],
  "conflicts": ["list any contradictions found in the evidence"],
  "complete": true_or_false,
  "failure_mode": "NO_HITS | LOW_RELEVANCE | ... or null if complete",
  "strategy": "the recovery or next-step strategy, or null if complete"
}

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


def _format_evidence_for_prompt(observations: list[Observation], max_docs: int = 15, max_chars: int = 400) -> str:
    """Format deduplicated evidence from all observations for the sub-agent prompt.

    Mirrors the main agent's _format_retrieved_docs_for_think: extracts evidence
    from all search/read observations, deduplicates by content_hash, and formats
    top N docs with title + score + content preview.
    """
    evidence = _extract_evidence_from_observations(observations)
    if not evidence:
        return ""
    # Sort by score descending
    evidence.sort(key=lambda e: e.get("score", 0.0), reverse=True)
    parts: list[str] = []
    for i, doc in enumerate(evidence[:max_docs], 1):
        title = (doc.get("title") or doc.get("file_name") or "Unknown")[:60]
        doc_id = doc.get("document_id", "")
        score = doc.get("score", 0.0)
        content = (doc.get("content") or "")[:max_chars].replace("\n", " ")
        parts.append(f"[E{i}] {title} (doc_id={doc_id}, score={score:.2f})\n  {content}")
    return "\n\n".join(parts)


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
                # Compact metadata for search tools (like main agent's
                # _observations_metadata_text). Actual evidence content
                # is shown in the separate "Evidence found so far" section
                # below, deduplicated across all tool calls.
                result = obs.result or {}
                if "hits" in result:
                    hits = result["hits"]
                    best_score = max((h.get("score", 0) or 0) for h in hits) if hits else 0
                    search_type = result.get("search_type", "")
                    type_text = f" type={search_type}" if search_type else ""
                    parts.append(f"     → hit_count={len(hits)} best_score={best_score:.3f}{type_text}\n")
                elif "docs" in result:
                    docs = result["docs"]
                    doc_count = len(docs)
                    confidence = result.get("confidence", "N/A")
                    parts.append(f"     → doc_count={doc_count} confidence={confidence}\n")
                elif obs.tool == "file_read":
                    # Show line range + content preview + continuation hint
                    # so the LLM can use what it read and page further if needed.
                    offset = obs.arguments.get("offset", 1)
                    limit = obs.arguments.get("limit", 200)
                    title = result.get("title", "")
                    total_lines = result.get("total_lines", "?")
                    end_line = result.get("end_line", offset + limit - 1)
                    content_preview = (result.get("content", "") or "")[:300].replace("\n", " ")
                    hint = result.get("continuation_hint", "")
                    parts.append(f"     → read lines {offset}-{end_line}/{total_lines} of '{title[:50]}' (doc_id={obs.arguments.get('document_id')})\n")
                    parts.append(f"       content: {content_preview}…\n")
                    if hint:
                        parts.append(f"       {hint}\n")
                elif obs.tool == "kb_grep" and "matches" in result:
                    matches = result.get("matches", [])
                    parts.append(f"     → {result.get('total_matches', len(matches))} matches in {result.get('documents_searched', '?')} docs:\n")
                    for m in matches[:5]:
                        parts.append(f"       • doc={m.get('document_id')} line={m.get('line_number')}: {(m.get('line_text', '') or '')[:80]}\n")
                elif obs.tool == "kb_outline" and "headings" in result:
                    headings = result.get("headings", [])
                    title = result.get("title", "")
                    parts.append(f"     → outline of '{title[:50]}' (doc_id={result.get('document_id')}): {len(headings)} headings\n")
                    for h in headings[:5]:
                        parts.append(f"       • {h.get('level', '?')}: {(h.get('text', '') or '')[:60]}\n")
                else:
                    parts.append(f"     → {json.dumps(result, default=str)[:150]}\n")
        parts.append("\n")

    # Evidence section: deduplicated content from all search/read calls.
    # This mirrors the main agent's _format_retrieved_docs_for_think —
    # the LLM sees actual evidence content to assess relevance without
    # calling another tool.
    evidence_text = _format_evidence_for_prompt(observations)
    if evidence_text:
        parts.append(f"Evidence found so far (deduplicated, cite by [E1], [E2], etc.):\n{evidence_text}\n\n")

    remaining = tool_budget - calls_used
    parts.append(f"Tool calls remaining: {remaining}/{tool_budget}\n")
    if remaining <= 0:
        parts.append("\nYou have exhausted your tool-call budget. Write your JSON summary now.")
    else:
        total_hits = sum(
            len(o.result.get("hits", o.result.get("docs", [])))
            for o in observations
            if not o.error and isinstance(o.result, dict)
        )
        distinct_tools = len({o.tool for o in observations if not o.error})
        if distinct_tools >= 5:
            # The LLM has tried many different tools. Even with hits, if the
            # evidence isn't relevant to the sub-query, it's time to finalize.
            parts.append(
                f"\nYou have tried {distinct_tools} different tools with {total_hits} total hits. "
                "Assess honestly: are the hits actually about the sub-query topic? "
                "If the results keep pointing to a different topic, the document likely doesn't "
                "contain what you need. Write your final JSON now with failure_mode=LOW_RELEVANCE "
                "and describe what you found in gaps."
            )
        elif total_hits >= 3:
            parts.append(f"\nYou already have {total_hits} hits from prior searches. If the evidence is relevant, write your final JSON now. Only call another tool if the results so far are clearly insufficient.")
        else:
            parts.append("\nCall the next tool, or write your final JSON in the required output format if you have enough evidence.")
    return "".join(parts)


def _extract_json_from_text(text: str) -> str:
    """Strip markdown fences and return the inner JSON body."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


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
        # {"page_content": "...", "metadata": {"document_id": N, "title": "...", ...}
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
    tool_budget: int = 10,
    subagent_id: str = "",
) -> dict:
    """Run a single retrieval sub-agent loop.

    Args:
        ctx: ToolContext (shared with main agent).
        sub_query: A single sub-query to retrieve evidence for.
        tool_budget: Per-subagent tool-call budget (from SUBAGENT_TOOL_BUDGET).
        subagent_id: Unique identifier for progress event streaming.

    Returns:
        dict with keys: ok, evidence (list of dicts), summary, query
    """
    # Lazy imports
    from app.services.agentic_rag.agent_graph.tooling import _run_tool
    from app.services.agentic_rag.agent_graph.observations import _tool_descriptions_text
    from app.services.agentic_rag.tools import build_tools
    from app.services.settings_service import get_setting
    from app.services.agentic_rag.agent_graph.helpers import _emit_timeline

    # Build search/read tools only
    all_tools = build_tools(ctx)
    retrieval_tool_names = {
        "keyword_search", "semantic_search",
        "title_search", "file_read", "kb_outline", "kb_grep",
        "graph_expand",
    }
    tools = {t.name: t for t in all_tools if t.name in retrieval_tool_names}
    tools_list = list(tools.values())
    tools_text = _tool_descriptions_text(tools_list)

    system = RETRIEVAL_SUBAGENT_PROMPT
    observations: list[Observation] = []
    counts: dict[str, int] = {}
    final_state = {}
    seen_signatures: set[str] = set()  # persists across iterations

    _emit_timeline(type="subagent_start", subagent_id=subagent_id,
                   subagent_type="retrieval", label=sub_query)

    iteration = 0
    while True:
        iteration += 1
        calls_used = sum(counts.values())
        # Budget counts ALL tool call attempts (successful + dedup-blocked +
        # errors). This prevents the LLM from wasting iterations on blocked
        # calls after the budget is notionally exhausted.
        total_attempts = len(observations)
        if calls_used >= tool_budget or total_attempts >= tool_budget or iteration > tool_budget + 2:
            break

        user = _build_retrieval_user_prompt(
            sub_query, tools_text, iteration, tool_budget, calls_used, observations,
        )

        try:
            tool_temp = get_setting(ctx.db, "TOOL_CALL_TEMPERATURE", ctx.org_id)
            llm = build_chat_llm(ctx.org_id, ctx.db, role="chat", temperature=tool_temp)
            if iteration == 1:
                from app.services.agentic_rag.llm_factory import get_org_llm
                cfg = get_org_llm(ctx.org_id, ctx.db, role="chat")
                logger.info("[retrieval_subagent %s] model=%s base=%s",
                            subagent_id, cfg["model_name"], cfg["api_base"])
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
                    json_text = _extract_json_from_text(parsed.final_answer)
                    final_state = json.loads(json_text)
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

            # Dedup guard: skip identical tool+key-arg combinations.
            # The signature uses the primary search key for each tool:
            #   query for keyword/semantic search
            #   title_contains for title_search
            #   pattern for kb_grep
            #   document_id for kb_outline
            #   document_id:offset for file_read (allows paging)
            # This prevents the LLM from repeating the same search with
            # cosmetic variations (e.g. metadata_only=true vs false).
            if name == "file_read":
                sig_key = f"{args.get('document_id')}:{args.get('offset', 1)}"
            else:
                sig_key = (
                    args.get("query")
                    or args.get("title_contains")
                    or args.get("pattern")
                    or str(args.get("document_id") or "")
                )
            sig = f"{name}:{sig_key}"
            if sig in seen_signatures:
                observations.append(Observation(
                    tool=name, arguments=args, result={},
                    error=f"Duplicate call: {sig} already tried. Try a different tool or query.",
                    tokens=0,
                ))
                continue
            seen_signatures.add(sig)

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

            label = getattr(tool, "ui_label", name)
            tool_step = _emit_timeline(type="subagent_step", subagent_id=subagent_id,
                                       step_type="tool", tool=name, label=label,
                                       status="active")

            result = await _run_tool(tool, name, args)
            obs = Observation(
                tool=result["tool"], arguments=result["arguments"],
                result=result.get("result", {}), error=result.get("error"),
                tokens=result.get("tokens", 0),
            )
            observations.append(obs)
            counts[name] = counts.get(name, 0) + 1
            calls_used += 1

            hit_count = 0
            if not obs.error and isinstance(obs.result, dict):
                hit_count = obs.result.get("count", 0)
            if obs.error:
                logger.warning("[retrieval_subagent %s] tool %s failed: %s",
                               subagent_id, name, obs.error)
            _emit_timeline(id=tool_step, type="subagent_step", subagent_id=subagent_id,
                           step_type="tool", tool=name, label=label,
                           hit_count=hit_count, error=bool(obs.error),
                           status="complete")

    # Extract evidence from all observations and merge with any citations
    # the sub-agent explicitly included in its final JSON.
    extracted = _extract_evidence_from_observations(observations)
    explicit_citations = {c.get("document_id") for c in final_state.get("evidence", []) if c.get("document_id")}
    if explicit_citations and final_state.get("evidence"):
        # Re-order extracted evidence so explicitly cited documents appear first,
        # while still keeping the full content for the parent agent.
        ordered = sorted(
            extracted,
            key=lambda e: (e.get("document_id") in explicit_citations),
            reverse=True,
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

    _emit_timeline(type="subagent_done", subagent_id=subagent_id,
                   subagent_type="retrieval", label=sub_query,
                   succeeded=len(evidence) > 0, evidence_count=len(evidence))

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
    tool_budget: int = 10,
) -> list[dict]:
    """Run multiple retrieval sub-agents in parallel.

    Args:
        ctx: ToolContext (shared — each sub-agent gets its own copy of tools).
        sub_queries: List of independent sub-queries.
        tool_budget: Per-subagent tool-call budget (from SUBAGENT_TOOL_BUDGET).

    Returns:
        list of dicts (one per sub-query): ok, evidence, summary, query
    """
    import uuid

    subagent_ids = [str(uuid.uuid4())[:8] for _ in sub_queries]
    tasks = [
        run_retrieval_subagent(ctx, q, tool_budget=tool_budget, subagent_id=sid)
        for q, sid in zip(sub_queries, subagent_ids)
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
