"""Shared tooling helpers for the agent graph.

Provides utility functions used by v2 tooling and subagents:
- _run_tool: execute a single tool call
- _summarize_result: human-readable result summary
- _merge_retrieved_docs: promote search/read docs into graph state
- _hit_to_doc_dict: convert a search hit to a doc dict
"""

from __future__ import annotations

import logging

from app.services.agentic_rag.schemas import Observation

logger = logging.getLogger(__name__)


def _summarize_result(obs: Observation) -> str:
    """Build a one-line summary of a tool observation result for the UI."""
    if obs.error:
        return obs.error[:120]
    r = obs.result
    if not isinstance(r, dict):
        return ""
    if "hits" in r and isinstance(r["hits"], list):
        n = len(r["hits"])
        return f"{n} {'hit' if n == 1 else 'hits'} retrieved"
    if "docs" in r and isinstance(r["docs"], list):
        n = len(r["docs"])
        return f"{n} {'doc' if n == 1 else 'docs'} retrieved"
    if "matches" in r and isinstance(r["matches"], list):
        n = len(r["matches"])
        return f"{n} {'match' if n == 1 else 'matches'} found"
    if "content" in r:
        tokens = r.get("total_tokens", "?")
        return f"Read {tokens} tokens"
    if "headings" in r and isinstance(r["headings"], list):
        n = len(r["headings"])
        return f"{n} {'heading' if n == 1 else 'headings'}"
    if "points" in r and isinstance(r["points"], list):
        n = len(r["points"])
        return f"{n} {'data point' if n == 1 else 'data points'} extracted"
    if "chart_option" in r:
        return "Chart generated"
    if "file_id" in r and r.get("file_id") and "format" in r and r.get("format"):
        fmt = r["format"].upper()
        name = r.get("file_name", "")
        charts = r.get("chart_count", 0)
        suffix = f" ({charts} charts)" if charts else ""
        return f"{fmt} generated: {name}{suffix}"
    if "mode" in r and r.get("mode") in ("issues", "screenshot", "outline", "validate", "annotated", "text", "get", "query"):
        mode = r["mode"]
        output = r.get("output", "")
        if mode == "issues":
            issue_count = output.count("issue") if isinstance(output, str) else 0
            return f"QA: {issue_count} issues" if issue_count else "QA: no issues"
        return f"Inspected ({mode})"
    if "commands_applied" in r:
        return f"Edited: {r['commands_applied']} commands applied"
    if "result" in r and isinstance(r["result"], str):
        return r["result"][:120]
    return ""


def _seed_existing_docs(existing_docs, seen_hashes, merged_docs):
    from app.services.infrastructure import content_hash as _ch
    for doc in existing_docs or []:
        if not isinstance(doc, dict):
            continue
        h = doc.get("metadata", {}).get("content_hash") or _ch(doc.get("page_content", ""))
        if h not in seen_hashes:
            seen_hashes.add(h)
            merged_docs.append(doc)


# Tools that return hits in the new atomic search format: {"hits": [...]}
_SEARCH_TOOLS = frozenset({"keyword_search", "semantic_search", "graph_expand", "retrieve_parallel"})


def _hit_to_doc_dict(hit: dict) -> dict:
    """Convert a search tool hit (flat dict) to the standard doc dict shape."""
    return {
        "page_content": hit.get("content", ""),
        "metadata": {
            "document_id": hit.get("document_id"),
            "chunk_index": hit.get("chunk_index"),
            "page": hit.get("page"),
            "title": hit.get("title", ""),
            "file_name": hit.get("file_name", ""),
            "content_hash": hit.get("content_hash", ""),
            "qdrant_point_id": hit.get("qdrant_point_id", ""),
            "_reranker_score": hit.get("_reranker_score", hit.get("score", 0.0)),
            "citation_ref": hit.get("citation_ref", {}),
        },
    }


def _merge_observation_docs(all_observations, seen_hashes, merged_docs):
    from app.services.infrastructure import content_hash as _ch
    best_confidence = 0.0
    for obs in all_observations:
        if obs.tool in _SEARCH_TOOLS and not obs.error:
            hits = obs.result.get("hits")
            if isinstance(hits, list):
                for hit in hits:
                    if not isinstance(hit, dict):
                        continue
                    doc_dict = _hit_to_doc_dict(hit)
                    h = doc_dict["metadata"].get("content_hash") or _ch(doc_dict.get("page_content", ""))
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        merged_docs.append(doc_dict)
                # Search hits with reranker scores or dense scores contribute confidence.
                # _reranker_score (from cross-encoder reranking inside search tools)
                # that can be negative; normalize via sigmoid to 0-1.
                # score from semantic_search is cosine similarity (0-1).
                # score from keyword_search (exact leg) is MySQL FTS score (0-10+); clamp to 0-1.
                # score from keyword_search (sparse leg) is SPLADE dot product (0-10+); clamp to 0-1.
                for h in hits:
                    rs = h.get("_reranker_score")
                    if rs is not None:
                        norm = 1.0 / (1.0 + pow(2.718281828, -rs))
                    else:
                        raw_score = h.get("score", 0.0)
                        # Dense cosine similarity is 0-1; SPLADE/FTS scores can be >1.
                        # Clamp to 0-1 range.
                        norm = min(raw_score, 1.0) if raw_score > 0 else 0.0
                    if norm > best_confidence:
                        best_confidence = norm
                logger.debug(
                    "[tool_node] merged search hits: tool=%s hits=%d best_confidence=%.3f",
                    obs.tool, len(hits), best_confidence,
                )
        elif obs.tool == "title_search" and not obs.error:
            docs = obs.result.get("docs")
            if isinstance(docs, list):
                for doc in docs:
                    if not isinstance(doc, dict):
                        continue
                    h = doc.get("metadata", {}).get("content_hash") or _ch(doc.get("page_content", ""))
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        merged_docs.append(doc)
                # Document-level matches are high-confidence by definition.
                if best_confidence < 0.9:
                    best_confidence = 0.9
        elif obs.tool == "file_read" and not obs.error:
            # file_read returns a single document/file's content, not a docs list.
            # Convert to the standard doc dict shape so it gets a [KB-N]
            # label in the finalize prompt and becomes citable evidence.
            content = obs.result.get("content", "")
            if content:
                citation_ref = obs.result.get("citation_ref", {})
                doc_dict = {
                    "page_content": content,
                    "metadata": {
                        "document_id": obs.result.get("document_id"),
                        "file_id": obs.result.get("file_id"),
                        "source_type": obs.result.get("source_type"),
                        "title": obs.result.get("title") or obs.result.get("file_name"),
                        "file_name": obs.result.get("file_name"),
                        "source": "file_read",
                        "_reranker_score": 1.0,
                        "truncated": obs.result.get("truncated", False),
                        "citation_ref": citation_ref,
                    },
                }
                h = _ch(content)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    merged_docs.append(doc_dict)
                if best_confidence < 0.9:
                    best_confidence = 0.9
    return best_confidence


def _merge_retrieved_docs(
    all_observations: list[Observation],
    existing_docs: list[dict],
) -> tuple[list[dict], float]:
    """Promote all search/read docs into graph state (deduplicated across
    observations by content_hash).

    `observations` uses the append-style `accumulate` reducer, so tool_node
    must return ONLY the observations it created. Returning prior + new made
    the channel grow 1 \u2192 3 \u2192 7 \u2192 15 across tool rounds.
    """
    merged_docs: list[dict] = []
    seen_hashes: set[str] = set()
    _seed_existing_docs(existing_docs, seen_hashes, merged_docs)
    best_confidence = _merge_observation_docs(all_observations, seen_hashes, merged_docs)
    return merged_docs, best_confidence


async def _run_tool(tool, name: str, args: dict) -> dict:
    try:
        # Normalize nested dict keys — some LLM providers return keys with
        # extra quotes (e.g. '"title"' instead of 'title'). Call the tool's
        # prepare_arguments if it exists, otherwise pass through.
        if hasattr(tool, "prepare_arguments"):
            args = tool.prepare_arguments(args)
        raw = await tool.arun(args)
        # Tools return {"ok": bool, "result": {...}, "error": str|None, "tokens": int, "terminate": bool}.
        # Unwrap the envelope so obs.result is the inner payload (e.g. {"docs": [...], ...}).
        if isinstance(raw, dict) and "result" in raw:
            return {
                "tool": name,
                "arguments": args,
                "result": raw.get("result", {}),
                "error": raw.get("error"),
                "tokens": raw.get("tokens", 0),
                "terminate": raw.get("terminate", False),
            }
        return {"tool": name, "arguments": args, "result": raw, "error": None, "tokens": 0, "terminate": False}
    except Exception as exc:
        # GraphInterrupt must propagate to LangGraph so it can checkpoint
        # and pause the graph. Do not turn it into an error observation.
        from langgraph.errors import GraphInterrupt
        if isinstance(exc, GraphInterrupt):
            raise
        logger.warning("[_run_tool] %s failed: %s", name, exc)
        return {"tool": name, "arguments": args, "result": {}, "error": str(exc), "tokens": 0, "terminate": False}
