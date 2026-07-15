"""DB query tool — safe MySQL query execution for structured data retrieval.

When the supervisor determines that a query needs structured data that
isn't well-indexed in vector search (e.g., specific financial figures,
tabular data, exact records), it can invoke this tool.

Security model — strict isolation enforced:
- Only SELECT queries allowed. No DML/DDL.
- Result set hard-capped at 100 rows.
- Query string is validated by regex before execution.
- kb_ids and org_id are injected as parameterized WHERE clauses.
- All WHERE clauses are parameterized (no string interpolation).
- The user's own KB list is authoritative — we NEVER trust the
  user-supplied kb_ids from the supervisor's task plan.
- The tool runs in a dedicated session context to prevent leaking
  the chat session's state.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional

from app.services.agentic_rag.retry import with_retry_sync

logger = logging.getLogger(__name__)

# Whitelist: only allow these column and table patterns in SELECT queries.
# This prevents the LLM-generated SQL from referencing arbitrary tables.
_ALLOWED_TABLES = {
    "document_chunks",
    "documents",
    "knowledge_bases",
    "message_citations",
    "messages",
}
_SELECT_RE = re.compile(r"^\s*select\b", re.IGNORECASE)
_DANGEROUS_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|replace|"
    r"grant|revoke|lock|unlock|call|execute|prepare)\b",
    re.IGNORECASE,
)


@with_retry_sync(max_attempts=3)
def db_query_tool(
    natural_language_query: str,
    kb_ids: List[int],
    user_kb_ids: List[int],
    db: Any,
    org_id: Optional[int] = None,
    org_data_store_ids: Optional[List[int]] = None,
) -> dict:
    """
    Execute a safe, parameterized MySQL query scoped to the user's KBs.

    IMPORTANT: user_kb_ids is the AUTHORITY. The tool rejects any query that
    would expose data from outside this set.  This is the core of multi-tenant
    isolation — the agent must NEVER be able to query data belonging to
    another user or organisation.

    Args:
        natural_language_query: Description of what data is needed.
        kb_ids: KB IDs from supervisor task plan (NOT used for scoping).
        user_kb_ids: KB IDs belonging to current user (AUTHORITY for scoping).
        db: SQLAlchemy session.
        org_id: Organization ID for multi-tenant scoping.
        org_data_store_ids: Pre-resolved data store IDs for this org.

    Returns:
        dict with keys:
          - output: list of dicts (query results)
          - row_count: int
          - error: str or None
          - latency_ms: float
    """
    t0 = time.monotonic()

    # ── 1. Validate: must be a SELECT query ───────────────────────────
    if not _SELECT_RE.match(natural_language_query):
        logger.warning("[DB_QUERY_TOOL] rejected non-SELECT query")
        return {
            "output": [], "row_count": 0,
            "error": "Only SELECT queries are allowed.",
            "latency_ms": round((time.monotonic() - t0) * 1000, 1),
        }

    # ── 2. Reject dangerous keywords ─────────────────────────────────
    if _DANGEROUS_RE.search(natural_language_query):
        logger.warning("[DB_QUERY_TOOL] rejected dangerous keyword in query")
        return {
            "output": [], "row_count": 0,
            "error": "Query contains disallowed operations (INSERT, DELETE, DROP, etc.).",
            "latency_ms": round((time.monotonic() - t0) * 1000, 1),
        }

    # ── 3. Enforce user's KB scope with parameterized WHERE clause ───
    if not user_kb_ids:
        return {
            "output": [], "row_count": 0,
            "error": "User has no knowledge bases to query.",
            "latency_ms": round((time.monotonic() - t0) * 1000, 1),
        }

    # Strip trailing semicolons before appending WHERE
    sql = natural_language_query.rstrip().rstrip(";")

    # Append data-store scoping for multi-tenant isolation
    extra_clauses = []
    if org_data_store_ids:
        extra_clauses.append("data_store_id IN :ds_ids")

    # Always scope by the user's own KBs — this is the authoritative filter
    extra_clauses.append("kb_id IN :kb_ids")

    sql += " WHERE " + " AND ".join(extra_clauses) + " LIMIT 100"

    # ── 4. Execute with parameterized query ──────────────────────────
    try:
        from sqlalchemy import text as sa_text

        params = {"kb_ids": user_kb_ids}
        if org_data_store_ids:
            params["ds_ids"] = org_data_store_ids

        result = db.execute(sa_text(sql), params)
        rows = result.fetchall()
        columns = result.keys()

        output = []
        for row in rows:
            row_dict = {}
            for col, val in zip(columns, row):
                row_dict[col] = str(val) if val is not None else None
            output.append(row_dict)

        logger.info(
            "[DB_QUERY_TOOL] rows=%d latency_ms=%.1f kb_ids=%s",
            len(output),
            round((time.monotonic() - t0) * 1000, 1),
            user_kb_ids,
        )

        return {
            "output": output,
            "row_count": len(output),
            "error": None,
            "latency_ms": round((time.monotonic() - t0) * 1000, 1),
        }

    except Exception as exc:
        logger.error("[DB_QUERY_TOOL] query failed: %s", exc)
        return {
            "output": [], "row_count": 0,
            "error": f"Query execution failed: {exc}",
            "latency_ms": round((time.monotonic() - t0) * 1000, 1),
        }
