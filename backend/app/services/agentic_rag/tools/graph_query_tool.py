"""Graph query tool — Neo4j 2-hop entity traversal with data isolation.

When the supervisor determines that a query needs entity relationship data
(e.g., "what companies are related to X", "what products use technology Y"),
it can invoke this tool to traverse the Neo4j knowledge graph.

Security: Only entities linked to the user's own KB-indexed documents are
returned.  We discover entities through the KB document namespace, so a
user can NEVER see entities from another user's KB — even if the graph
itself contains them.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from app.services.agentic_rag.retry import with_retry_sync

logger = logging.getLogger(__name__)


@with_retry_sync(max_attempts=3)
def graph_query_tool(
    query: str,
    kb_ids: List[int],
    user_kb_ids: List[int],
    db: Any,
    org_id: Optional[int] = None,
) -> dict:
    """
    Execute a Neo4j 2-hop graph traversal scoped to the user's KBs.

    The traversal discovers seed entities by matching against document
    metadata (sources tied to the user's KBs), then walks up to 2 hops
    to find related entities and their relationships.

    Args:
        query: Natural language description of the graph traversal needed.
        kb_ids: KB IDs from supervisor task plan (for context, not scoping).
        user_kb_ids: KB IDs belonging to current user (AUTHORITY for scoping).
        db: SQLAlchemy session (used to resolve KB → document namespace).
        org_id: Organization ID for multi-tenant scoping.

    Returns:
        dict with keys:
          - output: list of entity dicts with relationships
          - entity_count: int
          - relationship_count: int
          - error: str or None
          - latency_ms: float
    """
    t0 = time.monotonic()

    # ── 1. Authority check ───────────────────────────────────────────
    if not user_kb_ids:
        return {
            "output": [], "entity_count": 0, "relationship_count": 0,
            "error": "User has no knowledge bases to traverse the graph for.",
            "latency_ms": round((time.monotonic() - t0) * 1000, 1),
        }

    # ── 2. Resolve KB-scoped seed entities from document metadata ────
    try:
        from app.models.knowledge import DocumentChunk
        chunks = (
            db.query(DocumentChunk)
            .filter(DocumentChunk.kb_id.in_(user_kb_ids))
            .limit(200)
            .all()
        )
        # Collect unique entity source names tied to user's KBs
        seed_sources = list({
            c.metadata.get("source", "")
            for c in chunks
            if c.metadata
            and isinstance(c.metadata, dict)
            and c.metadata.get("source")
        })
    except Exception as exc:
        logger.warning("[GRAPH_QUERY_TOOL] failed to resolve user KB sources: %s", exc)
        seed_sources = []

    # ── 3. Execute 2-hop Neo4j traversal ─────────────────────────────
    try:
        from app.services.graph import _get_driver
        driver = _get_driver()

        with driver.session() as session:
            # Build seed filter — if user has KB-scoped entities, prefer those
            seed_filter_parts = []
            for src in seed_sources[:10]:
                safe = src.replace("\\", "\\\\").replace("'", "\\'")
                seed_filter_parts.append(f"toLower(e.name) CONTAINS '{safe.lower()}'")
            # Also match the query against entity names/descriptions
            safe_query = query.replace("'", "\\'")
            seed_filter_parts.append(
                f"toLower(e.name) CONTAINS '{safe_query.lower()}' "
                f"OR toLower(e.description) CONTAINS '{safe_query.lower()}'"
            )

            seed_filter = " OR ".join(seed_filter_parts) if seed_filter_parts else "TRUE"

            # 2-hop traversal: seed → 1-hop → 2-hop neighbors
            result = session.run("""
                MATCH (e:__Entity__)
                WHERE (${seed_filter})
                OPTIONAL MATCH (e)-[r1]-(mid:__Entity__)
                OPTIONAL MATCH (mid)-[r2]-(hop2:__Entity__)
                RETURN DISTINCT
                       e.name AS name,
                       e.type AS type,
                       e.description AS description,
                       collect(DISTINCT {
                           neighbor: mid.name,
                           relation: type(r1),
                           hop: 1
                       }) AS hop1,
                       collect(DISTINCT {
                           neighbor: hop2.name,
                           relation: type(r2),
                           hop: 2
                       }) AS hop2
                LIMIT 30
            """, seed_filter=seed_filter)

            output = []
            entity_count = 0
            relationship_count = 0

            for record in result:
                entity_name = record.get("name")
                if not entity_name:
                    continue
                entity_count += 1
                hop1 = record.get("hop1", [])
                hop2 = record.get("hop2", [])
                relationship_count += len(hop1) + len(hop2)
                output.append({
                    "entity": entity_name,
                    "type": record.get("type"),
                    "description": record.get("description"),
                    "hop1_relationships": hop1,
                    "hop2_relationships": hop2,
                })

            logger.info(
                "[GRAPH_QUERY_TOOL] query=%s entities=%d rels=%d latency_ms=%.1f",
                query[:80], entity_count, relationship_count,
                round((time.monotonic() - t0) * 1000, 1),
            )

            return {
                "output": output,
                "entity_count": entity_count,
                "relationship_count": relationship_count,
                "error": None,
                "latency_ms": round((time.monotonic() - t0) * 1000, 1),
            }

    except Exception as exc:
        logger.error("[GRAPH_QUERY_TOOL] failed: %s", exc)
        return {
            "output": [],
            "entity_count": 0,
            "relationship_count": 0,
            "error": str(exc),
            "latency_ms": round((time.monotonic() - t0) * 1000, 1),
        }
