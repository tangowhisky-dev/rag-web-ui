"""Graph-expanded retrieval and entity context enrichment.

Two retrieval-time operations that use Neo4j graph topology to
improve search results beyond pure vector similarity:

  expand_docs_via_graph()
    Takes already-retrieved docs (from vector search + RRF), extracts
    their Qdrant point UUIDs, traverses Neo4j (chunk → entity → entity
    → chunk) to find RELATED chunks not in the original result set,
    fetches those chunks from Qdrant by UUID, and returns them as
    additional LangchainDocument objects to merge into the candidate
    pool BEFORE reranking.

  enrich_docs_with_graph()
    For the final top-k docs: append entity relationship triples as
    [Graph context] text so the LLM sees the graph alongside the chunk.
    Non-fatal per doc.

Supporting helpers:
  _build_graph_scope_filter()       — entity scope filter for traversal
  _build_traversal_patterns()       — Cypher path patterns for N-hop traversal
  _traverse_graph_for_expansion()   — executes the Neo4j traversal
  _fetch_expanded_docs_from_qdrant() — fetches chunk text from Qdrant by UUID
"""

import logging
from typing import Optional

import neo4j
from langchain_core.documents import Document as LangchainDocument
from sqlalchemy.orm import Session

from app.core.config import settings
from app.services.agentic_rag.retry import with_retry_sync
from app.services.settings_service import get_setting

from .setup import _get_driver
from .build import _extract_seen_point_ids

logger = logging.getLogger(__name__)


def _build_graph_scope_filter(
    kb_ids: list[int],
    datastore_ids: Optional[list[int]],
) -> tuple[list[str], list[str], str]:
    """Build entity scope filter for graph traversal.

    Ensures graph traversal stays within the queried KB(s)/datastore(s)
    and doesn't cross-contaminate via shared entity nodes from other KBs.
    Entities without scope props are from older ingestion runs —
    include them for backward compatibility.
    """
    kb_scope = [str(k) for k in kb_ids] if kb_ids else []
    ds_scope = [str(d) for d in datastore_ids] if datastore_ids else []
    scope_clauses = []
    if kb_scope:
        scope_clauses.append("e.kb_id IN $kb_scope")
    if ds_scope:
        scope_clauses.append("e.data_store_id IN $ds_scope")
    if scope_clauses:
        scope_filter = "(" + " OR ".join(scope_clauses) + " OR e.kb_id IS NULL AND e.data_store_id IS NULL)"
    else:
        scope_filter = "true"
    return kb_scope, ds_scope, scope_filter


def _build_traversal_patterns(hops_val: int) -> tuple[str, str]:
    """Build Cypher path pattern and intermediate entity filter for N-hop traversal.

    1 hop: (e)-[:FROM_CHUNK]->(c2)
    2 hops: (e)-[r1]-(e2)-[:FROM_CHUNK]->(c2)
    N hops: chain of N entity nodes with N-1 relationships
    """
    hops = max(1, hops_val)
    if hops == 1:
        rest_pattern = "(e)-[:FROM_CHUNK]->(c2)"
    else:
        parts = ["(e)"]
        for i in range(2, hops + 1):
            parts.append(f"-[r{i - 1}]-(e{i})")
        parts.append("-[:FROM_CHUNK]->(c2)")
        rest_pattern = " ".join(parts)

    if hops > 1:
        interm_clauses = []
        for i in range(2, hops + 1):
            interm_clauses.append(f"(e{i}.kb_id IN $kb_scope OR e{i}.data_store_id IN $ds_scope OR (e{i}.kb_id IS NULL AND e{i}.data_store_id IS NULL))")
        interm_filter = " AND ".join(interm_clauses)
    else:
        interm_filter = "true"
    return rest_pattern, interm_filter


def _traverse_graph_for_expansion(
    driver: neo4j.Driver,
    seen_point_ids: set[str],
    collections: list[str],
    scope_filter: str,
    rest_pattern: str,
    interm_filter: str,
    fanout_val: int,
    limit_val: int,
    kb_scope: list[str],
    ds_scope: list[str],
) -> list[tuple[str, str]]:
    """Traverse from seed chunks via entity relationships to connected chunks.

    The first hop's distinct entities are capped (GRAPHRAG_ENTITY_FANOUT_CAP)
    before expanding further, so a handful of highly-connected "hub"
    entities (e.g. generic terms shared by hundreds of chunks) can't blow
    up the traversal into a combinatorial cross product.
    Return qdrant_point_id + qdrant_collection of chunks NOT already seen.
    """
    with driver.session() as session:
        result = session.run(
            f"""
            MATCH (c:Chunk)
            WHERE c.qdrant_point_id IN $seen_ids
              AND c.qdrant_collection IN $collections
            MATCH (c)<-[:FROM_CHUNK]-(e)
            WHERE {scope_filter}
            WITH DISTINCT e LIMIT $entity_cap
            MATCH {rest_pattern}
            WHERE c2.qdrant_point_id IS NOT NULL
              AND NOT c2.qdrant_point_id IN $seen_ids
              AND c2.qdrant_collection IN $collections
              AND {interm_filter}
            RETURN DISTINCT c2.qdrant_point_id AS point_id,
                            c2.qdrant_collection AS collection
            LIMIT $limit
            """,
            seen_ids=list(seen_point_ids),
            collections=collections,
            entity_cap=max(1, fanout_val),
            limit=max(1, limit_val),
            kb_scope=kb_scope,
            ds_scope=ds_scope,
        )
        return [
            (rec["point_id"], rec["collection"]) for rec in result
        ]


def _fetch_expanded_docs_from_qdrant(
    expansion_targets: list[tuple[str, str]],
) -> list[LangchainDocument]:
    """Fetch chunk text from Qdrant by point UUID and build LangchainDocuments.

    Groups targets by collection and fetches text/payload only (no re-embedding).
    """
    from collections import defaultdict
    from qdrant_client import QdrantClient

    by_collection: dict[str, list[str]] = defaultdict(list)
    for point_id, collection in expansion_targets:
        by_collection[collection].append(point_id)

    qdrant = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    expanded_docs: list[LangchainDocument] = []

    for collection, point_ids in by_collection.items():
        try:
            points = qdrant.retrieve(
                collection_name=collection,
                ids=point_ids,
                with_payload=True,
                with_vectors=False,   # text only — Qdrant is the source of truth
            )
            for pt in points:
                payload = pt.payload or {}
                chunk_text = payload.get("chunk_text", "")
                if not chunk_text:
                    continue
                meta = {k: v for k, v in payload.items() if k != "chunk_text"}
                meta["_graph_expanded"] = True
                meta["qdrant_point_id"] = str(pt.id)
                expanded_docs.append(
                    LangchainDocument(page_content=chunk_text, metadata=meta)
                )
        except Exception as exc:
            logger.warning(
                "GraphService.expand: Qdrant retrieve failed for collection %s: %s",
                collection, exc,
            )
    return expanded_docs


@with_retry_sync(max_attempts=3)
def expand_docs_via_graph(
    docs: list[LangchainDocument],
    kb_ids: list[int],
    db: Optional[Session] = None,
    org_id: Optional[int] = None,
    datastore_ids: Optional[list[int]] = None,
) -> list[LangchainDocument]:
    """
    Graph-expanded retrieval: find additional chunks via Neo4j graph traversal
    and fetch their text from Qdrant.

    Flow:
      1. Extract Qdrant point UUIDs from the already-retrieved docs.
      2. Query Neo4j: traverse chunk → entity → entity → chunk to find
         entity-connected chunks whose qdrant_point_id is NOT already in
         the current result set.
      3. Fetch those new points from Qdrant by UUID (text/payload only,
         no vector computation needed).
      4. Return them as LangchainDocument objects with metadata flag
         `_graph_expanded=True` so the caller can annotate them.

    This surfaces chunks that are SEMANTICALLY linked via entity relationships
    but would not have been returned by vector similarity alone.

    When db and org_id are provided, org-overridable settings (hops, limit,
    fanout) are resolved via the settings service.

    Non-fatal — returns [] on any failure so the caller's pipeline continues
    with only the original vector search results.
    """
    if not get_setting(db, "GRAPHRAG_ENABLED", None) or not docs:
        return []

    from qdrant_client import QdrantClient

    # Resolve org-overridable settings
    hops_val = get_setting(db, "GRAPHRAG_RETRIEVAL_HOPS", org_id)
    fanout_val = get_setting(db, "GRAPHRAG_ENTITY_FANOUT_CAP", org_id)
    limit_val = get_setting(db, "GRAPHRAG_RETRIEVAL_LIMIT", org_id)

    seen_point_ids = _extract_seen_point_ids(docs)

    if not seen_point_ids:
        # Docs came from before the qdrant_point_id payload field was added —
        # fall back gracefully rather than blowing up.
        logger.debug("GraphService.expand: no qdrant_point_id in doc metadata, skipping expansion")
        return []

    try:
        driver = _get_driver()
        collections = [f"kb_{kb_id}" for kb_id in kb_ids]
        if datastore_ids:
            collections += [f"ds_{ds_id}" for ds_id in datastore_ids]

        kb_scope, ds_scope, scope_filter = _build_graph_scope_filter(kb_ids, datastore_ids)
        rest_pattern, interm_filter = _build_traversal_patterns(hops_val)

        expansion_targets = _traverse_graph_for_expansion(
            driver, seen_point_ids, collections, scope_filter,
            rest_pattern, interm_filter, fanout_val, limit_val,
            kb_scope, ds_scope,
        )
        if not expansion_targets:
            logger.debug("GraphService.expand: no graph-connected chunks found beyond current result set")
            return []

        logger.info(
            "GraphService.expand: found %d graph-connected chunks to fetch from Qdrant",
            len(expansion_targets),
        )

        expanded_docs = _fetch_expanded_docs_from_qdrant(expansion_targets)

        logger.info(
            "GraphService.expand: fetched %d graph-expanded docs from Qdrant",
            len(expanded_docs),
        )
        return expanded_docs

    except Exception as exc:
        logger.warning("GraphService.expand: expansion failed (non-fatal): %s", exc)
        return []


# ── Retrieval: entity context enrichment ──────────────────────────────────────

@with_retry_sync(max_attempts=3)
def enrich_docs_with_graph(
    docs: list[LangchainDocument],
    db: Optional[Session] = None,
    org_id: Optional[int] = None,
) -> list[LangchainDocument]:
    """
    Append [Graph context] entity relationship triples to each doc's text.

    Looks up each Chunk in Neo4j by qdrant_point_id (primary key), traverses
    FROM_CHUNK edges to Entity nodes, collects entity-to-entity relationship
    triples, and appends them as a compact text block.

    This gives the LLM explicit relationship context alongside the chunk text.
    Called AFTER expansion+reranking so only the final top-k docs are enriched.

    Falls back to (document_id, chunk_index) lookup for docs that predate the
    qdrant_point_id field being added to Qdrant payloads.

    Non-fatal per doc — failures return the doc unchanged.
    """
    if not get_setting(db, "GRAPHRAG_ENABLED", None) or not docs:
        return docs

    driver = _get_driver()
    enriched = []
    graph_hits = 0

    for doc in docs:
        point_id  = doc.metadata.get("qdrant_point_id")
        doc_id    = doc.metadata.get("document_id")
        chunk_idx = doc.metadata.get("chunk_index")

        try:
            with driver.session() as session:
                if point_id:
                    # Primary path: lookup by Qdrant point UUID (O(1) index hit).
                    # Neighbor traversal is scoped to entities that also have a
                    # FROM_CHUNK edge into this same Qdrant collection — prevents
                    # stale inter-entity edges from deleted KBs bleeding in.
                    result = session.run(
                        """
                        MATCH (c:Chunk {qdrant_point_id: $point_id})
                        OPTIONAL MATCH (e)-[:FROM_CHUNK]->(c)
                        OPTIONAL MATCH (e)-[r]-(neighbor)
                        WHERE EXISTS { MATCH (neighbor)-[:FROM_CHUNK]->(:Chunk {qdrant_collection: c.qdrant_collection}) }
                        WITH e.name AS ename, type(r) AS rel, neighbor.name AS nname
                        WHERE ename IS NOT NULL AND nname IS NOT NULL
                        RETURN collect(DISTINCT [ename, rel, nname])[..40] AS triples
                        """,
                        point_id=point_id,
                    )
                elif doc_id is not None and chunk_idx is not None:
                    # Fallback: legacy lookup by (document_id, chunk_index).
                    # No collection scoping available — best-effort.
                    result = session.run(
                        """
                        MATCH (c:Chunk {document_id: $document_id, chunk_index: $chunk_index})
                        OPTIONAL MATCH (e)-[:FROM_CHUNK]->(c)
                        OPTIONAL MATCH (e)-[r]-(neighbor)
                        WITH e.name AS ename, type(r) AS rel, neighbor.name AS nname
                        WHERE ename IS NOT NULL AND nname IS NOT NULL
                        RETURN collect(DISTINCT [ename, rel, nname])[..40] AS triples
                        """,
                        document_id=str(doc_id),
                        chunk_index=int(chunk_idx),
                    )
                else:
                    enriched.append(doc)
                    continue

                record = result.single()
                triples = record["triples"] if record else []

            if triples:
                graph_ctx = "\n[Graph context]\n" + "".join(
                    f"{t[0]} -[{t[1]}]-> {t[2]}\n" for t in triples
                )
                enriched_doc = LangchainDocument(
                    page_content=doc.page_content + graph_ctx,
                    metadata={**doc.metadata, "_graph_triples": len(triples)},
                )
                graph_hits += 1
            else:
                enriched_doc = doc

        except Exception as exc:
            logger.warning(
                "GraphService.enrich: failed for point_id=%s doc_id=%s chunk=%s: %s",
                point_id, doc_id, chunk_idx, exc,
            )
            enriched_doc = doc

        enriched.append(enriched_doc)

    logger.info(
        "GraphService.enrich: %d/%d docs enriched with entity triples",
        graph_hits, len(docs),
    )
    return enriched
