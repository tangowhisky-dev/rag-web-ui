"""
Neo4j knowledge graph service.

Architecture
------------
Strict separation of concerns:
  Qdrant  — source of truth for all chunk TEXT and VECTORS
  Neo4j   — source of truth for GRAPH TOPOLOGY (entities, relationships, chunk linkage)

Cross-reference
---------------
The link between the two systems is the Qdrant point UUID, stored as
`qdrant_point_id` on every Chunk node in Neo4j. This enables:

  1. Graph-expanded retrieval — vector search returns point UUIDs → traverse
     Neo4j to find entity-connected Chunk nodes NOT in the original result set
     → fetch those Chunks from Qdrant by UUID → merge into candidate pool.

  2. Entity context enrichment — for the final top-k docs, traverse FROM_CHUNK
     edges to Entity nodes, collect relationship triples, append a [Graph context]
     block to each chunk's text so the LLM sees the entity graph.

No vectors are stored in Neo4j. All text retrieval goes through Qdrant. Neo4j
is only traversed for graph topology.

Extraction backends
-------------------
  ReLiK  (default, GRAPHRAG_LLM unset):
    Calls the dockerized ReLiK service (GET /api/relik).
    Fast, local, deterministic, zero API cost.

  LLM pipeline  (GRAPHRAG_LLM=<model_name>):
    neo4j-graphrag Pipeline + LLMEntityRelationExtractor with
    use_structured_output=True — JSON Schema-constrained output, no free-form
    JSON drift.

Ingestion (build_graph_for_document)
-------------------------------------
  1. Write Chunk nodes keyed by qdrant_point_id (the UUID Qdrant uses as point ID).
     Also carry document_id + chunk_index for backward-compatible lookups.
  2. Run extraction → write Entity nodes + typed relationship edges.
  3. Link Entity nodes to Chunk nodes via FROM_CHUNK edges.

Retrieval
---------
  expand_docs_via_graph(docs, kb_ids)
    Takes already-retrieved docs (from vector search + RRF).
    Extracts their Qdrant point IDs → traverses Neo4j graph
    (chunk → entity → entity → chunk) to find RELATED chunks.
    Fetches those related chunks from Qdrant by UUID.
    Returns them as additional LangchainDocument objects to merge into
    the candidate pool BEFORE reranking.

  enrich_docs_with_graph(docs)
    For the final top-k docs: append entity relationship triples as
    [Graph context] text so the LLM sees the graph alongside the chunk.
    Non-fatal per doc.

Deletion
--------
  delete_graph_for_document / delete_graph_for_kb
  Always run regardless of GRAPHRAG_ENABLED — data may exist from a prior
  ingest run. Also cleans up orphaned Entity nodes with no remaining chunk links.
"""

import logging
import uuid
from typing import Optional

import httpx
import neo4j
from langchain_core.documents import Document as LangchainDocument

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Module-level singletons (lazy) ────────────────────────────────────────────
_neo4j_driver: Optional[neo4j.Driver] = None
_llm_pipeline = None   # neo4j_graphrag Pipeline — only built when GRAPHRAG_LLM is set


def _get_driver() -> neo4j.Driver:
    global _neo4j_driver
    if _neo4j_driver is None:
        _neo4j_driver = neo4j.GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
    return _neo4j_driver


def _chunk_id_to_point_id(chunk_id: str) -> str:
    """
    Convert a SHA-256 hex chunk ID to the deterministic UUID Qdrant uses.

    Mirrors document_processor._chunk_id_to_point_id so the UUID written
    to Neo4j is always the exact UUID stored as the Qdrant point ID.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_OID, chunk_id))


# ── ReLiK extraction path ─────────────────────────────────────────────────────

async def _call_relik(text: str) -> dict:
    """
    Call the dockerized ReLiK service.
    GET /api/relik?text=...&annotation_type=char&relation_threshold=0.5
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(
            f"{settings.RELIK_URL}/api/relik",
            params={"text": text, "annotation_type": "char", "relation_threshold": 0.5},
        )
        resp.raise_for_status()
        return resp.json()


async def _extract_with_relik(
    session: neo4j.Session,
    document_id: int,
    chunks: list[str],
    qdrant_point_ids: list[str],
) -> tuple[int, int]:
    """
    Run ReLiK on each chunk, write Entity nodes + typed relationships to Neo4j.
    Links Entity nodes to Chunk nodes via FROM_CHUNK edges using qdrant_point_id.
    Returns (total_entities, total_relations).
    """
    total_entities = 0
    total_relations = 0

    for idx, (text, point_id) in enumerate(zip(chunks, qdrant_point_ids)):
        try:
            out = await _call_relik(text)
        except Exception as exc:
            logger.warning(
                "GraphService[relik]: call failed for doc %d chunk %d: %s",
                document_id, idx, exc,
            )
            continue

        # spans → Entity nodes + FROM_CHUNK edges
        for span in out.get("spans", []):
            entity_name = (span.get("text") or "").strip()
            entity_type = span.get("label") or "Entity"
            if entity_type == "--NME--":
                entity_type = "Entity"
            if not entity_name:
                continue
            session.run(
                """
                MERGE (e:Entity {name: $name})
                SET e.type = $type
                WITH e
                MATCH (c:Chunk {qdrant_point_id: $point_id})
                MERGE (e)-[:FROM_CHUNK]->(c)
                """,
                name=entity_name,
                type=entity_type,
                point_id=point_id,
            )
            total_entities += 1

        # triplets → typed relationship edges between Entity pairs
        for triplet in out.get("triplets", []):
            subj = (triplet.get("subject", {}).get("text") or "").strip()
            obj  = (triplet.get("object",  {}).get("text") or "").strip()
            rel  = triplet.get("relation", {}).get("label") or "RELATED_TO"
            rel  = rel.upper().replace(" ", "_")
            if not subj or not obj:
                continue
            rel_safe = "".join(c if c.isalnum() or c == "_" else "_" for c in rel)
            session.run(
                f"""
                MERGE (a:Entity {{name: $subj}})
                MERGE (b:Entity {{name: $obj}})
                MERGE (a)-[:`{rel_safe}`]->(b)
                """,
                subj=subj,
                obj=obj,
            )
            total_relations += 1

    return total_entities, total_relations


# ── LLM pipeline path ─────────────────────────────────────────────────────────

def _get_llm_pipeline():
    """
    Build (once) a neo4j-graphrag Pipeline with LLMEntityRelationExtractor.

    Pipeline topology:
      extractor (LLMEntityRelationExtractor)
          └─[graph: Neo4jGraph]──► writer (Neo4jWriter)

    use_structured_output=True makes the LLM return a JSON Schema-validated
    Neo4jGraph object — no regex post-processing, no json.loads guesswork.
    The LLM must support response_format=json_schema (OpenAI GPT-4 class and
    most modern OpenAI-compatible endpoints do).
    """
    global _llm_pipeline
    if _llm_pipeline is not None:
        return _llm_pipeline

    from neo4j_graphrag.llm import OpenAILLM
    from neo4j_graphrag.experimental.components.entity_relation_extractor import (
        LLMEntityRelationExtractor,
        OnError,
    )
    from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
    from neo4j_graphrag.experimental.pipeline.pipeline import Pipeline

    llm = OpenAILLM(
        model_name=settings.GRAPHRAG_LLM,
        model_params={"temperature": 0},
        base_url=settings.OPENAI_API_BASE,
        api_key=settings.OPENAI_API_KEY,
    )

    extractor = LLMEntityRelationExtractor(
        llm=llm,
        use_structured_output=True,
        on_error=OnError.IGNORE,
        create_lexical_graph=False,
        max_concurrency=3,
    )

    writer = Neo4jWriter(
        driver=_get_driver(),
        neo4j_database="neo4j",
        batch_size=500,
    )

    pipe = Pipeline()
    pipe.add_component(extractor, "extractor")
    pipe.add_component(writer, "writer")
    pipe.connect("extractor", "writer", input_config={"graph": "extractor"})

    _llm_pipeline = pipe
    logger.info("GraphService[llm]: pipeline built with model=%s", settings.GRAPHRAG_LLM)
    return _llm_pipeline


async def _extract_with_llm(
    document_id: int,
    file_name: str,
    chunks: list[str],
    qdrant_point_ids: list[str],
) -> tuple[int, int]:
    """
    Run neo4j-graphrag LLM pipeline per chunk.

    After the pipeline writes Entity nodes, an additional pass links them to
    our Chunk nodes via FROM_CHUNK edges using qdrant_point_id — consistent
    with the ReLiK path so retrieval expansion works the same way regardless
    of which extraction backend was used.
    """
    from neo4j_graphrag.experimental.components.types import TextChunks, TextChunk

    pipe = _get_llm_pipeline()
    driver = _get_driver()
    total_nodes = 0  # overridden by post-loop Neo4j query; kept for early-return path
    total_rels = 0

    for idx, (text, point_id) in enumerate(zip(chunks, qdrant_point_ids)):
        try:
            pipe_result = await pipe.run({
                "extractor": {
                    "chunks": TextChunks(chunks=[TextChunk(text=text, index=idx)]),
                    "examples": "",
                },
                "writer": {},
            })
            # PipelineResult.result is a dict keyed by component name.
            # The writer output is KGWriterModel(status="SUCCESS"|"FAILURE").
            # Any other shape (None, unexpected) — we still attempt the
            # FROM_CHUNK linking because the Cypher may have run regardless.
            writer_output = None
            raw = getattr(pipe_result, "result", None)
            if isinstance(raw, dict):
                writer_output = raw.get("writer")
            elif hasattr(raw, "status"):
                writer_output = raw  # direct KGWriterModel (future-proofing)

            status = getattr(writer_output, "status", None)
            if status == "FAILURE":
                logger.warning(
                    "GraphService[llm]: writer returned FAILURE for doc %d chunk %d — skipping FROM_CHUNK link",
                    document_id, idx,
                )
                continue
            # status == "SUCCESS" or None (unknown shape) — proceed either way
            # since Neo4j Cypher already ran if no exception was raised.

            # Link entities written by this chunk run to our Chunk node.
            # Use __Entity__ — neo4j-graphrag's internal label convention.
            # Only link entities not already connected to this specific chunk
            # (idempotent on re-ingest).
            with driver.session() as session:
                rec = session.run(
                    """
                    MATCH (e:__Entity__)
                    WHERE NOT EXISTS {
                        MATCH (e)-[:FROM_CHUNK]->(:Chunk {qdrant_point_id: $point_id})
                    }
                    WITH e LIMIT 500
                    MATCH (c:Chunk {qdrant_point_id: $point_id})
                    MERGE (e)-[:FROM_CHUNK]->(c)
                    RETURN count(e) AS linked
                    """,
                    point_id=point_id,
                ).single()
                total_nodes += rec["linked"] if rec else 0

            meta = getattr(pipe_result, "metadata", {}) or {}
            # relationship_count is not reliably present on PipelineResult;
            # we do a single accurate query at the end instead.

        except Exception as exc:
            logger.warning(
                "GraphService[llm]: pipeline failed for doc %d chunk %d: %s",
                document_id, idx, exc,
            )
            continue

    # Query actual counts from Neo4j scoped to this document.
    # total_nodes = unique __Entity__ nodes linked to any of this doc's chunks.
    # total_rels  = relationships between those entities.
    with driver.session() as session:
        rec = session.run(
            """
            MATCH (c:Chunk {document_id: $doc_id})<-[:FROM_CHUNK]-(e)
            WITH collect(DISTINCT e) AS entities
            UNWIND entities AS e
            OPTIONAL MATCH (e)-[r]-(e2)
            WHERE e2 IN entities
            RETURN size(entities) AS node_count,
                   count(r)      AS rel_count
            """,
            doc_id=str(document_id),
        ).single()
        total_nodes = rec["node_count"] if rec else 0
        total_rels  = rec["rel_count"]  if rec else 0

    return total_nodes, total_rels


# ── Ingest ─────────────────────────────────────────────────────────────────────

async def build_graph_for_document(
    kb_id: int,
    document_id: int,
    file_name: str,
    chunks: list[str],
    chunk_ids: list[str],
) -> None:
    """
    Extract entity/relationship graph from document chunks and store in Neo4j.

    chunk_ids are the SHA-256 hex strings from document_processor — we convert
    them to the actual Qdrant point UUIDs here via the same deterministic
    uuid5 transform, so qdrant_point_id in Neo4j always matches the Qdrant
    point ID exactly.

    Chunk nodes carry:
      qdrant_point_id    — primary cross-reference key (UUID string)
      qdrant_collection  — which Qdrant collection (kb_<kb_id>)
      document_id        — for deletion queries
      chunk_index        — ordinal position within the document
      kb_id              — knowledge base
      file_name          — human-readable source
    """
    if not settings.GRAPHRAG_ENABLED:
        return

    driver = _get_driver()
    backend = "llm" if settings.GRAPHRAG_LLM else "relik"

    # Convert SHA-256 chunk IDs → the actual UUIDs Qdrant uses as point IDs
    qdrant_point_ids = [_chunk_id_to_point_id(cid) for cid in chunk_ids]
    collection_name = f"kb_{kb_id}"

    logger.info(
        "GraphService[%s]: writing %d Chunk nodes for doc %d (kb=%d)",
        backend, len(chunks), document_id, kb_id,
    )

    with driver.session() as session:
        for idx, (text, point_id) in enumerate(zip(chunks, qdrant_point_ids)):
            session.run(
                """
                MERGE (c:Chunk {qdrant_point_id: $point_id})
                SET c.qdrant_collection = $collection,
                    c.document_id       = $document_id,
                    c.chunk_index       = $chunk_index,
                    c.kb_id             = $kb_id,
                    c.file_name         = $file_name
                """,
                point_id=point_id,
                collection=collection_name,
                document_id=str(document_id),
                chunk_index=idx,
                kb_id=str(kb_id),
                file_name=file_name,
            )

    # Extract entities and relationships
    if settings.GRAPHRAG_LLM:
        total_entities, total_relations = await _extract_with_llm(
            document_id=document_id,
            file_name=file_name,
            chunks=chunks,
            qdrant_point_ids=qdrant_point_ids,
        )
    else:
        with driver.session() as session:
            total_entities, total_relations = await _extract_with_relik(
                session=session,
                document_id=document_id,
                chunks=chunks,
                qdrant_point_ids=qdrant_point_ids,
            )

    logger.info(
        "GraphService[%s]: doc %d — %d entities, %d relations written to Neo4j",
        backend, document_id, total_entities, total_relations,
    )


# ── Retrieval: graph expansion ─────────────────────────────────────────────────

def expand_docs_via_graph(
    docs: list[LangchainDocument],
    kb_ids: list[int],
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

    Non-fatal — returns [] on any failure so the caller's pipeline continues
    with only the original vector search results.
    """
    if not settings.GRAPHRAG_ENABLED or not docs:
        return []

    from qdrant_client import QdrantClient

    # Extract the Qdrant point UUIDs from the retrieved docs
    seen_point_ids = set()
    for doc in docs:
        pid = doc.metadata.get("qdrant_point_id")
        if pid:
            seen_point_ids.add(pid)

    if not seen_point_ids:
        # Docs came from before the qdrant_point_id payload field was added —
        # fall back gracefully rather than blowing up.
        logger.debug("GraphService.expand: no qdrant_point_id in doc metadata, skipping expansion")
        return []

    try:
        driver = _get_driver()
        collections = [f"kb_{kb_id}" for kb_id in kb_ids]

        # Traverse the graph: found_chunk → entity → entity → connected_chunk
        # Return qdrant_point_id + qdrant_collection of chunks NOT already seen.
        with driver.session() as session:
            result = session.run(
                """
                MATCH (c:Chunk)
                WHERE c.qdrant_point_id IN $seen_ids
                  AND c.qdrant_collection IN $collections
                MATCH (c)<-[:FROM_CHUNK]-(e)-[r]-(e2)-[:FROM_CHUNK]->(c2:Chunk)
                WHERE c2.qdrant_point_id IS NOT NULL
                  AND NOT c2.qdrant_point_id IN $seen_ids
                  AND c2.qdrant_collection IN $collections
                RETURN DISTINCT c2.qdrant_point_id AS point_id,
                                c2.qdrant_collection AS collection
                LIMIT 40
                """,
                seen_ids=list(seen_point_ids),
                collections=collections,
            )
            expansion_targets = [(r["point_id"], r["collection"]) for r in result]

        if not expansion_targets:
            logger.debug("GraphService.expand: no graph-connected chunks found beyond current result set")
            return []

        logger.info(
            "GraphService.expand: found %d graph-connected chunks to fetch from Qdrant",
            len(expansion_targets),
        )

        # Group by collection and fetch from Qdrant (text only, no re-embedding)
        from collections import defaultdict
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

        logger.info(
            "GraphService.expand: fetched %d graph-expanded docs from Qdrant",
            len(expanded_docs),
        )
        return expanded_docs

    except Exception as exc:
        logger.warning("GraphService.expand: expansion failed (non-fatal): %s", exc)
        return []


# ── Retrieval: entity context enrichment ──────────────────────────────────────

def enrich_docs_with_graph(docs: list[LangchainDocument]) -> list[LangchainDocument]:
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
    if not settings.GRAPHRAG_ENABLED or not docs:
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
                    # Primary path: lookup by Qdrant point UUID (O(1) index hit)
                    result = session.run(
                        """
                        MATCH (c:Chunk {qdrant_point_id: $point_id})
                        OPTIONAL MATCH (e)-[:FROM_CHUNK]->(c)
                        OPTIONAL MATCH (e)-[r]-(neighbor)
                        WITH e.name AS ename, type(r) AS rel, neighbor.name AS nname
                        WHERE ename IS NOT NULL AND nname IS NOT NULL
                        RETURN collect(DISTINCT [ename, rel, nname])[..40] AS triples
                        """,
                        point_id=point_id,
                    )
                elif doc_id is not None and chunk_idx is not None:
                    # Fallback: legacy lookup by (document_id, chunk_index)
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


# ── Deletion ───────────────────────────────────────────────────────────────────

def delete_graph_for_document(kb_id: int, document_id: int) -> None:
    """
    Remove all Neo4j Chunk nodes for a deleted document, and clean up
    any Entity nodes that no longer have any Chunk connections.

    NOT gated on GRAPHRAG_ENABLED — data may exist from a prior ingest
    run when the flag was on.
    """
    if not settings.NEO4J_URI:
        return

    driver = _get_driver()
    with driver.session() as session:
        rec = session.run(
            """
            MATCH (c:Chunk {document_id: $doc_id})
            DETACH DELETE c
            RETURN count(c) AS deleted
            """,
            doc_id=str(document_id),
        ).single()
        logger.info(
            "GraphService: deleted %d Chunk nodes for doc %d",
            rec["deleted"] if rec else 0, document_id,
        )

        rec = session.run(
            """
            MATCH (e:__Entity__)
            WHERE NOT EXISTS { MATCH (:Chunk)-[:FROM_CHUNK]->(e) }
            DETACH DELETE e
            RETURN count(e) AS cleaned
            """
        ).single()
        logger.info(
            "GraphService: cleaned %d orphaned Entity nodes after doc %d deletion",
            rec["cleaned"] if rec else 0, document_id,
        )


def delete_graph_for_kb(kb_id: int) -> None:
    """
    Remove all Neo4j nodes for an entire deleted knowledge base.

    NOT gated on GRAPHRAG_ENABLED — same reasoning as delete_graph_for_document.
    """
    if not settings.NEO4J_URI:
        return

    driver = _get_driver()
    with driver.session() as session:
        rec = session.run(
            """
            MATCH (n {kb_id: $kb_id})
            DETACH DELETE n
            RETURN count(n) AS deleted
            """,
            kb_id=str(kb_id),
        ).single()
        logger.info(
            "GraphService: deleted %d nodes for kb_%d",
            rec["deleted"] if rec else 0, kb_id,
        )

        rec = session.run(
            """
            MATCH (e:__Entity__)
            WHERE NOT EXISTS { MATCH (:Chunk)-[:FROM_CHUNK]->(e) }
            DETACH DELETE e
            RETURN count(e) AS cleaned
            """
        ).single()
        logger.info(
            "GraphService: cleaned %d orphaned Entity nodes after kb_%d deletion",
            rec["cleaned"] if rec else 0, kb_id,
        )
